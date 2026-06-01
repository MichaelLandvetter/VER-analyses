import unittest
from unittest.mock import patch

import numpy as np

import ver_peaks
from ver_filter import BandpassFilter
from ver_peaks import detect_ver_peaks
from ver_scope import VERScopeProcessor
from ver_wavelet import compute_wavelet_scalogram


class VERProcessingTests(unittest.TestCase):
    @staticmethod
    def _process_signal(scope, trigger, signal):
        results = []
        for tr, eeg in zip(trigger, signal):
            results.append(scope.process_sample(tr, eeg))
        return results

    def test_rising_edge_detection_counts_once_per_pulse(self):
        bp = BandpassFilter()
        scope = VERScopeProcessor(
            bp,
            epoch_config={
                "pre_stim_ms": 20,
                "post_stim_ms": 40,
                "flashes_per_session": 2,
                "num_sessions": 1,
            },
        )

        total_triggers = 0
        signal = np.sin(np.linspace(0, 10, 200))
        trigger = np.zeros(200)
        trigger[40:45] = 1  # one multi-sample pulse
        trigger[120:125] = 1

        for tr, eeg in zip(trigger, signal):
            result = scope.process_sample(tr, eeg)
            if result["trigger_detected"]:
                total_triggers += 1

        self.assertEqual(total_triggers, 2)
        self.assertEqual(len(scope.session_averages), 1)

    def test_completed_session_number_matches_finished_session(self):
        bp = BandpassFilter()
        scope = VERScopeProcessor(
            bp,
            epoch_config={
                "pre_stim_ms": 20,
                "post_stim_ms": 40,
                "flashes_per_session": 2,
                "num_sessions": 2,
            },
        )

        signal = np.sin(np.linspace(0, 20, 260))
        trigger = np.zeros(260)
        trigger[40:45] = 1
        trigger[120:125] = 1
        trigger[200:205] = 1

        results = self._process_signal(scope, trigger, signal)
        completed = [result for result in results if result["session_complete"]]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["completed_session_number"], 1)
        self.assertEqual(completed[0]["session_number"], 1)

        third_epoch = next(result for result in results if result["epoch_complete"] and result["flash_count"] == 1 and result["session_number"] == 2)
        self.assertFalse(third_epoch["session_complete"])

    def test_partial_session_can_be_saved_after_half_the_flashes(self):
        bp = BandpassFilter()
        scope = VERScopeProcessor(
            bp,
            epoch_config={
                "pre_stim_ms": 20,
                "post_stim_ms": 40,
                "flashes_per_session": 4,
                "num_sessions": 2,
            },
        )

        signal = np.sin(np.linspace(0, 20, 260))
        trigger = np.zeros(260)
        trigger[40:45] = 1
        trigger[120:125] = 1
        trigger[200:205] = 1

        self._process_signal(scope, trigger, signal)
        partial = scope.save_partial_session()

        self.assertIsNotNone(partial)
        self.assertEqual(partial["session_number"], 1)
        self.assertEqual(partial["flash_count"], 3)
        self.assertEqual(len(scope.session_averages), 1)
        self.assertEqual(scope.session_index, 1)
        self.assertEqual(scope.flash_count, 0)
        self.assertIsNone(scope.running_average)

    def test_wavelet_output_shapes(self):
        epoch = np.sin(2 * np.pi * 10 * np.arange(125) / 250.0)
        power, freqs = compute_wavelet_scalogram(epoch)
        self.assertEqual(power.shape[1], epoch.shape[0])
        self.assertGreater(power.shape[0], 0)
        self.assertEqual(power.shape[0], freqs.shape[0])

    def test_wavelet_power_returns_positive_values(self):
        epoch = np.sin(2 * np.pi * 10 * np.arange(125) / 250.0)
        power, _ = compute_wavelet_scalogram(epoch)
        self.assertGreaterEqual(float(np.min(power)), 0.0)
        self.assertGreater(float(np.max(power)), 0.0)

    def test_wavelet_power_scales_with_signal_amplitude(self):
        t = np.arange(125) / 250.0
        epoch_large = 10.0 * np.sin(2 * np.pi * 10 * t)
        epoch_small = 0.02 * np.sin(2 * np.pi * 10 * t)
        power_large, _ = compute_wavelet_scalogram(epoch_large)
        power_small, _ = compute_wavelet_scalogram(epoch_small)
        self.assertGreater(float(np.max(power_large)), float(np.max(power_small)))

    def test_detect_ver_peaks_finds_expected_peaks(self):
        """Synthetic waveform with known peaks — three largest in 0–200ms, sorted by latency."""
        sample_rate = 250.0
        t = np.arange(-100, 300, 1000.0 / sample_rate)  # ms axis
        # Three peaks in 0–200ms: negative at ~75ms, positive at ~100ms, negative at ~135ms
        epoch = (
            -1.5 * np.exp(-((t - 75) ** 2) / (2 * 10 ** 2))   # largest negative peak at 75ms
            + 2.0 * np.exp(-((t - 100) ** 2) / (2 * 10 ** 2))  # largest positive peak at 100ms
            - 1.0 * np.exp(-((t - 135) ** 2) / (2 * 10 ** 2))  # smaller negative peak at 135ms
        )
        peaks = detect_ver_peaks(epoch, t)

        self.assertTrue(peaks['Peak-1']['found'])
        self.assertTrue(peaks['Peak-2']['found'])
        self.assertTrue(peaks['Peak-3']['found'])

        # Peaks are sorted by latency: ~75ms, ~100ms, ~135ms
        self.assertAlmostEqual(peaks['Peak-1']['latency_ms'], 75.0, delta=5.0)
        self.assertAlmostEqual(peaks['Peak-2']['latency_ms'], 100.0, delta=5.0)
        self.assertAlmostEqual(peaks['Peak-3']['latency_ms'], 135.0, delta=5.0)

        # Amplitudes retain sign
        self.assertLess(peaks['Peak-1']['amplitude'], 0)
        self.assertGreater(peaks['Peak-2']['amplitude'], 0)
        self.assertLess(peaks['Peak-3']['amplitude'], 0)

    def test_detect_ver_peaks_polarity_agnostic(self):
        """Inverted polarity waveform (P-N-P) should also detect all three peaks."""
        sample_rate = 250.0
        t = np.arange(-100, 300, 1000.0 / sample_rate)  # ms axis
        # Inverted: positive at ~75ms, negative at ~100ms, positive at ~135ms
        epoch = (
            1.5 * np.exp(-((t - 75) ** 2) / (2 * 10 ** 2))
            - 2.0 * np.exp(-((t - 100) ** 2) / (2 * 10 ** 2))
            + 1.0 * np.exp(-((t - 135) ** 2) / (2 * 10 ** 2))
        )
        peaks = detect_ver_peaks(epoch, t)

        self.assertTrue(peaks['Peak-1']['found'])
        self.assertTrue(peaks['Peak-2']['found'])
        self.assertTrue(peaks['Peak-3']['found'])

        self.assertAlmostEqual(peaks['Peak-1']['latency_ms'], 75.0, delta=5.0)
        self.assertAlmostEqual(peaks['Peak-2']['latency_ms'], 100.0, delta=5.0)
        self.assertAlmostEqual(peaks['Peak-3']['latency_ms'], 135.0, delta=5.0)

    def test_detect_ver_peaks_not_found_when_window_missing(self):
        """If time axis doesn't cover the 0–200ms window, found should be False."""
        t = np.linspace(-100, -10, 100)  # only negative time values, no 0–200ms range
        epoch = np.zeros(100)
        peaks = detect_ver_peaks(epoch, t)
        # 0–200ms window is entirely outside the time axis → all not found
        self.assertFalse(peaks['Peak-1']['found'])
        self.assertTrue(np.isnan(peaks['Peak-1']['latency_ms']))

    def test_detect_ver_peaks_applies_minus_100_to_0_baseline(self):
        """Default config keeps baseline correction in the -100..0ms window."""
        sample_rate = 250.0
        t = np.arange(-100, 300, 1000.0 / sample_rate)
        baseline_offset = 5.0
        epoch = (
            baseline_offset
            + 1.0 * np.exp(-((t - 70) ** 2) / (2 * 8 ** 2))
            - 1.2 * np.exp(-((t - 110) ** 2) / (2 * 8 ** 2))
            + 0.8 * np.exp(-((t - 150) ** 2) / (2 * 8 ** 2))
        )
        peaks = detect_ver_peaks(epoch, t)

        self.assertAlmostEqual(peaks['Peak-1']['latency_ms'], 70.0, delta=6.0)
        self.assertAlmostEqual(peaks['Peak-2']['latency_ms'], 110.0, delta=6.0)
        self.assertAlmostEqual(peaks['Peak-3']['latency_ms'], 150.0, delta=6.0)
        self.assertAlmostEqual(peaks['Peak-1']['amplitude'], 1.0, delta=0.25)
        self.assertAlmostEqual(peaks['Peak-2']['amplitude'], -1.2, delta=0.25)
        self.assertAlmostEqual(peaks['Peak-3']['amplitude'], 0.8, delta=0.25)

    def test_detect_ver_peaks_uses_configured_baseline_period(self):
        t = np.arange(-100, 500, 4.0)
        epoch = (
            2.0
            + 3.0 * ((t >= -100) & (t < 0)).astype(float)
            + 3.0 * np.exp(-((t - 70) ** 2) / (2 * 8 ** 2))
        )

        with patch.object(ver_peaks.ver_config, "BASELINE_START_MS", 250), patch.object(ver_peaks.ver_config, "BASELINE_END_MS", 450):
            peaks = detect_ver_peaks(epoch, t)

        self.assertAlmostEqual(peaks["Peak-1"]["latency_ms"], 70.0, delta=6.0)
        self.assertAlmostEqual(peaks["Peak-1"]["amplitude"], 3.0, delta=0.3)

    def test_detect_ver_peaks_adds_snr_and_ver_detected(self):
        t = np.arange(-100, 300, 4.0)
        epoch = (
            0.05 * np.sin(2 * np.pi * t / 30.0)
            + 2.0 * np.exp(-((t - 90) ** 2) / (2 * 7 ** 2))
            - 1.5 * np.exp(-((t - 130) ** 2) / (2 * 9 ** 2))
        )
        peaks = detect_ver_peaks(epoch, t)

        self.assertGreater(peaks["noise_rms"], 0.0)
        self.assertTrue(peaks["VER_detected"])
        self.assertTrue(peaks["Peak-1"]["found"])
        self.assertIn("snr", peaks["Peak-1"])
        self.assertIn("above_threshold", peaks["Peak-1"])
        self.assertTrue(any(peaks[name]["above_threshold"] for name in ("Peak-1", "Peak-2", "Peak-3")))

    def test_detect_ver_peaks_uses_configured_snr_threshold(self):
        t = np.arange(-100, 300, 4.0)
        baseline_mask = (t >= -100) & (t < 0)
        baseline = np.sin(np.linspace(0, 4 * np.pi, np.count_nonzero(baseline_mask)))
        baseline *= 0.25 / np.sqrt(np.mean(baseline ** 2))
        epoch = np.zeros_like(t)
        epoch[baseline_mask] = baseline
        epoch += 0.7 * np.exp(-((t - 90) ** 2) / (2 * 7 ** 2))

        with patch.object(ver_peaks.ver_config, "SNR_THRESHOLD", 3.0):
            peaks = detect_ver_peaks(epoch, t)

        self.assertGreater(peaks["Peak-1"]["snr"], 2.0)
        self.assertLess(peaks["Peak-1"]["snr"], 3.0)
        self.assertFalse(peaks["Peak-1"]["above_threshold"])
        self.assertFalse(peaks["VER_detected"])

    def test_detect_ver_peaks_marks_no_ver_for_noise_only(self):
        rng = np.random.default_rng(7)
        t = np.arange(-100, 300, 4.0)
        epoch = np.zeros(t.size)
        baseline_mask = (t >= -100) & (t < 0)
        post_mask = (t >= 0) & (t <= 200)
        epoch[baseline_mask] = 0.2 * rng.standard_normal(np.count_nonzero(baseline_mask))
        epoch[post_mask] = 0.01 * rng.standard_normal(np.count_nonzero(post_mask))
        peaks = detect_ver_peaks(epoch, t)

        self.assertFalse(peaks["VER_detected"])
        self.assertGreater(peaks["noise_rms"], 0.0)
        self.assertFalse(any(peaks[name]["above_threshold"] for name in ("Peak-1", "Peak-2", "Peak-3")))


if __name__ == "__main__":
    unittest.main()
