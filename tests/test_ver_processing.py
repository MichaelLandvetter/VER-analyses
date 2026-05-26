import unittest

import numpy as np

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

    def test_wavelet_power_normalised_to_0_1(self):
        epoch = np.sin(2 * np.pi * 10 * np.arange(125) / 250.0)
        power, _ = compute_wavelet_scalogram(epoch)
        self.assertAlmostEqual(float(np.max(power)), 1.0, places=6)
        self.assertGreaterEqual(float(np.min(power)), 0.0)

    def test_wavelet_power_comparable_across_amplitude_scales(self):
        """Power should be normalised so SD-card and LabChart amplitudes give same max."""
        t = np.arange(125) / 250.0
        epoch_large = 10.0 * np.sin(2 * np.pi * 10 * t)
        epoch_small = 0.02 * np.sin(2 * np.pi * 10 * t)
        power_large, _ = compute_wavelet_scalogram(epoch_large)
        power_small, _ = compute_wavelet_scalogram(epoch_small)
        self.assertAlmostEqual(float(np.max(power_large)), 1.0, places=6)
        self.assertAlmostEqual(float(np.max(power_small)), 1.0, places=6)

    def test_detect_ver_peaks_finds_expected_peaks(self):
        """Synthetic waveform with known peaks in expected windows."""
        sample_rate = 250.0
        t = np.arange(-100, 300, 1000.0 / sample_rate)  # ms axis
        # N75 at 75 ms (negative), P100 at 100 ms (positive), N135 at 135 ms (negative)
        epoch = (
            -1.5 * np.exp(-((t - 75) ** 2) / (2 * 10 ** 2))   # N75
            + 2.0 * np.exp(-((t - 100) ** 2) / (2 * 10 ** 2))  # P100
            - 1.0 * np.exp(-((t - 135) ** 2) / (2 * 10 ** 2))  # N135
        )
        peaks = detect_ver_peaks(epoch, t)

        self.assertTrue(peaks['N75']['found'])
        self.assertTrue(peaks['P100']['found'])
        self.assertTrue(peaks['N135']['found'])

        self.assertAlmostEqual(peaks['N75']['latency_ms'], 75.0, delta=5.0)
        self.assertAlmostEqual(peaks['P100']['latency_ms'], 100.0, delta=5.0)
        self.assertAlmostEqual(peaks['N135']['latency_ms'], 135.0, delta=5.0)

        self.assertLess(peaks['N75']['amplitude'], 0)
        self.assertGreater(peaks['P100']['amplitude'], 0)
        self.assertLess(peaks['N135']['amplitude'], 0)

    def test_detect_ver_peaks_not_found_when_window_missing(self):
        """If time axis doesn't cover the window, found should be False."""
        t = np.linspace(0, 50, 100)  # only 0–50 ms, so N75 window 50–100 is at edge
        epoch = np.zeros(100)
        peaks = detect_ver_peaks(epoch, t)
        # P100 window is 80–130 ms, entirely outside 0–50 ms range → not found
        self.assertFalse(peaks['P100']['found'])
        self.assertTrue(np.isnan(peaks['P100']['latency_ms']))


if __name__ == "__main__":
    unittest.main()
