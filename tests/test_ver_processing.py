import unittest

import numpy as np

from ver_filter import BandpassFilter
from ver_scope import VERScopeProcessor
from ver_wavelet import compute_wavelet_scalogram


class VERProcessingTests(unittest.TestCase):
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

    def test_wavelet_output_shapes(self):
        epoch = np.sin(2 * np.pi * 10 * np.arange(125) / 250.0)
        power, freqs = compute_wavelet_scalogram(epoch)
        self.assertEqual(power.shape[1], epoch.shape[0])
        self.assertGreater(power.shape[0], 0)
        self.assertEqual(power.shape[0], freqs.shape[0])


if __name__ == "__main__":
    unittest.main()
