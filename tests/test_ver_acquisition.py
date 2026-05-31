import tempfile
import unittest
from pathlib import Path

from ver_acquisition import FileAcquisitionSimulator
from ver_config import FILE_CONFIG, FILE_FORMATS


class FileAcquisitionSimulatorTests(unittest.TestCase):
    def setUp(self):
        self._original_config = dict(FILE_CONFIG)

    def tearDown(self):
        FILE_CONFIG.clear()
        FILE_CONFIG.update(self._original_config)

    def test_sd_card_format_uses_threshold_trigger_and_eeg_column_2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = Path(tmp_dir) / "sd.txt"
            data_file.write_text("0.0\t9\t1.5\t0\t0\n1.0\t9\t-2.0\t0\t0\n", encoding="utf-8")

            FILE_CONFIG.clear()
            FILE_CONFIG.update(FILE_FORMATS["SD-card"])
            samples = list(FileAcquisitionSimulator(str(data_file), speed_factor=None).stream_samples())

            self.assertEqual(samples[0][0], 0.0)
            self.assertEqual(samples[0][1], 1.5)
            self.assertEqual(samples[1][0], 1.0)
            self.assertEqual(samples[1][1], -2.0)

    def test_labchart_format_uses_interval_trigger_and_eeg_column_1(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = Path(tmp_dir) / "lab.txt"
            data_file.write_text("0.04\t0.02\n0.43\t-0.015\n", encoding="utf-8")

            FILE_CONFIG.clear()
            FILE_CONFIG.update(FILE_FORMATS["LabChart"])
            samples = list(FileAcquisitionSimulator(str(data_file), speed_factor=None).stream_samples())

            self.assertEqual(samples[0][0], 0.0)
            self.assertAlmostEqual(samples[0][1], 0.02)
            self.assertEqual(samples[1][0], 1.0)
            self.assertAlmostEqual(samples[1][1], -0.015)

    def test_file_config_initializes_from_sd_card_format(self):
        self.assertEqual(FILE_CONFIG, FILE_FORMATS["SD-card"])
        self.assertIsNot(FILE_CONFIG, FILE_FORMATS["SD-card"])


if __name__ == "__main__":
    unittest.main()
