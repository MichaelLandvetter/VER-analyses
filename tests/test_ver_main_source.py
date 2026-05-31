import unittest
from pathlib import Path


class VERMainSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main_path = Path(__file__).resolve().parents[1] / "ver_main.py"
        cls.source = main_path.read_text(encoding="utf-8")

    def test_file_format_selector_is_present(self):
        self.assertIn("self.format_combo = QComboBox()", self.source)
        self.assertIn("self.format_combo.addItems(list(FILE_FORMATS.keys()))", self.source)
        self.assertIn("def _on_format_changed(self, format_name: str):", self.source)
        self.assertIn("FILE_CONFIG.update(FILE_FORMATS[format_name])", self.source)
        self.assertIn("self.reset_all()", self.source)
        self.assertIn('downsample_action = QAction("Downsample LabChart file (1000 Hz → 250 Hz)...", self)', self.source)
        self.assertIn("downsample_action.triggered.connect(self._on_downsample)", self.source)
        self.assertIn("def _on_downsample(self):", self.source)

    def test_downsample_helper_uses_decimate(self):
        self.assertIn("def downsample_labchart_file(input_filepath: str) -> str:", self.source)
        self.assertIn("from scipy.signal import decimate", self.source)
        self.assertIn("decimate(col, q=4, ftype=\"fir\", zero_phase=True)", self.source)

    def test_speed_selector_combo_is_wired_to_speed_factor(self):
        self.assertIn('self.speed_combo = QComboBox()', self.source)
        self.assertIn('"Real-time (1×)"', self.source)
        self.assertIn('"Fast (10×)"', self.source)
        self.assertIn('"Maximum speed"', self.source)
        self.assertIn("def _get_speed_factor(self)", self.source)
        self.assertIn("speed_factor", self.source)

    def test_wavelet_stats_are_computed_and_sent_to_display(self):
        self.assertIn("peak_idx = np.unravel_index(np.argmax(power), power.shape)", self.source)
        self.assertIn("self.display.update_wavelet_stats(peak_freq, peak_latency_ms, peak_power, session_num, ver_peaks=ver_peaks)", self.source)
        self.assertIn("ver_peaks=ver_peaks", self.source)

    def test_report_success_message_shows_report_directory(self):
        self.assertIn('f"Reports saved to:\\n{report_dir}', self.source)


if __name__ == "__main__":
    unittest.main()
