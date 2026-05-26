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

    def test_fast_mode_checkbox_is_wired_to_simulate_realtime(self):
        self.assertIn('self.fast_mode_check = QCheckBox("Fast mode")', self.source)
        self.assertIn("simulate_rt = not self.fast_mode_check.isChecked()", self.source)
        self.assertIn("self._start_worker(simulate_realtime=simulate_rt)", self.source)

    def test_wavelet_stats_are_computed_and_sent_to_display(self):
        self.assertIn("peak_idx = np.unravel_index(np.argmax(power), power.shape)", self.source)
        self.assertIn("self.display.update_wavelet_stats(peak_freq, peak_latency_ms, peak_power, session_num)", self.source)


if __name__ == "__main__":
    unittest.main()
