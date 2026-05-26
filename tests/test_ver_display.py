import unittest
from pathlib import Path


class VERDisplaySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        display_path = Path(__file__).resolve().parents[1] / "ver_display.py"
        cls.source = display_path.read_text(encoding="utf-8")

    def test_flash_markers_use_filtered_trace_height(self):
        self.assertIn("filt_max = float(np.max(y_filt))", self.source)
        self.assertIn("filt_min = float(np.min(y_filt))", self.source)
        self.assertIn("y_dot = filt_max + 0.1 * filt_range", self.source)
        self.assertNotIn("y_max = float(np.max(y_raw)) if len(y_raw) else 1.0", self.source)
        # Verify no hardcoded absolute offset of 0.5 is used as minimum
        self.assertNotIn("max(0.1 * filt_range, 0.5)", self.source)

    def test_sessions_panel_removes_legend_and_sets_fixed_x_range(self):
        self.assertIn('self.plot_sessions.setXRange(-100, 400, padding=0)', self.source)
        self.assertNotIn("self.plot_sessions.addLegend()", self.source)

    def test_session_average_plot_has_no_legend_name(self):
        plot_block = """        self.plot_sessions.plot(
            epoch_time_ms,
            session_avg + offset,
            pen=pg.mkPen(color, width=2),
        )"""
        self.assertIn(plot_block, self.source)

    def test_wavelet_stats_label_and_update_method_exist(self):
        self.assertIn('self.wavelet_stats_label = QLabel("Peak: — Hz | — ms | Power: —")', self.source)
        self.assertIn(
            'f"M{session_number} — Peak: {peak_freq:.1f} Hz | {peak_latency_ms:.0f} ms | Power: {peak_power:.2f}"',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
