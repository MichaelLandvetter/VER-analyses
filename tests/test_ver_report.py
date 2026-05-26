import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

from ver_report import save_ver_report

matplotlib.use("Agg")


class VERReportTests(unittest.TestCase):
    def test_save_ver_report_returns_png_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "sample.txt"
            input_file.write_text("placeholder\n", encoding="utf-8")

            epoch_time_ms = np.linspace(-50, 400, 126)
            session_averages = [
                np.sin(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
                np.cos(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
            ]

            result = save_ver_report(str(input_file), session_averages, epoch_time_ms)

            self.assertIsNotNone(result)
            self.assertEqual(set(result.keys()), {"png"})
            self.assertTrue(result["png"].endswith(".png"))
            self.assertTrue(Path(result["png"]).exists())

    def test_save_ver_report_uses_sequential_minute_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "sample.txt"
            input_file.write_text("placeholder\n", encoding="utf-8")

            epoch_time_ms = np.linspace(-50, 400, 126)
            session_averages = [
                np.sin(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
                np.cos(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
            ]
            captured = {}

            def capture_savefig(fig, path, *args, **kwargs):
                captured["figure"] = fig
                captured["path"] = str(path)

            with patch("matplotlib.figure.Figure.savefig", autospec=True, side_effect=capture_savefig), patch(
                "matplotlib.pyplot.close"
            ):
                result = save_ver_report(str(input_file), session_averages, epoch_time_ms)

            self.assertEqual(set(result.keys()), {"png"})
            self.assertTrue(result["png"].endswith(".png"))

            fig = captured["figure"]
            ax1, ax2, colorbar_ax = fig.axes
            self.assertEqual(fig.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
            self.assertEqual(ax1.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
            self.assertEqual(ax2.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
            self.assertEqual(ax1.get_title(), "VER Evolution — Minute by Minute")
            self.assertEqual(ax1.get_xlabel(), "Minute")
            self.assertEqual(ax1.get_ylabel(), "Amplitude (µV)")
            self.assertEqual([tick.get_text() for tick in ax1.get_xticklabels()], ["M1", "M2"])

            trace_lines = [line for line in ax1.lines if len(line.get_xdata()) == epoch_time_ms.size]
            self.assertEqual(len(trace_lines), 2)
            self.assertAlmostEqual(float(trace_lines[0].get_xdata()[0]), 0.0)
            self.assertAlmostEqual(float(trace_lines[1].get_xdata()[0]), 450.0)

            self.assertEqual(ax2.get_title(), "Wavelet Scalograms by Minute")
            self.assertEqual([tick.get_text() for tick in ax2.get_xticklabels()], ["M1", "M2"])
            self.assertEqual(colorbar_ax.get_ylabel(), "Power")
            self.assertGreater(len(fig.texts), 0)
            stats_text = fig.texts[0].get_text()
            self.assertIn("M1:", stats_text)
            self.assertIn("Hz |", stats_text)
            self.assertIn("Power", stats_text)


if __name__ == "__main__":
    unittest.main()
