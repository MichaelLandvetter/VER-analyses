import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

from ver_report import save_ver_report

matplotlib.use("Agg")


class VERReportTests(unittest.TestCase):
    def test_save_ver_report_returns_png_and_pdf(self):
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
            self.assertIn("png", result)
            self.assertIn("pdf", result)
            self.assertTrue(result["png"].endswith(".png"))
            self.assertTrue(result["pdf"].endswith(".pdf"))
            self.assertTrue(Path(result["png"]).exists())
            self.assertTrue(Path(result["pdf"]).exists())

    def test_save_ver_report_uses_sequential_minute_layout(self):
        import matplotlib.pyplot as plt
        from ver_report import _build_figures_page
        from ver_wavelet import compute_wavelet_scalogram

        epoch_time_ms = np.linspace(-50, 400, 126)
        session_averages_list = [
            np.sin(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
            np.cos(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
        ]
        averages = np.asarray(session_averages_list)

        session_wavelets = []
        session_wavelet_freqs = None
        for avg in session_averages_list:
            power, freqs = compute_wavelet_scalogram(avg)
            session_wavelets.append(power)
            session_wavelet_freqs = freqs
        session_wavelets_arr = np.asarray(session_wavelets)
        freq_min = float(session_wavelet_freqs[0])
        freq_max = float(session_wavelet_freqs[-1])
        labels = ["Minute 1", "Minute 2"]

        fig = _build_figures_page(
            averages,
            epoch_time_ms,
            session_wavelets,
            session_wavelets_arr,
            session_wavelet_freqs,
            freq_min,
            freq_max,
            labels,
        )

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

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
