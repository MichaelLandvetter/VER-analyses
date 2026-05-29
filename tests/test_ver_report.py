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
            self.assertIn("report_dir", result)
            self.assertTrue(result["png"].endswith(".png"))
            self.assertTrue(result["pdf"].endswith(".pdf"))
            self.assertTrue(Path(result["png"]).exists())
            self.assertTrue(Path(result["pdf"]).exists())
            expected_dir = Path(tmp_dir) / "Reports" / "sample"
            self.assertEqual(Path(result["report_dir"]), expected_dir)
            self.assertEqual(Path(result["png"]), expected_dir / "sample.png")
            self.assertEqual(Path(result["pdf"]), expected_dir / "sample.pdf")
            self.assertTrue(input_file.exists())

    def test_save_ver_report_creates_csv_files(self):
        import csv as csv_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "subject01.txt"
            input_file.write_text("placeholder\n", encoding="utf-8")

            epoch_time_ms = np.linspace(-100, 400, 126)
            session_averages = [
                np.sin(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
                np.cos(np.linspace(0, 3 * np.pi, epoch_time_ms.size)),
            ]
            session_ver_peaks = [
                {
                    "Peak-1": {"found": True, "latency_ms": 75.0, "amplitude": -0.5, "snr": 2.4, "above_threshold": True},
                    "Peak-2": {"found": True, "latency_ms": 100.0, "amplitude": 0.8, "snr": 3.1, "above_threshold": True},
                    "Peak-3": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "VER_detected": True,
                    "noise_rms": 0.2,
                },
                {
                    "Peak-1": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "Peak-2": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "Peak-3": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "VER_detected": False,
                    "noise_rms": 0.3,
                },
            ]
            session_flash_counts = [120, 65]

            result = save_ver_report(
                str(input_file),
                session_averages,
                epoch_time_ms,
                session_ver_peaks=session_ver_peaks,
                session_flash_counts=session_flash_counts,
            )

            self.assertIsNotNone(result)
            self.assertIn("summary_csv", result)
            self.assertIn("waveforms_csv", result)

            summary_path = Path(result["summary_csv"])
            waveforms_path = Path(result["waveforms_csv"])
            expected_dir = Path(tmp_dir) / "Reports" / "subject01"
            self.assertEqual(summary_path, expected_dir / "subject01_summary.csv")
            self.assertEqual(waveforms_path, expected_dir / "subject01_waveforms.csv")
            self.assertTrue(summary_path.exists())
            self.assertTrue(waveforms_path.exists())

            # Verify summary CSV structure and content
            with open(summary_path, encoding="utf-8", newline="") as f:
                rows = list(csv_mod.reader(f))
            self.assertEqual(rows[0], [
                "Minute", "N_flashes", "Peak_power", "VER_detected", "Noise_RMS",
                "Peak1_latency_ms", "Peak1_amplitude", "Peak1_SNR",
                "Peak2_latency_ms", "Peak2_amplitude", "Peak2_SNR",
                "Peak3_latency_ms", "Peak3_amplitude", "Peak3_SNR",
            ])
            self.assertEqual(len(rows), 3)  # header + 2 data rows
            # Minute 1
            self.assertEqual(rows[1][0], "1")
            self.assertEqual(rows[1][1], "120")
            self.assertNotEqual(rows[1][2], "")  # peak_power is present
            self.assertEqual(rows[1][3], "True")
            self.assertEqual(rows[1][4], "0.2")
            self.assertEqual(rows[1][5], "75.0")  # Peak-1 latency
            self.assertEqual(rows[1][8], "100.0")  # Peak-2 latency
            self.assertEqual(rows[1][11], "")  # Peak-3 latency empty (not found)
            # Minute 2
            self.assertEqual(rows[2][0], "2")
            self.assertEqual(rows[2][1], "65")
            self.assertEqual(rows[2][3], "False")
            self.assertEqual(rows[2][5], "")  # Peak-1 latency empty (not found)

            # Verify waveforms CSV structure and content
            with open(waveforms_path, encoding="utf-8", newline="") as f:
                wrows = list(csv_mod.reader(f))
            self.assertEqual(wrows[0], ["Time_ms", "Minute_1", "Minute_2"])
            self.assertEqual(len(wrows), epoch_time_ms.size + 1)  # header + one row per time point
            # First time value should match epoch_time_ms[0]
            self.assertAlmostEqual(float(wrows[1][0]), float(epoch_time_ms[0]), places=5)
            # Two amplitude columns per row
            self.assertEqual(len(wrows[1]), 3)

    def test_save_ver_report_csv_missing_values_are_empty(self):
        """Missing peak values (not found) are written as empty strings, not 'nan'."""
        import csv as csv_mod

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "test.txt"
            input_file.write_text("placeholder\n", encoding="utf-8")

            epoch_time_ms = np.linspace(-100, 400, 126)
            session_averages = [np.zeros(epoch_time_ms.size)]
            session_ver_peaks = [
                {
                    "Peak-1": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "Peak-2": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "Peak-3": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
                    "VER_detected": False,
                    "noise_rms": 0.1,
                }
            ]

            result = save_ver_report(
                str(input_file),
                session_averages,
                epoch_time_ms,
                session_ver_peaks=session_ver_peaks,
            )

            with open(result["summary_csv"], encoding="utf-8", newline="") as f:
                rows = list(csv_mod.reader(f))
            # N_flashes not provided -> empty string
            self.assertEqual(rows[1][1], "")
            # All peaks not found -> empty strings
            self.assertEqual(rows[1][3], "False")
            self.assertEqual(rows[1][5], "")
            self.assertEqual(rows[1][6], "")
            self.assertEqual(rows[1][8], "")
            self.assertEqual(rows[1][11], "")

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
        freq_min = float(session_wavelet_freqs[0])
        freq_max = float(session_wavelet_freqs[-1])
        labels = ["Minute 1", "Minute 2"]
        session_ver_peaks = [
            {
                "Peak-1": {"found": True, "latency_ms": 75.0, "amplitude": -0.5},
                "Peak-2": {"found": True, "latency_ms": 100.0, "amplitude": 0.8},
                "Peak-3": {"found": True, "latency_ms": 135.0, "amplitude": -0.3},
            },
            {
                "Peak-1": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan")},
                "Peak-2": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan")},
                "Peak-3": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan")},
            },
        ]

        fig = _build_figures_page(
            averages,
            epoch_time_ms,
            session_wavelets,
            session_wavelet_freqs,
            freq_min,
            freq_max,
            labels,
            session_ver_peaks=session_ver_peaks,
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
        self.assertIsNotNone(ax1.get_legend())
        self.assertEqual([text.get_text() for text in ax1.get_legend().get_texts()], ["Peak-1", "Peak-2", "Peak-3"])

        plt.close(fig)

    def test_report_ignores_nan_peak_markers(self):
        import matplotlib.pyplot as plt
        from ver_report import _build_figures_page
        from ver_wavelet import compute_wavelet_scalogram

        epoch_time_ms = np.linspace(-50, 400, 126)
        session_averages_list = [np.sin(np.linspace(0, 3 * np.pi, epoch_time_ms.size))]
        averages = np.asarray(session_averages_list)

        session_wavelets = []
        session_wavelet_freqs = None
        for avg in session_averages_list:
            power, freqs = compute_wavelet_scalogram(avg)
            session_wavelets.append(power)
            session_wavelet_freqs = freqs

        fig = _build_figures_page(
            averages,
            epoch_time_ms,
            session_wavelets,
            session_wavelet_freqs,
            float(session_wavelet_freqs[0]),
            float(session_wavelet_freqs[-1]),
            ["Minute 1"],
            session_ver_peaks=[{
                "Peak-1": {"found": True, "latency_ms": float("nan"), "amplitude": -0.5},
                "Peak-2": {"found": True, "latency_ms": 100.0, "amplitude": float("nan")},
                "Peak-3": {"found": True, "latency_ms": float("nan"), "amplitude": float("nan")},
            }],
        )

        ax1 = fig.axes[0]
        for line in ax1.lines:
            self.assertTrue(np.all(np.isfinite(line.get_xdata())))
            self.assertTrue(np.all(np.isfinite(line.get_ydata())))

        plt.close(fig)

    def test_stats_table_includes_ver_column_and_snr_text(self):
        import matplotlib.pyplot as plt
        from ver_report import _build_stats_table_page

        session_wavelets = [np.ones((3, 4))]
        session_wavelet_freqs = np.array([5.0, 10.0, 15.0])
        epoch_time_ms = np.array([-100.0, 0.0, 100.0, 200.0])
        labels = ["Minute 1"]
        session_ver_peaks = [{
            "Peak-1": {"found": True, "latency_ms": 70.0, "amplitude": 0.5, "snr": 2.5, "above_threshold": True},
            "Peak-2": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
            "Peak-3": {"found": False, "latency_ms": float("nan"), "amplitude": float("nan"), "snr": float("nan"), "above_threshold": False},
            "VER_detected": True,
            "noise_rms": 0.2,
        }]

        fig = _build_stats_table_page(
            session_wavelets,
            session_wavelet_freqs,
            epoch_time_ms,
            labels,
            session_ver_peaks=session_ver_peaks,
        )

        table = fig.axes[0].tables[0]
        header_cols = sorted(col for (row, col) in table.get_celld() if row == 0)
        headers = {table.get_celld()[(0, col)].get_text().get_text(): col for col in header_cols}
        self.assertIn("VER?", headers)
        self.assertIn("Peak-1 Amp", headers)
        self.assertEqual(table.get_celld()[(1, headers["VER?"])].get_text().get_text(), "Yes")
        self.assertEqual(table.get_celld()[(1, headers["Peak-1 Amp"])].get_text().get_text(), "0.5000 (SNR=2.5)")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
