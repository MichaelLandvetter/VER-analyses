import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
