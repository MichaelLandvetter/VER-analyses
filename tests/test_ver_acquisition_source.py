import unittest
from pathlib import Path


class VERAcquisitionSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1] / "ver_acquisition.py"
        cls.source = source_path.read_text(encoding="utf-8")

    def test_waveshare_stream_is_rate_limited_to_sample_rate(self):
        self.assertIn("sample_interval = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0", self.source)
        self.assertIn("next_sample_time = time.perf_counter()", self.source)
        self.assertIn("sleep_for = next_sample_time - time.perf_counter()", self.source)
        self.assertIn("if sleep_for > 0:", self.source)
        self.assertIn("time.sleep(sleep_for)", self.source)

    def test_waveshare_trigger_uses_hysteresis_and_min_interval(self):
        self.assertIn("trigger_high_threshold", self.source)
        self.assertIn("trigger_low_threshold", self.source)
        self.assertIn("trigger_min_interval_s", self.source)
        self.assertIn("rising_edge = self._trigger_high and not prev_trigger_high", self.source)
        self.assertIn("or now - self._last_trigger_time >= self.trigger_min_interval_s", self.source)


if __name__ == "__main__":
    unittest.main()
