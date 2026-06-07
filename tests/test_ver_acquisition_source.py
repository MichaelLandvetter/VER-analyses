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

    def test_serial_source_class_is_present(self):
        self.assertIn("class SerialAcquisitionSource:", self.source)

    def test_serial_source_opens_pyserial_port(self):
        self.assertIn("import serial  # pyserial", self.source)
        self.assertIn("serial.Serial(self.port, baudrate=self.baud_rate, timeout=self.timeout)", self.source)

    def test_serial_source_is_binary_only(self):
        self.assertNotIn("def _try_parse_ascii_sample(self) -> Optional[np.ndarray]:", self.source)
        self.assertNotIn("parts = text.split(\",\")", self.source)
        self.assertNotIn("trigger = float(parts[0])", self.source)
        self.assertNotIn("eeg = float(parts[1])", self.source)

    def test_serial_source_parses_binary_packets(self):
        self.assertIn("def _try_parse_binary_sample(self) -> Optional[np.ndarray]:", self.source)
        self.assertIn("def _decode_serial_trigger(self, trigger_state: int) -> float:", self.source)
        self.assertIn("self._binary_header = b\"\\xA5\\x5A\"", self.source)
        self.assertIn("self._binary_packet_size = 9", self.source)
        self.assertIn("struct.unpack(\"<2sHf1s\", packet)", self.source)
        self.assertIn("trigger_level = self._decode_serial_trigger(trigger_state)", self.source)

    def test_serial_source_skips_malformed_packets(self):
        self.assertIn("if packet[-1] != self._binary_footer:", self.source)

    def test_serial_source_has_close_method(self):
        self.assertIn("def close(self) -> None:", self.source)
        self.assertIn("self._serial.close()", self.source)

    def test_serial_source_yields_trigger_eeg_array(self):
        self.assertIn("return np.asarray([trigger_level, float(eeg)], dtype=float)", self.source)
        self.assertIn("self._serial_trigger_high_threshold", self.source)
        self.assertIn("self._serial_trigger_low_threshold", self.source)

    def test_serial_config_is_imported(self):
        self.assertIn("SERIAL_CONFIG", self.source)


if __name__ == "__main__":
    unittest.main()
