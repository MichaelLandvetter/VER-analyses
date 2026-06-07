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

    def test_downsample_opens_reusable_dialog(self):
        self.assertIn("class DownsampleDialog(QDialog):", self.source)
        self.assertIn("def _select_and_downsample(self):", self.source)
        self.assertIn("def _on_downsample(self):", self.source)
        self.assertIn("DownsampleDialog(self)", self.source)
        self.assertIn("dlg.exec()", self.source)

    def test_downsample_dialog_has_fixed_status_area(self):
        self.assertIn("self.setFixedSize(560, 300)", self.source)
        self.assertIn("layout.addStretch(1)", self.source)
        self.assertIn('status_title = QLabel("Status")', self.source)
        self.assertIn("self._status_label = QTextBrowser()", self.source)
        self.assertIn("self._status_label.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)", self.source)
        self.assertIn("self._status_label.setFixedHeight(58)", self.source)
        self.assertIn('self._status_label.setPlainText(f"Saved: {output_path}")', self.source)

    def test_no_startup_file_prompt(self):
        self.assertNotIn("self._select_data_file(initial=True)", self.source)

    def test_downsample_helper_uses_decimate(self):
        self.assertIn("def downsample_labchart_file(input_filepath: str) -> str:", self.source)
        self.assertIn("from scipy.signal import decimate", self.source)
        self.assertIn("source_rate_hz = 1000", self.source)
        self.assertIn("target_rate_hz = 250", self.source)
        self.assertIn("decimation_factor = source_rate_hz // target_rate_hz", self.source)
        self.assertIn("decimate(col, q=decimation_factor, ftype=\"fir\", zero_phase=True)", self.source)

    def test_speed_selector_combo_is_wired_to_speed_factor(self):
        self.assertIn('self.speed_combo = QComboBox()', self.source)
        self.assertIn('"Real-time (1×)"', self.source)
        self.assertIn('"Fast (10×)"', self.source)
        self.assertIn('"Maximum speed"', self.source)
        self.assertIn("def _get_speed_factor(self)", self.source)
        self.assertIn("speed_factor", self.source)

    def test_serial_port_combo_supports_manual_port_entry(self):
        self.assertIn("self.serial_port_combo.setEditable(True)", self.source)
        self.assertIn("self.serial_port_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)", self.source)
        self.assertIn('self.serial_port_combo.setPlaceholderText("Select or type port")', self.source)
        self.assertIn("self.serial_port_combo.setEditText(current)", self.source)
        self.assertIn('configured_port = str(SERIAL_CONFIG.get("port", "")).strip()', self.source)

    def test_wavelet_stats_are_computed_and_sent_to_display(self):
        self.assertIn("peak_idx = np.unravel_index(np.argmax(power), power.shape)", self.source)
        self.assertIn("self.display.update_wavelet_stats(peak_freq, peak_latency_ms, peak_power, session_num, ver_peaks=ver_peaks)", self.source)
        self.assertIn("ver_peaks=ver_peaks", self.source)

    def test_report_success_message_shows_report_directory(self):
        self.assertIn('f"Reports saved to:\\n{report_dir}', self.source)
        self.assertIn('Summary CSV: {summary_csv_name}', self.source)
        self.assertIn('Waveforms CSV: {waveforms_csv_name}', self.source)

    def test_close_event_prompts_save_before_shutdown_and_supports_cancel(self):
        self.assertIn('def closeEvent(self, event):', self.source)
        self.assertIn('"Save before exit?"', self.source)
        self.assertIn('"You have session data. Save a report before exiting?"', self.source)
        self.assertIn("QMessageBox.StandardButton.Cancel", self.source)
        self.assertIn("if resp == QMessageBox.StandardButton.Cancel:", self.source)
        self.assertIn("event.ignore()", self.source)
        self.assertIn("if resp == QMessageBox.StandardButton.Yes:", self.source)
        self.assertIn("self.save_report()", self.source)

    def test_close_event_saves_prompt_before_worker_shutdown(self):
        close_event_start = self.source.index("def closeEvent(self, event):")
        close_event_source = self.source[close_event_start:]
        prompt_marker = '"Save before exit?"'
        shutdown_marker = "self._shutdown_worker()"
        self.assertIn(prompt_marker, close_event_source)
        self.assertIn(shutdown_marker, close_event_source)
        self.assertLess(close_event_source.index(prompt_marker), close_event_source.index(shutdown_marker))

    def test_worker_emits_eof_only_while_running(self):
        self.assertIn("if self._running:", self.source)
        self.assertIn("self.eof_reached.emit()", self.source)
        self.assertIn("def stop(self):", self.source)
        self.assertIn("self._running = False", self.source)


if __name__ == "__main__":
    unittest.main()
