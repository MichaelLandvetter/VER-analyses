"""Main application entry point for modular VER analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ver_acquisition import FileAcquisitionSimulator, WaveshareAcquisitionSource
from ver_config import ACQ_CONFIG, EPOCH_CONFIG, FILE_CONFIG, FILE_FORMATS, FILTER_CONFIG, HARDWARE_CONFIG
from ver_display import VERDisplayWidget
from ver_filter import BandpassFilter
from ver_peaks import detect_ver_peaks
from ver_report import save_ver_report
from ver_scope import VERScopeProcessor
from ver_wavelet import compute_wavelet_scalogram


def downsample_labchart_file(input_filepath: str) -> str:
    """
    Downsample a LabChart .txt file from 1000 Hz to 250 Hz using anti-alias decimation.
    Saves the result as <original_name>_250_Hz.txt in the same directory.
    Returns the output file path.
    """
    from scipy.signal import decimate

    input_path = Path(input_filepath)
    output_path = input_path.parent / f"{input_path.stem}_250_Hz{input_path.suffix}"
    source_rate_hz = 1000
    target_rate_hz = 250
    decimation_factor = source_rate_hz // target_rate_hz

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_rows = []
    header_lines = []
    parsing_started = False
    for line in lines:
        parts = line.strip().split("\t")
        try:
            numeric_parts = [float(p) for p in parts if p.strip()]
            data_rows.append(numeric_parts)
            parsing_started = True
        except ValueError:
            if not parsing_started:
                header_lines.append(line)

    if not data_rows:
        raise ValueError("No numeric data rows found in file.")

    array = np.array(data_rows)

    decimated_cols = []
    for col_idx in range(array.shape[1]):
        col = array[:, col_idx]
        try:
            dec = decimate(col, q=decimation_factor, ftype="fir", zero_phase=True)
        except Exception:
            dec = col[::decimation_factor]
        decimated_cols.append(dec)

    decimated = np.column_stack(decimated_cols)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line)
        for row in decimated:
            f.write("\t".join(f"{v:.6g}" for v in row) + "\n")

    return str(output_path)


class DownsampleDialog(QDialog):
    """Modal dialog for downsampling LabChart files (1000 Hz → 250 Hz).

    Opens as a modal window (blocking the main window) and stays open after
    each conversion so the user can process multiple files in sequence.
    Close the dialog when all files have been downsampled.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downsample LabChart File")
        self.setFixedSize(560, 300)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info_label = QLabel(
            "This tool downsamples LabChart .txt files from 1000 Hz to 250 Hz "
            "using anti-alias decimation.\n\n"
            "Click the button to select a file. The output is saved in the same "
            "directory with '_250_Hz' added to the filename.\n\n"
            "You can downsample multiple files before closing this window."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        select_btn = QPushButton("Select file to downsample")
        select_btn.clicked.connect(self._select_and_downsample)
        layout.addWidget(select_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        layout.addStretch(1)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        status_title = QLabel("Status")
        layout.addWidget(status_title)

        self._status_label = QTextBrowser()
        self._status_label.setReadOnly(True)
        self._status_label.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self._status_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._status_label.setFixedHeight(58)
        layout.addWidget(self._status_label)

    def _select_and_downsample(self):
        input_filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select LabChart file to downsample (1000 Hz → 250 Hz)",
            "",
            "Text files (*.txt);;All files (*.*)",
        )
        if not input_filepath:
            return
        try:
            output_path = downsample_labchart_file(input_filepath)
            self._status_label.setPlainText(f"Saved: {output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Downsampling failed:\n{e}")


class AcquisitionWorker(QObject):
    sample_ready = pyqtSignal(object)
    eof_reached = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = False
        self._paused = True
        self._batch_size = 8
        self._batch_max_latency_s = 0.03

    def run(self):
        try:
            self._running = True
            batch = []
            last_emit = time.perf_counter()
            for row in self.source.stream_samples():
                if not self._running:
                    break
                while self._paused and self._running:
                    if batch:
                        self.sample_ready.emit(np.vstack(batch))
                        batch = []
                    time.sleep(0.02)
                if not self._running:
                    break
                batch.append(np.asarray(row, dtype=float))
                now = time.perf_counter()
                if len(batch) >= self._batch_size or (now - last_emit) >= self._batch_max_latency_s:
                    self.sample_ready.emit(np.vstack(batch))
                    batch = []
                    last_emit = now
            if batch:
                self.sample_ready.emit(np.vstack(batch))
            self.eof_reached.emit()
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))
        finally:
            close_fn = getattr(self.source, "close", None)
            if callable(close_fn):
                close_fn()

    def start_stream(self):
        self._paused = False

    def pause_stream(self):
        self._paused = True

    def stop(self):
        self._running = False
        self._paused = False


class VERMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VER Analysis")
        self.resize(1280, 920)

        self.data_file = None
        self.worker = None
        self.worker_thread = None
        self.acquisition_source_mode = ACQ_CONFIG.get("source_mode", "File")
        self.session_wavelets = []
        self.session_wavelet_freqs = None
        self.session_labels = []
        self.session_ver_peaks = []
        self.session_flash_counts = []
        self._scope_panel_session = None

        self.bandpass = BandpassFilter()
        self.scope = VERScopeProcessor(self.bandpass)

        self._build_ui()
        self._build_menu()

    def _build_ui(self):
        central = QWidget(self)
        root = QVBoxLayout(central)

        controls_row = QHBoxLayout()

        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected")
        open_btn = QPushButton("Open Data File")
        open_btn.clicked.connect(lambda: self._select_data_file(initial=False))
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(open_btn)
        self.format_combo = QComboBox()
        self.format_combo.addItems(list(FILE_FORMATS.keys()))
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        file_layout.addWidget(QLabel("File format:"))
        file_layout.addWidget(self.format_combo)

        filter_group = QGroupBox("Filter Settings")
        filter_layout = QFormLayout(filter_group)
        self.low_spin = QSpinBox()
        self.low_spin.setRange(1, 120)
        self.low_spin.setValue(int(FILTER_CONFIG["lowcut_hz"]))
        self.high_spin = QSpinBox()
        self.high_spin.setRange(2, 124)
        self.high_spin.setValue(int(FILTER_CONFIG["highcut_hz"]))
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self._apply_filter_settings)
        filter_layout.addRow("Low cut (Hz)", self.low_spin)
        filter_layout.addRow("High cut (Hz)", self.high_spin)
        filter_layout.addRow(apply_filter_btn)

        run_group = QGroupBox("Controls")
        run_layout = QHBoxLayout(run_group)
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.reset_btn = QPushButton("Reset")
        self.save_btn = QPushButton("Save Report")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Real-time (1×)", "Fast (10×)", "Maximum speed"])
        self.speed_combo.setToolTip("Replay speed")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["File Replay", "Waveshare Live (CH0/CH1 @ 250 Hz)"])
        self.source_combo.currentTextChanged.connect(self._on_source_mode_changed)
        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.reset_btn.clicked.connect(self.reset_all)
        self.save_btn.clicked.connect(self.save_report)
        run_layout.addWidget(self.start_btn)
        run_layout.addWidget(self.stop_btn)
        run_layout.addWidget(self.reset_btn)
        run_layout.addWidget(self.save_btn)
        run_layout.addWidget(self.source_combo)
        run_layout.addWidget(self.speed_combo)

        self.progress_label = QLabel("Minute 0/10 | Flash 0/120")

        controls_row.addWidget(file_group)
        controls_row.addWidget(filter_group)
        controls_row.addWidget(run_group)
        controls_row.addWidget(self.progress_label)

        root.addLayout(controls_row)

        self.display = VERDisplayWidget(self)
        root.addWidget(self.display)

        self.setCentralWidget(central)
        self._set_current_format()
        self._set_current_source_mode()

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Data File", self)
        open_action.triggered.connect(lambda: self._select_data_file(initial=False))
        save_action = QAction("Save Report", self)
        save_action.triggered.connect(self.save_report)
        downsample_action = QAction("Downsample LabChart file (1000 Hz → 250 Hz)...", self)
        downsample_action.triggered.connect(self._on_downsample)
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(downsample_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        settings_menu = menubar.addMenu("Settings")
        apply_filter_action = QAction("Filter Settings", self)
        apply_filter_action.triggered.connect(self._apply_filter_settings)
        settings_menu.addAction(apply_filter_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        QMessageBox.information(
            self,
            "About VER Analysis",
            "Modular VER analysis with real-time replay, trigger-locked averaging, wavelet analysis, and report export.",
        )

    def _on_downsample(self):
        dlg = DownsampleDialog(self)
        dlg.exec()

    def _select_data_file(self, initial: bool = False):
        default_path = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(self, "Select raw data file", default_path, "Text Files (*.txt);;All Files (*)")
        if selected:
            self.data_file = selected
            self.file_label.setText(f"Selected: {Path(selected).name}")
            self.display.set_status(f"Loaded file: {Path(selected).name}")
            if not initial:
                self.reset_all()
            if self.worker is not None:
                self._restart_worker_with_file()
        elif initial:
            fallback = Path(__file__).with_name("RAW_files_combined.txt")
            if fallback.exists():
                self.data_file = str(fallback)
                self.file_label.setText(f"Selected: {fallback.name}")
                self.display.set_status(f"Loaded file: {fallback.name}")

    def _restart_worker_with_file(self):
        self._shutdown_worker()
        self._start_worker(self._get_speed_factor())

    def _on_source_mode_changed(self, mode: str):
        self.acquisition_source_mode = "Waveshare" if mode.startswith("Waveshare") else "File"
        ACQ_CONFIG["source_mode"] = self.acquisition_source_mode
        is_hardware = self.acquisition_source_mode == "Waveshare"
        self.speed_combo.setEnabled(not is_hardware)
        self.format_combo.setEnabled(not is_hardware)
        if is_hardware:
            self.display.set_status("Source: Waveshare live (CH0 EEG, CH1 trigger)")
        else:
            self.display.set_status("Source: File replay")
        if self.worker is not None:
            self._shutdown_worker()

    def _set_current_source_mode(self):
        if self.acquisition_source_mode == "Waveshare":
            self.source_combo.setCurrentText("Waveshare Live (CH0/CH1 @ 250 Hz)")
        else:
            self.source_combo.setCurrentText("File Replay")

    def _get_speed_factor(self) -> float | None:
        speed_map = {"Real-time (1×)": 1.0, "Fast (10×)": 10.0, "Maximum speed": None}
        return speed_map.get(self.speed_combo.currentText(), 1.0)

    def _build_acquisition_source(self, speed_factor: float | None):
        if self.acquisition_source_mode == "Waveshare":
            return WaveshareAcquisitionSource(
                sample_rate=ACQ_CONFIG["sample_rate"],
                waveshare_dir=str(Path(__file__).with_name(HARDWARE_CONFIG["waveshare_dir"])),
                eeg_channel=HARDWARE_CONFIG["eeg_channel"],
                trigger_channel=HARDWARE_CONFIG["trigger_channel"],
                trigger_threshold=HARDWARE_CONFIG["trigger_threshold"],
                adc_gain=HARDWARE_CONFIG["adc_gain"],
                adc_rate=HARDWARE_CONFIG["adc_rate"],
                voltage_ref=HARDWARE_CONFIG["voltage_ref"],
            )

        if not self.data_file:
            QMessageBox.warning(self, "No file", "Please select a data file first.")
            return None
        return FileAcquisitionSimulator(
            self.data_file,
            sample_rate=ACQ_CONFIG["sample_rate"],
            speed_factor=speed_factor,
        )

    def _start_worker(self, speed_factor: float | None = 1.0):
        source = self._build_acquisition_source(speed_factor)
        if source is None:
            return

        self.worker_thread = QThread(self)
        self.worker = AcquisitionWorker(source)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.sample_ready.connect(self._handle_sample)
        self.worker.eof_reached.connect(self._handle_eof)
        self.worker.error.connect(self._handle_worker_error)
        self.worker_thread.start()

    def _shutdown_worker(self):
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        self.worker = None
        self.worker_thread = None

    def start_acquisition(self):
        if self.worker is None:
            if self.scope.flash_count > 0 or self.scope.session_averages:
                resp = QMessageBox.question(
                    self, "Reset?",
                    "There is data from a previous run. Reset before starting?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if resp == QMessageBox.StandardButton.Yes:
                    self.reset_all()
            self._start_worker(self._get_speed_factor())
        if self.worker is not None:
            self.worker.start_stream()

    def stop_acquisition(self):
        if self.worker is not None:
            self.worker.pause_stream()

    def reset_all(self):
        self.bandpass = BandpassFilter({
            "lowcut_hz": float(self.low_spin.value()),
            "highcut_hz": float(self.high_spin.value()),
            "sample_rate": ACQ_CONFIG["sample_rate"],
            "order": FILTER_CONFIG["order"],
        })
        self.scope = VERScopeProcessor(self.bandpass)
        self.session_wavelets = []
        self.session_wavelet_freqs = None
        self.session_labels = []
        self.session_ver_peaks = []
        self.session_flash_counts = []
        self._scope_panel_session = None
        self.display.reset_all()
        self._set_progress(0, 0)

    def _apply_filter_settings(self):
        low = float(self.low_spin.value())
        high = float(self.high_spin.value())
        if low >= high:
            QMessageBox.warning(self, "Invalid filter", "Low cut must be less than high cut.")
            return
        self.bandpass.redesign(low, high)
        self.display.set_status(f"Filter updated: {low:.1f}-{high:.1f} Hz")

    def _handle_sample(self, row: np.ndarray):
        samples = np.asarray(row, dtype=float)
        if samples.ndim == 1:
            self._handle_single_sample(samples)
            return
        for sample in samples:
            self._handle_single_sample(sample)

    def _handle_single_sample(self, sample: np.ndarray):
        trigger = bool(sample[0])
        eeg = float(sample[1])
        filtered = self.bandpass.process_sample(eeg)

        scope_result = self.scope.process_sample(trigger, eeg)
        self.display.update_scroll_panel(eeg, filtered, scope_result["trigger_detected"])

        current_session = scope_result["session_number"]
        self._set_progress(current_session, scope_result["flash_count"])

        if scope_result["epoch_complete"]:
            if self._scope_panel_session != current_session:
                self.display.clear_scope_panel()
                self._scope_panel_session = current_session
            self.display.update_scope_panel(
                self.scope.epoch_time_ms,
                scope_result["completed_epoch"],
                scope_result["running_average"],
                scope_result["flash_count"],
                current_session,
            )

        if scope_result["session_complete"]:
            session_avg = scope_result["completed_session_average"]
            session_num = scope_result["completed_session_number"]
            self._record_session(session_avg, session_num)
            self.display.clear_scope_panel()
            self._scope_panel_session = None

            if not self.scope.has_completed_all_sessions():
                self._set_progress(min(EPOCH_CONFIG["num_sessions"], session_num + 1), 0)

            if self.scope.has_completed_all_sessions():
                self.stop_acquisition()
                self.save_report()

    def _set_progress(self, session_number: int, flash_count: int):
        self.progress_label.setText(
            f"Minute {session_number}/{EPOCH_CONFIG['num_sessions']} | Flash {flash_count}/{EPOCH_CONFIG['flashes_per_session']}"
        )

    def _handle_eof(self):
        self.stop_acquisition()
        partial_session = self.scope.save_partial_session(EPOCH_CONFIG["flashes_per_session"] // 2)
        if partial_session is not None:
            self._record_session(
                partial_session["session_average"],
                partial_session["session_number"],
                flash_count=partial_session["flash_count"],
            )
        if self.scope.session_averages:
            resp = QMessageBox.question(
                self,
                "End of file",
                "Reached end of file. Save report for collected minutes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if resp == QMessageBox.StandardButton.Yes:
                self.save_report()
        self.display.set_status("End of file reached")

    def _handle_worker_error(self, message: str):
        QMessageBox.critical(self, "Acquisition error", message)

    def _record_session(self, session_avg: np.ndarray, session_num: int, flash_count: int | None = None):
        power, freqs = compute_wavelet_scalogram(session_avg)
        self.session_wavelets.append(power)
        self.session_wavelet_freqs = freqs
        peak_idx = np.unravel_index(np.argmax(power), power.shape)
        peak_freq = float(freqs[peak_idx[0]])
        peak_latency_ms = float(self.scope.epoch_time_ms[peak_idx[1]])
        peak_power = float(power[peak_idx])

        ver_peaks = detect_ver_peaks(session_avg, self.scope.epoch_time_ms)
        self.session_ver_peaks.append(ver_peaks)
        self.session_flash_counts.append(flash_count)

        label = f"Minute {session_num}"
        if flash_count is not None and flash_count != EPOCH_CONFIG["flashes_per_session"]:
            label = f"{label} ({flash_count}/{EPOCH_CONFIG['flashes_per_session']})"
        self.session_labels.append(label)

        self.display.update_wavelet_panel(power, freqs, self.scope.epoch_time_ms, session_num)
        self.display.update_wavelet_stats(peak_freq, peak_latency_ms, peak_power, session_num, ver_peaks=ver_peaks)
        self.display.add_session_average(
            self.scope.epoch_time_ms,
            session_avg,
            session_num,
            session_label=label,
            ver_peaks=ver_peaks,
        )

    def _on_format_changed(self, format_name: str):
        FILE_CONFIG.update(FILE_FORMATS[format_name])
        self.display.set_status(f"File format: {format_name}")
        self.reset_all()

    def _set_current_format(self):
        current = {key: FILE_CONFIG.get(key) for key in ("delimiter", "trigger_column", "eeg_column", "skip_header", "trigger_mode", "trigger_threshold")}
        for format_name, cfg in FILE_FORMATS.items():
            if all(current.get(key) == cfg.get(key) for key in cfg):
                self.format_combo.setCurrentText(format_name)
                return
        default_name = next(iter(FILE_FORMATS))
        self.format_combo.setCurrentText(default_name)

    def save_report(self):
        report_input = self.data_file
        if report_input is None:
            report_input = str(Path.cwd() / "waveshare_live_report.txt")
        result = save_ver_report(
            report_input,
            self.scope.session_averages,
            self.scope.epoch_time_ms,
            session_wavelets=self.session_wavelets if self.session_wavelets else None,
            session_wavelet_freqs=self.session_wavelet_freqs,
            session_labels=self.session_labels if self.session_labels else None,
            session_ver_peaks=self.session_ver_peaks if self.session_ver_peaks else None,
            session_flash_counts=self.session_flash_counts if self.session_flash_counts else None,
        )
        if result is None:
            QMessageBox.information(self, "No data", "No completed minutes available yet.")
            return
        report_dir = result.get("report_dir", str(Path(result["png"]).parent))
        png_name = Path(result["png"]).name
        pdf_name = Path(result["pdf"]).name if "pdf" in result else "—"
        summary_csv_name = Path(result["summary_csv"]).name if "summary_csv" in result else "—"
        waveforms_csv_name = Path(result["waveforms_csv"]).name if "waveforms_csv" in result else "—"
        QMessageBox.information(
            self,
            "Report saved",
            f"Reports saved to:\n{report_dir}\n\nPNG: {png_name}\nPDF: {pdf_name}\nSummary CSV: {summary_csv_name}\nWaveforms CSV: {waveforms_csv_name}",
        )

    def closeEvent(self, event):
        self._shutdown_worker()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = VERMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
