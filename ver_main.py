"""Main application entry point for modular VER analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ver_acquisition import FileAcquisitionSimulator
from ver_config import ACQ_CONFIG, EPOCH_CONFIG, FILE_CONFIG, FILTER_CONFIG
from ver_display import VERDisplayWidget
from ver_filter import BandpassFilter
from ver_report import save_ver_report
from ver_scope import VERScopeProcessor
from ver_wavelet import compute_wavelet_scalogram


class AcquisitionWorker(QObject):
    sample_ready = pyqtSignal(object)
    eof_reached = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file_path: str, sample_rate: float, simulate_realtime: bool):
        super().__init__()
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.simulate_realtime = simulate_realtime
        self._running = False
        self._paused = True

    def run(self):
        try:
            simulator = FileAcquisitionSimulator(
                self.file_path,
                sample_rate=self.sample_rate,
                simulate_realtime=self.simulate_realtime,
            )
            self._running = True
            for row in simulator.stream_samples():
                if not self._running:
                    break
                while self._paused and self._running:
                    time.sleep(0.02)
                if not self._running:
                    break
                self.sample_ready.emit(np.asarray(row, dtype=float))
            self.eof_reached.emit()
        except Exception as exc:  # pragma: no cover
            self.error.emit(str(exc))

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
        self.session_wavelets = []
        self.session_wavelet_freqs = None
        self.session_labels = []
        self._scope_panel_session = None

        self.bandpass = BandpassFilter()
        self.scope = VERScopeProcessor(self.bandpass)

        self._build_ui()
        self._build_menu()
        self._select_data_file(initial=True)

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
        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.reset_btn.clicked.connect(self.reset_all)
        self.save_btn.clicked.connect(self.save_report)
        run_layout.addWidget(self.start_btn)
        run_layout.addWidget(self.stop_btn)
        run_layout.addWidget(self.reset_btn)
        run_layout.addWidget(self.save_btn)

        self.progress_label = QLabel("Minute 0/10 | Flash 0/120")

        controls_row.addWidget(file_group)
        controls_row.addWidget(filter_group)
        controls_row.addWidget(run_group)
        controls_row.addWidget(self.progress_label)

        root.addLayout(controls_row)

        self.display = VERDisplayWidget(self)
        root.addWidget(self.display)

        self.setCentralWidget(central)

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Data File", self)
        open_action.triggered.connect(lambda: self._select_data_file(initial=False))
        save_action = QAction("Save Report", self)
        save_action.triggered.connect(self.save_report)
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
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

    def _select_data_file(self, initial: bool = False):
        default_path = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(self, "Select raw data file", default_path, "Text Files (*.txt);;All Files (*)")
        if selected:
            self.data_file = selected
            self.file_label.setText(f"Selected: {Path(selected).name}")
            self.display.set_status(f"Loaded file: {Path(selected).name}")
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
        self._start_worker()

    def _start_worker(self):
        if not self.data_file:
            QMessageBox.warning(self, "No file", "Please select a data file first.")
            return

        self.worker_thread = QThread(self)
        self.worker = AcquisitionWorker(self.data_file, ACQ_CONFIG["sample_rate"], ACQ_CONFIG["simulate_realtime"])
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
            self._start_worker()
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
        trigger = float(row[FILE_CONFIG["trigger_column"]])
        eeg = float(row[FILE_CONFIG["eeg_column"]])
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

        label = f"Minute {session_num}"
        if flash_count is not None and flash_count != EPOCH_CONFIG["flashes_per_session"]:
            label = f"{label} ({flash_count}/{EPOCH_CONFIG['flashes_per_session']})"
        self.session_labels.append(label)

        self.display.update_wavelet_panel(power, freqs, self.scope.epoch_time_ms, session_num)
        self.display.add_session_average(self.scope.epoch_time_ms, session_avg, session_num, session_label=label)

    def save_report(self):
        if not self.data_file:
            QMessageBox.warning(self, "No file", "Please select a data file first.")
            return
        result = save_ver_report(
            self.data_file,
            self.scope.session_averages,
            self.scope.epoch_time_ms,
            session_wavelets=self.session_wavelets if self.session_wavelets else None,
            session_wavelet_freqs=self.session_wavelet_freqs,
            session_labels=self.session_labels if self.session_labels else None,
        )
        if result is None:
            QMessageBox.information(self, "No data", "No completed minutes available yet.")
            return
        QMessageBox.information(self, "Report saved", f"Saved:\n{result['png']}")

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
