"""PyQtGraph display components for live VER visualization."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ver_config import ACQ_CONFIG, DISPLAY_CONFIG, EPOCH_CONFIG


class VERDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_rate = ACQ_CONFIG["sample_rate"]
        self.scroll_seconds = DISPLAY_CONFIG["scroll_seconds"]
        self.max_scroll_samples = int(self.scroll_seconds * self.sample_rate)

        self.raw_buffer = deque(maxlen=self.max_scroll_samples)
        self.filtered_buffer = deque(maxlen=self.max_scroll_samples)
        self.time_buffer = deque(maxlen=self.max_scroll_samples)
        self.flash_times = deque(maxlen=500)
        self.sample_index = 0

        self.session_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        layout = QVBoxLayout(self)
        self.status_label = QLabel("No data loaded")
        layout.addWidget(self.status_label)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)

        self._init_panels()

    def _init_panels(self):
        self.plot_raw = self.graphics.addPlot(row=0, col=0, title="Raw + Filtered EEG")
        self.plot_raw.showGrid(x=True, y=True, alpha=0.3)
        self.plot_raw.setLabel("bottom", "Time", "s")
        self.plot_raw.setLabel("left", "Amplitude")
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen((170, 170, 170), width=1))
        self.curve_filtered = self.plot_raw.plot(pen=pg.mkPen((0, 220, 120), width=1.5))
        self.flash_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 0, 0, 180), pen=pg.mkPen(None))
        self.plot_raw.addItem(self.flash_scatter)

        self.plot_scope = self.graphics.addPlot(row=1, col=0, title="Scope View")
        self.plot_scope.showGrid(x=True, y=True, alpha=0.3)
        self.plot_scope.setLabel("bottom", "Time", "ms")
        self.plot_scope.setLabel("left", "Amplitude")
        self.scope_avg_curve = self.plot_scope.plot(pen=pg.mkPen((255, 200, 0), width=3))
        self.scope_overlay_curves: List[pg.PlotCurveItem] = []

        self.plot_wavelet = self.graphics.addPlot(row=2, col=0, title="Wavelet Scalogram")
        self.plot_wavelet.setLabel("bottom", "Time", "ms")
        self.plot_wavelet.setLabel("left", "Frequency", "Hz")
        self.wavelet_image = pg.ImageItem()
        self.plot_wavelet.addItem(self.wavelet_image)

        self.plot_sessions = self.graphics.addPlot(row=3, col=0, title="Session Averages Overlay")
        self.plot_sessions.showGrid(x=True, y=True, alpha=0.3)
        self.plot_sessions.setLabel("bottom", "Time", "ms")
        self.plot_sessions.setLabel("left", "Amplitude")
        self.plot_sessions.addLegend()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def update_scroll_panel(self, raw_sample: float, filtered_sample: float, trigger_detected: bool) -> None:
        t = self.sample_index / self.sample_rate
        self.sample_index += 1

        self.time_buffer.append(t)
        self.raw_buffer.append(float(raw_sample))
        self.filtered_buffer.append(float(filtered_sample))

        x = np.asarray(self.time_buffer, dtype=float)
        y_raw = np.asarray(self.raw_buffer, dtype=float)
        y_filt = np.asarray(self.filtered_buffer, dtype=float)

        self.curve_raw.setData(x, y_raw)
        self.curve_filtered.setData(x, y_filt)

        if trigger_detected:
            self.flash_times.append(t)

        if self.flash_times:
            y_max = float(np.max(y_raw)) if len(y_raw) else 1.0
            visible = [ft for ft in self.flash_times if x[0] <= ft <= x[-1]]
            if visible:
                fx = np.array(visible, dtype=float)
                fy = np.full(len(visible), y_max, dtype=float)
                self.flash_scatter.setData(x=fx, y=fy)
            else:
                self.flash_scatter.setData(x=[], y=[])

    def update_scope_panel(self, epoch_time_ms: np.ndarray, latest_epoch: np.ndarray, running_average: np.ndarray, flash_count: int, session_number: int) -> None:
        curve = self.plot_scope.plot(epoch_time_ms, latest_epoch, pen=pg.mkPen((180, 180, 180, 120), width=1))
        self.scope_overlay_curves.append(curve)
        if len(self.scope_overlay_curves) > DISPLAY_CONFIG["max_epoch_overlays"]:
            old = self.scope_overlay_curves.pop(0)
            self.plot_scope.removeItem(old)

        self.scope_avg_curve.setData(epoch_time_ms, running_average)
        self.plot_scope.setTitle(
            f"Scope View - Flash {flash_count}/{EPOCH_CONFIG['flashes_per_session']} | Session {session_number}/{EPOCH_CONFIG['num_sessions']}"
        )

    def clear_scope_panel(self):
        for curve in self.scope_overlay_curves:
            self.plot_scope.removeItem(curve)
        self.scope_overlay_curves = []
        self.scope_avg_curve.setData([], [])

    def update_wavelet_panel(self, power: np.ndarray, freqs: np.ndarray, epoch_time_ms: np.ndarray, session_number: int) -> None:
        self.wavelet_image.setImage(power.T, autoLevels=True)
        self.wavelet_image.resetTransform()
        x0 = float(epoch_time_ms[0])
        y0 = float(freqs[0])
        dx = float(epoch_time_ms[-1] - epoch_time_ms[0]) / max(1, power.shape[1] - 1)
        dy = float(freqs[-1] - freqs[0]) / max(1, power.shape[0] - 1)
        self.wavelet_image.setPos(x0, y0)
        self.wavelet_image.scale(dx, dy)
        self.plot_wavelet.setTitle(f"Wavelet Scalogram - Session {session_number}")

    def add_session_average(self, epoch_time_ms: np.ndarray, session_avg: np.ndarray, session_number: int) -> None:
        color = self.session_colors[(session_number - 1) % len(self.session_colors)]
        self.plot_sessions.plot(
            epoch_time_ms,
            session_avg,
            pen=pg.mkPen(color, width=2),
            name=f"Session {session_number}",
        )

    def reset_all(self):
        self.raw_buffer.clear()
        self.filtered_buffer.clear()
        self.time_buffer.clear()
        self.flash_times.clear()
        self.sample_index = 0
        self.curve_raw.setData([], [])
        self.curve_filtered.setData([], [])
        self.flash_scatter.setData(x=[], y=[])
        self.clear_scope_panel()
        self.wavelet_image.setImage(np.zeros((2, 2)))
        self.plot_sessions.clear()
        self.plot_sessions.addLegend()
