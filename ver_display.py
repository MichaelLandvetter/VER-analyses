"""PyQtGraph display components for live VER visualization."""

from __future__ import annotations

from collections import deque
import math
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTransform
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

        self.wavelet_stats_label = QLabel("Peak: — Hz | — ms | Power: —")
        self.wavelet_stats_label.setStyleSheet("color: white; font-size: 11px;")
        layout.addWidget(self.wavelet_stats_label)

        self._init_panels()

    def _reset_sessions_panel(self) -> None:
        self.plot_sessions.clear()
        self.plot_sessions.showGrid(x=True, y=False, alpha=0.3)
        self.plot_sessions.setLabel("bottom", "Time", "ms")
        self.plot_sessions.setLabel("left", "Minute")
        self.plot_sessions.setTitle("VER Evolution — Minute by Minute")
        self.plot_sessions.setXRange(-200, 400, padding=0)
        self.plot_sessions.enableAutoRange('y', True)
        self.plot_sessions.getAxis("left").setTicks([[]])
        self._offset_step = None
        self._session_ticks: List[tuple[float, str]] = []
        self._sessions_y_min = None
        self._sessions_y_max = None

    def _init_panels(self):
        self.plot_sessions = self.graphics.addPlot(row=0, col=0, rowspan=3, title="VER Evolution — Minute by Minute")
        self.plot_sessions.getViewBox().setMouseEnabled(x=False, y=True)
        self._reset_sessions_panel()

        self.plot_raw = self.graphics.addPlot(row=0, col=1, title="Raw + Filtered EEG")
        self.plot_raw.getViewBox().setMouseEnabled(x=False, y=True)
        self.plot_raw.showGrid(x=True, y=True, alpha=0.3)
        self.plot_raw.setLabel("bottom", "Time", "s")
        self.plot_raw.setLabel("left", "Amplitude")
        self.plot_raw.enableAutoRange('x', True)
        self.plot_raw.setYRange(-1, 1, padding=0)
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen((170, 170, 170), width=1))
        self.curve_filtered = self.plot_raw.plot(pen=pg.mkPen((0, 220, 120), width=1.5))
        self.flash_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 0, 0, 180), pen=pg.mkPen(None))
        self.plot_raw.addItem(self.flash_scatter)

        self.plot_scope = self.graphics.addPlot(row=1, col=1, title="Scope View")
        self.plot_scope.getViewBox().setMouseEnabled(x=False, y=True)
        self.plot_scope.showGrid(x=True, y=True, alpha=0.3)
        self.plot_scope.setLabel("bottom", "Time", "ms")
        self.plot_scope.setLabel("left", "Amplitude")
        self.plot_scope.setXRange(-EPOCH_CONFIG["pre_stim_ms"], EPOCH_CONFIG["post_stim_ms"], padding=0)
        self.plot_scope.enableAutoRange('y', True)
        self.scope_avg_curve = self.plot_scope.plot(pen=pg.mkPen((255, 200, 0), width=3))
        self.scope_overlay_curves: List[pg.PlotCurveItem] = []

        self.plot_wavelet = self.graphics.addPlot(row=2, col=1, title="Wavelet Scalogram")
        self.plot_wavelet.getViewBox().setMouseEnabled(x=False, y=True)
        self.plot_wavelet.setLabel("bottom", "Time", "ms")
        self.plot_wavelet.setLabel("left", "Frequency", "Hz")
        self.plot_wavelet.setXRange(-EPOCH_CONFIG["pre_stim_ms"], EPOCH_CONFIG["post_stim_ms"], padding=0)
        self.plot_wavelet.setYRange(0, 50, padding=0)
        self.wavelet_image = pg.ImageItem()
        self.plot_wavelet.addItem(self.wavelet_image)

        self.graphics.ci.layout.setColumnStretchFactor(0, 1)
        self.graphics.ci.layout.setColumnStretchFactor(1, 1)

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
            if len(y_filt) > 0:
                filt_max = float(np.max(y_filt))
                filt_min = float(np.min(y_filt))
                filt_range = filt_max - filt_min
                if filt_range > 0:
                    y_dot = filt_max + 0.1 * filt_range
                elif filt_max != 0:
                    y_dot = filt_max + 0.1 * abs(filt_max)
                else:
                    y_dot = 1.0
            else:
                y_dot = 1.0
            visible = [ft for ft in self.flash_times if x[0] <= ft <= x[-1]]
            if visible:
                fx = np.array(visible, dtype=float)
                fy = np.full(len(visible), y_dot, dtype=float)
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
            f"Scope View - Flash {flash_count}/{EPOCH_CONFIG['flashes_per_session']} | Minute {session_number}/{EPOCH_CONFIG['num_sessions']}"
        )

    def clear_scope_panel(self):
        for curve in self.scope_overlay_curves:
            self.plot_scope.removeItem(curve)
        self.scope_overlay_curves = []
        self.scope_avg_curve.setData([], [])

    def update_wavelet_panel(self, power: np.ndarray, freqs: np.ndarray, epoch_time_ms: np.ndarray, session_number: int) -> None:
        display_power = np.asarray(power, dtype=float)
        power_max = float(np.max(display_power)) if display_power.size else 0.0
        if power_max > 0:
            display_power = display_power / power_max
        self.wavelet_image.setImage(display_power.T, autoLevels=True)
        x0 = float(epoch_time_ms[0])
        y0 = float(freqs[0])
        dx = float(epoch_time_ms[-1] - epoch_time_ms[0]) / max(1, power.shape[1] - 1)
        dy = float(freqs[-1] - freqs[0]) / max(1, power.shape[0] - 1)
        tr = QTransform()
        tr.translate(x0, y0)
        tr.scale(dx, dy)
        self.wavelet_image.setTransform(tr)
        self.plot_wavelet.setTitle(f"Wavelet Scalogram - Minute {session_number}")

    def update_wavelet_stats(self, peak_freq: float, peak_latency_ms: float, peak_power: float, session_number: int, ver_peaks=None) -> None:
        wavelet_text = f"M{session_number} — Wavelet peak: {peak_freq:.1f} Hz | {peak_latency_ms:.0f} ms | Power: {peak_power:.3e}"
        if ver_peaks:
            def fmt(p): return f"{p['latency_ms']:.0f} ms ({p['amplitude']:.4f})" if p.get('found') else "\u2014"
            peaks_text = (f"  |  Peak-1: {fmt(ver_peaks.get('Peak-1', {}))}  "
                          f"Peak-2: {fmt(ver_peaks.get('Peak-2', {}))}  "
                          f"Peak-3: {fmt(ver_peaks.get('Peak-3', {}))}")
            wavelet_text += peaks_text
        self.wavelet_stats_label.setText(wavelet_text)

    def _compute_offset_step(self, session_avg: np.ndarray) -> float:
        if self._offset_step is None:
            peak_to_peak = float(np.ptp(session_avg)) if len(session_avg) else 0.0
            if peak_to_peak > 0:
                self._offset_step = 2.5 * peak_to_peak
            else:
                self._offset_step = 1.0
        return self._offset_step

    def add_session_average(
        self,
        epoch_time_ms: np.ndarray,
        session_avg: np.ndarray,
        session_number: int,
        session_label: str | None = None,
        ver_peaks: Optional[dict] = None,
    ) -> None:
        offset_step = self._compute_offset_step(session_avg)
        offset = (EPOCH_CONFIG["num_sessions"] - session_number) * offset_step
        color = self.session_colors[(session_number - 1) % len(self.session_colors)]
        short_label = f"M{session_number}"
        label_text = session_label or short_label

        ref_line = pg.InfiniteLine(
            pos=offset,
            angle=0,
            pen=pg.mkPen((80, 80, 80), style=Qt.PenStyle.DashLine),
        )
        self.plot_sessions.addItem(ref_line)
        self.plot_sessions.plot(
            epoch_time_ms,
            session_avg + offset,
            pen=pg.mkPen(color, width=2),
        )
        if ver_peaks:
            peak_styles = {
                "Peak-1": {"color": "#4488FF"},
                "Peak-2": {"color": "#FF4444"},
                "Peak-3": {"color": "#44FF88"},
            }
            for peak_name, style in peak_styles.items():
                peak = ver_peaks.get(peak_name)
                if peak and peak.get("found"):
                    marker_x = float(peak["latency_ms"])
                    marker_y = float(peak["amplitude"]) + offset
                    if math.isnan(marker_x) or math.isnan(marker_y):
                        continue
                    symbol = 't' if peak["amplitude"] >= 0 else 't1'
                    if peak.get("above_threshold", False):
                        brush = pg.mkBrush(style["color"])
                        pen = pg.mkPen(None)
                    else:
                        brush = pg.mkBrush(None)
                        pen = pg.mkPen("#888888", width=1)
                    scatter = pg.ScatterPlotItem(
                        x=[marker_x],
                        y=[marker_y],
                        symbol=symbol,
                        size=10,
                        brush=brush,
                        pen=pen,
                    )
                    self.plot_sessions.addItem(scatter)
            if not ver_peaks.get("VER_detected", True):
                no_ver_text = pg.TextItem("No VER", color="#888888", anchor=(0.0, 0.5))
                no_ver_text.setPos(5.0, offset)
                self.plot_sessions.addItem(no_ver_text)
        text = pg.TextItem(label_text, color=color, anchor=(1, 0.5))
        text.setPos(float(epoch_time_ms[0]) - 5.0, offset)
        self.plot_sessions.addItem(text)

        self._session_ticks.append((offset, short_label))
        self.plot_sessions.getAxis("left").setTicks([sorted(self._session_ticks, key=lambda tick: tick[0])])

        shifted = session_avg + offset
        session_min = float(np.min(shifted))
        session_max = float(np.max(shifted))
        self._sessions_y_min = session_min if self._sessions_y_min is None else min(self._sessions_y_min, session_min)
        self._sessions_y_max = session_max if self._sessions_y_max is None else max(self._sessions_y_max, session_max)
        margin = offset_step * 0.6
        self.plot_sessions.setYRange(self._sessions_y_min - margin, self._sessions_y_max + margin, padding=0)

    def reset_all(self):
        self._offset_step = None
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
        self.wavelet_stats_label.setText("Peak: — Hz | — ms | Power: —")
        self._reset_sessions_panel()
        self.plot_raw.enableAutoRange('x', True)
        self.plot_raw.setYRange(-1, 1, padding=0)
        self.plot_scope.setXRange(-EPOCH_CONFIG["pre_stim_ms"], EPOCH_CONFIG["post_stim_ms"], padding=0)
        self.plot_scope.enableAutoRange('y', True)
        self.plot_wavelet.setXRange(-EPOCH_CONFIG["pre_stim_ms"], EPOCH_CONFIG["post_stim_ms"], padding=0)
        self.plot_wavelet.setYRange(0, 50, padding=0)
