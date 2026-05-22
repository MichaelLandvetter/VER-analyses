"""Bandpass filter utilities for real-time and offline VER processing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from ver_config import FILTER_CONFIG


class BandpassFilter:
    """Butterworth bandpass filter with both causal and zero-phase paths."""

    def __init__(self, config: Optional[dict] = None):
        self.config = dict(FILTER_CONFIG)
        if config:
            self.config.update(config)
        self._dc_mean = 0.0
        self._dc_count = 0
        self.redesign(self.config["lowcut_hz"], self.config["highcut_hz"])

    def redesign(self, lowcut_hz: float, highcut_hz: float, order: Optional[int] = None) -> None:
        self.config["lowcut_hz"] = float(lowcut_hz)
        self.config["highcut_hz"] = float(highcut_hz)
        if order is not None:
            self.config["order"] = int(order)

        sample_rate = float(self.config["sample_rate"])
        nyquist = sample_rate / 2.0
        low = self.config["lowcut_hz"] / nyquist
        high = self.config["highcut_hz"] / nyquist
        if not (0.0 < low < high < 1.0):
            raise ValueError("Invalid bandpass bounds. Require 0 < lowcut < highcut < Nyquist")

        self.sos = butter(self.config["order"], [low, high], btype="band", output="sos")
        self._zi = sosfilt_zi(self.sos)
        self._dc_mean = 0.0
        self._dc_count = 0

    def process_sample(self, sample: float) -> float:
        """Causal real-time filter path preserving state between samples."""
        self._dc_count += 1
        self._dc_mean += (float(sample) - self._dc_mean) / self._dc_count
        centered = float(sample) - self._dc_mean
        y, self._zi = sosfilt(self.sos, np.array([centered], dtype=float), zi=self._zi)
        return float(y[0])

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            return samples
        centered = samples - np.mean(samples)
        y, self._zi = sosfilt(self.sos, centered, zi=self._zi)
        return y

    def apply_zero_phase(self, samples: np.ndarray, baseline_mean: Optional[float] = None) -> np.ndarray:
        """Zero-phase offline filtering path for extracted epochs."""
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            return samples
        dc = np.mean(samples) if baseline_mean is None else float(baseline_mean)
        centered = samples - dc
        try:
            return sosfiltfilt(self.sos, centered)
        except ValueError:
            return sosfilt(self.sos, centered)
