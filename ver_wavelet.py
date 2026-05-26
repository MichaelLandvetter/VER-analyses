"""Wavelet analysis utilities for VER averaged epochs."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pywt

from ver_config import ACQ_CONFIG, WAVELET_CONFIG


def compute_wavelet_scalogram(epoch: np.ndarray, sample_rate: Optional[float] = None, config: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    cfg = dict(WAVELET_CONFIG)
    if config:
        cfg.update(config)

    fs = float(sample_rate if sample_rate is not None else ACQ_CONFIG["sample_rate"])
    data = np.asarray(epoch, dtype=float)
    freqs = np.linspace(cfg["freq_min"], cfg["freq_max"], cfg["num_freqs"])

    central_freq = pywt.central_frequency(cfg["wavelet"])
    scales = central_freq * fs / freqs

    coeffs, _ = pywt.cwt(data, scales, cfg["wavelet"], sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2

    # Normalise power to 0–1 so values are comparable across file types
    # (SD-card amplitudes ~10 units vs LabChart ~0.02 units)
    power_max = np.max(power)
    if power_max > 0:
        power = power / power_max

    return power, freqs


if __name__ == "__main__":
    t = np.arange(0, 0.5, 1.0 / ACQ_CONFIG["sample_rate"])
    synthetic = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    pwr, f = compute_wavelet_scalogram(synthetic)
    print("Synthetic test scalogram shape:", pwr.shape, "Frequency range:", (f.min(), f.max()))
