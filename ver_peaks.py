"""Time-domain peak detection for VER waveforms (Peak-1, Peak-2, Peak-3)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.signal import find_peaks

import ver_config

MIN_NOISE_RMS = 1e-10


class VERPeak(TypedDict):
    latency_ms: float   # time in ms where peak occurs
    amplitude: float    # amplitude value at the peak
    found: bool         # False if no clear peak found in window
    snr: float
    above_threshold: bool


VERPeaksResult = TypedDict(
    "VERPeaksResult",
    {
        "Peak-1": VERPeak,
        "Peak-2": VERPeak,
        "Peak-3": VERPeak,
        "VER_detected": bool,
        "noise_rms": float,
    },
)


def detect_ver_peaks(epoch_avg: np.ndarray, epoch_time_ms: np.ndarray) -> VERPeaksResult:
    """
    Detect the three largest peaks (any polarity) between 0 and 200ms post-stimulus.
    Uses baseline correction and prominence/distance constraints for robustness.
    Returns Peak-1, Peak-2, Peak-3 sorted by latency (earliest first).

    Works for any species regardless of polarity convention.

    Parameters
    ----------
    epoch_avg : np.ndarray
        Averaged epoch waveform (same length as epoch_time_ms)
    epoch_time_ms : np.ndarray
        Time axis in milliseconds (negative values = pre-stimulus)

    Returns
    -------
    dict with Peak-1/2/3 plus SNR flags, VER_detected and noise_rms
    """
    def _empty_peak() -> VERPeak:
        return {
            "latency_ms": float("nan"),
            "amplitude": float("nan"),
            "found": False,
            "snr": float("nan"),
            "above_threshold": False,
        }

    # Use the configured baseline window for both baseline correction and noise estimation.
    baseline_mask = (
        (epoch_time_ms >= ver_config.BASELINE_START_MS)
        & (epoch_time_ms < ver_config.BASELINE_END_MS)
    )
    baseline = float(np.mean(epoch_avg[baseline_mask])) if np.any(baseline_mask) else 0.0

    noise_mask = (
        (epoch_time_ms >= ver_config.BASELINE_START_MS)
        & (epoch_time_ms < ver_config.BASELINE_END_MS)
    )
    baseline_segment = epoch_avg[noise_mask]
    noise_rms = float(np.sqrt(np.mean(baseline_segment ** 2))) if np.any(noise_mask) else MIN_NOISE_RMS
    noise_rms = max(noise_rms, MIN_NOISE_RMS)

    mask = (epoch_time_ms >= 0) & (epoch_time_ms <= 200)
    if not np.any(mask):
        return {
            'Peak-1': _empty_peak(),
            'Peak-2': _empty_peak(),
            'Peak-3': _empty_peak(),
            'VER_detected': False,
            'noise_rms': noise_rms,
        }

    segment = epoch_avg[mask] - baseline
    seg_times = epoch_time_ms[mask]
    signal_range = float(np.max(segment) - np.min(segment))
    # Require peaks to stand out by at least 10% of segment range.
    # 1e-10 prevents zero-prominence calls for near-flat numeric input; it has no physiological meaning.
    min_prominence = max(0.1 * signal_range, 1e-10)
    # Keep detected peaks at least ~20ms apart at 250Hz (5 samples).
    min_distance = 5

    # Find robust local maxima and minima
    pos_peaks, _ = find_peaks(segment, prominence=min_prominence, distance=min_distance)
    neg_peaks, _ = find_peaks(-segment, prominence=min_prominence, distance=min_distance)

    all_peak_indices = np.concatenate([pos_peaks, neg_peaks])

    if len(all_peak_indices) == 0:
        # No local extrema found (e.g. flat signal) — fall back to the samples with the
        # largest absolute values. These may not be strict local maxima, but they represent
        # the most prominent features in a signal with no clear peaks.
        all_peak_indices = np.argsort(np.abs(segment))[-3:]

    # Rank by absolute amplitude, take top 3
    ranked = sorted(all_peak_indices, key=lambda i: abs(segment[i]), reverse=True)
    top3 = ranked[:3]

    # Sort by latency (time order)
    top3_sorted = sorted(top3, key=lambda i: seg_times[i])

    result: VERPeaksResult = {
        "Peak-1": _empty_peak(),
        "Peak-2": _empty_peak(),
        "Peak-3": _empty_peak(),
        "VER_detected": False,
        "noise_rms": noise_rms,
    }
    peak_names = ['Peak-1', 'Peak-2', 'Peak-3']
    for i, name in enumerate(peak_names):
        if i < len(top3_sorted):
            idx = top3_sorted[i]
            result[name] = {
                "latency_ms": float(seg_times[idx]),
                "amplitude": float(segment[idx]),
                "found": True,
                "snr": float("nan"),
                "above_threshold": False,
            }
        else:
            result[name] = _empty_peak()

    for name in peak_names:
        if result[name]["found"]:
            snr = abs(result[name]["amplitude"]) / noise_rms
            result[name]["snr"] = snr
            result[name]["above_threshold"] = snr >= ver_config.SNR_THRESHOLD

    result["VER_detected"] = any(result[name]["above_threshold"] for name in peak_names)
    result["noise_rms"] = noise_rms

    return result
