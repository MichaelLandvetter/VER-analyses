"""Time-domain peak detection for VER waveforms (Peak-1, Peak-2, Peak-3)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.signal import find_peaks

from ver_config import EPOCH_CONFIG


class VERPeak(TypedDict):
    latency_ms: float   # time in ms where peak occurs
    amplitude: float    # amplitude value at the peak
    found: bool         # False if no clear peak found in window


def detect_ver_peaks(epoch_avg: np.ndarray, epoch_time_ms: np.ndarray) -> dict[str, VERPeak]:
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
    dict with keys 'Peak-1', 'Peak-2', 'Peak-3', each a VERPeak dict
    """
    empty = VERPeak(latency_ms=float('nan'), amplitude=float('nan'), found=False)

    # pre_stim_ms is stored as a positive duration; negate it to get the start of pre-stimulus time.
    baseline_mask = (epoch_time_ms >= -EPOCH_CONFIG["pre_stim_ms"]) & (epoch_time_ms < 0)
    baseline = float(np.mean(epoch_avg[baseline_mask])) if np.any(baseline_mask) else 0.0

    mask = (epoch_time_ms >= 0) & (epoch_time_ms <= 200)
    if not np.any(mask):
        return {'Peak-1': empty, 'Peak-2': empty, 'Peak-3': empty}

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

    result: dict[str, VERPeak] = {}
    peak_names = ['Peak-1', 'Peak-2', 'Peak-3']
    for i, name in enumerate(peak_names):
        if i < len(top3_sorted):
            idx = top3_sorted[i]
            result[name] = VERPeak(
                latency_ms=float(seg_times[idx]),
                amplitude=float(segment[idx]),
                found=True,
            )
        else:
            result[name] = empty

    return result
