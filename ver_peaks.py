"""Time-domain peak detection for VER waveforms (Peak-1, Peak-2, Peak-3)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.signal import find_peaks


class VERPeak(TypedDict):
    latency_ms: float   # time in ms where peak occurs
    amplitude: float    # amplitude value at the peak
    found: bool         # False if no clear peak found in window


def detect_ver_peaks(epoch_avg: np.ndarray, epoch_time_ms: np.ndarray) -> dict[str, VERPeak]:
    """
    Detect the three largest peaks (any polarity) between 0 and 200ms post-stimulus.
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

    mask = (epoch_time_ms >= 0) & (epoch_time_ms <= 200)
    if not np.any(mask):
        return {'Peak-1': empty, 'Peak-2': empty, 'Peak-3': empty}

    segment = epoch_avg[mask]
    seg_times = epoch_time_ms[mask]

    # Find local maxima and minima
    pos_peaks, _ = find_peaks(segment)
    neg_peaks, _ = find_peaks(-segment)

    all_peak_indices = np.concatenate([pos_peaks, neg_peaks])

    if len(all_peak_indices) == 0:
        # No local extrema found — fall back to top 3 absolute values
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
