"""Time-domain peak detection for VER waveforms (N75, P100, N135)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class VERPeak(TypedDict):
    latency_ms: float   # time in ms where peak occurs
    amplitude: float    # amplitude value at the peak
    found: bool         # False if no clear peak found in window


def detect_ver_peaks(epoch_avg: np.ndarray, epoch_time_ms: np.ndarray) -> dict[str, VERPeak]:
    """
    Detect N75, P100, N135 peaks in the averaged VER waveform.

    Parameters
    ----------
    epoch_avg : np.ndarray
        Averaged epoch waveform (same length as epoch_time_ms)
    epoch_time_ms : np.ndarray
        Time axis in milliseconds (negative values = pre-stimulus)

    Returns
    -------
    dict with keys 'N75', 'P100', 'N135', each a VERPeak dict
    """
    def _find_peak(data, times, t_start, t_end, polarity):
        """Find peak of given polarity within time window."""
        mask = (times >= t_start) & (times <= t_end)
        if not np.any(mask):
            return VERPeak(latency_ms=float('nan'), amplitude=float('nan'), found=False)
        segment = data[mask]
        seg_times = times[mask]
        if polarity == 'negative':
            idx = np.argmin(segment)
        else:
            idx = np.argmax(segment)
        return VERPeak(
            latency_ms=float(seg_times[idx]),
            amplitude=float(segment[idx]),
            found=True,
        )

    return {
        'N75':  _find_peak(epoch_avg, epoch_time_ms, 50,  100, 'negative'),
        'P100': _find_peak(epoch_avg, epoch_time_ms, 80,  130, 'positive'),
        'N135': _find_peak(epoch_avg, epoch_time_ms, 110, 170, 'negative'),
    }
