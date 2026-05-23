"""Generate final summary VER report figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ver_config import EPOCH_CONFIG
from ver_wavelet import compute_wavelet_scalogram

_SESSION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def save_ver_report(
    input_file: str,
    session_averages: List[np.ndarray],
    epoch_time_ms: np.ndarray,
    session_wavelets: Optional[List[np.ndarray]] = None,
    session_wavelet_freqs: Optional[np.ndarray] = None,
    session_labels: Optional[List[str]] = None,
) -> Optional[dict]:
    if not session_averages:
        return None

    averages = np.asarray(session_averages, dtype=float)

    if session_wavelets is None or session_wavelet_freqs is None:
        computed_wavelets = []
        freqs = None
        for avg in averages:
            power, freqs = compute_wavelet_scalogram(avg)
            computed_wavelets.append(power)
        session_wavelets = computed_wavelets
        session_wavelet_freqs = freqs

    session_wavelets_arr = np.asarray(session_wavelets)
    freq_min = float(session_wavelet_freqs[0])
    freq_max = float(session_wavelet_freqs[-1])
    labels = session_labels or [f"Minute {idx}" for idx in range(1, len(averages) + 1)]
    offset_step = max(15.0, 2.5 * float(np.ptp(averages[0]))) if len(averages) else 15.0

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1)

    # Row 1: Stacked VER averages on the standard epoch time axis
    ax1 = fig.add_subplot(gs[0, 0])
    y_ticks = []
    y_min = None
    y_max = None
    for idx, (avg, label) in enumerate(zip(averages, labels), start=1):
        offset = (EPOCH_CONFIG["num_sessions"] - idx) * offset_step
        color = _SESSION_COLORS[(idx - 1) % len(_SESSION_COLORS)]
        shifted = avg + offset
        ax1.axhline(y=offset, color="grey", linestyle="--", linewidth=0.8, alpha=0.8)
        ax1.plot(
            epoch_time_ms,
            shifted,
            linewidth=1.8,
            label=label,
            color=color,
        )
        ax1.text(float(epoch_time_ms[0]) - 10.0, offset, label, color=color, ha="right", va="center", fontsize=8)
        y_ticks.append((offset, f"M{idx}"))
        session_min = float(np.min(shifted))
        session_max = float(np.max(shifted))
        y_min = session_min if y_min is None else min(y_min, session_min)
        y_max = session_max if y_max is None else max(y_max, session_max)
    ax1.set_xlim(-EPOCH_CONFIG["pre_stim_ms"], EPOCH_CONFIG["post_stim_ms"])
    ax1.set_title("VER Evolution — Minute by Minute")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Minute")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.set_yticks([tick for tick, _ in y_ticks])
    ax1.set_yticklabels([label for _, label in y_ticks])
    if y_min is not None and y_max is not None:
        margin = offset_step * 0.6
        ax1.set_ylim(y_min - margin, y_max + margin)

    # Row 2: Wavelet scalograms sequentially in one wide panel
    vmin = float(np.min(session_wavelets_arr))
    vmax = float(np.max(session_wavelets_arr))
    ax2 = fig.add_subplot(gs[1, 0])
    combined_wavelets = np.hstack(session_wavelets)
    segment_width = float(epoch_time_ms[-1] - epoch_time_ms[0])
    total_width = segment_width * len(session_wavelets)
    im = ax2.imshow(
        combined_wavelets,
        extent=[0.0, total_width, freq_min, freq_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    tick_positions = [(idx + 0.5) * segment_width for idx in range(len(labels))]
    tick_labels = [f"M{idx + 1}" for idx in range(len(labels))]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    for idx in range(1, len(labels)):
        ax2.axvline(idx * segment_width, color="white", linestyle="--", linewidth=0.6, alpha=0.7)
    ax2.set_title("Wavelet Scalograms by Minute")
    ax2.set_xlabel("Minute")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax2, label="Power", shrink=0.9)

    input_path = Path(input_file)
    out_dir = input_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem
    png_path = out_dir / f"{stem}_ver_report_{ts}.png"

    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {"png": str(png_path)}
