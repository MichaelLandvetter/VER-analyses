"""Generate final summary VER report figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ver_wavelet import compute_wavelet_scalogram

MINUTE_COLORS = [
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
    fig = plt.figure(figsize=(18, 10), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 1)

    # Row 1: Sequential VER averages on a minute axis
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")
    epoch_width = float(epoch_time_ms[-1] - epoch_time_ms[0])
    epoch_start = float(epoch_time_ms[0])
    for idx, avg in enumerate(averages):
        x_offset = idx * epoch_width
        x_plot = epoch_time_ms - epoch_start + x_offset
        ax1.plot(
            x_plot,
            avg,
            linewidth=1.2,
            alpha=0.9,
            color=MINUTE_COLORS[idx % len(_MINUTE_COLORS)],
        )
        if idx > 0:
            ax1.axvline(x=x_offset, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    total_width = epoch_width * len(averages)
    tick_positions = [(idx + 0.5) * epoch_width for idx in range(len(averages))]
    tick_labels = [f"M{idx + 1}" for idx in range(len(averages))]
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlim(0.0, total_width)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_title("VER Evolution — Minute by Minute")
    ax1.set_xlabel("Minute")
    ax1.set_ylabel("Amplitude (µV)")

    # Row 2: Wavelet scalograms sequentially in one wide panel
    vmin = float(np.min(session_wavelets_arr))
    vmax = float(np.max(session_wavelets_arr))
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("white")
    combined_wavelets = np.hstack(session_wavelets)
    segment_width = epoch_width
    im = ax2.imshow(
        combined_wavelets,
        extent=[0.0, total_width, freq_min, freq_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    for idx in range(1, len(labels)):
        ax2.axvline(idx * segment_width, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
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
