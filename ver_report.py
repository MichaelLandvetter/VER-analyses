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
    labels = session_labels or [f"Session {idx}" for idx in range(1, len(averages) + 1)]
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
        y_ticks.append((offset, f"S{idx}"))
        session_min = float(np.min(shifted))
        session_max = float(np.max(shifted))
        y_min = session_min if y_min is None else min(y_min, session_min)
        y_max = session_max if y_max is None else max(y_max, session_max)
    ax1.set_xlim(float(epoch_time_ms[0]), float(epoch_time_ms[-1]))
    ax1.set_title("VER Evolution — Session by Session")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Session")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.set_yticks([tick for tick, _ in y_ticks])
    ax1.set_yticklabels([label for _, label in y_ticks])
    if y_min is not None and y_max is not None:
        margin = offset_step * 0.6
        ax1.set_ylim(y_min - margin, y_max + margin)

    # Row 2: Wavelet scalograms side by side
    vmin = float(np.min(session_wavelets_arr))
    vmax = float(np.max(session_wavelets_arr))
    last_im = None
    wavelet_gs = gs[1, 0].subgridspec(1, len(session_wavelets), wspace=0.05)
    wavelet_axes = []
    for i, (power, label) in enumerate(zip(session_wavelets, labels)):
        ax = fig.add_subplot(wavelet_gs[0, i])
        last_im = ax.imshow(
            power,
            extent=[float(epoch_time_ms[0]), float(epoch_time_ms[-1]), freq_min, freq_max],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Time (ms)")
        if i == 0:
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_yticklabels([])
        wavelet_axes.append(ax)
    if last_im is not None:
        fig.colorbar(last_im, ax=wavelet_axes, label="Power", shrink=0.9)

    input_path = Path(input_file)
    out_dir = input_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem
    png_path = out_dir / f"{stem}_ver_report_{ts}.png"
    pdf_path = out_dir / f"{stem}_ver_report_{ts}.pdf"

    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}
