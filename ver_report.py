"""Generate final summary VER report figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

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
    epoch_time_s = epoch_time_ms / 1000.0
    freq_min = float(session_wavelet_freqs[0])
    freq_max = float(session_wavelet_freqs[-1])

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1)

    # Row 1: VER averages on 10-minute time axis
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, avg in enumerate(averages, start=1):
        offset_s = (idx - 1) * 60.0
        ax1.plot(
            epoch_time_s + offset_s,
            avg,
            linewidth=1.2,
            label=f"Session {idx}",
            color=_SESSION_COLORS[(idx - 1) % len(_SESSION_COLORS)],
        )
    for boundary in range(60, 600, 60):
        ax1.axvline(x=boundary, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_xlim(0, 600)
    ax1.set_title("VER Averages — 10 Minute Recording")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.legend(fontsize=8, ncol=5)

    # Row 2: Wavelet scalograms on 10-minute time axis
    ax2 = fig.add_subplot(gs[1, 0])
    vmin = float(np.min(session_wavelets_arr))
    vmax = float(np.max(session_wavelets_arr))
    last_im = None
    for i, power in enumerate(session_wavelets):
        offset = i * 60.0
        last_im = ax2.imshow(
            power,
            extent=[offset, offset + 60, freq_min, freq_max],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
    for boundary in range(60, 600, 60):
        ax2.axvline(x=boundary, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xlim(0, 600)
    ax2.set_title("Wavelet Scalograms — 10 Minute Recording")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    if last_im is not None:
        fig.colorbar(last_im, ax=ax2, label="Power")

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
