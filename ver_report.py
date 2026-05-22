"""Generate final summary VER report figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ver_config import WAVELET_CONFIG
from ver_wavelet import compute_wavelet_scalogram


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
    grand_avg = np.mean(averages, axis=0)
    grand_sd = np.std(averages, axis=0)

    if session_wavelets is None or session_wavelet_freqs is None:
        session_wavelets = []
        for avg in averages:
            power, freqs = compute_wavelet_scalogram(avg)
            session_wavelets.append(power)
        session_wavelet_freqs = freqs

    session_wavelets = np.asarray(session_wavelets)
    grand_wavelet = np.mean(session_wavelets, axis=0)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    for idx, avg in enumerate(averages, start=1):
        ax1.plot(epoch_time_ms, avg, linewidth=1.2, label=f"Session {idx}")
    ax1.set_title("Session VER Averages")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epoch_time_ms, grand_avg, color="black", linewidth=2, label="Grand Average")
    ax2.fill_between(epoch_time_ms, grand_avg - grand_sd, grand_avg + grand_sd, color="tab:blue", alpha=0.25, label="±1 SD")
    ax2.set_title("Grand Average ±1 SD")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()

    wavelet_grid = gs[1, 0].subgridspec(2, 5)
    vmin = float(np.min(session_wavelets))
    vmax = float(np.max(session_wavelets))
    for i in range(10):
        ax = fig.add_subplot(wavelet_grid[i // 5, i % 5])
        if i < len(session_wavelets):
            im = ax.imshow(
                session_wavelets[i],
                extent=[epoch_time_ms[0], epoch_time_ms[-1], session_wavelet_freqs[0], session_wavelet_freqs[-1]],
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"S{i+1}", fontsize=9)
        else:
            ax.axis("off")
        if i % 5 == 0:
            ax.set_ylabel("Hz")
        if i // 5 == 1:
            ax.set_xlabel("ms")

    ax4 = fig.add_subplot(gs[1, 1])
    grand_im = ax4.imshow(
        grand_wavelet,
        extent=[epoch_time_ms[0], epoch_time_ms[-1], WAVELET_CONFIG["freq_min"], WAVELET_CONFIG["freq_max"]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax4.set_title("Grand Average Wavelet")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Frequency (Hz)")
    fig.colorbar(grand_im, ax=ax4, label="Power")

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
