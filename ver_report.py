"""Generate final summary VER report figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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

    fig1 = _build_figures_page(averages, epoch_time_ms, session_wavelets, session_wavelets_arr, session_wavelet_freqs, freq_min, freq_max, labels)

    fig2 = _build_stats_table_page(session_wavelets, session_wavelet_freqs, epoch_time_ms, labels)

    input_path = Path(input_file)
    out_dir = input_path.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_path.stem
    png_path = out_dir / f"{stem}_ver_report_{ts}.png"
    pdf_path = out_dir / f"{stem}_ver_report_{ts}.pdf"

    fig1.savefig(png_path, dpi=150)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def _build_figures_page(
    averages: np.ndarray,
    epoch_time_ms: np.ndarray,
    session_wavelets: List[np.ndarray],
    session_wavelets_arr: np.ndarray,
    session_wavelet_freqs: np.ndarray,
    freq_min: float,
    freq_max: float,
    labels: List[str],
) -> plt.Figure:
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
            color=MINUTE_COLORS[idx % len(MINUTE_COLORS)],
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
        ax2.axvline(idx * epoch_width, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_title("Wavelet Scalograms by Minute")
    ax2.set_xlabel("Minute")
    ax2.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax2, label="Power", shrink=0.9)

    return fig


def _build_stats_table_page(
    session_wavelets: List[np.ndarray],
    session_wavelet_freqs: np.ndarray,
    epoch_time_ms: np.ndarray,
    labels: List[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, max(4, 1 + 0.5 * len(session_wavelets))), facecolor="white")
    ax.axis("off")
    ax.set_title("VER Analysis — Peak Statistics", fontsize=14, fontweight="bold", pad=20)

    col_labels = ["Minute", "Label", "Peak Freq (Hz)", "Peak Latency (ms)", "Peak Power"]
    rows = []
    for idx, wavelet in enumerate(session_wavelets):
        peak_idx = np.unravel_index(np.argmax(wavelet), wavelet.shape)
        peak_freq = float(session_wavelet_freqs[peak_idx[0]])
        peak_latency_ms = float(epoch_time_ms[peak_idx[1]])
        peak_power = float(wavelet[peak_idx])
        rows.append([
            str(idx + 1),
            labels[idx] if idx < len(labels) else f"Minute {idx + 1}",
            f"{peak_freq:.1f}",
            f"{peak_latency_ms:.0f}",
            f"{peak_power:.4f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("lightgray")
        if row_idx == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#E9EFF7")
        else:
            cell.set_facecolor("white")

    return fig
