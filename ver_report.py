"""Generate final summary VER report figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from ver_wavelet import compute_wavelet_scalogram

MINUTE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _get_report_dir(data_file_path: str) -> Path:
    """Create and return the report output directory."""
    data_path = Path(data_file_path)
    report_dir = data_path.parent / "Reports" / data_path.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def save_ver_report(
    input_file: str,
    session_averages: List[np.ndarray],
    epoch_time_ms: np.ndarray,
    session_wavelets: Optional[List[np.ndarray]] = None,
    session_wavelet_freqs: Optional[np.ndarray] = None,
    session_labels: Optional[List[str]] = None,
    session_ver_peaks: Optional[List[dict]] = None,
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

    freq_min = float(session_wavelet_freqs[0])
    freq_max = float(session_wavelet_freqs[-1])
    labels = session_labels or [f"Minute {idx}" for idx in range(1, len(averages) + 1)]

    fig1 = _build_figures_page(
        averages,
        epoch_time_ms,
        session_wavelets,
        session_wavelet_freqs,
        freq_min,
        freq_max,
        labels,
        session_ver_peaks=session_ver_peaks,
    )

    fig2 = _build_stats_table_page(session_wavelets, session_wavelet_freqs, epoch_time_ms, labels, session_ver_peaks)

    report_dir = _get_report_dir(input_file)
    stem = Path(input_file).stem
    png_path = report_dir / f"{stem}.png"
    pdf_path = report_dir / f"{stem}.pdf"

    fig1.savefig(png_path, dpi=150)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)

    return {"png": str(png_path), "pdf": str(pdf_path), "report_dir": str(report_dir)}


def _build_figures_page(
    averages: np.ndarray,
    epoch_time_ms: np.ndarray,
    session_wavelets: List[np.ndarray],
    session_wavelet_freqs: np.ndarray,
    freq_min: float,
    freq_max: float,
    labels: List[str],
    session_ver_peaks: Optional[List[dict]] = None,
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
        ver_peaks = session_ver_peaks[idx] if session_ver_peaks and idx < len(session_ver_peaks) else None
        if ver_peaks:
            for peak_name, color in [("Peak-1", "blue"), ("Peak-2", "red"), ("Peak-3", "green")]:
                peak = ver_peaks.get(peak_name)
                if peak and peak.get("found"):
                    marker_x = float(peak["latency_ms"]) - epoch_start + x_offset
                    marker_y = float(peak["amplitude"])
                    if math.isnan(marker_x) or math.isnan(marker_y):
                        continue
                    marker = '^' if peak["amplitude"] >= 0 else 'v'
                    ax1.plot(
                        marker_x,
                        marker_y,
                        marker=marker,
                        color=color,
                        markersize=6,
                        linestyle="None",
                    )
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
    ax1.legend(
        handles=[
            Line2D([0], [0], marker='D', color="blue", linestyle="None", markersize=6, label="Peak-1"),
            Line2D([0], [0], marker='D', color="red", linestyle="None", markersize=6, label="Peak-2"),
            Line2D([0], [0], marker='D', color="green", linestyle="None", markersize=6, label="Peak-3"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
    )

    # Row 2: Wavelet scalograms sequentially in one wide panel
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("white")
    normalised_wavelets = []
    for wavelet in session_wavelets:
        display_wavelet = np.asarray(wavelet, dtype=float)
        wavelet_max = float(np.max(display_wavelet)) if display_wavelet.size else 0.0
        if wavelet_max > 0:
            display_wavelet = display_wavelet / wavelet_max
        normalised_wavelets.append(display_wavelet)
    combined_wavelets = np.hstack(normalised_wavelets)
    vmin = float(np.min(combined_wavelets))
    vmax = float(np.max(combined_wavelets))
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
    session_ver_peaks: Optional[List[dict]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, max(4, 1 + 0.5 * len(session_wavelets))), facecolor="white")
    ax.axis("off")
    ax.set_title("VER Analysis — Peak Statistics", fontsize=14, fontweight="bold", pad=20)

    col_labels = [
        "Minute", "Label",
        "Peak-1 Latency (ms)", "Peak-1 Amp",
        "Peak-2 Latency (ms)", "Peak-2 Amp",
        "Peak-3 Latency (ms)", "Peak-3 Amp",
        "Peak Freq (Hz)", "Peak Latency (ms)", "Peak Power",
    ]
    rows = []
    for idx, wavelet in enumerate(session_wavelets):
        peak_idx = np.unravel_index(np.argmax(wavelet), wavelet.shape)
        peak_freq = float(session_wavelet_freqs[peak_idx[0]])
        peak_latency_ms = float(epoch_time_ms[peak_idx[1]])
        peak_power = float(wavelet[peak_idx])

        def _fmt_peak(peaks, key):
            if peaks is None:
                return "\u2014", "\u2014"
            p = peaks.get(key, {})
            if p.get('found'):
                return f"{p['latency_ms']:.0f}", f"{p['amplitude']:.4f}"
            return "\u2014", "\u2014"

        ver_peaks = session_ver_peaks[idx] if session_ver_peaks and idx < len(session_ver_peaks) else None
        p1_lat, p1_amp = _fmt_peak(ver_peaks, 'Peak-1')
        p2_lat, p2_amp = _fmt_peak(ver_peaks, 'Peak-2')
        p3_lat, p3_amp = _fmt_peak(ver_peaks, 'Peak-3')

        rows.append([
            str(idx + 1),
            labels[idx] if idx < len(labels) else f"Minute {idx + 1}",
            p1_lat, p1_amp,
            p2_lat, p2_amp,
            p3_lat, p3_amp,
            f"{peak_freq:.1f}",
            f"{peak_latency_ms:.0f}",
            f"{peak_power:.3e}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

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
