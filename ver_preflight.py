"""Whole-file pre-analysis helpers for artifact exclusion threshold suggestion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ver_acquisition import FileAcquisitionSimulator
from ver_config import FILE_CONFIG, FILTER_CONFIG
from ver_filter import BandpassFilter
from ver_scope import VERScopeProcessor

MAD_TO_SIGMA = 1.4826  # Converts MAD to std-dev equivalent assuming normal distribution.
ROBUST_SIGMA_MULTIPLIER = 3.0
MIN_THRESHOLD_UV = 1e-6


@dataclass(frozen=True)
class ExclusionSuggestion:
    suggested_threshold_uv: float
    total_epochs: int
    accepted_epochs: int
    rejected_epochs: int


def _suggest_threshold_from_peaks(peak_values: np.ndarray) -> float:
    if peak_values.size == 0:
        raise ValueError("No complete epochs were detected. Verify trigger settings and file format.")

    median_peak = float(np.median(peak_values))
    mad = float(np.median(np.abs(peak_values - median_peak)))
    robust_sigma = MAD_TO_SIGMA * mad

    if robust_sigma <= 0:
        robust_sigma = float(np.std(peak_values))

    # Suggest threshold at median peak plus 3 robust standard deviations.
    suggested = median_peak + (ROBUST_SIGMA_MULTIPLIER * robust_sigma)
    if suggested < MIN_THRESHOLD_UV:
        suggested = max(median_peak, MIN_THRESHOLD_UV)
    return float(suggested)


def suggest_exclusion_from_file(
    file_path: str,
    *,
    epoch_config: dict,
    file_config: dict | None = None,
    bandpass_filter=None,
) -> ExclusionSuggestion:
    active_file_config = dict(FILE_CONFIG)
    if file_config:
        active_file_config.update(file_config)

    scope_filter = bandpass_filter or BandpassFilter(
        {
            "lowcut_hz": float(FILTER_CONFIG["lowcut_hz"]),
            "highcut_hz": float(FILTER_CONFIG["highcut_hz"]),
            "sample_rate": float(FILTER_CONFIG["sample_rate"]),
            "order": int(FILTER_CONFIG["order"]),
        }
    )

    scope = VERScopeProcessor(scope_filter, epoch_config=epoch_config)
    simulator = FileAcquisitionSimulator(file_path, speed_factor=None, file_config=active_file_config)
    epoch_peak_abs: list[float] = []

    for sample in simulator.stream_samples():
        result = scope.process_sample(bool(sample[0]), float(sample[1]))
        if result["epoch_complete"] and result["completed_epoch"] is not None:
            epoch_peak_abs.append(float(np.max(np.abs(result["completed_epoch"]))))

    peak_values = np.asarray(epoch_peak_abs, dtype=float)
    suggested_threshold = _suggest_threshold_from_peaks(peak_values)
    rejected = int(np.count_nonzero(peak_values > suggested_threshold))
    total = int(peak_values.size)
    accepted = total - rejected

    return ExclusionSuggestion(
        suggested_threshold_uv=suggested_threshold,
        total_epochs=total,
        accepted_epochs=accepted,
        rejected_epochs=rejected,
    )
