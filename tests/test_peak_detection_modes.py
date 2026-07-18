import json
from pathlib import Path

import numpy as np

from ver_peaks import detect_ver_peaks
from ver_settings import SettingsManager


def _epoch_with_peaks():
    epoch_time_ms = np.arange(-100.0, 201.0, 1.0)
    epoch_avg = np.zeros_like(epoch_time_ms, dtype=float)

    def _set_peak(latency_ms, amplitude):
        idx = int(np.where(epoch_time_ms == latency_ms)[0][0])
        epoch_avg[idx] = amplitude

    _set_peak(45.0, 4.0)
    _set_peak(90.0, -8.0)
    _set_peak(120.0, -6.0)
    _set_peak(130.0, 3.5)
    return epoch_avg, epoch_time_ms


def test_detect_ver_peaks_legacy_mode_preserves_top3_ranking():
    epoch_avg, epoch_time_ms = _epoch_with_peaks()

    result = detect_ver_peaks(
        epoch_avg,
        epoch_time_ms,
        classifier_cfg={"snr_threshold": 2.0, "peak_detection_mode": "legacy_top3"},
    )

    assert result["Peak-1"]["latency_ms"] == 45.0
    assert result["Peak-2"]["latency_ms"] == 90.0
    assert result["Peak-3"]["latency_ms"] == 120.0


def test_detect_ver_peaks_dominant_mode_uses_opposite_polarity_neighbors():
    epoch_avg, epoch_time_ms = _epoch_with_peaks()

    result = detect_ver_peaks(
        epoch_avg,
        epoch_time_ms,
        classifier_cfg={
            "snr_threshold": 2.0,
            "peak_detection_mode": "dominant_opposite_neighbors",
        },
    )

    assert result["Peak-1"]["latency_ms"] == 45.0
    assert result["Peak-2"]["latency_ms"] == 90.0
    assert result["Peak-3"]["latency_ms"] == 130.0


def test_settings_manager_backfills_peak_detection_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy_settings = SettingsManager().default_settings
    legacy_settings["CLASSIFIER_CONFIG"].pop("peak_detection_mode", None)
    (tmp_path / "user_settings.json").write_text(json.dumps(legacy_settings), encoding="utf-8")

    reloaded = SettingsManager().load_settings()

    assert reloaded["CLASSIFIER_CONFIG"]["peak_detection_mode"] == "legacy_top3"


def test_ver_main_source_includes_three_way_completion_dialog():
    src = Path("ver_main.py").read_text(encoding="utf-8")

    assert 'def prompt_analysis_complete_action(parent) -> str:' in src
    assert '"Proceed to Human Validation"' in src
    assert '"Back to Analysis"' in src
    assert '"Cancel"' in src
    assert 'next_action = prompt_analysis_complete_action(self)' in src
    assert 'if next_action == "proceed_to_validation":' in src
    assert "self.save_report()" in src
    assert 'elif next_action == "back_to_analysis":' in src
    assert 'status_message = "Analysis complete. Adjust settings and rerun when ready."' in src
