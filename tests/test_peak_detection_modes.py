import json
from pathlib import Path

import numpy as np

from ver_analysis_flow import (
    BACK_TO_ANALYSIS,
    CANCEL_ANALYSIS,
    PROCEED_TO_VALIDATION,
    normalize_analysis_complete_action,
    should_proceed_to_human_validation,
    status_message_for_analysis_complete_action,
)
from ver_peaks import detect_ver_peaks, refresh_classifier_cfg
from ver_settings import SettingsManager

REPO_ROOT = Path(__file__).resolve().parents[1]


def _epoch_with_peaks():
    epoch_time_ms = np.arange(-100.0, 201.0, 1.0)
    epoch_avg = np.zeros_like(epoch_time_ms, dtype=float)

    def _set_peak(latency_ms, amplitude):
        idx = int(np.searchsorted(epoch_time_ms, latency_ms))
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


def test_detect_ver_peaks_uses_refreshed_cached_classifier_settings():
    epoch_avg, epoch_time_ms = _epoch_with_peaks()
    epoch_avg[epoch_time_ms < 0] = 1.0
    epoch_avg[np.searchsorted(epoch_time_ms, 45.0)] = 3.6
    epoch_avg[np.searchsorted(epoch_time_ms, 90.0)] = -4.0
    epoch_avg[np.searchsorted(epoch_time_ms, 120.0)] = -1.2
    epoch_avg[np.searchsorted(epoch_time_ms, 130.0)] = 3.1

    refresh_classifier_cfg({"snr_threshold": 3.0, "peak_detection_mode": "legacy_top3"})
    first_run = detect_ver_peaks(epoch_avg, epoch_time_ms)

    refresh_classifier_cfg(
        {
            "snr_threshold": 2.0,
            "peak_detection_mode": "dominant_opposite_neighbors",
        }
    )
    rerun = detect_ver_peaks(epoch_avg, epoch_time_ms)

    assert first_run["Peak-3"]["latency_ms"] == 120.0
    assert first_run["Peak-1"]["above_threshold"] is False
    assert rerun["Peak-3"]["latency_ms"] == 130.0
    assert rerun["Peak-1"]["above_threshold"] is True


def test_settings_manager_backfills_peak_detection_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy_settings = SettingsManager().default_settings
    legacy_settings["CLASSIFIER_CONFIG"].pop("peak_detection_mode", None)
    (tmp_path / "user_settings.json").write_text(json.dumps(legacy_settings), encoding="utf-8")

    reloaded = SettingsManager().load_settings()

    assert reloaded["CLASSIFIER_CONFIG"]["peak_detection_mode"] == "legacy_top3"


def test_analysis_complete_action_helpers_cover_all_choices():
    assert normalize_analysis_complete_action(PROCEED_TO_VALIDATION) == PROCEED_TO_VALIDATION
    assert normalize_analysis_complete_action(BACK_TO_ANALYSIS) == BACK_TO_ANALYSIS
    assert normalize_analysis_complete_action("unexpected") == CANCEL_ANALYSIS
    assert should_proceed_to_human_validation(PROCEED_TO_VALIDATION) is True
    assert should_proceed_to_human_validation(BACK_TO_ANALYSIS) is False
    assert should_proceed_to_human_validation(None) is False


def test_analysis_complete_status_messages_match_choice():
    assert status_message_for_analysis_complete_action(
        PROCEED_TO_VALIDATION,
        has_session_averages=True,
    ) == "End of file reached"
    assert status_message_for_analysis_complete_action(
        BACK_TO_ANALYSIS,
        has_session_averages=True,
    ) == "Analysis complete. Adjust settings and rerun when ready."
    assert status_message_for_analysis_complete_action(
        CANCEL_ANALYSIS,
        has_session_averages=True,
    ) == "Analysis complete."
    assert status_message_for_analysis_complete_action(
        None,
        has_session_averages=False,
    ) == "End of file reached"


def test_ver_main_refreshes_classifier_settings_on_save_and_start():
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")

    assert "_refresh_runtime_classifier_settings(self.cfg)" in src
    assert '_refresh_runtime_classifier_settings(self.settings_manager.settings.get("CLASSIFIER_CONFIG", {}))' in src


def test_classifier_settings_save_message_mentions_next_start_without_restart():
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")

    assert "Classifier settings saved." in src
    assert "Changes apply the next time you click Start." in src
    assert "The current graph stays unchanged until then." in src
    assert "Restart the application for all changes to take effect." not in src
