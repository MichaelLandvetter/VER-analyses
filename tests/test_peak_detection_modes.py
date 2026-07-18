import ast
import json
from pathlib import Path
from types import SimpleNamespace

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


def _load_ver_main_symbol(name: str, class_name: str | None = None, extra_globals: dict | None = None):
    tree = ast.parse((REPO_ROOT / "ver_main.py").read_text(encoding="utf-8"))
    target = None
    if class_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                target = node
                break
    else:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == name:
                        target = child
                        break
            if target is not None:
                break

    if target is None:
        raise AssertionError(f"Could not find {class_name + '.' if class_name else ''}{name} in ver_main.py")

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module, filename="ver_main.py", mode="exec"), namespace)
    return namespace[name]


def _capture_config(target: dict, key: str):
    def _capture(cfg):
        target[key] = dict(cfg)

    return _capture


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


def test_refresh_runtime_classifier_settings_updates_both_modules():
    captured = {}
    helper = _load_ver_main_symbol(
        "_refresh_runtime_classifier_settings",
        extra_globals={
            "ver_classifier": SimpleNamespace(
                refresh_classifier_cfg=_capture_config(captured, "classifier")
            ),
            "ver_peaks": SimpleNamespace(
                refresh_classifier_cfg=_capture_config(captured, "peaks")
            ),
        },
    )

    cfg = {"snr_threshold": 4.2, "peak_detection_mode": "dominant_opposite_neighbors"}
    helper(cfg)

    assert captured["classifier"] == cfg
    assert captured["peaks"] == cfg


def test_classifier_settings_save_updates_runtime_config_and_message(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    cfg = dict(manager.settings["CLASSIFIER_CONFIG"])
    cfg["snr_threshold"] = 2.0
    cfg["peak_detection_mode"] = "legacy_top3"

    class _Spin:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    class _Combo:
        def __init__(self, value):
            self._value = value

        def currentData(self):
            return self._value

    captured = {}
    save_settings = _load_ver_main_symbol(
        "save_settings",
        class_name="ClassifierSettingsTab",
        extra_globals={
            "_refresh_runtime_classifier_settings": _capture_config(captured, "cfg"),
            "QMessageBox": SimpleNamespace(
                information=lambda *args: captured.__setitem__("message", args[2])
            ),
        },
    )
    stub = SimpleNamespace(
        inputs={
            "snr_threshold": _Spin(4.5),
            "min_power": _Spin(2.5),
        },
        cfg=cfg,
        peak_detection_mode_combo=_Combo("dominant_opposite_neighbors"),
        sm=manager,
    )

    save_settings(stub)

    reloaded = SettingsManager().load_settings()
    assert manager.settings["CLASSIFIER_CONFIG"]["snr_threshold"] == 4.5
    assert manager.settings["CLASSIFIER_CONFIG"]["peak_detection_mode"] == "dominant_opposite_neighbors"
    assert manager.settings["CLASSIFIER_CONFIG"]["min_power"] == 2.5e-7
    assert reloaded["CLASSIFIER_CONFIG"]["snr_threshold"] == 4.5
    assert reloaded["CLASSIFIER_CONFIG"]["peak_detection_mode"] == "dominant_opposite_neighbors"
    assert captured["cfg"]["snr_threshold"] == 4.5
    assert captured["cfg"]["peak_detection_mode"] == "dominant_opposite_neighbors"
    assert captured["message"] == (
        "Classifier settings saved.\n\n"
        "Changes apply the next time you click Start. The current graph stays unchanged until then."
    )


def test_start_acquisition_refreshes_live_classifier_settings_before_rerun():
    captured = {}
    start_acquisition = _load_ver_main_symbol(
        "start_acquisition",
        class_name="VERMainWindow",
        extra_globals={
            "_refresh_runtime_classifier_settings": _capture_config(captured, "cfg"),
            "QMessageBox": SimpleNamespace(
                StandardButton=SimpleNamespace(Yes=1, No=2),
                question=lambda *args, **kwargs: 1,
            ),
        },
    )

    class _DummyButton:
        def setText(self, _text):
            return None

    class _DummyDisplay:
        def set_status(self, _text):
            return None

    class _DummyTabs:
        def setCurrentIndex(self, _index):
            return None

    class _DummyCombo:
        def currentText(self):
            return "Filtered"

    class _DummyBandpass:
        def set_scope_mode(self, _mode):
            return None

    class _DummyWorker:
        def start_stream(self):
            captured["started"] = True

    scope = SimpleNamespace(
        flash_count=0,
        session_averages=[],
        bandpass_filter=_DummyBandpass(),
    )
    stub = SimpleNamespace(
        _get_speed_factor=lambda: 1.0,
        _sync_artifact_settings_from_ui=lambda: None,
        settings_manager=SimpleNamespace(
            settings={
                "CLASSIFIER_CONFIG": {
                    "snr_threshold": 5.0,
                    "peak_detection_mode": "dominant_opposite_neighbors",
                }
            }
        ),
        scope_filter_combo=_DummyCombo(),
        scope=scope,
        worker=None,
        _start_worker=lambda speed: setattr(stub, "worker", _DummyWorker()),
        start_btn=_DummyButton(),
        _update_warning_visibility=lambda: None,
        display=_DummyDisplay(),
        tabs=_DummyTabs(),
    )

    start_acquisition(stub)

    assert captured["cfg"] == stub.settings_manager.settings["CLASSIFIER_CONFIG"]
    assert captured["started"] is True
