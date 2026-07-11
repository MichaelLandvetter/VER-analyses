import csv

import numpy as np

from ver_report import save_ver_report
from ver_scope import VERScopeProcessor


class DummyFilter:
    def apply_zero_phase(self, epoch, baseline_mean=0.0):
        return np.asarray(epoch, dtype=float) - baseline_mean


TEST_SCOPE_CONFIG = {
    "pre_stim_ms": 4,
    "post_stim_ms": 8,
    "flashes_per_session": 2,
    "num_sessions": 1,
    "artifact_exclusion_uv": 0.5,
}

TEST_SAMPLES = [
    (0, 0.0),
    (1, 0.0),
    (0, 1.0),
    (0, 0.0),
    (1, 0.0),
    (0, 0.2),
    (0, 0.0),
]


def _run_scope(artifact_enabled: bool):
    scope = VERScopeProcessor(
        DummyFilter(),
        {**TEST_SCOPE_CONFIG, "artifact_rejection_enabled": artifact_enabled},
    )
    last_result = None
    for trigger, eeg in TEST_SAMPLES:
        result = scope.process_sample(trigger, eeg)
        if result["session_complete"]:
            last_result = result
    assert last_result is not None
    return scope, last_result


def test_artifact_toggle_changes_epoch_selection_and_average():
    _, disabled_result = _run_scope(False)
    _, enabled_result = _run_scope(True)

    assert disabled_result["completed_session_flash_count"] == 2
    assert disabled_result["completed_session_flash_count_accepted"] == 2
    assert disabled_result["artifact_rejection_enabled"] is False
    np.testing.assert_allclose(disabled_result["completed_session_average"], np.array([0.0, 0.0, 0.6]))

    assert enabled_result["completed_session_flash_count"] == 2
    assert enabled_result["completed_session_flash_count_accepted"] == 1
    assert enabled_result["artifact_rejection_enabled"] is True
    np.testing.assert_allclose(enabled_result["completed_session_average"], np.array([0.0, 0.0, 0.2]))


def test_summary_csv_records_mode_specific_counts_and_provenance(tmp_path):
    scope_off, disabled_result = _run_scope(False)
    _, enabled_result = _run_scope(True)

    data_file = tmp_path / "artifact_case.txt"
    data_file.write_text("placeholder\n", encoding="utf-8")

    result = save_ver_report(
        str(data_file),
        [
            disabled_result["completed_session_average"],
            enabled_result["completed_session_average"],
        ],
        scope_off.epoch_time_ms,
        session_flash_counts=[2, 2],
        session_flash_counts_accepted=[2, 1],
        session_artifact_rejection_enabled=[False, True],
        session_artifact_exclusion_thresholds=[0.5, 0.5],
    )

    with open(result["summary_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["N_flashes_accepted"] for row in rows] == ["2", "1"]
    assert [row["N_flashes_rejected"] for row in rows] == ["0", "1"]
    assert [row["Exclusion_Enabled"] for row in rows] == ["False", "True"]
    assert [row["Exclusion_Threshold"] for row in rows] == ["0.5", "0.5"]
