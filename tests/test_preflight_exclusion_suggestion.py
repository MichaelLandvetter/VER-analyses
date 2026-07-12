import numpy as np

from ver_preflight import suggest_exclusion_from_file

EXPECTED_MIN_THRESHOLD = 0.5
EXPECTED_MAX_THRESHOLD = 1.0
FLASHES_PER_SESSION_LIMIT = 9999


class DummyFilter:
    def apply_zero_phase(self, epoch, baseline_mean=0.0):
        return np.asarray(epoch, dtype=float) - baseline_mean


def _write_test_file(path, amplitudes):
    rows = []
    pre_samples = 4
    post_samples = 8
    for amp in amplitudes:
        rows.extend([[0.0, 0.0]] * pre_samples)
        rows.append([1.0, 0.0])
        rows.append([0.0, float(amp)])
        rows.extend([[0.0, 0.0]] * (post_samples - 2))
        rows.append([0.0, 0.0])
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter="\t", fmt="%.6f")


def test_suggest_exclusion_uses_whole_file_epochs(tmp_path):
    data_path = tmp_path / "preflight_input.txt"
    _write_test_file(data_path, amplitudes=[0.2, 0.25, 0.3, 0.35, 1.2, 1.5])

    suggestion = suggest_exclusion_from_file(
        str(data_path),
        epoch_config={
            "pre_stim_ms": 4,
            "post_stim_ms": 8,
            "flashes_per_session": FLASHES_PER_SESSION_LIMIT,
            "num_sessions": 1,
            "artifact_rejection_enabled": True,
            "artifact_exclusion_uv": 0.01,
        },
        file_config={
            "delimiter": "\t",
            "trigger_column": 0,
            "eeg_column": 1,
            "skip_header": 0,
            "trigger_mode": "threshold",
            "trigger_threshold": 0.5,
        },
        bandpass_filter=DummyFilter(),
    )

    assert suggestion.total_epochs == 6
    assert suggestion.accepted_epochs == 4
    assert suggestion.rejected_epochs == 2
    assert EXPECTED_MIN_THRESHOLD < suggestion.suggested_threshold_uv < EXPECTED_MAX_THRESHOLD
