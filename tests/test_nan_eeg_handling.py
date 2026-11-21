import numpy as np

from ver_acquisition import repair_nonfinite_eeg
from ver_scope import VERScopeProcessor


class _PassThroughFilter:
    def apply_zero_phase(self, epoch, baseline_mean=0.0):
        return epoch


def test_repair_keeps_length_and_interpolates_gap():
    eeg = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
    repaired, bad = repair_nonfinite_eeg(eeg)
    assert repaired.shape == eeg.shape
    assert np.all(np.isfinite(repaired))
    assert np.allclose(repaired[2:4], [3.0, 4.0])
    assert bad.tolist() == [False, False, True, True, False]


def test_epoch_overlapping_gap_is_rejected_not_averaged():
    config = {"pre_stim_ms": 8, "post_stim_ms": 12, "artifact_exclusion_uv": 100.0,
              "artifact_rejection_enabled": True, "flashes_per_session": 1000}
    scope = VERScopeProcessor(_PassThroughFilter(), epoch_config=config)
    scope.pre_buffer.clear()
    scope.pre_bad.clear()
    scope.pre_buffer.extend([0.0, 0.0])
    scope.pre_bad.extend([False, False])

    samples = [(1, 0.0, True), (0, 0.0, False), (0, 0.0, False), (0, 0.0, False)]
    results = [scope.process_sample(bool(t), v, is_bad=b) for t, v, b in samples]
    completed = [r for r in results if r.get("epoch_complete")]

    assert len(completed) == 1
    assert completed[0]["epoch_has_gap"] is True
    assert completed[0]["epoch_rejected"] is True
    assert scope.flash_count == 1
    assert scope.flash_count_accepted == 0


def test_gap_epochs_are_counted_separately_per_session():
    config = {"pre_stim_ms": 8, "post_stim_ms": 12, "artifact_exclusion_uv": 100.0,
              "artifact_rejection_enabled": True, "flashes_per_session": 2}
    scope = VERScopeProcessor(_PassThroughFilter(), epoch_config=config)
    scope.pre_buffer.extend([0.0, 0.0])
    scope.pre_bad.extend([False, False])

    samples = [(1, 0.0, True), (0, 0.0, False), (0, 0.0, False), (0, 0.0, False),
               (1, 0.0, False), (0, 0.0, False), (0, 0.0, False), (0, 0.0, False)]
    results = [scope.process_sample(bool(t), v, is_bad=b) for t, v, b in samples]
    completed = [r for r in results if r.get("session_complete")]

    assert len(completed) == 1
    assert completed[0]["completed_session_flash_count_gap"] == 1
    assert scope.flash_count_gap == 0


def test_summary_csv_reports_gap_excluded_count(tmp_path):
    from ver_report import _write_summary_csv

    wavelet = np.ones((3, 10))
    freqs = np.array([1.0, 2.0, 3.0])
    time_ms = np.arange(10) * 4.0
    path = tmp_path / "summary.csv"
    _write_summary_csv(
        path, [wavelet], freqs, time_ms, None, [10],
        session_flash_counts_accepted=[8],
        session_flash_counts_gap=[2],
    )
    header, row = path.read_text(encoding="utf-8").splitlines()[:2]
    assert header.split(",")[-1] == "N_flashes_gap_excluded"
    assert row.split(",")[-1] == "2"
