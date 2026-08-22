import ast
import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_filter_compare(extra_globals: dict):
    tree = ast.parse((REPO_ROOT / "ver_main.py").read_text(encoding="utf-8"))
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERMainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_run_filter_compare":
                    target = child
                    break
    if target is None:
        raise AssertionError("Could not find VERMainWindow._run_filter_compare")

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    namespace.update(extra_globals)
    exec(compile(module, filename="ver_main.py", mode="exec"), namespace)
    return namespace["_run_filter_compare"]


class _DummySpin:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _FakeTable:
    def auto_set_font_size(self, _value):
        return None

    def set_fontsize(self, _value):
        return None


class _FakeAxes:
    def plot(self, *_args, **_kwargs):
        return None

    def axvline(self, *_args, **_kwargs):
        return None

    def axhline(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def set_title(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None

    def annotate(self, *_args, **_kwargs):
        return None

    def table(self, *_args, **_kwargs):
        return _FakeTable()


class _FakeFigure:
    def savefig(self, path, **_kwargs):
        Path(path).write_bytes(b"fake-png")

    def subplots_adjust(self, **_kwargs):
        return None


class _FakePlot:
    @staticmethod
    def subplots(**_kwargs):
        return _FakeFigure(), _FakeAxes()

    @staticmethod
    def tight_layout(**_kwargs):
        return None

    @staticmethod
    def close(_fig):
        return None


class _DummyMessageBox:
    @staticmethod
    def information(*_args, **_kwargs):
        return None

    @staticmethod
    def warning(*_args, **_kwargs):
        return None

    @staticmethod
    def critical(*_args, **_kwargs):
        return None


class _DummyFileDialog:
    @staticmethod
    def getExistingDirectory(*_args, **_kwargs):
        return ""


class _IdentityBandpassFilter:
    def __init__(self, _cfg):
        self.mode = None

    def set_scope_mode(self, mode):
        self.mode = mode

    def apply_zero_phase(self, samples, baseline_mean=None):
        arr = np.asarray(samples, dtype=float)
        dc = float(np.mean(arr) if baseline_mean is None else baseline_mean)
        return arr - dc


def _detect_stub(epoch_avg, epoch_time_ms, classifier_cfg=None):
    _ = classifier_cfg
    mask = (epoch_time_ms >= 0) & (epoch_time_ms <= 200)
    seg = np.asarray(epoch_avg, dtype=float)[mask]
    seg_t = np.asarray(epoch_time_ms, dtype=float)[mask]
    dominant = int(np.argmax(np.abs(seg)))
    dominant_sign = np.sign(seg[dominant]) if seg.size else 0.0
    before = None
    after = None
    for idx in range(seg.size):
        if dominant_sign == 0 or seg[idx] * dominant_sign >= 0:
            continue
        if idx < dominant:
            before = idx
        elif idx > dominant and after is None:
            after = idx
    noise = max(float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0, 1e-10)

    def _peak(idx):
        if idx is None:
            return {"latency_ms": float("nan"), "amplitude": float("nan"), "found": False, "snr": float("nan"), "above_threshold": False}
        amp = float(seg[idx])
        snr = abs(amp) / noise
        return {"latency_ms": float(seg_t[idx]), "amplitude": amp, "found": True, "snr": snr, "above_threshold": True}

    return {
        "Peak-1": _peak(before),
        "Peak-2": _peak(dominant if seg.size else None),
        "Peak-3": _peak(after),
        "VER_detected": True,
        "noise_rms": noise,
    }


def _wavelet_stub(epoch_avg):
    arr = np.abs(np.asarray(epoch_avg, dtype=float))
    if arr.size == 0:
        arr = np.zeros(1, dtype=float)
    return arr.reshape(1, -1), np.array([12.0], dtype=float)


def _build_compare_callable():
    globals_map = {
        "np": np,
        "csv": csv,
        "datetime": datetime,
        "Path": Path,
        "hashlib": __import__("hashlib"),
        "BandpassFilter": _IdentityBandpassFilter,
        "QMessageBox": _DummyMessageBox,
        "QFileDialog": _DummyFileDialog,
        "plt": _FakePlot,
        "compute_wavelet_scalogram": _wavelet_stub,
        "detect_ver_peaks": _detect_stub,
        "ver_classifier": SimpleNamespace(
            evaluate_ver_peak=lambda *args, **kwargs: (
                True,
                {"Scale Range": True, "Minimum Power": True, "P2 Latency": True, "Peak Structure": True, "SNR": True},
            )
        ),
        "ver_peaks": SimpleNamespace(
            DEFAULT_PEAK_DETECTION_MODE="dominant_opposite_neighbors",
            DOMINANT_OPPOSITE_NEIGHBORS_MODE="dominant_opposite_neighbors",
            LEGACY_PEAK_DETECTION_MODE="legacy_top3",
        ),
        "SCOPE_FILTER_BUTTERWORTH": "butter",
        "SCOPE_FILTER_FIR": "fir",
        "SCOPE_FILTER_SAVGOL": "savgol",
        "BASELINE_START_MS": -100.0,
        "BASELINE_END_MS": 0.0,
        "ACQ_CONFIG": {"sample_rate": 250.0},
        "log": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    return _load_run_filter_compare(globals_map)


def _run_case(tmp_path: Path, raw_session_averages, session_averages, accepted_counts):
    run_compare = _build_compare_callable()
    epoch_time_ms = np.array([-100.0, 0.0, 100.0, 200.0], dtype=float)
    stub = SimpleNamespace(
        scope=SimpleNamespace(
            epoch_time_ms=epoch_time_ms,
            pre_samples=1,
            raw_session_averages=[np.asarray(v, dtype=float) for v in raw_session_averages],
            session_averages=[np.asarray(v, dtype=float) for v in session_averages],
            raw_session_epochs=[],
            session_epochs=[],
        ),
        low_spin=_DummySpin(12.0),
        high_spin=_DummySpin(32.0),
        _validate_filter_bounds=lambda _low, _high: (True, ""),
        settings_manager=SimpleNamespace(settings={"CLASSIFIER_CONFIG": {"peak_detection_mode": "dominant_opposite_neighbors"}}),
        bandpass=SimpleNamespace(scope_mode="butter"),
        session_flash_counts_accepted=accepted_counts,
        data_file=str(tmp_path / "input.txt"),
    )
    Path(stub.data_file).write_text("dummy", encoding="utf-8")
    run_compare(stub)
    csv_files = sorted(tmp_path.glob("*_filter_compare_*.csv"))
    assert csv_files, "Filter Compare CSV was not created"
    with open(csv_files[-1], newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_filter_compare_single_ver_case_keeps_report_basis_checksum(tmp_path):
    rows = _run_case(
        tmp_path,
        raw_session_averages=[[1.0, 5.0, -3.0, 2.0]],
        session_averages=[[0.0, 4.0, -4.0, 1.0]],
        accepted_counts=[120],
    )
    butter = next(row for row in rows if row["mode"] == "butter")
    assert butter["source_basis"] == "report_waveforms_mean"
    assert butter["n_waveforms_used"] == "1"
    assert butter["accepted_indices"] == "1"
    assert butter["waveform_checksum"] == butter["report_basis_checksum"]


def test_filter_compare_multi_ver_case_uses_all_completed_sessions(tmp_path):
    rows = _run_case(
        tmp_path,
        raw_session_averages=[
            [1.0, 5.0, -3.0, 2.0],
            [1.0, 3.0, -5.0, 2.0],
        ],
        session_averages=[
            [0.0, 4.0, -4.0, 1.0],
            [0.0, 2.0, -6.0, 1.0],
        ],
        accepted_counts=[118, 120],
    )
    butter = next(row for row in rows if row["mode"] == "butter")
    assert butter["source_basis"] == "report_waveforms_mean"
    assert butter["n_waveforms_used"] == "2"
    assert butter["accepted_indices"] == "1,2"
    assert butter["accepted_index_count"] == "238"
    assert butter["waveform_checksum"] == butter["report_basis_checksum"]
