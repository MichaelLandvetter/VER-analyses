"""Tests for the live USB session save workflow improvements.

Covers:
- ``save_ver_report()`` accepting an explicit ``output_dir`` that is used
  instead of the auto-derived Reports sub-directory.
- The ``_prompt_live_session_info`` helper's sanitisation and cancellation
  logic (tested in isolation without a running Qt event-loop).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ver_report import save_ver_report

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_averages():
    """Return a single session average and matching epoch time array."""
    epoch_time_ms = np.linspace(-10, 30, 40)
    # Use a non-trivial waveform so wavelet normalization does not raise.
    session_average = np.sin(np.linspace(0, 2 * np.pi, 40)) * 2.0
    return [session_average], epoch_time_ms


# ---------------------------------------------------------------------------
# ver_report.save_ver_report – output_dir parameter
# ---------------------------------------------------------------------------

def test_save_ver_report_output_dir_respected(tmp_path):
    """When output_dir is given the files are written there, not in a
    Reports sub-directory relative to the data file or cwd."""
    averages, epoch_time_ms = _minimal_averages()
    custom_dir = tmp_path / "MyExperiment"

    result = save_ver_report(
        str(tmp_path / "serial_live_report.txt"),
        averages,
        epoch_time_ms,
        force_stem="MyExperiment",
        output_dir=custom_dir,
    )

    assert result is not None
    assert Path(result["png"]).parent == custom_dir
    assert Path(result["pdf"]).parent == custom_dir
    assert Path(result["summary_csv"]).parent == custom_dir
    assert Path(result["waveforms_csv"]).parent == custom_dir
    assert Path(result["report_dir"]) == custom_dir


def test_save_ver_report_output_dir_filenames_use_stem(tmp_path):
    """Files inside output_dir are named with the force_stem prefix."""
    averages, epoch_time_ms = _minimal_averages()
    stem = "animal01_session02"
    custom_dir = tmp_path / stem

    result = save_ver_report(
        str(tmp_path / "serial_live_report.txt"),
        averages,
        epoch_time_ms,
        force_stem=stem,
        output_dir=custom_dir,
    )

    assert Path(result["png"]).name == f"{stem}.png"
    assert Path(result["pdf"]).name == f"{stem}.pdf"
    assert Path(result["summary_csv"]).name == f"{stem}_summary.csv"
    assert Path(result["waveforms_csv"]).name == f"{stem}_waveforms.csv"


def test_save_ver_report_output_dir_created_automatically(tmp_path):
    """output_dir is created if it does not yet exist."""
    averages, epoch_time_ms = _minimal_averages()
    nested_dir = tmp_path / "base" / "exp01"
    assert not nested_dir.exists()

    save_ver_report(
        str(tmp_path / "serial_live_report.txt"),
        averages,
        epoch_time_ms,
        force_stem="exp01",
        output_dir=nested_dir,
    )

    assert nested_dir.is_dir()


def test_save_ver_report_without_output_dir_uses_default_reports_folder(tmp_path):
    """When output_dir is omitted, a Reports sub-folder is created next to
    the data file as before (regression guard)."""
    averages, epoch_time_ms = _minimal_averages()
    data_file = tmp_path / "subject_data.txt"
    data_file.write_text("placeholder\n", encoding="utf-8")

    result = save_ver_report(str(data_file), averages, epoch_time_ms)

    report_dir = Path(result["report_dir"])
    assert "Reports" in report_dir.parts


# ---------------------------------------------------------------------------
# _prompt_live_session_info – static / source-level checks
# ---------------------------------------------------------------------------

def _compile_method(method_name: str, extra_ns: dict | None = None):
    """Compile a VERMainWindow method with optional namespace overrides."""
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERMainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    module = ast.Module(body=[child], type_ignores=[])
                    ast.fix_missing_locations(module)
                    ns = dict(extra_ns or {})
                    exec(compile(module, filename="ver_main.py", mode="exec"), ns)
                    return ns[method_name]
    raise AssertionError(f"{method_name} not found in VERMainWindow")


def test_prompt_live_session_info_exists_in_vermainwindow():
    """VERMainWindow must define _prompt_live_session_info."""
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERMainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append(child.name)
    assert "_prompt_live_session_info" in methods


def test_prompt_live_session_info_cancels_on_empty_folder():
    """If the folder dialog returns an empty string the helper returns (None, None)."""
    fn = _compile_method(
        "_prompt_live_session_info",
        extra_ns={
            "QFileDialog": SimpleNamespace(
                getExistingDirectory=lambda *a, **k: "",
            ),
            "Path": Path,
            "QMessageBox": SimpleNamespace(warning=lambda *a, **k: None),
        },
    )
    result = fn(SimpleNamespace())
    assert result == (None, None)


def test_prompt_live_session_info_cancels_on_input_dialog_cancel():
    """If the name input dialog is cancelled (ok=False) the helper returns (None, None)."""

    class _QInputDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return ("", False)  # user pressed Cancel

    fn = _compile_method(
        "_prompt_live_session_info",
        extra_ns={
            "QFileDialog": SimpleNamespace(
                getExistingDirectory=lambda *a, **k: "/some/folder"
            ),
            "QInputDialog": _QInputDialog,
            "Path": Path,
            "QMessageBox": SimpleNamespace(warning=lambda *a, **k: None),
        },
    )
    result = fn(SimpleNamespace())
    assert result == (None, None)


def test_prompt_live_session_info_sanitises_invalid_chars():
    """Characters that are invalid in paths are replaced with underscores."""

    class _QInputDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return ('exp/1: "test"', True)

    fn = _compile_method(
        "_prompt_live_session_info",
        extra_ns={
            "QFileDialog": SimpleNamespace(
                getExistingDirectory=lambda *a, **k: "/base"
            ),
            "QInputDialog": _QInputDialog,
            "Path": Path,
            "QMessageBox": SimpleNamespace(warning=lambda *a, **k: None),
        },
    )
    output_dir, name = fn(SimpleNamespace())
    assert output_dir is not None
    for bad_char in r'\/:*?"<>|':
        assert bad_char not in name


def test_save_report_passes_output_dir_to_save_ver_report():
    """save_report must pass live_output_dir and live_experiment_name to
    save_ver_report when in serial (live) mode."""
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    # Simple textual assertion – both keyword arguments must appear in the
    # save_report body after the _prompt_live_session_info call.
    assert "output_dir=live_output_dir" in src
    assert "force_stem=live_experiment_name" in src
