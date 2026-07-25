"""Tests for the Space-bar Stop/Resume shortcut in VERMainWindow.

These tests verify source-level contracts without instantiating Qt widgets.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_VER_MAIN_SRC = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
_VER_MAIN_TREE = ast.parse(_VER_MAIN_SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_method_names(class_name: str) -> list[str]:
    for node in _VER_MAIN_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return []


def _method_src(class_name: str, method_name: str) -> str:
    for node in _VER_MAIN_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(_VER_MAIN_SRC, child) or ""
    return ""


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_ver_main_has_key_press_event():
    assert "keyPressEvent" in _class_method_names("VERMainWindow"), (
        "VERMainWindow must implement keyPressEvent to handle the Space shortcut"
    )


def test_key_press_event_checks_space_key():
    src = _method_src("VERMainWindow", "keyPressEvent")
    assert "Key_Space" in src, (
        "keyPressEvent must check for Qt.Key.Key_Space"
    )


def test_key_press_event_guards_editable_widgets():
    src = _method_src("VERMainWindow", "keyPressEvent")
    # Must check focused widget type to avoid triggering in text-entry fields
    assert "focusWidget" in src, (
        "keyPressEvent must call QApplication.focusWidget() "
        "to detect focus in text-entry controls"
    )
    assert "isinstance" in src, (
        "keyPressEvent must use isinstance() to guard editable widget types"
    )


def test_key_press_event_calls_stop_acquisition_when_running():
    src = _method_src("VERMainWindow", "keyPressEvent")
    assert "stop_acquisition" in src, (
        "keyPressEvent must call stop_acquisition() when analysis is running"
    )


def test_key_press_event_calls_start_acquisition_when_paused():
    src = _method_src("VERMainWindow", "keyPressEvent")
    assert "start_acquisition" in src, (
        "keyPressEvent must call start_acquisition() when analysis is paused/resumed"
    )


def test_key_press_event_does_not_trigger_initial_start():
    src = _method_src("VERMainWindow", "keyPressEvent")
    # Space must only fire for Running or Resume states, never for the Start state.
    # The guard uses startswith("Running") and startswith("Resume") — "Start" must not appear
    # as a condition that triggers action.
    assert "Start" not in src or "startswith" in src, (
        "keyPressEvent must not map Space to the initial Start action"
    )
    # More precise: check that the code branches on "Running" or "Resume", not "Start"
    assert 'startswith("Running")' in src or '"Running"' in src, (
        "keyPressEvent must check for the Running state"
    )
    assert 'startswith("Resume")' in src or '"Resume"' in src, (
        "keyPressEvent must check for the Resume state"
    )


def test_stop_btn_label_includes_space_hint():
    src = _method_src("VERMainWindow", "_build_ui")
    assert "Stop  (Space)" in src or "Stop (Space)" in src, (
        "_build_ui must set the stop button label to include '(Space)' hint"
    )


def test_stop_acquisition_sets_resume_with_space_hint():
    src = _method_src("VERMainWindow", "stop_acquisition")
    assert "Resume  (Space)" in src or "Resume (Space)" in src, (
        "stop_acquisition must set start_btn text to 'Resume  (Space)' "
        "to inform users about the keyboard shortcut"
    )


def test_key_press_event_delegates_to_super_when_not_handled():
    src = _method_src("VERMainWindow", "keyPressEvent")
    assert "super().keyPressEvent" in src, (
        "keyPressEvent must call super().keyPressEvent(event) for unhandled keys "
        "to preserve default Qt behavior"
    )
