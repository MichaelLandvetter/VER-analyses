"""Tests for the focused-view toggle on the Raw + Filtered EEG panel.

These tests exercise the toggle state-machine logic and source-level contracts
without instantiating Qt widgets (so they run headlessly in CI).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_VER_DISPLAY_SRC = (REPO_ROOT / "ver_display.py").read_text(encoding="utf-8")
_VER_DISPLAY_TREE = ast.parse(_VER_DISPLAY_SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_level_names() -> set[str]:
    return {node.name for node in _VER_DISPLAY_TREE.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def _class_method_names(class_name: str) -> list[str]:
    for node in _VER_DISPLAY_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return []


def _class_src(class_name: str) -> str:
    """Return the source text of a top-level class."""
    for node in _VER_DISPLAY_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(_VER_DISPLAY_SRC, node) or ""
    return ""


def _method_src(class_name: str, method_name: str) -> str:
    for node in _VER_DISPLAY_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(_VER_DISPLAY_SRC, child) or ""
    return ""


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_focusable_viewbox_class_exists():
    assert "_FocusableViewBox" in _top_level_names(), (
        "_FocusableViewBox class must be defined at module level in ver_display.py"
    )


def test_focusable_viewbox_has_sig_double_clicked():
    src = _class_src("_FocusableViewBox")
    assert "sigDoubleClicked" in src, (
        "_FocusableViewBox must declare a sigDoubleClicked pyqtSignal"
    )


def test_focusable_viewbox_overrides_mouse_click_event():
    src = _class_src("_FocusableViewBox")
    assert "mouseClickEvent" in src, (
        "_FocusableViewBox must override mouseClickEvent to detect double-clicks"
    )


def test_focusable_viewbox_checks_ev_double():
    src = _class_src("_FocusableViewBox")
    assert "ev.double()" in src, (
        "_FocusableViewBox.mouseClickEvent must check ev.double() "
        "to use PyQtGraph's double-click routing"
    )


def test_ver_display_has_toggle_raw_focus_method():
    assert "toggle_raw_focus" in _class_method_names("VERDisplayWidget"), (
        "VERDisplayWidget must expose a toggle_raw_focus() method"
    )


def test_toggle_raw_focus_mutates_raw_focused_flag():
    src = _method_src("VERDisplayWidget", "toggle_raw_focus")
    assert "_raw_focused" in src, (
        "toggle_raw_focus must read/write self._raw_focused"
    )


def test_toggle_raw_focus_uses_layout_max_width_constraint():
    src = _method_src("VERDisplayWidget", "toggle_raw_focus")
    assert "setColumnMaximumWidth" in src, (
        "toggle_raw_focus must call setColumnMaximumWidth to collapse the "
        "VER-evolution column when entering focused mode"
    )


def test_toggle_raw_focus_uses_layout_max_height_constraint():
    src = _method_src("VERDisplayWidget", "toggle_raw_focus")
    assert "setRowMaximumHeight" in src, (
        "toggle_raw_focus must call setRowMaximumHeight to collapse "
        "Scope/Wavelet rows when entering focused mode"
    )


def test_toggle_raw_focus_hides_and_shows_other_panels():
    src = _method_src("VERDisplayWidget", "toggle_raw_focus")
    assert "plot_sessions" in src, (
        "toggle_raw_focus must reference plot_sessions to hide/show it"
    )
    assert "plot_scope" in src, (
        "toggle_raw_focus must reference plot_scope to hide/show it"
    )
    assert "plot_wavelet" in src, (
        "toggle_raw_focus must reference plot_wavelet to hide/show it"
    )


def test_toggle_raw_focus_does_not_call_data_processing():
    """Confirm toggle_raw_focus contains no data-processing calls."""
    src = _method_src("VERDisplayWidget", "toggle_raw_focus")
    forbidden = ("setData", "apply_filter", "update_scroll_panel", "compute_wavelet")
    for name in forbidden:
        assert name not in src, (
            f"toggle_raw_focus must not call {name!r} — "
            "no data reprocessing during a layout toggle"
        )


def test_raw_focused_flag_initialised_to_false():
    src = _method_src("VERDisplayWidget", "__init__")
    assert "_raw_focused = False" in src, (
        "VERDisplayWidget.__init__ must initialise _raw_focused to False"
    )


def test_reset_all_restores_focused_layout():
    src = _method_src("VERDisplayWidget", "reset_all")
    assert "_raw_focused" in src, (
        "reset_all must check _raw_focused and restore the normal layout "
        "if the panel was enlarged"
    )
    assert "toggle_raw_focus" in src, (
        "reset_all must call toggle_raw_focus() to cleanly restore the layout"
    )


def test_init_panels_passes_focusable_viewbox_to_plot_raw():
    src = _method_src("VERDisplayWidget", "_init_panels")
    assert "_FocusableViewBox()" in src, (
        "_init_panels must instantiate _FocusableViewBox for plot_raw"
    )
    assert "sigDoubleClicked.connect" in src, (
        "_init_panels must connect sigDoubleClicked to toggle_raw_focus"
    )


def test_layout_unconstrained_constant_defined():
    names = {
        node.targets[0].id
        for node in _VER_DISPLAY_TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
    }
    assert "_LAYOUT_UNCONSTRAINED" in names, (
        "_LAYOUT_UNCONSTRAINED constant must be defined at module level"
    )


def test_discoverability_hints_in_titles():
    assert "_RAW_TITLE_NORMAL" in _VER_DISPLAY_SRC, (
        "_RAW_TITLE_NORMAL must be defined so users see a hint in the plot title"
    )
    assert "_RAW_TITLE_FOCUSED" in _VER_DISPLAY_SRC, (
        "_RAW_TITLE_FOCUSED must be defined so users see a restore hint"
    )
    # Both hint strings must mention double-click
    for const_name in ("_RAW_TITLE_NORMAL", "_RAW_TITLE_FOCUSED"):
        # Extract the string value from the AST
        for node in _VER_DISPLAY_TREE.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == const_name
            ):
                value = ast.literal_eval(node.value)
                assert "double-click" in value.lower(), (
                    f"{const_name} must mention 'double-click' to aid discoverability"
                )
