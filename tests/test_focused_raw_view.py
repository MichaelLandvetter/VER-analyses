"""Tests for the focused-view toggle on the Raw + Filtered EEG panel.

These tests exercise the toggle state-machine logic and source-level contracts
without instantiating Qt widgets (so they run headlessly in CI).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

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


def _load_display_method(method_name: str, extra_globals: dict | None = None):
    """Compile a single VERDisplayWidget method as a standalone callable.

    The compiled function accepts *self* as its first argument, so it can be
    called with a stub object instead of a real widget.  Module-level constants
    from ver_display.py are injected automatically.
    """
    # Collect module-level name→value assignments (constants used by methods)
    module_consts: dict = {}
    for node in _VER_DISPLAY_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                module_consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass

    for node in _VER_DISPLAY_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERDisplayWidget":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    module = ast.Module(body=[child], type_ignores=[])
                    ast.fix_missing_locations(module)
                    ns: dict = dict(module_consts)
                    if extra_globals:
                        ns.update(extra_globals)
                    exec(compile(module, filename="ver_display.py", mode="exec"), ns)
                    return ns[method_name]
    raise AssertionError(f"Method VERDisplayWidget.{method_name} not found")


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


def test_layout_unconstrained_value_is_qwidgetsize_max():
    """_LAYOUT_UNCONSTRAINED must equal Qt's QWIDGETSIZE_MAX (16 777 215)."""
    for node in _VER_DISPLAY_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_LAYOUT_UNCONSTRAINED"
        ):
            value = ast.literal_eval(node.value)
            assert value == 16_777_215.0, (
                f"_LAYOUT_UNCONSTRAINED should equal QWIDGETSIZE_MAX (16 777 215), got {value}"
            )
            return
    raise AssertionError("_LAYOUT_UNCONSTRAINED not found in module body")


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


# ---------------------------------------------------------------------------
# Behavioural / integration test (no Qt widgets required)
# ---------------------------------------------------------------------------

def _build_stub():
    """Return a stub that mimics the parts of VERDisplayWidget used by toggle_raw_focus."""
    calls = []

    class _RecordingLayout:
        def setColumnMaximumWidth(self, col, w):
            calls.append(("setColumnMaximumWidth", col, w))

        def setRowMaximumHeight(self, row, h):
            calls.append(("setRowMaximumHeight", row, h))

    class _RecordingPlot:
        def __init__(self, name):
            self._name = name
            self.visible = True

        def hide(self):
            self.visible = False
            calls.append(("hide", self._name))

        def show(self):
            self.visible = True
            calls.append(("show", self._name))

        def setTitle(self, t):
            calls.append(("setTitle", self._name, t))

    class _RecordingLabel:
        def __init__(self):
            self.visible = True

        def hide(self):
            self.visible = False
            calls.append(("hide", "wavelet_stats_label"))

        def show(self):
            self.visible = True
            calls.append(("show", "wavelet_stats_label"))

    layout = _RecordingLayout()
    stub = SimpleNamespace(
        _raw_focused=False,
        graphics=SimpleNamespace(ci=SimpleNamespace(layout=layout)),
        plot_sessions=_RecordingPlot("sessions"),
        plot_scope=_RecordingPlot("scope"),
        plot_wavelet=_RecordingPlot("wavelet"),
        wavelet_stats_label=_RecordingLabel(),
        plot_raw=_RecordingPlot("raw"),
    )
    return stub, calls


def test_toggle_raw_focus_behaviour_focus_then_restore():
    """toggle_raw_focus changes only layout/visibility state and toggles correctly."""
    toggle_raw_focus = _load_display_method("toggle_raw_focus")
    stub, calls = _build_stub()

    # --- First call: enter focused mode ---
    toggle_raw_focus(stub)

    assert stub._raw_focused is True, "Flag must flip to True after first toggle"
    assert not stub.plot_sessions.visible, "plot_sessions must be hidden in focused mode"
    assert not stub.plot_scope.visible, "plot_scope must be hidden in focused mode"
    assert not stub.plot_wavelet.visible, "plot_wavelet must be hidden in focused mode"
    assert not stub.wavelet_stats_label.visible, "wavelet_stats_label must be hidden in focused mode"

    # Layout must have collapsed col 0 and rows 1/2
    assert any(c == ("setColumnMaximumWidth", 0, 0.0) for c in calls), (
        "Column 0 max-width must be set to 0 when entering focused mode"
    )
    assert any(c == ("setRowMaximumHeight", 1, 0.0) for c in calls), (
        "Row 1 max-height must be set to 0 when entering focused mode"
    )
    assert any(c == ("setRowMaximumHeight", 2, 0.0) for c in calls), (
        "Row 2 max-height must be set to 0 when entering focused mode"
    )

    # --- Second call: restore normal mode ---
    calls.clear()
    toggle_raw_focus(stub)

    assert stub._raw_focused is False, "Flag must flip back to False after second toggle"
    assert stub.plot_sessions.visible, "plot_sessions must be shown after restore"
    assert stub.plot_scope.visible, "plot_scope must be shown after restore"
    assert stub.plot_wavelet.visible, "plot_wavelet must be shown after restore"
    assert stub.wavelet_stats_label.visible, "wavelet_stats_label must be shown after restore"

    # Layout must have removed constraints for col 0 and rows 1/2
    unconstrained = 16_777_215.0
    assert any(c == ("setColumnMaximumWidth", 0, unconstrained) for c in calls), (
        "Column 0 max-width must be restored to QWIDGETSIZE_MAX after restore"
    )
    assert any(c == ("setRowMaximumHeight", 1, unconstrained) for c in calls), (
        "Row 1 max-height must be restored to QWIDGETSIZE_MAX after restore"
    )
    assert any(c == ("setRowMaximumHeight", 2, unconstrained) for c in calls), (
        "Row 2 max-height must be restored to QWIDGETSIZE_MAX after restore"
    )


def test_toggle_raw_focus_behaviour_preserves_data_attributes():
    """toggle_raw_focus must not touch any data buffers or analysis state."""
    toggle_raw_focus = _load_display_method("toggle_raw_focus")
    stub, _ = _build_stub()

    # Add data-like attributes that should remain untouched
    sentinel = object()
    stub.raw_buffer = sentinel
    stub.filtered_buffer = sentinel
    stub.time_buffer = sentinel
    stub.sample_index = 42
    stub._last_scroll_draw = 1.234

    toggle_raw_focus(stub)  # enter focused
    toggle_raw_focus(stub)  # restore

    assert stub.raw_buffer is sentinel, "raw_buffer must not be modified by toggle_raw_focus"
    assert stub.filtered_buffer is sentinel, "filtered_buffer must not be modified by toggle_raw_focus"
    assert stub.time_buffer is sentinel, "time_buffer must not be modified by toggle_raw_focus"
    assert stub.sample_index == 42, "sample_index must not be modified by toggle_raw_focus"
    assert stub._last_scroll_draw == 1.234, "_last_scroll_draw must not be modified by toggle_raw_focus"
