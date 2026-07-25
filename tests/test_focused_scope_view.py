"""Tests for the focused-view toggle on the Scope View panel.

These tests exercise the toggle state-machine logic and source-level contracts
without instantiating Qt widgets (so they run headlessly in CI).

The test structure mirrors tests/test_focused_raw_view.py for consistency.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
_VER_DISPLAY_SRC = (REPO_ROOT / "ver_display.py").read_text(encoding="utf-8")
_VER_DISPLAY_TREE = ast.parse(_VER_DISPLAY_SRC)


def _get_module_constant(name: str):
    """Extract a literal module-level constant from the parsed ver_display.py AST."""
    for node in _VER_DISPLAY_TREE.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Module constant {name!r} not found in ver_display.py")


# Read directly from the source so the test stays in sync if the value ever changes.
_LAYOUT_UNCONSTRAINED = _get_module_constant("_LAYOUT_UNCONSTRAINED")


# ---------------------------------------------------------------------------
# Helpers (shared with test_focused_raw_view.py by convention)
# ---------------------------------------------------------------------------

def _top_level_names() -> set[str]:
    return {node.name for node in _VER_DISPLAY_TREE.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def _class_method_names(class_name: str) -> list[str]:
    for node in _VER_DISPLAY_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
    return []


def _class_src(class_name: str) -> str:
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
    """Compile a single VERDisplayWidget method as a standalone callable."""
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

def test_ver_display_has_toggle_scope_focus_method():
    assert "toggle_scope_focus" in _class_method_names("VERDisplayWidget"), (
        "VERDisplayWidget must expose a toggle_scope_focus() method"
    )


def test_toggle_scope_focus_mutates_scope_focused_flag():
    src = _method_src("VERDisplayWidget", "toggle_scope_focus")
    assert "_scope_focused" in src, (
        "toggle_scope_focus must read/write self._scope_focused"
    )


def test_toggle_scope_focus_uses_layout_max_width_constraint():
    src = _method_src("VERDisplayWidget", "toggle_scope_focus")
    assert "setColumnMaximumWidth" in src, (
        "toggle_scope_focus must call setColumnMaximumWidth to collapse the "
        "VER-evolution column when entering focused mode"
    )


def test_toggle_scope_focus_uses_layout_max_height_constraint():
    src = _method_src("VERDisplayWidget", "toggle_scope_focus")
    assert "setRowMaximumHeight" in src, (
        "toggle_scope_focus must call setRowMaximumHeight to collapse "
        "Raw/Wavelet rows when entering focused mode"
    )


def test_toggle_scope_focus_hides_and_shows_other_panels():
    src = _method_src("VERDisplayWidget", "toggle_scope_focus")
    assert "plot_sessions" in src, (
        "toggle_scope_focus must reference plot_sessions to hide/show it"
    )
    assert "plot_raw" in src, (
        "toggle_scope_focus must reference plot_raw to hide/show it"
    )
    assert "plot_wavelet" in src, (
        "toggle_scope_focus must reference plot_wavelet to hide/show it"
    )


def test_toggle_scope_focus_does_not_call_data_processing():
    """Confirm toggle_scope_focus contains no data-processing calls."""
    src = _method_src("VERDisplayWidget", "toggle_scope_focus")
    forbidden = ("setData", "apply_filter", "update_scroll_panel", "compute_wavelet")
    for name in forbidden:
        assert name not in src, (
            f"toggle_scope_focus must not call {name!r} — "
            "no data reprocessing during a layout toggle"
        )


def test_scope_focused_flag_initialised_to_false():
    src = _method_src("VERDisplayWidget", "__init__")
    assert "_scope_focused = False" in src, (
        "VERDisplayWidget.__init__ must initialise _scope_focused to False"
    )


def test_reset_all_restores_scope_focused_layout():
    src = _method_src("VERDisplayWidget", "reset_all")
    assert "_scope_focused" in src, (
        "reset_all must check _scope_focused and restore the normal layout "
        "if the scope panel was enlarged"
    )
    assert "toggle_scope_focus" in src, (
        "reset_all must call toggle_scope_focus() to cleanly restore the layout"
    )


def test_init_panels_passes_focusable_viewbox_to_plot_scope():
    src = _method_src("VERDisplayWidget", "_init_panels")
    assert "_FocusableViewBox()" in src, (
        "_init_panels must instantiate _FocusableViewBox (for plot_raw and/or plot_scope)"
    )
    assert "toggle_scope_focus" in src, (
        "_init_panels must connect a sigDoubleClicked signal to toggle_scope_focus"
    )


def test_scope_title_constants_defined():
    names = {
        node.targets[0].id
        for node in _VER_DISPLAY_TREE.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
    }
    assert "_SCOPE_TITLE_NORMAL" in names, (
        "_SCOPE_TITLE_NORMAL constant must be defined at module level in ver_display.py"
    )
    assert "_SCOPE_TITLE_FOCUSED" in names, (
        "_SCOPE_TITLE_FOCUSED constant must be defined at module level in ver_display.py"
    )


def test_scope_title_constants_mention_double_click():
    for const_name in ("_SCOPE_TITLE_NORMAL", "_SCOPE_TITLE_FOCUSED"):
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
# Behavioural / integration tests (no Qt widgets required)
# ---------------------------------------------------------------------------

def _build_stub():
    """Return a stub that mimics the parts of VERDisplayWidget used by toggle_scope_focus."""
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
        _scope_focused=False,
        graphics=SimpleNamespace(ci=SimpleNamespace(layout=layout)),
        plot_sessions=_RecordingPlot("sessions"),
        plot_scope=_RecordingPlot("scope"),
        plot_wavelet=_RecordingPlot("wavelet"),
        wavelet_stats_label=_RecordingLabel(),
        plot_raw=_RecordingPlot("raw"),
    )
    return stub, calls


def test_toggle_scope_focus_behaviour_focus_then_restore():
    """toggle_scope_focus changes only layout/visibility state and toggles correctly."""
    toggle_scope_focus = _load_display_method("toggle_scope_focus")
    stub, calls = _build_stub()

    # --- First call: enter focused mode ---
    toggle_scope_focus(stub)

    assert stub._scope_focused is True, "Flag must flip to True after first toggle"
    assert not stub.plot_sessions.visible, "plot_sessions must be hidden in focused mode"
    assert not stub.plot_raw.visible, "plot_raw must be hidden in focused mode"
    assert not stub.plot_wavelet.visible, "plot_wavelet must be hidden in focused mode"
    assert not stub.wavelet_stats_label.visible, "wavelet_stats_label must be hidden in focused mode"
    assert stub.plot_scope.visible, "plot_scope must remain visible in focused mode"

    # Layout must have collapsed col 0 and rows 0/2
    assert any(c == ("setColumnMaximumWidth", 0, 0) for c in calls), (
        "Column 0 max-width must be set to 0 when entering focused mode"
    )
    assert any(c == ("setRowMaximumHeight", 0, 0) for c in calls), (
        "Row 0 max-height must be set to 0 when entering focused mode"
    )
    assert any(c == ("setRowMaximumHeight", 2, 0) for c in calls), (
        "Row 2 max-height must be set to 0 when entering focused mode"
    )

    # --- Second call: restore normal mode ---
    calls.clear()
    toggle_scope_focus(stub)

    assert stub._scope_focused is False, "Flag must flip back to False after second toggle"
    assert stub.plot_sessions.visible, "plot_sessions must be shown after restore"
    assert stub.plot_raw.visible, "plot_raw must be shown after restore"
    assert stub.plot_wavelet.visible, "plot_wavelet must be shown after restore"
    assert stub.wavelet_stats_label.visible, "wavelet_stats_label must be shown after restore"

    # Layout must have removed constraints for col 0 and rows 0/2
    assert any(c == ("setColumnMaximumWidth", 0, _LAYOUT_UNCONSTRAINED) for c in calls), (
        "Column 0 max-width must be restored to QWIDGETSIZE_MAX after restore"
    )
    assert any(c == ("setRowMaximumHeight", 0, _LAYOUT_UNCONSTRAINED) for c in calls), (
        "Row 0 max-height must be restored to QWIDGETSIZE_MAX after restore"
    )
    assert any(c == ("setRowMaximumHeight", 2, _LAYOUT_UNCONSTRAINED) for c in calls), (
        "Row 2 max-height must be restored to QWIDGETSIZE_MAX after restore"
    )


def test_toggle_scope_focus_behaviour_preserves_data_attributes():
    """toggle_scope_focus must not touch any data buffers or analysis state."""
    toggle_scope_focus = _load_display_method("toggle_scope_focus")
    stub, _ = _build_stub()

    sentinel = object()
    stub.raw_buffer = sentinel
    stub.filtered_buffer = sentinel
    stub.time_buffer = sentinel
    stub.sample_index = 99
    stub._last_scroll_draw = 3.14

    toggle_scope_focus(stub)  # enter focused
    toggle_scope_focus(stub)  # restore

    assert stub.raw_buffer is sentinel, "raw_buffer must not be modified by toggle_scope_focus"
    assert stub.filtered_buffer is sentinel, "filtered_buffer must not be modified by toggle_scope_focus"
    assert stub.time_buffer is sentinel, "time_buffer must not be modified by toggle_scope_focus"
    assert stub.sample_index == 99, "sample_index must not be modified by toggle_scope_focus"
    assert stub._last_scroll_draw == 3.14, "_last_scroll_draw must not be modified by toggle_scope_focus"
