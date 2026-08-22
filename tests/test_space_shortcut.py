"""Source-level contracts for the Space-bar transport shortcut in VERMainWindow."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_VER_MAIN_SRC = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
_VER_MAIN_TREE = ast.parse(_VER_MAIN_SRC)


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


def test_ver_main_has_space_shortcut_methods():
    method_names = _class_method_names("VERMainWindow")
    assert "_configure_space_shortcut" in method_names
    assert "handle_space_toggle" in method_names
    assert "_focused_widget_blocks_space_shortcut" in method_names
    assert "_update_transport_button_labels" in method_names


def test_init_installs_space_shortcut_and_event_filter():
    src = _method_src("VERMainWindow", "__init__")
    assert "self._configure_space_shortcut()" in src
    assert "installEventFilter(self)" in src


def test_configure_space_shortcut_uses_single_window_shortcut():
    src = _method_src("VERMainWindow", "_configure_space_shortcut")
    assert "QShortcut" in src
    assert "Key_Space" in src
    assert "WindowShortcut" in src
    assert "activated.connect(self.handle_space_toggle)" in src


def test_handle_space_toggle_routes_by_transport_state():
    src = _method_src("VERMainWindow", "handle_space_toggle")
    assert "_transport_state()" in src
    assert 'state == "running"' in src
    assert "stop_acquisition()" in src
    assert "start_acquisition()" in src
    assert "log.debug" in src


def test_space_shortcut_guard_checks_editable_focus():
    src = _method_src("VERMainWindow", "_focused_widget_blocks_space_shortcut")
    assert "focusWidget" in src
    assert "QLineEdit" in src
    assert "QAbstractSpinBox" in src
    assert "QTextEdit" in src


def test_event_filter_blocks_shortcut_override_for_editable_focus():
    src = _method_src("VERMainWindow", "eventFilter")
    assert "ShortcutOverride" in src
    assert "Key_Space" in src
    assert "_focused_widget_blocks_space_shortcut()" in src
    assert "event.accept()" in src


def test_build_ui_disables_default_button_behavior_for_space_conflicts():
    src = _method_src("VERMainWindow", "_build_ui")
    assert 'QPushButton("Open Data File")' in src
    assert "open_btn.setAutoDefault(False)" in src
    assert "open_btn.setDefault(False)" in src
    assert "self.start_btn.setAutoDefault(False)" in src
    assert "self.start_btn.setDefault(False)" in src
    assert "self.stop_btn.setAutoDefault(False)" in src
    assert "self.stop_btn.setDefault(False)" in src


def test_transport_button_hints_remain_state_specific():
    src = _method_src("VERMainWindow", "_update_transport_button_labels")
    assert '"Stop  (Space)"' in src
    assert '"Resume  (Space)"' in src
    assert '"Start  (Space)"' in src
    assert 'self.stop_btn.setText("Stop")' in src


def test_running_and_paused_transitions_use_shared_label_helper():
    assert "_update_transport_button_labels()" in _method_src("VERMainWindow", "start_acquisition")
    assert "_update_transport_button_labels()" in _method_src("VERMainWindow", "stop_acquisition")
    assert "_update_transport_button_labels()" in _method_src("VERMainWindow", "reset_all")
    assert "_update_transport_button_labels()" in _method_src("VERMainWindow", "_handle_eof")


def test_key_press_event_no_longer_handles_space_transport():
    src = _method_src("VERMainWindow", "keyPressEvent")
    assert src == "", "Space transport should be handled only by the dedicated QShortcut"
