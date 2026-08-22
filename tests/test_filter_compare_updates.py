import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VER_MAIN = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
TREE = ast.parse(VER_MAIN)


def _class_method(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERMainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"Method {name} not found")


def test_filter_compare_moved_to_file_menu_and_removed_from_box3():
    build_menu = _class_method("_build_menu")
    build_ui = _class_method("_build_ui")

    menu_strings = {
        node.value
        for node in ast.walk(build_menu)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    ui_strings = {
        node.value
        for node in ast.walk(build_ui)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "Filter Compare" in menu_strings
    assert "Filter Compare" not in ui_strings


def test_low_and_high_cut_controls_use_decimal_steps():
    build_ui = _class_method("_build_ui")
    calls = [node for node in ast.walk(build_ui) if isinstance(node, ast.Call)]

    def _has_assignment(attr_name: str, constructor: str) -> bool:
        for node in ast.walk(build_ui):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id == constructor
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                    and node.targets[0].attr == attr_name
                ):
                    return True
        return False

    def _has_set_step(attr_name: str, step: float) -> bool:
        for call in calls:
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "setSingleStep"
                and isinstance(call.func.value, ast.Attribute)
                and isinstance(call.func.value.value, ast.Name)
                and call.func.value.value.id == "self"
                and call.func.value.attr == attr_name
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == step
            ):
                return True
        return False

    assert _has_assignment("low_spin", "QDoubleSpinBox")
    assert _has_assignment("high_spin", "QDoubleSpinBox")
    assert _has_set_step("low_spin", 0.1)
    assert _has_set_step("high_spin", 0.5)
    assert "def _validate_filter_bounds(self, low: float, high: float) -> tuple[bool, str]:" in VER_MAIN


def test_filter_compare_csv_contains_provenance_and_confidence_fields():
    required_fields = [
        "classification_pass",
        "scale_range_pass",
        "minimum_power_pass",
        "p2_latency_pass",
        "peak_structure_pass",
        "snr_pass",
        "source_basis",
        "n_waveforms_used",
        "time_start_ms",
        "time_end_ms",
        "dt_ms",
        "baseline_window_ms",
        "accepted_indices",
        "accepted_index_count",
        "accepted_index_hash",
        "p1_index",
        "p2_index",
        "p3_index",
        "p1_window_ms",
        "p2_window_ms",
        "p3_window_ms",
        "waveform_checksum",
        "report_basis_checksum",
        "low_confidence",
        "low_confidence_reason",
        "peak_detection_mode",
        "lowcut_hz",
        "highcut_hz",
    ]
    run_compare = _class_method("_run_filter_compare")
    csv_fields_values = set()
    for node in ast.walk(run_compare):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "csv_fields"
                and isinstance(node.value, ast.List)
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        csv_fields_values.add(elt.value)
    for field in required_fields:
        assert field in csv_fields_values


def test_validate_filter_bounds_enforces_nyquist_and_ordering_rules():
    target = _class_method("_validate_filter_bounds")

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"ACQ_CONFIG": {"sample_rate": 250}}
    exec(compile(module, filename="ver_main.py", mode="exec"), namespace)
    fn = namespace["_validate_filter_bounds"]

    assert fn(object(), 0.0, 32.0) == (True, "")
    ok, msg = fn(object(), -0.1, 32.0)
    assert ok is False and "0 Hz or higher" in msg
    ok, msg = fn(object(), 0.0, 0.0)
    assert ok is False and "above 0 Hz" in msg
    ok, msg = fn(object(), 12.0, 12.0)
    assert ok is False and "less than high cut" in msg
    ok, msg = fn(object(), 0.0, 125.0)
    assert ok is False and "Nyquist" in msg
