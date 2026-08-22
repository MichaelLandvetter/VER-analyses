from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VER_MAIN = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")


def test_filter_compare_moved_to_file_menu_and_removed_from_box3():
    assert 'filter_compare_action = QAction("Filter Compare", self)' in VER_MAIN
    assert "file_menu.addAction(filter_compare_action)" in VER_MAIN
    assert 'QPushButton("Filter Compare")' not in VER_MAIN


def test_low_and_high_cut_controls_use_decimal_steps():
    assert "self.low_spin = QDoubleSpinBox()" in VER_MAIN
    assert "self.low_spin.setSingleStep(0.1)" in VER_MAIN
    assert "self.high_spin = QDoubleSpinBox()" in VER_MAIN
    assert "self.high_spin.setSingleStep(0.5)" in VER_MAIN
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
        "time_axis_start_ms",
        "time_axis_end_ms",
        "time_axis_step_ms",
        "peak_detection_mode",
        "lowcut_hz",
        "highcut_hz",
    ]
    for field in required_fields:
        assert f'"{field}"' in VER_MAIN
