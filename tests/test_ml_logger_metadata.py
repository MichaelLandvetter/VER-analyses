import ast
from pathlib import Path

from ver_settings import SettingsManager


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ml_logger_source_includes_metadata_fields_and_compact_table():
    src = (REPO_ROOT / "ver_ml_logger.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    table_headers = None
    csv_headers = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setHorizontalHeaderLabels"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            table_headers = [elt.value for elt in node.args[0].elts if isinstance(elt, ast.Constant)]

        # The header is written via writer.writerow(_NEW_CSV_HEADER), so look for
        # the module-level _NEW_CSV_HEADER assignment to obtain the canonical list.
        if (
            isinstance(node, ast.Assign)
            and node.targets
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_NEW_CSV_HEADER"
            and isinstance(node.value, ast.List)
        ):
            values = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            if "Block" in values and "Human_Label" in values:
                csv_headers = values

    assert table_headers is not None
    assert csv_headers is not None
    assert '"Human Validation"' in src
    assert '"Computer Reason"' in src
    assert '"Human Reason"' in src
    assert '"Review Confidence"' in src
    assert "Observer ID" in src
    assert "filename" not in table_headers
    assert "species" not in table_headers
    assert "Source file:</b>" in src
    assert "Species:</b>" in src
    assert csv_headers[-2:] == ["File name", "Species"]
    assert "self.filename, self.species" in src


def test_settings_manager_default_metadata_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    assert manager.default_settings["METADATA_CONFIG"]["species"] == ""
    assert manager.settings["METADATA_CONFIG"]["species"] == ""


def test_ver_main_source_moves_species_selector_into_data_file_group():
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    self_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    selected_species_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_selected_species_value"
    ]

    assert 'self.file_species_combo = QComboBox()' in src
    assert 'self.file_species_combo.addItem("(not set)")' in src
    assert 'self.file_species_combo.addItems(self._species_options())' in src
    assert 'species_layout.addWidget(QLabel("Species:"))' in src
    assert "layout2.addLayout(species_layout)" in src
    assert "file_species_combo" in self_attrs
    assert "set_species" not in self_attrs
    assert "self.set_species" not in src
    assert 'return "" if species_value == "(not set)" else species_value' in src
    assert 'new_settings["METADATA_CONFIG"]["species"] = self._selected_species_value()' in src
    assert "species=self._selected_species_value()," in src
    assert len(selected_species_calls) >= 2
    # Persistence: species change must be wired to save immediately
    assert "self.file_species_combo.currentTextChanged.connect(self._on_species_changed)" in src
    assert "def _on_species_changed(" in src
    assert 'metadata["species"] = self._selected_species_value()' in src
    assert "self.settings_manager.save_settings()" in src
    # Ordering: connect must come after the initial restore to prevent a spurious save on startup
    connect_pos = src.find("self.file_species_combo.currentTextChanged.connect(self._on_species_changed)")
    restore_pos = src.find("self._set_species_selection(saved_species)")
    assert restore_pos != -1, "_set_species_selection(saved_species) not found in source"
    assert connect_pos != -1, "currentTextChanged.connect(_on_species_changed) not found in source"
    assert restore_pos < connect_pos, "_set_species_selection must appear before the signal connect in _build_ui"


def test_settings_manager_species_round_trip(monkeypatch, tmp_path):
    """Species written via SettingsManager.save_settings() is reloaded correctly."""
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    manager.settings.setdefault("METADATA_CONFIG", {})["species"] = "Cat"
    manager.save_settings()

    manager2 = SettingsManager()
    assert manager2.settings["METADATA_CONFIG"]["species"] == "Cat"
