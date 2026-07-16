import ast
import json
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


def test_settings_manager_first_run_seeds_core_editable_sections(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    settings_path = tmp_path / "user_settings.json"
    persisted = json.loads(settings_path.read_text(encoding="utf-8"))

    assert "FILE_FORMATS" in persisted
    assert "SPECIES" in persisted
    assert "ML_LOGGER" in persisted
    assert persisted["ML_LOGGER"]["observer_id"] == ""
    assert manager.settings["SPECIES"] == persisted["SPECIES"]


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


def test_settings_manager_observer_id_round_trip(monkeypatch, tmp_path):
    """Observer_ID written to ML_LOGGER section survives a settings reload."""
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    all_settings = manager.load_settings()
    all_settings.setdefault("ML_LOGGER", {})["observer_id"] = "DR_SMITH"
    manager.save_settings(all_settings)

    manager2 = SettingsManager()
    reloaded = manager2.load_settings()
    assert reloaded.get("ML_LOGGER", {}).get("observer_id") == "DR_SMITH"


def test_settings_manager_observer_id_absent_defaults_empty(monkeypatch, tmp_path):
    """When ML_LOGGER key is absent, get() with default '' returns empty string (backward-compat)."""
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    all_settings = manager.load_settings()
    # Simulate older files where ML_LOGGER was absent.
    all_settings.pop("ML_LOGGER", None)
    assert all_settings.get("ML_LOGGER", {}).get("observer_id", "") == ""


def test_settings_manager_backfills_missing_sections_without_overwriting_user_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy_settings = {
        "FILE_FORMATS": {
            "SD-card": {
                "delimiter": ","
            }
        },
        "SPECIES": {
            "Custom species": "Custom species"
        }
    }
    (tmp_path / "user_settings.json").write_text(json.dumps(legacy_settings), encoding="utf-8")

    manager = SettingsManager()
    reloaded = manager.load_settings()

    assert reloaded["FILE_FORMATS"]["SD-card"]["delimiter"] == ","
    assert "trigger_column" in reloaded["FILE_FORMATS"]["SD-card"]
    assert "LabChart" in reloaded["FILE_FORMATS"]
    assert reloaded["SPECIES"]["Custom species"] == "Custom species"
    assert "ML_LOGGER" in reloaded


def test_ml_logger_source_persists_and_prefills_observer_id():
    """Verify the source contains the patterns for observer_id persistence and prefill."""
    src = (REPO_ROOT / "ver_ml_logger.py").read_text(encoding="utf-8")
    # Prefill: settings loaded in __init__ and used to set text
    assert 'ML_LOGGER' in src
    assert 'observer_id' in src
    assert '_default_observer_id' in src
    assert 'self.observer_id_input.setText(self._default_observer_id)' in src
    # Persist: observer_id saved after successful CSV write
    assert 'all_settings.setdefault("ML_LOGGER", {})["observer_id"] = observer_id' in src
    assert 'self._settings_manager.save_settings(all_settings)' in src


def test_ml_logger_source_requires_human_reason_when_labels_differ():
    """Verify save_data validates Human_Reason before any CSV write when labels differ."""
    src = (REPO_ROOT / "ver_ml_logger.py").read_text(encoding="utf-8")
    # Validation must happen before the file is opened for writing
    assert 'human_label != comp_label and not human_reason' in src
    assert 'Human Reason Required' in src
    assert 'rows_to_write' in src
    # The CSV open must come after the validation loop (rows_to_write collected first)
    validation_pos = src.find('rows_to_write = []')
    csv_open_pos = src.find('open(csv_path, mode="a"')
    assert validation_pos != -1
    assert csv_open_pos != -1
    assert validation_pos < csv_open_pos, "Validation loop must precede CSV file open"
