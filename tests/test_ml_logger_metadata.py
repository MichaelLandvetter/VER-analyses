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

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "writerow"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            values = [elt.value for elt in node.args[0].elts if isinstance(elt, ast.Constant)]
            if "Block" in values and "Human_Label" in values:
                csv_headers = values

    assert table_headers is not None
    assert csv_headers is not None
    assert '"Human Validation"' in src
    assert '"Computer Reason"' in src
    assert "filename" not in table_headers
    assert "species" not in table_headers
    assert "Source file:</b>" in src
    assert "Species:</b>" in src
    assert csv_headers[-2:] == ["filename", "species"]
    assert "self.filename, self.species" in src


def test_settings_manager_default_metadata_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = SettingsManager()
    assert manager.default_settings["METADATA_CONFIG"]["species"] == ""
    assert manager.settings["METADATA_CONFIG"]["species"] == ""
