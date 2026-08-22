import ast
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _compile_method(method_name, extra_ns=None):
    src = (REPO_ROOT / "ver_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VERMainWindow":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    module = ast.Module(body=[child], type_ignores=[])
                    ast.fix_missing_locations(module)
                    ns = dict(extra_ns or {})
                    exec(compile(module, filename="ver_main.py", mode="exec"), ns)
                    return ns[method_name]
    raise AssertionError(f"{method_name} not found in VERMainWindow")


def test_select_data_file_uses_last_open_dir_when_available():
    calls = []

    def _get_open_file_name(_self, _title, start_path, _filters):
        calls.append(start_path)
        return ("", "")

    fn = _compile_method(
        "_select_data_file",
        extra_ns={"QFileDialog": SimpleNamespace(getOpenFileName=_get_open_file_name), "Path": Path},
    )
    stub = SimpleNamespace(_last_open_dir="/tmp/folder")

    fn(stub, initial=False)

    assert calls == ["/tmp/folder"]


def test_select_data_file_defaults_to_cwd_and_updates_last_open_dir():
    selected_file = "/tmp/folder_b/input.txt"
    calls = []

    def _get_open_file_name(_self, _title, start_path, _filters):
        calls.append(start_path)
        return (selected_file, "")

    fn = _compile_method(
        "_select_data_file",
        extra_ns={
            "QFileDialog": SimpleNamespace(getOpenFileName=_get_open_file_name),
            "Path": Path,
            "auto_detect_file_format": lambda _path: None,
        },
    )

    stub = SimpleNamespace(
        _last_open_dir=None,
        data_file=None,
        suggest_exclusion_btn=SimpleNamespace(setEnabled=lambda *_: None),
        file_label=SimpleNamespace(setText=lambda *_: None),
        display=SimpleNamespace(set_status=lambda *_: None),
        format_combo=SimpleNamespace(findText=lambda *_: -1, setCurrentIndex=lambda *_: None),
        reset_all=lambda: None,
        worker=None,
        _restart_worker_with_file=lambda: None,
    )

    fn(stub, initial=False)

    assert calls == [str(Path.cwd())]
    assert stub._last_open_dir == str(Path(selected_file).parent)

