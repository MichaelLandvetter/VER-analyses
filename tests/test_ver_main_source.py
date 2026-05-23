import unittest
from pathlib import Path


class VERMainSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main_path = Path(__file__).resolve().parents[1] / "ver_main.py"
        cls.source = main_path.read_text(encoding="utf-8")

    def test_file_menu_has_exit_action(self):
        self.assertIn('exit_action = QAction("Exit", self)', self.source)
        self.assertIn('exit_action.setShortcut("Ctrl+Q")', self.source)
        self.assertIn("exit_action.triggered.connect(self.close)", self.source)
        self.assertIn("file_menu.addSeparator()", self.source)
        self.assertIn("file_menu.addAction(exit_action)", self.source)

    def test_new_session_scope_clear_happens_before_scope_update(self):
        self.assertIn("if self._scope_panel_session != current_session:", self.source)
        self.assertIn("self.display.clear_scope_panel()", self.source)
        self.assertIn("self._scope_panel_session = current_session", self.source)


if __name__ == "__main__":
    unittest.main()
