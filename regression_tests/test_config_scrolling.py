"""Real Qt resize checks for the shared configuration tab wrappers."""
import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PyQt5 import QtCore, QtWidgets
except ImportError:
    QtCore = QtWidgets = None


@unittest.skipIf(QtWidgets is None, "PyQt5 is required for widget resize tests")
class ConfigScrollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        path = Path(__file__).resolve().parents[1] / "patcherbot/gui/camera.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        camera = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CameraGui")
        methods = [node for node in camera.body if isinstance(node, ast.FunctionDef) and node.name in ("add_tab", "add_config_gui")]
        namespace = {"QtWidgets": QtWidgets, "Qt": QtCore.Qt, "logging": logging}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        cls.host_type = type("TabHost", (), {name: namespace[name] for name in ("add_tab", "add_config_gui")})

    def setUp(self):
        self.host = self.host_type()
        self.host.config_tab = QtWidgets.QTabWidget()
        self.host.config_tab.resize(280, 220)

    def tearDown(self):
        self.host.config_tab.close()
        self.host.config_tab.deleteLater()
        self.app.processEvents()

    def panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        for index in range(20):
            editor = QtWidgets.QLineEdit("setting {}".format(index))
            editor.setMinimumWidth(400)
            layout.addWidget(editor)
        return panel

    def test_overflow_scrolls_both_axes_and_preserves_setting_values(self):
        panel = self.panel()
        self.host.add_tab(panel, "Patch")
        self.host.config_tab.show()
        self.app.processEvents()
        scroll = self.host.config_tab.widget(0)
        self.assertIs(scroll.widget(), panel)
        self.assertEqual(scroll.frameShape(), QtWidgets.QFrame.NoFrame)
        self.assertTrue(scroll.widgetResizable())
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        self.assertGreater(scroll.horizontalScrollBar().maximum(), 0)
        last = panel.findChildren(QtWidgets.QLineEdit)[-1]
        scroll.ensureWidgetVisible(last)
        self.app.processEvents()
        self.assertGreater(scroll.verticalScrollBar().value(), 0)
        self.assertEqual(last.text(), "setting 19")
        self.host.config_tab.resize(900, 1600)
        self.app.processEvents()
        self.assertEqual(scroll.verticalScrollBar().maximum(), 0)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)

    def test_config_return_value_and_classic_first_tab_are_preserved(self):
        panels = []
        for name in ("Calibration", "Patch", "Protocols", "Experiment book"):
            result = self.host.add_config_gui(SimpleNamespace(name=name), gui_class=lambda config: self.panel())
            panels.append(result)
            self.assertIs(self.host.config_tab.widget(len(panels) - 1).widget(), result)
        classic = self.panel()
        self.host.add_tab(classic, "PatcherBot Agent", index=0)
        self.assertEqual([self.host.config_tab.tabText(i) for i in range(5)],
                         ["PatcherBot Agent", "Calibration", "Patch", "Protocols", "Experiment book"])
        self.assertIs(self.host.config_tab.widget(0).widget(), classic)
        for i, panel in enumerate(panels, start=1):
            self.assertIs(self.host.config_tab.widget(i).widget(), panel)


if __name__ == "__main__":
    unittest.main()
