"""
GUI utilities for loading .ui files and applying input masks on table cells.
"""

import os
import sys
from tqdm import tqdm
import subprocess
import inspect
from sys import platform as _platform

from PyQt6.QtWidgets import QStyledItemDelegate, QLineEdit
from PyQt6 import QtWidgets


def debug_trace(type="interactive"):
    """Set a tracepoint in the Python debugger that works with Qt."""
    from PyQt6.QtCore import pyqtRemoveInputHook

    if type == "interactive":
        from ipdb import set_trace
    else:
        from pdb import set_trace

    pyqtRemoveInputHook()
    set_trace()


def module_path():
    """Return this module’s directory path using inspect."""
    return os.path.dirname(inspect.getfile(inspect.currentframe()))


class UiLoader:
    """Compile .ui files in ui_interfaces/ to Python modules via pyuic6."""

    def __init__(self, path=None):
        if path is None:
            self.path = module_path() + "/ui_interfaces/"
        else:
            self.path = path
        self.count = True
        print("Loading UIs at : {0}".format(self.path))
        self.update_Ui_Interfaces()

    def update_Ui_Interfaces(self, pyuic="pyuic6"):
        """Run pyuic for each .ui file to generate *_ui_mod.py files."""
        repository = self.path
        uipathes = sorted([f for f in os.listdir(repository) if f.endswith(".ui")])

        if _platform == "linux" or _platform == "linux2" or _platform == "darwin":
            for f in tqdm(uipathes):
                pyfilename = f.split(".ui")[0] + "_ui_mod.py"
                command = [f"{pyuic}", "-x", f, "-o", pyfilename]
                query = subprocess.Popen(
                    command,
                    cwd=repository,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                (out_txt, error) = query.communicate()
                out_txt = out_txt.decode("utf-8")
        else:
            for f in tqdm(uipathes):
                pyfilename = f.split(".ui")[0] + "_ui_mod.py"
                command = [f"{pyuic}.bat", "-x", f, "-o", pyfilename]
                query = subprocess.Popen(
                    command,
                    cwd=repository,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                (out_txt, error) = query.communicate()
                out_txt = out_txt.decode("utf-8")


class MaskDelegate(QStyledItemDelegate):
    """Delegate applying a per-column input mask to QLineEdit editors."""

    def __init__(self, masks, parent=None):
        super().__init__(parent)
        self.masks = masks

    def createEditor(self, parent, option, index):
        """Create a masked QLineEdit for columns defined in self.masks."""
        editor = QLineEdit(parent)
        column = index.column()
        if column in self.masks:
            mask = self.masks[column]
            editor.setInputMask(mask)
        return editor
