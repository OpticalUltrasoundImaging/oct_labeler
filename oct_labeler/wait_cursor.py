from contextlib import contextmanager

from PySide6 import QtWidgets
from PySide6.QtCore import Qt


@contextmanager
def wait_cursor():
    QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
