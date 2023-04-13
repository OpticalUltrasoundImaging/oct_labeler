from contextlib import contextmanager

from PySide6 import QtWidgets
from PySide6.QtCore import Qt


@contextmanager
def WaitCursor(self: QtWidgets.QWidget | None = None):
    QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    if self is not None:
        self.repaint()
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
