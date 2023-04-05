from typing import Sequence

from PySide6 import QtWidgets


def wrap_boxlayout(*widgets: QtWidgets.QWidget | list | tuple, boxdir="v"):
    if boxdir == "v":
        layout = QtWidgets.QVBoxLayout()
        boxdir = "h"
    else:
        layout = QtWidgets.QHBoxLayout()
        boxdir = "v"

    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            layout.addLayout(w)
        elif isinstance(w, QtWidgets.QWidget):
            layout.addWidget(w)
        else:
            assert isinstance(w, (list, tuple))
            layout.addLayout(wrap_boxlayout(*w, boxdir=boxdir))
    return layout


def wrap_groupbox(name: str, *widgets: QtWidgets.QWidget | Sequence[QtWidgets.QWidget]):
    """
    Wrap widgets in a QGroupBox

    Usage:
    >>> wrap_groupbox("Tasks", [widget1, widget2])
    """
    gb = QtWidgets.QGroupBox(name)
    gb.setLayout(wrap_boxlayout(*widgets, boxdir="v"))
    return gb
