from typing import Iterable
from PySide6 import QtWidgets
from PySide6.QtWidgets import QWidget, QBoxLayout


_WidgetOrLayout = QWidget | QBoxLayout
_WidgetsOrLayouts = _WidgetOrLayout | Iterable[_WidgetOrLayout]


def wrap_boxlayout(*widgets: _WidgetsOrLayouts, boxdir="v"):
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
        elif isinstance(w, (list, tuple)):
            layout.addLayout(wrap_boxlayout(*w, boxdir=boxdir))
        else:
            raise TypeError(f"Unknown type for widget {w}")
    return layout


def wrap_groupbox(name: str, *widgets: _WidgetsOrLayouts, boxdir="v"):
    """
    Wrap widgets in a QGroupBox

    Usage:
    >>> wrap_groupbox("Tasks", [widget1, widget2])
    """
    gb = QtWidgets.QGroupBox(name)
    gb.setLayout(wrap_boxlayout(*widgets, boxdir=boxdir))
    return gb
