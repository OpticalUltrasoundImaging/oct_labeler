from typing import TypeGuard

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal


def is_str_list(l: list) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in l)


class CheckableList(QtWidgets.QListWidget):
    """
    Create a QListWidget with checkable items.

    emits stateChanged when any items are checked/unchecked
    """

    Checked = Qt.CheckState.Checked
    Unchecked = Qt.CheckState.Unchecked

    def __init__(self, items: list[str] | list[tuple[str, bool]] = []):
        super().__init__()
        _size_policy = self.sizePolicy()
        _size_policy.setVerticalPolicy(QtWidgets.QSizePolicy.Minimum)
        self.setUniformItemSizes(True)

        self._items_l: list[QtWidgets.QListWidgetItem] = []
        self._items_d: dict[str, QtWidgets.QListWidgetItem] = {}

        # note: calling _item.setCheckState makes the item checkable
        if is_str_list(items):
            for name in items:
                _item = QtWidgets.QListWidgetItem(name)
                _item.setCheckState(self.Unchecked)
                self._items_l.append(_item)
                self._items_d[name] = _item
                self.addItem(_item)
        else:
            for name, state in items:
                _item = QtWidgets.QListWidgetItem(name)
                _item.setCheckState(self.Checked if state else self.Unchecked)
                self._items_l.append(_item)
                self._items_d[name] = _item
                self.addItem(_item)

    def set_item_tooltip(self, name: str, txt: str):
        if _item := self._items_d.get(name):
            _item.setToolTip(txt)
        else:
            raise ValueError(
                f"Attemping to set tooltip to item that doesn't exist: {name}"
            )

    def get_state(self, name: str) -> bool:
        """
        Get the checkState of the item with name `name`
        """
        if _item := self._items_d.get(name):
            return _item.checkState() == self.Checked
        raise ValueError(f"Trying to get state of an item that doesn't exist: {name}")

    def get_states(self) -> list[tuple[str, bool]]:
        """
        Get checkState of all items, e.g. `[("hello", True), ("world", False)]`
        """
        return [
            (_item.text(), _item.checkState() == self.Checked)
            for _item in self._items_l
        ]

    def reset_states(self):
        """
        Set checkState of all items to be Unchecked
        """
        for _item in self._items_l:
            _item.setCheckState(self.Unchecked)

    def set_state(self, name: str, state: bool):
        if _item := self._items_d.get(name):
            _item.setCheckState(self.Checked if state else self.Unchecked)
        else:
            raise ValueError(
                f"Trying to set state of an item that doesn't exist: {name}"
            )

    def set_states(self, items: list[tuple[str, bool]]):
        self.reset_states()
        for name, state in items:
            self.set_state(name, state)

    def get_checked_str(self) -> str:
        """
        Return a comma-separated-string of all checked labels
        """
        return ",".join((i.text() for i in self._items_l if i.checkState() == self.Checked))


if __name__ == "__main__":

    class AppWin(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()

            w = QtWidgets.QWidget()
            self.setCentralWidget(w)
            layout = QtWidgets.QVBoxLayout(w)

            cl = CheckableList([("Hello", True), ("World", False)])
            layout.addWidget(cl)
            self.cl = cl

            btn = QtWidgets.QPushButton("Break")
            btn.clicked.connect(self.break_here)
            layout.addWidget(btn)

        def break_here(self):
            breakpoint()
            print("")

    import sys

    app = QtWidgets.QApplication([])
    win = AppWin()
    win.show()
    sys.exit(app.exec())
