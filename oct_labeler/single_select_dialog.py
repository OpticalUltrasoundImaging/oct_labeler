from typing import Optional
from PySide6 import QtWidgets


class SingleSelectDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        options: list[str] = [],
        msg="",
        gbtitle="",
        selected: list = [],
    ):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        menubar = QtWidgets.QMenuBar()
        layout.addWidget(menubar)

        text_msg = QtWidgets.QLabel(msg)
        layout.addWidget(text_msg)

        groupbox = QtWidgets.QGroupBox()
        layout.addWidget(groupbox)

        if gbtitle:
            groupbox.setTitle(gbtitle)

        btns = [QtWidgets.QRadioButton(s) for s in options]
        btns[0].setChecked(True)
        selected.append(btns[0].text())

        def make_cb(s):
            def cb():
                selected[0] = s

            return cb

        [bn.clicked.connect(make_cb(s)) for bn, s in zip(btns, options)]

        vbox = QtWidgets.QVBoxLayout()
        [vbox.addWidget(bt) for bt in btns]
        groupbox.setLayout(vbox)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)


if __name__ == "__main__":
    a = QtWidgets.QApplication()

    key = "I_updated"
    from pathlib import Path

    fname = "boo"

    selected = []
    d = SingleSelectDialog(
        options=["o1", "option 2", "laksjdlajslkajsd"],
        msg=f'Key "{key}" not found in "{Path(fname).name}".',
        gbtitle="Available keys",
        selected=selected,
    )
    ret = d.exec()

    print("Dialog returned", ret, selected[0])
