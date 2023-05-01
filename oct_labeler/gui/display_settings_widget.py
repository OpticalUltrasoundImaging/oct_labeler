from PySide6 import QtWidgets, QtCore

from .qt_utils import wrap_boxlayout, wrap_groupbox


class DisplaySettingsWidget(QtWidgets.QGroupBox):
    sigToggleLogCompression = QtCore.Signal(bool)
    sigDynamicRangeChanged = QtCore.Signal()

    def __init__(self, parent=None, default_dr=50):
        super().__init__("Display", parent)

        self._log_comp_cb = QtWidgets.QCheckBox()
        self._log_comp_cb.setText("Use log compression")
        self._log_comp_cb.setDown(False)
        self._log_comp_cb.stateChanged.connect(self._toggle_log_compression)

        self._drange_lbl = QtWidgets.QLabel()
        self._drange_lbl.setText("Dynamic range")
        self._drange_lbl.setEnabled(False)
        self._drange_sb = QtWidgets.QSpinBox()
        self._drange_sb.setMaximum(200)
        self._drange_sb.setMinimum(5)
        self._drange_sb.setValue(default_dr)
        self._drange_sb.setEnabled(False)
        self._old_dr = default_dr
        self._drange_sb.editingFinished.connect(self._handle_drange_edit_finished)

        _log_compression_gb = wrap_groupbox(
            "Log compression",
            self._log_comp_cb,
            [self._drange_lbl, self._drange_sb],
        )

        self.setLayout(wrap_boxlayout(_log_compression_gb, boxdir="v"))

    @QtCore.Slot()
    def _toggle_log_compression(self, check_state: int):
        checked = check_state != 0
        self._drange_lbl.setEnabled(checked)
        self._drange_sb.setEnabled(checked)
        self.sigToggleLogCompression.emit(checked)

    @QtCore.Slot()
    def _handle_drange_edit_finished(self):
        "Prevent emit when the value didn't change."
        new_dir = self.getDynamicRange()
        if new_dir != self._old_dr:
            self._old_dr = new_dir
            self.sigDynamicRangeChanged.emit()

    def logCompressionEnabled(self) -> bool:
        return self._log_comp_cb.isChecked()

    def getDynamicRange(self) -> int:
        return self._drange_sb.value()
