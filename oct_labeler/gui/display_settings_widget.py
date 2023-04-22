from PySide6 import QtWidgets, QtCore

from .qt_utils import wrap_boxlayout, wrap_groupbox


class DisplaySettingsWidget(QtWidgets.QGroupBox):
    sigToggleLogCompression = QtCore.Signal(bool)
    sigDynamicRangeChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__("Display", parent)

        log_compression_cb = QtWidgets.QCheckBox()
        log_compression_cb.setText("Use log compression")
        log_compression_cb.setDown(False)
        log_compression_cb.stateChanged.connect(self._toggle_log_compression)

        self._drange_label = dynamic_range_label = QtWidgets.QLabel()
        dynamic_range_label.setText("Dynamic range")
        dynamic_range_label.setEnabled(False)
        self._drange_spinbox = dynamic_range_sb = QtWidgets.QSpinBox()
        dynamic_range_sb.setValue(120)
        dynamic_range_sb.setMaximum(200)
        dynamic_range_sb.setMinimum(5)
        dynamic_range_sb.setEnabled(False)
        dynamic_range_sb.valueChanged.connect(self.sigDynamicRangeChanged)

        _log_compression_gb = wrap_groupbox(
            "Log compression",
            log_compression_cb,
            [dynamic_range_label, dynamic_range_sb],
        )

        self.setLayout(wrap_boxlayout(_log_compression_gb, boxdir="v"))

    @QtCore.Slot()
    def _toggle_log_compression(self, check_state: int):
        checked = check_state != 0
        self._drange_label.setEnabled(checked)
        self._drange_spinbox.setEnabled(checked)
        self.sigToggleLogCompression.emit(checked)
