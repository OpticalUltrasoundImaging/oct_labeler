from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QPixmap
import numpy as np


from .qt_utils import wrap_boxlayout, wrap_groupbox
from ..imgproc import polar2cart
from .fix_offcenter import FixOffcenterGui


def qimg_from_np(img: np.ndarray):
    if len(img.shape) == 2:  # grayscale
        h, w = img.shape
        bytes_per_line = w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

    h, w, ch = img.shape
    assert ch == 3
    bytes_per_line = w * ch
    return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


class WarpDisp(QtWidgets.QWidget):
    def __init__(self, img: np.ndarray):
        super().__init__()
        self.pic = QtWidgets.QLabel(self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pic)
        self.setLayout(layout)

        self.pad: int = 250
        self.scale: float = 0.5
        self._img = None
        self.update_img(img)

    def update_img(self, img=None):
        if img is None:
            if self._img is None:
                return
            img = self._img
        else:
            self._img = img
        cart = polar2cart(img, self.pad, self.scale)
        qimg = qimg_from_np(cart)
        self.pic.setPixmap(QPixmap.fromImage(qimg))


class DisplaySettingsWidget(QtWidgets.QGroupBox):
    sigToggleLogCompression = QtCore.Signal(bool)
    sigDynamicRangeChanged = QtCore.Signal()
    sigOpenOffcenterGui = QtCore.Signal()

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

        self._show_warp = QtWidgets.QPushButton()
        self._show_warp.setText("Show warp")

        _log_compression_gb = wrap_groupbox(
            "",
            self._log_comp_cb,
            [self._drange_lbl, self._drange_sb],
        )

        self._warp_pad_lbl = QtWidgets.QLabel()
        self._warp_pad_lbl.setText("Padding top (px)")
        self._warp_pad_sb = QtWidgets.QSpinBox()
        self._warp_pad_sb.setMinimum(0)
        self._warp_pad_sb.setMaximum(1000)
        self._warp_pad_sb.setValue(250)
        self._warp_pad_sb.editingFinished.connect(self._handle_pad_changed)

        _warp_gb = wrap_groupbox(
            "",
            self._show_warp,
            [self._warp_pad_lbl, self._warp_pad_sb],
        )

        self._fix_offcenter = QtWidgets.QPushButton()
        self._fix_offcenter.setText("Fix offcenter")
        self._fix_offcenter.clicked.connect(self._handle_fix_offcenter)
        self._fix_offcenter_gui = None

        self.setLayout(
            wrap_boxlayout(
                _log_compression_gb, _warp_gb, self._fix_offcenter, boxdir="v"
            )
        )

        self._warp_disp = None

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

    @QtCore.Slot()
    def _handle_pad_changed(self):
        pad = self._warp_pad_sb.value()
        if self._warp_disp:
            self._warp_disp.pad = pad
            self._warp_disp.update_img()

    @QtCore.Slot()
    def _handle_fix_offcenter(self):
        self.sigOpenOffcenterGui.emit()

    def fix_offcenter_callback(self, image: np.ndarray):
        self._fix_offcenter_gui = FixOffcenterGui(image)
        self._fix_offcenter_gui.showMaximized()

    def logCompressionEnabled(self) -> bool:
        return self._log_comp_cb.isChecked()

    def getDynamicRange(self) -> int:
        return self._drange_sb.value()

    def show_warp_callback(self, img: np.ndarray | None = None):
        """
        If img give, try to show img in a WarpDisp window.
        If img is None, close the WarpDisp if exists
        """
        if img is not None:
            if self._warp_disp is None:
                self._warp_disp = WarpDisp(img)
                self._warp_disp.pad = self._warp_pad_sb.value()
            else:
                self._warp_disp.update_img(img)
            self._warp_disp.show()
        else:
            if self._warp_disp is not None:
                del self._warp_disp
                self._warp_disp = None
