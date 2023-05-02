from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QPixmap
import cv2
import numba as nb
import numpy as np


from .qt_utils import wrap_boxlayout, wrap_groupbox


def qimg_from_np(img: np.ndarray) -> QImage:
    Format = QImage.Format
    img = np.ascontiguousarray(img, dtype=np.uint8)
    breakpoint()
    if len(img.shape) == 2:
        format = Format.Format_Grayscale8
    else:
        assert len(img.shape) == 3
        assert img.shape[-1] == 3
        format = Format.Format_RGB888
    return QImage(img.data, img.shape[1], img.shape[0], format)


def polar2cart(img, pad: int = 250, scale=1.0):
    """
    Polar (linear) to cartesian (circular) image.
    pad: padding at the top to compensate for probe radius
    scale: default 1.0. Use smaller to get smaller image output
    """
    img = cv2.copyMakeBorder(img, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
    h, w = img.shape[:2]
    r = round(min(h, w) * scale)
    sz = r * 2
    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS
    img = cv2.warpPolar(img.T, (sz, sz), (r, r), r, flags)
    return img


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
        cv2.imwrite("cart.png", cart)
        # qimg = qimg_from_np(cart)
        # qimg.save("wtf.png")
        qimg = QImage("cart.png")
        self.pic.setPixmap(QPixmap.fromImage(qimg))


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

        self._show_warp = QtWidgets.QPushButton()
        self._show_warp.setText("Show warp")

        _log_compression_gb = wrap_groupbox(
            "Log compression",
            self._log_comp_cb,
            [self._drange_lbl, self._drange_sb],
        )

        self._warp_pad_lbl = QtWidgets.QLabel()
        self._warp_pad_lbl.setText("Padding top (px)")
        self._warp_pad_sb = QtWidgets.QSpinBox()
        self._warp_pad_sb.setMinimum(0)
        self._warp_pad_sb.setMaximum(1000)
        self._warp_pad_sb.editingFinished.connect

        _warp_gb = wrap_groupbox(
            "Warp polar",
            [self._warp_pad_lbl, self._warp_pad_sb],
            self._show_warp,
        )

        self.setLayout(wrap_boxlayout(_log_compression_gb, _warp_gb, boxdir="v"))

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

    def logCompressionEnabled(self) -> bool:
        return self._log_comp_cb.isChecked()

    def getDynamicRange(self) -> int:
        return self._drange_sb.value()

    def show_warp_callback(self, img: np.ndarray):
        if self._warp_disp is None:
            self._warp_disp = WarpDisp(img)
        else:
            self._warp_disp.update_img(img)
        self._warp_disp.show()


@nb.njit(
    (nb.float64[:, :, ::1], nb.float64),
    nogil=True,
    fastmath=True,
    parallel=True,
    cache=True,
)
def log_compress_par(x, dB: float):
    "Log compression with dynamic range dB"
    maxval = 255.0
    res = np.empty(x.shape, dtype=np.uint8)
    l = len(x)
    for i in nb.prange(l):
        xmax = np.percentile(x[i], 99.9)
        lc = 20.0 / dB * np.log10(x[i] / xmax) + 1.0
        lc = np.clip(lc, 0.0, 1.0)
        res[i] = (maxval * lc).astype(np.uint8)
    return res
