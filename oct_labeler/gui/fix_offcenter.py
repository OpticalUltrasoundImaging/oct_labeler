from typing import NamedTuple
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QImage, QPainter, QPen
import numpy as np
import cv2

from oct_labeler.imgproc import polar2cart, cart2polar


def qimg_from_np(img: np.ndarray):
    assert img.dtype == np.uint8
    img = np.ascontiguousarray(img)
    if len(img.shape) == 2:  # grayscale
        h, w = img.shape
        bytes_per_line = w
        return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

    h, w, ch = img.shape
    bytes_per_line = w * ch
    return QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)


def shift_img(img: np.ndarray, dx: int = 0, dy: int = 0):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy] = 0
    elif dy < 0:
        img[dy:] = 0
    if dx > 0:
        img[:, :dx] = 0
    if dx < 0:
        img[:, dx:] = 0
    return img


def correct_offcenter(img: np.ndarray, dx: float, dy: float, pad=250, scale=1.0):
    """
    `dx`, `dy`, computed using circ image scaled at `scale`.
    """
    c = polar2cart(img, pad, scale=1.0)  # always use 1.0 scale here
    cshift = shift_img(c, round(dx / scale), round(dy / scale))
    return cart2polar(cshift, (img.shape[1], img.shape[0]), pad)


class CircleInfo(NamedTuple):
    c: QPointF  # center
    r: float  # radius


class Canvas(QtWidgets.QWidget):
    sigUpdateOffset = QtCore.Signal(float, float)

    def __init__(self, image: QImage | np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = self.moving = False
        self.circle_info: list[CircleInfo] = []

        self.txt = QtWidgets.QLabel(self)
        self.txt.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.update_image(image)

        self._msg_end = ""

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, self.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.reset)

    def update_image(self, image: QImage | np.ndarray):
        if isinstance(image, np.ndarray):
            image = qimg_from_np(image)
        self.image = image
        self.setFixedSize(self.image.width(), self.image.height())
        self.txt.setFixedSize(self.image.width(), self.image.height())

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
            self.circle_info.append(CircleInfo(event.position(), 0))
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position()
            self.moving = True
            c = self.circle_info.pop().c
            r = (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2
            r = r**0.5
            self.circle_info.append(CircleInfo(c, r))
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = self.moving = False
            self.update()

    def move_circle(self, dx: float, dy: float):
        if self.circle_info:
            c = self.circle_info[-1].c
            if dx != 0:
                c.setX(c.x() + dx)
            if dy != 0:
                c.setY(c.y() + dy)
            self.update()

    def resize_circle(self, dr: float):
        if self.circle_info:
            ci = self.circle_info.pop()
            self.circle_info.append(CircleInfo(ci.c, ci.r + dr))
            self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        match event.key(), event.modifiers():
            # shift circle
            case (Qt.Key.Key_Left | Qt.Key.Key_H, Qt.KeyboardModifier.NoModifier):
                self.move_circle(-1, 0)
            case (Qt.Key.Key_Right | Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier):
                self.move_circle(1, 0)
            case (Qt.Key.Key_Up | Qt.Key.Key_K, Qt.KeyboardModifier.NoModifier):
                self.move_circle(0, -1)
            case (Qt.Key.Key_Down | Qt.Key.Key_J, Qt.KeyboardModifier.NoModifier):
                self.move_circle(0, 1)

            # resize radius
            case (Qt.Key.Key_Up | Qt.Key.Key_K, Qt.KeyboardModifier.ShiftModifier):
                self.resize_circle(1)
            case (Qt.Key.Key_Down | Qt.Key.Key_J, Qt.KeyboardModifier.ShiftModifier):
                self.resize_circle(-1)

            case (Qt.Key.Key_Enter | Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier):
                dx, dy = self.calc_offset()
                self.sigUpdateOffset.emit(dx, dy)

        return super().keyPressEvent(event)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = event.rect()
        qp.drawImage(rect, self.image, rect)
        for c, r in self.circle_info:
            if r == 0:
                self.draw_point(qp, c)
            else:
                self.draw_circle(qp, c, r)

        self.update_txt()

    def calc_offset(self) -> tuple[float, float]:
        if self.circle_info:
            c = self.circle_info[-1].c
            dx = self.image.width() / 2 - c.x()
            dy = self.image.height() / 2 - c.y()
            return dx, dy
        return 0.0, 0.0

    def update_txt(self):
        l = [f"({int(c.x())}, {int(c.y())}, r={int(r)})" for c, r in self.circle_info]

        # offset from center of the last circle
        if self.circle_info:
            dx, dy = self.calc_offset()
            self._msg_end = f" {dx=}, {dy=}"

        t = " ".join(l) + self._msg_end
        self.txt.setText(t)

    def draw_point(self, qp, c: QPointF):
        qp.setPen(QPen(Qt.green, 4))
        qp.drawPoint(c)

    def draw_circle(self, qp, c: QPointF, r: float):
        qp.setPen(QPen(Qt.green, 2, Qt.DashLine))
        qp.drawEllipse(c, r, r)

    def undo(self):
        if self.circle_info:
            self.circle_info.pop()
            self.update()

    def reset(self):
        if self.circle_info:
            self.circle_info.clear()
            self._msg_end = ""
            self.update()


class FixOffcenterGui(QtWidgets.QWidget):
    def __init__(self, image: np.ndarray | None = None):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # these QLabels are just placeholders so that QBoxLayout
        # can scale them, and we override paintEvent to draw images
        # in them
        self.disp_linear = QtWidgets.QLabel()
        self.disp_linear_fix = QtWidgets.QLabel()
        layout.addWidget(self.disp_linear)
        layout.addWidget(self.disp_linear_fix)

        self.disp_circ = None
        self.circ_scale: float = 0.5
        self.dx = 0.0
        self.dy = 0.0

        self.img_linear = None
        self.img_linear_fix = None
        self.img_circ = None

        self.qimg_linear = None
        self.qimg_linear_fix = None

        if image is None:
            btn = QtWidgets.QPushButton()
            btn.setText("Open image")
            btn.clicked.connect(self.open_file)
            layout.addWidget(btn)
        else:
            assert isinstance(image, np.ndarray)
            self.update_image(image)

    def open_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open linear image", filter="(*.png)"
        )
        if not fname:
            return

        self.update_image(cv2.imread(fname, cv2.IMREAD_GRAYSCALE))

    def update_image(self, image: np.ndarray):
        self.img_linear = image
        self.img_circ = polar2cart(self.img_linear, 250, self.circ_scale)

        self.qimg_linear = qimg_from_np(self.img_linear)
        self.qimg_linear_fix = self.qimg_linear

        self.disp_circ = Canvas(self.img_circ)
        self.disp_circ.sigUpdateOffset.connect(self.update_offset)

        self.disp_circ.show()

    @QtCore.Slot()
    def update_offset(self, dx: float, dy: float):
        assert self.img_linear is not None
        self.dx = dx
        self.dy = dy
        img_linear_fix = correct_offcenter(
            self.img_linear, dx, dy, scale=self.circ_scale
        )
        self.qimg_linear_fix = qimg_from_np(img_linear_fix)
        self.update()

    @staticmethod
    def _paint_img_in_widget(qp: QPainter, widget: QtWidgets.QWidget, img: QImage):
        img = img.scaled(widget.size(), Qt.AspectRatioMode.KeepAspectRatio)
        qp.drawImage(widget.frameGeometry(), img)

    def paintEvent(self, event: QtGui.QPaintEvent):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.qimg_linear:
            self._paint_img_in_widget(qp, self.disp_linear, self.qimg_linear)
        if self.qimg_linear_fix:
            self._paint_img_in_widget(qp, self.disp_linear_fix, self.qimg_linear_fix)
        return super().paintEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.disp_circ:
            self.disp_circ.close()
            del self.disp_circ
        return super().closeEvent(event)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = FixOffcenterGui()
    win.showMaximized()
    app.exec()
