from pathlib import Path
from dataclasses import dataclass
from configparser import ConfigParser
from copy import deepcopy
import pickle

import scipy.io as sio

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np

__version__ = (0, 1, 0)

INI_FILE = "oct_labeler.ini"

config = ConfigParser()
config.read(INI_FILE)
try:
    LABELS = config.get("labeler", "labels").split(",")
except Exception as e:
    print(f"Failed to read labels from '{INI_FILE}'")
    print(e)
    LABELS = ["normal", "polyp", "cancer"]

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class WindowMixin:
    def error_dialog(self, msg: str):
        """
        Display `msg` in a popup error dialog
        """
        attr_name = "_err_dialog"
        if not hasattr(self, attr_name):
            err_dialog = QtWidgets.QErrorMessage(self)
            err_dialog.setWindowTitle("Error")
            setattr(self, attr_name, err_dialog)
        else:
            err_dialog = getattr(self, attr_name)

        err_dialog.showMessage(msg)


Labels = list[tuple[tuple[int, int], str]]


@dataclass
class OctData:
    path: str | Path  # path to the image mat file
    mat: dict  # original mat dict from sio.loadmat
    imgs: np.ndarray  # ref to image array
    labels: list[Labels]  # [[((10, 20), "normal")]]

    def save_labels(self) -> Path:
        label_path = self.get_label_fname_from_img_path(self.path)
        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)
        return label_path

    def load_labels(self):
        label_path = self.get_label_fname_from_img_path(self.path)
        with open(label_path, "rb") as fp:
            self.labels = pickle.load(fp)

    @classmethod
    def from_mat_path(cls, path: str):
        ...

    @staticmethod
    def get_label_fname_from_img_path(path: str | Path, ext=".pkl") -> Path:
        path = Path(path)
        return path.parent / (path.stem + "_label" + ext)


class LinearRegionItemClickable(pg.LinearRegionItem):
    clicked = QtCore.Signal(pg.LinearRegionItem)

    def mousePressEvent(self, e):
        self.clicked.emit(self)
        super().mousePressEvent(e)


class AppWin(QtWidgets.QMainWindow, WindowMixin):
    def __init__(self):
        super().__init__()

        ### Top horizontal Layout
        self.file_dialog_btn = QtWidgets.QPushButton("Select mat file", self)
        self.file_dialog_btn.clicked.connect(self.open_file_dialog)
        self.fname: str | Path = ""

        self.text = QtWidgets.QLabel("Welcome to OCT Image Labeler")

        ### Second horizontal layout
        self.time_dec_btn = QtWidgets.QPushButton("<", self)
        self.time_inc_btn = QtWidgets.QPushButton(">", self)
        def _save_labels():
            if self.oct_data:
                label_path = self.oct_data.save_labels()
                msg = f"Saved labels to {label_path}"
                self.status_msg(msg)
        self.save_label_btn = QtWidgets.QPushButton("Save labels", self)
        self.save_label_btn.clicked.connect(_save_labels)
        self.remove_label_btn = QtWidgets.QPushButton("Remove last touched label", self)
        self.add_label_btn = QtWidgets.QPushButton("Add label", self)
        self.add_label_btn.clicked.connect(self._add_label)

        self.label_combo_box = QtWidgets.QComboBox()
        self.label_combo_box.addItems(LABELS)
        self.label_combo_box.setCurrentText(LABELS[0])

        ### image view area
        self.imv = pg.ImageView(self, name="ImageView")
        self.imv.sigTimeChanged.connect(self._imv_time_changed)

        self.curr_label_region: LinearRegionItemClickable | None = None

        # https://github.com/pyqtgraph/pyqtgraph/issues/523
        self.imv.roi.sigRegionChanged.disconnect(self.imv.roiChanged)
        self.imv.roi.sigRegionChangeFinished.connect(self.imv.roiChanged)

        # Keep references of line regions for labels
        # so we can remove them from the ViewBox later
        self.imv_region2label: dict[pg.LinearRegionItem, str] = {}
        self.imv_region2textItem: dict[pg.LinearRegionItem, pg.TextItem] = {}

        def _remove_last_clicked_label():
            if self.curr_label_region is None:
                return
            label = self.imv_region2label.pop(self.curr_label_region)
            ti = self.imv_region2textItem.pop(self.curr_label_region)
            view_box = self.imv.getView()
            view_box.removeItem(self.curr_label_region)
            view_box.removeItem(ti)
            self.curr_label_region = None

            self._imv_linear_region_change_finished()

        self.remove_label_btn.clicked.connect(_remove_last_clicked_label)

        self.oct_data: OctData | None = None

        ### Define layout
        # top horizontal layout
        hlayout1 = QtWidgets.QHBoxLayout()
        hlayout1.addWidget(self.file_dialog_btn)
        hlayout1.addWidget(self.text)

        hlayout2 = QtWidgets.QHBoxLayout()
        hlayout2.addWidget(self.time_dec_btn)
        hlayout2.addWidget(self.time_inc_btn)
        hlayout2.addWidget(self.save_label_btn)
        hlayout2.addWidget(self.remove_label_btn)
        hlayout2.addWidget(self.add_label_btn)
        hlayout2.addWidget(self.label_combo_box)

        # main layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addLayout(hlayout1)
        layout.addLayout(hlayout2)
        layout.addWidget(self.imv)

        self.status_msg("Ready")

        ### Debug use
        # self.oct_data = self._load_oct_data("~/box/oct invivo/10 polyps a & b/raw_data_image_polypa_cut_aligned.mat")
        # if self.oct_data:
            # self._disp_image()

    def status_msg(self, msg: str):
        print(msg)
        self.statusBar().showMessage(msg)

    @QtCore.Slot()
    def open_file_dialog(self):
        # Get filename from File Dialog
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open OCT aligned Mat file", filter="Mat files (*.mat)"
        )
        if not self.fname:
            return

        # Load matfile
        self.status_msg(f"Loading {self.fname}")
        self.repaint()  # force render the status message
        QtWidgets.QApplication.setOverrideCursor(QtGui.Qt.WaitCursor)
        try:
            self.oct_data = self._load_oct_data(self.fname)
            if self.oct_data:
                # show images
                self._disp_image()
                # create LinearRegionItem if labels 
                self._imv_update_linear_regions_from_labels()
        except Exception as e:
            print(e)
            self.error_dialog("Unknown exception while reading file. Check logs.")
            self.status_msg(f"Failed to load {self.fname}")
        else:
            self.status_msg(f"Loaded {self.fname}")
        QtWidgets.QApplication.restoreOverrideCursor()

    def _load_oct_data(self, fname: str | Path) -> OctData | None:
        fname = Path(fname)
        assert fname.exists()
        mat = sio.loadmat(fname)
        keys = [s for s in mat.keys() if not s.startswith("__")]

        print(f"Available keys in data file: {keys}")
        key = "I_updated"
        if key in keys:
            scans = mat[key]
            scans = np.moveaxis(scans, -1, 0)
            assert len(scans) > 0
        else:
            self.error_dialog(
                f'Key "{key}" not found in "{Path(fname).name}". Available keys are {keys}. Please load the cut/aligned Mat file.'
            )
            return None

        oct_data = OctData(path=fname, mat=mat, imgs=scans, labels=[None] * len(scans))

        # try to load labels if they exist
        try:
            oct_data.load_labels()
        except FileNotFoundError:
            pass

        return oct_data

    def _disp_image(self):
        assert self.oct_data is not None
        self.imv.setImage(self.oct_data.imgs)

    @QtCore.Slot()
    def _add_label(self, rgn: tuple[int, int] | None = None, label: str | None = None):
        if not self.oct_data:
            return

        x_max = self.oct_data.imgs.shape[-1]

        if rgn is None:
            rgn = (0, x_max // 2)

        if label is None:
            label = self.label_combo_box.currentText()

        print(f"_add_label {rgn=} {label=}")

        viewbox = self.imv.getView()

        # add LinearRegionItem to represent label region
        lri = LinearRegionItemClickable(
            values=rgn, orientation="vertical", bounds=(0, x_max)
        )
        lri.sigRegionChangeFinished.connect(self._imv_linear_region_change_finished)
        lri.sigRegionChanged.connect(self._imv_linear_region_changed)
        lri.clicked.connect(self._update_curr_label_region)

        viewbox.addItem(lri)

        # add text label for LinearRegionItem
        ti = pg.TextItem(text=label)
        ti.setPos(rgn[0], 0)
        viewbox.addItem(ti)

        self.imv_region2label[lri] = label
        self.imv_region2textItem[lri] = ti

    @QtCore.Slot()
    def _imv_time_changed(self, ind, _):
        """
        callback for when ImageView's time changes (moved to a new image)
        """
        assert self.oct_data is not None

        # if current label is None, copy previous label
        if ind > 0 and not self.oct_data.labels[ind]:
            print("Copying last labels")
            self._imv_copy_last_label()

        self._imv_update_linear_regions_from_labels(ind)

    def _imv_copy_last_label(self):
        assert self.oct_data is not None

        ind = self.imv.currentIndex
        self.oct_data.labels[ind] = deepcopy(self.oct_data.labels[ind - 1])

    def _imv_update_linear_regions_from_labels(self, ind: int | None = None):
        """
        Update the LinearRegionItem from OctData.labels
        """
        assert self.oct_data is not None

        if ind is None:
            ind = self.imv.currentIndex

        # remove current LinearRegionItem and TextItem from the
        # view_box and from the imv_region2label cache
        view_box = self.imv.getView()
        for imv_region, ti in self.imv_region2textItem.items():
            view_box.removeItem(imv_region)
            view_box.removeItem(ti)
        self.imv_region2textItem.clear()
        self.imv_region2label.clear()

        # add current labels from oct_data
        labels: Labels = self.oct_data.labels[ind]
        if labels:
            for rgn, label in labels:
                self._add_label(rgn, label)

    @QtCore.Slot()
    def _imv_linear_region_changed(self, lnr_rgn: pg.LinearRegionItem):
        "during drag of linear region, update text item position"
        ti = self.imv_region2textItem[lnr_rgn]
        ti.setPos(lnr_rgn.getRegion()[0], 0)

    @QtCore.Slot()
    def _update_curr_label_region(self, lnr_rgn):
        "Slot to handle click on a linear region"
        self.curr_label_region = lnr_rgn

    @QtCore.Slot()
    def _imv_linear_region_change_finished(self, lnr_rgn: pg.LinearRegionItem = None):
        "update oct_data labels after dragging linear region"
        ind = self.imv.currentIndex

        # get labes for this ind
        labels = self.oct_data.labels[ind]
        if labels is None:
            self.oct_data.labels[ind] = []
            labels = self.oct_data.labels[ind]
        else:
            labels.clear()

        for lnr_rgn, label in self.imv_region2label.items():
            rgn = lnr_rgn.getRegion()
            labels.append((rgn, label))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication([])
    app.setApplicationDisplayName(f"OCT Image Labeler ({'.'.join((str(i) for i in __version__))})")

    win = AppWin()
    win.showMaximized()

    sys.exit(app.exec())
