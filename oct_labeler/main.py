from typing import Sequence
from pathlib import Path
from copy import deepcopy

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
import pyqtgraph as pg
import scipy.io as sio
import numpy as np

from oct_labeler.checkable_list import CheckableList
from oct_labeler.oct_data import OctData, OctDataHdf5
from oct_labeler.version import __version__
from oct_labeler.single_select_dialog import SingleSelectDialog
from oct_labeler.wait_cursor import wait_cursor

from oct_labeler.qt_utils import wrap_boxlayout, wrap_groupbox


LABELS = ["normal", "polyp", "cancer", "scar", "other"]
POLYP_TYPES = [
    ("TA", "Tubular adenoma"),
    ("TVA", "Tubulovillous adenoma"),
    ("VA", "Villous adenoma"),
    ("HP", "Hyperplastic polyp"),
    ("SSP", "Sessile serrated polyp"),
    "Adenocarcinoma",
]

pg.setConfigOption("imageAxisOrder", "row-major")
# pg.setConfigOption("background", "w")
# pg.setConfigOption("foreground", "k")

OCT_LABELER_DEBUG = False


class WindowMixin:
    def error_dialog(self, msg: str):
        """
        Display `msg` in a popup error dialog
        """
        err_dialog = QtWidgets.QErrorMessage()
        err_dialog.setWindowTitle("Error")
        err_dialog.showMessage(msg)
        err_dialog.exec()


LRI_brush = pg.mkBrush((0, 255, 0, 10))
LRI_hoverBrush = pg.mkBrush((0, 255, 0, 30))


class LinearRegionItemClickable(pg.LinearRegionItem):
    clicked = QtCore.Signal(pg.LinearRegionItem)

    def mousePressEvent(self, e):
        self.clicked.emit(self)
        super().mousePressEvent(e)


class AppWin(QtWidgets.QMainWindow, WindowMixin):
    def __init__(self):
        super().__init__()

        # flag to mark if there are unsaved changes
        self.dirty: bool = False

        ### Top horizontal Layout
        file_dialog_btn = QtWidgets.QPushButton("&Load data file", self)
        file_dialog_btn.clicked.connect(self.open_file_dialog)
        self.fname: str | Path = ""

        self.text_msg = QtWidgets.QLabel("Welcome to OCT Image Labeler")

        ### Second horizontal layout
        area_label = QtWidgets.QLabel()
        area_label.setText("Current area:")
        area_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.area_select = QtWidgets.QComboBox()
        self.area_select.setDisabled(True)
        self.area_select.currentIndexChanged.connect(self._area_changed)

        toggle_binimg_btn = QtWidgets.QPushButton("&Toggle bin image", self)
        self._is_bin_img = False
        toggle_binimg_btn.clicked.connect(
            lambda: self._toggle_binimg(not self._is_bin_img)
        )

        time_dec_btn = QtWidgets.QPushButton("&Back", self)
        time_dec_btn.clicked.connect(lambda: self.oct_data and self.imv.jumpFrames(-1))
        time_inc_btn = QtWidgets.QPushButton("&Forward", self)
        time_inc_btn.clicked.connect(lambda: self.oct_data and self.imv.jumpFrames(1))

        if OCT_LABELER_DEBUG:
            debug_btn = QtWidgets.QPushButton("Breakpoint")
            debug_btn.clicked.connect(self.breakpoint)

            nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                toggle_binimg_btn,
                time_dec_btn,
                time_inc_btn,
                debug_btn,
            )
        else:
            nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                toggle_binimg_btn,
                time_dec_btn,
                time_inc_btn,
            )

        save_label_btn = QtWidgets.QPushButton("&Save labels", self)
        save_label_btn.clicked.connect(self._save_labels)
        duplicate_labels_btn = QtWidgets.QPushButton("&Copy last labels", self)
        duplicate_labels_btn.clicked.connect(self._imv_copy_last_label)
        remove_label_btn = QtWidgets.QPushButton("&Delete last touched label", self)
        add_label_btn = QtWidgets.QPushButton("&Add label", self)
        add_label_btn.clicked.connect(self._add_label)

        label_list = CheckableList(LABELS)
        self.label_list = label_list

        _polyp_types = [i if isinstance(i, str) else i[0] for i in POLYP_TYPES]
        polyp_type_list = CheckableList(_polyp_types)
        for i in POLYP_TYPES:
            if isinstance(i, tuple):
                polyp_type_list.set_item_tooltip(i[0], i[1])
        self.polyp_type_list = polyp_type_list

        def _tmp(item: QtWidgets.QListWidgetItem):
            name = item.text()
            checked = item.checkState() == Qt.CheckState.Checked
            print(name, checked)

        polyp_type_list.itemChanged.connect(_tmp)

        def _calc_ListWidget_size(ql: QtWidgets.QListWidget) -> tuple[int, int]:
            height = ql.sizeHintForRow(0) * (ql.count() + 1)
            width = ql.sizeHintForColumn(0) + 10
            return width, height

        w1, h1 = _calc_ListWidget_size(label_list)
        w2, h2 = _calc_ListWidget_size(polyp_type_list)

        _max_height = max(h1, h2)
        _max_width = max(w1, w2)

        label_list.setMaximumHeight(_max_height)
        polyp_type_list.setMaximumHeight(_max_height)

        label_list.setMaximumWidth(_max_width)
        polyp_type_list.setMaximumWidth(_max_width)

        labels_gb = wrap_groupbox(
            "Labels",
            save_label_btn,
            add_label_btn,
            duplicate_labels_btn,
            remove_label_btn,
        )

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

        remove_label_btn.clicked.connect(self._remove_last_clicked_label)

        self.oct_data: OctData | OctDataHdf5 | None = None
        self.curr_area = 0

        ### Define layout
        # top horizontal layout
        hlayout1 = wrap_boxlayout(file_dialog_btn, self.text_msg, boxdir="h")
        hlayout2 = wrap_boxlayout(
            nav_gb, labels_gb, label_list, polyp_type_list, boxdir="h"
        )

        # main layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        w.setLayout(wrap_boxlayout(hlayout1, hlayout2, self.imv, boxdir="v"))

        self.status_msg("Ready")

    def hdf5_check(self):
        # TODO: remove after deprecating .mat
        return isinstance(self.oct_data, OctDataHdf5)

    def _area_changed(self, idx: int):
        self.status_msg(f"Loading Area {idx + 1}")
        self.curr_area = idx
        self._after_load_show()

    def status_msg(self, msg: str):
        """
        Display a msg in the bottom status bar of the main window
        """
        print("status_msg:", msg)
        self.statusBar().showMessage(msg)

    @QtCore.Slot()
    def open_file_dialog(self):
        if self.dirty and not self._handle_dirty_close():
            return

        self.dirty = False

        # Get filename from File Dialog
        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="Open OCT aligned Mat file", filter="(*.mat *.hdf5)"
        )
        if not self.fname:
            return

        # Load matfile
        self.status_msg(f"Loading {self.fname}")
        self.repaint()  # force render the status message
        try:
            if Path(self.fname).suffix == ".hdf5":
                self.oct_data = self._load_oct_data_hdf5(self.fname)
                n_areas = len(self.oct_data.imgs)
                self.area_select.clear()
                self.area_select.addItems([str(i + 1) for i in range(n_areas)])
                # This calls self._after_load_show due to the callback
                self.area_select.setDisabled(False)
            else:
                self.oct_data = self._load_oct_data_mat(self.fname)
                self._after_load_show()

        except Exception as e:
            print(e)
            self.error_dialog("Unknown exception while reading file. Check logs.")
            self.status_msg(f"Failed to load {self.fname}")

        self.text_msg.setText("Opened " + self.fname)

    def _after_load_show(self):
        assert self.oct_data is not None
        if self.hdf5_check():
            self.status_msg(
                f"Loaded {self.fname} area={self.curr_area} {self.oct_data.imgs[self.curr_area].shape}"
            )
        else:
            self.status_msg(f"Loaded {self.fname} {self.oct_data.imgs.shape}")

        # show images
        self._disp_image()
        # create LinearRegionItem if labels
        self._imv_update_linear_regions_from_labels()

    def _load_oct_data_hdf5(self, fname: str | Path):
        fname = Path(fname)
        assert fname.exists()

        with wait_cursor():
            hdf5_data = OctDataHdf5(fname)

        return hdf5_data

    def _load_oct_data_mat(self, fname: str | Path) -> OctData | None:
        fname = Path(fname)
        assert fname.exists()

        with wait_cursor():
            mat = sio.loadmat(fname)

        keys = [s for s in mat.keys() if not s.startswith("__")]

        print(f"Available keys in data file: {keys}")
        key = "I_updated"
        if key not in keys:
            d = SingleSelectDialog(
                msg=f'Key "{key}" not found in "{Path(fname).name}".',
                options=keys,
                gbtitle="Keys",
            )
            ret = d.exec()
            key = d.get_selected()

            if ret:
                print(f"Using {key=}")
            else:
                self.error_dialog(
                    f'Key "{key}" not found in "{Path(fname).name}". Available keys are {keys}. Please load the cut/aligned Mat file.'
                )
                return None

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0

        oct_data = OctData(
            path=fname,
            label_path=OctData.label_path_from_img_path(fname),
            imgs=scans,
            labels=[None] * len(scans),
        )

        # try to load labels if they exist
        try:
            oct_data.load_labels()
        except FileNotFoundError:
            pass

        return oct_data

    def _save_labels(self):
        if self.oct_data:
            label_path = self.oct_data.save_labels()
            self.dirty = False
            msg = f"Saved labels to {label_path}"
            self.status_msg(msg)

    def _disp_image(self):
        assert self.oct_data is not None
        if self.hdf5_check():
            with wait_cursor():
                self.imv.setImage(self.oct_data.imgs[self.curr_area])
        else:
            self.imv.setImage(self.oct_data.imgs)

    @QtCore.Slot()
    def _add_label(
        self, rgn: tuple[int, int] | None = None, label: str | None = None, _dirty=True
    ):
        """
        To add label without setting self.dirty, pass `_dirty = False` in the parameters
        (for switching between frames and loading existing labels).
        """
        if not self.oct_data:
            return

        if self.hdf5_check():
            x_max = self.oct_data.imgs[self.curr_area].shape[-1]
        else:
            x_max = self.oct_data.imgs.shape[-1]

        if rgn is None:
            rgn = (0, x_max // 2)

        if label is None:
            label = self.label_list.get_checked_str()
            if not label:
                self.error_dialog("Please select a label first")
                return

            if label == "polyp":
                _polyp_type = self.polyp_type_list.get_checked_str()
                if not _polyp_type:
                    self.error_dialog("Please select a polyp type")
                    return
                label += ";" + _polyp_type

        print(f"_add_label {rgn=} {label=}")
        if _dirty:
            self.dirty = True

        viewbox = self.imv.getView()

        # add LinearRegionItem to represent label region
        lri = LinearRegionItemClickable(
            values=rgn,
            orientation="vertical",
            bounds=(0, x_max),
            brush=LRI_brush,
            hoverBrush=LRI_hoverBrush,
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
        self.curr_label_region = lri

    @QtCore.Slot()
    def _remove_last_clicked_label(self):
        """
        Remove `self.curr_label_region` from the plot and from
        oct_data (handled by `_imv_linear_region_change_finished`)
        """
        if self.curr_label_region is None:
            return

        self.dirty = True

        self.imv_region2label.pop(self.curr_label_region)
        ti = self.imv_region2textItem.pop(self.curr_label_region)

        view_box = self.imv.getView()
        view_box.removeItem(self.curr_label_region)
        view_box.removeItem(ti)
        self.curr_label_region = None

        self._imv_linear_region_change_finished()

    @QtCore.Slot()
    def _imv_time_changed(self, ind, _):
        """
        callback for when ImageView's time changes (moved to a new image)
        """
        if self.oct_data is None:
            return
        self._imv_update_linear_regions_from_labels(ind)

    def _imv_copy_last_label(self):
        """
        For the current frame, try to copy the labels from the last (previous) frame.
        If the previous frame doesn't have labels, try to copy labels from the next frame.
        Otherwise do nothing.
        """
        assert self.oct_data

        ind = self.imv.currentIndex
        labels = self.oct_data.labels  # ref

        if ind > 0 and labels[ind - 1]:
            labels[ind] = deepcopy(labels[ind - 1])
        elif ind < len(labels) - 1 and labels[ind + 1]:
            labels[ind] = deepcopy(labels[ind + 1])

        # update display
        self._imv_update_linear_regions_from_labels(ind)

    def _remove_displayed_linear_regions(self):
        # remove current LinearRegionItem and TextItem from the
        # view_box and from the imv_region2label cache
        view_box = self.imv.getView()
        for imv_region, ti in self.imv_region2textItem.items():
            view_box.removeItem(imv_region)
            view_box.removeItem(ti)
        self.imv_region2textItem.clear()
        self.imv_region2label.clear()

    def _imv_update_linear_regions_from_labels(self, ind: int | None = None):
        """
        Update the LinearRegionItem from OctData.labels
        """
        if not self.oct_data:
            return

        if ind is None:
            ind = int(self.imv.currentIndex)

        self._remove_displayed_linear_regions()

        # add current labels from oct_data
        if self.hdf5_check():
            labels = self.oct_data.labels[self.curr_area][ind]
        else:
            labels = self.oct_data.labels[ind]

        if labels:
            for rgn, label in labels:
                self._add_label(rgn, label, _dirty=False)

    @QtCore.Slot()
    def _update_curr_label_region(self, lnr_rgn):
        "Slot to handle click on a linear region"
        self.curr_label_region = lnr_rgn

    @QtCore.Slot()
    def _imv_linear_region_changed(self, lnr_rgn: pg.LinearRegionItem):
        "During drag of linear region, update text item position"
        ti = self.imv_region2textItem[lnr_rgn]
        ti.setPos(lnr_rgn.getRegion()[0], 0)
        self.dirty = True

    @QtCore.Slot()
    def _imv_linear_region_change_finished(
        self, lnr_rgn: pg.LinearRegionItem | None = None
    ):
        """
        Update oct_data labels after dragging linear region
        """
        ind = self.imv.currentIndex

        # get labes for this ind
        assert self.oct_data is not None
        if self.hdf5_check():
            labels = self.oct_data.labels[self.curr_area][ind]
        else:
            labels = self.oct_data.labels[ind]

        if labels is None:
            if self.hdf5_check():
                self.oct_data.labels[self.curr_area][ind] = []
                labels = self.oct_data.labels[self.curr_area][ind]
            else:
                self.oct_data.labels[ind] = []
                labels = self.oct_data.labels[ind]
        else:
            labels.clear()

        for lnr_rgn, label in self.imv_region2label.items():
            rgn = lnr_rgn.getRegion()
            labels.append((rgn, label))

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        match (event.text(), event.isAutoRepeat()):
            case ("h", False):
                # hide current linear region labels to reveal image
                self._remove_displayed_linear_regions()
            # case ("t", False):
            # self._toggle_binimg(True)
            # self.repaint()

        return super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        match (event.text(), event.isAutoRepeat()):
            case ("h", False):
                # restore linear region labels
                self._imv_update_linear_regions_from_labels()
            # case ("t", False):
            # self._toggle_binimg(False)
            # self.repaint()

        return super().keyPressEvent(event)

    def _toggle_binimg(self, b=True):
        if self.hdf5_check():
            with wait_cursor():
                self._is_bin_img = b
                self.oct_data.imgs = (
                    self.oct_data._binimgs if b else self.oct_data._imgs
                )
                # self._disp_image()
                idx = self.imv.currentIndex
                self.imv.setImage(self.oct_data.imgs[self.curr_area])
                self.imv.setCurrentIndex(idx)
                # self._imv_time_changed(idx, None)

    def _handle_dirty_close(self) -> bool:
        """
        Pop up a dialog to ask if {save,discard,cancel}

        If {save,discard}, return True
        If {cancel}, return False
        """
        sb = QtWidgets.QMessageBox.StandardButton

        # popup message box dialog
        dl = QtWidgets.QMessageBox(self)
        dl.setText("The labels have been modified.")
        dl.setInformativeText("Do you want to save your changes?")
        dl.setStandardButtons(sb.Save | sb.Discard | sb.Cancel)
        dl.setDefaultButton(sb.Save)

        ret = dl.exec()
        if ret == sb.Save:
            self._save_labels()
            return True
        elif ret == sb.Discard:
            return True

        return False

    def closeEvent(self, event: QtGui.QCloseEvent):
        """
        Override closeEvent to handle unsaved changes
        """
        # popup dialog to remind user to save data
        if self.dirty and not self._handle_dirty_close():
            # cancel pressed. ignore closeEvent
            event.ignore()
            return

        return super().closeEvent(event)

    def breakpoint(self):
        """
        Debug slot
        """
        breakpoint()
        print("")


def main():
    import sys
    import os

    if os.name == "nt":
        # Dark mode for Windows
        sys.argv += ["-platform", "windows:darkmode=2"]

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName(
        f"OCT Image Labeler ({'.'.join((str(i) for i in __version__))})"
    )

    win = AppWin()
    win.showMaximized()

    sys.exit(app.exec())
