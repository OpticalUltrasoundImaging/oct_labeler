from pathlib import Path
from copy import deepcopy
import logging

from PIL import Image
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
import pyqtgraph as pg
import scipy.io as sio
import numpy as np

from .checkable_list import CheckableList
from ..data import OctData, OctDataHdf5, AREA_LABELS
from .. import __version__
from .single_select_dialog import SingleSelectDialog
from .wait_cursor import WaitCursor

from .qt_utils import wrap_boxlayout, wrap_groupbox


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)

APP_NAME = f"OCT Image Labeler ({__version__})"

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


class AppWin(QtWidgets.QMainWindow, WindowMixin):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)

        # flag to mark if there are unsaved changes
        self.dirty: bool = False

        ### Top horizontal Layout
        file_dialog_btn = QtWidgets.QPushButton("&Load data file", self)
        file_dialog_btn.clicked.connect(self.open_file_dialog)
        self.fname: str | Path = ""

        self.text_msg = QtWidgets.QLabel("Welcome to OCT Image Labeler")

        ############################
        ### Second horizontal layout
        ############################

        ########################
        ### Navigation GroupBox
        ########################
        # Area select combobox
        self._area_label = area_label = QtWidgets.QLabel()
        area_label.setEnabled(False)
        area_label.setText("Current area:")
        area_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.area_select = QtWidgets.QComboBox()
        self.area_select.setEnabled(False)
        self.area_select.currentIndexChanged.connect(self._area_changed)

        # data select combobox (I_imgs, mu_imgs, etc.)
        self._dataselect_label = dataselect_label = QtWidgets.QLabel()
        dataselect_label.setEnabled(False)
        dataselect_label.setText("Current data:")
        dataselect_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.data_select = QtWidgets.QComboBox()
        self.data_select.setEnabled(False)
        self.data_select.currentTextChanged.connect(self._data_select_changed)

        time_dec_btn = QtWidgets.QPushButton("&Back", self)
        time_dec_btn.clicked.connect(lambda: self.oct_data and self.imv.jumpFrames(-1))
        time_inc_btn = QtWidgets.QPushButton("&Forward", self)
        time_inc_btn.clicked.connect(lambda: self.oct_data and self.imv.jumpFrames(1))

        export_img_btn = QtWidgets.QPushButton("&Export Image", self)
        export_img_btn.clicked.connect(self._export_image)

        if OCT_LABELER_DEBUG:
            debug_btn = QtWidgets.QPushButton("Breakpoint")
            debug_btn.clicked.connect(self.breakpoint)

            self.nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                [dataselect_label, self.data_select],
                time_dec_btn,
                time_inc_btn,
                export_img_btn,
                debug_btn,
            )
        else:
            self.nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                [dataselect_label, self.data_select],
                time_dec_btn,
                time_inc_btn,
                export_img_btn,
            )

        #####################
        ### Display GroupBox
        #####################
        self.disp_settings = DisplaySettingsWidget()

        ###################
        ### Labels GroupBox
        ###################
        save_label_btn = QtWidgets.QPushButton("&Save labels", self)
        save_label_btn.clicked.connect(self._save_labels)
        duplicate_labels_btn = QtWidgets.QPushButton("&Copy last labels", self)
        duplicate_labels_btn.clicked.connect(self._imv_copy_last_label)
        remove_label_btn = QtWidgets.QPushButton("&Delete last touched label", self)
        add_label_btn = QtWidgets.QPushButton("&Add label", self)
        add_label_btn.clicked.connect(self._add_label)

        self.label_list = CheckableList(LABELS)

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

        w1, h1 = _calc_ListWidget_size(self.label_list)
        w2, h2 = _calc_ListWidget_size(polyp_type_list)

        _max_height = max(h1, h2)
        _max_width = max(w1, w2)

        self.label_list.setMaximumHeight(_max_height)
        self.polyp_type_list.setMaximumHeight(_max_height)

        self.label_list.setMaximumWidth(_max_width)
        self.polyp_type_list.setMaximumWidth(_max_width)

        self.labels_gb = wrap_groupbox(
            "Labels",
            save_label_btn,
            add_label_btn,
            duplicate_labels_btn,
            remove_label_btn,
        )

        ###################
        ### image view area
        ###################
        self.imv = pg.ImageView(name="ImageView")
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
        self.curr_area = 1

        #################
        ### Main layout
        #################
        # top horizontal layout
        hlayout1 = wrap_boxlayout(file_dialog_btn, self.text_msg, boxdir="h")
        hlayout2 = wrap_boxlayout(
            self.nav_gb,
            self.disp_settings,
            self.labels_gb,
            self.label_list,
            self.polyp_type_list,
            boxdir="h",
        )

        # main layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        w.setLayout(wrap_boxlayout(hlayout1, hlayout2, self.imv, boxdir="v"))

        self.status_msg("Ready")

    def _area_changed(self, idx: int):
        """
        Handle when the area_select QComboBox is changed (by the user or programmatically).

            The item's index is passed or -1 if the combobox becomes empty or the currentIndex was reset.
        """
        if idx == -1:  # updated programmatically
            return

        self.status_msg(f"Loading Area {idx + 1}")
        self.curr_area = idx
        self._after_load_show()

    def _data_select_changed(self, txt: str):
        if not txt:
            return

        if isinstance(self.oct_data, OctDataHdf5):
            self.status_msg(f"Loading data {txt}")
            self.oct_data.set_key(txt)
            self._after_load_show()

    def status_msg(self, msg: str):
        """
        Display a msg in the bottom status bar of the main window
        """
        logging.info("status_msg: " + msg)
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
                # Setup all states for HDF5 data

                self.oct_data = self._load_oct_data_hdf5(self.fname)
                n_areas = self.oct_data.n_areas

                self.area_select.clear()
                self.area_select.addItems([str(i + 1) for i in range(n_areas)])
                self.area_select.setEnabled(True)
                self._area_label.setEnabled(True)

                self.data_select.clear()
                self.data_select.addItems(self.oct_data.get_keys())
                self.data_select.setEnabled(True)

                self._dataselect_label.setEnabled(True)

            else:  # .mat
                self.oct_data = self._load_oct_data_mat(self.fname)
                self._after_load_show()

                self._area_label.setEnabled(False)
                self.area_select.setEnabled(False)

                self._dataselect_label.setEnabled(False)
                self.data_select.setEnabled(False)

        except Exception as e:
            import traceback

            print(e)
            traceback.print_stack()
            self.error_dialog("Unknown exception while reading file. Check logs.")
            self.status_msg(f"Failed to load {self.fname}")
        else:
            if isinstance(self.oct_data, OctDataHdf5):
                self.status_msg(
                    f"Loaded {self.fname} area={self.curr_area + 1} {self.oct_data.imgs[self.curr_area].shape}"
                )
            elif isinstance(self.oct_data, OctData):
                self.status_msg(f"Loaded {self.fname} {self.oct_data.imgs.shape}")

        self.text_msg.setText("Opened " + self.fname)

    @QtCore.Slot()
    def _export_image(self):
        frame_idx = int(self.imv.currentIndex)

        if isinstance(self.oct_data, OctDataHdf5):  # HDF5 version
            area_idx = self.curr_area
            img = self.oct_data.imgs[area_idx][frame_idx]  # type: ignore
            pid = self.oct_data.hdf5path.parent.stem  # type: ignore
        else:  # Old mat format
            area_idx = 0
            img = self.oct_data.imgs[frame_idx]  # type: ignore
            pid = self.oct_data.path.parent.stem.replace(" ", "-")  # type: ignore

        path = Path.home() / "Desktop"
        if isinstance(self.oct_data, OctDataHdf5):  # HDF5 version
            path /= (
                f"export_p{pid}_a{area_idx + 1}_f{frame_idx}_{self.oct_data.key}.png"
            )
        else:
            path /= f"export_p{pid}_a{area_idx + 1}_f{frame_idx}.png"

        if len(img.shape) == 2:
            pil_img = Image.fromarray(img, "L")
        else:
            pil_img = Image.fromarray(img, "RGB")

        pil_img.save(path)
        self.status_msg(f"Exported image to {path}")

    def _after_load_show(self):
        assert self.oct_data is not None

        # show images
        self._disp_image()
        # create LinearRegionItem if labels
        self._imv_update_linear_regions_from_labels()

    def _load_oct_data_hdf5(self, fname: str | Path):
        fname = Path(fname)
        assert fname.exists()

        with WaitCursor():
            hdf5_data = OctDataHdf5(fname)

        return hdf5_data

    def _load_oct_data_mat(self, mat_path: str | Path) -> OctData | None:
        mat_path = Path(mat_path)
        assert mat_path.exists()

        with WaitCursor():
            mat = sio.loadmat(mat_path)

        keys = [s for s in mat.keys() if not s.startswith("__")]

        key = "I_updated"
        if key not in keys:
            d = SingleSelectDialog(
                msg=f'Key "{key}" not found in "{Path(mat_path).name}".',
                options=keys,
                gbtitle="Keys",
            )
            ret = d.exec()
            key = d.get_selected()

            if ret:
                print(f"Using {key=}")
            else:
                self.error_dialog(
                    f'Key "{key}" not found in "{Path(mat_path).name}". Available keys are {keys}. Please load the cut/aligned Mat file.'
                )
                return None

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0

        return OctData.from_mat_path(mat_path, _imgs=scans)

    def _save_labels(self):
        if self.oct_data:
            label_path = self.oct_data.save_labels()
            self.dirty = False
            msg = f"Saved labels to {label_path}"
            self.status_msg(msg)

    def _disp_image(self):
        assert self.oct_data is not None
        if isinstance(self.oct_data, OctDataHdf5):
            with WaitCursor():
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

        if isinstance(self.oct_data, OctDataHdf5):
            x_max = self.oct_data.imgs[self.curr_area].shape[-1]
        else:
            x_max = self.oct_data.imgs.shape[-1]

        if rgn is None:
            rgn = (0, x_max // 2)
        else:
            rgn = int(rgn[0]), int(rgn[1])

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

        # print(f"_add_label {rgn=} {label=}")
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

        if isinstance(self.oct_data, OctDataHdf5):
            labels: AREA_LABELS = self.oct_data.labels[self.curr_area]  # ref
        else:
            labels: AREA_LABELS = self.oct_data.labels  # ref

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
        if isinstance(self.oct_data, OctDataHdf5):
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
        if isinstance(self.oct_data, OctDataHdf5):
            labels = self.oct_data.labels[self.curr_area][ind]
        else:
            labels = self.oct_data.labels[ind]

        if labels is None:
            if isinstance(self.oct_data, OctDataHdf5):
                self.oct_data.labels[self.curr_area][ind] = []
                labels = self.oct_data.labels[self.curr_area][ind]
            else:
                self.oct_data.labels[ind] = []
                labels = self.oct_data.labels[ind]
        else:
            labels.clear()

        assert labels is not None
        for lnr_rgn, label in self.imv_region2label.items():
            rgn = lnr_rgn.getRegion()
            rgn = (round(rgn[0]), round(rgn[1]))  # type: ignore
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


def gui_main():
    import sys
    import os

    if os.name == "nt":
        # Dark mode for Windows
        sys.argv += ["-platform", "windows:darkmode=2"]

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationDisplayName(APP_NAME)

    if os.name == "posix":
        # Dark mode for posix
        # https://doc.qt.io/qt-6/qpalette.html
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        p = app.palette()
        QPalette = QtGui.QPalette
        QColor = QtGui.QColor
        # Background
        p.setColor(QPalette.Window, QColor(53, 53, 53))
        p.setColor(QPalette.Base, QColor(53, 53, 53))
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.Highlight, QColor(142, 45, 197))
        # Foreground
        p.setColor(QPalette.WindowText, QColor(255, 255, 255))
        p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        p.setColor(QPalette.Text, QColor(255, 255, 255))

        p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(255, 255, 255, 128))
        p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(255, 255, 255, 128))
        p.setColor(QPalette.Disabled, QPalette.Text, QColor(255, 255, 255, 128))

        app.setPalette(p)

    win = AppWin()
    win.showMaximized()

    sys.exit(app.exec())
