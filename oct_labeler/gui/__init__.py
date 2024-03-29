from pathlib import Path
from copy import deepcopy
from functools import partial
import logging

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np

from oct_labeler.gui.checkable_list import CheckableList
from oct_labeler.gui.display_settings_widget import DisplaySettingsWidget
from oct_labeler.gui.wait_cursor import WaitCursor
from oct_labeler.gui.qt_utils import wrap_boxlayout, wrap_groupbox
from oct_labeler.gui.v3d import get_3d_canvas

from oct_labeler import __version__
from oct_labeler.data import (
    AreaLabel,
    Label,
    ScanData,
    ScanDataMat,
    ScanDataHdf5,
    Span,
)
import oct_labeler.data as odata
from oct_labeler.imgproc import log_compress_par


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
logging.getLogger("numba").setLevel(logging.INFO)

APP_NAME = f"OCT Image Labeler ({__version__})"

LABELS = ["normal", "polyp", "cancer", "scar", "other"]

# list of pairs (name, description)
POLYP_TYPES = [
    ("TA", "Tubular adenoma"),
    ("TVA", "Tubulovillous adenoma"),
    ("VA", "Villous adenoma"),
    ("HP", "Hyperplastic polyp"),
    ("SSP", "Sessile serrated polyp"),
    ("Adenocarcinoma", "Adenocarcinoma"),
]

T_GRADES = [
    "T0",
    "Tis",
    "T1",
    "T2",
    "T3",
    "T4",
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)

        # flag to mark if there are unsaved changes
        self.dirty: bool = False

        ### Top horizontal Layout
        file_dialog_btn = QtWidgets.QPushButton("&Load data file", self)
        file_dialog_btn.clicked.connect(self.open_file_dialog)
        self.fname: str | Path = ""

        self.text_msg = QtWidgets.QLabel("Welcome to OCT Image Labeler")

        self._imgs_orig: np.ndarray | None = None
        self._imgs: np.ndarray | None = None

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
        time_dec_btn.clicked.connect(lambda: self.imv.jumpFrames(-1))
        time_inc_btn = QtWidgets.QPushButton("&Forward", self)
        time_inc_btn.clicked.connect(lambda: self.imv.jumpFrames(1))

        view_3d_btn = QtWidgets.QPushButton("&View 3D", self)
        view_3d_btn.clicked.connect(self._view_3d)

        export_img_btn = QtWidgets.QPushButton("&Export Image", self)
        export_img_btn.clicked.connect(self._export_image)
        export_img_stack_btn = QtWidgets.QPushButton("Export Image Stack", self)
        export_img_stack_btn.clicked.connect(self._export_image_stack)

        if OCT_LABELER_DEBUG:
            debug_btn = QtWidgets.QPushButton("Breakpoint")
            debug_btn.clicked.connect(self.breakpoint)

            self.nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                [dataselect_label, self.data_select],
                time_dec_btn,
                time_inc_btn,
                view_3d_btn,
                export_img_btn,
                export_img_stack_btn,
                debug_btn,
            )
        else:
            self.nav_gb = wrap_groupbox(
                "Navigation",
                [area_label, self.area_select],
                [dataselect_label, self.data_select],
                time_dec_btn,
                time_inc_btn,
                view_3d_btn,
                export_img_btn,
                export_img_stack_btn,
            )

        self.nav_gb.setEnabled(False)

        #####################
        ### Display GroupBox
        #####################
        self.disp_settings = DisplaySettingsWidget()
        self.disp_settings.setEnabled(False)
        self.disp_settings.sigToggleLogCompression.connect(self._toggle_dynamic_range)
        self.disp_settings.sigDynamicRangeChanged.connect(
            partial(self._toggle_dynamic_range, True)
        )
        self.disp_settings._show_radial.clicked.connect(self._show_radial_cb)
        self.disp_settings.sigOpenOffcenterGui.connect(
            lambda: self.disp_settings.fix_offcenter_callback(
                self._imgs[self.imv.currentIndex]
            )
        )

        ###################
        ### Labels GroupBox
        ###################
        save_label_btn = QtWidgets.QPushButton("&Save labels", self)
        save_label_btn.clicked.connect(self._save_labels)
        duplicate_labels_btn = QtWidgets.QPushButton("&Copy previous labels", self)
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

        self.t_grade_list = CheckableList(T_GRADES)

        def _calc_ListWidget_size(ql: QtWidgets.QListWidget) -> tuple[int, int]:
            height = ql.sizeHintForRow(0) * (ql.count() + 1)
            width = ql.sizeHintForColumn(0) + 10
            return width, height

        w1, h1 = _calc_ListWidget_size(self.label_list)
        w2, h2 = _calc_ListWidget_size(polyp_type_list)
        w3, h3 = _calc_ListWidget_size(self.t_grade_list)

        _max_height = max(h1, h2, h3)
        _max_width = max(w1, w2, w3)

        self.label_list.setMaximumHeight(_max_height)
        self.polyp_type_list.setMaximumHeight(_max_height)
        self.t_grade_list.setMaximumHeight(_max_height)

        self.label_list.setMaximumWidth(_max_width)
        self.polyp_type_list.setMaximumWidth(_max_width)
        self.t_grade_list.setMaximumWidth(_max_width)

        self.label_list.setEnabled(False)
        self.polyp_type_list.setEnabled(False)
        self.t_grade_list.setEnabled(False)

        self.labels_gb = wrap_groupbox(
            "Labels",
            save_label_btn,
            add_label_btn,
            duplicate_labels_btn,
            remove_label_btn,
        )
        self.labels_gb.setEnabled(False)

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
        self.imv_region2label: dict[pg.LinearRegionItem, Label] = {}
        self.imv_region2textItem: dict[pg.LinearRegionItem, pg.TextItem] = {}

        remove_label_btn.clicked.connect(self._remove_last_clicked_label)

        self.oct_data: ScanData | None = None
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
            self.t_grade_list,
            boxdir="h",
        )

        # main layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        w.setLayout(wrap_boxlayout(hlayout1, hlayout2, self.imv, boxdir="v"))

        self.status_msg("Ready")

    @QtCore.Slot()
    def _toggle_dynamic_range(self, on: bool):
        if on:
            assert self._imgs_orig is not None
            self._imgs_orig[
                :, :50
            ] = 1.0  # get rid of noise line at the top of the images

            dr = self.disp_settings.getDynamicRange()

            with WaitCursor():
                self._imgs = log_compress_par(self._imgs_orig, float(dr))

            logging.info(f"Applied dynamic range {dr} dB")
        else:
            self._imgs = self._imgs_orig

        frame_idx = self.imv.currentIndex
        self._after_load_show()
        self.imv.setCurrentIndex(frame_idx)

    @QtCore.Slot()
    def _show_radial_cb(self):
        assert self._imgs is not None
        img = self._imgs[self.imv.currentIndex]
        self.disp_settings.show_radial_callback(img)

    def _area_changed(self, idx: int = 0):
        """
        Handle when the area_select QComboBox is changed (by the user or programmatically).

            The item's index is passed or -1 if the combobox becomes empty or the currentIndex was reset.
        """
        if idx == -1:  # updated programmatically
            return

        self.status_msg(f"Loading Area {idx + 1}")
        self.curr_area = idx

        with WaitCursor():
            if isinstance(self.oct_data, ScanDataHdf5):
                self._imgs_orig = self.oct_data.imgs[self.curr_area]
            elif isinstance(self.oct_data, ScanDataMat):
                self._imgs_orig = self.oct_data.imgs

        # self._toggle_dynamic_range(self.disp_settings.logCompressionEnabled())
        self.disp_settings._toggle_log_compression(0)  # disable the UI
        self.disp_settings._log_comp_cb.setChecked(False)
        # self._after_load_show()

    def _data_select_changed(self, txt: str):
        if not txt:
            return
        assert self.oct_data

        self.status_msg(f"Loading data {txt}")
        self.oct_data.set_key(txt)
        self._area_changed(self.curr_area)

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
                self.oct_data = ScanDataHdf5(self.fname)
                n_areas = self.oct_data.n_areas

                self.area_select.clear()
                self.area_select.addItems([str(i + 1) for i in range(n_areas)])
                self.area_select.setEnabled(True)
                self._area_label.setEnabled(True)

                self.data_select.clear()
                self.data_select.addItems(self.oct_data.get_keys())
                self.data_select.setEnabled(True)
                self._dataselect_label.setEnabled(True)

                self._area_changed(self.curr_area)  # disp data updated here

            else:  # .mat
                self.oct_data = ScanDataMat(self.fname)

                self._area_label.setEnabled(False)
                self.area_select.setEnabled(False)

                self.data_select.clear()
                self.data_select.addItems(self.oct_data.get_keys())
                self.data_select.setEnabled(True)
                self._dataselect_label.setEnabled(True)

                self._area_changed(self.curr_area)  # disp data updated here

        except Exception as e:
            import traceback

            print(e)
            traceback.print_exc()
            self.error_dialog("Unknown exception while reading file. Check logs.")
            self.status_msg(f"Failed to load {self.fname}")
        else:
            self.status_msg(
                f"Loaded {self.fname} area={self.curr_area + 1} {self._imgs[self.curr_area].shape}"
            )
        self.nav_gb.setEnabled(True)
        self.disp_settings.setEnabled(True)
        self.labels_gb.setEnabled(True)
        self.label_list.setEnabled(True)
        self.polyp_type_list.setEnabled(True)
        self.t_grade_list.setEnabled(True)

        self.text_msg.setText("Opened " + self.fname)

    def get_area_label(self) -> AreaLabel:
        """
        Get AreaLabel for the currently area.
        For HDF5 (multi area) this is the currently active area.
        For Mat (single area), this is just the labels.
        """
        area_label: AreaLabel
        if isinstance(self.oct_data, ScanDataHdf5):
            area_label = self.oct_data.labels[f"area_{self.curr_area+1}"]
        elif isinstance(self.oct_data, ScanDataMat):
            area_label = self.oct_data.labels
        else:
            raise ValueError(f"Unknown data type: {self.oct_data}")
        return area_label

    @QtCore.Slot()
    def _view_3d(self) -> None:
        self.canvas_3d = get_3d_canvas(self._imgs)

    @QtCore.Slot()
    def _export_image(self) -> None:
        # Compute filename
        path = Path.home() / "Desktop"
        frame_idx = int(self.imv.currentIndex)
        if isinstance(self.oct_data, ScanDataHdf5):  # HDF5 version
            pid = self.oct_data.path.parent.stem  # type: ignore
            path /= f"export_p{pid}_a{self.curr_area + 1}_f{frame_idx}_{self.oct_data.key}.png"
        else:  # Old mat format
            pid = self.oct_data.path.parent.stem.replace(" ", "-")  # type: ignore
            path /= f"export_p{pid}_a1_f{frame_idx}.png"

        # save image
        self.imv.imageItem.save(str(path))
        self.status_msg(f"Exported image to {path}")

    @QtCore.Slot()
    def _export_image_stack(self) -> None:
        assert self.oct_data is not None
        name = "stack"
        path = Path.home() / "Desktop"
        if (path / name).exists():
            for i in range(10):
                newpath = path / (name + str(i))
                if (newpath).exists():
                    continue
                else:
                    path = newpath
                    break
        else:
            newpath = path / name
        self.oct_data.export_image_stack(path)

    def _after_load_show(self) -> None:
        # show images
        self.imv.setImage(self._imgs)

        # create LinearRegionItem if labels
        self._imv_update_linear_regions_from_labels()

    def _save_labels(self) -> None:
        if self.oct_data:
            label_path = self.oct_data.save_labels()
            self.dirty = False
            msg = f"Saved labels to {label_path}"
            self.status_msg(msg)

    @QtCore.Slot()
    def _add_label(
        self, xspan: Span | None = None, label: Label | None = None, _dirty=True
    ):
        """
        To add label without setting self.dirty, pass `_dirty = False` in the parameters
        (for switching between frames and loading existing labels).
        """
        if not self.oct_data:
            return

        assert self._imgs is not None
        x_max = self._imgs.shape[-1]

        if xspan is None:
            xspan = (0, x_max // 2)
        else:
            xspan = int(xspan[0]), int(xspan[1])

        if label is None:
            label = Label(name=self.label_list.get_checked_str())
            if not label["name"]:
                self.error_dialog("Please select a label first")
                return

            if label["name"] == "polyp":
                _polyp_type = self.polyp_type_list.get_checked_str()
                if not _polyp_type:
                    self.error_dialog("Please select a polyp type")
                    return
                label["histology"] = _polyp_type

            t_grade = self.t_grade_list.get_checked_str()
            if t_grade:
                label["T"] = t_grade

        # print(f"_add_label {rgn=} {label=}")
        if _dirty:
            self.dirty = True

        viewbox = self.imv.getView()

        # add LinearRegionItem to represent label region
        lri = LinearRegionItemClickable(
            values=xspan,
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
        label_str = "\n".join((f"{k}: {v}" for k, v in label.items()))
        ti = pg.TextItem(text=label_str)
        ti.setPos(xspan[0], 0)
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
    def _imv_time_changed(self, ind: int, _) -> None:
        """
        callback for when ImageView's time changes (moved to a new image)
        """
        if self.oct_data is None:
            return
        assert self._imgs is not None
        self._imv_update_linear_regions_from_labels(ind)

        if self.disp_settings._radial_disp:
            self.disp_settings.show_radial_callback(self._imgs[ind])

    def _imv_copy_last_label(self) -> None:
        """
        For the current frame, try to copy the labels from the last (previous) frame.
        If the previous frame doesn't have labels, try to copy labels from the next frame.
        Otherwise do nothing.
        """
        assert self.oct_data

        ind = self.imv.currentIndex

        labels: AreaLabel = self.get_area_label()
        annos = labels["annotations"]

        if ind > 0 and annos[ind - 1]:
            annos[ind] = deepcopy(annos[ind - 1])
        elif ind < len(annos) - 1 and annos[ind + 1]:
            annos[ind] = deepcopy(annos[ind + 1])

        # update display
        self._imv_update_linear_regions_from_labels(ind)

    def _remove_displayed_linear_regions(self) -> None:
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
        area_label: AreaLabel = self.get_area_label()
        frame_labels = area_label["annotations"][ind]

        if frame_labels:
            for anno in frame_labels:
                self._add_label(anno["xspan"], anno["label"], _dirty=False)

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

        # get frame annotations at this ind
        assert self.oct_data is not None
        area_label = self.get_area_label()
        labels = area_label["annotations"][ind]
        labels.clear()

        assert labels is not None
        for lnr_rgn, label in self.imv_region2label.items():
            rgn = lnr_rgn.getRegion()
            rgn = (round(rgn[0]), round(rgn[1]))  # type: ignore
            anno = odata.make_annotation(rgn, label)
            labels.append(anno)

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

        # close 2nd window showing Warp
        self.disp_settings.show_radial_callback(None)

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

    # Fix display not set on linux
    if os.name == "posix" and os.environ.get("DISPLAY") is None:
        os.environ["DISPLAY"] = ":1"

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
