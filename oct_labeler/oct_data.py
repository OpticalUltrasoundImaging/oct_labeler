from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle

import h5py
import numpy as np


RANGE_T = tuple[int, int]
NAME_T = str
ONE_LABEL = tuple[RANGE_T, NAME_T]
Labels = list[ONE_LABEL]


from typing import Callable, Generic, TypeVar

VT = TypeVar("VT")


class LazyList(Generic[VT]):
    def __init__(
        self, n, get_func: Callable[[int], VT], lst: list[VT | None] | None = None
    ):
        if lst is None:
            self._l = [None] * n
        else:
            self._l = lst
        self._get_func = get_func

    def __len__(self):
        return len(self._l)

    def __getitem__(self, i: int) -> VT:
        if self._l[i] is None:
            self._l[i] = self._get_func(i)
        return self._l[i]


from functools import partial


class OctDataHdf5:
    def __init__(self, hdf5path: str | Path):
        self.hdf5path = Path(hdf5path)
        self._hdf5file = h5py.File(hdf5path, "r")

        self.n_areas = len(self._hdf5file["areas"])

        self._imgs: list[np.ndarray | None] = [None] * self.n_areas
        self._binimgs: list[np.ndarray | None] = [None] * self.n_areas

        # labels[area_idx][img_idx]
        self._labels: list[list[Labels] | None]

        def _get_area(name: str, i: int) -> np.ndarray:
            return self._hdf5file["areas"][str(i + 1)][name][...]

        self.imgs = LazyList(self.n_areas, partial(_get_area, "imgs"))
        self.binimgs = LazyList(self.n_areas, partial(_get_area, "binimgs"))

        self.load_labels()

        def _init_labels(i: int) -> list[Labels]:
            return [None] * len(self.imgs[i])

        self.labels = LazyList(self.n_areas, _init_labels, self._labels)

    def __del__(self):
        self._hdf5file.close()

    @property
    def label_path(self):
        p = self.hdf5path
        return p.parent / (p.stem + "_labels.pkl")

    def load_labels(self):
        p = self.label_path
        if p.exists():
            with open(self.label_path, "rb") as fp:
                self._labels = pickle.load(fp)
        else:
            print(f"{self.__class__.__name__}: Label file not found: {p}")
            self._labels = [None] * self.n_areas

    def save_labels(self):
        p = self.label_path
        with open(p, "wb") as fp:
            pickle.dump(self._labels, fp)
        return p


def _merge_neighbours(ls: list[ONE_LABEL]):
    prev: ONE_LABEL | None = None
    prev = ls[0]
    for curr in ls[1:]:
        if prev is None:
            prev = curr
            continue

        prev_r = prev[0]
        curr_r = curr[0]
        prev_name = prev[1]
        curr_name = curr[1]

        # If prev and curr have different labels,
        # or if prev and curr don't overlap, return the prev label,
        # and set curr to prev.
        if prev_name != curr_name or prev_r[1] < curr_r[0] - 1:
            yield prev
            prev = curr

        # Merge prev and curr
        prev_r[1] = curr_r[1]
        continue

    yield prev


@dataclass
class OctData:
    path: str | Path  # path to the image mat file
    label_path: str | Path
    imgs: np.ndarray  # ref to image array
    labels: list[Labels]  # [[((10, 20), "normal")]]

    all_areas: bool = False

    def save_labels(self, label_path: str | Path | None = None):
        if label_path is None:
            label_path = self.label_path
        self.path = self.img_path_from_label_path(label_path)

        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)

        return label_path

    def load_labels(self, label_path: str | Path | None = None):
        if label_path is None:
            label_path = self.label_path

        # self.path = self.img_path_from_label_path(label_path)

        with open(label_path, "rb") as fp:
            self.labels = pickle.load(fp)

    @classmethod
    def from_label_path(cls, label_path: str | Path) -> OctData:
        """
        Note: this doesn't load the images, and just load the labels for manipulation
        """
        oct_data = OctData(
            path=cls.img_path_from_label_path(label_path),
            label_path=label_path,
            imgs=None,
            labels=None,
        )
        oct_data.load_labels()
        return oct_data

    @classmethod
    def from_mat_path(cls, fname: str | Path) -> OctData:
        scans = cls.load_imgs(fname)

        oct_data = OctData(
            path=fname,
            label_path=cls.label_path_from_img_path(fname),
            imgs=scans,
            labels=[None] * len(scans),
        )
        return oct_data

    @staticmethod
    def load_imgs(fname):
        import scipy.io as sio

        mat = sio.loadmat(fname)

        keys = [s for s in mat.keys() if not s.startswith("__")]
        print(f"Available keys in data file: {keys}")
        key = "I_updated"
        assert key in keys

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0
        return scans

    def load_imgs_(self):
        self.imgs = self.load_imgs(self.img_path_from_label_path(self.label_path))

    @staticmethod
    def label_path_from_img_path(path: str | Path, ext=".pkl") -> Path:
        path = Path(path)
        return path.parent / (path.stem + "_label" + ext)

    @staticmethod
    def img_path_from_label_path(label_path: str | Path) -> Path:
        label_path = Path(label_path)
        return label_path.parent / (label_path.stem.rsplit("_label", 1)[0] + ".mat")

    def shift_x(self, dx: int | list[int]):
        if self.imgs is None:
            xlim = 2000
        else:
            xlim = self.imgs.shape[-1]

        def _s(x: int, dx: int):
            return (round(x) + dx + xlim) % xlim

        def mv_one(l: ONE_LABEL, dx: int):
            (x1, x2), name = l
            x1, x2 = _s(x1, dx), _s(x2, dx)
            if x1 < x2:
                return (((x1, x2), name),)
            elif x1 > x2:
                return (((x1, xlim - 1), name), ((0, x2), name))
            raise ValueError("x1 == x2")

        flatten = lambda l: sorted(x for ll in l for x in ll)

        new_labels = [None] * len(self.labels)
        if isinstance(dx, int):
            for i, ls in enumerate(self.labels):
                if ls is not None:
                    new_labels[i] = flatten(mv_one(l, dx) for l in ls)
        else:
            for i, ls in enumerate(self.labels):
                if ls is not None:
                    new_labels[i] = flatten(mv_one(l, dx[i]) for l in ls)

        # TODO: merge two labels if they overlap
        for i, ls in enumerate(new_labels):
            new_labels[i] = tuple(_merge_neighbours(ls))

        self.labels = new_labels

    def count(self):  # const
        from collections import Counter, defaultdict

        total_width = defaultdict(int)
        count = defaultdict(int)
        for l in self.labels:
            if not l:
                continue
            for ll in l:
                total_width[ll[1]] += abs(round(ll[0][1] - ll[0][0]))
                count[ll[1]] += 1

        c = Counter(ll[1] for l in self.labels if l for ll in l)
        return Counter(count), Counter(total_width)
