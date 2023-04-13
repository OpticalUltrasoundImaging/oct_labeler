from __future__ import annotations
from typing import Callable, Iterable, Sequence, TypeVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle

import h5py
import numpy as np


RANGE_T = tuple[int, int]  # (1, 1)
ONE_LABEL = tuple[RANGE_T, str]
FRAME_LABEL = list[ONE_LABEL]
AREA_LABELS = list[FRAME_LABEL | None]
AREAS_LABELS = list[AREA_LABELS | None]


VT = TypeVar("VT")


class LazyList(Sequence[VT]):
    def __init__(self, n, get_func: Callable[[int], VT], lst: list[VT | None] = []):
        self.list: list[VT | None] = lst if lst else [None] * n
        self._get_func = get_func

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i: int) -> VT:
        item = self.list[i]
        if item is None:
            item = self.list[i] = self._get_func(i)
        return item

    def __setitem__(self, i: int, v: VT):
        self.list[i] = v


class OctDataHdf5:
    def __init__(self, hdf5path: str | Path):
        self.hdf5path = Path(hdf5path)
        self._hdf5file = h5py.File(hdf5path, "r")

        self.n_areas = len(self._hdf5file["areas"])  # type: ignore

        def _get_area(name: str, i: int) -> np.ndarray:
            # Slicing a h5py dataset produces an np.ndarray
            return self._hdf5file["areas"][str(i + 1)][name][...]  # type: ignore

        self._imgs = LazyList(self.n_areas, partial(_get_area, "imgs"))
        self._binimgs = LazyList(self.n_areas, partial(_get_area, "binimgs"))
        self.imgs: LazyList = self._imgs

        # labels[area_idx][img_idx]
        self._labels: AREAS_LABELS = self._load_labels()

        def _init_labels(i: int) -> AREA_LABELS:
            return [None] * len(self.imgs[i])  # type: ignore

        self.labels: LazyList[AREA_LABELS] = LazyList(
            self.n_areas, _init_labels, self._labels
        )

    @classmethod
    def from_label_path(cls, p: Path):
        return cls(p.parent / p.name.replace("_labels.pkl", ".hdf5"))

    @property
    def label_path(self):
        p = self.hdf5path
        return p.parent / (p.stem + "_labels.pkl")

    def _load_labels(self) -> AREAS_LABELS:
        p = self.label_path
        if p.exists():
            with open(self.label_path, "rb") as fp:
                lbls = pickle.load(fp)
                return lbls

        print(f"{self.__class__.__name__}: Label file not found: {p}")
        return [None] * self.n_areas

    def save_labels(self):
        p = self.label_path
        with open(p, "wb") as fp:
            pickle.dump(self._labels, fp)
        return p


def _merge_neighbours(ls: FRAME_LABEL):
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
        if prev_name != curr_name or prev_r[1] <= curr_r[0]:
            yield prev
            prev = curr
            continue

        # Merge prev and curr
        breakpoint()
        merged_r = (prev_r[0], curr_r[1])
        prev = (merged_r, prev[1])
        continue

    yield prev


@dataclass
class OctData:
    path: Path  # path to the image mat file
    labels: AREA_LABELS  # [[((10, 20), "normal")]]

    all_areas: bool = False

    @property
    def imgs(self) -> np.ndarray:
        if hasattr(self, "_imgs"):
            return self._imgs
        self._imgs = self._load_imgs(self.path)
        return self._imgs

    @property
    def label_path(self) -> Path:
        p = self.path
        return p.parent / (p.stem + "_label.pkl")

    def save_labels(self):
        label_path = self.label_path
        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)
        return label_path

    def load_labels(self):
        p = self.label_path
        if p.exists():
            with open(self.label_path, "rb") as fp:
                self.labels = pickle.load(fp)

    @classmethod
    def from_label_path(cls, label_path: str | Path):
        """
        Note: this doesn't load the images, and just load the labels for manipulation
        """
        oct_data = OctData(
            path=cls.img_path_from_label_path(label_path),
            labels=[],  # load below with .load_labels()
        )
        oct_data.load_labels()
        return oct_data

    @classmethod
    def from_mat_path(cls, p: str | Path, _imgs: np.ndarray | None = None):
        oct_data = OctData(
            path=Path(p),
            labels=[],  # load below with .load_labels()
        )
        oct_data.load_labels()
        if _imgs is not None:
            oct_data._imgs = _imgs
        return oct_data

    @staticmethod
    def _load_imgs(fname):
        import scipy.io as sio

        mat = sio.loadmat(fname)

        keys = [s for s in mat.keys() if not s.startswith("__")]
        key = "I_updated"
        assert key in keys, f"Available keys in data file: {keys}"

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0
        return scans

    @staticmethod
    def img_path_from_label_path(p: str | Path) -> Path:
        p = Path(p)
        return p.parent / (p.stem.replace("_label", "") + ".mat")

    def shift_x(self, dx: int | Iterable[int] | Callable[[int], int]):
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

        from tqdm import tqdm

        new_labels: AREA_LABELS = [None] * len(self.labels)

        def _tqdm(it):
            return tqdm(it, total=len(self.labels), desc="shift_x")

        if isinstance(dx, int):
            for i, ls in _tqdm(enumerate(self.labels)):
                if ls is not None:
                    new_labels[i] = flatten(mv_one(l, dx) for l in ls)
        elif isinstance(dx, Iterable):
            for i, (ls, _dx) in _tqdm(enumerate(zip(self.labels, dx))):
                if ls is not None:
                    new_labels[i] = flatten(mv_one(l, _dx) for l in ls)
        elif callable(dx):
            for i, ls in _tqdm(enumerate(self.labels)):
                if ls is not None:
                    _dx = dx(i)
                    new_labels[i] = flatten(mv_one(l, _dx) for l in ls)

        # merge two labels if they overlap
        for i, ls in enumerate(new_labels):
            if ls:
                new_labels[i] = sorted(_merge_neighbours(ls))

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

        return Counter(count), Counter(total_width)
