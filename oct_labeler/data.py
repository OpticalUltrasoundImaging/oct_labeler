from __future__ import annotations
from typing import Callable, Sequence, Mapping, NamedTuple, TypeVar, Final
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
import pickle
import logging

import scipy.io as sio
import h5py
import numpy as np
import cv2


RANGE_T = tuple[int, int]  # (1, 1)
ONE_LABEL = tuple[RANGE_T, str]
FRAME_LABEL = list[ONE_LABEL]
AREA_LABELS = list[FRAME_LABEL | None]
AREAS_LABELS = list[AREA_LABELS | None]


KT = TypeVar("KT")
VT = TypeVar("VT")

LABELS_EXT = "_labels.pkl"


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


class LazyDict(Mapping[KT, VT]):
    def __init__(self, get_func: Callable[[KT], VT], d: dict[KT, VT] = {}):
        self._d = d
        self._get_func = get_func

    def __len__(self):
        return len(self._d)

    def __getitem__(self, k: KT) -> VT:
        v = self._d.get(k)
        if v is None:
            v = self._d[k] = self._get_func(k)
        return v

    def __setitem__(self, k: KT, v: VT):
        self._d[k] = v


import abc


class ScanData(abc.ABC):
    @abc.abstractmethod
    def __init__(self, path: Path | str):
        ...

    @abc.abstractmethod
    def set_key(self, key: str):
        ...

    @abc.abstractmethod
    def get_keys(self) -> list[str]:
        ...

    @classmethod
    @abc.abstractmethod
    def from_label_path(cls, path: Path) -> ScanData:
        ...

    @abc.abstractmethod
    def save_labels(self) -> Path:
        ...


class ScanDataHdf5(ScanData):
    def __init__(self, hdf5path: str | Path, default_key="I_imgs"):
        self.path = Path(hdf5path)

        self._hdf5file = h5py.File(hdf5path, "r")

        self.n_areas = len(self._hdf5file["areas"])  # type: ignore
        self._areas = list(self._hdf5file["areas"].keys())  # type: ignore

        # get keys
        assert "1" in self._hdf5file["areas"]  # type: ignore
        self._keys = list(self._hdf5file["areas"]["1"].keys())  # type: ignore
        for i in range(self.n_areas):
            idx = str(i + 1)
            for k in self._keys:
                assert k in self._hdf5file["areas"][idx]  # type: ignore

        # labels[area_idx][img_idx]
        self._labels: AREAS_LABELS = self._load_labels()

        def _init_labels(i: int) -> AREA_LABELS:
            return [None] * len(self.imgs[i])  # type: ignore

        self.labels: LazyList[AREA_LABELS] = LazyList(
            self.n_areas, _init_labels, self._labels
        )

        self.key = default_key  # default key
        if self.key not in self._keys:
            self.key = self._keys[0]
            logging.info(
                f"Default key {default_key} not in HDF5 file {hdf5path}. Using {self.key}."
            )

        self.imgs = LazyList(self.n_areas, partial(self._get_area, self.key))

    def set_key(self, k: str):
        "Set the key in 'areas/idx/I_img' currently used for self.imgs"
        self.key = k
        self.imgs = LazyList(self.n_areas, partial(self._get_area, k))

    def get_keys(self):
        return self._keys

    def save_labels(self):
        p = self.label_path
        with open(p, "wb") as fp:
            pickle.dump(self._labels, fp)
        return p

    @classmethod
    def from_label_path(cls, p: Path):
        return cls(p.parent / p.name.replace(LABELS_EXT, ".hdf5"))

    @property
    def label_path(self):
        p = self.path
        return p.parent / (p.stem + LABELS_EXT)

    def _get_area(self, name: str, i: int) -> np.ndarray:
        # Slicing a h5py dataset produces an np.ndarray
        return self._hdf5file["areas"][str(i + 1)][name][...]  # type: ignore

    def _load_labels(self) -> AREAS_LABELS:
        p = self.label_path
        if p.exists():
            with open(self.label_path, "rb") as fp:
                lbls = pickle.load(fp)
                return lbls

        logging.info(f"{self.__class__.__name__}: Label file not found: {p}")
        return [None] * self.n_areas


class ScanDataMat(ScanData):
    def __init__(self, mat_path: Path | str, default_key="I_updated"):
        self.path = Path(mat_path)

        # load mat
        self._mat = sio.loadmat(mat_path)
        self._keys: list[str] = [s for s in self._mat.keys() if not s.startswith("__")]
        self.key = default_key
        if default_key not in self._keys:
            self.key = self._keys[0]

        scans = self._mat[self.key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0
        self.imgs = scans

        # load labels
        self._load_labels()

    def set_key(self, key: str):
        "Set the key in 'areas/idx/I_img' currently used for self.imgs"
        self.key = key
        scans = self._mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0
        self.imgs = scans

    def get_keys(self):
        return self._keys

    @property
    def label_path(self) -> Path:
        p = self.path
        return p.parent / (p.stem + LABELS_EXT)

    def save_labels(self, path=None) -> Path:
        label_path = self.label_path if path is None else path
        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)
        return label_path

    def update_imgs(self, imgs: np.ndarray):
        self.imgs = imgs
        self._mat[self.key] = imgs

    def save_imgs(self, path=None) -> Path:
        path = self.label_path if path is None else path

        _savemat = {}
        for k, v in self._mat.items():
            if isinstance(v, np.ndarray):
                v = np.moveaxis(v, 0, -1)
            _savemat[k] = v
        sio.savemat(path, _savemat)
        return path

    def _load_labels(self):
        "Internal helper to load labels"
        p = self.label_path
        if p.exists():
            with open(self.label_path, "rb") as fp:
                self.labels = pickle.load(fp)
        else:
            self.labels = [None] * len(self.imgs)

    @classmethod
    def from_label_path(cls, p: Path):
        return cls(p.parent / p.name.replace(LABELS_EXT, ".mat"))


class LabelCounts(NamedTuple):
    count: Counter[str]  # number of ROIs
    width: Counter[str]  # total width of the label in pixels


def count_labels(labels: AREA_LABELS):
    count: defaultdict[str, int] = defaultdict(int)
    total_width: defaultdict[str, int] = defaultdict(int)
    for l in labels:
        if not l:
            continue
        for ll in l:
            total_width[ll[1]] += abs(round(ll[0][1] - ll[0][0]))
            count[ll[1]] += 1

    return LabelCounts(count=Counter(count), width=Counter(total_width))


def _merge_neighbours(ls: FRAME_LABEL):
    ls = sorted(ls)
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
        if prev_name != curr_name or curr_r[0] - prev_r[1] > 1:
            yield prev
            prev = curr
            continue

        # Merge prev and curr
        merged_r = (prev_r[0], curr_r[1])
        prev = (merged_r, prev[1])
        continue

    yield prev


def calc_offset_x(a: np.ndarray, b: np.ndarray, yslice=None):
    if yslice is not None:
        a, b = a[yslice].astype(np.double), b[yslice].astype(np.double)
    elif yslice is None and a.shape[0] > 400:
        yslice = slice(50, 300)  # default for OCT tube images
        a, b = a[yslice].astype(np.double), b[yslice].astype(np.double)
    else:
        a, b = a.astype(np.double), b.astype(np.double)
    return round(cv2.phaseCorrelate(a, b)[0][0])


def _s(x: int, dx: int, xlim: int):
    return (round(x) + dx + xlim) % xlim


def mv_one(l: ONE_LABEL, dx: int, xlim: int) -> tuple[ONE_LABEL, ...]:
    (x1, x2), name = l
    x1, x2 = _s(x1, dx, xlim), _s(x2, dx, xlim)
    if x1 < x2:
        return (((x1, x2), name),)
    elif x1 > x2:
        if x2 == 0:
            return (((x1, xlim), name),)
        return (((x1, xlim), name), ((0, x2), name))
    raise ValueError("x1 == x2")


def shift_x(
    old_imgs: np.ndarray | list[np.ndarray],
    new_imgs: np.ndarray | list[np.ndarray],
    old_labels: AREA_LABELS,
):
    """
    Shift x for one scan area.
    """
    # (n_imgs, y, x)
    assert old_imgs[0].shape == new_imgs[0].shape
    assert len(old_imgs) == len(new_imgs)
    xlim = old_imgs[0].shape[-1]

    flatten = lambda l: sorted(x for ll in l for x in ll)

    from tqdm import tqdm

    new_labels: AREA_LABELS = [None] * len(old_labels)

    for i, ls in tqdm(enumerate(old_labels), total=len(old_labels), desc="shift_x"):
        # per frame
        if ls is not None:
            img1, img2 = old_imgs[i].astype(np.double), new_imgs[i].astype(np.double)
            dx = calc_offset_x(img1, img2)
            new_labels[i] = flatten(mv_one(l, dx, xlim) for l in ls)

    # merge two labels if they overlap
    for i, ls in enumerate(new_labels):
        if ls:
            new_labels[i] = list(_merge_neighbours(ls))

    return new_labels


import unittest


class Test_label_shift(unittest.TestCase):
    def test_mv_one(self):
        # Shift left over
        self.assertEqual(
            mv_one(((0, 10), "a"), dx=-10, xlim=2000), (((1990, 2000), "a"),)
        )

        # Shift left split
        self.assertEqual(
            mv_one(((0, 20), "a"), dx=-10, xlim=2000),
            (((1990, 2000), "a"), ((0, 10), "a")),
        )

        # Shift right
        self.assertEqual(mv_one(((0, 10), "a"), dx=10, xlim=2000), (((10, 20), "a"),))

        # Shift right over
        self.assertEqual(
            mv_one(((1990, 2000), "a"), dx=10, xlim=2000), (((0, 10), "a"),)
        )

        # Shift right split
        self.assertEqual(
            mv_one(((1980, 2000), "a"), dx=10, xlim=2000),
            (((1990, 2000), "a"), ((0, 10), "a")),
        )

    def test_merge_neighbours(self):
        # No merge. only sort
        inp = [((1990, 2000), "a"), ((0, 10), "a")]
        exp = [((0, 10), "a"), ((1990, 2000), "a")]
        self.assertEqual(list(_merge_neighbours(inp)), exp)

        # merge
        inp = [((0, 10), "a"), ((11, 20), "a")]
        exp = [((0, 20), "a")]
        self.assertEqual(list(_merge_neighbours(inp)), exp)

    def test_shift_x(self):
        # (y=10, x=20) image
        img1: Final = np.repeat(np.expand_dims(np.arange(20), 0), 10, axis=0)

        offset = 5
        img2 = np.roll(img1, offset)
        old_label: FRAME_LABEL = [((2, 5), "a")]
        new_label: FRAME_LABEL = [((2 + offset, 5 + offset), "a")]
        self.assertEqual(shift_x([img1], [img2], [old_label]), [new_label])

        offset = 3
        img2 = np.roll(img1, offset)
        old_label: FRAME_LABEL = [((15, 20), "a")]
        new_label: FRAME_LABEL = [((0, 3), "a"), ((18, 20), "a")]
        self.assertEqual(shift_x([img1], [img2], [old_label]), [new_label])


import tempfile


class TestHdf5Data(unittest.TestCase):
    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile()

        self.data = [
            {
                "imgs": np.random.random((20, 30)),
                "bin_imgs": np.random.random((10, 20)),
            },
            {
                "imgs": np.random.random((20, 30)),
                "bin_imgs": np.random.random((10, 20)),
            },
            {
                "imgs": np.random.random((20, 30)),
                "bin_imgs": np.random.random((10, 20)),
            },
        ]

        with h5py.File(self.fp.name, "w") as hdf5file:
            for i, ds in enumerate(self.data):
                group = hdf5file.create_group(f"areas/{i+1}")
                for k, v in ds.items():
                    group.create_dataset(k, data=v)

    def test_hdf5_data(self):
        d = ScanDataHdf5(self.fp.name)
        self.assertEqual(d.n_areas, len(self.data))
        self.assertEqual(d._keys, ["bin_imgs", "imgs"])

        for i, ds in enumerate(self.data):
            for k, v in ds.items():
                d.set_key(k)
                self.assertTrue(np.allclose(v, d.imgs[i]))


if __name__ == "__main__":
    unittest.main()
