from __future__ import annotations
from typing import Callable, Iterable, Sequence, Mapping, TypeVar, Final
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import pickle
import logging

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


class OctDataHdf5:
    def __init__(self, hdf5path: str | Path, default_key="I_imgs"):
        self.hdf5path = Path(hdf5path)
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

    def _get_area(self, name: str, i: int) -> np.ndarray:
        # Slicing a h5py dataset produces an np.ndarray
        return self._hdf5file["areas"][str(i + 1)][name][...]  # type: ignore

    def set_key(self, k: str):
        "Set the key in 'areas/idx/I_img' currently used for self.imgs"
        self.key = k
        self.imgs = LazyList(self.n_areas, partial(self._get_area, k))

    def get_keys(self):
        return self._keys

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

        logging.info(f"{self.__class__.__name__}: Label file not found: {p}")
        return [None] * self.n_areas

    def save_labels(self):
        p = self.label_path
        with open(p, "wb") as fp:
            pickle.dump(self._labels, fp)
        return p


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
        else:
            self.labels = [None] * len(self.imgs)

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
        if _imgs is not None:
            oct_data._imgs = _imgs
        oct_data.load_labels()
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


if __name__ == "__main__":
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
            self.assertEqual(
                mv_one(((0, 10), "a"), dx=10, xlim=2000), (((10, 20), "a"),)
            )

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

    unittest.main()
