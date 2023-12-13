from __future__ import annotations
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Sequence,
    Mapping,
    NamedTuple,
    TypeVar,
    Final,
    TypedDict,
    NotRequired,
)
from collections import Counter, defaultdict
from functools import partial, singledispatchmethod
from pathlib import Path
import json
import logging
import shutil

from tqdm import tqdm
import scipy.io as sio
import h5py
import numpy as np
import cv2


KT = TypeVar("KT")
VT = TypeVar("VT")

LABELS_EXT = "_labels.json"


class Label(TypedDict):
    """
    Label of an annotation
    """

    name: str
    histology: NotRequired[Any]
    T: NotRequired[Any]


Span = tuple[int, int]  # (1, 1)


class Annotation(TypedDict):
    xspan: Span
    label: Label


def make_annotation(span: Span, label: Label):
    return Annotation(xspan=span, label=label)


def annotation_sort_key(a: Annotation):
    return a["xspan"]


FrameLabel = list[Annotation]


class AreaLabel(TypedDict):
    annotations: list[FrameLabel]


MultiAreaLabel = dict[str, AreaLabel]


class LazyList(Sequence[VT]):
    def __init__(self, n, get_func: Callable[[int], VT], lst: list[VT | None] = []):
        self.list: list[VT | None] = lst if lst else [None] * n
        self._get_func = get_func

    def __len__(self):
        return len(self.list)

    @singledispatchmethod
    def __getitem__(self, _):
        raise NotImplementedError()

    @__getitem__.register
    def _(self, i: int) -> VT:
        item = self.list[i]
        if item is None:
            item = self.list[i] = self._get_func(i)
        return item

    @__getitem__.register
    def _(self, s: slice) -> list[VT]:
        res = []
        for j in range(*s.indices(len(self.list))):
            res.append(self[j])
        return res

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
    labels: Any

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
    def from_label_path(cls, path: Path | str) -> ScanData:
        ...

    @abc.abstractmethod
    def save_labels(self) -> Path:
        ...

    @abc.abstractmethod
    def export_image_stack(self, dest: Path):
        ...


class ScanDataHdf5(ScanData):
    def __init__(self, hdf5path: str | Path, default_key="I_imgs"):
        self.path = Path(hdf5path)

        self._hdf5file = h5py.File(hdf5path, "r")  # open readonly

        self.n_areas = len(self._hdf5file["areas"])  # type: ignore
        self._areas = list(self._hdf5file["areas"].keys())  # type: ignore

        # get keys
        assert "1" in self._hdf5file["areas"]  # type: ignore
        self._keys = list(self._hdf5file["areas"]["1"].keys())  # type: ignore
        for i in range(self.n_areas):
            idx = str(i + 1)
            for k in self._keys:
                assert k in self._hdf5file["areas"][idx]  # type: ignore

        self.key = default_key  # default key
        if self.key not in self._keys:
            self.key = self._keys[0]
            logging.info(
                f"Default key {default_key} not in HDF5 file {hdf5path}. Using {self.key}."
            )
        self.imgs = LazyList(self.n_areas, partial(self._get_area, self.key))

        self.labels = self.load_labels()

        # def _init_labels(i: int) -> AreaLabel:
        # return {"annotations": [[] for _ in range(len(self.imgs[i]))]}

        # self.labels: LazyList[AreaLabel] = LazyList(
        # self.n_areas, _init_labels, self._labels
        # )
        # self.labels = {f"area_{i+1}": _init_labels(i) for i in range(self.n_areas)}

    def set_key(self, k: str):
        "Set the key in 'areas/idx/I_img' currently used for self.imgs"
        self.key = k
        self.imgs = LazyList(self.n_areas, partial(self._get_area, k))

    def get_keys(self):
        return self._keys

    def save_labels(self):
        p = self.label_path
        with open(p, "w") as fp:
            json.dump(self.labels, fp)
        return p

    @classmethod
    def from_label_path(cls, p: Path | str):
        p = Path(p)
        return cls(p.parent / p.name.replace(LABELS_EXT, ".hdf5"))

    @property
    def label_path(self):
        p = self.path
        return p.parent / (p.stem + LABELS_EXT)

    def _get_area(self, name: str, i: int) -> np.ndarray:
        # Slicing a h5py dataset produces an np.ndarray
        return self._hdf5file["areas"][str(i + 1)][name][...]  # type: ignore

    def update_area(self, i: int, imgs: np.ndarray):
        self.imgs[i] = imgs

    def save_imgs(self, newpath: Path) -> Path:
        "creates a copy of the HDF5 and copies the current data to it."
        assert newpath != self.path
        shutil.copy(self.path, newpath)
        newh5 = h5py.File(newpath, "r+")  # open read write
        for i in range(self.n_areas):
            newh5["areas"][str(i + 1)][self.key][...] = self.imgs[i]  # type: ignore
        newh5.close()
        return newpath

    def load_labels(self) -> MultiAreaLabel:
        if self.label_path.exists():
            with open(self.label_path, "r") as fp:
                return json.load(fp)

        logging.info(
            f"{self.__class__.__name__}: Label file not found: {self.label_path}"
        )
        return {}

    def export_image_stack(self, dest: Path | str, ext: str = "tiff", stack=True):
        dest = Path(dest)
        if stack:
            assert ext in ("tiff", "tif"), "Only tiff support multiple images."
            for i in tqdm(range(self.n_areas)):
                fname = dest / f"{i+1}.{ext}"
                cv2.imwritemulti(str(fname), self.imgs[i])
        else:
            for i in range(self.n_areas):
                thisdir = dest / str(i + 1)
                thisdir.mkdir(exist_ok=True, parents=True)
                for j, img in tqdm(
                    enumerate(self.imgs[i]),
                    desc=f"Area {i}/{self.n_areas}",
                    leave=False,
                    total=len(self.imgs[i]),
                ):
                    cv2.imwrite(str(thisdir / f"{j:03d}.{ext}"), img)


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
        self.labels = self.load_labels()

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
        with open(label_path, "w") as fp:
            json.dump(self.labels, fp)
        return label_path

    def load_labels(self) -> AreaLabel:
        "Internal helper to load labels"
        if self.label_path.exists():
            with open(self.label_path, "r") as fp:
                return json.load(fp)

        logging.info(
            f"{self.__class__.__name__}: Label file not found: {self.label_path}"
        )
        return {"annotations": [[] for _ in range(len(self.imgs))]}

    def update_imgs(self, imgs: np.ndarray):
        self.imgs = imgs
        self._mat[self.key] = imgs

    def save_imgs(self, path: Path) -> Path:
        _savemat = {}
        for k, v in self._mat.items():
            if isinstance(v, np.ndarray):
                v = np.moveaxis(v, 0, -1)
            _savemat[k] = v
        sio.savemat(path, _savemat)
        return path

    @classmethod
    def from_label_path(cls, p: Path | str):
        p = Path(p)
        return cls(p.parent / p.name.replace(LABELS_EXT, ".mat"))

    def export_image_stack(self, dest: Path | str):
        dest = Path(dest)
        dest.mkdir(exist_ok=True, parents=True)
        for j, img in tqdm(enumerate(self.imgs), leave=False, total=len(self.imgs)):
            cv2.imwrite(str(dest / f"{j:03d}.png"), img)


class LabelCounts(NamedTuple):
    c: Counter[str]  # number of ROIs
    width: Counter[str]  # total width of the label in pixels


def count_labels(labels: AreaLabel):
    count: defaultdict[str, int] = defaultdict(int)
    total_width: defaultdict[str, int] = defaultdict(int)
    for l in labels["annotations"]:
        if not l:
            continue
        for ll in l:
            span = ll["xspan"]
            name = ll["label"]["name"]
            total_width[name] += abs(round(span[1] - span[0]))
            count[name] += 1

    return LabelCounts(c=Counter(count), width=Counter(total_width))


def _merge_neighbours(ls: FrameLabel):
    ls = sorted(ls, key=annotation_sort_key)
    prev = ls[0]
    for curr in ls[1:]:
        if prev is None:
            prev = curr
            continue

        # If prev and curr have different labels,
        # or if prev and curr don't overlap, return the prev label,
        # and set curr to prev.

        if prev["label"] != curr["label"] or curr["xspan"][0] - prev["xspan"][1] > 1:
            yield prev
            prev = curr
            continue

        # Merge prev and curr
        merged_span = (prev["xspan"][0], curr["xspan"][1])
        prev = Annotation(xspan=merged_span, label=prev["label"])
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
    "Shift with wrap around"
    return (round(x) + dx + xlim) % xlim


from functools import wraps


def trace_fn(f):
    """
    A decorator to print the function name and its arguments when its called.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        s = f"{f.__name__}("
        s += ", ".join([f"#{i}={x}" for i, x in enumerate(args)])
        s += ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        s += ")\n"
        print(s)
        return f(*args, **kwargs)

    return wrapper


def mv_one(l: Annotation, dx: int, xlim: int) -> tuple[Annotation, ...]:
    (x1, x2) = l["xspan"]

    assert x1 != x2
    x1, x2 = _s(x1, dx, xlim), _s(x2, dx, xlim)
    if x1 < x2:
        return (Annotation(xspan=(x1, x2), label=l["label"]),)
    elif x1 > x2:
        if x2 == 0:
            return (Annotation(xspan=(x1, xlim), label=l["label"]),)
        return (
            Annotation(xspan=(x1, xlim), label=l["label"]),
            Annotation(xspan=(0, x2), label=l["label"]),
        )
    else:  # x1 == x2
        # The whole span is included, since after shift, one of the
        # bounds wrapped around to be equal to the other.
        return (Annotation(xspan=(0, xlim), label=l["label"]),)


def shift_x(
    old_imgs: np.ndarray | list[np.ndarray],
    new_imgs: np.ndarray | list[np.ndarray],
    old_labels: AreaLabel,
):
    """
    Shift x for one scan area.
    """
    assert len(old_imgs) == len(new_imgs)
    # (n_imgs, y, x)
    if old_imgs[0].shape != new_imgs[0].shape:
        # migrate mat images of height 625 to HDF5 images of height 624
        assert old_imgs.shape[1] == 625
        assert new_imgs.shape[1] == 624
        old_imgs = old_imgs[:, :624, :]

    xlim = old_imgs[0].shape[-1]

    flatten = lambda l: sorted(x for ll in l for x in ll)

    new_labels: AreaLabel = deepcopy(old_labels)

    for i, ls in enumerate(tqdm(old_labels["annotations"], desc="shift_x")):
        # per frame
        if ls is not None:
            img1, img2 = old_imgs[i].astype(np.double), new_imgs[i].astype(np.double)
            dx = calc_offset_x(img1, img2)
            new_labels["annotations"][i] = flatten(mv_one(l, dx, xlim) for l in ls)

    # merge two labels if they overlap
    for i, ls in enumerate(new_labels["annotations"]):
        if ls:
            new_labels["annotations"][i] = list(_merge_neighbours(ls))

    return new_labels


import unittest


class Test_label_shift(unittest.TestCase):
    def test_mv_one(self) -> None:
        # Shift left over
        self.assertEqual(
            mv_one(Annotation(xspan=(0, 10), label={"name": "a"}), dx=-10, xlim=2000),
            (dict(xspan=(1990, 2000), label={"name": "a"}),),
        )

        # Shift left split
        self.assertEqual(
            mv_one(Annotation(xspan=(0, 20), label={"name": "a"}), dx=-10, xlim=2000),
            (
                Annotation(xspan=(1990, 2000), label={"name": "a"}),
                (Annotation(xspan=(0, 10), label={"name": "a"})),
            ),
        )

        # Shift right
        self.assertEqual(
            mv_one(Annotation(xspan=(0, 10), label={"name": "a"}), dx=10, xlim=2000),
            (Annotation(xspan=(10, 20), label={"name": "a"}),),
        )

        # Shift right over
        self.assertEqual(
            mv_one(
                Annotation(xspan=(1990, 2000), label={"name": "a"}), dx=10, xlim=2000
            ),
            (Annotation(xspan=(0, 10), label={"name": "a"}),),
        )

        # Shift right split
        self.assertEqual(
            mv_one(
                Annotation(xspan=(1980, 2000), label={"name": "a"}), dx=10, xlim=2000
            ),
            (
                Annotation(xspan=(1990, 2000), label={"name": "a"}),
                Annotation(xspan=(0, 10), label={"name": "a"}),
            ),
        )

    def test_merge_neighbours(self) -> None:
        # No merge. only sort
        inp = [((1990, 2000), "a"), ((0, 10), "a")]
        exp = [((0, 10), "a"), ((1990, 2000), "a")]
        self.assertEqual(list(_merge_neighbours(inp)), exp)

        # merge
        inp = [((0, 10), "a"), ((11, 20), "a")]
        exp = [((0, 20), "a")]
        self.assertEqual(list(_merge_neighbours(inp)), exp)

    def test_shift_x(self) -> None:
        # (y=10, x=20) image
        img1: Final = np.repeat(np.expand_dims(np.arange(20), 0), 10, axis=0)

        offset = 5
        img2 = np.roll(img1, offset)
        old_label: FrameLabel = [Annotation(xspan=(2, 5), label={"name": "a"})]
        new_label: FrameLabel = [
            Annotation(xspan=(2 + offset, 5 + offset), label={"name": "a"})
        ]
        self.assertEqual(
            shift_x([img1], [img2], {"annotations": [old_label]}), [new_label]
        )

        offset = 3
        img2 = np.roll(img1, offset)
        old_label = [Annotation(xspan=(15, 20), label={"name": "a"})]
        new_label = [
            Annotation(xspan=(0, 3), label={"name": "a"}),
            Annotation(xspan=(18, 20), label={"name": "a"}),
        ]
        self.assertEqual(
            shift_x([img1], [img2], {"annotations": [old_label]}), [new_label]
        )


import tempfile


class TestHdf5Data(unittest.TestCase):
    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile(delete=False)

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

    def tearDown(self) -> None:
        self.fp.close()
        return super().tearDown()

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
