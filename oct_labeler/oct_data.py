from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle

import h5py
import numpy as np


ONE_LABEL = tuple[tuple[int, int], str]
Labels = list[ONE_LABEL]


@dataclass
class OctDataHdf5:
    # everything here indexed by area idx
    imgs: list[np.ndarray]
    _imgs: list[np.ndarray]
    _binimgs: list[np.ndarray]
    labels: list[list[Labels]]

    hdf5path: Path

    all_areas: bool = True

    @classmethod
    def from_path(cls, path: str | Path):
        path = Path(path)
        imgs = []
        binimgs = []
        labels = []
        with h5py.File(path, "r") as f:
            for _i in f["areas"]:
                imgs.append(f["areas"][_i]["imgs"][...])
                binimgs.append(f["areas"][_i]["binimgs"][...])
                labels.append([None for _ in range(len(imgs[-1]))])
        return OctDataHdf5(
            imgs=imgs, _imgs=imgs, _binimgs=binimgs, labels=labels, hdf5path=path
        )

    def label_path(self):
        p = self.hdf5path
        return p.parent / (p.stem + "_labels.pkl")

    def load_labels(self):
        with open(self.label_path(), "rb") as fp:
            self.labels = pickle.load(fp)

    def save_labels(self):
        p = self.label_path()
        with open(p, "wb") as fp:
            pickle.dump(self.labels, fp)
        return p


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
        import scipy.io as sio

        mat = sio.loadmat(fname)

        keys = [s for s in mat.keys() if not s.startswith("__")]
        print(f"Available keys in data file: {keys}")
        key = "I_updated"
        assert key in keys

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0

        oct_data = OctData(
            path=fname,
            label_path=cls.label_path_from_img_path(fname),
            imgs=scans,
            labels=[None] * len(scans),
        )
        return oct_data

    @staticmethod
    def label_path_from_img_path(path: str | Path, ext=".pkl") -> Path:
        path = Path(path)
        return path.parent / (path.stem + "_label" + ext)

    @staticmethod
    def img_path_from_label_path(label_path: str | Path) -> Path:
        label_path = Path(label_path)
        return label_path.parent / (label_path.stem.rsplit("_label", 1)[0] + ".mat")

    def shift_x(self, dx):
        def m_one(l: ONE_LABEL):
            return ((l[0][0] + dx, l[0][1] + dx), l[1])

        self.labels = [[m_one(l) for l in ls] for ls in self.labels]

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
