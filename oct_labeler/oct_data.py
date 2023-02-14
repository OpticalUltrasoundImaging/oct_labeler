from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


ONE_LABEL = tuple[tuple[int, int], str]
Labels = list[ONE_LABEL]


@dataclass
class OctData:
    path: str | Path  # path to the image mat file
    label_path: str | Path
    imgs: np.ndarray  # ref to image array
    labels: list[Labels]  # [[((10, 20), "normal")]]

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

        self.path = self.img_path_from_label_path(label_path)

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
        from collections import Counter

        return Counter([ll[1] for l in self.labels if l for ll in l])
