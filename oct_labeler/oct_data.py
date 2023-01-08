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
    imgs: np.ndarray  # ref to image array
    labels: list[Labels]  # [[((10, 20), "normal")]]

    def save_labels(self, label_path: Path | None = None) -> Path:
        if label_path is None:
            label_path = self.get_label_fname_from_img_path(self.path)

        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)

        return label_path

    def load_labels(self, label_path: Path | None = None):
        if label_path is None:
            label_path = self.get_label_fname_from_img_path(self.path)

        with open(label_path, "rb") as fp:
            self.labels = pickle.load(fp)

    @classmethod
    def from_mat_path(cls, fname: str) -> OctData:
        import scipy.io as sio

        mat = sio.loadmat(fname)

        keys = [s for s in mat.keys() if not s.startswith("__")]
        print(f"Available keys in data file: {keys}")
        key = "I_updated"
        assert key in keys

        scans = mat[key]
        scans = np.moveaxis(scans, -1, 0)
        assert len(scans) > 0

        oct_data = OctData(path=fname, imgs=scans, labels=[None] * len(scans))
        return oct_data

    @staticmethod
    def get_label_fname_from_img_path(path: str | Path, ext=".pkl") -> Path:
        path = Path(path)
        return path.parent / (path.stem + "_label" + ext)

    def shift_x(self, dx):
        def m_one(l: ONE_LABEL):
            return ((l[0][0] + dx, l[0][1] + dx), l[1])

        self.labels = [[m_one(l) for l in ls] for ls in self.labels]
