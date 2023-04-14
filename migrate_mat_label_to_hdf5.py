# %%
from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

from oct_labeler.oct_data import OctDataHdf5, OctData

from tqdm import tqdm

# from tqdm.notebook import tqdm

# %load_ext autoreload
# %autoreload 2

# %%
# new_dir = Path("C:/src/OCT_proc/imgs2/")
new_dir = Path("G:/data/OCT_invivo/imgs2/")


# %%
def get_first_num(s: str | Path) -> int:
    """
    Get the leading number in a string.
    e.g. "9 polyp TA" -> 9
    """
    s = s.name if isinstance(s, Path) else s
    if match_obj := re.match(r"^(\d+)", s):
        return int(match_obj[0])
    raise ValueError(f"Failed to find the leading number in '{s}'")


old_dir = Path.home() / "Box/OCT invivo"
old_dirs = []
for fd in old_dir.glob("*"):
    if not fd.is_dir():
        continue
    try:
        area_i = get_first_num(fd)
        old_dirs.append((area_i, fd))
    except Exception as e:
        pass

old_dirs = sorted(old_dirs)
old_dirs


# %%
def calc_offset(a, b):
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h].astype(np.double), b[:h].astype(np.double)
    return round(cv2.phaseCorrelate(a, b)[0][0])


# %%
"""
Iterate through old directories.

For each found label file:
    Get old data file
    Calculate offset of old data from new data
    Apply offset to label data
    write label to new label file
"""

# %%
for pid, old_d in old_dirs:
    # pid, old_d = old_dirs[0]
    print(pid, old_d)

    # find old data labels
    label_files = list(old_d.glob("*.pkl"))
    if not label_files:
        continue

    # load new data
    new_data = OctDataHdf5(new_dir / str(pid) / "areas.hdf5")

    for label_file in label_files:
        print("Working on ", label_file)
        if label_file.name.startswith("4_rectal_cancer"):
            # skip this as raw data missing.
            print("Skipped.")
            continue

        curr_area_idx = get_first_num(label_file) - 1

        # load old labels and data
        old_data = OctData.from_label_path(label_file)
        old_data.load_imgs_()

        n_imgs = len(old_data.imgs)

        # xoffsets = [
        #     -calc_offset(new_data.imgs[curr_area_idx][i], old_data.imgs[i])
        #     for i in tqdm(range(n_imgs), desc="corr", leave=False)
        # ]
        # xoffset = calc_offset(new_data.imgs[curr_area_idx][0], old_data.imgs[0])
        def get_offset(i: int):
            return -calc_offset(new_data.imgs[curr_area_idx][i], old_data.imgs[i])

        # plt.figure()
        # plt.subplot(211)
        # plt.imshow(old_data.imgs[0])
        # plt.title(f"Old ({xoffsets[0]=})")
        # plt.subplot(212)
        # plt.imshow(new_data.imgs[curr_area_idx][0])
        # plt.title("New")

        old_data.shift_x(get_offset)
        new_data.labels[curr_area_idx] = old_data.labels

    new_data.save_labels()

# %%
