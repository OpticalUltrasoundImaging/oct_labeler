# %%
from pathlib import Path
from oct_labeler.data import ScanDataMat, ScanDataHdf5, shift_x

# %load_ext autoreload
# %autoreload 2

root = Path("/media/tnie/TigerDrive/Data/OCT_invivo/")
root.exists()

old_dir = root / "mat_imgs"
# new_dir = root / "imgs3"
new_dir = Path.home() / "code/oct_proc/redata"
new_dirs = [d for d in new_dir.glob("*") if d.is_dir()]

# %%
old_dir

# %%
"""
Iterate through old directories.

For each found label file:
    Get old data file
    Calculate offset of old data from new data
    Apply offset to label data
    write label to new label file
"""

# for nd in new_dirs:

# old_data = ScanDataMat(old_dir / nd.name / "areas.hdf5")
old_data = ScanDataMat(old_dir / "13 polyp TA/3normal_cut_aligned.mat")
new_data = ScanDataHdf5(new_dir / "13/areas.hdf5")
area_i = 2

new_data.labels[area_i] = shift_x(old_data.imgs, new_data.imgs[area_i], old_data.labels)

new_data.save_labels()
