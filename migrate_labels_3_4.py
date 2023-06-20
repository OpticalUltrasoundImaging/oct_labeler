# %%
from pathlib import Path
from oct_labeler.data import ScanDataHdf5, shift_x

root = Path("/media/tnie/TigerDrive/Data/OCT_invivo/")
root.exists()

old_dir = root / "imgs3"
new_dir = root / "imgs4"
new_dirs = [d for d in new_dir.glob("*") if d.is_dir()]

# %%
# new_dirs = [new_dirs[0], new_dirs[-1]]
new_dirs = [new_dir / "2"]
new_dirs

# %%
"""
Iterate through old directories.

For each found label file:
    Get old data file
    Calculate offset of old data from new data
    Apply offset to label data
    write label to new label file
"""

for new_dir in new_dirs:
    print(new_dir)
    new_data = ScanDataHdf5(new_dir / "areas.hdf5")
    old_data = ScanDataHdf5(old_dir / new_dir.name / "areas.hdf5")

    for area_i in range(new_data.n_areas):
        new_data.labels[area_i] = shift_x(
            old_data.imgs[area_i], new_data.imgs[area_i], old_data.labels[area_i]
        )

    new_data.save_labels()

# %%
