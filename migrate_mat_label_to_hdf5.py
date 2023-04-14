# %%
from pathlib import Path
from oct_labeler.data import OctDataHdf5, shift_x

root = Path("/media/tnie/TigerDrive/Data/OCT_invivo/")
root.exists()

old_dir = root / "imgs2"
new_dir = root / "imgs3"
new_dirs = [d for d in new_dir.glob("*") if d.is_dir()]

"""
Iterate through old directories.

For each found label file:
    Get old data file
    Calculate offset of old data from new data
    Apply offset to label data
    write label to new label file
"""

for nd in new_dirs:
    print(nd)
    new_data = OctDataHdf5(nd / "areas.hdf5")
    old_data = OctDataHdf5(old_dir / nd.name / "areas.hdf5")

    for area_i in range(new_data.n_areas):
        new_data.labels[area_i] = shift_x(
            old_data.imgs[area_i], new_data.imgs[area_i], old_data.labels[area_i]
        )

    new_data.save_labels()
