from oct_labeler.oct_data import OctData
from oct_labeler.version import __version__


def main():
    import argparse

    parser = argparse.ArgumentParser(
        __file__, description=f"OCT label migration {__version__}"
    )
    parser.add_argument("label_path")
    parser.add_argument("polyp_type")
    args = parser.parse_args()

    label_path = args.label_path
    polyp_type = args.polyp_type
    print(args)

    oct_data = OctData(None, None, None)  # type: ignore
    oct_data.load_labels(label_path)

    n_modified = 0
    labels = oct_data.labels  # ref
    for i, label in enumerate(labels):
        new_label = []
        if not label:
            continue

        for rgn, name in label:
            if name.startswith("polyp"):
                new_label.append((rgn, f"polyp;{polyp_type}"))
                n_modified += 1
            else:
                new_label.append((rgn, name))
        labels[i] = new_label

    print(f"Modified {n_modified} label regions.")
    oct_data.save_labels(label_path)
