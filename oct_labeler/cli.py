from pathlib import Path
import click

from oct_labeler.gui import gui_main
from oct_labeler.data import (
    ScanData,
    ScanDataHdf5,
    ScanDataMat,
    count_labels,
    LABELS_EXT,
)


@click.group()
def cli():
    pass


@click.command()
def gui():
    gui_main()


@click.command()
@click.argument("data_path")
def interact(data_path: str):
    path = Path(data_path)
    if path.suffix == ".hdf5":
        data: ScanData = ScanDataHdf5(path)
    elif path.suffix == ".mat":
        data = ScanDataMat(path)
    else:
        raise TypeError(f"{path} is not a valid data file.")
    breakpoint()


@click.command()
@click.argument("label_path")
def inspect(label_path: str):
    click.echo(f"Inspecting {label_path}")

    oct_data = ScanDataMat.from_label_path(Path(label_path))
    click.echo(count_labels(oct_data.labels))


@click.command()
@click.option("-r", "--recursive", is_flag=True)
@click.argument("path")
def inspect_all(recursive: bool, path: str):
    root = Path(path)
    if recursive:
        label_files = sorted(root.glob("**/*" + LABELS_EXT))
    else:
        label_files = sorted(root.glob("*" + LABELS_EXT))

    click.echo(f"Found {len(label_files)} label files in {path}")

    from collections import Counter

    c_all: Counter[str] = Counter()
    width_all: Counter[str] = Counter()
    for label_file in label_files:
        oct_data = ScanDataMat.from_label_path(label_file)
        res = count_labels(oct_data.labels)
        c_all += res.c
        width_all += res.width
        click.echo(f"{label_file.relative_to(root)}:\t{res.c}")

    click.echo(f"\nTotal count: {c_all}")
    click.echo(f"Total width: {width_all}")


@click.command()
@click.argument("label_path")
@click.option("--dry", "-d", is_flag=True, help="Dry run")
@click.argument("old_name")
@click.argument("new_name")
def migrate(dry: bool, label_path: str, old_name: str, new_name: str):
    click.echo(
        f"Migrating {label_path}: {old_name} -> {new_name} {'(dry run)' if dry else ''}"
    )

    oct_data = ScanDataMat.from_label_path(label_path)

    n_modified = 0
    labels = oct_data.labels  # ref
    for i, label in enumerate(labels):
        if not label:
            continue

        new_label = []
        for rgn, name in label:
            if name == old_name:
                new_label.append((rgn, new_name))
                n_modified += 1
            else:
                new_label.append((rgn, name))

        labels[i] = new_label

    print(f"Modified {n_modified} label regions.")
    if not dry:
        oct_data.save_labels()


cli.add_command(gui)
cli.add_command(inspect)
cli.add_command(inspect_all)
cli.add_command(migrate)
cli.add_command(interact)

if __name__ == "__main__":
    cli()
