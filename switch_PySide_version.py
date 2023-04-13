from pathlib import Path
import re
import sys


root = Path("./oct_labeler/gui/")
files = list(root.glob("*.py"))
files.append(Path("./pyproject.toml"))


def _sub_in_file(file: Path, pat, repl):
    with open(file, "r") as fp:
        txt = fp.read()

    res = re.sub(pat, repl, txt)

    with open(file, "w") as fp:
        fp.write(res)



assert len(sys.argv) > 1
pysidever = sys.argv[1]
if pysidever == "PySide2":
    print("Switching to", pysidever)
    for file in files:
        _sub_in_file(file, "PySide6", "PySide2")

    _sub_in_file(Path("./oct_labeler/gui/__init__.py"), r"app\.exec\(\)", r"app\.exec_\(\)")

elif pysidever == "PySide6":
    print("Switching to", pysidever)
    for file in files:
        _sub_in_file(file, "PySide2", "PySide6")

    _sub_in_file(Path("./oct_labeler/gui/__init__.py"), "app.exec_", "app.exec")
