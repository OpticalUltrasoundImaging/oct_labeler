# Assume we are already in a conda environment

# if dist exists, remove it
$DistDirectory = "./dist"
if (Test-Path $DistDirectory) { Remove-Item $DistDirectory -Recurse }

# Install oct_labeler
pip install -e .
if (!$?) { Exit $LASTEXITCODE }

# build binary
pyinstaller main.spec
