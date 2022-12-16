$DistDirectory = "./dist"
if (Test-Path $DistDirectory) {
  Remove-Item $DistDirectory -Recurse
}

conda activate oct_labeler
pip install -e .
pyinstaller main.spec
