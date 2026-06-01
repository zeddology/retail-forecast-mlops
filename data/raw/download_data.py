import kagglehub
import shutil
import os
from pathlib import Path

# Download (goes to kagglehub's cache)
path = kagglehub.competition_download('demand-forecasting-kernels-only')
print("Downloaded to:", path)

# List what came down
downloaded = Path(path)
for f in downloaded.iterdir():
    print("  -", f.name)

# Copy train.csv into data/raw/ (relative to where you run this)
dest = Path("data") / "raw" / "train.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(downloaded / "train.csv", dest)
print("Copied train.csv to:", dest.resolve())