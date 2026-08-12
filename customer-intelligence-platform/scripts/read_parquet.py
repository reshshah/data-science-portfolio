
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/rshah/customer intelligence platform")

TRAIN_FILE = PROJECT_ROOT / "data" / "ml" / "train.parquet"

df = pd.read_parquet(TRAIN_FILE)

print(df.head())
print(df.info())