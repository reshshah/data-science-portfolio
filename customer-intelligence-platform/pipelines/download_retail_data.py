"""Download the UCI Online Retail II dataset into data/raw/.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
1,067,371 transactions from a UK online retailer, Dec 2009 - Dec 2011.
Licensed CC BY 4.0 (Chen, D. 2012, DOI 10.24432/C5CG6D).

Run from the project root:

    python3 pipelines/download_retail_data.py

The raw file is gitignored -- this script is how anyone reproduces it.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXCEL_NAME = "online_retail_II.xlsx"
CSV_NAME = "online_retail_II.csv"


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Download + unzip the dataset, then convert to CSV for fast reloads."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME
    if csv_path.exists():
        print(f"Already present: {csv_path}")
        return csv_path

    print(f"Downloading {URL} (~44 MB, this takes a minute)...")
    with urllib.request.urlopen(URL) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(EXCEL_NAME, raw_dir)
    xlsx_path = raw_dir / EXCEL_NAME
    print(f"Extracted {xlsx_path}")

    # The xlsx has two sheets (2009-2010, 2010-2011). Concatenate into one CSV.
    import pandas as pd

    print("Converting to CSV (reading Excel is slow; this is a one-time cost)...")
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df):,} transactions to {csv_path}")
    return csv_path


if __name__ == "__main__":
    download()
