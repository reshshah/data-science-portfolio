import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_split(data_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / f"{name}.parquet")
    logger.info("Loaded %s: %s", name, df.shape)
    return df


def load_feature_metadata(metadata_file: Path) -> dict:
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)
