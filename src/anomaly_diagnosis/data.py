from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def split_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "ID",
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    drop_train = [c for c in [id_col, target_col] if c in train_df.columns]
    drop_test = [c for c in [id_col] if c in test_df.columns]

    x_train = train_df.drop(columns=drop_train)
    y = train_df[target_col].copy()
    x_test = test_df.drop(columns=drop_test)

    x_train = (
        x_train.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    x_test = (
        x_test.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return x_train, y, x_test


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("X_")]
