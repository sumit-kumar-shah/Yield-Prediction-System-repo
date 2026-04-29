import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import pytest
from feature_engineering import add_lag_features, normalize_year, FEATURE_COLS


@pytest.fixture
def base_df():
    rows = []
    for yr in range(2010, 2016):
        rows.append({"State":"Punjab","Crop":"Wheat","Season":"Rabi",
                     "Year":yr,"Yield_kg_per_ha":3000.0 + yr*10})
    return pd.DataFrame(rows)


def test_lag_features_created(base_df):
    df = add_lag_features(base_df)
    assert "Yield_lag1" in df.columns
    assert "Yield_lag2" in df.columns
    assert "Yield_roll3" in df.columns


def test_lag1_is_previous_year(base_df):
    df = add_lag_features(base_df).sort_values("Year").reset_index(drop=True)
    for i in range(1, len(df)):
        assert df.loc[i, "Yield_lag1"] == df.loc[i-1, "Yield_kg_per_ha"]


def test_no_nulls_after_lag(base_df):
    df = add_lag_features(base_df)
    assert df[["Yield_lag1","Yield_lag2","Yield_roll3"]].isna().sum().sum() == 0


def test_year_norm_range(base_df):
    df = normalize_year(base_df)
    assert df["Year_norm"].min() >= 0
    assert df["Year_norm"].max() <= 1.1  # slight over for future years


def test_feature_col_count():
    assert len(FEATURE_COLS) == 12
