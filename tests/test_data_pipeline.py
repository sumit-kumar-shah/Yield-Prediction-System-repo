import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import pytest
from data_pipeline import clean, validate


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "State": ["Punjab","Punjab","Maharashtra"],
        "Crop":  ["Wheat","Wheat","Rice"],
        "Season":["Rabi","Rabi","Kharif"],
        "Year":  [2020, 2021, 2020],
        "Area_Hectares":        [1000.0, 1200.0, 800.0],
        "Rainfall_mm":          [600.0,   700.0, 900.0],
        "Fertilizer_kg_per_ha": [180.0,   200.0, 150.0],
        "Avg_Temp_C":           [22.0,    23.0,  28.0],
        "Irrigation_pct":       [75.0,    80.0,  60.0],
        "Yield_kg_per_ha":      [3200.0, 3400.0, 2100.0],
    })


def test_clean_removes_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df.iloc[:1]])
    cleaned = clean(df_dup)
    assert len(cleaned) == len(sample_df)


def test_clean_drops_zero_yield(sample_df):
    df_bad = sample_df.copy()
    df_bad.loc[0, "Yield_kg_per_ha"] = 0
    cleaned = clean(df_bad)
    assert (cleaned["Yield_kg_per_ha"] > 0).all()


def test_clean_handles_nulls(sample_df):
    df_null = sample_df.copy()
    df_null.loc[1, "Rainfall_mm"] = np.nan
    cleaned = clean(df_null)
    assert cleaned["Rainfall_mm"].isna().sum() == 0


def test_validate_passes(sample_df):
    cleaned = clean(sample_df)
    validate(cleaned)  # Should not raise


def test_validate_fails_missing_col(sample_df):
    df_bad = sample_df.drop(columns=["Rainfall_mm"])
    with pytest.raises(AssertionError):
        validate(df_bad)


def test_clean_strips_whitespace(sample_df):
    df_ws = sample_df.copy()
    df_ws["State"] = "  Punjab  "
    cleaned = clean(df_ws)
    assert (cleaned["State"] == "Punjab").all()
