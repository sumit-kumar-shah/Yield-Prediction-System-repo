"""
feature_engineering.py
Lag features, rolling averages, label encoding, train/val/test split.
Split is computed dynamically based on whatever years exist in the data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
FEATURE_COLS = [
    "Area_Hectares", "Rainfall_mm", "Fertilizer_kg_per_ha",
    "Avg_Temp_C", "Irrigation_pct",
    "State_enc", "Crop_enc", "Season_enc",
    "Yield_lag1", "Yield_lag2", "Yield_roll3",
    "Year_norm",
]
TARGET_COL = "Yield_kg_per_ha"


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["State", "Crop", "Season", "Year"]).copy()
    grp = df.groupby(["State", "Crop", "Season"])["Yield_kg_per_ha"]
    df["Yield_lag1"]  = grp.shift(1)
    df["Yield_lag2"]  = grp.shift(2)
    df["Yield_roll3"] = grp.shift(1).rolling(3, min_periods=1).mean().values
    # Fill first-year NaNs with per-crop mean
    for col in ["Yield_lag1", "Yield_lag2", "Yield_roll3"]:
        fill = df.groupby("Crop")["Yield_kg_per_ha"].transform("mean")
        df[col] = df[col].fillna(fill)
    return df


def encode_categoricals(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for col in ["State", "Crop", "Season"]:
        enc_col  = f"{col}_enc"
        enc_path = MODELS_DIR / f"le_{col.lower()}.pkl"
        if fit:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col])
            joblib.dump(le, enc_path)
        else:
            le = joblib.load(enc_path)
            df[enc_col] = le.transform(df[col])
    return df


def normalize_year(df: pd.DataFrame) -> pd.DataFrame:
    min_yr = df["Year"].min()
    max_yr = df["Year"].max()
    rng = max_yr - min_yr if max_yr != min_yr else 1
    df["Year_norm"] = (df["Year"] - min_yr) / rng
    return df


def build_features(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    df = add_lag_features(df)
    df = encode_categoricals(df, fit=fit)
    df = normalize_year(df)
    return df


def split_data(df: pd.DataFrame):
    """
    Dynamic temporal split based on actual years in the data.
      Train : first 70% of years
      Val   : next 15% of years
      Test  : last 15% of years
    """
    years      = sorted(df["Year"].unique())
    n          = len(years)
    train_end  = years[int(n * 0.70) - 1]
    val_end    = years[int(n * 0.85) - 1]

    train = df[df["Year"] <= train_end]
    val   = df[(df["Year"] > train_end) & (df["Year"] <= val_end)]
    test  = df[df["Year"] > val_end]

    print(f"[features] Years  → Train: {years[0]}–{train_end} "
          f"| Val: {train_end+1}–{val_end} "
          f"| Test: {val_end+1}–{years[-1]}")
    print(f"[features] Rows   → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    X_tr  = train[FEATURE_COLS].values;  y_tr  = train[TARGET_COL].values
    X_val = val[FEATURE_COLS].values;    y_val = val[TARGET_COL].values
    X_te  = test[FEATURE_COLS].values;   y_te  = test[TARGET_COL].values
    return X_tr, y_tr, X_val, y_val, X_te, y_te


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    df = run_pipeline()
    df = build_features(df, fit=True)
    split_data(df)
