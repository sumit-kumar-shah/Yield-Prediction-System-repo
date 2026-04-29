"""
data_pipeline.py
ETL: Load raw CSV, clean, validate, and save processed dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_raw(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "crop_data.csv"
    df = pd.read_csv(path)
    print(f"[pipeline] Loaded {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(subset=["Yield_kg_per_ha"])
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    df["Year"]   = df["Year"].astype(int)
    df["State"]  = df["State"].str.strip()
    df["Crop"]   = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()
    df = df[(df["Yield_kg_per_ha"] > 0) & (df["Area_Hectares"] > 0)]
    for col in ["Yield_kg_per_ha", "Fertilizer_kg_per_ha"]:
        cap = df[col].quantile(0.99)
        df[col] = np.clip(df[col], 0, cap)
    print(f"[pipeline] After cleaning: {len(df):,} rows")
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    required = ["State","Crop","Season","Year","Area_Hectares",
                "Rainfall_mm","Fertilizer_kg_per_ha","Avg_Temp_C",
                "Irrigation_pct","Yield_kg_per_ha"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing cols: {missing}"
    assert df["Yield_kg_per_ha"].isna().sum() == 0
    print("[pipeline] Validation passed ✓")


def save_processed(df: pd.DataFrame) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "crop_data_clean.csv"
    df.to_csv(out, index=False)
    print(f"[pipeline] Saved → {out}")
    return out


def run_pipeline() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    validate(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_pipeline()
