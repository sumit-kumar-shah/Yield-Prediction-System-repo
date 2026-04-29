"""
predict.py
Load trained model pipeline and run inference for a single input record.
The imputer is baked into each pipeline inside the VotingRegressor,
so no separate NaN handling is needed here.
"""
import sys
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model():
    model = joblib.load(MODELS_DIR / "ensemble_model.pkl")
    meta  = joblib.load(MODELS_DIR / "model_meta.pkl")
    return model, meta


def encode_input(state: str, crop: str, season: str) -> tuple:
    encoders = {}
    for col in ["State", "Crop", "Season"]:
        p = MODELS_DIR / f"le_{col.lower()}.pkl"
        encoders[col] = joblib.load(p)
    try:
        s_enc  = int(encoders["State"].transform([state])[0])
        c_enc  = int(encoders["Crop"].transform([crop])[0])
        se_enc = int(encoders["Season"].transform([season])[0])
    except ValueError as e:
        raise ValueError(f"Unknown category: {e}. Check state/crop/season spelling.")
    return s_enc, c_enc, se_enc


def get_lag_features(state: str, crop: str, season: str, year: int) -> tuple:
    proc = Path(__file__).parent.parent / "data" / "processed" / "crop_data_clean.csv"
    try:
        df   = pd.read_csv(proc)
        hist = df[(df["State"] == state) & (df["Crop"] == crop) & (df["Season"] == season)]
        hist = hist.sort_values("Year")
        prev = hist[hist["Year"] < year]["Yield_kg_per_ha"]
        lag1  = float(prev.iloc[-1])  if len(prev) >= 1 else float(hist["Yield_kg_per_ha"].mean())
        lag2  = float(prev.iloc[-2])  if len(prev) >= 2 else lag1
        roll3 = float(prev.tail(3).mean()) if len(prev) >= 1 else lag1
    except Exception:
        # Fallback: return NaN — the imputer inside the pipeline will handle it
        lag1 = lag2 = roll3 = float("nan")
    return lag1, lag2, roll3


def predict(
    state: str,
    crop: str,
    season: str,
    year: int,
    area_hectares: float,
    rainfall_mm: float,
    fertilizer_kg_per_ha: float,
    avg_temp_c: float,
    irrigation_pct: float,
) -> dict:
    model, meta = load_model()
    s_enc, c_enc, se_enc = encode_input(state, crop, season)
    lag1, lag2, roll3    = get_lag_features(state, crop, season, year)
    year_norm = (year - 2006) / (2023 - 2006)

    row = np.array([[
        area_hectares, rainfall_mm, fertilizer_kg_per_ha,
        avg_temp_c, irrigation_pct,
        s_enc, c_enc, se_enc,
        lag1, lag2, roll3,
        year_norm,
    ]], dtype=float)

    # NaNs in lag features are fine — the SimpleImputer inside each
    # sub-pipeline of the VotingRegressor handles them automatically
    pred = float(model.predict(row)[0])
    pred = max(pred, 0.0)  # yield can't be negative

    return {
        "predicted_yield_kg_per_ha": round(pred, 2),
        "confidence_low":            round(pred * 0.92, 2),
        "confidence_high":           round(pred * 1.08, 2),
        "model_r2":                  round(meta["test_r2"], 4),
        "model_rmse":                round(meta["test_rmse"], 2),
    }


if __name__ == "__main__":
    result = predict(
        state="Punjab", crop="Wheat", season="Rabi",
        year=2024, area_hectares=1500, rainfall_mm=600,
        fertilizer_kg_per_ha=200, avg_temp_c=24, irrigation_pct=80
    )
    print(result)
