"""
api/app.py
FastAPI application — REST endpoints + HTML dashboard serving.
"""
import sys, json
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, insert_prediction, get_history
import predict as predictor

STATIC_DIR    = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"
PROC_DATA     = Path(__file__).parent.parent / "data" / "processed" / "crop_data_clean.csv"
MODELS_DIR    = Path(__file__).parent.parent / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[app] Database initialised ✓")
    yield


app = FastAPI(
    title="CropYield AI",
    description="Crop yield prediction using ensemble ML (RF + GBR + XGBoost + LightGBM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    state:                str   = Field(..., example="Punjab")
    crop:                 str   = Field(..., example="Wheat")
    season:               str   = Field(..., example="Rabi")
    year:                 int   = Field(..., ge=2000, le=2030, example=2024)
    area_hectares:        float = Field(..., gt=0, example=1200.0)
    rainfall_mm:          float = Field(..., ge=0, example=650.0)
    fertilizer_kg_per_ha: float = Field(..., ge=0, example=180.0)
    avg_temp_c:           float = Field(..., example=24.0)
    irrigation_pct:       float = Field(..., ge=0, le=100, example=75.0)


class PredictResponse(BaseModel):
    predicted_yield_kg_per_ha: float
    confidence_low:  float
    confidence_high: float
    model_r2:   float
    model_rmse: float


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Dashboard UI")
async def dashboard(request: Request):
    # Read template manually to avoid Jinja2 LRU cache dict-hashing bug
    template_path = TEMPLATES_DIR / "index.html"
    html_content  = template_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


@app.post("/predict", response_model=PredictResponse, summary="Predict crop yield")
async def predict_endpoint(body: PredictRequest):
    try:
        result = predictor.predict(
            state=body.state,
            crop=body.crop,
            season=body.season,
            year=body.year,
            area_hectares=body.area_hectares,
            rainfall_mm=body.rainfall_mm,
            fertilizer_kg_per_ha=body.fertilizer_kg_per_ha,
            avg_temp_c=body.avg_temp_c,
            irrigation_pct=body.irrigation_pct,
        )
        insert_prediction({**body.model_dump(), **result})
        return result
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python src/train.py")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/crops", summary="List supported crops and seasons")
async def list_crops():
    if PROC_DATA.exists():
        df = pd.read_csv(PROC_DATA)
        return {
            "crops":   sorted(df["Crop"].unique().tolist()),
            "states":  sorted(df["State"].unique().tolist()),
            "seasons": sorted(df["Season"].unique().tolist()),
        }
    return {
        "crops":   ["Rice","Wheat","Maize","Soybean","Sugarcane","Cotton",
                    "Mustard","Chickpea","Groundnut","Bajra"],
        "states":  ["Uttar Pradesh","Punjab","Haryana","Maharashtra","Madhya Pradesh"],
        "seasons": ["Kharif","Rabi","Whole Year"],
    }


@app.get("/history", summary="Past predictions from DB")
async def history(limit: int = 50):
    return {"predictions": get_history(limit)}


@app.get("/trend", summary="Historical yield trend for a crop+state")
async def trend(state: str, crop: str, season: str):
    if not PROC_DATA.exists():
        raise HTTPException(503, "Processed data not found. Run training first.")
    df  = pd.read_csv(PROC_DATA)
    sub = df[(df["State"]==state) & (df["Crop"]==crop) & (df["Season"]==season)].sort_values("Year")
    if sub.empty:
        raise HTTPException(404, "No data for this combination.")
    return {
        "years":  sub["Year"].tolist(),
        "yields": sub["Yield_kg_per_ha"].round(2).tolist(),
    }


@app.get("/metrics", summary="Trained model evaluation metrics")
async def metrics():
    import joblib
    mp = MODELS_DIR / "model_meta.pkl"
    if not mp.exists():
        raise HTTPException(503, "Model not trained yet. Run: python src/train.py")
    meta = joblib.load(mp)
    return {k: v for k, v in meta.items() if k != "feature_cols"}


@app.get("/health")
async def health():
    model_ready = (MODELS_DIR / "ensemble_model.pkl").exists()
    return {"status": "ok", "model_ready": model_ready}
