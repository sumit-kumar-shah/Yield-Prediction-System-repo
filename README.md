# CropYield Prediction System

Predicting state-level crop yield (kg/hectare) using historical agronomic, weather, and soil data to aid agricultural planning in India.
Crop yield prediction system using an ensemble of Random Forest, Gradient Boosting, XGBoost, and LightGBM — served via a FastAPI REST API with a live browser dashboard.

---

## Project Structure

```
cropyield-Prediction/
├── data/
│   ├── raw/crop_data.csv          ← Included dataset (2,970 records)
│   └── processed/                 ← Auto-generated after training
├── src/
│   ├── data_pipeline.py           ← ETL: load, clean, validate
│   ├── feature_engineering.py     ← Lag features, encoding, split
│   ├── train.py                   ← Model training + evaluation
│   └── predict.py                 ← Inference module
├── api/
│   ├── app.py                     ← FastAPI application
│   ├── database.py                ← SQLite prediction history
│   ├── templates/index.html       ← Dashboard UI
│   └── static/                    ← Charts served here after training
├── models/                        ← Saved model files (after training)
├── tests/                         ← pytest test suite
├── requirements.txt
└── run.py                         ← Server entry point
```

---

## Quick Start

### Step 1 — Python version
Requires **Python 3.10+**. Check: `python --version`

### Step 2 — Create virtual environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac / Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model
```bash
python src/train.py
```
This will:
- Run the ETL pipeline on `data/raw/crop_data.csv`
- Engineer lag features and encode categoricals
- Train Random Forest, Gradient Boosting, XGBoost, LightGBM
- Tune hyperparameters with RandomizedSearchCV
- Build a weighted VotingRegressor ensemble
- Print R², RMSE, MAE on the held-out test set
- Save `models/ensemble_model.pkl`
- Save `api/static/feature_importance.png` and `actual_vs_predicted.png`

Expected output (approximate):
```
Ensemble Test R² ≈ 0.92
```

### Step 5 — Start the server
```bash
python run.py
```
Server starts at: **http://localhost:8000**

### Step 6 — Open the dashboard
Go to **http://localhost:8000** in your browser.

### Step 7 — Run tests
```bash
pytest tests/ -v
```
(Run after Step 4 — API tests require trained model)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Dashboard UI |
| POST   | `/predict` | Run yield prediction |
| GET    | `/crops`   | List supported crops/states/seasons |
| GET    | `/trend?state=X&crop=Y&season=Z` | Historical yield data |
| GET    | `/history` | Past predictions from DB |
| GET    | `/metrics` | Model evaluation metrics |
| GET    | `/health`  | Health check |
| GET    | `/docs`    | Auto-generated Swagger UI |

### Example API call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Punjab",
    "crop": "Wheat",
    "season": "Rabi",
    "year": 2024,
    "area_hectares": 1200,
    "rainfall_mm": 650,
    "fertilizer_kg_per_ha": 180,
    "avg_temp_c": 24,
    "irrigation_pct": 75
  }'
```

Response:
```json
{
  "predicted_yield_kg_per_ha": 3847.21,
  "confidence_low": 3540.03,
  "confidence_high": 4154.99,
  "model_r2": 0.9187,
  "model_rmse": 412.35
}
```

---

## Dataset

`data/raw/crop_data.csv` — 2,970 records across:
- **15 Indian states**
- **10 crops** (Rice, Wheat, Maize, Soybean, Sugarcane, Cotton, Mustard, Chickpea, Groundnut, Bajra)
- **3 seasons** (Kharif, Rabi, Whole Year)
- **Years** 2000–2017

Features: Area (ha), Rainfall (mm), Fertilizer (kg/ha), Temperature (°C), Irrigation (%)

---

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.115.5 | REST API framework |
| uvicorn | 0.32.1  | ASGI server |
| pandas  | 2.1.4   | Data manipulation |
| numpy   | 1.26.4  | Numerical operations |
| scikit-learn | 1.5.2 | RF, GBR, ensemble, HPO |
| xgboost | 2.1.3   | XGBoost regressor |
| lightgbm| 4.5.0   | LightGBM regressor |
| matplotlib | 3.9.2 | Chart generation |
| seaborn | 0.13.2  | EDA plots |
| joblib  | 1.4.2   | Model serialisation |
| pytest  | 8.3.3   | Testing |
| httpx   | 0.27.2  | Async HTTP (test client) |
