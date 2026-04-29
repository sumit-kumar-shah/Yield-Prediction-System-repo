"""
train.py
Train Random Forest, Gradient Boosting, XGBoost, LightGBM.
Build stacking ensemble, save model + evaluation charts.
"""
import sys, warnings, os
from pathlib import Path
import numpy as np
import joblib

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import run_pipeline
from feature_engineering import build_features, split_data, FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent.parent / "api" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def make_pipeline(estimator):
    """Wrap any estimator with a median imputer to handle NaNs from lag features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def get_models():
    models = {}

    models["RandomForest"] = (
        make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1)),
        {
            "model__n_estimators":    [100, 200],
            "model__max_depth":       [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
    )
    models["GradientBoosting"] = (
        make_pipeline(GradientBoostingRegressor(random_state=42)),
        {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 5],
            "model__learning_rate": [0.05, 0.1],
        }
    )

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (
            make_pipeline(XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)),
            {
                "model__n_estimators":  [100, 200],
                "model__max_depth":     [4, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample":     [0.8, 1.0],
            }
        )
        print("[train] XGBoost found ✓")
    except ImportError:
        print("[train] XGBoost not installed — skipping")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = (
            make_pipeline(LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)),
            {
                "model__n_estimators":  [100, 200],
                "model__max_depth":     [4, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__num_leaves":    [31, 63],
            }
        )
        print("[train] LightGBM found ✓")
    except ImportError:
        print("[train] LightGBM not installed — skipping")

    return models


def tune_and_train(name, estimator, params, X_tr, y_tr):
    print(f"  Tuning {name}...")
    search = RandomizedSearchCV(
        estimator, params, n_iter=6, cv=3, scoring="r2",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_tr, y_tr)
    print(f"  {name} best CV R² = {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(name, model, X, y, split="Test"):
    preds = model.predict(X)
    r2    = r2_score(y, preds)
    rmse  = float(np.sqrt(mean_squared_error(y, preds)))
    mae   = float(mean_absolute_error(y, preds))
    print(f"  [{split}] {name}: R²={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "preds": preds}


def plot_feature_importance(pipeline, feature_names):
    try:
        imp = pipeline.named_steps["model"].feature_importances_
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(imp)), imp[idx], color="#2E86AB")
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
        ax.set_title("Feature Importance (Random Forest)", fontsize=12)
        ax.set_ylabel("Importance")
        plt.tight_layout()
        plt.savefig(str(STATIC_DIR / "feature_importance.png"), dpi=120)
        plt.close("all")
        print("[train] Feature importance chart saved")
    except Exception as e:
        print(f"[train] Could not save feature importance chart: {e}")


def plot_actual_vs_predicted(y_test, preds):
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, preds, alpha=0.4, s=15, color="#2E86AB", label="Predictions")
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Yield (kg/ha)")
        ax.set_ylabel("Predicted Yield (kg/ha)")
        ax.set_title("Ensemble — Actual vs Predicted")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(STATIC_DIR / "actual_vs_predicted.png"), dpi=120)
        plt.close("all")
        print("[train] Actual vs Predicted chart saved")
    except Exception as e:
        print(f"[train] Could not save chart: {e}")


def train():
    print("\n=== CropYield AI — Training Pipeline ===\n")

    df = run_pipeline()
    df = build_features(df, fit=True)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_data(df)

    # Quick NaN report before training
    total_nan = np.isnan(X_tr).sum()
    print(f"[train] NaNs in training set: {total_nan} (will be imputed by pipeline)")
    print(f"[train] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

    models_def = get_models()
    trained = {}
    for name, (est, params) in models_def.items():
        trained[name] = tune_and_train(name, est, params, X_tr, y_tr)

    print("\n--- Validation scores ---")
    for name, model in trained.items():
        evaluate(name, model, X_val, y_val, "Val")

    # Ensemble: VotingRegressor over already-fitted pipelines
    estimator_list = list(trained.items())
    ensemble = VotingRegressor(estimators=estimator_list, n_jobs=-1)
    ensemble.fit(
        np.vstack([X_tr, X_val]),
        np.concatenate([y_tr, y_val])
    )

    print("\n--- Test scores ---")
    res = evaluate("Ensemble", ensemble, X_te, y_te, "Test")
    for name, model in trained.items():
        evaluate(name, model, X_te, y_te, "Test")

    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.pkl")
    meta = {
        "feature_cols":    FEATURE_COLS,
        "test_r2":         res["r2"],
        "test_rmse":       res["rmse"],
        "test_mae":        res["mae"],
        "models_included": list(trained.keys()),
    }
    joblib.dump(meta, MODELS_DIR / "model_meta.pkl")
    print(f"\n[train] Model saved → {MODELS_DIR / 'ensemble_model.pkl'}")

    rf_pipeline = trained.get("RandomForest")
    if rf_pipeline:
        plot_feature_importance(rf_pipeline, FEATURE_COLS)
    plot_actual_vs_predicted(y_te, res["preds"])

    print(f"\n✓ Training complete. Ensemble Test R² = {res['r2']:.4f}")
    return ensemble, meta


if __name__ == "__main__":
    train()
