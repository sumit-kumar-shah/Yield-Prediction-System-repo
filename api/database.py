"""
database.py
SQLite persistence for prediction history.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT,
            state     TEXT,
            crop      TEXT,
            season    TEXT,
            year      INTEGER,
            area      REAL,
            rainfall  REAL,
            fertilizer REAL,
            temp      REAL,
            irrigation REAL,
            predicted REAL,
            conf_low  REAL,
            conf_high REAL
        )
    """)
    con.commit()
    con.close()


def insert_prediction(data: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO predictions
        (ts, state, crop, season, year, area, rainfall, fertilizer, temp,
         irrigation, predicted, conf_low, conf_high)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        data["state"], data["crop"], data["season"], data["year"],
        data["area_hectares"], data["rainfall_mm"],
        data["fertilizer_kg_per_ha"], data["avg_temp_c"],
        data["irrigation_pct"],
        data["predicted_yield_kg_per_ha"],
        data["confidence_low"], data["confidence_high"],
    ))
    con.commit()
    con.close()


def get_history(limit: int = 50) -> list:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
