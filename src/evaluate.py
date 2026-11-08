"""Evaluarea pe setul de test (zone ținute deoparte): B0, B1, LightGBM.

Rând (stație × oră × tip zi): MAE, RMSE, R² pe y.
Stație: MAE pe orele de încărcare pe zi = (5·Σ weekday + 2·Σ weekend) / 7, plus Spearman
        între cererea zilnică prezisă și cea reală (contează pentru clasarea locațiilor).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as C
from src.baselines import B0, B1
from src.split import split_table
from src.train import load_model

MODEL_NAMES = {"B0": "B0 (profil oraș)", "B1": "B1 (ocupare × n_piles)", "LGBM": "LightGBM"}


def daily_hours(df: pd.DataFrame, col: str) -> pd.Series:
    """Ore de încărcare pe zi per stație, doar pentru stațiile cu toate cele 48 de celule."""
    full = df.groupby("station_id")["hour"].transform("size") == 48
    s = df[full].groupby(["station_id", "is_weekend"])[col].sum().unstack("is_weekend")
    return (5 * s[0] + 2 * s[1]) / 7


def predict_all(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    booster, meta = load_model()
    out = test.copy()
    out["pred_B0"] = B0().fit(train).predict(test)
    out["pred_B1"] = B1().fit(train).predict(test)
    out["pred_LGBM"] = booster.predict(test[meta["features"]])
    return out


def compute_metrics(pred: pd.DataFrame) -> dict:
    res = {}
    real_daily = daily_hours(pred, "y")
    for m in MODEL_NAMES:
        p = pred[f"pred_{m}"]
        d = daily_hours(pred, f"pred_{m}")
        res[m] = {
            "row_MAE": float(mean_absolute_error(pred["y"], p)),
            "row_RMSE": float(np.sqrt(mean_squared_error(pred["y"], p))),
            "row_R2": float(r2_score(pred["y"], p)),
            "station_daily_hours_MAE": float(np.mean(np.abs(d - real_daily))),
            # B0 prezice aceeași valoare pentru toate stațiile -> clasarea nu e definită
            "station_daily_spearman": (float(spearmanr(d, real_daily).statistic)
                                       if d.nunique() > 1 else None),
        }
    return res


def table_md(metrics: dict, info: dict) -> str:
    lines = [
        "| Model | MAE (rând) | RMSE (rând) | R² (rând) | MAE ore încărcare/zi (stație) | Spearman cerere zilnică |",
        "|---|---|---|---|---|---|",
    ]
    for m, name in MODEL_NAMES.items():
        r = metrics[m]
        sp = "—" if r["station_daily_spearman"] is None else f"{r['station_daily_spearman']:.3f}"
        lines.append(f"| {name} | {r['row_MAE']:.3f} | {r['row_RMSE']:.3f} | {r['row_R2']:.3f} | "
                     f"{r['station_daily_hours_MAE']:.2f} | {sp} |")
    lines.append("")
    lines.append(f"Set de test: {info['n_test_rows']:,} rânduri, {info['n_test_stations']} stații, "
                 f"{info['n_test_zones']} zone (metricile pe stație: {info['n_test_stations_full']} stații "
                 f"cu toate cele 48 de celule). Media lui y în test: {info['y_mean_test']:.3f}; "
                 f"ore de încărcare pe zi, în medie: {info['daily_mean_test']:.1f}.")
    return "\n".join(lines)


def evaluate():
    table = pd.read_parquet(C.TRAIN_TABLE)
    train, test = split_table(table)
    pred = predict_all(train, test)
    metrics = compute_metrics(pred)
    _, meta = load_model()
    info = {
        "n_test_rows": int(len(test)), "n_test_stations": int(test["station_id"].nunique()),
        "n_test_zones": int(test["zone_id"].nunique()),
        "n_test_stations_full": int(len(daily_hours(pred, "y"))),
        "y_mean_test": float(test["y"].mean()), "daily_mean_test": float(daily_hours(pred, "y").mean()),
        "model": {k: meta[k] for k in ("feature_set", "monotone", "num_boost_round", "cv_mae", "b1_cv_mae")},
        "model_params": {k: meta["params"][k] for k in ("objective", "num_leaves", "min_child_samples")},
    }
    out = {"metrics": metrics, "info": info}
    C.METRICS_FILE.write_text(json.dumps(out, indent=2))
    md = table_md(metrics, info)
    C.METRICS_TABLE.write_text(md + "\n")
    pred.to_parquet(C.PREDICTIONS_FILE, index=False)
    print(md)
    return out


if __name__ == "__main__":
    evaluate()
