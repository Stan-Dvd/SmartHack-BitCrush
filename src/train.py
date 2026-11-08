"""Antrenarea LightGBM: tuning cu GroupKFold pe zone (doar train), apoi modelul final pe tot train-ul.

Două moduri pentru țintă, comparate în CV (MAE se calculează mereu pe y, în puncte ocupate):
  - "y":   modelul prezice direct y (puncte ocupate în medie);
  - "occ": modelul prezice rata de ocupare y / n_piles, cu pondere n_piles, iar predicția se înmulțește
           cu n_piles. Cu ponderea n_piles, L1 ponderat pe ocupare e proporțional cu MAE pe y.

Usage:
    python -m src.train                      # modelul final: grid complet, constrângerile din config.MONOTONE
    python -m src.train --no-monotone        # fără constrângeri monotone (ablație)
    python -m src.train --feature-set base   # fără POI și populație (ablație)
"""
import argparse
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config as C
from src.baselines import B1
from src.split import split_table

# (mod țintă, obiectiv): pentru "y" L2 și tweedie (planul), pentru "occ" L2, cross-entropy și tweedie
OBJECTIVES = [
    ("y", {"objective": "regression"}),
    ("y", {"objective": "tweedie", "tweedie_variance_power": 1.2}),
    ("y", {"objective": "tweedie", "tweedie_variance_power": 1.5}),
    ("occ", {"objective": "regression"}),
    ("occ", {"objective": "cross_entropy"}),
    ("occ", {"objective": "tweedie", "tweedie_variance_power": 1.2}),
]
# rândurile unei stații sunt puternic corelate (48 per stație), deci min_child_samples mare (~10-20 stații)
# și arbori mici sunt regularizarea naturală; grila inițială (15-63 frunze, 50-200) era la marginea ei
GRID = {"num_leaves": [4, 7, 15, 31], "min_child_samples": [200, 500, 1000]}


def feature_list(feature_set: str) -> list[str]:
    if feature_set == "full":
        return list(C.FEATURES)
    if feature_set == "base":
        return [f for f in C.FEATURES if f not in C.CONTEXT_FEATURES]
    raise ValueError(feature_set)


def monotone_vector(features, use: bool):
    return [C.MONOTONE.get(f, 0) if use else 0 for f in features]


def make_params(obj: dict, extra: dict, features, monotone: bool) -> dict:
    p = {**C.LGBM_BASE_PARAMS, **obj, **extra, "metric": "l1"}
    if monotone:
        p["monotone_constraints"] = monotone_vector(features, True)
    return p


def label_weight(df: pd.DataFrame, target_mode: str):
    if target_mode == "occ":
        return (df[C.TARGET] / df["n_piles"]).clip(0, 1), df["n_piles"].astype(float)
    return df[C.TARGET], None


def predict_y(booster, df: pd.DataFrame, features, target_mode: str, num_iteration=None) -> np.ndarray:
    """Predicția în unitatea lui y (puncte ocupate), oricare ar fi modul țintei."""
    p = booster.predict(df[features], num_iteration=num_iteration)
    return p * df["n_piles"].to_numpy() if target_mode == "occ" else p


def fit_booster(df, features, params, target_mode, num_boost_round, valid=None):
    lab, w = label_weight(df, target_mode)
    dtr = lgb.Dataset(df[features], lab, weight=w)
    kw = {}
    if valid is not None:
        vlab, vw = label_weight(valid, target_mode)
        kw = dict(valid_sets=[lgb.Dataset(valid[features], vlab, weight=vw, reference=dtr)],
                  callbacks=[lgb.early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False)])
    return lgb.train(params, dtr, num_boost_round=num_boost_round, **kw)


def cv_folds(train: pd.DataFrame):
    return list(GroupKFold(n_splits=C.N_FOLDS).split(train, groups=train["zone_id"]))


def run_cv(train, features, params, folds, target_mode="y"):
    maes, iters = [], []
    for tr_idx, va_idx in folds:
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        booster = fit_booster(tr, features, params, target_mode, C.MAX_BOOST_ROUNDS, valid=va)
        pred = predict_y(booster, va, features, target_mode, num_iteration=booster.best_iteration)
        maes.append(float(np.mean(np.abs(pred - va[C.TARGET]))))
        iters.append(int(booster.best_iteration))
    return maes, iters


def baseline_cv(train, folds):
    maes = []
    for tr_idx, va_idx in folds:
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        maes.append(float(np.mean(np.abs(B1().fit(tr).predict(va) - va[C.TARGET]))))
    return maes


def train(feature_set="full", monotone=C.USE_MONOTONE, save=True, quick=False):
    table = pd.read_parquet(C.TRAIN_TABLE)
    tr, _ = split_table(table)
    features = feature_list(feature_set)
    folds = cv_folds(tr)
    b1_cv = baseline_cv(tr, folds)
    print(f"B1 CV MAE: {np.mean(b1_cv):.4f}")

    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    if quick:
        grid = [{"num_leaves": 31, "min_child_samples": 50}]
    rows = []
    for (mode, obj), extra in itertools.product(OBJECTIVES, grid):
        params = make_params(obj, extra, features, monotone)
        t = time.time()
        maes, iters = run_cv(tr, features, params, folds, mode)
        rows.append({"target_mode": mode, **obj, **extra, "cv_mae": np.mean(maes), "cv_mae_std": np.std(maes),
                     "best_iter_mean": np.mean(iters), "fold_maes": maes, "fold_iters": iters,
                     "secs": round(time.time() - t, 1)})
        print(f"{mode:3s} {json.dumps({**obj, **extra})}: CV MAE {np.mean(maes):.4f} ± {np.std(maes):.4f}, "
              f"iter {np.mean(iters):.0f}")
    cv = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    best = cv.iloc[0]
    mode = best["target_mode"]
    obj = {"objective": best["objective"]}
    if best["objective"] == "tweedie":
        obj["tweedie_variance_power"] = float(best["tweedie_variance_power"])
    extra = {k: int(best[k]) for k in GRID}
    params = make_params(obj, extra, features, monotone)
    n_rounds = int(round(best["best_iter_mean"]))

    booster = fit_booster(tr, features, params, mode, n_rounds)
    meta = {
        "features": features, "feature_set": feature_set, "monotone": monotone,
        "monotone_dict": C.MONOTONE if monotone else {}, "target_mode": mode,
        "params": params, "num_boost_round": n_rounds,
        "cv_mae": float(best["cv_mae"]), "cv_mae_std": float(best["cv_mae_std"]),
        "cv_fold_maes": list(best["fold_maes"]), "b1_cv_mae": float(np.mean(b1_cv)), "b1_cv_fold_maes": b1_cv,
        "n_train_rows": int(len(tr)), "n_train_stations": int(tr["station_id"].nunique()),
        "n_train_zones": int(tr["zone_id"].nunique()),
    }
    print(f"\nCel mai bun: {mode} {json.dumps({**obj, **extra})} -> CV MAE {best['cv_mae']:.4f} "
          f"(B1 {np.mean(b1_cv):.4f}), {n_rounds} runde")
    if save:
        C.MODELS.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(C.MODEL_FILE))
        C.PARAMS_FILE.write_text(json.dumps(meta, indent=2))
        cv.assign(feature_set=feature_set, monotone=monotone).to_csv(C.CV_RESULTS_FILE, index=False)
    return booster, meta, cv


def load_model():
    return lgb.Booster(model_file=str(C.MODEL_FILE)), json.loads(C.PARAMS_FILE.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="full", choices=["full", "base"])
    ap.add_argument("--no-monotone", action="store_true", help="fără constrângerile din config.MONOTONE")
    ap.add_argument("--quick", action="store_true", help="doar parametrii de pornire, fără grid")
    a = ap.parse_args()
    train(a.feature_set, not a.no_monotone, quick=a.quick)


if __name__ == "__main__":
    main()
