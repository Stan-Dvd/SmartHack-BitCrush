"""Serviciul de predicție: încarcă modelul LightGBM activ și transformă feature-urile statice în profiluri orare.

Logica de predicție e identică cu ml/src/train.py::predict_y: în modul „occ” modelul prezice rata de ocupare,
care se înmulțește cu n_piles. Ordinea feature-urilor și modul țintei vin din params.json, nu sunt hardcodate.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ROOT, settings

TEMPORAL = ("hour", "is_weekend")
HOURS = np.arange(24)


@dataclass
class LoadedModel:
    model_id: int | None
    booster: lgb.Booster
    features: list[str]
    target_mode: str

    @property
    def static_features(self) -> list[str]:
        return [f for f in self.features if f not in TEMPORAL]


def load_booster(path: Path) -> lgb.Booster:
    # Pe Windows, git poate converti fișierul la CRLF, iar LightGBM nu mai recunoaște formatul.
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return lgb.Booster(model_str=text)


def load_from_files(model_path: Path, params_path: Path, model_id: int | None = None) -> LoadedModel:
    params = json.loads(params_path.read_text(encoding="utf-8"))
    booster = load_booster(model_path)
    features = list(params["features"])
    if booster.feature_name() != features:
        raise ValueError(f"Feature-urile din {model_path.name} nu corespund cu {params_path.name}: "
                         f"{booster.feature_name()} vs {features}")
    return LoadedModel(model_id, booster, features, params.get("target_mode", "y"))


_cache: dict[int, LoadedModel] = {}
_lock = threading.Lock()


def active_model(conn) -> LoadedModel:
    """Modelul marcat activ în model_ml (încărcat o singură dată per model_id)."""
    row = conn.execute(
        "SELECT model_id, cale_fisier, features, target_mode FROM model_ml WHERE activ"
    ).fetchone()
    if not row:
        raise RuntimeError("Nu există niciun model activ în model_ml (rulați python -m etl.register_model).")
    with _lock:
        if row["model_id"] not in _cache:
            booster = load_booster(ROOT / row["cale_fisier"])
            features = list(row["features"])
            if booster.feature_name() != features:
                raise ValueError("Feature-urile modelului nu corespund cu model_ml.features")
            _cache[row["model_id"]] = LoadedModel(row["model_id"], booster, features, row["target_mode"])
        return _cache[row["model_id"]]


def expand_hours(static: pd.DataFrame) -> pd.DataFrame:
    """Fiecare rând static devine 48 de rânduri (24 de ore x is_weekend). Păstrează indexul original în `_row`."""
    n = len(static)
    rep = static.loc[static.index.repeat(48)].copy()
    rep["_row"] = np.repeat(np.arange(n), 48)
    rep["is_weekend"] = np.tile(np.repeat([0, 1], 24), n)
    rep["hour"] = np.tile(np.concatenate([HOURS, HOURS]), n)
    return rep.reset_index(drop=True)


def predict_profiles(model: LoadedModel, static: pd.DataFrame) -> pd.DataFrame:
    """static: un rând per locație, cu coloanele statice ale modelului (inclusiv n_piles și power_kw).

    Întoarce 48 de rânduri per locație: _row, hour, is_weekend, busy_pred, ocupare_pred, kwh_pred.
    """
    missing = [f for f in model.static_features if f not in static.columns]
    if missing:
        raise ValueError(f"Lipsesc feature-urile: {missing}")
    static = static.reset_index(drop=True)
    rows = expand_hours(static)
    X = rows[model.features].astype(float)
    pred = model.booster.predict(X)
    n_piles = rows["n_piles"].to_numpy(dtype=float)
    busy = pred * n_piles if model.target_mode == "occ" else pred
    busy = np.clip(busy, 0.0, n_piles)
    out = rows[["_row", "hour", "is_weekend"]].copy()
    out["busy_pred"] = busy
    out["ocupare_pred"] = busy / n_piles
    out["kwh_pred"] = busy * rows["power_kw"].to_numpy(dtype=float)  # o oră
    return out


def summarize(profile: pd.DataFrame, sat_occ: float = settings.sat_occ) -> pd.DataFrame:
    """Sumarul per locație. Orele de încărcare pe zi = (5·Σ zi lucrătoare + 2·Σ weekend) / 7,
    ca daily_hours din ml/src/evaluate.py."""
    p = profile
    wk = p["is_weekend"].astype(bool)
    w = np.where(wk, 2.0 / 7.0, 5.0 / 7.0)
    agg = pd.DataFrame({
        "_row": p["_row"],
        "ore": p["busy_pred"] * w,
        "kwh": p["kwh_pred"] * w,
        "occ": p["ocupare_pred"] * w / 24.0,
    }).groupby("_row").sum()
    peak = p.loc[p.groupby("_row")["ocupare_pred"].idxmax(), ["_row", "ocupare_pred", "hour", "is_weekend"]]
    peak = peak.set_index("_row")
    out = pd.DataFrame({
        "ore_incarcare_zi": agg["ore"],
        "kwh_zi": agg["kwh"],
        "ocupare_medie": agg["occ"],
        "ocupare_varf": peak["ocupare_pred"],
        "ora_varf": peak["hour"].astype(int),
        "weekend_varf": peak["is_weekend"].astype(bool),
    })
    out["saturata_prezis"] = out["ocupare_varf"] >= sat_occ
    return out.reset_index()
