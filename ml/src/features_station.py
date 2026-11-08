"""Feature-uri despre stație: n_piles și power_kw.

- n_piles = charge_count din station_information (egal cu busy + idle în serie).
- fast_share = fast_count / charge_count (candidat).
- power_kw = media puterii nominale pe punctele stației (pile_rated_power.csv); punctele cu 0 kW
  sunt tratate ca lipsă. Pentru stațiile fără niciun punct cu putere cunoscută:
  (slow_count * median_AC + fast_count * median_DC) / charge_count, cu medianele din fișierul de puncte.
"""
import numpy as np
import pandas as pd

from src.load_urbanev import load_piles


def station_features(info: pd.DataFrame, piles: pd.DataFrame | None = None) -> pd.DataFrame:
    piles = load_piles() if piles is None else piles
    p = piles[piles["power"] > 0]
    med_ac = float(p.loc[p["pileType"] == "AC", "power"].median())
    med_dc = float(p.loc[p["pileType"] == "DC", "power"].median())
    mean_power = p.groupby("station_id")["power"].mean()

    out = info[["station_id"]].copy()
    out["n_piles"] = info["charge_count"].astype(int)
    out["power_kw"] = info["station_id"].map(mean_power)
    out["power_imputed"] = out["power_kw"].isna().astype(int)
    fallback = (info["slow_count"] * med_ac + info["fast_count"] * med_dc) / info["charge_count"]
    out["power_kw"] = out["power_kw"].fillna(fallback)
    out["fast_share"] = info["fast_count"] / info["charge_count"]  # ponderea punctelor rapide (DC)
    assert out["power_kw"].notna().all() and np.isfinite(out["power_kw"]).all()
    out.attrs["median_ac_kw"] = med_ac
    out.attrs["median_dc_kw"] = med_dc
    return out
