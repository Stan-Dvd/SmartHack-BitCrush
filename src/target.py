"""Ținta: busy la 5 min -> număr mediu de puncte ocupate per (stație, oră, is_weekend).

Ieșiri:
  data/processed/target.parquet        station_id, hour, is_weekend, y, n_obs
  data/processed/station_stats.parquet coverage, sat_share, n_piles_series etc. (doar pentru analiză)
  reports/data_quality.json            ce s-a eliminat și de ce
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config as C
from src.features_temporal import add_temporal
from src.load_urbanev import list_station_ids, load_station_info, read_station_series


def _station_aggregate(station_id: int):
    s = read_station_series(station_id)
    n_total = len(s)
    ok = s["busy"].notna()
    s = s[ok]
    stats = dict(
        station_id=station_id,
        n_intervals=n_total,
        n_available=int(ok.sum()),
        coverage=float(ok.mean()) if n_total else 0.0,
    )
    if s.empty:
        stats.update(n_piles_series=np.nan, n_piles_series_nunique=0, sat_share=np.nan, zero_share=np.nan,
                     first_time=None, last_time=None, busy_mean=np.nan)
        return stats, None

    total = s["busy"] + s["idle"]
    n_piles_series = float(total.mode().iloc[0])
    stats.update(
        n_piles_series=n_piles_series,
        n_piles_series_nunique=int(total.nunique()),
        sat_share=float((s["busy"] >= total).mean()),  # toate punctele ocupate
        zero_share=float((s["busy"] == 0).mean()),
        first_time=str(s["time"].iloc[0]),
        last_time=str(s["time"].iloc[-1]),
        busy_mean=float(s["busy"].mean()),
    )
    cells = (
        add_temporal(s)
        .groupby(["hour", "is_weekend"], as_index=False)
        .agg(y=("busy", "mean"), n_obs=("busy", "size"))
    )
    cells.insert(0, "station_id", station_id)
    return stats, cells


def compute_raw_aggregates():
    ids = list_station_ids()
    with Pool(C.N_JOBS) as pool:
        res = pool.map(_station_aggregate, ids, chunksize=8)
    stats = pd.DataFrame([r[0] for r in res])
    cells = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    return stats, cells


def build_target():
    info = load_station_info()
    stats, cells = compute_raw_aggregates()
    stats = stats.merge(info[["station_id", "charge_count"]], on="station_id", how="left")

    n_all = len(stats)
    no_data = stats["n_available"] == 0
    low_cov = (~no_data) & (stats["coverage"] < C.COVERAGE_MIN_EXTRA)
    # serii „blocate pe plin”: toate punctele ocupate aproape permanent timp de luni -> artefact de raportare
    stuck_full = (~no_data) & (~low_cov) & (stats["sat_share"] > C.STUCK_FULL_SHARE)
    keep_ids = set(stats.loc[~no_data & ~low_cov & ~stuck_full, "station_id"])
    # acoperire între COVERAGE_MIN_EXTRA și COVERAGE_MIN: doar pentru antrenare, niciodată în evaluare
    extra_ids = set(stats.loc[stats["coverage"] < C.COVERAGE_MIN, "station_id"]) & keep_ids

    cells_kept_st = cells[cells["station_id"].isin(keep_ids)]
    few_obs = cells_kept_st["n_obs"] < C.MIN_CELL_OBS
    target = cells_kept_st[~few_obs].reset_index(drop=True)
    target["hour"] = target["hour"].astype(int)
    target["is_weekend"] = target["is_weekend"].astype(int)
    target["is_extra"] = target["station_id"].isin(extra_ids).astype(int)

    stats["kept"] = stats["station_id"].isin(set(target["station_id"]))
    stats["is_extra"] = stats["station_id"].isin(extra_ids)
    mismatch = stats["n_piles_series"].notna() & (stats["n_piles_series"] != stats["charge_count"])

    cells_per_station = target.groupby("station_id").size()
    quality = {
        "coverage_min": C.COVERAGE_MIN,
        "coverage_min_extra_train_only": C.COVERAGE_MIN_EXTRA,
        "min_cell_obs": C.MIN_CELL_OBS,
        "stations_raw": int(n_all),
        "stations_removed_no_data": int(no_data.sum()),
        "stations_removed_low_coverage": int(low_cov.sum()),
        "stations_extra_train_only": int((stats["kept"] & stats["is_extra"]).sum()),
        "stuck_full_share": C.STUCK_FULL_SHARE,
        "stations_removed_stuck_full": int(stuck_full.sum()),
        "stations_removed_stuck_full_ids": [int(s) for s in stats.loc[stuck_full, "station_id"]],
        "stations_kept": int(stats["kept"].sum()),
        "stations_kept_evaluable": int((stats["kept"] & ~stats["is_extra"]).sum()),
        "stations_kept_almost_always_zero": int((stats["kept"] & (stats["zero_share"] > 0.99)).sum()),
        "cells_possible_for_kept_stations": int(len(keep_ids) * 48),
        "cells_present_for_kept_stations": int(len(cells_kept_st)),
        "cells_missing_entirely": int(len(keep_ids) * 48 - len(cells_kept_st)),
        "cells_removed_few_obs": int(few_obs.sum()),
        "cells_kept": int(len(target)),
        "stations_with_all_48_cells": int((cells_per_station == 48).sum()),
        "stations_busy_plus_idle_not_constant": int((stats["n_piles_series_nunique"] > 1).sum()),
        "stations_busy_plus_idle_ne_charge_count": int(mismatch.sum()),
        "coverage_kept_stations": stats.loc[stats["kept"], "coverage"].describe().round(4).to_dict(),
        "n_obs_per_cell": target["n_obs"].describe().round(2).to_dict(),
        "y_summary": target["y"].describe().round(4).to_dict(),
    }

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    target.to_parquet(C.TARGET_FILE, index=False)
    stats.to_parquet(C.STATION_STATS_FILE, index=False)
    C.DATA_QUALITY_FILE.write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in quality.items() if not isinstance(v, dict)}, indent=2))
    return target, stats


if __name__ == "__main__":
    build_target()
