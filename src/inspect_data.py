"""Pasul 0: inspectarea datelor brute. Răspunsurile sunt notate în docs/data_notes.md."""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config as C
from src.features_spatial import haversine_m
from src.load_urbanev import (
    list_station_ids,
    load_piles,
    load_poi,
    load_station_distance,
    load_station_info,
)
from src.target import _station_aggregate

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def describe(name: str, df: pd.DataFrame, show_cols: int | None = None):
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    print(f"rânduri: {len(df):,}   coloane: {df.shape[1]:,}")
    cols = df.columns[:show_cols] if show_cols else df.columns
    print("tipuri:\n", df[cols].dtypes.to_string())
    print("primele rânduri:\n", df[cols].head().to_string())
    miss = df.isna().mean().mul(100)
    print(f"% lipsă (max pe coloane): {miss.max():.2f}%")
    if show_cols is None:
        print(miss.round(2).to_string())


def main():
    info = load_station_info()
    piles = load_piles()
    poi = load_poi()
    dist = load_station_distance()
    describe("station_information.csv", info)
    describe("pile_rated_power.csv", piles)
    dist_df = pd.concat([pd.Series(dist.index, name="ID"), dist.reset_index(drop=True)], axis=1)
    describe("station_distance.csv (primele 6 coloane)", dist_df, show_cols=6)
    describe("poi.csv", poi)

    ids = list_station_ids()
    sample = pd.read_csv(C.CHARGE_5MIN_DIR / f"{ids[0]}.csv")
    describe(f"charge_5min/{ids[0]}.csv (exemplu)", sample)

    print("\nAgregare pe toate fișierele charge_5min ...")
    with Pool(C.N_JOBS) as pool:
        stats = pd.DataFrame([r[0] for r in pool.map(_station_aggregate, ids, chunksize=8)])
    stats = stats.merge(info[["station_id", "charge_count"]], on="station_id")
    print(stats[["n_intervals", "coverage", "n_piles_series", "sat_share", "busy_mean"]].describe().round(3))
    for t in (0.3, 0.5, 0.7, 0.9):
        print(f"stații cu acoperire >= {t:.0%}: {(stats['coverage'] >= t).sum()}")
    print("stații fără nicio observație:", int((stats["n_available"] == 0).sum()))
    print("prima observație (top):\n", stats["first_time"].value_counts().head(3).to_string())
    print("ultima observație (top):\n", stats["last_time"].value_counts().head(3).to_string())

    print(f"\n{'=' * 80}\nRĂSPUNSURI PASUL 0\n{'=' * 80}")
    has = stats["n_piles_series"].notna()
    print("Q1 busy: format lung, un fișier per stație; busy = nr. puncte ocupate per stație.")
    print(f"   busy+idle constant în timp: {(stats.loc[has, 'n_piles_series_nunique'] == 1).mean():.2%} din stațiile cu date")
    print(f"   busy+idle == charge_count: {(stats.loc[has, 'n_piles_series'] == stats.loc[has, 'charge_count']).mean():.2%}")
    print(f"   busy maxim > busy+idle: {int((sample['busy'] > sample['busy'] + sample['idle']).sum())} (exemplu)")

    pile_st = piles.groupby("station_id").size()
    print("Q2 putere per PUNCT (pile_rated_power.csv), număr de puncte în station_information (charge_count).")
    print(f"   puncte: {len(piles):,}; cu putere 0: {(piles['power'] == 0).sum()}; tipuri: {piles['pileType'].value_counts().to_dict()}")
    print(f"   stații cu puncte în fișier: {pile_st.size} / {len(info)}; fără: {len(info) - pile_st.size}")
    match = info.set_index("station_id")["charge_count"].reindex(pile_st.index) == pile_st
    print(f"   nr. puncte în fișier == charge_count: {match.mean():.2%}")

    print(f"Q3 zonă: TAZID, {info['TAZID'].nunique()} zone unice, {info['TAZID'].isna().sum()} lipsă.")

    print(f"Q4 POI: puncte cu coordonate ({len(poi):,}), categorii: {poi['primary_types'].value_counts().to_dict()}")

    s_info, s_files = set(info["station_id"]), set(ids)
    s_piles, s_dist = set(piles["station_id"]), set(dist.index)
    print("Q5 ID-uri: station_id întreg; fișierele charge_5min se numesc {station_id}.csv.")
    print(f"   info == fișiere serie: {s_info == s_files}; info == matrice distanțe: {s_info == s_dist}; "
          f"piles ⊆ info: {s_piles <= s_info}")

    # verificare: haversine vs station_distance.csv
    st = info.set_index("station_id").loc[dist.index]
    lat, lon = st["latitude"].to_numpy(), st["longitude"].to_numpy()
    hv = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    diff = np.abs(hv - dist.to_numpy())
    rel = diff[dist.to_numpy() > 100] / dist.to_numpy()[dist.to_numpy() > 100]
    print(f"\nhaversine vs station_distance.csv: diferență absolută medie {diff.mean():.1f} m, "
          f"max {diff.max():.1f} m; relativă mediană {np.median(rel):.4%}, p99 {np.quantile(rel, 0.99):.4%}")
    dup = info.duplicated(["longitude", "latitude"], keep=False)
    print(f"stații cu coordonate identice cu altă stație: {int(dup.sum())}")
    print("bbox stații:", info[["longitude", "latitude"]].agg(["min", "max"]).to_dict())


if __name__ == "__main__":
    main()
