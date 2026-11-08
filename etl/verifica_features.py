r"""Verifică paritatea feature-urilor calculate în PostGIS (features_locatie) cu cele din setul românesc
care a fost generat offline:
  * --spatial: romania_spatial_features.csv (pop_1km pe grila WorldPop brută, poi_*_500m);
  * --ev-csv:  romania_ev_5min.csv (concurență, n_piles, power_kw; primul rând al fiecărei stații).

    python -m etl.verifica_features --spatial D:\hack\romania_spatial_features.csv --ev-csv D:\hack\romania_ev_5min.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from app.db import close_pool, get_connection, open_pool

COMPETITION = ["dist_nearest_station_m", "n_stations_500m", "n_stations_1km", "n_stations_3km",
               "competitor_piles_1km", "n_piles", "power_kw"]
SPATIAL = ["pop_1km", "poi_food_500m", "poi_business_res_500m", "poi_lifestyle_500m"]


def first_rows(path: Path) -> pd.DataFrame:
    """Primul rând al fiecărei stații din CSV-ul de 5 minute (fișierul e ordonat pe stații)."""
    rows, seen = [], set()
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["station_id"] not in seen:
                seen.add(row["station_id"])
                rows.append({"cod": row["station_id"], **{c: float(row[c]) for c in COMPETITION}})
    return pd.DataFrame(rows)


def compare(db: pd.DataFrame, ref: pd.DataFrame, cols, rel_tol=0.005, abs_tol=1e-6):
    m = db.merge(ref, on="cod", suffixes=("", "_ref"))
    print(f"  stații comparate: {len(m)}")
    ok_all = True
    for c in cols:
        a, b = m[c].astype(float), m[f"{c}_ref"].astype(float)
        bad = ~np.isclose(a, b, rtol=rel_tol, atol=abs_tol)
        ok_all &= not bad.any()
        print(f"  {c:24s} diferențe: {int(bad.sum()):4d}   max |dif| = {float((a - b).abs().max()):.4g}")
        if bad.any():
            print(m.loc[bad, ["cod", c, f"{c}_ref"]].head(5).to_string(index=False))
    return ok_all


def main(spatial: Path | None, ev_csv: Path | None):
    open_pool()
    try:
        with get_connection() as conn:
            # pop_1km din model folosește populația ajustată la INS; setul original a folosit grila brută,
            # deci paritatea se verifică pe suma brută (aceeași definiție spațială).
            db = pd.DataFrame(conn.execute("""
                SELECT s.cod, f.*, f.pop_1km AS pop_1km_ajustat,
                       (SELECT COALESCE(SUM(g.populatie), 0) FROM populatie_grid g
                        WHERE ST_DWithin(g.locatie, s.locatie, 1000, false))::float AS pop_1km_brut
                FROM features_locatie f JOIN statie_incarcare s USING (statie_id)
            """).fetchall())
            db["pop_1km"] = db["pop_1km_brut"]
    finally:
        close_pool()
    ok = True
    if spatial:
        print(f"Populație și POI vs {spatial.name}:")
        ok &= compare(db, pd.read_csv(spatial).rename(columns={"station_id": "cod"}), SPATIAL)
    if ev_csv:
        print(f"Concurență și stație vs {ev_csv.name} (citire completă, ~1 min):")
        ok &= compare(db, first_rows(ev_csv), COMPETITION)
    print("REZULTAT:", "paritate" if ok else "există diferențe (vezi mai sus)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spatial", type=Path)
    ap.add_argument("--ev-csv", type=Path)
    a = ap.parse_args()
    main(a.spatial, a.ev_csv)
