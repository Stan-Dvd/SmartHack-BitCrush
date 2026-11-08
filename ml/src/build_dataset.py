"""Construiește tabelul de antrenare: un rând per (stație, oră, is_weekend).

    station_id | zone_id | [feature-uri spațiale] | n_piles | power_kw | hour | is_weekend | y

Feature-urile de stație și cele spațiale se calculează pentru toate cele 1.682 de stații brute
(salvate în station_features.parquet), iar competiția folosește toate stațiile ca infrastructură.
"""
import pandas as pd

from src import config as C
from src.features_spatial import (
    competition_features,
    poi_features,
    population_features,
    ratio_features,
    weighted_competition,
)
from src.features_osm import osm_features
from src.features_station import station_features
from src.load_urbanev import load_poi, load_station_info
from src.target import build_target


def build_station_features() -> pd.DataFrame:
    info = load_station_info()
    st = station_features(info)
    comp = competition_features(info, info)
    poi_df = load_poi()
    poi = poi_features(info, poi_df)
    poi_1km = poi_features(info, poi_df, radius_m=1000)
    wcomp = weighted_competition(info, info)
    if C.WORLDPOP_CROP.exists():
        pop = population_features(info, radii=C.POP_RADII_M)
    else:
        print(f"AVERTISMENT: lipsește {C.WORLDPOP_CROP.relative_to(C.ROOT)} (rulează python -m src.download); "
              "pop_* = NaN, se poate antrena doar cu --feature-set base")
        pop = pd.DataFrame({f"pop_{n}": float("nan") for n in ("500m", "1km", "3km")}, index=info.index)
    feats = pd.concat(
        [info[["station_id", "TAZID", "latitude", "longitude"]].rename(columns={"TAZID": "zone_id"}),
         comp, wcomp, pop, poi, poi_1km, st.drop(columns="station_id")],
        axis=1,
    )
    feats = pd.concat([feats, ratio_features(feats)], axis=1)
    if C.OSM_PBF.exists():
        feats = pd.concat([feats, osm_features(info)], axis=1)
    else:
        print(f"AVERTISMENT: lipsește {C.OSM_PBF.relative_to(C.ROOT)} (rulează python -m src.download); "
              "feature-urile OSM = NaN")
        feats = feats.assign(**{f: float("nan") for f in C.CANDIDATE_OSM})
    feats.to_parquet(C.STATION_FEATURES_FILE, index=False)
    return feats


def build_dataset(rebuild_target: bool = True) -> pd.DataFrame:
    target = build_target()[0] if rebuild_target or not C.TARGET_FILE.exists() else pd.read_parquet(C.TARGET_FILE)
    feats = build_station_features()
    # feature-urile modelului + candidații (pentru experimente), fără duplicate, fără cele temporale
    static = [f for f in dict.fromkeys([*C.FEATURES, *C.CANDIDATE_FEATURES]) if f not in C.TEMPORAL_FEATURES]
    cols = ["station_id", "zone_id", *static, "power_imputed"]
    table = target.merge(feats[cols], on="station_id", how="left", validate="many_to_one")
    table = table[[*C.ID_COLS, *static, "power_imputed", *C.TEMPORAL_FEATURES, C.TARGET, "n_obs", "is_extra"]]
    table = table.sort_values(["station_id", "is_weekend", "hour"]).reset_index(drop=True)

    # verificări anti-leakage și de integritate
    assert not (set(C.FEATURES + C.CANDIDATE_FEATURES) & C.FORBIDDEN_FEATURES), "feature interzis"
    must_be_complete = [f for f in [*C.FEATURES, *C.CANDIDATE_FEATURES]
                        if (not f.startswith("pop_") or C.WORLDPOP_CROP.exists())
                        and (f not in C.CANDIDATE_OSM or C.OSM_PBF.exists())]
    assert table[must_be_complete + [C.TARGET]].notna().all().all(), "NaN în feature-uri sau țintă"
    assert not table.duplicated(["station_id", "hour", "is_weekend"]).any()
    assert table.groupby("station_id")["zone_id"].nunique().max() == 1

    table.to_parquet(C.TRAIN_TABLE, index=False)
    n_extra = table.loc[table["is_extra"] == 1, "station_id"].nunique()
    print(f"train_table: {len(table):,} rânduri, {table['station_id'].nunique()} stații "
          f"({n_extra} extra, doar pentru train), {table['zone_id'].nunique()} zone "
          f"-> {C.TRAIN_TABLE.relative_to(C.ROOT)}")
    return table


if __name__ == "__main__":
    import sys
    build_dataset(rebuild_target="--reuse-target" not in sys.argv)
