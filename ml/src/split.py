"""Split pe zone: 20% din zone în test, ales o singură dată cu seed 42.

Toate rândurile unei stații și toate stațiile unei zone ajung în aceeași parte.
Fișierul data/processed/split.json nu se suprascrie fără --force.
"""
import json
import sys

import numpy as np
import pandas as pd

from src import config as C


def _plain(v):
    """ID-uri serializabile în JSON: numerele numpy devin int, textul rămâne text (ex. județe, „RO0001”)."""
    return int(v) if isinstance(v, (int, np.integer)) else v


def make_split(table: pd.DataFrame) -> dict:
    if "is_extra" in table:
        table = table[table["is_extra"] == 0]
    zones = np.sort(table["zone_id"].unique())
    rng = np.random.default_rng(C.SEED)
    n_test = int(round(C.TEST_FRAC * len(zones)))
    test_zones = np.sort(rng.choice(zones, size=n_test, replace=False))
    st_zone = table.drop_duplicates("station_id").set_index("station_id")["zone_id"]
    test_st = st_zone[st_zone.isin(test_zones)].index
    train_st = st_zone[~st_zone.isin(test_zones)].index
    split = {
        "seed": C.SEED,
        "test_frac_zones": C.TEST_FRAC,
        "n_zones": int(len(zones)),
        "test_zones": [_plain(z) for z in test_zones],
        "train_zones": [_plain(z) for z in zones if z not in set(test_zones)],
        "test_stations": [_plain(s) for s in sorted(test_st)],
        "train_stations": [_plain(s) for s in sorted(train_st)],
    }
    assert not set(split["test_zones"]) & set(split["train_zones"])
    assert not set(split["test_stations"]) & set(split["train_stations"])
    return split


def load_split() -> dict:
    return json.loads(C.SPLIT_FILE.read_text())


def split_table(table: pd.DataFrame, split: dict | None = None, include_extra: bool = C.USE_EXTRA_TRAIN):
    """Train/test după split.json.

    Stațiile „extra” (acoperire între COVERAGE_MIN_EXTRA și COVERAGE_MIN) nu sunt evaluate niciodată:
    intră în train doar dacă include_extra și doar dacă zona lor NU e în test; altfel sunt eliminate.
    """
    split = split or load_split()
    extra = table["is_extra"] == 1 if "is_extra" in table else pd.Series(False, index=table.index)
    unknown = set(table.loc[~extra, "station_id"]) - set(split["train_stations"]) - set(split["test_stations"])
    assert not unknown, f"stații care nu apar în split.json: {sorted(unknown)[:10]}"
    in_test_zone = table["zone_id"].isin(split["test_zones"])
    is_test = ~extra & table["station_id"].isin(split["test_stations"])
    is_train = (~extra & ~is_test) | (extra & ~in_test_zone & include_extra)
    train, test = table[is_train].reset_index(drop=True), table[is_test].reset_index(drop=True)
    assert not set(train["zone_id"]) & set(test["zone_id"])
    assert not set(train["station_id"]) & set(test["station_id"])
    return train, test


def main():
    if C.SPLIT_FILE.exists() and "--force" not in sys.argv:
        s = load_split()
        print(f"split.json există deja (nu se suprascrie fără --force): "
              f"{len(s['test_zones'])} zone / {len(s['test_stations'])} stații în test")
        return
    table = pd.read_parquet(C.TRAIN_TABLE)
    split = make_split(table)
    C.SPLIT_FILE.write_text(json.dumps(split, indent=1))
    train, test = split_table(table, split)
    print(f"zone: {len(split['train_zones'])} train / {len(split['test_zones'])} test; "
          f"stații: {len(split['train_stations'])} / {len(split['test_stations'])}; "
          f"rânduri: {len(train):,} / {len(test):,} ({len(test) / len(table):.1%} test)")


if __name__ == "__main__":
    main()
