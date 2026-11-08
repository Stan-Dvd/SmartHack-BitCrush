"""Încărcarea fișierelor UrbanEV (date brute pe stații) și a POI."""
import pandas as pd

from src import config as C


def load_station_info() -> pd.DataFrame:
    """Toate cele 1.682 de stații brute: coordonate, număr de puncte, zonă (TAZID)."""
    df = pd.read_csv(C.STATION_INFO)
    df["station_id"] = df["station_id"].astype(int)
    df["TAZID"] = df["TAZID"].astype(int)
    return df


def load_piles() -> pd.DataFrame:
    """Puterea nominală per punct de încărcare (kW), tip AC/DC."""
    df = pd.read_csv(C.PILE_POWER)
    df["station_id"] = df["station_id"].astype(int)
    df["pile_id"] = df["pile_id"].astype("int64")
    return df


def load_poi() -> pd.DataFrame:
    """POI ca puncte WGS84, coloana `primary_types` are 3 categorii."""
    return pd.read_csv(C.POI_FILE)


def load_station_distance() -> pd.DataFrame:
    """Matricea de distanțe între stații (metri), indexată după station_id pe ambele axe."""
    df = pd.read_csv(C.STATION_DISTANCE, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def list_station_ids() -> list[int]:
    return sorted(int(p.stem) for p in C.CHARGE_5MIN_DIR.glob("*.csv"))


def read_station_series(station_id: int, usecols=("time", "busy", "idle")) -> pd.DataFrame:
    """Seria la 5 minute a unei stații. `busy` = numărul de puncte ocupate."""
    df = pd.read_csv(C.CHARGE_5MIN_DIR / f"{station_id}.csv", usecols=list(usecols))
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    return df
