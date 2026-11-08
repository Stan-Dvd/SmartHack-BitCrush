"""Feature-uri spațiale: competiție (alte stații), populație WorldPop, POI.

Toate distanțele sunt haversine, în metri. Funcțiile primesc puncte arbitrare (lat, lon), deci se pot
folosi identic și pentru locații candidate noi. Pentru stațiile existente se exclude stația însăși.
"""
import numpy as np
import pandas as pd
import rasterio
from sklearn.neighbors import BallTree

from src import config as C


def _rad(lat, lon) -> np.ndarray:
    return np.deg2rad(np.c_[np.asarray(lat, float), np.asarray(lon, float)])


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * C.EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def competition_features(query: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Competiția față de `stations` (toate stațiile brute).

    query: coloane latitude, longitude și opțional station_id (pentru excluderea stației însăși).
    stations: coloane station_id, latitude, longitude, charge_count.
    """
    tree = BallTree(_rad(stations["latitude"], stations["longitude"]), metric="haversine")
    st_ids = stations["station_id"].to_numpy()
    st_piles = stations["charge_count"].to_numpy()
    q = _rad(query["latitude"], query["longitude"])
    self_ids = query["station_id"].to_numpy() if "station_id" in query else np.full(len(query), -1)

    out = {}
    # cea mai apropiată ALTĂ stație: cerem k vecini, eliminăm stația însăși
    k = min(len(stations), 5)
    dist, idx = tree.query(q, k=k)
    nearest = []
    for d_row, i_row, sid in zip(dist, idx, self_ids):
        mask = st_ids[i_row] != sid
        nearest.append(d_row[mask][0] * C.EARTH_RADIUS_M)
    out["dist_nearest_station_m"] = np.array(nearest)

    for r in C.STATION_RADII_M:
        ind = tree.query_radius(q, r=r / C.EARTH_RADIUS_M)
        name = f"n_stations_{r}m" if r < 1000 else f"n_stations_{r // 1000}km"
        out[name] = np.array([np.sum(st_ids[i] != sid) for i, sid in zip(ind, self_ids)])
        if r == C.COMPETITOR_PILES_RADIUS_M:
            out["competitor_piles_1km"] = np.array(
                [st_piles[i][st_ids[i] != sid].sum() for i, sid in zip(ind, self_ids)]
            )
    return pd.DataFrame(out, index=query.index)


def _worldpop_points():
    with rasterio.open(C.WORLDPOP_CROP) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
    valid = arr > 0
    if nodata is not None:
        valid &= arr != nodata
    rows, cols = np.nonzero(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(ys), np.asarray(xs), arr[rows, cols]


def _radius_name(prefix: str, r: int) -> str:
    return f"{prefix}_{r // 1000}km" if r >= 1000 and r % 1000 == 0 else f"{prefix}_{r}m"


def population_features(query: pd.DataFrame, radii=(C.POP_RADIUS_M,)) -> pd.DataFrame:
    """pop_<rază>: suma populației WorldPop (pixeli de 100 m cu centrul în rază)."""
    lat, lon, val = _worldpop_points()
    tree = BallTree(_rad(lat, lon), metric="haversine")
    q = _rad(query["latitude"], query["longitude"])
    out = {}
    for r in radii:
        ind = tree.query_radius(q, r=r / C.EARTH_RADIUS_M)
        out[_radius_name("pop", r)] = [val[i].sum() for i in ind]
    return pd.DataFrame(out, index=query.index)


def poi_features(query: pd.DataFrame, poi: pd.DataFrame, radius_m: int = C.POI_RADIUS_M) -> pd.DataFrame:
    """Numărul de POI pe categorie pe rază (implicit 500 m, numele din config.POI_CATEGORIES)."""
    q = _rad(query["latitude"], query["longitude"])
    out = {}
    for cat, name in C.POI_CATEGORIES.items():
        if radius_m != C.POI_RADIUS_M:
            name = name.replace("_500m", "_" + _radius_name("x", radius_m).split("_", 1)[1])
        p = poi[poi["primary_types"] == cat]
        tree = BallTree(_rad(p["latitude"], p["longitude"]), metric="haversine")
        out[name] = tree.query_radius(q, r=radius_m / C.EARTH_RADIUS_M, count_only=True)
    return pd.DataFrame(out, index=query.index)


def weighted_competition(query: pd.DataFrame, stations: pd.DataFrame, scale_m: float = 1000,
                         max_m: float = 5000) -> pd.DataFrame:
    """Σ punctele celorlalte stații · exp(−d / scale): competiție netedă, fără praguri de rază."""
    tree = BallTree(_rad(stations["latitude"], stations["longitude"]), metric="haversine")
    st_ids = stations["station_id"].to_numpy()
    st_piles = stations["charge_count"].to_numpy()
    self_ids = query["station_id"].to_numpy() if "station_id" in query else np.full(len(query), -1)
    ind, dist = tree.query_radius(_rad(query["latitude"], query["longitude"]), r=max_m / C.EARTH_RADIUS_M,
                                  return_distance=True)
    vals = [float((st_piles[i] * np.exp(-d * C.EARTH_RADIUS_M / scale_m))[st_ids[i] != sid].sum())
            for i, d, sid in zip(ind, dist, self_ids)]
    return pd.DataFrame({"competition_weighted": vals}, index=query.index)


def ratio_features(f: pd.DataFrame) -> pd.DataFrame:
    """Raporturi cerere/ofertă: populația și POI-urile din jur împărțite la punctele de încărcare din jur."""
    supply = f["competitor_piles_1km"] + f["n_piles"]
    poi_1km = f[[n.replace("_500m", "_1km") for n in C.POI_FEATURES]].sum(axis=1)
    return pd.DataFrame({"pop_per_pile_1km": f["pop_1km"] / supply, "poi_per_pile_1km": poi_1km / supply},
                        index=f.index)
