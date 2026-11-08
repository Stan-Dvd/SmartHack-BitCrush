"""Feature-uri OpenStreetMap despre tipul locului (candidați; intră în model doar dacă câștigă în CV).

Extragerea (o singură dată, cache în data/raw/osm/):
  - destinația terenului (landuse) clasificată pe o grilă de ~100 m peste BBOX: residential / industrial / commercial
  - drumuri mari (motorway, trunk, primary + link-uri), eșantionate la ~25 m
  - parcări, gări (tren și metrou), autogări, clădiri: câte un punct per obiect

Feature-urile (haversine, BallTree), calculabile pentru orice punct, deci și pentru locații noi și pentru România:
  landuse_<clasă>_500m   fracțiunea din cercul de 500 m cu acea destinație
  dist_major_road_m, major_road_km_1km
  n_parking_500m, dist_parking_m, dist_rail_station_m, dist_bus_station_m, n_buildings_500m
"""
import numpy as np
import osmium
import pandas as pd
import shapely
from shapely import wkb as shapely_wkb
from sklearn.neighbors import BallTree

from src import config as C
from src.features_spatial import _rad

LANDUSE_CLASSES = {
    "residential": {"residential"},
    "industrial": {"industrial", "port", "railway"},
    "commercial": {"commercial", "retail"},
}
MAJOR_ROADS = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
ROAD_STEP_M = 25
GRID_STEP_M = 100
POINTS_FILE = C.OSM_DIR / "shenzhen_points.parquet"
GRID_FILE = C.OSM_DIR / "shenzhen_landuse_grid.parquet"


def _in_bbox(lon, lat) -> bool:
    b = C.BBOX
    return b["lon_min"] <= lon <= b["lon_max"] and b["lat_min"] <= lat <= b["lat_max"]


def _point_kind(tags) -> str | None:
    if tags.get("amenity") == "parking":
        return "parking"
    if tags.get("railway") in ("station", "halt") or tags.get("public_transport") == "station" and (
            "train" in tags or "subway" in tags or tags.get("station") in ("subway", "train", "light_rail")):
        return "rail_station"
    if tags.get("amenity") == "bus_station":
        return "bus_station"
    return None


def extract_osm():
    """Parcurge extractul Guangdong și salvează punctele și grila de landuse pentru BBOX."""
    wkbf = osmium.geom.WKBFactory()
    points, landuse_polys, landuse_cls = [], [], []
    road_lines = []
    fp = (osmium.FileProcessor(str(C.OSM_PBF))
          .with_locations()
          .with_areas()
          .with_filter(osmium.filter.KeyFilter("landuse", "highway", "amenity", "building", "railway",
                                               "public_transport")))
    for obj in fp:
        tags = obj.tags
        if obj.is_node():
            loc = obj.location
            if not loc.valid() or not _in_bbox(loc.lon, loc.lat):
                continue
            kind = _point_kind(tags)
            if kind:
                points.append((kind, loc.lon, loc.lat))
            elif "building" in tags:
                points.append(("building", loc.lon, loc.lat))
        elif obj.is_way():
            try:
                loc = obj.nodes[0].location
            except Exception:  # noqa: BLE001
                continue
            if not loc.valid() or not _in_bbox(loc.lon, loc.lat):
                continue
            if tags.get("highway") in MAJOR_ROADS:
                try:
                    road_lines.append(shapely_wkb.loads(wkbf.create_linestring(obj), hex=True))
                except Exception:  # noqa: BLE001
                    pass
            elif obj.is_closed() and "building" in tags:
                # clădirile: un punct per obiect (primul nod), suficient pentru numărători pe 500 m
                points.append(("building", loc.lon, loc.lat))
        elif obj.is_area():
            lu = tags.get("landuse")
            kind = _point_kind(tags)
            if lu is None and kind is None:
                continue
            try:
                geom = shapely_wkb.loads(wkbf.create_multipolygon(obj), hex=True)
            except Exception:  # noqa: BLE001
                continue
            c = geom.representative_point()
            if not _in_bbox(c.x, c.y):
                continue
            if kind:
                points.append((kind, c.x, c.y))
            for cls, values in LANDUSE_CLASSES.items():
                if lu in values:
                    landuse_polys.append(geom)
                    landuse_cls.append(cls)

    # drumurile mari: puncte la ~25 m de-a lungul fiecărei linii
    step_deg = ROAD_STEP_M / 111_000
    for line in road_lines:
        n = max(int(line.length / step_deg), 1)
        for p in shapely.line_interpolate_point(line, np.linspace(0, line.length, n + 1)):
            points.append(("major_road", p.x, p.y))
    pts = pd.DataFrame(points, columns=["kind", "lon", "lat"])

    # grila de landuse
    b = C.BBOX
    lat_step = GRID_STEP_M / 111_000
    lon_step = GRID_STEP_M / (111_000 * np.cos(np.deg2rad((b["lat_min"] + b["lat_max"]) / 2)))
    glon, glat = np.meshgrid(np.arange(b["lon_min"], b["lon_max"], lon_step),
                             np.arange(b["lat_min"], b["lat_max"], lat_step))
    grid = pd.DataFrame({"lon": glon.ravel(), "lat": glat.ravel(), "landuse": "none"})
    if landuse_polys:
        tree = shapely.STRtree(landuse_polys)
        gp, pi = tree.query(shapely.points(grid["lon"], grid["lat"]), predicate="within")
        first = pd.Series(pi, index=gp).groupby(level=0).first()  # la suprapuneri: primul poligon
        grid.loc[first.index, "landuse"] = np.array(landuse_cls)[first.to_numpy()]

    C.OSM_DIR.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(POINTS_FILE, index=False)
    grid.to_parquet(GRID_FILE, index=False)
    print("[OSM] puncte:", pts["kind"].value_counts().to_dict())
    print("[OSM] grila landuse:", grid["landuse"].value_counts().to_dict())
    return pts, grid


def osm_features(query: pd.DataFrame) -> pd.DataFrame:
    if not (POINTS_FILE.exists() and GRID_FILE.exists()):
        extract_osm()
    pts = pd.read_parquet(POINTS_FILE)
    grid = pd.read_parquet(GRID_FILE)
    q = _rad(query["latitude"], query["longitude"])
    out = {}

    gtree = BallTree(_rad(grid["lat"], grid["lon"]), metric="haversine")
    ind = gtree.query_radius(q, r=500 / C.EARTH_RADIUS_M)
    lu = grid["landuse"].to_numpy()
    for cls in LANDUSE_CLASSES:
        out[f"landuse_{cls}_500m"] = [float(np.mean(lu[i] == cls)) if len(i) else 0.0 for i in ind]

    def tree_for(kind):
        p = pts[pts["kind"] == kind]
        return BallTree(_rad(p["lat"], p["lon"]), metric="haversine") if len(p) else None

    def nearest_m(tree):
        return tree.query(q, k=1)[0][:, 0] * C.EARTH_RADIUS_M

    def count(tree, r):
        return tree.query_radius(q, r=r / C.EARTH_RADIUS_M, count_only=True)

    t = tree_for("major_road")
    out["dist_major_road_m"] = nearest_m(t)
    out["major_road_km_1km"] = count(t, 1000) * ROAD_STEP_M / 1000
    t = tree_for("parking")
    out["n_parking_500m"] = count(t, 500)
    out["dist_parking_m"] = nearest_m(t)
    out["dist_rail_station_m"] = nearest_m(tree_for("rail_station"))
    out["dist_bus_station_m"] = nearest_m(tree_for("bus_station"))
    out["n_buildings_500m"] = count(tree_for("building"), 500)
    return pd.DataFrame(out, index=query.index)


if __name__ == "__main__":
    extract_osm()
