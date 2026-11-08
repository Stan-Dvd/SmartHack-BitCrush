r"""Importă POI-urile OpenStreetMap în poi_osm, cu aceleași categorii ca feature-urile modelului.

Port exact al enrich_romania_features.py (scriptul care a calculat poi_*_500m pentru stațiile românești):
același filtru pyrosm (amenity/shop/office/leisure/tourism), aceeași funcție de categorie, geometria în
EPSG:3035. Categoria e calculată vectorizat, cu aceeași ordine de reguli (food > business_res > lifestyle).

    pip install -r requirements-etl.txt
    python -m etl.import_poi D:\hack\romania-261105.osm.pbf
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.db import close_pool, get_connection, open_pool

FOOD_AMENITY = {"restaurant", "cafe", "fast_food", "food_court", "ice_cream", "bakery"}
BUSINESS_AMENITY = {"bank", "marketplace", "post_office", "bureau_de_change"}
RESIDENTIAL_BUILDING = {"apartments", "residential", "house", "detached"}
LIFESTYLE_AMENITY = {
    "cinema", "theatre", "nightclub", "bar", "pub", "library",
    "arts_centre", "community_centre", "social_centre",
}


def _tag(df: pd.DataFrame, name: str) -> pd.Series:
    """Ca value() din scriptul original: None / NaN -> șir gol."""
    if name not in df.columns:
        return pd.Series("", index=df.index)
    s = df[name]
    return s.where(s.notna(), "").astype(str).replace({"nan": "", "None": ""})


def categorize(df: pd.DataFrame) -> pd.Series:
    amenity, shop, leisure = _tag(df, "amenity"), _tag(df, "shop"), _tag(df, "leisure")
    tourism, office, building = _tag(df, "tourism"), _tag(df, "office"), _tag(df, "building")
    food = amenity.isin(FOOD_AMENITY)
    business = (office != "") | (shop != "") | amenity.isin(BUSINESS_AMENITY) | building.isin(RESIDENTIAL_BUILDING)
    lifestyle = (leisure != "") | (tourism != "") | amenity.isin(LIFESTYLE_AMENITY)
    return pd.Series(np.select([food, business, lifestyle], ["food", "business_res", "lifestyle"], default=None),
                     index=df.index)


def main(pbf: Path):
    from pyrosm import OSM  # dependență doar pentru ETL

    t0 = time.time()
    print(f"Citesc POI-urile din {pbf} (durează câteva minute)...")
    pois = OSM(str(pbf)).get_pois(custom_filter={
        "amenity": True, "shop": True, "office": True, "leisure": True, "tourism": True,
    })
    print(f"  {len(pois):,} obiecte OSM în {time.time() - t0:.0f} s")
    pois = pois.to_crs("EPSG:3035")
    pois["categorie"] = categorize(pois)
    pois = pois[pois["categorie"].notna() & pois.geometry.notna() & ~pois.geometry.is_empty]
    print(f"  {len(pois):,} POI cu categorie: {pois['categorie'].value_counts().to_dict()}")

    ids = pois["id"] if "id" in pois.columns else pd.Series([None] * len(pois), index=pois.index)
    types = pois["osm_type"] if "osm_type" in pois.columns else pd.Series([None] * len(pois), index=pois.index)
    open_pool()
    try:
        with get_connection() as conn, conn.transaction():
            conn.execute("TRUNCATE poi_osm RESTART IDENTITY")
            with conn.cursor() as cur, cur.copy("COPY poi_osm (osm_id, osm_type, categorie, geom) FROM STDIN") as cp:
                for oid, otype, cat, geom in zip(ids, types, pois["categorie"], pois.geometry):
                    cp.write_row((None if pd.isna(oid) else int(oid), None if pd.isna(otype) else str(otype),
                                  cat, f"SRID=3035;{geom.wkt}"))
            conn.execute("ANALYZE poi_osm")
        print(f"POI importate: {len(pois):,} în {time.time() - t0:.0f} s")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pbf", type=Path)
    main(ap.parse_args().pbf)
