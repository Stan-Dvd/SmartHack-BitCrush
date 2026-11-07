"""Importă inf.csv; coloane așteptate: station_id, longitude, latitude,
charge_count, TAZID, area, perimeter."""
import argparse
import csv

from app.db import open_pool, close_pool, get_connection
from app.h3_utils import station_cells
from app.repository import ensure_h3_cell

def main(path):
    open_pool()
    try:
        with open(path, newline="", encoding="utf-8-sig") as f, get_connection() as conn:
            for row in csv.DictReader(f):
                station_id = int(row["station_id"])
                lon, lat = float(row["longitude"]), float(row["latitude"])
                h3_8, h3_9 = station_cells(lat, lon)
                with conn.transaction():
                    ensure_h3_cell(conn, h3_8, 8)
                    ensure_h3_cell(conn, h3_9, 9)
                    conn.execute("""
                        INSERT INTO statie_incarcare
                        (statie_id, locatie, h3_res8, h3_res9, charge_count,
                         taz_id, area, perimeter)
                        VALUES (
                            %s, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,
                            %s,%s,%s,%s,%s,%s
                        )
                        ON CONFLICT (statie_id) DO UPDATE SET
                            locatie=EXCLUDED.locatie,
                            h3_res8=EXCLUDED.h3_res8,
                            h3_res9=EXCLUDED.h3_res9,
                            charge_count=EXCLUDED.charge_count,
                            taz_id=EXCLUDED.taz_id,
                            area=EXCLUDED.area,
                            perimeter=EXCLUDED.perimeter
                    """, [
                        station_id, lon, lat, h3_8, h3_9,
                        int(float(row.get("charge_count") or 1)),
                        int(float(row["TAZID"])) if row.get("TAZID") else None,
                        float(row["area"]) if row.get("area") else None,
                        float(row["perimeter"]) if row.get("perimeter") else None,
                    ])
        print("Import stații finalizat.")
    finally:
        close_pool()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    main(parser.parse_args().csv_path)
