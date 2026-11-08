r"""Importă județele (contur OSM admin_level 4), localitățile (admin_level 8) și populația oficială INS
(POP107D), apoi calculează raportul populație WorldPop / INS pe județ și ajustează grila
(populatie_ajustata). Se rulează DUPĂ
etl.import_populatie (care golește grila) și ÎNAINTE de etl.recalculeaza.

    python -m etl.import_judete D:\hack\romania-261105.osm.pbf "D:\hack\exportPivot_POP107D (1).csv"
"""
import argparse
import csv
import re
import unicodedata
from pathlib import Path

from shapely.geometry import MultiPolygon

from app.db import close_pool, get_connection, open_pool


def ascii_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"^Municipiul\s+", "", s.strip(), flags=re.I)
    return s.lower()


def read_ins(path: Path) -> tuple[dict, int | None]:
    """{judet_ascii: populație} pentru rândurile Total/Total/<județ>/TOTAL."""
    out, year = {}, None
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            row = [x.strip() for x in row]
            if len(row) < 7 or row[0] != "Total" or row[1] != "Total" or row[3] != "TOTAL" or row[2] == "TOTAL":
                continue
            out[ascii_name(row[2])] = int(float(row[6]))
            m = re.search(r"\d{4}", row[4])
            year = int(m.group()) if m else year
    return out, year


def main(pbf: Path, ins_csv: Path):
    from pyrosm import OSM  # dependență doar pentru ETL

    osm = OSM(str(pbf))
    b = osm.get_boundaries(boundary_type="administrative", custom_filter={"admin_level": ["4"]})
    b = b[b.geometry.notna()].to_crs("EPSG:4326")
    loc = osm.get_boundaries(boundary_type="administrative", custom_filter={"admin_level": ["8"]})
    loc = loc[loc.geometry.notna() & loc["name"].notna()].to_crs("EPSG:4326")
    ins, year = read_ins(ins_csv)
    open_pool()
    try:
        with get_connection() as conn, conn.transaction():
            conn.execute("UPDATE populatie_grid SET judet_id = NULL, populatie_ajustata = NULL")
            conn.execute("DELETE FROM judet")
            conn.execute("ALTER SEQUENCE judet_judet_id_seq RESTART")
            missing = []
            for name, geom in zip(b["name"], b.geometry):
                key = ascii_name(name)
                if key not in ins:
                    missing.append(name)
                g = geom if isinstance(geom, MultiPolygon) else MultiPolygon([geom])
                conn.execute("""
                    INSERT INTO judet (nume, nume_ascii, geom, populatie_ins, an_ins)
                    VALUES (%s, %s, ST_Multi(ST_MakeValid(ST_GeomFromText(%s, 4326))), %s, %s)
                """, [name, key, g.wkt, ins.get(key), year])
            conn.execute("""
                INSERT INTO judet_bucata (judet_id, geom)
                SELECT judet_id, ST_Subdivide(geom, 256) FROM judet
            """)
            # localitățile (admin_level 8), legate de județul care conține un punct interior al conturului
            conn.execute("ALTER SEQUENCE localitate_localitate_id_seq RESTART")
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO localitate (nume, nume_ascii, geom)
                    VALUES (%s, %s, ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_GeomFromText(%s, 4326)), 3)))
                """, [(name, ascii_name(name), (g if isinstance(g, MultiPolygon) else MultiPolygon([g])).wkt)
                      for name, g in zip(loc["name"], loc.geometry)])
            conn.execute("""
                UPDATE localitate l SET judet_id = jb.judet_id
                FROM judet_bucata jb
                WHERE ST_Intersects(jb.geom, ST_PointOnSurface(l.geom))
            """)
            conn.execute("ANALYZE localitate")
            n_loc = conn.execute("SELECT COUNT(*) AS n, COUNT(judet_id) AS cu_judet FROM localitate").fetchone()
            conn.execute("""
                UPDATE judet j SET populatie_worldpop = x.pop,
                       raport_worldpop_ins = x.pop / NULLIF(j.populatie_ins, 0)
                FROM (
                    SELECT j2.judet_id, SUM(g.populatie)::double precision AS pop
                    FROM judet j2 JOIN populatie_grid g ON ST_Intersects(j2.geom, g.locatie::geometry)
                    GROUP BY j2.judet_id
                ) x WHERE x.judet_id = j.judet_id
            """)
            # ajustarea grilei: suma pe județ = populația INS (în afara contururilor rămâne valoarea brută)
            conn.execute("""
                UPDATE populatie_grid g SET judet_id = j.judet_id,
                       populatie_ajustata = g.populatie * j.populatie_ins / NULLIF(j.populatie_worldpop, 0)
                FROM judet j WHERE ST_Intersects(j.geom, g.locatie::geometry)
            """)
            conn.execute("ANALYZE populatie_grid")
            tot = conn.execute("""
                SELECT SUM(populatie)::bigint AS brut, SUM(COALESCE(populatie_ajustata, populatie))::bigint AS ajustat,
                       COUNT(*) FILTER (WHERE judet_id IS NULL) AS fara_judet
                FROM populatie_grid""").fetchone()
            rows = conn.execute("""
                SELECT nume, populatie_ins, populatie_worldpop::bigint AS wp, round(raport_worldpop_ins::numeric, 2) AS r
                FROM judet ORDER BY raport_worldpop_ins DESC NULLS LAST
            """).fetchall()
        print(f"Județe importate: {len(rows)} (INS {year}); fără populație INS: {missing or 'niciunul'}")
        print(f"Localități importate: {n_loc['n']:,} ({n_loc['cu_judet']:,} cu județ)")
        print(f"Populație grilă: brută {tot['brut']:,}, ajustată {tot['ajustat']:,} "
              f"({tot['fara_judet']} puncte în afara contururilor, nemodificate)")
        for r in rows:
            print(f"  {r['nume']:18s} INS {r['populatie_ins'] or 0:>10,}  WorldPop {r['wp'] or 0:>10,}  raport {r['r']}")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pbf", type=Path)
    ap.add_argument("ins_csv", type=Path)
    a = ap.parse_args()
    main(a.pbf, a.ins_csv)
