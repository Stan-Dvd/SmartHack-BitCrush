r"""Importă grila WorldPop 2020 la 1 km (ppp_ROU_2020_1km_Aggregated_UNadj, coloane X,Y,Z) în populatie_grid.

    python -m etl.import_populatie D:\hack\romania_worldpop.zip
"""
import argparse
import csv
import io
import zipfile
from pathlib import Path

from app.db import close_pool, get_connection, open_pool


def _rows(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
            with archive.open(name) as handle:
                yield from csv.DictReader(io.TextIOWrapper(handle, encoding="ascii"))
    else:
        with open(path, encoding="ascii", newline="") as handle:
            yield from csv.DictReader(handle)


def main(path: Path):
    open_pool()
    n = 0
    try:
        with get_connection() as conn, conn.transaction():
            conn.execute("TRUNCATE populatie_grid RESTART IDENTITY")
            with conn.cursor() as cur, cur.copy("COPY populatie_grid (locatie, populatie) FROM STDIN") as cp:
                for row in _rows(path):
                    cp.write_row((f"SRID=4326;POINT({float(row['X'])} {float(row['Y'])})", float(row["Z"])))
                    n += 1
            conn.execute("ANALYZE populatie_grid")
        print(f"Puncte de populație importate: {n:,} (din {path})")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    main(ap.parse_args().path)
