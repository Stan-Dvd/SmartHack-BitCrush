"""Importă stațiile românești în statie_incarcare.

ID-urile și regulile sunt identice cu load_romania() din scriptul care a generat romania_ev_5min.csv:
  statie_id = indexul rândului (1-based), cod = RO{index:04d},
  n_piles = max(1, round(capacity)) (1 dacă lipsește),
  power_kw = media(power_min_kw, power_max_kw) (22 kW dacă lipsesc ambele).
Diferență intenționată: puterile 0 sunt tratate ca lipsă (ca în pipeline-ul Shenzhen, docs/data_notes.md),
nu ca 0 kW.
Stațiile fără operator în sursă primesc OPERATOR_IMPLICIT (decizie de proiect, nu dată din sursă) și
operator_implicit = true, ca atribuirea să rămână vizibilă.

    python -m etl.import_statii [data/romania/statii_electrice_ML_cu_localitate.csv]
"""
import argparse
import csv

from app.config import ROOT
from app.db import close_pool, get_connection, open_pool
from app.h3_utils import station_cells
from app.repository import ensure_h3_cell

DEFAULT_CSV = ROOT / "data" / "romania" / "statii_electrice_ML_cu_localitate.csv"
OPERATOR_IMPLICIT = "PPC"  # operatorul atribuit stațiilor care nu au operator în sursă


def _num(value):
    value = (value or "").strip()
    return float(value) if value else None


def read_stations(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            capacity = _num(row.get("capacity"))
            pmin, pmax = _num(row.get("power_min_kw")), _num(row.get("power_max_kw"))
            powers = [p for p in (pmin, pmax) if p]  # None și 0 = lipsă
            operator = (row.get("operator") or "").strip() or None
            yield {
                "statie_id": index,
                "cod": f"RO{index:04d}",
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "judet": row.get("judet") or None,
                "localitate": row.get("localitate") or None,
                "operator": operator or OPERATOR_IMPLICIT,
                "operator_implicit": operator is None,
                "n_piles": max(1, round(capacity)) if capacity is not None else 1,
                "power_kw": sum(powers) / len(powers) if powers else 22.0,
                "power_min_kw": pmin,
                "power_max_kw": pmax,
                "is_open_24_7": (row.get("is_open_24_7") or "0").strip() in ("1", "1.0", "True", "true"),
            }


def main(path):
    stations = list(read_stations(path))
    open_pool()
    try:
        with get_connection() as conn, conn.transaction():
            seen = set()
            for st in stations:
                st["h3_8"], st["h3_9"] = station_cells(st["lat"], st["lon"])
                for cell, res in ((st["h3_8"], 8), (st["h3_9"], 9)):
                    if cell not in seen:
                        ensure_h3_cell(conn, cell, res)
                        seen.add(cell)
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO statie_incarcare
                        (statie_id, cod, locatie, judet, localitate, operator, operator_implicit, n_piles,
                         power_kw, power_min_kw, power_max_kw, is_open_24_7, h3_res8, h3_res9)
                    VALUES (%(statie_id)s, %(cod)s,
                            ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography,
                            %(judet)s, %(localitate)s, %(operator)s, %(operator_implicit)s, %(n_piles)s,
                            %(power_kw)s, %(power_min_kw)s, %(power_max_kw)s, %(is_open_24_7)s,
                            %(h3_8)s, %(h3_9)s)
                    ON CONFLICT (statie_id) DO UPDATE SET
                        cod = EXCLUDED.cod, locatie = EXCLUDED.locatie, judet = EXCLUDED.judet,
                        localitate = EXCLUDED.localitate, operator = EXCLUDED.operator,
                        operator_implicit = EXCLUDED.operator_implicit,
                        n_piles = EXCLUDED.n_piles, power_kw = EXCLUDED.power_kw,
                        power_min_kw = EXCLUDED.power_min_kw, power_max_kw = EXCLUDED.power_max_kw,
                        is_open_24_7 = EXCLUDED.is_open_24_7,
                        h3_res8 = EXCLUDED.h3_res8, h3_res9 = EXCLUDED.h3_res9
                """, stations)
        n_impl = sum(st["operator_implicit"] for st in stations)
        print(f"Stații importate: {len(stations)} (din {path}); {n_impl} fără operator -> {OPERATOR_IMPLICIT}")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", nargs="?", default=str(DEFAULT_CSV))
    main(ap.parse_args().csv_path)
