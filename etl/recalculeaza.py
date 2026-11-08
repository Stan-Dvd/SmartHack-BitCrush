"""Recalculează feature-urile și predicțiile modelului activ.

    python -m etl.recalculeaza --statii          # toate stațiile active (profil de 48 de ore + sumar)
    python -m etl.recalculeaza --h3              # grila H3 de oportunități (stație de referință, doar sumar)
    python -m etl.recalculeaza --statii --h3 --min-pop 200
"""
import argparse
import time

from app.config import settings
from app.db import close_pool, get_connection, open_pool
from app.predictii import h3_grid_cells, recalc_h3, recalc_statii


def main(statii: bool, grila: bool, res: int, min_pop: float):
    open_pool()
    try:
        if statii:
            t0 = time.time()
            with get_connection() as conn, conn.transaction():
                n = recalc_statii(conn)
            print(f"Stații: {n} recalculate în {time.time() - t0:.1f} s")
        if grila:
            t0 = time.time()
            with get_connection() as conn, conn.transaction():
                conn.execute("DELETE FROM features_locatie WHERE tip = 'h3'")
                cells = h3_grid_cells(conn, res, min_pop)
                print(f"Grila H3 res {res}: {len(cells):,} celule cu populație >= {min_pop:g}")
                recalc_h3(conn, cells, progress=lambda d, t: print(f"  {d:,}/{t:,} ({time.time() - t0:.0f} s)"))
            print(f"Grila H3: {len(cells):,} celule în {time.time() - t0:.1f} s")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--statii", action="store_true")
    ap.add_argument("--h3", action="store_true")
    ap.add_argument("--res", type=int, default=settings.h3_grid_res)
    ap.add_argument("--min-pop", type=float, default=settings.h3_grid_min_pop)
    a = ap.parse_args()
    if not (a.statii or a.h3):
        ap.error("alegeți --statii și/sau --h3")
    main(a.statii, a.h3, a.res, a.min_pop)
