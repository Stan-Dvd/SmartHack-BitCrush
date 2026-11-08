"""Înregistrează modelul LightGBM din ml/models în model_ml și îl marchează activ.

    python -m etl.register_model
    python -m etl.register_model --model ml/models/lgbm.txt --params ml/models/params.json --nume lgbm_cerere
"""
import argparse
import hashlib
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from app.config import ROOT, settings
from app.db import close_pool, get_connection, open_pool
from app.ml_service import load_from_files

LIMITARI = (
    "Antrenat pe UrbanEV Shenzhen (1.406 stații, sep. 2022 - feb. 2023) și aplicat pe România fără reantrenare: "
    "nivelurile absolute sunt orientative, clasarea locațiilor e mai robustă. Ținta e cererea servită "
    "(puncte ocupate în medie per oră și tip de zi), nu cererea totală. Efectul lui n_piles și power_kw nu e "
    "cauzal (selection bias): locațiile se compară la aceeași configurație. Modelul nu folosește istoricul de "
    "ocupare; pentru stațiile existente, prognoza se poate calibra cu observațiile reale (fn_prognoza_statie). "
    "Setul românesc romania_ev_5min*.csv are busy sintetic și nu e folosit."
)


def main(model_path: Path, params_path: Path, metrics_path: Path, nume: str):
    loaded = load_from_files(model_path, params_path)  # validează formatul și ordinea feature-urilor
    params = json.loads(params_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    raw = model_path.read_bytes().replace(b"\r\n", b"\n")
    versiune = hashlib.sha256(raw).hexdigest()[:12]
    rel = model_path.resolve().relative_to(ROOT).as_posix()
    open_pool()
    try:
        with get_connection() as conn, conn.transaction():
            conn.execute("UPDATE model_ml SET activ = FALSE WHERE activ")
            row = conn.execute("""
                INSERT INTO model_ml (nume, versiune, cale_fisier, features, target_mode, params, metrici,
                                      antrenat_pe, limitari, activ)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (nume, versiune) DO UPDATE SET
                    cale_fisier = EXCLUDED.cale_fisier, features = EXCLUDED.features,
                    target_mode = EXCLUDED.target_mode, params = EXCLUDED.params,
                    metrici = EXCLUDED.metrici, limitari = EXCLUDED.limitari, activ = TRUE
                RETURNING model_id
            """, [nume, versiune, rel, Jsonb(loaded.features), loaded.target_mode, Jsonb(params),
                  Jsonb(metrics) if metrics else None,
                  "UrbanEV Shenzhen, 2022-09-01 - 2023-02-28 (CC0); WorldPop 2020; POI UrbanEV", LIMITARI]).fetchone()
        print(f"Model activ: model_id={row['model_id']} {nume} v{versiune} ({rel}), "
              f"{loaded.booster.num_trees()} arbori, target_mode={loaded.target_mode}")
    finally:
        close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=settings.ml_model_dir / "lgbm.txt")
    ap.add_argument("--params", type=Path, default=settings.ml_model_dir / "params.json")
    ap.add_argument("--metrics", type=Path, default=settings.ml_reports_dir / "metrics.json")
    ap.add_argument("--nume", default="lgbm_cerere_shenzhen")
    a = ap.parse_args()
    main(a.model, a.params, a.metrics, a.nume)
