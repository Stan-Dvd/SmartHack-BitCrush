"""Calculul feature-urilor (în PostGIS) și al predicțiilor (LightGBM), plus salvarea lor în baza de date."""
from __future__ import annotations

from typing import Iterable, Sequence

import h3
import numpy as np
import pandas as pd

from .config import settings
from .h3_utils import station_cells
from .ml_service import LoadedModel, active_model, predict_profiles, summarize
from .repository import ensure_h3_cell

LOC_FEATURES = [
    "dist_nearest_station_m", "n_stations_500m", "n_stations_1km", "n_stations_3km",
    "competitor_piles_1km", "pop_1km", "poi_food_500m", "poi_business_res_500m", "poi_lifestyle_500m",
]
SUBJECT_COL = {"statie": "statie_id", "candidat": "candidat_id", "h3": "h3_index"}
SUMMARY_COLS = ["ore_incarcare_zi", "kwh_zi", "ocupare_medie", "ocupare_varf", "ora_varf",
                "weekend_varf", "saturata_prezis"]


def _point_wkt(lon: float, lat: float) -> str:
    return f"SRID=4326;POINT({lon} {lat})"


# ----------------------------------------------------------------------------- feature-uri
def compute_features(conn, lons: Sequence[float], lats: Sequence[float],
                     exclude_ids: Sequence[int | None] | None = None,
                     extra: Sequence[tuple[float, float, int]] | None = None) -> pd.DataFrame:
    """Feature-urile de locație pentru o listă de puncte, calculate cu fn_features_punct.

    extra = [(lon, lat, n_piles), ...]: stații ipotetice adăugate la concurența tuturor punctelor.
    """
    n = len(lons)
    if n == 0:
        return pd.DataFrame(columns=[*LOC_FEATURES, "pop_max_celula", "judet", "raport_pop_judet", "pop_suspect"])
    exclude_ids = list(exclude_ids) if exclude_ids is not None else [None] * n
    extra = list(extra or [])
    extra_geo = [_point_wkt(e[0], e[1]) for e in extra] or None
    extra_piles = [int(e[2]) for e in extra] or None
    rows = conn.execute("""
        SELECT t.i, f.*
        FROM unnest(%s::float8[], %s::float8[], %s::bigint[]) WITH ORDINALITY AS t(lon, lat, excl, i)
        CROSS JOIN LATERAL fn_features_punct(
            t.lon, t.lat, t.excl,
            -- trimis ca text[]: separatorul de elemente al tipului geography e „:”, nu „,”
            (SELECT array_agg(ST_GeogFromText(e.w) ORDER BY e.i) FROM unnest(%s::text[]) WITH ORDINALITY AS e(w, i)),
            %s::int[]) f
        ORDER BY t.i
    """, [list(map(float, lons)), list(map(float, lats)), exclude_ids, extra_geo, extra_piles]).fetchall()
    df = pd.DataFrame(rows).drop(columns="i")
    df["dist_nearest_station_m"] = df["dist_nearest_station_m"].astype(float)
    df["pop_suspect"] = df["raport_pop_judet"].astype(float).fillna(1.0) > settings.raport_pop_judet_suspect
    return df


def upsert_features(conn, tip: str, subjects: Sequence, feats: pd.DataFrame) -> dict:
    """Scrie feature-urile (inclusiv n_piles și power_kw) și întoarce {subiect: feature_id}."""
    col = SUBJECT_COL[tip]
    cols = [*LOC_FEATURES, "n_piles", "power_kw", "judet", "raport_pop_judet", "pop_suspect"]
    records = []
    for subj, (_, r) in zip(subjects, feats.iterrows()):
        vals = [None if (r[c] is None or (isinstance(r[c], float) and np.isnan(r[c]))) else r[c] for c in cols]
        records.append([tip, subj, *[v.item() if hasattr(v, "item") else v for v in vals]])
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
    with conn.cursor() as cur:
        cur.executemany(f"""
            INSERT INTO features_locatie (tip, {col}, {", ".join(cols)})
            VALUES (%s, %s, {", ".join(["%s"] * len(cols))})
            ON CONFLICT ({col}) DO UPDATE SET {set_clause}, calculat_la = NOW()
        """, records)
    rows = conn.execute(
        f"SELECT {col} AS s, feature_id FROM features_locatie WHERE {col} = ANY(%s)", [list(subjects)]
    ).fetchall()
    return {r["s"]: r["feature_id"] for r in rows}


# ----------------------------------------------------------------------------- predicții
def save_predictions(conn, model: LoadedModel, feature_ids: Sequence[int],
                     profile: pd.DataFrame | None, summary: pd.DataFrame):
    """Înlocuiește predicțiile modelului pentru feature_ids. profile=None => doar sumarul (celule H3)."""
    ids = [int(x) for x in feature_ids]
    conn.execute("DELETE FROM predictie_profil WHERE model_id = %s AND feature_id = ANY(%s)", [model.model_id, ids])
    conn.execute("DELETE FROM predictie_sumar WHERE model_id = %s AND feature_id = ANY(%s)", [model.model_id, ids])
    with conn.cursor() as cur:
        if profile is not None:
            fid = np.asarray(ids)[profile["_row"].to_numpy()]
            with cur.copy("COPY predictie_profil (feature_id, model_id, hour, is_weekend, busy_pred, "
                          "ocupare_pred, kwh_pred) FROM STDIN") as cp:
                for f, h, w, b, o, k in zip(fid, profile["hour"], profile["is_weekend"], profile["busy_pred"],
                                            profile["ocupare_pred"], profile["kwh_pred"]):
                    cp.write_row((int(f), model.model_id, int(h), bool(w), float(b), float(o), float(k)))
        with cur.copy(f"COPY predictie_sumar (feature_id, model_id, {', '.join(SUMMARY_COLS)}) FROM STDIN") as cp:
            for _, r in summary.iterrows():
                cp.write_row((ids[int(r["_row"])], model.model_id, float(r["ore_incarcare_zi"]), float(r["kwh_zi"]),
                              float(r["ocupare_medie"]), float(r["ocupare_varf"]), int(r["ora_varf"]),
                              bool(r["weekend_varf"]), bool(r["saturata_prezis"])))


def update_percentiles(conn, tip: str, model_id: int):
    conn.execute("""
        UPDATE predictie_sumar ps SET percentila_nationala = r.pr
        FROM (
            SELECT ps.feature_id, percent_rank() OVER (ORDER BY ps.ocupare_medie) AS pr
            FROM predictie_sumar ps JOIN features_locatie f USING (feature_id)
            WHERE f.tip = %s AND ps.model_id = %s
        ) r
        WHERE ps.feature_id = r.feature_id AND ps.model_id = %s
    """, [tip, model_id, model_id])


def _predict_and_save(conn, model, tip, subjects, feats, with_profile=True):
    fmap = upsert_features(conn, tip, subjects, feats)
    profile = predict_profiles(model, feats[model.static_features])
    summary = summarize(profile)
    save_predictions(conn, model, [fmap[s] for s in subjects], profile if with_profile else None, summary)
    return summary


# ----------------------------------------------------------------------------- stații existente
def recalc_statii(conn, statie_ids: Iterable[int] | None = None, percentile: bool = True) -> int:
    """Recalculează feature-urile și predicțiile pentru stațiile date (implicit: toate cele active)."""
    model = active_model(conn)
    params, where = [], "WHERE activa"
    if statie_ids is not None:
        statie_ids = [int(x) for x in statie_ids]
        if not statie_ids:
            return 0
        where += " AND statie_id = ANY(%s)"
        params.append(statie_ids)
    st = pd.DataFrame(conn.execute(
        f"SELECT statie_id, longitude, latitude, n_piles, power_kw FROM statie_incarcare {where} ORDER BY statie_id",
        params).fetchall())
    if st.empty:
        return 0
    feats = compute_features(conn, st["longitude"], st["latitude"], st["statie_id"].tolist())
    feats["n_piles"] = st["n_piles"].to_numpy()
    feats["power_kw"] = st["power_kw"].to_numpy()
    _predict_and_save(conn, model, "statie", st["statie_id"].tolist(), feats)
    if percentile:
        update_percentiles(conn, "statie", model.model_id)
    return len(st)


def statii_in_raza(conn, lon: float, lat: float, raza_m: float = settings.raza_impact_m,
                   exclude_id: int | None = None) -> list[int]:
    rows = conn.execute("""
        SELECT statie_id FROM statie_incarcare
        WHERE activa AND statie_id IS DISTINCT FROM %s
          AND ST_DWithin(locatie, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s, false)
    """, [exclude_id, lon, lat, raza_m]).fetchall()
    return [r["statie_id"] for r in rows]


# ----------------------------------------------------------------------------- grila H3 de oportunități
def h3_grid_cells(conn, res: int = settings.h3_grid_res, min_pop: float = settings.h3_grid_min_pop) -> list[str]:
    """Celulele H3 care conțin cel puțin un punct WorldPop cu populație >= min_pop."""
    rows = conn.execute("""
        SELECT ST_X(locatie::geometry) AS lon, ST_Y(locatie::geometry) AS lat
        FROM populatie_grid WHERE COALESCE(populatie_ajustata, populatie) >= %s
    """, [min_pop]).fetchall()
    return sorted({h3.latlng_to_cell(r["lat"], r["lon"], res) for r in rows})


def recalc_h3(conn, cells: Sequence[str], n_piles: int = settings.predictie_n_piles_default,
              power_kw: float = settings.predictie_power_kw_default, chunk: int = 2000,
              progress=None) -> int:
    """Evaluează o stație de referință (n_piles, power_kw) în centrul fiecărei celule. Stochează doar sumarul."""
    model = active_model(conn)
    done = 0
    for i in range(0, len(cells), chunk):
        part = list(cells[i:i + chunk])
        for c in part:
            ensure_h3_cell(conn, c, h3.get_resolution(c))
        centers = [h3.cell_to_latlng(c) for c in part]
        feats = compute_features(conn, [lon for _, lon in centers], [lat for lat, _ in centers])
        feats["n_piles"] = n_piles
        feats["power_kw"] = power_kw
        _predict_and_save(conn, model, "h3", part, feats, with_profile=False)
        done += len(part)
        if progress:
            progress(done, len(cells))
    update_percentiles(conn, "h3", model.model_id)
    return done


# ----------------------------------------------------------------------------- locații noi
def _reason(f: dict) -> str:
    parts = [f"populație ~{f['pop_1km']:,.0f} pe 1 km",
             f"{f['poi_food_500m'] + f['poi_business_res_500m'] + f['poi_lifestyle_500m']} POI pe 500 m",
             f"{f['n_stations_3km']} stații pe 3 km"]
    if f.get("dist_nearest_station_m") is not None:
        parts.append(f"cea mai apropiată la {f['dist_nearest_station_m'] / 1000:.1f} km")
    return "; ".join(parts)


def evalueaza_punct(conn, lon: float, lat: float, n_piles: int | None = None, power_kw: float | None = None,
                    cu_impact: bool = True) -> dict:
    """Performanța prezisă a unei stații noi în (lon, lat) și impactul ei asupra stațiilor existente pe 3 km."""
    model = active_model(conn)
    n_piles = int(n_piles or settings.predictie_n_piles_default)
    power_kw = float(power_kw or settings.predictie_power_kw_default)
    feats = compute_features(conn, [lon], [lat])
    feats["n_piles"] = n_piles
    feats["power_kw"] = power_kw
    profile = predict_profiles(model, feats[model.static_features])
    summary = summarize(profile).iloc[0]
    f = feats.iloc[0].to_dict()
    f = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v))
         for k, v in f.items()}

    ref = conn.execute("""
        SELECT
          (SELECT AVG((ps.ocupare_medie <= %(occ)s)::int) FROM predictie_sumar ps
             JOIN features_locatie fl USING (feature_id)
             WHERE fl.tip = 'statie' AND ps.model_id = %(m)s) AS pct_statii,
          (SELECT AVG((ps.ore_incarcare_zi <= %(ore)s)::int) FROM predictie_sumar ps
             JOIN features_locatie fl USING (feature_id)
             WHERE fl.tip = 'h3' AND ps.model_id = %(m)s) AS pct_h3
    """, {"occ": float(summary["ocupare_medie"]), "ore": float(summary["ore_incarcare_zi"]),
          "m": model.model_id}).fetchone()
    reference_config = (n_piles == settings.predictie_n_piles_default
                        and power_kw == settings.predictie_power_kw_default)

    result = {
        "lon": lon, "lat": lat, "n_piles": n_piles, "power_kw": power_kw, "model_id": model.model_id,
        "features": f,
        "sumar": {c: (summary[c].item() if hasattr(summary[c], "item") else summary[c]) for c in SUMMARY_COLS},
        "profil": [
            {"hour": int(r.hour), "is_weekend": bool(r.is_weekend), "busy_pred": float(r.busy_pred),
             "ocupare_pred": float(r.ocupare_pred), "kwh_pred": float(r.kwh_pred)}
            for r in profile.itertuples()
        ],
        "percentila_ocupare_vs_statii": float(ref["pct_statii"]) if ref["pct_statii"] is not None else None,
        # comparabil doar la configurația de referință a grilei (aceleași n_piles și power_kw)
        "percentila_vs_grila_h3": (float(ref["pct_h3"]) if ref["pct_h3"] is not None and reference_config
                                   else None),
        "motiv": _reason(f),
    }
    if cu_impact:
        result["impact_vecini"] = impact_vecini(conn, model, lon, lat, n_piles)
    return result


def impact_vecini(conn, model: LoadedModel, lon: float, lat: float, n_piles: int) -> dict:
    """Ore de încărcare/zi prezise pentru stațiile pe 3 km, fără și cu noua stație (canibalizare)."""
    ids = statii_in_raza(conn, lon, lat)
    if not ids:
        return {"statii": [], "total_ore_inainte": 0.0, "total_ore_dupa": 0.0, "pierdere_ore_zi": 0.0}
    st = pd.DataFrame(conn.execute("""
        SELECT statie_id, cod, longitude, latitude, n_piles, power_kw,
               ST_Distance(locatie, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, false) AS dist_m
        FROM statie_incarcare WHERE statie_id = ANY(%s) ORDER BY statie_id
    """, [lon, lat, ids]).fetchall())
    out = {}
    for label, extra in (("inainte", None), ("dupa", [(lon, lat, n_piles)])):
        feats = compute_features(conn, st["longitude"], st["latitude"], st["statie_id"].tolist(), extra=extra)
        feats["n_piles"] = st["n_piles"].to_numpy()
        feats["power_kw"] = st["power_kw"].to_numpy()
        out[label] = summarize(predict_profiles(model, feats[model.static_features]))["ore_incarcare_zi"].to_numpy()
    statii = [
        {"statie_id": int(r.statie_id), "cod": r.cod, "dist_m": float(r.dist_m),
         "lat": float(r.latitude), "lon": float(r.longitude),
         "ore_inainte": float(b), "ore_dupa": float(a), "delta_ore": float(a - b)}
        for r, b, a in zip(st.itertuples(), out["inainte"], out["dupa"])
    ]
    statii.sort(key=lambda x: x["delta_ore"])
    total_b, total_a = float(out["inainte"].sum()), float(out["dupa"].sum())
    return {"statii": statii, "total_ore_inainte": total_b, "total_ore_dupa": total_a,
            "pierdere_ore_zi": total_b - total_a}


def salveaza_candidat(conn, lon: float, lat: float, n_piles: int, power_kw: float,
                      nume: str | None = None, judet: str | None = None) -> int:
    """Persistă un candidat, cu feature-urile, profilul și sumarul lui."""
    model = active_model(conn)
    h3_8, h3_9 = station_cells(lat, lon)
    ensure_h3_cell(conn, h3_8, settings.h3_res8)
    ensure_h3_cell(conn, h3_9, settings.h3_res9)
    cid = conn.execute("""
        INSERT INTO locatie_candidat (nume, locatie, judet, n_piles, power_kw, h3_res8, h3_res9)
        VALUES (%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s, %s, %s, %s, %s)
        RETURNING candidat_id
    """, [nume, lon, lat, judet, n_piles, power_kw, h3_8, h3_9]).fetchone()["candidat_id"]
    feats = compute_features(conn, [lon], [lat])
    feats["n_piles"] = n_piles
    feats["power_kw"] = power_kw
    _predict_and_save(conn, model, "candidat", [cid], feats)
    return cid
