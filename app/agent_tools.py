"""Tool-urile deterministe ale agentului AI. Toate cifrele din rapoartele agentului vin de aici.

Fiecare tool e o funcție `tool(conn, **params) -> dict`, înregistrată în TOOLS cu descrierea și schema JSON a
parametrilor (folosite și de Gemini, și de endpoint-ul POST /api/tools/{nume}). Cheile care încep cu „_” sunt
doar pentru hartă/UI și nu se trimit modelului de limbaj.
"""
from __future__ import annotations

import difflib
import inspect
import json
import math
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

import h3
import numpy as np
import pandas as pd

from .config import settings
from .ml_service import active_model, expand_hours, predict_profiles, summarize
from .predictii import LOC_FEATURES, compute_features, evalueaza_punct

EARTH_R = 6_371_008.8
RAZA_CONCURENTA_M = 3000.0

FEATURE_LABELS = {
    "dist_nearest_station_m": "distanța până la cea mai apropiată stație",
    "n_stations_500m": "stații concurente pe 500 m",
    "n_stations_1km": "stații concurente pe 1 km",
    "n_stations_3km": "stații concurente pe 3 km",
    "competitor_piles_1km": "puncte de încărcare concurente pe 1 km",
    "pop_1km": "populația pe 1 km",
    "poi_food_500m": "POI alimentație pe 500 m",
    "poi_business_res_500m": "POI business/rezidențial pe 500 m",
    "poi_lifestyle_500m": "POI servicii/lifestyle pe 500 m",
    "n_piles": "numărul de puncte",
    "power_kw": "puterea pe punct",
    "hour": "ora din zi",
    "is_weekend": "tipul zilei",
}


class ToolError(ValueError):
    """Eroare de utilizare a unui tool (parametri greșiți, zonă inexistentă); mesajul ajunge la agent."""


# ============================================================================= utilitare
def _r(x, nd=2):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def _config(n_piles, power_kw):
    n = int(n_piles or settings.predictie_n_piles_default)
    p = float(power_kw or settings.predictie_power_kw_default)
    if n <= 0 or p <= 0:
        raise ToolError("n_piles și power_kw trebuie să fie pozitive.")
    return n, p


def _is_reference(n, p):
    return n == settings.predictie_n_piles_default and p == float(settings.predictie_power_kw_default)


PREFIXE = r"^(judetul|judet|jud|municipiul|municipiu|mun|orasul|oras|comuna|com|sectorul)\s+"


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(PREFIXE, "", s)
    return re.sub(r"[^a-z0-9]", "", s)


# ============================================================================= zone
@dataclass
class Zona:
    tip: str          # 'judet' | 'localitate'
    id: int
    nume: str
    judet: str

    def filtru(self, geom_expr: str) -> tuple[str, list]:
        """Condiție SQL: geometria (EPSG:4326, tip geometry) se află în zonă."""
        if self.tip == "judet":
            return (f"EXISTS (SELECT 1 FROM judet_bucata jb WHERE jb.judet_id = %s "
                    f"AND ST_Intersects(jb.geom, {geom_expr}))", [self.id])
        return (f"EXISTS (SELECT 1 FROM localitate lz WHERE lz.localitate_id = %s "
                f"AND ST_Intersects(lz.geom, {geom_expr}))", [self.id])

    def to_dict(self):
        return {"tip": self.tip, "id": self.id, "nume": self.nume, "judet": self.judet}


_index_lock = threading.Lock()
_index: dict | None = None


def _name_index(conn):
    global _index
    with _index_lock:
        if _index is None:
            idx: dict[str, list[dict]] = {}
            for r in conn.execute("SELECT judet_id AS id, nume, nume AS judet FROM judet").fetchall():
                idx.setdefault(normalize_name(r["nume"]), []).append({"tip": "judet", **r})
            for r in conn.execute("""
                SELECT l.localitate_id AS id, l.nume, j.nume AS judet
                FROM localitate l LEFT JOIN judet j USING (judet_id)
            """).fetchall():
                idx.setdefault(normalize_name(r["nume"]), []).append({"tip": "localitate", **r})
            _index = idx
        return _index


def _find_zone(conn, nume: str, judet: str | None = None, tip: str = "auto") -> tuple[Zona | None, dict]:
    """Întoarce (zona, info). info conține variantele dacă numele e ambiguu sau sugestii dacă nu există."""
    if not nume or not str(nume).strip():
        raise ToolError("Numele zonei lipsește.")
    idx = _name_index(conn)
    raw = str(nume).strip().lower()
    if tip == "auto":
        if re.match(r"^(jude|jud\b)", unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()):
            tip = "judet"
        elif re.match(r"^(municip|mun\b|oras|comun|sector)", raw):
            tip = "localitate"
    key = normalize_name(nume)
    hits = list(idx.get(key, []))
    if tip in ("judet", "localitate"):
        hits = [h for h in hits if h["tip"] == tip]
    if judet:
        jk = normalize_name(judet)
        hits = [h for h in hits if h["judet"] and normalize_name(h["judet"]) == jk] or hits
    if not hits:
        sugestii = difflib.get_close_matches(key, list(idx.keys()), n=5, cutoff=0.75)
        variante = [h for s in sugestii for h in idx[s]][:8]
        return None, {"gasit": False, "sugestii": [f"{v['nume']} ({v['tip']}, {v['judet']})" for v in variante]}
    judete = [h for h in hits if h["tip"] == "judet"]
    if judete:  # „Iași” = județul, dacă nu s-a cerut explicit localitatea
        h = judete[0]
        alt = [f"{x['nume']} (localitate, {x['judet']})" for x in hits if x["tip"] == "localitate"]
        return Zona("judet", h["id"], h["nume"], h["nume"]), {"alternative": alt}
    if len(hits) > 1:
        return None, {"gasit": False, "ambiguu": True,
                      "variante": [f"{h['nume']} (localitate, județul {h['judet']})" for h in hits],
                      "mesaj": "Numele corespunde mai multor localități; precizați parametrul judet."}
    h = hits[0]
    return Zona("localitate", h["id"], h["nume"], h["judet"] or ""), {}


def _zone(conn, zona: str, judet: str | None = None, tip: str = "auto") -> Zona:
    z, info = _find_zone(conn, zona, judet, tip)
    if z is None:
        raise ToolError(json.dumps({"mesaj": f"Zona „{zona}” nu a fost găsită.", **info}, ensure_ascii=False))
    return z


# ============================================================================= predicții pe grila H3
_grid_cache: dict[tuple, pd.DataFrame] = {}


def _grid(conn, zona: Zona | None, n_piles: int, power_kw: float, exclude_pop_suspect=True) -> pd.DataFrame:
    """Celulele grilei H3 (din zonă sau toate), cu predicția pentru configurația cerută."""
    filters, params = ["f.tip = 'h3'"], []
    if zona is not None:
        cond, p = zona.filtru("c.center::geometry")
        filters.append(cond)
        params += p
    if exclude_pop_suspect:
        filters.append("NOT f.pop_suspect")
    df = pd.DataFrame(conn.execute(f"""
        SELECT f.h3_index, ST_Y(c.center::geometry) AS lat, ST_X(c.center::geometry) AS lon, f.judet,
               {", ".join("f." + c for c in LOC_FEATURES)}, f.pop_suspect,
               ps.ore_incarcare_zi, ps.kwh_zi, ps.ocupare_medie, ps.ocupare_varf, ps.ora_varf
        FROM features_locatie f
        JOIN celule_h3 c ON c.h3_index = f.h3_index
        JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
        WHERE {" AND ".join(filters)}
    """, params).fetchall())
    if df.empty or _is_reference(n_piles, power_kw):
        return df
    model = active_model(conn)
    static = df[LOC_FEATURES].astype(float).copy()
    static["n_piles"], static["power_kw"] = n_piles, power_kw
    s = summarize(predict_profiles(model, static)).set_index("_row").sort_index()
    for c in ("ore_incarcare_zi", "kwh_zi", "ocupare_medie", "ocupare_varf", "ora_varf"):
        df[c] = s[c].to_numpy()
    return df


_baseline_cache: dict[tuple, float] = {}
SORT_COLS = ["ore_incarcare_zi", "pop_1km", "poi_total", "h3_index"]
SORT_ASC = [False, False, False, True]


def _baseline(conn, n_piles: int, power_kw: float) -> float:
    """Nivelul de bază al modelului = mediana națională a orelor/zi pe grila H3 (în ~93% din celulele populate
    modelul prezice exact această valoare: diferențiază doar zonele urbane dense)."""
    model = active_model(conn)
    key = (model.model_id, n_piles, power_kw)
    if key not in _baseline_cache:
        g = _grid(conn, None, n_piles, power_kw)
        _baseline_cache[key] = float(g["ore_incarcare_zi"].median())
    return _baseline_cache[key]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinea: orele prezise; la egalitate (nivelul de bază), populația și apoi numărul de POI."""
    df = df.copy()
    df["poi_total"] = df["poi_food_500m"] + df["poi_business_res_500m"] + df["poi_lifestyle_500m"]
    return df.sort_values(SORT_COLS, ascending=SORT_ASC)


def _cell_out(r, baseline: float | None = None) -> dict:
    return {
        "h3_index": r["h3_index"], "lat": _r(r["lat"], 5), "lon": _r(r["lon"], 5), "judet": r.get("judet"),
        "ore_incarcare_zi": _r(r["ore_incarcare_zi"]), "kwh_zi": _r(r["kwh_zi"], 1),
        "ocupare_medie": _r(r["ocupare_medie"], 3), "ocupare_varf": _r(r["ocupare_varf"], 3),
        "ora_varf": int(r["ora_varf"]), "pop_1km": _r(r["pop_1km"], 0),
        "poi_500m": int(r["poi_food_500m"] + r["poi_business_res_500m"] + r["poi_lifestyle_500m"]),
        "n_stations_3km": int(r["n_stations_3km"]),
        "dist_nearest_station_m": _r(r["dist_nearest_station_m"], 0),
        **({"peste_nivelul_de_baza": bool(r["ore_incarcare_zi"] > baseline + 0.01)} if baseline is not None else {}),
    }


NOTA_NIVEL_BAZA = (
    "Modelul (antrenat pe Shenzhen) prezice aceeași valoare, nivelul de bază, pentru zonele mai puțin dense "
    "decât cele mai puțin dense zone din datele de antrenare; diferențiază doar zonele urbane dense. Între "
    "celulele aflate la nivelul de bază, ordinea e dată de criteriul secundar: populația pe 1 km, apoi POI.")


# ============================================================================= 1. resolve_area
def resolve_area(conn, nume: str, judet: str | None = None, tip: str = "auto") -> dict:
    z, info = _find_zone(conn, nume, judet, tip)
    if z is None:
        return {"gasit": False, **info}
    table, key = ("judet", "judet_id") if z.tip == "judet" else ("localitate", "localitate_id")
    g = conn.execute(f"""
        SELECT ST_XMin(geom) AS x0, ST_YMin(geom) AS y0, ST_XMax(geom) AS x1, ST_YMax(geom) AS y1,
               ST_X(ST_PointOnSurface(geom)) AS cx, ST_Y(ST_PointOnSurface(geom)) AS cy,
               ST_Area(geom::geography) / 1e6 AS km2,
               ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, 0.002), 5)::json AS contur
        FROM {table} WHERE {key} = %s
    """, [z.id]).fetchone()
    cond_c, p_c = z.filtru("c.center::geometry")
    n_cel = conn.execute(f"""
        SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE f.pop_suspect) AS suspect
        FROM features_locatie f JOIN celule_h3 c ON c.h3_index = f.h3_index
        WHERE f.tip = 'h3' AND {cond_c}
    """, p_c).fetchone()
    cond_s, p_s = z.filtru("s.locatie::geometry")
    n_st = conn.execute(f"SELECT COUNT(*) AS n FROM statie_incarcare s WHERE s.activa AND {cond_s}", p_s).fetchone()
    pop = None
    if z.tip == "judet":
        pop = conn.execute("SELECT populatie_ins FROM judet WHERE judet_id = %s", [z.id]).fetchone()["populatie_ins"]
    return {
        "gasit": True, **z.to_dict(), "suprafata_km2": _r(g["km2"], 0), "populatie_ins": pop,
        "bbox": [_r(g["x0"], 4), _r(g["y0"], 4), _r(g["x1"], 4), _r(g["y1"], 4)],
        "centru": {"lat": _r(g["cy"], 5), "lon": _r(g["cx"], 5)},
        "n_celule_h3_evaluate": n_cel["n"], "n_celule_populatie_suspecta": n_cel["suspect"],
        "n_statii_existente": n_st["n"], **info,
        "_harta": {"contur": g["contur"], "centru": [g["cy"], g["cx"]], "bbox": [g["x0"], g["y0"], g["x1"], g["y1"]]},
    }


# ============================================================================= 2. get_existing_stations
def get_existing_stations(conn, zona: str, judet: str | None = None, tip: str = "auto", top: int = 15) -> dict:
    z = _zone(conn, zona, judet, tip)
    cond, p = z.filtru("ST_SetSRID(ST_MakePoint(v.longitude, v.latitude), 4326)")
    rows = conn.execute(f"""
        SELECT v.statie_id, v.cod, v.latitude AS lat, v.longitude AS lon, v.localitate, v.operator,
               v.operator_implicit, v.n_piles, v.power_kw, v.ore_incarcare_zi, v.ocupare_medie, v.kwh_zi
        FROM v_statie_predictie v WHERE v.activa AND {cond}
        ORDER BY v.ore_incarcare_zi DESC NULLS LAST
    """, p).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return {"zona": z.to_dict(), "n_statii": 0, "statii": [], "_harta": {"statii": []}}
    pe_operator = df.groupby("operator").agg(statii=("statie_id", "size"), puncte=("n_piles", "sum")) \
        .sort_values("statii", ascending=False).head(10)
    return {
        "zona": z.to_dict(),
        "n_statii": int(len(df)), "n_puncte": int(df["n_piles"].sum()),
        "putere_medie_kw": _r(df["power_kw"].mean(), 1),
        "ore_incarcare_zi_medie": _r(df["ore_incarcare_zi"].mean()),
        "ocupare_medie": _r(df["ocupare_medie"].mean(), 3),
        "statii_cu_operator_atribuit": int(df["operator_implicit"].sum()),
        "pe_operator": [{"operator": k, "statii": int(v.statii), "puncte": int(v.puncte)}
                        for k, v in pe_operator.iterrows()],
        "cele_mai_utilizate": [
            {"cod": r.cod, "localitate": r.localitate, "operator": r.operator, "n_piles": int(r.n_piles),
             "power_kw": _r(r.power_kw, 0), "ore_incarcare_zi": _r(r.ore_incarcare_zi),
             "ocupare_medie": _r(r.ocupare_medie, 3)}
            for r in df.head(int(top)).itertuples()
        ],
        "_harta": {"statii": [[float(r.lat), float(r.lon), r.cod] for r in df.itertuples()]},
    }


# ============================================================================= 3. score_candidates
def score_candidates(conn, zona: str, judet: str | None = None, tip: str = "auto", n_piles: int | None = None,
                     power_kw: float | None = None, min_dist_statie_m: float = 0, top_k: int = 10) -> dict:
    z = _zone(conn, zona, judet, tip)
    n, p = _config(n_piles, power_kw)
    df = _grid(conn, z, n, p)
    if df.empty:
        return {"zona": z.to_dict(), "n_celule": 0, "candidati": [],
                "mesaj": "Zona nu are celule populate evaluate (sau doar celule cu populație suspectă)."}
    total = len(df)
    base = _baseline(conn, n, p)
    df = df[df["dist_nearest_station_m"].fillna(1e12) >= float(min_dist_statie_m)]
    df = _sorted(df)
    top = [_cell_out(r, base) for _, r in df.head(int(top_k)).iterrows()]
    n_peste = int((df["ore_incarcare_zi"] > base + 0.01).sum())
    return {
        "zona": z.to_dict(), "configuratie": {"n_piles": n, "power_kw": p},
        "n_celule_evaluate": total, "n_celule_dupa_filtre": int(len(df)),
        "nivel_de_baza_ore_zi": _r(base), "n_celule_peste_nivelul_de_baza": n_peste,
        "nota": NOTA_NIVEL_BAZA,
        "ore_incarcare_zi_mediana_zona": _r(df["ore_incarcare_zi"].median()) if len(df) else None,
        "candidati": top,
        "_harta": {"candidati": [[c["lat"], c["lon"], c["h3_index"], c["ore_incarcare_zi"]] for c in top]},
    }


# ============================================================================= 4. explain_cell
def explain_cell(conn, h3_index: str | None = None, lat: float | None = None, lon: float | None = None,
                 n_piles: int | None = None, power_kw: float | None = None) -> dict:
    n, p = _config(n_piles, power_kw)
    if h3_index:
        if not h3.is_valid_cell(h3_index):
            raise ToolError(f"h3_index invalid: {h3_index}")
        lat, lon = h3.cell_to_latlng(h3_index)
    if lat is None or lon is None:
        raise ToolError("Dați h3_index sau lat și lon.")
    feats = compute_features(conn, [float(lon)], [float(lat)])
    feats["n_piles"], feats["power_kw"] = n, p
    model = active_model(conn)
    rows = expand_hours(feats[model.static_features].astype(float))
    X = rows[model.features].astype(float)
    contrib = model.booster.predict(X, pred_contrib=True)  # TreeSHAP; ultima coloană = valoarea de bază
    raw = contrib.sum(axis=1)
    mean_c = contrib[:, :-1].mean(axis=0)
    base = float(contrib[:, -1].mean())
    # obiectivul tweedie are link log: contribuțiile se adună în log, deci efectul e multiplicativ.
    # Verificat empiric: exp(Σ contribuții) trebuie să reproducă predicția.
    link_log = bool(np.allclose(np.exp(raw), model.booster.predict(X), rtol=1e-6, atol=1e-9))
    profile = predict_profiles(model, feats[model.static_features].astype(float))
    summary = summarize(profile).iloc[0]
    effects = []
    for f, c in zip(model.features, mean_c):
        if f in ("hour", "is_weekend"):
            continue
        effects.append({
            "feature": f, "descriere": FEATURE_LABELS.get(f, f),
            "valoare": _r(float(feats.iloc[0][f]), 1),
            "contributie_log": _r(c, 4),
            "efect_ocupare": (f"x{math.exp(c):.2f}" if link_log else f"{c:+.3f}"),
        })
    effects.sort(key=lambda e: -abs(e["contributie_log"]))
    f0 = feats.iloc[0]
    judet = f0.get("judet")
    comp = None
    if judet:
        avg = conn.execute(f"""
            SELECT COUNT(*) AS n, AVG(ps.ore_incarcare_zi) AS ore,
                   {", ".join(f"AVG(f.{c}) AS {c}" for c in LOC_FEATURES)}
            FROM features_locatie f
            JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
            WHERE f.tip = 'h3' AND f.judet = %s AND NOT f.pop_suspect
        """, [judet]).fetchone()
        if avg and avg["n"]:
            comp = {
                "judet": judet, "n_celule": avg["n"],
                "ore_incarcare_zi_medie_judet_statie_referinta": _r(avg["ore"]),
                "medii_feature": {c: _r(avg[c], 1) for c in LOC_FEATURES},
            }
    return {
        "punct": {"lat": _r(lat, 5), "lon": _r(lon, 5), "h3_index": h3_index or h3.latlng_to_cell(lat, lon, 8),
                  "judet": judet},
        "configuratie": {"n_piles": n, "power_kw": p},
        "predictie": {"ore_incarcare_zi": _r(summary["ore_incarcare_zi"]), "ocupare_medie": _r(summary["ocupare_medie"], 3),
                      "kwh_zi": _r(summary["kwh_zi"], 1)},
        "feature_uri": {c: _r(float(f0[c]), 1) for c in LOC_FEATURES},
        "pop_suspect": bool(f0.get("pop_suspect")),
        "top_3_contributii": effects[:3],
        "toate_contributiile": effects,
        "explicatie_metoda": ("Contribuții TreeSHAP (LightGBM pred_contrib) mediate pe cele 48 de ore x tip de zi; "
                              "efect_ocupare = factorul multiplicativ față de valoarea de bază a modelului."),
        "ocupare_de_baza_model": _r(math.exp(base), 3) if link_log else _r(base, 3),
        "verificare_reconstructie_max_dif": _r(float(np.abs(np.exp(raw) - model.booster.predict(X)).max()), 8)
        if link_log else None,
        "comparatie_judet": comp,
        "_harta": {"punct": [float(lat), float(lon)]},
    }


# ============================================================================= 5. plan_network
def _hours_for(conn, model, lons, lats, n_piles_list, power_list, exclude_ids=None, extra=None) -> np.ndarray:
    if len(lons) == 0:
        return np.array([])
    feats = compute_features(conn, lons, lats, exclude_ids, extra=extra)
    feats["n_piles"] = np.asarray(n_piles_list, dtype=float)
    feats["power_kw"] = np.asarray(power_list, dtype=float)
    s = summarize(predict_profiles(model, feats[model.static_features].astype(float)))
    return s.sort_values("_row")["ore_incarcare_zi"].to_numpy()


def plan_network(conn, zona: str, k: int = 3, judet: str | None = None, tip: str = "auto",
                 min_dist_m: float = 1000, n_piles: int | None = None, power_kw: float | None = None,
                 min_dist_statie_m: float = 0, pool: int = 150, shortlist: int = 8) -> dict:
    """Greedy: la fiecare pas alege celula cu cel mai mare câștig net (ore proprii − pierderea vecinilor),
    recalculând concurența cu stațiile deja alese."""
    z = _zone(conn, zona, judet, tip)
    n, p = _config(n_piles, power_kw)
    k = max(1, min(int(k), 10))
    model = active_model(conn)
    grid = _grid(conn, z, n, p)
    if grid.empty:
        raise ToolError("Zona nu are celule populate evaluate.")
    grid = grid[grid["dist_nearest_station_m"].fillna(1e12) >= float(min_dist_statie_m)]
    base = _baseline(conn, n, p)
    cand = _sorted(grid).head(int(pool)).reset_index(drop=True)
    own = dict(zip(cand["h3_index"], cand["ore_incarcare_zi"]))  # ore proprii, cu concurența curentă
    pos = {r.h3_index: (float(r.lon), float(r.lat)) for r in cand.itertuples()}
    tie = {r.h3_index: (float(r.pop_1km), float(r.poi_total)) for r in cand.itertuples()}  # departajare

    st = pd.DataFrame(conn.execute("""
        SELECT statie_id, cod, longitude AS lon, latitude AS lat, n_piles, power_kw FROM statie_incarcare WHERE activa
    """).fetchall())

    def stations_near(lon, lat):
        d = np.array([_haversine_m(lon, lat, a, b) for a, b in zip(st["lon"], st["lat"])])
        return st[d <= RAZA_CONCURENTA_M]

    chosen: list[dict] = []
    impact_total: dict[int, dict] = {}
    for step in range(k):
        extra_now = [(c["lon"], c["lat"], n) for c in chosen]
        if chosen:  # celulele din raza ultimei alegeri își schimbă concurența
            last = chosen[-1]
            upd = [h for h, (lo, la) in pos.items() if h not in {c["h3_index"] for c in chosen}
                   and _haversine_m(lo, la, last["lon"], last["lat"]) <= RAZA_CONCURENTA_M]
            if upd:
                vals = _hours_for(conn, model, [pos[h][0] for h in upd], [pos[h][1] for h in upd],
                                  [n] * len(upd), [p] * len(upd), extra=extra_now)
                own.update(dict(zip(upd, vals)))
        eligible = [h for h in pos if h not in {c["h3_index"] for c in chosen}
                    and all(_haversine_m(pos[h][0], pos[h][1], c["lon"], c["lat"]) >= float(min_dist_m) for c in chosen)]
        if not eligible:
            break
        short = sorted(eligible, key=lambda h: (-round(own[h], 6), -tie[h][0], -tie[h][1], h))[: int(shortlist)]
        best = None
        for h in short:
            lo, la = pos[h]
            # pierderea stațiilor existente din raza candidatului
            near = stations_near(lo, la)
            loss_exist, detalii = 0.0, []
            if len(near):
                ids = near["statie_id"].tolist()
                before = _hours_for(conn, model, near["lon"], near["lat"], near["n_piles"], near["power_kw"], ids, extra_now)
                after = _hours_for(conn, model, near["lon"], near["lat"], near["n_piles"], near["power_kw"], ids,
                                   extra_now + [(lo, la, n)])
                loss_exist = float((before - after).sum())
                detalii = [{"statie_id": int(i), "cod": c, "lat": float(a), "lon": float(o), "inainte": float(b), "dupa": float(f)}
                           for i, c, a, o, b, f in zip(ids, near["cod"], near["lat"], near["lon"], before, after) if b - f > 1e-9]
            # pierderea stațiilor alese anterior din raza candidatului
            loss_chosen = 0.0
            for c in chosen:
                if _haversine_m(lo, la, c["lon"], c["lat"]) > RAZA_CONCURENTA_M:
                    continue
                others = [(o["lon"], o["lat"], n) for o in chosen if o is not c]
                b = _hours_for(conn, model, [c["lon"]], [c["lat"]], [n], [p], extra=others)[0]
                a = _hours_for(conn, model, [c["lon"]], [c["lat"]], [n], [p], extra=others + [(lo, la, n)])[0]
                loss_chosen += float(b - a)
            net = own[h] - loss_exist - loss_chosen
            if best is None or net > best["castig_net_ore_zi"] + 1e-9:  # la egalitate rămâne ordinea din shortlist
                best = {"h3_index": h, "lon": lo, "lat": la, "ore_proprii_zi": float(own[h]),
                        "pierdere_statii_existente_ore_zi": loss_exist, "pierdere_statii_propuse_ore_zi": loss_chosen,
                        "castig_net_ore_zi": net, "_detalii": detalii}
        best["pas"] = step + 1
        for d in best.pop("_detalii"):
            cur = impact_total.setdefault(d["statie_id"], {**d})
            cur["dupa"] = min(cur["dupa"], d["dupa"])
        chosen.append(best)

    row = cand.set_index("h3_index")
    alese = []
    for c in chosen:
        r = row.loc[c["h3_index"]]
        alese.append({
            "pas": c["pas"], "h3_index": c["h3_index"], "lat": _r(c["lat"], 5), "lon": _r(c["lon"], 5),
            "judet": r["judet"], "ore_proprii_zi": _r(c["ore_proprii_zi"]),
            "pierdere_statii_existente_ore_zi": _r(c["pierdere_statii_existente_ore_zi"], 3),
            "pierdere_statii_propuse_ore_zi": _r(c["pierdere_statii_propuse_ore_zi"], 3),
            "castig_net_ore_zi": _r(c["castig_net_ore_zi"]),
            "kwh_zi_estimat": _r(c["ore_proprii_zi"] * p, 1),
            "pop_1km": _r(r["pop_1km"], 0),
            "poi_500m": int(r["poi_food_500m"] + r["poi_business_res_500m"] + r["poi_lifestyle_500m"]),
            "dist_nearest_station_m_initial": _r(r["dist_nearest_station_m"], 0),
            "peste_nivelul_de_baza": bool(c["ore_proprii_zi"] > base + 0.01),
        })
    afectate = sorted(impact_total.values(), key=lambda d: d["dupa"] - d["inainte"])
    return {
        "zona": z.to_dict(), "configuratie": {"n_piles": n, "power_kw": p}, "k_cerut": k,
        "min_dist_intre_statii_m": float(min_dist_m), "candidati_analizati": int(len(cand)),
        "nivel_de_baza_ore_zi": _r(base), "nota": NOTA_NIVEL_BAZA,
        "locatii_alese": alese,
        "total_ore_noi_zi": _r(sum(a["ore_proprii_zi"] for a in alese)),
        "total_castig_net_ore_zi": _r(sum(a["castig_net_ore_zi"] for a in alese)),
        "statii_existente_afectate": [
            {"cod": d["cod"], "ore_inainte": _r(d["inainte"]), "ore_dupa": _r(d["dupa"]),
             "delta": _r(d["dupa"] - d["inainte"], 3)} for d in afectate[:10]],
        "metoda": ("Greedy: la fiecare pas, din cele mai bune celule ale zonei (stație de referință), se alege celula "
                   "cu câștig net maxim = ore proprii - pierderea stațiilor existente și propuse pe 3 km; "
                   "concurența se recalculează după fiecare alegere."),
        "_harta": {"recomandari": [[a["lat"], a["lon"], a["pas"], a["h3_index"], a["ore_proprii_zi"]] for a in alese],
                   "afectate": [[d["lat"], d["lon"], d["cod"], _r(d["dupa"] - d["inainte"], 3)] for d in afectate]},
    }


# ============================================================================= 6. simulate_station
def simulate_station(conn, lat: float | None = None, lon: float | None = None, h3_index: str | None = None,
                     n_piles: int | None = None, power_kw: float | None = None) -> dict:
    if h3_index:
        if not h3.is_valid_cell(h3_index):
            raise ToolError(f"h3_index invalid: {h3_index}")
        lat, lon = h3.cell_to_latlng(h3_index)
    if lat is None or lon is None:
        raise ToolError("Dați lat și lon sau h3_index.")
    n, p = _config(n_piles, power_kw)
    r = evalueaza_punct(conn, float(lon), float(lat), n, p, cu_impact=True)
    im = r["impact_vecini"]
    return {
        "punct": {"lat": _r(lat, 5), "lon": _r(lon, 5), "judet": r["features"].get("judet")},
        "configuratie": {"n_piles": n, "power_kw": p},
        "sumar": {k: (_r(v, 3 if k.startswith("ocupare") else 1) if isinstance(v, float) else v)
                  for k, v in r["sumar"].items()},
        "percentila_ocupare_vs_statii_existente": _r(r["percentila_ocupare_vs_statii"], 3),
        "percentila_vs_grila_h3": _r(r["percentila_vs_grila_h3"], 3),
        "pop_suspect": bool(r["features"].get("pop_suspect")),
        "motiv": r["motiv"],
        "canibalizare": {
            "statii_pe_3km": len(im["statii"]),
            "pierdere_totala_ore_zi": _r(im["pierdere_ore_zi"], 3),
            "cele_mai_afectate": [{"cod": s["cod"], "dist_m": _r(s["dist_m"], 0), "ore_inainte": _r(s["ore_inainte"]),
                                   "ore_dupa": _r(s["ore_dupa"]), "delta": _r(s["delta_ore"], 3)}
                                  for s in im["statii"][:8]],
        },
        "_harta": {"punct": [float(lat), float(lon)],
                   "afectate": [[s["lat"], s["lon"], s["cod"], _r(s["delta_ore"], 3)]
                                for s in im["statii"] if s["delta_ore"] < -1e-9]},
    }


# ============================================================================= 7. economics
def _capex_default(power_kw):
    if power_kw <= 22:
        return settings.econ_capex_ac_per_punct
    if power_kw <= 60:
        return settings.econ_capex_dc50_per_punct
    return settings.econ_capex_dc150_per_punct


def economics(conn, kwh_zi: float, n_piles: int | None = None, power_kw: float | None = None,
              pret_kwh: float | None = None, cost_energie_kwh: float | None = None,
              capex_per_punct: float | None = None, opex_procent_capex: float | None = None,
              factor_putere_medie: float | None = None, factor_ajustare_cerere: float | None = None) -> dict:
    """Aritmetică simplă, cu ipotezele afișate. kwh_zi = valoarea prezisă de model (plafon: puncte ocupate x putere)."""
    n, p = _config(n_piles, power_kw)
    if kwh_zi is None or float(kwh_zi) < 0:
        raise ToolError("kwh_zi trebuie să fie un număr pozitiv (din predicție).")
    ip = {
        "pret_kwh_ron": float(pret_kwh if pret_kwh is not None else
                              (settings.econ_pret_kwh_ac if p <= 22 else settings.econ_pret_kwh_dc)),
        "cost_energie_kwh_ron": float(cost_energie_kwh if cost_energie_kwh is not None else settings.econ_cost_energie_kwh),
        "capex_per_punct_ron": float(capex_per_punct if capex_per_punct is not None else _capex_default(p)),
        "opex_procent_capex_an": float(opex_procent_capex if opex_procent_capex is not None else settings.econ_opex_procent),
        "factor_putere_medie": float(factor_putere_medie if factor_putere_medie is not None else settings.econ_factor_putere),
        "factor_ajustare_cerere": float(factor_ajustare_cerere if factor_ajustare_cerere is not None
                                        else settings.econ_factor_cerere),
    }
    kwh_an = float(kwh_zi) * ip["factor_putere_medie"] * ip["factor_ajustare_cerere"] * 365
    venit = kwh_an * ip["pret_kwh_ron"]
    cost_en = kwh_an * ip["cost_energie_kwh_ron"]
    capex = n * ip["capex_per_punct_ron"]
    opex = capex * ip["opex_procent_capex_an"] / 100
    profit = venit - cost_en - opex
    return {
        "configuratie": {"n_piles": n, "power_kw": p},
        "ipoteze": ip,
        "ipoteze_sunt_ilustrative": True,
        "avertisment": ("Nivelul cererii vine dintr-un model antrenat pe Shenzhen (piață EV mult mai matură decât "
                        "România); cifrele sunt utile pentru a compara locații, nu ca prognoză financiară. "
                        "factor_ajustare_cerere scalează cererea pentru piața locală."),
        "kwh_livrati_an": _r(kwh_an, 0), "venit_an_ron": _r(venit, 0), "cost_energie_an_ron": _r(cost_en, 0),
        "opex_an_ron": _r(opex, 0), "profit_operational_an_ron": _r(profit, 0), "capex_ron": _r(capex, 0),
        "ani_amortizare": _r(capex / profit, 1) if profit > 0 else None,
        "formule": {
            "kwh_livrati_an": "kwh_zi_prezis x factor_putere_medie x factor_ajustare_cerere x 365",
            "venit_an": "kwh_livrati_an x pret_kwh",
            "profit_operational_an": "venit - kwh_livrati_an x cost_energie_kwh - capex x opex%",
            "ani_amortizare": "capex / profit_operational_an",
        },
    }


# ============================================================================= 8. rank_areas
def rank_areas(conn, n_piles: int | None = None, power_kw: float | None = None, top: int = 10,
               min_dist_statie_m: float = 1000) -> dict:
    n, p = _config(n_piles, power_kw)
    g = _grid(conn, None, n, p, exclude_pop_suspect=False)
    ok = g[~g["pop_suspect"]]
    prag = _baseline(conn, n, p) + 0.01
    opp = ok[(ok["ore_incarcare_zi"] > prag) & (ok["dist_nearest_station_m"].fillna(1e12) >= float(min_dist_statie_m))]
    st = pd.DataFrame(conn.execute("""
        SELECT f.judet, COUNT(*) AS n_statii, AVG(ps.ocupare_medie) AS ocupare_statii
        FROM features_locatie f
        JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
        WHERE f.tip = 'statie' GROUP BY f.judet
    """).fetchall())
    jud = pd.DataFrame(conn.execute("SELECT nume AS judet, populatie_ins, raport_worldpop_ins FROM judet").fetchall())
    agg = opp.groupby("judet").agg(celule_oportunitate=("h3_index", "size"), ore_max=("ore_incarcare_zi", "max"))
    best = opp.sort_values("ore_incarcare_zi", ascending=False).groupby("judet").head(1).set_index("judet")
    t = jud.merge(st, on="judet", how="left").merge(agg, left_on="judet", right_index=True, how="left")
    t[["n_statii", "celule_oportunitate"]] = t[["n_statii", "celule_oportunitate"]].fillna(0).astype(int)
    t["statii_la_100k"] = t["n_statii"] / t["populatie_ins"] * 1e5
    t["pop_suspect"] = t["raport_worldpop_ins"] > settings.raport_pop_judet_suspect
    t = t.sort_values(["celule_oportunitate", "ore_max"], ascending=[False, False]).reset_index(drop=True)
    out = []
    for i, r in t.iterrows():
        b = best.loc[r["judet"]] if r["judet"] in best.index else None
        out.append({
            "loc": i + 1, "judet": r["judet"], "celule_oportunitate": int(r["celule_oportunitate"]),
            "ore_max_zi": _r(r["ore_max"]), "populatie_ins": int(r["populatie_ins"]) if r["populatie_ins"] == r["populatie_ins"] else None,
            "statii_existente": int(r["n_statii"]), "statii_la_100k_loc": _r(r["statii_la_100k"], 1),
            "ocupare_medie_prezisa_statii": _r(r["ocupare_statii"], 3),
            "populatie_suspecta": bool(r["pop_suspect"]),
            "cea_mai_buna_celula": ({"h3_index": b["h3_index"], "lat": _r(b["lat"], 5), "lon": _r(b["lon"], 5)}
                                    if b is not None else None),
        })
    return {
        "configuratie": {"n_piles": n, "power_kw": p},
        "definitie": (f"celulă de oportunitate = celulă H3 populată, nesuspectă, unde modelul prezice peste nivelul "
                      f"de bază ({prag - 0.01:.2f} ore/zi) și aflată la cel puțin {min_dist_statie_m:g} m de orice stație "
                      f"existentă; ordinea: numărul de celule, apoi orele maxime"),
        "nivel_de_baza_ore_zi": _r(prag - 0.01), "nota": NOTA_NIVEL_BAZA,
        "total_judete": len(out), "clasament": out[: int(top)],
        "_harta": {"recomandari": [[o["cea_mai_buna_celula"]["lat"], o["cea_mai_buna_celula"]["lon"], o["loc"],
                                    o["judet"], o["ore_max_zi"]]
                                   for o in out[: int(top)] if o["cea_mai_buna_celula"]]},
    }


# ============================================================================= registrul tool-urilor
_ZONA_PARAMS = {
    "zona": {"type": "string", "description": "Numele județului sau al localității (ex. „Cluj”, „Cluj-Napoca”, „județul Iași”)."},
    "judet": {"type": "string", "description": "Opțional: județul, pentru a dezambiguiza localitățile cu același nume."},
    "tip": {"type": "string", "enum": ["auto", "judet", "localitate"], "description": "Tipul zonei; implicit auto."},
}
_CONFIG_PARAMS = {
    "n_piles": {"type": "integer", "description": "Numărul de puncte de încărcare ale stației noi (implicit 2). Nu-l optimiza: efectul învățat nu e cauzal."},
    "power_kw": {"type": "number", "description": "Puterea pe punct, kW (implicit 50)."},
}


@dataclass
class ToolSpec:
    fn: Callable
    description: str
    parameters: dict


TOOLS: dict[str, ToolSpec] = {
    "resolve_area": ToolSpec(resolve_area,
        "Găsește un județ sau o localitate după nume. Întoarce tipul, județul, populația INS (pentru județe), "
        "suprafața, numărul de celule H3 evaluate și de stații existente. Dacă numele e ambiguu, întoarce variantele.",
        {"type": "object", "properties": {"nume": _ZONA_PARAMS["zona"], "judet": _ZONA_PARAMS["judet"],
                                          "tip": _ZONA_PARAMS["tip"]}, "required": ["nume"]}),
    "get_existing_stations": ToolSpec(get_existing_stations,
        "Stațiile de încărcare existente din zonă: număr, puncte, operatori, utilizarea prezisă (ore de încărcare/zi, "
        "ocupare) și cele mai utilizate stații.",
        {"type": "object", "properties": {**_ZONA_PARAMS, "top": {"type": "integer", "description": "Câte stații listate (implicit 15)."}},
         "required": ["zona"]}),
    "score_candidates": ToolSpec(score_candidates,
        "Celulele H3 din zonă ordonate după utilizarea prezisă a unei stații noi (ore de încărcare/zi, ocupare, kWh/zi). "
        "Nu ține cont de canibalizarea dintre ele; pentru un set de locații folosește plan_network.",
        {"type": "object", "properties": {**_ZONA_PARAMS, **_CONFIG_PARAMS,
            "min_dist_statie_m": {"type": "number", "description": "Exclude celulele mai apropiate de o stație existentă (m)."},
            "top_k": {"type": "integer", "description": "Câte celule întoarce (implicit 10)."}}, "required": ["zona"]}),
    "explain_cell": ToolSpec(explain_cell,
        "Explică predicția pentru o celulă H3 sau un punct: feature-urile, top 3 contribuții SHAP (efect multiplicativ "
        "asupra ocupării) și comparația cu media județului.",
        {"type": "object", "properties": {"h3_index": {"type": "string"}, "lat": {"type": "number"},
                                          "lon": {"type": "number"}, **_CONFIG_PARAMS}}),
    "plan_network": ToolSpec(plan_network,
        "Alege setul de k locații noi pentru zonă (greedy cu recalcularea concurenței după fiecare alegere), "
        "maximizând câștigul net = ore de încărcare proprii - pierderea stațiilor vecine. Tool-ul principal pentru recomandări.",
        {"type": "object", "properties": {**_ZONA_PARAMS, **_CONFIG_PARAMS,
            "k": {"type": "integer", "description": "Câte stații noi (1-10, implicit 3)."},
            "min_dist_m": {"type": "number", "description": "Distanța minimă între stațiile propuse (m, implicit 1000)."},
            "min_dist_statie_m": {"type": "number", "description": "Distanța minimă față de stațiile existente (m, implicit 0)."}},
         "required": ["zona"]}),
    "simulate_station": ToolSpec(simulate_station,
        "Simulează o stație nouă într-un punct (lat/lon sau h3_index): cererea prezisă, percentila și cât pierd "
        "stațiile vecine pe 3 km (canibalizare).",
        {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"},
                                          "h3_index": {"type": "string"}, **_CONFIG_PARAMS}}),
    "economics": ToolSpec(economics,
        "Calcul economic simplu pentru o stație: venit, cost energie, opex, capex, ani până la amortizare. "
        "Primește kwh_zi din predicție (sumar.kwh_zi sau kwh_zi_estimat); ipotezele nespecificate au valori ilustrative.",
        {"type": "object", "properties": {"kwh_zi": {"type": "number", "description": "kWh/zi prezis de model."},
            **_CONFIG_PARAMS,
            "pret_kwh": {"type": "number", "description": "Preț de vânzare, RON/kWh."},
            "cost_energie_kwh": {"type": "number", "description": "Cost energie, RON/kWh."},
            "capex_per_punct": {"type": "number", "description": "Investiție per punct, RON."},
            "opex_procent_capex": {"type": "number", "description": "Cost anual de operare, % din capex."},
            "factor_putere_medie": {"type": "number", "description": "Putere medie livrată / putere nominală (0-1)."},
            "factor_ajustare_cerere": {"type": "number", "description": "Scalarea cererii prezise pentru piața locală (implicit din config)."}},
         "required": ["kwh_zi"]}),
    "rank_areas": ToolSpec(rank_areas,
        "Clasamentul județelor după potențialul neacoperit: celule de oportunitate (ore/zi >= P90 național, departe de "
        "stații), stații la 100k locuitori, ocuparea prezisă a stațiilor existente.",
        {"type": "object", "properties": {**_CONFIG_PARAMS,
            "top": {"type": "integer", "description": "Câte județe în clasament (implicit 10)."},
            "min_dist_statie_m": {"type": "number", "description": "Distanța minimă față de stații existente (m, implicit 1000)."}}}),
}


def run_tool(conn, name: str, args: dict | None) -> dict:
    if name not in TOOLS:
        raise ToolError(f"Tool necunoscut: {name}")
    spec = TOOLS[name]
    args = {k: v for k, v in (args or {}).items() if v is not None}
    sig = inspect.signature(spec.fn)
    unknown = [k for k in args if k not in sig.parameters]
    if unknown:
        raise ToolError(f"Parametri necunoscuți pentru {name}: {unknown}")
    try:
        sig.bind(conn, **args)
    except TypeError as e:
        raise ToolError(f"Parametri invalizi pentru {name}: {e}") from None
    return spec.fn(conn, **args)


def for_llm(result: Any) -> Any:
    """Elimină cheile „_*” (doar pentru hartă) înainte de a trimite rezultatul modelului."""
    return to_jsonable(result, drop_private=True)


def to_jsonable(result: Any, drop_private: bool = False) -> Any:
    """Convertește tipurile numpy/pandas în tipuri JSON; opțional elimină cheile „_*”."""
    if isinstance(result, dict):
        return {k: to_jsonable(v, drop_private) for k, v in result.items()
                if not (drop_private and str(k).startswith("_"))}
    if isinstance(result, (list, tuple)):
        return [to_jsonable(v, drop_private) for v in result]
    if isinstance(result, float) and math.isnan(result):
        return None
    if isinstance(result, (np.integer,)):
        return int(result)
    if isinstance(result, (np.floating,)):
        return float(result)
    if isinstance(result, (np.bool_,)):
        return bool(result)
    return result
