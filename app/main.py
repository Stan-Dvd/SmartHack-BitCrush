from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import h3
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, Field

from .config import settings
from .db import close_pool, get_connection, open_pool
from .h3_utils import station_cells
from .ml_service import active_model
from .predictii import (
    _reason,
    evalueaza_punct,
    h3_grid_cells,
    recalc_h3,
    recalc_statii,
    salveaza_candidat,
    statii_in_raza,
)
from .repository import ensure_h3_cell, ensure_temporal_partition
from . import agent as agent_mod
from .agent_tools import TOOLS, ToolError, run_tool, to_jsonable
from .llm_gemini import LLMUnavailable


@asynccontextmanager
async def lifespan(app: FastAPI):
    open_pool()
    yield
    close_pool()


app = FastAPI(
    title="UrbanEV API",
    version="3.0.0",
    description=(
        "FastAPI + PostgreSQL/PostGIS + H3 + LightGBM. Predicții de utilizare pentru stațiile de încărcare "
        "existente și evaluarea locațiilor noi."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================= modele de request
class StationCreate(BaseModel):
    statie_id: Optional[int] = Field(default=None, gt=0)
    cod: Optional[str] = None
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    n_piles: int = Field(default=1, gt=0)
    power_kw: float = Field(default=22.0, gt=0)
    judet: Optional[str] = None
    localitate: Optional[str] = None
    operator: Optional[str] = None
    is_open_24_7: bool = False


class SnapshotCreate(BaseModel):
    timestamp: datetime
    busy: int = Field(ge=0)
    e_price: Optional[float] = Field(default=None, ge=0)
    s_price: Optional[float] = Field(default=None, ge=0)
    # Opțional: putere diferită pentru această observație; implicit power_kw al stației.
    pump_rating_kw: Optional[float] = Field(default=None, gt=0)


class CandidateEvaluate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    n_piles: Optional[int] = Field(default=None, gt=0)
    power_kw: Optional[float] = Field(default=None, gt=0)
    cu_impact: bool = True       # calculează canibalizarea stațiilor existente pe 3 km
    salveaza: bool = False       # persistă candidatul în locatie_candidat
    nume: Optional[str] = None
    judet: Optional[str] = None


class CandidateStatus(BaseModel):
    status: Literal["propus", "evaluat", "aprobat", "respins"]


class CandidateBatchItem(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    n_piles: Optional[int] = Field(default=None, gt=0)
    power_kw: Optional[float] = Field(default=None, gt=0)
    nume: Optional[str] = None
    judet: Optional[str] = None


class CandidateBatch(BaseModel):
    candidati: list[CandidateBatchItem] = Field(min_length=1, max_length=50)
    status: Literal["propus", "evaluat", "aprobat", "respins"] = "propus"


class AgentRun(BaseModel):
    actiune: Literal["sugereaza_statii", "evalueaza_punct", "compara_judete"]
    parametri: dict = Field(default_factory=dict)


def normalize_timestamp(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _bbox_filter(prefix, min_lon, min_lat, max_lon, max_lat, filters, params, geo="locatie"):
    if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
        filters.append(f"ST_Intersects({prefix}{geo}::geometry, ST_MakeEnvelope(%s,%s,%s,%s,4326))")
        params.extend([min_lon, min_lat, max_lon, max_lat])


# ============================================================================= sistem
@app.get("/health")
def health():
    with get_connection() as conn:
        conn.execute("SELECT 1")
        model = conn.execute("SELECT model_id, nume, versiune FROM model_ml WHERE activ").fetchone()
    return {
        "status": "ok",
        "database": "connected",
        "model_activ": model,
        "h3_res8": settings.h3_res8,
        "h3_res9": settings.h3_res9,
        "snapshot_minutes": settings.snapshot_minutes,
        "timezone": settings.timezone,
    }


@app.get("/api/models")
def list_models():
    with get_connection() as conn:
        return conn.execute("""
            SELECT model_id, nume, versiune, cale_fisier, features, target_mode, antrenat_pe, limitari,
                   activ, created_at, metrici -> 'metrics' -> 'LGBM' AS metrici_test
            FROM model_ml ORDER BY model_id DESC
        """).fetchall()


@app.post("/api/admin/recalculeaza")
def recalculeaza(statii: bool = True, grila_h3: bool = False,
                 min_pop: float = Query(settings.h3_grid_min_pop, ge=0)):
    """Recalculează feature-urile și predicțiile (de ex. după înregistrarea unui model nou)."""
    result = {}
    with get_connection() as conn:
        if statii:
            result["statii"] = recalc_statii(conn)
        if grila_h3:
            conn.execute("DELETE FROM features_locatie WHERE tip = 'h3'")
            result["celule_h3"] = recalc_h3(conn, h3_grid_cells(conn, settings.h3_grid_res, min_pop))
    return result


# ============================================================================= stații existente
STATION_COLS = """
    statie_id, cod, longitude AS lng, latitude AS lat, judet, localitate, operator, operator_implicit,
    n_piles, power_kw,
    power_min_kw, power_max_kw, is_open_24_7, activa, h3_res8, h3_res9,
    ore_incarcare_zi, kwh_zi, ocupare_medie, ocupare_varf, ora_varf, weekend_varf, saturata_prezis,
    percentila_nationala, pop_suspect
"""


@app.get("/api/stations")
def list_stations(
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    judet: Optional[str] = None,
    h3_res8: Optional[str] = None,
    doar_active: bool = False,
    order_by: Literal["statie_id", "ore_incarcare_zi", "ocupare_medie"] = "statie_id",
    limit: int = Query(5000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
):
    filters, params = [], []
    if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
        filters.append("longitude BETWEEN %s AND %s AND latitude BETWEEN %s AND %s")
        params.extend([min_lon, max_lon, min_lat, max_lat])
    if judet:
        filters.append("judet = %s")
        params.append(judet)
    if h3_res8:
        filters.append("h3_res8 = %s")
        params.append(h3_res8)
    if doar_active:
        filters.append("activa")
    where = "WHERE " + " AND ".join(filters) if filters else ""
    order = "statie_id" if order_by == "statie_id" else f"{order_by} DESC NULLS LAST"
    with get_connection() as conn:
        return conn.execute(f"""
            SELECT {STATION_COLS} FROM v_statie_predictie {where}
            ORDER BY {order} LIMIT %s OFFSET %s
        """, [*params, limit, offset]).fetchall()


def _station_or_404(conn, station_id: int):
    station = conn.execute(f"""
        SELECT {STATION_COLS}, feature_id, dist_nearest_station_m, n_stations_500m, n_stations_1km,
               n_stations_3km, competitor_piles_1km, pop_1km, poi_food_500m, poi_business_res_500m,
               poi_lifestyle_500m, features_calculat_la, model_id
        FROM v_statie_predictie WHERE statie_id = %s
    """, [station_id]).fetchone()
    if not station:
        raise HTTPException(404, "Stația nu există.")
    return station


@app.get("/api/stations/{station_id}")
def get_station(station_id: int, observatii: int = Query(500, ge=0, le=10000)):
    with get_connection() as conn:
        station = _station_or_404(conn, station_id)
        station["profil"] = conn.execute("""
            SELECT hour, is_weekend, busy_pred, ocupare_pred, kwh_pred
            FROM predictie_profil
            WHERE feature_id = %s AND model_id = fn_model_activ()
            ORDER BY is_weekend, hour
        """, [station["feature_id"]]).fetchall()
        station["observations"] = conn.execute("""
            SELECT timestamp, busy, volume, e_price, s_price
            FROM date_temporale_statie WHERE statie_id = %s
            ORDER BY timestamp DESC LIMIT %s
        """, [station_id, observatii]).fetchall()
        return station


@app.get("/api/stations/{station_id}/forecast")
def station_forecast(
    station_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    calibrat: bool = True,
):
    """Prognoza orară a utilizării: profilul prezis de model, proiectat pe calendar (ora locală) și
    calibrat cu observațiile din ultimele 28 de zile, dacă există cel puțin 7 zile de date."""
    start = normalize_timestamp(start) if start else datetime.now(timezone.utc).replace(minute=0, second=0,
                                                                                         microsecond=0)
    end = normalize_timestamp(end) if end else start + timedelta(days=7)
    if end <= start:
        raise HTTPException(422, "end trebuie să fie după start.")
    if end - start > timedelta(days=62):
        raise HTTPException(422, "Intervalul maxim este de 62 de zile.")
    with get_connection() as conn:
        _station_or_404(conn, station_id)
        rows = conn.execute("""
            SELECT * FROM fn_prognoza_statie(%s, %s, %s, %s, %s)
        """, [station_id, start, end, calibrat, settings.timezone]).fetchall()
    total_kwh = sum(r["kwh_prognoza"] for r in rows)
    return {
        "statie_id": station_id,
        "start": start, "end": end,
        "calibrat": bool(rows and rows[0]["are_calibrare"]),
        "factor_calibrare": rows[0]["factor_calibrare"] if rows else None,
        "total_ore_incarcare": sum(r["busy_prognoza"] for r in rows),
        "total_kwh": total_kwh,
        "ore": rows,
    }


@app.post("/api/stations")
def create_station(payload: StationCreate):
    """Adaugă o stație, îi calculează predicția și recalculează concurența stațiilor pe 3 km."""
    h3_8, h3_9 = station_cells(payload.latitude, payload.longitude)
    try:
        with get_connection() as conn:
            with conn.transaction():
                ensure_h3_cell(conn, h3_8, settings.h3_res8)
                ensure_h3_cell(conn, h3_9, settings.h3_res9)
                sid = payload.statie_id or conn.execute(
                    "SELECT COALESCE(MAX(statie_id), 0) + 1 AS id FROM statie_incarcare").fetchone()["id"]
                conn.execute("""
                    INSERT INTO statie_incarcare
                        (statie_id, cod, locatie, judet, localitate, operator, n_piles, power_kw,
                         power_max_kw, is_open_24_7, sursa, h3_res8, h3_res9)
                    VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                            %s, %s, %s, %s, %s, %s, %s, 'api', %s, %s)
                """, [sid, payload.cod or f"RO{sid:04d}", payload.longitude, payload.latitude,
                      payload.judet, payload.localitate, payload.operator, payload.n_piles, payload.power_kw,
                      payload.power_kw, payload.is_open_24_7, h3_8, h3_9])
                vecini = statii_in_raza(conn, payload.longitude, payload.latitude, exclude_id=sid)
                recalc_statii(conn, [sid, *vecini])
            station = _station_or_404(conn, sid)
        return {"created": True, "station": station, "vecini_recalculati": len(vecini)}
    except UniqueViolation:
        raise HTTPException(409, "Există deja o stație cu acest statie_id sau cod.")


@app.post("/api/stations/{station_id}/snapshot")
def add_snapshot(station_id: int, payload: SnapshotCreate):
    timestamp = normalize_timestamp(payload.timestamp)
    with get_connection() as conn:
        st = conn.execute(
            "SELECT n_piles, power_kw FROM statie_incarcare WHERE statie_id = %s", [station_id]
        ).fetchone()
        if not st:
            raise HTTPException(404, "Stația nu există.")
        if payload.busy > st["n_piles"]:
            raise HTTPException(422, f"busy ({payload.busy}) nu poate depăși n_piles ({st['n_piles']}).")
        rating = payload.pump_rating_kw or st["power_kw"] or settings.pump_rating_kw
        volume_kwh = payload.busy * rating * settings.snapshot_minutes / 60.0
        with conn.transaction():
            ensure_temporal_partition(conn, timestamp)
            row = conn.execute("""
                INSERT INTO date_temporale_statie (statie_id, timestamp, busy, volume, e_price, s_price)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (statie_id, timestamp) DO UPDATE SET
                    busy = EXCLUDED.busy, volume = EXCLUDED.volume,
                    e_price = EXCLUDED.e_price, s_price = EXCLUDED.s_price
                RETURNING statie_id, timestamp, busy, volume, e_price, s_price
            """, [station_id, timestamp, payload.busy, volume_kwh, payload.e_price, payload.s_price]).fetchone()
    return {
        "saved": True,
        "observation": row,
        "calculation": {
            "busy": payload.busy,
            "pump_rating_kw": rating,
            "interval_minutes": settings.snapshot_minutes,
            "formula": "busy * pump_rating_kw * interval_minutes / 60",
            "volume_unit": "kWh",
        },
    }


# ============================================================================= locații noi
@app.post("/api/candidates/evaluate")
def evaluate_candidate(payload: CandidateEvaluate):
    """Performanța prezisă a unei stații noi în punctul dat și impactul asupra stațiilor vecine."""
    with get_connection() as conn:
        result = evalueaza_punct(conn, payload.longitude, payload.latitude, payload.n_piles, payload.power_kw,
                                 payload.cu_impact)
        if payload.salveaza:
            result["candidat_id"] = salveaza_candidat(
                conn, payload.longitude, payload.latitude, result["n_piles"], result["power_kw"],
                payload.nume, payload.judet)
        result["limitari"] = conn.execute("SELECT limitari FROM model_ml WHERE activ").fetchone()["limitari"]
    return result


CANDIDATE_COLS = """
    c.candidat_id, c.nume, c.longitude AS lng, c.latitude AS lat, c.judet, c.n_piles, c.power_kw,
    c.status, c.h3_res8, c.created_at, f.feature_id,
    ps.ore_incarcare_zi, ps.kwh_zi, ps.ocupare_medie, ps.ocupare_varf, ps.ora_varf, ps.saturata_prezis
"""
CANDIDATE_FROM = """
    FROM locatie_candidat c
    LEFT JOIN features_locatie f ON f.candidat_id = c.candidat_id
    LEFT JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
"""


@app.get("/api/candidates")
def list_candidates(status: Optional[str] = None, limit: int = Query(500, ge=1, le=5000)):
    where, params = ("WHERE c.status = %s", [status]) if status else ("", [])
    with get_connection() as conn:
        return conn.execute(f"""
            SELECT {CANDIDATE_COLS} {CANDIDATE_FROM} {where}
            ORDER BY ps.ore_incarcare_zi DESC NULLS LAST LIMIT %s
        """, [*params, limit]).fetchall()


@app.get("/api/candidates/{candidat_id}")
def get_candidate(candidat_id: int):
    with get_connection() as conn:
        cand = conn.execute(f"SELECT {CANDIDATE_COLS}, f.* {CANDIDATE_FROM} WHERE c.candidat_id = %s",
                            [candidat_id]).fetchone()
        if not cand:
            raise HTTPException(404, "Candidatul nu există.")
        cand["profil"] = conn.execute("""
            SELECT hour, is_weekend, busy_pred, ocupare_pred, kwh_pred FROM predictie_profil
            WHERE feature_id = %s AND model_id = fn_model_activ() ORDER BY is_weekend, hour
        """, [cand["feature_id"]]).fetchall()
        return cand


@app.patch("/api/candidates/{candidat_id}")
def update_candidate(candidat_id: int, payload: CandidateStatus):
    with get_connection() as conn:
        row = conn.execute("UPDATE locatie_candidat SET status = %s WHERE candidat_id = %s RETURNING candidat_id, status",
                           [payload.status, candidat_id]).fetchone()
    if not row:
        raise HTTPException(404, "Candidatul nu există.")
    return row


@app.post("/api/candidates/batch")
def save_candidates(payload: CandidateBatch):
    """Salvează mai multe locații (de ex. recomandările agentului) ca locatie_candidat, cu predicția lor."""
    ids = []
    with get_connection() as conn:
        with conn.transaction():
            for c in payload.candidati:
                n = c.n_piles or settings.predictie_n_piles_default
                p = c.power_kw or settings.predictie_power_kw_default
                cid = salveaza_candidat(conn, c.longitude, c.latitude, n, p, c.nume, c.judet)
                conn.execute("UPDATE locatie_candidat SET status = %s WHERE candidat_id = %s", [payload.status, cid])
                ids.append(cid)
    return {"salvati": len(ids), "candidat_ids": ids}


# ============================================================================= agent AI și tool-urile lui
@app.get("/api/agent/status")
def agent_status():
    return agent_mod.status()


@app.post("/api/agent/run")
def agent_run(payload: AgentRun):
    """Rulează o acțiune a agentului AI (Gemini + tool-uri deterministe). 503 dacă LLM-ul nu e disponibil."""
    try:
        return to_jsonable(agent_mod.run(payload.actiune, payload.parametri))
    except LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/tools")
def list_tools():
    return [{"nume": n, "descriere": s.description, "parametri": s.parameters} for n, s in TOOLS.items()]


@app.post("/api/tools/{nume}")
def call_tool(nume: str, argumente: dict | None = None):
    """Rulează direct un tool al agentului (fără LLM). Cheile „_harta” sunt straturile pentru hartă."""
    if nume not in TOOLS:
        raise HTTPException(404, f"Tool necunoscut: {nume}. Disponibile: {', '.join(TOOLS)}")
    try:
        with get_connection() as conn:
            return to_jsonable(run_tool(conn, nume, argumente or {}))
    except ToolError as e:
        raise HTTPException(422, str(e))


# ============================================================================= grila H3 de oportunități
OPPORTUNITY_SQL = """
    SELECT f.h3_index, ST_Y(c.center::geometry) AS lat, ST_X(c.center::geometry) AS lon,
           ps.ore_incarcare_zi, ps.kwh_zi, ps.ocupare_medie, ps.ocupare_varf, ps.ora_varf,
           ps.percentila_nationala, f.pop_1km, f.poi_food_500m, f.poi_business_res_500m,
           f.poi_lifestyle_500m, f.n_stations_1km, f.n_stations_3km, f.dist_nearest_station_m,
           f.pop_suspect, f.n_piles, f.power_kw, f.judet
    FROM features_locatie f
    JOIN celule_h3 c ON c.h3_index = f.h3_index
    JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
"""


@app.get("/api/opportunities")
def list_opportunities(
    judet: Optional[str] = None,
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_dist_statie_m: float = Query(0, ge=0, description="exclude celulele mai apropiate de o stație existentă"),
    exclude_pop_suspect: bool = True,
    limit: int = Query(20, ge=1, le=1000),
):
    """Celulele H3 unde o stație de referință (PREDICTIE_N_PILES_DEFAULT x PREDICTIE_POWER_KW_DEFAULT)
    ar avea cea mai mare utilizare prezisă. Celulele cu pop_suspect (județ cu populație WorldPop
    nesigură) sunt excluse implicit."""
    filters, params = ["f.tip = 'h3'"], []
    _bbox_filter("c.", min_lon, min_lat, max_lon, max_lat, filters, params, geo="center")
    if min_dist_statie_m:
        filters.append("COALESCE(f.dist_nearest_station_m, 1e12) >= %s")
        params.append(min_dist_statie_m)
    if exclude_pop_suspect:
        filters.append("NOT f.pop_suspect")
    if judet:
        filters.append("f.judet = %s")
        params.append(judet)
    with get_connection() as conn:
        rows = conn.execute(f"""
            {OPPORTUNITY_SQL}
            WHERE {" AND ".join(filters)}
            ORDER BY ps.ore_incarcare_zi DESC, f.h3_index
            LIMIT %s
        """, [*params, limit]).fetchall()
    for r in rows:
        r["motiv"] = _reason(r)
    return rows


@app.get("/api/heatmap")
def get_heatmap(
    sursa: Literal["predictie", "observatii"] = "predictie",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    resolution: int = Query(settings.h3_grid_res, ge=4, le=9),
    limit: int = Query(50000, ge=1, le=200000),
):
    """sursa=predictie: utilizarea prezisă a unei stații de referință în fiecare celulă H3 populată
    (agregată prin medie la rezoluții mai mici). sursa=observatii: agregarea observațiilor reale."""
    if sursa == "predictie":
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT f.h3_index, ps.ore_incarcare_zi
                FROM features_locatie f
                JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
                WHERE f.tip = 'h3' AND NOT f.pop_suspect
            """).fetchall()
        groups: dict[str, list[float]] = {}
        for r in rows:
            cell = r["h3_index"]
            if resolution < h3.get_resolution(cell):
                cell = h3.cell_to_parent(cell, resolution)
            groups.setdefault(cell, []).append(r["ore_incarcare_zi"])
        vals = {c: sum(v) / len(v) for c, v in groups.items()}
        top = max(vals.values(), default=0) or 1.0
        out = []
        for cell, v in sorted(vals.items(), key=lambda kv: -kv[1])[:limit]:
            lat, lon = h3.cell_to_latlng(cell)
            out.append({"h3_index": cell, "lat": lat, "lon": lon, "ore_incarcare_zi": v, "score": v / top})
        return out

    if resolution not in (8, 9):
        raise HTTPException(400, "Pentru observații, rezoluțiile disponibile sunt 8 și 9.")
    time_filters, params = [], []
    if start:
        time_filters.append("d.timestamp >= %s")
        params.append(normalize_timestamp(start))
    if end:
        time_filters.append("d.timestamp < %s")
        params.append(normalize_timestamp(end))
    where = "WHERE " + " AND ".join(time_filters) if time_filters else ""
    h3_col = "s.h3_res8" if resolution == 8 else "s.h3_res9"
    with get_connection() as conn:
        rows = conn.execute(f"""
            WITH grouped AS (
                SELECT {h3_col} AS h3_index, AVG(d.busy) AS avg_busy, SUM(d.volume) AS total_volume,
                       COUNT(DISTINCT s.statie_id)::int AS station_count,
                       ST_Y(ST_Centroid(ST_Collect(s.locatie::geometry))) AS lat,
                       ST_X(ST_Centroid(ST_Collect(s.locatie::geometry))) AS lon
                FROM statie_incarcare s
                JOIN date_temporale_statie d ON d.statie_id = s.statie_id
                {where}
                GROUP BY {h3_col}
            )
            SELECT h3_index, lat, lon, avg_busy::float AS avg_busy, total_volume::float AS total_volume,
                   station_count,
                   CASE WHEN MAX(total_volume) OVER () > 0
                        THEN (total_volume / MAX(total_volume) OVER ())::float ELSE 0 END AS score
            FROM grouped ORDER BY score DESC LIMIT %s
        """, [*params, limit]).fetchall()
    return rows


@app.get("/api/insights")
def get_insights(limit: int = Query(10, ge=1, le=100), min_dist_statie_m: float = Query(1000, ge=0)):
    """Top oportunități pentru stații noi (celule H3 la cel puțin min_dist_statie_m de o stație existentă)."""
    rows = list_opportunities(min_dist_statie_m=min_dist_statie_m, limit=limit, judet=None, min_lon=None,
                              min_lat=None, max_lon=None, max_lat=None, exclude_pop_suspect=True)
    return [{
        "h3_index": r["h3_index"], "lat": r["lat"], "lon": r["lon"], "judet": r["judet"],
        "type": "NEW_SITE",
        "title": f"{r['judet'] or 'România'}: celula H3 {r['h3_index']}",
        "score": r["ore_incarcare_zi"],
        "ocupare_medie": r["ocupare_medie"],
        "reason": f"{r['ore_incarcare_zi']:.1f} ore de încărcare/zi prezise pentru o stație de "
                  f"{r['n_piles']} x {r['power_kw']:g} kW; {r['motiv']}",
    } for r in rows]


@app.get("/api/stats")
def get_stats():
    with get_connection() as conn:
        stations = conn.execute("""
            SELECT COUNT(*)::int AS station_count,
                   COUNT(*) FILTER (WHERE activa)::int AS active_count,
                   COALESCE(SUM(n_piles), 0)::bigint AS total_piles,
                   COUNT(DISTINCT judet)::int AS judet_count,
                   COUNT(DISTINCT h3_res8)::int AS h3_res8_count
            FROM statie_incarcare
        """).fetchone()
        predictii = conn.execute("""
            SELECT f.tip, COUNT(*)::int AS n, AVG(ps.ore_incarcare_zi) AS ore_incarcare_zi_medie,
                   SUM(ps.kwh_zi) AS kwh_zi_total, AVG(ps.ocupare_medie) AS ocupare_medie,
                   COUNT(*) FILTER (WHERE ps.saturata_prezis)::int AS saturate
            FROM predictie_sumar ps JOIN features_locatie f USING (feature_id)
            WHERE ps.model_id = fn_model_activ()
            GROUP BY f.tip
        """).fetchall()
        temporal = conn.execute("""
            SELECT COUNT(*)::bigint AS observation_count, AVG(busy)::float AS avg_busy,
                   COALESCE(SUM(volume), 0)::float AS total_volume_kwh
            FROM date_temporale_statie
        """).fetchone()
    return {"stations": stations, "predictii": predictii, "temporal": temporal}


# ============================================================================= celule H3
@app.get("/api/h3/cells")
def list_h3_cells(
    resolution: int = Query(8, ge=0, le=15),
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    limit: int = Query(5000, ge=1, le=50000),
):
    filters, params = ["resolution = %s"], [resolution]
    _bbox_filter("", min_lon, min_lat, max_lon, max_lat, filters, params, geo="boundary")
    with get_connection() as conn:
        return conn.execute(f"""
            SELECT h3_index, resolution, ST_AsGeoJSON(boundary)::json AS boundary,
                   ST_AsGeoJSON(center::geometry)::json AS center
            FROM celule_h3 WHERE {" AND ".join(filters)}
            ORDER BY h3_index LIMIT %s
        """, [*params, limit]).fetchall()


@app.get("/api/h3/cells/{h3_index}")
def get_h3_cell(h3_index: str):
    with get_connection() as conn:
        cell = conn.execute("""
            SELECT h3_index, resolution, ST_AsGeoJSON(boundary)::json AS boundary,
                   ST_AsGeoJSON(center::geometry)::json AS center
            FROM celule_h3 WHERE h3_index = %s
        """, [h3_index]).fetchone()
        if not cell:
            raise HTTPException(404, "Celula H3 nu există.")
        col = {8: "h3_res8", 9: "h3_res9"}.get(cell["resolution"])
        cell["stations"] = conn.execute(f"""
            SELECT statie_id, cod, longitude AS lng, latitude AS lat, n_piles, power_kw, judet,
                   ore_incarcare_zi, ocupare_medie
            FROM v_statie_predictie WHERE {col} = %s ORDER BY statie_id
        """, [h3_index]).fetchall() if col else []
        cell["predictie"] = conn.execute("""
            SELECT f.*, ps.ore_incarcare_zi, ps.kwh_zi, ps.ocupare_medie, ps.ocupare_varf, ps.percentila_nationala
            FROM features_locatie f
            JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ()
            WHERE f.h3_index = %s
        """, [h3_index]).fetchone()
        return cell
