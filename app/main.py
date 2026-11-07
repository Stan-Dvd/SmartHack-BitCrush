from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from psycopg.errors import UniqueViolation

from .config import settings
from .db import get_connection, open_pool, close_pool
from .h3_utils import station_cells
from .repository import ensure_h3_cell, ensure_temporal_partition

@asynccontextmanager
async def lifespan(app: FastAPI):
    open_pool()
    yield
    close_pool()

app = FastAPI(
    title="UrbanEV API",
    version="2.0.0",
    description="FastAPI + PostgreSQL/PostGIS + H3 pentru infrastructura EV.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StationCreate(BaseModel):
    statie_id: int
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    charge_count: int = Field(default=1, gt=0)
    taz_id: Optional[int] = None
    area: Optional[float] = Field(default=None, ge=0)
    perimeter: Optional[float] = Field(default=None, ge=0)

class SnapshotCreate(BaseModel):
    timestamp: datetime
    busy: int = Field(ge=0)
    e_price: Optional[float] = Field(default=None, ge=0)
    s_price: Optional[float] = Field(default=None, ge=0)
    # Optional: permite un rating diferit pentru o anumită stație/observație.
    # Dacă lipsește, se folosește PUMP_RATING_KW din .env.
    pump_rating_kw: Optional[float] = Field(default=None, gt=0)

def normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value

@app.get("/health")
def health():
    with get_connection() as conn:
        conn.execute("SELECT 1")
    return {
        "status": "ok",
        "database": "connected",
        "h3_res8": settings.h3_res8,
        "h3_res9": settings.h3_res9,
        "pump_rating_kw": settings.pump_rating_kw,
        "snapshot_minutes": settings.snapshot_minutes,
    }

@app.get("/api/stations")
def list_stations(
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
    taz_id: Optional[int] = None,
    h3_res8: Optional[str] = None,
    limit: int = Query(5000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
):
    filters, params = [], []
    if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
        filters.append("""
            ST_Intersects(
                s.locatie::geometry,
                ST_MakeEnvelope(%s,%s,%s,%s,4326)
            )
        """)
        params.extend([min_lon, min_lat, max_lon, max_lat])
    if taz_id is not None:
        filters.append("s.taz_id = %s")
        params.append(taz_id)
    if h3_res8:
        filters.append("s.h3_res8 = %s")
        params.append(h3_res8)

    where = "WHERE " + " AND ".join(filters) if filters else ""
    sql = f"""
        SELECT s.statie_id, s.longitude AS lng, s.latitude AS lat,
               s.charge_count, s.taz_id, s.area, s.perimeter,
               s.h3_res8, s.h3_res9,
               ST_AsGeoJSON(s.locatie::geometry)::json AS geometry
        FROM statie_incarcare s
        {where}
        ORDER BY s.statie_id
        LIMIT %s OFFSET %s
    """
    with get_connection() as conn:
        return conn.execute(sql, [*params, limit, offset]).fetchall()

@app.get("/api/stations/{station_id}")
def get_station(station_id: int):
    with get_connection() as conn:
        station = conn.execute("""
            SELECT statie_id, longitude AS lng, latitude AS lat,
                   charge_count, taz_id, area, perimeter, h3_res8, h3_res9,
                   ST_AsGeoJSON(locatie::geometry)::json AS geometry
            FROM statie_incarcare
            WHERE statie_id = %s
        """, [station_id]).fetchone()
        if not station:
            raise HTTPException(404, "Stația nu există.")
        station["observations"] = conn.execute("""
            SELECT timestamp, busy, volume, e_price, s_price
            FROM date_temporale_statie
            WHERE statie_id = %s
            ORDER BY timestamp DESC
            LIMIT 500
        """, [station_id]).fetchall()
        return station

@app.get("/api/stats")
def get_stats():
    with get_connection() as conn:
        stations = conn.execute("""
            SELECT COUNT(*)::int AS station_count,
                   COALESCE(SUM(charge_count),0)::bigint AS total_piles,
                   COUNT(DISTINCT taz_id)::int AS taz_count,
                   COUNT(DISTINCT h3_res8)::int AS h3_res8_count,
                   COUNT(DISTINCT h3_res9)::int AS h3_res9_count
            FROM statie_incarcare
        """).fetchone()
        temporal = conn.execute("""
            SELECT COUNT(*)::bigint AS observation_count,
                   COALESCE(SUM(busy),0) AS busy_sum,
                   COALESCE(SUM(volume),0) AS total_volume_kwh,
                   AVG(busy) AS avg_busy,
                   AVG(volume) AS avg_volume_kwh,
                   AVG(e_price) AS avg_e_price,
                   AVG(s_price) AS avg_s_price
            FROM date_temporale_statie
        """).fetchone()
    return {"stations": stations, "temporal": temporal}

@app.get("/api/heatmap")
def get_heatmap(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    resolution: int = Query(9, ge=0, le=15),
    limit: int = Query(10000, ge=1, le=50000),
):
    if resolution not in (8, 9):
        raise HTTPException(400, "Rezoluțiile disponibile pentru stații sunt 8 și 9.")
    time_filters, params = [], []
    if start:
        time_filters.append("d.timestamp >= %s")
        params.append(normalize_timestamp(start))
    if end:
        time_filters.append("d.timestamp < %s")
        params.append(normalize_timestamp(end))
    where = "WHERE " + " AND ".join(time_filters) if time_filters else ""
    h3_col = "s.h3_res8" if resolution == 8 else "s.h3_res9"
    sql = f"""
        WITH grouped AS (
            SELECT {h3_col} AS h3_index,
                   AVG(d.busy) AS avg_busy,
                   SUM(d.volume) AS total_volume,
                   COUNT(DISTINCT s.statie_id)::int AS station_count,
                   ST_Y(ST_Centroid(ST_Collect(s.locatie::geometry))) AS lat,
                   ST_X(ST_Centroid(ST_Collect(s.locatie::geometry))) AS lon
            FROM statie_incarcare s
            JOIN date_temporale_statie d ON d.statie_id = s.statie_id
            {where}
            GROUP BY {h3_col}
        )
        SELECT h3_index, lat, lon, avg_busy, total_volume, station_count,
               CASE WHEN MAX(total_volume) OVER () > 0
                    THEN total_volume / MAX(total_volume) OVER ()
                    ELSE 0 END AS score
        FROM grouped
        ORDER BY score DESC
        LIMIT %s
    """
    with get_connection() as conn:
        rows = conn.execute(sql, [*params, limit]).fetchall()
    return [
        {**r, "lat": float(r["lat"]), "lon": float(r["lon"]),
         "score": float(r["score"] or 0),
         "avg_busy": float(r["avg_busy"]) if r["avg_busy"] is not None else None,
         "total_volume": float(r["total_volume"]) if r["total_volume"] is not None else None}
        for r in rows
    ]

@app.get("/api/h3/cells")
def list_h3_cells(
    resolution: int = Query(8, ge=0, le=15),
    min_lon: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lon: Optional[float] = None,
    max_lat: Optional[float] = None,
):
    filters, params = ["resolution = %s"], [resolution]
    if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
        filters.append("ST_Intersects(boundary, ST_MakeEnvelope(%s,%s,%s,%s,4326))")
        params.extend([min_lon, min_lat, max_lon, max_lat])
    with get_connection() as conn:
        return conn.execute(f"""
            SELECT h3_index, resolution,
                   ST_AsGeoJSON(boundary)::json AS boundary,
                   ST_AsGeoJSON(center::geometry)::json AS center
            FROM celule_h3
            WHERE {" AND ".join(filters)}
            ORDER BY h3_index
        """, params).fetchall()

@app.get("/api/h3/cells/{h3_index}")
def get_h3_cell(h3_index: str):
    with get_connection() as conn:
        cell = conn.execute("""
            SELECT h3_index, resolution,
                   ST_AsGeoJSON(boundary)::json AS boundary,
                   ST_AsGeoJSON(center::geometry)::json AS center
            FROM celule_h3 WHERE h3_index = %s
        """, [h3_index]).fetchone()
        if not cell:
            raise HTTPException(404, "Celula H3 nu există.")
        h3_column = "h3_res8" if cell["resolution"] == 8 else "h3_res9"
        cell["stations"] = conn.execute(f"""
            SELECT statie_id, longitude AS lng, latitude AS lat,
                   charge_count, taz_id
            FROM statie_incarcare WHERE {h3_column} = %s
            ORDER BY statie_id
        """, [h3_index]).fetchall()
        cell["aggregates"] = conn.execute(f"""
            SELECT COUNT(DISTINCT s.statie_id)::int AS station_count,
                   COUNT(d.timestamp)::bigint AS observation_count,
                   AVG(d.busy) AS avg_busy,
                   SUM(d.volume) AS total_volume_kwh,
                   AVG(d.e_price) AS avg_e_price,
                   AVG(d.s_price) AS avg_s_price
            FROM statie_incarcare s
            LEFT JOIN date_temporale_statie d ON d.statie_id = s.statie_id
            WHERE s.{h3_column} = %s
        """, [h3_index]).fetchone()
        return cell

@app.get("/api/insights")
def get_insights(
    resolution: int = Query(8, ge=8, le=9),
    limit: int = Query(10, ge=1, le=100),
):
    h3_col = "s.h3_res8" if resolution == 8 else "s.h3_res9"
    with get_connection() as conn:
        return conn.execute(f"""
            SELECT {h3_col} AS h3_index,
                   COUNT(DISTINCT s.statie_id)::int AS station_count,
                   AVG(d.busy) AS avg_busy,
                   SUM(d.volume) AS total_volume_kwh,
                   AVG(d.e_price) AS avg_e_price,
                   AVG(d.s_price) AS avg_s_price
            FROM statie_incarcare s
            JOIN date_temporale_statie d ON d.statie_id = s.statie_id
            GROUP BY {h3_col}
            ORDER BY SUM(d.volume) DESC NULLS LAST
            LIMIT %s
        """, [limit]).fetchall()

@app.post("/api/stations")
def create_station(payload: StationCreate):
    h3_8, h3_9 = station_cells(payload.latitude, payload.longitude)
    try:
        with get_connection() as conn:
            with conn.transaction():
                ensure_h3_cell(conn, h3_8, 8)
                ensure_h3_cell(conn, h3_9, 9)
                row = conn.execute("""
                    INSERT INTO statie_incarcare
                    (statie_id, locatie, h3_res8, h3_res9, charge_count,
                     taz_id, area, perimeter)
                    VALUES (
                        %s,
                        ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,
                        %s,%s,%s,%s,%s,%s
                    )
                    RETURNING statie_id, longitude, latitude, h3_res8, h3_res9
                """, [payload.statie_id, payload.longitude, payload.latitude,
                      h3_8, h3_9, payload.charge_count, payload.taz_id,
                      payload.area, payload.perimeter]).fetchone()
        return {"created": True, "station": row}
    except UniqueViolation:
        raise HTTPException(409, "Există deja o stație cu acest statie_id.")

@app.post("/api/stations/{station_id}/snapshot")
def add_snapshot(station_id: int, payload: SnapshotCreate):
    timestamp = normalize_timestamp(payload.timestamp)
    rating = payload.pump_rating_kw or settings.pump_rating_kw
    volume_kwh = payload.busy * rating * settings.snapshot_minutes / 60.0

    with get_connection() as conn:
        exists = conn.execute(
            "SELECT charge_count FROM statie_incarcare WHERE statie_id = %s",
            [station_id],
        ).fetchone()
        if not exists:
            raise HTTPException(404, "Stația nu există.")
        if payload.busy > exists["charge_count"]:
            raise HTTPException(
                422,
                f"busy ({payload.busy}) nu poate depăși charge_count ({exists['charge_count']}).",
            )
        with conn.transaction():
            ensure_temporal_partition(conn, timestamp)
            row = conn.execute("""
                INSERT INTO date_temporale_statie
                    (statie_id, timestamp, busy, volume, e_price, s_price)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (statie_id, timestamp) DO UPDATE SET
                    busy = EXCLUDED.busy,
                    volume = EXCLUDED.volume,
                    e_price = EXCLUDED.e_price,
                    s_price = EXCLUDED.s_price
                RETURNING statie_id, timestamp, busy, volume, e_price, s_price
            """, [station_id, timestamp, payload.busy, volume_kwh,
                  payload.e_price, payload.s_price]).fetchone()

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
