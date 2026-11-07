from psycopg import sql
from .h3_utils import cell_geometry_wkt, cell_center_wkt

def ensure_h3_cell(conn, cell: str, resolution: int):
    conn.execute("""
        INSERT INTO celule_h3 (h3_index, resolution, boundary, center)
        VALUES (
            %s, %s,
            ST_GeomFromText(%s, 4326),
            ST_SetSRID(ST_GeomFromText(%s), 4326)::geography
        )
        ON CONFLICT (h3_index) DO NOTHING
    """, [cell, resolution, cell_geometry_wkt(cell), cell_center_wkt(cell)])

def ensure_temporal_partition(conn, timestamp):
    # Creează automat partiția lunară înainte de INSERT.
    month_start = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    partition_name = f"date_temporale_statie_{month_start:%Y_%m}"
    # SQL identifiers și valori sunt compuse prin psycopg.sql, nu prin interpolare nesigură.
    statement = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} PARTITION OF date_temporale_statie
        FOR VALUES FROM ({}) TO ({})
    """).format(
        sql.Identifier(partition_name),
        sql.Literal(month_start),
        sql.Literal(next_month),
    )
    conn.execute(statement)
