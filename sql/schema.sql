CREATE EXTENSION IF NOT EXISTS postgis;

-- ATENȚIE: comenzile DROP șterg datele existente.
DROP TABLE IF EXISTS date_temporale_statie CASCADE;
DROP TABLE IF EXISTS statie_incarcare CASCADE;
DROP TABLE IF EXISTS celule_h3 CASCADE;

CREATE TABLE celule_h3 (
    h3_index VARCHAR(16) PRIMARY KEY,
    resolution SMALLINT NOT NULL,
    boundary GEOMETRY(POLYGON, 4326) NOT NULL,
    center GEOGRAPHY(POINT, 4326) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_h3_resolution CHECK (resolution BETWEEN 0 AND 15)
);

CREATE INDEX idx_celule_h3_boundary ON celule_h3 USING GIST (boundary);
CREATE INDEX idx_celule_h3_center ON celule_h3 USING GIST (center);

CREATE TABLE statie_incarcare (
    statie_id BIGINT PRIMARY KEY,
    locatie GEOGRAPHY(POINT, 4326) NOT NULL,
    longitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_X(locatie::geometry)) STORED,
    latitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_Y(locatie::geometry)) STORED,
    h3_res8 VARCHAR(16) NOT NULL,
    h3_res9 VARCHAR(16),
    charge_count INTEGER NOT NULL DEFAULT 1,
    -- Păstrate deoarece datele fizice inițiale includ TAZID, area și perimeter,
    -- iar schema trimisă le referențiază în indexuri și CHECK-uri.
    taz_id INTEGER,
    area DOUBLE PRECISION,
    perimeter DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_statie_charge_count CHECK (charge_count > 0),
    CONSTRAINT chk_statie_area CHECK (area IS NULL OR area >= 0),
    CONSTRAINT chk_statie_perimeter CHECK (perimeter IS NULL OR perimeter >= 0)
);

CREATE INDEX idx_statie_locatie ON statie_incarcare USING GIST (locatie);
CREATE INDEX idx_statie_h3_res8 ON statie_incarcare (h3_res8);
CREATE INDEX idx_statie_h3_res9 ON statie_incarcare (h3_res9);
CREATE INDEX idx_statie_taz_id ON statie_incarcare (taz_id);

CREATE TABLE date_temporale_statie (
    statie_id BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    busy INTEGER,
    volume NUMERIC(14,6),
    e_price NUMERIC(10,4),
    s_price NUMERIC(10,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (statie_id, timestamp),
    CONSTRAINT fk_temporal_statie
        FOREIGN KEY (statie_id)
        REFERENCES statie_incarcare (statie_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_metrics_non_negative CHECK (
        (busy IS NULL OR busy >= 0) AND
        (volume IS NULL OR volume >= 0) AND
        (e_price IS NULL OR e_price >= 0) AND
        (s_price IS NULL OR s_price >= 0)
    )
) PARTITION BY RANGE (timestamp);

-- Partiția implicită permite inserarea oricărui timestamp.
-- API-ul creează automat partiții lunare înainte de inserare.
CREATE TABLE date_temporale_statie_default
    PARTITION OF date_temporale_statie DEFAULT;

CREATE INDEX idx_temporal_station_time
    ON date_temporale_statie (statie_id, timestamp DESC);
CREATE INDEX idx_temporal_timestamp
    ON date_temporale_statie (timestamp DESC);
