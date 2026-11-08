-- UrbanEV v3: baza de date construită în jurul modelului ML de cerere (LightGBM, ml/models/lgbm.txt).
--
-- Două utilizări:
--   1. utilizarea viitoare a stațiilor existente: profil orar prezis (oră x zi lucrătoare/weekend),
--      proiectat pe calendar și calibrat cu observațiile reale (fn_prognoza_statie);
--   2. performanța unei stații noi în orice punct: feature-urile se calculează live (fn_features_punct),
--      inclusiv efectul asupra stațiilor vecine (canibalizare), plus grila H3 de oportunități.
--
-- ATENȚIE: comenzile DROP șterg datele existente.

CREATE EXTENSION IF NOT EXISTS postgis;

DROP VIEW IF EXISTS v_calibrare_statie CASCADE;
DROP VIEW IF EXISTS v_statie_predictie CASCADE;
DROP FUNCTION IF EXISTS fn_prognoza_statie CASCADE;
DROP FUNCTION IF EXISTS fn_features_punct CASCADE;
DROP FUNCTION IF EXISTS fn_model_activ CASCADE;
DROP TABLE IF EXISTS predictie_sumar CASCADE;
DROP TABLE IF EXISTS predictie_profil CASCADE;
DROP TABLE IF EXISTS features_locatie CASCADE;
DROP TABLE IF EXISTS model_ml CASCADE;
DROP TABLE IF EXISTS date_temporale_statie CASCADE;
DROP TABLE IF EXISTS locatie_candidat CASCADE;
DROP TABLE IF EXISTS statie_incarcare CASCADE;
DROP TABLE IF EXISTS poi_osm CASCADE;
DROP TABLE IF EXISTS populatie_grid CASCADE;
DROP TABLE IF EXISTS localitate CASCADE;
DROP TABLE IF EXISTS judet_bucata CASCADE;
DROP TABLE IF EXISTS judet CASCADE;
DROP TABLE IF EXISTS celule_h3 CASCADE;

-- =====================================================================
-- Celule H3 (rezoluțiile 8 și 9)
-- =====================================================================
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

-- =====================================================================
-- Stații existente (România, statii_electrice_ML_cu_localitate.csv)
-- =====================================================================
CREATE TABLE statie_incarcare (
    statie_id BIGINT PRIMARY KEY,                 -- indexul (1-based) din CSV
    cod TEXT NOT NULL UNIQUE,                     -- RO0001 ... (același ID ca în romania_ev_5min.csv)
    locatie GEOGRAPHY(POINT, 4326) NOT NULL,
    longitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_X(locatie::geometry)) STORED,
    latitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_Y(locatie::geometry)) STORED,
    judet TEXT,
    localitate TEXT,
    operator TEXT,
    operator_implicit BOOLEAN NOT NULL DEFAULT FALSE,  -- operatorul lipsea din sursă și a fost atribuit la import
    n_piles INTEGER NOT NULL DEFAULT 1,          -- puncte de încărcare (feature n_piles)
    power_kw DOUBLE PRECISION NOT NULL,          -- putere medie per punct (feature power_kw)
    power_min_kw DOUBLE PRECISION,
    power_max_kw DOUBLE PRECISION,
    is_open_24_7 BOOLEAN NOT NULL DEFAULT FALSE,
    activa BOOLEAN NOT NULL DEFAULT TRUE,         -- doar stațiile active intră în concurență
    sursa TEXT NOT NULL DEFAULT 'statii_electrice_ML_cu_localitate.csv',
    h3_res8 VARCHAR(16) NOT NULL REFERENCES celule_h3 (h3_index),
    h3_res9 VARCHAR(16) REFERENCES celule_h3 (h3_index),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_statie_n_piles CHECK (n_piles > 0),
    CONSTRAINT chk_statie_power CHECK (power_kw > 0)
);
CREATE INDEX idx_statie_locatie ON statie_incarcare USING GIST (locatie);
CREATE INDEX idx_statie_h3_res8 ON statie_incarcare (h3_res8);
CREATE INDEX idx_statie_h3_res9 ON statie_incarcare (h3_res9);
CREATE INDEX idx_statie_judet ON statie_incarcare (judet);

-- =====================================================================
-- Locații candidat (stații propuse, evaluate cu modelul)
-- =====================================================================
CREATE TABLE locatie_candidat (
    candidat_id BIGSERIAL PRIMARY KEY,
    nume TEXT,
    locatie GEOGRAPHY(POINT, 4326) NOT NULL,
    longitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_X(locatie::geometry)) STORED,
    latitude DOUBLE PRECISION GENERATED ALWAYS AS (ST_Y(locatie::geometry)) STORED,
    judet TEXT,
    n_piles INTEGER NOT NULL,
    power_kw DOUBLE PRECISION NOT NULL,
    h3_res8 VARCHAR(16) REFERENCES celule_h3 (h3_index),
    h3_res9 VARCHAR(16) REFERENCES celule_h3 (h3_index),
    status TEXT NOT NULL DEFAULT 'evaluat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_candidat_n_piles CHECK (n_piles > 0),
    CONSTRAINT chk_candidat_power CHECK (power_kw > 0),
    CONSTRAINT chk_candidat_status CHECK (status IN ('propus', 'evaluat', 'aprobat', 'respins'))
);
CREATE INDEX idx_candidat_locatie ON locatie_candidat USING GIST (locatie);

-- =====================================================================
-- Straturi de referință pentru feature-uri
-- =====================================================================
-- POI din OpenStreetMap, aceleași categorii ca în feature-urile modelului.
-- Geometria e păstrată în EPSG:3035 (LAEA Europa), ca în enrich_romania_features.py:
-- distanța de 500 m se măsoară până la geometria POI-ului (punct sau poligon).
CREATE TABLE poi_osm (
    poi_id BIGSERIAL PRIMARY KEY,
    osm_id BIGINT,
    osm_type TEXT,
    categorie TEXT NOT NULL,
    geom GEOMETRY(GEOMETRY, 3035) NOT NULL,
    CONSTRAINT chk_poi_categorie CHECK (categorie IN ('food', 'business_res', 'lifestyle'))
);
CREATE INDEX idx_poi_geom ON poi_osm USING GIST (geom);

-- Județe: contur OSM (admin_level 4) + populația oficială INS (POP107D). raport_worldpop_ins compară
-- populația WorldPop din județ cu cea oficială; un raport mare indică populație WorldPop mutată greșit
-- (de ex. Călărași: milioane de persoane în plus), deci pop_1km nu e de încredere acolo.
CREATE TABLE judet (
    judet_id SERIAL PRIMARY KEY,
    nume TEXT NOT NULL UNIQUE,
    nume_ascii TEXT NOT NULL UNIQUE,
    geom GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    populatie_ins BIGINT,
    an_ins SMALLINT,
    populatie_worldpop DOUBLE PRECISION,
    raport_worldpop_ins DOUBLE PRECISION
);
CREATE INDEX idx_judet_geom ON judet USING GIST (geom);

-- Contururile subdivizate (ST_Subdivide), pentru testul rapid punct-în-județ.
CREATE TABLE judet_bucata (
    judet_id INTEGER NOT NULL REFERENCES judet (judet_id) ON DELETE CASCADE,
    geom GEOMETRY(POLYGON, 4326) NOT NULL
);
CREATE INDEX idx_judet_bucata_geom ON judet_bucata USING GIST (geom);

-- Localități (municipii, orașe, comune): OSM admin_level 8. Folosite de agent pentru a rezolva
-- o zonă după nume („Cluj-Napoca”); judet_id = județul care conține un punct interior al conturului.
CREATE TABLE localitate (
    localitate_id SERIAL PRIMARY KEY,
    nume TEXT NOT NULL,
    nume_ascii TEXT NOT NULL,
    judet_id INTEGER REFERENCES judet (judet_id) ON DELETE CASCADE,
    geom GEOMETRY(MULTIPOLYGON, 4326) NOT NULL
);
CREATE INDEX idx_localitate_geom ON localitate USING GIST (geom);
CREATE INDEX idx_localitate_nume ON localitate (nume_ascii);

-- WorldPop 2020, grila de 1 km (persoane per celulă), centrul celulei.
-- Fișierul WorldPop are populația mutată greșit (Călărași ~8,6 mil. față de 0,29 mil. INS; restul județelor
-- ~0,5 din INS). populatie_ajustata = populatie x populatie_ins / populatie_worldpop a județului, astfel încât
-- suma pe județ să fie cea oficială. Feature-ul pop_1km folosește valoarea ajustată (brută în afara județelor).
CREATE TABLE populatie_grid (
    grid_id BIGSERIAL PRIMARY KEY,
    locatie GEOGRAPHY(POINT, 4326) NOT NULL,
    populatie REAL NOT NULL,
    populatie_ajustata REAL,
    judet_id INTEGER REFERENCES judet (judet_id) ON DELETE SET NULL,
    sursa TEXT NOT NULL DEFAULT 'WorldPop 2020 1km UNadj (ppp_ROU_2020_1km_Aggregated_UNadj)',
    CONSTRAINT chk_populatie CHECK (populatie >= 0)
);
CREATE INDEX idx_populatie_locatie ON populatie_grid USING GIST (locatie);

-- =====================================================================
-- Observații la 5 minute (colectate prin API; partiționate lunar)
-- =====================================================================
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
        FOREIGN KEY (statie_id) REFERENCES statie_incarcare (statie_id) ON DELETE CASCADE,
    CONSTRAINT chk_metrics_non_negative CHECK (
        (busy IS NULL OR busy >= 0) AND
        (volume IS NULL OR volume >= 0) AND
        (e_price IS NULL OR e_price >= 0) AND
        (s_price IS NULL OR s_price >= 0)
    )
) PARTITION BY RANGE (timestamp);

CREATE TABLE date_temporale_statie_default PARTITION OF date_temporale_statie DEFAULT;
CREATE INDEX idx_temporal_station_time ON date_temporale_statie (statie_id, timestamp DESC);
CREATE INDEX idx_temporal_timestamp ON date_temporale_statie (timestamp DESC);

-- =====================================================================
-- Registrul modelelor ML
-- =====================================================================
CREATE TABLE model_ml (
    model_id SERIAL PRIMARY KEY,
    nume TEXT NOT NULL,
    versiune TEXT NOT NULL,
    cale_fisier TEXT NOT NULL,           -- relativ la rădăcina proiectului
    features JSONB NOT NULL,             -- ordinea exactă a coloanelor de intrare
    target_mode TEXT NOT NULL,           -- 'occ' (rata de ocupare x n_piles) sau 'y'
    params JSONB NOT NULL,
    metrici JSONB,
    antrenat_pe TEXT NOT NULL,
    limitari TEXT,
    activ BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_versiune UNIQUE (nume, versiune),
    CONSTRAINT chk_target_mode CHECK (target_mode IN ('occ', 'y'))
);
CREATE UNIQUE INDEX uq_model_activ ON model_ml (activ) WHERE activ;

-- =====================================================================
-- Feature-uri per locație (stație, candidat sau centrul unei celule H3)
-- Numele coloanelor sunt identice cu ml/src/config.py (FEATURES).
-- =====================================================================
CREATE TABLE features_locatie (
    feature_id BIGSERIAL PRIMARY KEY,
    tip TEXT NOT NULL,
    statie_id BIGINT UNIQUE REFERENCES statie_incarcare (statie_id) ON DELETE CASCADE,
    candidat_id BIGINT UNIQUE REFERENCES locatie_candidat (candidat_id) ON DELETE CASCADE,
    h3_index VARCHAR(16) UNIQUE REFERENCES celule_h3 (h3_index) ON DELETE CASCADE,

    dist_nearest_station_m DOUBLE PRECISION,
    n_stations_500m INTEGER NOT NULL,
    n_stations_1km INTEGER NOT NULL,
    n_stations_3km INTEGER NOT NULL,
    competitor_piles_1km INTEGER NOT NULL,
    pop_1km DOUBLE PRECISION NOT NULL,
    poi_food_500m INTEGER NOT NULL,
    poi_business_res_500m INTEGER NOT NULL,
    poi_lifestyle_500m INTEGER NOT NULL,
    n_piles INTEGER NOT NULL,
    power_kw DOUBLE PRECISION NOT NULL,

    judet TEXT,                                   -- județul (din conturul OSM) în care se află punctul
    raport_pop_judet DOUBLE PRECISION,            -- judet.raport_worldpop_ins
    -- pop_1km nu e de încredere: județ cu WorldPop mult peste populația INS (distribuția internă e greșită)
    pop_suspect BOOLEAN NOT NULL DEFAULT FALSE,
    calculat_la TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_features_tip CHECK (tip IN ('statie', 'candidat', 'h3')),
    CONSTRAINT chk_features_subiect CHECK (
        (tip = 'statie'   AND statie_id IS NOT NULL AND candidat_id IS NULL AND h3_index IS NULL) OR
        (tip = 'candidat' AND candidat_id IS NOT NULL AND statie_id IS NULL AND h3_index IS NULL) OR
        (tip = 'h3'       AND h3_index IS NOT NULL AND statie_id IS NULL AND candidat_id IS NULL)
    )
);
CREATE INDEX idx_features_tip ON features_locatie (tip);

-- =====================================================================
-- Predicții
-- =====================================================================
-- Profilul orar: 48 de rânduri (24 ore x is_weekend) per stație sau candidat.
CREATE TABLE predictie_profil (
    feature_id BIGINT NOT NULL REFERENCES features_locatie (feature_id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES model_ml (model_id) ON DELETE CASCADE,
    hour SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    busy_pred DOUBLE PRECISION NOT NULL,      -- puncte ocupate în medie în acea oră
    ocupare_pred DOUBLE PRECISION NOT NULL,   -- busy_pred / n_piles
    kwh_pred DOUBLE PRECISION NOT NULL,       -- busy_pred x power_kw x 1 h (plafon superior)
    PRIMARY KEY (feature_id, model_id, hour, is_weekend),
    CONSTRAINT chk_profil_hour CHECK (hour BETWEEN 0 AND 23)
);

-- Sumarul pe locație (stocat pentru toate tipurile, inclusiv celulele H3).
CREATE TABLE predictie_sumar (
    feature_id BIGINT NOT NULL REFERENCES features_locatie (feature_id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES model_ml (model_id) ON DELETE CASCADE,
    ore_incarcare_zi DOUBLE PRECISION NOT NULL,   -- (5·Σ zi lucrătoare + 2·Σ weekend) / 7
    kwh_zi DOUBLE PRECISION NOT NULL,
    ocupare_medie DOUBLE PRECISION NOT NULL,
    ocupare_varf DOUBLE PRECISION NOT NULL,
    ora_varf SMALLINT NOT NULL,
    weekend_varf BOOLEAN NOT NULL,
    saturata_prezis BOOLEAN NOT NULL,             -- ocupare_varf >= 0.9 (SAT_OCC din ml/src/config.py)
    percentila_nationala DOUBLE PRECISION,        -- percent_rank după ocupare_medie, în cadrul aceluiași tip
    calculat_la TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (feature_id, model_id)
);
CREATE INDEX idx_sumar_model_ore ON predictie_sumar (model_id, ore_incarcare_zi DESC);

-- =====================================================================
-- Funcții
-- =====================================================================
CREATE FUNCTION fn_model_activ() RETURNS INTEGER
LANGUAGE sql STABLE AS $$
    SELECT model_id FROM model_ml WHERE activ
$$;

-- Feature-urile de locație pentru un punct oarecare. Definițiile urmează ml/src/config.py
-- și scripturile care au generat setul românesc (D:\hack\generate_ev_csv.py, enrich_romania_features.py):
--   * concurența: doar stațiile active, fără stația însăși; distanțe pe sferă (≈ haversine), inclusiv la limită;
--   * pop_1km: Σ celulelor WorldPop (ajustate la populația INS pe județ) cu centrul la cel mult 1 km;
--   * POI: numărul de obiecte pe categorie la cel mult 500 m (EPSG:3035).
-- p_extra_locatii / p_extra_piles adaugă stații ipotetice în concurență (canibalizare, planificarea unei rețele);
-- cele două array-uri au aceeași lungime și ordine.
-- pop_max_celula: cea mai mare celulă WorldPop folosită; judet / raport_pop_judet: județul punctului
-- și raportul WorldPop/INS al județului (ambele pentru marcarea valorilor suspecte, nu intră în model).
CREATE FUNCTION fn_features_punct(
    p_lon DOUBLE PRECISION,
    p_lat DOUBLE PRECISION,
    p_exclude_statie_id BIGINT DEFAULT NULL,
    p_extra_locatii GEOGRAPHY[] DEFAULT NULL,
    p_extra_piles INTEGER[] DEFAULT NULL
)
RETURNS TABLE (
    dist_nearest_station_m DOUBLE PRECISION,
    n_stations_500m INTEGER,
    n_stations_1km INTEGER,
    n_stations_3km INTEGER,
    competitor_piles_1km INTEGER,
    pop_1km DOUBLE PRECISION,
    poi_food_500m INTEGER,
    poi_business_res_500m INTEGER,
    poi_lifestyle_500m INTEGER,
    pop_max_celula DOUBLE PRECISION,
    judet TEXT,
    raport_pop_judet DOUBLE PRECISION
)
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    WITH p AS (
        SELECT ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography AS g,
               ST_Transform(ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326), 3035) AS g3035
    ),
    extra AS (
        SELECT e.g, e.n_piles FROM unnest(p_extra_locatii, p_extra_piles) AS e(g, n_piles)
        WHERE e.g IS NOT NULL
    ),
    concurenti AS (
        SELECT ST_Distance(s.locatie, p.g, false) AS d, s.n_piles
        FROM statie_incarcare s, p
        WHERE s.activa
          AND s.statie_id IS DISTINCT FROM p_exclude_statie_id
          AND ST_DWithin(s.locatie, p.g, 3000, false)
        UNION ALL
        SELECT ST_Distance(e.g, p.g, false), e.n_piles
        FROM extra e, p
        WHERE ST_DWithin(e.g, p.g, 3000, false)
    ),
    cel_mai_apropiat AS (
        SELECT MIN(d) AS d FROM (
            (SELECT ST_Distance(s.locatie, p.g, false) AS d
             FROM statie_incarcare s, p
             WHERE s.activa AND s.statie_id IS DISTINCT FROM p_exclude_statie_id
             ORDER BY s.locatie <-> p.g
             LIMIT 5)
            UNION ALL
            SELECT ST_Distance(e.g, p.g, false) FROM extra e, p
        ) x
    ),
    pop AS (
        SELECT COALESCE(SUM(COALESCE(g.populatie_ajustata, g.populatie)), 0)::double precision AS total,
               COALESCE(MAX(COALESCE(g.populatie_ajustata, g.populatie)), 0)::double precision AS maxim
        FROM populatie_grid g, p
        WHERE ST_DWithin(g.locatie, p.g, 1000, false)
    ),
    poi AS (
        SELECT COUNT(*) FILTER (WHERE o.categorie = 'food')::int AS food,
               COUNT(*) FILTER (WHERE o.categorie = 'business_res')::int AS business_res,
               COUNT(*) FILTER (WHERE o.categorie = 'lifestyle')::int AS lifestyle
        FROM poi_osm o, p
        WHERE ST_DWithin(o.geom, p.g3035, 500)
    )
    SELECT
        (SELECT d FROM cel_mai_apropiat),
        (SELECT COUNT(*) FILTER (WHERE d <= 500) FROM concurenti)::int,
        (SELECT COUNT(*) FILTER (WHERE d <= 1000) FROM concurenti)::int,
        (SELECT COUNT(*) FROM concurenti)::int,
        (SELECT COALESCE(SUM(n_piles) FILTER (WHERE d <= 1000), 0) FROM concurenti)::int,
        pop.total, poi.food, poi.business_res, poi.lifestyle, pop.maxim, j.nume, j.raport_worldpop_ins
    FROM pop, poi
    LEFT JOIN LATERAL (
        SELECT jj.nume, jj.raport_worldpop_ins
        FROM judet_bucata jb JOIN judet jj USING (judet_id)
        WHERE ST_Intersects(jb.geom, ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326))
        LIMIT 1
    ) j ON TRUE
$$;

-- Prognoza orară pentru o stație existentă, pe intervalul [p_de_la, p_pana_la).
-- Ora și tipul de zi se iau în ora locală (p_tz). Calibrarea: factor = medie busy observată /
-- medie busy prezisă pe aceleași intervale din ultimele p_zile_calibrare zile, limitat la [0.25, 4];
-- se aplică doar dacă există observații în cel puțin p_min_zile zile distincte.
CREATE FUNCTION fn_prognoza_statie(
    p_statie_id BIGINT,
    p_de_la TIMESTAMPTZ,
    p_pana_la TIMESTAMPTZ,
    p_calibrat BOOLEAN DEFAULT TRUE,
    p_tz TEXT DEFAULT 'Europe/Bucharest',
    p_zile_calibrare INTEGER DEFAULT 28,
    p_min_zile INTEGER DEFAULT 7
)
RETURNS TABLE (
    ora TIMESTAMPTZ,
    ora_locala TIMESTAMP,
    hour SMALLINT,
    is_weekend BOOLEAN,
    busy_model DOUBLE PRECISION,
    busy_prognoza DOUBLE PRECISION,
    ocupare_prognoza DOUBLE PRECISION,
    kwh_prognoza DOUBLE PRECISION,
    factor_calibrare DOUBLE PRECISION,
    are_calibrare BOOLEAN
)
LANGUAGE sql STABLE AS $$
    WITH f AS (
        SELECT fl.feature_id, fl.n_piles, fl.power_kw
        FROM features_locatie fl
        WHERE fl.statie_id = p_statie_id
    ),
    prof AS (
        SELECT pp.hour, pp.is_weekend, pp.busy_pred, pp.ocupare_pred, pp.kwh_pred
        FROM predictie_profil pp JOIN f USING (feature_id)
        WHERE pp.model_id = fn_model_activ()
    ),
    obs AS (
        SELECT d.busy,
               EXTRACT(hour FROM d.timestamp AT TIME ZONE p_tz)::smallint AS h,
               EXTRACT(isodow FROM d.timestamp AT TIME ZONE p_tz) >= 6 AS w,
               (d.timestamp AT TIME ZONE p_tz)::date AS zi
        FROM date_temporale_statie d
        WHERE d.statie_id = p_statie_id
          AND d.busy IS NOT NULL
          AND d.timestamp >= NOW() - make_interval(days => p_zile_calibrare)
    ),
    cal AS (
        SELECT COUNT(DISTINCT obs.zi) AS zile,
               AVG(obs.busy)::double precision AS obs_medie,
               AVG(prof.busy_pred) AS pred_medie
        FROM obs JOIN prof ON prof.hour = obs.h AND prof.is_weekend = obs.w
    ),
    factor AS (
        SELECT CASE WHEN p_calibrat AND cal.zile >= p_min_zile AND cal.pred_medie > 0
                    THEN LEAST(4.0, GREATEST(0.25, cal.obs_medie / cal.pred_medie))
                    ELSE 1.0 END AS k,
               (p_calibrat AND cal.zile >= p_min_zile AND cal.pred_medie > 0) AS ok
        FROM cal
    ),
    ore AS (
        SELECT t AS ora, (t AT TIME ZONE p_tz) AS ora_locala
        FROM generate_series(date_trunc('hour', p_de_la), p_pana_la - interval '1 microsecond',
                             interval '1 hour') AS t
    )
    SELECT ore.ora, ore.ora_locala,
           EXTRACT(hour FROM ore.ora_locala)::smallint,
           EXTRACT(isodow FROM ore.ora_locala) >= 6,
           prof.busy_pred,
           LEAST(prof.busy_pred * factor.k, f.n_piles),
           LEAST(prof.busy_pred * factor.k, f.n_piles) / f.n_piles,
           LEAST(prof.busy_pred * factor.k, f.n_piles) * f.power_kw,
           factor.k, factor.ok
    FROM ore
    CROSS JOIN f
    JOIN prof ON prof.hour = EXTRACT(hour FROM ore.ora_locala)
             AND prof.is_weekend = (EXTRACT(isodow FROM ore.ora_locala) >= 6)
    CROSS JOIN factor
    ORDER BY ore.ora
$$;

-- =====================================================================
-- View-uri
-- =====================================================================
-- Stațiile cu feature-urile și sumarul predicției pentru modelul activ.
CREATE VIEW v_statie_predictie AS
SELECT s.statie_id, s.cod, s.longitude, s.latitude, s.judet, s.localitate, s.operator,
       s.n_piles, s.power_kw, s.power_min_kw, s.power_max_kw, s.is_open_24_7, s.activa,
       s.h3_res8, s.h3_res9,
       f.feature_id, f.dist_nearest_station_m, f.n_stations_500m, f.n_stations_1km, f.n_stations_3km,
       f.competitor_piles_1km, f.pop_1km, f.poi_food_500m, f.poi_business_res_500m,
       f.poi_lifestyle_500m, f.pop_suspect, f.calculat_la AS features_calculat_la,
       ps.model_id, ps.ore_incarcare_zi, ps.kwh_zi, ps.ocupare_medie, ps.ocupare_varf, ps.ora_varf,
       ps.weekend_varf, ps.saturata_prezis, ps.percentila_nationala, s.operator_implicit
FROM statie_incarcare s
LEFT JOIN features_locatie f ON f.statie_id = s.statie_id
LEFT JOIN predictie_sumar ps ON ps.feature_id = f.feature_id AND ps.model_id = fn_model_activ();

-- Observat vs prezis, per stație × oră × tip de zi (ora locală București), pentru monitorizarea modelului.
CREATE VIEW v_calibrare_statie AS
WITH obs AS (
    SELECT d.statie_id,
           EXTRACT(hour FROM d.timestamp AT TIME ZONE 'Europe/Bucharest')::smallint AS hour,
           EXTRACT(isodow FROM d.timestamp AT TIME ZONE 'Europe/Bucharest') >= 6 AS is_weekend,
           AVG(d.busy)::double precision AS busy_observat,
           COUNT(*) AS n_obs,
           MIN(d.timestamp) AS prima_obs,
           MAX(d.timestamp) AS ultima_obs
    FROM date_temporale_statie d
    WHERE d.busy IS NOT NULL
    GROUP BY 1, 2, 3
)
SELECT obs.statie_id, obs.hour, obs.is_weekend, obs.busy_observat, obs.n_obs,
       pp.busy_pred, obs.busy_observat - pp.busy_pred AS eroare,
       obs.prima_obs, obs.ultima_obs
FROM obs
JOIN features_locatie f ON f.statie_id = obs.statie_id
JOIN predictie_profil pp ON pp.feature_id = f.feature_id
                        AND pp.model_id = fn_model_activ()
                        AND pp.hour = obs.hour AND pp.is_weekend = obs.is_weekend;
