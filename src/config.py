"""Constante globale: căi, raze, praguri, seed. Toate scripturile le citesc de aici."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- căi ---
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"

URBANEV_RAW = RAW / "UrbanEVDataset" / "20220901-20230228_station-raw"
URBANEV_CLEANED = RAW / "UrbanEVDataset" / "20220901-20230228_zone-cleaned-aggregated"
URBANEV_SUPP = RAW / "UrbanEVSupplemental"

STATION_INFO = URBANEV_RAW / "station_information.csv"
PILE_POWER = URBANEV_RAW / "pile_rated_power.csv"
STATION_DISTANCE = URBANEV_RAW / "station_distance.csv"
CHARGE_5MIN_DIR = URBANEV_RAW / "charge_5min"
POI_FILE = URBANEV_SUPP / "20221201-shenzhen-poi.csv"

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/CHN/chn_ppp_2020_UNadj.tif"
)
WORLDPOP_DIR = RAW / "worldpop"
WORLDPOP_CROP = WORLDPOP_DIR / "shenzhen_ppp_2020_UNadj.tif"

TARGET_FILE = PROCESSED / "target.parquet"
STATION_STATS_FILE = PROCESSED / "station_stats.parquet"
STATION_FEATURES_FILE = PROCESSED / "station_features.parquet"
TRAIN_TABLE = PROCESSED / "train_table.parquet"
SPLIT_FILE = PROCESSED / "split.json"

MODEL_FILE = MODELS / "lgbm.txt"
PARAMS_FILE = MODELS / "params.json"
METRICS_FILE = REPORTS / "metrics.json"
METRICS_TABLE = REPORTS / "metrics_table.md"
DATA_QUALITY_FILE = REPORTS / "data_quality.json"
CV_RESULTS_FILE = REPORTS / "cv_results.csv"
PREDICTIONS_FILE = REPORTS / "test_predictions.parquet"
ANALYSIS_FILE = REPORTS / "analysis.json"

# --- zona Shenzhen (WGS84) ---
BBOX = dict(lon_min=113.70, lon_max=114.70, lat_min=22.35, lat_max=22.90)

# --- ținta ---
COVERAGE_MIN = 0.70  # stațiile cu acoperire sub prag sunt eliminate
MIN_CELL_OBS = 10  # celulele (stație, oră, is_weekend) cu mai puține intervale sunt eliminate

# --- feature-uri pe raze (metri) ---
EARTH_RADIUS_M = 6_371_008.8
STATION_RADII_M = (500, 1000, 3000)
COMPETITOR_PILES_RADIUS_M = 1000
POP_RADIUS_M = 1000
POI_RADIUS_M = 500
POI_CATEGORIES = {
    "food and beverage services": "poi_food_500m",
    "business and residential": "poi_business_res_500m",
    "lifestyle services": "poi_lifestyle_500m",
}

COMPETITION_FEATURES = [
    "dist_nearest_station_m",
    "n_stations_500m",
    "n_stations_1km",
    "n_stations_3km",
    "competitor_piles_1km",
]
POI_FEATURES = list(POI_CATEGORIES.values())
CONTEXT_FEATURES = ["pop_1km", *POI_FEATURES]
STATION_FEATURES = ["n_piles", "power_kw"]
TEMPORAL_FEATURES = ["hour", "is_weekend"]

FEATURES = [*COMPETITION_FEATURES, *CONTEXT_FEATURES, *STATION_FEATURES, *TEMPORAL_FEATURES]
TARGET = "y"
ID_COLS = ["station_id", "zone_id"]

# coloane care NU au voie să intre în model (verificat prin aserțiuni)
FORBIDDEN_FEATURES = {
    "station_id", "zone_id", "TAZID", "longitude", "latitude", "lon", "lat",
    "y", "busy", "idle", "occupancy", "sat_share", "coverage", "n_obs",
}

# --- split și antrenare ---
SEED = 42
TEST_FRAC = 0.20
N_FOLDS = 5

LGBM_BASE_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=1.0,
    seed=42,
    deterministic=True,
    force_col_wise=True,
    num_threads=8,
    verbose=-1,
)
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100

# constrângeri monotone (folosite doar cu `python -m src.train --monotone`)
MONOTONE = {"n_piles": 1}

# --- saturare ---
SAT_OCC = 0.9  # ocupare prezisă la ora de vârf peste prag => stație saturată (prezis)
SAT_SHARE_REAL = 0.10  # % intervale cu toate punctele ocupate peste prag => saturată (real)

N_JOBS = 12
