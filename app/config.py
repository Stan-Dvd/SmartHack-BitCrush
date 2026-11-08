from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5433/urbanev"
    h3_res8: int = 8
    h3_res9: int = 9
    # Fallback pentru volumul observațiilor când stația nu are putere; predicțiile folosesc power_kw al stației.
    pump_rating_kw: float = 50.0
    snapshot_minutes: int = 5
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    timezone: str = "Europe/Bucharest"

    # --- model ML ---
    ml_model_dir: Path = ROOT / "ml" / "models"
    ml_reports_dir: Path = ROOT / "ml" / "reports"
    # configurația de referință pentru o stație nouă (candidat sau celulă H3 de oportunitate);
    # modelul nu e cauzal în n_piles, deci locațiile se compară la aceeași configurație
    predictie_n_piles_default: int = 2
    predictie_power_kw_default: float = 50.0
    sat_occ: float = 0.9  # ocupare la vârf peste prag => saturată (SAT_OCC din ml/src/config.py)
    raza_impact_m: float = 3000.0  # raza maximă a feature-urilor de concurență

    # --- grila H3 de oportunități ---
    h3_grid_res: int = 8
    h3_grid_min_pop: float = 100.0  # celule care conțin un punct WorldPop 1 km cu cel puțin atâția locuitori
    # pop_1km e marcat suspect în județele unde WorldPop depășește de peste atâtea ori populația INS
    # (Călărași: 29,6x; populația e corectată la totalul județului, dar distribuția internă rămâne greșită)
    raport_pop_judet_suspect: float = 2.0

    # --- agent AI (Gemini, free tier: https://aistudio.google.com) ---
    # cheia se poate numi în .env GEMINI_API_KEY (recomandat), GOOGLE_API_KEY sau GOOGLE_MAPS_API_KEY
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_MAPS_API_KEY"))
    gemini_model: str = "gemini-3.6-flash"  # stabil și disponibil pe free tier
    # folosit dacă modelul principal e supraîncărcat (503) și după reîncercări; gol = fără rezervă
    gemini_model_rezerva: str | None = "gemini-3.5-flash-lite"
    agent_max_pasi: int = 8  # câte runde de apeluri de tool-uri are voie agentul

    # --- ipoteze economice implicite (ILUSTRATIVE; se pot suprascrie din UI / .env) ---
    econ_pret_kwh_ac: float = 1.9  # RON/kWh, stații AC (<= 22 kW)
    econ_pret_kwh_dc: float = 2.6  # RON/kWh, stații DC
    econ_cost_energie_kwh: float = 1.1  # RON/kWh, cost de achiziție
    econ_capex_ac_per_punct: float = 20_000  # RON / punct AC
    econ_capex_dc50_per_punct: float = 160_000  # RON / punct DC <= 60 kW
    econ_capex_dc150_per_punct: float = 280_000  # RON / punct DC > 60 kW
    econ_opex_procent: float = 6.0  # % din capex pe an
    # kwh prezis = puncte ocupate x putere nominală (plafon); energia livrată real e mai mică
    econ_factor_putere: float = 0.6
    # cererea prezisă e la nivel Shenzhen; 1.0 = fără ajustare (se poate scădea pentru piața din România)
    econ_factor_cerere: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


settings = Settings()
