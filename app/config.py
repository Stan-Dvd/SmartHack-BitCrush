from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/urbanev"
    h3_res8: int = 8
    h3_res9: int = 9
    pump_rating_kw: float = 50.0
    snapshot_minutes: int = 5
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

settings = Settings()
