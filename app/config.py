from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Theft Detection API"
    app_version: str = "0.1.0"
    debug: bool = True

    transactional_database_url: str = "mysql+pymysql://root:root@localhost:3306/sesamedb"
    vector_database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/theft_detection_gallery"
    database_url: str | None = None

    base_dir: Path = Path(__file__).resolve().parents[1]
    repo_root: Path = Path(__file__).resolve().parents[2]
    entry_script_path: Path = Path(__file__).resolve().parents[1] / "DetectEntry.py"
    kiosk_script_path: Path = Path(__file__).resolve().parents[1] / "DetectKiosk.py"
    python_bin: str = "python"
    video_storage_dir: Path = Path(__file__).resolve().parents[2] / "session"
    transactional_table_prefix: str = "tds_"
    location_table_name: str = "location"
    location_id_column: str = "id"
    location_name_column: str = "name"

    model_config = SettingsConfigDict(
        env_prefix="THEFT_API_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


if settings.database_url:
    # Backward compatibility for older env files that only define one DB URL.
    settings.transactional_database_url = settings.database_url
