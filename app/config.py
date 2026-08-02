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
    ffmpeg_bin: str = "ffmpeg"
    video_storage_dir: Path = Path(__file__).resolve().parents[2] / "session"
    credential_secret: str = "change_me_please_use_a_long_random_secret"
    transactional_table_prefix: str = "tds_"
    location_table_name: str = "location"
    location_id_column: str = "id"
    location_name_column: str = "name"
    dahua_rtsp_port: int = 554
    dahua_playback_subtype: int = 0
    dahua_output_video_codec: str = "libx264"
    dahua_output_preset: str = "veryfast"
    dahua_output_crf: int = 23
    spaces_endpoint_url: str | None = None
    spaces_region: str = "sgp1"
    spaces_bucket: str | None = None
    spaces_access_key: str | None = None
    spaces_secret_key: str | None = None
    spaces_key_prefix: str = "tds"
    spaces_presign_ttl_seconds: int = 600
    spaces_public_read: bool = False
    spaces_public_base_url: str | None = None
    runner_base_url: str | None = None
    runner_timeout_seconds: int = 7200
    runner_input_key_prefix: str = "runner_inputs"
    runpod_endpoint_id: str | None = None
    runpod_entry_endpoint_id: str | None = None
    runpod_kiosk_endpoint_id: str | None = None
    runpod_api_key: str | None = None
    runpod_webhook_base_url: str | None = None
    runpod_webhook_secret: str | None = None
    retrieval_poll_seconds: int = 10
    retrieval_max_global_workers: int = 2
    retrieval_max_per_location: int = 1
    analysis_poll_seconds: int = 10
    analysis_max_global_workers: int = 1
    analysis_cooldown_seconds: int = 10
    whitelist_qrentry_table_name: str = "phonenumber"
    whitelist_qrentry_value_column: str = "participantId"
    whitelist_qrentry_label_column: str = "participantId"
    whitelist_qrentry_display_column: str = "participantId"
    whitelist_qrentry_create_column: str = "participantId"
    whitelist_entrylogs_table_name: str = "fingerprint"
    whitelist_entrylogs_value_column: str = "fingerprint"
    whitelist_entrylogs_label_column: str = "fingerprint"
    whitelist_entrylogs_display_column: str = "fingerprint"
    theft_transaction_table_name: str = "transaction"
    theft_transaction_status_column: str = "status"
    theft_transaction_status_value: str = "theft"
    paid_transaction_table_name: str = "transaction"
    paid_transaction_detail_table_name: str = "transactionDetail"
    paid_transaction_id_column: str = "id"
    paid_transaction_location_id_column: str = "locationId"
    paid_transaction_time_column: str = "createdAt"
    paid_transaction_status_column: str = "status"
    paid_transaction_status_value: str = "paid"
    paid_transaction_receipt_column: str = "receiptNumber"
    paid_transaction_total_amount_column: str = "totalAmount"
    paid_transaction_detail_transaction_id_column: str = "transactionId"
    paid_transaction_detail_quantity_column: str = "quantity"
    paid_transaction_detail_item_name_column: str = "name"

    model_config = SettingsConfigDict(
        env_prefix="THEFT_API_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


if settings.database_url:
    # Backward compatibility for older env files that only define one DB URL.
    settings.transactional_database_url = settings.database_url
