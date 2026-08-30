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
    dahua_output_scale: float = 0.5
    dahua_ffmpeg_threads: int = 2
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
    runpod_enqueue_timeout_seconds: int = 15
    runpod_status_timeout_seconds: int = 15
    runner_input_key_prefix: str = "runner_inputs"
    runpod_endpoint_id: str | None = None
    runpod_entry_endpoint_id: str | None = None
    runpod_kiosk_endpoint_id: str | None = None
    runpod_grouping_endpoint_id: str | None = None
    runpod_api_key: str | None = None
    runpod_webhook_base_url: str | None = None
    runpod_webhook_secret: str | None = None
    runpod_cost_per_second_usd: float = 0.0
    retrieval_poll_seconds: int = 10
    retrieval_max_global_workers: int = 1
    retrieval_max_per_location: int = 1
    retrieval_ffmpeg_timeout_seconds: int = 60
    retrieval_stale_seconds: int = 900
    trigger_frame_count: int = 20
    trigger_frame_gap: int = 12
    trigger_frame_fps: int = 25
    trigger_frame_crop_left_fraction: float = 0.40
    grouping_open_entry_stale_hours: int = 6
    grouping_open_entry_max_wait_minutes: int = 30
    grouping_carry_forward_buffer_minutes: int = 30
    grouping_window_grace_minutes: int = 20
    time_period_timezone: str = "Asia/Kuala_Lumpur"
    grouping_poll_seconds: int = 30
    grouping_max_global_workers: int = 1
    theft_confidence_poll_seconds: int = 30
    theft_confidence_max_global_workers: int = 1
    filter_long_stay_seconds: int = 300
    filter_low_purchase_quantity: int = 1
    filter_low_purchase_value: int = 1000
    filter_transaction_issue_short_period_seconds: int = 120
    filter_carry_score_threshold: float = 40
    filter_unusual_group_size: int = 2
    filter_unusual_group_size_min_history: int = 2
    filter_unusual_group_size_delta: int = 2
    analysis_poll_seconds: int = 10
    analysis_max_global_workers: int = 1
    analysis_cooldown_seconds: int = 10
    kiosk_analysis_poll_seconds: int = 10
    kiosk_analysis_max_global_workers: int = 1
    kiosk_analysis_cooldown_seconds: int = 10
    entrance_trigger_extra_before_seconds: int = 10
    entrance_trigger_extra_after_seconds: int = 40
    kiosk_transaction_extra_before_seconds: int = 10
    kiosk_transaction_extra_after_seconds: int = -10
    whitelist_qrentry_table_name: str = "phonenumber"
    whitelist_qrentry_id_column: str = "id"
    whitelist_qrentry_value_column: str = "id"
    whitelist_qrentry_label_column: str = "participantId"
    whitelist_qrentry_display_column: str = "participantId"
    whitelist_qrentry_create_column: str = "participantId"
    whitelist_entrylogs_table_name: str = "fingerprint"
    whitelist_entrylogs_id_column: str = "id"
    whitelist_entrylogs_value_column: str = "id"
    whitelist_entrylogs_label_column: str = "fingerprint"
    whitelist_entrylogs_display_column: str = "fingerprint"
    stripe_secret_key: str | None = None
    stripe_api_base_url: str = "https://api.stripe.com/v1"
    stripe_lookup_timeout_seconds: int = 12
    theft_transaction_table_name: str = "transaction"
    theft_transaction_status_column: str = "status"
    theft_transaction_status_value: str = "theft"
    thief_alert_table_name: str = "tds_thief_alert"
    thief_alert_checked_column: str = "checked"
    paid_transaction_table_name: str = "transaction"
    paid_transaction_detail_table_name: str = "transactionDetail"
    paid_transaction_location_id_column: str = "locationId"
    paid_transaction_status_column: str = "status"
    paid_transaction_status_value: str = "paid"
    paid_transaction_receipt_column: str = "receiptNumber"
    paid_transaction_total_amount_column: str = "totalAmount"
    paid_transaction_detail_transaction_id_column: str = "receiptNumber"
    paid_transaction_detail_quantity_column: str = "quantity"
    paid_transaction_detail_item_name_column: str = "name"
    gemini_api_key: str | None = None
    kiosk_gemini_model: str = "gemini-3-flash-preview"
    grouping_gemini_model: str = "gemini-3.5-flash-lite"
    grouping_gemini_frames_per_trigger: int = 4
    grouping_gemini_image_scale: float = 0.35
    grouping_gemini_max_images_per_request: int = 40
    kiosk_gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    kiosk_gemini_timeout_seconds: int = 180
    gemini_input_cost_per_1m_tokens_usd: float = 0.0
    gemini_output_cost_per_1m_tokens_usd: float = 0.0
    gemini_cached_input_cost_per_1m_tokens_usd: float = 0.0

    model_config = SettingsConfigDict(
        env_prefix="THEFT_API_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()


if settings.database_url:
    # Backward compatibility for older env files that only define one DB URL.
    settings.transactional_database_url = settings.database_url
