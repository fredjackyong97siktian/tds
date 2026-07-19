from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str


class CctvBase(BaseModel):
    location_id: int
    section: str
    stream_name: str | None = None
    recorder_channel: str | None = None
    delayed_seconds: int = 0


class CctvCreate(CctvBase):
    pass


class CctvUpdate(CctvBase):
    pass


class CctvResponse(CctvBase):
    id: int
    created_at: datetime
    updated_at: datetime


class LocationOption(BaseModel):
    id: int
    name: str
    dahua_host: str | None = None
    dahua_username: str | None = None
    dahua_password: str | None = None
    rtsp_port: int | None = None
    notes: str | None = None
    has_endpoint_config: bool = False
    has_password_config: bool = False


class LocationEndpointUpsert(BaseModel):
    dahua_host: str
    dahua_username: str
    dahua_password: str | None = None
    rtsp_port: int = Field(default=554, ge=1, le=65535)
    notes: str | None = None


class WhitelistEntryCreate(BaseModel):
    method: str = Field(pattern="^(qrentry|entrylogs)$")
    entry_id: str
    status: str = Field(default="active", pattern="^(active|inactive)$")


class WhitelistEntryResponse(BaseModel):
    id: int
    method: str
    entry_id: str
    status: str
    resolved_value: str | None = None
    created_at: datetime
    updated_at: datetime


class WhitelistSourceOption(BaseModel):
    value: str
    label: str
    secondary_label: str | None = None
    method: str | None = None


class PhoneNumberCreate(BaseModel):
    phone_number: str


class PhoneNumberResponse(BaseModel):
    value: str
    label: str
    secondary_label: str | None = None


class TriggerCreate(BaseModel):
    location_id: int
    aqara_event_id: str | None = None
    trigger_source: str = "aqara"
    trigger_time: datetime
    raw_payload: dict[str, Any] | None = None


class TriggerResponse(BaseModel):
    id: int
    location_id: int
    status: str
    trigger_time: datetime


class TriggerListItem(TriggerResponse):
    aqara_event_id: str | None = None
    trigger_source: str
    entry_source_type: str
    entry_match_status: str
    whitelist_hit: bool = False
    issue_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class SessionCreate(BaseModel):
    entry_trigger_id: int
    location_id: int
    exit_trigger_id: int | None = None
    start_time: datetime | None = None


class SessionResponse(BaseModel):
    id: int
    entry_trigger_id: int
    exit_trigger_id: int | None = None
    location_id: int
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_item_brought: int = 0
    actual_items_brought: int = 0
    transaction_total_items: int = 0
    total_customer: int = 0


class SessionListItem(SessionResponse):
    linked_customer_count: int = 0
    linked_video_count: int = 0
    created_at: datetime
    updated_at: datetime


class SessionCustomerCreate(BaseModel):
    person_id: int
    enter_time: datetime | None = None
    kiosk_start_time: datetime | None = None
    leave_time: datetime | None = None
    match_status: str = "tracked"


class VideoAssetCreate(BaseModel):
    section: str = Field(pattern="^(entrance|kiosk)$")
    video_url: str
    file_path: str | None = None
    sequence_no: int | None = None
    captured_start_time: datetime | None = None
    captured_end_time: datetime | None = None
    retrieved_at: datetime | None = None
    analyzed_at: datetime | None = None
    retention_until: datetime | None = None
    status: str = Field(default="not_retrieved", pattern="^(not_retrieved|retrieving|ready|processing|processed|deleted|issue)$")
    trigger_id: int | None = None
    link_section: str | None = Field(default=None, pattern="^(entrance|kiosk)$")
    link_sequence_no: int | None = None
    clip_start_time: datetime | None = None
    clip_end_time: datetime | None = None
    is_primary: bool = False
    metadata: dict[str, Any] | None = None


class TransactionCreate(BaseModel):
    receipt_number: str
    transaction_time: datetime | None = None
    total_items: int = 0
    total_amount: float | None = None
    raw_payload: dict[str, Any] | None = None


class ScriptRunResponse(BaseModel):
    script_name: str
    model_name: str | None = None
    status: str
    command: list[str]
    stdout: str
    stderr: str


class EntryRunRequest(BaseModel):
    video_path: str
    model_name: str | None = None
    output_dir: str | None = None
    gallery_state_path: str | None = None


class KioskRunRequest(BaseModel):
    video_path: str
    model_name: str | None = None
    output_dir: str | None = None
    gallery_state_path: str | None = None


class RetrievalRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    location_id: int


class RetrievalAcceptedResponse(BaseModel):
    ok: bool = True
    message: str
    video_asset_id: int
    trigger_id: int | None = None
    session_id: int | None = None
    location_id: int
    section: str
    status: str
    video_url: str
    file_path: str
    requested_start_time: datetime
    requested_end_time: datetime


class SessionFinalizeRequest(BaseModel):
    kiosk_total_items: int = Field(ge=0)
    actual_items_brought: int | None = Field(default=None, ge=0)


class SessionFinalizeResponse(BaseModel):
    session_id: int
    status: str
    kiosk_total_items: int
    transaction_total_items: int
    actual_items_brought: int
    result_summary: dict[str, Any]


class CustomerGalleryCreate(BaseModel):
    location_id: int
    person_id: int
    session_customer_id: int | None = None
    image_url: str | None = None
    image_kind: str = "reid_view"
    embedding_osnet: list[float] | None = None
    embedding_fashion: list[float] | None = None
    metadata: dict[str, Any] | None = None


class CustomerGalleryResponse(BaseModel):
    id: int
    location_id: int
    session_id: int
    session_customer_id: int | None = None
    person_id: int
    image_url: str | None = None
    image_kind: str
    embedding_osnet: Any | None = None
    embedding_fashion: Any | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class ActiveGalleryUpsert(BaseModel):
    session_id: int | None = None


class TheftListItem(BaseModel):
    id: str
    reference: str | None = None
    location_id: str | None = None
    status: str
    total_amount: float | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] | None = None
    session_customer_id: int
    person_id: int | None = None
    state_kind: str = "active_gallery"
    state_payload: dict[str, Any]
    metadata: dict[str, Any] | None = None


class ActiveGalleryResponse(BaseModel):
    id: int
    location_id: int
    session_id: int | None = None
    session_customer_id: int
    person_id: int | None = None
    state_kind: str
    state_payload: dict[str, Any]
    metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class VideoAssetListItem(BaseModel):
    id: int
    trigger_id: int | None = None
    location_id: int | None = None
    section: str
    sequence_no: int | None = None
    video_url: str
    file_path: str | None = None
    captured_start_time: datetime | None = None
    captured_end_time: datetime | None = None
    retrieved_at: datetime | None = None
    analyzed_at: datetime | None = None
    retention_until: datetime | None = None
    status: str
    metadata: dict[str, Any] | None = None
    created_at: datetime
    session_link_count: int = 0
    primary_session_id: int | None = None
    session_ids: str | None = None
