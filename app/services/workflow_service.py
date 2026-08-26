from __future__ import annotations
import base64
import json
import logging
import mimetypes
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from os.path import basename, splitext
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import text
from sqlalchemy.orm import Session
from PIL import Image

from ..config import settings
from ..crypto import decrypt_secret
from .. import repositories
from ..db import TransactionalSessionLocal, VectorSessionLocal
from ..spaces import (
    generate_presigned_download_url,
    generate_public_object_url,
    is_spaces_configured,
    is_spaces_public_read_enabled,
    upload_private_file,
)
from .. import vector_repositories
from ..storage import (
    alert_tmp_video_path,
    build_spaces_object_key,
    guess_media_type,
    gallery_state_path as build_private_gallery_state_path,
    processed_video_spaces_key,
    source_video_spaces_key,
    session_logs_root,
    session_root,
    tmp_media_root,
    trigger_gallery_state_path,
    trigger_processed_root,
    trigger_tmp_video_path,
    session_tmp_video_path,
)


UTC = timezone.utc
SCRIPT_RUN_COMMAND_REDACTED = "[redacted]"
logger = logging.getLogger("tds.workflow_service")
KIOSK_OWNERSHIP_MIN_MARGIN_SECONDS = 10.0
NO_KIOSK_VIDEO_REASON = "No kiosk video was queued because no paid transactions were found inside the session window."
DEFAULT_FILTER_FACTORS: dict[str, dict[str, Any]] = {
    "long_stay_low_purchase": {"enabled": True},
    "transaction_issue_low_purchase": {"enabled": True},
    "multiple_transaction_issues": {"enabled": True},
    "multiple_minus_button_alert": {"enabled": True},
    "carry_item_signal": {"enabled": True},
    "country_code_check": {"enabled": True},
    "unusual_group_size": {"enabled": True},
    "customer_risk_history": {"enabled": False},
}
GEMINI_COSTS_PER_1M_TOKENS_USD: dict[str, dict[str, float]] = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached_input": 0.15},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached_input": 0.025},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
}
DEFAULT_RUNPOD_SERVERLESS_COST_PER_SECOND_USD = 0.69 / 3600


@dataclass
class ScriptExecutionResult:
    script_run_id: int
    script_name: str
    model_name: str | None
    status: str
    command: list[str]
    stdout: str
    stderr: str
    runner_job_id: str | None = None
    message: str | None = None


class GeminiKioskSummaryError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def get_script_run_details(db: Session, script_run_id: int) -> dict[str, Any]:
    record = repositories.get_script_run(db, script_run_id)
    if not record:
        raise ValueError(f"Script run {script_run_id} was not found.")
    return _build_script_run_details(record)


def get_script_run_details_by_runner_job_id(db: Session, runner_job_id: str) -> dict[str, Any]:
    record = repositories.get_script_run_by_runner_job_id(db, runner_job_id)
    if not record:
        raise ValueError(f"Script run with runner job id {runner_job_id} was not found.")
    return _build_script_run_details(record)


def list_script_run_details(
    db: Session,
    limit: int = 100,
    *,
    script_name: str | None = None,
    script_type: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _build_script_run_details(record)
        for record in repositories.list_script_runs(
            db,
            limit=limit,
            script_name=script_name,
            script_type=script_type,
            model_name=model_name,
        )
    ]


def get_latest_script_run_details_for_session(
    db: Session,
    session_id: int,
    *,
    script_name: str | None = None,
) -> dict[str, Any]:
    record = repositories.get_latest_script_run_for_session(db, session_id, script_name=script_name)
    if not record:
        scope = f" and script {script_name}" if script_name else ""
        raise ValueError(f"Script run for session {session_id}{scope} was not found.")
    return _build_script_run_details(record)


def get_latest_script_run_details_for_video_asset(
    db: Session,
    video_asset_id: int,
) -> dict[str, Any]:
    record = repositories.get_latest_script_run_for_video_asset(db, video_asset_id)
    if not record:
        raise ValueError(f"Script run for video asset {video_asset_id} was not found.")
    return _build_script_run_details(record)


def get_session_pipeline_log_details(db: Session, session_id: int) -> dict[str, Any]:
    session = repositories.get_session(db, session_id)
    session_videos = repositories.list_session_video_assets(db, session_id=session_id)

    entry_videos = [row for row in session_videos if str(row.get("section") or "").strip().lower() == "entry"]
    kiosk_videos = [row for row in session_videos if str(row.get("section") or "").strip().lower() == "kiosk"]
    exit_videos = [row for row in session_videos if str(row.get("section") or "").strip().lower() == "exit"]

    entry_run = None
    entry_trigger_id = session.get("entry_trigger_id")
    if entry_trigger_id is not None:
        try:
            entry_record = repositories.get_latest_script_run_for_trigger(db, int(entry_trigger_id), script_name="entry")
        except ValueError:
            entry_record = {}
        if entry_record:
            entry_run = _build_script_run_details(entry_record)

    kiosk_run = None
    try:
        kiosk_record = repositories.get_latest_script_run_for_session(db, session_id, script_name="kiosk")
    except ValueError:
        kiosk_record = {}
    if kiosk_record:
        kiosk_run = _build_script_run_details(kiosk_record)

    exit_run = None
    exit_trigger_id = session.get("exit_trigger_id")
    if exit_trigger_id is not None:
        try:
            exit_record = repositories.get_latest_script_run_for_trigger(db, int(exit_trigger_id), script_name="retrieve_video")
        except ValueError:
            exit_record = {}
        if exit_record:
            exit_run = _build_script_run_details(exit_record)

    return {
        "session_id": session_id,
        "entry": {
            "label": "Entry",
            "script_run": entry_run,
            "video_assets": entry_videos,
        },
        "kiosk": {
            "label": "Kiosk",
            "script_run": kiosk_run,
            "video_assets": kiosk_videos,
        },
        "exit": {
            "label": "Exit",
            "script_run": exit_run,
            "video_assets": exit_videos,
        },
    }


def _build_script_run_details(record: Mapping[str, Any]) -> dict[str, Any]:
    runner_payload = record.get("runner_payload")
    if not isinstance(runner_payload, dict):
        runner_payload = {}

    command = record.get("command")
    if isinstance(command, list):
        command_list = [str(item) for item in command]
    elif isinstance(command, str):
        command_list = [command]
    else:
        command_list = []

    return {
        "script_run_id": int(record["id"]),
        "session_id": record.get("session_id"),
        "trigger_id": record.get("trigger_id"),
        "runner_job_id": record.get("runner_job_id"),
        "script_name": str(record.get("script_name") or ""),
        "model_name": record.get("model_name"),
        "status": str(record.get("status") or ""),
        "command": command_list,
        "stdout": str(record.get("stdout_log") or ""),
        "stderr": str(record.get("stderr_log") or ""),
        "runner_payload": runner_payload,
        "cost_amount": record.get("cost_amount"),
        "cost_currency": record.get("cost_currency") or "USD",
        "cost_source": record.get("cost_source"),
        "log_object_key": runner_payload.get("log_object_key"),
        "log_url": runner_payload.get("log_url"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
    }


@dataclass
class VideoRetrievalResult:
    video_asset_id: int | None
    session_id: int | None
    trigger_id: int | None
    location_id: int
    section: str
    requested_start_time: str
    requested_end_time: str
    output_path: str
    rtsp_url: str
    command: list[str]
    status: str
    stdout: str
    stderr: str


@dataclass
class VideoRetrievalQueued:
    video_asset_id: int
    session_id: int | None
    trigger_id: int | None
    location_id: int
    section: str
    requested_start_time: datetime
    requested_end_time: datetime
    delayed_seconds: int
    adjusted_start_time: datetime
    adjusted_end_time: datetime
    output_path: str
    rtsp_url: str
    dahua_host: str
    dahua_username: str
    rtsp_port: int
    status: str
    video_url: str
    retrieval_mode: str | None = None


@dataclass
class TriggerFrameAssetRetrievalQueued:
    frame_asset_id: int
    trigger_id: int
    location_id: int
    section: str
    requested_start_time: datetime
    requested_end_time: datetime
    adjusted_start_time: datetime
    adjusted_end_time: datetime
    output_dir: str
    rtsp_url: str


@dataclass
class EntranceAnalysisQueued:
    video_asset_id: int
    trigger_id: int
    session_id: int | None
    location_id: int
    video_path: str
    model_name: str | None = None


@dataclass
class KioskAnalysisQueued:
    video_asset_id: int
    session_id: int
    location_id: int
    video_path: str
    model_name: str | None = None


@dataclass
class GroupingAnalysisQueued:
    batch_id: int
    location_id: int
    period_code: str
    window_start: datetime
    window_end: datetime
    manifest_url: str
    manifest_object_key: str
    model_name: str | None = None


@dataclass
class RemoteRunnerResult:
    status: str
    stdout: str
    stderr: str
    processed_video_object_key: str | None
    processed_video_url: str | None
    log_object_key: str | None = None
    log_url: str | None = None
    tracking_summary: dict[str, Any] | None = None
    reid_views_summary: dict[str, Any] | None = None
    kiosk_summary: dict[str, Any] | None = None
    transaction_match_summary: dict[str, Any] | None = None
    grouping_summary: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass
class RunpodEnqueueResult:
    job_id: str


def build_session_workdir(location_id: int, session_id: int) -> Path:
    workdir = session_root(location_id, session_id)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def build_session_output_root(location_id: int, session_id: int) -> Path:
    output_root = build_session_workdir(location_id, session_id) / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def build_logs_root(location_id: int, session_id: int) -> Path:
    logs_root = session_logs_root(location_id, session_id)
    logs_root.mkdir(parents=True, exist_ok=True)
    return logs_root


def default_video_output_dir(location_id: int, session_id: int, video_path: str) -> Path:
    stem = splitext(basename(video_path))[0]
    out_dir = build_logs_root(location_id, session_id) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def default_trigger_output_dir(location_id: int, trigger_id: int) -> Path:
    return trigger_processed_root(location_id, trigger_id, "entrance")


def _runpod_dispatch_busy(db: Session) -> bool:
    return repositories.has_active_remote_analysis_script_run(db)


def _runner_enabled() -> bool:
    return _runpod_runner_enabled()


def _http_runner_enabled() -> bool:
    return False


def _runpod_runner_enabled() -> bool:
    has_endpoint = any(
        str(value or "").strip()
        for value in (
            settings.runpod_endpoint_id,
            settings.runpod_entry_endpoint_id,
            settings.runpod_kiosk_endpoint_id,
            settings.runpod_grouping_endpoint_id,
        )
    )
    return bool(has_endpoint and str(settings.runpod_api_key or "").strip())


def _spaces_download_url_for_object_key(object_key: str) -> str:
    if is_spaces_public_read_enabled():
        return generate_public_object_url(object_key)
    return generate_presigned_download_url(object_key, expires_seconds=settings.runner_timeout_seconds)


def _runner_input_object_key(
    *,
    kind: str,
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    section: str | None,
    filename: str,
) -> str:
    segments = [f"location_{location_id}"]
    if trigger_id is not None:
        segments.append(f"trigger_{trigger_id}")
    elif session_id is not None:
        segments.append(f"session_{session_id}")
    if section:
        segments.append(section)
    segments.append(kind)
    segments.append(filename)
    return build_spaces_object_key(*segments)


def _upload_runner_input_file(
    local_path: Path,
    *,
    kind: str,
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    section: str | None = None,
) -> tuple[str, str]:
    if not is_spaces_configured():
        raise RuntimeError(
            "Remote runner execution requires DigitalOcean Spaces. Configure Spaces in `tds/.env`."
        )
    object_key = _runner_input_object_key(
        kind=kind,
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
        section=section,
        filename=local_path.name,
    )
    upload_private_file(local_path, object_key, content_type=guess_media_type(str(local_path)))
    return object_key, _spaces_download_url_for_object_key(object_key)


def _upload_customer_gallery_image_to_spaces(
    image_url: str | None,
    *,
    location_id: int,
    session_id: int,
    session_customer_id: int,
    person_id: int,
    output_dir: Path,
) -> str | None:
    if not image_url:
        return None

    parsed = urlparse(image_url)
    if parsed.scheme in {"http", "https"}:
        return image_url
    if not is_spaces_configured():
        return None

    candidate_path = Path(image_url)
    if not candidate_path.exists():
        candidate_path = output_dir / Path(image_url).name
    if not candidate_path.exists() or not candidate_path.is_file():
        logger.warning(
            "Customer gallery image could not be uploaded because file was not found image_url=%s fallback=%s",
            image_url,
            candidate_path,
        )
        return None

    object_key = build_spaces_object_key(
        f"location_{location_id}",
        f"session_{session_id}",
        "customer_gallery",
        f"sc_{session_customer_id}",
        f"person_{person_id}",
        candidate_path.name,
    )
    upload_result = upload_private_file(
        candidate_path,
        object_key,
        content_type=guess_media_type(str(candidate_path)),
    )
    return upload_result.get("public_url") or generate_public_object_url(object_key)


def _build_processed_video_upload_target(
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    script_name: str,
    processed_video_path: Path | None = None,
    source_video_path: str | None = None,
) -> dict[str, str]:
    if processed_video_path is not None:
        source_stem = processed_video_path.stem
        suffix = processed_video_path.suffix
    else:
        source_stem = Path(source_video_path or "video").stem
        suffix = ".mp4"
    if source_stem.endswith("_output"):
        source_stem = source_stem[: -len("_output")]

    shortened_stem = source_stem
    section_name = str(video_asset_row.get("section") or script_name or "video").strip().lower()
    section_prefix = f"{section_name}_playback_"
    if shortened_stem.startswith(section_prefix):
        shortened_stem = shortened_stem[len(section_prefix) :]

    timestamp_match = re.fullmatch(
        r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})",
        shortened_stem,
    )
    if timestamp_match:
        (
            start_year,
            start_month,
            start_day,
            start_hour,
            start_minute,
            start_second,
            end_year,
            end_month,
            end_day,
            end_hour,
            end_minute,
            end_second,
        ) = timestamp_match.groups()
        shortened_stem = (
            f"{start_day}_{start_month}_{start_year[-2:]}"
            f"_{start_hour}{start_minute}{start_second}"
            f"_{end_day}_{end_month}_{end_year[-2:]}"
            f"_{end_hour}{end_minute}{end_second}"
        )

    run_suffix = datetime.now(UTC).strftime("r%Y%m%d%H%M%S")
    upload_filename = f"{section_name[:1] or 'v'}_{shortened_stem}_{run_suffix}{suffix}"

    object_key = processed_video_spaces_key(
        location_id=location_id,
        section=str(video_asset_row.get("section") or script_name),
        filename=upload_filename,
        session_id=session_id,
        trigger_id=trigger_id,
    )
    return {
        "object_key": object_key,
        "video_url": (
            generate_public_object_url(object_key)
            if is_spaces_public_read_enabled()
            else f"/api/v1/videos/assets/{int(video_asset_row['id'])}/content"
        ),
        "file_path": f"spaces://{object_key}",
    }


def _apply_processed_video_upload_result(
    db: Session,
    *,
    video_asset_row: dict[str, Any],
    object_key: str,
    video_url: str | None,
) -> None:
    repositories.update_video_asset(
        db,
        int(video_asset_row["id"]),
        {
            "video_url": str(video_url or f"/api/v1/videos/assets/{int(video_asset_row['id'])}/content"),
            # Keep file_path pointing at the raw/source video. Processed playback lives in video_url only.
            "file_path": video_asset_row.get("file_path"),
            "captured_start_time": video_asset_row.get("captured_start_time"),
            "captured_end_time": video_asset_row.get("captured_end_time"),
            "retrieved_at": video_asset_row.get("retrieved_at"),
            "analyzed_at": datetime.now(UTC),
            "retention_until": video_asset_row.get("retention_until"),
            "status": "processed",
            "metadata": None,
        },
    )


def _build_source_video_upload_target(
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    source_video_path: str,
) -> dict[str, str]:
    source_filename = Path(source_video_path).name or f"video_asset_{int(video_asset_row['id'])}.mp4"
    object_key = source_video_spaces_key(
        location_id=location_id,
        section=str(video_asset_row.get("section") or "video"),
        filename=source_filename,
        session_id=session_id,
        trigger_id=trigger_id,
    )
    return {
        "object_key": object_key,
        "video_url": (
            generate_public_object_url(object_key)
            if is_spaces_public_read_enabled()
            else _spaces_download_url_for_object_key(object_key)
        ),
        "file_path": f"spaces://{object_key}",
    }


def _ensure_source_video_ready_for_runner(
    db: Session,
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
) -> tuple[dict[str, Any], str]:
    source_video_url = str(video_asset_row.get("video_url") or "").strip()
    source_file_path = str(video_asset_row.get("file_path") or "").strip()
    if source_file_path.startswith("spaces://"):
        source_object_key = source_file_path.removeprefix("spaces://").lstrip("/")
        if is_spaces_public_read_enabled():
            source_url = generate_public_object_url(source_object_key)
        else:
            source_url = _spaces_download_url_for_object_key(source_object_key)
        return video_asset_row, source_url

    local_path = Path(source_file_path)
    if not local_path.exists():
        if source_video_url:
            return video_asset_row, source_video_url
        raise RuntimeError(
            f"Runpod analysis requires a retrievable source video, but local file does not exist: {source_file_path}"
        )

    upload_target = _build_source_video_upload_target(
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
        source_video_path=str(local_path),
    )
    upload_result = upload_private_file(
        local_path,
        upload_target["object_key"],
        content_type=guess_media_type(str(local_path)),
    )
    repositories.update_video_asset(
        db,
        int(video_asset_row["id"]),
        {
            "video_url": str(upload_result.get("public_url") or upload_target["video_url"]),
            "file_path": f"spaces://{upload_target['object_key']}",
            "captured_start_time": video_asset_row.get("captured_start_time"),
            "captured_end_time": video_asset_row.get("captured_end_time"),
            "retrieved_at": video_asset_row.get("retrieved_at"),
            "analyzed_at": video_asset_row.get("analyzed_at"),
            "retention_until": video_asset_row.get("retention_until"),
            "status": str(video_asset_row.get("status") or "ready"),
            "metadata": None,
        },
    )
    refreshed_row = repositories.get_video_asset(db, int(video_asset_row["id"]))
    return refreshed_row, str(refreshed_row.get("video_url") or upload_target["video_url"])


def _ensure_analysis_uses_source_video(video_path: str, *, video_asset_id: int) -> None:
    normalized = video_path.replace("\\", "/")
    if normalized.startswith("spaces://"):
        object_key = normalized.removeprefix("spaces://").lstrip("/")
        if "/source/" not in object_key:
            raise ValueError(
                f"Video asset {video_asset_id} does not point to a raw source video. "
                "Refusing to analyze processed output. Re-retrieve the raw video first."
            )
        return
    if "/processed/" in normalized or Path(normalized).stem.endswith("_output"):
        raise ValueError(
            f"Video asset {video_asset_id} points to a processed video. "
            "Refusing to analyze processed output. Re-retrieve the raw video first."
        )


def _repair_video_asset_source_file_path_for_analysis(
    db: Session,
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
) -> dict[str, Any]:
    file_path = str(video_asset_row.get("file_path") or "").strip()
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("spaces://") and "/source/" in normalized:
        return video_asset_row
    if normalized and not normalized.startswith("spaces://") and "/processed/" not in normalized and not Path(normalized).stem.endswith("_output"):
        return video_asset_row

    captured_start = video_asset_row.get("captured_start_time")
    captured_end = video_asset_row.get("captured_end_time")
    section = str(video_asset_row.get("section") or "video").strip().lower()
    if not isinstance(captured_start, datetime) or not isinstance(captured_end, datetime):
        return video_asset_row

    source_filename = (
        f"{section}_playback_"
        f"{_format_dahua_playback_time(captured_start)}_"
        f"{_format_dahua_playback_time(captured_end)}.mp4"
    )
    source_object_key = source_video_spaces_key(
        location_id=location_id,
        section=section,
        filename=source_filename,
        session_id=session_id,
        trigger_id=trigger_id,
    )
    repaired_file_path = f"spaces://{source_object_key}"
    repositories.update_video_asset(
        db,
        int(video_asset_row["id"]),
        {
            "video_url": video_asset_row.get("video_url"),
            "file_path": repaired_file_path,
            "captured_start_time": captured_start,
            "captured_end_time": captured_end,
            "retrieved_at": video_asset_row.get("retrieved_at"),
            "analyzed_at": video_asset_row.get("analyzed_at"),
            "retention_until": video_asset_row.get("retention_until"),
            "status": video_asset_row.get("status"),
            "metadata": video_asset_row.get("metadata"),
        },
    )
    logger.warning(
        "Repaired video_asset source file_path before analysis video_asset_id=%s old_file_path=%s repaired_file_path=%s",
        video_asset_row.get("id"),
        file_path,
        repaired_file_path,
    )
    return repositories.get_video_asset(db, int(video_asset_row["id"]))


def _write_remote_entry_summaries(
    *,
    video_path: str,
    output_dir: Path,
    tracking_summary: dict[str, Any],
    reid_views_summary: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _tracking_summary_path(video_path, output_dir).write_text(json.dumps(tracking_summary, indent=2))
    _reid_views_summary_path(video_path, output_dir).write_text(
        json.dumps(reid_views_summary or {"customers": []}, indent=2)
    )


def _remote_runner_result_from_runpod_body(body: dict[str, Any]) -> tuple[str, RemoteRunnerResult]:
    status = str(body.get("status") or "").upper()
    output = body.get("output")
    if status == "COMPLETED":
        if not isinstance(output, dict):
            raise RuntimeError(f"Runpod completed without structured output: {body}")
        return (
            status,
            RemoteRunnerResult(
                status=str(output.get("status") or "success"),
                stdout=str(output.get("stdout") or ""),
                stderr=str(output.get("stderr") or ""),
                processed_video_object_key=output.get("processed_video_object_key"),
                processed_video_url=output.get("processed_video_url"),
                log_object_key=output.get("log_object_key"),
                log_url=output.get("log_url"),
                tracking_summary=output.get("tracking_summary"),
                reid_views_summary=output.get("reid_views_summary"),
                kiosk_summary=output.get("kiosk_summary"),
                transaction_match_summary=output.get("transaction_match_summary"),
                grouping_summary=output.get("grouping_summary"),
                meta=output.get("meta") if isinstance(output.get("meta"), dict) else None,
            ),
        )

    error_detail = body.get("error") or body.get("message") or body
    if isinstance(output, dict):
        return (
            status,
            RemoteRunnerResult(
                status="failed",
                stdout=str(output.get("stdout") or ""),
                stderr=str(output.get("stderr") or error_detail),
                processed_video_object_key=output.get("processed_video_object_key"),
                processed_video_url=output.get("processed_video_url"),
                log_object_key=output.get("log_object_key"),
                log_url=output.get("log_url"),
                tracking_summary=output.get("tracking_summary"),
                reid_views_summary=output.get("reid_views_summary"),
                kiosk_summary=output.get("kiosk_summary"),
                transaction_match_summary=output.get("transaction_match_summary"),
                grouping_summary=output.get("grouping_summary"),
                meta=output.get("meta") if isinstance(output.get("meta"), dict) else None,
            ),
        )
    return (
        status,
        RemoteRunnerResult(
            status="failed",
            stdout="",
            stderr=str(error_detail),
            processed_video_object_key=None,
            processed_video_url=None,
        ),
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _gemini_model_key(model_name: Any) -> str:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return ""
    normalized = normalized.removeprefix("models/")
    normalized = normalized.split("/")[-1]
    return normalized


def _gemini_model_cost_rates(model_name: Any) -> dict[str, float]:
    model_key = _gemini_model_key(model_name)
    if model_key in GEMINI_COSTS_PER_1M_TOKENS_USD:
        return GEMINI_COSTS_PER_1M_TOKENS_USD[model_key]
    for known_model, rates in GEMINI_COSTS_PER_1M_TOKENS_USD.items():
        if known_model in model_key:
            return rates
    return {}


def _remote_runner_cost(remote_result: RemoteRunnerResult) -> tuple[float | None, str | None]:
    meta = remote_result.meta if isinstance(remote_result.meta, Mapping) else {}
    for key in ("cost_amount", "cost_usd", "runpod_cost_usd", "estimated_cost_usd"):
        amount = _positive_float(meta.get(key))
        if amount is not None:
            return amount, "runpod_returned"

    duration_seconds = None
    for key in ("duration_seconds", "job_seconds", "runtime_seconds", "execution_seconds"):
        duration_seconds = _positive_float(meta.get(key))
        if duration_seconds is not None:
            break
    if duration_seconds is None:
        for key in ("executionTime", "execution_time_ms", "runtime_ms"):
            duration_milliseconds = _positive_float(meta.get(key))
            if duration_milliseconds is not None:
                duration_seconds = duration_milliseconds / 1000
                break
    cost_per_second = (
        _positive_float(settings.runpod_cost_per_second_usd)
        or DEFAULT_RUNPOD_SERVERLESS_COST_PER_SECOND_USD
    )
    if duration_seconds is None or cost_per_second is None:
        return None, None
    return duration_seconds * cost_per_second, "runpod_estimate"


def _record_remote_runner_cost(db: Session, script_run_id: int, remote_result: RemoteRunnerResult) -> None:
    amount, source = _remote_runner_cost(remote_result)
    repositories.add_script_run_cost(
        db,
        script_run_id,
        cost_amount=amount,
        cost_currency="USD",
        cost_source=source,
    )


def _gemini_usage_cost(gemini_meta: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    raw_usage = gemini_meta.get("raw_usage") if isinstance(gemini_meta.get("raw_usage"), Mapping) else {}
    usage = gemini_meta.get("usage") if isinstance(gemini_meta.get("usage"), Mapping) else {}
    model_name = gemini_meta.get("model") or gemini_meta.get("model_name")
    model_rates = _gemini_model_cost_rates(model_name)
    input_tokens = (
        _positive_float(raw_usage.get("promptTokenCount"))
        or _positive_float(usage.get("input_tokens"))
        or 0.0
    )
    output_tokens = (
        _positive_float(raw_usage.get("candidatesTokenCount"))
        or _positive_float(usage.get("output_tokens"))
        or 0.0
    )
    cached_input_tokens = (
        _positive_float(raw_usage.get("cachedContentTokenCount"))
        or _positive_float(usage.get("cached_input_tokens"))
        or 0.0
    )
    input_rate = _positive_float(settings.gemini_input_cost_per_1m_tokens_usd) or model_rates.get("input", 0.0)
    output_rate = _positive_float(settings.gemini_output_cost_per_1m_tokens_usd) or model_rates.get("output", 0.0)
    cached_rate = _positive_float(settings.gemini_cached_input_cost_per_1m_tokens_usd)
    if cached_rate is None:
        cached_rate = model_rates.get("cached_input")
    billable_input_tokens = max(0.0, input_tokens - cached_input_tokens)
    amount = (billable_input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    if cached_rate is not None:
        amount += cached_input_tokens * cached_rate / 1_000_000
    detail = {
        "model": model_name,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "input_cost_per_1m_tokens_usd": input_rate,
        "output_cost_per_1m_tokens_usd": output_rate,
        "cached_input_cost_per_1m_tokens_usd": cached_rate,
        "estimated_cost_usd": amount,
    }
    if amount <= 0:
        return None, detail
    return amount, detail


def _record_gemini_cost(db: Session, script_run_id: int, gemini_meta: Mapping[str, Any]) -> dict[str, Any]:
    amount, detail = _gemini_usage_cost(gemini_meta)
    repositories.add_script_run_cost(
        db,
        script_run_id,
        cost_amount=amount,
        cost_currency="USD",
        cost_source="gemini_estimate",
    )
    return detail


def _compact_gemini_meta_for_log(gemini_meta: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(gemini_meta.get("prompt") or "")
    compact: dict[str, Any] = {
        "provider": gemini_meta.get("provider"),
        "model": gemini_meta.get("model") or gemini_meta.get("model_name"),
        "image_count": gemini_meta.get("image_count"),
        "image_resize_scale": gemini_meta.get("image_resize_scale"),
        "prompt_chars": len(prompt),
        "prompt_preview": prompt[:800],
        "raw_usage": gemini_meta.get("raw_usage") or {},
    }
    image_urls = gemini_meta.get("image_urls")
    if isinstance(image_urls, list):
        compact["image_urls"] = image_urls
    chunks = gemini_meta.get("chunks")
    if isinstance(chunks, list):
        compact["chunks"] = [
            _compact_gemini_meta_for_log(chunk)
            for chunk in chunks
            if isinstance(chunk, Mapping)
        ]
    return compact


def _create_gemini_script_run(
    db: Session,
    *,
    session_id: int | None,
    trigger_id: int | None,
    script_name: str,
    model_name: str,
    runner_payload: Mapping[str, Any],
) -> int:
    return repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        model_name=model_name,
        runner_payload=runner_payload,
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
        stdout_log="",
        stderr_log="",
    )


def _record_gemini_log_cost(db: Session, script_run_id: int, gemini_log: Mapping[str, Any]) -> dict[str, Any]:
    cost_details: list[dict[str, Any]] = []
    total_amount = 0.0
    groups = gemini_log.get("groups") if isinstance(gemini_log.get("groups"), list) else []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        meta = group.get("meta") if isinstance(group.get("meta"), Mapping) else None
        if not meta:
            continue
        amount, detail = _gemini_usage_cost(meta)
        detail["group_id"] = group.get("group_id")
        cost_details.append(detail)
        if amount is not None:
            total_amount += amount
    if total_amount > 0:
        repositories.add_script_run_cost(
            db,
            script_run_id,
            cost_amount=total_amount,
            cost_currency="USD",
            cost_source="gemini_estimate",
        )
    return {
        "source": "gemini_estimate",
        "currency": "USD",
        "amount": total_amount,
        "groups": cost_details,
    }


def _extract_json_object(text_value: str) -> dict[str, Any]:
    text_value = str(text_value or "").strip()
    if text_value.startswith("```"):
        text_value = re.sub(r"^```(?:json)?\s*", "", text_value, flags=re.IGNORECASE)
        text_value = re.sub(r"\s*```$", "", text_value)
    try:
        parsed = json.loads(text_value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text_value.find("{")
    end = text_value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text_value[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Gemini response did not contain a JSON object.")


def _download_image_for_gemini(image_url: str, *, resize_scale: float | None = None) -> dict[str, Any]:
    with urlopen(image_url, timeout=settings.kiosk_gemini_timeout_seconds) as response:
        payload = response.read()
        content_type = response.headers.get_content_type()
    if not content_type or content_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(image_url)
        content_type = guessed or "image/jpeg"
    scale = _coerce_number(resize_scale, 1.0) if resize_scale is not None else 1.0
    if 0 < scale < 1:
        try:
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized = image.convert("RGB").resize(resized_size, Image.Resampling.LANCZOS)
                output = BytesIO()
                resized.save(output, format="JPEG", quality=85, optimize=True)
                payload = output.getvalue()
                content_type = "image/jpeg"
        except Exception:
            logger.exception("Could not resize Gemini image before sending url=%s", image_url)
    return {
        "inline_data": {
            "mime_type": content_type,
            "data": base64.b64encode(payload).decode("ascii"),
        }
    }


def _call_kiosk_gemini_summary(
    *,
    prompt: str,
    image_urls: list[str],
    model_name: str | None = None,
    image_resize_scale: float | None = None,
    allow_text_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = str(settings.gemini_api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured in tds_api. Set THEFT_API_GEMINI_API_KEY.")
    if not image_urls and not allow_text_only:
        raise RuntimeError("No kiosk evidence image URLs were returned by the runner.")

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for image_url in image_urls:
        parts.append(_download_image_for_gemini(image_url, resize_scale=image_resize_scale))

    selected_model = str(model_name or settings.kiosk_gemini_model).strip()
    request_body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    url = (
        f"{str(settings.kiosk_gemini_base_url).rstrip('/')}/models/"
        f"{selected_model}:generateContent?key={quote(api_key)}"
    )
    request = Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.kiosk_gemini_timeout_seconds) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {detail[:1000]}") from exc

    parsed = json.loads(raw_body)
    candidates = parsed.get("candidates") or []
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
    text_parts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    result = _extract_json_object("\n".join(text_parts))
    return result, {
        "provider": "tds_api_gemini",
        "model": selected_model,
        "image_count": len(image_urls),
        "image_resize_scale": image_resize_scale,
        "prompt": prompt,
        "image_urls": image_urls,
        "raw_response": parsed,
        "raw_usage": parsed.get("usageMetadata") or {},
    }


def _count_items_from_kiosk_vlm_result(result: Mapping[str, Any]) -> int:
    for key in ("suspected_total_count", "confirmed_visible_count", "total_items_taken_out"):
        try:
            value = int(result.get(key) or 0)
            if value >= 0:
                return value
        except (TypeError, ValueError):
            continue
    total = 0
    for item in result.get("customers_left_with_items") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            total += max(0, int(item.get("carried_out_count") or 0))
        except (TypeError, ValueError):
            continue
    return total


def _complete_kiosk_summary_with_tds_gemini(
    kiosk_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not kiosk_summary:
        return kiosk_summary, {
            "status": "skipped",
            "skip_reason": "no_kiosk_summary",
            "provider": "tds_api_gemini",
            "groups": [],
            "detected_total_items": 0,
        }
    groups = kiosk_summary.get("groups")
    if not isinstance(groups, list):
        return kiosk_summary, {
            "status": "skipped",
            "skip_reason": "no_kiosk_groups",
            "provider": "tds_api_gemini",
            "groups": [],
            "detected_total_items": 0,
        }

    detected_total = 0
    completed_groups: list[dict[str, Any]] = []
    diagnostics_groups: list[dict[str, Any]] = []
    match_diagnostics = (
        dict(kiosk_summary.get("match_diagnostics") or {})
        if isinstance(kiosk_summary.get("match_diagnostics"), Mapping)
        else {}
    )
    for group in groups:
        if not isinstance(group, dict):
            continue
        enriched_group = dict(group)
        diagnostics_group = {
            "group_id": enriched_group.get("group_id"),
            "vlm_pending": bool(enriched_group.get("vlm_pending")),
        }
        prompt = str(enriched_group.get("vlm_prompt") or "").strip()
        image_urls = [
            str(url).strip()
            for url in (enriched_group.get("llm_input_image_urls") or [])
            if str(url).strip()
        ]
        diagnostics_group["prompt"] = prompt
        if not image_urls:
            image_urls = [
                _spaces_download_url_for_object_key(str(upload["object_key"]))
                for upload in (enriched_group.get("llm_input_image_uploads") or [])
                if isinstance(upload, Mapping) and str(upload.get("object_key") or "").strip()
            ]
            if image_urls:
                enriched_group["llm_input_image_urls"] = image_urls
        diagnostics_group["image_urls"] = image_urls
        if enriched_group.get("vlm_pending") and prompt and not image_urls:
            diagnostics_group["status"] = "failed"
            diagnostics_group["error"] = (
                f"Kiosk group {enriched_group.get('group_id')} requires TDS Gemini, "
                "but runner returned no evidence image URLs or Spaces object keys."
            )
            diagnostics_groups.append(diagnostics_group)
            raise GeminiKioskSummaryError(
                diagnostics_group["error"],
                diagnostics={
                    "provider": "tds_api_gemini",
                    "groups": diagnostics_groups,
                    "detected_total_items": detected_total,
                },
            )
        if enriched_group.get("vlm_pending") and prompt and image_urls:
            try:
                vlm_result, vlm_meta = _call_kiosk_gemini_summary(prompt=prompt, image_urls=image_urls)
            except Exception as exc:
                diagnostics_group["status"] = "failed"
                diagnostics_group["error"] = str(exc)
                diagnostics_groups.append(diagnostics_group)
                raise GeminiKioskSummaryError(
                    f"TDS Gemini failed for kiosk group {enriched_group.get('group_id')}: {exc}",
                    diagnostics={
                        "provider": "tds_api_gemini",
                        "groups": diagnostics_groups,
                        "detected_total_items": detected_total,
                    },
                ) from exc
            enriched_group["vlm_result"] = vlm_result
            enriched_group["vlm_meta"] = vlm_meta
            enriched_group["vlm_pending"] = False
            diagnostics_group["status"] = "success"
            diagnostics_group["result"] = vlm_result
            diagnostics_group["meta"] = vlm_meta
            kiosk_event_summary = dict(enriched_group.get("kiosk_event_summary") or {})
            kiosk_event_summary["total_items_taken_out"] = _count_items_from_kiosk_vlm_result(vlm_result)
            kiosk_event_summary["vlm_result"] = vlm_result
            enriched_group["kiosk_event_summary"] = kiosk_event_summary
        else:
            meta_status = str((enriched_group.get("vlm_meta") or {}).get("status") or "").strip()
            diagnostics_group["status"] = "skipped"
            diagnostics_group["skip_reason"] = meta_status or "runner_completed_without_tds_gemini"
            diagnostics_group["result"] = (enriched_group.get("kiosk_event_summary") or {}).get("vlm_result")
            diagnostics_group["meta"] = enriched_group.get("vlm_meta")
        detected_total += int((enriched_group.get("kiosk_event_summary") or {}).get("total_items_taken_out") or 0)
        completed_groups.append(enriched_group)
        diagnostics_group["group_detected_total_items"] = int(
            (enriched_group.get("kiosk_event_summary") or {}).get("total_items_taken_out") or 0
        )
        diagnostics_groups.append(diagnostics_group)

    completed_summary = {
        **kiosk_summary,
        "groups": completed_groups,
        "detected_total_items": detected_total,
        "vlm_completed_by": "tds_api",
    }
    diagnostics = {
        "status": "success",
        "provider": "tds_api_gemini",
        "model": settings.kiosk_gemini_model,
        "groups": diagnostics_groups,
        "detected_total_items": detected_total,
        "match_diagnostics": match_diagnostics,
    }
    if not completed_groups:
        diagnostics["status"] = "skipped"
        diagnostics["skip_reason"] = str(match_diagnostics.get("gemini_skip_reason") or "no_kiosk_groups")
        diagnostics["message"] = str(
            match_diagnostics.get("gemini_skip_message")
            or "Gemini skipped because the kiosk runner produced no groups to summarize."
        )
    elif all(str(group.get("status") or "") == "skipped" for group in diagnostics_groups):
        diagnostics["status"] = "skipped"
        diagnostics["skip_reason"] = str(
            match_diagnostics.get("gemini_skip_reason")
            or next(
                (
                    str(group.get("skip_reason") or "")
                    for group in diagnostics_groups
                    if str(group.get("skip_reason") or "").strip()
                ),
                "runner_skipped_all_groups",
            )
        )
        diagnostics["message"] = str(
            match_diagnostics.get("gemini_skip_message")
            or "Gemini skipped because the kiosk runner did not produce usable evidence groups."
        )
    return completed_summary, diagnostics


def _kiosk_summary_requires_tds_gemini(kiosk_summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(kiosk_summary, Mapping):
        return False
    groups = kiosk_summary.get("groups")
    if not isinstance(groups, list):
        return False
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        prompt = str(group.get("vlm_prompt") or "").strip()
        image_urls = [
            str(url).strip()
            for url in (group.get("llm_input_image_urls") or [])
            if str(url).strip()
        ]
        image_uploads = [
            upload
            for upload in (group.get("llm_input_image_uploads") or [])
            if isinstance(upload, Mapping) and str(upload.get("object_key") or "").strip()
        ]
        if group.get("vlm_pending") and prompt and (image_urls or image_uploads):
            return True
    return False


def _finalize_remote_entry_script_run(
    db: Session,
    *,
    script_run: dict[str, Any],
    remote_result: RemoteRunnerResult,
) -> ScriptExecutionResult:
    if str(script_run.get("status") or "").strip().lower() != "running":
        return ScriptExecutionResult(
            script_run_id=int(script_run["id"]),
            runner_job_id=str(script_run.get("runner_job_id") or ""),
            script_name="entry",
            model_name=script_run.get("model_name"),
            status=str(script_run.get("status") or "success"),
            command=["runpod_serverless", "entry"],
            stdout=str(script_run.get("stdout_log") or ""),
            stderr=str(script_run.get("stderr_log") or ""),
            message="Runpod callback already processed for this script run.",
        )
    runner_payload = dict(script_run.get("runner_payload") or {})
    script_run_id = int(script_run["id"])
    session_id = int(script_run["session_id"]) if script_run.get("session_id") is not None else None
    trigger_id = int(script_run["trigger_id"]) if script_run.get("trigger_id") is not None else None
    location_id = int(runner_payload["location_id"])
    video_path = str(runner_payload["video_path"])
    output_dir = Path(str(runner_payload["output_dir"]))
    gallery_state_path = Path(str(runner_payload["gallery_state_path"]))
    video_asset_id = int(runner_payload["video_asset_id"])
    processed_video_url = str(remote_result.processed_video_url or runner_payload.get("processed_video_url") or "")
    repositories.assign_script_run_runner_job(
        db,
        script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        runner_payload={
            **runner_payload,
            "processed_video_url": processed_video_url or runner_payload.get("processed_video_url"),
            "log_object_key": remote_result.log_object_key or runner_payload.get("log_object_key"),
            "log_url": remote_result.log_url or runner_payload.get("log_url"),
        },
    )
    video_asset_row = repositories.get_video_asset(db, video_asset_id)
    trigger = repositories.get_trigger(db, int(trigger_id)) if trigger_id is not None else None
    session = repositories.get_session(db, session_id) if session_id is not None else None

    remote_status = "success" if remote_result.status == "success" else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=remote_status,
        stdout_log=remote_result.stdout,
        stderr_log=remote_result.stderr,
    )
    _record_remote_runner_cost(db, script_run_id, remote_result)
    result = ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        script_name="entry",
        model_name=script_run.get("model_name"),
        status=remote_status,
        command=["runpod_serverless", "entry"],
        stdout=remote_result.stdout,
        stderr=remote_result.stderr,
    )
    if result.status != "success":
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        return result
    if not remote_result.tracking_summary:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nRemote runner did not return tracking_summary.".strip()
        _record_followup_failure(
            db,
            script_run_id=result.script_run_id,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=str(script_run.get("model_name") or "remote_runner_tracking_summary_missing"),
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    _write_remote_entry_summaries(
        video_path=video_path,
        output_dir=output_dir,
        tracking_summary=remote_result.tracking_summary,
        reid_views_summary=remote_result.reid_views_summary,
    )
    try:
        _sync_gallery_state_after_entry(
            location_id=location_id,
            session_id=session_id,
            video_asset_id=video_asset_id,
            exit_trigger_id=trigger_id,
            video_path=video_path,
            output_dir=output_dir,
            gallery_state_path=gallery_state_path,
            enter_time=(
                session.get("start_time")
                if session
                else (trigger.get("trigger_time") if trigger else video_asset_row.get("captured_start_time"))
            ),
            leave_time=(trigger.get("trigger_time") if trigger else video_asset_row.get("captured_end_time")),
            captured_start_time=video_asset_row.get("captured_start_time"),
            captured_end_time=video_asset_row.get("captured_end_time"),
        )
    except Exception as exc:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nGallery persistence failed: {exc}".strip()
        _record_followup_failure(
            db,
            script_run_id=result.script_run_id,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=str(script_run.get("model_name") or "remote_runner_gallery_persistence"),
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    if not remote_result.processed_video_object_key:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nRemote runner did not return processed video object key.".strip()
        _record_followup_failure(
            db,
            script_run_id=result.script_run_id,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=str(script_run.get("model_name") or "remote_runner_processed_video_missing"),
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    _apply_processed_video_upload_result(
        db,
        video_asset_row=video_asset_row,
        object_key=remote_result.processed_video_object_key,
        video_url=processed_video_url,
    )
    session_id = script_run.get("session_id")
    try:
        completed_kiosk_summary = _complete_kiosk_summary_with_tds_gemini(remote_result.kiosk_summary)
    except Exception as exc:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nTDS API Gemini kiosk summary failed: {exc}".strip()
        repositories.revise_script_run(
            db,
            result.script_run_id,
            status="failed",
            stdout_log=result.stdout,
            stderr_log=stderr,
        )
        if session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=f"TDS API Gemini kiosk summary failed: {exc}",
            )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    if session_id is not None and completed_kiosk_summary:
        session_row = repositories.get_session(db, int(session_id))
        existing_summary = dict(session_row.get("result_summary") or {})
        kiosk_runs = dict(existing_summary.get("kiosk_runs") or {})
        kiosk_runs[str(video_asset_id)] = completed_kiosk_summary
        cumulative_detected_total = sum(
            int(run.get("detected_total_items") or 0)
            for run in kiosk_runs.values()
            if isinstance(run, dict)
        )
        merged_summary = {
            **existing_summary,
            "kiosk_runs": kiosk_runs,
            "kiosk_detected_total_items": cumulative_detected_total,
            "last_kiosk_video_asset_id": video_asset_id,
        }
        repositories.update_session_summary(
            db,
            session_id=int(session_id),
            status="pending",
            result_summary=merged_summary,
        )
        session_videos = repositories.list_session_video_assets(
            db,
            session_id=int(session_id),
            section="kiosk",
        )
        has_pending_kiosk = any(
            str(item.get("video_status") or "").strip().lower()
            in {"not_retrieved", "retrieving", "ready", "processing"}
            for item in session_videos
        )
        if not has_pending_kiosk:
            repositories.finalize_session_result(
                db,
                session_id=int(session_id),
                kiosk_total_items=cumulative_detected_total,
                tolerance=1,
                extra_result_summary=merged_summary,
            )
    return result


def _finalize_remote_kiosk_script_run(
    db: Session,
    *,
    script_run: dict[str, Any],
    remote_result: RemoteRunnerResult,
) -> ScriptExecutionResult:
    if str(script_run.get("status") or "").strip().lower() != "running":
        return ScriptExecutionResult(
            script_run_id=int(script_run["id"]),
            runner_job_id=str(script_run.get("runner_job_id") or ""),
            script_name="kiosk",
            model_name=script_run.get("model_name"),
            status=str(script_run.get("status") or "success"),
            command=["runpod_serverless", "kiosk"],
            stdout=str(script_run.get("stdout_log") or ""),
            stderr=str(script_run.get("stderr_log") or ""),
            message="Runpod callback already processed for this script run.",
        )
    runner_payload = dict(script_run.get("runner_payload") or {})
    script_run_id = int(script_run["id"])
    video_asset_id = int(runner_payload["video_asset_id"])
    processed_video_url = str(remote_result.processed_video_url or runner_payload.get("processed_video_url") or "")
    repositories.assign_script_run_runner_job(
        db,
        script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        runner_payload={
            **runner_payload,
            "processed_video_url": processed_video_url or runner_payload.get("processed_video_url"),
            "log_object_key": remote_result.log_object_key or runner_payload.get("log_object_key"),
            "log_url": remote_result.log_url or runner_payload.get("log_url"),
        },
    )
    video_asset_row = repositories.get_video_asset(db, video_asset_id)

    remote_status = "success" if remote_result.status == "success" else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=remote_status,
        stdout_log=remote_result.stdout,
        stderr_log=remote_result.stderr,
    )
    _record_remote_runner_cost(db, script_run_id, remote_result)
    result = ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        script_name="kiosk",
        model_name=script_run.get("model_name"),
        status=remote_status,
        command=["runpod_serverless", "kiosk"],
        stdout=remote_result.stdout,
        stderr=remote_result.stderr,
    )
    session_id = script_run.get("session_id")

    def persist_gemini_log(payload: Mapping[str, Any]) -> None:
        repositories.assign_script_run_runner_job(
            db,
            script_run_id,
            runner_job_id=str(script_run.get("runner_job_id") or ""),
            runner_payload={
                **dict(script_run.get("runner_payload") or {}),
                **dict(payload),
            },
        )

    if result.status != "success":
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        if session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=remote_result.stderr or "Kiosk analysis failed in the remote runner.",
        )
        return result
    persist_gemini_log(
        {
            "gemini_log": {
                "status": "pending",
                "message": "Waiting for TDS Gemini kiosk enrichment.",
            }
        }
    )
    if not remote_result.processed_video_object_key:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nRemote runner did not return processed video object key.".strip()
        repositories.revise_script_run(
            db,
            result.script_run_id,
            status="failed",
            stdout_log=result.stdout,
            stderr_log=stderr,
        )
        if session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason="Kiosk analysis finished without a processed video object key.",
            )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    if session_id is not None and not remote_result.kiosk_summary:
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nRemote runner did not return kiosk summary.".strip()
        repositories.revise_script_run(
            db,
            result.script_run_id,
            status="failed",
            stdout_log=result.stdout,
            stderr_log=stderr,
        )
        repositories.update_session_fields(
            db,
            session_id=int(session_id),
            status="issue",
            issue_reason="Kiosk analysis finished without kiosk summary data.",
        )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    gemini_script_run_id: int | None = None
    if _kiosk_summary_requires_tds_gemini(remote_result.kiosk_summary):
        gemini_script_run_id = _create_gemini_script_run(
            db,
            session_id=int(session_id) if session_id is not None else None,
            trigger_id=None,
            script_name="kiosk",
            model_name="gemini_kiosk_summary",
            runner_payload={
                "parent_script_run_id": result.script_run_id,
                "video_asset_id": video_asset_id,
                "session_id": int(session_id) if session_id is not None else None,
                "source": "tds_api_kiosk_enrichment",
            },
        )
    try:
        completed_kiosk_summary, gemini_log = _complete_kiosk_summary_with_tds_gemini(remote_result.kiosk_summary)
        gemini_status = str(gemini_log.get("status") or "success")
        gemini_cost = (
            _record_gemini_log_cost(db, gemini_script_run_id, gemini_log)
            if gemini_script_run_id is not None
            else {}
        )
        if gemini_script_run_id is not None:
            repositories.finish_script_run(
                db,
                gemini_script_run_id,
                status="success" if gemini_status != "failed" else "failed",
                stdout_log=json.dumps(gemini_log, indent=2, default=str),
                stderr_log="" if gemini_status != "failed" else str(gemini_log.get("error") or "Gemini kiosk summary failed."),
            )
        persist_gemini_log(
            {
                "gemini_log": {
                    "script_run_id": gemini_script_run_id,
                    "status": gemini_status,
                    "cost": gemini_cost,
                    **{key: value for key, value in gemini_log.items() if key != "status"},
                }
            }
        )
    except GeminiKioskSummaryError as exc:
        if gemini_script_run_id is not None:
            repositories.finish_script_run(
                db,
                gemini_script_run_id,
                status="failed",
                stdout_log=json.dumps(dict(exc.diagnostics or {}), indent=2, default=str),
                stderr_log=str(exc),
            )
        persist_gemini_log(
            {
                "gemini_log": {
                    "script_run_id": gemini_script_run_id,
                    "status": "failed",
                    "error": str(exc),
                    **dict(exc.diagnostics or {}),
                }
            }
        )
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nTDS API Gemini kiosk summary failed: {exc}".strip()
        repositories.revise_script_run(
            db,
            result.script_run_id,
            status="failed",
            stdout_log=result.stdout,
            stderr_log=stderr,
        )
        if session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=f"TDS API Gemini kiosk summary failed: {exc}",
            )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    except Exception as exc:
        if gemini_script_run_id is not None:
            repositories.finish_script_run(
                db,
                gemini_script_run_id,
                status="failed",
                stdout_log="",
                stderr_log=str(exc),
            )
        persist_gemini_log(
            {
                "gemini_log": {
                    "script_run_id": gemini_script_run_id,
                    "status": "failed",
                    "error": str(exc),
                }
            }
        )
        repositories.update_video_asset_status(db, video_asset_id, "issue")
        stderr = f"{result.stderr}\nTDS API Gemini kiosk summary failed: {exc}".strip()
        repositories.revise_script_run(
            db,
            result.script_run_id,
            status="failed",
            stdout_log=result.stdout,
            stderr_log=stderr,
        )
        if session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=f"TDS API Gemini kiosk summary failed: {exc}",
            )
        return ScriptExecutionResult(
            script_run_id=result.script_run_id,
            runner_job_id=result.runner_job_id,
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    if session_id is not None and completed_kiosk_summary:
        detected_total_items = int(completed_kiosk_summary.get("detected_total_items") or 0)
        repositories.finalize_session_result(
            db,
            session_id=int(session_id),
            kiosk_total_items=detected_total_items,
            tolerance=1,
            extra_result_summary={
                "kiosk_summary": completed_kiosk_summary,
            },
        )
    _apply_processed_video_upload_result(
        db,
        video_asset_row=video_asset_row,
        object_key=remote_result.processed_video_object_key,
        video_url=processed_video_url,
    )
    return result


def _finalize_remote_kiosk_match_script_run(
    db: Session,
    *,
    script_run: dict[str, Any],
    remote_result: RemoteRunnerResult,
) -> ScriptExecutionResult:
    if str(script_run.get("status") or "").strip().lower() != "running":
        return ScriptExecutionResult(
            script_run_id=int(script_run["id"]),
            runner_job_id=str(script_run.get("runner_job_id") or ""),
            script_name="kiosk_match",
            model_name=script_run.get("model_name"),
            status=str(script_run.get("status") or "success"),
            command=["runpod_serverless", "kiosk_match"],
            stdout=str(script_run.get("stdout_log") or ""),
            stderr=str(script_run.get("stderr_log") or ""),
            message="Runpod callback already processed for this script run.",
        )
    script_run_id = int(script_run["id"])
    runner_payload = dict(script_run.get("runner_payload") or {})
    session_id = int(script_run["session_id"]) if script_run.get("session_id") is not None else None
    location_id = int(runner_payload.get("location_id") or 0) if runner_payload.get("location_id") is not None else None

    remote_status = "success" if remote_result.status == "success" else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=remote_status,
        stdout_log=remote_result.stdout,
        stderr_log=remote_result.stderr,
    )
    _record_remote_runner_cost(db, script_run_id, remote_result)
    result = ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        script_name="kiosk_match",
        model_name=script_run.get("model_name"),
        status=remote_status,
        command=["runpod_serverless", "kiosk_match"],
        stdout=remote_result.stdout,
        stderr=remote_result.stderr,
    )
    if session_id is None or location_id is None:
        return result

    session_row = repositories.get_session(db, session_id)
    existing_summary = dict(session_row.get("result_summary") or {})
    pipeline = dict(existing_summary.get("session_close_pipeline") or {})
    transaction_identification = dict(remote_result.transaction_match_summary or {})
    pipeline["transaction_identification"] = transaction_identification
    existing_summary["session_close_pipeline"] = pipeline

    if result.status != "success":
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            result_summary=existing_summary,
            issue_reason=remote_result.stderr or "Kiosk transaction matching failed in the remote runner.",
        )
        return result

    transaction_results = transaction_identification.get("transactions")
    if not isinstance(transaction_results, list):
        transaction_results = []

    chosen_candidate_index = transaction_identification.get("chosen_candidate_index")
    if chosen_candidate_index is None:
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            result_summary=existing_summary,
            issue_reason="No kiosk transaction matched confidently for this session.",
        )
        return result

    matched_result = next(
        (
            row
            for row in transaction_results
            if int(row.get("candidate_index") or 0) == int(chosen_candidate_index)
        ),
        None,
    )
    if not isinstance(matched_result, dict):
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            result_summary=existing_summary,
            issue_reason="Matched kiosk transaction could not be found in candidate transactions.",
        )
        return result

    raw_payload = dict(matched_result.get("raw_payload") or {})
    raw_payload["transaction_identification"] = matched_result
    window_start = _coerce_datetime_value(raw_payload.get("window_start"))
    window_end = _coerce_datetime_value(raw_payload.get("window_end"))
    if window_start is None or window_end is None:
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            result_summary=existing_summary,
            issue_reason="Matched kiosk transaction is missing window bounds.",
        )
        return result

    repositories.delete_session_transactions(db, session_id)
    selected_transaction_id = repositories.create_transaction(
        db,
        session_id,
        {
            "receipt_number": str(matched_result.get("receipt_number") or matched_result.get("transaction_id") or ""),
            "transaction_time": _coerce_datetime_value(matched_result.get("transaction_time")),
            "total_items": int(matched_result.get("total_items") or 0),
            "total_amount": matched_result.get("total_amount"),
            "raw_payload": raw_payload,
        },
    )
    transaction_identification["chosen_session_transaction_id"] = selected_transaction_id
    pipeline["paid_transactions"] = [matched_result]
    existing_summary["session_close_pipeline"] = pipeline

    queued = retrieve_kiosk_video_window(
        db,
        session_id=session_id,
        location_id=location_id,
        start_time=window_start,
        end_time=window_end,
    )
    pipeline["selected_kiosk_windows"] = [
        {
            "start_time": window_start.isoformat(),
            "end_time": window_end.isoformat(),
        }
    ]
    pipeline["queued_kiosk_video_asset_ids"] = [int(queued.video_asset_id)]
    existing_summary["session_close_pipeline"] = pipeline
    repositories.update_session_fields(
        db,
        session_id=session_id,
        status="pending",
        transaction_total_items=int(matched_result.get("total_items") or 0),
        result_summary=existing_summary,
        issue_reason=None,
    )
    return result


def process_runpod_webhook(
    db: Session,
    *,
    kind: str,
    body: dict[str, Any],
    token: str | None = None,
) -> dict[str, Any]:
    secret = str(settings.runpod_webhook_secret or "").strip()
    if secret and token != secret:
        raise PermissionError("Invalid Runpod webhook token.")

    job_id = str(body.get("id") or body.get("jobId") or "").strip()
    script_run_id_value = body.get("script_run_id")
    script_run = None
    if job_id:
        script_run = repositories.get_script_run_by_runner_job_id(db, job_id)
    elif script_run_id_value is not None:
        try:
            script_run = repositories.get_script_run(db, int(script_run_id_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Runpod webhook payload has invalid script_run_id.") from exc
    else:
        raise ValueError("Runpod webhook payload is missing job id.")
    if script_run is None:
        raise ValueError("Runpod webhook does not match any script_run.")

    effective_body = body
    runner_job_id = str(script_run.get("runner_job_id") or job_id or "").strip()
    normalized_kind = str(kind or "").strip().lower()
    if not isinstance(body.get("output"), dict):
        if not runner_job_id:
            raise ValueError("Runpod webhook is missing runner job id for status fetch.")
        script_name = str(script_run.get("script_name") or normalized_kind or "").strip().lower()
        if script_name not in {"entry", "kiosk", "kiosk_match", "grouping"}:
            raise ValueError("Unsupported Runpod webhook kind.")
        effective_body = _fetch_runpod_status_with_retries(
            runner_job_id=runner_job_id,
            script_name=script_name,
            attempts=3,
            sleep_seconds=2.0,
        )

    runpod_status, remote_result = _remote_runner_result_from_runpod_body(effective_body)
    if not _is_runpod_terminal_status(runpod_status):
        logger.info(
            "Runpod webhook received before terminal state for job_id=%s script_run_id=%s kind=%s status=%s",
            runner_job_id or job_id,
            script_run.get("id"),
            normalized_kind,
            runpod_status or "UNKNOWN",
        )
        return {
            "ok": True,
            "job_id": runner_job_id or job_id,
            "script_run_id": int(script_run["id"]),
            "runpod_status": runpod_status,
            "status": str(script_run.get("status") or "running"),
            "message": "Runpod job is not terminal yet; reconciliation worker will retry.",
        }
    if normalized_kind == "entry":
        result = _finalize_remote_entry_script_run(db, script_run=script_run, remote_result=remote_result)
    elif normalized_kind == "kiosk":
        result = _finalize_remote_kiosk_script_run(db, script_run=script_run, remote_result=remote_result)
    elif normalized_kind == "kiosk_match":
        result = _finalize_remote_kiosk_match_script_run(db, script_run=script_run, remote_result=remote_result)
    elif normalized_kind == "grouping":
        result = _finalize_remote_grouping_script_run(db, script_run=script_run, remote_result=remote_result)
    else:
        raise ValueError("Unsupported Runpod webhook kind.")

    return {
        "ok": True,
        "job_id": job_id,
        "script_run_id": result.script_run_id,
        "runpod_status": runpod_status,
        "status": result.status,
    }


def _is_runpod_terminal_status(status: str) -> bool:
    normalized = str(status or "").strip().upper()
    return normalized in {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "TIMED_OUT", "ABORTED"}


def _fetch_runpod_status_with_retries(
    *,
    runner_job_id: str,
    script_name: str,
    attempts: int = 3,
    sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    last_body: dict[str, Any] | None = None
    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        body = _runpod_request(
            method="GET",
            path=f"/status/{quote(runner_job_id)}",
            kind=script_name,
            timeout_seconds=settings.runpod_status_timeout_seconds,
        )
        last_body = body
        runpod_status = str(body.get("status") or "").strip().upper()
        if _is_runpod_terminal_status(runpod_status):
            if attempt > 1:
                logger.info(
                    "Runpod status reached terminal state on retry %s for job_id=%s script=%s status=%s",
                    attempt,
                    runner_job_id,
                    script_name,
                    runpod_status,
                )
            return body
        if attempt < total_attempts:
            logger.info(
                "Runpod status still pending for job_id=%s script=%s attempt=%s/%s status=%s; retrying in %.1fs",
                runner_job_id,
                script_name,
                attempt,
                total_attempts,
                runpod_status or "UNKNOWN",
                sleep_seconds,
            )
            time.sleep(max(0.0, sleep_seconds))
    assert last_body is not None
    return last_body


def reconcile_running_remote_analysis_script_runs(db: Session) -> list[dict[str, Any]]:
    reconciled: list[dict[str, Any]] = []
    for script_run in repositories.list_running_remote_analysis_script_runs(db):
        job_id = str(script_run.get("runner_job_id") or "").strip()
        script_name = str(script_run.get("script_name") or "").strip().lower()
        if not job_id or script_name not in {"entry", "kiosk", "kiosk_match", "grouping"}:
            continue
        try:
            body = _runpod_request(
                method="GET",
                path=f"/status/{quote(job_id)}",
                kind=script_name,
                timeout_seconds=settings.runpod_status_timeout_seconds,
            )
        except Exception:
            continue
        runpod_status = str(body.get("status") or "").strip().upper()
        if not _is_runpod_terminal_status(runpod_status):
            continue
        remote_status, remote_result = _remote_runner_result_from_runpod_body(body)
        if script_name == "entry":
            result = _finalize_remote_entry_script_run(db, script_run=script_run, remote_result=remote_result)
        elif script_name == "kiosk_match":
            result = _finalize_remote_kiosk_match_script_run(db, script_run=script_run, remote_result=remote_result)
        elif script_name == "grouping":
            result = _finalize_remote_grouping_script_run(db, script_run=script_run, remote_result=remote_result)
        else:
            result = _finalize_remote_kiosk_script_run(db, script_run=script_run, remote_result=remote_result)
        reconciled.append(
            {
                "script_run_id": result.script_run_id,
                "runner_job_id": result.runner_job_id,
                "script_name": script_name,
                "runpod_status": remote_status,
                "status": result.status,
            }
        )
    return reconciled


def _post_remote_runner_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.runner_timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runner request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Runner request failed: {exc}") from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Runner returned invalid JSON: {response_text[:1000]}") from exc


def _download_remote_json(url: str) -> dict[str, Any]:
    request = Request(url, method="GET")
    with urlopen(request, timeout=settings.runner_timeout_seconds) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    payload = json.loads(response_text)
    if not isinstance(payload, dict):
        raise RuntimeError("Remote JSON artifact did not contain an object.")
    return payload


def _resolve_grouping_summary_from_remote_result(
    remote_result: RemoteRunnerResult,
) -> tuple[dict[str, Any], str | None]:
    grouping_summary = dict(remote_result.grouping_summary or {})
    meta = dict(remote_result.meta or {})
    summary_url = str(meta.get("grouping_summary_url") or "").strip()
    summary_object_key = str(meta.get("grouping_summary_object_key") or "").strip()
    if not summary_url and summary_object_key:
        try:
            summary_url = generate_public_object_url(summary_object_key)
        except Exception:
            logger.exception("Could not build public grouping summary URL for %s", summary_object_key)
    if not summary_url:
        return grouping_summary, None
    try:
        full_summary = _download_remote_json(summary_url)
        full_summary.setdefault("artifact_url", summary_url)
        if summary_object_key:
            full_summary.setdefault("artifact_object_key", summary_object_key)
        return full_summary, None
    except Exception as exc:
        error = f"Could not load grouping summary artifact: {exc}"
        logger.exception("Could not load grouping summary artifact url=%s", summary_url)
        grouping_summary["artifact_url"] = summary_url
        if summary_object_key:
            grouping_summary["artifact_object_key"] = summary_object_key
        grouping_summary["artifact_fetch_error"] = error
        return grouping_summary, error


def _group_trigger_id_list(value: Any) -> list[int]:
    raw_values = value if isinstance(value, list) else ([] if value is None else [value])
    trigger_ids: list[int] = []
    for raw_value in raw_values:
        try:
            trigger_id = int(raw_value)
        except (TypeError, ValueError):
            continue
        if trigger_id not in trigger_ids:
            trigger_ids.append(trigger_id)
    return trigger_ids


def _person_frame_urls_by_trigger(grouping_summary: Mapping[str, Any]) -> dict[int, list[str]]:
    frames_by_trigger: dict[int, list[str]] = {}
    diagnostics = grouping_summary.get("diagnostics")
    if not isinstance(diagnostics, list):
        return frames_by_trigger
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        try:
            trigger_id = int(diagnostic.get("trigger_id"))
        except (TypeError, ValueError):
            continue
        frame_urls: list[str] = []
        for sample in diagnostic.get("samples") or []:
            if not isinstance(sample, Mapping):
                continue
            try:
                person_count = int(sample.get("person_count") or 0)
            except (TypeError, ValueError):
                person_count = 0
            image_url = str(sample.get("image_url") or "").strip()
            if person_count > 0 and image_url and image_url not in frame_urls:
                frame_urls.append(image_url)
        if frame_urls:
            frames_by_trigger[trigger_id] = frame_urls[:6]
    return frames_by_trigger


def _repair_grouping_with_gemini(
    db: Session,
    *,
    parent_script_run_id: int | None,
    batch_id: int | None = None,
    location_id: int | None = None,
    grouping_summary: dict[str, Any],
) -> dict[str, Any]:
    existing_repair = grouping_summary.get("gemini_repair")
    if isinstance(existing_repair, Mapping) and existing_repair.get("status"):
        return grouping_summary
    groups = grouping_summary.get("groups") or grouping_summary.get("Groups") or []
    if not isinstance(groups, list):
        return grouping_summary
    unknown_trigger_ids = _group_trigger_id_list(grouping_summary.get("unknown"))
    completed_groups: list[dict[str, Any]] = []
    open_groups: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, Mapping):
            continue
        normalized_group = dict(group)
        normalized_group.setdefault("group_id", index)
        entry_ids = _group_trigger_id_list(normalized_group.get("entry"))
        exit_ids = _group_trigger_id_list(normalized_group.get("exit"))
        if entry_ids and exit_ids:
            completed_groups.append(normalized_group)
        elif entry_ids:
            open_groups.append(normalized_group)

    if not open_groups and not unknown_trigger_ids:
        return grouping_summary

    frames_by_trigger = _person_frame_urls_by_trigger(grouping_summary)
    candidate_trigger_ids: list[int] = []
    for group in open_groups:
        for trigger_id in _group_trigger_id_list(group.get("entry")):
            if trigger_id not in candidate_trigger_ids:
                candidate_trigger_ids.append(trigger_id)
    for trigger_id in unknown_trigger_ids:
        if trigger_id not in candidate_trigger_ids:
            candidate_trigger_ids.append(trigger_id)

    image_urls: list[str] = []
    image_notes: list[dict[str, Any]] = []
    for trigger_id in candidate_trigger_ids:
        for image_url in frames_by_trigger.get(trigger_id, [])[:4]:
            if image_url in image_urls:
                continue
            image_urls.append(image_url)
            image_notes.append(
                {
                    "image_number": len(image_urls),
                    "trigger_id": trigger_id,
                    "role_hint": "open_entry" if any(trigger_id in _group_trigger_id_list(group.get("entry")) for group in open_groups) else "unknown",
                }
            )

    if len(candidate_trigger_ids) < 2 or not image_urls:
        grouping_summary["gemini_repair"] = {
            "status": "skipped",
            "reason": "not_enough_problematic_person_frames",
            "candidate_trigger_ids": candidate_trigger_ids,
        }
        return grouping_summary

    repair_script_run_id = _create_gemini_script_run(
        db,
        session_id=None,
        trigger_id=None,
        script_name="grouping",
        model_name="gemini_grouping_repair",
        runner_payload={
            "batch_id": batch_id,
            "location_id": location_id,
            "parent_script_run_id": parent_script_run_id,
            "candidate_trigger_ids": candidate_trigger_ids,
            "image_count": len(image_urls),
        },
    )
    prompt = (
        "You are repairing retail entrance/exit grouping. Each trigger is a door event. "
        "A group should identify the same person across triggers. A trigger must never be both entry and exit in the same group. "
        "Use the image-number mapping to compare people by clothing, body shape, bags, and direction. "
        "Direction is important: walking into the store means entry; walking out of or away from the store means exit. "
        "If clear walking direction conflicts with simple timestamp assumptions, prioritize the walking direction. "
        "Return strict JSON only with schema: "
        '{"groups":[{"entry":[integer],"exit":[integer],"confidence":number,"reason":string}],'
        '"unknown":[integer],"notes":[string]}. '
        f"Open entry groups needing exit: {json.dumps([{'group_id': group.get('group_id'), 'entry': _group_trigger_id_list(group.get('entry'))} for group in open_groups])}. "
        f"Unknown triggers: {json.dumps(unknown_trigger_ids)}. "
        f"Image mapping: {json.dumps(image_notes)}. "
        "Only create an exit match if the same person is clearly visible. If unsure, leave the trigger in unknown."
    )
    try:
        repair_result, repair_meta = _call_kiosk_gemini_summary(prompt=prompt, image_urls=image_urls)
        repair_cost = _record_gemini_cost(db, repair_script_run_id, repair_meta)
    except Exception as exc:
        repositories.finish_script_run(
            db,
            repair_script_run_id,
            status="failed",
            stdout_log=json.dumps(
                {
                    "candidate_trigger_ids": candidate_trigger_ids,
                    "image_count": len(image_urls),
                    "image_mapping": image_notes,
                },
                indent=2,
                default=str,
            ),
            stderr_log=str(exc),
        )
        grouping_summary["gemini_repair"] = {
            "status": "failed",
            "script_run_id": repair_script_run_id,
            "error": str(exc),
            "candidate_trigger_ids": candidate_trigger_ids,
            "image_urls": image_urls,
        }
        logger.exception("Gemini grouping repair failed")
        return grouping_summary

    repaired_groups: list[dict[str, Any]] = []
    consumed_trigger_ids: set[int] = set()
    next_group_id = len(completed_groups) + 1
    for repaired in repair_result.get("groups") or []:
        if not isinstance(repaired, Mapping):
            continue
        entry_ids = _group_trigger_id_list(repaired.get("entry"))
        exit_ids = _group_trigger_id_list(repaired.get("exit"))
        if not entry_ids:
            continue
        entry_ids = entry_ids[:1]
        exit_ids = [trigger_id for trigger_id in exit_ids if trigger_id not in entry_ids]
        if not exit_ids:
            continue
        confidence = _coerce_number(repaired.get("confidence"), 0.0)
        if confidence < 0.75:
            continue
        group_payload = {
            "group_id": next_group_id,
            "entry": entry_ids,
            "exit": exit_ids[:1],
            "score": confidence,
            "repair_source": "gemini",
            "reason": str(repaired.get("reason") or "gemini_grouping_repair"),
        }
        next_group_id += 1
        repaired_groups.append(group_payload)
        consumed_trigger_ids.update(entry_ids)
        consumed_trigger_ids.update(exit_ids[:1])

    repositories.finish_script_run(
        db,
        repair_script_run_id,
        status="success",
        stdout_log=json.dumps(
            {
                "candidate_trigger_ids": candidate_trigger_ids,
                "image_mapping": image_notes,
                "result": repair_result,
                "applied_groups": repaired_groups,
                "remaining_unknown": sorted(
                    {
                        trigger_id
                        for trigger_id in candidate_trigger_ids + unknown_trigger_ids
                        if trigger_id not in consumed_trigger_ids
                    }
                ),
            },
            indent=2,
            default=str,
        ),
        stderr_log="",
    )
    remaining_open_entries = sorted(
        {
            trigger_id
            for group in open_groups
            for trigger_id in _group_trigger_id_list(group.get("entry"))
            if trigger_id not in consumed_trigger_ids
        }
    )
    repaired_unknown = set(_group_trigger_id_list(repair_result.get("unknown")))
    remaining_unknown = sorted(
        {
            trigger_id
            for trigger_id in candidate_trigger_ids + unknown_trigger_ids
            if trigger_id not in consumed_trigger_ids
        }
        | {trigger_id for trigger_id in repaired_unknown if trigger_id not in consumed_trigger_ids}
    )
    grouping_summary["groups"] = completed_groups + repaired_groups
    grouping_summary["open_entries"] = sorted(
        set(_group_trigger_id_list(grouping_summary.get("open_entries"))) | set(remaining_open_entries)
    )
    grouping_summary["unknown"] = remaining_unknown
    grouping_summary["gemini_repair"] = {
        "status": "success",
        "script_run_id": repair_script_run_id,
        "input": {
            "open_groups": open_groups,
            "unknown": unknown_trigger_ids,
            "image_mapping": image_notes,
        },
        "result": repair_result,
        "cost": repair_cost,
        "applied_group_count": len(repaired_groups),
    }
    return grouping_summary


def _persist_grouping_items_from_summary(
    db: Session,
    *,
    batch_id: int,
    grouping_summary: Mapping[str, Any],
) -> None:
    groups = grouping_summary.get("groups") or grouping_summary.get("Groups") or []
    grouped_trigger_ids: set[int] = set()
    if isinstance(groups, list):
        for group_index, group in enumerate(groups, start=1):
            if not isinstance(group, Mapping):
                continue
            group_key = str(group.get("group_id") or group.get("id") or group_index)
            for role in ("entry", "exit"):
                trigger_ids = group.get(role) or []
                if not isinstance(trigger_ids, list):
                    continue
                for trigger_id in trigger_ids:
                    try:
                        normalized_trigger_id = int(trigger_id)
                        grouped_trigger_ids.add(normalized_trigger_id)
                        repositories.upsert_grouping_item(
                            db,
                            batch_id=batch_id,
                            trigger_id=normalized_trigger_id,
                            video_asset_id=None,
                            group_key=group_key,
                            role=role,
                            status="grouped",
                            score=float(group.get("score")) if group.get("score") is not None else None,
                            result_payload=dict(group),
                        )
                    except Exception:
                        logger.exception("Could not persist grouping item batch_id=%s trigger_id=%s", batch_id, trigger_id)
    open_entries = grouping_summary.get("open_entries") or []
    if isinstance(open_entries, list):
        for trigger_id in open_entries:
            try:
                normalized_trigger_id = int(trigger_id)
                if normalized_trigger_id in grouped_trigger_ids:
                    continue
                repositories.upsert_grouping_item(
                    db,
                    batch_id=batch_id,
                    trigger_id=normalized_trigger_id,
                    video_asset_id=None,
                    group_key=None,
                    role="unknown",
                    status="unknown",
                    result_payload={"reason": "open_entry_waiting_for_exit"},
                )
            except Exception:
                logger.exception("Could not persist open grouping item batch_id=%s trigger_id=%s", batch_id, trigger_id)
    unknown = grouping_summary.get("unknown") or []
    if isinstance(unknown, list):
        for trigger_id in unknown:
            try:
                normalized_trigger_id = int(trigger_id)
                if normalized_trigger_id in grouped_trigger_ids:
                    continue
                repositories.upsert_grouping_item(
                    db,
                    batch_id=batch_id,
                    trigger_id=normalized_trigger_id,
                    video_asset_id=None,
                    group_key=None,
                    role="unknown",
                    status="unknown",
                    result_payload={"reason": "runner_or_repair_returned_unknown"},
                )
            except Exception:
                logger.exception("Could not persist unknown grouping item batch_id=%s trigger_id=%s", batch_id, trigger_id)


def _invoke_remote_runner(
    *,
    endpoint: str,
    payload: dict[str, Any],
) -> RemoteRunnerResult:
    base_url = str(settings.runner_base_url or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("Remote runner base URL is not configured.")
    body = _post_remote_runner_json(f"{base_url}{endpoint}", payload)
    return RemoteRunnerResult(
        status=str(body.get("status") or "failed"),
        stdout=str(body.get("stdout") or ""),
        stderr=str(body.get("stderr") or ""),
        processed_video_object_key=body.get("processed_video_object_key"),
        processed_video_url=body.get("processed_video_url"),
        tracking_summary=body.get("tracking_summary"),
        reid_views_summary=body.get("reid_views_summary"),
        kiosk_summary=body.get("kiosk_summary"),
        transaction_match_summary=body.get("transaction_match_summary"),
        grouping_summary=body.get("grouping_summary"),
        meta=body.get("meta") if isinstance(body.get("meta"), dict) else None,
    )


def _runpod_endpoint_id(kind: str | None = None) -> str:
    normalized_kind = str(kind or "").strip().lower()
    endpoint_id = ""
    if normalized_kind == "entry":
        endpoint_id = str(settings.runpod_entry_endpoint_id or "").strip()
    elif normalized_kind in {"kiosk", "kiosk_match"}:
        endpoint_id = str(settings.runpod_kiosk_endpoint_id or "").strip()
    elif normalized_kind == "grouping":
        endpoint_id = str(settings.runpod_grouping_endpoint_id or "").strip()
    if not endpoint_id:
        endpoint_id = str(settings.runpod_endpoint_id or "").strip()
    if not endpoint_id:
        raise RuntimeError("Runpod endpoint id is not configured.")
    return endpoint_id


def _runpod_endpoint_url(path: str, *, kind: str | None = None) -> str:
    endpoint_id = _runpod_endpoint_id(kind)
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"https://api.runpod.ai/v2/{quote(endpoint_id)}{normalized_path}"


def _runpod_request(
    *,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    kind: str | None = None,
    timeout_seconds: int | float | None = None,
) -> dict[str, Any]:
    api_key = str(settings.runpod_api_key or "").strip()
    if not api_key:
        raise RuntimeError("Runpod API key is not configured.")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        _runpod_endpoint_url(path, kind=kind),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method=method,
    )
    request_timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else settings.runner_timeout_seconds
    )
    try:
        with urlopen(request, timeout=request_timeout) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Runpod request failed: {exc}") from exc
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Runpod returned invalid JSON: {response_text[:1000]}") from exc


def _build_runpod_webhook_url(kind: str) -> str:
    base_url = str(settings.runpod_webhook_base_url or "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("Runpod webhook base URL is not configured.")
    webhook_url = f"{base_url}/api/v1/runpod/webhooks/{quote(kind)}"
    secret = str(settings.runpod_webhook_secret or "").strip()
    if secret:
        webhook_url = f"{webhook_url}?token={quote(secret, safe='')}"
    return webhook_url


def _enqueue_runpod_runner(
    *,
    kind: str,
    payload: dict[str, Any],
) -> RunpodEnqueueResult:
    enqueue_body = _runpod_request(
        method="POST",
        path="/run",
        payload={
            "input": payload,
            "webhook": _build_runpod_webhook_url(kind),
        },
        kind=kind,
        timeout_seconds=settings.runpod_enqueue_timeout_seconds,
    )
    job_id = str(enqueue_body.get("id") or "").strip()
    if not job_id:
        raise RuntimeError(f"Runpod did not return a job id: {enqueue_body}")
    return RunpodEnqueueResult(job_id=job_id)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _is_under_tmp_media_root(path: Path) -> bool:
    try:
        resolved = path.resolve()
        root = tmp_media_root().resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _expected_processed_video_path(video_path: str, output_dir: Path) -> Path:
    stem = Path(video_path).stem
    if stem.endswith("_output"):
        return output_dir / f"{stem}.mp4"
    return output_dir / f"{stem}_output.mp4"


def _resolve_processed_video_path(video_path: str, output_dir: Path) -> Path | None:
    expected_path = _expected_processed_video_path(video_path, output_dir)
    if expected_path.exists():
        return expected_path

    stem = Path(video_path).stem
    candidate_names = [
        f"{stem}_output.mp4",
        f"{stem}.mp4",
    ]
    for candidate_name in candidate_names:
        candidate_path = output_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    candidates = sorted(
        path
        for path in output_dir.glob(f"{stem}*.mp4")
        if path.is_file()
    )
    if candidates:
        return candidates[0]

    return None


def _tracking_summary_path(video_path: str, output_dir: Path) -> Path:
    return output_dir / f"{Path(video_path).stem}_tracking_summary.json"


def _reid_views_summary_path(video_path: str, output_dir: Path) -> Path:
    return output_dir / f"{Path(video_path).stem}_reid_views_summary.json"


def _load_tracking_summary(video_path: str, output_dir: Path) -> dict[str, Any]:
    summary_path = _tracking_summary_path(video_path, output_dir)
    if not summary_path.exists():
        raise FileNotFoundError(f"Tracking summary not found at {summary_path}")
    return json.loads(summary_path.read_text())


def _load_reid_views_summary(video_path: str, output_dir: Path) -> dict[str, Any]:
    summary_path = _reid_views_summary_path(video_path, output_dir)
    if not summary_path.exists():
        return {"customers": []}
    return json.loads(summary_path.read_text())


def _customer_event_time(
    *,
    customer: dict[str, Any],
    tracking_summary: dict[str, Any],
    captured_start_time: datetime | None,
    fallback_time: datetime | None,
    event_key: str,
) -> datetime | None:
    if captured_start_time is None:
        return fallback_time

    offset_value = customer.get(f"{event_key}_time_offset_seconds")
    if offset_value is None:
        frame_value = customer.get(f"{event_key}_frame")
        fps_value = tracking_summary.get("fps")
        try:
            if frame_value is not None and fps_value is not None and float(fps_value) > 0:
                offset_value = float(frame_value) / float(fps_value)
        except (TypeError, ValueError, ZeroDivisionError):
            offset_value = None

    try:
        if offset_value is not None:
            return captured_start_time + timedelta(seconds=max(0.0, float(offset_value)))
    except (TypeError, ValueError):
        pass
    return fallback_time


def _load_cross_state_pickle(gallery_state_path: Path) -> dict[str, Any]:
    import pickle

    if not gallery_state_path.exists():
        return {"next_gid": 1, "persistent_gallery": {}, "persistent_gallery_view_paths": {}}
    with gallery_state_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        return {"next_gid": 1, "persistent_gallery": {}, "persistent_gallery_view_paths": {}}
    data.setdefault("persistent_gallery", {})
    data.setdefault("persistent_gallery_view_paths", {})
    return data


def _write_cross_state_pickle(gallery_state_path: Path, cross_state: dict[str, Any]) -> None:
    import pickle

    gallery_state_path.parent.mkdir(parents=True, exist_ok=True)
    with gallery_state_path.open("wb") as handle:
        pickle.dump(cross_state, handle)


def _tensor_like_to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return None


def _float_list_to_tensor(value: Any):
    import torch

    floats = _tensor_like_to_float_list(value)
    if not floats:
        return None
    return torch.tensor(floats, dtype=torch.float32)


def _combine_fashion_embedding(upper: Any, lower: Any) -> list[float] | None:
    upper_list = _tensor_like_to_float_list(upper)
    lower_list = _tensor_like_to_float_list(lower)
    if upper_list and lower_list:
        return upper_list + lower_list
    return upper_list or lower_list


def _split_fashion_embedding(value: Any) -> tuple[list[float] | None, list[float] | None]:
    combined = _tensor_like_to_float_list(value)
    if not combined:
        return None, None
    if len(combined) < 2 or len(combined) % 2 != 0:
        return combined, None
    midpoint = len(combined) // 2
    return combined[:midpoint], combined[midpoint:]


def _merge_metadata(base: Any, extra: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(base) if isinstance(base, Mapping) else {}
    payload.update(extra)
    return payload


def _resolve_session_effective_end_time(db: Session, session_id: int) -> datetime | None:
    session = repositories.get_session(db, session_id)
    end_time = _coerce_datetime_value(session.get("end_time"))
    if end_time is not None:
        return end_time
    session_customers = repositories.list_session_customers(db, session_id)
    leave_times = [
        _coerce_datetime_value(row.get("leave_time"))
        for row in session_customers
        if row.get("leave_time") is not None
    ]
    leave_times = [value for value in leave_times if value is not None]
    if leave_times:
        return max(leave_times)
    return _coerce_datetime_value(session.get("start_time"))


def _time_window_overlap_seconds(
    first_start: datetime | None,
    first_end: datetime | None,
    second_start: datetime | None,
    second_end: datetime | None,
) -> float:
    if first_start is None or first_end is None or second_start is None or second_end is None:
        return 0.0
    overlap_start = max(first_start, second_start)
    overlap_end = min(first_end, second_end)
    return max(0.0, (overlap_end - overlap_start).total_seconds())


def _seconds_to_nearest_transaction(db: Session, session_id: int, anchor_time: datetime) -> float:
    transaction_rows = repositories.list_session_transactions(db, session_id)
    transaction_times = [
        _coerce_datetime_value(row.get("transaction_time"))
        for row in transaction_rows
        if row.get("transaction_time") is not None
    ]
    transaction_times = [value for value in transaction_times if value is not None]
    if not transaction_times:
        return float("inf")
    return min(abs((value - anchor_time).total_seconds()) for value in transaction_times)


def _capture_snapshot_frame(
    *,
    location_id: int,
    session_id: int,
    section: str,
    recorder_channel: str,
    start_time: datetime,
    delayed_seconds: int,
    snapshot_path: Path,
) -> dict[str, Any]:
    session_db = TransactionalSessionLocal()
    try:
        location = repositories.get_location_endpoint(session_db, location_id)
    finally:
        session_db.close()
    adjusted_start = start_time - timedelta(seconds=delayed_seconds)
    adjusted_end = adjusted_start + timedelta(seconds=2)
    rtsp_url = _build_dahua_rtsp_playback_url(
        host=str(location.get("dahua_host") or "").strip(),
        username=str(location.get("dahua_username") or "").strip(),
        password=decrypt_secret(str(location.get("dahua_password_encrypted") or "").strip()),
        rtsp_port=int(location.get("rtsp_port") or settings.dahua_rtsp_port),
        channel=recorder_channel,
        start_time=adjusted_start,
        end_time=adjusted_end,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.ffmpeg_bin,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(snapshot_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "status": "ok" if completed.returncode == 0 and snapshot_path.exists() else "failed",
        "stderr": str(completed.stderr or "").strip()[-1000:],
    }


def _build_kiosk_transaction_match_manifest(
    db: Session,
    *,
    session_id: int,
    location_id: int,
    transaction_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    session_customers = repositories.list_session_customers(db, session_id)
    session_customer_ids = [
        int(row["id"])
        for row in session_customers
        if row.get("id") is not None
    ]
    vector_db = VectorSessionLocal()
    try:
        history_rows = vector_repositories.list_history_gallery_records(
            vector_db,
            location_id=location_id,
            session_customer_ids=session_customer_ids,
            limit=500,
        )
    finally:
        vector_db.close()

    cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section="kiosk")
    recorder_channel = str(cctv.get("recorder_channel") or "").strip()
    if not recorder_channel:
        raise ValueError("Kiosk CCTV record does not have a recorder_channel.")
    delayed_seconds = int(cctv.get("delayed_seconds") or 0)

    snapshot_root = build_session_workdir(location_id, session_id) / "kiosk" / "transaction_match_inputs"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    transactions_payload: list[dict[str, Any]] = []
    for transaction in transaction_summaries:
        receipt_number = str(transaction.get("receipt_number") or transaction.get("transaction_id") or "transaction")
        window_start = _coerce_datetime_value(transaction.get("window_start"))
        window_end = _coerce_datetime_value(transaction.get("window_end"))
        if window_start is None or window_end is None:
            continue
        total_seconds = max(1.0, (window_end - window_start).total_seconds())
        sample_count = 10
        sample_offsets = [total_seconds * (index / max(1, sample_count - 1)) for index in range(sample_count)]
        samples_payload: list[dict[str, Any]] = []
        safe_receipt = re.sub(r"[^A-Za-z0-9._-]+", "_", receipt_number) or "transaction"
        for index, offset_seconds in enumerate(sample_offsets, start=1):
            sample_time = window_start + timedelta(seconds=float(offset_seconds))
            snapshot_path = snapshot_root / safe_receipt / f"sample_{index:02d}_{sample_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            capture_meta = _capture_snapshot_frame(
                location_id=location_id,
                session_id=session_id,
                section="kiosk",
                recorder_channel=recorder_channel,
                start_time=sample_time,
                delayed_seconds=delayed_seconds,
                snapshot_path=snapshot_path,
            )
            if capture_meta["status"] != "ok":
                samples_payload.append(
                    {
                        "sample_index": index,
                        "sample_time": sample_time.isoformat(),
                        "status": "failed",
                        "stderr": capture_meta["stderr"],
                    }
                )
                continue
            object_key, image_url = _upload_runner_input_file(
                snapshot_path,
                kind="kiosk_match_image",
                location_id=location_id,
                session_id=session_id,
                trigger_id=None,
                section="kiosk",
            )
            samples_payload.append(
                {
                    "sample_index": index,
                    "sample_time": sample_time.isoformat(),
                    "status": "ok",
                    "image_object_key": object_key,
                    "image_url": image_url,
                }
            )
        transactions_payload.append(
            {
                "candidate_index": transaction.get("candidate_index"),
                "transaction_id": transaction.get("transaction_id"),
                "receipt_number": transaction.get("receipt_number"),
                "transaction_time": transaction.get("transaction_time"),
                "total_items": transaction.get("total_items"),
                "total_amount": transaction.get("total_amount"),
                "window_start": transaction.get("window_start"),
                "window_end": transaction.get("window_end"),
                "raw_payload": transaction.get("raw_payload"),
                "samples": samples_payload,
            }
        )

    return {
        "session_id": session_id,
        "location_id": location_id,
        "min_score": 0.74,
        "min_margin": 0.03,
        "target_history_rows": [
            {
                "history_gallery_id": row.get("id"),
                "session_customer_id": row.get("session_customer_id"),
                "embedding_osnet": row.get("embedding_osnet"),
                "embedding_fashion": row.get("embedding_fashion"),
            }
            for row in history_rows
        ],
        "transactions": transactions_payload,
    }


def _queue_kiosk_transaction_match_for_session(
    db: Session,
    *,
    session_id: int,
    location_id: int,
    session_close_summary: dict[str, Any],
) -> dict[str, Any]:
    pipeline = dict(session_close_summary.get("session_close_pipeline") or {})
    transaction_rows = list(pipeline.get("paid_transactions") or [])
    manifest_payload = _build_kiosk_transaction_match_manifest(
        db,
        session_id=session_id,
        location_id=location_id,
        transaction_summaries=transaction_rows,
    )
    manifest_path = build_session_workdir(location_id, session_id) / "kiosk" / "transaction_match_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str))
    manifest_object_key, manifest_url = _upload_runner_input_file(
        manifest_path,
        kind="kiosk_match_manifest",
        location_id=location_id,
        session_id=session_id,
        trigger_id=None,
        section="kiosk",
    )
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=None,
        script_name="kiosk_match",
        model_name="runpod_runner",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    enqueue_result = _enqueue_runpod_runner(
        kind="kiosk_match",
        payload={
            "kind": "kiosk_match",
            "manifest_url": manifest_url,
            "callback_url": _build_runpod_webhook_url("kiosk_match"),
            "script_run_id": script_run_id,
        },
    )
    runner_payload = {
        "session_id": session_id,
        "location_id": location_id,
        "manifest_object_key": manifest_object_key,
        "manifest_url": manifest_url,
    }
    repositories.assign_script_run_runner_job(
        db,
        script_run_id,
        runner_job_id=enqueue_result.job_id,
        runner_payload=runner_payload,
    )
    return {
        "method": "runpod_snapshot_embedding_match",
        "status": "running",
        "script_run_id": script_run_id,
        "runner_job_id": enqueue_result.job_id,
        "manifest_object_key": manifest_object_key,
        "manifest_url": manifest_url,
    }


def _time_value_to_parts(value: Any) -> tuple[int, int, int]:
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return int(value.hour), int(value.minute), int(getattr(value, "second", 0) or 0)
    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds())
        return (total_seconds // 3600) % 24, (total_seconds // 60) % 60, total_seconds % 60
    text_value = str(value or "00:00:00")
    parts = [int(part) for part in text_value.split(":")[:3]]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _combine_local_datetime(day: datetime, value: Any) -> datetime:
    hour, minute, second = _time_value_to_parts(value)
    return day.replace(hour=hour, minute=minute, second=second, microsecond=0)


def _time_period_zoneinfo() -> ZoneInfo:
    try:
        return ZoneInfo(settings.time_period_timezone)
    except ZoneInfoNotFoundError:
        logger.warning(
            "Invalid THEFT_API_TIME_PERIOD_TIMEZONE=%s; falling back to Asia/Kuala_Lumpur",
            settings.time_period_timezone,
        )
        return ZoneInfo("Asia/Kuala_Lumpur")


def _time_period_now() -> datetime:
    return datetime.now(_time_period_zoneinfo()).replace(tzinfo=None, microsecond=0)


def _to_time_period_local_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=None, microsecond=0)
    return value.astimezone(_time_period_zoneinfo()).replace(tzinfo=None, microsecond=0)


def _last_completed_period_window(period: Mapping[str, Any], *, now: datetime | None = None) -> tuple[datetime, datetime]:
    current = _to_time_period_local_naive(now) if now is not None else _time_period_now()
    today = current.replace(hour=0, minute=0, second=0, microsecond=0)
    start_today = _combine_local_datetime(today, period.get("start_time"))
    end_today = _combine_local_datetime(today, period.get("end_time"))
    if end_today <= start_today:
        end_today += timedelta(days=1)
    if current >= end_today:
        return start_today, end_today
    return start_today - timedelta(days=1), end_today - timedelta(days=1)


def _period_window_for_local_datetime(period: Mapping[str, Any], value: datetime) -> tuple[datetime, datetime] | None:
    current = value.replace(tzinfo=None, microsecond=0)
    day = current.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = _combine_local_datetime(day, period.get("start_time"))
    window_end = _combine_local_datetime(day, period.get("end_time"))
    if window_end <= window_start:
        if current < window_end:
            window_start -= timedelta(days=1)
        else:
            window_end += timedelta(days=1)
    if window_start <= current < window_end:
        return window_start, window_end
    return None


def _period_window_for_datetime(period: Mapping[str, Any], value: datetime) -> tuple[datetime, datetime] | None:
    return _period_window_for_local_datetime(period, _to_time_period_local_naive(value))


def _period_code_for_datetime(db: Session, location_id: int, value: datetime | None) -> str | None:
    if value is None:
        return None
    periods = repositories.list_filter_time_periods(db, selected_only=False)
    scoped_periods = [
        period
        for period in periods
        if period.get("location_id") is None or int(period["location_id"]) == int(location_id)
    ]
    scoped_periods.sort(key=lambda period: 0 if period.get("location_id") is not None else 1)
    for period in scoped_periods:
        if _period_window_for_datetime(period, value) is not None:
            return str(period.get("period_code") or "period")
    return None


def _selected_grouping_periods_for_location(periods: list[dict[str, Any]], location_id: int) -> list[dict[str, Any]]:
    local_periods = [
        period
        for period in periods
        if period.get("location_id") is not None and int(period["location_id"]) == int(location_id)
    ]
    candidate_periods = local_periods if local_periods else [
        period for period in periods if period.get("location_id") is None
    ]
    return [
        period
        for period in candidate_periods
        if bool(period.get("selected"))
    ]


def _grouping_time_from_trigger_frame_asset(row: Mapping[str, Any]) -> datetime | None:
    trigger_time = _coerce_datetime_value(row.get("trigger_time"))
    if trigger_time is not None:
        # trigger_event.trigger_time is the store event time used for period grouping.
        return trigger_time.replace(tzinfo=None, microsecond=0)
    frame_start_time = _coerce_datetime_value(row.get("frame_asset_start_time"))
    return frame_start_time.replace(tzinfo=None, microsecond=0) if frame_start_time is not None else None


def _frame_urls_from_video_asset(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("video_asset_metadata")
    if not isinstance(metadata, Mapping):
        return []
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        return []
    payload: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        image_url = str(frame.get("image_url") or "").strip()
        if not image_url:
            continue
        payload.append(
            {
                "index": frame.get("index"),
                "sample_time": frame.get("sample_time"),
                "offset_seconds": frame.get("offset_seconds"),
                "image_object_key": frame.get("image_object_key"),
                "image_url": image_url,
            }
        )
    return payload


def _frame_urls_from_trigger_frame_asset(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = row.get("trigger_frames")
    if not isinstance(frames, list):
        return []
    payload: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        image_url = str(frame.get("image_url") or "").strip()
        if not image_url:
            continue
        payload.append(
            {
                "index": frame.get("frame_index"),
                "sample_time": frame.get("sample_time"),
                "image_url": image_url,
            }
        )
    return payload


def _grouping_frames_per_trigger() -> int:
    return max(1, int(settings.grouping_gemini_frames_per_trigger or 5))


def _grouping_gemini_max_images_per_request() -> int:
    return max(1, int(settings.grouping_gemini_max_images_per_request or 90))


def _grouping_gemini_resize_scale() -> float | None:
    scale = _coerce_number(settings.grouping_gemini_image_scale, 1.0)
    return scale if 0 < scale < 1 else None


def _first_trigger_frame_payload(frames: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    if limit is None:
        limit = _grouping_frames_per_trigger()
    return frames[: max(1, int(limit))]


def prepare_due_grouping_batches(db: Session) -> list[dict[str, Any]]:
    periods = repositories.list_filter_time_periods(db, selected_only=False)
    if not periods:
        return []
    locations = repositories.list_locations(db)
    location_ids = [int(row["id"]) for row in locations if row.get("id") is not None]
    prepared: list[dict[str, Any]] = []
    current = _time_period_now()
    for location_id in location_ids:
        selected_periods = _selected_grouping_periods_for_location(periods, location_id)
        if not selected_periods:
            continue
        ready_assets = repositories.list_manual_grouping_ready_trigger_frame_assets(
            db,
            location_id=location_id,
            limit=1000,
        )
        if not ready_assets:
            continue
        for period in selected_periods:
            assets_by_window: dict[tuple[datetime, datetime], list[dict[str, Any]]] = {}
            for row in ready_assets:
                grouping_time = _grouping_time_from_trigger_frame_asset(row)
                if grouping_time is None:
                    continue
                window = _period_window_for_local_datetime(period, grouping_time)
                if window is None:
                    continue
                window_start, window_end = window
                if window_end > current:
                    continue
                assets_by_window.setdefault(window, []).append(row)

            for (window_start, window_end), _ready_rows in sorted(assets_by_window.items()):
                carryover_start = window_start - timedelta(hours=1)
                stale_count = repositories.mark_stale_open_entry_frame_assets_issue(
                    db,
                    location_id=location_id,
                    cutoff_time=carryover_start,
                )
                if stale_count:
                    logger.info(
                        "Marked stale open entry frame assets issue location_id=%s cutoff_time=%s count=%s",
                        location_id,
                        carryover_start,
                        stale_count,
                    )
                ready_trigger_ids = {int(row["trigger_id"]) for row in _ready_rows if row.get("trigger_id") is not None}
                trigger_assets = list(_ready_rows)
                for row in ready_assets:
                    try:
                        trigger_id = int(row["trigger_id"])
                    except (TypeError, ValueError):
                        continue
                    if trigger_id in ready_trigger_ids:
                        continue
                    grouping_time = _grouping_time_from_trigger_frame_asset(row)
                    if grouping_time is None:
                        continue
                    if carryover_start <= grouping_time < window_start:
                        trigger_assets.append(row)
                        ready_trigger_ids.add(trigger_id)
                period_code = str(period.get("period_code") or "period")
                existing = repositories.get_grouping_batch_by_window(
                    db,
                    location_id=location_id,
                    period_code=period_code,
                    window_start=window_start,
                    window_end=window_end,
                )
                if existing is not None:
                    continue
                if not trigger_assets:
                    continue
                batch = repositories.create_grouping_batch(
                    db,
                    location_id=location_id,
                    period_code=period_code,
                    window_start=window_start,
                    window_end=window_end,
                )
                for row in trigger_assets:
                    repositories.upsert_grouping_item(
                        db,
                        batch_id=int(batch["id"]),
                        trigger_id=int(row["trigger_id"]),
                        video_asset_id=None,
                        frame_payload={"frames": _first_trigger_frame_payload(_frame_urls_from_trigger_frame_asset(row))},
                    )
                prepared.append(batch)
                logger.info(
                    "Prepared grouping catch-up batch location_id=%s period_code=%s window_start=%s window_end=%s trigger_count=%s batch_id=%s",
                    location_id,
                    period_code,
                    window_start,
                    window_end,
                    len(trigger_assets),
                    batch.get("id"),
                )
    return prepared


def prepare_manual_grouping_batches(db: Session) -> list[dict[str, Any]]:
    locations = repositories.list_locations(db)
    prepared: list[dict[str, Any]] = []
    for location in locations:
        location_id = int(location["id"])
        ready_assets = repositories.list_manual_grouping_ready_trigger_frame_assets(
            db,
            location_id=location_id,
            limit=200,
        )
        if not ready_assets:
            continue
        newest_trigger_time = max(
            (_coerce_datetime_value(row.get("trigger_time")) for row in ready_assets),
            default=None,
        )
        if newest_trigger_time is None:
            continue
        window_start = newest_trigger_time - timedelta(hours=1)
        window_end = newest_trigger_time + timedelta(seconds=1)
        trigger_assets = [
            row
            for row in ready_assets
            if (grouping_time := _coerce_datetime_value(row.get("trigger_time"))) is not None
            and window_start <= grouping_time < window_end
        ]
        if not trigger_assets:
            continue
        batch = repositories.create_grouping_batch(
            db,
            location_id=location_id,
            period_code="manual",
            window_start=window_start,
            window_end=window_end,
        )
        if str(batch.get("status") or "").strip().lower() in {"failed", "issue"}:
            batch = repositories.update_grouping_batch(
                db,
                int(batch["id"]),
                {
                    "status": "pending",
                    "issue_reason": "Manual grouping retry queued.",
                },
            )
        for row in trigger_assets:
            repositories.upsert_grouping_item(
                db,
                batch_id=int(batch["id"]),
                trigger_id=int(row["trigger_id"]),
                video_asset_id=None,
                frame_payload={"frames": _first_trigger_frame_payload(_frame_urls_from_trigger_frame_asset(row))},
            )
        prepared.append(batch)
    return prepared


def prepare_manual_grouping_batches_for_range(
    db: Session,
    *,
    start_time: Any,
    end_time: Any,
    location_id: int | None = None,
) -> list[dict[str, Any]]:
    window_start = _coerce_datetime_value(start_time)
    window_end = _coerce_datetime_value(end_time)
    if window_start is None or window_end is None:
        raise ValueError("Start time and end time are required.")
    if window_end <= window_start:
        raise ValueError("End time must be after start time.")

    locations = repositories.list_locations(db)
    if location_id is not None:
        locations = [row for row in locations if int(row["id"]) == int(location_id)]
    prepared: list[dict[str, Any]] = []
    for location in locations:
        current_location_id = int(location["id"])
        ready_assets = repositories.list_manual_grouping_ready_trigger_frame_assets(
            db,
            location_id=current_location_id,
            limit=1000,
        )
        trigger_assets = [
            row
            for row in ready_assets
            if (grouping_time := _coerce_datetime_value(row.get("trigger_time"))) is not None
            and window_start <= grouping_time < window_end
        ]
        if not trigger_assets:
            continue
        batch = repositories.create_grouping_batch(
            db,
            location_id=current_location_id,
            period_code="manual_range",
            window_start=window_start,
            window_end=window_end,
        )
        if str(batch.get("status") or "").strip().lower() in {"failed", "issue"}:
            batch = repositories.update_grouping_batch(
                db,
                int(batch["id"]),
                {
                    "status": "pending",
                    "issue_reason": "Manual time-range grouping retry queued.",
                },
            )
        for row in trigger_assets:
            repositories.upsert_grouping_item(
                db,
                batch_id=int(batch["id"]),
                trigger_id=int(row["trigger_id"]),
                video_asset_id=None,
                frame_payload={"frames": _first_trigger_frame_payload(_frame_urls_from_trigger_frame_asset(row))},
            )
        prepared.append(batch)
    return prepared


def build_grouping_analysis_job_from_batch(db: Session, batch_id: int) -> GroupingAnalysisQueued:
    batch = repositories.get_grouping_batch(db, batch_id)
    items = repositories.list_grouping_items(db, batch_id)
    triggers_payload: list[dict[str, Any]] = []
    for item in items:
        trigger = repositories.get_trigger(db, int(item["trigger_id"]))
        frame_payload = item.get("frame_payload") if isinstance(item.get("frame_payload"), Mapping) else {}
        triggers_payload.append(
            {
                "trigger_id": int(item["trigger_id"]),
                "video_asset_id": int(item["video_asset_id"]) if item.get("video_asset_id") is not None else None,
                "phone_entry_id": trigger.get("phone_entry_id"),
                "credit_card_entry_id": trigger.get("credit_card_entry_id"),
                "entry_source_type": trigger.get("entry_source_type"),
                "trigger_time": trigger.get("trigger_time").isoformat() if hasattr(trigger.get("trigger_time"), "isoformat") else trigger.get("trigger_time"),
                "frames": _first_trigger_frame_payload(list(frame_payload.get("frames") or [])),
            }
        )
    manifest_payload = {
        "batch_id": int(batch["id"]),
        "location_id": int(batch["location_id"]),
        "period_code": batch.get("period_code"),
        "window_start": batch.get("window_start").isoformat() if hasattr(batch.get("window_start"), "isoformat") else batch.get("window_start"),
        "window_end": batch.get("window_end").isoformat() if hasattr(batch.get("window_end"), "isoformat") else batch.get("window_end"),
        "min_score": 0.74,
        "min_consecutive_matches": 3,
        "triggers": triggers_payload,
    }
    manifest_path = (
        tmp_media_root()
        / "grouping"
        / f"location_{int(batch['location_id'])}"
        / f"batch_{int(batch['id'])}"
        / "grouping_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str))
    manifest_object_key, manifest_url = _upload_runner_input_file(
        manifest_path,
        kind="grouping_manifest",
        location_id=int(batch["location_id"]),
        session_id=None,
        trigger_id=None,
        section="entrance",
    )
    repositories.update_grouping_batch(
        db,
        int(batch["id"]),
        {
            "manifest_object_key": manifest_object_key,
            "manifest_url": manifest_url,
        },
    )
    return GroupingAnalysisQueued(
        batch_id=int(batch["id"]),
        location_id=int(batch["location_id"]),
        period_code=str(batch.get("period_code") or "period"),
        window_start=_coerce_datetime_value(batch.get("window_start")) or datetime.now(),
        window_end=_coerce_datetime_value(batch.get("window_end")) or datetime.now(),
        manifest_url=manifest_url,
        manifest_object_key=manifest_object_key,
        model_name=None,
    )


def _run_gemini_grouping_for_batch(db: Session, *, batch_id: int, script_run_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = repositories.get_grouping_batch(db, batch_id)
    items = repositories.list_grouping_items(db, batch_id)
    trigger_inputs: list[dict[str, Any]] = []
    for item in items:
        trigger = repositories.get_trigger(db, int(item["trigger_id"]))
        frame_payload = item.get("frame_payload") if isinstance(item.get("frame_payload"), Mapping) else {}
        frames = _first_trigger_frame_payload(list(frame_payload.get("frames") or []))
        trigger_inputs.append(
            {
                "trigger_id": int(item["trigger_id"]),
                "trigger_time": trigger.get("trigger_time").isoformat() if hasattr(trigger.get("trigger_time"), "isoformat") else trigger.get("trigger_time"),
                "phone_entry_id": trigger.get("phone_entry_id"),
                "credit_card_entry_id": trigger.get("credit_card_entry_id"),
                "entry_source_type": trigger.get("entry_source_type"),
                "frames": frames,
            }
        )

    if not any(trigger.get("frames") for trigger in trigger_inputs):
        raise RuntimeError("No trigger frame images are available for Gemini grouping.")

    max_images = _grouping_gemini_max_images_per_request()
    pending_triggers: list[dict[str, Any]] = list(trigger_inputs)
    carry_forward_entries: dict[int, dict[str, Any]] = {}
    chunks: list[list[dict[str, Any]]] = []

    model_name = str(settings.grouping_gemini_model or "gemini-3.5-flash-lite").strip()
    resize_scale = _grouping_gemini_resize_scale()
    normalized_groups: list[dict[str, Any]] = []
    grouped_trigger_ids: set[int] = set()
    normalized_unknown: set[int] = set()
    normalized_open_entries: set[int] = set()
    trigger_has_identity: dict[int, bool] = {
        int(trigger["trigger_id"]): trigger.get("phone_entry_id") is not None or trigger.get("credit_card_entry_id") is not None
        for trigger in trigger_inputs
    }
    notes: list[str] = []
    chunk_results: list[dict[str, Any]] = []
    raw_metas: list[dict[str, Any]] = []
    next_group_id = 1

    chunk_index = 0
    while pending_triggers:
        chunk_index += 1
        # Reserve enough budget for at least one unseen trigger so a pile-up of
        # carried-forward open entries can never starve the loop of new triggers
        # to compare against (which would otherwise never terminate).
        next_frame_count = len(pending_triggers[0].get("frames") or [])
        carry_budget = max(max_images - next_frame_count, 0)
        chunk: list[dict[str, Any]] = []
        carry_image_count = 0
        for trigger_id in sorted(carry_forward_entries):
            trigger_input = carry_forward_entries[trigger_id]
            frame_count = len(trigger_input.get("frames") or [])
            if chunk and carry_image_count + frame_count > carry_budget:
                continue
            chunk.append(trigger_input)
            carry_image_count += frame_count
        current_image_count = carry_image_count
        while pending_triggers:
            frame_count = len(pending_triggers[0].get("frames") or [])
            if chunk and current_image_count + frame_count > max_images:
                break
            chunk.append(pending_triggers.pop(0))
            current_image_count += frame_count
        chunks.append(chunk)

        trigger_notes: list[dict[str, Any]] = []
        image_urls: list[str] = []
        image_mapping: list[dict[str, Any]] = []
        for trigger_input in chunk:
            note = {
                "trigger_id": trigger_input["trigger_id"],
                "trigger_time": trigger_input.get("trigger_time"),
                "phone_entry_id": trigger_input.get("phone_entry_id"),
                "credit_card_entry_id": trigger_input.get("credit_card_entry_id"),
                "entry_source_type": trigger_input.get("entry_source_type"),
                "image_numbers": [],
            }
            for frame in trigger_input.get("frames") or []:
                if not isinstance(frame, Mapping):
                    continue
                image_url = str(frame.get("image_url") or "").strip()
                if not image_url or image_url in image_urls:
                    continue
                image_urls.append(image_url)
                image_number = len(image_urls)
                note["image_numbers"].append(image_number)
                image_mapping.append(
                    {
                        "image_number": image_number,
                        "trigger_id": int(trigger_input["trigger_id"]),
                        "frame_index": frame.get("index"),
                        "sample_time": frame.get("sample_time"),
                    }
                )
            trigger_notes.append(note)
        if not image_urls:
            continue
        prompt = (
            "You are grouping retail door trigger events into customer sessions. "
            "The trigger list is in chronological order from earliest to latest. "
            "Each trigger is a door-opening event. A trigger may be an entry or an exit. Do not assume every door-opening trigger is an entry. "
            "Primary actor rule: a trigger's images can contain more than one person, for example a bystander walking past on the sidewalk, or a different "
            "customer from another trigger crossing through the background. For each trigger, identify the primary actor: the person who is directly interacting "
            "with the door, card reader, or QR scanner, or who is clearly the one passing through the doorway during that trigger's own frame sequence. Only the "
            "primary actor determines this trigger's entry/exit label and identity. A person who merely appears somewhere in the background of this trigger's "
            "images, without interacting with the door themselves, is not this trigger's subject even if you recognize them from another trigger; if you recognize "
            "a background person as the subject of a different trigger, use that observation only to help label that other trigger, not this one. "
            "Scene-consistency guard: before closing an entry with a candidate exit, check that the exit trigger's scene is actually consistent with the same "
            "single customer, matching clothing, build, and carried items, and not a visibly different number of people or a different group entirely. "
            "If the candidate exit shows a different number of people, unrelated new people, or no visible match to the entry customer's specific clothing, do not "
            "force the pairing. Leave the entry in open_entries and the exit trigger in unknown instead of guessing. Never invent or assume clothing, color, or "
            "item details that are not clearly visible in the specific images for that trigger. "
            "Identity matching is the gate for grouping: two triggers may only be placed in the same group if you can visually confirm they show the same physical "
            "person. Compare clothing color and pattern, pants/skirt color, shoes, bags or carried items, body shape and height, hair, and any other distinguishing "
            "visual detail across the trigger images before deciding two triggers belong together. "
            "Do not group two triggers together just because their timestamps or order look like a plausible entry-then-exit pair. Timing and order are supporting "
            "evidence only, never a substitute for a visual identity match. If you cannot visually confirm the same person, do not form a group even if the timeline fits. "
            "Direction rule: once you have visually confirmed a candidate match, use customer movement relative to the store entrance to label each trigger. "
            "A person walking into the store should be treated as entry. A person walking out of the store or away from the store should be treated as exit. "
            "If clear walking direction conflicts with simple timestamp assumptions, prioritize the walking direction. "
            "Identity rule: Entry triggers must have phone_entry_id or credit_card_entry_id. "
            "Triggers without phone_entry_id and without credit_card_entry_id can only be exit or unknown. "
            "A trigger with phone_entry_id or credit_card_entry_id can still be exit if visual direction evidence clearly supports it. "
            "Grouping rule: A complete session group must contain exactly one entry trigger and one or more later exit triggers, and every trigger in the group must "
            "be a confirmed visual match of the same person. A trigger must never be both entry and exit in the same group. "
            "Return confident complete entry+exit groups first even when other customers still have no visible exit yet. "
            "If an entry customer has not exited yet, put that entry trigger in open_entries instead of creating an entry-only group. "
            "Put exit-only and uncertain triggers into unknown. "
            "If a customer has been inside for more than 1 hour with no matching exit, keep it in open_entries and mention the issue in notes. "
            "If two triggers are visually confirmed as the same customer, the earlier trigger should normally be entry and the later trigger should normally be exit, "
            "with walking direction used to break ties. "
            "If door direction is visually ambiguous but the visual identity match is confirmed, prefer the chronological session pattern: earlier identity trigger "
            "opens the session, later same-person trigger closes it. "
            "Carry observation rule: for every grouped customer, describe what they visibly carry at entry and at exit. "
            "Count bags, plastic bags, woven/reusable bags, backpacks, boxes, cartons, bottles, and loose items only when visibly held, worn, or moving with the person. "
            "Record color, type, approximate size, count, and confidence. Use 0 count when the customer appears empty-handed. "
            "Important: Do not create a separate group for every trigger, and do not create a group from two triggers that merely fit a plausible timeline. Only group "
            "triggers when you have visually confirmed they show the same customer across time. "
            "If the same customer appears in trigger 73 and later trigger 74, return entry [73], exit [74]. "
            "If trigger 76 is an entry and trigger 77 is another entry before trigger 78 exits, return the complete matched pair and put still-waiting entries in open_entries. "
            "If a trigger cannot be confidently and visually matched into a complete entry+exit pair and is not a likely open entry, put it in unknown. "
            "Return strict JSON only with schema: "
            '{"groups":[{"entry":[integer],"exit":[integer],"confidence":number,"reason":string,'
            '"entry_carry":{"bag_count":integer,"item_count":integer,"items":[{"type":string,"color":string,"size":string,"count":integer,"confidence":number}],"summary":string},'
            '"exit_carry":{"bag_count":integer,"item_count":integer,"items":[{"type":string,"color":string,"size":string,"count":integer,"confidence":number}],"summary":string},'
            '"carry_change_summary":string,"total_customer":integer}],'
            '"open_entries":[integer],"unknown":[integer],"notes":[string]}. '
            f"Batch: {json.dumps({'batch_id': batch_id, 'location_id': batch.get('location_id'), 'period_code': batch.get('period_code'), 'window_start': batch.get('window_start'), 'window_end': batch.get('window_end'), 'chunk': chunk_index}, default=str)}. "
            f"Triggers: {json.dumps(trigger_notes, default=str)}. "
            f"Image mapping: {json.dumps(image_mapping, default=str)}."
        )
        gemini_result, gemini_meta = _call_kiosk_gemini_summary(
            prompt=prompt,
            image_urls=image_urls,
            model_name=model_name,
            image_resize_scale=resize_scale,
        )
        _record_gemini_cost(db, script_run_id, gemini_meta)
        raw_metas.append(gemini_meta)
        chunk_groups = gemini_result.get("groups") if isinstance(gemini_result.get("groups"), list) else []
        chunk_open_entries = gemini_result.get("open_entries") if isinstance(gemini_result.get("open_entries"), list) else []
        chunk_unknown = gemini_result.get("unknown") if isinstance(gemini_result.get("unknown"), list) else []
        chunk_grouped_trigger_ids: set[int] = set()
        for group in chunk_groups:
            if not isinstance(group, Mapping):
                continue
            entry_ids = _group_trigger_id_list(group.get("entry"))
            exit_ids = [trigger_id for trigger_id in _group_trigger_id_list(group.get("exit")) if trigger_id not in entry_ids]
            if not entry_ids:
                continue
            entry_ids = entry_ids[:1]
            invalid_entry_ids = [trigger_id for trigger_id in entry_ids if not trigger_has_identity.get(trigger_id, False)]
            if invalid_entry_ids:
                normalized_unknown.update(entry_ids)
                normalized_unknown.update(exit_ids)
                continue
            if not exit_ids:
                normalized_unknown.update(entry_ids)
                continue
            chunk_grouped_trigger_ids.update(entry_ids)
            chunk_grouped_trigger_ids.update(exit_ids)
            grouped_trigger_ids.update(entry_ids)
            grouped_trigger_ids.update(exit_ids)
            # An entry carried forward from an earlier chunk may only now be closing;
            # drop it from open/unknown so it doesn't also linger in those buckets.
            normalized_open_entries.difference_update(entry_ids)
            normalized_open_entries.difference_update(exit_ids)
            normalized_unknown.difference_update(entry_ids)
            normalized_unknown.difference_update(exit_ids)
            normalized_groups.append(
                {
                    "group_id": next_group_id,
                    "entry": entry_ids,
                    "exit": exit_ids,
                    "score": _coerce_number(group.get("confidence"), 0.0),
                    "reason": str(group.get("reason") or "gemini_grouping_direct"),
                    "entry_carry": group.get("entry_carry") if isinstance(group.get("entry_carry"), Mapping) else None,
                    "exit_carry": group.get("exit_carry") if isinstance(group.get("exit_carry"), Mapping) else None,
                    "carry_change_summary": str(group.get("carry_change_summary") or ""),
                    "total_customer": _coerce_int(group.get("total_customer"), 0),
                    "source": "gemini_grouping_direct",
                    "chunk": chunk_index,
                }
            )
            next_group_id += 1
        for trigger_id in _group_trigger_id_list(chunk_open_entries):
            if trigger_id in grouped_trigger_ids:
                continue
            if trigger_has_identity.get(trigger_id, False):
                normalized_open_entries.add(trigger_id)
            else:
                normalized_unknown.add(trigger_id)
        normalized_unknown.update(_group_trigger_id_list(chunk_unknown))
        if isinstance(gemini_result.get("notes"), list):
            notes.extend(str(note) for note in gemini_result.get("notes") or [])
        chunk_results.append(
            {
                "chunk": chunk_index,
                "trigger_ids": [int(trigger["trigger_id"]) for trigger in chunk],
                "image_count": len(image_urls),
                "image_mapping": image_mapping,
                "raw_result": gemini_result,
                "grouped_trigger_ids": sorted(chunk_grouped_trigger_ids),
            }
        )

        # Only triggers the model just reconfirmed as open_entries get another look in
        # a later chunk; anything grouped or marked unknown this round is final and is
        # dropped from the carry-forward set (an "all unknown" chunk simply stops there
        # instead of being retried).
        chunk_trigger_by_id = {int(trigger_input["trigger_id"]): trigger_input for trigger_input in chunk}
        for trigger_id in chunk_trigger_by_id:
            if trigger_id in normalized_open_entries:
                carry_forward_entries[trigger_id] = chunk_trigger_by_id[trigger_id]
            else:
                carry_forward_entries.pop(trigger_id, None)

    all_trigger_ids = [int(item["trigger_id"]) for item in items if item.get("trigger_id") is not None]
    normalized_unknown = {
        trigger_id
        for trigger_id in (list(normalized_unknown) + all_trigger_ids)
        if trigger_id not in grouped_trigger_ids and trigger_id not in normalized_open_entries
    }
    total_image_count = sum(len(trigger.get("frames") or []) for trigger in trigger_inputs)
    cost_details = [_gemini_usage_cost(meta)[1] for meta in raw_metas]
    grouping_summary = {
        "batch_id": batch_id,
        "location_id": int(batch["location_id"]),
        "period_code": batch.get("period_code"),
        "window_start": batch.get("window_start").isoformat() if hasattr(batch.get("window_start"), "isoformat") else batch.get("window_start"),
        "window_end": batch.get("window_end").isoformat() if hasattr(batch.get("window_end"), "isoformat") else batch.get("window_end"),
        "groups": normalized_groups,
        "open_entries": sorted(normalized_open_entries),
        "unknown": sorted(normalized_unknown),
        "notes": notes,
        "diagnostics": {
            "mode": "gemini_grouping_direct",
            "temporary_runpod_grouping_disabled": True,
            "model": model_name,
            "image_resize_scale": resize_scale,
            "max_frames_per_trigger": _grouping_frames_per_trigger(),
            "max_images_per_request": max_images,
            "chunk_count": len(chunks),
            "trigger_count": len(all_trigger_ids),
            "image_count": total_image_count,
            "chunks": chunk_results,
            "cost_details": cost_details,
        },
    }
    return grouping_summary, {
        "provider": "tds_api_gemini",
        "model": model_name,
        "image_resize_scale": resize_scale,
        "chunk_count": len(chunks),
        "image_count": total_image_count,
        "chunks": raw_metas,
    }


def _run_confidence_after_grouping_success(db: Session, *, batch_id: int) -> dict[str, Any]:
    deleted_count = repositories.delete_filter_confidence_results_for_batch(db, batch_id)
    try:
        result = run_theft_confidence_for_grouping_batch(db, batch_id=batch_id)
        return {
            "status": "success",
            "deleted_previous_confidence_result_count": deleted_count,
            **result,
        }
    except Exception as exc:
        logger.exception("Theft confidence failed immediately after grouping batch_id=%s", batch_id)
        try:
            batch = repositories.get_grouping_batch(db, batch_id)
            repositories.upsert_filter_confidence_result(
                db,
                batch_id=batch_id,
                group_key="__error__",
                location_id=int(batch.get("location_id") or 0),
                score=0,
                need_deep_analysis=False,
                reason="confidence_error",
                factor_payload={"error": str(exc), "source": "grouping_completion"},
            )
        except Exception:
            logger.exception("Could not persist immediate confidence error for batch_id=%s", batch_id)
        return {
            "status": "failed",
            "batch_id": batch_id,
            "deleted_previous_confidence_result_count": deleted_count,
            "error": str(exc),
        }


def start_grouping_analysis_job(job: GroupingAnalysisQueued) -> ScriptExecutionResult:
    db = TransactionalSessionLocal()
    try:
        script_run_id = repositories.create_script_run_started(
            db,
            session_id=None,
            trigger_id=None,
            script_name="grouping",
            model_name="gemini_grouping_direct",
            status="running",
            command=SCRIPT_RUN_COMMAND_REDACTED,
        )
        repositories.update_grouping_batch(
            db,
            job.batch_id,
            {
                "script_run_id": script_run_id,
                "status": "running",
                "started_at": datetime.now(UTC),
                "manifest_url": job.manifest_url,
                "manifest_object_key": job.manifest_object_key,
            },
        )
        processing_count = repositories.mark_grouping_batch_frame_assets_processing(db, job.batch_id)
        logger.info("Marked grouping frame assets processing batch_id=%s count=%s", job.batch_id, processing_count)
        grouping_summary, gemini_meta = _run_gemini_grouping_for_batch(db, batch_id=job.batch_id, script_run_id=script_run_id)
        _persist_grouping_items_from_summary(db, batch_id=job.batch_id, grouping_summary=grouping_summary)
        processed_count = repositories.mark_grouping_batch_frame_assets_processed(db, job.batch_id)
        logger.info("Marked grouping frame assets processed batch_id=%s count=%s", job.batch_id, processed_count)
        open_count = repositories.mark_grouping_batch_frame_assets_retrieved(
            db,
            job.batch_id,
            error=None,
        )
        logger.info("Kept unresolved grouping frame assets reusable batch_id=%s count=%s", job.batch_id, open_count)
        repositories.update_grouping_batch(
            db,
            job.batch_id,
            {
                "status": "success",
                "result_payload": grouping_summary,
                "issue_reason": None,
                "finished_at": datetime.now(UTC),
            },
        )
        repositories.assign_script_run_runner_job(
            db,
            script_run_id,
            runner_job_id=None,
            runner_payload={
                "batch_id": job.batch_id,
                "location_id": job.location_id,
                "period_code": job.period_code,
                "window_start": job.window_start.isoformat(),
                "window_end": job.window_end.isoformat(),
                "manifest_object_key": job.manifest_object_key,
                "manifest_url": job.manifest_url,
                "provider": "tds_api_gemini",
                "model": settings.grouping_gemini_model,
                "mode": "gemini_grouping_direct",
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="success",
            stdout_log=json.dumps(
                {
                    "grouping_summary": grouping_summary,
                    "gemini_meta": _compact_gemini_meta_for_log(gemini_meta),
                    "confidence_result": {
                        "status": "queued",
                        "message": "Theft confidence worker will process this successful grouping batch.",
                    },
                },
                indent=2,
                default=str,
            ),
            stderr_log="",
        )
        return ScriptExecutionResult(
            script_run_id=script_run_id,
            runner_job_id=None,
            script_name="grouping",
            model_name="gemini_grouping_direct",
            status="success",
            command=["tds_api_gemini", "grouping"],
            stdout=json.dumps(grouping_summary, default=str),
            stderr="",
        )
    except Exception as exc:
        try:
            repositories.mark_grouping_batch_frame_assets_retrieved(
                db,
                job.batch_id,
                error=f"Gemini grouping failed: {exc}",
            )
        except Exception:
            logger.exception("Could not restore grouping frame assets after Gemini failure batch_id=%s", job.batch_id)
        if "script_run_id" in locals():
            repositories.finish_script_run(
                db,
                script_run_id,
                status="failed",
                stdout_log="",
                stderr_log=str(exc),
            )
        repositories.update_grouping_batch(
            db,
            job.batch_id,
            {
                "status": "issue",
                "issue_reason": str(exc),
                "finished_at": datetime.now(UTC),
            },
        )
        raise
    finally:
        db.close()


def _refresh_grouping_item_frame_payloads(db: Session, *, batch: Mapping[str, Any]) -> int:
    location_id = int(batch["location_id"])
    window_start = batch.get("window_start")
    window_end = batch.get("window_end")
    if window_start is None or window_end is None:
        return 0
    existing_items = repositories.list_grouping_items(db, int(batch["id"]))
    existing_trigger_ids = {
        int(item["trigger_id"])
        for item in existing_items
        if item.get("trigger_id") is not None
    }
    if not existing_trigger_ids:
        return 0
    frame_assets = repositories.list_trigger_frame_assets_for_window(
        db,
        location_id=location_id,
        window_start=window_start,
        window_end=window_end,
    )
    refreshed_count = 0
    for row in frame_assets:
        trigger_id = int(row["trigger_id"])
        if trigger_id not in existing_trigger_ids:
            continue
        frames = _first_trigger_frame_payload(_frame_urls_from_trigger_frame_asset(row))
        if not frames:
            continue
        repositories.upsert_grouping_item(
            db,
            batch_id=int(batch["id"]),
            trigger_id=trigger_id,
            video_asset_id=None,
            group_key=None,
            role="unknown",
            status="pending",
            frame_payload={"frames": frames},
            result_payload=None,
        )
        refreshed_count += 1
    return refreshed_count


def retry_grouping_batch_now(db: Session, *, batch_id: int) -> dict[str, Any]:
    batch = repositories.get_grouping_batch(db, batch_id)
    status = str(batch.get("status") or "").strip().lower()
    if status in {"pending", "dispatching", "running"}:
        raise ValueError(f"Grouping batch {batch_id} is {status or 'unknown'} and cannot be rerun yet.")
    if status not in {"success", "failed", "issue", "cancel", "canceled", "cancelled"}:
        raise ValueError(f"Grouping batch {batch_id} is {status or 'unknown'} and cannot be rerun.")
    active_batches = [
        row
        for row in repositories.list_running_grouping_batches(db)
        if int(row["id"]) != batch_id
    ]
    if active_batches or repositories.has_active_remote_analysis_script_run(db, script_names=["grouping"]):
        raise ValueError("Grouping is already dispatching or running.")

    repositories.mark_grouping_batch_frame_assets_retrieved(
        db,
        batch_id,
        error="Rerun queued for grouping batch.",
    )
    deleted_confidence_count = repositories.delete_filter_confidence_results_for_batch(db, batch_id)
    repositories.reset_grouping_batch_for_retry(db, batch_id)
    refreshed_count = _refresh_grouping_item_frame_payloads(db, batch=batch)
    return {
        "ok": True,
        "batch_id": batch_id,
        "status": "pending",
        "queued": True,
        "message": "Grouping batch queued for background rerun.",
        "refreshed_frame_payload_count": refreshed_count,
        "deleted_confidence_result_count": deleted_confidence_count,
    }


def _finalize_remote_grouping_script_run(
    db: Session,
    *,
    script_run: dict[str, Any],
    remote_result: RemoteRunnerResult,
) -> ScriptExecutionResult:
    script_run_id = int(script_run["id"])
    runner_payload = dict(script_run.get("runner_payload") or {})
    batch_id = int(runner_payload["batch_id"]) if runner_payload.get("batch_id") is not None else None
    remote_status = "success" if remote_result.status == "success" else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=remote_status,
        stdout_log=remote_result.stdout,
        stderr_log=remote_result.stderr,
    )
    _record_remote_runner_cost(db, script_run_id, remote_result)
    result = ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=str(script_run.get("runner_job_id") or ""),
        script_name="grouping",
        model_name=script_run.get("model_name"),
        status=remote_status,
        command=["runpod_serverless", "grouping"],
        stdout=remote_result.stdout,
        stderr=remote_result.stderr,
    )
    if batch_id is None:
        return result
    grouping_summary, grouping_summary_fetch_error = _resolve_grouping_summary_from_remote_result(remote_result)
    repositories.update_grouping_batch(
        db,
        batch_id,
        {
            "status": remote_status,
            "result_payload": grouping_summary,
            "issue_reason": remote_result.stderr if remote_status != "success" else grouping_summary_fetch_error,
            "finished_at": datetime.now(UTC),
        },
    )
    if remote_status != "success":
        try:
            restored_count = repositories.mark_grouping_batch_frame_assets_retrieved(
                db,
                batch_id,
                error=remote_result.stderr or "Grouping RunPod job failed.",
            )
            logger.info("Restored grouping frame assets after failure batch_id=%s count=%s", batch_id, restored_count)
        except Exception:
            logger.exception("Could not restore grouping frame assets after failure batch_id=%s", batch_id)
        return result
    _persist_grouping_items_from_summary(db, batch_id=batch_id, grouping_summary=grouping_summary)
    try:
        processed_count = repositories.mark_grouping_batch_frame_assets_processed(db, batch_id)
        logger.info("Marked grouping frame assets processed batch_id=%s count=%s", batch_id, processed_count)
        open_count = repositories.mark_grouping_batch_frame_assets_retrieved(
            db,
            batch_id,
            error=None,
        )
        logger.info("Kept unresolved grouping frame assets reusable batch_id=%s count=%s", batch_id, open_count)
    except Exception:
        logger.exception("Could not mark grouping frame assets processed batch_id=%s", batch_id)
    return result


def _load_filter_factor_settings(db: Session, location_id: int) -> dict[str, dict[str, Any]]:
    settings_by_code = {code: dict(value) for code, value in DEFAULT_FILTER_FACTORS.items()}
    rows = repositories.list_filter_factors(db)
    for row in rows:
        row_location_id = row.get("location_id")
        if row_location_id is not None and int(row_location_id) != int(location_id):
            continue
        factor_code = str(row.get("factor_code") or "").strip()
        if not factor_code:
            continue
        settings_by_code[factor_code] = {
            **settings_by_code.get(factor_code, {}),
            "enabled": row.get("enabled") in (True, 1),
            "config": row.get("config"),
            "location_id": row_location_id,
        }
    return settings_by_code


def _apply_filter_factor(
    *,
    reasons: list[str],
    factor_details: dict[str, Any],
    factors: Mapping[str, Mapping[str, Any]],
    factor_code: str,
    hit: bool,
    reason: str,
    evidence: Mapping[str, Any] | None = None,
) -> bool:
    factor = dict(factors.get(factor_code) or DEFAULT_FILTER_FACTORS.get(factor_code) or {})
    enabled = factor.get("enabled") in (True, 1)
    triggered = enabled and hit
    factor_details[factor_code] = {
        "enabled": enabled,
        "hit": bool(hit),
        "triggered": triggered,
        "reason": reason,
        "evidence": dict(evidence or {}),
    }
    if triggered:
        reasons.append(reason)
    return triggered


def _mark_filter_factor_skipped(
    *,
    factor_details: dict[str, Any],
    factors: Mapping[str, Mapping[str, Any]],
    factor_code: str,
    reason: str,
) -> None:
    factor = dict(factors.get(factor_code) or DEFAULT_FILTER_FACTORS.get(factor_code) or {})
    factor_details[factor_code] = {
        "enabled": factor.get("enabled") in (True, 1),
        "hit": False,
        "triggered": False,
        "reason": reason,
        "evidence": {"skipped": True, "reason": reason},
    }


def _filter_factor_enabled(factors: Mapping[str, Mapping[str, Any]], factor_code: str) -> bool:
    config = factors.get(factor_code)
    if not isinstance(config, Mapping):
        return False
    return _as_boolish(config.get("enabled"))


def _read_group_value(group: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in group and group[key] is not None:
            return group[key]
    for container_key in ("gemini", "gemini_result", "result", "group", "summary"):
        container = group.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
    return None


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _as_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_phone_prefix(value: Any) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _normalize_country_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _extract_payment_intent_id(value: Any) -> str | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    match = re.search(r"(pi_[A-Za-z0-9]+)", raw_value)
    return match.group(1) if match else None


def _extract_stripe_id(value: Any, prefix: str) -> str | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    match = re.search(rf"({re.escape(prefix)}_[A-Za-z0-9]+)", raw_value)
    return match.group(1) if match else None


def _nested_get(value: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _stripe_get(path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    secret_key = str(settings.stripe_secret_key or os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not secret_key:
        raise RuntimeError("Stripe API key is not configured. Set THEFT_API_STRIPE_SECRET_KEY.")
    base_url = str(settings.stripe_api_base_url or "https://api.stripe.com/v1").rstrip("/")
    query = f"?{urlencode(params, doseq=True)}" if params else ""
    request = Request(
        f"{base_url}/{path.lstrip('/')}{query}",
        headers={"Authorization": f"Bearer {secret_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=settings.stripe_lookup_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Stripe request failed with HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Stripe request failed: {exc}") from exc


def _card_country_from_stripe_payload(payload: Mapping[str, Any]) -> str | None:
    return (
        _nested_get(payload, "card", "country")
        or _nested_get(payload, "card_present", "country")
        or _nested_get(payload, "payment_method_details", "card", "country")
        or _nested_get(payload, "payment_method_details", "card_present", "country")
        or _nested_get(payload, "latest_charge", "payment_method_details", "card", "country")
        or _nested_get(payload, "latest_charge", "payment_method_details", "card_present", "country")
        or _nested_get(payload, "payment_method", "card", "country")
        or _nested_get(payload, "payment_method", "card_present", "country")
    )


def _resolve_stripe_card_country(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        return {"country": None, "source": "none", "error": "missing_credit_card_identity"}
    payment_method_id = _extract_stripe_id(identity.get("payment_method_id"), "pm")
    charge_id = _extract_stripe_id(identity.get("charge_id"), "ch")
    payment_intent_id = (
        _extract_payment_intent_id(identity.get("payment_intent_id"))
        or _extract_payment_intent_id(identity.get("client_secret"))
    )
    attempts: list[dict[str, Any]] = []

    if payment_method_id:
        try:
            payload = _stripe_get(f"payment_methods/{quote(payment_method_id, safe='')}")
            country = _card_country_from_stripe_payload(payload)
            attempts.append({"source": "payment_method", "id": payment_method_id, "country": country})
            if country:
                return {
                    "country": country,
                    "source": "stripe_payment_method",
                    "stripe_id": payment_method_id,
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"source": "payment_method", "id": payment_method_id, "error": str(exc)})

    if charge_id:
        try:
            payload = _stripe_get(f"charges/{quote(charge_id, safe='')}")
            country = _card_country_from_stripe_payload(payload)
            attempts.append({"source": "charge", "id": charge_id, "country": country})
            if country:
                return {
                    "country": country,
                    "source": "stripe_charge",
                    "stripe_id": charge_id,
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"source": "charge", "id": charge_id, "error": str(exc)})

    if payment_intent_id:
        try:
            payload = _stripe_get(
                f"payment_intents/{quote(payment_intent_id, safe='')}",
                params={"expand[]": ["latest_charge", "payment_method"]},
            )
            country = _card_country_from_stripe_payload(payload)
            attempts.append({"source": "payment_intent", "id": payment_intent_id, "country": country})
            if country:
                return {
                    "country": country,
                    "source": "stripe_payment_intent",
                    "stripe_id": payment_intent_id,
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"source": "payment_intent", "id": payment_intent_id, "error": str(exc)})

    return {
        "country": None,
        "source": "stripe_lookup_failed" if attempts else "stripe_ids_missing",
        "attempts": attempts,
    }


def resolve_stripe_card_country_for_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    return _resolve_stripe_card_country(identity)


def _evaluate_country_code_check(
    db: Session,
    *,
    location_id: int,
    trigger_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rules = repositories.list_filter_country_code_checks(db, location_id=location_id, enabled_only=True)
    evidence: dict[str, Any] = {
        "rules_checked": len(rules),
        "matches": [],
        "identities_checked": [],
        "missing_card_country": [],
    }
    if not rules:
        return {**evidence, "hit": False, "reason": "no_country_rules_configured"}

    phone_rules = [
        {**rule, "normalized_phone_prefix": _normalize_phone_prefix(rule.get("phone_prefix"))}
        for rule in rules
        if _normalize_phone_prefix(rule.get("phone_prefix"))
    ]
    card_rules = [
        {**rule, "normalized_card_country": _normalize_country_token(rule.get("card_country") or rule.get("country_code"))}
        for rule in rules
        if _normalize_country_token(rule.get("card_country") or rule.get("country_code"))
    ]

    seen_identity_keys: set[tuple[str, str]] = set()
    for trigger in trigger_rows:
        trigger_id = int(trigger.get("id") or 0)
        phone_entry_id = trigger.get("phone_entry_id")
        if phone_entry_id is not None:
            identity_key = ("phone", str(phone_entry_id))
            if identity_key not in seen_identity_keys:
                seen_identity_keys.add(identity_key)
                identity = repositories.get_phone_entry_identity(db, phone_entry_id)
                phone_number = identity.get("phone_number") if identity else None
                normalized_phone = _normalize_phone_prefix(phone_number)
                identity_evidence = {
                    "trigger_id": trigger_id,
                    "type": "phone",
                    "phone_entry_id": phone_entry_id,
                    "phone_number": phone_number,
                }
                evidence["identities_checked"].append(identity_evidence)
                for rule in phone_rules:
                    prefix = rule["normalized_phone_prefix"]
                    if normalized_phone.startswith(prefix):
                        evidence["matches"].append(
                            {
                                "trigger_id": trigger_id,
                                "type": "phone",
                                "phone_entry_id": phone_entry_id,
                                "phone_number": phone_number,
                                "matched_value": f"+{prefix}",
                                "rule_id": rule.get("id"),
                                "country_code": rule.get("country_code"),
                                "country_name": rule.get("country_name"),
                            }
                        )

        credit_card_entry_id = trigger.get("credit_card_entry_id")
        if credit_card_entry_id is not None:
            identity_key = ("credit_card", str(credit_card_entry_id))
            if identity_key in seen_identity_keys:
                continue
            seen_identity_keys.add(identity_key)
            identity = repositories.get_credit_card_entry_identity(db, credit_card_entry_id)
            stored_card_country = identity.get("country") if identity else None
            stripe_lookup = (
                {"country": stored_card_country, "source": "stored_entrylogs_country"}
                if stored_card_country
                else _resolve_stripe_card_country(identity)
            )
            card_country = stored_card_country or stripe_lookup.get("country")
            normalized_card_country = _normalize_country_token(card_country)
            fingerprint = identity.get("fingerprint") if identity else None
            identity_evidence = {
                "trigger_id": trigger_id,
                "type": "credit_card",
                "credit_card_entry_id": credit_card_entry_id,
                "fingerprint": fingerprint,
                "card_country": card_country,
                "stored_card_country": stored_card_country,
                "stripe_country_lookup": stripe_lookup,
                "payment_method_id": identity.get("payment_method_id") if identity else None,
                "charge_id": identity.get("charge_id") if identity else None,
                "payment_intent_id": identity.get("payment_intent_id") if identity else None,
                "last4": identity.get("last4") if identity else None,
            }
            evidence["identities_checked"].append(identity_evidence)
            if not normalized_card_country:
                evidence["missing_card_country"].append(identity_evidence)
                continue
            for rule in card_rules:
                if normalized_card_country == rule["normalized_card_country"]:
                    evidence["matches"].append(
                        {
                            "trigger_id": trigger_id,
                            "type": "credit_card",
                            "credit_card_entry_id": credit_card_entry_id,
                            "fingerprint": fingerprint,
                            "matched_value": card_country,
                            "country_source": stripe_lookup.get("source"),
                            "rule_id": rule.get("id"),
                            "country_code": rule.get("country_code"),
                            "country_name": rule.get("country_name"),
                        }
                    )

    return {
        **evidence,
        "hit": bool(evidence["matches"]),
        "reason": "country_code_match" if evidence["matches"] else "no_country_code_match",
    }


def _group_entry_identity_values(
    db: Session,
    *,
    trigger_rows: list[dict[str, Any]],
    entry_trigger_ids: list[int],
) -> dict[str, list[str]]:
    entry_id_set = {int(trigger_id) for trigger_id in entry_trigger_ids}
    candidate_rows = [
        row
        for row in trigger_rows
        if not entry_id_set or int(row.get("id") or 0) in entry_id_set
    ]
    phone_numbers: list[str] = []
    card_fingerprints: list[str] = []

    for trigger in candidate_rows:
        phone_entry_id = trigger.get("phone_entry_id")
        if phone_entry_id is not None:
            identity = repositories.get_phone_entry_identity(db, phone_entry_id)
            value = str((identity or {}).get("phone_number") or "").strip()
            if value and value not in phone_numbers:
                phone_numbers.append(value)

        credit_card_entry_id = trigger.get("credit_card_entry_id")
        if credit_card_entry_id is not None:
            identity = repositories.get_credit_card_entry_identity(db, credit_card_entry_id)
            value = str((identity or {}).get("fingerprint") or "").strip()
            if value and value not in card_fingerprints:
                card_fingerprints.append(value)

    return {
        "phone_numbers": phone_numbers,
        "card_fingerprints": card_fingerprints,
    }


def _evaluate_unusual_group_size(
    db: Session,
    *,
    batch_id: int,
    location_id: int,
    trigger_rows: list[dict[str, Any]],
    entry_trigger_ids: list[int],
    total_customer: int,
) -> dict[str, Any]:
    identities = _group_entry_identity_values(
        db,
        trigger_rows=trigger_rows,
        entry_trigger_ids=entry_trigger_ids,
    )
    evidence: dict[str, Any] = {
        "current_total_customer": total_customer,
        "identities_checked": identities,
        "history": [],
        "history_count": 0,
        "historical_average": None,
        "historical_max": None,
        "min_history": settings.filter_unusual_group_size_min_history,
        "delta_threshold": settings.filter_unusual_group_size_delta,
    }
    if total_customer <= 0:
        return {**evidence, "hit": False, "reason": "missing_current_group_size"}
    if not identities["phone_numbers"] and not identities["card_fingerprints"]:
        return {**evidence, "hit": False, "reason": "missing_entry_identity"}

    history_rows = repositories.list_identity_group_size_history(
        db,
        location_id=location_id,
        phone_numbers=identities["phone_numbers"],
        card_fingerprints=identities["card_fingerprints"],
        exclude_batch_id=batch_id,
        limit=20,
    )
    history_sizes = [
        int(size)
        for row in history_rows
        if (size := _coerce_int(row.get("total_customer"), 0)) and int(size) > 0
    ]
    evidence["history"] = [
        {
            "batch_id": row.get("batch_id"),
            "group_key": row.get("group_key"),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "total_customer": row.get("total_customer"),
        }
        for row in history_rows
    ]
    evidence["history_count"] = len(history_sizes)
    if not history_sizes:
        return {**evidence, "hit": False, "reason": "no_identity_group_size_history"}

    historical_average = sum(history_sizes) / len(history_sizes)
    historical_max = max(history_sizes)
    evidence["historical_average"] = round(historical_average, 2)
    evidence["historical_max"] = historical_max

    min_history = max(1, int(settings.filter_unusual_group_size_min_history or 1))
    if len(history_sizes) < min_history:
        return {**evidence, "hit": False, "reason": "insufficient_identity_group_size_history"}

    delta_threshold = max(1, int(settings.filter_unusual_group_size_delta or 1))
    hit = total_customer >= historical_max + delta_threshold
    evidence["increase_from_historical_max"] = total_customer - historical_max
    evidence["increase_from_historical_average"] = round(total_customer - historical_average, 2)
    return {
        **evidence,
        "hit": hit,
        "reason": "identity_group_size_unusual" if hit else "identity_group_size_within_history",
    }


def _transaction_event_time(row: Mapping[str, Any]) -> datetime | None:
    return (
        _coerce_datetime_value(row.get("created_at"))
        or _coerce_datetime_value(row.get("createdAt"))
        or _coerce_datetime_value(row.get("transaction_time"))
    )


def _seconds_between(left: datetime, right: datetime) -> float:
    if left.tzinfo is not None and right.tzinfo is None:
        right = right.replace(tzinfo=left.tzinfo)
    elif left.tzinfo is None and right.tzinfo is not None:
        left = left.replace(tzinfo=right.tzinfo)
    return (left - right).total_seconds()


def _transaction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        details = row.get("details") if isinstance(row.get("details"), list) else []
        payload.append(
            {
                "receipt_number": row.get("receipt_number"),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "transaction_time": row.get("transaction_time"),
                "total_amount": row.get("total_amount"),
                "total_items": row.get("total_items"),
                "details": [
                    {
                        "item_name": detail.get("item_name"),
                        "barcode": detail.get("barcode"),
                        "quantity": detail.get("quantity"),
                        "price": detail.get("price"),
                        "subtotal": detail.get("subtotal"),
                    }
                    for detail in details
                    if isinstance(detail, Mapping)
                ],
            }
        )
    return payload


def _has_short_period_transaction_issues(rows: list[dict[str, Any]], threshold_seconds: int) -> tuple[bool, dict[str, Any]]:
    issue_times = sorted(
        event_time for row in rows if (event_time := _transaction_event_time(row)) is not None
    )
    if len(issue_times) < 2:
        return False, {"issue_count": len(rows), "matched_window_seconds": None}
    best_window: tuple[datetime, datetime] | None = None
    best_seconds: float | None = None
    for previous_time, current_time in zip(issue_times, issue_times[1:]):
        seconds = abs(_seconds_between(current_time, previous_time))
        if best_seconds is None or seconds < best_seconds:
            best_seconds = seconds
            best_window = (previous_time, current_time)
    hit = best_seconds is not None and best_seconds <= threshold_seconds
    return hit, {
        "issue_count": len(rows),
        "short_period_seconds": threshold_seconds,
        "matched_window_seconds": best_seconds,
        "matched_start": best_window[0].isoformat() if best_window else None,
        "matched_end": best_window[1].isoformat() if best_window else None,
    }


def _match_alert_ids_near_transactions(
    alerts: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    threshold_seconds: int,
) -> list[int]:
    transaction_times = [
        event_time for row in transactions if (event_time := _transaction_event_time(row)) is not None
    ]
    matched_ids: list[int] = []
    for alert in alerts:
        alert_time = _coerce_datetime_value(alert.get("created_at") or alert.get("createdAt"))
        if alert_time is None:
            continue
        for transaction_time in transaction_times:
            if abs(_seconds_between(alert_time, transaction_time)) <= threshold_seconds:
                try:
                    matched_ids.append(int(alert["id"]))
                except (KeyError, TypeError, ValueError):
                    pass
                break
    return sorted(set(matched_ids))


def _group_carry_evidence(group: Mapping[str, Any]) -> dict[str, Any]:
    entry_carry = _read_group_value(group, "entry_carry")
    exit_carry = _read_group_value(group, "exit_carry")
    return {
        "entry_carry": entry_carry if isinstance(entry_carry, Mapping) else None,
        "exit_carry": exit_carry if isinstance(exit_carry, Mapping) else None,
        "carry_change_summary": _read_group_value(group, "carry_change_summary"),
        "legacy_carry_score": _coerce_number(_read_group_value(group, "carry_something_from_store_score", "carry_score"), 0.0),
        "before_is_yellow_bag": _read_group_value(group, "before_is_yellow_bag"),
        "after_is_yellow_bag": _read_group_value(group, "after_is_yellow_bag"),
    }


def _evaluate_carry_item_signal_with_ai(
    db: Session,
    *,
    batch_id: int,
    group_key: str,
    location_id: int,
    trigger_ids: list[int],
    group: Mapping[str, Any],
    transactions: list[dict[str, Any]],
    total_quantity: int,
    total_value: float,
) -> dict[str, Any]:
    carry_evidence = _group_carry_evidence(group)
    legacy_hit = (
        carry_evidence["legacy_carry_score"] >= settings.filter_carry_score_threshold
        or not _as_boolish(carry_evidence["before_is_yellow_bag"])
        and _as_boolish(carry_evidence["after_is_yellow_bag"])
    )
    model_name = str(settings.grouping_gemini_model or "gemini-3.5-flash-lite").strip()
    transaction_payload = _transaction_summary(transactions)
    runner_payload = {
        "batch_id": batch_id,
        "group_key": group_key,
        "location_id": location_id,
        "trigger_ids": trigger_ids,
        "model": model_name,
        "carry_evidence": carry_evidence,
        "transactions": transaction_payload,
        "total_quantity": total_quantity,
        "total_value": total_value,
        "transaction_details": transaction_payload,
    }
    script_run_id = _create_gemini_script_run(
        db,
        session_id=None,
        trigger_id=trigger_ids[0] if trigger_ids else None,
        script_name="carry_confidence",
        model_name=model_name,
        runner_payload=runner_payload,
    )
    prompt = (
        "You are a retail theft confidence reviewer. Decide whether a customer's exit carrying state is suspicious "
        "after comparing entry carry evidence, exit carry evidence, and paid receipt item details. "
        "Flag only when the exit bag/plastic bag/container/items are not reasonably explained by the paid transaction. "
        "If visual carry evidence is missing or weak, return hit=false and reason='insufficient_visual_carry_evidence' instead of guessing. "
        "Examples: entry empty-handed then exit with a large red woven bag while receipt has only one small drink = suspicious. "
        "Entry already carrying the same bag and exit still carrying it = usually not suspicious. "
        "Exit carrying a normal plastic bag with several paid items that fit inside = usually reasonable. "
        "Use item names, quantity, likely physical size, and total value. Be conservative when evidence is unclear. "
        "Return strict JSON only with schema: "
        '{"hit":true|false,"score":number,"reason":string,"entry_bag_count":integer,'
        '"exit_bag_count":integer,"reasonable_with_receipt":true|false,'
        '"evidence_summary":string,"suspicious_objects":[string]}. '
        f"Input: {json.dumps(runner_payload, default=str)}"
    )
    try:
        result, meta = _call_kiosk_gemini_summary(
            prompt=prompt,
            image_urls=[],
            model_name=model_name,
            allow_text_only=True,
        )
        cost_detail = _record_gemini_cost(db, script_run_id, meta)
        normalized = {
            "hit": _as_boolish(result.get("hit")),
            "score": _coerce_number(result.get("score"), 0.0),
            "reason": str(result.get("reason") or "carry_ai"),
            "source": "gemini_carry_confidence",
            "entry_bag_count": _coerce_int(result.get("entry_bag_count"), 0),
            "exit_bag_count": _coerce_int(result.get("exit_bag_count"), 0),
            "reasonable_with_receipt": _as_boolish(result.get("reasonable_with_receipt")),
            "evidence_summary": result.get("evidence_summary"),
            "suspicious_objects": result.get("suspicious_objects") if isinstance(result.get("suspicious_objects"), list) else [],
            "evidence": carry_evidence,
            "raw_result": result,
            "cost_detail": cost_detail,
            "script_run_id": script_run_id,
            "transactions": transaction_payload,
            "total_quantity": total_quantity,
            "total_value": total_value,
        }
        repositories.finish_script_run(
            db,
            script_run_id,
            status="success",
            stdout_log=json.dumps(normalized, indent=2, default=str),
            stderr_log="",
        )
        return normalized
    except Exception as exc:
        logger.exception("Carry confidence AI failed batch_id=%s group_key=%s", batch_id, group_key)
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="",
            stderr_log=str(exc),
        )
        return {
            "hit": legacy_hit,
            "score": carry_evidence["legacy_carry_score"],
            "reason": "carry_ai_failed_legacy_fallback" if legacy_hit else "carry_ai_failed",
            "source": "legacy_carry_fallback",
            "error": str(exc),
            "evidence": carry_evidence,
            "script_run_id": script_run_id,
        }


def _ensure_session_for_confidence_group(
    db: Session,
    *,
    location_id: int,
    entry_trigger_ids: list[int],
    exit_trigger_ids: list[int],
) -> dict[str, Any] | None:
    trigger_rows: dict[int, dict[str, Any]] = {}
    for trigger_id in sorted(set(entry_trigger_ids + exit_trigger_ids)):
        try:
            trigger_rows[trigger_id] = repositories.get_trigger(db, trigger_id)
        except Exception:
            logger.exception("Could not load trigger while creating confidence session trigger_id=%s", trigger_id)

    entry_candidates = [
        row
        for trigger_id in entry_trigger_ids
        if (row := trigger_rows.get(trigger_id)) is not None
        and int(row.get("location_id") or 0) == location_id
        and _trigger_has_required_entry_identity(row)
    ]
    if not entry_candidates:
        logger.info(
            "Layer 0 confidence could not create session because no entry trigger has phone/credit identity "
            "location_id=%s entry_trigger_ids=%s exit_trigger_ids=%s",
            location_id,
            entry_trigger_ids,
            exit_trigger_ids,
        )
        return None

    def _trigger_time(row: Mapping[str, Any]) -> datetime | None:
        return _coerce_datetime_value(row.get("trigger_time"))

    def _trigger_time_sort_value(row: Mapping[str, Any], default: float) -> float:
        value = _trigger_time(row)
        return value.timestamp() if value is not None else default

    entry_trigger = min(entry_candidates, key=lambda row: _trigger_time_sort_value(row, float("inf")))
    exit_candidates = [
        row
        for trigger_id in exit_trigger_ids
        if (row := trigger_rows.get(trigger_id)) is not None
        and int(row.get("location_id") or 0) == location_id
    ]
    exit_trigger = (
        max(exit_candidates, key=lambda row: _trigger_time_sort_value(row, float("-inf")))
        if exit_candidates
        else None
    )

    if exit_trigger is not None:
        try:
            return repositories.get_session_by_trigger_pair(
                db,
                entry_trigger_id=int(entry_trigger["id"]),
                exit_trigger_id=int(exit_trigger["id"]),
            )
        except ValueError:
            pass

    session, _created = _get_or_create_session_for_entry_trigger(
        db,
        entry_trigger_id=int(entry_trigger["id"]),
        location_id=location_id,
        start_time=_trigger_time(entry_trigger),
    )

    if exit_trigger is not None:
        exit_time = _trigger_time(exit_trigger)
        if exit_time is not None:
            session = repositories.close_session(
                db,
                int(session["id"]),
                exit_time,
                exit_trigger_id=int(exit_trigger["id"]),
            )

    return session


def _queue_l1_video_for_trigger(
    db: Session,
    *,
    session_id: int,
    location_id: int,
    trigger: Mapping[str, Any],
    video_section: str,
    link_section: str,
) -> int | None:
    normalized_video_section = video_section.strip().lower()
    normalized_link_section = link_section.strip().lower()
    if normalized_video_section not in {"entrance", "kiosk"} or normalized_link_section not in {"entry", "exit"}:
        return None
    trigger_time = _coerce_datetime_value(trigger.get("trigger_time"))
    if trigger_time is None:
        return None
    trigger_id = int(trigger["id"])
    start_time = trigger_time - timedelta(seconds=40)
    end_time = trigger_time + timedelta(seconds=10)
    file_section = "exit" if normalized_link_section == "exit" else normalized_video_section
    filename = f"{file_section}_playback_{_format_dahua_playback_time(start_time)}_{_format_dahua_playback_time(end_time)}.mp4"
    output_path = session_tmp_video_path(location_id, session_id, file_section, filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_asset = repositories.find_video_asset_by_window(
        db,
        section=normalized_video_section,
        location_id=location_id,
        start_time=start_time,
        end_time=end_time,
    )
    if existing_asset is not None:
        video_asset_id = int(existing_asset["id"])
    else:
        video_asset_id = repositories.create_video_asset(
            db,
            {
                "trigger_id": trigger_id,
                "section": normalized_video_section,
                "sequence_no": None,
                "video_url": "",
                "file_path": str(output_path),
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": "not_retrieved",
                "metadata": {
                    "location_id": location_id,
                    "retrieval_source": "layer0_deep_analysis",
                    "promoted_from_layer0": True,
                    "retrieval_mode": "full_video",
                    "full_video_window_source": "trigger_time",
                    "full_video_before_seconds": 40,
                    "full_video_after_seconds": 10,
                    "session_link_section": normalized_link_section,
                },
            },
        )
        repositories.update_video_asset_url(db, video_asset_id, f"/api/v1/videos/assets/{video_asset_id}/content")
    repositories.update_video_asset_status(
        db,
        video_asset_id,
        "not_retrieved",
        metadata={
            "location_id": location_id,
            "retrieval_source": "layer0_deep_analysis",
            "promoted_from_layer0": True,
            "retrieval_mode": "full_video",
            "full_video_window_source": "trigger_time",
            "full_video_before_seconds": 40,
            "full_video_after_seconds": 10,
            "session_link_section": normalized_link_section,
        },
    )
    repositories.create_session_video_asset_link(
        db,
        session_id,
        video_asset_id,
        {
            "section": normalized_link_section,
            "sequence_no": None,
            "clip_start_time": start_time,
            "clip_end_time": end_time,
            "is_primary": normalized_link_section == "entry",
            "metadata": {
                "source": "layer0_deep_analysis",
                "trigger_id": trigger_id,
                "video_section": normalized_video_section,
            },
        },
    )
    return video_asset_id


def run_theft_confidence_for_grouping_batch(
    db: Session,
    *,
    batch_id: int,
) -> dict[str, Any]:
    lock_name = f"tds_theft_confidence_batch_{int(batch_id)}"
    lock_row = db.execute(text("select get_lock(:lock_name, 0) as acquired"), {"lock_name": lock_name}).mappings().first()
    if int((lock_row or {}).get("acquired") or 0) != 1:
        return {
            "ok": True,
            "batch_id": batch_id,
            "analyzed_count": 0,
            "promoted_count": 0,
            "skipped": True,
            "reason": "theft_confidence_batch_already_running",
        }
    try:
        return _run_theft_confidence_for_grouping_batch_locked(db, batch_id=batch_id)
    finally:
        db.execute(text("select release_lock(:lock_name)"), {"lock_name": lock_name})


def _run_theft_confidence_for_grouping_batch_locked(
    db: Session,
    *,
    batch_id: int,
) -> dict[str, Any]:
    batch = repositories.get_grouping_batch(db, batch_id)
    grouping_summary = batch.get("result_payload")
    if not isinstance(grouping_summary, Mapping):
        raise ValueError(f"Grouping batch {batch_id} does not have a result payload.")
    grouping_summary = dict(grouping_summary)
    grouping_summary_before_repair = json.dumps(grouping_summary, sort_keys=True, default=str)
    repaired_grouping_summary = _repair_grouping_with_gemini(
        db,
        parent_script_run_id=int(batch["script_run_id"]) if batch.get("script_run_id") is not None else None,
        batch_id=batch_id,
        location_id=int(batch.get("location_id") or 0) or None,
        grouping_summary=grouping_summary,
    )
    grouping_summary_after_repair = json.dumps(repaired_grouping_summary, sort_keys=True, default=str)
    if grouping_summary_after_repair != grouping_summary_before_repair:
        grouping_summary = repaired_grouping_summary
        repositories.update_grouping_batch(
            db,
            batch_id,
            {
                "result_payload": grouping_summary,
                "issue_reason": None,
            },
        )
        _persist_grouping_items_from_summary(db, batch_id=batch_id, grouping_summary=grouping_summary)
    location_id = int(grouping_summary.get("location_id") or 0)
    if location_id <= 0:
        location_id = int(batch.get("location_id") or 0)
    if location_id <= 0:
        raise ValueError(f"Grouping batch {batch_id} does not have a location_id.")
    groups = grouping_summary.get("groups") or grouping_summary.get("Groups") or []
    if not isinstance(groups, list):
        raise ValueError(f"Grouping batch {batch_id} result payload does not contain groups.")
    factor_settings = _load_filter_factor_settings(db, location_id)
    if not groups:
        repositories.upsert_filter_confidence_result(
            db,
            batch_id=batch_id,
            group_key="__no_groups__",
            location_id=location_id,
            score=0,
            need_deep_analysis=False,
            reason="no_groups",
            factor_payload={"message": "Grouping completed but returned no groups."},
        )
        return {
            "ok": True,
            "batch_id": batch_id,
            "analyzed_count": 0,
            "promoted_count": 0,
        }

    def _trigger_id_list(value: Any) -> list[int]:
        if isinstance(value, list):
            raw_values = value
        elif value is None:
            raw_values = []
        else:
            raw_values = [value]
        ids: list[int] = []
        for raw_value in raw_values:
            try:
                ids.append(int(raw_value))
            except (TypeError, ValueError):
                continue
        return ids

    analyzed_count = 0
    promoted_count_total = 0
    created_session_ids: set[int] = set()
    queued_video_asset_ids: set[int] = set()
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, Mapping):
            continue
        group_key = str(group.get("group_id") or group.get("id") or group_index)
        entry_trigger_ids = _trigger_id_list(group.get("entry"))
        exit_trigger_ids = _trigger_id_list(group.get("exit"))
        trigger_ids = entry_trigger_ids + exit_trigger_ids
        if not trigger_ids:
            repositories.upsert_filter_confidence_result(
                db,
                batch_id=batch_id,
                group_key=group_key,
                location_id=location_id,
                score=0,
                need_deep_analysis=False,
                reason="no_trigger_ids",
                factor_payload={"group": dict(group)},
            )
            analyzed_count += 1
            continue
        trigger_rows: list[dict[str, Any]] = []
        for trigger_id in trigger_ids:
            try:
                trigger_rows.append(repositories.get_trigger(db, trigger_id))
            except Exception:
                logger.exception("Could not load trigger for confidence analysis trigger_id=%s", trigger_id)
        trigger_times = [
            _coerce_datetime_value(row.get("trigger_time"))
            for row in trigger_rows
            if row.get("trigger_time") is not None
        ]
        trigger_times = [value for value in trigger_times if value is not None]
        if len(trigger_times) < 2:
            repositories.upsert_filter_confidence_result(
                db,
                batch_id=batch_id,
                group_key=group_key,
                location_id=location_id,
                score=0,
                need_deep_analysis=False,
                reason="insufficient_trigger_times",
                factor_payload={"trigger_ids": trigger_ids},
            )
            analyzed_count += 1
            continue
        start_time = min(trigger_times)
        end_time = max(trigger_times)
        transactions = repositories.list_paid_transactions_for_session_window(
            db,
            location_id=location_id,
            start_time=start_time,
            end_time=end_time,
        )
        issue_transactions = repositories.list_non_paid_transactions_for_session_window(
            db,
            location_id=location_id,
            start_time=start_time,
            end_time=end_time,
        )
        minus_alerts = repositories.list_minus_button_alerts_for_window(
            db,
            location_id=location_id,
            start_time=start_time,
            end_time=end_time,
        )
        total_quantity = sum(_coerce_int(row.get("total_items")) for row in transactions)
        total_value = 0.0
        for row in transactions:
            total_value += _coerce_number(row.get("total_amount"), 0.0)
        duration_seconds = max(0.0, (end_time - start_time).total_seconds())
        carry_score = _coerce_number(
            _read_group_value(group, "carry_something_from_store_score", "carry_score"),
            0.0,
        )
        before_is_yellow_bag = _read_group_value(group, "before_is_yellow_bag")
        after_is_yellow_bag = _read_group_value(group, "after_is_yellow_bag")

        paid_transaction_count = len(transactions)
        issue_transaction_count = len(issue_transactions)
        low_purchase = (
            paid_transaction_count == 0
            or total_quantity <= settings.filter_low_purchase_quantity
            or (0 < total_value <= settings.filter_low_purchase_value)
        )
        final_paid_receipt = transactions[-1] if transactions else None
        final_paid_time = _transaction_event_time(final_paid_receipt) if final_paid_receipt else None
        final_paid_is_low = bool(
            final_paid_receipt
            and (
                _coerce_int(final_paid_receipt.get("total_items")) <= settings.filter_low_purchase_quantity
                or (
                    0
                    < _coerce_number(final_paid_receipt.get("total_amount"), 0.0)
                    <= settings.filter_low_purchase_value
                )
            )
        )
        issue_before_final_paid = bool(
            final_paid_time
            and any(
                _seconds_between(final_paid_time, issue_time) >= 0
                for row in issue_transactions
                if (issue_time := _transaction_event_time(row)) is not None
            )
        )
        multiple_issue_hit, multiple_issue_evidence = _has_short_period_transaction_issues(
            issue_transactions,
            settings.filter_transaction_issue_short_period_seconds,
        )
        related_minus_alert_ids = _match_alert_ids_near_transactions(
            minus_alerts,
            issue_transactions,
            settings.filter_transaction_issue_short_period_seconds,
        )
        long_stay = duration_seconds >= settings.filter_long_stay_seconds
        total_customer = _coerce_int(_read_group_value(group, "total_customer", "customer_count"), 0)
        unusual_group_size_result = (
            _evaluate_unusual_group_size(
                db,
                batch_id=batch_id,
                location_id=location_id,
                trigger_rows=trigger_rows,
                entry_trigger_ids=entry_trigger_ids,
                total_customer=total_customer,
            )
            if _filter_factor_enabled(factor_settings, "unusual_group_size")
            else {"hit": False, "reason": "unusual_group_size_disabled", "current_total_customer": total_customer}
        )
        country_code_result: dict[str, Any] = {"hit": False, "reason": "not_evaluated"}
        carry_ai_result: dict[str, Any] = {"hit": False, "reason": "not_evaluated"}

        reasons: list[str] = []
        factor_details: dict[str, Any] = {}
        triggered_factors: list[str] = []
        should_continue_filtering = True

        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="long_stay_low_purchase",
            hit=long_stay and low_purchase,
            reason="long_stay_low_purchase",
            evidence={
                "duration_seconds": duration_seconds,
                "paid_transaction_count": paid_transaction_count,
                "total_quantity": total_quantity,
                "total_value": total_value,
            },
        ):
            triggered_factors.append("long_stay_low_purchase")
            should_continue_filtering = False
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="transaction_issue_low_purchase",
            hit=issue_before_final_paid and final_paid_is_low,
            reason="transaction_issue_low_purchase",
            evidence={
                "issue_transaction_count": issue_transaction_count,
                "paid_transaction_count": paid_transaction_count,
                "total_quantity": total_quantity,
                "total_value": total_value,
                "final_paid_receipt": _transaction_summary([final_paid_receipt])[0] if final_paid_receipt else None,
                "issue_before_final_paid": issue_before_final_paid,
            },
        ):
            triggered_factors.append("transaction_issue_low_purchase")
            should_continue_filtering = False
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="multiple_transaction_issues",
            hit=multiple_issue_hit,
            reason="multiple_transaction_issues",
            evidence={
                **multiple_issue_evidence,
                "related_minus_alert_ids": related_minus_alert_ids,
                "issue_transactions": _transaction_summary(issue_transactions),
            },
        ):
            triggered_factors.append("multiple_transaction_issues")
            should_continue_filtering = False
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="multiple_minus_button_alert",
            hit=len(minus_alerts) > 0,
            reason="multiple_minus_button_alert",
            evidence={"alert_count": len(minus_alerts), "alert_ids": [row.get("id") for row in minus_alerts]},
        ):
            triggered_factors.append("multiple_minus_button_alert")
            should_continue_filtering = False
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="unusual_group_size",
            hit=_as_boolish(unusual_group_size_result.get("hit")),
            reason="unusual_group_size",
            evidence=unusual_group_size_result,
        ):
            triggered_factors.append("unusual_group_size")
            should_continue_filtering = False
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="customer_risk_history",
            hit=False,
            reason="customer_risk_history",
            evidence={"implemented": False, "message": "Skipped until grouping returns stable customer identity."},
        ):
            triggered_factors.append("customer_risk_history")
            should_continue_filtering = False

        if should_continue_filtering and _filter_factor_enabled(factor_settings, "country_code_check"):
            country_code_result = _evaluate_country_code_check(
                db,
                location_id=location_id,
                trigger_rows=trigger_rows,
            )
        elif not should_continue_filtering:
            _mark_filter_factor_skipped(
                factor_details=factor_details,
                factors=factor_settings,
                factor_code="country_code_check",
                reason="skipped_after_previous_factor_hit",
            )
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="country_code_check",
            hit=_as_boolish(country_code_result.get("hit")),
            reason="country_code_check",
            evidence=country_code_result,
        ):
            triggered_factors.append("country_code_check")
            should_continue_filtering = False

        if should_continue_filtering:
            carry_signal = (
                carry_score >= settings.filter_carry_score_threshold
                or not _as_boolish(before_is_yellow_bag)
                and _as_boolish(after_is_yellow_bag)
            )
            carry_ai_result = (
                _evaluate_carry_item_signal_with_ai(
                    db,
                    batch_id=batch_id,
                    group_key=group_key,
                    location_id=location_id,
                    trigger_ids=trigger_ids,
                    group=group,
                    transactions=transactions,
                    total_quantity=total_quantity,
                    total_value=total_value,
                )
                if _filter_factor_enabled(factor_settings, "carry_item_signal")
                else {
                    "hit": carry_signal,
                    "score": carry_score,
                    "reason": "carry_item_signal_disabled",
                    "source": "legacy_carry_skipped_disabled",
                    "evidence": _group_carry_evidence(group),
                }
            )
        else:
            _mark_filter_factor_skipped(
                factor_details=factor_details,
                factors=factor_settings,
                factor_code="carry_item_signal",
                reason="skipped_after_previous_factor_hit",
            )
        if should_continue_filtering and _apply_filter_factor(
            reasons=reasons,
            factor_details=factor_details,
            factors=factor_settings,
            factor_code="carry_item_signal",
            hit=_as_boolish(carry_ai_result.get("hit")),
            reason="carry_item_signal",
            evidence=carry_ai_result,
        ):
            triggered_factors.append("carry_item_signal")

        score = float(len(triggered_factors))
        need_deep_analysis = bool(triggered_factors)
        repositories.upsert_filter_confidence_result(
            db,
            batch_id=batch_id,
            group_key=group_key,
            location_id=location_id,
            score=score,
            need_deep_analysis=need_deep_analysis,
            reason=", ".join(reasons) if reasons else "low_confidence",
            factor_payload={
                "duration_seconds": duration_seconds,
                "transaction_count": len(transactions),
                "issue_transaction_count": issue_transaction_count,
                "minus_button_alert_count": len(minus_alerts),
                "total_quantity": total_quantity,
                "total_value": total_value,
                "low_purchase": low_purchase,
                "final_paid_is_low": final_paid_is_low,
                "issue_before_final_paid": issue_before_final_paid,
                "carry_something_from_store_score": carry_score,
                "before_is_yellow_bag": before_is_yellow_bag,
                "after_is_yellow_bag": after_is_yellow_bag,
                "carry_ai_result": carry_ai_result,
                "country_code_result": country_code_result,
                "unusual_group_size_result": unusual_group_size_result,
                "total_customer": total_customer,
                "transactions": _transaction_summary(transactions),
                "issue_transactions": _transaction_summary(issue_transactions),
                "minus_alerts": [
                    {
                        "id": row.get("id"),
                        "method": row.get("method"),
                        "detail": row.get("detail"),
                        "created_at": row.get("created_at"),
                    }
                    for row in minus_alerts
                ],
                "trigger_ids": trigger_ids,
                "entry_trigger_ids": entry_trigger_ids,
                "exit_trigger_ids": exit_trigger_ids,
                "session_window_start": start_time.isoformat(),
                "session_window_end": end_time.isoformat(),
                "factor_settings": factor_settings,
                "factor_details": factor_details,
                "triggered_factors": triggered_factors,
                "decision_rule": "any_enabled_factor_hit",
            },
        )
        analyzed_count += 1
        if need_deep_analysis:
            session = _ensure_session_for_confidence_group(
                db,
                location_id=location_id,
                entry_trigger_ids=entry_trigger_ids,
                exit_trigger_ids=exit_trigger_ids,
            )
            if session:
                session_id = int(session["id"])
                session = repositories.update_session_grouping_link(
                    db,
                    session_id=session_id,
                    grouping_id=batch_id,
                )
                created_session_ids.add(session_id)
                for trigger_id in entry_trigger_ids:
                    trigger = next((row for row in trigger_rows if int(row.get("id") or 0) == trigger_id), None)
                    if trigger is None:
                        continue
                    video_asset_id = _queue_l1_video_for_trigger(
                        db,
                        session_id=session_id,
                        location_id=location_id,
                        trigger=trigger,
                        video_section="entrance",
                        link_section="entry",
                    )
                    if video_asset_id is not None:
                        queued_video_asset_ids.add(video_asset_id)
                for trigger_id in exit_trigger_ids:
                    trigger = next((row for row in trigger_rows if int(row.get("id") or 0) == trigger_id), None)
                    if trigger is None:
                        continue
                    video_asset_id = _queue_l1_video_for_trigger(
                        db,
                        session_id=session_id,
                        location_id=location_id,
                        trigger=trigger,
                        video_section="entrance",
                        link_section="exit",
                    )
                    if video_asset_id is not None:
                        queued_video_asset_ids.add(video_asset_id)
            promoted_count = repositories.promote_trigger_video_assets_to_full_retrieval(db, trigger_ids)
            promoted_count_total += promoted_count
            logger.info(
                "Layer 0 confidence promoted group_key=%s batch_id=%s trigger_ids=%s session_id=%s queued_video_assets=%s promoted_count=%s score=%.2f",
                group_key,
                batch_id,
                trigger_ids,
                session.get("id") if session else None,
                sorted(queued_video_asset_ids),
                promoted_count,
                score,
            )
    return {
        "ok": True,
        "batch_id": batch_id,
        "analyzed_count": analyzed_count,
        "promoted_count": promoted_count_total,
        "session_count": len(created_session_ids),
        "session_ids": sorted(created_session_ids),
        "queued_video_asset_ids": sorted(queued_video_asset_ids),
    }


def run_pending_theft_confidence_batches(db: Session, *, limit: int = 10) -> dict[str, Any]:
    candidates = repositories.list_pending_theft_confidence_batches(db, limit=max(1, min(limit, 50)))
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        batch_id = int(candidate["id"])
        try:
            result = run_theft_confidence_for_grouping_batch(db, batch_id=batch_id)
            results.append({"batch_id": batch_id, "status": "success", **result})
        except Exception as exc:
            logger.exception("Theft confidence failed for batch_id=%s", batch_id)
            try:
                repositories.upsert_filter_confidence_result(
                    db,
                    batch_id=batch_id,
                    group_key="__error__",
                    location_id=int(candidate.get("location_id") or 0),
                    score=0,
                    need_deep_analysis=False,
                    reason="confidence_error",
                    factor_payload={"error": str(exc)},
                )
            except Exception:
                logger.exception("Could not persist theft confidence error for batch_id=%s", batch_id)
            results.append({"batch_id": batch_id, "status": "failed", "error": str(exc)})
    return {
        "ok": True,
        "requested_limit": limit,
        "queued_count": len(candidates),
        "processed_count": len(results),
        "success_count": sum(1 for item in results if item.get("status") == "success"),
        "failed_count": sum(1 for item in results if item.get("status") == "failed"),
        "results": results,
    }


def _prepare_session_kiosk_pipeline(
    db: Session,
    *,
    session_id: int,
    location_id: int,
    session_start_time: datetime,
    session_end_time: datetime,
) -> tuple[int, dict[str, Any], list[tuple[datetime, datetime]]]:
    paid_transactions = repositories.list_paid_transactions_for_session_window(
        db,
        location_id=location_id,
        start_time=session_start_time,
        end_time=session_end_time,
    )

    repositories.delete_session_transactions(db, session_id)
    total_transaction_items = 0
    raw_windows: list[tuple[datetime, datetime]] = []
    transaction_summaries: list[dict[str, Any]] = []
    for candidate_index, transaction in enumerate(paid_transactions, start=1):
        transaction_time = _coerce_datetime_value(transaction.get("transaction_time"))
        if transaction_time is None:
            continue
        total_items = int(transaction.get("total_items") or 0)
        total_transaction_items += total_items
        window_start, window_end = _build_transaction_window_bounds(transaction_time, total_items)
        raw_windows.append((window_start, window_end))
        transaction_summaries.append(
            {
                "candidate_index": candidate_index,
                "transaction_id": transaction.get("transaction_id"),
                "receipt_number": transaction.get("receipt_number"),
                "transaction_time": transaction_time.isoformat() if hasattr(transaction_time, "isoformat") else transaction_time,
                "total_items": total_items,
                "total_amount": transaction.get("total_amount"),
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "raw_payload": {
                    **dict(transaction),
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                },
            }
        )

    merged_windows = _merge_time_windows(raw_windows)
    identification_summary: dict[str, Any] | None = None
    selected_windows = list(merged_windows)
    if len(transaction_summaries) == 1:
        selected = transaction_summaries[0]
        selected_transaction_id = repositories.create_transaction(
            db,
            session_id,
            {
                "receipt_number": str(selected.get("receipt_number") or selected.get("transaction_id") or ""),
                "transaction_time": _coerce_datetime_value(selected.get("transaction_time")),
                "total_items": int(selected.get("total_items") or 0),
                "total_amount": selected.get("total_amount"),
                "raw_payload": dict(selected.get("raw_payload") or {}),
            },
        )
        selected["session_transaction_id"] = selected_transaction_id
    elif len(transaction_summaries) > 1:
        identification_summary = _queue_kiosk_transaction_match_for_session(
            db,
            session_id=session_id,
            location_id=location_id,
            session_close_summary={
                "session_close_pipeline": {
                    "paid_transactions": transaction_summaries,
                }
            },
        )
        selected_windows = []

    session_close_summary = {
        "session_close_pipeline": {
            "session_start_time": session_start_time.isoformat() if hasattr(session_start_time, "isoformat") else session_start_time,
            "session_end_time": session_end_time.isoformat() if hasattr(session_end_time, "isoformat") else session_end_time,
            "paid_transaction_count": len(transaction_summaries),
            "paid_transactions": transaction_summaries,
            "merged_kiosk_windows": [
                {
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                }
                for start, end in merged_windows
            ],
            "selected_kiosk_windows": [
                {
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                }
                for start, end in selected_windows
            ],
            "transaction_identification": identification_summary,
            "queued_kiosk_video_asset_ids": [],
        }
    }
    return total_transaction_items, session_close_summary, selected_windows


def _resolve_shared_kiosk_video_session(
    db: Session,
    *,
    video_asset_id: int,
    video_path: str,
    session_ids: list[int],
) -> tuple[int | None, dict[str, Any]]:
    unique_session_ids = list(dict.fromkeys(int(session_id) for session_id in session_ids))
    video_asset = repositories.get_video_asset(db, video_asset_id)
    captured_start = _coerce_datetime_value(video_asset.get("captured_start_time"))
    captured_end = _coerce_datetime_value(video_asset.get("captured_end_time"))
    video_midpoint = None
    if captured_start is not None and captured_end is not None:
        video_midpoint = captured_start + (captured_end - captured_start) / 2
    details: dict[str, Any] = {
        "video_asset_id": int(video_asset_id),
        "candidate_session_ids": unique_session_ids,
        "method": "temporal_heuristic",
        "session_scores": {},
    }
    if len(unique_session_ids) == 1:
        chosen_session_id = unique_session_ids[0]
        details["chosen_session_id"] = chosen_session_id
        details["reason"] = "single_candidate"
        return chosen_session_id, details

    score_rows: list[tuple[int, tuple[int, float, float]]] = []
    for session_id in unique_session_ids:
        session = repositories.get_session(db, session_id)
        session_start = _coerce_datetime_value(session.get("start_time"))
        session_end = _resolve_session_effective_end_time(db, session_id)
        overlap_seconds = _time_window_overlap_seconds(
            captured_start,
            captured_end,
            session_start,
            session_end,
        )
        transaction_distance_seconds = (
            _seconds_to_nearest_transaction(db, session_id, video_midpoint)
            if video_midpoint is not None
            else float("inf")
        )
        transaction_hit = 1 if transaction_distance_seconds <= 60.0 else 0
        details["session_scores"][str(session_id)] = {
            "transaction_hit": transaction_hit,
            "overlap_seconds": overlap_seconds,
            "transaction_distance_seconds": None if transaction_distance_seconds == float("inf") else transaction_distance_seconds,
            "session_start": session_start.isoformat() if session_start is not None else None,
            "session_end": session_end.isoformat() if session_end is not None else None,
        }
        score_rows.append((session_id, (transaction_hit, overlap_seconds, -transaction_distance_seconds)))

    score_rows.sort(key=lambda item: item[1], reverse=True)
    if not score_rows:
        details["reason"] = "no_candidate_scores"
        return None, details

    best_session_id, best_score = score_rows[0]
    second_score = score_rows[1][1] if len(score_rows) > 1 else None
    details["best_score"] = {
        "transaction_hit": best_score[0],
        "overlap_seconds": best_score[1],
        "negative_transaction_distance_seconds": best_score[2],
    }
    if second_score is not None:
        details["second_best_score"] = {
            "transaction_hit": second_score[0],
            "overlap_seconds": second_score[1],
            "negative_transaction_distance_seconds": second_score[2],
        }
    if second_score is not None and best_score == second_score:
        details["reason"] = "score_margin_too_small"
        return None, details
    if best_score[0] == 0 and best_score[1] <= KIOSK_OWNERSHIP_MIN_MARGIN_SECONDS:
        details["reason"] = "insufficient_temporal_signal"
        return None, details
    details["chosen_session_id"] = int(best_session_id)
    details["reason"] = "resolved"
    return int(best_session_id), details


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return default


def _candidate_gallery_image_paths(
    *,
    cross_state: dict[str, Any],
    output_dir: Path,
    gallery_id: int,
) -> list[str]:
    image_paths = cross_state.get("persistent_gallery_view_paths", {}).get(gallery_id) or []
    if image_paths:
        return sorted(str(path) for path in image_paths)
    reid_dir = output_dir.parent / "reid_views" / f"ID{gallery_id}"
    if not reid_dir.exists():
        return []
    return sorted(str(path) for path in reid_dir.glob("*.jpg"))


def _resolve_runtime_gallery_entry(
    *,
    persistent_gallery: dict[int, dict[str, Any]],
    runtime_person_id: int,
) -> dict[str, Any]:
    gallery_entry = persistent_gallery.get(runtime_person_id) or {}
    if gallery_entry:
        return gallery_entry
    return {}


TERMINAL_SESSION_STATUSES = {"detected", "not_detected", "closed", "issue", "whitelisted"}


def _trigger_has_required_entry_identity(trigger: Mapping[str, Any] | None) -> bool:
    if not isinstance(trigger, Mapping):
        return False
    if trigger.get("phone_entry_id") is not None:
        return True
    if trigger.get("credit_card_entry_id") is not None:
        return True
    return False


def _get_or_create_session_for_entry_trigger(
    db: Session,
    *,
    entry_trigger_id: int,
    location_id: int,
    start_time: datetime | None,
) -> tuple[dict[str, Any], bool]:
    try:
        session = repositories.get_session_by_entry_trigger_id(db, entry_trigger_id)
        return session, False
    except ValueError:
        session = repositories.create_session(
            db,
            {
                "entry_trigger_id": entry_trigger_id,
                "exit_trigger_id": None,
                "location_id": location_id,
                "start_time": start_time,
            },
        )
        return session, True


def _resolve_open_session_customer_for_gallery(
    db: Session,
    *,
    location_id: int,
    session_customer_id: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        session_customer = repositories.get_session_customer(db, session_customer_id)
        session = repositories.get_session(db, int(session_customer["session_id"]))
    except ValueError:
        return None

    session_status = str(session.get("status") or "").strip().lower()
    if int(session.get("location_id") or 0) != location_id:
        return None
    if session_status in TERMINAL_SESSION_STATUSES:
        return None
    if session.get("end_time") is not None:
        return None
    if session_customer.get("leave_time") is not None:
        return None
    return session, session_customer


def _resolve_session_customer_identity_from_active_gallery(
    db: Session,
    *,
    location_id: int,
    session_customer_id: int,
    fallback_session_id: int | None = None,
    fallback_person_id: int | None = None,
) -> tuple[int | None, int | None] | None:
    try:
        session_customer = repositories.get_session_customer(db, session_customer_id)
        session = repositories.get_session(db, int(session_customer["session_id"]))
    except ValueError:
        if fallback_session_id is None and fallback_person_id is None:
            return None
        return fallback_session_id, fallback_person_id

    if int(session.get("location_id") or 0) != location_id:
        return None
    if session_customer.get("merged_into_session_customer_id") is not None:
        return None
    if session_customer.get("leave_time") is not None:
        return None
    return (
        int(session["id"]),
        _coerce_int(session_customer.get("person_id")) or fallback_person_id,
    )


def _extract_identity_id_from_metadata(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    return _coerce_int(value.get("identity_id"))


def _ensure_identity_for_session_customer(
    vector_db: Session,
    *,
    location_id: int,
    session_id: int | None,
    session_customer_id: int,
    person_id: int | None,
    seen_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = vector_repositories.find_identity_by_current_session_customer(
        vector_db,
        location_id=location_id,
        current_session_customer_id=session_customer_id,
    )
    payload_metadata = dict(metadata or {})
    if identity is not None:
        merged_metadata = (
            _merge_metadata(identity.get("metadata"), payload_metadata)
            if payload_metadata
            else identity.get("metadata")
        )
        return vector_repositories.update_identity_record(
            vector_db,
            int(identity["id"]),
            status="active",
            current_session_id=session_id,
            current_session_customer_id=session_customer_id,
            person_id=person_id,
            last_seen_at=seen_at,
            metadata=merged_metadata,
        )
    return vector_repositories.create_identity(
        vector_db,
        location_id=location_id,
        status="active",
        current_session_id=session_id,
        current_session_customer_id=session_customer_id,
        person_id=person_id,
        first_seen_at=seen_at,
        last_seen_at=seen_at,
        metadata=payload_metadata or None,
    )


def _canonicalize_session_customer_for_person(
    transactional_db: Session,
    vector_db: Session,
    *,
    session_id: int,
    person_id: int,
    merge_reason: str,
) -> dict[str, Any]:
    rows = repositories.list_session_customers_by_session_person(
        transactional_db,
        session_id,
        person_id,
    )
    if not rows:
        raise ValueError(
            f"Session customer for session_id={session_id} person_id={person_id} was not found."
        )

    canonical_row = next(
        (row for row in rows if row.get("merged_into_session_customer_id") is None),
        rows[0],
    )
    alias_ids = [
        int(row["id"])
        for row in rows
        if int(row["id"]) != int(canonical_row["id"])
    ]
    if alias_ids:
        vector_repositories.reassign_session_customer_aliases(
            vector_db,
            canonical_session_customer_id=int(canonical_row["id"]),
            alias_session_customer_ids=alias_ids,
            person_id=person_id,
        )
        repositories.merge_session_customer_aliases(
            transactional_db,
            canonical_session_customer_id=int(canonical_row["id"]),
            alias_session_customer_ids=alias_ids,
            merge_reason=merge_reason,
        )
        canonical_row = repositories.get_session_customer(
            transactional_db,
            int(canonical_row["id"]),
        )
        logger.info(
            "Merged duplicate session_customer rows session_id=%s person_id=%s canonical_session_customer_id=%s alias_session_customer_ids=%s reason=%s",
            session_id,
            person_id,
            canonical_row["id"],
            alias_ids,
            merge_reason,
        )
    return canonical_row


def _build_cross_state_from_active_gallery(
    location_id: int,
    *,
    gallery_date: Any | None = None,
    period_code: str | None = None,
) -> dict[str, Any]:
    vector_db = VectorSessionLocal()
    transactional_db = TransactionalSessionLocal()
    try:
        active_session_customer_ids = {
            int(row["id"])
            for row in repositories.list_active_session_customers_for_location(
                transactional_db,
                location_id=location_id,
            )
            if row.get("id") is not None
        }
        active_rows = vector_repositories.list_active_gallery_records(
            vector_db,
            location_id=location_id,
            gallery_date=gallery_date,
            period_code=period_code,
            limit=5000,
        )
        persistent_gallery: dict[int, dict[str, Any]] = {}
        persistent_gallery_view_paths: dict[int, list[str]] = {}
        next_gid = 1
        hydrated_rows = 0
        skipped_rows = 0

        for row in active_rows:
            session_customer_id = _coerce_int(row.get("session_customer_id"))
            if session_customer_id is None:
                skipped_rows += 1
                continue
            if session_customer_id not in active_session_customer_ids:
                skipped_rows += 1
                continue
            resolved_identity = _resolve_session_customer_identity_from_active_gallery(
                transactional_db,
                location_id=location_id,
                session_customer_id=session_customer_id,
                fallback_session_id=_coerce_int(row.get("session_id")),
                fallback_person_id=_coerce_int(row.get("person_id")),
            )
            if resolved_identity is None:
                skipped_rows += 1
                continue
            resolved_session_id, resolved_person_id = resolved_identity
            metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
            identity_id = _extract_identity_id_from_metadata(metadata)
            if identity_id is None:
                identity = _ensure_identity_for_session_customer(
                    vector_db,
                    location_id=location_id,
                    session_id=resolved_session_id,
                    session_customer_id=session_customer_id,
                    person_id=resolved_person_id,
                    metadata={
                        "source": "active_gallery_hydration",
                        "legacy_person_id": _coerce_int(row.get("person_id")),
                    },
                )
                identity_id = int(identity["id"])
            gallery_id = identity_id

            next_gid = max(next_gid, gallery_id + 1)

            image_paths = [str(row["image_url"])] if row.get("image_url") else []
            osnet_views = []
            osnet_tensor = _float_list_to_tensor(row.get("embedding_osnet"))
            if osnet_tensor is not None:
                osnet_views.append(osnet_tensor)

            fashion_upper, fashion_lower = _split_fashion_embedding(row.get("embedding_fashion"))

            if not osnet_views and fashion_upper is None and fashion_lower is None and not image_paths:
                continue

            gallery_entry = persistent_gallery.setdefault(
                gallery_id,
                {
                    "views": [],
                    "session_id": resolved_session_id,
                    "session_customer_id": session_customer_id,
                    "person_id": resolved_person_id,
                    "identity_id": identity_id,
                    "location_id": _coerce_int(row.get("location_id")),
                    "source": "postgresql_active_gallery",
                },
            )
            gallery_entry["views"].extend(osnet_views)
            if fashion_upper is not None:
                fashion_upper_tensor = _float_list_to_tensor(fashion_upper)
                if fashion_upper_tensor is not None and "fashion_upper_init" not in gallery_entry:
                    gallery_entry["fashion_upper_init"] = fashion_upper_tensor
            if fashion_lower is not None:
                fashion_lower_tensor = _float_list_to_tensor(fashion_lower)
                if fashion_lower_tensor is not None and "fashion_lower_init" not in gallery_entry:
                    gallery_entry["fashion_lower_init"] = fashion_lower_tensor

            if image_paths:
                existing_paths = persistent_gallery_view_paths.setdefault(gallery_id, [])
                for image_path in image_paths:
                    if image_path not in existing_paths:
                        existing_paths.append(image_path)
            hydrated_rows += 1

        logger.info(
            "Hydrated entrance persistent gallery from tds_active_gallery location_id=%s gallery_date=%s period_code=%s active_rows=%s hydrated_rows=%s skipped_rows=%s persistent_ids=%s",
            location_id,
            gallery_date,
            period_code,
            len(active_rows),
            hydrated_rows,
            skipped_rows,
            sorted(int(gid) for gid in persistent_gallery.keys()),
        )

        return {
            "next_gid": next_gid,
            "persistent_gallery": persistent_gallery,
            "persistent_gallery_view_paths": persistent_gallery_view_paths,
        }
    finally:
        vector_db.close()
        transactional_db.close()


def _hydrate_gallery_state_from_active_gallery(
    location_id: int,
    gallery_state_path: Path,
    *,
    gallery_date: Any | None = None,
    period_code: str | None = None,
) -> None:
    cross_state = _build_cross_state_from_active_gallery(
        location_id,
        gallery_date=gallery_date,
        period_code=period_code,
    )
    _write_cross_state_pickle(gallery_state_path, cross_state)


def _find_open_active_session_for_location(db: Session, location_id: int) -> dict[str, Any] | None:
    vector_db = VectorSessionLocal()
    try:
        active_session_customer_ids = {
            int(row["id"])
            for row in repositories.list_active_session_customers_for_location(
                db,
                location_id=location_id,
            )
            if row.get("id") is not None
        }
        active_rows = vector_repositories.list_active_gallery_records(
            vector_db,
            location_id=location_id,
            limit=5000,
        )
    finally:
        vector_db.close()

    seen_session_ids: set[int] = set()
    for row in active_rows:
        session_customer_id = _coerce_int(row.get("session_customer_id"))
        if session_customer_id is None or session_customer_id not in active_session_customer_ids:
            continue
        active_session_id = _coerce_int(row.get("session_id"))
        if active_session_id is None or active_session_id in seen_session_ids:
            continue
        seen_session_ids.add(active_session_id)
        try:
            session = repositories.get_session(db, active_session_id)
        except ValueError:
            continue
        status = str(session.get("status") or "").strip().lower()
        if int(session.get("location_id") or 0) != location_id:
            continue
        if status in {"detected", "not_detected", "closed", "issue", "whitelisted"}:
            continue
        return session
    try:
        return repositories.get_latest_open_session_by_location(db, location_id)
    except ValueError:
        return None


def _build_cross_state_from_session_customer_gallery(
    *,
    session_id: int,
    location_id: int,
) -> dict[str, Any]:
    vector_db = VectorSessionLocal()
    try:
        rows = vector_repositories.list_customer_gallery_records(
            vector_db,
            session_id=session_id,
        )
        persistent_gallery: dict[int, dict[str, Any]] = {}
        persistent_gallery_view_paths: dict[int, list[str]] = {}
        next_gid = 1

        for row in rows:
            gallery_id = _coerce_int(row.get("person_id"))
            if gallery_id is None:
                continue
            next_gid = max(next_gid, gallery_id + 1)

            image_paths = [str(row["image_url"])] if row.get("image_url") else []
            osnet_views = []
            osnet_tensor = _float_list_to_tensor(row.get("embedding_osnet"))
            if osnet_tensor is not None:
                osnet_views.append(osnet_tensor)

            fashion_upper, fashion_lower = _split_fashion_embedding(row.get("embedding_fashion"))
            if not osnet_views and fashion_upper is None and fashion_lower is None and not image_paths:
                continue

            gallery_entry = persistent_gallery.setdefault(
                gallery_id,
                {
                    "views": [],
                    "session_id": session_id,
                    "session_customer_id": _coerce_int(row.get("session_customer_id")),
                    "person_id": gallery_id,
                    "location_id": location_id,
                    "source": "session_customer_gallery",
                },
            )
            gallery_entry["views"].extend(osnet_views)
            if fashion_upper is not None:
                fashion_upper_tensor = _float_list_to_tensor(fashion_upper)
                if fashion_upper_tensor is not None and "fashion_upper_init" not in gallery_entry:
                    gallery_entry["fashion_upper_init"] = fashion_upper_tensor
            if fashion_lower is not None:
                fashion_lower_tensor = _float_list_to_tensor(fashion_lower)
                if fashion_lower_tensor is not None and "fashion_lower_init" not in gallery_entry:
                    gallery_entry["fashion_lower_init"] = fashion_lower_tensor

            if image_paths:
                existing_paths = persistent_gallery_view_paths.setdefault(gallery_id, [])
                for image_path in image_paths:
                    if image_path not in existing_paths:
                        existing_paths.append(image_path)

        return {
            "next_gid": next_gid,
            "persistent_gallery": persistent_gallery,
            "persistent_gallery_view_paths": persistent_gallery_view_paths,
        }
    finally:
        vector_db.close()


def _hydrate_gallery_state_from_session_customer_gallery(
    *,
    location_id: int,
    session_id: int,
    gallery_state_path: Path,
) -> None:
    cross_state = _build_cross_state_from_session_customer_gallery(
        session_id=session_id,
        location_id=location_id,
    )
    _write_cross_state_pickle(gallery_state_path, cross_state)


def _coerce_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None

    text_value = str(value).strip()
    if not text_value:
        return None

    normalized_value = text_value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized_value)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(text_value, fmt)
        except ValueError:
            continue
    return None


def _build_transaction_window_bounds(
    transaction_time: datetime,
    total_items: int,
) -> tuple[datetime, datetime]:
    extra_seconds = max(0, int(total_items) - 3) * 5
    base_padding_seconds = min(15 + extra_seconds, 40)
    before_padding_seconds = max(
        0,
        base_padding_seconds + int(settings.kiosk_transaction_extra_before_seconds),
    )
    after_padding_seconds = max(
        0,
        base_padding_seconds + int(settings.kiosk_transaction_extra_after_seconds),
    )
    return (
        transaction_time - timedelta(seconds=before_padding_seconds),
        transaction_time + timedelta(seconds=after_padding_seconds),
    )


def _merge_time_windows(windows: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not windows:
        return []
    sorted_windows = sorted(windows, key=lambda item: (item[0], item[1]))
    merged: list[tuple[datetime, datetime]] = []
    current_start, current_end = sorted_windows[0]
    for start, end in sorted_windows[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _maybe_close_session_and_prepare_kiosk(
    db: Session,
    *,
    session_id: int,
    exit_trigger_id: int | None = None,
) -> None:
    session = repositories.get_session(db, session_id)
    session_status = str(session.get("status") or "").strip().lower()
    if session_status in {"detected", "not_detected"}:
        return
    existing_kiosk_videos = repositories.list_session_video_assets(
        db,
        session_id=session_id,
        section="kiosk",
    )
    if session.get("end_time") is not None and existing_kiosk_videos:
        return

    session_customers = repositories.list_session_customers(db, session_id)
    if not session_customers:
        return
    if any(customer.get("leave_time") is None for customer in session_customers):
        return

    leave_times = [customer.get("leave_time") for customer in session_customers if customer.get("leave_time") is not None]
    if not leave_times:
        return

    session_end_time = session.get("end_time") or max(leave_times)
    session_start_time = session.get("start_time")
    if session_start_time is None:
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            end_time=session_end_time,
            exit_trigger_id=exit_trigger_id,
            total_customer=len(session_customers),
            issue_reason="Session start_time is missing; kiosk analysis could not be prepared.",
        )
        return

    total_transaction_items, session_close_summary, selected_windows = _prepare_session_kiosk_pipeline(
        db,
        session_id=session_id,
        location_id=int(session["location_id"]),
        session_start_time=session_start_time,
        session_end_time=session_end_time,
    )
    repositories.update_session_fields(
        db,
        session_id=session_id,
        end_time=session_end_time,
        exit_trigger_id=exit_trigger_id,
        total_customer=len(session_customers),
        transaction_total_items=total_transaction_items,
        result_summary=session_close_summary,
        issue_reason=None,
    )

    if not selected_windows and session_close_summary.get("session_close_pipeline", {}).get("transaction_identification"):
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="pending",
            end_time=session_end_time,
            exit_trigger_id=exit_trigger_id,
            total_customer=len(session_customers),
            transaction_total_items=total_transaction_items,
            result_summary=session_close_summary,
            issue_reason=None,
        )
        return

    if not selected_windows:
        repositories.finalize_session_result(
            db,
            session_id=session_id,
            kiosk_total_items=0,
            actual_items_brought=0,
            tolerance=1,
            extra_result_summary=session_close_summary,
        )
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="closed",
            end_time=session_end_time,
            exit_trigger_id=exit_trigger_id,
            total_customer=len(session_customers),
            transaction_total_items=total_transaction_items,
            result_summary=session_close_summary,
            issue_reason=NO_KIOSK_VIDEO_REASON,
        )
        return

    queued_video_asset_ids: list[int] = []
    try:
        for window_start, window_end in selected_windows:
            queued = retrieve_kiosk_video_window(
                db,
                session_id=session_id,
                location_id=int(session["location_id"]),
                start_time=window_start,
                end_time=window_end,
            )
            queued_video_asset_ids.append(int(queued.video_asset_id))
    except Exception as exc:
        session_close_summary["session_close_pipeline"]["queued_kiosk_video_asset_ids"] = queued_video_asset_ids
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            end_time=session_end_time,
            exit_trigger_id=exit_trigger_id,
            total_customer=len(session_customers),
            transaction_total_items=total_transaction_items,
            result_summary=session_close_summary,
            issue_reason=f"Kiosk preparation failed: {exc}",
        )
        return

    session_close_summary["session_close_pipeline"]["queued_kiosk_video_asset_ids"] = queued_video_asset_ids
    repositories.update_session_fields(
        db,
        session_id=session_id,
        status="pending",
        end_time=session_end_time,
        exit_trigger_id=exit_trigger_id,
        total_customer=len(session_customers),
        transaction_total_items=total_transaction_items,
        result_summary=session_close_summary,
        issue_reason=None,
    )


def _sync_gallery_state_after_entry(
    *,
    location_id: int,
    session_id: int | None,
    video_asset_id: int,
    exit_trigger_id: int | None,
    video_path: str,
    output_dir: Path,
    gallery_state_path: Path,
    enter_time: datetime | None,
    leave_time: datetime | None,
    captured_start_time: datetime | None,
    captured_end_time: datetime | None,
) -> None:
    tracking_summary = _load_tracking_summary(video_path, output_dir)
    reid_views_summary = _load_reid_views_summary(video_path, output_dir)
    cross_state = _load_cross_state_pickle(gallery_state_path)
    persistent_gallery = cross_state.get("persistent_gallery", {})
    reid_views_by_person: dict[int, list[dict[str, Any]]] = {}
    for customer_views in reid_views_summary.get("customers", []):
        try:
            summary_person_id = int(customer_views["person_id"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_views = customer_views.get("views") or []
        if isinstance(raw_views, list):
            reid_views_by_person[summary_person_id] = [
                view for view in raw_views if isinstance(view, dict)
            ]

    summary_customers = tracking_summary.get("customers", [])
    if not summary_customers and reid_views_by_person:
        summary_customers = [
            {
                "person_id": person_id,
                "entered": False,
                "exited": False,
                "group_id": None,
            }
            for person_id in sorted(reid_views_by_person)
        ]
        logger.info(
            "Tracking summary had no customers for video=%s; falling back to reid_views_summary customers=%s",
            Path(video_path).name,
            [customer["person_id"] for customer in summary_customers],
        )

    transactional_db = TransactionalSessionLocal()
    vector_db = VectorSessionLocal()
    try:
        trigger_row = repositories.get_trigger(transactional_db, int(exit_trigger_id)) if exit_trigger_id is not None else None
        can_create_entry_session = _trigger_has_required_entry_identity(trigger_row)
        allowed_video_link_section = "entry" if can_create_entry_session else "exit"
        entrance_mode = allowed_video_link_section == "entry"
        gallery_scope_time = (
            _coerce_datetime_value(trigger_row.get("trigger_time")) if trigger_row else None
        ) or captured_start_time
        gallery_date = _to_time_period_local_naive(gallery_scope_time).date() if gallery_scope_time is not None else None
        gallery_period_code = _period_code_for_datetime(transactional_db, location_id, gallery_scope_time)
        current_entry_session_id = session_id
        sessions_to_close: set[int] = set()
        linked_session_video_keys: set[tuple[int, str]] = set()
        resolved_exit_session_customer_ids: set[int] = set()
        resolved_exit_person_ids: set[int] = set()

        def ensure_entry_session(start_time: datetime | None) -> int:
            nonlocal current_entry_session_id
            if current_entry_session_id is not None:
                return int(current_entry_session_id)
            if not can_create_entry_session:
                raise ValueError(
                    "Cannot create session from entry analysis because trigger_event does not have a matched "
                    "phone_entry_id or credit_card_entry_id."
                )
            session, created = _get_or_create_session_for_entry_trigger(
                transactional_db,
                entry_trigger_id=int(exit_trigger_id),
                location_id=location_id,
                start_time=start_time,
            )
            current_entry_session_id = int(session["id"])
            if exit_trigger_id is not None:
                repositories.update_trigger_status(transactional_db, int(exit_trigger_id), "video_pending")
            if created:
                logger.info(
                    "Created session after unmatched entry was confirmed video=%s location_id=%s session_id=%s trigger_id=%s start_time=%s",
                    Path(video_path).name,
                    location_id,
                    current_entry_session_id,
                    exit_trigger_id,
                    start_time,
                )
            else:
                logger.info(
                    "Reused existing session after unmatched entry was confirmed video=%s location_id=%s session_id=%s trigger_id=%s",
                    Path(video_path).name,
                    location_id,
                    current_entry_session_id,
                    exit_trigger_id,
                )
            return current_entry_session_id

        def link_session_video_asset(event_session_id: int | None, section: str, metadata: dict[str, Any]) -> None:
            if event_session_id is None:
                return
            normalized_session_id = int(event_session_id)
            normalized_section = section.strip().lower()
            if normalized_section not in {"entry", "exit"}:
                return
            if normalized_section != allowed_video_link_section:
                logger.info(
                    "Skipping mixed video section link video=%s location_id=%s session_id=%s requested_section=%s allowed_section=%s",
                    Path(video_path).name,
                    location_id,
                    normalized_session_id,
                    normalized_section,
                    allowed_video_link_section,
                )
                return
            key = (normalized_session_id, normalized_section)
            if key in linked_session_video_keys:
                return
            repositories.create_session_video_asset_link(
                transactional_db,
                normalized_session_id,
                video_asset_id,
                {
                    "link_section": normalized_section,
                    "link_sequence_no": None,
                    "clip_start_time": captured_start_time,
                    "clip_end_time": captured_end_time,
                    "is_primary": normalized_section == "entry",
                    "metadata": {
                        "source": "entry_analysis",
                        "video": Path(video_path).name,
                        **metadata,
                    },
                },
            )
            linked_session_video_keys.add(key)

        raw_exit_customer_ids = tracking_summary.get("exit_customer") or []
        if entrance_mode and raw_exit_customer_ids:
            logger.info(
                "Ignoring exit_customer ids from entrance analysis video=%s location_id=%s exit_customer_ids=%s",
                Path(video_path).name,
                location_id,
                raw_exit_customer_ids,
            )
        if not entrance_mode and isinstance(raw_exit_customer_ids, list):
            for raw_session_customer_id in raw_exit_customer_ids:
                session_customer_id = _coerce_int(raw_session_customer_id)
                if session_customer_id is None:
                    continue
                try:
                    session_customer = repositories.get_session_customer(
                        transactional_db,
                        session_customer_id,
                    )
                except ValueError:
                    logger.warning(
                        "Tracking summary exit_customer id does not exist video=%s session_customer_id=%s",
                        Path(video_path).name,
                        session_customer_id,
                    )
                    continue
                active_owner = _resolve_open_session_customer_for_gallery(
                    transactional_db,
                    location_id=location_id,
                    session_customer_id=session_customer_id,
                )
                if active_owner is None:
                    logger.info(
                        "Ignoring stale tracking_summary exit_customer for closed/left session video=%s location_id=%s session_customer_id=%s",
                        Path(video_path).name,
                        location_id,
                        session_customer_id,
                    )
                    continue
                repositories.update_session_customer_leave_time(
                    transactional_db,
                    session_customer_id=session_customer_id,
                    leave_time=leave_time,
                    match_status="resolved",
                )
                exit_person_id = _coerce_int(session_customer.get("person_id"))
                sessions_to_close.add(int(session_customer["session_id"]))
                link_session_video_asset(
                    int(session_customer["session_id"]),
                    "exit",
                    {
                        "session_customer_id": session_customer_id,
                        "person_id": session_customer.get("person_id"),
                        "source_event": "tracking_summary.exit_customer",
                    },
                )
                logger.info(
                    "Updated session customer from tracking_summary exit_customer video=%s session_id=%s session_customer_id=%s leave_time=%s",
                    Path(video_path).name,
                    session_customer.get("session_id"),
                    session_customer_id,
                    leave_time,
                )
                resolved_exit_session_customer_ids.add(int(session_customer_id))
                if exit_person_id is not None:
                    resolved_exit_person_ids.add(exit_person_id)
                archived_count = vector_repositories.archive_active_gallery_by_aliases(
                    vector_db,
                    location_id=location_id,
                    session_customer_ids=[int(session_customer_id)],
                    archived_reason="customer_exited",
                    metadata_extra={
                        "source": "entry_analysis",
                        "video": Path(video_path).name,
                        "session_id": session_customer.get("session_id"),
                        "session_customer_id": session_customer_id,
                        "person_id": exit_person_id,
                        "source_event": "tracking_summary.exit_customer",
                    },
                )
                logger.info(
                    "Archived active gallery rows from tracking_summary exit_customer video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s archived_count=%s",
                    Path(video_path).name,
                    location_id,
                    session_customer.get("session_id"),
                    session_customer_id,
                    exit_person_id,
                    archived_count,
                )

        for customer in summary_customers:
            person_id = int(customer["person_id"])
            entered = bool(customer.get("entered"))
            exited = bool(customer.get("exited"))
            if entrance_mode and exited:
                logger.info(
                    "Ignoring exited flag from entrance analysis video=%s location_id=%s runtime_person_id=%s group_id=%s",
                    Path(video_path).name,
                    location_id,
                    person_id,
                    customer.get("group_id"),
                )
                exited = False
            view_rows = reid_views_by_person.get(person_id) or []
            if not entered and not exited and not view_rows:
                continue
            if not entered and not exited:
                logger.info(
                    "Skipping gallery-only customer without entry/exit event video=%s location_id=%s runtime_person_id=%s view_count=%s",
                    Path(video_path).name,
                    location_id,
                    person_id,
                    len(view_rows),
                )
                continue

            customer_enter_time = (
                _customer_event_time(
                    customer=customer,
                    tracking_summary=tracking_summary,
                    captured_start_time=captured_start_time,
                    fallback_time=enter_time,
                    event_key="entry",
                )
                if entered
                else enter_time
            )
            customer_leave_time = (
                _customer_event_time(
                    customer=customer,
                    tracking_summary=tracking_summary,
                    captured_start_time=captured_start_time,
                    fallback_time=leave_time,
                    event_key="exit",
                )
                if exited
                else None
            )

            gallery_entry = _resolve_runtime_gallery_entry(
                persistent_gallery=persistent_gallery,
                runtime_person_id=person_id,
            )
            source_session_id = _coerce_int(customer.get("session_id")) or _coerce_int(gallery_entry.get("session_id"))
            source_session_customer_id = _coerce_int(customer.get("session_customer_id")) or _coerce_int(gallery_entry.get("session_customer_id"))
            source_person_id = (
                _coerce_int(customer.get("matched_person_id"))
                or _coerce_int(gallery_entry.get("person_id"))
                or person_id
            )
            if source_session_customer_id is not None:
                resolved_identity = _resolve_session_customer_identity_from_active_gallery(
                    transactional_db,
                    location_id=location_id,
                    session_customer_id=source_session_customer_id,
                    fallback_session_id=source_session_id,
                    fallback_person_id=source_person_id,
                )
                if resolved_identity is None:
                    logger.info(
                        "Ignoring active-gallery match because session identity could not be resolved video=%s location_id=%s runtime_person_id=%s session_customer_id=%s",
                        Path(video_path).name,
                        location_id,
                        person_id,
                        source_session_customer_id,
                    )
                    source_session_customer_id = None
                    source_session_id = None
                    source_person_id = person_id
                else:
                    source_session_id, resolved_person_id = resolved_identity
                    source_person_id = resolved_person_id or source_person_id
            if exited and source_session_customer_id is None:
                fallback_person_ids = [
                    value
                    for value in (source_person_id, person_id)
                    if value is not None
                ]
                for fallback_person_id in dict.fromkeys(fallback_person_ids):
                    try:
                        fallback_session_customer = repositories.get_latest_open_session_customer_by_location_person(
                            transactional_db,
                            location_id=location_id,
                            person_id=int(fallback_person_id),
                        )
                    except ValueError:
                        continue
                    source_session_customer_id = int(fallback_session_customer["id"])
                    source_session_id = int(fallback_session_customer["session_id"])
                    source_person_id = int(fallback_session_customer["person_id"])
                    logger.info(
                        "Resolved exit customer from MySQL open session fallback video=%s location_id=%s runtime_person_id=%s session_id=%s session_customer_id=%s person_id=%s",
                        Path(video_path).name,
                        location_id,
                        person_id,
                        source_session_id,
                        source_session_customer_id,
                        source_person_id,
                    )
                    break

            if (
                not exited
                and (
                    (source_session_customer_id is not None and source_session_customer_id in resolved_exit_session_customer_ids)
                    or (source_person_id is not None and source_person_id in resolved_exit_person_ids)
                    or person_id in resolved_exit_person_ids
                )
            ):
                logger.info(
                    "Skipping customer that was already marked exited in this video to avoid recreating active gallery "
                    "video=%s location_id=%s runtime_person_id=%s source_session_customer_id=%s source_person_id=%s",
                    Path(video_path).name,
                    location_id,
                    person_id,
                    source_session_customer_id,
                    source_person_id,
                )
                continue

            if exited and source_session_customer_id is not None:
                repositories.update_session_customer_leave_time(
                    transactional_db,
                    session_customer_id=source_session_customer_id,
                    leave_time=customer_leave_time,
                    match_status="resolved",
                )
                customer_session_id = source_session_id or session_id
                session_customer = _canonicalize_session_customer_for_person(
                    transactional_db,
                    vector_db,
                    session_id=int(customer_session_id),
                    person_id=int(source_person_id or person_id),
                    merge_reason="duplicate_person_same_session",
                )
                session_customer_id = int(session_customer["id"])
                resolved_exit_session_customer_ids.add(int(session_customer_id))
                if source_person_id is not None:
                    resolved_exit_person_ids.add(int(source_person_id))
                sessions_to_close.add(customer_session_id)
                link_session_video_asset(
                    customer_session_id,
                    "exit",
                    {
                        "session_customer_id": session_customer_id,
                        "person_id": source_person_id,
                        "runtime_person_id": person_id,
                        "source_event": "tracking_summary.customers.exited",
                    },
                )
            elif exited and not entered:
                logger.warning(
                    "Exit-only customer had no matching active session; skipping new session_customer creation video=%s location_id=%s runtime_person_id=%s",
                    Path(video_path).name,
                    location_id,
                    person_id,
                )
                continue
            elif entered and source_session_customer_id is not None:
                customer_session_id = source_session_id or session_id
                session_customer = _canonicalize_session_customer_for_person(
                    transactional_db,
                    vector_db,
                    session_id=int(customer_session_id),
                    person_id=int(source_person_id or person_id),
                    merge_reason="duplicate_person_same_session",
                )
                session_customer_id = int(session_customer["id"])
                logger.info(
                    "Entered customer matched active gallery; not creating a new session customer video=%s location_id=%s runtime_person_id=%s active_session_id=%s active_session_customer_id=%s active_person_id=%s",
                    Path(video_path).name,
                    location_id,
                    person_id,
                    customer_session_id,
                    session_customer_id,
                    source_person_id,
                )
            else:
                if not can_create_entry_session:
                    logger.info(
                        "Skipping unmatched entry customer because trigger_event lacks matched phone/card identity "
                        "video=%s location_id=%s runtime_person_id=%s trigger_id=%s",
                        Path(video_path).name,
                        location_id,
                        person_id,
                        exit_trigger_id,
                    )
                    continue
                entry_session_id = ensure_entry_session(customer_enter_time)
                repositories.create_session_customer(
                    transactional_db,
                    entry_session_id,
                    {
                        "person_id": person_id,
                        "enter_time": customer_enter_time,
                        "kiosk_start_time": None,
                        "leave_time": customer_leave_time,
                        "match_status": "resolved" if exited else "tracked",
                    },
                )
                session_customer = _canonicalize_session_customer_for_person(
                    transactional_db,
                    vector_db,
                    session_id=int(entry_session_id),
                    person_id=person_id,
                    merge_reason="duplicate_person_same_session",
                )
                session_customer_id = int(session_customer["id"])
                customer_session_id = entry_session_id
                if exited:
                    sessions_to_close.add(entry_session_id)
                if entered:
                    link_session_video_asset(
                        customer_session_id,
                        "entry",
                        {
                            "session_customer_id": session_customer_id,
                            "person_id": person_id,
                            "runtime_person_id": person_id,
                            "source_event": "tracking_summary.customers.entered",
                        },
                    )
                if exited:
                    link_session_video_asset(
                        customer_session_id,
                        "exit",
                        {
                            "session_customer_id": session_customer_id,
                            "person_id": person_id,
                            "runtime_person_id": person_id,
                            "source_event": "tracking_summary.customers.exited",
                        },
                    )

            if entered and source_session_customer_id is not None:
                active_session_id = customer_session_id
                active_session_customer_id = session_customer_id
                active_person_id = source_person_id or person_id
            elif entered:
                active_session_id = customer_session_id
                active_session_customer_id = session_customer_id
                active_person_id = person_id
            else:
                active_session_id = source_session_id or customer_session_id
                active_session_customer_id = source_session_customer_id or session_customer_id
                active_person_id = source_person_id or person_id
            delete_session_customer_ids = [active_session_customer_id, session_customer_id]
            identity_seen_at = customer_leave_time if exited else customer_enter_time
            identity_row = _ensure_identity_for_session_customer(
                vector_db,
                location_id=location_id,
                session_id=active_session_id,
                session_customer_id=int(active_session_customer_id),
                person_id=active_person_id,
                seen_at=identity_seen_at,
                metadata={
                    "source": "entry_analysis",
                    "runtime_person_id": person_id,
                    "group_id": customer.get("group_id"),
                },
            )
            identity_id = int(identity_row["id"])

            vector_repositories.delete_customer_gallery_records_for_session_customer(
                vector_db,
                session_customer_id=session_customer_id,
            )

            osnet_views = gallery_entry.get("views") or []
            fashion_embedding = _combine_fashion_embedding(
                gallery_entry.get("fashion_upper_init"),
                gallery_entry.get("fashion_lower_init"),
            )
            image_paths = _candidate_gallery_image_paths(
                cross_state=cross_state,
                output_dir=output_dir,
                gallery_id=person_id,
            )
            canonical_view = view_rows[0] if view_rows else None
            canonical_image_url = (
                canonical_view.get("image_url")
                if canonical_view and canonical_view.get("image_url")
                else (image_paths[0] if image_paths else None)
            )
            canonical_image_public_url = (
                canonical_view.get("image_public_url")
                if canonical_view and canonical_view.get("image_public_url")
                else None
            )
            if not canonical_image_public_url and canonical_view and canonical_view.get("image_object_key"):
                canonical_image_public_url = _spaces_download_url_for_object_key(str(canonical_view["image_object_key"]))
            canonical_osnet = (
                canonical_view.get("embedding_osnet")
                if canonical_view
                else (_tensor_like_to_float_list(osnet_views[0]) if osnet_views else None)
            )
            canonical_fashion = (
                canonical_view.get("embedding_fashion")
                if canonical_view
                else fashion_embedding
            )
            if not canonical_image_public_url:
                canonical_image_public_url = _upload_customer_gallery_image_to_spaces(
                    canonical_image_url,
                    location_id=location_id,
                    session_id=customer_session_id,
                    session_customer_id=session_customer_id,
                    person_id=active_person_id,
                    output_dir=output_dir,
                )
            if (
                canonical_osnet is not None
                or canonical_fashion is not None
                or canonical_image_url is not None
            ):
                vector_repositories.create_customer_gallery_record(
                    vector_db,
                    location_id=location_id,
                    session_id=customer_session_id,
                    session_customer_id=session_customer_id,
                    person_id=active_person_id,
                    image_url=canonical_image_url,
                    image_public_url=canonical_image_public_url,
                    image_kind="reid_view" if canonical_osnet is not None else "fashion_view",
                    embedding_osnet=canonical_osnet,
                    embedding_fashion=canonical_fashion,
                    metadata={
                        "source": "entry_analysis",
                        "identity_id": identity_id,
                        "exited": bool(customer.get("exited")),
                        "group_id": customer.get("group_id"),
                        "active_view_count": len(view_rows) if view_rows else len(osnet_views),
                        "active_image_count": len(view_rows) if view_rows else len(image_paths),
                    },
                )
            if exited:
                vector_repositories.update_identity_record(
                    vector_db,
                    identity_id,
                    status="exited",
                    current_session_id=active_session_id,
                    current_session_customer_id=active_session_customer_id,
                    person_id=active_person_id,
                    last_seen_at=customer_leave_time,
                    exited_at=customer_leave_time,
                    metadata={
                        "source": "entry_analysis_exit",
                        "runtime_person_id": person_id,
                        "group_id": customer.get("group_id"),
                    },
                )
                archived_count = vector_repositories.archive_active_gallery_by_aliases(
                    vector_db,
                    location_id=location_id,
                    session_customer_ids=delete_session_customer_ids,
                    archived_reason="customer_exited",
                    metadata_extra={
                        "source": "entry_analysis",
                        "identity_id": identity_id,
                        "video": Path(video_path).name,
                        "session_id": active_session_id,
                        "session_customer_id": active_session_customer_id,
                        "person_id": active_person_id,
                    },
                )
                logger.info(
                    "Archived active gallery rows for exited customer video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s archived_count=%s",
                    Path(video_path).name,
                    location_id,
                    active_session_id,
                    active_session_customer_id,
                    active_person_id,
                    archived_count,
                )
                continue

            if view_rows or osnet_views or fashion_embedding is not None or image_paths:
                vector_repositories.delete_active_gallery_by_aliases(
                    vector_db,
                    location_id=location_id,
                    session_customer_ids=delete_session_customer_ids,
                )
                active_metadata = {
                    "source": "entry_analysis",
                    "identity_id": identity_id,
                    "group_id": customer.get("group_id"),
                    "entered": bool(customer.get("entered")),
                    "exited": False,
                    "gallery_date": gallery_date.isoformat() if gallery_date is not None else None,
                    "period_code": gallery_period_code,
                }
                if view_rows:
                    for index, view_row in enumerate(view_rows):
                        vector_repositories.create_active_gallery_record(
                            vector_db,
                            location_id=location_id,
                            session_id=active_session_id,
                            session_customer_id=active_session_customer_id,
                            person_id=active_person_id,
                            image_url=view_row.get("image_url"),
                            image_kind=str(view_row.get("image_kind") or "reid_view"),
                            gallery_date=gallery_date,
                            period_code=gallery_period_code,
                            embedding_osnet=view_row.get("embedding_osnet"),
                            embedding_fashion=view_row.get("embedding_fashion"),
                            metadata={**active_metadata, "view_index": index},
                        )
                    logger.info(
                        "Inserted active gallery rows from reid_views video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s row_count=%s entered=%s exited=%s",
                        Path(video_path).name,
                        location_id,
                        active_session_id,
                        active_session_customer_id,
                        active_person_id,
                        len(view_rows),
                        entered,
                        exited,
                    )
                elif osnet_views:
                    for index, osnet_view in enumerate(osnet_views):
                        image_url = image_paths[index] if index < len(image_paths) else (image_paths[0] if image_paths else None)
                        vector_repositories.create_active_gallery_record(
                            vector_db,
                            location_id=location_id,
                            session_id=active_session_id,
                            session_customer_id=active_session_customer_id,
                            person_id=active_person_id,
                            image_url=image_url,
                            image_kind="reid_view",
                            gallery_date=gallery_date,
                            period_code=gallery_period_code,
                            embedding_osnet=_tensor_like_to_float_list(osnet_view),
                            embedding_fashion=fashion_embedding,
                            metadata={**active_metadata, "view_index": index},
                        )
                    logger.info(
                        "Inserted active gallery rows from runtime gallery video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s row_count=%s entered=%s exited=%s",
                        Path(video_path).name,
                        location_id,
                        active_session_id,
                        active_session_customer_id,
                        active_person_id,
                        len(osnet_views),
                        entered,
                        exited,
                    )
                elif fashion_embedding is not None or canonical_image_url is not None:
                    vector_repositories.create_active_gallery_record(
                        vector_db,
                        location_id=location_id,
                        session_id=active_session_id,
                        session_customer_id=active_session_customer_id,
                        person_id=active_person_id,
                        image_url=canonical_image_url,
                        image_kind="fashion_view",
                        gallery_date=gallery_date,
                        period_code=gallery_period_code,
                        embedding_osnet=None,
                        embedding_fashion=fashion_embedding,
                        metadata=active_metadata,
                    )
                    logger.info(
                        "Inserted active gallery fallback row video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s entered=%s exited=%s",
                        Path(video_path).name,
                        location_id,
                        active_session_id,
                        active_session_customer_id,
                        active_person_id,
                        entered,
                        exited,
                    )
            else:
                logger.info(
                    "Preserving active gallery because there is no exit event and no fresh embeddings to refresh "
                    "video=%s location_id=%s session_id=%s session_customer_id=%s person_id=%s entered=%s exited=%s",
                    Path(video_path).name,
                    location_id,
                    active_session_id,
                    active_session_customer_id,
                    active_person_id,
                    entered,
                    exited,
                )

        for close_session_id in sorted(sessions_to_close):
            _maybe_close_session_and_prepare_kiosk(
                transactional_db,
                session_id=close_session_id,
                exit_trigger_id=exit_trigger_id,
            )
    finally:
        vector_db.close()
        transactional_db.close()


def _lookup_video_asset_by_file_path(db: Session, video_path: str) -> dict[str, Any] | None:
    try:
        return repositories.get_video_asset_by_file_path(db, video_path)
    except ValueError:
        return None


def _upload_processed_video_for_asset(
    db: Session,
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    processed_video_path: Path,
    source_video_path: str,
    output_dir: Path,
    script_name: str,
    model_name: str | None,
) -> None:
    target = _build_processed_video_upload_target(
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        processed_video_path=processed_video_path,
        source_video_path=source_video_path,
    )
    upload_result = upload_private_file(
        processed_video_path,
        target["object_key"],
        content_type=guess_media_type(str(processed_video_path)),
    )
    _apply_processed_video_upload_result(
        db,
        video_asset_row=video_asset_row,
        object_key=upload_result["object_key"],
        video_url=str(upload_result.get("public_url") or target["video_url"]),
    )


def _record_followup_failure(
    db: Session,
    *,
    script_run_id: int,
    session_id: int | None,
    trigger_id: int | None,
    script_name: str,
    model_name: str | None,
    stdout: str,
    stderr: str,
) -> None:
    repositories.revise_script_run(
        db,
        script_run_id,
        status="failed",
        stdout_log=stdout,
        stderr_log=stderr,
    )


def create_trigger_and_session(
    db: Session,
    *,
    location_id: int,
    aqara_event_id: str | None,
    trigger_time: datetime,
    raw_payload: dict | None,
    whitelist_hit: bool,
    create_session: bool = True,
    trigger_source: str = "aqara",
) -> dict:
    trigger = repositories.create_trigger(
        db,
        {
            "location_id": location_id,
            "aqara_event_id": aqara_event_id,
            "trigger_source": trigger_source,
            "trigger_time": trigger_time,
            "raw_payload": raw_payload,
        },
    )
    if whitelist_hit:
        repositories.update_trigger_status(db, trigger["id"], "whitelisted")
        trigger = repositories.get_trigger(db, trigger["id"])
        return {"trigger": trigger, "session": None, "message": "Whitelist hit. Downstream LLM flow can be skipped."}

    if not _trigger_has_required_entry_identity(trigger):
        repositories.update_trigger_status(db, trigger["id"], "pending")
        trigger = repositories.get_trigger(db, trigger["id"])
        return {
            "trigger": trigger,
            "session": None,
            "message": "Trigger created. Session creation blocked until trigger_event has matched phone_entry_id or credit_card_entry_id.",
        }

    if not create_session:
        repositories.update_trigger_status(db, trigger["id"], "pending")
        trigger = repositories.get_trigger(db, trigger["id"])
        return {"trigger": trigger, "session": None, "message": "Trigger created. Session creation deferred."}

    session, created = _get_or_create_session_for_entry_trigger(
        db,
        entry_trigger_id=int(trigger["id"]),
        location_id=location_id,
        start_time=trigger_time - timedelta(seconds=int(settings.entrance_trigger_extra_before_seconds)),
    )
    repositories.update_trigger_status(db, trigger["id"], "video_pending")
    trigger = repositories.get_trigger(db, trigger["id"])
    return {
        "trigger": trigger,
        "session": session,
        "message": "Trigger and session created." if created else "Trigger updated and existing session reused.",
    }


def _format_dahua_playback_time(value: datetime) -> str:
    return value.strftime("%Y_%m_%d_%H_%M_%S")


def _build_dahua_rtsp_playback_url(
    *,
    host: str,
    username: str,
    password: str,
    rtsp_port: int,
    channel: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    if not host or not username or not password:
        raise ValueError("Dahua RTSP settings are incomplete. Set location Dahua host, username, and password.")

    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    start = _format_dahua_playback_time(start_time)
    end = _format_dahua_playback_time(end_time)
    return (
        f"rtsp://{encoded_username}:{encoded_password}@{host}:{rtsp_port}"
        f"/cam/playback?channel={channel}&subtype={settings.dahua_playback_subtype}"
        f"&starttime={start}&endtime={end}"
    )


def _build_retrieval_command(rtsp_url: str, output_path: Path) -> list[str]:
    codec = settings.dahua_output_video_codec.strip()
    if codec == "copy":
        return [
            settings.ffmpeg_bin,
            "-y",
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-c",
            "copy",
            str(output_path),
        ]

    return [
        settings.ffmpeg_bin,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-c:v",
        codec,
        "-preset",
        settings.dahua_output_preset,
        "-crf",
        str(settings.dahua_output_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]


def _prepare_video_retrieval(
    db: Session,
    *,
    section: str,
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    alert_id: int | None,
    start_time: datetime,
    end_time: datetime,
) -> VideoRetrievalQueued:
    existing_asset = None
    if section == "kiosk":
        existing_asset = repositories.find_video_asset_by_window(
            db,
            section=section,
            location_id=location_id,
            start_time=start_time,
            end_time=end_time,
        )
    if existing_asset is not None:
        existing_video_asset_id = int(existing_asset["id"])
        access_url = str(existing_asset.get("video_url") or f"/api/v1/videos/assets/{existing_video_asset_id}/content")
        output_path = _resolve_local_video_retrieval_output_path(
            video_asset_row=existing_asset,
            location_id=location_id,
            session_id=session_id,
            trigger_id=trigger_id,
            alert_id=alert_id,
        )
        if session_id is not None:
            repositories.create_session_video_asset_link(
                db,
                session_id,
                existing_video_asset_id,
                {
                    "section": section,
                    "sequence_no": None,
                    "clip_start_time": start_time,
                    "clip_end_time": end_time,
                    "is_primary": False,
                    "metadata": {
                        "retrieval_source": "shared_existing_video_asset",
                        "reused_video_asset_id": existing_video_asset_id,
                    },
                },
            )
        cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section=section)
        location = repositories.get_location_endpoint(db, location_id)
        delayed_seconds = int(cctv.get("delayed_seconds") or 0)
        adjusted_start_time = start_time - timedelta(seconds=delayed_seconds)
        adjusted_end_time = end_time - timedelta(seconds=delayed_seconds)
        rtsp_url = _build_dahua_rtsp_playback_url(
            host=str(location.get("dahua_host") or "").strip(),
            username=str(location.get("dahua_username") or "").strip(),
            password=decrypt_secret(str(location.get("dahua_password_encrypted") or "").strip()),
            rtsp_port=int(location.get("rtsp_port") or settings.dahua_rtsp_port),
            channel=str(cctv.get("recorder_channel") or "").strip(),
            start_time=adjusted_start_time,
            end_time=adjusted_end_time,
        )
        return VideoRetrievalQueued(
            video_asset_id=existing_video_asset_id,
            session_id=session_id,
            trigger_id=trigger_id,
            location_id=location_id,
            section=section,
            requested_start_time=start_time,
            requested_end_time=end_time,
            delayed_seconds=delayed_seconds,
            adjusted_start_time=adjusted_start_time,
            adjusted_end_time=adjusted_end_time,
            output_path=output_path,
            rtsp_url=rtsp_url,
            dahua_host=str(location.get("dahua_host") or "").strip(),
            dahua_username=str(location.get("dahua_username") or "").strip(),
            rtsp_port=int(location.get("rtsp_port") or settings.dahua_rtsp_port),
            status=str(existing_asset.get("status") or "not_retrieved"),
            video_url=access_url,
        )

    cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section=section)
    location = repositories.get_location_endpoint(db, location_id)
    channel = str(cctv.get("recorder_channel") or "").strip()
    if not channel:
        raise ValueError(f"{section.capitalize()} CCTV record does not have a recorder_channel.")
    dahua_host = str(location.get("dahua_host") or "").strip()
    dahua_username = str(location.get("dahua_username") or "").strip()
    dahua_password_encrypted = str(location.get("dahua_password_encrypted") or "").strip()
    if not dahua_host or not dahua_username or not dahua_password_encrypted:
        raise ValueError(f"Location {location_id} does not have complete Dahua host credentials configured.")
    dahua_password = decrypt_secret(dahua_password_encrypted)
    rtsp_port = int(location.get("rtsp_port") or settings.dahua_rtsp_port)
    delayed_seconds = int(cctv.get("delayed_seconds") or 0)
    adjusted_start_time = start_time - timedelta(seconds=delayed_seconds)
    adjusted_end_time = end_time - timedelta(seconds=delayed_seconds)

    rtsp_url = _build_dahua_rtsp_playback_url(
        host=dahua_host,
        username=dahua_username,
        password=dahua_password,
        rtsp_port=rtsp_port,
        channel=channel,
        start_time=adjusted_start_time,
        end_time=adjusted_end_time,
    )
    filename = f"{section}_playback_{_format_dahua_playback_time(start_time)}_{_format_dahua_playback_time(end_time)}.mp4"
    if session_id is not None:
        output_path = session_tmp_video_path(location_id, session_id, section, filename)
    elif trigger_id is not None:
        output_path = trigger_tmp_video_path(location_id, trigger_id, section, filename)
    elif alert_id is not None:
        output_path = alert_tmp_video_path(location_id, alert_id, section, filename)
    else:
        raise ValueError("Either session_id, trigger_id, or alert_id is required for video retrieval.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retention_until = end_time + timedelta(days=3)
    video_asset_id = repositories.create_video_asset(
        db,
        {
            "trigger_id": trigger_id,
            "section": section,
            "sequence_no": None,
            "video_url": "",
            "file_path": str(output_path),
            "captured_start_time": start_time,
            "captured_end_time": end_time,
            "retrieved_at": None,
            "analyzed_at": None,
            "retention_until": retention_until,
            "status": "not_retrieved",
            "metadata": {
                "location_id": location_id,
                "alert_id": alert_id,
                "retrieval_source": "dahua_rtsp_playback",
            },
        },
    )
    access_url = f"/api/v1/videos/assets/{video_asset_id}/content"
    repositories.update_video_asset_url(db, video_asset_id, access_url)
    if session_id is not None:
        repositories.create_session_video_asset_link(
            db,
            session_id,
            video_asset_id,
            {
                "section": section,
                "sequence_no": None,
                "clip_start_time": start_time,
                "clip_end_time": end_time,
                "is_primary": True,
                "metadata": {
                    "retrieval_source": "dahua_rtsp_playback",
                },
            },
        )
    return VideoRetrievalQueued(
        video_asset_id=video_asset_id,
        session_id=session_id,
        trigger_id=trigger_id,
        location_id=location_id,
        section=section,
        requested_start_time=start_time,
        requested_end_time=end_time,
        delayed_seconds=delayed_seconds,
        adjusted_start_time=adjusted_start_time,
        adjusted_end_time=adjusted_end_time,
        output_path=str(output_path),
        rtsp_url=rtsp_url,
        dahua_host=dahua_host,
        dahua_username=dahua_username,
        rtsp_port=rtsp_port,
        status="not_retrieved",
        video_url=access_url,
    )


def _resolve_local_video_retrieval_output_path(
    *,
    video_asset_row: dict[str, Any],
    location_id: int,
    session_id: int | None,
    trigger_id: int | None,
    alert_id: int | None,
) -> str:
    existing_file_path = str(video_asset_row.get("file_path") or "").strip()
    if existing_file_path and not existing_file_path.startswith("spaces://"):
        return existing_file_path

    captured_start = video_asset_row.get("captured_start_time")
    captured_end = video_asset_row.get("captured_end_time")
    section = str(video_asset_row.get("section") or "").strip()
    if not section:
        raise ValueError(f"Video asset {int(video_asset_row['id'])} does not have a section.")
    if not isinstance(captured_start, datetime) or not isinstance(captured_end, datetime):
        raise ValueError(f"Video asset {int(video_asset_row['id'])} is missing capture timestamps.")

    filename = f"{section}_playback_{_format_dahua_playback_time(captured_start)}_{_format_dahua_playback_time(captured_end)}.mp4"
    if session_id is not None:
        return str(session_tmp_video_path(location_id, session_id, section, filename))
    if trigger_id is not None:
        return str(trigger_tmp_video_path(location_id, trigger_id, section, filename))
    if alert_id is not None:
        return str(alert_tmp_video_path(location_id, alert_id, section, filename))
    raise ValueError("Either session_id, trigger_id, or alert_id is required for video retrieval.")


def build_retrieval_job_from_video_asset(db: Session, video_asset_id: int) -> VideoRetrievalQueued:
    video_asset = repositories.get_video_asset(db, video_asset_id)
    section = str(video_asset.get("section") or "").strip()
    if not section:
        raise ValueError(f"Video asset {video_asset_id} does not have a section.")

    trigger_id = video_asset.get("trigger_id")
    session_id = None
    location_id = None

    if trigger_id is not None:
        trigger = repositories.get_trigger(db, int(trigger_id))
        location_id = int(trigger["location_id"])
    else:
        candidates = repositories.list_pending_video_asset_retrievals(db, limit=500)
        matched = next((row for row in candidates if int(row["id"]) == video_asset_id), None)
        if matched is None:
            matched = next((row for row in repositories.list_running_video_asset_retrievals(db) if int(row["id"]) == video_asset_id), None)
        if matched is None:
            raise ValueError(f"Could not resolve session/location for video asset {video_asset_id}.")
        session_id = int(matched["session_id"]) if matched.get("session_id") is not None else None
        location_id = int(matched["location_id"]) if matched.get("location_id") is not None else None

    if location_id is None:
        raise ValueError(f"Could not resolve location_id for video asset {video_asset_id}.")

    start_time = video_asset.get("captured_start_time")
    end_time = video_asset.get("captured_end_time")
    if start_time is None or end_time is None:
        raise ValueError(f"Video asset {video_asset_id} is missing capture timestamps.")

    cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section=section)
    location = repositories.get_location_endpoint(db, location_id)
    channel = str(cctv.get("recorder_channel") or "").strip()
    if not channel:
        raise ValueError(f"{section.capitalize()} CCTV record does not have a recorder_channel.")
    dahua_host = str(location.get("dahua_host") or "").strip()
    dahua_username = str(location.get("dahua_username") or "").strip()
    dahua_password_encrypted = str(location.get("dahua_password_encrypted") or "").strip()
    if not dahua_host or not dahua_username or not dahua_password_encrypted:
        raise ValueError(f"Location {location_id} does not have complete Dahua host credentials configured.")
    dahua_password = decrypt_secret(dahua_password_encrypted)
    rtsp_port = int(location.get("rtsp_port") or settings.dahua_rtsp_port)
    delayed_seconds = int(cctv.get("delayed_seconds") or 0)
    adjusted_start_time = start_time - timedelta(seconds=delayed_seconds)
    adjusted_end_time = end_time - timedelta(seconds=delayed_seconds)
    rtsp_url = _build_dahua_rtsp_playback_url(
        host=dahua_host,
        username=dahua_username,
        password=dahua_password,
        rtsp_port=rtsp_port,
        channel=channel,
        start_time=adjusted_start_time,
        end_time=adjusted_end_time,
    )
    output_path = _resolve_local_video_retrieval_output_path(
        video_asset_row=video_asset,
        location_id=location_id,
        session_id=session_id,
        trigger_id=int(trigger_id) if trigger_id is not None else None,
        alert_id=(
            int(video_asset.get("metadata", {}).get("alert_id"))
            if isinstance(video_asset.get("metadata"), dict) and video_asset.get("metadata", {}).get("alert_id") is not None
            else None
        ),
    )
    metadata = video_asset.get("metadata") if isinstance(video_asset.get("metadata"), Mapping) else {}
    claimed_from_status = str(metadata.get("claimed_from_status") or video_asset.get("status") or "retrieving")
    retrieval_mode = str(metadata.get("retrieval_mode") or "").strip().lower() or None

    return VideoRetrievalQueued(
        video_asset_id=video_asset_id,
        session_id=session_id,
        trigger_id=int(trigger_id) if trigger_id is not None else None,
        location_id=location_id,
        section=section,
        requested_start_time=start_time,
        requested_end_time=end_time,
        delayed_seconds=delayed_seconds,
        adjusted_start_time=adjusted_start_time,
        adjusted_end_time=adjusted_end_time,
        output_path=output_path,
        rtsp_url=rtsp_url,
        dahua_host=dahua_host,
        dahua_username=dahua_username,
        rtsp_port=rtsp_port,
        status=claimed_from_status,
        video_url=str(video_asset.get("video_url") or f"/api/v1/videos/assets/{video_asset_id}/content"),
        retrieval_mode=retrieval_mode,
    )


def build_retrieval_job_from_trigger_frame_asset(db: Session, frame_asset_id: int) -> TriggerFrameAssetRetrievalQueued:
    frame_asset = repositories.get_trigger_frame_asset(db, frame_asset_id)
    trigger_id = int(frame_asset["trigger_id"])
    location_id = int(frame_asset["location_id"])
    start_time = frame_asset.get("start_time")
    end_time = frame_asset.get("end_time")
    if start_time is None or end_time is None:
        raise ValueError(f"Frame asset {frame_asset_id} is missing capture timestamps.")

    section = "entrance"
    cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section=section)
    location = repositories.get_location_endpoint(db, location_id)
    channel = str(cctv.get("recorder_channel") or "").strip()
    if not channel:
        raise ValueError("Entrance CCTV record does not have a recorder_channel.")
    dahua_host = str(location.get("dahua_host") or "").strip()
    dahua_username = str(location.get("dahua_username") or "").strip()
    dahua_password_encrypted = str(location.get("dahua_password_encrypted") or "").strip()
    if not dahua_host or not dahua_username or not dahua_password_encrypted:
        raise ValueError(f"Location {location_id} does not have complete Dahua host credentials configured.")
    dahua_password = decrypt_secret(dahua_password_encrypted)
    rtsp_port = int(location.get("rtsp_port") or settings.dahua_rtsp_port)
    delayed_seconds = int(cctv.get("delayed_seconds") or 0)
    adjusted_start_time = start_time - timedelta(seconds=delayed_seconds)
    adjusted_end_time = end_time - timedelta(seconds=delayed_seconds)
    rtsp_url = _build_dahua_rtsp_playback_url(
        host=dahua_host,
        username=dahua_username,
        password=dahua_password,
        rtsp_port=rtsp_port,
        channel=channel,
        start_time=adjusted_start_time,
        end_time=adjusted_end_time,
    )
    output_dir = (
        tmp_media_root()
        / f"location_{location_id}"
        / f"trigger_{trigger_id}"
        / section
        / "frames"
        / f"frame_asset_{frame_asset_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    return TriggerFrameAssetRetrievalQueued(
        frame_asset_id=frame_asset_id,
        trigger_id=trigger_id,
        location_id=location_id,
        section=section,
        requested_start_time=start_time,
        requested_end_time=end_time,
        adjusted_start_time=adjusted_start_time,
        adjusted_end_time=adjusted_end_time,
        output_dir=str(output_dir),
        rtsp_url=rtsp_url,
    )


def build_entrance_analysis_job_from_video_asset(db: Session, video_asset_id: int) -> EntranceAnalysisQueued:
    video_asset = repositories.get_video_asset(db, video_asset_id)
    trigger_id = video_asset.get("trigger_id")
    if trigger_id is None:
        raise ValueError(f"Video asset {video_asset_id} does not have a related trigger.")
    if str(video_asset.get("section") or "") != "entrance":
        raise ValueError(f"Video asset {video_asset_id} is not an entrance video.")
    video_path = str(video_asset.get("file_path") or "").strip()
    if not video_path:
        raise ValueError(f"Video asset {video_asset_id} does not have a file path.")
    trigger = repositories.get_trigger(db, int(trigger_id))
    session_id = None
    try:
        existing_session = repositories.get_session_by_entry_trigger_id(db, int(trigger_id))
        session_id = int(existing_session["id"])
    except ValueError:
        session_id = None
    video_asset = _repair_video_asset_source_file_path_for_analysis(
        db,
        video_asset_row=video_asset,
        location_id=int(trigger["location_id"]),
        session_id=session_id,
        trigger_id=int(trigger_id),
    )
    video_path = str(video_asset.get("file_path") or "").strip()
    _ensure_analysis_uses_source_video(video_path, video_asset_id=video_asset_id)
    return EntranceAnalysisQueued(
        video_asset_id=video_asset_id,
        trigger_id=int(trigger_id),
        session_id=session_id,
        location_id=int(trigger["location_id"]),
        video_path=video_path,
        model_name=None,
    )


def _trigger_frame_spaces_key(
    *,
    location_id: int,
    trigger_id: int,
    section: str,
    filename: str,
) -> str:
    return build_spaces_object_key(
        f"location_{location_id}",
        f"trigger_{trigger_id}",
        section,
        "frames",
        filename,
    )


def _build_frame_batch_capture_command(
    rtsp_url: str,
    *,
    start_offset_seconds: float,
    gap_seconds: float,
    frame_count: int,
    output_pattern: Path,
) -> list[str]:
    return [
        settings.ffmpeg_bin,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-ss",
        f"{max(0.0, start_offset_seconds):.3f}",
        "-vf",
        f"fps={1 / max(0.04, gap_seconds):.6f}",
        "-frames:v",
        str(max(1, frame_count)),
        "-q:v",
        "2",
        str(output_pattern),
    ]


def _build_frame_capture_command(rtsp_url: str, offset_seconds: float, output_path: Path) -> list[str]:
    return [
        settings.ffmpeg_bin,
        "-y",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-ss",
        f"{max(0.0, offset_seconds):.3f}",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]


def _trigger_frame_offsets(duration_seconds: float, frame_count: int) -> list[float]:
    frame_count = max(1, int(frame_count))
    duration_seconds = max(0.0, float(duration_seconds))
    if frame_count == 1 or duration_seconds <= 0:
        return [0.0 for _ in range(frame_count)]
    step_seconds = duration_seconds / max(1, frame_count - 1)
    return [min(duration_seconds, index * step_seconds) for index in range(frame_count)]


def _selected_trigger_frame_offsets(duration_seconds: float) -> tuple[list[float], int]:
    planned_frame_count = max(1, int(settings.trigger_frame_count))
    selected_frame_count = min(planned_frame_count, _grouping_frames_per_trigger())
    return _trigger_frame_offsets(duration_seconds, planned_frame_count)[:selected_frame_count], planned_frame_count


def _run_trigger_frame_retrieval_job(
    *,
    video_asset_id: int,
    trigger_id: int | None,
    location_id: int,
    section: str,
    start_time: datetime,
    end_time: datetime,
    rtsp_url: str,
    output_path: str,
) -> None:
    db = TransactionalSessionLocal()
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=None,
        trigger_id=trigger_id,
        script_name="retrieve_video",
        model_name="dahua_rtsp_playback:trigger_frames",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    try:
        if trigger_id is None:
            raise ValueError("Frame retrieval requires a trigger_id.")
        if not is_spaces_configured():
            raise RuntimeError("Frame retrieval requires DigitalOcean Spaces.")

        frame_root = Path(output_path).with_suffix("") / "frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        duration_seconds = max(0.0, (end_time - start_time).total_seconds())
        offsets, planned_frame_count = _selected_trigger_frame_offsets(duration_seconds)
        frame_count = len(offsets)
        gap_seconds = max(0.04, offsets[1] - offsets[0] if len(offsets) > 1 else duration_seconds or 1.0)

        output_pattern = frame_root / f"trigger_{trigger_id}_frame_%02d.jpg"
        command = _build_frame_batch_capture_command(
            rtsp_url,
            start_offset_seconds=offsets[0] if offsets else 0.0,
            gap_seconds=gap_seconds,
            frame_count=frame_count,
            output_pattern=output_pattern,
        )
        completed_stderr = ""
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(1, int(settings.retrieval_ffmpeg_timeout_seconds)),
            )
            stdout_parts.append(completed.stdout or "")
            stderr_parts.append(completed.stderr or "")
            completed_stderr = str(completed.stderr or "").strip()
        except subprocess.TimeoutExpired as exc:
            completed_stderr = str(exc.stderr or f"ffmpeg timed out after {settings.retrieval_ffmpeg_timeout_seconds}s")
            stdout_parts.append(str(exc.stdout or ""))
            stderr_parts.append(completed_stderr)

        batch_paths = [
            frame_root / f"trigger_{trigger_id}_frame_{index:02d}.jpg"
            for index in range(1, frame_count + 1)
        ]
        missing_batch_paths = [
            (index, offsets[index - 1], path)
            for index, path in enumerate(batch_paths, start=1)
            if not path.exists()
        ]
        if missing_batch_paths:
            stderr_parts.append(
                f"Batch frame capture produced {frame_count - len(missing_batch_paths)}/{frame_count} images; "
                "falling back to per-frame capture for missing images."
            )
            for index, offset_seconds, local_path in missing_batch_paths:
                fallback_command = _build_frame_capture_command(rtsp_url, offset_seconds, local_path)
                try:
                    completed = subprocess.run(
                        fallback_command,
                        capture_output=True,
                        text=True,
                        timeout=max(1, int(settings.retrieval_ffmpeg_timeout_seconds)),
                    )
                    stdout_parts.append(completed.stdout or "")
                    stderr_parts.append(completed.stderr or "")
                except subprocess.TimeoutExpired as exc:
                    stdout_parts.append(str(exc.stdout or ""))
                    stderr_parts.append(str(exc.stderr or f"ffmpeg timed out after {settings.retrieval_ffmpeg_timeout_seconds}s"))

        frame_payload: list[dict[str, Any]] = []
        for index, offset_seconds in enumerate(offsets, start=1):
            sample_time = start_time + timedelta(seconds=offset_seconds)
            filename = f"trigger_{trigger_id}_frame_{index:02d}_{sample_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            local_path = frame_root / f"trigger_{trigger_id}_frame_{index:02d}.jpg"
            frame_record: dict[str, Any] = {
                "index": index,
                "sample_time": sample_time.isoformat(),
                "offset_seconds": round(float(offset_seconds), 3),
                "status": "ok" if local_path.exists() else "failed",
                "stderr": completed_stderr[-1000:],
            }
            if frame_record["status"] == "ok":
                object_key = _trigger_frame_spaces_key(
                    location_id=location_id,
                    trigger_id=trigger_id,
                    section=section,
                    filename=filename,
                )
                upload_result = upload_private_file(local_path, object_key, content_type=guess_media_type(str(local_path)))
                frame_record["image_object_key"] = object_key
                frame_record["image_url"] = str(upload_result.get("public_url") or _spaces_download_url_for_object_key(object_key))
            frame_payload.append(frame_record)

        ok_frames = [frame for frame in frame_payload if frame.get("status") == "ok"]
        final_status = "retrieved" if ok_frames else "issue"
        metadata = {
            "retrieval_mode": "trigger_frames",
            "sampling_strategy": "first_n_from_even_window",
            "planned_frame_count": planned_frame_count,
            "frame_count_requested": frame_count,
            "first_frame_count_from_planned_window": frame_count,
            "frame_gap_seconds": round(float(gap_seconds), 3),
            "frames_retrieved_count": len(ok_frames),
            "frames": frame_payload,
        }
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": str(ok_frames[-1].get("image_url") if ok_frames else f"/api/v1/videos/assets/{video_asset_id}/content"),
                "file_path": output_path,
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": datetime.now(UTC) if ok_frames else None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": final_status,
                "metadata": metadata,
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="success" if ok_frames else "failed",
            stdout_log="\n".join(part for part in stdout_parts if part),
            stderr_log="\n".join(part for part in stderr_parts if part),
        )
    except Exception as exc:
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": f"/api/v1/videos/assets/{video_asset_id}/content",
                "file_path": output_path,
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": "issue",
                "metadata": {"retrieval_mode": "trigger_frames", "error": str(exc)},
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="\n".join(part for part in stdout_parts if part),
            stderr_log=str(exc),
        )
    finally:
        db.close()


def _run_video_retrieval_job(
    *,
    video_asset_id: int,
    session_id: int | None,
    trigger_id: int | None,
    location_id: int,
    section: str,
    start_time: datetime,
    end_time: datetime,
    delayed_seconds: int,
    adjusted_start_time: datetime,
    adjusted_end_time: datetime,
    output_path: str,
    rtsp_url: str,
    dahua_host: str,
    dahua_username: str,
    rtsp_port: int,
) -> None:
    db = TransactionalSessionLocal()
    command: list[str] = []
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name="retrieve_video",
        model_name=f"dahua_rtsp_playback:{settings.dahua_output_video_codec}",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    try:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        command = _build_retrieval_command(rtsp_url, target_path)
        completed = subprocess.run(command, capture_output=True, text=True)
        status = "success" if completed.returncode == 0 else "failed"
        source_upload_url = f"/api/v1/videos/assets/{video_asset_id}/content"
        source_file_path = output_path
        metadata: dict[str, Any] | None = None
        if status == "success":
            try:
                video_asset_row = repositories.get_video_asset(db, video_asset_id)
                upload_target = _build_source_video_upload_target(
                    video_asset_row=video_asset_row,
                    location_id=location_id,
                    session_id=session_id,
                    trigger_id=trigger_id,
                    source_video_path=output_path,
                )
                upload_result = upload_private_file(
                    target_path,
                    upload_target["object_key"],
                    content_type=guess_media_type(str(target_path)),
                )
                source_upload_url = str(upload_result.get("public_url") or upload_target["video_url"])
                source_file_path = f"spaces://{upload_target['object_key']}"
            except Exception as upload_exc:
                metadata = {
                    "warning": "spaces_upload_failed_after_retrieval",
                    "spaces_upload_error": str(upload_exc),
                }
                logger.warning(
                    "Spaces upload failed after successful retrieval for video_asset_id=%s; keeping local file ready. error=%s",
                    video_asset_id,
                    upload_exc,
                )
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": source_upload_url,
                "file_path": source_file_path,
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": datetime.now(UTC) if status == "success" else None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": "ready" if status == "success" else "issue",
                "metadata": metadata,
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status=status,
            stdout_log=completed.stdout,
            stderr_log=completed.stderr,
        )
        if status != "success" and section == "kiosk" and session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=(completed.stderr or completed.stdout or "Kiosk video retrieval failed.").strip(),
            )
    except Exception as exc:
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": f"/api/v1/videos/assets/{video_asset_id}/content",
                "file_path": output_path,
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": "issue",
                "metadata": None,
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="",
            stderr_log=str(exc),
        )
    finally:
        db.close()


def start_trigger_frame_asset_retrieval_job(job: TriggerFrameAssetRetrievalQueued) -> None:
    db = TransactionalSessionLocal()
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=None,
        trigger_id=job.trigger_id,
        script_name="retrieve_video",
        model_name="dahua_rtsp_playback:trigger_frame_asset",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
        runner_payload={
            "frame_asset_id": job.frame_asset_id,
            "location_id": job.location_id,
            "start_time": job.requested_start_time.isoformat(),
            "end_time": job.requested_end_time.isoformat(),
        },
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    try:
        if not is_spaces_configured():
            raise RuntimeError("Frame retrieval requires DigitalOcean Spaces.")

        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration_seconds = max(0.0, (job.requested_end_time - job.requested_start_time).total_seconds())
        offsets, planned_frame_count = _selected_trigger_frame_offsets(duration_seconds)
        frame_count = len(offsets)
        gap_seconds = max(0.04, offsets[1] - offsets[0] if len(offsets) > 1 else duration_seconds or 1.0)

        output_pattern = output_dir / f"trigger_{job.trigger_id}_asset_{job.frame_asset_id}_frame_%02d.jpg"
        command = _build_frame_batch_capture_command(
            job.rtsp_url,
            start_offset_seconds=offsets[0] if offsets else 0.0,
            gap_seconds=gap_seconds,
            frame_count=frame_count,
            output_pattern=output_pattern,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(1, int(settings.retrieval_ffmpeg_timeout_seconds)),
            )
            stdout_parts.append(completed.stdout or "")
            stderr_parts.append(completed.stderr or "")
        except subprocess.TimeoutExpired as exc:
            stdout_parts.append(str(exc.stdout or ""))
            stderr_parts.append(str(exc.stderr or f"ffmpeg timed out after {settings.retrieval_ffmpeg_timeout_seconds}s"))

        batch_paths = [
            output_dir / f"trigger_{job.trigger_id}_asset_{job.frame_asset_id}_frame_{index:02d}.jpg"
            for index in range(1, frame_count + 1)
        ]
        missing_batch_paths = [
            (index, offsets[index - 1], path)
            for index, path in enumerate(batch_paths, start=1)
            if not path.exists()
        ]
        if missing_batch_paths:
            stderr_parts.append(
                f"Batch frame capture produced {frame_count - len(missing_batch_paths)}/{frame_count} images; "
                "falling back to per-frame capture for missing images."
            )
            for index, offset_seconds, local_path in missing_batch_paths:
                fallback_command = _build_frame_capture_command(job.rtsp_url, offset_seconds, local_path)
                try:
                    completed = subprocess.run(
                        fallback_command,
                        capture_output=True,
                        text=True,
                        timeout=max(1, int(settings.retrieval_ffmpeg_timeout_seconds)),
                    )
                    stdout_parts.append(completed.stdout or "")
                    stderr_parts.append(completed.stderr or "")
                except subprocess.TimeoutExpired as exc:
                    stdout_parts.append(str(exc.stdout or ""))
                    stderr_parts.append(str(exc.stderr or f"ffmpeg timed out after {settings.retrieval_ffmpeg_timeout_seconds}s"))

        frame_rows: list[dict[str, Any]] = []
        for index, offset_seconds in enumerate(offsets, start=1):
            sample_time = job.requested_start_time + timedelta(seconds=offset_seconds)
            local_path = output_dir / f"trigger_{job.trigger_id}_asset_{job.frame_asset_id}_frame_{index:02d}.jpg"
            filename = (
                f"trigger_{job.trigger_id}_asset_{job.frame_asset_id}_"
                f"frame_{index:02d}_{sample_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            row: dict[str, Any] = {
                "frame_index": index,
                "sample_time": sample_time,
                "image_url": None,
                "status": "failed",
            }
            if local_path.exists():
                object_key = _trigger_frame_spaces_key(
                    location_id=job.location_id,
                    trigger_id=job.trigger_id,
                    section=job.section,
                    filename=filename,
                )
                upload_result = upload_private_file(local_path, object_key, content_type=guess_media_type(str(local_path)))
                row["image_url"] = str(upload_result.get("public_url") or _spaces_download_url_for_object_key(object_key))
                row["status"] = "ok"
            frame_rows.append(row)

        repositories.replace_trigger_frame_rows(
            db,
            frame_asset_id=job.frame_asset_id,
            trigger_id=job.trigger_id,
            frames=frame_rows,
        )
        ok_count = sum(1 for row in frame_rows if row.get("status") == "ok")
        final_status = "retrieved" if ok_count else "issue"
        if ok_count >= frame_count:
            error = None
        elif ok_count:
            error = (
                f"Retrieved {ok_count}/{frame_count} trigger frames from the first {frame_count} of {planned_frame_count} planned samples. "
                "Check the retrieve_video script log for missing-frame ffmpeg errors."
            )
        else:
            error = "No trigger frames were retrieved."
        repositories.update_trigger_frame_asset_status(db, job.frame_asset_id, final_status, error=error)
        repositories.finish_script_run(
            db,
            script_run_id,
            status="success" if ok_count else "failed",
            stdout_log="\n".join(part for part in stdout_parts if part),
            stderr_log="\n".join(part for part in stderr_parts if part),
        )
    except Exception as exc:
        repositories.update_trigger_frame_asset_status(db, job.frame_asset_id, "issue", error=str(exc))
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="\n".join(part for part in stdout_parts if part),
            stderr_log="\n".join(part for part in stderr_parts if part) or str(exc),
        )
    finally:
        db.close()


def start_video_retrieval_job(job: VideoRetrievalQueued) -> None:
    if job.status == "not_retrieved" and job.section == "entrance" and job.retrieval_mode != "full_video":
        _run_trigger_frame_retrieval_job(
            video_asset_id=job.video_asset_id,
            trigger_id=job.trigger_id,
            location_id=job.location_id,
            section=job.section,
            start_time=job.requested_start_time,
            end_time=job.requested_end_time,
            rtsp_url=job.rtsp_url,
            output_path=job.output_path,
        )
        return
    _run_video_retrieval_job(
        video_asset_id=job.video_asset_id,
        session_id=job.session_id,
        trigger_id=job.trigger_id,
        location_id=job.location_id,
        section=job.section,
        start_time=job.requested_start_time,
        end_time=job.requested_end_time,
        delayed_seconds=job.delayed_seconds,
        adjusted_start_time=job.adjusted_start_time,
        adjusted_end_time=job.adjusted_end_time,
        output_path=job.output_path,
        rtsp_url=job.rtsp_url,
        dahua_host=job.dahua_host,
        dahua_username=job.dahua_username,
        rtsp_port=job.rtsp_port,
    )


def start_entrance_analysis_job(job: EntranceAnalysisQueued) -> ScriptExecutionResult:
    db = TransactionalSessionLocal()
    try:
        result = run_entry_for_trigger(
            db,
            trigger_id=job.trigger_id,
            session_id=job.session_id,
            video_path=job.video_path,
            model_name=job.model_name,
        )
        if result.status == "failed":
            repositories.update_video_asset_status(db, job.video_asset_id, "issue")
        elif result.status == "pending":
            repositories.update_video_asset_status(db, job.video_asset_id, "ready")
        return result
    except Exception as exc:
        repositories.update_video_asset_status(db, job.video_asset_id, "issue")
        script_run_id = repositories.create_script_run_started(
            db,
            session_id=job.session_id,
            trigger_id=job.trigger_id,
            script_name="entry",
            model_name=job.model_name or "analysis_worker",
            command=SCRIPT_RUN_COMMAND_REDACTED,
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="",
            stderr_log=str(exc),
        )
        if section == "kiosk" and session_id is not None:
            repositories.update_session_fields(
                db,
                session_id=int(session_id),
                status="issue",
                issue_reason=str(exc),
            )
        raise
    finally:
        db.close()


def retrieve_entrance_video_window(
    db: Session,
    *,
    trigger_id: int,
    location_id: int,
    start_time: datetime,
    end_time: datetime,
) -> VideoRetrievalQueued:
    return _prepare_video_retrieval(
        db,
        section="entrance",
        location_id=location_id,
        session_id=None,
        trigger_id=trigger_id,
        alert_id=None,
        start_time=start_time,
        end_time=end_time,
    )


def retrieve_kiosk_video_window(
    db: Session,
    *,
    session_id: int,
    location_id: int,
    start_time: datetime,
    end_time: datetime,
) -> VideoRetrievalQueued:
    return _prepare_video_retrieval(
        db,
        section="kiosk",
        location_id=location_id,
        session_id=session_id,
        trigger_id=None,
        alert_id=None,
        start_time=start_time,
        end_time=end_time,
    )


def retrieve_alert_kiosk_video_window(
    db: Session,
    *,
    alert_id: int,
    location_id: int,
    start_time: datetime,
    end_time: datetime,
) -> VideoRetrievalQueued:
    return _prepare_video_retrieval(
        db,
        section="kiosk",
        location_id=location_id,
        session_id=None,
        trigger_id=None,
        alert_id=alert_id,
        start_time=start_time,
        end_time=end_time,
    )


def ensure_kiosk_video_assets_for_session(db: Session, session_id: int) -> list[int]:
    session = repositories.get_session(db, session_id)
    existing_kiosk_videos = repositories.list_session_video_assets(
        db,
        session_id=session_id,
        section="kiosk",
    )
    if existing_kiosk_videos:
        return [int(row["video_asset_id"]) for row in existing_kiosk_videos if row.get("video_asset_id") is not None]

    summary = dict(session.get("result_summary") or {})
    pipeline = dict(summary.get("session_close_pipeline") or {})
    selected_windows = pipeline.get("selected_kiosk_windows")
    merged_windows = selected_windows if isinstance(selected_windows, list) else (pipeline.get("merged_kiosk_windows") or [])
    if isinstance(selected_windows, list) and not selected_windows and pipeline.get("transaction_identification"):
        return []
    if not isinstance(merged_windows, list) or not merged_windows:
        session_start_time = session.get("start_time")
        session_end_time = session.get("end_time")
        if session_start_time is None or session_end_time is None:
            repositories.update_session_fields(
                db,
                session_id=session_id,
                status="issue",
                issue_reason="Kiosk retry failed because session start_time or end_time is missing.",
            )
            return []

        total_transaction_items, prepared_summary, recomputed_windows = _prepare_session_kiosk_pipeline(
            db,
            session_id=session_id,
            location_id=int(session["location_id"]),
            session_start_time=session_start_time,
            session_end_time=session_end_time,
        )
        prepared_pipeline = dict(prepared_summary.get("session_close_pipeline") or {})
        if not recomputed_windows and prepared_pipeline.get("transaction_identification"):
            summary["session_close_pipeline"] = prepared_pipeline
            repositories.update_session_fields(
                db,
                session_id=session_id,
                status="pending",
                transaction_total_items=total_transaction_items,
                result_summary=summary,
                issue_reason=None,
            )
            return []
        if not recomputed_windows:
            merged_summary = {**summary, **prepared_summary}
            repositories.finalize_session_result(
                db,
                session_id=session_id,
                kiosk_total_items=0,
                actual_items_brought=0,
                tolerance=1,
                extra_result_summary=merged_summary,
            )
            repositories.update_session_fields(
                db,
                session_id=session_id,
                status="closed",
                transaction_total_items=total_transaction_items,
                result_summary=merged_summary,
                issue_reason=NO_KIOSK_VIDEO_REASON,
            )
            return []

        pipeline = dict(prepared_summary.get("session_close_pipeline") or {})
        summary["session_close_pipeline"] = pipeline
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="pending",
            transaction_total_items=total_transaction_items,
            result_summary=summary,
            issue_reason=None,
        )
        merged_windows = pipeline.get("selected_kiosk_windows") or pipeline.get("merged_kiosk_windows") or []

    location_id = int(session["location_id"])
    queued_video_asset_ids: list[int] = []
    try:
        for window in merged_windows:
            if not isinstance(window, dict):
                continue
            start_value = window.get("start_time")
            end_value = window.get("end_time")
            if not start_value or not end_value:
                continue
            start_time = _coerce_datetime_value(start_value)
            end_time = _coerce_datetime_value(end_value)
            if start_time is None or end_time is None:
                continue
            queued = retrieve_kiosk_video_window(
                db,
                session_id=session_id,
                location_id=location_id,
                start_time=start_time,
                end_time=end_time,
            )
            queued_video_asset_ids.append(int(queued.video_asset_id))
    except Exception as exc:
        pipeline["queued_kiosk_video_asset_ids"] = queued_video_asset_ids
        summary["session_close_pipeline"] = pipeline
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="issue",
            result_summary=summary,
            issue_reason=f"Kiosk retry failed while creating video asset: {exc}",
        )
        return queued_video_asset_ids

    if queued_video_asset_ids:
        pipeline["queued_kiosk_video_asset_ids"] = queued_video_asset_ids
        summary["session_close_pipeline"] = pipeline
        repositories.update_session_fields(
            db,
            session_id=session_id,
            result_summary=summary,
        )
    return queued_video_asset_ids


def run_entry_for_trigger(
    db: Session,
    *,
    trigger_id: int,
    session_id: int | None = None,
    video_path: str,
    model_name: str | None = None,
    output_dir: str | None = None,
    gallery_state_path: str | None = None,
) -> ScriptExecutionResult:
    trigger = repositories.get_trigger(db, trigger_id)
    location_id = int(trigger["location_id"])
    if session_id is None:
        try:
            existing_session = repositories.get_session_by_entry_trigger_id(db, trigger_id)
            session_id = int(existing_session["id"])
        except ValueError:
            session_id = None
    if session_id is not None:
        session = repositories.get_session(db, session_id)
        location_id = int(session["location_id"])
    resolved_output_dir = (
        Path(output_dir)
        if output_dir
        else default_trigger_output_dir(location_id, trigger_id)
    )
    resolved_gallery_state = (
        Path(gallery_state_path)
        if gallery_state_path
        else trigger_gallery_state_path(location_id, trigger_id, "entrance")
    )
    if not _runpod_runner_enabled():
        raise RuntimeError(
            "Runpod entry analysis is not configured. Set THEFT_API_RUNPOD_ENTRY_ENDPOINT_ID "
            "and THEFT_API_RUNPOD_API_KEY in the API environment."
        )
    trigger_time = _coerce_datetime_value(trigger.get("trigger_time"))
    gallery_date = _to_time_period_local_naive(trigger_time).date() if trigger_time is not None else None
    gallery_period_code = _period_code_for_datetime(db, location_id, trigger_time)
    _hydrate_gallery_state_from_active_gallery(
        location_id,
        resolved_gallery_state,
        gallery_date=gallery_date,
        period_code=gallery_period_code,
    )
    video_asset_row = _lookup_video_asset_by_file_path(db, video_path)
    if video_asset_row is None:
        raise RuntimeError("Runpod entry analysis requires a matching video_asset row for the source video.")
    if _runpod_dispatch_busy(db):
        return ScriptExecutionResult(
            script_run_id=None,
            runner_job_id=None,
            script_name="entry",
            model_name=model_name,
            status="pending",
            command=["runpod_serverless", "entry"],
            stdout="",
            stderr="",
            message="Runpod analysis worker is busy. Entry job was not enqueued yet; retry after the current analysis finishes.",
        )
    repositories.update_video_asset_status(db, int(video_asset_row["id"]), "processing")
    video_asset_row, source_video_url = _ensure_source_video_ready_for_runner(
        db,
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
    )
    gallery_state_url = _upload_runner_input_file(
        resolved_gallery_state,
        kind="gallery_state",
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
        section="entrance",
    )[1]
    upload_target = _build_processed_video_upload_target(
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name="entry",
        source_video_path=video_path,
    )
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name="entry",
        model_name=model_name or "runpod_runner",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    enqueue_result = _enqueue_runpod_runner(
        kind="entry",
        payload={
            "kind": "entry",
            "video_url": source_video_url,
            "gallery_state_url": gallery_state_url,
            "processed_video_object_key": upload_target["object_key"],
            "callback_url": _build_runpod_webhook_url("entry"),
            "script_run_id": script_run_id,
            "session_id": session_id,
            "location_id": location_id,
            "model_name": model_name,
        },
    )
    repositories.assign_script_run_runner_job(
        db,
        script_run_id,
        runner_job_id=enqueue_result.job_id,
        runner_payload={
            "video_asset_id": int(video_asset_row["id"]),
            "location_id": location_id,
            "session_id": session_id,
            "trigger_id": trigger_id,
            "video_path": video_path,
            "output_dir": str(resolved_output_dir),
            "gallery_state_path": str(resolved_gallery_state),
            "processed_video_url": upload_target["video_url"],
        },
    )
    return ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=enqueue_result.job_id,
        script_name="entry",
        model_name=model_name,
        status="running",
        command=["runpod_serverless", "entry"],
        stdout="",
        stderr="",
        message="Runpod entry job queued. FastAPI will update MySQL when the webhook completes.",
    )


def run_kiosk_for_session(
    db: Session,
    *,
    session_id: int,
    video_path: str,
    model_name: str | None = None,
    output_dir: str | None = None,
    gallery_state_path: str | None = None,
) -> ScriptExecutionResult:
    session = repositories.get_session(db, session_id)
    location_id = int(session["location_id"])
    workdir = build_session_workdir(location_id, session_id)
    resolved_output_dir = Path(output_dir) if output_dir else default_video_output_dir(location_id, session_id, video_path)
    resolved_gallery_state = (
        Path(gallery_state_path)
        if gallery_state_path
        else build_private_gallery_state_path(location_id, session_id)
    )
    if not _runpod_runner_enabled():
        raise RuntimeError(
            "Runpod kiosk analysis is not configured. Set THEFT_API_RUNPOD_KIOSK_ENDPOINT_ID "
            "and THEFT_API_RUNPOD_API_KEY in the API environment."
        )
    _hydrate_gallery_state_from_session_customer_gallery(
        location_id=location_id,
        session_id=session_id,
        gallery_state_path=resolved_gallery_state,
    )
    video_asset_row = _lookup_video_asset_by_file_path(db, video_path)
    if video_asset_row is None:
        raise RuntimeError("Runpod kiosk analysis requires a matching video_asset row for the source video.")
    if _runpod_dispatch_busy(db):
        return ScriptExecutionResult(
            script_run_id=None,
            runner_job_id=None,
            script_name="kiosk",
            model_name=model_name,
            status="pending",
            command=["runpod_serverless", "kiosk"],
            stdout="",
            stderr="",
            message="Runpod analysis worker is busy. Kiosk job was not enqueued yet; retry after the current analysis finishes.",
        )
    repositories.update_video_asset_status(db, int(video_asset_row["id"]), "processing")
    video_asset_row, source_video_url = _ensure_source_video_ready_for_runner(
        db,
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=None,
    )
    gallery_state_url = _upload_runner_input_file(
        resolved_gallery_state,
        kind="gallery_state",
        location_id=location_id,
        session_id=session_id,
        trigger_id=None,
        section=str(video_asset_row.get("section") or "kiosk"),
    )[1]
    upload_target = _build_processed_video_upload_target(
        video_asset_row=video_asset_row,
        location_id=location_id,
        session_id=session_id,
        trigger_id=None,
        script_name="kiosk",
        source_video_path=video_path,
    )
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=None,
        script_name="kiosk",
        model_name=model_name or "runpod_runner",
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    enqueue_result = _enqueue_runpod_runner(
        kind="kiosk",
        payload={
            "kind": "kiosk",
            "video_url": source_video_url,
            "gallery_state_url": gallery_state_url,
            "processed_video_object_key": upload_target["object_key"],
            "callback_url": _build_runpod_webhook_url("kiosk"),
            "script_run_id": script_run_id,
            "model_name": model_name,
        },
    )
    repositories.assign_script_run_runner_job(
        db,
        script_run_id,
        runner_job_id=enqueue_result.job_id,
        runner_payload={
            "video_asset_id": int(video_asset_row["id"]),
            "location_id": location_id,
            "session_id": session_id,
            "trigger_id": None,
            "video_path": video_path,
            "output_dir": str(resolved_output_dir),
            "gallery_state_path": str(resolved_gallery_state),
            "processed_video_url": upload_target["video_url"],
        },
    )
    return ScriptExecutionResult(
        script_run_id=script_run_id,
        runner_job_id=enqueue_result.job_id,
        script_name="kiosk",
        model_name=model_name,
        status="running",
        command=["runpod_serverless", "kiosk"],
        stdout="",
        stderr="",
        message="Runpod kiosk job queued. FastAPI will update MySQL when the webhook completes.",
    )


def build_kiosk_analysis_job_from_video_asset(db: Session, video_asset_id: int) -> KioskAnalysisQueued:
    video_asset = repositories.get_video_asset(db, video_asset_id)
    if str(video_asset.get("section") or "") != "kiosk":
        raise ValueError(f"Video asset {video_asset_id} is not a kiosk video.")
    video_path = str(video_asset.get("file_path") or "").strip()
    if not video_path:
        raise ValueError(f"Video asset {video_asset_id} does not have a file path.")
    session_links = [
        row
        for row in repositories.list_video_asset_session_links(db, video_asset_id)
        if str(row.get("section") or "").strip().lower() == "kiosk"
        and str(row.get("session_status") or "").strip().lower() == "pending"
    ]
    session_ids = [int(row["session_id"]) for row in session_links if row.get("session_id") is not None]
    session_id: int | None
    if not session_ids:
        raise ValueError(f"Video asset {video_asset_id} does not have a related session.")
    if len(dict.fromkeys(session_ids)) > 1:
        chosen_session_id, ownership_meta = _resolve_shared_kiosk_video_session(
            db,
            video_asset_id=video_asset_id,
            video_path=video_path,
            session_ids=session_ids,
        )
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": video_asset.get("video_url"),
                "file_path": video_asset.get("file_path"),
                "captured_start_time": video_asset.get("captured_start_time"),
                "captured_end_time": video_asset.get("captured_end_time"),
                "retrieved_at": video_asset.get("retrieved_at"),
                "analyzed_at": video_asset.get("analyzed_at"),
                "retention_until": video_asset.get("retention_until"),
                "status": video_asset.get("status"),
                "metadata": _merge_metadata(video_asset.get("metadata"), {"kiosk_ownership_resolution": ownership_meta}),
            },
        )
        if chosen_session_id is None:
            raise ValueError(
                f"Video asset {video_asset_id} is linked to multiple pending sessions "
                f"{sorted(dict.fromkeys(session_ids))}, but ownership is ambiguous."
            )
        repositories.delete_session_video_asset_links_for_video_asset_except(
            db,
            video_asset_id=video_asset_id,
            keep_session_id=chosen_session_id,
            section="kiosk",
        )
        session_id = chosen_session_id
    else:
        session_id = int(session_ids[0])
    session = repositories.get_session(db, session_id)
    video_asset = _repair_video_asset_source_file_path_for_analysis(
        db,
        video_asset_row=video_asset,
        location_id=int(session["location_id"]),
        session_id=session_id,
        trigger_id=None,
    )
    video_path = str(video_asset.get("file_path") or "").strip()
    _ensure_analysis_uses_source_video(video_path, video_asset_id=video_asset_id)
    if str(session.get("status") or "").strip().lower() != "pending":
        raise ValueError(
            f"Video asset {video_asset_id} belongs to session {session_id} with status "
            f"{session.get('status')!r}, expected 'pending'."
        )
    return KioskAnalysisQueued(
        video_asset_id=video_asset_id,
        session_id=session_id,
        location_id=int(session["location_id"]),
        video_path=video_path,
        model_name=None,
    )


def start_kiosk_analysis_job(job: KioskAnalysisQueued) -> ScriptExecutionResult:
    db = TransactionalSessionLocal()
    try:
        result = run_kiosk_for_session(
            db,
            session_id=job.session_id,
            video_path=job.video_path,
            model_name=job.model_name,
        )
        if result.status == "failed":
            repositories.update_video_asset_status(db, job.video_asset_id, "issue")
        elif result.status == "pending":
            repositories.update_video_asset_status(db, job.video_asset_id, "ready")
        return result
    except Exception as exc:
        repositories.update_video_asset_status(db, job.video_asset_id, "issue")
        script_run_id = repositories.create_script_run_started(
            db,
            session_id=job.session_id,
            trigger_id=None,
            script_name="kiosk",
            model_name=job.model_name or "analysis_worker",
            command=SCRIPT_RUN_COMMAND_REDACTED,
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status="failed",
            stdout_log="",
            stderr_log=str(exc),
        )
        raise
    finally:
        db.close()


def check_video_ready_policy(created_time: datetime, retries_used: int) -> dict:
    retry_limit = 3
    wait_minutes = 5
    ready_after = created_time + timedelta(minutes=wait_minutes * (retries_used + 1))
    should_mark_issue = retries_used >= retry_limit
    return {
        "retries_used": retries_used,
        "retry_limit": retry_limit,
        "wait_minutes_between_retries": wait_minutes,
        "next_retry_after": ready_after.astimezone(UTC).isoformat(),
        "should_mark_issue": should_mark_issue,
        "recommended_action": "mark_trigger_issue" if should_mark_issue else "retry_when_ready",
        "explanation": (
            "Video is still within retry budget. Wait the suggested interval and check again."
            if not should_mark_issue
            else "Retry limit reached. Mark the trigger as issue and stop downstream automation."
        ),
    }
