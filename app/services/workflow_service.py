from __future__ import annotations
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from os.path import basename, splitext
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from sqlalchemy.orm import Session

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


@dataclass
class EntranceAnalysisQueued:
    video_asset_id: int
    trigger_id: int
    session_id: int
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
    session_id = int(script_run["session_id"])
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
    session = repositories.get_session(db, session_id)
    trigger = repositories.get_trigger(db, int(trigger_id)) if trigger_id is not None else None

    remote_status = "success" if remote_result.status == "success" else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=remote_status,
        stdout_log=remote_result.stdout,
        stderr_log=remote_result.stderr,
    )
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
            enter_time=session.get("start_time"),
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
    if session_id is not None and remote_result.kiosk_summary:
        session_row = repositories.get_session(db, int(session_id))
        existing_summary = dict(session_row.get("result_summary") or {})
        kiosk_runs = dict(existing_summary.get("kiosk_runs") or {})
        kiosk_runs[str(video_asset_id)] = remote_result.kiosk_summary
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
    if session_id is not None and remote_result.kiosk_summary:
        detected_total_items = int(remote_result.kiosk_summary.get("detected_total_items") or 0)
        repositories.finalize_session_result(
            db,
            session_id=int(session_id),
            kiosk_total_items=detected_total_items,
            tolerance=1,
            extra_result_summary={
                "kiosk_summary": remote_result.kiosk_summary,
            },
        )
    _apply_processed_video_upload_result(
        db,
        video_asset_row=video_asset_row,
        object_key=remote_result.processed_video_object_key,
        video_url=processed_video_url,
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
        if script_name not in {"entry", "kiosk"}:
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
    return normalized in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "ABORTED"}


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
        if not job_id or script_name not in {"entry", "kiosk"}:
            continue
        try:
            body = _runpod_request(
                method="GET",
                path=f"/status/{quote(job_id)}",
                kind=script_name,
            )
        except Exception:
            continue
        runpod_status = str(body.get("status") or "").strip().upper()
        if not _is_runpod_terminal_status(runpod_status):
            continue
        remote_status, remote_result = _remote_runner_result_from_runpod_body(body)
        if script_name == "entry":
            result = _finalize_remote_entry_script_run(db, script_run=script_run, remote_result=remote_result)
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
    )


def _runpod_endpoint_id(kind: str | None = None) -> str:
    normalized_kind = str(kind or "").strip().lower()
    endpoint_id = ""
    if normalized_kind == "entry":
        endpoint_id = str(settings.runpod_entry_endpoint_id or "").strip()
    elif normalized_kind == "kiosk":
        endpoint_id = str(settings.runpod_kiosk_endpoint_id or "").strip()
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
    try:
        with urlopen(request, timeout=settings.runner_timeout_seconds) as response:
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


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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

    # Backward-compatible fallback for older cross-state layouts that may have
    # stored the same runtime person under a nested person_id field.
    for entry in persistent_gallery.values():
        if _coerce_int(entry.get("person_id")) == runtime_person_id:
            return entry
    return {}


TERMINAL_SESSION_STATUSES = {"detected", "not_detected", "closed", "issue", "whitelisted"}


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


def _build_cross_state_from_active_gallery(location_id: int) -> dict[str, Any]:
    vector_db = VectorSessionLocal()
    transactional_db = TransactionalSessionLocal()
    try:
        active_rows = vector_repositories.list_active_gallery_records(
            vector_db,
            location_id=location_id,
            limit=5000,
        )
        persistent_gallery: dict[int, dict[str, Any]] = {}
        persistent_gallery_view_paths: dict[int, list[str]] = {}
        next_gid = 1

        for row in active_rows:
            session_customer_id = _coerce_int(row.get("session_customer_id"))
            if session_customer_id is None:
                continue
            active_owner = _resolve_open_session_customer_for_gallery(
                transactional_db,
                location_id=location_id,
                session_customer_id=session_customer_id,
            )
            if active_owner is None:
                continue
            active_session, active_session_customer = active_owner
            gallery_id = _coerce_int(row.get("person_id"))
            if gallery_id is None:
                gallery_id = _coerce_int(active_session_customer.get("person_id"))
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
                    "session_id": int(active_session["id"]),
                    "session_customer_id": session_customer_id,
                    "person_id": gallery_id,
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

        return {
            "next_gid": next_gid,
            "persistent_gallery": persistent_gallery,
            "persistent_gallery_view_paths": persistent_gallery_view_paths,
        }
    finally:
        vector_db.close()
        transactional_db.close()


def _hydrate_gallery_state_from_active_gallery(location_id: int, gallery_state_path: Path) -> None:
    cross_state = _build_cross_state_from_active_gallery(location_id)
    _write_cross_state_pickle(gallery_state_path, cross_state)


def _find_open_active_session_for_location(db: Session, location_id: int) -> dict[str, Any] | None:
    vector_db = VectorSessionLocal()
    try:
        active_rows = vector_repositories.list_active_gallery_records(
            vector_db,
            location_id=location_id,
            limit=5000,
        )
    finally:
        vector_db.close()

    seen_session_ids: set[int] = set()
    for row in active_rows:
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
    padding_seconds = min(15 + extra_seconds, 40)
    return (
        transaction_time - timedelta(seconds=padding_seconds),
        transaction_time + timedelta(seconds=padding_seconds),
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

    paid_transactions = repositories.list_paid_transactions_for_session_window(
        db,
        location_id=int(session["location_id"]),
        start_time=session_start_time,
        end_time=session_end_time,
    )

    repositories.delete_session_transactions(db, session_id)
    total_transaction_items = 0
    raw_windows: list[tuple[datetime, datetime]] = []
    transaction_summaries: list[dict[str, Any]] = []
    for transaction in paid_transactions:
        transaction_time = _coerce_datetime_value(transaction.get("transaction_time"))
        if transaction_time is None:
            continue
        total_items = int(transaction.get("total_items") or 0)
        total_transaction_items += total_items
        window_start, window_end = _build_transaction_window_bounds(transaction_time, total_items)
        raw_windows.append((window_start, window_end))
        repositories.create_transaction(
            db,
            session_id,
            {
                "receipt_number": str(transaction.get("receipt_number") or transaction.get("transaction_id") or ""),
                "transaction_time": transaction_time,
                "total_items": total_items,
                "total_amount": transaction.get("total_amount"),
                "raw_payload": transaction,
            },
        )
        transaction_summaries.append(
            {
                "transaction_id": transaction.get("transaction_id"),
                "receipt_number": transaction.get("receipt_number"),
                "transaction_time": transaction_time.isoformat() if hasattr(transaction_time, "isoformat") else transaction_time,
                "total_items": total_items,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
            }
        )

    merged_windows = _merge_time_windows(raw_windows)
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
            "queued_kiosk_video_asset_ids": [],
        }
    }
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

    if not merged_windows:
        repositories.finalize_session_result(
            db,
            session_id=session_id,
            kiosk_total_items=0,
            actual_items_brought=0,
            tolerance=1,
            extra_result_summary=session_close_summary,
        )
        return

    queued_video_asset_ids: list[int] = []
    try:
        for window_start, window_end in merged_windows:
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
    session_id: int,
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
        sessions_to_close: set[int] = set()
        linked_session_video_keys: set[tuple[int, str]] = set()

        def link_session_video_asset(event_session_id: int | None, section: str, metadata: dict[str, Any]) -> None:
            if event_session_id is None:
                return
            normalized_session_id = int(event_session_id)
            normalized_section = section.strip().lower()
            if normalized_section not in {"entry", "exit"}:
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
        if isinstance(raw_exit_customer_ids, list):
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

        for customer in summary_customers:
            person_id = int(customer["person_id"])
            entered = bool(customer.get("entered"))
            exited = bool(customer.get("exited"))
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
                active_owner = _resolve_open_session_customer_for_gallery(
                    transactional_db,
                    location_id=location_id,
                    session_customer_id=source_session_customer_id,
                )
                if active_owner is None:
                    logger.info(
                        "Ignoring stale active-gallery match for closed/left session video=%s location_id=%s runtime_person_id=%s session_customer_id=%s",
                        Path(video_path).name,
                        location_id,
                        person_id,
                        source_session_customer_id,
                    )
                    source_session_customer_id = None
                    source_session_id = None
                    source_person_id = person_id
                else:
                    source_session, source_session_customer = active_owner
                    source_session_id = int(source_session["id"])
                    source_person_id = _coerce_int(source_session_customer.get("person_id")) or source_person_id
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

            if exited and source_session_customer_id is not None:
                repositories.update_session_customer_leave_time(
                    transactional_db,
                    session_customer_id=source_session_customer_id,
                    leave_time=customer_leave_time,
                    match_status="resolved",
                )
                session_customer_id = source_session_customer_id
                customer_session_id = source_session_id or session_id
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
                session_customer_id = source_session_customer_id
                customer_session_id = source_session_id or session_id
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
                repositories.create_session_customer(
                    transactional_db,
                    session_id,
                    {
                        "person_id": person_id,
                        "enter_time": customer_enter_time,
                        "kiosk_start_time": None,
                        "leave_time": customer_leave_time,
                        "match_status": "resolved" if exited else "tracked",
                    },
                )
                session_customer = repositories.get_session_customer_by_session_person(
                    transactional_db,
                    session_id,
                    person_id,
                )
                session_customer_id = int(session_customer["id"])
                customer_session_id = session_id
                if exited:
                    sessions_to_close.add(session_id)
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
            delete_person_ids = [person_id, active_person_id, source_person_id]

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
                        "exited": bool(customer.get("exited")),
                        "group_id": customer.get("group_id"),
                        "active_view_count": len(view_rows) if view_rows else len(osnet_views),
                        "active_image_count": len(view_rows) if view_rows else len(image_paths),
                    },
                )
            if exited:
                # Temporarily keep active gallery rows during session-close testing.
                # TODO: Re-enable once entrance exit/session matching is verified.
                # vector_repositories.delete_active_gallery_by_aliases(
                #     vector_db,
                #     location_id=location_id,
                #     session_customer_ids=delete_session_customer_ids,
                #     person_ids=delete_person_ids,
                # )
                pass

            if view_rows or osnet_views or fashion_embedding is not None or image_paths:
                # Temporarily keep active gallery rows during session-close testing.
                # TODO: Re-enable once entrance exit/session matching is verified.
                # vector_repositories.delete_active_gallery_by_aliases(
                #     vector_db,
                #     location_id=location_id,
                #     session_customer_ids=delete_session_customer_ids,
                #     person_ids=delete_person_ids,
                # )
                active_metadata = {
                    "source": "entry_analysis",
                    "group_id": customer.get("group_id"),
                    "entered": bool(customer.get("entered")),
                    "exited": False,
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
                # Temporarily keep active gallery rows during session-close testing.
                # TODO: Re-enable once entrance exit/session matching is verified.
                # vector_repositories.delete_active_gallery_by_aliases(
                #     vector_db,
                #     location_id=location_id,
                #     session_customer_ids=delete_session_customer_ids,
                #     person_ids=delete_person_ids,
                # )
                pass

        if not sessions_to_close:
            sessions_to_close.add(session_id)
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
    session_id: int,
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

    if not create_session:
        repositories.update_trigger_status(db, trigger["id"], "pending")
        trigger = repositories.get_trigger(db, trigger["id"])
        return {"trigger": trigger, "session": None, "message": "Trigger created. Session creation deferred."}

    session = repositories.create_session(
        db,
        {
            "entry_trigger_id": trigger["id"],
            "exit_trigger_id": None,
            "location_id": location_id,
            "start_time": trigger_time,
        },
    )
    repositories.update_trigger_status(db, trigger["id"], "video_pending")
    trigger = repositories.get_trigger(db, trigger["id"])
    return {"trigger": trigger, "session": session, "message": "Trigger and session created."}


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
    start_time: datetime,
    end_time: datetime,
) -> VideoRetrievalQueued:
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
    else:
        raise ValueError("Either session_id or trigger_id is required for video retrieval.")
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
            "metadata": None,
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
        output_path=str(video_asset.get("file_path") or ""),
        rtsp_url=rtsp_url,
        dahua_host=dahua_host,
        dahua_username=dahua_username,
        rtsp_port=rtsp_port,
        status=str(video_asset.get("status") or "retrieving"),
        video_url=str(video_asset.get("video_url") or f"/api/v1/videos/assets/{video_asset_id}/content"),
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
    video_asset = _repair_video_asset_source_file_path_for_analysis(
        db,
        video_asset_row=video_asset,
        location_id=int(trigger["location_id"]),
        session_id=None,
        trigger_id=int(trigger_id),
    )
    video_path = str(video_asset.get("file_path") or "").strip()
    _ensure_analysis_uses_source_video(video_path, video_asset_id=video_asset_id)
    try:
        session = repositories.get_session_by_entry_trigger_id(db, int(trigger_id))
    except ValueError:
        session = repositories.create_session(
            db,
            {
                "entry_trigger_id": int(trigger_id),
                "exit_trigger_id": None,
                "location_id": int(trigger["location_id"]),
                "start_time": trigger.get("trigger_time"),
            },
        )
        repositories.update_trigger_status(db, int(trigger_id), "video_pending")
    return EntranceAnalysisQueued(
        video_asset_id=video_asset_id,
        trigger_id=int(trigger_id),
        session_id=int(session["id"]),
        location_id=int(trigger["location_id"]),
        video_path=video_path,
        model_name=None,
    )


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


def start_video_retrieval_job(job: VideoRetrievalQueued) -> None:
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
    merged_windows = pipeline.get("merged_kiosk_windows") or []
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

        paid_transactions = repositories.list_paid_transactions_for_session_window(
            db,
            location_id=int(session["location_id"]),
            start_time=session_start_time,
            end_time=session_end_time,
        )
        repositories.delete_session_transactions(db, session_id)
        total_transaction_items = 0
        raw_windows: list[tuple[datetime, datetime]] = []
        transaction_summaries: list[dict[str, Any]] = []
        for transaction in paid_transactions:
            transaction_time = _coerce_datetime_value(transaction.get("transaction_time"))
            if transaction_time is None:
                continue
            total_items = int(transaction.get("total_items") or 0)
            total_transaction_items += total_items
            window_start, window_end = _build_transaction_window_bounds(transaction_time, total_items)
            raw_windows.append((window_start, window_end))
            repositories.create_transaction(
                db,
                session_id,
                {
                    "receipt_number": str(transaction.get("receipt_number") or transaction.get("transaction_id") or ""),
                    "transaction_time": transaction_time,
                    "total_items": total_items,
                    "total_amount": transaction.get("total_amount"),
                    "raw_payload": transaction,
                },
            )
            transaction_summaries.append(
                {
                    "transaction_id": transaction.get("transaction_id"),
                    "receipt_number": transaction.get("receipt_number"),
                    "transaction_time": transaction_time.isoformat() if hasattr(transaction_time, "isoformat") else transaction_time,
                    "total_items": total_items,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                }
            )

        recomputed_windows = _merge_time_windows(raw_windows)
        if not recomputed_windows:
            repositories.update_session_fields(
                db,
                session_id=session_id,
                status="issue",
                transaction_total_items=total_transaction_items,
                result_summary={
                    **summary,
                    "session_close_pipeline": {
                        "session_start_time": session_start_time.isoformat()
                        if hasattr(session_start_time, "isoformat")
                        else session_start_time,
                        "session_end_time": session_end_time.isoformat()
                        if hasattr(session_end_time, "isoformat")
                        else session_end_time,
                        "paid_transaction_count": len(transaction_summaries),
                        "paid_transactions": transaction_summaries,
                        "merged_kiosk_windows": [],
                        "queued_kiosk_video_asset_ids": [],
                    },
                },
                issue_reason="Kiosk retry failed because no paid transactions were found inside the session window.",
            )
            return []

        pipeline = {
            "session_start_time": session_start_time.isoformat()
            if hasattr(session_start_time, "isoformat")
            else session_start_time,
            "session_end_time": session_end_time.isoformat()
            if hasattr(session_end_time, "isoformat")
            else session_end_time,
            "paid_transaction_count": len(transaction_summaries),
            "paid_transactions": transaction_summaries,
            "merged_kiosk_windows": [
                {
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                }
                for start, end in recomputed_windows
            ],
            "queued_kiosk_video_asset_ids": [],
        }
        summary["session_close_pipeline"] = pipeline
        repositories.update_session_fields(
            db,
            session_id=session_id,
            status="pending",
            transaction_total_items=total_transaction_items,
            result_summary=summary,
            issue_reason=None,
        )
        merged_windows = pipeline["merged_kiosk_windows"]

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
    session_id: int,
    video_path: str,
    model_name: str | None = None,
    output_dir: str | None = None,
    gallery_state_path: str | None = None,
) -> ScriptExecutionResult:
    session = repositories.get_session(db, session_id)
    location_id = int(session["location_id"])
    workdir = build_session_workdir(location_id, session_id)
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
    _hydrate_gallery_state_from_active_gallery(location_id, resolved_gallery_state)
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
    session_id = repositories.get_primary_session_id_for_video_asset(db, video_asset_id)
    if session_id is None:
        raise ValueError(f"Video asset {video_asset_id} does not have a related session.")
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
