from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from os.path import basename, splitext
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sqlalchemy.orm import Session

from ..config import settings
from ..crypto import decrypt_secret
from .. import repositories
from ..db import TransactionalSessionLocal, VectorSessionLocal
from ..spaces import upload_private_file
from .. import vector_repositories
from ..storage import (
    guess_media_type,
    gallery_state_path as build_private_gallery_state_path,
    processed_video_spaces_key,
    session_logs_root,
    session_root,
    tmp_media_root,
    trigger_processed_root,
    trigger_tmp_video_path,
    session_tmp_video_path,
)


UTC = timezone.utc
SCRIPT_RUN_COMMAND_REDACTED = "[redacted]"


@dataclass
class ScriptExecutionResult:
    script_name: str
    model_name: str | None
    status: str
    command: list[str]
    stdout: str
    stderr: str


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
    return output_dir / f"{Path(video_path).stem}_output.mp4"


def _tracking_summary_path(video_path: str, output_dir: Path) -> Path:
    return output_dir / f"{Path(video_path).stem}_tracking_summary.json"


def _load_tracking_summary(video_path: str, output_dir: Path) -> dict[str, Any]:
    summary_path = _tracking_summary_path(video_path, output_dir)
    if not summary_path.exists():
        raise FileNotFoundError(f"Tracking summary not found at {summary_path}")
    return json.loads(summary_path.read_text())


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


def _combine_fashion_embedding(upper: Any, lower: Any) -> list[float] | None:
    upper_list = _tensor_like_to_float_list(upper)
    lower_list = _tensor_like_to_float_list(lower)
    if upper_list and lower_list:
        return upper_list + lower_list
    return upper_list or lower_list


def _candidate_gallery_image_paths(
    *,
    cross_state: dict[str, Any],
    output_dir: Path,
    person_id: int,
) -> list[str]:
    image_paths = cross_state.get("persistent_gallery_view_paths", {}).get(person_id) or []
    if image_paths:
        return sorted(str(path) for path in image_paths)
    reid_dir = output_dir.parent / "reid_views" / f"ID{person_id}"
    if not reid_dir.exists():
        return []
    return sorted(str(path) for path in reid_dir.glob("*.jpg"))


def _sync_gallery_state_after_entry(
    *,
    location_id: int,
    session_id: int,
    video_path: str,
    output_dir: Path,
    gallery_state_path: Path,
    enter_time: datetime | None,
    leave_time: datetime | None,
) -> None:
    tracking_summary = _load_tracking_summary(video_path, output_dir)
    cross_state = _load_cross_state_pickle(gallery_state_path)
    persistent_gallery = cross_state.get("persistent_gallery", {})

    transactional_db = TransactionalSessionLocal()
    vector_db = VectorSessionLocal()
    try:
        for customer in tracking_summary.get("customers", []):
            person_id = int(customer["person_id"])
            repositories.create_session_customer(
                transactional_db,
                session_id,
                {
                    "person_id": person_id,
                    "enter_time": enter_time,
                    "kiosk_start_time": None,
                    "leave_time": leave_time if bool(customer.get("exited")) else None,
                    "match_status": "resolved" if bool(customer.get("exited")) else "tracked",
                },
            )
            session_customer = repositories.get_session_customer_by_session_person(
                transactional_db,
                session_id,
                person_id,
            )

            vector_repositories.delete_customer_gallery_records_for_session_customer(
                vector_db,
                session_customer_id=int(session_customer["id"]),
            )

            gallery_entry = persistent_gallery.get(person_id) or {}
            osnet_views = gallery_entry.get("views") or []
            fashion_embedding = _combine_fashion_embedding(
                gallery_entry.get("fashion_upper_init"),
                gallery_entry.get("fashion_lower_init"),
            )
            image_paths = _candidate_gallery_image_paths(
                cross_state=cross_state,
                output_dir=output_dir,
                person_id=person_id,
            )

            created_gallery_ids: list[int] = []
            for index, osnet_view in enumerate(osnet_views):
                image_url = image_paths[index] if index < len(image_paths) else (image_paths[0] if image_paths else None)
                row = vector_repositories.create_customer_gallery_record(
                    vector_db,
                    location_id=location_id,
                    session_id=session_id,
                    session_customer_id=int(session_customer["id"]),
                    person_id=person_id,
                    image_url=image_url,
                    image_kind="reid_view",
                    embedding_osnet=_tensor_like_to_float_list(osnet_view),
                    embedding_fashion=fashion_embedding,
                    metadata={
                        "source": "entry_analysis",
                        "exited": bool(customer.get("exited")),
                        "group_id": customer.get("group_id"),
                        "view_index": index,
                    },
                )
                created_gallery_ids.append(int(row["id"]))

            if not created_gallery_ids and fashion_embedding is not None:
                row = vector_repositories.create_customer_gallery_record(
                    vector_db,
                    location_id=location_id,
                    session_id=session_id,
                    session_customer_id=int(session_customer["id"]),
                    person_id=person_id,
                    image_url=image_paths[0] if image_paths else None,
                    image_kind="fashion_view",
                    embedding_osnet=None,
                    embedding_fashion=fashion_embedding,
                    metadata={
                        "source": "entry_analysis",
                        "exited": bool(customer.get("exited")),
                        "group_id": customer.get("group_id"),
                    },
                )
                created_gallery_ids.append(int(row["id"]))

            if bool(customer.get("exited")):
                vector_repositories.delete_active_gallery(
                    vector_db,
                    location_id=location_id,
                    session_customer_id=int(session_customer["id"]),
                )
                continue

            if created_gallery_ids:
                vector_repositories.upsert_active_gallery(
                    vector_db,
                    location_id=location_id,
                    session_id=session_id,
                    session_customer_id=int(session_customer["id"]),
                    person_id=person_id,
                    state_kind="active_gallery",
                    state_payload={
                        "customer_gallery_ids": created_gallery_ids,
                        "primary_gallery_entry_id": created_gallery_ids[0],
                        "is_active": True,
                    },
                    metadata={
                        "source": "entry_analysis",
                        "group_id": customer.get("group_id"),
                        "entered": bool(customer.get("entered")),
                        "exited": False,
                    },
                )
            else:
                vector_repositories.delete_active_gallery(
                    vector_db,
                    location_id=location_id,
                    session_customer_id=int(session_customer["id"]),
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
    object_key = processed_video_spaces_key(
        location_id=location_id,
        section=str(video_asset_row.get("section") or script_name),
        filename=processed_video_path.name,
        session_id=session_id,
        trigger_id=trigger_id,
    )
    upload_result = upload_private_file(
        processed_video_path,
        object_key,
        content_type=guess_media_type(str(processed_video_path)),
    )

    raw_input_path = Path(source_video_path)
    raw_removed = False
    if _is_under_tmp_media_root(raw_input_path):
        _safe_unlink(raw_input_path)
        raw_removed = not raw_input_path.exists()

    processed_local_path = processed_video_path
    _safe_unlink(processed_local_path)
    processed_removed = not processed_local_path.exists()

    repositories.update_video_asset(
        db,
        int(video_asset_row["id"]),
        {
            "video_url": f"/api/v1/videos/assets/{int(video_asset_row['id'])}/content",
            "file_path": f"spaces://{upload_result['object_key']}",
            "captured_start_time": video_asset_row.get("captured_start_time"),
            "captured_end_time": video_asset_row.get("captured_end_time"),
            "retrieved_at": video_asset_row.get("retrieved_at"),
            "analyzed_at": datetime.now(UTC),
            "retention_until": video_asset_row.get("retention_until"),
            "status": "processed",
            "metadata": None,
        },
    )


def _record_followup_failure(
    db: Session,
    *,
    session_id: int,
    trigger_id: int | None,
    script_name: str,
    model_name: str | None,
    stdout: str,
    stderr: str,
) -> None:
    repositories.create_script_run(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        model_name=model_name,
        status="failed",
        command=SCRIPT_RUN_COMMAND_REDACTED,
        stdout_log=stdout,
        stderr_log=stderr,
    )


def run_script(
    db: Session,
    *,
    script_name: str,
    model_name: str | None,
    script_path: Path,
    args: list[str],
    session_id: int | None = None,
    trigger_id: int | None = None,
    cwd: Path | None = None,
) -> ScriptExecutionResult:
    command = [settings.python_bin, str(script_path), *args]
    script_run_id = repositories.create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        model_name=model_name,
        status="running",
        command=SCRIPT_RUN_COMMAND_REDACTED,
    )
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    status = "success" if completed.returncode == 0 else "failed"
    repositories.finish_script_run(
        db,
        script_run_id,
        status=status,
        stdout_log=completed.stdout,
        stderr_log=completed.stderr,
    )
    return ScriptExecutionResult(
        script_name=script_name,
        model_name=model_name,
        status=status,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
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
        repositories.update_video_asset(
            db,
            video_asset_id,
            {
                "video_url": f"/api/v1/videos/assets/{video_asset_id}/content",
                "file_path": output_path,
                "captured_start_time": start_time,
                "captured_end_time": end_time,
                "retrieved_at": datetime.now(UTC) if status == "success" else None,
                "analyzed_at": None,
                "retention_until": end_time + timedelta(days=3),
                "status": "ready" if status == "success" else "issue",
                "metadata": None,
            },
        )
        repositories.finish_script_run(
            db,
            script_run_id,
            status=status,
            stdout_log=completed.stdout,
            stderr_log=completed.stderr,
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
        if result.status != "success":
            repositories.update_video_asset_status(db, job.video_asset_id, "issue")
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
    resolved_gallery_state = Path(gallery_state_path) if gallery_state_path else build_private_gallery_state_path(location_id, session_id)
    video_asset_row = _lookup_video_asset_by_file_path(db, video_path)
    if video_asset_row is not None:
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "processing")

    result = run_script(
        db,
        script_name="entry",
        model_name=model_name,
        script_path=settings.entry_script_path,
        args=[
            "--video",
            str(video_path),
            "--output-dir",
            str(resolved_output_dir),
            "--gallery-state",
            str(resolved_gallery_state),
        ],
        session_id=session_id,
        trigger_id=trigger_id,
        cwd=workdir,
    )
    if video_asset_row is None:
        return result
    if result.status != "success":
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        return result

    processed_video_path = _expected_processed_video_path(video_path, resolved_output_dir)
    if not processed_video_path.exists():
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        stderr = f"{result.stderr}\nProcessed video not found at {processed_video_path}".strip()
        _record_followup_failure(
            db,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=model_name or "postprocess_processed_video_missing",
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )

    try:
        _sync_gallery_state_after_entry(
            location_id=location_id,
            session_id=session_id,
            video_path=video_path,
            output_dir=resolved_output_dir,
            gallery_state_path=resolved_gallery_state,
            enter_time=session.get("start_time"),
            leave_time=video_asset_row.get("captured_end_time"),
        )
    except Exception as exc:
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        stderr = f"{result.stderr}\nGallery persistence failed: {exc}".strip()
        _record_followup_failure(
            db,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=model_name or "postprocess_gallery_persistence",
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )

    try:
        _upload_processed_video_for_asset(
            db,
            video_asset_row=video_asset_row,
            location_id=location_id,
            session_id=session_id,
            trigger_id=trigger_id,
            processed_video_path=processed_video_path,
            source_video_path=video_path,
            output_dir=resolved_output_dir,
            script_name="entry",
            model_name=model_name,
        )
    except Exception as exc:
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        stderr = f"{result.stderr}\nDigitalOcean Spaces upload failed: {exc}".strip()
        _record_followup_failure(
            db,
            session_id=session_id,
            trigger_id=trigger_id,
            script_name="entry",
            model_name=model_name or "postprocess_spaces_upload",
            stdout=result.stdout,
            stderr=stderr,
        )
        return ScriptExecutionResult(
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=stderr,
        )
    return result


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
    resolved_gallery_state = Path(gallery_state_path) if gallery_state_path else build_private_gallery_state_path(location_id, session_id)
    video_asset_row = _lookup_video_asset_by_file_path(db, video_path)
    if video_asset_row is not None:
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "processing")

    result = run_script(
        db,
        script_name="kiosk",
        model_name=model_name,
        script_path=settings.kiosk_script_path,
        args=[
            "--video",
            str(video_path),
            "--output-dir",
            str(resolved_output_dir),
            "--gallery-state",
            str(resolved_gallery_state),
        ],
        session_id=session_id,
        trigger_id=None,
        cwd=workdir,
    )
    if video_asset_row is None:
        return result
    if result.status != "success":
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        return result

    processed_video_path = _expected_processed_video_path(video_path, resolved_output_dir)
    if not processed_video_path.exists():
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        return ScriptExecutionResult(
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=f"{result.stderr}\nProcessed video not found at {processed_video_path}".strip(),
        )

    try:
        _upload_processed_video_for_asset(
            db,
            video_asset_row=video_asset_row,
            location_id=location_id,
            session_id=session_id,
            trigger_id=None,
            processed_video_path=processed_video_path,
            source_video_path=video_path,
            output_dir=resolved_output_dir,
            script_name="kiosk",
            model_name=model_name,
        )
    except Exception as exc:
        repositories.update_video_asset_status(db, int(video_asset_row["id"]), "issue")
        return ScriptExecutionResult(
            script_name=result.script_name,
            model_name=result.model_name,
            status="failed",
            command=result.command,
            stdout=result.stdout,
            stderr=f"{result.stderr}\nDigitalOcean Spaces upload failed: {exc}".strip(),
        )
    return result


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
