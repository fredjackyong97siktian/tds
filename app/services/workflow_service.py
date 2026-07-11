from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from os.path import basename, splitext
from pathlib import Path

from sqlalchemy.orm import Session

from ..config import settings
from .. import repositories
from ..storage import gallery_state_path as build_private_gallery_state_path, session_logs_root, session_root


UTC = timezone.utc


@dataclass
class ScriptExecutionResult:
    script_name: str
    model_name: str | None
    status: str
    command: list[str]
    stdout: str
    stderr: str


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
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    status = "success" if completed.returncode == 0 else "failed"
    repositories.create_script_run(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        model_name=model_name,
        status=status,
        command=" ".join(command),
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


def retrieve_kiosk_video_window(*, start_time: datetime, end_time: datetime) -> dict:
    return {
        "requested_start_time": start_time.isoformat(),
        "requested_end_time": end_time.isoformat(),
        "recommended_padding_seconds": 10,
        "note": "Hook this into your Da Hua/NVR retrieval service.",
    }


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
    resolved_output_dir = Path(output_dir) if output_dir else default_video_output_dir(location_id, session_id, video_path)
    resolved_gallery_state = Path(gallery_state_path) if gallery_state_path else build_private_gallery_state_path(location_id, session_id)
    return run_script(
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
    return run_script(
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
