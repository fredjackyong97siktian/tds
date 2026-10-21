"""Short identifier-camera evidence sequences for visits without a paid receipt."""

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

from .. import repositories
from ..storage import tmp_trigger_root

FRAME_COUNT = 12
FRAME_INTERVAL_SECONDS = 0.5
LAST_FRAME_BEFORE_EXIT_SECONDS = 3


def sample_times(exit_time: datetime) -> list[datetime]:
    first = exit_time - timedelta(seconds=LAST_FRAME_BEFORE_EXIT_SECONDS + (FRAME_COUNT - 1) * FRAME_INTERVAL_SECONDS)
    return [first + timedelta(seconds=index * FRAME_INTERVAL_SECONDS) for index in range(FRAME_COUNT)]


def capture_exit(db: Any, *, location_id: int, exit_trigger: dict[str, Any],
                 parent_script_run_id: int) -> dict[str, Any]:
    return capture_sequence(db, location_id=location_id, trigger=exit_trigger,
                            parent_script_run_id=parent_script_run_id, section="identifier",
                            times=sample_times(exit_trigger["trigger_time"] if isinstance(exit_trigger["trigger_time"], datetime)
                                               else datetime.fromisoformat(str(exit_trigger["trigger_time"]))))


def capture_sequence(db: Any, *, location_id: int, trigger: dict[str, Any],
                     parent_script_run_id: int, section: str, times: list[datetime]) -> dict[str, Any]:
    """Capture one evenly sampled NVR sequence, without the entrance crop."""
    from . import workflow_service as workflow

    trigger_id = int(trigger["trigger_id"])
    frame_count = len(times)
    if frame_count < 2:
        raise ValueError("At least two camera sample times are required.")
    interval = (times[1] - times[0]).total_seconds()
    if interval <= 0 or any(abs((right - left).total_seconds() - interval) > 0.0001 for left, right in zip(times, times[1:])):
        raise ValueError("Camera sample times must be increasing and evenly spaced.")
    run_id = repositories.create_script_run_started(
        db, session_id=None, trigger_id=trigger_id, script_name="retrieve_video",
        model_name=f"dahua_rtsp_playback:{'identifier_frames' if section == 'identifier' else 'kiosk_payment_frames'}", command=workflow.SCRIPT_RUN_COMMAND_REDACTED,
        runner_payload={"parent_script_run_id": parent_script_run_id, "location_id": location_id,
                        "section": section, "trigger_id": trigger_id, "frame_count": frame_count,
                        "interval_seconds": interval, "sample_times": [value.isoformat() for value in times]},
    )
    frames: list[dict[str, Any]] = []
    rtsp_url = ""
    try:
        exit_time = workflow._coerce_datetime_value(trigger.get("trigger_time"))
        if exit_time is None:
            raise ValueError(f"{section} capture requires a trigger/payment timestamp.")
        cctv = repositories.get_cctv_by_location_section(db, location_id=location_id, section=section)
        channel = str(cctv.get("recorder_channel") or "").strip()
        if not channel:
            raise ValueError(f"Configure the {section} CCTV recorder channel.")
        direction = str(cctv.get("exit_direction") or "").strip()
        if not direction and section == "identifier":
            raise ValueError("Select the identifier camera's Exit Direction in CCTV setup.")
        if not workflow.is_spaces_configured():
            raise ValueError(f"{section} frame retrieval requires DigitalOcean Spaces.")
        location = repositories.get_location_endpoint(db, location_id)
        delay = workflow._current_delayed_seconds(db, location_id=location_id, section=section, cctv=cctv)
        camera_times = [value - timedelta(seconds=delay) for value in times]
        # Dahua URL timestamps have whole-second precision. Seek the fractional
        # remainder after opening playback to preserve half-second sample times.
        playback_start = camera_times[0].replace(microsecond=0)
        offset = (camera_times[0] - playback_start).total_seconds()
        rtsp_url = workflow._build_dahua_rtsp_playback_url(
            host=str(location.get("dahua_host") or "").strip(),
            username=str(location.get("dahua_username") or "").strip(),
            password=workflow.decrypt_secret(str(location.get("dahua_password_encrypted") or "")),
            rtsp_port=int(location.get("rtsp_port") or workflow.settings.dahua_rtsp_port), channel=channel,
            start_time=playback_start, end_time=camera_times[-1] + timedelta(seconds=1),
        )
        root = tmp_trigger_root(location_id, trigger_id) / section / "frames"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"run_{run_id}_", dir=root) as directory:
            pattern = Path(directory) / "frame_%02d.jpg"
            command = workflow._build_frame_batch_capture_command(
                rtsp_url, start_offset_seconds=offset, gap_seconds=interval,
                frame_count=frame_count, output_pattern=pattern, apply_entrance_crop=False,
            )
            lock = workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=workflow.settings.retrieval_ffmpeg_lock_wait_seconds)
            if lock is None:
                raise RuntimeError(f"{section} retrieval could not acquire the NVR/FFmpeg lock.")
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=max(1, int(workflow.settings.retrieval_ffmpeg_timeout_seconds)))
            finally:
                workflow._release_retrieval_ffmpeg_lock(lock)
            paths = [Path(directory) / f"frame_{index:02d}.jpg" for index in range(1, frame_count + 1)]
            available = sum(path.is_file() and path.stat().st_size > 0 for path in paths)
            if completed.returncode != 0 or available != frame_count:
                raise RuntimeError(f"{section} capture returned {available}/{frame_count} frames. "
                                   + str(completed.stderr or "")[-1500:].replace(rtsp_url, "[REDACTED RTSP]"))
            for index, (path, time, camera_time) in enumerate(zip(paths, times, camera_times), start=1):
                key = workflow._trigger_frame_spaces_key(
                    location_id=location_id, trigger_id=trigger_id, section=section,
                    filename=f"{section}_run_{run_id}_frame_{index:02d}_{time.strftime('%Y%m%d_%H%M%S_%f')}.jpg",
                )
                uploaded = workflow.upload_private_file(path, key, content_type="image/jpeg")
                frames.append({"frame_index": index, "sample_time": time.isoformat(),
                               "camera_sample_time": camera_time.isoformat(), "status": "ok",
                               "image_object_key": key,
                               "image_url": str(uploaded.get("public_url") or workflow._spaces_download_url_for_object_key(key))})
        result = {"trigger_id": trigger_id, "trigger_time": exit_time.isoformat(),
                  "section": section, "recorder_channel": channel, "exit_direction": direction,
                  "delayed_seconds": delay, "frames": frames, "script_run_id": run_id}
        repositories.finish_script_run(db, run_id, status="success", stderr_log="",
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, result))
        return result
    except Exception as exc:
        error = str(exc).replace(rtsp_url, "[REDACTED RTSP]") if rtsp_url else str(exc)
        repositories.finish_script_run(db, run_id, status="failed", stderr_log=error,
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, {"error": error, "frames": frames}))
        raise RuntimeError(f"{section.title()} retrieval failed for trigger {trigger_id}: {error}") from None
