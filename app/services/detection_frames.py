"""Immutable copies of the exact JPEG bytes prepared for a detection request."""

import base64
import hashlib
import json
from pathlib import Path

from sqlalchemy import text

from .. import repositories
from ..config import settings

MAX_FRAME_BYTES = 8 * 1024 * 1024
FIELDS = {"role", "trigger_id", "video_asset_id", "sample_time", "playback_offset_seconds", "camera_clock"}


def _directory(location_id, session_id, run_id):
    root = Path(settings.video_storage_dir).resolve()
    directory = root / "locations" / f"location_{int(location_id)}" / f"session_{int(session_id)}" / "detection_analysis" / f"run_{int(run_id)}"
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Detection frame storage is outside the permitted root.")
    return directory


def preserve(frames, *, location_id, session_id, run_id):
    if not 1 <= len(frames) <= 40:
        raise ValueError("Detection frame snapshots require 1 to 40 images.")
    prepared = []
    for index, frame in enumerate(frames, start=1):
        value = str(frame.get("image_url") or "")
        prefix = "data:image/jpeg;base64,"
        if not value.startswith(prefix) or len(value) > MAX_FRAME_BYTES * 2:
            raise ValueError("Detection snapshots must be bounded JPEG data URLs.")
        raw = base64.b64decode(value[len(prefix):], validate=True)
        if len(raw) > MAX_FRAME_BYTES or not raw.startswith(b'\xff\xd8'):
            raise ValueError("Invalid or oversized detection JPEG.")
        if frame.get("role") not in {"entry", "kiosk", "exit"}:
            raise ValueError("Unknown detection frame phase.")
        item = {key: value for key, value in frame.items() if key in FIELDS}
        item.update(index=index, sha256=hashlib.sha256(raw).hexdigest(),
                    image_url=f"/api/v1/sessions/{session_id}/detection-analysis/frames/{run_id}/{index}")
        prepared.append((item, raw))
    directory = _directory(location_id, session_id, run_id)
    # A run ID never overwrites an earlier attempt's evidence.
    directory.mkdir(parents=True, exist_ok=False)
    for item, raw in prepared:
        (directory / f"frame_{item['index']:02d}.jpg").write_bytes(raw)
    manifest = [item for item, _ in prepared]
    (directory / "manifest.json").write_text(json.dumps(manifest, default=str))
    return manifest


def _run(db, session_id, run_id):
    session = repositories.get_session(db, session_id)
    summary = session.get("result_summary") or {}
    if isinstance(summary, str):
        summary = json.loads(summary)
    if run_id is None:
        run_id = (summary.get("detection_analysis") or {}).get("script_run_id")
    if not run_id:
        return session, None, {}
    row = db.execute(text(f"""select id, script_name, model_name, runner_payload
        from {repositories._table('script_run')} where id=:id and session_id=:session_id"""),
        {"id": int(run_id), "session_id": int(session_id)}).mappings().first()
    if not row or not (row["script_name"] == "detection_analysis" or
                       (row["script_name"] == "carry_confidence" and str(row.get("model_name") or "").endswith("_detection_analysis"))):
        raise ValueError("Detection Script Run was not found for this session.")
    payload = row.get("runner_payload") or {}
    return session, int(row["id"]), json.loads(payload) if isinstance(payload, str) else dict(payload)


def list_frames(db, session_id, run_id=None):
    session, run_id, payload = _run(db, session_id, run_id)
    result = {"session_id": session_id, "script_run_id": run_id, "frames": [], "request_started_at": payload.get("request_started_at")}
    if payload.get("frame_snapshot_version") != 1:
        result["message"] = "Exact request images were not retained for this attempt. Older frames will not be regenerated or guessed."
        return result
    directory = _directory(session["location_id"], session_id, run_id)
    for item in payload.get("frames", []):
        index = int(item["index"])
        if not 1 <= index <= 40:
            raise ValueError("Invalid saved frame index.")
        row = {key: value for key, value in item.items() if key in FIELDS | {"index", "sha256"}}
        row["image_url"] = f"/api/v1/sessions/{session_id}/detection-analysis/frames/{run_id}/{index}"
        row["available"] = (directory / f"frame_{index:02d}.jpg").is_file()
        result["frames"].append(row)
    if any(not frame["available"] for frame in result["frames"]):
        result["message"] = "Some saved images are unavailable or have expired under the media retention policy."
    return result


def image_path(db, session_id, run_id, index):
    session, matched_id, payload = _run(db, session_id, run_id)
    if payload.get("frame_snapshot_version") != 1:
        raise FileNotFoundError("Exact frames were not retained for this attempt.")
    item = next((row for row in payload.get("frames", []) if row.get("index") == index), None)
    if not item or not 1 <= index <= 40:
        raise FileNotFoundError("Saved detection frame was not found.")
    path = _directory(session["location_id"], session_id, matched_id) / f"frame_{index:02d}.jpg"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FRAME_BYTES:
        raise FileNotFoundError("Saved detection frame is unavailable or has expired.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
        raise ValueError("Saved detection frame no longer matches the original request image.")
    return path
