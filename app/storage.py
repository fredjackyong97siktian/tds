from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

from .config import settings


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "file"


def media_root() -> Path:
    root = settings.video_storage_dir / "locations"
    root.mkdir(parents=True, exist_ok=True)
    return root


def tmp_media_root() -> Path:
    root = settings.video_storage_dir / "tmp_video" / "locations"
    root.mkdir(parents=True, exist_ok=True)
    return root


def location_root(location_id: int) -> Path:
    root = media_root() / f"location_{location_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def tmp_location_root(location_id: int) -> Path:
    root = tmp_media_root() / f"location_{location_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def trigger_root(location_id: int, trigger_id: int) -> Path:
    root = location_root(location_id) / "triggers" / f"trigger_{trigger_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def tmp_trigger_root(location_id: int, trigger_id: int) -> Path:
    root = tmp_location_root(location_id) / "triggers" / f"trigger_{trigger_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def trigger_video_path(location_id: int, trigger_id: int, section: str, filename: str) -> Path:
    directory = trigger_root(location_id, trigger_id) / section / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_name(filename)


def trigger_tmp_video_path(location_id: int, trigger_id: int, section: str, filename: str) -> Path:
    directory = tmp_trigger_root(location_id, trigger_id) / section / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_name(filename)


def session_root(location_id: int, session_id: int) -> Path:
    root = location_root(location_id) / "sessions" / f"session_{session_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def tmp_session_root(location_id: int, session_id: int) -> Path:
    root = tmp_location_root(location_id) / "sessions" / f"session_{session_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_video_path(location_id: int, session_id: int, section: str, filename: str) -> Path:
    directory = session_root(location_id, session_id) / "videos" / section / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_name(filename)


def session_tmp_video_path(location_id: int, session_id: int, section: str, filename: str) -> Path:
    directory = tmp_session_root(location_id, session_id) / "videos" / section / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_name(filename)


def session_scripts_root(location_id: int, session_id: int) -> Path:
    root = session_root(location_id, session_id) / "scripts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_logs_root(location_id: int, session_id: int) -> Path:
    root = session_scripts_root(location_id, session_id) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def gallery_state_path(location_id: int, session_id: int) -> Path:
    root = session_scripts_root(location_id, session_id)
    return root / "active_gallery_state.pkl"


def session_customer_reid_path(
    location_id: int,
    session_id: int,
    session_customer_id: int,
    stage: str,
    filename: str,
) -> Path:
    directory = session_root(location_id, session_id) / "reid" / "session_customers" / f"sc_{session_customer_id}" / stage
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _safe_name(filename)


def infer_filename(source: str | None, fallback_stem: str, default_extension: str) -> str:
    if source:
        parsed = urlparse(source)
        candidate = Path(parsed.path).name
        if candidate:
            return _safe_name(candidate)
    return _safe_name(f"{fallback_stem}{default_extension}")


def resolve_private_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser().resolve()
    allowed_root = settings.video_storage_dir.resolve()
    if allowed_root not in candidate.parents and candidate != allowed_root:
        raise ValueError("Requested file is outside the private media root.")
    return candidate


def guess_media_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"
