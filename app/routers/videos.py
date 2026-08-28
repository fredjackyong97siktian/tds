import logging

from fastapi import APIRouter, Depends, HTTPException

from fastapi.responses import FileResponse, RedirectResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..spaces import (
    _public_base_url,
    generate_presigned_download_url,
    generate_public_object_url,
    is_spaces_public_read_enabled,
)
from ..schemas import VideoAssetCreate, VideoAssetListItem
from ..storage import (
    guess_media_type,
    infer_filename,
    resolve_private_path,
    session_video_path,
    trigger_video_path,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


def _known_video_base_urls() -> tuple[str, ...]:
    bases: list[str] = []
    try:
        bases.append(_public_base_url())
    except RuntimeError:
        pass
    return tuple(bases)


@router.get("/assets", response_model=list[VideoAssetListItem])
def list_video_assets(limit: int = 50, db: Session = Depends(get_transaction_db)) -> list[VideoAssetListItem]:
    rows = repositories.list_video_assets(db, limit=limit)
    known_bases = _known_video_base_urls()
    for row in rows:
        file_path = row.get("file_path")
        if (
            str(row.get("status") or "") == "processed"
            and not row.get("video_url")
            and isinstance(file_path, str)
            and file_path.startswith("spaces://")
        ):
            spaces_object_key = file_path.removeprefix("spaces://").lstrip("/")
            if spaces_object_key:
                try:
                    row["video_url"] = (
                        generate_public_object_url(spaces_object_key)
                        if is_spaces_public_read_enabled()
                        else generate_presigned_download_url(spaces_object_key)
                    )
                except Exception:
                    logger.exception("Could not build video URL for video_asset_id=%s", row.get("id"))
                    row["video_url"] = row.get("video_url") or ""
        video_url = row.get("video_url")
        # A relative path (our own /api/... fallback) is fine; anything else that
        # doesn't point at our known Spaces base is unrecognized/malformed - don't
        # surface it as a playable link, just treat the video as not-yet-available.
        if (
            known_bases
            and isinstance(video_url, str)
            and video_url
            and not video_url.startswith("/")
            and not video_url.startswith(known_bases)
        ):
            row["video_url"] = None

    items: list[VideoAssetListItem] = []
    for row in rows:
        try:
            items.append(VideoAssetListItem(**row))
        except ValidationError:
            logger.exception("Skipping malformed video_asset row id=%s", row.get("id"))
    return items


@router.post("/assets/{video_asset_id}/retry-issue")
def retry_video_asset_issue(video_asset_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return repositories.retry_video_asset_issue(db, video_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/assets/{video_asset_id}/restart-analysis")
def restart_video_asset_analysis(video_asset_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return repositories.restart_video_asset_analysis(db, video_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/assets/{video_asset_id}", status_code=204)
def delete_video_asset(video_asset_id: int, db: Session = Depends(get_transaction_db)) -> None:
    try:
        repositories.get_video_asset(db, video_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    repositories.delete_video_asset(db, video_asset_id)


@router.post("/triggers/{trigger_id}")
def create_trigger_video_asset(trigger_id: int, payload: VideoAssetCreate, db: Session = Depends(get_transaction_db)) -> dict:
    trigger = repositories.get_trigger(db, trigger_id)
    filename = infer_filename(payload.file_path or payload.video_url, f"trigger_{trigger_id}_{payload.section}", ".mp4")
    canonical_path = str(trigger_video_path(trigger["location_id"], trigger_id, payload.section, filename))
    video_asset_id = repositories.create_video_asset(
        db,
        {
            **payload.model_dump(),
            "trigger_id": trigger_id,
            "file_path": canonical_path,
            "metadata": None,
        },
    )
    access_url = f"/api/v1/videos/assets/{video_asset_id}/content"
    repositories.update_video_asset_url(db, video_asset_id, access_url)
    return {
        "ok": True,
        "trigger_id": trigger_id,
        "video_asset_id": video_asset_id,
        "section": payload.section,
        "video_url": access_url,
        "file_path": canonical_path,
    }


@router.post("/sessions/{session_id}")
def create_video_asset(session_id: int, payload: VideoAssetCreate, db: Session = Depends(get_transaction_db)) -> dict:
    session = repositories.get_session(db, session_id)
    filename = infer_filename(payload.file_path or payload.video_url, f"session_{session_id}_{payload.section}", ".mp4")
    canonical_path = str(session_video_path(session["location_id"], session_id, payload.section, filename))
    video_asset_id = repositories.create_video_asset(
        db,
        {
            **payload.model_dump(),
            "file_path": canonical_path,
            "metadata": None,
        },
    )
    repositories.create_session_video_asset_link(db, session_id, video_asset_id, payload.model_dump())
    access_url = f"/api/v1/videos/assets/{video_asset_id}/content"
    repositories.update_video_asset_url(db, video_asset_id, access_url)
    return {
        "ok": True,
        "session_id": session_id,
        "video_asset_id": video_asset_id,
        "section": payload.section,
        "video_url": access_url,
        "file_path": canonical_path,
    }


@router.get("/assets/{video_asset_id}/content")
def get_video_asset_content(video_asset_id: int, db: Session = Depends(get_transaction_db)) -> FileResponse:
    row = repositories.get_video_asset(db, video_asset_id)
    file_path = row.get("file_path")
    if isinstance(file_path, str) and file_path.startswith("spaces://"):
        spaces_object_key = file_path.removeprefix("spaces://").lstrip("/")
        if not spaces_object_key:
            raise HTTPException(status_code=404, detail="Spaces object key is missing for this video asset.")
        try:
            resolved_url = (
                generate_public_object_url(spaces_object_key)
                if is_spaces_public_read_enabled()
                else generate_presigned_download_url(spaces_object_key)
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return RedirectResponse(url=resolved_url, status_code=307)

    if file_path:
        try:
            resolved = resolve_private_path(file_path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if resolved.exists() and resolved.is_file():
            return FileResponse(path=resolved, media_type=guess_media_type(str(resolved)), filename=resolved.name)

    if not file_path:
        raise HTTPException(status_code=404, detail="Video asset does not have a private file path.")
    raise HTTPException(status_code=404, detail="Private video file not found.")
