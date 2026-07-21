from fastapi import APIRouter, Depends, HTTPException

from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..spaces import generate_presigned_download_url
from ..schemas import VideoAssetCreate, VideoAssetListItem
from ..storage import (
    guess_media_type,
    infer_filename,
    resolve_private_path,
    session_video_path,
    trigger_video_path,
)


router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


@router.get("/assets", response_model=list[VideoAssetListItem])
def list_video_assets(limit: int = 50, db: Session = Depends(get_transaction_db)) -> list[VideoAssetListItem]:
    rows = repositories.list_video_assets(db, limit=limit)
    for row in rows:
        file_path = row.get("file_path")
        if str(row.get("status") or "") == "processed" and isinstance(file_path, str) and file_path.startswith("spaces://"):
            spaces_object_key = file_path.removeprefix("spaces://").lstrip("/")
            if spaces_object_key:
                try:
                    row["video_url"] = generate_presigned_download_url(spaces_object_key)
                except RuntimeError:
                    row["video_url"] = row.get("video_url") or ""
    return [VideoAssetListItem(**row) for row in rows]


@router.post("/assets/{video_asset_id}/retry-issue")
def retry_video_asset_issue(video_asset_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return repositories.retry_video_asset_issue(db, video_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
            presigned_url = generate_presigned_download_url(spaces_object_key)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return RedirectResponse(url=presigned_url, status_code=307)

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
