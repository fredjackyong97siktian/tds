from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
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
    return [VideoAssetListItem(**row) for row in rows]


@router.post("/triggers/{trigger_id}")
def create_trigger_video_asset(trigger_id: int, payload: VideoAssetCreate, db: Session = Depends(get_transaction_db)) -> dict:
    trigger = repositories.get_trigger(db, trigger_id)
    filename = infer_filename(payload.file_path or payload.video_url, f"trigger_{trigger_id}_{payload.section}", ".mp4")
    canonical_path = str(trigger_video_path(trigger["location_id"], trigger_id, payload.section, filename))
    original_video_url = payload.video_url
    original_file_path = payload.file_path
    video_asset_id = repositories.create_video_asset(
        db,
        {
            **payload.model_dump(),
            "trigger_id": trigger_id,
            "file_path": canonical_path,
            "metadata": {
                **(payload.metadata or {}),
                "original_video_url": original_video_url,
                "original_file_path": original_file_path,
            },
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
    original_video_url = payload.video_url
    original_file_path = payload.file_path
    video_asset_id = repositories.create_video_asset(
        db,
        {
            **payload.model_dump(),
            "file_path": canonical_path,
            "metadata": {
                **(payload.metadata or {}),
                "original_video_url": original_video_url,
                "original_file_path": original_file_path,
            },
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
    if not file_path:
        raise HTTPException(status_code=404, detail="Video asset does not have a private file path.")
    try:
        resolved = resolve_private_path(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Private video file not found.")
    return FileResponse(path=resolved, media_type=guess_media_type(str(resolved)), filename=resolved.name)
