from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_transaction_db
from .. import repositories


router = APIRouter(prefix="/api/v1/workers", tags=["workers"])


class WorkerControlRequest(BaseModel):
    paused: bool


@router.get("/retrieval-status")
def get_retrieval_status(db: Session = Depends(get_transaction_db)) -> dict:
    pending_rows = repositories.list_pending_video_asset_retrievals(
        db,
        limit=max(settings.retrieval_max_global_workers * 50, 500),
    )
    running_rows = repositories.list_running_video_asset_retrievals(db)

    per_location: dict[int, dict] = {}

    for row in pending_rows:
        location_id = row.get("location_id")
        if location_id is None:
            continue
        location_id = int(location_id)
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_video_asset_ids": [],
                "queued_video_asset_ids": [],
            },
        )
        current["queued_count"] += 1
        current["queued_video_asset_ids"].append(int(row["id"]))

    for row in running_rows:
        location_id = row.get("location_id")
        if location_id is None:
            continue
        location_id = int(location_id)
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_video_asset_ids": [],
                "queued_video_asset_ids": [],
            },
        )
        current["running_count"] += 1
        current["is_busy"] = current["running_count"] > 0
        current["running_video_asset_ids"].append(int(row["id"]))

    locations = sorted(per_location.values(), key=lambda item: int(item["location_id"]))

    return {
        "poll_seconds": settings.retrieval_poll_seconds,
        "max_global_workers": settings.retrieval_max_global_workers,
        "max_per_location": settings.retrieval_max_per_location,
        "queued_count": len(pending_rows),
        "running_count": len(running_rows),
        "paused": repositories.is_worker_paused(db, "retrieval"),
        "locations": locations,
    }


@router.post("/retrieval-control")
def update_retrieval_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "retrieval", payload.paused)
    return {
        "ok": True,
        **state,
    }


@router.get("/analysis-status")
def get_analysis_status(db: Session = Depends(get_transaction_db)) -> dict:
    pending_rows = repositories.list_pending_video_asset_analyses(
        db,
        limit=max(settings.analysis_max_global_workers * 50, 500),
    )
    running_rows = repositories.list_running_video_asset_analyses(db)

    per_location: dict[int, dict] = {}

    for row in pending_rows:
        location_id = row.get("location_id")
        if location_id is None:
            continue
        location_id = int(location_id)
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_video_asset_ids": [],
                "queued_video_asset_ids": [],
            },
        )
        current["queued_count"] += 1
        current["queued_video_asset_ids"].append(int(row["id"]))

    for row in running_rows:
        location_id = row.get("location_id")
        if location_id is None:
            continue
        location_id = int(location_id)
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_video_asset_ids": [],
                "queued_video_asset_ids": [],
            },
        )
        current["running_count"] += 1
        current["is_busy"] = current["running_count"] > 0
        current["running_video_asset_ids"].append(int(row["id"]))

    locations = sorted(per_location.values(), key=lambda item: int(item["location_id"]))

    return {
        "poll_seconds": settings.analysis_poll_seconds,
        "max_global_workers": settings.analysis_max_global_workers,
        "cooldown_seconds": settings.analysis_cooldown_seconds,
        "queued_count": len(pending_rows),
        "running_count": len(running_rows),
        "remote_dispatch_busy": repositories.has_active_remote_analysis_script_run(db),
        "paused": repositories.is_worker_paused(db, "analysis"),
        "locations": locations,
    }


@router.post("/analysis-control")
def update_analysis_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "analysis", payload.paused)
    return {
        "ok": True,
        **state,
    }
