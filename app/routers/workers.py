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
    pending_video_rows = repositories.list_pending_video_asset_retrievals(
        db,
        limit=max(settings.retrieval_max_global_workers * 50, 500),
    )
    pending_frame_rows = repositories.list_pending_trigger_frame_asset_retrievals(
        db,
        limit=max(settings.retrieval_max_global_workers * 50, 500),
    )
    running_video_rows = repositories.list_running_video_asset_retrievals(db)
    running_frame_rows = repositories.list_running_trigger_frame_asset_retrievals(db)
    pending_rows = [*pending_frame_rows, *pending_video_rows]
    running_rows = [*running_frame_rows, *running_video_rows]

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
                "running_frame_asset_ids": [],
                "queued_frame_asset_ids": [],
            },
        )
        current["queued_count"] += 1
        if row in pending_frame_rows:
            current["queued_frame_asset_ids"].append(int(row["id"]))
        else:
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
                "running_frame_asset_ids": [],
                "queued_frame_asset_ids": [],
            },
        )
        current["running_count"] += 1
        current["is_busy"] = current["running_count"] > 0
        if row in running_frame_rows:
            current["running_frame_asset_ids"].append(int(row["id"]))
        else:
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


def _build_entrance_analysis_status(db: Session) -> dict:
    pending_rows = repositories.list_pending_video_asset_analyses(
        db,
        limit=max(settings.analysis_max_global_workers * 50, 500),
    )
    running_rows = [
        row
        for row in repositories.list_running_video_asset_analyses(db)
        if str(row.get("section") or "").strip().lower() == "entrance"
    ]

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
        "paused": repositories.is_worker_paused(db, "entrance_analysis"),
        "locations": locations,
    }


@router.get("/entrance-analysis-status")
def get_entrance_analysis_status(db: Session = Depends(get_transaction_db)) -> dict:
    return _build_entrance_analysis_status(db)


@router.get("/analysis-status")
def get_analysis_status(db: Session = Depends(get_transaction_db)) -> dict:
    return _build_entrance_analysis_status(db)


@router.post("/entrance-analysis-control")
def update_entrance_analysis_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "entrance_analysis", payload.paused)
    return {
        "ok": True,
        **state,
    }


@router.post("/analysis-control")
def update_analysis_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "entrance_analysis", payload.paused)
    return {
        "ok": True,
        **state,
    }


@router.get("/grouping-status")
def get_grouping_status(db: Session = Depends(get_transaction_db)) -> dict:
    pending_rows = repositories.list_pending_grouping_batches(
        db,
        limit=max(settings.grouping_max_global_workers * 50, 500),
    )
    running_rows = repositories.list_running_grouping_batches(db)

    per_location: dict[int, dict] = {}
    for row in pending_rows:
        location_id = int(row["location_id"])
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_batch_ids": [],
                "queued_batch_ids": [],
            },
        )
        current["queued_count"] += 1
        current["queued_batch_ids"].append(int(row["id"]))

    for row in running_rows:
        location_id = int(row["location_id"])
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_batch_ids": [],
                "queued_batch_ids": [],
            },
        )
        current["running_count"] += 1
        current["is_busy"] = current["running_count"] > 0
        current["running_batch_ids"].append(int(row["id"]))

    return {
        "poll_seconds": settings.grouping_poll_seconds,
        "max_global_workers": settings.grouping_max_global_workers,
        "model": (
            settings.deepseek_vision_model
            if str(settings.grouping_provider or "gemini").strip().lower() == "deepseek"
            else settings.grouping_gemini_model
        ),
        "queued_count": len(pending_rows),
        "running_count": len(running_rows),
        "remote_dispatch_busy": repositories.has_active_remote_analysis_script_run(db, script_names=["grouping"]),
        "paused": repositories.is_worker_paused(db, "grouping"),
        "locations": sorted(per_location.values(), key=lambda item: int(item["location_id"])),
    }


@router.post("/grouping-control")
def update_grouping_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "grouping", payload.paused)
    return {
        "ok": True,
        **state,
    }


@router.get("/theft-confidence-status")
def get_theft_confidence_status(db: Session = Depends(get_transaction_db)) -> dict:
    pending_rows = repositories.list_pending_theft_confidence_batches(
        db,
        limit=max(settings.theft_confidence_max_global_workers * 50, 500),
    )
    per_location: dict[int, dict] = {}
    for row in pending_rows:
        location_id = int(row["location_id"])
        current = per_location.setdefault(
            location_id,
            {
                "location_id": location_id,
                "queued_count": 0,
                "running_count": 0,
                "is_busy": False,
                "running_batch_ids": [],
                "queued_batch_ids": [],
            },
        )
        current["queued_count"] += 1
        current["queued_batch_ids"].append(int(row["id"]))

    return {
        "poll_seconds": settings.theft_confidence_poll_seconds,
        "max_global_workers": settings.theft_confidence_max_global_workers,
        "queued_count": len(pending_rows),
        "running_count": 0,
        "remote_dispatch_busy": False,
        "paused": repositories.is_worker_paused(db, "theft_confidence_analysis"),
        "locations": sorted(per_location.values(), key=lambda item: int(item["location_id"])),
    }


@router.post("/theft-confidence-control")
def update_theft_confidence_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "theft_confidence_analysis", payload.paused)
    return {
        "ok": True,
        **state,
    }


@router.get("/kiosk-analysis-status")
def get_kiosk_analysis_status(db: Session = Depends(get_transaction_db)) -> dict:
    pending_rows = repositories.list_pending_kiosk_video_asset_analyses(
        db,
        limit=max(settings.kiosk_analysis_max_global_workers * 50, 500),
    )
    running_rows = [
        row
        for row in repositories.list_running_video_asset_analyses(db)
        if str(row.get("section") or "").strip().lower() == "kiosk"
    ]

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
        "poll_seconds": settings.kiosk_analysis_poll_seconds,
        "max_global_workers": settings.kiosk_analysis_max_global_workers,
        "cooldown_seconds": settings.kiosk_analysis_cooldown_seconds,
        "queued_count": len(pending_rows),
        "running_count": len(running_rows),
        "remote_dispatch_busy": repositories.has_active_remote_analysis_script_run(db),
        "paused": repositories.is_worker_paused(db, "kiosk_analysis"),
        "locations": locations,
    }


@router.post("/kiosk-analysis-control")
def update_kiosk_analysis_control(payload: WorkerControlRequest, db: Session = Depends(get_transaction_db)) -> dict:
    state = repositories.set_worker_paused(db, "kiosk_analysis", payload.paused)
    return {
        "ok": True,
        **state,
    }
