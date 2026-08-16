from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..schemas import (
    AlertKioskRetrievalRequest,
    RetrievalAcceptedResponse,
    ThiefAlertCheckResponse,
    ThiefAlertItem,
    ThiefAlertSummary,
)
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=list[ThiefAlertItem])
def list_alerts(limit: int = 100, db: Session = Depends(get_transaction_db)) -> list[ThiefAlertItem]:
    rows = repositories.list_thief_alerts(db, limit=limit)
    return [ThiefAlertItem(**row) for row in rows]


@router.get("/summary", response_model=ThiefAlertSummary)
def get_alert_summary(db: Session = Depends(get_transaction_db)) -> ThiefAlertSummary:
    return ThiefAlertSummary(unchecked_count=repositories.get_unchecked_thief_alert_count(db))


@router.get("/{alert_id}", response_model=ThiefAlertItem)
def get_alert(alert_id: int, db: Session = Depends(get_transaction_db)) -> ThiefAlertItem:
    try:
        row = repositories.get_thief_alert(db, alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ThiefAlertItem(**row)


@router.post("/{alert_id}/check", response_model=ThiefAlertCheckResponse)
def mark_alert_checked(alert_id: int, db: Session = Depends(get_transaction_db)) -> ThiefAlertCheckResponse:
    try:
        row = repositories.mark_thief_alert_checked(db, alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ThiefAlertCheckResponse(id=int(row["id"]), checked=bool(row["checked"]))


@router.post("/{alert_id}/retrieve-kiosk-video", response_model=RetrievalAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def retrieve_alert_kiosk_video(
    alert_id: int,
    payload: AlertKioskRetrievalRequest,
    db: Session = Depends(get_transaction_db),
) -> RetrievalAcceptedResponse:
    try:
        alert = repositories.get_thief_alert(db, alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if str(alert.get("method") or "").strip().lower() != "kiosk":
        raise HTTPException(status_code=400, detail="Only Kiosk alerts can queue kiosk video retrieval.")
    anchor_time = alert.get("created_at")
    if anchor_time is None:
        raise HTTPException(status_code=400, detail="Alert does not have a createdAt timestamp.")
    result = workflow_service.retrieve_alert_kiosk_video_window(
        db,
        alert_id=alert_id,
        location_id=int(alert["location_id"]),
        start_time=anchor_time - timedelta(seconds=int(payload.before_seconds)),
        end_time=anchor_time + timedelta(seconds=int(payload.after_seconds)),
    )
    return RetrievalAcceptedResponse(
        message="Alert kiosk video retrieval queued for worker pickup.",
        video_asset_id=result.video_asset_id,
        trigger_id=result.trigger_id,
        session_id=result.session_id,
        location_id=result.location_id,
        section=result.section,
        status=result.status,
        video_url=result.video_url,
        file_path=result.output_path,
        requested_start_time=result.requested_start_time,
        requested_end_time=result.requested_end_time,
    )
