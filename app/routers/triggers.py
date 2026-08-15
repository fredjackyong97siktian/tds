from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..schemas import TriggerCreate, TriggerListItem, TriggerResponse


router = APIRouter(prefix="/api/v1/triggers", tags=["triggers"])


@router.post("", response_model=TriggerResponse)
def create_trigger(payload: TriggerCreate, db: Session = Depends(get_transaction_db)) -> TriggerResponse:
    row = repositories.create_trigger(db, payload.model_dump())
    return TriggerResponse(**row)


@router.get("", response_model=list[TriggerListItem])
def list_triggers(limit: int = 50, db: Session = Depends(get_transaction_db)) -> list[TriggerListItem]:
    rows = repositories.list_triggers(db, limit=limit)
    return [TriggerListItem(**row) for row in rows]


@router.get("/{trigger_id}", response_model=TriggerListItem)
def get_trigger(trigger_id: int, db: Session = Depends(get_transaction_db)) -> TriggerListItem:
    try:
        row = repositories.get_trigger(db, trigger_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = {
        **row,
        "latest_script_name": None,
        "latest_script_status": None,
        "latest_script_finished_at": None,
        "latest_error_log": None,
        "latest_video_asset_id": None,
        "latest_video_status": None,
        "can_retry": False,
        "retry_to_status": None,
    }
    return TriggerListItem(**payload)


@router.post("/{trigger_id}/retry-issue")
def retry_trigger_issue(trigger_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return repositories.retry_trigger_issue(db, trigger_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
