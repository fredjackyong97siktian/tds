from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..schemas import TriggerCreate, TriggerListItem, TriggerResponse
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/triggers", tags=["triggers"])


@router.post("", response_model=TriggerResponse)
def create_trigger(payload: TriggerCreate, db: Session = Depends(get_transaction_db)) -> TriggerResponse:
    row = repositories.create_trigger(db, payload.model_dump())
    return TriggerResponse(**row)


@router.get("", response_model=list[TriggerListItem])
def list_triggers(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_transaction_db),
) -> list[TriggerListItem]:
    rows = repositories.list_triggers(db, limit=max(1, min(limit, 500)), offset=max(0, offset))
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


@router.get("/{trigger_id}/card-country")
def get_trigger_card_country(trigger_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        trigger = repositories.get_trigger(db, trigger_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    credit_card_entry_id = trigger.get("credit_card_entry_id")
    if credit_card_entry_id is None:
        return {
            "trigger_id": trigger_id,
            "credit_card_entry_id": None,
            "fingerprint": None,
            "country": None,
            "source": "no_credit_card_entry",
        }
    identity = repositories.get_credit_card_entry_identity(db, credit_card_entry_id)
    stored_country = identity.get("country") if identity else None
    lookup = (
        {"country": stored_country, "source": "stored_entrylogs_country"}
        if stored_country
        else workflow_service.resolve_stripe_card_country_for_identity(identity)
    )
    return {
        "trigger_id": trigger_id,
        "credit_card_entry_id": credit_card_entry_id,
        "fingerprint": identity.get("fingerprint") if identity else None,
        "country": lookup.get("country"),
        "source": lookup.get("source"),
        "attempts": lookup.get("attempts", []),
        "error": lookup.get("error"),
    }


@router.post("/{trigger_id}/retry-issue")
def retry_trigger_issue(trigger_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return repositories.retry_trigger_issue(db, trigger_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
