from fastapi import APIRouter, Depends
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
