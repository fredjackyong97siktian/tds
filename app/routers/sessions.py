from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..schemas import (
    SessionCreate,
    SessionCustomerCreate,
    SessionFinalizeRequest,
    SessionFinalizeResponse,
    SessionListItem,
    SessionResponse,
    TransactionCreate,
)


router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(payload: SessionCreate, db: Session = Depends(get_transaction_db)) -> SessionResponse:
    row = repositories.create_session(db, payload.model_dump())
    return SessionResponse(**row)


@router.get("", response_model=list[SessionListItem])
def list_sessions(limit: int = 50, db: Session = Depends(get_transaction_db)) -> list[SessionListItem]:
    rows = repositories.list_sessions(db, limit=limit)
    return [SessionListItem(**row) for row in rows]


@router.post("/{session_id}/customers")
def upsert_session_customer(session_id: int, payload: SessionCustomerCreate, db: Session = Depends(get_transaction_db)) -> dict:
    repositories.create_session_customer(db, session_id, payload.model_dump())
    return {"ok": True, "session_id": session_id, "person_id": payload.person_id}


@router.post("/{session_id}/transactions")
def add_transaction(session_id: int, payload: TransactionCreate, db: Session = Depends(get_transaction_db)) -> dict:
    repositories.create_transaction(db, session_id, payload.model_dump())
    return {"ok": True, "session_id": session_id, "receipt_number": payload.receipt_number}


@router.post("/{session_id}/close", response_model=SessionResponse)
def close_session(session_id: int, exit_trigger_id: int | None = None, db: Session = Depends(get_transaction_db)) -> SessionResponse:
    row = repositories.close_session(db, session_id, datetime.utcnow(), exit_trigger_id)
    return SessionResponse(**row)


@router.post("/{session_id}/finalize", response_model=SessionFinalizeResponse)
def finalize_session(session_id: int, payload: SessionFinalizeRequest, db: Session = Depends(get_transaction_db)) -> SessionFinalizeResponse:
    row = repositories.finalize_session_result(
        db,
        session_id=session_id,
        kiosk_total_items=payload.kiosk_total_items,
        actual_items_brought=payload.actual_items_brought,
    )
    return SessionFinalizeResponse(**row)
