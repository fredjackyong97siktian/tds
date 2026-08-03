from datetime import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories
from ..services import workflow_service
from ..schemas import (
    SessionCreate,
    SessionCustomerCreate,
    SessionEndTimeUpdateRequest,
    SessionFinalizeRequest,
    SessionFinalizeResponse,
    SessionListItem,
    SessionResponse,
    TransactionCreate,
)


router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
logger = logging.getLogger("tds.sessions_router")


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


@router.post("/{session_id}/end-time")
def update_session_end_time(
    session_id: int,
    payload: SessionEndTimeUpdateRequest,
    db: Session = Depends(get_transaction_db),
) -> dict:
    try:
        session = repositories.get_session(db, session_id)
        if payload.exit_trigger_id is not None:
            trigger = repositories.get_trigger(db, payload.exit_trigger_id)
            if int(trigger["location_id"]) != int(session["location_id"]):
                raise HTTPException(status_code=400, detail="Exit trigger must belong to the same location as the session.")
        row = repositories.update_session_fields(
            db,
            session_id=session_id,
            status="pending",
            end_time=payload.end_time,
            exit_trigger_id=payload.exit_trigger_id,
            issue_reason=None,
        )
        inserted_video_asset_ids = workflow_service.ensure_kiosk_video_assets_for_session(db, session_id)
        return {
            "session": row,
            "inserted_video_asset_ids": inserted_video_asset_ids,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Session end-time update failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Session end-time update failed: {exc}") from exc


@router.post("/{session_id}/finalize", response_model=SessionFinalizeResponse)
def finalize_session(session_id: int, payload: SessionFinalizeRequest, db: Session = Depends(get_transaction_db)) -> SessionFinalizeResponse:
    row = repositories.finalize_session_result(
        db,
        session_id=session_id,
        kiosk_total_items=payload.kiosk_total_items,
        actual_items_brought=payload.actual_items_brought,
    )
    return SessionFinalizeResponse(**row)


@router.post("/{session_id}/retry-issue")
def retry_session_issue(session_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        result = repositories.retry_session_issue(db, session_id)
        inserted_video_asset_ids = workflow_service.ensure_kiosk_video_assets_for_session(db, session_id)
        return {
            **result,
            "inserted_video_asset_ids": inserted_video_asset_ids,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Session retry failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Session retry failed: {exc}") from exc
