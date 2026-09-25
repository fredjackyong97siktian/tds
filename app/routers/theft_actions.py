import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/sessions/{session_id}/theft-action", tags=["theft-actions"])
logger = logging.getLogger("tds.theft_actions_router")


class TheftActionLineItem(BaseModel):
    name: str
    price: float
    quantity: int = 1


class TheftActionChargeRequest(BaseModel):
    line_items: list[TheftActionLineItem] = Field(min_length=1)
    triggered_by: str | None = None


class TheftActionWhatsappRequest(BaseModel):
    message_text: str = Field(min_length=1)
    triggered_by: str | None = None


@router.get("/worksheet")
def get_worksheet(session_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        return workflow_service.get_theft_action_worksheet(db, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/fallback-phone")
def get_fallback_phone(session_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    phone_number = workflow_service.resolve_theft_action_fallback_phone(db, session_id)
    return {"session_id": session_id, "phone_number": phone_number}


@router.post("/charge")
def charge(
    session_id: int,
    payload: TheftActionChargeRequest,
    db: Session = Depends(get_transaction_db),
) -> dict:
    try:
        return workflow_service.trigger_theft_action_charge(
            db,
            session_id=session_id,
            line_items=[item.model_dump() for item in payload.line_items],
            triggered_by=payload.triggered_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Theft action Stripe charge failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Theft action charge failed: {exc}") from exc


@router.post("/whatsapp")
def send_whatsapp(
    session_id: int,
    payload: TheftActionWhatsappRequest,
    db: Session = Depends(get_transaction_db),
) -> dict:
    try:
        return workflow_service.trigger_theft_action_whatsapp(
            db,
            session_id=session_id,
            message_text=payload.message_text,
            triggered_by=payload.triggered_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Theft action WhatsApp send failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Theft action WhatsApp send failed: {exc}") from exc
