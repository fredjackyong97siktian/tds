from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from ..schemas import (
    EntryRunRequest,
    KioskRunRequest,
    RetrievalRequest,
    ScriptRunResponse,
)
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


@router.post("/triggers/{trigger_id}/run-entry", response_model=ScriptRunResponse)
def run_entry(
    trigger_id: int,
    session_id: int,
    payload: EntryRunRequest,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunResponse:
    result = workflow_service.run_entry_for_trigger(
        db,
        trigger_id=trigger_id,
        session_id=session_id,
        video_path=payload.video_path,
        model_name=payload.model_name,
        output_dir=payload.output_dir,
        gallery_state_path=payload.gallery_state_path,
    )
    return ScriptRunResponse(**result.__dict__)


@router.post("/sessions/{session_id}/run-kiosk", response_model=ScriptRunResponse)
def run_kiosk(
    session_id: int,
    payload: KioskRunRequest,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunResponse:
    result = workflow_service.run_kiosk_for_session(
        db,
        session_id=session_id,
        video_path=payload.video_path,
        model_name=payload.model_name,
        output_dir=payload.output_dir,
        gallery_state_path=payload.gallery_state_path,
    )
    return ScriptRunResponse(**result.__dict__)


@router.post("/sessions/{session_id}/retrieve-kiosk-video")
def retrieve_kiosk_video(session_id: int, payload: RetrievalRequest) -> dict:
    return {
        "session_id": session_id,
        **workflow_service.retrieve_kiosk_video_window(
            start_time=payload.start_time,
            end_time=payload.end_time,
        ),
    }


@router.get("/triggers/{trigger_id}/video-ready-policy")
def video_ready_policy(trigger_id: int, created_time: str, retries_used: int = 0) -> dict:
    from datetime import datetime

    parsed = datetime.fromisoformat(created_time)
    return {"trigger_id": trigger_id, **workflow_service.check_video_ready_policy(parsed, retries_used)}
