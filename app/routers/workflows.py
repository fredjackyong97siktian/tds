from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from ..schemas import (
    EntryRunRequest,
    KioskRunRequest,
    RetrievalAcceptedResponse,
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


@router.post("/sessions/{session_id}/retrieve-kiosk-video", response_model=RetrievalAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def retrieve_kiosk_video(
    session_id: int,
    payload: RetrievalRequest,
    db: Session = Depends(get_transaction_db),
) -> RetrievalAcceptedResponse:
    try:
        result = workflow_service.retrieve_kiosk_video_window(
            db,
            session_id=session_id,
            location_id=payload.location_id,
            start_time=payload.start_time,
            end_time=payload.end_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalAcceptedResponse(
        message="Kiosk video retrieval queued for worker pickup.",
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


@router.post("/triggers/{trigger_id}/retrieve-entrance-video", response_model=RetrievalAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
def retrieve_entrance_video(
    trigger_id: int,
    payload: RetrievalRequest,
    db: Session = Depends(get_transaction_db),
) -> RetrievalAcceptedResponse:
    try:
        result = workflow_service.retrieve_entrance_video_window(
            db,
            trigger_id=trigger_id,
            location_id=payload.location_id,
            start_time=payload.start_time,
            end_time=payload.end_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalAcceptedResponse(
        message="Entrance video retrieval queued for worker pickup.",
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


@router.get("/triggers/{trigger_id}/video-ready-policy")
def video_ready_policy(trigger_id: int, created_time: str, retries_used: int = 0) -> dict:
    from datetime import datetime

    parsed = datetime.fromisoformat(created_time)
    return {"trigger_id": trigger_id, **workflow_service.check_video_ready_policy(parsed, retries_used)}
