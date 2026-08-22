from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import repositories
from ..config import settings
from ..db import get_transaction_db
from ..schemas import (
    EntryRunRequest,
    KioskRunRequest,
    RetrievalAcceptedResponse,
    RetrievalRequest,
    SessionPipelineLogResponse,
    ScriptRunDetailResponse,
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


@router.get("/script-runs/{script_run_id}", response_model=ScriptRunDetailResponse)
def get_script_run(
    script_run_id: int,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunDetailResponse:
    try:
        result = workflow_service.get_script_run_details(db, script_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScriptRunDetailResponse(**result)


@router.get("/script-runs/by-runner-job/{runner_job_id}", response_model=ScriptRunDetailResponse)
def get_script_run_by_runner_job_id(
    runner_job_id: str,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunDetailResponse:
    try:
        result = workflow_service.get_script_run_details_by_runner_job_id(db, runner_job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScriptRunDetailResponse(**result)


@router.get("/sessions/{session_id}/script-runs/latest", response_model=ScriptRunDetailResponse)
def get_latest_script_run_for_session(
    session_id: int,
    script_name: str | None = None,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunDetailResponse:
    try:
        result = workflow_service.get_latest_script_run_details_for_session(
            db,
            session_id,
            script_name=script_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScriptRunDetailResponse(**result)


@router.get("/videos/{video_asset_id}/script-runs/latest", response_model=ScriptRunDetailResponse)
def get_latest_script_run_for_video_asset(
    video_asset_id: int,
    db: Session = Depends(get_transaction_db),
) -> ScriptRunDetailResponse:
    try:
        result = workflow_service.get_latest_script_run_details_for_video_asset(db, video_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScriptRunDetailResponse(**result)


@router.get("/sessions/{session_id}/pipeline-log", response_model=SessionPipelineLogResponse)
def get_session_pipeline_log(
    session_id: int,
    db: Session = Depends(get_transaction_db),
) -> SessionPipelineLogResponse:
    try:
        result = workflow_service.get_session_pipeline_log_details(db, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionPipelineLogResponse(**result)


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
    payload: RetrievalRequest | None = None,
    db: Session = Depends(get_transaction_db),
) -> RetrievalAcceptedResponse:
    try:
        if payload is None:
            trigger = repositories.get_trigger(db, trigger_id)
            trigger_time = trigger.get("trigger_time")
            if isinstance(trigger_time, str):
                trigger_time = datetime.fromisoformat(trigger_time)
            if not isinstance(trigger_time, datetime):
                raise ValueError(f"Trigger {trigger_id} does not have a valid trigger_time.")
            location_id = int(trigger["location_id"])
            start_time = trigger_time - timedelta(seconds=int(settings.entrance_trigger_extra_before_seconds))
            end_time = trigger_time + timedelta(seconds=int(settings.entrance_trigger_extra_after_seconds))
        else:
            location_id = payload.location_id
            start_time = payload.start_time
            end_time = payload.end_time
        result = workflow_service.retrieve_entrance_video_window(
            db,
            trigger_id=trigger_id,
            location_id=location_id,
            start_time=start_time,
            end_time=end_time,
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
