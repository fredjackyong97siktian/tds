from datetime import datetime
from datetime import timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import VectorSessionLocal, get_transaction_db
from .. import repositories, vector_repositories
from ..services import workflow_service
from ..schemas import (
    SessionCreate,
    SessionCustomerResponse,
    SessionCustomerCreate,
    SessionEndTimeUpdateRequest,
    SessionFinalizeRequest,
    SessionFinalizeResponse,
    SessionListItem,
    SessionResponse,
    SessionStatusUpdateRequest,
    SessionTransactionDetailResponse,
    SessionTransactionResponse,
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


@router.get("/{session_id}/customers", response_model=list[SessionCustomerResponse])
def list_session_customers(session_id: int, db: Session = Depends(get_transaction_db)) -> list[SessionCustomerResponse]:
    rows = repositories.list_session_customers(db, session_id)
    return [SessionCustomerResponse(**row) for row in rows]


@router.get("/{session_id}/transactions", response_model=list[SessionTransactionResponse])
def list_session_transactions(session_id: int, db: Session = Depends(get_transaction_db)) -> list[SessionTransactionResponse]:
    rows = repositories.list_session_transactions(db, session_id)
    return [SessionTransactionResponse(**row) for row in rows]


@router.get("/{session_id}/transaction-details", response_model=list[SessionTransactionDetailResponse])
def list_session_transaction_details(session_id: int, db: Session = Depends(get_transaction_db)) -> list[SessionTransactionDetailResponse]:
    rows = repositories.list_session_transaction_details(db, session_id)
    return [SessionTransactionDetailResponse(**row) for row in rows]


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
        exit_video_asset_id = None
        if payload.exit_trigger_id is not None:
            trigger = repositories.get_trigger(db, payload.exit_trigger_id)
            try:
                exit_video = repositories.get_latest_video_asset_for_trigger(
                    db,
                    trigger_id=int(payload.exit_trigger_id),
                    section="entrance",
                )
            except ValueError:
                trigger_time = trigger["trigger_time"]
                queued = workflow_service.retrieve_entrance_video_window(
                    db,
                    trigger_id=int(payload.exit_trigger_id),
                    location_id=int(session["location_id"]),
                    start_time=trigger_time - timedelta(seconds=int(workflow_service.settings.entrance_trigger_extra_before_seconds)),
                    end_time=trigger_time + timedelta(seconds=int(workflow_service.settings.entrance_trigger_extra_after_seconds)),
                )
                exit_video_asset_id = int(queued.video_asset_id)
                exit_clip_start_time = queued.requested_start_time
                exit_clip_end_time = queued.requested_end_time
            else:
                exit_video_asset_id = int(exit_video["id"])
                exit_clip_start_time = exit_video.get("captured_start_time")
                exit_clip_end_time = exit_video.get("captured_end_time")

            if exit_video_asset_id is not None:
                repositories.create_session_video_asset_link(
                    db,
                    session_id,
                    exit_video_asset_id,
                    {
                        "link_section": "exit",
                        "link_sequence_no": None,
                        "clip_start_time": exit_clip_start_time,
                        "clip_end_time": exit_clip_end_time,
                        "is_primary": False,
                        "metadata": {
                            "source": "session_end_time_update",
                            "exit_trigger_id": int(payload.exit_trigger_id),
                        },
                    },
                )
        inserted_video_asset_ids = workflow_service.ensure_kiosk_video_assets_for_session(db, session_id)
        return {
            "session": row,
            "inserted_video_asset_ids": inserted_video_asset_ids,
            "exit_video_asset_id": exit_video_asset_id,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Session end-time update failed for session_id=%s", session_id)
        raise HTTPException(status_code=500, detail=f"Session end-time update failed: {exc}") from exc


@router.patch("/{session_id}/status", response_model=SessionResponse)
def update_session_status(
    session_id: int,
    payload: SessionStatusUpdateRequest,
    db: Session = Depends(get_transaction_db),
) -> SessionResponse:
    row = repositories.update_session_fields(
        db,
        session_id=session_id,
        status=payload.status,
        issue_reason=None if payload.status != "issue" else repositories.get_session(db, session_id).get("issue_reason"),
    )
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


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: int, db: Session = Depends(get_transaction_db)) -> None:
    try:
        session = repositories.get_session(db, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session_customers = repositories.list_session_customers(db, session_id)
    session_customer_ids = [int(row["id"]) for row in session_customers if row.get("id") is not None]
    with VectorSessionLocal() as vector_db:
        if session_customer_ids:
            vector_repositories.purge_session_customer_records(
                vector_db,
                session_customer_ids=session_customer_ids,
            )
        vector_repositories.purge_session_records(vector_db, session_id=session_id)
    repositories.delete_session(db, session_id)
    logger.info(
        "Deleted session session_id=%s location_id=%s customers=%s",
        session_id,
        session.get("location_id"),
        len(session_customer_ids),
    )


@router.delete("/customers/{session_customer_id}", status_code=204)
def delete_session_customer(session_customer_id: int, db: Session = Depends(get_transaction_db)) -> None:
    try:
        row = repositories.get_session_customer(db, session_customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session_id = int(row["session_id"])
    repositories.delete_session_customer(db, session_customer_id)
    with VectorSessionLocal() as vector_db:
        vector_repositories.purge_session_customer_records(
            vector_db,
            session_customer_ids=[session_customer_id],
        )
    repositories.update_session_fields(
        db,
        session_id=session_id,
        total_customer=repositories.get_session_customer_count(db, session_id),
    )


@router.post("/customers/{session_customer_id}/mark-exit")
def mark_session_customer_exit(session_customer_id: int, db: Session = Depends(get_transaction_db)) -> dict:
    try:
        session_customer = repositories.get_session_customer(db, session_customer_id)
        session_id = int(session_customer["session_id"])
        session = repositories.get_session(db, session_id)
        location_id = int(session["location_id"])
        person_id = session_customer.get("person_id")
        leave_time = session.get("end_time") or datetime.utcnow()

        archived_count = 0
        identity_id = None
        with VectorSessionLocal() as vector_db:
            identity = vector_repositories.find_identity_by_current_session_customer(
                vector_db,
                location_id=location_id,
                current_session_customer_id=session_customer_id,
            )
            if identity is not None:
                identity_id = int(identity["id"])
                vector_repositories.update_identity_record(
                    vector_db,
                    identity_id,
                    status="exited",
                    current_session_id=session_id,
                    current_session_customer_id=session_customer_id,
                    person_id=int(person_id) if person_id is not None else None,
                    last_seen_at=leave_time,
                    exited_at=leave_time,
                    metadata={
                        "source": "dashboard_manual_exit",
                        "session_id": session_id,
                        "session_customer_id": session_customer_id,
                        "person_id": person_id,
                    },
                )
            archived_count = vector_repositories.archive_active_gallery_by_aliases(
                vector_db,
                location_id=location_id,
                session_customer_ids=[session_customer_id],
                archived_reason="manual_exit",
                metadata_extra={
                    "source": "dashboard_manual_exit",
                    "session_id": session_id,
                    "session_customer_id": session_customer_id,
                    "person_id": person_id,
                    "identity_id": identity_id,
                },
            )

        repositories.update_session_customer_leave_time(
            db,
            session_customer_id=session_customer_id,
            leave_time=leave_time,
            match_status="resolved",
        )

        workflow_service._maybe_close_session_and_prepare_kiosk(  # noqa: SLF001
            db,
            session_id=session_id,
            exit_trigger_id=session.get("exit_trigger_id"),
        )
        updated_customer = repositories.get_session_customer(db, session_customer_id)
        updated_session = repositories.get_session(db, session_id)
        return {
            "ok": True,
            "session_customer": updated_customer,
            "session": updated_session,
            "archived_count": archived_count,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        logger.exception("Manual exit failed due to vector gallery permission issue for session_customer_id=%s", session_customer_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Manual exit failed for session_customer_id=%s", session_customer_id)
        raise HTTPException(status_code=500, detail=f"Manual exit failed: {exc}") from exc
