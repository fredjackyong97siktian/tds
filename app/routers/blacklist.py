from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import BlockedEntryResponse, UpdateEntryStatusRequest, WhitelistSourceOption


router = APIRouter(prefix="/api/v1/blacklist", tags=["blacklist"])


@router.get("/_debug/entrylogs-status-counts")
def debug_entrylogs_status_counts(db: Session = Depends(get_transaction_db)) -> dict:
    return repositories.debug_entrylogs_status_counts(db)


@router.get("", response_model=list[BlockedEntryResponse])
def list_blacklist(db: Session = Depends(get_transaction_db)) -> list[BlockedEntryResponse]:
    rows = repositories.list_blocked_entries(db)
    return [BlockedEntryResponse(**row) for row in rows]


@router.patch("/status", response_model=BlockedEntryResponse)
def update_status(payload: UpdateEntryStatusRequest, db: Session = Depends(get_transaction_db)) -> BlockedEntryResponse:
    try:
        row = repositories.update_entry_status(
            db,
            method=payload.method,
            entry_id=payload.entry_id,
            status_value=payload.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BlockedEntryResponse(**row)


@router.get("/source-options", response_model=list[WhitelistSourceOption])
def list_source_options(
    method: str = Query(pattern="^(qrentry|entrylogs)$"),
    search: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_transaction_db),
) -> list[WhitelistSourceOption]:
    try:
        rows = repositories.list_whitelist_source_options(db, method, search=search, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [WhitelistSourceOption(**row) for row in rows]
