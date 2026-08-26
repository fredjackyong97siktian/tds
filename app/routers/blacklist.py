from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import BlacklistEntryCreate, BlacklistEntryResponse, WhitelistSourceOption


router = APIRouter(prefix="/api/v1/blacklist", tags=["blacklist"])


@router.get("", response_model=list[BlacklistEntryResponse])
def list_blacklist(db: Session = Depends(get_transaction_db)) -> list[BlacklistEntryResponse]:
    rows = repositories.list_blacklist_entries(db)
    return [BlacklistEntryResponse(**row) for row in rows]


@router.post("", response_model=BlacklistEntryResponse, status_code=status.HTTP_201_CREATED)
def create_blacklist(payload: BlacklistEntryCreate, db: Session = Depends(get_transaction_db)) -> BlacklistEntryResponse:
    try:
        row = repositories.create_blacklist_entry(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BlacklistEntryResponse(**row)


@router.delete("/{blacklist_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_blacklist(blacklist_id: int, db: Session = Depends(get_transaction_db)) -> Response:
    deleted = repositories.delete_blacklist_entry(db, blacklist_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Blacklist entry not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
