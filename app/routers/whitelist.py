from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import (
    PhoneNumberCreate,
    PhoneNumberResponse,
    WhitelistEntryCreate,
    WhitelistEntryResponse,
    WhitelistSourceOption,
)


router = APIRouter(prefix="/api/v1/whitelist", tags=["whitelist"])


@router.get("", response_model=list[WhitelistEntryResponse])
def list_whitelist(db: Session = Depends(get_transaction_db)) -> list[WhitelistEntryResponse]:
    rows = repositories.list_whitelist_entries(db)
    return [WhitelistEntryResponse(**row) for row in rows]


@router.post("", response_model=WhitelistEntryResponse, status_code=status.HTTP_201_CREATED)
def create_whitelist(payload: WhitelistEntryCreate, db: Session = Depends(get_transaction_db)) -> WhitelistEntryResponse:
    try:
        row = repositories.create_whitelist_entry(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WhitelistEntryResponse(**row)


@router.delete("/{whitelist_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_whitelist(whitelist_id: int, db: Session = Depends(get_transaction_db)) -> Response:
    deleted = repositories.delete_whitelist_entry(db, whitelist_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Whitelist entry not found.")
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


@router.post("/phonenumbers", response_model=PhoneNumberResponse, status_code=status.HTTP_201_CREATED)
def create_phone_number(payload: PhoneNumberCreate, db: Session = Depends(get_transaction_db)) -> PhoneNumberResponse:
    try:
        row = repositories.create_phone_number_source(db, payload.phone_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PhoneNumberResponse(**row)
