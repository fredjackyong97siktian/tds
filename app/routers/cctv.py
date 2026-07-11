from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import CctvCreate, CctvResponse, CctvUpdate


router = APIRouter(prefix="/api/v1/cctv", tags=["cctv"])


@router.get("", response_model=list[CctvResponse])
def list_cctv(location_id: int | None = None, db: Session = Depends(get_transaction_db)) -> list[CctvResponse]:
    rows = repositories.list_cctv(db, location_id=location_id)
    return [CctvResponse(**row) for row in rows]


@router.get("/{cctv_id}", response_model=CctvResponse)
def get_cctv(cctv_id: int, db: Session = Depends(get_transaction_db)) -> CctvResponse:
    try:
        row = repositories.get_cctv(db, cctv_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="CCTV record not found.") from exc
    return CctvResponse(**row)


@router.post("", response_model=CctvResponse, status_code=status.HTTP_201_CREATED)
def create_cctv(payload: CctvCreate, db: Session = Depends(get_transaction_db)) -> CctvResponse:
    row = repositories.create_cctv(db, payload.model_dump())
    return CctvResponse(**row)


@router.put("/{cctv_id}", response_model=CctvResponse)
def update_cctv(cctv_id: int, payload: CctvUpdate, db: Session = Depends(get_transaction_db)) -> CctvResponse:
    try:
        row = repositories.update_cctv(db, cctv_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="CCTV record not found.") from exc
    return CctvResponse(**row)


@router.delete("/{cctv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cctv(cctv_id: int, db: Session = Depends(get_transaction_db)) -> Response:
    deleted = repositories.delete_cctv(db, cctv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="CCTV record not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
