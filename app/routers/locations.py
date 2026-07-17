from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repositories
from ..crypto import encrypt_secret
from ..db import get_transaction_db
from ..schemas import LocationEndpointUpsert, LocationOption


router = APIRouter(prefix="/api/v1/locations", tags=["locations"])


@router.get("", response_model=list[LocationOption])
def list_locations(db: Session = Depends(get_transaction_db)) -> list[LocationOption]:
    rows = repositories.list_locations(db)
    return [LocationOption(**row) for row in rows]


@router.get("/{location_id}", response_model=LocationOption)
def get_location(location_id: int, db: Session = Depends(get_transaction_db)) -> LocationOption:
    try:
        row = repositories.get_location_endpoint(db, location_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Location not found.") from exc
    return LocationOption(**row)


@router.put("/{location_id}/endpoint", response_model=LocationOption)
def upsert_location_endpoint(
    location_id: int,
    payload: LocationEndpointUpsert,
    db: Session = Depends(get_transaction_db),
) -> LocationOption:
    try:
        row = repositories.upsert_location_endpoint(
            db,
            location_id,
            {
                "dahua_host": payload.dahua_host,
                "dahua_username": payload.dahua_username,
                "dahua_password_encrypted": encrypt_secret(payload.dahua_password) if payload.dahua_password else None,
                "rtsp_port": payload.rtsp_port,
                "notes": payload.notes,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Location not found.") from exc
    return LocationOption(**row)
