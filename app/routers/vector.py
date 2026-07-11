from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_vector_db
from ..schemas import (
    ActiveGalleryResponse,
    ActiveGalleryUpsert,
    CustomerGalleryCreate,
    CustomerGalleryResponse,
)
from .. import vector_repositories
from ..storage import guess_media_type, resolve_private_path


router = APIRouter(prefix="/api/v1/vector", tags=["vector"])


@router.post("/sessions/{session_id}/customer-gallery", response_model=CustomerGalleryResponse)
def create_customer_gallery(
    session_id: int,
    payload: CustomerGalleryCreate,
    db: Session = Depends(get_vector_db),
) -> CustomerGalleryResponse:
    row = vector_repositories.create_customer_gallery_record(
        db,
        location_id=payload.location_id,
        session_id=session_id,
        person_id=payload.person_id,
        session_customer_id=payload.session_customer_id,
        image_url=payload.image_url,
        image_kind=payload.image_kind,
        embedding_osnet=payload.embedding_osnet,
        embedding_fashion=payload.embedding_fashion,
        metadata=payload.metadata,
    )
    return CustomerGalleryResponse(**row)


@router.get("/sessions/{session_id}/customer-gallery", response_model=list[CustomerGalleryResponse])
def list_customer_gallery(session_id: int, db: Session = Depends(get_vector_db)) -> list[CustomerGalleryResponse]:
    rows = vector_repositories.list_customer_gallery_records(db, session_id=session_id)
    return [CustomerGalleryResponse(**row) for row in rows]


@router.put("/locations/{location_id}/active-gallery/{session_customer_id}", response_model=ActiveGalleryResponse)
def upsert_active_gallery(
    location_id: int,
    session_customer_id: int,
    payload: ActiveGalleryUpsert,
    db: Session = Depends(get_vector_db),
) -> ActiveGalleryResponse:
    row = vector_repositories.upsert_and_get_active_gallery(
        db,
        location_id=location_id,
        session_id=payload.session_id,
        session_customer_id=session_customer_id,
        person_id=payload.person_id,
        state_kind=payload.state_kind,
        state_payload=payload.state_payload,
        metadata=payload.metadata,
    )
    return ActiveGalleryResponse(**row)


@router.get("/locations/{location_id}/active-gallery/{session_customer_id}/{state_kind}", response_model=ActiveGalleryResponse)
def get_active_gallery(
    location_id: int,
    session_customer_id: int,
    state_kind: str,
    db: Session = Depends(get_vector_db),
) -> ActiveGalleryResponse:
    row = vector_repositories.get_active_gallery(
        db,
        location_id=location_id,
        session_customer_id=session_customer_id,
        state_kind=state_kind,
    )
    return ActiveGalleryResponse(**row)


@router.get("/active-gallery", response_model=list[ActiveGalleryResponse])
def list_active_gallery(
    location_id: int | None = None,
    limit: int = 100,
    db: Session = Depends(get_vector_db),
) -> list[ActiveGalleryResponse]:
    rows = vector_repositories.list_active_gallery_records(db, location_id=location_id, limit=limit)
    return [ActiveGalleryResponse(**row) for row in rows]


@router.get("/customer-gallery/{gallery_id}/image")
def get_customer_gallery_image(gallery_id: int, db: Session = Depends(get_vector_db)) -> FileResponse:
    row = vector_repositories.get_customer_gallery_record(db, gallery_id)
    metadata = row.get("metadata") or {}
    image_path = metadata.get("image_path") or row.get("image_url")
    if not image_path:
        raise HTTPException(status_code=404, detail="Customer gallery record does not have a private image path.")
    try:
        resolved = resolve_private_path(image_path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Private gallery image not found.")
    return FileResponse(path=resolved, media_type=guess_media_type(str(resolved)), filename=resolved.name)
