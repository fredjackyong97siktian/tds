from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import LocationOption


router = APIRouter(prefix="/api/v1/locations", tags=["locations"])


@router.get("", response_model=list[LocationOption])
def list_locations(db: Session = Depends(get_transaction_db)) -> list[LocationOption]:
    rows = repositories.list_locations(db)
    return [LocationOption(**row) for row in rows]
