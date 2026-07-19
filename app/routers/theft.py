from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import TheftListItem


router = APIRouter(prefix="/api/v1/theft", tags=["theft"])


@router.get("", response_model=list[TheftListItem])
def list_theft_transactions(limit: int = 50, db: Session = Depends(get_transaction_db)) -> list[TheftListItem]:
    rows = repositories.list_theft_transactions(db, limit=limit)
    return [TheftListItem(**row) for row in rows]
