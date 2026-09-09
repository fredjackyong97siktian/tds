from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/activity-timeseries")
def get_activity_timeseries(days: int = 30, db: Session = Depends(get_transaction_db)) -> list[dict]:
    return repositories.get_dashboard_activity_timeseries(db, days=days)
