from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from .. import repositories


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/activity-timeseries")
def get_activity_timeseries(hours: int = 24, db: Session = Depends(get_transaction_db)) -> list[dict]:
    return repositories.get_dashboard_activity_timeseries(db, hours=hours)


@router.get("/customer-group-type-breakdown")
def get_customer_group_type_breakdown(hours: int = 24, db: Session = Depends(get_transaction_db)) -> dict[str, int]:
    return repositories.get_customer_group_type_breakdown(db, hours=hours)
