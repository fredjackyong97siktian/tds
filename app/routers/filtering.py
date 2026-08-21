from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/filtering", tags=["filtering"])


class TimePeriodPayload(BaseModel):
    location_id: int | None = None
    period_code: str
    label: str
    start_time: str
    end_time: str
    selected: bool = False
    metadata: dict[str, Any] | None = None


class FilterFactorPayload(BaseModel):
    location_id: int | None = None
    factor_code: str
    label: str
    enabled: bool = True
    weight: float = 1
    config: dict[str, Any] | None = None


@router.get("/time-periods")
def list_time_periods(db: Session = Depends(get_transaction_db)) -> list[dict[str, Any]]:
    return repositories.list_filter_time_periods(db)


@router.put("/time-periods/{period_code}")
def upsert_time_period(
    period_code: str,
    payload: TimePeriodPayload,
    db: Session = Depends(get_transaction_db),
) -> dict[str, Any]:
    data = payload.model_dump()
    data["period_code"] = period_code
    try:
        return repositories.upsert_filter_time_period(db, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/factors")
def list_factors(db: Session = Depends(get_transaction_db)) -> list[dict[str, Any]]:
    return repositories.list_filter_factors(db)


@router.put("/factors/{factor_code}")
def upsert_factor(
    factor_code: str,
    payload: FilterFactorPayload,
    db: Session = Depends(get_transaction_db),
) -> dict[str, Any]:
    data = payload.model_dump()
    data["factor_code"] = factor_code
    try:
        return repositories.upsert_filter_factor(db, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/grouping-batches")
def list_grouping_batches(db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    return {
        "pending": repositories.list_pending_grouping_batches(db, limit=200),
        "running": repositories.list_running_grouping_batches(db),
    }


@router.post("/grouping-batches/prepare")
def prepare_grouping_batches(db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        batches = workflow_service.prepare_due_grouping_batches(db)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "prepared_count": len(batches),
        "batches": batches,
    }
