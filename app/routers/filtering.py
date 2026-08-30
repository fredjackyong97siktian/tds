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


class CountryCodeCheckPayload(BaseModel):
    location_id: int | None = None
    country_code: str
    country_name: str | None = None
    phone_prefix: str | None = None
    card_country: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] | None = None


class GroupingRangePayload(BaseModel):
    location_id: int | None = None
    start_time: str
    end_time: str


class SelfGroupingGroupPayload(BaseModel):
    group_id: str | int | None = None
    entry: list[int] = []
    exit: list[int] = []
    total_customer: int | None = None


class SelfGroupingPayload(BaseModel):
    location_id: int
    start_time: str
    end_time: str
    groups: list[SelfGroupingGroupPayload]


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


@router.delete("/time-periods/{period_code}")
def delete_time_period(
    period_code: str,
    location_id: int | None = None,
    db: Session = Depends(get_transaction_db),
) -> dict[str, Any]:
    try:
        repositories.delete_filter_time_period(db, period_code=period_code, location_id=location_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


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


@router.get("/country-code-checks")
def list_country_code_checks(db: Session = Depends(get_transaction_db)) -> list[dict[str, Any]]:
    return repositories.list_filter_country_code_checks(db)


@router.post("/country-code-checks")
def create_country_code_check(
    payload: CountryCodeCheckPayload,
    db: Session = Depends(get_transaction_db),
) -> dict[str, Any]:
    try:
        return repositories.create_filter_country_code_check(db, payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/country-code-checks/{rule_id}")
def update_country_code_check(
    rule_id: int,
    payload: CountryCodeCheckPayload,
    db: Session = Depends(get_transaction_db),
) -> dict[str, Any]:
    try:
        return repositories.update_filter_country_code_check(db, rule_id, payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/country-code-checks/{rule_id}")
def delete_country_code_check(rule_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        repositories.delete_filter_country_code_check(db, rule_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/grouping-batches")
def list_grouping_batches(db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    return {
        "pending": repositories.list_pending_grouping_batches(db, limit=200),
        "running": repositories.list_running_grouping_batches(db),
        "recent": repositories.list_recent_grouping_batches(db, limit=100),
    }


@router.get("/grouping-batches/{batch_id}")
def get_grouping_batch(batch_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return repositories.get_grouping_batch(db, batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/confidence-results")
def list_confidence_results(
    limit: int = 100,
    offset: int = 0,
    batch_id: int | None = None,
    db: Session = Depends(get_transaction_db),
) -> list[dict[str, Any]]:
    return repositories.list_filter_confidence_results(
        db,
        limit=max(1, min(limit, 500)),
        offset=max(0, offset),
        batch_id=batch_id,
    )


@router.get("/confidence-results/count")
def count_confidence_results(
    batch_id: int | None = None,
    db: Session = Depends(get_transaction_db),
) -> dict[str, int]:
    return {"total": repositories.count_filter_confidence_results(db, batch_id=batch_id)}


@router.post("/confidence-results/{confidence_result_id}/retry")
def retry_confidence_result(confidence_result_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return repositories.retry_filter_confidence_result(db, confidence_result_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/confidence-results/{confidence_result_id}/force-deep-analysis")
def force_deep_analysis(confidence_result_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return workflow_service.force_deep_analysis_for_confidence_result(db, confidence_result_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/confidence-results/{confidence_result_id}")
def delete_confidence_result(confidence_result_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return repositories.delete_filter_confidence_result(db, confidence_result_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@router.post("/grouping-batches/run-now")
def run_grouping_now(db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        prepared_batches = workflow_service.prepare_manual_grouping_batches(db)
        if repositories.has_active_remote_analysis_script_run(db, script_names=["grouping"]):
            raise HTTPException(status_code=409, detail="Grouping is already dispatching or running.")
        pending_batches = repositories.list_pending_grouping_batches(db, limit=20)
        for batch in pending_batches:
            batch_id = int(batch["id"])
            if not repositories.claim_grouping_batch_for_dispatch(db, batch_id):
                continue
            try:
                job = workflow_service.build_grouping_analysis_job_from_batch(db, batch_id)
                result = workflow_service.start_grouping_analysis_job(job)
            except Exception as exc:
                repositories.update_grouping_batch(
                    db,
                    batch_id,
                    {
                        "status": "issue",
                        "issue_reason": str(exc),
                    },
                )
                raise
            return {
                "ok": True,
                "prepared_count": len(prepared_batches),
                "dispatched": True,
                "batch_id": batch_id,
                "script_run_id": result.script_run_id,
                "runner_job_id": result.runner_job_id,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "prepared_count": len(prepared_batches),
        "dispatched": False,
        "message": "No retrieved, ungrouped trigger frames are ready for manual grouping.",
    }


@router.post("/grouping-batches/run-range")
def run_grouping_range(payload: GroupingRangePayload, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        prepared_batches = workflow_service.prepare_manual_grouping_batches_for_range(
            db,
            start_time=payload.start_time,
            end_time=payload.end_time,
            location_id=payload.location_id,
        )
        if repositories.has_active_remote_analysis_script_run(db, script_names=["grouping"]):
            raise HTTPException(status_code=409, detail="Grouping is already dispatching or running.")
        pending_batches = repositories.list_pending_grouping_batches(db, limit=20)
        for batch in pending_batches:
            batch_id = int(batch["id"])
            if not repositories.claim_grouping_batch_for_dispatch(db, batch_id):
                continue
            try:
                job = workflow_service.build_grouping_analysis_job_from_batch(db, batch_id)
                result = workflow_service.start_grouping_analysis_job(job)
            except Exception as exc:
                repositories.update_grouping_batch(
                    db,
                    batch_id,
                    {
                        "status": "issue",
                        "issue_reason": str(exc),
                    },
                )
                raise
            return {
                "ok": True,
                "prepared_count": len(prepared_batches),
                "dispatched": True,
                "batch_id": batch_id,
                "script_run_id": result.script_run_id,
                "runner_job_id": result.runner_job_id,
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "prepared_count": len(prepared_batches),
        "dispatched": False,
        "message": "No retrieved, ungrouped trigger frames were found inside that time range.",
    }


@router.post("/grouping-batches/self-group")
def create_self_grouping(payload: SelfGroupingPayload, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return workflow_service.create_self_grouping_batch(
            db,
            location_id=payload.location_id,
            start_time=payload.start_time,
            end_time=payload.end_time,
            groups=[group.model_dump() for group in payload.groups],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/grouping-batches/{batch_id}/retry")
def retry_grouping_batch(batch_id: int, db: Session = Depends(get_transaction_db)) -> dict[str, Any]:
    try:
        return workflow_service.retry_grouping_batch_now(db, batch_id=batch_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
