from fastapi import APIRouter, Depends, HTTPException
from typing import Literal
import os
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_transaction_db
from .. import repositories
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class FramesPerTriggerRequest(BaseModel):
    frames_per_trigger: int


class GroupingExpandDurationRequest(BaseModel):
    duration_minutes: int = Field(ge=1, le=120, strict=True)


@router.get("/grouping-expand-duration")
def get_grouping_expand_duration(db: Session = Depends(get_transaction_db)) -> dict:
    from ..services import grouping_expand
    return {"duration_minutes": grouping_expand.duration_minutes(db), "default": grouping_expand.DEFAULT_MINUTES}


@router.put("/grouping-expand-duration")
def update_grouping_expand_duration(payload: GroupingExpandDurationRequest,
                                   db: Session = Depends(get_transaction_db)) -> dict:
    from ..services import grouping_expand
    repositories.set_app_setting(db, grouping_expand.SETTING_KEY, str(payload.duration_minutes))
    return {"ok": True, "duration_minutes": payload.duration_minutes}


class KioskAnalysisDisabledRequest(BaseModel):
    disabled: bool


class JevStageProviderRequest(BaseModel):
    provider: Literal["jev", "inherit"]


@router.get("/jev-stage-providers")
def get_jev_stage_providers(db: Session = Depends(get_transaction_db)) -> dict:
    return {
        "adjacent": workflow_service._current_jev_stage_provider(db, "adjacent"),
        "carry": workflow_service._current_jev_stage_provider(db, "carry"),
        "presence": workflow_service._current_jev_stage_provider(db, "presence"),
        "openrouter_key_configured": bool(str(settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()),
        "model": "typesafe/jev-1.14",
        "minimum_choice_confidence": workflow_service.jev_pipeline.MIN_CHOICE_CONFIDENCE,
    }


@router.put("/jev-stage-providers/{stage}")
def update_jev_stage_provider(stage: Literal["adjacent", "carry", "presence"], payload: JevStageProviderRequest,
                              db: Session = Depends(get_transaction_db)) -> dict:
    repositories.set_app_setting(db, f"{stage}_provider", payload.provider)
    return {"ok": True, "stage": stage, "provider": payload.provider}


@router.get("/grouping-frames-per-trigger")
def get_grouping_frames_per_trigger(db: Session = Depends(get_transaction_db)) -> dict:
    return {
        "frames_per_trigger": workflow_service._grouping_frames_per_trigger(db),  # noqa: SLF001
        "default": settings.grouping_gemini_frames_per_trigger,
    }


@router.put("/grouping-frames-per-trigger")
def update_grouping_frames_per_trigger(
    payload: FramesPerTriggerRequest,
    db: Session = Depends(get_transaction_db),
) -> dict:
    if not (1 <= payload.frames_per_trigger <= 20):
        raise HTTPException(status_code=400, detail="frames_per_trigger must be between 1 and 20.")
    repositories.set_app_setting(
        db,
        workflow_service._GROUPING_FRAMES_PER_TRIGGER_APP_SETTING_KEY,  # noqa: SLF001
        str(payload.frames_per_trigger),
    )
    return {"ok": True, "frames_per_trigger": payload.frames_per_trigger}


@router.get("/kiosk-analysis-disabled")
def get_kiosk_analysis_disabled(db: Session = Depends(get_transaction_db)) -> dict:
    return {"disabled": workflow_service._is_kiosk_analysis_disabled(db)}  # noqa: SLF001


@router.put("/kiosk-analysis-disabled")
def update_kiosk_analysis_disabled(
    payload: KioskAnalysisDisabledRequest,
    db: Session = Depends(get_transaction_db),
) -> dict:
    repositories.set_app_setting(
        db,
        workflow_service._KIOSK_ANALYSIS_DISABLED_APP_SETTING_KEY,  # noqa: SLF001
        "true" if payload.disabled else "false",
    )
    return {"ok": True, "disabled": payload.disabled}
