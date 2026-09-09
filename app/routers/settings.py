from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_transaction_db
from .. import repositories
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


class FramesPerTriggerRequest(BaseModel):
    frames_per_trigger: int


class KioskAnalysisDisabledRequest(BaseModel):
    disabled: bool


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
