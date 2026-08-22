from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import repositories
from ..db import get_transaction_db
from ..schemas import TriggerFrameAssetListItem

router = APIRouter(prefix="/api/v1/frames", tags=["frames"])


@router.get("/assets", response_model=list[TriggerFrameAssetListItem])
def list_frame_assets(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_transaction_db),
) -> list[TriggerFrameAssetListItem]:
    rows = repositories.list_trigger_frame_assets(db, limit=limit)
    return [TriggerFrameAssetListItem(**row) for row in rows]
