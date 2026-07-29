from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from ..db import get_transaction_db
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/runpod", tags=["runpod"])


@router.post("/webhooks/{kind}")
async def runpod_webhook(
    kind: str,
    request: Request,
    token: str | None = Query(default=None),
    db: Session = Depends(get_transaction_db),
) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    try:
        return workflow_service.process_runpod_webhook(
            db,
            kind=kind,
            body=payload,
            token=token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
