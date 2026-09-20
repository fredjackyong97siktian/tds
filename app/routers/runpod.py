from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

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
        # process_runpod_webhook is a plain blocking function - for the
        # kiosk kind, it can synchronously call all the way down into a
        # Gemini/OpenRouter vision call that's now bounded at up to ~12
        # minutes (3 retries x the 240s hard deadline) rather than instant.
        # Calling it directly here, unawaited, would run it straight on this
        # single asyncio event loop thread - blocking EVERY other request
        # this whole process is handling, website included, for that entire
        # duration. Confirmed live: the site went fully unresponsive (not
        # just slow) during exactly this. run_in_threadpool moves the actual
        # blocking work off the event loop so other requests keep flowing
        # while this one is still in progress.
        return await run_in_threadpool(
            workflow_service.process_runpod_webhook,
            db,
            kind=kind,
            body=payload,
            token=token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
