import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..db import get_transaction_db
from ..services import workflow_service


router = APIRouter(prefix="/api/v1/runpod", tags=["runpod"])

# Serializes webhook processing to exactly one at a time, in arrival order -
# without this, run_in_threadpool would let several arrive-close-together
# webhooks each grab their own thread and run their (now up to ~12-minute-
# bounded) vision calls fully in parallel, multiplying the CPU/network load
# on a single-vCPU host instead of spreading it out. A queued webhook still
# finishes eventually; several genuinely concurrent ones competing for the
# same limited CPU could each take longer AND worsen the exact contention
# this host is already tight on.
_webhook_lock = asyncio.Lock()


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
        # while this one is still in progress; the lock above then makes
        # sure only one such call is actually executing at a time, queuing
        # the rest instead of running them all concurrently.
        async with _webhook_lock:
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
