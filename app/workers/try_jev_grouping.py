"""Run one opt-in Jev experiment, without changing any groups or sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from time import monotonic

from .. import repositories
from ..config import settings
from ..db import TransactionalSessionLocal
from ..services import jev_matching, workflow_service


def load_trigger(db, trigger_id: int, frame_count: int) -> dict:
    trigger = repositories.get_trigger(db, trigger_id)
    assets = repositories.list_trigger_frame_assets(db, limit=1, trigger_id=trigger_id)
    frames = (assets[0].get("frames") or []) if assets else []
    frames = [frame for frame in frames if frame.get("status") == "ok" and frame.get("image_url")]
    return {
        "trigger_id": trigger_id, "trigger_time": trigger["trigger_time"], "location_id": trigger["location_id"],
        "phone_entry_id": trigger.get("phone_entry_id"), "credit_card_entry_id": trigger.get("credit_card_entry_id"),
        "frames": sorted(frames, key=lambda frame: int(frame.get("frame_index") or frame.get("index") or 0))[:frame_count],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=int, required=True, help="Known entry trigger ID")
    parser.add_argument("--candidates", type=int, nargs="+", required=True, help="One or two later candidate exit IDs")
    parser.add_argument("--frames", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--dry-run", action="store_true", help="Validate triggers and show the plan without downloading images, calling AI or writing script_run")
    args = parser.parse_args()
    if not 1 <= len(args.candidates) <= 2:
        parser.error("Use one or two candidate exits.")
    api_key = str(settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not args.dry_run and not api_key:
        parser.error("No OpenRouter key configured. Set THEFT_API_OPENROUTER_API_KEY or OPENROUTER_API_KEY in the server environment.")
    db = TransactionalSessionLocal()
    run_id = None
    call_index = None
    response_logged = False
    started = monotonic()
    try:
        entry = load_trigger(db, args.entry, args.frames)
        candidates = [load_trigger(db, value, args.frames) for value in args.candidates]
        if entry.get("phone_entry_id") is None and entry.get("credit_card_entry_id") is None:
            raise ValueError("Use a confirmed entry with phone_entry_id or credit_card_entry_id for this first experiment.")
        if args.dry_run:
            _, urls = jev_matching.build_request(entry, candidates, frames_per_trigger=args.frames, image_loader=lambda url: "data:image/jpeg;base64,DRY_RUN")
            print(json.dumps({"model": jev_matching.MODEL, "entry": args.entry, "candidates": args.candidates, "image_count": len(urls), "test_only": True, "dry_run": True}))
            return 0
        run_id = repositories.create_script_run_started(
            db, session_id=None, trigger_id=args.entry, script_name="grouping", model_name="openrouter-jev-1.14_matching_test",
            command="python -m app.workers.try_jev_grouping", runner_payload={
                "test_only": True, "model": jev_matching.MODEL, "entry_trigger_id": args.entry,
                "candidate_trigger_ids": args.candidates, "frames_per_trigger": args.frames,
            },
        )
        body, urls = jev_matching.build_request(entry, candidates, frames_per_trigger=args.frames)
        call_index = repositories.append_script_run_call(db, run_id, {"outcome": "in_progress", "request": {
            "model": jev_matching.MODEL, "image_urls": urls, "questions": body["questions"],
        }})
        response, meta = jev_matching.call_openrouter(body, api_key=api_key)
        repositories.replace_script_run_call(db, run_id, call_index, {
            "outcome": "success", "response": response, "meta": meta,
            "request": {"model": jev_matching.MODEL, "image_urls": urls, "questions": body["questions"]},
        })
        response_logged = True
        # Record actual spend even if answer validation subsequently fails.
        workflow_service._record_openrouter_cost(db, run_id, meta)
        result = jev_matching.interpret_response(response, entry_id=args.entry, candidate_ids=args.candidates)
        result.update(script_run_id=run_id, model=jev_matching.MODEL, image_count=len(urls),
                      api_seconds=meta["duration_seconds"], total_seconds=round(monotonic() - started, 3),
                      cost_usd=meta["usage"]["actual_cost_usd"])
        repositories.finish_script_run(db, run_id, status="success", stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, {"result": result}), stderr_log="")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        if run_id is not None:
            if call_index is not None and not response_logged:
                repositories.replace_script_run_call(db, run_id, call_index, {
                    "outcome": "error", "error": message,
                    "duration_seconds": round(monotonic() - started, 3),
                    "request": {"model": jev_matching.MODEL, "image_urls": urls, "questions": body["questions"]},
                })
            repositories.finish_script_run(db, run_id, status="failed", stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, {}), stderr_log=message)
        print(json.dumps({"error": message, "script_run_id": run_id, "test_only": True}), file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
