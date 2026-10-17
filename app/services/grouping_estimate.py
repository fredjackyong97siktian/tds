"""Read-only grouping previews. No model calls, downloads or batch writes."""

import json
import math
from datetime import timedelta
from statistics import median
from typing import Any

from .. import repositories
from . import workflow_service as workflow


def _chunk_count(triggers: list[dict[str, Any]], max_images: int) -> int:
    count = images = 0
    for trigger in triggers:
        size = len(trigger.get("frames") or [])
        if not size:
            continue
        if not images or images + size > max_images:
            count += 1
            images = 0
        images += size
    return count


def _timing(db: Any, stage: str) -> dict[str, Any]:
    samples = repositories.list_grouping_timing_samples(db, workflow._grouping_stage_label(db, stage))
    unit_key = "entry_count" if stage == "adjacent" else "chunk_count"
    per_unit = []
    for sample in samples:
        try:
            seconds, units = float(sample["seconds"]), int(sample.get(unit_key) or 0)
        except (ValueError, TypeError, KeyError):
            continue
        if seconds > 0 and units > 0:
            per_unit.append(seconds / units)
    return {
        "sample_runs": len(per_unit),
        # Includes normal verification, request spacing and retries in past runs.
        "seconds_per_unit": median(per_unit) if per_unit else None,
    }


def estimate_inputs(db: Any, batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    framed_batches = [[item for item in batch if item.get("frames")] for batch in batches]
    checks = sum(len(workflow._plan_grouping_adjacent_checks(batch)) for batch in framed_batches)
    direct_enabled = workflow.settings.grouping_direct_enabled
    chunks = sum(_chunk_count(batch, workflow._grouping_gemini_max_images_per_request()) for batch in framed_batches) if direct_enabled else 0
    adjacent = _timing(db, "adjacent")
    direct = _timing(db, "direct")
    required = [(checks, adjacent), (chunks, direct)]
    has_history = all(not units or timing["seconds_per_unit"] is not None for units, timing in required)
    seconds = sum(units * (timing["seconds_per_unit"] or 60) for units, timing in required)
    return {
        "batch_count": sum(bool(batch) for batch in framed_batches),
        "trigger_count": sum(len(batch) for batch in framed_batches),
        "image_count": sum(len(item["frames"]) for batch in framed_batches for item in batch),
        "adjacent_checks": checks,
        "direct_chunks_estimate": chunks,
        "direct_enabled": direct_enabled,
        "estimated_seconds_min": math.ceil(seconds * 0.7),
        "estimated_seconds_max": math.ceil(seconds * (1.6 if has_history else 2)),
        "timing_basis": "recent_runs" if has_history else "rough_fallback",
        "adjacent_sample_runs": adjacent["sample_runs"],
        "direct_sample_runs": direct["sample_runs"],
        "note": "Direct chunks assume no earlier matches. Matching can reduce them; carried entries, verification, retries and provider delays can increase the wait. Text/repair and theft-confidence analysis are not included. This is not a maximum or a reservation.",
    }


def preview_grouping(db: Any, *, mode: str, batch_id: int | None = None,
                     location_id: int | None = None, start_time: str | None = None,
                     end_time: str | None = None) -> dict[str, Any]:
    if mode == "retry":
        if batch_id is None:
            raise ValueError("Batch ID is required.")
        batch = repositories.get_grouping_batch(db, batch_id)
        triggers = repositories.list_grouping_estimate_triggers(db, batch_id)
        for item in triggers:
            payload = item.get("frame_payload") or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                payload = {}
            item["frames"] = workflow._first_trigger_frame_payload(payload.get("frames") or [])
        by_id = {int(item["trigger_id"]): item for item in triggers}
        if batch.get("window_start") is not None and batch.get("window_end") is not None:
            # Mirror execution's live snapshot refresh, but never upsert items.
            load = (repositories.list_trigger_frame_assets_for_window if triggers
                    else repositories.list_ready_trigger_frame_assets_in_window)
            live_rows = load(db, location_id=int(batch["location_id"]),
                             window_start=batch["window_start"], window_end=batch["window_end"])
            for row in live_rows:
                if (row.get("trigger_source") == workflow._MOTION_SENSOR_TRIGGER_SOURCE
                        and row.get("motion_person_present") is not None and not bool(row["motion_person_present"])):
                    continue
                frames = workflow._first_trigger_frame_payload(workflow._frame_urls_from_trigger_frame_asset(row))
                if frames:
                    by_id[int(row["trigger_id"])] = {**row, "frames": frames}
        result = estimate_inputs(db, [sorted(by_id.values(), key=lambda item: workflow._coerce_datetime_value(item.get("trigger_time")) or workflow.datetime.min)])
        result["note"] += " Retry inputs include currently available frames in this batch's window; retrieval still in progress can change them."
        return result

    start = workflow._coerce_datetime_value(start_time)
    end = workflow._coerce_datetime_value(end_time)
    if mode == "range" and (start is None or end is None or end <= start):
        raise ValueError("A valid start and end time are required.")
    batches = []
    for location in repositories.list_locations(db):
        current_id = int(location["id"])
        if location_id is not None and current_id != location_id:
            continue
        rows = repositories.list_manual_grouping_ready_trigger_frame_assets(
            db, location_id=current_id, limit=1000 if mode == "range" else 200,
        )
        # The real preparation filter may call AI and cache person-presence.
        # A preview must not spend money or mutate triggers: apply cached
        # exclusions only, leaving unchecked motion triggers provisional.
        rows = [row for row in rows if row.get("trigger_source") != workflow._MOTION_SENSOR_TRIGGER_SOURCE
                or row.get("motion_person_present") is None or bool(row["motion_person_present"])]
        if not rows:
            continue
        times = [workflow._coerce_datetime_value(row.get("trigger_time")) for row in rows]
        valid_times = [value for value in times if value is not None]
        if not valid_times:
            continue
        window_start = start if mode == "range" else max(valid_times) - timedelta(hours=1)
        window_end = end if mode == "range" else max(valid_times) + timedelta(seconds=1)
        selected = []
        for row, timestamp in zip(rows, times):
            if timestamp is None or not (window_start <= timestamp < window_end):
                continue
            selected.append({**row, "frames": workflow._first_trigger_frame_payload(workflow._frame_urls_from_trigger_frame_asset(row))})
        batches.append(selected)
    result = estimate_inputs(db, batches)
    result["note"] += " Unchecked motion triggers are provisional and may be excluded after person detection."
    return result
