"""Typed Jev adapters for adjacent matching and image-based carry assessment."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from . import jev_matching as jev

# Separate from MiMo's similarity cutoff. These trial thresholds are not yet
# calibrated on the store's CCTV; preserve the complete distributions for review.
MIN_CHOICE_CONFIDENCE = 0.5


def encode_sequences(triggers: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    frames_by_trigger = []
    urls = []
    for trigger in triggers:
        frames = sorted(trigger.get("frames") or [], key=lambda item: int(item.get("index") or item.get("frame_index") or 0))
        frames = [frame for frame in frames if frame.get("image_url") and frame.get("status", "ok") == "ok"]
        if not frames:
            raise ValueError(f"Trigger {trigger['trigger_id']} has no usable frames.")
        frames_by_trigger.append(frames)
        for frame in frames:
            if frame["image_url"] not in urls:
                urls.append(frame["image_url"])
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(urls)))) as pool:
        encoded = dict(zip(urls, pool.map(jev.download_image, urls)))
    sequences = []
    for trigger, frames in zip(triggers, frames_by_trigger):
        sequences.append({
            "trigger_id": int(trigger["trigger_id"]), "trigger_time": str(trigger.get("trigger_time") or ""),
            "frames": [{"frame_index": frame.get("index") or frame.get("frame_index"),
                        "sample_time": str(frame.get("sample_time") or ""),
                        "image": {"type": "image_url", "image_url": {"url": encoded[frame["image_url"]]}}}
                       for frame in frames],
        })
    return sequences, urls


def read_choice(response: Mapping[str, Any], question: str, options: set[str]) -> dict[str, Any]:
    answer = (response.get("answers") or {}).get(question) or {}
    if answer.get("type") != "choice" or answer.get("choice") not in options:
        raise ValueError(f"Invalid Jev answer for {question}.")
    probabilities = answer.get("probabilities") or {}
    if set(probabilities) != options:
        raise ValueError(f"Incomplete Jev probabilities for {question}.")
    probabilities = {key: jev._probability(value) for key, value in probabilities.items()}
    if abs(sum(probabilities.values()) - 1) > 0.02:
        raise ValueError(f"Invalid Jev probability sum for {question}.")
    if probabilities[answer["choice"]] + 0.0001 < max(probabilities.values()):
        raise ValueError(f"Jev choice does not agree with its probabilities for {question}.")
    return {"choice": answer["choice"], "confidence": jev._probability(answer.get("confidence")),
            "probabilities": probabilities}


def adjacent_request(entry: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    if not 1 <= len(candidates) <= 2:
        raise ValueError("Adjacent Jev matching expects one or two candidates.")
    ids = [int(item["trigger_id"]) for item in [entry, *candidates]]
    if len(set(ids)) != len(ids):
        raise ValueError("Adjacent trigger IDs must be unique.")
    entry_time = datetime.fromisoformat(str(entry["trigger_time"]))
    for candidate in candidates:
        if datetime.fromisoformat(str(candidate["trigger_time"])) <= entry_time:
            raise ValueError("Exit candidates must be later than entry.")
    sequences, urls = encode_sequences([entry, *candidates])
    options = {
        f"candidate_{index + 1}": f"Candidate {index + 1}, trigger {candidate['trigger_id']}, contains this same customer exiting; the other candidate does not."
        for index, candidate in enumerate(candidates)
    }
    if len(candidates) == 2:
        options["both"] = "BOTH candidate sequences contain this same customer exiting: duplicate/overlapping exit triggers or repeated views of this exit."
    options["none"] = "Neither candidate clearly shows this entry customer exiting, or evidence is insufficient to identify a match."
    prompt = (
        "Compare this one confirmed entry customer against their nearby exit candidates. Use each sequence in frame order. "
        "Identify the entry customer from entry images alone, comparing visible clothing, build, hair, footwear and carried items. "
        "Carried items may change after shopping. Never use timing alone to establish identity or invent hidden details. "
        "A candidate can show multiple people: compare the entry customer against EACH visible person, including background people. "
        "A candidate matches if any visible person is the same physical customer and the sequence shows that customer leaving. "
        "There may be two matching exit triggers; choose both when both show that customer's exit, not merely because both have people. "
        "If the customer's direction is visibly entering again, that is not a matching exit. Use doorway movement across the frames. "
        "Choose none rather than guessing. Which option describes the matched exit candidate(s)?"
    )
    return {"model": jev.MODEL, "state": {"entry": sequences[0], "candidates": sequences[1:]},
            "questions": {"matched_exits": {"type": "choice", "instructions": prompt, "criteria": options}}}, urls


def adjacent_result(response: Mapping[str, Any], candidate_count: int) -> dict[str, Any]:
    options = {f"candidate_{index + 1}" for index in range(candidate_count)} | {"none"}
    if candidate_count == 2:
        options.add("both")
    answer = read_choice(response, "matched_exits", options)
    selected = answer["choice"]
    matches = []
    if selected == "both":
        matches = [1, 2]
    elif selected != "none":
        matches = [int(selected.split("_")[1])]
    return {
        "matched_candidates": matches,
        "matched_candidate": matches[0] if len(matches) == 1 else None,
        "confidence": answer["confidence"], "minimum_confidence": None,
        "reason": f"Jev selected {selected}; choice confidence {answer['confidence']:.2f}.",
        "jev_decision": answer, "source": "openrouter_jev_adjacent",
        "entry_carry": None, "exit_carry": None,
    }


def carry_request(entries: list[Mapping[str, Any]], exits: list[Mapping[str, Any]],
                  transactions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    if not entries or not exits:
        raise ValueError("Carry assessment requires both entry and exit frames.")
    entry_ids = {int(row["trigger_id"]) for row in entries}
    if entry_ids & {int(row["trigger_id"]) for row in exits}:
        raise ValueError("A carry trigger cannot be both entry and exit.")
    sequences, urls = encode_sequences([*entries, *exits])
    state = {
        "entry_sequences": sequences[:len(entries)], "exit_sequences": sequences[len(entries):],
        "paid_transactions_in_visit_window": transactions,
        "receipt_attribution": "time_window_only_not_confirmed_customer_ownership",
        "receipt_query_complete": True,
    }
    instructions = (
        "Assess only the grouped customer's visible carrying state, comparing all entry sequences, all exit sequences and receipt item details. "
        "Ignore unrelated bystanders and do not treat repeated frames or two exit triggers as two sets of purchases/items. "
        "Suspicious criteria: a visible newly acquired carried load not reasonably explained by purchases; "
        "a bag held folded/scrunched at entry that visibly becomes larger/fuller/loaded at exit; "
        "or a visibly yellow bag/plastic bag/carried container belonging to this customer at entry or exit (store policy override). "
        "Yellow objects in the background or yellow clothing are not carried yellow containers. "
        "A pre-existing bag alone, a large bag alone or no paid receipt alone is not sufficient evidence of theft. "
        "No paid receipts plus clearly acquired store goods can be suspicious; delivery of a pre-existing box into the store is not. "
        "Use item names, quantities, likely sizes and paid values; monetary values are in the source table's units, not necessarily RM. "
        "Do not guess concealed contents or exact item counts from opaque bags. Receipts are window candidates, not confirmed customer ownership. "
        "If receipt ownership is needed to distinguish suspicious from reasonable but unclear, choose insufficient_evidence. "
        "Reasonable requires clear evidence consistent with the customer's existing belongings or attributable purchases. "
        "If visibility, customer identity, entry/exit comparison or required transaction attribution is insufficient, choose insufficient_evidence. "
        "Suspicious is a screening flag for deeper analysis, not a confirmed theft finding. Which status applies?"
    )
    reasons = {
        "yellow_bag": "Clearly visible carried yellow bag/container: store screening policy.",
        "bag_enlarged": "An entry bag visibly becomes fuller/larger/loaded at exit.",
        "unexplained_carry_change": "Clearly acquired carried goods are not reasonably explained by purchases.",
        "consistent_with_entry": "Clear carrying state is explained by pre-existing belongings or delivery into store.",
        "consistent_with_receipt": "Clear carrying state is reasonably explained by attributable paid items.",
        "unclear_images": "Customer carrying state cannot be compared clearly from the images.",
        "uncertain_receipt_assignment": "Receipt ownership or relevant transaction evidence is uncertain.",
        "other": "None of the listed reasons is supported clearly enough.",
    }
    return {"model": jev.MODEL, "state": state, "questions": {
        "carry_status": {"type": "choice", "instructions": instructions, "criteria": {
            "suspicious": "Visible evidence clearly meets at least one configured suspicious carrying criterion.",
            "reasonable": "Enough evidence to judge; visible carrying state is reasonably explained and no suspicious criterion is met.",
            "insufficient_evidence": "Not enough visible or transaction-attribution evidence to decide either way. Do not clear the customer or flag theft by guessing.",
        }},
        "carry_reason": {"type": "choice", "instructions": instructions + " Independently select the best-supported reason from the observations; do not assume any other question's answer.",
                         "criteria": reasons},
    }}, urls


CARRY_REASONS = {"yellow_bag", "bag_enlarged", "unexplained_carry_change", "consistent_with_entry",
                 "consistent_with_receipt", "unclear_images", "uncertain_receipt_assignment", "other"}


def carry_result(response: Mapping[str, Any]) -> dict[str, Any]:
    status = read_choice(response, "carry_status", {"suspicious", "reasonable", "insufficient_evidence"})
    reason = read_choice(response, "carry_reason", CARRY_REASONS)
    effective = status["choice"] if status["confidence"] >= MIN_CHOICE_CONFIDENCE else "insufficient_evidence"
    allowed = {
        "suspicious": {"yellow_bag", "bag_enlarged", "unexplained_carry_change"},
        "reasonable": {"consistent_with_entry", "consistent_with_receipt"},
        "insufficient_evidence": {"unclear_images", "uncertain_receipt_assignment"},
    }
    reason_code = reason["choice"] if reason["choice"] in allowed[effective] and reason["confidence"] >= MIN_CHOICE_CONFIDENCE else f"carry_{effective}"
    messages = {
        "suspicious": "Jev found a suspicious carrying state under the configured screening criteria; deeper analysis is required, not confirmed theft.",
        "reasonable": "Jev found enough evidence for a reasonable carrying state under the configured criteria. This clears only the carry factor.",
        "insufficient_evidence": "Carry evidence is unclear. This is not a confirmed hit or an all-clear result, and does not trigger the carry factor.",
    }
    return {"status": effective, "hit": effective == "suspicious", "insufficient_evidence": effective == "insufficient_evidence",
            "reason": reason_code, "message": messages[effective], "score": status["probabilities"]["suspicious"],
            "confidence": status["confidence"], "minimum_confidence": MIN_CHOICE_CONFIDENCE,
            "source": "openrouter_jev_carry", "jev_decision": status, "jev_reason": reason}


def logged_call(db: Any, script_run_id: int, body: dict[str, Any], urls: list[str], *, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # Late import avoids a workflow -> adapter -> workflow import cycle.
    from . import workflow_service as workflow
    import json
    import os

    api_key = str(workflow.settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OpenRouter key is missing. Configure THEFT_API_OPENROUTER_API_KEY to use Jev.")
    def dispatch():
        response, meta = jev.call_openrouter(body, api_key=api_key, timeout_seconds=workflow.settings.openrouter_timeout_seconds)
        return response, {**meta, "image_urls": urls, "image_count": len(urls), "questions": body["questions"]}
    response, meta = workflow._call_vision_with_retry(
        dispatch, description=description, prompt=json.dumps(body["questions"]), image_urls=urls,
        db=db, script_run_id=script_run_id,
    )
    # Keep DB writes on the caller thread, and charge before typed validation.
    workflow._record_openrouter_cost(db, script_run_id, meta)
    return response, meta


def verify_direct_matches(db: Any, script_run_id: int, candidates: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    from . import workflow_service as workflow

    triggers: dict[int, dict[str, Any]] = {}
    pairs: dict[str, tuple[int, int, int]] = {}
    questions = {}
    for candidate in candidates:
        number = int(candidate["match_number"])
        entry = candidate["entry"]
        exits = candidate["exit_triggers"]
        if not exits:
            raise ValueError("Direct verification needs at least one exit per match.")
        for trigger in [entry, *exits]:
            trigger_id = int(trigger["trigger_id"])
            triggers[trigger_id] = {**trigger, "frames": workflow._select_verification_frames(trigger)}
        for index, exit_trigger in enumerate(exits):
            entry_id, exit_id = int(entry["trigger_id"]), int(exit_trigger["trigger_id"])
            if entry_id == exit_id:
                raise ValueError("An entry cannot verify against itself as an exit.")
            entry_time = datetime.fromisoformat(str(entry["trigger_time"]))
            if datetime.fromisoformat(str(exit_trigger["trigger_time"])) <= entry_time:
                raise ValueError("Verification exits must follow their entry.")
            key = f"match_{number}_exit_{index}"
            if key in pairs:
                raise ValueError("Duplicate direct verification match number.")
            pairs[key] = (number, entry_id, exit_id)
            questions[key] = {
                "type": "noul",
                "instructions": f"Do `triggers.t{entry_id}` and `triggers.t{exit_id}` show the SAME physical customer "
                "entering and subsequently exiting this store? Judge only these two ordered frame sequences, "
                "never another match's images. Identify the actor interacting with the door or passing through "
                "the doorway, not an unrelated background passerby. Compare visible clothing, patterns, "
                "pants, shoes, hair and build. Carried purchases may change. Do not assume a proposed match "
                "is correct and do not invent hidden details. Unclear identity or direction is uncertain, "
                "not a confirmed match.",
                "criteria": {"true": "Visible identity and direction support the same customer entering then exiting.",
                             "false": "Different customer or inconsistent entry/exit direction."},
            }
    sequences, urls = encode_sequences(list(triggers.values()))
    body = {"model": jev.MODEL, "state": {"triggers": {f"t{item['trigger_id']}": item for item in sequences}},
            "questions": questions}
    response, meta = logged_call(db, script_run_id, body, urls, description="Jev direct-match verification")
    results: dict[int, dict[str, Any]] = {}
    for key, (number, entry_id, exit_id) in pairs.items():
        answer = (response.get("answers") or {}).get(key) or {}
        if answer.get("type") != "noul":
            raise ValueError(f"Jev returned no valid verification for {key}.")
        probability = jev._probability(answer.get("noul"))
        result = results.setdefault(number, {"confidence": 1.0, "error": None,
                                             "source": "openrouter_jev_direct_verification", "exit_checks": []})
        result["confidence"] = min(result["confidence"], probability)
        result["exit_checks"].append({"entry_trigger_id": entry_id, "exit_trigger_id": exit_id,
                                     "same_customer_probability": probability})
    return results, meta


def check_person_presence(db: Any, script_run_id: int, urls: list[str]) -> dict[str, Any]:
    sequences, urls = encode_sequences([{"trigger_id": 0, "frames": [
        {"frame_index": index + 1, "image_url": url} for index, url in enumerate(urls)
    ]}])
    criteria = {
        "present": "At least one frame clearly shows a real human physically present in front of the store door, in the immediate doorway/entrance area.",
        "absent": "No real human is physically in front of the store door. Any visible people are only farther away on the sidewalk, road, or in the background; objects, pets, shadows, and lighting changes do not count.",
        "unclear": "The store door area cannot be identified reliably, or visibility, blur, or occlusion prevents determining whether a real human is there.",
    }
    body = {"model": jev.MODEL, "state": {"trigger": sequences[0]}, "questions": {
        "person_presence": {"type": "choice", "instructions":
            "Inspect all ordered trigger frames. Is there any real human physically present in front of "
            "the door of the store, in the immediate doorway/entrance area, in any frame? "
            "A person need not open the door or enter the store. Do not count people visible only farther "
            "away on the sidewalk, road, or in the background. Do not count posters, screen images, "
            "reflections of posters, carts, pets, shadows, or lighting changes as a real human.",
            "criteria": criteria},
    }}
    response, _ = logged_call(db, script_run_id, body, urls, description="Jev sensor person presence")
    decision = read_choice(response, "person_presence", set(criteria))
    uncertain = decision["choice"] == "unclear" or decision["confidence"] < MIN_CHOICE_CONFIDENCE
    return {"person_present": uncertain or decision["choice"] != "absent",
            "confidence": decision["confidence"], "uncertain": uncertain,
            "jev_decision": decision, "source": "openrouter_jev_person_presence"}


def verify_adjacent(db: Any, script_run_id: int, entry_groups: list[dict[str, Any]]):
    if len(entry_groups) != 1:
        raise ValueError("Each Jev adjacent call must contain exactly one independent entry group.")
    group = entry_groups[0]
    body, urls = adjacent_request(group["entry"], group["candidates"])
    response, meta = logged_call(db, script_run_id, body, urls, description="Jev adjacent matching")
    result = adjacent_result(response, len(group["candidates"]))
    return {int(group["group_number"]): result}, {}, {}, meta


def evaluate_carry(db: Any, *, batch_id: int, group_key: str, location_id: int,
                   group: Mapping[str, Any], transactions: list[dict[str, Any]]) -> dict[str, Any]:
    from .. import repositories
    from . import workflow_service as workflow

    entry_ids = [int(value) for value in (workflow._read_group_value(group, "entry") or [])]
    exit_ids = [int(value) for value in (workflow._read_group_value(group, "exit") or [])]
    run_id = repositories.create_script_run_started(
        db, session_id=None, trigger_id=entry_ids[0] if entry_ids else None,
        script_name="carry_confidence", model_name="openrouter-jev-1.14_carry_confidence",
        command=workflow.SCRIPT_RUN_COMMAND_REDACTED,
        runner_payload={"batch_id": batch_id, "group_key": group_key, "location_id": location_id,
                        "entry_trigger_ids": entry_ids, "exit_trigger_ids": exit_ids, "model": jev.MODEL},
    )
    try:
        triggers = []
        for trigger_id in [*entry_ids, *exit_ids]:
            trigger = repositories.get_trigger(db, trigger_id)
            if int(trigger["location_id"]) != int(location_id):
                raise ValueError("Carry trigger belongs to another location.")
            assets = repositories.list_trigger_frame_assets(db, limit=1, trigger_id=trigger_id)
            frames = [frame for frame in ((assets[0].get("frames") or []) if assets else [])
                      if frame.get("status") == "ok" and frame.get("image_url")]
            frames = sorted(frames, key=lambda frame: int(frame.get("frame_index") or 0))[:workflow._grouping_frames_per_trigger(db)]
            triggers.append({"trigger_id": trigger_id, "trigger_time": trigger.get("trigger_time"), "frames": frames})
        body, urls = carry_request(triggers[:len(entry_ids)], triggers[len(entry_ids):], workflow._transaction_summary(transactions))
        response, meta = logged_call(db, run_id, body, urls, description="Jev carry assessment")
        result = {**carry_result(response), "script_run_id": run_id,
                  "image_urls": urls, "image_count": len(urls), "transactions": workflow._transaction_summary(transactions),
                  "cost_detail": meta.get("usage"), "receipt_attribution": "time_window_only"}
    except Exception as exc:
        result = {"status": "insufficient_evidence", "hit": False, "insufficient_evidence": True,
                  "reason": "jev_carry_failed", "message": "Carry assessment could not complete; this is not an all-clear or a carry-factor hit.",
                  "source": "openrouter_jev_carry", "error": str(exc), "script_run_id": run_id}
        repositories.finish_script_run(db, run_id, status="failed",
                                     stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, result), stderr_log=str(exc))
        return result
    repositories.finish_script_run(db, run_id, status="success",
                                 stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, result), stderr_log="")
    return result
