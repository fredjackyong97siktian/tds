"""Assign each paid receipt to at most one active visit, before scoring factors."""

from datetime import datetime, timedelta
import hashlib
import json
import re

from .. import repositories
from . import transaction_assignment_store as store

POLICY_VERSION = 2
ENABLED_SETTING_KEY = "paid_transaction_assignment_enabled"


class AssignmentPaused(RuntimeError):
    pass


def is_enabled(db):
    # Explicit opt-in: keep work queued while the assignment issue is investigated.
    return str(repositories.get_app_setting(db, ENABLED_SETTING_KEY) or "false").strip().lower() == "true"


def require_enabled(db):
    if not is_enabled(db):
        raise AssignmentPaused("Transaction Assignment is paused. Confidence analysis remains queued; no purchase decision was made.")


def payment_time(row):
    from . import workflow_service as workflow

    value = workflow._coerce_datetime_value(row.get("payment_attempt_at"))
    if value is not None:
        return workflow._utc_naive_to_local(value), "payment_attempt_at"
    return workflow._coerce_datetime_value(row.get("transaction_time")), "transaction_time_fallback"


def active_groups(groups, paid_at):
    return [group for group in groups if group["entry_time"] <= paid_at
            and ((group.get("exit_time") is not None and paid_at <= group["exit_time"])
                 or (group.get("exit_time") is None
                     and paid_at - group["entry_time"] <= timedelta(minutes=store.OPEN_ENTRY_LOOKBACK_MINUTES)))]


def machine_side(location_name, row):
    raw = row.get("raw_payload") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    name = str(row.get("machine_name") or raw.get("machineName") or raw.get("machine_name") or "").strip()
    side = None
    if "edumetro" in str(location_name).lower():
        tokens = set(re.findall(r"(?<![a-z])(left|right)(?![a-z])", name.lower()))
        if len(tokens) == 1:
            side = {"left": "right", "right": "left"}[tokens.pop()]
    return {"machine_name": name, "image_side": side,
            "mapping": "Edumetro: Left machine = image-right; Right machine = image-left."
            if "edumetro" in str(location_name).lower() else "Machine-to-image mapping not configured; do not assume a side."}


def _ids(value):
    return [int(item) for item in str(value or "").split(",") if item]


def collect_groups(db, *, location_id, start, end, batch_id, current_group):
    from . import workflow_service as workflow

    rows, opens = store.list_overlapping_groups(db, location_id=location_id, start=start, end=end)
    by_entry = {}
    for row in sorted(rows, key=lambda item: int(item["batch_id"])):
        entries = _ids(row["entry_ids"])
        if len(entries) != 1:
            raise ValueError("Transaction assignment requires one entry trigger per group.")
        by_entry[entries[0]] = {"entry_trigger_id": entries[0], "exit_trigger_ids": _ids(row["exit_ids"]),
                              "entry_time": row["entry_time"], "exit_time": row["exit_time"],
                              "batch_id": row["batch_id"], "group_key": row["group_key"]}
    for row in opens:
        entry_id = int(row["trigger_id"])
        by_entry.setdefault(entry_id, {"entry_trigger_id": entry_id, "exit_trigger_ids": [],
                                      "entry_time": row["entry_time"], "exit_time": None,
                                      "batch_id": row["batch_id"], "group_key": None})
    batch = repositories.get_grouping_batch(db, batch_id)
    payload = batch.get("result_payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    # Include all newly formed groups even before their item rows are committed.
    pending_entries = [{"entry": [value], "exit": []} for value in workflow._group_trigger_id_list(payload.get("open_entries"))
                       if value not in by_entry]
    for group in [*pending_entries, *(payload.get("groups") or []), current_group]:
        entry_ids = workflow._group_trigger_id_list(group.get("entry"))
        exit_ids = workflow._group_trigger_id_list(group.get("exit"))
        if len(entry_ids) != 1:
            raise ValueError("Transaction assignment requires one entry trigger per group.")
        entry_time = workflow._coerce_datetime_value(repositories.get_trigger(db, entry_ids[0])["trigger_time"])
        exit_times = [workflow._coerce_datetime_value(repositories.get_trigger(db, value)["trigger_time"]) for value in exit_ids]
        if entry_time is None or any(value is None or value <= entry_time for value in exit_times):
            raise ValueError("Invalid group timestamps for transaction assignment.")
        by_entry[entry_ids[0]] = {"entry_trigger_id": entry_ids[0], "exit_trigger_ids": exit_ids,
                                "entry_time": entry_time, "exit_time": max(exit_times) if exit_times else None,
                                "batch_id": batch_id, "group_key": str(group.get("group_id") or group.get("id") or "")}
    # Apply the same bounds to current-batch open entries, which may not yet
    # exist in the stored item query, and to older/cached batch snapshots.
    eligible = [value for value in by_entry.values() if value["entry_time"] <= end
                and ((value["exit_time"] is not None and value["exit_time"] >= start)
                     or (value["exit_time"] is None
                         and value["entry_time"] >= start - timedelta(minutes=store.OPEN_ENTRY_LOOKBACK_MINUTES)))]
    return sorted(eligible, key=lambda value: value["entry_trigger_id"])


def _references(db, group):
    from . import workflow_service as workflow

    def sequence(trigger_id):
        trigger = repositories.get_trigger(db, trigger_id)
        urls = workflow._trigger_frame_image_urls_for_identity(db, trigger_id, limit=6)
        return {"trigger_id": trigger_id, "trigger_time": trigger["trigger_time"],
                "frames": [{"frame_index": index, "image_url": url} for index, url in enumerate(urls, 1)]}
    return [sequence(group["entry_trigger_id"]), *[sequence(value) for value in group["exit_trigger_ids"]]]


def decide_with_jev(db, *, location_id, batch_id, receipt, row, paid_at, candidates, mapping):
    from . import identifier_capture, jev_pipeline, workflow_service as workflow

    require_enabled(db)

    run_id = repositories.create_script_run_started(
        db, session_id=None, trigger_id=candidates[0]["entry_trigger_id"], script_name="grouping",
        model_name="openrouter-jev-1.14_transaction_assignment", command=workflow.SCRIPT_RUN_COMMAND_REDACTED,
        runner_payload={"batch_id": batch_id, "location_id": location_id, "receipt_number": receipt,
                        "candidate_entry_trigger_ids": [value["entry_trigger_id"] for value in candidates],
                        "payment_time": paid_at.isoformat(), **mapping})
    evidence = {"script_run_id": run_id, "method": "jev", "media_owner_trigger_id": candidates[0]["entry_trigger_id"]}
    try:
        probe = identifier_capture.capture_sequence(
            db, location_id=location_id, trigger={"trigger_id": candidates[0]["entry_trigger_id"], "trigger_time": paid_at},
            parent_script_run_id=run_id, section="kiosk",
            times=[paid_at + timedelta(seconds=index - 5) for index in range(10)])
        evidence["kiosk_evidence"] = probe
        group_states, all_urls = [], []
        for group in candidates:
            sequences, urls = jev_pipeline.encode_sequences(_references(db, group))
            all_urls.extend(urls)
            group_states.append({"entry_trigger_id": group["entry_trigger_id"],
                                 "entry": sequences[0], "exits": sequences[1:]})
        kiosk, urls = jev_pipeline.encode_sequences([probe])
        all_urls.extend(urls)
        options = {f"entry_{group['entry_trigger_id']}": f"The paying customer is the customer of entry trigger {group['entry_trigger_id']}."
                   for group in candidates}
        options.update(none="The paying customer clearly belongs to none of these groups.",
                       uncertain="The paying customer or their group cannot be determined clearly from these images.")
        body = {"model": jev_pipeline.jev.MODEL,
                "state": {"active_groups": group_states, "payment_camera": kiosk[0],
                          "paid_transaction": workflow._transaction_summary([row])[0], **mapping},
                "questions": {"receipt_owner": {"type": "choice", "criteria": options, "instructions": (
                    "Identify who made this ONE paid transaction using payment_camera frames in time order and active_groups entry/exit references. "
                    "Match visible clothing, build, hair and movement to the person actually operating the relevant payment machine. "
                    "Use machine_name and image_side when supplied: Edumetro Left machine is on image-right, Right machine on image-left. "
                    "A bystander near the kiosk is not necessarily the payer. Do not assign by timing alone: every candidate was active at payment. "
                    "Do not invent hidden faces or items. Choose exactly one owner, none if clearly another customer, "
                    "or uncertain when identity/visibility/machine association is unclear. Other receipts may belong to the same group.")}}}
        require_enabled(db)
        response, _meta = jev_pipeline.logged_call(db, run_id, body, list(dict.fromkeys(all_urls)), description="Paid transaction assignment")
        answer = jev_pipeline.read_choice(response, "receipt_owner", set(options))
        choice = answer["choice"] if answer["confidence"] >= jev_pipeline.MIN_CHOICE_CONFIDENCE else "uncertain"
        evidence.update(status="assigned" if choice.startswith("entry_") else choice,
                        entry_trigger_id=int(choice[6:]) if choice.startswith("entry_") else None,
                        jev_decision=answer, reason=f"Jev selected {choice} for this paid receipt.")
        repositories.finish_script_run(db, run_id, status="success", stderr_log="",
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, evidence))
    except Exception as exc:
        evidence.update(status="error", entry_trigger_id=None, reason=str(exc))
        repositories.finish_script_run(db, run_id, status="failed", stderr_log=str(exc),
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, evidence))
        if isinstance(exc, AssignmentPaused):
            raise
    return evidence


def assign(db, *, batch_id, location_id, group, start_time, end_time):
    from . import workflow_service as workflow

    require_enabled(db)
    entries = workflow._group_trigger_id_list(group.get("entry"))
    if len(entries) != 1:
        raise ValueError("Transaction assignment requires exactly one entry trigger.")
    owner = entries[0]
    # A failed query or missing migration must stop scoring, not masquerade as no purchase.
    with store.location_lock(db, location_id):
        rows = repositories.list_paid_transactions_for_assignment(
            db, location_id=location_id, start_time=start_time, end_time=end_time)
        rows = list({str(row.get("receipt_number") or row.get("transaction_id")): row for row in rows}.values())
        if not rows:
            return [], _report(owner, [], [], [])
        payment_times = [payment_time(row)[0] for row in rows]
        if any(value is None for value in payment_times):
            raise ValueError("Paid receipt has no usable payment timestamp.")
        groups = collect_groups(db, location_id=location_id, start=min(payment_times), end=max(payment_times),
                                batch_id=batch_id, current_group=group)
        receipts = [str(row.get("receipt_number") or row.get("transaction_id") or "") for row in rows]
        if any(not value or value == "None" for value in receipts):
            raise ValueError("Paid transaction is missing its receipt identity.")
        cached = store.load(db, location_id, receipts)
        location = repositories.get_location_endpoint(db, location_id)
        decisions, assigned = [], []
        for receipt, row in zip(receipts, rows):
            require_enabled(db)
            paid_at, source = payment_time(row)
            if paid_at is None:
                raise ValueError(f"Paid receipt {receipt} has no usable payment timestamp.")
            eligible = active_groups(groups, paid_at)
            mapping = machine_side(location.get("name"), row)
            context = {"version": POLICY_VERSION, "paid_at": paid_at,
                       "groups": [{key: value[key] for key in ("entry_trigger_id", "exit_trigger_ids", "entry_time", "exit_time")}
                                  for value in eligible], "mapping": mapping}
            context_hash = hashlib.sha256(json.dumps(context, default=str, sort_keys=True).encode()).hexdigest()
            previous = cached.get(receipt)
            failed_run_id = None
            if previous and previous["status"] == "error" and len(eligible) <= 1:
                # Re-evaluate a failed visual attempt deterministically when the
                # corrected time window proves there is no ambiguity. No resend.
                failed_run_id = previous["evidence"].get("script_run_id")
                previous = None
            if previous and (previous["context_hash"] == context_hash or previous["status"] == "error"):
                decision = {**previous["evidence"], "cached": True}
            elif previous and previous["status"] == "assigned":
                # Never silently move a receipt after another confidence result used it.
                decision = {**previous["evidence"], "cached": True, "context_changed": True,
                            "reason": "Existing receipt owner retained; candidate context changed. Review before reassigning."}
            else:
                if len(eligible) == 1:
                    decision = {"status": "assigned", "method": "time", "entry_trigger_id": eligible[0]["entry_trigger_id"],
                                "reason": "Only this group was active at payment time."}
                elif not eligible:
                    decision = {"status": "none", "method": "time", "entry_trigger_id": None,
                                "reason": "No group was active at payment time."}
                else:
                    decision = decide_with_jev(db, location_id=location_id, batch_id=batch_id, receipt=receipt,
                                               row=row, paid_at=paid_at, candidates=eligible, mapping=mapping)
                decision.update(receipt_number=receipt, transaction_id=row.get("transaction_id"), payment_time=paid_at.isoformat(),
                                payment_time_source=source, active_groups=eligible, **mapping)
                if failed_run_id is not None:
                    decision["supersedes_failed_script_run_id"] = failed_run_id
                store.save(db, location_id=location_id, receipt=receipt, payment_time_utc=workflow._local_naive_to_utc(paid_at),
                           context_hash=context_hash, decision=decision)
            decisions.append(decision)
            if decision["status"] == "assigned" and decision.get("entry_trigger_id") == owner and not decision.get("context_changed"):
                assigned.append({**row, "receipt_attribution": "confirmed_group_ownership"})
    return assigned, _report(owner, rows, assigned, decisions)


def _report(owner, rows, assigned, decisions):
    unresolved = sum(value["status"] in {"uncertain", "error"} or value.get("context_changed", False) for value in decisions)
    return {"version": POLICY_VERSION, "entry_trigger_id": owner,
            "open_entry_lookback_minutes": store.OPEN_ENTRY_LOOKBACK_MINUTES,
            "status": "needs_review" if unresolved else "complete", "candidate_count": len(rows),
            "assigned_count": len(assigned), "unresolved_count": unresolved,
            "selection": "none" if not assigned else "all" if len(assigned) == len(rows) else "some",
            "assigned_receipts": [str(row.get("receipt_number") or row.get("transaction_id")) for row in assigned],
            "decisions": decisions}


def for_session(db, session_id):
    """None means a legacy session; [] means authoritative no assigned paid receipt."""
    session = repositories.get_session(db, session_id)
    result_id = session.get("confidence_result_id")
    if not result_id:
        return None
    result = repositories.get_filter_confidence_result(db, int(result_id))
    payload = (result or {}).get("factor_payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    if "transaction_assignment" not in payload:
        return None
    receipts = payload["transaction_assignment"].get("assigned_receipts") or []
    owners = store.load(db, int(session["location_id"]), receipts)
    entry = payload["transaction_assignment"]["entry_trigger_id"]
    return [row for row in payload.get("transactions", [])
            if (owners.get(str(row.get("receipt_number"))) or {}).get("entry_trigger_id") == entry
            and (owners.get(str(row.get("receipt_number"))) or {}).get("status", "assigned") == "assigned"]


def sync_session(db, session_id, transactions):
    from . import workflow_service as workflow

    repositories.delete_session_transactions(db, session_id)
    for row in transactions:
        repositories.create_transaction(db, session_id, {
            "receipt_number": str(row.get("receipt_number") or row.get("transaction_id")),
            "transaction_time": workflow._coerce_datetime_value(row.get("transaction_time")),
            "total_items": row.get("total_items") or 0, "total_amount": row.get("total_amount"),
            "raw_payload": dict(row)})
