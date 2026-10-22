"""Assign each paid receipt to at most one active visit, before scoring factors."""

from datetime import datetime, timedelta
import hashlib
import json
import re

from .. import repositories
from . import transaction_assignment_store as store

POLICY_VERSION = 2
ENABLED_SETTING_KEY = "paid_transaction_assignment_enabled"
PAYMENT_FRAME_OFFSETS_SECONDS = (-4, -2, 0, 2, 4)
PAYMENT_FRAME_IMAGE_SCALE = 0.5


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
    return [group for group in groups if group.get("exit_trigger_ids")
            and group.get("exit_time") is not None
            and group["entry_time"] <= paid_at <= group["exit_time"]]


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

    rows, _ = store.list_overlapping_groups(db, location_id=location_id, start=start, end=end)
    by_entry = {}
    for row in sorted(rows, key=lambda item: int(item["batch_id"])):
        entries = _ids(row["entry_ids"])
        if len(entries) != 1:
            raise ValueError("Transaction assignment requires one entry trigger per group.")
        by_entry[entries[0]] = {"entry_trigger_id": entries[0], "exit_trigger_ids": _ids(row["exit_ids"]),
                              "entry_time": row["entry_time"], "exit_time": row["exit_time"],
                              "batch_id": row["batch_id"], "group_key": row["group_key"]}
    batch = repositories.get_grouping_batch(db, batch_id)
    if batch.get("status") != "success" or int(batch.get("location_id") or 0) != location_id:
        raise ValueError("Transaction assignment requires a successful grouping batch at this location.")
    payload = batch.get("result_payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    # Current-batch groups can be scored before their membership rows commit,
    # but the saved grouping output must actually contain the group.
    def membership(group):
        return (tuple(sorted(workflow._group_trigger_id_list(group.get("entry")))),
                tuple(sorted(workflow._group_trigger_id_list(group.get("exit")))))

    recorded = {membership(group) for group in payload.get("groups") or []}
    recorded.update((tuple(_ids(row["entry_ids"])), tuple(_ids(row["exit_ids"])))
                    for row in rows if int(row["batch_id"]) == batch_id)
    if membership(current_group) not in recorded or not membership(current_group)[1]:
        raise ValueError("Transaction assignment requires an existing group with an entry and an exit.")
    for group in [*(payload.get("groups") or []), current_group]:
        entry_ids = workflow._group_trigger_id_list(group.get("entry"))
        exit_ids = workflow._group_trigger_id_list(group.get("exit"))
        if not exit_ids:
            continue
        if len(entry_ids) != 1:
            raise ValueError("Transaction assignment requires one entry trigger per group.")
        entry_time = workflow._coerce_datetime_value(repositories.get_trigger(db, entry_ids[0])["trigger_time"])
        exit_times = [workflow._coerce_datetime_value(repositories.get_trigger(db, value)["trigger_time"]) for value in exit_ids]
        if entry_time is None or any(value is None or value <= entry_time for value in exit_times):
            raise ValueError("Invalid group timestamps for transaction assignment.")
        by_entry[entry_ids[0]] = {"entry_trigger_id": entry_ids[0], "exit_trigger_ids": exit_ids,
                                "entry_time": entry_time, "exit_time": max(exit_times) if exit_times else None,
                                "batch_id": batch_id, "group_key": str(group.get("group_id") or group.get("id") or "")}
    eligible = [value for value in by_entry.values() if value["exit_trigger_ids"]
                and value["exit_time"] is not None and value["entry_time"] <= end
                and value["exit_time"] >= start]
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
                        "payment_time": paid_at.isoformat(), "payment_frame_count": len(PAYMENT_FRAME_OFFSETS_SECONDS),
                        "payment_frame_offsets_seconds": list(PAYMENT_FRAME_OFFSETS_SECONDS),
                        "payment_image_scale": PAYMENT_FRAME_IMAGE_SCALE, **mapping})
    evidence = {"script_run_id": run_id, "method": "jev", "media_owner_trigger_id": candidates[0]["entry_trigger_id"]}
    try:
        store.update_retry_progress(db, location_id=location_id, receipt=receipt, state="retrieving_frames", script_run_id=run_id)
        probe = identifier_capture.capture_sequence(
            db, location_id=location_id, trigger={"trigger_id": candidates[0]["entry_trigger_id"], "trigger_time": paid_at},
            parent_script_run_id=run_id, section="kiosk",
            times=[paid_at + timedelta(seconds=offset) for offset in PAYMENT_FRAME_OFFSETS_SECONDS],
            image_scale=PAYMENT_FRAME_IMAGE_SCALE)
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
        store.update_retry_progress(db, location_id=location_id, receipt=receipt, state="calling_jev", script_run_id=run_id)
        response, _meta = jev_pipeline.logged_call(db, run_id, body, list(dict.fromkeys(all_urls)), description="Paid transaction assignment")
        answer = jev_pipeline.read_choice(response, "receipt_owner", set(options))
        choice = answer["choice"]
        evidence.update(status="assigned" if choice.startswith("entry_") else choice,
                        entry_trigger_id=int(choice[6:]) if choice.startswith("entry_") else None,
                        jev_decision=answer, reason=f"Jev selected {choice} for this paid receipt.",
                        api_succeeded=True, persistence_status="awaiting_receipt_commit")
        store.checkpoint_run(db, run_id, evidence)
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
    if len(entries) != 1 or not workflow._group_trigger_id_list(group.get("exit")):
        raise ValueError("Transaction assignment requires exactly one entry trigger and at least one exit in a group.")
    owner = entries[0]
    # A failed query or missing migration must stop scoring, not masquerade as no purchase.
    with store.location_lock(db, location_id):
        # Config reads may have opened a repeatable-read snapshot BEFORE the
        # location lock was acquired. Refresh it to see the prior owner's commit.
        db.commit()
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
            retry = (previous or {}).get("evidence", {}).get("retry")
            if retry and retry.get("finished_at"):
                retry = None
            if previous and previous["context_hash"] == context_hash:
                updated = store.honor_recorded_choice(previous["evidence"], {item["entry_trigger_id"] for item in eligible})
                if updated != previous["evidence"]:
                    _commit_decision(db, location_id=location_id, receipt=receipt,
                        payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash, decision=updated)
                    previous = {**previous, "status": updated["status"], "evidence": updated}
            if previous and previous["evidence"].get("retry_authorized"):
                # One explicit user retry. The new attempt barrier replaces this
                # authorization before a paid request, so polling cannot loop it.
                from datetime import timezone
                retry = {**(retry or {}), "state": "processing", "batch_id": batch_id,
                         "started_at": datetime.now(timezone.utc).isoformat()}
                retry.setdefault("requested_at", retry["started_at"])
                retry.setdefault("source_script_run_id", previous["evidence"].get("script_run_id"))
                # Consume authorization durably before recovery/capture. A
                # killed retry cannot silently authorize another paid call.
                store.save(db, location_id=location_id, receipt=receipt,
                    payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash,
                    decision={"status": "error", "method": "jev", "entry_trigger_id": None,
                              "attempt_reserved": True, "retry": retry,
                              "reason": "Retry processing: checking saved answers before a new request."})
                previous = None
            if previous and previous["evidence"].get("attempt_reserved"):
                recovered = store.recover_decision(db, location_id=location_id, receipt=receipt)
                if recovered is not None:
                    recovered.update(receipt_number=receipt, transaction_id=row.get("transaction_id"),
                                     payment_time=paid_at.isoformat(), payment_time_source=source,
                                     active_groups=eligible, **mapping)
                    if set(recovered["recorded_candidate_entry_ids"]) != {item["entry_trigger_id"] for item in eligible}:
                        recovered["context_changed"] = True
                    if retry:
                        recovered["retry"] = retry
                    _commit_decision(db, location_id=location_id, receipt=receipt,
                        payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash, decision=recovered)
                    previous = {"status": recovered["status"], "context_hash": context_hash, "evidence": recovered}
            failed_run_id = None
            if previous and previous["status"] == "error" and len(eligible) <= 1:
                # Re-evaluate a failed visual attempt deterministically when the
                # corrected time window proves there is no ambiguity. No resend.
                failed_run_id = previous["evidence"].get("script_run_id")
                previous = None
            if previous and (previous["context_hash"] == context_hash or previous["status"] == "error"):
                decision = {**previous["evidence"], "cached": True}
            elif previous:
                # A successful none/uncertain answer must not be bought again
                # merely because another grouping batch changed the candidates.
                decision = {**previous["evidence"], "cached": True, "context_changed": True,
                            "reason": "Recorded receipt decision retained; candidate context changed. Review before reassigning."}
            else:
                if len(eligible) == 1:
                    decision = {"status": "assigned", "method": "time", "entry_trigger_id": eligible[0]["entry_trigger_id"],
                                "reason": "Only this group was active at payment time."}
                elif not eligible:
                    decision = {"status": "none", "method": "time", "entry_trigger_id": None,
                                "reason": "No group was active at payment time."}
                else:
                    decision = store.recover_decision(db, location_id=location_id, receipt=receipt)
                    if decision is None:
                        # Commit a durable attempt barrier BEFORE any NVR/API
                        # work. A killed child or failed final write cannot resend.
                        store.save(db, location_id=location_id, receipt=receipt,
                            payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash,
                            decision={"status": "error", "method": "jev", "entry_trigger_id": None,
                                      "attempt_reserved": True,
                                      **({"retry": retry} if retry else {}),
                                      "reason": "Assignment attempt reserved; outcome not committed. Automatic resend blocked."})
                        decision = decide_with_jev(db, location_id=location_id, batch_id=batch_id, receipt=receipt,
                                                   row=row, paid_at=paid_at, candidates=eligible, mapping=mapping)
                    elif set(decision["recorded_candidate_entry_ids"]) != {item["entry_trigger_id"] for item in eligible}:
                        decision["context_changed"] = True
                decision.update(receipt_number=receipt, transaction_id=row.get("transaction_id"), payment_time=paid_at.isoformat(),
                                payment_time_source=source, active_groups=eligible, **mapping)
                if failed_run_id is not None:
                    decision["supersedes_failed_script_run_id"] = failed_run_id
                if retry:
                    decision["retry"] = retry
                _commit_decision(db, location_id=location_id, receipt=receipt,
                    payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash, decision=decision)
            if retry and not (decision.get("retry") or {}).get("finished_at"):
                decision["retry"] = retry
                _commit_decision(db, location_id=location_id, receipt=receipt,
                    payment_time_utc=workflow._local_naive_to_utc(paid_at), context_hash=context_hash, decision=decision)
            decisions.append(decision)
            if decision["status"] == "assigned" and decision.get("entry_trigger_id") == owner and not decision.get("context_changed"):
                assigned.append({**row, "receipt_attribution": "confirmed_group_ownership"})
    return assigned, _report(owner, rows, assigned, decisions)


def _commit_decision(db, *, location_id, receipt, payment_time_utc, context_hash, decision):
    from datetime import timezone
    if decision.get("retry"):
        state = ("failed" if decision["status"] == "error" else "needs_review"
                 if decision["status"] == "uncertain" or decision.get("context_changed") else "completed")
        source = ("saved_answer" if decision.get("recovered_from_script_run") or decision.get("cached")
                  else "timing" if decision.get("method") == "time" else "new_request")
        decision["retry"] = {**decision["retry"], "state": state, "source": source,
                             "finished_at": datetime.now(timezone.utc).isoformat(),
                             "script_run_id": decision.get("script_run_id")}
    run_id = decision.get("script_run_id")
    try:
        store.save(db, location_id=location_id, receipt=receipt, payment_time_utc=payment_time_utc,
                   context_hash=context_hash, decision={**decision, "persistence_status": "committed"})
    except Exception as exc:
        db.rollback()
        if run_id:
            repositories.finish_script_run(db, run_id, status="failed", stderr_log=f"Receipt decision persistence failed: {exc}",
                stdout_log=repositories.merge_script_run_stdout_fields(db, run_id,
                    {**decision, "api_succeeded": decision.get("status") != "error", "persistence_status": "failed",
                     "persistence_error": str(exc)}))
        raise
    decision["persistence_status"] = "committed"
    if run_id:
        repositories.finish_script_run(db, run_id, status="failed" if decision["status"] == "error" else "success",
            stderr_log=str(decision.get("reason") or "") if decision["status"] == "error" else "",
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id,
                {**decision, "persistence_status": "committed"}))


def _report(owner, rows, assigned, decisions):
    unresolved = sum(value["status"] in {"uncertain", "error"} or value.get("context_changed", False) for value in decisions)
    return {"version": POLICY_VERSION, "entry_trigger_id": owner,
            "candidate_policy": "completed_groups_only",
            "status": "needs_review" if unresolved else "complete", "candidate_count": len(rows),
            "assigned_count": len(assigned), "unresolved_count": unresolved,
            "selection": "none" if not assigned else "all" if len(assigned) == len(rows) else "some",
            "assigned_receipts": [str(row.get("receipt_number") or row.get("transaction_id")) for row in assigned],
            "decisions": decisions}


def retry_from_script_run(db, script_run_id):
    from sqlalchemy import text
    from . import workflow_service as workflow

    require_enabled(db)
    run = repositories.get_script_run(db, script_run_id)
    if run.get("model_name") != "openrouter-jev-1.14_transaction_assignment":
        raise ValueError("This Script Run is not Transaction Assignment.")
    payload = run.get("runner_payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise ValueError("This Script Run has an invalid assignment payload.") from exc
    if not isinstance(payload, dict):
        raise ValueError("This Script Run has an invalid assignment payload.")
    batch_id, location_id = int(payload.get("batch_id") or 0), int(payload.get("location_id") or 0)
    receipt = str(payload.get("receipt_number") or "").strip()
    paid_at = workflow._coerce_datetime_value(payload.get("payment_time"))
    if not batch_id or not location_id or not receipt or paid_at is None:
        raise ValueError("This Script Run is missing its batch, location, receipt or payment time.")
    batch = repositories.get_grouping_batch(db, batch_id)
    if int(batch["location_id"]) != location_id or batch["status"] != "success":
        raise ValueError("The original grouping batch must be successful before retrying its assignment.")
    lock = workflow._try_acquire_theft_confidence_lock(batch_id, wait_seconds=0)
    if lock is None:
        raise RuntimeError("This confidence batch is already running. Wait for it to finish before retrying.")
    try:
        with store.location_lock(db, location_id, wait_seconds=0):
            db.commit()
            previous = store.load(db, location_id, [receipt]).get(receipt)
            from datetime import timezone
            retry = {"state": "queued", "requested_at": datetime.now(timezone.utc).isoformat(),
                     "batch_id": batch_id, "source_script_run_id": script_run_id}
            table = repositories._table("filter_confidence_result")
            # Keep existing result IDs and group memberships. Only affected
            # scores become eligible again; the worker does the expensive work.
            db.execute(text(f"""
                update {table}
                set factor_payload=json_set(coalesce(factor_payload, json_object()), '$.assignment_retry_requested', json_extract('true', '$'))
                where batch_id=:batch_id and json_search(factor_payload, 'one', :receipt, null,
                    '$.transaction_assignment.decisions[*].receipt_number') is not null
            """), {"batch_id": batch_id, "receipt": receipt})
            db.execute(text(f"delete from {table} where batch_id=:batch_id and group_key='__error__'"), {"batch_id": batch_id})
            if not previous or previous["status"] != "assigned":
                store.save(db, location_id=location_id, receipt=receipt,
                    payment_time_utc=workflow._local_naive_to_utc(paid_at),
                    context_hash=(previous or {}).get("context_hash") or hashlib.sha256(receipt.encode()).hexdigest(),
                    decision={"status": "error", "method": "jev", "entry_trigger_id": None,
                              "retry_authorized": True, "requested_from_script_run_id": script_run_id,
                              "script_run_id": script_run_id, "batch_id": batch_id,
                              "retry": retry,
                              "reason": "User queued one retry. Recover a saved answer first; otherwise allow one new attempt."})
            else:
                store.save(db, location_id=location_id, receipt=receipt,
                    payment_time_utc=previous["payment_time_utc"], context_hash=previous["context_hash"],
                    decision={**previous["evidence"], "retry": retry})
            paused = repositories.is_worker_paused(db, "theft_confidence_analysis")
            return {"ok": True, "batch_id": batch_id, "receipt_number": receipt, "queued": True,
                    "retry": retry,
                    "message": "Retry queued. Turn on Theft Confidence Analysis in Workers." if paused else
                               "Retry queued. Saved answers will be reused before any new paid request."}
    except Exception:
        db.rollback()
        raise
    finally:
        workflow._release_theft_confidence_lock(lock, batch_id)


def list_stuck_assignments(db, *, limit=20):
    rows = store.list_unresolved(db, limit=limit)
    items = []
    for row in rows:
        evidence = row.get("evidence") or {}
        payload = row.get("runner_payload") or {}
        for name, value in (("evidence", evidence), ("payload", payload)):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = {}
            if not isinstance(value, dict):
                value = {}
            if name == "evidence":
                evidence = value
            else:
                payload = value
        retry = evidence.get("retry") or {}
        if retry.get("state") in {"processing", "retrieving_frames", "calling_jev"} and not row.get("confidence_running"):
            retry = {**retry, "state": "interrupted"}
        items.append({"location_id": row["location_id"], "receipt_number": row["receipt_number"],
                      "status": "queued" if evidence.get("retry_authorized") else row["status"],
                      "batch_id": payload.get("batch_id") or evidence.get("batch_id"),
                      "script_run_id": row.get("latest_script_run_id") or row.get("script_run_id"),
                      "payment_time_utc": row.get("payment_time_utc"), "updated_at": row.get("updated_at"),
                      "last_error": (row.get("last_error") or None) if row["status"] == "error" else None,
                      "retry": retry or None, "confidence_running": bool(row.get("confidence_running")),
                      "entry_trigger_id": row.get("entry_trigger_id"),
                      "reason": str(evidence.get("reason") or "Assignment requires review or recovery.")[:1000]})
    return {"items": items, "limit": limit, "enabled": is_enabled(db),
            "worker_paused": repositories.is_worker_paused(db, "theft_confidence_analysis")}


def resolve_manually(db, *, confidence_result_id, receipt, entry_trigger_id):
    """Record a human ownership decision; never capture frames or call AI."""
    from contextlib import ExitStack
    from datetime import timezone
    from sqlalchemy import text
    from . import workflow_service as workflow

    result = repositories.get_filter_confidence_result(db, confidence_result_id)
    if not result:
        raise ValueError("Confidence result was not found.")
    batch_id, location_id = int(result["batch_id"]), int(result["location_id"])
    lock = workflow._try_acquire_theft_confidence_lock(batch_id, wait_seconds=0)
    if lock is None:
        raise RuntimeError("This confidence batch is running. Wait for it to finish before assigning manually.")
    locks = ExitStack()
    locks.callback(workflow._release_theft_confidence_lock, lock, batch_id)
    try:
        with store.location_lock(db, location_id, wait_seconds=0):
            db.commit()
            result = repositories.get_filter_confidence_result(db, confidence_result_id)
            if not result:
                raise ValueError("Confidence result was removed. Refresh the page.")
            payload = result.get("factor_payload") or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            report = payload.get("transaction_assignment") or {}
            snapshot = next((decision for decision in report.get("decisions", [])
                             if str(decision.get("receipt_number")) == receipt), None)
            if snapshot is None:
                raise ValueError("This receipt does not belong to this confidence result's assignment audit.")
            previous = store.load(db, location_id, [receipt]).get(receipt)
            if not previous:
                raise ValueError("No saved receipt assignment exists. Retry confidence analysis first.")
            evidence = previous.get("evidence") or {}
            if isinstance(evidence, str):
                evidence = json.loads(evidence)
            if previous["status"] == "assigned" and previous.get("entry_trigger_id") != entry_trigger_id:
                raise RuntimeError("This receipt already has a confirmed owner. Refresh before making another change.")
            if previous["status"] not in {"error", "uncertain", "assigned"}:
                raise ValueError("This receipt is already resolved. Refresh its confidence result.")
            paid_at = workflow._utc_naive_to_local(previous["payment_time_utc"])
            if paid_at is None:
                raise ValueError("The receipt has no valid payment timestamp.")
            current_group = {"entry": payload.get("entry_trigger_ids") or [report.get("entry_trigger_id")],
                             "exit": payload.get("exit_trigger_ids") or []}
            groups = collect_groups(db, location_id=location_id, start=paid_at, end=paid_at,
                                    batch_id=batch_id, current_group=current_group)
            eligible = active_groups(groups, paid_at)
            if entry_trigger_id is not None and not any(group["entry_trigger_id"] == entry_trigger_id for group in eligible):
                raise ValueError("Select an existing completed group that was active when this receipt was paid.")
            related = repositories._fetch_all_dicts(db.execute(text(f"""
                select distinct batch_id from {repositories._table('filter_confidence_result')}
                where location_id=:location_id and json_search(factor_payload, 'one', :receipt, null,
                    '$.transaction_assignment.decisions[*].receipt_number') is not null
            """), {"location_id": location_id, "receipt": receipt}))
            affected_batches = {int(row["batch_id"]) for row in related}
            affected_batches.update(int(group["batch_id"]) for group in eligible)
            for related_batch in sorted(affected_batches - {batch_id}):
                related_lock = workflow._try_acquire_theft_confidence_lock(related_batch, wait_seconds=0)
                if related_lock is None:
                    raise RuntimeError("Another affected confidence batch is running. Wait for it to finish before assigning manually.")
                locks.callback(workflow._release_theft_confidence_lock, related_lock, related_batch)
            context = {"version": POLICY_VERSION, "paid_at": paid_at,
                       "groups": [{key: group[key] for key in ("entry_trigger_id", "exit_trigger_ids", "entry_time", "exit_time")}
                                  for group in eligible],
                       "mapping": {key: snapshot.get(key) for key in ("machine_name", "image_side", "mapping")}}
            context_hash = hashlib.sha256(json.dumps(context, default=str, sort_keys=True).encode()).hexdigest()
            decision = {**evidence, **context["mapping"], "status": "assigned" if entry_trigger_id is not None else "none",
                        "method": "manual", "entry_trigger_id": entry_trigger_id, "context_changed": False,
                        "receipt_number": receipt, "payment_time": paid_at.isoformat(), "active_groups": eligible,
                        "reason": f"Manually assigned to the group with entry #{entry_trigger_id}." if entry_trigger_id is not None
                                  else "Manually confirmed that none of the candidate groups owns this paid receipt.",
                        "manual_resolution": {"confidence_result_id": confidence_result_id,
                            "resolved_at": datetime.now(timezone.utc).isoformat(),
                            "previous_status": previous["status"], "previous_reason": evidence.get("reason")},
                        "persistence_status": "committed"}
            if decision.get("retry"):
                decision["retry"] = {**decision["retry"], "state": "completed", "source": "manual",
                                     "finished_at": datetime.now(timezone.utc).isoformat()}
            for key in ("attempt_reserved", "retry_authorized", "requested_from_script_run_id"):
                decision.pop(key, None)
            # Queue every existing group that saw this receipt, not just the
            # chosen group. Their totals and carry checks must agree on ownership.
            db.execute(text(f"""
                update {repositories._table('filter_confidence_result')}
                set factor_payload=json_set(coalesce(factor_payload, json_object()),
                    '$.assignment_retry_requested', json_extract('true', '$'))
                where location_id=:location_id and json_search(factor_payload, 'one', :receipt, null,
                    '$.transaction_assignment.decisions[*].receipt_number') is not null
            """), {"location_id": location_id, "receipt": receipt})
            store.save(db, location_id=location_id, receipt=receipt, payment_time_utc=previous["payment_time_utc"],
                       context_hash=context_hash, decision=decision)
            held = not is_enabled(db) or repositories.is_worker_paused(db, "theft_confidence_analysis")
            return {"ok": True, "status": decision["status"], "entry_trigger_id": entry_trigger_id,
                    "message": "Manual ownership saved without an AI request. Confidence checks are queued."
                    + (" Resume Transaction Assignment and the confidence worker to process them." if held else "")}
    except Exception:
        db.rollback()
        raise
    finally:
        locks.close()


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
