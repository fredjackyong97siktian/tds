"""Lightweight successful-run audit for one confidence group, without log blobs."""

import json

from sqlalchemy import bindparam, text

from .. import repositories


def _object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def _ids(values):
    result = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            pass
    return result


def _recorded_runs(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("script_run_id"):
                found.update(_ids([item]))
            elif key.endswith("script_run_ids") and isinstance(item, list):
                found.update(_ids(item))
            elif isinstance(item, (dict, list)):
                found.update(_recorded_runs(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_recorded_runs(item))
    return found


def _scope(run, *, row, trigger_ids, run_ids, receipts):
    payload = _object(run.get("runner_payload"))
    if run.get("status") != "success":
        return None
    if run["script_run_id"] in run_ids:
        return "Recorded for this group"
    if run.get("batch_root") and run.get("script_name") == "grouping":
        return "Shared grouping batch"
    model = str(run.get("model_name") or "")
    if model.endswith("_transaction_assignment"):
        if str(payload.get("receipt_number")) in receipts and str(payload.get("location_id")) == str(row["location_id"]):
            return "Receipt assignment"
        return None
    if run.get("script_name") in {"retrieve_video", "retrieve_frame"} or model.endswith("_person_presence"):
        if run.get("trigger_id") in trigger_ids:
            return "Trigger evidence"
    if row.get("session_id") and run.get("session_id") == row["session_id"]:
        return "Session"
    if str(payload.get("batch_id")) == str(row["batch_id"]):
        if run.get("script_name") == "carry_confidence":
            return "Group analysis" if str(payload.get("group_key")) == str(row["group_key"]) else None
        if run.get("script_name") == "grouping" and not model.endswith("_person_presence"):
            return "Shared grouping batch"
    if _ids([payload.get("parent_script_run_id")]) & run_ids:
        return "Linked evidence"
    return None


def list_successful_runs(db, confidence_result_id):
    rows = repositories.list_filter_confidence_results(db, limit=1, confidence_result_id=confidence_result_id)
    if not rows:
        return None
    row = rows[0]
    payload = _object(row.get("factor_payload"))
    trigger_ids = _ids([value for key in ("entry_trigger_ids", "exit_trigger_ids", "trigger_ids")
                        for value in payload.get(key, [])])
    run_ids = _recorded_runs(payload)
    assignment = _object(payload.get("transaction_assignment"))
    receipts = {str(item["receipt_number"]) for item in assignment.get("decisions", [])
                if isinstance(item, dict) and item.get("receipt_number")}
    table = repositories._table("script_run")
    batch_root = f"(select script_run_id from {repositories._table('filter_grouping_batch')} where id = :batch_id)"
    costs = repositories._script_run_cost_select(db, table)
    query = text(f"""
        select id as script_run_id, session_id, trigger_id, script_name, model_name,
               status, runner_payload, {costs}, started_at, finished_at,
               (id = {batch_root}) as batch_root
        from {table}
        where status = 'success' and (
            trigger_id in :trigger_ids or session_id = :session_id or id in :run_ids
            or id = {batch_root}
            or json_unquote(json_extract(runner_payload, '$.batch_id')) = :batch_id
            or json_unquote(json_extract(runner_payload, '$.parent_script_run_id')) in :parent_ids
            or (json_unquote(json_extract(runner_payload, '$.location_id')) = :location_id
                and json_unquote(json_extract(runner_payload, '$.receipt_number')) in :receipts)
        )
        order by coalesce(started_at, finished_at) asc, id asc
    """).bindparams(*(bindparam(key, expanding=True) for key in
                       ("trigger_ids", "run_ids", "parent_ids", "receipts")))
    candidates = db.execute(query, {
        "trigger_ids": sorted(trigger_ids), "run_ids": sorted(run_ids),
        "parent_ids": [str(value) for value in sorted(run_ids)], "receipts": sorted(receipts),
        "session_id": row.get("session_id"), "batch_id": str(row["batch_id"]),
        "location_id": str(row["location_id"]),
    }).mappings().all()
    result = []
    for candidate in candidates:
        run = dict(candidate)
        scope = _scope(run, row=row, trigger_ids=trigger_ids, run_ids=run_ids, receipts=receipts)
        if scope:
            run.pop("runner_payload", None)
            run.pop("batch_root", None)
            run["scope"] = scope
            result.append(run)
    return result
