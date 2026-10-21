"""Post-direct recovery of open identity entries, including shared exit evidence."""

from datetime import timedelta

from .. import repositories
from . import jev_pipeline

SETTING_KEY = "grouping_expand_duration_minutes"
DEFAULT_MINUTES = 20


def duration_minutes(db):
    value = repositories.get_app_setting(db, SETTING_KEY)
    try:
        minutes = int(value) if value is not None else DEFAULT_MINUTES
    except (TypeError, ValueError):
        minutes = DEFAULT_MINUTES
    return max(1, min(120, minutes))


def plan(triggers, groups, *, window_end, minutes, excluded_ids=()):
    from . import workflow_service as workflow

    end = workflow._coerce_datetime_value(window_end)
    if end is None:
        raise ValueError("Expand requires the running grouping period's end timestamp.")
    entries_used = {value for group in groups for value in workflow._group_trigger_id_list(group.get("entry"))}
    by_id = {int(row["trigger_id"]): row for row in triggers if row.get("frames") and int(row["trigger_id"]) not in excluded_ids
             and row.get("motion_person_present") not in (False, 0)
             and workflow._coerce_datetime_value(row.get("trigger_time")) is not None}
    ordered = sorted(by_id.values(), key=lambda row: (workflow._coerce_datetime_value(row.get("trigger_time")), int(row["trigger_id"])))
    checks = []
    for entry in ordered:
        entry_id = int(entry["trigger_id"])
        if entry_id in entries_used or (entry.get("phone_entry_id") is None and entry.get("credit_card_entry_id") is None):
            continue
        start = workflow._coerce_datetime_value(entry.get("trigger_time"))
        if start is None or start >= end:
            continue
        cutoff = min(start + timedelta(minutes=minutes), end)
        candidates = [row for row in ordered if int(row["trigger_id"]) != entry_id
                      and (stamp := workflow._coerce_datetime_value(row.get("trigger_time"))) is not None
                      and start < stamp <= cutoff and stamp < end]
        if candidates:
            checks.append({"entry": entry, "candidates": candidates, "cutoff": cutoff})
    return checks


def run(db, *, batch_id, location_id, triggers, groups, window_end, excluded_ids=()):
    from . import workflow_service as workflow

    minutes = duration_minutes(db)
    non_customer_ids = repositories.list_confirmed_non_customer_trigger_ids(
        db, [int(row["trigger_id"]) for row in triggers])
    non_customer_ids.update(int(row["trigger_id"]) for row in triggers if row.get("motion_person_present") in (False, 0))
    checks = plan(triggers, groups, window_end=window_end, minutes=minutes,
                  excluded_ids=set(excluded_ids) | non_customer_ids)
    diagnostics = {"duration_minutes": minutes, "eligible_entry_count": len(checks),
                   "planned_attempts": sum((len(check["candidates"]) + 1) // 2 for check in checks),
                   "attempts": [], "matched_groups": [], "status": "skipped",
                   "excluded_non_customer_trigger_ids": sorted(non_customer_ids)}
    if not checks:
        return [], diagnostics, []
    # The old batch+trigger key would overwrite another group's membership.
    if not repositories._column_exists(db, repositories._table("filter_grouping_item"), "membership_key"):
        raise RuntimeError("Expand requires migrations/20261021_grouping_shared_memberships.sql before it can safely share triggers.")
    run_id = workflow._create_gemini_script_run(
        db, session_id=None, trigger_id=None, script_name="grouping",
        model_name="openrouter-jev-1.14_grouping_expand",
        runner_payload={"batch_id": batch_id, "location_id": location_id, "duration_minutes": minutes,
                        "period_end": str(window_end), "planned_attempts": diagnostics["planned_attempts"],
                        "entry_trigger_ids": [int(check["entry"]["trigger_id"]) for check in checks]})
    diagnostics.update(script_run_id=run_id, status="success")
    added, notes = [], []
    next_id = max([int(group.get("group_id") or group.get("id") or index)
                   for index, group in enumerate(groups, 1)] or [0]) + 1
    try:
        for check in checks:
            entry = check["entry"]
            entry_id = int(entry["trigger_id"])
            candidates = check["candidates"]
            # Exactly the same focused entry-vs-one/two-candidates prompt and
            # selected-option behavior as Jev adjacent. Stop at the first match.
            for offset in range(0, len(candidates), 2):
                pair = candidates[offset:offset + 2]
                attempt = {"entry_trigger_id": entry_id, "candidate_trigger_ids": [int(row["trigger_id"]) for row in pair],
                           "cutoff": check["cutoff"].isoformat()}
                try:
                    body, urls = jev_pipeline.adjacent_request(entry, pair)
                    response, _meta = jev_pipeline.logged_call(db, run_id, body, urls, description="Jev expand matching")
                    result = jev_pipeline.adjacent_result(response, len(pair))
                    exit_ids = [int(pair[index - 1]["trigger_id"]) for index in result["matched_candidates"]]
                    attempt.update(matched_trigger_ids=exit_ids, jev_decision=result["jev_decision"])
                except Exception as exc:
                    attempt.update(error=str(exc), matched_trigger_ids=[])
                    diagnostics["status"] = "partial_error"
                    diagnostics["attempts"].append(attempt)
                    repositories.merge_script_run_stdout_fields(db, run_id, diagnostics)
                    # Do not jump past an unavailable earlier comparison and
                    # silently claim a later exit instead.
                    break
                diagnostics["attempts"].append(attempt)
                repositories.annotate_latest_script_run_call(db, run_id, attempt)
                repositories.merge_script_run_stdout_fields(db, run_id, diagnostics)
                if not exit_ids:
                    continue
                shared = sorted(set(exit_ids) & {value for group in [*groups, *added]
                                                 for role in ("entry", "exit")
                                                 for value in workflow._group_trigger_id_list(group.get(role))})
                match = {"group_id": next_id, "entry": [entry_id], "exit": exit_ids,
                         "score": result["confidence"], "reason": result["reason"], "total_customer": 1,
                         "entry_has_identity": True, "source": "jev_grouping_expand", "verified": True,
                         "jev_decision": result["jev_decision"],
                         "verification": {"confidence": result["confidence"], "error": None},
                         "shared_trigger_ids": shared, "expand_script_run_id": run_id,
                         "expand_duration_minutes": minutes, "expand_cutoff": check["cutoff"].isoformat()}
                added.append(match)
                diagnostics["matched_groups"] = list(added)
                repositories.merge_script_run_stdout_fields(db, run_id, diagnostics)
                next_id += 1
                notes.append(f"Expand matched entry {entry_id} to exit(s) {exit_ids}; shared evidence {shared}.")
                break
        diagnostics["matched_groups"] = added
        repositories.finish_script_run(db, run_id, status="failed" if diagnostics["status"] == "partial_error" else "success",
            stderr_log="Some Expand comparisons failed; unresolved entries were preserved." if diagnostics["status"] == "partial_error" else "",
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, diagnostics))
    except Exception as exc:
        repositories.finish_script_run(db, run_id, status="failed", stderr_log=str(exc),
            stdout_log=repositories.merge_script_run_stdout_fields(db, run_id, diagnostics))
        raise
    return added, diagnostics, notes
