"""Batch trigger references used for non-customer exclusions in result summaries."""


def trigger_ids(values):
    if not isinstance(values, list):
        return set()
    found = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("trigger_id", value.get("entry_trigger_id", value.get("id")))
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
            if number > 0:
                found.add(number)
        except (TypeError, ValueError):
            pass
    return found


def result_trigger_ids(payload):
    if not isinstance(payload, dict):
        return set()
    result = set()
    for key in ("open_entries", "unknown", "issues", "excluded_non_customer_trigger_ids"):
        result.update(trigger_ids(payload.get(key)))
    groups = payload.get("groups")
    for group in groups if isinstance(groups, list) else []:
        if isinstance(group, dict):
            result.update(trigger_ids(group.get("entry")))
            result.update(trigger_ids(group.get("exit")))
    return result


def annotate_batches(db, rows):
    from .. import repositories

    references = {value for row in rows for value in result_trigger_ids(row.get("result_payload"))}
    if not references:
        return
    excluded = repositories.list_confirmed_non_customer_trigger_ids(db, sorted(references))
    for row in rows:
        payload = row.get("result_payload")
        if isinstance(payload, dict):
            relevant = result_trigger_ids(payload)
            saved = trigger_ids(payload.get("excluded_non_customer_trigger_ids"))
            row["result_payload"] = {**payload, "excluded_non_customer_trigger_ids": sorted(saved | (excluded & relevant))}
