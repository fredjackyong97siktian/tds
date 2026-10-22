"""Temporary hold on L1 sessions; L0 scoring and evidence retrieval continue."""

PAUSED = True


def require_active():
    if PAUSED:
        raise ValueError("Session analysis is paused. No new L1 or no-transaction API request was started.")


def no_transaction_check(db, *, row, payload, session_id):
    require_active()
    if not row.get("need_deep_analysis") or payload.get("transactions"):
        return None
    existing = payload.get("no_transaction_group")
    if existing:
        return existing
    from . import jev_pipeline
    return jev_pipeline.evaluate_carry(
        db, batch_id=row["batch_id"], group_key=row["group_key"], location_id=row["location_id"],
        group={"entry": payload["entry_trigger_ids"], "exit": payload["exit_trigger_ids"]},
        transactions=[], session_id=session_id,
    )
