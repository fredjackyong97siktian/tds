"""Durable paid-receipt ownership, independent of session creation."""

from contextlib import contextmanager
from datetime import timedelta
import json

from sqlalchemy import bindparam, text

from .. import repositories

OPEN_ENTRY_LOOKBACK_MINUTES = 60


@contextmanager
def location_lock(db, location_id, *, wait_seconds=30):
    with db.get_bind().connect() as connection:
        name = f"tds_paid_assignment_{int(location_id)}"
        acquired = connection.execute(text("select get_lock(:name, :wait)"), {"name": name, "wait": wait_seconds}).scalar() == 1
        if not acquired:
            raise RuntimeError("Paid transaction assignment is busy for this location. Retry the confidence result.")
        try:
            yield
        finally:
            connection.execute(text("select release_lock(:name)"), {"name": name})


def load(db, location_id, receipts):
    if not receipts:
        return {}
    statement = text(f"select * from {repositories._table('paid_transaction_assignment')} "
                     "where location_id=:location_id and receipt_number in :receipts").bindparams(bindparam("receipts", expanding=True))
    rows = repositories._fetch_all_dicts(db.execute(statement, {"location_id": location_id, "receipts": receipts}))
    for row in rows:
        if isinstance(row.get("evidence"), str):
            row["evidence"] = json.loads(row["evidence"])
    return {str(row["receipt_number"]): row for row in rows}


def list_unresolved(db, *, limit=20):
    # Bound the ledger first; never transfer full Script Run logs to this panel.
    return repositories._fetch_all_dicts(db.execute(text(f"""
        select a.*, s.id as latest_script_run_id, s.runner_payload,
               left(s.stderr_log, 1000) as last_error
        from (
            select * from {repositories._table('paid_transaction_assignment')}
            where status in ('error', 'uncertain')
            order by updated_at desc, location_id, receipt_number limit :limit
        ) a
        left join {repositories._table('script_run')} s on s.id = coalesce(
          a.script_run_id,
          cast(json_unquote(json_extract(a.evidence, '$.requested_from_script_run_id')) as unsigned), (
            select max(r.id) from {repositories._table('script_run')} r
            where r.script_name='grouping' and r.model_name='openrouter-jev-1.14_transaction_assignment'
              and json_unquote(json_extract(r.runner_payload, '$.receipt_number'))=a.receipt_number
              and cast(json_unquote(json_extract(r.runner_payload, '$.location_id')) as unsigned)=a.location_id
        ))
        order by a.updated_at desc, a.location_id, a.receipt_number
    """), {"limit": max(1, min(int(limit), 50))}))


def save(db, *, location_id, receipt, payment_time_utc, context_hash, decision):
    db.execute(text(f"""
        insert into {repositories._table('paid_transaction_assignment')}
            (location_id, receipt_number, payment_time_utc, entry_trigger_id, status, method, context_hash, evidence, script_run_id)
        values (:location_id, :receipt, :payment_time, :entry, :status, :method, :hash, :evidence, :run)
        on duplicate key update payment_time_utc=values(payment_time_utc), entry_trigger_id=values(entry_trigger_id),
            status=values(status), method=values(method), context_hash=values(context_hash),
            evidence=values(evidence), script_run_id=values(script_run_id)
    """), {"location_id": location_id, "receipt": receipt, "payment_time": payment_time_utc,
           "entry": decision.get("entry_trigger_id"), "status": decision["status"],
           "method": decision["method"], "hash": context_hash,
           "evidence": json.dumps(decision, default=str), "run": decision.get("script_run_id")})
    db.commit()


def checkpoint_run(db, run_id, decision):
    stdout = repositories.merge_script_run_stdout_fields(db, run_id, decision)
    db.execute(text(f"update {repositories._table('script_run')} set stdout_log=:stdout where id=:id"),
               {"id": run_id, "stdout": stdout})
    db.commit()


def recover_decision(db, *, location_id, receipt):
    """Recover a paid answer whose receipt commit was interrupted or failed."""
    rows = repositories._fetch_all_dicts(db.execute(text(f"""
        select id, status, runner_payload, stdout_log from {repositories._table('script_run')}
        where script_name='grouping' and model_name='openrouter-jev-1.14_transaction_assignment'
          and status in ('success', 'running', 'failed')
          and cast(json_unquote(json_extract(runner_payload, '$.location_id')) as unsigned)=:location_id
          and json_unquote(json_extract(runner_payload, '$.receipt_number'))=:receipt
        order by id desc limit 20
    """), {"location_id": location_id, "receipt": receipt}))
    fields = ("status", "entry_trigger_id", "jev_decision", "reason", "kiosk_evidence", "media_owner_trigger_id")
    for row in rows:
        try:
            payload = row.get("runner_payload") or {}
            payload = json.loads(payload) if isinstance(payload, str) else payload
            result = row.get("stdout_log") or {}
            result = json.loads(result) if isinstance(result, str) else result
            if result.get("status") not in {"assigned", "none", "uncertain"}:
                continue
            if row["status"] != "success" and result.get("api_succeeded") is not True:
                continue
            if result["status"] == "assigned" and result.get("entry_trigger_id") not in payload.get("candidate_entry_trigger_ids", []):
                continue
            return {**{key: result[key] for key in fields if key in result}, "method": "jev",
                    "script_run_id": int(row["id"]), "recovered_from_script_run": True,
                    "recorded_candidate_entry_ids": payload.get("candidate_entry_trigger_ids") or []}
        except (ValueError, TypeError, AttributeError):
            continue
    return None


def list_overlapping_groups(db, *, location_id, start, end):
    table = repositories._table
    # Group across every successful batch, not only the batch currently being scored.
    rows = repositories._fetch_all_dicts(db.execute(text(f"""
        select gi.batch_id, gi.group_key,
               min(case when gi.role='entry' then te.trigger_time end) as entry_time,
               max(case when gi.role='exit' then te.trigger_time end) as exit_time,
               group_concat(distinct case when gi.role='entry' then te.id end order by te.id) as entry_ids,
               group_concat(distinct case when gi.role='exit' then te.id end order by te.id) as exit_ids
        from {table('filter_grouping_item')} gi
        join {table('filter_grouping_batch')} b on b.id=gi.batch_id
        join {table('trigger_event')} te on te.id=gi.trigger_id
        where te.location_id=:location_id and b.status='success'
          and gi.role in ('entry','exit') and gi.status='grouped'
        group by gi.batch_id, gi.group_key
        having entry_time <= :end_time and exit_time >= :start_time
    """), {"location_id": location_id, "start_time": start, "end_time": end}))
    # A missing exit is not evidence that a customer remained inside forever.
    # Bound unresolved entries by the payment window, never wall-clock now.
    opens = repositories._fetch_all_dicts(db.execute(text(f"""
        select gi.batch_id, gi.trigger_id, te.trigger_time as entry_time
        from {table('filter_grouping_item')} gi
        join {table('filter_grouping_batch')} b on b.id=gi.batch_id
        join {table('trigger_event')} te on te.id=gi.trigger_id
        where te.location_id=:location_id and b.status='success' and gi.role='unknown'
          and json_unquote(json_extract(gi.result_payload, '$.reason'))='open_entry_waiting_for_exit'
          and te.trigger_time >= :open_since and te.trigger_time <= :end_time
          and not exists (
              select 1 from {table('filter_grouping_item')} used
              join {table('filter_grouping_batch')} ub on ub.id=used.batch_id
              where used.trigger_id=gi.trigger_id and used.role='entry'
                and used.status='grouped' and ub.status='success')
    """), {"location_id": location_id, "open_since": start - timedelta(minutes=OPEN_ENTRY_LOOKBACK_MINUTES), "end_time": end}))
    return rows, opens
