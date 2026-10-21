"""Durable paid-receipt ownership, independent of session creation."""

from contextlib import contextmanager
from datetime import timedelta
import json

from sqlalchemy import bindparam, text

from .. import repositories

OPEN_ENTRY_LOOKBACK_MINUTES = 60


@contextmanager
def location_lock(db, location_id):
    with db.get_bind().connect() as connection:
        name = f"tds_paid_assignment_{int(location_id)}"
        acquired = connection.execute(text("select get_lock(:name, 30)"), {"name": name}).scalar() == 1
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
