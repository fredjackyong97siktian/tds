from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from .. import repositories
from ..config import settings
from ..storage import trigger_tmp_video_path

logger = logging.getLogger(__name__)
READY_STATUSES = {"ready", "retrieved", "processing", "processed"}


@contextmanager
def group_lock(db, result_id):
    # Keep the named lock on one connection while repository calls commit.
    with db.get_bind().connect() as connection:
        name = f"tds_confidence_media_{result_id}"
        acquired = connection.execute(text("select get_lock(:name, 0)"), {"name": name}).scalar() == 1
        try:
            yield acquired
        finally:
            if acquired:
                connection.execute(text("select release_lock(:name)"), {"name": name})


def _payload(row):
    value = row.get("factor_payload") or {}
    return json.loads(value) if isinstance(value, str) else dict(value)


def _save(db, result_id, plan):
    plan["checked_at"] = datetime.now(timezone.utc).isoformat()
    db.execute(text(f"""
        update {repositories._table('filter_confidence_result')}
        set factor_payload = json_set(coalesce(factor_payload, json_object()), '$.media_retrieval', json_extract(:plan, '$'))
        where id = :id
    """), {"id": result_id, "plan": json.dumps(plan, default=str)})
    db.commit()


def _time(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def required_windows(triggers, entry_ids, exit_ids):
    if not entry_ids or not exit_ids:
        raise ValueError("A completed group must have entry and exit triggers before media retrieval.")
    entries = [triggers[value] for value in entry_ids]
    exits = [triggers[value] for value in exit_ids]
    start = min(_time(row["trigger_time"]) for row in entries)
    end = max(_time(row["trigger_time"]) for row in exits)
    if set(entry_ids) & set(exit_ids) or min(_time(row["trigger_time"]) for row in exits) <= max(_time(row["trigger_time"]) for row in entries):
        raise ValueError("Exit must follow entry before media retrieval.")
    windows = []
    for role, ids in (("entry", entry_ids), ("exit", exit_ids)):
        for trigger_id in ids:
            anchor = _time(triggers[trigger_id]["trigger_time"])
            windows.append({"role": role, "section": "entrance", "trigger_id": trigger_id,
                            "start": anchor - timedelta(seconds=40), "end": anchor + timedelta(seconds=10)})
    # No receipt is needed to preserve the kiosk evidence for a visit.
    windows.append({"role": "kiosk", "section": "kiosk", "trigger_id": entry_ids[0], "start": start, "end": end})
    return windows


def queue_for_group(db, *, batch_id, group_key):
    row = db.execute(text(f"select * from {repositories._table('filter_confidence_result')} where batch_id=:batch_id and group_key=:group_key"),
                     {"batch_id": batch_id, "group_key": str(group_key)}).mappings().first()
    if row is None:
        raise ValueError("Confidence result must be saved before queuing its media.")
    with group_lock(db, row["id"]) as acquired:
        if not acquired:
            return []
        current = repositories.get_filter_confidence_result(db, row["id"])
        return _queue(db, current) if current else []


def _queue(db, row):
    payload = _payload(row)
    entry_ids = [int(value) for value in payload.get("entry_trigger_ids", [])]
    exit_ids = [int(value) for value in payload.get("exit_trigger_ids", [])]
    plan = payload.get("media_retrieval") or {}
    if plan.get("status") in {"pending", "waiting", "complete"}:
        return [link["video_asset_id"] for link in plan.get("links", [])]
    plan = {"status": "preparing", "links": []}
    _save(db, row["id"], plan)
    try:
        triggers = {value: repositories.get_trigger(db, value) for value in set(entry_ids + exit_ids)}
        if any(int(value["location_id"]) != int(row["location_id"]) for value in triggers.values()):
            raise ValueError("Group media cannot reference another location's trigger.")
        for window in required_windows(triggers, entry_ids, exit_ids):
            asset = repositories.find_video_asset_by_window(
                db, section=window["section"], location_id=row["location_id"], start_time=window["start"], end_time=window["end"], full_video_only=True)
            if asset is None:
                filename = f"cr_{row['id']}_{window['role']}_{window['start']:%Y%m%d_%H%M%S}_{window['end']:%Y%m%d_%H%M%S}.mp4"
                path = trigger_tmp_video_path(row["location_id"], window["trigger_id"], window["section"], filename)
                asset_id = repositories.create_video_asset(db, {
                    "trigger_id": window["trigger_id"], "section": window["section"], "sequence_no": None,
                    "video_url": "", "file_path": str(path), "captured_start_time": window["start"],
                    "captured_end_time": window["end"], "retrieved_at": None, "analyzed_at": None,
                    "retention_until": None, "status": "not_retrieved",
                    "metadata": {"location_id": row["location_id"], "retrieval_source": "confidence_evidence",
                                 "retrieval_mode": "full_video", "analysis_eligible": False,
                                 "confidence_result_id": row["id"], "grouping_id": row["batch_id"]},
                })
                repositories.update_video_asset_url(db, asset_id, f"/api/v1/videos/assets/{asset_id}/content")
            else:
                asset_id = int(asset["id"])
            plan["links"].append({"video_asset_id": asset_id, "section": window["role"],
                                  "trigger_id": window["trigger_id"], "start": window["start"].isoformat(), "end": window["end"].isoformat()})
            _save(db, row["id"], plan)
        plan["status"] = "pending"
        _save(db, row["id"], plan)
        return [link["video_asset_id"] for link in plan["links"]]
    except Exception as exc:
        db.rollback()
        plan.update(status="planning_error", error=str(exc))
        _save(db, row["id"], plan)
        raise


def finish_pending(db, limit=50):
    from . import session_analysis
    rows = db.execute(text(f"""
        select * from {repositories._table('filter_confidence_result')}
        where json_unquote(json_extract(factor_payload, '$.media_retrieval.status'))
              in ('preparing', 'planning_error', 'pending', 'waiting')
           or (:resume and json_unquote(json_extract(factor_payload, '$.media_retrieval.session_analysis_status')) = 'paused')
        order by updated_at asc, id asc limit :limit
    """), {"limit": limit, "resume": not session_analysis.PAUSED}).mappings().all()
    for row in rows:
        try:
            with group_lock(db, row["id"]) as acquired:
                if acquired:
                    current = repositories.get_filter_confidence_result(db, row["id"])
                    plan = _payload(current).get("media_retrieval", {}) if current else {}
                    if current and (plan.get("status") != "complete" or (not session_analysis.PAUSED and plan.get("session_analysis_status") == "paused")):
                        _finish(db, current)
        except Exception:
            db.rollback()
            logger.exception("Could not finish confidence media result_id=%s", row["id"])


def _finish(db, row):
    from . import workflow_service, session_analysis, detection_analysis

    payload = _payload(row)
    plan = payload["media_retrieval"]
    if plan["status"] in {"preparing", "planning_error"}:
        _queue(db, row)
        return
    links = plan.get("links", [])
    assets = [repositories.get_video_asset(db, int(link["video_asset_id"])) for link in links]
    for link, asset in zip(links, assets):
        link["status"] = asset["status"]
    if not links or any(asset["status"] not in READY_STATUSES or not asset.get("retrieved_at") for asset in assets):
        errors = [int(asset["id"]) for asset in assets if asset["status"] in {"issue", "deleted"}]
        plan.update(status="waiting" if errors else "pending", issue_video_asset_ids=errors)
        _save(db, row["id"], plan)
        return
    if not row.get("need_deep_analysis"):
        plan.update(status="complete", session_analysis_status="not_required")
        _save(db, row["id"], plan)
        return
    if session_analysis.PAUSED and not detection_analysis.ENABLED:
        plan.update(status="complete", session_analysis_status="paused")
        _save(db, row["id"], plan)
        return
    session = workflow_service._ensure_session_for_confidence_group(
        db, location_id=int(row["location_id"]), entry_trigger_ids=payload["entry_trigger_ids"],
        exit_trigger_ids=payload["exit_trigger_ids"], entry_has_identity=payload.get("entry_has_identity", True))
    if session is None:
        _save(db, row["id"], plan)
        return
    session_id = int(session["id"])
    no_transaction = (session_analysis.no_transaction_check(db, row=row, payload=payload, session_id=session_id)
                      if not detection_analysis.ENABLED else None)
    if no_transaction:
        db.execute(text(f"""
            update {repositories._table('filter_confidence_result')}
            set factor_payload=json_set(coalesce(factor_payload,json_object()), '$.no_transaction_group', json_extract(:value,'$'))
            where id=:id
        """), {"id": row["id"], "value": json.dumps(no_transaction, default=str)})
        db.commit()
    repositories.update_session_grouping_link(db, session_id=session_id, grouping_id=int(row["batch_id"]))
    if "transaction_assignment" in payload:
        from . import transaction_assignment
        owned = transaction_assignment.for_session(db, session_id)
        if owned is not None:
            transaction_assignment.sync_session(db, session_id, owned)
    for link, asset in zip(links, assets):
        repositories.create_session_video_asset_link(db, session_id, asset["id"], {
            "section": link["section"], "sequence_no": None, "clip_start_time": _time(link["start"]),
            "clip_end_time": _time(link["end"]), "is_primary": link["section"] == "entry",
            "metadata": {"source": "confidence_evidence", "confidence_result_id": row["id"]}})
    for asset in assets:
        # Collecting evidence for a safe group must not start paid L1 analysis.
        metadata = dict(asset.get("metadata") or {})
        if metadata.get("retrieval_source") == "confidence_evidence":
            metadata["analysis_eligible"] = False if detection_analysis.ENABLED else (bool(metadata.get("analysis_eligible")) or bool(row["need_deep_analysis"]))
            repositories.update_video_asset_status(db, asset["id"], asset["status"], metadata=metadata)
    if detection_analysis.ENABLED:
        detection_analysis.queue_for_confidence(db, session_id, row)
    plan.update(status="complete", session_analysis_status="detection_queued" if detection_analysis.ENABLED else "ready", session_id=session_id, completed_at=datetime.now(timezone.utc).isoformat())
    _save(db, row["id"], plan)
