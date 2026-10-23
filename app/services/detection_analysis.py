"""One explicit, recoverable frame-based detection attempt for an L1 session."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
import time

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from .. import repositories
from ..config import settings
from .detection_evidence import MODEL
from .detection_settings import current_model

logger = logging.getLogger(__name__)
ENABLED = True
DECISIONS = {"reasonable": "not_detected", "suspicious": "detected", "insufficient_evidence": "need_review"}


def obj(value):
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def session_lock(db, session_id):
    with db.get_bind().connect() as connection:
        name = f"tds_detection_{int(session_id)}"
        acquired = connection.execute(text("select get_lock(:name, 0)"), {"name": name}).scalar() == 1
        try:
            yield acquired
        finally:
            if acquired:
                connection.execute(text("select release_lock(:name)"), {"name": name})


def save(db, session_id, analysis, *, session_status=None):
    db.execute(text(f"""update {repositories._table('session')}
        set result_summary=json_set(coalesce(result_summary,json_object()), '$.detection_analysis', json_extract(:value,'$')),
            status=coalesce(:status,status)
        where id=:id"""), {"id": session_id, "value": json.dumps(analysis, default=str), "status": session_status})
    db.commit()


def queue_for_confidence(db, session_id, row):
    if not row.get("need_deep_analysis"):
        return None
    session = repositories.get_session(db, session_id)
    existing = obj(session.get("result_summary")).get("detection_analysis")
    if existing:
        return existing
    analysis = {"status": "queued", "model": current_model(db), "confidence_result_id": int(row["id"]),
                "queued_at": now(), "attempt": 0}
    save(db, session_id, analysis, session_status="pending")
    return analysis


def retry(db, session_id):
    with session_lock(db, session_id) as acquired:
        if not acquired:
            raise ValueError("Detection Analysis is running; wait for it to finish.")
        session = repositories.get_session(db, session_id)
        analysis = obj(session.get("result_summary")).get("detection_analysis")
        if not analysis:
            raise ValueError("This session has no queued Detection Analysis. Older sessions are not automatically backfilled.")
        if analysis.get("status") in {"queued", "running"}:
            return analysis
        if analysis.get("status") == "success" and analysis.get("decision") != "insufficient_evidence":
            raise ValueError("Detection Analysis has already completed. No duplicate paid request was queued.")
        analysis = {**analysis, "status": "queued", "queued_at": now(), "retry_requested": True,
                    "new_attempt_requested": analysis.get("status") == "success"}
        analysis.pop("error", None)
        save(db, session_id, analysis, session_status="pending")
        return analysis


def pending(db, limit=10):
    return db.execute(text(f"""select s.id from {repositories._table('session')} s
        left join {repositories._table('filter_confidence_result')} c
          on c.id=cast(json_unquote(json_extract(s.result_summary,'$.detection_analysis.confidence_result_id')) as unsigned)
        where json_unquote(json_extract(s.result_summary,'$.detection_analysis.status')) = 'running'
           or (json_unquote(json_extract(s.result_summary,'$.detection_analysis.status')) = 'queued'
               and coalesce(json_unquote(json_extract(c.factor_payload,'$.transaction_assignment.status')),'complete') in ('complete','completed')
               and coalesce(json_extract(c.factor_payload,'$.transaction_assignment.unresolved_count'),0)=0)
        order by s.id asc limit :limit"""), {"limit": limit}).scalars().all()


def worker_status(db):
    rows = db.execute(text(f"""select location_id,
        sum(json_unquote(json_extract(result_summary,'$.detection_analysis.status'))='queued') as queued_count,
        sum(json_unquote(json_extract(result_summary,'$.detection_analysis.status'))='running') as running_count
        from {repositories._table('session')}
        where json_unquote(json_extract(result_summary,'$.detection_analysis.status')) in ('queued','running')
        group by location_id""")).mappings().all()
    locations = [{"location_id": int(row["location_id"]), "queued_count": int(row["queued_count"] or 0),
                  "running_count": int(row["running_count"] or 0), "is_busy": bool(row["running_count"]),
                  "queued_video_asset_ids": [], "running_video_asset_ids": []} for row in rows]
    return {"poll_seconds": 10, "max_global_workers": 1, "cooldown_seconds": 0,
            "queued_count": sum(row["queued_count"] for row in locations),
            "running_count": sum(row["running_count"] for row in locations),
            "remote_dispatch_busy": False, "paused": repositories.is_worker_paused(db, "entrance_analysis"),
            "locations": locations}


def list_issues(db, *, limit=10, offset=0):
    limit, offset = max(1, min(int(limit), 50)), max(0, int(offset))
    table = repositories._table("session")
    analysis = "json_extract(result_summary,'$.detection_analysis')"
    lifecycle = f"json_unquote(json_extract({analysis},'$.status'))"
    decision = f"json_unquote(json_extract({analysis},'$.decision'))"
    retrying = f"json_unquote(json_extract({analysis},'$.retry_requested')) in ('true','1')"
    where = f"""({lifecycle} = 'failed'
        or ({lifecycle} = 'success' and {decision} = 'insufficient_evidence')
        or ({lifecycle} in ('queued','running') and {retrying}))"""
    total = int(db.execute(text(f"select count(*) from {table} where {where}")).scalar() or 0)
    rows = db.execute(text(f"""select id as session_id, location_id, {analysis} as analysis,
        greatest(coalesce(json_unquote(json_extract({analysis},'$.queued_at')),''),
                 coalesce(json_unquote(json_extract({analysis},'$.started_at')),''),
                 coalesce(json_unquote(json_extract({analysis},'$.completed_at')),'')) as updated_at
        from {table} where {where} order by updated_at desc, id desc limit :limit offset :offset"""),
        {"limit": limit, "offset": offset}).mappings().all()
    items = [{**dict(row), "analysis": obj(row["analysis"])} for row in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset,
            "worker_paused": repositories.is_worker_paused(db, "entrance_analysis")}


def worker_activity(db):
    return repositories._fill_hourly_activity_buckets(repositories._hourly_status_activity(
        db, table_name="script_run", time_column="finished_at", processed_statuses=("success",),
        failed_statuses=("failed",), extra_where="(script_name = 'detection_analysis' or (script_name = 'carry_confidence' and right(model_name,19) = '_detection_analysis'))"))


def parse_decision(response):
    if response.get("error"):
        raise ValueError("MiMo returned a provider error, not a completed detection decision.")
    choices = response.get("choices") or []
    finish_reason = response.get("finish_reason") or (choices[0].get("finish_reason") if choices else None)
    if finish_reason and finish_reason != "stop":
        raise ValueError("MiMo did not finish a complete Detection Analysis response.")
    content = response.get("content")
    if content is None and choices:
        content = (choices[0].get("message") or {}).get("content")
    from .workflow_service import _extract_json_object
    result = _extract_json_object(str(content or ""))
    decision = result.get("status")
    if decision not in DECISIONS:
        raise ValueError("MiMo did not return a valid Detection Analysis status.")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("MiMo did not provide an evidence-based explanation.")
    limitations = result.get("limitations") or ""
    if not isinstance(limitations, (str, list)):
        limitations = str(limitations)
    if isinstance(limitations, list):
        limitations = [str(item) for item in limitations]
    return {"decision": decision, "reason": result["reason"], "limitations": limitations}


def _cost(response):
    from .workflow_service import _openrouter_usage_cost
    return _openrouter_usage_cost({"model": response.get("model", MODEL),
                                   "usage": {"actual_cost_usd": (response.get("usage") or {}).get("cost")}})[0]


def stream_response(chunks):
    result = {key: next((chunk[key] for chunk in reversed(chunks) if chunk.get(key) is not None), None)
              for key in ("id", "model", "provider", "usage")}
    result["content"] = ''.join(choice.get("delta", {}).get("content") or '' for chunk in chunks for choice in chunk.get("choices", []))
    result["finish_reason"] = next((choice["finish_reason"] for chunk in reversed(chunks)
                                    for choice in chunk.get("choices", []) if choice.get("finish_reason")), None)
    if any(chunk.get("error") for chunk in chunks):
        result["error"] = "Provider stream failed"
    return result


def _persist_response(db, run_id, response, *, complete):
    # Persist billed usage BEFORE validating or applying the model's decision.
    _write_journal(run_id, response, complete=complete)
    db.execute(text(f"update {repositories._table('script_run')} set stdout_log=:log where id=:id"),
               {"id": run_id, "log": json.dumps({"response": response, "response_complete": complete}, default=str)})
    db.commit()
    amount = _cost(response)
    if amount is not None:
        repositories.update_script_run_cost(db, run_id, cost_amount=amount, cost_currency="USD", cost_source="openrouter_actual")


def _journal_path(run_id):
    return Path(settings.video_storage_dir) / "detection_analysis" / f"script_run_{int(run_id)}.json"


def _write_journal(run_id, response, *, complete):
    # This mounted-volume journal survives a database outage and temp cleanup.
    try:
        path = _journal_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"response": response, "response_complete": complete}, default=str))
        temporary.replace(path)
    except OSError:
        logger.warning("Could not persist detection journal run_id=%s; retaining database audit", run_id, exc_info=True)


def _saved_response(db, run_id):
    if not run_id:
        return None
    row = db.execute(text(f"select stdout_log from {repositories._table('script_run')} where id=:id"), {"id": run_id}).mappings().first()
    sources = [row["stdout_log"]] if row else []
    try:
        sources.append(_journal_path(run_id).read_text())
    except OSError:
        pass
    for source in sources:
        try:
            saved = obj(source)
            response = saved.get("response")
            if response and (saved.get("response_complete") or response.get("finish_reason") == "stop"):
                parse_decision(response)
                return response
        except (ValueError, TypeError):
            pass
    return None


def _apply(db, session_id, analysis, response):
    result = parse_decision(response)
    analysis = {**analysis, **result, "status": "success", "completed_at": now(),
                "model": response.get("model") or MODEL, "cost_amount": _cost(response),
                "cost_currency": "USD", "cost_source": "openrouter_actual"}
    analysis.pop("error", None)
    analysis.pop("new_attempt_requested", None)
    analysis.pop("retry_requested", None)
    save(db, session_id, analysis, session_status=DECISIONS[result["decision"]])
    return analysis


def create_run(db, *, model_name, **kwargs):
    try:
        return repositories.create_script_run_started(db, script_name="detection_analysis", model_name=model_name, **kwargs)
    except DBAPIError as exc:
        original = getattr(exc, "orig", None)
        if not getattr(original, "args", ()) or original.args[0] != 3819 or "chk_script_run_name" not in str(original):
            raise
        # Compatibility only for the known old script-name check. No inference
        # has happened yet; this is not a second paid API attempt.
        db.rollback()
        return repositories.create_script_run_started(db, script_name="carry_confidence",
            model_name=f"{model_name}_detection_analysis", **kwargs)


def run_session(db, session_id):
    from . import detection_evidence, detection_frames
    with session_lock(db, session_id) as acquired:
        if not acquired:
            return None
        session = repositories.get_session(db, session_id)
        analysis = obj(session.get("result_summary")).get("detection_analysis") or {}
        if analysis.get("status") not in {"queued", "running"}:
            return analysis
        previous_run = analysis.get("script_run_id")
        response = _saved_response(db, previous_run) if not (analysis.get("status") == "queued" and analysis.get("new_attempt_requested")) else None
        if response:
            # Explicit retry/restart reuses a saved paid answer; no extra charge.
            _persist_response(db, previous_run, response, complete=True)
            result = _apply(db, session_id, analysis, response)
            repositories.finish_script_run(db, previous_run, status="success",
                stdout_log=json.dumps({"response": response, "response_complete": True}), stderr_log="")
            return result
        if analysis.get("status") == "running":
            # The named lock was free: the previous process is gone. Never repay automatically.
            analysis.update(status="failed", error="Previous detection process stopped without a recoverable answer. Retry explicitly; the upstream request may have been billed.")
            save(db, session_id, analysis, session_status="issue")
            if previous_run:
                db.execute(text(f"update {repositories._table('script_run')} set status='failed', stderr_log=:error, finished_at=now() where id=:id and status='running'"),
                           {"id": previous_run, "error": analysis["error"]})
                db.commit()
            return analysis
        run_id = None
        try:
            row = repositories.get_filter_confidence_result(db, int(analysis["confidence_result_id"]))
            if not row or not row.get("need_deep_analysis"):
                raise ValueError("A factor hit or explicit L1 request is required for Detection Analysis.")
            payload = obj(row.get("factor_payload"))
            assignment = obj(payload.get("transaction_assignment"))
            if assignment.get("status", "complete") not in {"complete", "completed"} or assignment.get("unresolved_count", 0):
                analysis.update(status="queued", message="Waiting for paid receipt ownership to be resolved before Detection Analysis.")
                save(db, session_id, analysis)
                return analysis
            if not payload.get("media_retrieval", {}).get("links"):
                raise ValueError("Detection evidence has not been attached to this confidence result.")
            analysis.update(status="running", started_at=now(), attempt=int(analysis.get("attempt", 0)) + 1)
            analysis.pop("message", None)
            analysis.pop("new_attempt_requested", None)
            save(db, session_id, analysis, session_status="pending")
            selected_model = current_model(db)
            analysis["model"] = selected_model
            run_id = create_run(db, session_id=session_id, trigger_id=session.get("entry_trigger_id"),
                model_name=selected_model,
                runner_payload={"confidence_result_id": row["id"], "batch_id": row["batch_id"], "group_key": row["group_key"],
                                "location_id": row["location_id"], "analysis_type": "detection_analysis", "requested_model": selected_model,
                                "attempt": analysis["attempt"], "previous_script_run_id": previous_run},
                command="[redacted]", status="running")
            analysis["script_run_id"] = run_id
            save(db, session_id, analysis)
            chunks = []
            last_save = 0.0

            def capture(chunk):
                nonlocal last_save
                chunks.append(chunk)
                if chunk.get("usage") or time.monotonic() - last_save > 3:
                    _persist_response(db, run_id, stream_response(chunks), complete=False)
                    last_save = time.monotonic()

            with tempfile.TemporaryDirectory(prefix=f"tds-detection-{session_id}-") as directory:
                context, frames = detection_evidence.build_evidence(db, session, row, Path(directory))
                snapshots = detection_frames.preserve(frames, location_id=row["location_id"], session_id=session_id, run_id=run_id)
                analysis["image_count"] = len(frames)
                save(db, session_id, analysis)
                db.execute(text(f"""update {repositories._table('script_run')}
                    set runner_payload=json_set(coalesce(runner_payload,json_object()),
                        '$.image_count', :count, '$.request_context', json_extract(:context,'$'),
                        '$.frames', json_extract(:frames,'$'), '$.frame_snapshot_version', 1,
                        '$.request_started_at', :started) where id=:id"""),
                    {"id": run_id, "count": len(frames), "context": json.dumps(context, default=str),
                     "frames": json.dumps(snapshots, default=str), "started": now()})
                db.commit()
                response = detection_evidence.call_mimo(context, frames, on_chunk=capture, model_name=selected_model)
            _persist_response(db, run_id, response, complete=True)
            result = _apply(db, session_id, analysis, response)
            repositories.finish_script_run(db, run_id, status="success",
                stdout_log=json.dumps({"response": response, "response_complete": True}), stderr_log="")
            return result
        except Exception as exc:
            db.rollback()
            logger.exception("Detection Analysis failed session_id=%s", session_id)
            analysis.update(status="failed", error=str(exc), completed_at=now())
            save(db, session_id, analysis, session_status="issue")
            if run_id:
                # Do not overwrite a saved raw answer or its already-recorded cost.
                db.execute(text(f"update {repositories._table('script_run')} set status='failed', stderr_log=:error, finished_at=now() where id=:id"),
                           {"id": run_id, "error": str(exc)})
                db.commit()
            return analysis
