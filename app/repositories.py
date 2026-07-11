import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import settings


def _table(name: str) -> str:
    return f"{settings.transactional_table_prefix}{name}"


def _fetch_one_dict(result) -> dict[str, Any]:
    row = result.mappings().first()
    if row is None:
        raise ValueError("Expected a row but query returned nothing.")
    return dict(row)


def _fetch_all_dicts(result) -> list[dict[str, Any]]:
    return [dict(row) for row in result.mappings().all()]


def get_cctv(db: Session, cctv_id: int) -> dict[str, Any]:
    cctv_table = _table("cctv")
    result = db.execute(
        text(
            f"""
            select id, location_id, section, stream_name, recorder_channel, delayed_seconds, created_at, updated_at
            from {cctv_table}
            where id = :cctv_id
            """
        ),
        {"cctv_id": cctv_id},
    )
    return _fetch_one_dict(result)


def list_cctv(db: Session, location_id: int | None = None) -> list[dict[str, Any]]:
    cctv_table = _table("cctv")
    if location_id is None:
        result = db.execute(
            text(
                f"""
                select id, location_id, section, stream_name, recorder_channel, delayed_seconds, created_at, updated_at
                from {cctv_table}
                order by location_id asc, section asc, id asc
                """
            )
        )
        return _fetch_all_dicts(result)

    result = db.execute(
        text(
            f"""
            select id, location_id, section, stream_name, recorder_channel, delayed_seconds, created_at, updated_at
            from {cctv_table}
            where location_id = :location_id
            order by section asc, id asc
            """
        ),
        {"location_id": location_id},
    )
    return _fetch_all_dicts(result)


def list_locations(db: Session) -> list[dict[str, Any]]:
    table_name = settings.location_table_name
    id_column = settings.location_id_column
    name_column = settings.location_name_column

    result = db.execute(
        text(
            f"""
            select {id_column} as id, {name_column} as name
            from {table_name}
            order by {name_column} asc, {id_column} asc
            """
        )
    )
    return _fetch_all_dicts(result)


def get_trigger(db: Session, trigger_id: int) -> dict[str, Any]:
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select id, location_id, status, trigger_time
            from {trigger_table}
            where id = :trigger_id
            """
        ),
        {"trigger_id": trigger_id},
    )
    return _fetch_one_dict(result)


def list_triggers(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select id, location_id, aqara_event_id, trigger_source, trigger_time,
                   entry_source_type, entry_match_status, status, whitelist_hit,
                   issue_reason, created_at, updated_at
            from {trigger_table}
            order by trigger_time desc, id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def get_video_asset(db: Session, video_asset_id: int) -> dict[str, Any]:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            select id, trigger_id, section, sequence_no, video_url, file_path,
                   captured_start_time, captured_end_time, retention_until, status, metadata, created_at
            from {video_asset_table}
            where id = :video_asset_id
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    return _fetch_one_dict(result)


def list_video_assets(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    session_video_asset_table = _table("session_video_asset")
    result = db.execute(
        text(
            f"""
            select va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                   va.captured_start_time, va.captured_end_time, va.retention_until, va.status,
                   va.metadata, va.created_at,
                   count(distinct sva.id) as session_link_count,
                   min(sva.session_id) as primary_session_id,
                   group_concat(distinct sva.session_id order by sva.session_id separator ',') as session_ids
            from {video_asset_table} va
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            group by va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                     va.captured_start_time, va.captured_end_time, va.retention_until, va.status,
                     va.metadata, va.created_at
            order by va.created_at desc, va.id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def get_session(db: Session, session_id: int) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer
            from {session_table}
            where id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    return _fetch_one_dict(result)


def list_sessions(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    session_table = _table("session")
    session_customer_table = _table("session_customer")
    session_video_asset_table = _table("session_video_asset")
    result = db.execute(
        text(
            f"""
            select s.id, s.entry_trigger_id, s.exit_trigger_id, s.location_id, s.status,
                   s.start_time, s.end_time, s.total_item_brought, s.actual_items_brought,
                   s.transaction_total_items, s.total_customer, s.created_at, s.updated_at,
                   count(distinct sc.id) as linked_customer_count,
                   count(distinct sva.id) as linked_video_count
            from {session_table} s
            left join {session_customer_table} sc on sc.session_id = s.id
            left join {session_video_asset_table} sva on sva.session_id = s.id
            group by s.id, s.entry_trigger_id, s.exit_trigger_id, s.location_id, s.status,
                     s.start_time, s.end_time, s.total_item_brought, s.actual_items_brought,
                     s.transaction_total_items, s.total_customer, s.created_at, s.updated_at
            order by s.created_at desc, s.id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def get_transaction_total_items(db: Session, session_id: int) -> int:
    transaction_table = _table("session_transaction")
    result = db.execute(
        text(
            f"""
            select coalesce(sum(total_items), 0) as transaction_total_items
            from {transaction_table}
            where session_id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    row = _fetch_one_dict(result)
    return int(row["transaction_total_items"] or 0)


def create_trigger(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            insert into {trigger_table} (
                location_id, aqara_event_id, trigger_source, trigger_time, raw_payload
            )
            values (
                :location_id, :aqara_event_id, :trigger_source, :trigger_time, :raw_payload
            )
            """
        ),
        {
            **payload,
            "raw_payload": json.dumps(payload.get("raw_payload")) if payload.get("raw_payload") is not None else None,
        },
    )
    db.commit()
    return get_trigger(db, int(result.lastrowid))


def create_cctv(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    cctv_table = _table("cctv")
    result = db.execute(
        text(
            f"""
            insert into {cctv_table} (
                location_id, section, stream_name, recorder_channel, delayed_seconds
            )
            values (
                :location_id, :section, :stream_name, :recorder_channel, :delayed_seconds
            )
            """
        ),
        payload,
    )
    db.commit()
    return get_cctv(db, int(result.lastrowid))


def update_cctv(db: Session, cctv_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    cctv_table = _table("cctv")
    result = db.execute(
        text(
            f"""
            update {cctv_table}
            set location_id = :location_id,
                section = :section,
                stream_name = :stream_name,
                recorder_channel = :recorder_channel,
                delayed_seconds = :delayed_seconds
            where id = :cctv_id
            """
        ),
        {"cctv_id": cctv_id, **payload},
    )
    db.commit()
    if result.rowcount == 0:
        raise ValueError("CCTV record not found.")
    return get_cctv(db, cctv_id)


def delete_cctv(db: Session, cctv_id: int) -> bool:
    cctv_table = _table("cctv")
    result = db.execute(
        text(
            f"""
            delete from {cctv_table}
            where id = :cctv_id
            """
        ),
        {"cctv_id": cctv_id},
    )
    db.commit()
    return bool(result.rowcount)


def update_trigger_status(db: Session, trigger_id: int, status: str, issue_reason: str | None = None) -> None:
    trigger_table = _table("trigger_event")
    db.execute(
        text(
            f"""
            update {trigger_table}
            set status = :status, issue_reason = :issue_reason
            where id = :trigger_id
            """
        ),
        {"trigger_id": trigger_id, "status": status, "issue_reason": issue_reason},
    )
    db.commit()


def create_session(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            insert into {session_table} (
                entry_trigger_id, exit_trigger_id, location_id, start_time
            )
            values (
                :entry_trigger_id, :exit_trigger_id, :location_id, :start_time
            )
            """
        ),
        payload,
    )
    db.commit()
    return get_session(db, int(result.lastrowid))


def close_session(db: Session, session_id: int, end_time, exit_trigger_id: int | None = None) -> dict[str, Any]:
    session_table = _table("session")
    db.execute(
        text(
            f"""
            update {session_table}
            set end_time = :end_time,
                exit_trigger_id = coalesce(:exit_trigger_id, exit_trigger_id),
                status = 'closed'
            where id = :session_id
            """
        ),
        {"session_id": session_id, "end_time": end_time, "exit_trigger_id": exit_trigger_id},
    )
    db.commit()
    return get_session(db, session_id)


def create_session_customer(db: Session, session_id: int, payload: Mapping[str, Any]) -> None:
    session_customer_table = _table("session_customer")
    db.execute(
        text(
            f"""
            insert into {session_customer_table} (
                session_id, person_id, enter_time, kiosk_start_time, leave_time, match_status
            )
            values (
                :session_id, :person_id, :enter_time, :kiosk_start_time, :leave_time, :match_status
            )
            on duplicate key update
                enter_time = values(enter_time),
                kiosk_start_time = values(kiosk_start_time),
                leave_time = values(leave_time),
                match_status = values(match_status)
            """
        ),
        {"session_id": session_id, **payload},
    )
    db.commit()


def create_video_asset(db: Session, payload: Mapping[str, Any]) -> int:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            insert into {video_asset_table} (
                trigger_id, section, sequence_no, video_url, file_path, captured_start_time,
                captured_end_time, retention_until, status, metadata
            )
            values (
                :trigger_id, :section, :sequence_no, :video_url, :file_path, :captured_start_time,
                :captured_end_time, :retention_until, :status, :metadata
            )
            """
        ),
        {
            **payload,
            "metadata": json.dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
        },
    )
    db.commit()
    return int(result.lastrowid)


def update_video_asset_url(db: Session, video_asset_id: int, video_url: str) -> None:
    video_asset_table = _table("video_asset")
    db.execute(
        text(
            f"""
            update {video_asset_table}
            set video_url = :video_url
            where id = :video_asset_id
            """
        ),
        {"video_asset_id": video_asset_id, "video_url": video_url},
    )
    db.commit()


def create_session_video_asset_link(db: Session, session_id: int, video_asset_id: int, payload: Mapping[str, Any]) -> None:
    session_video_asset_table = _table("session_video_asset")
    db.execute(
        text(
            f"""
            insert into {session_video_asset_table} (
                session_id, video_asset_id, section, sequence_no, clip_start_time, clip_end_time, is_primary, metadata
            )
            values (
                :session_id, :video_asset_id, :section, :sequence_no, :clip_start_time, :clip_end_time, :is_primary, :metadata
            )
            on duplicate key update
                section = values(section),
                sequence_no = values(sequence_no),
                clip_start_time = values(clip_start_time),
                clip_end_time = values(clip_end_time),
                is_primary = values(is_primary),
                metadata = values(metadata)
            """
        ),
        {
            "session_id": session_id,
            "video_asset_id": video_asset_id,
            "section": payload.get("link_section") or payload.get("section"),
            "sequence_no": payload.get("link_sequence_no", payload.get("sequence_no")),
            "clip_start_time": payload.get("clip_start_time"),
            "clip_end_time": payload.get("clip_end_time"),
            "is_primary": 1 if payload.get("is_primary") else 0,
            "metadata": json.dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
        },
    )
    db.commit()


def create_transaction(db: Session, session_id: int, payload: Mapping[str, Any]) -> None:
    transaction_table = _table("session_transaction")
    db.execute(
        text(
            f"""
            insert into {transaction_table} (
                session_id, receipt_number, transaction_time, total_items, total_amount, raw_payload
            )
            values (
                :session_id, :receipt_number, :transaction_time, :total_items, :total_amount, :raw_payload
            )
            """
        ),
        {
            "session_id": session_id,
            **payload,
            "raw_payload": json.dumps(payload.get("raw_payload")) if payload.get("raw_payload") is not None else None,
        },
    )
    db.commit()


def create_script_run(
    db: Session,
    *,
    session_id: int | None,
    trigger_id: int | None,
    script_name: str,
    model_name: str | None,
    status: str,
    command: str,
    stdout_log: str,
    stderr_log: str,
) -> None:
    script_run_table = _table("script_run")
    db.execute(
        text(
            f"""
            insert into {script_run_table} (
                session_id, trigger_id, script_name, model_name, status, command, stdout_log, stderr_log, finished_at
            )
            values (
                :session_id, :trigger_id, :script_name, :model_name, :status, :command, :stdout_log, :stderr_log, now()
            )
            """
        ),
        {
            "session_id": session_id,
            "trigger_id": trigger_id,
            "script_name": script_name,
            "model_name": model_name,
            "status": status,
            "command": command,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
        },
    )
    db.commit()


def finalize_session_result(
    db: Session,
    *,
    session_id: int,
    kiosk_total_items: int,
    actual_items_brought: int | None = None,
) -> dict[str, Any]:
    session_table = _table("session")
    transaction_total_items = get_transaction_total_items(db, session_id)
    actual_items = kiosk_total_items if actual_items_brought is None else actual_items_brought

    if kiosk_total_items == transaction_total_items:
        status = "not_detected"
    elif kiosk_total_items > transaction_total_items:
        status = "detected"
    else:
        status = "need_review"

    result_summary = {
        "kiosk_total_items": kiosk_total_items,
        "transaction_total_items": transaction_total_items,
        "actual_items_brought": actual_items,
        "difference": kiosk_total_items - transaction_total_items,
        "decision": status,
    }

    db.execute(
        text(
            f"""
            update {session_table}
            set total_item_brought = :kiosk_total_items,
                actual_items_brought = :actual_items_brought,
                transaction_total_items = :transaction_total_items,
                status = :status,
                result_summary = :result_summary
            where id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "kiosk_total_items": kiosk_total_items,
            "actual_items_brought": actual_items,
            "transaction_total_items": transaction_total_items,
            "status": status,
            "result_summary": json.dumps(result_summary),
        },
    )
    db.commit()

    return {
        "session_id": session_id,
        "status": status,
        "kiosk_total_items": kiosk_total_items,
        "transaction_total_items": transaction_total_items,
        "actual_items_brought": actual_items,
        "result_summary": result_summary,
    }
