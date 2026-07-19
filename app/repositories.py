import json
import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import settings


def _table(name: str) -> str:
    return f"{settings.transactional_table_prefix}{name}"


def _quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def _fetch_one_dict(result) -> dict[str, Any]:
    row = result.mappings().first()
    if row is None:
        raise ValueError("Expected a row but query returned nothing.")
    return dict(row)


def _fetch_all_dicts(result) -> list[dict[str, Any]]:
    return [dict(row) for row in result.mappings().all()]


def _pick_first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _whitelist_source_config(method: str) -> dict[str, str]:
    if method == "qrentry":
        return {
            "table_name": settings.whitelist_qrentry_table_name,
            "value_column": settings.whitelist_qrentry_value_column,
            "label_column": settings.whitelist_qrentry_label_column,
            "display_column": settings.whitelist_qrentry_display_column,
            "create_column": settings.whitelist_qrentry_create_column,
        }

    if method == "entrylogs":
        return {
            "table_name": settings.whitelist_entrylogs_table_name,
            "value_column": settings.whitelist_entrylogs_value_column,
            "label_column": settings.whitelist_entrylogs_label_column,
            "display_column": settings.whitelist_entrylogs_display_column,
        }

    raise ValueError("Unsupported whitelist method.")


def _validate_whitelist_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in {"qrentry", "entrylogs"}:
        raise ValueError("Unsupported whitelist method.")
    return normalized


def _normalize_international_phone_number(value: str) -> str:
    normalized = re.sub(r"[\s\-()]+", "", value.strip())
    if not normalized:
        raise ValueError("Phone number is required.")
    if normalized.startswith("+"):
        normalized = normalized[1:]
    if not re.fullmatch(r"\d{8,15}", normalized):
        raise ValueError("Phone number must include country code and contain only digits, for example 60123456789.")
    if normalized.startswith("0"):
        raise ValueError("Phone number must include country code and should not start with 0, for example 60123456789.")
    return normalized


def get_cctv(db: Session, cctv_id: int) -> dict[str, Any]:
    cctv_table = _table("cctv")
    location_endpoint_table = _table("location_endpoint")
    result = db.execute(
        text(
            f"""
            select c.id,
                   e.location_id as location_id,
                   c.location_endpoint_id,
                   c.section,
                   c.stream_name,
                   c.recorder_channel,
                   c.delayed_seconds,
                   c.created_at,
                   c.updated_at
            from {cctv_table} c
            join {location_endpoint_table} e on e.id = c.location_endpoint_id
            where c.id = :cctv_id
            """
        ),
        {"cctv_id": cctv_id},
    )
    return _fetch_one_dict(result)


def get_cctv_by_location_section(db: Session, *, location_id: int, section: str) -> dict[str, Any]:
    cctv_table = _table("cctv")
    location_endpoint_table = _table("location_endpoint")
    result = db.execute(
        text(
            f"""
            select c.id,
                   e.location_id as location_id,
                   c.location_endpoint_id,
                   c.section,
                   c.stream_name,
                   c.recorder_channel,
                   c.delayed_seconds,
                   c.created_at,
                   c.updated_at
            from {cctv_table} c
            join {location_endpoint_table} e on e.id = c.location_endpoint_id
            where e.location_id = :location_id and c.section = :section
            limit 1
            """
        ),
        {"location_id": location_id, "section": section},
    )
    return _fetch_one_dict(result)


def list_cctv(db: Session, location_id: int | None = None) -> list[dict[str, Any]]:
    cctv_table = _table("cctv")
    location_endpoint_table = _table("location_endpoint")
    if location_id is None:
        result = db.execute(
            text(
                f"""
                select c.id,
                       e.location_id as location_id,
                       c.location_endpoint_id,
                       c.section,
                       c.stream_name,
                       c.recorder_channel,
                       c.delayed_seconds,
                       c.created_at,
                       c.updated_at
                from {cctv_table} c
                join {location_endpoint_table} e on e.id = c.location_endpoint_id
                order by e.location_id asc, c.section asc, c.id asc
                """
            )
        )
        return _fetch_all_dicts(result)

    result = db.execute(
        text(
            f"""
            select c.id,
                   e.location_id as location_id,
                   c.location_endpoint_id,
                   c.section,
                   c.stream_name,
                   c.recorder_channel,
                   c.delayed_seconds,
                   c.created_at,
                   c.updated_at
            from {cctv_table} c
            join {location_endpoint_table} e on e.id = c.location_endpoint_id
            where e.location_id = :location_id
            order by c.section asc, c.id asc
            """
        ),
        {"location_id": location_id},
    )
    return _fetch_all_dicts(result)


def list_theft_transactions(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    table_name = _quote_identifier(settings.theft_transaction_table_name)
    status_column = _quote_identifier(settings.theft_transaction_status_column)

    result = db.execute(
        text(
            f"""
            select *
            from {table_name}
            where {status_column} = :status_value
            limit :limit
            """
        ),
        {
            "status_value": settings.theft_transaction_status_value,
            "limit": limit,
        },
    )
    rows = _fetch_all_dicts(result)

    def sort_key(row: Mapping[str, Any]) -> Any:
        return _pick_first(row, "created_at", "createdAt", "transaction_time", "transactionTime", "updated_at", "updatedAt")

    sorted_rows = sorted(rows, key=lambda row: sort_key(row) or 0, reverse=True)

    payload: list[dict[str, Any]] = []
    for row in sorted_rows:
        item_id = _pick_first(row, "id", "ID", "transaction_id", "transactionId", "receipt_number", "receiptNumber")
        payload.append(
            {
                "id": str(item_id) if item_id is not None else "-",
                "reference": (
                    str(
                        _pick_first(
                            row,
                            "receipt_number",
                            "receiptNumber",
                            "reference",
                            "reference_no",
                            "referenceNo",
                            "transaction_id",
                            "transactionId",
                        )
                    )
                    if _pick_first(
                        row,
                        "receipt_number",
                        "receiptNumber",
                        "reference",
                        "reference_no",
                        "referenceNo",
                        "transaction_id",
                        "transactionId",
                    )
                    is not None
                    else None
                ),
                "location_id": (
                    str(_pick_first(row, "location_id", "locationId", "store_id", "storeId"))
                    if _pick_first(row, "location_id", "locationId", "store_id", "storeId") is not None
                    else None
                ),
                "status": str(_pick_first(row, settings.theft_transaction_status_column) or settings.theft_transaction_status_value),
                "total_amount": _pick_first(row, "total_amount", "totalAmount", "amount", "grand_total", "grandTotal"),
                "created_at": _pick_first(row, "created_at", "createdAt", "transaction_time", "transactionTime", "updated_at", "updatedAt"),
                "metadata": {key: value for key, value in row.items()},
            }
        )

    return payload


def list_locations(db: Session) -> list[dict[str, Any]]:
    table_name = settings.location_table_name
    id_column = settings.location_id_column
    name_column = settings.location_name_column
    location_endpoint_table = _table("location_endpoint")

    result = db.execute(
        text(
            f"""
            select l.{id_column} as id,
                   l.{name_column} as name,
                   e.dahua_host,
                   e.dahua_username,
                   null as dahua_password,
                   e.rtsp_port,
                   e.notes,
                   case when e.id is null then 0 else 1 end as has_endpoint_config,
                   case when e.dahua_password_encrypted is null or e.dahua_password_encrypted = '' then 0 else 1 end as has_password_config
            from {table_name} l
            left join {location_endpoint_table} e on e.location_id = l.{id_column}
            order by {name_column} asc, {id_column} asc
            """
        )
    )
    return _fetch_all_dicts(result)


def get_location_endpoint(db: Session, location_id: int) -> dict[str, Any]:
    table_name = settings.location_table_name
    id_column = settings.location_id_column
    name_column = settings.location_name_column
    location_endpoint_table = _table("location_endpoint")

    result = db.execute(
        text(
            f"""
            select l.{id_column} as id,
                   l.{name_column} as name,
                   e.dahua_host,
                   e.dahua_username,
                   null as dahua_password,
                   e.dahua_password_encrypted,
                   e.rtsp_port,
                   e.notes,
                   case when e.id is null then 0 else 1 end as has_endpoint_config,
                   case when e.dahua_password_encrypted is null or e.dahua_password_encrypted = '' then 0 else 1 end as has_password_config
            from {table_name} l
            left join {location_endpoint_table} e on e.location_id = l.{id_column}
            where l.{id_column} = :location_id
            limit 1
            """
        ),
        {"location_id": location_id},
    )
    return _fetch_one_dict(result)


def upsert_location_endpoint(db: Session, location_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    location_endpoint_table = _table("location_endpoint")
    existing = db.execute(
        text(
            f"""
            select id, dahua_password_encrypted
            from {location_endpoint_table}
            where location_id = :location_id
            limit 1
            """
        ),
        {"location_id": location_id},
    ).mappings().first()

    if existing is None:
        if payload.get("dahua_password_encrypted") is None:
            raise ValueError("Password is required when creating a new location endpoint.")
        db.execute(
            text(
                f"""
                insert into {location_endpoint_table} (
                    location_id, dahua_host, dahua_username, dahua_password_encrypted, rtsp_port, notes
                )
                values (
                    :location_id, :dahua_host, :dahua_username, :dahua_password_encrypted, :rtsp_port, :notes
                )
                """
            ),
            {"location_id": location_id, **payload},
        )
    else:
        update_sql = f"""
            update {location_endpoint_table}
            set dahua_host = :dahua_host,
                dahua_username = :dahua_username,
                rtsp_port = :rtsp_port,
                notes = :notes
        """
        params = {
            "location_id": location_id,
            "dahua_host": payload["dahua_host"],
            "dahua_username": payload["dahua_username"],
            "rtsp_port": payload["rtsp_port"],
            "notes": payload["notes"],
        }
        if payload.get("dahua_password_encrypted") is not None:
            update_sql += ", dahua_password_encrypted = :dahua_password_encrypted"
            params["dahua_password_encrypted"] = payload["dahua_password_encrypted"]
        update_sql += " where location_id = :location_id"
        db.execute(text(update_sql), params)

    db.commit()
    return get_location_endpoint(db, location_id)


def get_location_endpoint_by_location_id(db: Session, location_id: int) -> dict[str, Any]:
    location_endpoint_table = _table("location_endpoint")
    result = db.execute(
        text(
            f"""
            select id, location_id, dahua_host, dahua_username, dahua_password_encrypted, rtsp_port, notes, created_at, updated_at
            from {location_endpoint_table}
            where location_id = :location_id
            limit 1
            """
        ),
        {"location_id": location_id},
    )
    return _fetch_one_dict(result)


def delete_location_endpoint(db: Session, location_id: int) -> bool:
    location_endpoint_table = _table("location_endpoint")
    cctv_table = _table("cctv")

    linked_cctv_count = db.execute(
        text(
            f"""
            select count(*) as total
            from {cctv_table} c
            join {location_endpoint_table} e on e.id = c.location_endpoint_id
            where e.location_id = :location_id
            """
        ),
        {"location_id": location_id},
    ).scalar_one()

    if int(linked_cctv_count or 0) > 0:
        raise ValueError("Delete the CCTV rows for this location before deleting the NVR.")

    result = db.execute(
        text(
            f"""
            delete from {location_endpoint_table}
            where location_id = :location_id
            """
        ),
        {"location_id": location_id},
    )
    db.commit()
    return bool(result.rowcount)


def list_whitelist_entries(db: Session) -> list[dict[str, Any]]:
    whitelist_table = _table("whitelist_entry")
    qrentry = _whitelist_source_config("qrentry")
    entrylogs = _whitelist_source_config("entrylogs")

    result = db.execute(
        text(
            f"""
            select w.id, w.method, w.entry_id, w.status, w.created_at, w.updated_at,
                   case
                       when w.method = 'qrentry' then (
                           select cast(q.{qrentry["display_column"]} as char)
                           from {qrentry["table_name"]} q
                           where cast(q.{qrentry["value_column"]} as char) = w.entry_id
                           limit 1
                       )
                       when w.method = 'entrylogs' then (
                           select cast(e.{entrylogs["display_column"]} as char)
                           from {entrylogs["table_name"]} e
                           where cast(e.{entrylogs["value_column"]} as char) = w.entry_id
                           limit 1
                       )
                       else null
                   end as resolved_value
            from {whitelist_table} w
            order by w.created_at desc, w.id desc
            """
        )
    )
    return _fetch_all_dicts(result)


def create_whitelist_entry(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    whitelist_table = _table("whitelist_entry")
    result = db.execute(
        text(
            f"""
            insert into {whitelist_table} (
                method, entry_id, status
            )
            values (
                :method, :entry_id, :status
            )
            """
        ),
        payload,
    )
    db.commit()
    whitelist_id = int(result.lastrowid)
    rows = [row for row in list_whitelist_entries(db) if int(row["id"]) == whitelist_id]
    if not rows:
        raise ValueError("Whitelist entry not found after create.")
    return rows[0]


def delete_whitelist_entry(db: Session, whitelist_id: int) -> bool:
    whitelist_table = _table("whitelist_entry")
    result = db.execute(
        text(
            f"""
            delete from {whitelist_table}
            where id = :whitelist_id
            """
        ),
        {"whitelist_id": whitelist_id},
    )
    db.commit()
    return bool(result.rowcount)


def list_whitelist_source_options(
    db: Session,
    method: str,
    *,
    search: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    method = _validate_whitelist_method(method)
    source = _whitelist_source_config(method)
    search_value = f"%{search.strip()}%" if search and search.strip() else None
    result = db.execute(
        text(
            f"""
            select cast({source["value_column"]} as char) as value,
                   cast({source["label_column"]} as char) as label,
                   case
                       when {source["display_column"]} = {source["label_column"]} then null
                       else cast({source["display_column"]} as char)
                   end as secondary_label,
                   :method as method
            from {source["table_name"]}
            where {source["value_column"]} is not null
              and (
                  :search_value is null
                  or cast({source["value_column"]} as char) like :search_value
                  or cast({source["label_column"]} as char) like :search_value
                  or cast({source["display_column"]} as char) like :search_value
              )
            order by {source["label_column"]} asc
            limit :limit_value
            """
        ),
        {
            "method": method,
            "search_value": search_value,
            "limit_value": limit,
        },
    )
    return _fetch_all_dicts(result)


def create_phone_number_source(db: Session, phone_number: str) -> dict[str, Any]:
    source = _whitelist_source_config("qrentry")
    normalized_phone_number = _normalize_international_phone_number(phone_number)

    existing_result = db.execute(
        text(
            f"""
            select cast({source["value_column"]} as char) as value,
                   cast({source["label_column"]} as char) as label,
                   case
                       when {source["display_column"]} = {source["label_column"]} then null
                       else cast({source["display_column"]} as char)
                   end as secondary_label,
                   'qrentry' as method
            from {source["table_name"]}
            where cast({source["create_column"]} as char) = :phone_number
            limit 1
            """
        ),
        {"phone_number": normalized_phone_number},
    ).mappings().first()

    if existing_result is not None:
        raise ValueError("This phone number already exists.")

    db.execute(
        text(
            f"""
            insert into {source["table_name"]} (
                {source["create_column"]}
            )
            values (
                :phone_number
            )
            """
        ),
        {"phone_number": normalized_phone_number},
    )
    db.commit()

    created_result = db.execute(
        text(
            f"""
            select cast({source["value_column"]} as char) as value,
                   cast({source["label_column"]} as char) as label,
                   case
                       when {source["display_column"]} = {source["label_column"]} then null
                       else cast({source["display_column"]} as char)
                   end as secondary_label,
                   'qrentry' as method
            from {source["table_name"]}
            where cast({source["create_column"]} as char) = :phone_number
            limit 1
            """
        ),
        {"phone_number": normalized_phone_number},
    )
    return _fetch_one_dict(created_result)


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


def get_worker_control(db: Session, worker_name: str) -> dict[str, Any]:
    worker_control_table = _table("worker_control")
    result = db.execute(
        text(
            f"""
            select worker_name, paused, paused_at, resumed_at, created_at, updated_at
            from {worker_control_table}
            where worker_name = :worker_name
            limit 1
            """
        ),
        {"worker_name": worker_name},
    )
    row = result.mappings().first()
    if row is None:
        return {
            "worker_name": worker_name,
            "paused": False,
            "paused_at": None,
            "resumed_at": None,
            "created_at": None,
            "updated_at": None,
        }
    return dict(row)


def is_worker_paused(db: Session, worker_name: str) -> bool:
    row = get_worker_control(db, worker_name)
    return bool(row.get("paused", False))


def set_worker_paused(db: Session, worker_name: str, paused: bool) -> dict[str, Any]:
    worker_control_table = _table("worker_control")
    db.execute(
        text(
            f"""
            insert into {worker_control_table} (
                worker_name, paused, paused_at, resumed_at
            )
            values (
                :worker_name,
                :paused,
                case when :paused = 1 then now() else null end,
                case when :paused = 0 then now() else null end
            )
            on duplicate key update
                paused = values(paused),
                paused_at = case when values(paused) = 1 then now() else paused_at end,
                resumed_at = case when values(paused) = 0 then now() else resumed_at end
            """
        ),
        {
            "worker_name": worker_name,
            "paused": 1 if paused else 0,
        },
    )
    db.commit()
    return get_worker_control(db, worker_name)


def list_triggers(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    trigger_table = _table("trigger_event")
    script_run_table = _table("script_run")
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            select id, location_id, aqara_event_id, trigger_source, trigger_time,
                   entry_source_type, entry_match_status, status, whitelist_hit,
                   issue_reason,
                   (
                       select sr.script_name
                       from {script_run_table} sr
                       where sr.trigger_id = te.id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_name,
                   (
                       select sr.status
                       from {script_run_table} sr
                       where sr.trigger_id = te.id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_status,
                   (
                       select sr.finished_at
                       from {script_run_table} sr
                       where sr.trigger_id = te.id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_finished_at,
                   (
                       select nullif(trim(sr.stderr_log), '')
                       from {script_run_table} sr
                       where sr.trigger_id = te.id
                         and sr.status = 'failed'
                       order by sr.id desc
                       limit 1
                   ) as latest_error_log,
                   (
                       select va.id
                       from {video_asset_table} va
                       where va.trigger_id = te.id
                       order by coalesce(va.captured_start_time, va.created_at) desc, va.id desc
                       limit 1
                   ) as latest_video_asset_id,
                   (
                       select va.status
                       from {video_asset_table} va
                       where va.trigger_id = te.id
                       order by coalesce(va.captured_start_time, va.created_at) desc, va.id desc
                       limit 1
                   ) as latest_video_status,
                   exists(
                       select 1
                       from {video_asset_table} issue_va
                       where issue_va.trigger_id = te.id
                         and issue_va.status = 'issue'
                       limit 1
                   ) as can_retry,
                   case
                       when (
                           select sr.script_name
                           from {script_run_table} sr
                           where sr.trigger_id = te.id
                             and sr.status = 'failed'
                           order by sr.id desc
                           limit 1
                       ) = 'entry' then 'ready'
                       when exists(
                           select 1
                           from {video_asset_table} issue_va
                           where issue_va.trigger_id = te.id
                             and issue_va.status = 'issue'
                           limit 1
                       ) then 'not_retrieved'
                       else null
                   end as retry_to_status,
                   created_at, updated_at
            from {trigger_table} te
            order by trigger_time desc, id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def retry_trigger_issue(db: Session, trigger_id: int) -> dict[str, Any]:
    trigger = get_trigger(db, trigger_id)
    video_asset_table = _table("video_asset")
    script_run_table = _table("script_run")
    latest_failed_script = db.execute(
        text(
            f"""
            select script_name
            from {script_run_table}
            where trigger_id = :trigger_id
              and status = 'failed'
            order by id desc
            limit 1
            """
        ),
        {"trigger_id": trigger_id},
    ).mappings().first()
    issue_video = db.execute(
        text(
            f"""
            select id, trigger_id, section, sequence_no, video_url, file_path,
                   captured_start_time, captured_end_time, retrieved_at, analyzed_at,
                   retention_until, status, metadata, created_at
            from {video_asset_table}
            where trigger_id = :trigger_id
              and status = 'issue'
            order by coalesce(captured_start_time, created_at) desc, id desc
            limit 1
            """
        ),
        {"trigger_id": trigger_id},
    ).mappings().first()
    if issue_video is None:
        raise ValueError("This trigger does not have an issue video to retry.")

    retry_to_status = "ready" if latest_failed_script and latest_failed_script["script_name"] == "entry" else "not_retrieved"
    update_video_asset(
        db,
        int(issue_video["id"]),
        {
            "video_url": issue_video.get("video_url"),
            "file_path": issue_video.get("file_path"),
            "captured_start_time": issue_video.get("captured_start_time"),
            "captured_end_time": issue_video.get("captured_end_time"),
            "retrieved_at": None if retry_to_status == "not_retrieved" else issue_video.get("retrieved_at"),
            "analyzed_at": None,
            "retention_until": issue_video.get("retention_until"),
            "status": retry_to_status,
            "metadata": issue_video.get("metadata"),
        },
    )
    return {
        "ok": True,
        "trigger_id": trigger_id,
        "location_id": trigger["location_id"],
        "video_asset_id": int(issue_video["id"]),
        "new_status": retry_to_status,
    }


def get_video_asset(db: Session, video_asset_id: int) -> dict[str, Any]:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            select id, trigger_id, section, sequence_no, video_url, file_path,
                   captured_start_time, captured_end_time, retrieved_at, analyzed_at,
                   retention_until, status, metadata, created_at
            from {video_asset_table}
            where id = :video_asset_id
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    return _fetch_one_dict(result)


def get_video_asset_by_file_path(db: Session, file_path: str) -> dict[str, Any]:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            select id, trigger_id, section, sequence_no, video_url, file_path,
                   captured_start_time, captured_end_time, retrieved_at, analyzed_at,
                   retention_until, status, metadata, created_at
            from {video_asset_table}
            where file_path = :file_path
            order by id desc
            limit 1
            """
        ),
        {"file_path": file_path},
    )
    return _fetch_one_dict(result)


def list_pending_video_asset_retrievals(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select va.id,
                   va.trigger_id,
                   va.section,
                   va.file_path,
                   va.captured_start_time,
                   va.captured_end_time,
                   va.retrieved_at,
                   va.analyzed_at,
                   va.created_at,
                   min(sva.session_id) as session_id,
                   coalesce(te.location_id, min(s.location_id)) as location_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.status = 'not_retrieved'
            group by va.id, va.trigger_id, va.section, va.file_path, va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.created_at, te.location_id
            order by va.created_at asc, va.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def list_running_video_asset_retrievals(db: Session) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select va.id,
                   va.trigger_id,
                   va.section,
                   va.file_path,
                   va.captured_start_time,
                   va.captured_end_time,
                   va.retrieved_at,
                   va.analyzed_at,
                   min(sva.session_id) as session_id,
                   coalesce(te.location_id, min(s.location_id)) as location_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.status = 'retrieving'
            group by va.id, va.trigger_id, va.section, va.file_path, va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, te.location_id
            order by va.id asc
            """
        )
    )
    return _fetch_all_dicts(result)


def claim_video_asset_for_retrieval(db: Session, video_asset_id: int) -> bool:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            update {video_asset_table}
            set status = 'retrieving'
            where id = :video_asset_id and status = 'not_retrieved'
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    db.commit()
    return bool(result.rowcount)


def list_pending_video_asset_analyses(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select va.id,
                   va.trigger_id,
                   va.section,
                   va.file_path,
                   va.video_url,
                   va.captured_start_time,
                   va.captured_end_time,
                   va.retrieved_at,
                   va.analyzed_at,
                   va.created_at,
                   te.location_id
            from {video_asset_table} va
            inner join {trigger_table} te on te.id = va.trigger_id
            where va.section = 'entrance'
              and va.status = 'ready'
              and not exists (
                  select 1
                  from {video_asset_table} prev
                  inner join {trigger_table} prev_te on prev_te.id = prev.trigger_id
                  where prev.section = 'entrance'
                    and prev_te.location_id = te.location_id
                    and (
                        coalesce(prev.captured_start_time, prev.created_at) < coalesce(va.captured_start_time, va.created_at)
                        or (
                            coalesce(prev.captured_start_time, prev.created_at) = coalesce(va.captured_start_time, va.created_at)
                            and prev.id < va.id
                        )
                    )
                    and prev.status in ('not_retrieved', 'retrieving', 'ready', 'processing', 'issue')
              )
            order by coalesce(va.captured_start_time, va.created_at) asc, va.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def list_running_video_asset_analyses(db: Session) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select va.id,
                   va.trigger_id,
                   va.section,
                   va.file_path,
                   va.video_url,
                   va.captured_start_time,
                   va.captured_end_time,
                   va.retrieved_at,
                   va.analyzed_at,
                   va.created_at,
                   te.location_id
            from {video_asset_table} va
            inner join {trigger_table} te on te.id = va.trigger_id
            where va.section = 'entrance'
              and va.status = 'processing'
            order by va.id asc
            """
        )
    )
    return _fetch_all_dicts(result)


def claim_video_asset_for_analysis(db: Session, video_asset_id: int) -> bool:
    video_asset_table = _table("video_asset")
    result = db.execute(
        text(
            f"""
            update {video_asset_table}
            set status = 'processing'
            where id = :video_asset_id
              and section = 'entrance'
              and status = 'ready'
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    db.commit()
    return bool(result.rowcount)


def list_location_analysis_heads(db: Session) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select head.id,
                   head.trigger_id,
                   head.status,
                   head.captured_start_time,
                   head.created_at,
                   te.location_id
            from {video_asset_table} head
            inner join {trigger_table} te on te.id = head.trigger_id
            where head.section = 'entrance'
              and head.status in ('not_retrieved', 'retrieving', 'ready', 'processing', 'issue')
              and not exists (
                  select 1
                  from {video_asset_table} prev
                  inner join {trigger_table} prev_te on prev_te.id = prev.trigger_id
                  where prev.section = 'entrance'
                    and prev.status in ('not_retrieved', 'retrieving', 'ready', 'processing', 'issue')
                    and prev_te.location_id = te.location_id
                    and (
                        coalesce(prev.captured_start_time, prev.created_at) < coalesce(head.captured_start_time, head.created_at)
                        or (
                            coalesce(prev.captured_start_time, prev.created_at) = coalesce(head.captured_start_time, head.created_at)
                            and prev.id < head.id
                        )
                    )
              )
            order by te.location_id asc
            """
        )
    )
    return _fetch_all_dicts(result)


def list_video_assets(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    session_video_asset_table = _table("session_video_asset")
    trigger_table = _table("trigger_event")
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                   va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.retention_until, va.status,
                   va.metadata, va.created_at,
                   coalesce(te.location_id, min(s.location_id)) as location_id,
                   count(distinct sva.id) as session_link_count,
                   min(sva.session_id) as primary_session_id,
                   group_concat(distinct sva.session_id order by sva.session_id separator ',') as session_ids
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            group by va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                     va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.retention_until, va.status,
                     va.metadata, va.created_at, te.location_id
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


def get_session_by_entry_trigger_id(db: Session, entry_trigger_id: int) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer
            from {session_table}
            where entry_trigger_id = :entry_trigger_id
            order by id asc
            limit 1
            """
        ),
        {"entry_trigger_id": entry_trigger_id},
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
    location_endpoint = get_location_endpoint_by_location_id(db, int(payload["location_id"]))
    result = db.execute(
        text(
            f"""
            insert into {cctv_table} (
                location_endpoint_id, section, stream_name, recorder_channel, delayed_seconds
            )
            values (
                :location_endpoint_id, :section, :stream_name, :recorder_channel, :delayed_seconds
            )
            """
        ),
        {
            "location_endpoint_id": int(location_endpoint["id"]),
            "section": payload["section"],
            "stream_name": payload.get("stream_name"),
            "recorder_channel": payload.get("recorder_channel"),
            "delayed_seconds": payload.get("delayed_seconds", 0),
        },
    )
    db.commit()
    return get_cctv(db, int(result.lastrowid))


def update_cctv(db: Session, cctv_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    cctv_table = _table("cctv")
    location_endpoint = get_location_endpoint_by_location_id(db, int(payload["location_id"]))
    result = db.execute(
        text(
            f"""
            update {cctv_table}
            set location_endpoint_id = :location_endpoint_id,
                section = :section,
                stream_name = :stream_name,
                recorder_channel = :recorder_channel,
                delayed_seconds = :delayed_seconds
            where id = :cctv_id
            """
        ),
        {
            "cctv_id": cctv_id,
            "location_endpoint_id": int(location_endpoint["id"]),
            "section": payload["section"],
            "stream_name": payload.get("stream_name"),
            "recorder_channel": payload.get("recorder_channel"),
            "delayed_seconds": payload.get("delayed_seconds", 0),
        },
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
    file_path = payload.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        try:
            existing_row = get_video_asset_by_file_path(db, file_path.strip())
        except ValueError:
            existing_row = None
        if existing_row is not None:
            return int(existing_row["id"])
    result = db.execute(
        text(
            f"""
            insert into {video_asset_table} (
                trigger_id, section, sequence_no, video_url, file_path, captured_start_time,
                captured_end_time, retrieved_at, analyzed_at, retention_until, status, metadata
            )
            values (
                :trigger_id, :section, :sequence_no, :video_url, :file_path, :captured_start_time,
                :captured_end_time, :retrieved_at, :analyzed_at, :retention_until, :status, :metadata
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


def update_video_asset_status(db: Session, video_asset_id: int, status: str, metadata: Mapping[str, Any] | None = None) -> None:
    video_asset_table = _table("video_asset")
    db.execute(
        text(
            f"""
            update {video_asset_table}
            set status = :status,
                metadata = coalesce(:metadata, metadata)
            where id = :video_asset_id
            """
        ),
        {
            "video_asset_id": video_asset_id,
            "status": status,
            "metadata": json.dumps(metadata) if metadata is not None else None,
        },
    )
    db.commit()


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


def update_video_asset(db: Session, video_asset_id: int, payload: Mapping[str, Any]) -> None:
    video_asset_table = _table("video_asset")
    db.execute(
        text(
            f"""
            update {video_asset_table}
            set video_url = :video_url,
                file_path = :file_path,
                captured_start_time = :captured_start_time,
                captured_end_time = :captured_end_time,
                retrieved_at = :retrieved_at,
                analyzed_at = :analyzed_at,
                retention_until = :retention_until,
                status = :status,
                metadata = :metadata
            where id = :video_asset_id
            """
        ),
        {
            "video_asset_id": video_asset_id,
            "video_url": payload.get("video_url"),
            "file_path": payload.get("file_path"),
            "captured_start_time": payload.get("captured_start_time"),
            "captured_end_time": payload.get("captured_end_time"),
            "retrieved_at": payload.get("retrieved_at"),
            "analyzed_at": payload.get("analyzed_at"),
            "retention_until": payload.get("retention_until"),
            "status": payload.get("status"),
            "metadata": json.dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
        },
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
