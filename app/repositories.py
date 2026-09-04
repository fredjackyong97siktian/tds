import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .config import settings

PAID_TRANSACTION_ID_COLUMN = "receiptNumber"
PAID_TRANSACTION_TIME_COLUMN = "Formatted Timestamp"
PAID_TRANSACTION_DATABASE = "sesamedb"
_COLUMN_EXISTS_CACHE: dict[tuple[str, str], bool] = {}


def _table(name: str) -> str:
    if name.startswith(settings.transactional_table_prefix):
        return name
    return f"{settings.transactional_table_prefix}{name}"


def _quote_identifier(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def _qualified_paid_table(name: str) -> str:
    return f"{_quote_identifier(PAID_TRANSACTION_DATABASE)}.{_quote_identifier(name)}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _column_exists(db: Session, table_name: str, column_name: str) -> bool:
    normalized_table = table_name.split(".")[-1].strip("`")
    cache_key = (normalized_table, column_name)
    if cache_key in _COLUMN_EXISTS_CACHE:
        return _COLUMN_EXISTS_CACHE[cache_key]
    result = db.execute(
        text(
            """
            select count(*) as column_count
            from information_schema.columns
            where table_schema = database()
              and table_name = :table_name
              and column_name = :column_name
            """
        ),
        {"table_name": normalized_table, "column_name": column_name},
    )
    exists = int(result.scalar() or 0) > 0
    _COLUMN_EXISTS_CACHE[cache_key] = exists
    return exists


def _script_run_has_cost_columns(db: Session, script_run_table: str) -> bool:
    return all(
        _column_exists(db, script_run_table, column_name)
        for column_name in ("cost_amount", "cost_currency", "cost_source")
    )


def _script_run_cost_select(db: Session, script_run_table: str) -> str:
    if _script_run_has_cost_columns(db, script_run_table):
        return "cost_amount, cost_currency, cost_source"
    return "null as cost_amount, 'USD' as cost_currency, null as cost_source"


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
            "id_column": settings.whitelist_qrentry_id_column,
            "value_column": settings.whitelist_qrentry_value_column,
            "label_column": settings.whitelist_qrentry_label_column,
            "display_column": settings.whitelist_qrentry_display_column,
            "create_column": settings.whitelist_qrentry_create_column,
        }

    if method == "entrylogs":
        return {
            "table_name": settings.whitelist_entrylogs_table_name,
            "id_column": settings.whitelist_entrylogs_id_column,
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


def _resolve_entry_source_value(
    db: Session,
    *,
    source: Mapping[str, str],
    entry_id: Any,
) -> str | None:
    row = db.execute(
        text(
            f"""
            select cast({source["display_column"]} as char) as resolved_value
            from {source["table_name"]}
            where cast({source["value_column"]} as char) = :entry_id
               or cast({source["id_column"]} as char) = :entry_id
            limit 1
            """
        ),
        {"entry_id": str(entry_id)},
    ).mappings().first()
    if row and row.get("resolved_value") is not None:
        return str(row["resolved_value"])
    return None


def _resolve_entry_source_row(
    db: Session,
    *,
    source: Mapping[str, str],
    entry_id: Any,
) -> dict[str, Any] | None:
    row = db.execute(
        text(
            f"""
            select *
            from {source["table_name"]}
            where cast({source["value_column"]} as char) = :entry_id
               or cast({source["id_column"]} as char) = :entry_id
            limit 1
            """
        ),
        {"entry_id": str(entry_id)},
    ).mappings().first()
    return dict(row) if row else None


def get_phone_entry_identity(db: Session, phone_entry_id: Any) -> dict[str, Any] | None:
    source = _whitelist_source_config("qrentry")
    row = _resolve_entry_source_row(db, source=source, entry_id=phone_entry_id)
    if row is None:
        try:
            fallback_row = db.execute(
                text(
                    """
                    select id, participantId
                    from qrentry
                    where cast(id as char) = :entry_id
                       or cast(participantId as char) = :entry_id
                    limit 1
                    """
                ),
                {"entry_id": str(phone_entry_id)},
            ).mappings().first()
        except SQLAlchemyError:
            fallback_row = None
        row = dict(fallback_row) if fallback_row else None
    if row is None:
        return None
    return {
        "id": row.get(source["id_column"]) or row.get("id"),
        "entry_id": phone_entry_id,
        "phone_number": row.get(source["display_column"]) or row.get("participantId"),
        "raw": dict(row),
    }


def get_credit_card_entry_identity(db: Session, credit_card_entry_id: Any) -> dict[str, Any] | None:
    source = _whitelist_source_config("entrylogs")
    row = _resolve_entry_source_row(db, source=source, entry_id=credit_card_entry_id)
    if row is None:
        return None
    country = _pick_first(row, "country", "country_code", "countryCode", "card_country", "issuer_country", "cardCountry")
    return {
        "id": row.get(source["id_column"]),
        "entry_id": credit_card_entry_id,
        "fingerprint": row.get(source["display_column"]) or row.get(source["value_column"]),
        "country": country,
        "payment_method_id": _pick_first(row, "paymentMethodId", "payment_method_id", "paymentMethod"),
        "charge_id": _pick_first(row, "charge", "chargeId", "charge_id"),
        "payment_intent_id": _pick_first(row, "paymentIntentId", "payment_intent_id", "payment_intent", "stripId"),
        "client_secret": _pick_first(row, "clientSecret", "client_secret"),
        "customer_stripe_id": _pick_first(row, "customerStripeId", "customer_stripe_id"),
        "last4": _pick_first(row, "last4"),
        "raw": dict(row),
    }


def _resolve_trigger_entry_identity(
    db: Session,
    *,
    phone_entry_id: Any | None,
    credit_card_entry_id: Any | None,
) -> tuple[str | None, str | None]:
    if phone_entry_id is not None:
        source = _whitelist_source_config("qrentry")
        resolved_value = _resolve_entry_source_value(db, source=source, entry_id=phone_entry_id)
        return "Phone Number", resolved_value if resolved_value is not None else str(phone_entry_id)

    if credit_card_entry_id is not None:
        source = _whitelist_source_config("entrylogs")
        resolved_value = _resolve_entry_source_value(db, source=source, entry_id=credit_card_entry_id)
        return "Fingerprint", resolved_value if resolved_value is not None else str(credit_card_entry_id)

    return None, None


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


def _parse_session_id_from_alert_detail(detail: Any) -> int | None:
    if detail is None:
        return None
    text = str(detail).strip()
    match = re.search(r"Session\s+(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def list_thief_alerts(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    result = db.execute(
        text(
            f"""
            select id,
                   locationId as location_id,
                   method,
                   detail,
                   {checked_column} as checked,
                   createdAt as created_at
            from {table_name}
            order by createdAt desc, id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        row["checked"] = bool(row.get("checked"))
        if str(row.get("method") or "").strip().lower() == "tds system":
            row["session_id"] = _parse_session_id_from_alert_detail(row.get("detail"))
        else:
            row["session_id"] = None
    return rows


def get_thief_alert(db: Session, alert_id: int) -> dict[str, Any]:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    result = db.execute(
        text(
            f"""
            select id,
                   locationId as location_id,
                   method,
                   detail,
                   {checked_column} as checked,
                   createdAt as created_at
            from {table_name}
            where id = :alert_id
            limit 1
            """
        ),
        {"alert_id": alert_id},
    )
    row = _fetch_one_dict(result)
    row["checked"] = bool(row.get("checked"))
    if str(row.get("method") or "").strip().lower() == "tds system":
        row["session_id"] = _parse_session_id_from_alert_detail(row.get("detail"))
    else:
        row["session_id"] = None
    return row


def get_unchecked_thief_alert_count(db: Session) -> int:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    result = db.execute(
        text(
            f"""
            select count(*) as alert_count
            from {table_name}
            where coalesce({checked_column}, 0) = 0
            """
        )
    )
    row = result.mappings().first()
    return int((row or {}).get("alert_count") or 0)


def mark_thief_alert_checked(db: Session, alert_id: int) -> dict[str, Any]:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    result = db.execute(
        text(
            f"""
            update {table_name}
            set {checked_column} = 1
            where id = :alert_id
            """
        ),
        {"alert_id": alert_id},
    )
    db.commit()
    if result.rowcount == 0:
        raise ValueError("Thief alert not found.")
    return get_thief_alert(db, alert_id)


def create_thief_alert(
    db: Session,
    *,
    location_id: int,
    method: str,
    detail: str,
) -> int:
    table_name = _table(settings.thief_alert_table_name)
    result = db.execute(
        text(
            f"""
            insert into {table_name} (
                locationId, method, detail, checked, createdAt
            )
            values (
                :location_id, :method, :detail, 0, utc_timestamp()
            )
            """
        ),
        {
            "location_id": location_id,
            "method": method,
            "detail": detail,
        },
    )
    db.commit()
    inserted_id = getattr(result, "lastrowid", None)
    if inserted_id is None:
        raise ValueError("Failed to create thief alert.")
    return int(inserted_id)


def create_thief_alert_if_missing(
    db: Session,
    *,
    location_id: int,
    method: str,
    detail: str,
) -> int | None:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    existing = db.execute(
        text(
            f"""
            select id
            from {table_name}
            where locationId = :location_id
              and method = :method
              and detail = :detail
              and coalesce({checked_column}, 0) = 0
            order by id desc
            limit 1
            """
        ),
        {
            "location_id": location_id,
            "method": method,
            "detail": detail,
        },
    ).mappings().first()
    if existing is not None:
        return int(existing["id"])
    return create_thief_alert(db, location_id=location_id, method=method, detail=detail)


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


def _ensure_list_entry_not_conflicting(
    db: Session,
    *,
    target_table: str,
    target_label: str,
    method: str,
    entry_id: str,
) -> None:
    result = db.execute(
        text(
            f"""
            select id
            from {target_table}
            where method = :method
              and entry_id = :entry_id
            limit 1
            """
        ),
        {"method": method, "entry_id": entry_id},
    )
    if result.mappings().first():
        raise ValueError(f"This entry is already active in the {target_label}. Remove it there before adding it here.")


def create_whitelist_entry(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    whitelist_table = _table("whitelist_entry")
    blacklist_table = _table("blacklist_entry")
    method = _validate_whitelist_method(str(payload.get("method") or ""))
    entry_id = str(payload.get("entry_id") or "").strip()
    if not entry_id:
        raise ValueError("Entry ID is required.")
    _ensure_list_entry_not_conflicting(
        db,
        target_table=blacklist_table,
        target_label="blacklist",
        method=method,
        entry_id=entry_id,
    )
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
        {
            "method": method,
            "entry_id": entry_id,
            "status": str(payload.get("status") or "active"),
        },
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


def list_blacklist_entries(db: Session) -> list[dict[str, Any]]:
    blacklist_table = _table("blacklist_entry")
    qrentry = _whitelist_source_config("qrentry")
    entrylogs = _whitelist_source_config("entrylogs")

    result = db.execute(
        text(
            f"""
            select b.id, b.method, b.entry_id, b.criteria, b.status, b.created_at, b.updated_at,
                   case
                       when b.method = 'qrentry' then (
                           select cast(q.{qrentry["display_column"]} as char)
                           from {qrentry["table_name"]} q
                           where cast(q.{qrentry["value_column"]} as char) = b.entry_id
                           limit 1
                       )
                       when b.method = 'entrylogs' then (
                           select cast(e.{entrylogs["display_column"]} as char)
                           from {entrylogs["table_name"]} e
                           where cast(e.{entrylogs["value_column"]} as char) = b.entry_id
                           limit 1
                       )
                       else null
                   end as resolved_value
            from {blacklist_table} b
            order by b.created_at desc, b.id desc
            """
        )
    )
    return _fetch_all_dicts(result)


def create_blacklist_entry(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    blacklist_table = _table("blacklist_entry")
    whitelist_table = _table("whitelist_entry")
    method = _validate_whitelist_method(str(payload.get("method") or ""))
    entry_id = str(payload.get("entry_id") or "").strip()
    criteria = str(payload.get("criteria") or "").strip()
    if not entry_id:
        raise ValueError("Entry ID is required.")
    if not criteria:
        raise ValueError("Blacklist criteria is required.")
    _ensure_list_entry_not_conflicting(
        db,
        target_table=whitelist_table,
        target_label="whitelist",
        method=method,
        entry_id=entry_id,
    )
    result = db.execute(
        text(
            f"""
            insert into {blacklist_table} (
                method, entry_id, criteria, status
            )
            values (
                :method, :entry_id, :criteria, :status
            )
            """
        ),
        {
            "method": method,
            "entry_id": entry_id,
            "criteria": criteria,
            "status": str(payload.get("status") or "active"),
        },
    )
    db.commit()
    blacklist_id = int(result.lastrowid)
    rows = [row for row in list_blacklist_entries(db) if int(row["id"]) == blacklist_id]
    if not rows:
        raise ValueError("Blacklist entry not found after create.")
    return rows[0]


def delete_blacklist_entry(db: Session, blacklist_id: int) -> bool:
    blacklist_table = _table("blacklist_entry")
    result = db.execute(
        text(
            f"""
            delete from {blacklist_table}
            where id = :blacklist_id
            """
        ),
        {"blacklist_id": blacklist_id},
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
            select id, location_id, phone_entry_id, credit_card_entry_id, aqara_event_id, trigger_source, trigger_time,
                   phone_entry_id, credit_card_entry_id, entry_source_type, entry_match_status,
                   status, whitelist_hit, raw_payload, issue_reason, created_at, updated_at,
                   unique_customer_count, unique_customer_count_confidence, unique_customer_count_source
            from {trigger_table}
            where id = :trigger_id
            """
        ),
        {"trigger_id": trigger_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("raw_payload"), str):
        try:
            row["raw_payload"] = json.loads(row["raw_payload"])
        except json.JSONDecodeError:
            pass
    label, value = _resolve_trigger_entry_identity(
        db,
        phone_entry_id=row.get("phone_entry_id"),
        credit_card_entry_id=row.get("credit_card_entry_id"),
    )
    row["resolved_entry_label"] = label
    row["resolved_entry_value"] = value
    phone_identity = get_phone_entry_identity(db, row.get("phone_entry_id")) if row.get("phone_entry_id") is not None else None
    card_identity = (
        get_credit_card_entry_identity(db, row.get("credit_card_entry_id"))
        if row.get("credit_card_entry_id") is not None
        else None
    )
    row["resolved_phone_number"] = phone_identity.get("phone_number") if phone_identity else None
    row["resolved_card_fingerprint"] = card_identity.get("fingerprint") if card_identity else None
    return row


def get_app_setting(db: Session, setting_key: str) -> str | None:
    setting_table = _table("app_setting")
    result = db.execute(
        text(f"select setting_value from {setting_table} where setting_key = :setting_key limit 1"),
        {"setting_key": setting_key},
    )
    row = result.first()
    return row[0] if row is not None else None


def set_app_setting(db: Session, setting_key: str, setting_value: str) -> None:
    setting_table = _table("app_setting")
    db.execute(
        text(
            f"""
            insert into {setting_table} (setting_key, setting_value)
            values (:setting_key, :setting_value)
            on duplicate key update setting_value = values(setting_value)
            """
        ),
        {"setting_key": setting_key, "setting_value": setting_value},
    )
    db.commit()


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


def list_triggers(db: Session, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
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
                   (
                       select va.metadata
                       from {video_asset_table} va
                       where va.trigger_id = te.id
                       order by coalesce(va.captured_start_time, va.created_at) desc, va.id desc
                       limit 1
                   ) as latest_video_metadata,
                   exists(
                       select 1
                       from {video_asset_table} issue_va
                       where issue_va.trigger_id = te.id
                         and issue_va.status = 'issue'
                       limit 1
                   ) as can_retry,
                   case
                       when lower(coalesce((
                           select va.section
                           from {video_asset_table} va
                           where va.trigger_id = te.id
                             and va.status = 'issue'
                           order by coalesce(va.captured_start_time, va.created_at) desc, va.id desc
                           limit 1
                       ), '')) = 'kiosk' then 'ready'
                       when lower(coalesce((
                           select sr.script_name
                           from {script_run_table} sr
                           where sr.trigger_id = te.id
                             and sr.status = 'failed'
                           order by sr.id desc
                           limit 1
                       ), '')) in ('entry', 'kiosk') then 'ready'
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
            offset :offset
            """
        ),
        {"limit": limit, "offset": max(0, offset)},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        metadata = row.pop("latest_video_metadata", None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None
        frames = metadata.get("frames") if isinstance(metadata, Mapping) else None
        row["trigger_frames"] = frames if isinstance(frames, list) else []
        label, value = _resolve_trigger_entry_identity(
            db,
            phone_entry_id=row.get("phone_entry_id"),
            credit_card_entry_id=row.get("credit_card_entry_id"),
        )
        row["resolved_entry_label"] = label
        row["resolved_entry_value"] = value
    return rows


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

    retry_to_status = _issue_video_retry_status(issue_video, latest_failed_script)
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


def _issue_video_retry_status(
    video_asset: Mapping[str, Any],
    latest_failed_script: Mapping[str, Any] | None,
) -> str:
    section = str(video_asset.get("section") or "").strip().lower()
    script_name = str((latest_failed_script or {}).get("script_name") or "").strip().lower()
    if section == "kiosk" or script_name in {"entry", "kiosk", "kiosk_match"}:
        return "ready"
    return "not_retrieved"


def list_trigger_frame_assets(
    db: Session,
    limit: int = 100,
    *,
    location_id: int | None = None,
    start_time: Any | None = None,
    end_time: Any | None = None,
    status: str | None = None,
    trigger_id: int | None = None,
) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    where_clauses = ["status <> 'deleted'"]
    params: dict[str, Any] = {"limit": limit}
    if location_id is not None:
        where_clauses.append("location_id = :location_id")
        params["location_id"] = location_id
    if start_time is not None:
        where_clauses.append("start_time >= :start_time")
        params["start_time"] = start_time
    if end_time is not None:
        where_clauses.append("start_time < :end_time")
        params["end_time"] = end_time
    if status:
        where_clauses.append("status = :status")
        params["status"] = status
    if trigger_id is not None:
        where_clauses.append("trigger_id = :trigger_id")
        params["trigger_id"] = trigger_id
    where_sql = " and ".join(where_clauses)
    result = db.execute(
        text(
            f"""
            select id, trigger_id, location_id, start_time, end_time, status, error, created_at, updated_at
            from {frame_asset_table}
            where {where_sql}
            order by created_at desc, id desc
            limit :limit
            """
        ),
        params,
    )
    assets = _fetch_all_dicts(result)
    if not assets:
        return []

    asset_ids = [int(asset["id"]) for asset in assets]
    frame_result = db.execute(
        text(
            f"""
            select id, frame_asset_id, trigger_id, frame_index, sample_time, image_url, status,
                   is_best_for_verification, created_at
            from {frame_table}
            where frame_asset_id in :asset_ids
              and status <> 'deleted'
            order by frame_asset_id asc, frame_index asc, id asc
            """
        ).bindparams(bindparam("asset_ids", expanding=True)),
        {"asset_ids": asset_ids},
    )
    frames_by_asset: dict[int, list[dict[str, Any]]] = {}
    for frame in _fetch_all_dicts(frame_result):
        frames_by_asset.setdefault(int(frame["frame_asset_id"]), []).append(frame)

    for asset in assets:
        asset["frames"] = frames_by_asset.get(int(asset["id"]), [])
    return assets


def set_trigger_best_verification_frames(db: Session, trigger_id: int, frame_indices: list[int]) -> None:
    # Persists grouping_adjacent's per-trigger "clearest 1-2 frames" pick so a
    # later, separate process (grouping_repair, which runs in its own worker
    # pass and has no access to adjacent's in-memory results) can still send a
    # cheap, curated frame subset to its own verification re-check instead of
    # every frame. Always resets the trigger's other frames to 0 first, so a
    # re-run's new pick fully replaces any stale prior selection.
    frame_table = _table("trigger_frame")
    db.execute(
        text(f"update {frame_table} set is_best_for_verification = 0 where trigger_id = :trigger_id"),
        {"trigger_id": trigger_id},
    )
    if frame_indices:
        db.execute(
            text(
                f"""
                update {frame_table}
                set is_best_for_verification = 1
                where trigger_id = :trigger_id
                  and frame_index in :frame_indices
                """
            ).bindparams(bindparam("frame_indices", expanding=True)),
            {"trigger_id": trigger_id, "frame_indices": frame_indices},
        )
    db.commit()


def list_best_verification_frame_urls_by_trigger(db: Session, trigger_ids: list[int]) -> dict[int, list[str]]:
    # Read side of set_trigger_best_verification_frames - a trigger with no
    # curated pick yet (never went through grouping_adjacent, or predates this
    # feature) simply has no entry here; callers should fall back to that
    # trigger's full frame list in that case, not treat an empty result as
    # "no usable frames".
    if not trigger_ids:
        return {}
    frame_table = _table("trigger_frame")
    result = db.execute(
        text(
            f"""
            select trigger_id, image_url
            from {frame_table}
            where trigger_id in :trigger_ids
              and is_best_for_verification = 1
              and status = 'ok'
            order by trigger_id asc, frame_index asc
            """
        ).bindparams(bindparam("trigger_ids", expanding=True)),
        {"trigger_ids": trigger_ids},
    )
    urls_by_trigger: dict[int, list[str]] = {}
    for row in _fetch_all_dicts(result):
        trigger_id = int(row["trigger_id"])
        image_url = str(row.get("image_url") or "").strip()
        if image_url:
            urls_by_trigger.setdefault(trigger_id, []).append(image_url)
    return urls_by_trigger


def set_trigger_unique_customer_count(
    db: Session,
    trigger_id: int,
    *,
    count: int,
    confidence: float,
    source: str,
) -> None:
    # source is whichever grouping stage produced this estimate (adjacent,
    # direct, or repair) - kept purely so a stale/odd count can be traced back
    # to where it came from, not used to gate anything itself.
    trigger_table = _table("trigger_event")
    db.execute(
        text(
            f"""
            update {trigger_table}
            set unique_customer_count = :count,
                unique_customer_count_confidence = :confidence,
                unique_customer_count_source = :source
            where id = :trigger_id
            """
        ),
        {"trigger_id": trigger_id, "count": count, "confidence": confidence, "source": source},
    )
    db.commit()


def get_trigger_unique_customer_counts(db: Session, trigger_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not trigger_ids:
        return {}
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select id, unique_customer_count, unique_customer_count_confidence, unique_customer_count_source
            from {trigger_table}
            where id in :trigger_ids
              and unique_customer_count is not null
            """
        ).bindparams(bindparam("trigger_ids", expanding=True)),
        {"trigger_ids": trigger_ids},
    )
    return {int(row["id"]): dict(row) for row in _fetch_all_dicts(result)}


def get_trigger_frame_asset(db: Session, frame_asset_id: int) -> dict[str, Any]:
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            select id, trigger_id, location_id, start_time, end_time, status, error, created_at, updated_at
            from {frame_asset_table}
            where id = :frame_asset_id
            """
        ),
        {"frame_asset_id": frame_asset_id},
    )
    return _fetch_one_dict(result)


def retry_trigger_frame_asset_issue(db: Session, frame_asset_id: int) -> dict[str, Any]:
    # Also allows retrying a 'retrieved' asset that only got a partial frame
    # set (e.g. 2/5 frames) - the retrieval job marks the whole asset
    # 'retrieved' once even one frame succeeds, so completeness has to be
    # judged by the caller (the frontend shows this button whenever the
    # frame count looks short) rather than gatekept purely on status here.
    frame_asset = get_trigger_frame_asset(db, frame_asset_id)
    if str(frame_asset.get("status") or "") not in {"issue", "retrieved"}:
        raise ValueError("This trigger frame asset is not in a retryable state.")
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table}
            set status = 'not_retrieved',
                error = null,
                updated_at = now()
            where id = :frame_asset_id
              and status in ('issue', 'retrieved')
            """
        ),
        {"frame_asset_id": frame_asset_id},
    )
    db.commit()
    if not result.rowcount:
        raise ValueError("This trigger frame asset is not in a retryable state.")
    return get_trigger_frame_asset(db, frame_asset_id)


def list_pending_trigger_frame_asset_retrievals(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select fa.id, fa.trigger_id, fa.location_id, fa.start_time, fa.end_time, fa.status, fa.error,
                   fa.created_at, fa.updated_at
            from {frame_asset_table} fa
            left join {trigger_table} te on te.id = fa.trigger_id
            where fa.status = 'not_retrieved'
              and (te.id is null or (te.whitelist_hit = 0 and te.status <> 'whitelisted'))
            order by fa.start_time asc, fa.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def list_running_trigger_frame_asset_retrievals(db: Session) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            select id, trigger_id, location_id, start_time, end_time, status, error, created_at, updated_at
            from {frame_asset_table}
            where status = 'retrieving'
            order by id asc
            """
        )
    )
    return _fetch_all_dicts(result)


def reset_stale_trigger_frame_asset_retrievals(db: Session, stale_seconds: int) -> int:
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table}
            set status = 'not_retrieved',
                error = concat('Reset stale retrieval after ', :stale_seconds, ' seconds.'),
                updated_at = now()
            where status = 'retrieving'
              and timestampdiff(second, updated_at, now()) > :stale_seconds
            """
        ),
        {"stale_seconds": max(1, int(stale_seconds))},
    )
    db.commit()
    return int(result.rowcount or 0)


def claim_trigger_frame_asset_for_retrieval(db: Session, frame_asset_id: int) -> bool:
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table}
            set status = 'retrieving',
                error = null,
                updated_at = now()
            where id = :frame_asset_id and status = 'not_retrieved'
            """
        ),
        {"frame_asset_id": frame_asset_id},
    )
    db.commit()
    return bool(result.rowcount)


def update_trigger_frame_asset_status(
    db: Session,
    frame_asset_id: int,
    status: str,
    *,
    error: str | None = None,
) -> None:
    frame_asset_table = _table("trigger_frame_asset")
    db.execute(
        text(
            f"""
            update {frame_asset_table}
            set status = :status,
                error = :error,
                updated_at = now()
            where id = :frame_asset_id
            """
        ),
        {"frame_asset_id": frame_asset_id, "status": status, "error": error},
    )
    db.commit()


def replace_trigger_frame_rows(
    db: Session,
    *,
    frame_asset_id: int,
    trigger_id: int,
    frames: list[Mapping[str, Any]],
) -> None:
    frame_table = _table("trigger_frame")
    db.execute(
        text(f"delete from {frame_table} where frame_asset_id = :frame_asset_id"),
        {"frame_asset_id": frame_asset_id},
    )
    if frames:
        db.execute(
            text(
                f"""
                insert into {frame_table} (
                    frame_asset_id, trigger_id, frame_index, sample_time, image_url, status
                )
                values (
                    :frame_asset_id, :trigger_id, :frame_index, :sample_time, :image_url, :status
                )
                """
            ),
            [
                {
                    "frame_asset_id": frame_asset_id,
                    "trigger_id": trigger_id,
                    "frame_index": frame.get("frame_index"),
                    "sample_time": frame.get("sample_time"),
                    "image_url": frame.get("image_url"),
                    "status": frame.get("status"),
                }
                for frame in frames
            ],
        )
    db.commit()


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
    row = _fetch_one_dict(result)
    if isinstance(row.get("metadata"), str):
        try:
            row["metadata"] = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass
    return row


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


def get_latest_video_asset_for_trigger(
    db: Session,
    *,
    trigger_id: int,
    section: str | None = None,
) -> dict[str, Any]:
    video_asset_table = _table("video_asset")
    where_section = ""
    params: dict[str, Any] = {"trigger_id": trigger_id}
    if section is not None:
        where_section = " and section = :section"
        params["section"] = section
    result = db.execute(
        text(
            f"""
            select id, trigger_id, section, sequence_no, video_url, file_path,
                   captured_start_time, captured_end_time, retrieved_at, analyzed_at,
                   retention_until, status, metadata, created_at
            from {video_asset_table}
            where trigger_id = :trigger_id
              and status <> 'deleted'
              {where_section}
            order by coalesce(captured_start_time, created_at) desc, id desc
            limit 1
            """
        ),
        params,
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("metadata"), str):
        try:
            row["metadata"] = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass
    return row


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
                   va.status,
                   va.created_at,
                   min(sva.session_id) as session_id,
                   coalesce(
                       te.location_id,
                       min(s.location_id),
                       cast(json_unquote(json_extract(va.metadata, '$.location_id')) as unsigned)
                   ) as location_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.status = 'not_retrieved'
            group by va.id, va.trigger_id, va.section, va.file_path, va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.status, va.created_at, te.location_id
            order by case when va.section = 'entrance' then 0 else 1 end asc,
                     coalesce(va.captured_start_time, va.created_at) asc,
                     va.id asc
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
                   va.status,
                   min(sva.session_id) as session_id,
                   coalesce(
                       te.location_id,
                       min(s.location_id),
                       cast(json_unquote(json_extract(va.metadata, '$.location_id')) as unsigned)
                   ) as location_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.status = 'retrieving'
            group by va.id, va.trigger_id, va.section, va.file_path, va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.status, te.location_id
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
            set metadata = json_set(coalesce(metadata, json_object()), '$.claimed_from_status', status),
                status = 'retrieving'
            where id = :video_asset_id and status = 'not_retrieved'
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    db.commit()
    return bool(result.rowcount)


def list_ready_trigger_frame_assets_for_window(
    db: Session,
    *,
    location_id: int,
    window_start: Any,
    window_end: Any,
) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select te.id as trigger_id,
                   te.location_id,
                   te.trigger_time,
                   te.phone_entry_id,
                   te.credit_card_entry_id,
                   te.entry_source_type,
                   fa.id as frame_asset_id,
                   fa.start_time as frame_asset_start_time,
                   fa.end_time as frame_asset_end_time,
                   fa.status as frame_asset_status,
                   fa.created_at as frame_asset_created_at
            from {trigger_table} te
            join {frame_asset_table} fa on fa.trigger_id = te.id
            where te.location_id = :location_id
              and te.trigger_time >= :window_start
              and te.trigger_time < :window_end
              and te.whitelist_hit = 0
              and te.status <> 'whitelisted'
              and fa.status = 'retrieved'
            order by te.trigger_time asc, te.id asc, fa.id asc
            """
        ),
        {
            "location_id": location_id,
            "window_start": window_start,
            "window_end": window_end,
        },
    )
    rows = _fetch_all_dicts(result)
    frame_asset_ids = [int(row["frame_asset_id"]) for row in rows if row.get("frame_asset_id") is not None]
    frames_by_asset: dict[int, list[dict[str, Any]]] = {}
    if frame_asset_ids:
        frame_result = db.execute(
            text(
                f"""
                select id, frame_asset_id, trigger_id, frame_index, sample_time, image_url, status, created_at
                from {frame_table}
                where frame_asset_id in :frame_asset_ids
                  and status <> 'deleted'
                order by frame_asset_id asc, frame_index asc, id asc
                """
            ).bindparams(bindparam("frame_asset_ids", expanding=True)),
            {"frame_asset_ids": frame_asset_ids},
        )
        for frame in _fetch_all_dicts(frame_result):
            frames_by_asset.setdefault(int(frame["frame_asset_id"]), []).append(frame)
    for row in rows:
        row["trigger_frames"] = frames_by_asset.get(int(row["frame_asset_id"]), [])
    return rows


def list_trigger_frame_assets_for_window(
    db: Session,
    *,
    location_id: int,
    window_start: Any,
    window_end: Any,
) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            select te.id as trigger_id,
                   te.location_id,
                   te.trigger_time,
                   te.phone_entry_id,
                   te.credit_card_entry_id,
                   te.entry_source_type,
                   fa.id as frame_asset_id,
                   fa.start_time as frame_asset_start_time,
                   fa.end_time as frame_asset_end_time,
                   fa.status as frame_asset_status,
                   fa.created_at as frame_asset_created_at
            from {trigger_table} te
            join {frame_asset_table} fa on fa.trigger_id = te.id
            where te.location_id = :location_id
              and te.trigger_time >= :window_start
              and te.trigger_time < :window_end
              and te.whitelist_hit = 0
              and te.status <> 'whitelisted'
              and fa.status <> 'deleted'
            order by te.trigger_time asc, te.id asc, fa.id asc
            """
        ),
        {
            "location_id": location_id,
            "window_start": window_start,
            "window_end": window_end,
        },
    )
    rows = _fetch_all_dicts(result)
    frame_asset_ids = [int(row["frame_asset_id"]) for row in rows if row.get("frame_asset_id") is not None]
    frames_by_asset: dict[int, list[dict[str, Any]]] = {}
    if frame_asset_ids:
        frame_result = db.execute(
            text(
                f"""
                select id, frame_asset_id, trigger_id, frame_index, sample_time, image_url, status, created_at
                from {frame_table}
                where frame_asset_id in :frame_asset_ids
                  and status <> 'deleted'
                order by frame_asset_id asc, frame_index asc, id asc
                """
            ).bindparams(bindparam("frame_asset_ids", expanding=True)),
            {"frame_asset_ids": frame_asset_ids},
        )
        for frame in _fetch_all_dicts(frame_result):
            frames_by_asset.setdefault(int(frame["frame_asset_id"]), []).append(frame)
    for row in rows:
        row["trigger_frames"] = frames_by_asset.get(int(row["frame_asset_id"]), [])
    return rows


def has_pending_trigger_frame_retrieval_in_window(
    db: Session,
    *,
    location_id: int,
    window_start: Any,
    window_end: Any,
    min_frame_count: int = 1,
) -> bool:
    # A trigger with no frame_asset row yet (retrieval not even queued), one
    # still short of a terminal state ('retrieved', 'processed', or 'issue'), or
    # one marked 'retrieved' with fewer than min_frame_count successfully-captured
    # frames (the retrieval job marks the whole asset 'retrieved' the moment even
    # one frame out of five succeeds) all mean this window's data isn't actually
    # complete yet - forming a batch now would silently snapshot a partial or
    # near-empty trigger set, exactly like trigger 454 and 410 slipping through.
    # 'processed' is included as terminal because a trigger just inside this
    # window's carry-forward buffer can already have been fully consumed by an
    # earlier, adjacent period's batch (mark_grouping_batch_frame_assets_processed)
    # - that's finished, not still pending, and must not block this window forever.
    trigger_table = _table("trigger_event")
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    result = db.execute(
        text(
            f"""
            select 1
            from {trigger_table} te
            left join {frame_asset_table} fa on fa.trigger_id = te.id
            left join (
                select frame_asset_id, count(*) as ok_count
                from {frame_table}
                where status = 'ok'
                group by frame_asset_id
            ) okc on okc.frame_asset_id = fa.id
            where te.location_id = :location_id
              and te.trigger_time >= :window_start
              and te.trigger_time < :window_end
              and te.whitelist_hit = 0
              and te.status <> 'whitelisted'
              and (
                  fa.id is null
                  or fa.status not in ('retrieved', 'processed', 'issue')
                  or (fa.status = 'retrieved' and coalesce(okc.ok_count, 0) < :min_frame_count)
              )
            limit 1
            """
        ),
        {
            "location_id": location_id,
            "window_start": window_start,
            "window_end": window_end,
            "min_frame_count": max(1, int(min_frame_count)),
        },
    )
    return result.first() is not None


def requeue_incomplete_trigger_frame_assets_in_window(
    db: Session,
    *,
    location_id: int,
    window_start: Any,
    window_end: Any,
    min_frame_count: int,
    cooldown_seconds: int = 120,
) -> int:
    # Give a partially-retrieved asset another shot at filling in the missing
    # frames, rate-limited so a persistently-broken trigger (e.g. the camera
    # genuinely had nothing for that offset) doesn't get re-run every single
    # poll tick for the whole grace window.
    trigger_table = _table("trigger_event")
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table} fa
            join {trigger_table} te on te.id = fa.trigger_id
            left join (
                select frame_asset_id, count(*) as ok_count
                from {frame_table}
                where status = 'ok'
                group by frame_asset_id
            ) okc on okc.frame_asset_id = fa.id
            set fa.status = 'not_retrieved',
                fa.error = null,
                fa.updated_at = now()
            where te.location_id = :location_id
              and te.trigger_time >= :window_start
              and te.trigger_time < :window_end
              and te.whitelist_hit = 0
              and te.status <> 'whitelisted'
              and fa.status = 'retrieved'
              and coalesce(okc.ok_count, 0) < :min_frame_count
              and fa.updated_at < date_sub(now(), interval :cooldown_seconds second)
            """
        ),
        {
            "location_id": location_id,
            "window_start": window_start,
            "window_end": window_end,
            "min_frame_count": max(1, int(min_frame_count)),
            "cooldown_seconds": max(1, int(cooldown_seconds)),
        },
    )
    db.commit()
    return int(result.rowcount or 0)


def list_manual_grouping_ready_trigger_frame_assets(
    db: Session,
    *,
    location_id: int,
    limit: int = 200,
) -> list[dict[str, Any]]:
    frame_asset_table = _table("trigger_frame_asset")
    frame_table = _table("trigger_frame")
    trigger_table = _table("trigger_event")
    grouping_item_table = _table("filter_grouping_item")
    grouping_batch_table = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select te.id as trigger_id,
                   te.location_id,
                   te.trigger_time,
                   te.phone_entry_id,
                   te.credit_card_entry_id,
                   te.entry_source_type,
                   fa.id as frame_asset_id,
                   fa.start_time as frame_asset_start_time,
                   fa.end_time as frame_asset_end_time,
                   fa.status as frame_asset_status,
                   fa.created_at as frame_asset_created_at
            from {trigger_table} te
            join {frame_asset_table} fa on fa.trigger_id = te.id
            where te.location_id = :location_id
              and te.whitelist_hit = 0
              and te.status <> 'whitelisted'
              and (
                  fa.status = 'retrieved'
                  -- mark_stale_open_entry_frame_assets_issue flips a perfectly
                  -- good, already-retrieved asset to 'issue' purely because it
                  -- sat too long without a match - its frames are fine, it was
                  -- never a real retrieval failure. Excluding it here (the same
                  -- query both the schedule and every manual rerun use) made it
                  -- permanently unusable forever, even on retry.
                  or (fa.status = 'issue' and fa.error = 'No matching exit found before the open-entry staleness cutoff.')
              )
              and not exists (
                  select 1
                  from {grouping_item_table} gi
                  join {grouping_batch_table} gb on gb.id = gi.batch_id
                  where gi.trigger_id = te.id
                    and (
                        gb.status in ('pending', 'dispatching', 'running')
                        or gi.status = 'grouped'
                    )
              )
            order by te.trigger_time asc, te.id asc, fa.id asc
            limit :limit
            """
        ),
        {"location_id": location_id, "limit": limit},
    )
    rows = _fetch_all_dicts(result)
    frame_asset_ids = [int(row["frame_asset_id"]) for row in rows if row.get("frame_asset_id") is not None]
    frames_by_asset: dict[int, list[dict[str, Any]]] = {}
    if frame_asset_ids:
        frame_result = db.execute(
            text(
                f"""
                select id, frame_asset_id, trigger_id, frame_index, sample_time, image_url, status, created_at
                from {frame_table}
                where frame_asset_id in :frame_asset_ids
                  and status <> 'deleted'
                order by frame_asset_id asc, frame_index asc, id asc
                """
            ).bindparams(bindparam("frame_asset_ids", expanding=True)),
            {"frame_asset_ids": frame_asset_ids},
        )
        for frame in _fetch_all_dicts(frame_result):
            frames_by_asset.setdefault(int(frame["frame_asset_id"]), []).append(frame)
    for row in rows:
        row["trigger_frames"] = frames_by_asset.get(int(row["frame_asset_id"]), [])
    return rows


def list_filter_time_periods(db: Session, *, selected_only: bool = False) -> list[dict[str, Any]]:
    table_name = _table("filter_time_period")
    where_selected = "where selected = 1" if selected_only else ""
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, label, start_time, end_time, selected, metadata, created_at, updated_at
            from {table_name}
            {where_selected}
            order by coalesce(location_id, 0) asc, start_time asc, id asc
            """
        )
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
        row["start_time"] = _mysql_time_value_to_clock_string(row.get("start_time"))
        row["end_time"] = _mysql_time_value_to_clock_string(row.get("end_time"))
    return rows


def _mysql_time_value_to_clock_string(value: Any) -> Any:
    if isinstance(value, timedelta):
        total_seconds = int(value.total_seconds()) % 86400
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return value


def _normalize_mysql_time_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value.strip(), flags=re.IGNORECASE)
    if match is None:
        return value
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def upsert_filter_time_period(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    table_name = _table("filter_time_period")
    params = {
        "location_id": payload.get("location_id"),
        "period_code": payload.get("period_code"),
        "label": payload.get("label"),
        "start_time": _normalize_mysql_time_value(payload.get("start_time")),
        "end_time": _normalize_mysql_time_value(payload.get("end_time")),
        "selected": 1 if payload.get("selected") else 0,
        "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
    }
    updated = db.execute(
        text(
            f"""
            update {table_name}
            set label = :label,
                start_time = :start_time,
                end_time = :end_time,
                selected = :selected,
                metadata = :metadata
            where ((location_id is null and :location_id is null) or location_id = :location_id)
              and period_code = :period_code
            """
        ),
        params,
    )
    if not updated.rowcount:
        db.execute(
            text(
                f"""
                insert into {table_name} (location_id, period_code, label, start_time, end_time, selected, metadata)
                values (:location_id, :period_code, :label, :start_time, :end_time, :selected, :metadata)
                """
            ),
            params,
        )
    db.commit()
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, label, start_time, end_time, selected, metadata, created_at, updated_at
            from {table_name}
            where ((location_id is null and :location_id is null) or location_id = :location_id)
              and period_code = :period_code
            order by id asc
            limit 1
            """
        ),
        {"location_id": payload.get("location_id"), "period_code": payload.get("period_code")},
    )
    return _fetch_one_dict(result)


def delete_filter_time_period(db: Session, *, period_code: str, location_id: int | None = None) -> None:
    table_name = _table("filter_time_period")
    db.execute(
        text(
            f"""
            delete from {table_name}
            where ((location_id is null and :location_id is null) or location_id = :location_id)
              and period_code = :period_code
            """
        ),
        {"location_id": location_id, "period_code": period_code},
    )
    db.commit()


def list_filter_factors(db: Session) -> list[dict[str, Any]]:
    table_name = _table("filter_factor")
    result = db.execute(
        text(
            f"""
            select id, location_id, factor_code, label, enabled, weight, config, created_at, updated_at
            from {table_name}
            order by coalesce(location_id, 0) asc, factor_code asc
            """
        )
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("config"), str):
            try:
                row["config"] = json.loads(row["config"])
            except json.JSONDecodeError:
                pass
    return rows


def upsert_filter_factor(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    table_name = _table("filter_factor")
    params = {
        "location_id": payload.get("location_id"),
        "factor_code": payload.get("factor_code"),
        "label": payload.get("label"),
        "enabled": 1 if payload.get("enabled") else 0,
        "weight": payload.get("weight", 1),
        "config": _json_dumps(payload.get("config")) if payload.get("config") is not None else None,
    }
    updated = db.execute(
        text(
            f"""
            update {table_name}
            set label = :label,
                enabled = :enabled,
                weight = :weight,
                config = :config
            where ((location_id is null and :location_id is null) or location_id = :location_id)
              and factor_code = :factor_code
            """
        ),
        params,
    )
    if not updated.rowcount:
        db.execute(
            text(
                f"""
                insert into {table_name} (location_id, factor_code, label, enabled, weight, config)
                values (:location_id, :factor_code, :label, :enabled, :weight, :config)
                """
            ),
            params,
        )
    db.commit()
    result = db.execute(
        text(
            f"""
            select id, location_id, factor_code, label, enabled, weight, config, created_at, updated_at
            from {table_name}
            where ((location_id is null and :location_id is null) or location_id = :location_id)
              and factor_code = :factor_code
            order by id asc
            limit 1
            """
        ),
        {"location_id": payload.get("location_id"), "factor_code": payload.get("factor_code")},
    )
    return _fetch_one_dict(result)


def list_filter_country_code_checks(
    db: Session,
    *,
    location_id: int | None = None,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    table_name = _table("filter_country_code_check")
    where_clauses = []
    params: dict[str, Any] = {}
    if location_id is not None:
        where_clauses.append("(location_id is null or location_id = :location_id)")
        params["location_id"] = location_id
    if enabled_only:
        where_clauses.append("enabled = 1")
    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    result = db.execute(
        text(
            f"""
            select id, location_id, country_code, country_name, phone_prefix, card_country, enabled,
                   metadata, created_at, updated_at
            from {table_name}
            {where_sql}
            order by coalesce(location_id, 0) asc, country_name asc, country_code asc, id asc
            """
        ),
        params,
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
    return rows


def create_filter_country_code_check(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    table_name = _table("filter_country_code_check")
    params = {
        "location_id": payload.get("location_id"),
        "country_code": str(payload.get("country_code") or "").strip().upper(),
        "country_name": str(payload.get("country_name") or "").strip() or None,
        "phone_prefix": str(payload.get("phone_prefix") or "").strip() or None,
        "card_country": str(payload.get("card_country") or "").strip().upper() or None,
        "enabled": 1 if payload.get("enabled", True) else 0,
        "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
    }
    if not params["country_code"]:
        raise ValueError("country_code is required.")
    if not params["phone_prefix"] and not params["card_country"]:
        raise ValueError("phone_prefix or card_country is required.")
    result = db.execute(
        text(
            f"""
            insert into {table_name} (location_id, country_code, country_name, phone_prefix, card_country, enabled, metadata)
            values (:location_id, :country_code, :country_name, :phone_prefix, :card_country, :enabled, :metadata)
            """
        ),
        params,
    )
    db.commit()
    rule_id = int(result.lastrowid)
    return get_filter_country_code_check(db, rule_id)


def get_filter_country_code_check(db: Session, rule_id: int) -> dict[str, Any]:
    table_name = _table("filter_country_code_check")
    result = db.execute(
        text(
            f"""
            select id, location_id, country_code, country_name, phone_prefix, card_country, enabled,
                   metadata, created_at, updated_at
            from {table_name}
            where id = :rule_id
            """
        ),
        {"rule_id": rule_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("metadata"), str):
        try:
            row["metadata"] = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass
    return row


def update_filter_country_code_check(db: Session, rule_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    table_name = _table("filter_country_code_check")
    params = {
        "rule_id": rule_id,
        "location_id": payload.get("location_id"),
        "country_code": str(payload.get("country_code") or "").strip().upper(),
        "country_name": str(payload.get("country_name") or "").strip() or None,
        "phone_prefix": str(payload.get("phone_prefix") or "").strip() or None,
        "card_country": str(payload.get("card_country") or "").strip().upper() or None,
        "enabled": 1 if payload.get("enabled", True) else 0,
        "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
    }
    if not params["country_code"]:
        raise ValueError("country_code is required.")
    db.execute(
        text(
            f"""
            update {table_name}
            set location_id = :location_id,
                country_code = :country_code,
                country_name = :country_name,
                phone_prefix = :phone_prefix,
                card_country = :card_country,
                enabled = :enabled,
                metadata = :metadata
            where id = :rule_id
            """
        ),
        params,
    )
    db.commit()
    return get_filter_country_code_check(db, rule_id)


def delete_filter_country_code_check(db: Session, rule_id: int) -> None:
    table_name = _table("filter_country_code_check")
    db.execute(text(f"delete from {table_name} where id = :rule_id"), {"rule_id": rule_id})
    db.commit()


def get_grouping_batch_by_window(
    db: Session,
    *,
    location_id: int,
    period_code: str,
    window_start: Any,
    window_end: Any,
) -> dict[str, Any] | None:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, window_start, window_end, script_run_id, status,
                   manifest_url, manifest_object_key, result_payload, issue_reason, started_at, finished_at, created_at, updated_at
            from {table_name}
            where location_id = :location_id
              and period_code = :period_code
              and window_start = :window_start
              and window_end = :window_end
            limit 1
            """
        ),
        {
            "location_id": location_id,
            "period_code": period_code,
            "window_start": window_start,
            "window_end": window_end,
        },
    )
    row = result.mappings().first()
    if row is None:
        return None
    payload = dict(row)
    if isinstance(payload.get("result_payload"), str):
        try:
            payload["result_payload"] = json.loads(payload["result_payload"])
        except json.JSONDecodeError:
            pass
    return payload


def create_grouping_batch(
    db: Session,
    *,
    location_id: int,
    period_code: str,
    window_start: Any,
    window_end: Any,
    status: str = "pending",
) -> dict[str, Any]:
    existing = get_grouping_batch_by_window(
        db,
        location_id=location_id,
        period_code=period_code,
        window_start=window_start,
        window_end=window_end,
    )
    if existing is not None:
        return existing
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            insert into {table_name} (location_id, period_code, window_start, window_end, status)
            values (:location_id, :period_code, :window_start, :window_end, :status)
            """
        ),
        {
            "location_id": location_id,
            "period_code": period_code,
            "window_start": window_start,
            "window_end": window_end,
            "status": status,
        },
    )
    db.commit()
    return get_grouping_batch(db, int(result.lastrowid))


def get_grouping_batch(db: Session, batch_id: int) -> dict[str, Any]:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, window_start, window_end, script_run_id, status,
                   manifest_url, manifest_object_key, result_payload, issue_reason, started_at, finished_at, created_at, updated_at
            from {table_name}
            where id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("result_payload"), str):
        try:
            row["result_payload"] = json.loads(row["result_payload"])
        except json.JSONDecodeError:
            pass
    return row


def update_grouping_batch(db: Session, batch_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    table_name = _table("filter_grouping_batch")
    db.execute(
        text(
            f"""
            update {table_name}
            set script_run_id = coalesce(:script_run_id, script_run_id),
                status = coalesce(:status, status),
                manifest_url = coalesce(:manifest_url, manifest_url),
                manifest_object_key = coalesce(:manifest_object_key, manifest_object_key),
                result_payload = coalesce(:result_payload, result_payload),
                issue_reason = coalesce(:issue_reason, issue_reason),
                started_at = coalesce(:started_at, started_at),
                finished_at = coalesce(:finished_at, finished_at)
            where id = :batch_id
            """
        ),
        {
            "batch_id": batch_id,
            "script_run_id": payload.get("script_run_id"),
            "status": payload.get("status"),
            "manifest_url": payload.get("manifest_url"),
            "manifest_object_key": payload.get("manifest_object_key"),
            "result_payload": _json_dumps(payload.get("result_payload")) if payload.get("result_payload") is not None else None,
            "issue_reason": payload.get("issue_reason"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
        },
    )
    db.commit()
    return get_grouping_batch(db, batch_id)


def reset_grouping_batch_for_retry(db: Session, batch_id: int) -> dict[str, Any]:
    table_name = _table("filter_grouping_batch")
    db.execute(
        text(
            f"""
            update {table_name}
            set status = 'pending',
                issue_reason = null,
                result_payload = null,
                started_at = null,
                finished_at = null,
                updated_at = now()
            where id = :batch_id
              and status in ('success', 'failed', 'issue', 'cancel', 'canceled', 'cancelled')
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return get_grouping_batch(db, batch_id)


def delete_filter_confidence_results_for_batch(db: Session, batch_id: int) -> int:
    table_name = _table("filter_confidence_result")
    result = db.execute(
        text(
            f"""
            delete from {table_name}
            where batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return int(result.rowcount or 0)


def delete_grouping_items_for_batch(db: Session, batch_id: int) -> int:
    table_name = _table("filter_grouping_item")
    result = db.execute(
        text(
            f"""
            delete from {table_name}
            where batch_id = :batch_id
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return int(result.rowcount or 0)


def upsert_grouping_item(
    db: Session,
    *,
    batch_id: int,
    trigger_id: int,
    video_asset_id: int | None,
    group_key: str | None = None,
    role: str = "unknown",
    status: str = "pending",
    score: float | None = None,
    frame_payload: Mapping[str, Any] | None = None,
    result_payload: Mapping[str, Any] | None = None,
) -> None:
    table_name = _table("filter_grouping_item")
    db.execute(
        text(
            f"""
            insert into {table_name} (
                batch_id, trigger_id, video_asset_id, group_key, role, status, score, frame_payload, result_payload
            )
            values (
                :batch_id, :trigger_id, :video_asset_id, :group_key, :role, :status, :score, :frame_payload, :result_payload
            )
            on duplicate key update
                video_asset_id = coalesce(values(video_asset_id), video_asset_id),
                group_key = values(group_key),
                role = values(role),
                status = values(status),
                score = values(score),
                frame_payload = coalesce(values(frame_payload), frame_payload),
                result_payload = values(result_payload)
            """
        ),
        {
            "batch_id": batch_id,
            "trigger_id": trigger_id,
            "video_asset_id": video_asset_id,
            "group_key": group_key,
            "role": role,
            "status": status,
            "score": score,
            "frame_payload": _json_dumps(frame_payload) if frame_payload is not None else None,
            "result_payload": _json_dumps(result_payload) if result_payload is not None else None,
        },
    )
    db.commit()


def list_pending_grouping_batches(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, window_start, window_end, script_run_id, status,
                   manifest_url, manifest_object_key, result_payload, issue_reason, started_at, finished_at, created_at, updated_at
            from {table_name}
            where status = 'pending'
            order by window_start asc, id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def claim_grouping_batch_for_dispatch(db: Session, batch_id: int) -> bool:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            update {table_name}
            set status = 'dispatching'
            where id = :batch_id and status = 'pending'
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return int(result.rowcount or 0) > 0


def list_running_grouping_batches(db: Session) -> list[dict[str, Any]]:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, window_start, window_end, script_run_id, status,
                   manifest_url, manifest_object_key, result_payload, issue_reason, started_at, finished_at, created_at, updated_at
            from {table_name}
            where status in ('dispatching', 'running')
            order by started_at asc, id asc
            """
        )
    )
    return _fetch_all_dicts(result)


def list_recent_grouping_batches(db: Session, limit: int = 50, *, offset: int = 0) -> list[dict[str, Any]]:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select id, location_id, period_code, window_start, window_end, script_run_id, status,
                   manifest_url, manifest_object_key, result_payload, issue_reason, started_at, finished_at, created_at, updated_at
            from {table_name}
            where status in ('success', 'failed', 'issue')
            order by coalesce(finished_at, updated_at, created_at) desc, id desc
            limit :limit offset :offset
            """
        ),
        {"limit": limit, "offset": max(0, offset)},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("result_payload"), str):
            try:
                row["result_payload"] = json.loads(row["result_payload"])
            except json.JSONDecodeError:
                pass
    return rows


def count_recent_grouping_batches(db: Session) -> int:
    table_name = _table("filter_grouping_batch")
    result = db.execute(
        text(
            f"""
            select count(*) as total
            from {table_name}
            where status in ('success', 'failed', 'issue')
            """
        )
    )
    row = result.mappings().first()
    return int((row or {}).get("total") or 0)


def list_grouping_items(db: Session, batch_id: int) -> list[dict[str, Any]]:
    table_name = _table("filter_grouping_item")
    result = db.execute(
        text(
            f"""
            select id, batch_id, trigger_id, video_asset_id, group_key, role, status, score,
                   frame_payload, result_payload, created_at, updated_at
            from {table_name}
            where batch_id = :batch_id
            order by trigger_id asc, id asc
            """
        ),
        {"batch_id": batch_id},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        for key in ("frame_payload", "result_payload"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
    return rows


def mark_grouping_batch_frame_assets_processed(db: Session, batch_id: int) -> int:
    grouping_item_table = _table("filter_grouping_item")
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table} fa
            join {grouping_item_table} gi on gi.trigger_id = fa.trigger_id
            set fa.status = 'processed',
                fa.error = null,
                fa.updated_at = now()
            where gi.batch_id = :batch_id
              and gi.status = 'grouped'
              and fa.status in ('retrieved', 'processing')
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return int(result.rowcount or 0)


def mark_grouping_batch_frame_assets_processing(db: Session, batch_id: int) -> int:
    grouping_item_table = _table("filter_grouping_item")
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table} fa
            join {grouping_item_table} gi on gi.trigger_id = fa.trigger_id
            set fa.status = 'processing',
                fa.error = null,
                fa.updated_at = now()
            where gi.batch_id = :batch_id
              and fa.status = 'retrieved'
            """
        ),
        {"batch_id": batch_id},
    )
    db.commit()
    return int(result.rowcount or 0)


def mark_grouping_batch_frame_assets_retrieved(db: Session, batch_id: int, *, error: str | None = None) -> int:
    grouping_item_table = _table("filter_grouping_item")
    frame_asset_table = _table("trigger_frame_asset")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table} fa
            join {grouping_item_table} gi on gi.trigger_id = fa.trigger_id
            set fa.status = 'retrieved',
                fa.error = :error,
                fa.updated_at = now()
            where gi.batch_id = :batch_id
              and fa.status = 'processing'
            """
        ),
        {"batch_id": batch_id, "error": error},
    )
    db.commit()
    return int(result.rowcount or 0)


def mark_stale_open_entry_frame_assets_issue(
    db: Session,
    *,
    location_id: int,
    cutoff_time: Any,
) -> int:
    # Catches any retrieved-but-never-resolved frame asset past the cutoff,
    # regardless of whether it ever made it into a completed batch - a trigger
    # that was retrieved but, for whatever reason, never got swept into a
    # successful batch at all previously fell through this check entirely
    # (no qualifying prior grouping_item row to match against) and would sit
    # as "ready" and keep getting pulled into every future batch forever.
    frame_asset_table = _table("trigger_frame_asset")
    grouping_item_table = _table("filter_grouping_item")
    result = db.execute(
        text(
            f"""
            update {frame_asset_table} fa
            set fa.status = 'issue',
                fa.error = 'No matching exit found before the open-entry staleness cutoff.',
                fa.updated_at = now()
            where fa.location_id = :location_id
              and fa.status = 'retrieved'
              and fa.start_time < :cutoff_time
              and not exists (
                  select 1
                  from {grouping_item_table} grouped_gi
                  where grouped_gi.trigger_id = fa.trigger_id
                    and grouped_gi.status = 'grouped'
              )
            """
        ),
        {"location_id": location_id, "cutoff_time": cutoff_time},
    )
    db.commit()
    return int(result.rowcount or 0)


def _script_run_list_filters(
    *,
    script_name: str | None,
    script_type: str | None,
    model_name: str | None,
) -> tuple[list[str], dict[str, Any]]:
    filters: list[str] = []
    params: dict[str, Any] = {}
    if script_name:
        filters.append("script_name = :script_name")
        params["script_name"] = script_name
    if model_name:
        filters.append("model_name = :model_name")
        params["model_name"] = model_name
    normalized_type = str(script_type or "").strip().lower()
    if normalized_type == "gemini":
        filters.append("model_name like 'gemini_%'")
    elif normalized_type == "runpod":
        filters.append("(model_name = 'runpod_runner' or runner_job_id is not null)")
    return filters, params


def list_script_runs(
    db: Session,
    limit: int = 100,
    *,
    offset: int = 0,
    script_name: str | None = None,
    script_type: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    filters, params = _script_run_list_filters(
        script_name=script_name,
        script_type=script_type,
        model_name=model_name,
    )
    params["limit"] = limit
    params["offset"] = offset
    where_clause = f"where {' and '.join(filters)}" if filters else ""
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            {where_clause}
            order by started_at desc, id desc
            limit :limit
            offset :offset
            """
        ),
        params,
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("runner_payload"), str):
            try:
                row["runner_payload"] = json.loads(row["runner_payload"])
            except json.JSONDecodeError:
                pass
    return rows


def count_script_runs(
    db: Session,
    *,
    script_name: str | None = None,
    script_type: str | None = None,
    model_name: str | None = None,
) -> int:
    script_run_table = _table("script_run")
    filters, params = _script_run_list_filters(
        script_name=script_name,
        script_type=script_type,
        model_name=model_name,
    )
    where_clause = f"where {' and '.join(filters)}" if filters else ""
    result = db.execute(
        text(
            f"""
            select count(*) as total
            from {script_run_table}
            {where_clause}
            """
        ),
        params,
    )
    row = result.mappings().first()
    return int(row["total"]) if row and row.get("total") is not None else 0


def list_pending_theft_confidence_batches(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    batch_table = _table("filter_grouping_batch")
    confidence_table = _table("filter_confidence_result")
    result = db.execute(
        text(
            f"""
            select b.id, b.location_id, b.period_code, b.window_start, b.window_end, b.script_run_id,
                   b.status, b.manifest_url, b.manifest_object_key, b.result_payload, b.issue_reason,
                   b.started_at, b.finished_at, b.created_at, b.updated_at
            from {batch_table} b
            where b.status = 'success'
              and b.result_payload is not null
              and not exists (
                  select 1
                  from {confidence_table} c
                  where c.batch_id = b.id
              )
            order by b.finished_at asc, b.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("result_payload"), str):
            try:
                row["result_payload"] = json.loads(row["result_payload"])
            except json.JSONDecodeError:
                pass
    return rows


def list_identity_group_size_history(
    db: Session,
    *,
    location_id: int,
    phone_numbers: list[str] | None = None,
    card_fingerprints: list[str] | None = None,
    exclude_batch_id: int | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    phone_values = [str(value).strip() for value in (phone_numbers or []) if str(value).strip()]
    card_values = [str(value).strip() for value in (card_fingerprints or []) if str(value).strip()]
    if not phone_values and not card_values:
        return []

    confidence_table = _table("filter_confidence_result")
    batch_table = _table("filter_grouping_batch")
    grouping_item_table = _table("filter_grouping_item")
    trigger_table = _table("trigger_event")
    qrentry = _whitelist_source_config("qrentry")
    entrylogs = _whitelist_source_config("entrylogs")

    identity_clauses: list[str] = []
    params: dict[str, Any] = {
        "location_id": location_id,
        "exclude_batch_id": exclude_batch_id,
        "limit": limit,
    }
    bind_params = []
    if phone_values:
        identity_clauses.append(f"cast(q.{qrentry['display_column']} as char) in :phone_values")
        params["phone_values"] = phone_values
        bind_params.append(bindparam("phone_values", expanding=True))
    if card_values:
        identity_clauses.append(f"cast(e.{entrylogs['display_column']} as char) in :card_values")
        params["card_values"] = card_values
        bind_params.append(bindparam("card_values", expanding=True))

    statement = text(
        f"""
        select distinct
               c.id as confidence_result_id,
               c.batch_id,
               c.group_key,
               b.window_start,
               b.window_end,
               cast(json_unquote(json_extract(c.factor_payload, '$.total_customer')) as unsigned) as total_customer
        from {confidence_table} c
        join {batch_table} b on b.id = c.batch_id
        join {grouping_item_table} gi on gi.batch_id = b.id and gi.group_key = c.group_key and gi.role = 'entry'
        join {trigger_table} te on te.id = gi.trigger_id
        left join {qrentry['table_name']} q
               on te.phone_entry_id is not null
              and (
                  cast(q.{qrentry['id_column']} as char) = cast(te.phone_entry_id as char)
                  or cast(q.{qrentry['display_column']} as char) = cast(te.phone_entry_id as char)
              )
        left join {entrylogs['table_name']} e
               on te.credit_card_entry_id is not null
              and (
                  cast(e.{entrylogs['id_column']} as char) = cast(te.credit_card_entry_id as char)
                  or cast(e.{entrylogs['display_column']} as char) = cast(te.credit_card_entry_id as char)
              )
        where b.location_id = :location_id
          and (:exclude_batch_id is null or b.id <> :exclude_batch_id)
          and c.group_key not in ('__error__', '__no_groups__')
          and json_extract(c.factor_payload, '$.total_customer') is not null
          and ({' or '.join(identity_clauses)})
        order by b.window_start desc, c.id desc
        limit :limit
        """
    )
    if bind_params:
        statement = statement.bindparams(*bind_params)
    result = db.execute(statement, params)
    return _fetch_all_dicts(result)


def upsert_filter_confidence_result(
    db: Session,
    *,
    batch_id: int,
    group_key: str,
    location_id: int,
    score: float,
    need_deep_analysis: bool,
    reason: str,
    factor_payload: Mapping[str, Any],
) -> None:
    table_name = _table("filter_confidence_result")
    db.execute(
        text(
            f"""
            insert into {table_name} (
                batch_id, group_key, location_id, score, need_deep_analysis, reason, factor_payload
            )
            values (
                :batch_id, :group_key, :location_id, :score, :need_deep_analysis, :reason, :factor_payload
            )
            on duplicate key update
                location_id = values(location_id),
                score = values(score),
                need_deep_analysis = values(need_deep_analysis),
                reason = values(reason),
                factor_payload = values(factor_payload)
            """
        ),
        {
            "batch_id": batch_id,
            "group_key": group_key,
            "location_id": location_id,
            "score": score,
            "need_deep_analysis": 1 if need_deep_analysis else 0,
            "reason": reason,
            "factor_payload": _json_dumps(dict(factor_payload)),
        },
    )
    db.commit()


def count_filter_confidence_results(db: Session, *, batch_id: int | None = None) -> int:
    confidence_table = _table("filter_confidence_result")
    where_clauses: list[str] = []
    params: dict[str, Any] = {}
    if batch_id is not None:
        where_clauses.append("batch_id = :batch_id")
        params["batch_id"] = batch_id
    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    result = db.execute(
        text(
            f"""
            select count(*) as total
            from {confidence_table}
            {where_sql}
            """
        ),
        params,
    )
    row = result.mappings().first()
    return int((row or {}).get("total") or 0)


def list_filter_confidence_results(
    db: Session,
    limit: int = 100,
    *,
    offset: int = 0,
    batch_id: int | None = None,
) -> list[dict[str, Any]]:
    confidence_table = _table("filter_confidence_result")
    batch_table = _table("filter_grouping_batch")
    trigger_table = _table("trigger_event")
    location_table = settings.location_table_name
    location_id_column = settings.location_id_column
    location_name_column = settings.location_name_column
    where_clauses: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": max(0, offset)}
    if batch_id is not None:
        where_clauses.append("c.batch_id = :batch_id")
        params["batch_id"] = batch_id
    where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""
    result = db.execute(
        text(
            f"""
            select c.id, c.batch_id, c.group_key, c.location_id, c.score, c.need_deep_analysis,
                   c.reason, c.factor_payload, c.created_at, c.updated_at,
                   b.period_code, b.window_start, b.window_end, b.status as batch_status,
                   b.result_payload as grouping_result_payload,
                   l.{location_name_column} as location_name
            from {confidence_table} c
            left join {batch_table} b on b.id = c.batch_id
            left join {location_table} l on l.{location_id_column} = c.location_id
            {where_sql}
            order by c.created_at desc, c.id desc
            limit :limit
            offset :offset
            """
        ),
        params,
    )
    rows = _fetch_all_dicts(result)
    batch_ids = sorted({int(row["batch_id"]) for row in rows if row.get("batch_id") is not None})
    grouping_items_by_result: dict[tuple[int, str], dict[str, list[int]]] = {}
    if batch_ids:
        grouping_item_table = _table("filter_grouping_item")
        item_result = db.execute(
            text(
                f"""
                select batch_id, group_key, trigger_id, role
                from {grouping_item_table}
                where batch_id in :batch_ids
                  and trigger_id is not null
                  and status <> 'deleted'
                order by batch_id asc, group_key asc, role asc, trigger_id asc
                """
            ).bindparams(bindparam("batch_ids", expanding=True)),
            {"batch_ids": batch_ids},
        )
        for item in _fetch_all_dicts(item_result):
            key = (int(item["batch_id"]), str(item.get("group_key") or ""))
            role = str(item.get("role") or "").strip().lower()
            if role not in {"entry", "exit", "unknown"}:
                continue
            bucket = grouping_items_by_result.setdefault(key, {"entry": [], "exit": [], "unknown": []})
            try:
                trigger_id = int(item["trigger_id"])
            except (TypeError, ValueError):
                continue
            if trigger_id not in bucket[role]:
                bucket[role].append(trigger_id)
    trigger_ids: set[int] = set()
    entry_trigger_ids: set[int] = set()
    for row in rows:
        if isinstance(row.get("factor_payload"), str):
            try:
                row["factor_payload"] = json.loads(row["factor_payload"])
            except json.JSONDecodeError:
                pass
        if isinstance(row.get("grouping_result_payload"), str):
            try:
                row["grouping_result_payload"] = json.loads(row["grouping_result_payload"])
            except json.JSONDecodeError:
                pass
        payload = row.get("factor_payload")
        if not isinstance(payload, Mapping):
            payload = {}
            row["factor_payload"] = payload
        grouping_key = (int(row["batch_id"]), str(row.get("group_key") or ""))
        grouping_item_ids = grouping_items_by_result.get(grouping_key)
        if grouping_item_ids:
            mutable_payload = dict(payload)
            if not isinstance(mutable_payload.get("entry_trigger_ids"), list) and grouping_item_ids["entry"]:
                mutable_payload["entry_trigger_ids"] = grouping_item_ids["entry"]
            if not isinstance(mutable_payload.get("exit_trigger_ids"), list) and grouping_item_ids["exit"]:
                mutable_payload["exit_trigger_ids"] = grouping_item_ids["exit"]
            if not isinstance(mutable_payload.get("trigger_ids"), list):
                mutable_payload["trigger_ids"] = grouping_item_ids["entry"] + grouping_item_ids["exit"]
            row["factor_payload"] = mutable_payload
            payload = mutable_payload
        for key in ("entry_trigger_ids", "exit_trigger_ids", "trigger_ids"):
            values = payload.get(key)
            if isinstance(values, list):
                for value in values:
                    try:
                        trigger_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    trigger_ids.add(trigger_id)
                    if key == "entry_trigger_ids":
                        entry_trigger_ids.add(trigger_id)
    trigger_times_by_id: dict[int, Any] = {}
    if trigger_ids:
        trigger_result = db.execute(
            text(
                f"""
                select id, trigger_time
                from {trigger_table}
                where id in :trigger_ids
                """
            ).bindparams(bindparam("trigger_ids", expanding=True)),
            {"trigger_ids": sorted(trigger_ids)},
        )
        trigger_times_by_id = {int(row["id"]): row.get("trigger_time") for row in _fetch_all_dicts(trigger_result)}
    session_ids_by_batch_entry: dict[tuple[int, int], int] = {}
    if batch_ids and entry_trigger_ids:
        session_table = _table("session")
        if _column_exists(db, session_table, "grouping_id"):
            session_result = db.execute(
                text(
                    f"""
                    select id, grouping_id, entry_trigger_id
                    from {session_table}
                    where grouping_id in :batch_ids
                      and entry_trigger_id in :entry_trigger_ids
                    order by id desc
                    """
                ).bindparams(
                    bindparam("batch_ids", expanding=True),
                    bindparam("entry_trigger_ids", expanding=True),
                ),
                {
                    "batch_ids": batch_ids,
                    "entry_trigger_ids": sorted(entry_trigger_ids),
                },
            )
            for session_row in _fetch_all_dicts(session_result):
                if session_row.get("grouping_id") is None or session_row.get("entry_trigger_id") is None:
                    continue
                key = (int(session_row["grouping_id"]), int(session_row["entry_trigger_id"]))
                session_ids_by_batch_entry.setdefault(key, int(session_row["id"]))
    for row in rows:
        payload = row.get("factor_payload")
        if not isinstance(payload, Mapping):
            continue
        row["session_id"] = None
        row["session_window_start"] = payload.get("session_window_start")
        row["session_window_end"] = payload.get("session_window_end")

        def _ids_from_payload(key: str) -> list[int]:
            values = payload.get(key)
            if not isinstance(values, list):
                return []
            ids: list[int] = []
            for value in values:
                try:
                    ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            return ids

        entry_times = [trigger_times_by_id[trigger_id] for trigger_id in _ids_from_payload("entry_trigger_ids") if trigger_id in trigger_times_by_id]
        exit_times = [trigger_times_by_id[trigger_id] for trigger_id in _ids_from_payload("exit_trigger_ids") if trigger_id in trigger_times_by_id]
        if entry_times:
            row["session_window_start"] = min(entry_times)
        for entry_trigger_id in _ids_from_payload("entry_trigger_ids"):
            session_id = session_ids_by_batch_entry.get((int(row["batch_id"]), entry_trigger_id))
            if session_id is not None:
                row["session_id"] = session_id
                break
        if exit_times:
            row["session_window_end"] = max(exit_times)
        elif not row.get("session_window_end"):
            fallback_times = [
                trigger_times_by_id[trigger_id]
                for trigger_id in _ids_from_payload("trigger_ids")
                if trigger_id in trigger_times_by_id
            ]
            if fallback_times:
                row["session_window_start"] = row.get("session_window_start") or min(fallback_times)
                row["session_window_end"] = max(fallback_times)
    return rows


def get_filter_confidence_result(db: Session, confidence_result_id: int) -> dict[str, Any] | None:
    confidence_table = _table("filter_confidence_result")
    result = db.execute(
        text(
            f"""
            select id, batch_id, group_key, location_id, score, need_deep_analysis, reason, factor_payload
            from {confidence_table}
            where id = :confidence_result_id
            limit 1
            """
        ),
        {"confidence_result_id": confidence_result_id},
    )
    row = result.mappings().first()
    if row is None:
        return None
    payload = dict(row)
    if isinstance(payload.get("factor_payload"), str):
        try:
            payload["factor_payload"] = json.loads(payload["factor_payload"])
        except json.JSONDecodeError:
            pass
    return payload


def retry_filter_confidence_result(db: Session, confidence_result_id: int) -> dict[str, Any]:
    confidence_table = _table("filter_confidence_result")
    result = db.execute(
        text(
            f"""
            select id, batch_id, group_key, reason
            from {confidence_table}
            where id = :confidence_result_id
            limit 1
            """
        ),
        {"confidence_result_id": confidence_result_id},
    )
    row = result.mappings().first()
    if row is None:
        raise ValueError(f"Confidence result {confidence_result_id} was not found.")
    payload = dict(row)
    group_key = str(payload.get("group_key") or "")
    reason = str(payload.get("reason") or "")
    if group_key != "__error__" and reason != "confidence_error":
        raise ValueError("Only issue confidence results can be retried.")
    db.execute(
        text(
            f"""
            delete from {confidence_table}
            where id = :confidence_result_id
            """
        ),
        {"confidence_result_id": confidence_result_id},
    )
    db.commit()
    return {
        "ok": True,
        "confidence_result_id": confidence_result_id,
        "batch_id": int(payload["batch_id"]),
    }


def delete_filter_confidence_result(db: Session, confidence_result_id: int) -> dict[str, Any]:
    confidence_table = _table("filter_confidence_result")
    result = db.execute(
        text(
            f"""
            delete from {confidence_table}
            where id = :confidence_result_id
            """
        ),
        {"confidence_result_id": confidence_result_id},
    )
    db.commit()
    if int(result.rowcount or 0) <= 0:
        raise ValueError(f"Confidence result {confidence_result_id} was not found.")
    return {"ok": True, "confidence_result_id": confidence_result_id}


def promote_trigger_video_assets_to_full_retrieval(db: Session, trigger_ids: list[int]) -> int:
    normalized_ids = sorted({int(trigger_id) for trigger_id in trigger_ids if trigger_id is not None})
    if not normalized_ids:
        return 0
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            update {video_asset_table} va
            join {trigger_table} te on te.id = va.trigger_id
            set va.captured_start_time = date_sub(te.trigger_time, interval 40 second),
                va.captured_end_time = date_add(te.trigger_time, interval 10 second),
                va.status = 'not_retrieved',
                va.metadata = json_set(
                    coalesce(va.metadata, json_object()),
                    '$.promoted_from_layer0',
                    true,
                    '$.retrieval_mode',
                    'full_video',
                    '$.full_video_window_source',
                    'trigger_time',
                    '$.full_video_before_seconds',
                    40,
                    '$.full_video_after_seconds',
                    10
                )
            where va.trigger_id in :trigger_ids
              and va.section = 'entrance'
              and va.status in ('frames_retrieved', '10_frames_retrieved')
            """
        ).bindparams(bindparam("trigger_ids", expanding=True)),
        {"trigger_ids": normalized_ids},
    )
    db.commit()
    return int(result.rowcount or 0)


def list_pending_video_asset_analyses(db: Session, limit: int = 50) -> list[dict[str, Any]]:
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
                   va.video_url,
                   va.captured_start_time,
                   va.captured_end_time,
                   va.retrieved_at,
                   va.analyzed_at,
                   va.created_at,
                   coalesce(te.location_id, min(s.location_id)) as location_id,
                   min(sva.session_id) as session_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
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
            group by va.id, va.trigger_id, va.section, va.file_path, va.video_url,
                     va.captured_start_time, va.captured_end_time, va.retrieved_at,
                     va.analyzed_at, va.created_at, te.location_id
            order by coalesce(va.captured_start_time, va.created_at) asc, va.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def list_pending_kiosk_video_asset_analyses(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
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
                   s.location_id as location_id,
                   min(sva.session_id) as session_id
            from {video_asset_table} va
            inner join {session_video_asset_table} sva on sva.video_asset_id = va.id
            inner join {session_table} s on s.id = sva.session_id
            where va.section = 'kiosk'
              and va.status = 'ready'
              and s.status = 'pending'
            group by va.id, va.trigger_id, va.section, va.file_path, va.video_url,
                     va.captured_start_time, va.captured_end_time, va.retrieved_at,
                     va.analyzed_at, va.created_at, s.location_id
            order by coalesce(va.captured_start_time, va.created_at) asc, va.id asc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    return _fetch_all_dicts(result)


def list_running_video_asset_analyses(
    db: Session,
    *,
    sections: list[str] | None = None,
) -> list[dict[str, Any]]:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
    normalized_sections = [
        str(section).strip().lower()
        for section in (sections or [])
        if str(section or "").strip()
    ]
    params: dict[str, Any] = {}
    section_clause = ""
    if normalized_sections:
        placeholders: list[str] = []
        for index, section in enumerate(normalized_sections):
            key = f"section_{index}"
            placeholders.append(f":{key}")
            params[key] = section
        section_clause = f" and lower(va.section) in ({', '.join(placeholders)})"
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
                   coalesce(te.location_id, min(s.location_id)) as location_id,
                   min(sva.session_id) as session_id
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.section in ('entrance', 'kiosk')
              and va.status = 'processing'
              {section_clause}
            group by va.id, va.trigger_id, va.section, va.file_path, va.video_url,
                     va.captured_start_time, va.captured_end_time, va.retrieved_at,
                     va.analyzed_at, va.created_at, te.location_id
            order by va.id asc
            """
        ),
        params,
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
              and section in ('entrance', 'kiosk')
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
    script_run_table = _table("script_run")
    result = db.execute(
        text(
            f"""
            select va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                   va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.retention_until, va.status,
                   va.metadata, va.created_at,
                   coalesce(te.location_id, min(s.location_id)) as location_id,
                   count(distinct sva.id) as session_link_count,
                   min(sva.session_id) as primary_session_id,
                   group_concat(distinct sva.session_id order by sva.session_id separator ',') as session_ids,
                   (
                       select sr.script_name
                       from {script_run_table} sr
                       where sr.trigger_id = va.trigger_id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_name,
                   (
                       select sr.status
                       from {script_run_table} sr
                       where sr.trigger_id = va.trigger_id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_status,
                   (
                       select sr.finished_at
                       from {script_run_table} sr
                       where sr.trigger_id = va.trigger_id
                       order by sr.id desc
                       limit 1
                   ) as latest_script_finished_at,
                   (
                       select nullif(trim(sr.stderr_log), '')
                       from {script_run_table} sr
                       where sr.trigger_id = va.trigger_id
                         and sr.status = 'failed'
                       order by sr.id desc
                       limit 1
                   ) as latest_error_log,
                   case when va.status = 'issue' then true else false end as can_retry,
                   case
                       when va.status <> 'issue' then null
                       when lower(coalesce(va.section, '')) = 'kiosk' then 'ready'
                       when lower(coalesce((
                           select sr.script_name
                           from {script_run_table} sr
                           where sr.trigger_id = va.trigger_id
                             and sr.status = 'failed'
                           order by sr.id desc
                           limit 1
                       ), '')) in ('entry', 'kiosk') then 'ready'
                       else 'not_retrieved'
                   end as retry_to_status
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            group by va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                     va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at, va.retention_until, va.status,
                     va.metadata, va.created_at, te.location_id
            order by coalesce(va.captured_start_time, va.created_at) desc, va.id desc
            limit :limit
            """
        ),
        {"limit": limit},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
    return rows


def retry_video_asset_issue(db: Session, video_asset_id: int) -> dict[str, Any]:
    video_asset = get_video_asset(db, video_asset_id)
    if str(video_asset.get("status") or "") != "issue":
        raise ValueError("This video asset is not in issue state.")
    trigger_id = video_asset.get("trigger_id")
    section = str(video_asset.get("section") or "").strip().lower()
    session_id = None
    session_video_asset_table = _table("session_video_asset")
    linked_session = db.execute(
        text(
            f"""
            select session_id
            from {session_video_asset_table}
            where video_asset_id = :video_asset_id
            order by id asc
            limit 1
            """
        ),
        {"video_asset_id": video_asset_id},
    ).mappings().first()
    if linked_session is not None:
        session_id = linked_session.get("session_id")
    if trigger_id is None and session_id is None and section != "kiosk":
        raise ValueError("This video asset is not linked to a trigger or session.")
    script_run_table = _table("script_run")
    latest_failed_script = None
    if trigger_id is not None:
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
    elif session_id is not None:
        latest_failed_script = db.execute(
            text(
                f"""
                select script_name
                from {script_run_table}
                where session_id = :session_id
                  and status = 'failed'
                order by id desc
                limit 1
                """
            ),
            {"session_id": session_id},
        ).mappings().first()
    retry_to_status = _issue_video_retry_status(video_asset, latest_failed_script)
    update_video_asset(
        db,
        video_asset_id,
        {
            "video_url": video_asset.get("video_url"),
            "file_path": video_asset.get("file_path"),
            "captured_start_time": video_asset.get("captured_start_time"),
            "captured_end_time": video_asset.get("captured_end_time"),
            "retrieved_at": None if retry_to_status == "not_retrieved" else video_asset.get("retrieved_at"),
            "analyzed_at": None,
            "retention_until": video_asset.get("retention_until"),
            "status": retry_to_status,
            "metadata": video_asset.get("metadata"),
        },
    )
    if section == "kiosk" and retry_to_status == "ready" and session_id is not None:
        update_session_fields(
            db,
            session_id=int(session_id),
            status="pending",
            issue_reason=None,
        )
    return {
        "ok": True,
        "video_asset_id": video_asset_id,
        "trigger_id": int(trigger_id) if trigger_id is not None else None,
        "session_id": int(session_id) if session_id is not None else None,
        "new_status": retry_to_status,
    }


def restart_video_asset_analysis(db: Session, video_asset_id: int) -> dict[str, Any]:
    video_asset = get_video_asset(db, video_asset_id)
    section = str(video_asset.get("section") or "").strip().lower()
    if section not in {"entrance", "kiosk"}:
        raise ValueError("Only entrance and kiosk video assets can be re-analyzed.")
    current_status = str(video_asset.get("status") or "").strip().lower()
    if current_status == "retrieving":
        raise ValueError("This video asset is still retrieving.")
    if current_status == "processing":
        raise ValueError("This video asset is already processing.")
    if not (video_asset.get("file_path") or video_asset.get("video_url")):
        raise ValueError("This video asset does not have a source path or URL to analyze.")
    session_id = get_primary_session_id_for_video_asset(db, video_asset_id) if section == "kiosk" else None

    update_video_asset(
        db,
        video_asset_id,
        {
            "video_url": video_asset.get("video_url"),
            "file_path": video_asset.get("file_path"),
            "captured_start_time": video_asset.get("captured_start_time"),
            "captured_end_time": video_asset.get("captured_end_time"),
            "retrieved_at": video_asset.get("retrieved_at"),
            "analyzed_at": None,
            "retention_until": video_asset.get("retention_until"),
            "status": "ready",
            "metadata": video_asset.get("metadata"),
        },
    )
    if section == "kiosk" and session_id is not None:
        update_session_fields(
            db,
            session_id=int(session_id),
            status="pending",
            issue_reason=None,
        )
    return {
        "ok": True,
        "video_asset_id": video_asset_id,
        "session_id": int(session_id) if session_id is not None else None,
        "new_status": "ready",
    }


def get_session(db: Session, session_id: int) -> dict[str, Any]:
    session_table = _table("session")
    has_grouping_id = _column_exists(db, session_table, "grouping_id")
    grouping_id_select = ", grouping_id" if has_grouping_id else ""
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer,
                   result_summary, issue_reason
                   {grouping_id_select}
            from {session_table}
            where id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("result_summary"), str):
        try:
            row["result_summary"] = json.loads(row["result_summary"])
        except json.JSONDecodeError:
            pass
    row["confidence_result_id"] = _find_confidence_result_id_for_session(db, row)
    return row


def _find_confidence_result_id_for_session(db: Session, session: Mapping[str, Any]) -> int | None:
    grouping_id = session.get("grouping_id")
    entry_trigger_id = session.get("entry_trigger_id")
    if grouping_id is None or entry_trigger_id is None:
        return None
    confidence_table = _table("filter_confidence_result")
    grouping_item_table = _table("filter_grouping_item")
    result = db.execute(
        text(
            f"""
            select c.id
            from {confidence_table} c
            inner join {grouping_item_table} gi
                    on gi.batch_id = c.batch_id
                   and gi.group_key = c.group_key
                   and gi.role = 'entry'
                   and gi.trigger_id = :entry_trigger_id
                   and gi.status <> 'deleted'
            where c.batch_id = :grouping_id
            order by c.created_at desc, c.id desc
            limit 1
            """
        ),
        {"grouping_id": int(grouping_id), "entry_trigger_id": int(entry_trigger_id)},
    )
    row = result.mappings().first()
    return int(row["id"]) if row else None


def get_session_by_entry_trigger_id(db: Session, entry_trigger_id: int) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer,
                   result_summary, issue_reason
            from {session_table}
            where entry_trigger_id = :entry_trigger_id
            order by id asc
            limit 1
            """
        ),
        {"entry_trigger_id": entry_trigger_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("result_summary"), str):
        try:
            row["result_summary"] = json.loads(row["result_summary"])
        except json.JSONDecodeError:
            pass
    return row


def get_session_by_trigger_pair(db: Session, entry_trigger_id: int, exit_trigger_id: int) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer,
                   result_summary, issue_reason
            from {session_table}
            where entry_trigger_id = :entry_trigger_id
              and exit_trigger_id = :exit_trigger_id
            order by id asc
            limit 1
            """
        ),
        {"entry_trigger_id": entry_trigger_id, "exit_trigger_id": exit_trigger_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("result_summary"), str):
        try:
            row["result_summary"] = json.loads(row["result_summary"])
        except json.JSONDecodeError:
            pass
    return row


def get_latest_open_session_by_location(db: Session, location_id: int) -> dict[str, Any]:
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select id, entry_trigger_id, exit_trigger_id, location_id, status, start_time, end_time,
                   total_item_brought, actual_items_brought, transaction_total_items, total_customer,
                   result_summary, issue_reason
            from {session_table}
            where location_id = :location_id
              and end_time is null
              and status not in ('detected', 'not_detected', 'closed', 'issue', 'whitelisted')
            order by created_at desc, id desc
            limit 1
            """
        ),
        {"location_id": location_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("result_summary"), str):
        try:
            row["result_summary"] = json.loads(row["result_summary"])
        except json.JSONDecodeError:
            pass
    return row


def update_session_summary(
    db: Session,
    *,
    session_id: int,
    status: str | None = None,
    total_customer: int | None = None,
    transaction_total_items: int | None = None,
    result_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session_table = _table("session")
    db.execute(
        text(
            f"""
            update {session_table}
            set status = coalesce(:status, status),
                total_customer = coalesce(:total_customer, total_customer),
                transaction_total_items = coalesce(:transaction_total_items, transaction_total_items),
                result_summary = coalesce(:result_summary, result_summary)
            where id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "status": status,
            "total_customer": total_customer,
            "transaction_total_items": transaction_total_items,
            "result_summary": _json_dumps(dict(result_summary)) if result_summary is not None else None,
        },
    )
    db.commit()
    return get_session(db, session_id)


def update_session_fields(
    db: Session,
    *,
    session_id: int,
    status: str | None = None,
    end_time: Any | None = None,
    exit_trigger_id: int | None = None,
    total_customer: int | None = None,
    transaction_total_items: int | None = None,
    result_summary: Mapping[str, Any] | None = None,
    issue_reason: str | None = None,
) -> dict[str, Any]:
    session_table = _table("session")
    db.execute(
        text(
            f"""
            update {session_table}
            set status = coalesce(:status, status),
                end_time = coalesce(:end_time, end_time),
                exit_trigger_id = coalesce(:exit_trigger_id, exit_trigger_id),
                total_customer = coalesce(:total_customer, total_customer),
                transaction_total_items = coalesce(:transaction_total_items, transaction_total_items),
                result_summary = coalesce(:result_summary, result_summary),
                issue_reason = :issue_reason
            where id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "status": status,
            "end_time": end_time,
            "exit_trigger_id": exit_trigger_id,
            "total_customer": total_customer,
            "transaction_total_items": transaction_total_items,
            "result_summary": _json_dumps(dict(result_summary)) if result_summary is not None else None,
            "issue_reason": issue_reason,
        },
    )
    db.commit()
    return get_session(db, session_id)


def update_session_grouping_link(
    db: Session,
    *,
    session_id: int,
    grouping_id: int,
) -> dict[str, Any]:
    session_table = _table("session")
    if not _column_exists(db, session_table, "grouping_id"):
        return get_session(db, session_id)
    db.execute(
        text(
            f"""
            update {session_table}
            set grouping_id = :grouping_id
            where id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "grouping_id": grouping_id,
        },
    )
    db.commit()
    return get_session(db, session_id)


def get_session_customer_count(db: Session, session_id: int) -> int:
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select count(*) as customer_count
            from {session_customer_table}
            where session_id = :session_id
              and merged_into_session_customer_id is null
            """
        ),
        {"session_id": session_id},
    )
    row = _fetch_one_dict(result)
    return int(row.get("customer_count") or 0)


def delete_session(db: Session, session_id: int) -> None:
    session_table = _table("session")
    db.execute(
        text(
            f"""
            delete from {session_table}
            where id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    db.commit()


def retry_session_issue(db: Session, session_id: int) -> dict[str, Any]:
    session = get_session(db, session_id)
    current_status = str(session.get("status") or "").strip().lower()
    if current_status not in {"issue", "closed", "pending", "need_review"}:
        raise ValueError("This session is not in issue, need_review, closed, or pending state.")

    # The pipeline is entrance -> kiosk. If entrance itself is broken, retrying
    # kiosk first is pointless (and used to be all this function did) - kiosk
    # can't produce a meaningful result without entrance having actually run.
    entrance_videos = list_session_video_assets(db, session_id=session_id, section="entrance")
    issue_entrance_video = next(
        (row for row in entrance_videos if str(row.get("video_status") or "") == "issue"),
        None,
    )
    if issue_entrance_video is not None:
        entrance_retry_result = retry_video_asset_issue(db, int(issue_entrance_video["video_asset_id"]))
        updated = update_session_fields(
            db,
            session_id=session_id,
            status="pending",
            issue_reason=None,
        )
        return {
            "ok": True,
            "session_id": session_id,
            "new_status": str(updated.get("status") or "pending"),
            "retried_stage": "entrance",
            "entrance_retry": entrance_retry_result,
        }

    # need_review means entrance and kiosk both ran fine, but the kiosk item
    # count didn't clearly match the transaction - there's no "issue" video to
    # reset here, so retrying means forcing kiosk analysis to run again.
    if current_status == "need_review":
        kiosk_videos = list_session_video_assets(db, session_id=session_id, section="kiosk")
        retriable_kiosk_video = next(
            (row for row in kiosk_videos if str(row.get("video_status") or "") not in {"retrieving", "processing"}),
            None,
        )
        if retriable_kiosk_video is not None:
            kiosk_retry_result = restart_video_asset_analysis(db, int(retriable_kiosk_video["video_asset_id"]))
            return {
                "ok": True,
                "session_id": session_id,
                "new_status": "pending",
                "retried_stage": "kiosk",
                "kiosk_retry": kiosk_retry_result,
            }

    updated = update_session_fields(
        db,
        session_id=session_id,
        status="pending",
        issue_reason=None,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "new_status": str(updated.get("status") or "pending"),
        "retried_stage": "kiosk",
    }


def list_sessions(
    db: Session, limit: int = 50, *, offset: int = 0, session_id: int | None = None
) -> list[dict[str, Any]]:
    session_table = _table("session")
    location_table = settings.location_table_name
    location_id_column = settings.location_id_column
    location_name_column = settings.location_name_column
    has_grouping_id = _column_exists(db, session_table, "grouping_id")
    grouping_id_select = ", s.grouping_id" if has_grouping_id else ""
    where_sql = "where s.id = :session_id" if session_id is not None else ""
    result = db.execute(
        text(
            f"""
            select s.id, s.entry_trigger_id, s.exit_trigger_id, s.location_id, s.status,
                   s.start_time, s.end_time, s.total_item_brought, s.actual_items_brought,
                   s.transaction_total_items, s.total_customer, s.issue_reason, s.result_summary{grouping_id_select},
                   l.{location_name_column} as location_name,
                   case when s.status in ('issue', 'closed', 'need_review')
                          or (s.status = 'pending' and s.end_time is not null)
                        then true else false end as can_retry,
                   s.created_at, s.updated_at
            from {session_table} s
            left join {location_table} l on l.{location_id_column} = s.location_id
            {where_sql}
            order by s.id desc
            limit :limit offset :offset
            """
        ),
        {"limit": limit, "offset": offset, "session_id": session_id},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("result_summary"), str):
            try:
                row["result_summary"] = json.loads(row["result_summary"])
            except json.JSONDecodeError:
                pass
        row["session_videos"] = list_session_video_assets(db, session_id=int(row["id"]))
        row["confidence_result_id"] = _find_confidence_result_id_for_session(db, row)
    return rows


def count_sessions(db: Session) -> int:
    session_table = _table("session")
    result = db.execute(text(f"select count(*) as total from {session_table}"))
    row = _fetch_one_dict(result)
    return int(row.get("total") or 0)


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


def delete_session_transactions(db: Session, session_id: int) -> None:
    transaction_table = _table("session_transaction")
    db.execute(
        text(
            f"""
            delete from {transaction_table}
            where session_id = :session_id
            """
        ),
        {"session_id": session_id},
    )
    db.commit()


def list_session_transactions(db: Session, session_id: int) -> list[dict[str, Any]]:
    transaction_table = _table("session_transaction")
    result = db.execute(
        text(
            f"""
            select id, session_id, receipt_number, transaction_time, total_items, total_amount, raw_payload
            from {transaction_table}
            where session_id = :session_id
            order by transaction_time asc, id asc
            """
        ),
        {"session_id": session_id},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("raw_payload"), str):
            try:
                row["raw_payload"] = json.loads(row["raw_payload"])
            except json.JSONDecodeError:
                pass
    return rows


def list_session_transaction_details(db: Session, session_id: int) -> list[dict[str, Any]]:
    rows = list_session_transactions(db, session_id)
    payload: list[dict[str, Any]] = []
    for row in rows:
        raw_payload = row.get("raw_payload") or {}
        details = raw_payload.get("details") if isinstance(raw_payload, Mapping) else None
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            detail_payload = detail.get("raw_payload") if isinstance(detail.get("raw_payload"), Mapping) else {}
            quantity_value = _pick_first(detail, "quantity", "qty")
            if quantity_value is None:
                quantity_value = _pick_first(detail_payload, "quantity", "qty")
            try:
                quantity = int(quantity_value or 0)
            except (TypeError, ValueError):
                quantity = 0
            price_value = _pick_first(detail, "price", "unit_price", "unitPrice", "sellingPrice", "priceAmount")
            if price_value is None:
                price_value = _pick_first(
                    detail_payload,
                    "price",
                    "unit_price",
                    "unitPrice",
                    "sellingPrice",
                    "priceAmount",
                )
            try:
                price = float(price_value) if price_value is not None else None
            except (TypeError, ValueError):
                price = None
            subtotal_value = _pick_first(detail, "subtotal", "subTotal", "total", "line_total", "lineTotal")
            if subtotal_value is None:
                subtotal_value = _pick_first(
                    detail_payload,
                    "subtotal",
                    "subTotal",
                    "total",
                    "line_total",
                    "lineTotal",
                )
            try:
                subtotal = float(subtotal_value) if subtotal_value is not None else None
            except (TypeError, ValueError):
                subtotal = None
            if subtotal is None and price is not None:
                subtotal = float(price) * max(0, quantity)
            payload.append(
                {
                    "session_transaction_id": int(row["id"]),
                    "session_id": int(row["session_id"]),
                    "receipt_number": row.get("receipt_number"),
                    "transaction_time": row.get("transaction_time"),
                    "transaction_total_amount": row.get("total_amount"),
                    "transaction_status": _pick_first(
                        raw_payload,
                        "status",
                        "Status",
                        "transaction_status",
                        "transactionStatus",
                    )
                    or settings.paid_transaction_status_value,
                    "item_name": _pick_first(
                        detail,
                        "item_name",
                        "itemName",
                        "name",
                        "text",
                        "product_name",
                        "productName",
                    )
                    or _pick_first(
                        detail_payload,
                        "item_name",
                        "itemName",
                        "name",
                        "text",
                        "product_name",
                        "productName",
                    ),
                    "barcode": _pick_first(detail, "barcode", "barCode", "sku", "SKU", "code")
                    or _pick_first(detail_payload, "barcode", "barCode", "sku", "SKU", "code"),
                    "quantity": max(0, quantity),
                    "price": price,
                    "subtotal": subtotal,
                    "raw_payload": dict(detail),
                }
            )
    return payload


def list_session_customers(db: Session, session_id: int) -> list[dict[str, Any]]:
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select id, session_id, person_id, merged_into_session_customer_id, enter_time,
                   kiosk_start_time, leave_time, match_status, merge_reason, merged_at,
                   created_at, updated_at
            from {session_customer_table}
            where session_id = :session_id
              and merged_into_session_customer_id is null
            order by person_id asc, id asc
            """
        ),
        {"session_id": session_id},
    )
    return _fetch_all_dicts(result)


def list_paid_transactions_for_session_window(
    db: Session,
    *,
    location_id: int,
    start_time,
    end_time,
) -> list[dict[str, Any]]:
    transaction_table = _qualified_paid_table(settings.paid_transaction_table_name)
    detail_table = _qualified_paid_table(settings.paid_transaction_detail_table_name)
    transaction_id_column = _quote_identifier(PAID_TRANSACTION_ID_COLUMN)
    location_id_column = _quote_identifier(settings.paid_transaction_location_id_column)
    transaction_time_column = _quote_identifier(PAID_TRANSACTION_TIME_COLUMN)
    transaction_status_column = _quote_identifier(settings.paid_transaction_status_column)
    receipt_column = _quote_identifier(settings.paid_transaction_receipt_column)
    total_amount_column = _quote_identifier(settings.paid_transaction_total_amount_column)
    detail_transaction_id_name = settings.paid_transaction_detail_transaction_id_column
    detail_transaction_id_column = _quote_identifier(detail_transaction_id_name)

    txn_result = db.execute(
        text(
            f"""
            select t.*
            from {transaction_table} t
            where t.{location_id_column} = :location_id
              and t.{transaction_status_column} = :status_value
              and t.{transaction_time_column} between :start_time and :end_time
            order by t.{transaction_time_column} asc, t.{transaction_id_column} asc
            """
        ),
        {
            "location_id": location_id,
            "status_value": settings.paid_transaction_status_value,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    transactions = _fetch_all_dicts(txn_result)
    if not transactions:
        return []

    transaction_ids = [
        row.get(PAID_TRANSACTION_ID_COLUMN)
        for row in transactions
        if row.get(PAID_TRANSACTION_ID_COLUMN) is not None
    ]
    details_by_transaction_id: dict[Any, list[dict[str, Any]]] = {}
    if transaction_ids:
        detail_result = db.execute(
            text(
                f"""
                select d.*
                from {detail_table} d
                where d.{detail_transaction_id_column} in :transaction_ids
                order by d.{detail_transaction_id_column} asc
                """
            ).bindparams(bindparam("transaction_ids", expanding=True)),
            {"transaction_ids": transaction_ids},
        )
        for row in _fetch_all_dicts(detail_result):
            txn_id = row.get(detail_transaction_id_name)
            details_by_transaction_id.setdefault(txn_id, []).append(row)

    payload_by_receipt: dict[Any, dict[str, Any]] = {}
    for row in transactions:
        txn_id = row.get(PAID_TRANSACTION_ID_COLUMN)
        details = details_by_transaction_id.get(txn_id, [])
        total_items = 0
        total_subtotal = 0.0
        normalized_details: list[dict[str, Any]] = []
        for detail in details:
            try:
                quantity = int(detail.get(settings.paid_transaction_detail_quantity_column) or 0)
            except (TypeError, ValueError):
                quantity = 0
            total_items += max(0, quantity)
            subtotal_value = _pick_first(detail, "subtotal", "subTotal", "total", "line_total", "lineTotal")
            try:
                subtotal = float(subtotal_value) if subtotal_value is not None else None
            except (TypeError, ValueError):
                subtotal = None
            if subtotal is not None:
                total_subtotal += subtotal
            normalized_details.append(
                {
                    "quantity": max(0, quantity),
                    "item_name": _pick_first(
                        detail,
                        settings.paid_transaction_detail_item_name_column,
                        "item_name",
                        "itemName",
                        "name",
                        "text",
                        "product_name",
                        "productName",
                    ),
                    "barcode": _pick_first(detail, "barcode", "barCode", "sku", "SKU", "code"),
                    "price": _pick_first(
                        detail,
                        "price",
                        "unit_price",
                        "unitPrice",
                        "sellingPrice",
                        "priceAmount",
                    ),
                    "subtotal": subtotal_value,
                    "raw_payload": detail,
                }
            )
        receipt_number = row.get(settings.paid_transaction_receipt_column) or txn_id
        total_amount_value = row.get(settings.paid_transaction_total_amount_column)
        try:
            total_amount = float(total_amount_value) if total_amount_value is not None else total_subtotal
        except (TypeError, ValueError):
            total_amount = total_subtotal
        existing = payload_by_receipt.get(receipt_number)
        if existing is None:
            payload_by_receipt[receipt_number] = {
                "transaction_id": txn_id,
                "receipt_number": receipt_number,
                "transaction_time": row.get(PAID_TRANSACTION_TIME_COLUMN),
                "created_at": _pick_first(row, "createdAt", "created_at", PAID_TRANSACTION_TIME_COLUMN),
                "location_id": row.get(settings.paid_transaction_location_id_column),
                "status": row.get(settings.paid_transaction_status_column),
                "total_amount": total_amount,
                "total_items": total_items,
                "raw_payload": row,
                "details": normalized_details,
            }
            continue
        if _pick_first(row, "createdAt", "created_at", PAID_TRANSACTION_TIME_COLUMN):
            existing["created_at"] = _pick_first(row, "createdAt", "created_at", PAID_TRANSACTION_TIME_COLUMN)
        if row.get(PAID_TRANSACTION_TIME_COLUMN):
            existing["transaction_time"] = row.get(PAID_TRANSACTION_TIME_COLUMN)
        try:
            existing_amount = float(existing.get("total_amount") or 0)
        except (TypeError, ValueError):
            existing_amount = 0.0
        if total_amount > existing_amount:
            existing["total_amount"] = total_amount
    return list(payload_by_receipt.values())


def list_non_paid_transactions_for_session_window(
    db: Session,
    *,
    location_id: int,
    start_time,
    end_time,
) -> list[dict[str, Any]]:
    transaction_table = _qualified_paid_table(settings.paid_transaction_table_name)
    transaction_id_column = _quote_identifier(PAID_TRANSACTION_ID_COLUMN)
    location_id_column = _quote_identifier(settings.paid_transaction_location_id_column)
    transaction_time_column = _quote_identifier(PAID_TRANSACTION_TIME_COLUMN)
    transaction_status_column = _quote_identifier(settings.paid_transaction_status_column)
    result = db.execute(
        text(
            f"""
            select t.*
            from {transaction_table} t
            where t.{location_id_column} = :location_id
              and coalesce(t.{transaction_status_column}, '') <> :paid_status
              and t.{transaction_time_column} between :start_time and :end_time
            order by t.{transaction_time_column} asc, t.{transaction_id_column} asc
            """
        ),
        {
            "location_id": location_id,
            "paid_status": settings.paid_transaction_status_value,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    rows = _fetch_all_dicts(result)
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "transaction_id": row.get(PAID_TRANSACTION_ID_COLUMN),
                "receipt_number": row.get(settings.paid_transaction_receipt_column),
                "transaction_time": row.get(PAID_TRANSACTION_TIME_COLUMN),
                "created_at": _pick_first(row, "createdAt", "created_at", PAID_TRANSACTION_TIME_COLUMN),
                "location_id": row.get(settings.paid_transaction_location_id_column),
                "status": row.get(settings.paid_transaction_status_column),
                "total_amount": row.get(settings.paid_transaction_total_amount_column),
                "raw_payload": row,
            }
        )
    return payload


def list_minus_button_alerts_for_window(
    db: Session,
    *,
    location_id: int,
    start_time,
    end_time,
) -> list[dict[str, Any]]:
    table_name = _table(settings.thief_alert_table_name)
    checked_column = _quote_identifier(settings.thief_alert_checked_column)
    result = db.execute(
        text(
            f"""
            select id,
                   locationId as location_id,
                   method,
                   detail,
                   {checked_column} as checked,
                   createdAt as created_at
            from {table_name}
            where locationId = :location_id
              and createdAt between :start_time and :end_time
              and lower(coalesce(method, '')) = 'kiosk'
              and lower(coalesce(detail, '')) like '%minus%'
            order by createdAt asc, id asc
            """
        ),
        {
            "location_id": location_id,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        row["checked"] = bool(row.get("checked"))
    return rows


def create_trigger(db: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    trigger_table = _table("trigger_event")
    result = db.execute(
        text(
            f"""
            insert into {trigger_table} (
                location_id, aqara_event_id, trigger_source, trigger_time,
                phone_entry_id, credit_card_entry_id, entry_source_type, entry_match_status, raw_payload
            )
            values (
                :location_id, :aqara_event_id, :trigger_source, :trigger_time,
                :phone_entry_id, :credit_card_entry_id, :entry_source_type, :entry_match_status, :raw_payload
            )
            """
        ),
        {
            **payload,
            "raw_payload": _json_dumps(payload.get("raw_payload")) if payload.get("raw_payload") is not None else None,
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
                status = case
                    when status in ('detected', 'not_detected', 'issue') then status
                    else 'pending'
                end,
                issue_reason = case
                    when status in ('detected', 'not_detected', 'issue') then issue_reason
                    else null
                end
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


def list_session_customers_by_session_person(
    db: Session,
    session_id: int,
    person_id: int,
) -> list[dict[str, Any]]:
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select id, session_id, person_id, merged_into_session_customer_id, enter_time,
                   kiosk_start_time, leave_time, match_status, merge_reason, merged_at,
                   created_at, updated_at
            from {session_customer_table}
            where session_id = :session_id and person_id = :person_id
            order by case when merged_into_session_customer_id is null then 0 else 1 end asc,
                     id asc
            """
        ),
        {"session_id": session_id, "person_id": person_id},
    )
    return _fetch_all_dicts(result)


def get_session_customer_by_session_person(db: Session, session_id: int, person_id: int) -> dict[str, Any]:
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select id, session_id, person_id, merged_into_session_customer_id, enter_time,
                   kiosk_start_time, leave_time, match_status, merge_reason, merged_at,
                   created_at, updated_at
            from {session_customer_table}
            where session_id = :session_id and person_id = :person_id
              and merged_into_session_customer_id is null
            order by id asc
            limit 1
            """
        ),
        {"session_id": session_id, "person_id": person_id},
    )
    return _fetch_one_dict(result)


def get_session_customer(db: Session, session_customer_id: int) -> dict[str, Any]:
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select id, session_id, person_id, merged_into_session_customer_id, enter_time,
                   kiosk_start_time, leave_time, match_status, merge_reason, merged_at,
                   created_at, updated_at
            from {session_customer_table}
            where id = :session_customer_id
            """
        ),
        {"session_customer_id": session_customer_id},
    )
    return _fetch_one_dict(result)


def delete_session_customer(db: Session, session_customer_id: int) -> dict[str, Any]:
    row = get_session_customer(db, session_customer_id)
    session_customer_table = _table("session_customer")
    db.execute(
        text(
            f"""
            delete from {session_customer_table}
            where id = :session_customer_id
            """
        ),
        {"session_customer_id": session_customer_id},
    )
    db.commit()
    return row


def get_latest_open_session_customer_by_location_person(
    db: Session,
    *,
    location_id: int,
    person_id: int,
) -> dict[str, Any]:
    session_table = _table("session")
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select sc.id, sc.session_id, sc.person_id, sc.merged_into_session_customer_id,
                   sc.enter_time, sc.kiosk_start_time, sc.leave_time, sc.match_status,
                   sc.merge_reason, sc.merged_at, sc.created_at, sc.updated_at
            from {session_customer_table} sc
            join {session_table} s on s.id = sc.session_id
            where s.location_id = :location_id
              and sc.person_id = :person_id
              and sc.merged_into_session_customer_id is null
              and sc.leave_time is null
              and s.end_time is null
              and s.status not in ('detected', 'not_detected', 'closed', 'issue', 'whitelisted')
            order by sc.created_at desc, sc.id desc
            limit 1
            """
        ),
        {"location_id": location_id, "person_id": person_id},
    )
    return _fetch_one_dict(result)


def list_active_session_customers_for_location(
    db: Session,
    *,
    location_id: int,
) -> list[dict[str, Any]]:
    session_table = _table("session")
    session_customer_table = _table("session_customer")
    result = db.execute(
        text(
            f"""
            select sc.id, sc.session_id, sc.person_id, sc.merged_into_session_customer_id,
                   sc.enter_time, sc.kiosk_start_time, sc.leave_time, sc.match_status,
                   sc.merge_reason, sc.merged_at, sc.created_at, sc.updated_at
            from {session_customer_table} sc
            join {session_table} s on s.id = sc.session_id
            where s.location_id = :location_id
              and sc.merged_into_session_customer_id is null
              and sc.leave_time is null
            order by sc.created_at desc, sc.id desc
            """
        ),
        {"location_id": location_id},
    )
    return _fetch_all_dicts(result)


def update_session_customer_leave_time(
    db: Session,
    *,
    session_customer_id: int,
    leave_time,
    match_status: str | None = None,
) -> None:
    session_customer_table = _table("session_customer")
    db.execute(
        text(
            f"""
            update {session_customer_table}
            set leave_time = case
                    when leave_time is null then :leave_time
                    when :leave_time is null then leave_time
                    when leave_time < :leave_time then :leave_time
                    else leave_time
                end,
                match_status = coalesce(:match_status, match_status)
            where id = :session_customer_id
            """
        ),
        {
            "session_customer_id": session_customer_id,
            "leave_time": leave_time,
            "match_status": match_status,
        },
    )
    db.commit()


def merge_session_customer_aliases(
    db: Session,
    *,
    canonical_session_customer_id: int,
    alias_session_customer_ids: list[int],
    merge_reason: str,
) -> None:
    normalized_alias_ids = sorted(
        {
            int(value)
            for value in alias_session_customer_ids
            if value is not None and int(value) != int(canonical_session_customer_id)
        }
    )
    if not normalized_alias_ids:
        return

    session_customer_table = _table("session_customer")
    db.execute(
        text(
            f"""
            update {session_customer_table}
            set merged_into_session_customer_id = :canonical_session_customer_id,
                merge_reason = :merge_reason,
                merged_at = now()
            where id = any(:alias_session_customer_ids)
            """
        ),
        {
            "canonical_session_customer_id": canonical_session_customer_id,
            "alias_session_customer_ids": normalized_alias_ids,
            "merge_reason": merge_reason,
        },
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
            "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
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
            "metadata": _json_dumps(metadata) if metadata is not None else None,
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
            "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
        },
    )
    db.commit()


def delete_video_asset(db: Session, video_asset_id: int) -> None:
    video_asset_table = _table("video_asset")
    db.execute(
        text(
            f"""
            delete from {video_asset_table}
            where id = :video_asset_id
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    db.commit()


def create_session_video_asset_link(db: Session, session_id: int, video_asset_id: int, payload: Mapping[str, Any]) -> None:
    session_video_asset_table = _table("session_video_asset")
    section_value = str(payload.get("link_section") or payload.get("section") or "").strip().lower()
    if section_value == "entry":
        db.execute(
            text(
                f"""
                delete from {session_video_asset_table}
                where session_id = :session_id
                  and section = 'entry'
                  and video_asset_id <> :video_asset_id
                """
            ),
            {
                "session_id": session_id,
                "video_asset_id": video_asset_id,
            },
        )
        db.execute(
            text(
                f"""
                delete from {session_video_asset_table}
                where session_id = :session_id
                  and section = 'entry'
                  and video_asset_id = :video_asset_id
                """
            ),
            {
                "session_id": session_id,
                "video_asset_id": video_asset_id,
            },
        )
        db.execute(
            text(
                f"""
                delete from {session_video_asset_table}
                where video_asset_id = :video_asset_id
                  and section = 'entry'
                  and session_id <> :session_id
                """
            ),
            {
                "session_id": session_id,
                "video_asset_id": video_asset_id,
            },
        )
    elif section_value:
        db.execute(
            text(
                f"""
                delete from {session_video_asset_table}
                where session_id = :session_id
                  and video_asset_id = :video_asset_id
                  and section = :section
                """
            ),
            {
                "session_id": session_id,
                "video_asset_id": video_asset_id,
                "section": section_value,
            },
        )
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
            "section": section_value,
            "sequence_no": payload.get("link_sequence_no", payload.get("sequence_no")),
            "clip_start_time": payload.get("clip_start_time"),
            "clip_end_time": payload.get("clip_end_time"),
            "is_primary": 1 if payload.get("is_primary") else 0,
            "metadata": _json_dumps(payload.get("metadata")) if payload.get("metadata") is not None else None,
        },
    )
    db.commit()


def find_video_asset_by_window(
    db: Session,
    *,
    section: str,
    location_id: int,
    start_time: Any,
    end_time: Any,
) -> dict[str, Any] | None:
    video_asset_table = _table("video_asset")
    trigger_table = _table("trigger_event")
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select va.id, va.trigger_id, va.section, va.sequence_no, va.video_url, va.file_path,
                   va.captured_start_time, va.captured_end_time, va.retrieved_at, va.analyzed_at,
                   va.retention_until, va.status, va.metadata, va.created_at
            from {video_asset_table} va
            left join {trigger_table} te on te.id = va.trigger_id
            left join {session_video_asset_table} sva on sva.video_asset_id = va.id
            left join {session_table} s on s.id = sva.session_id
            where va.section = :section
              and va.captured_start_time = :start_time
              and va.captured_end_time = :end_time
              and coalesce(
                    te.location_id,
                    s.location_id,
                    cast(json_unquote(json_extract(va.metadata, '$.location_id')) as unsigned)
                  ) = :location_id
              and va.status <> 'deleted'
            order by va.id asc
            limit 1
            """
        ),
        {
            "section": section,
            "start_time": start_time,
            "end_time": end_time,
            "location_id": location_id,
        },
    )
    row = result.mappings().first()
    if row is None:
        return None
    payload = dict(row)
    if isinstance(payload.get("metadata"), str):
        try:
            payload["metadata"] = json.loads(payload["metadata"])
        except json.JSONDecodeError:
            pass
    return payload


def list_video_asset_session_links(db: Session, video_asset_id: int) -> list[dict[str, Any]]:
    session_video_asset_table = _table("session_video_asset")
    session_table = _table("session")
    result = db.execute(
        text(
            f"""
            select sva.id, sva.session_id, sva.video_asset_id, sva.section, sva.sequence_no,
                   sva.clip_start_time, sva.clip_end_time, sva.is_primary, sva.metadata,
                   s.location_id, s.status as session_status, s.start_time as session_start_time,
                   s.end_time as session_end_time
            from {session_video_asset_table} sva
            join {session_table} s on s.id = sva.session_id
            where sva.video_asset_id = :video_asset_id
            order by sva.section asc, sva.is_primary desc, sva.session_id asc, sva.id asc
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
    return rows


def delete_session_video_asset_links_for_video_asset_except(
    db: Session,
    *,
    video_asset_id: int,
    keep_session_id: int,
    section: str | None = None,
) -> None:
    session_video_asset_table = _table("session_video_asset")
    params: dict[str, Any] = {
        "video_asset_id": video_asset_id,
        "keep_session_id": keep_session_id,
    }
    where_section = ""
    if section is not None:
        where_section = " and section = :section"
        params["section"] = section
    db.execute(
        text(
            f"""
            delete from {session_video_asset_table}
            where video_asset_id = :video_asset_id
              and session_id <> :keep_session_id
              {where_section}
            """
        ),
        params,
    )
    db.commit()


def delete_session_video_asset_link(
    db: Session,
    *,
    session_id: int,
    video_asset_id: int,
    section: str | None = None,
) -> bool:
    session_video_asset_table = _table("session_video_asset")
    params: dict[str, Any] = {
        "session_id": session_id,
        "video_asset_id": video_asset_id,
    }
    where_section = ""
    if section is not None:
        where_section = " and section = :section"
        params["section"] = section
    result = db.execute(
        text(
            f"""
            delete from {session_video_asset_table}
            where session_id = :session_id
              and video_asset_id = :video_asset_id
              {where_section}
            """
        ),
        params,
    )
    db.commit()
    return bool(result.rowcount)


def get_primary_session_id_for_video_asset(db: Session, video_asset_id: int) -> int | None:
    session_video_asset_table = _table("session_video_asset")
    result = db.execute(
        text(
            f"""
            select session_id
            from {session_video_asset_table}
            where video_asset_id = :video_asset_id
            order by is_primary desc, session_id asc
            limit 1
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    row = result.mappings().first()
    if row is None or row.get("session_id") is None:
        return None
    return int(row["session_id"])


def list_session_video_assets(
    db: Session,
    *,
    session_id: int,
    section: str | None = None,
) -> list[dict[str, Any]]:
    session_video_asset_table = _table("session_video_asset")
    video_asset_table = _table("video_asset")
    if section is None:
        result = db.execute(
            text(
                f"""
                select sva.id, sva.session_id, sva.video_asset_id, sva.section, sva.sequence_no,
                       sva.clip_start_time, sva.clip_end_time, sva.is_primary, sva.metadata,
                       va.status as video_status, va.file_path, va.video_url
                from {session_video_asset_table} sva
                join {video_asset_table} va on va.id = sva.video_asset_id
                where sva.session_id = :session_id
                order by sva.section asc, sva.sequence_no asc, sva.id asc
                """
            ),
            {"session_id": session_id},
        )
    else:
        result = db.execute(
            text(
                f"""
                select sva.id, sva.session_id, sva.video_asset_id, sva.section, sva.sequence_no,
                       sva.clip_start_time, sva.clip_end_time, sva.is_primary, sva.metadata,
                       va.status as video_status, va.file_path, va.video_url
                from {session_video_asset_table} sva
                join {video_asset_table} va on va.id = sva.video_asset_id
                where sva.session_id = :session_id and sva.section = :section
                order by sva.sequence_no asc, sva.id asc
                """
            ),
            {"session_id": session_id, "section": section},
        )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("metadata"), str):
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
    return rows


def create_transaction(db: Session, session_id: int, payload: Mapping[str, Any]) -> int:
    transaction_table = _table("session_transaction")
    result = db.execute(
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
            "raw_payload": _json_dumps(payload.get("raw_payload")) if payload.get("raw_payload") is not None else None,
        },
    )
    db.commit()
    inserted_id = getattr(result, "lastrowid", None)
    if inserted_id is None:
        row = db.execute(text("select last_insert_id()")).first()
        inserted_id = row[0] if row else None
    return int(inserted_id or 0)


def update_session_transaction_raw_payload(
    db: Session,
    session_transaction_id: int,
    raw_payload: Mapping[str, Any],
) -> None:
    transaction_table = _table("session_transaction")
    db.execute(
        text(
            f"""
            update {transaction_table}
            set raw_payload = :raw_payload
            where id = :session_transaction_id
            """
        ),
        {
            "session_transaction_id": session_transaction_id,
            "raw_payload": _json_dumps(dict(raw_payload)),
        },
    )
    db.commit()


def create_script_run_started(
    db: Session,
    *,
    session_id: int | None,
    trigger_id: int | None,
    script_name: str,
    model_name: str | None,
    runner_job_id: str | None = None,
    runner_payload: Mapping[str, Any] | None = None,
    status: str = "running",
    command: str,
    stdout_log: str = "",
    stderr_log: str = "",
) -> int:
    script_run_table = _table("script_run")
    result = db.execute(
        text(
            f"""
            insert into {script_run_table} (
                session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload, status, command, stdout_log, stderr_log
            )
            values (
                :session_id, :trigger_id, :script_name, :model_name, :runner_job_id, :runner_payload, :status, :command, :stdout_log, :stderr_log
            )
            """
        ),
        {
            "session_id": session_id,
            "trigger_id": trigger_id,
            "script_name": script_name,
            "model_name": model_name,
            "runner_job_id": runner_job_id,
            "runner_payload": _json_dumps(runner_payload) if runner_payload is not None else None,
            "status": status,
            "command": command,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
        },
    )
    db.commit()
    return int(result.lastrowid)


def finish_script_run(
    db: Session,
    script_run_id: int,
    *,
    status: str,
    stdout_log: str,
    stderr_log: str,
    cost_amount: float | None = None,
    cost_currency: str | None = None,
    cost_source: str | None = None,
) -> None:
    script_run_table = _table("script_run")
    cost_updates = ""
    params: dict[str, Any] = {
        "script_run_id": script_run_id,
        "status": status,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
    }
    if _script_run_has_cost_columns(db, script_run_table):
        cost_updates = """
                cost_amount = coalesce(:cost_amount, cost_amount),
                cost_currency = coalesce(:cost_currency, cost_currency),
                cost_source = coalesce(:cost_source, cost_source),
"""
        params.update(
            {
                "cost_amount": cost_amount,
                "cost_currency": cost_currency,
                "cost_source": cost_source,
            }
        )
    db.execute(
        text(
            f"""
            update {script_run_table}
            set status = :status,
                stdout_log = :stdout_log,
                stderr_log = :stderr_log,
{cost_updates}
                finished_at = now()
            where id = :script_run_id
            """
        ),
        params,
    )
    db.commit()


def update_script_run_cost(
    db: Session,
    script_run_id: int,
    *,
    cost_amount: float | None,
    cost_currency: str = "USD",
    cost_source: str | None,
) -> None:
    if cost_amount is None:
        return
    script_run_table = _table("script_run")
    if not _script_run_has_cost_columns(db, script_run_table):
        return
    db.execute(
        text(
            f"""
            update {script_run_table}
            set cost_amount = :cost_amount,
                cost_currency = :cost_currency,
                cost_source = :cost_source
            where id = :script_run_id
            """
        ),
        {
            "script_run_id": script_run_id,
            "cost_amount": cost_amount,
            "cost_currency": cost_currency,
            "cost_source": cost_source,
        },
    )
    db.commit()


def add_script_run_cost(
    db: Session,
    script_run_id: int,
    *,
    cost_amount: float | None,
    cost_currency: str = "USD",
    cost_source: str | None,
) -> None:
    if cost_amount is None or cost_amount <= 0:
        return
    script_run_table = _table("script_run")
    if not _script_run_has_cost_columns(db, script_run_table):
        return
    db.execute(
        text(
            f"""
            update {script_run_table}
            set cost_amount = coalesce(cost_amount, 0) + :cost_amount,
                cost_currency = :cost_currency,
                cost_source = case
                    when cost_source is null or cost_source = '' then :cost_source
                    when :cost_source is null or :cost_source = '' then cost_source
                    when instr(cost_source, :cost_source) > 0 then cost_source
                    else concat(cost_source, '+', :cost_source)
                end
            where id = :script_run_id
            """
        ),
        {
            "script_run_id": script_run_id,
            "cost_amount": cost_amount,
            "cost_currency": cost_currency,
            "cost_source": cost_source,
        },
    )
    db.commit()


def get_current_month_script_run_cost_summary(db: Session) -> dict[str, Any]:
    script_run_table = _table("script_run")
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    if not _script_run_has_cost_columns(db, script_run_table):
        return {
            "month": current_month,
            "currency": "USD",
            "total": 0.0,
            "gemini_total": 0.0,
            "deepseek_total": 0.0,
            "runpod_total": 0.0,
            "locations": [],
        }

    session_table = _table("session")
    trigger_table = _table("trigger_event")
    location_table = settings.location_table_name
    location_id_column = settings.location_id_column
    location_name_column = settings.location_name_column

    result = db.execute(
        text(
            f"""
            with cost_rows as (
                select sr.id,
                       coalesce(sr.cost_amount, 0) as cost_amount,
                       coalesce(sr.cost_currency, 'USD') as cost_currency,
                       lower(coalesce(sr.cost_source, '')) as cost_source,
                       lower(coalesce(sr.model_name, '')) as model_name,
                       lower(coalesce(sr.script_name, '')) as script_name,
                       coalesce(
                           s.location_id,
                           te.location_id,
                           case
                               when json_valid(sr.runner_payload)
                               then cast(nullif(json_unquote(json_extract(sr.runner_payload, '$.location_id')), '') as unsigned)
                               else null
                           end
                       ) as location_id
                from {script_run_table} sr
                left join {session_table} s on s.id = sr.session_id
                left join {trigger_table} te on te.id = sr.trigger_id
                where coalesce(sr.cost_amount, 0) > 0
                  and sr.started_at >= date_format(utc_timestamp(), '%Y-%m-01 00:00:00')
                  and sr.started_at < date_add(date_format(utc_timestamp(), '%Y-%m-01 00:00:00'), interval 1 month)
            )
            select date_format(utc_timestamp(), '%Y-%m') as month,
                   cr.cost_currency,
                   cr.location_id,
                   l.{location_name_column} as location_name,
                   sum(cr.cost_amount) as total,
                   sum(
                       case
                           when cr.cost_source like '%deepseek%' then cr.cost_amount
                           else 0
                       end
                   ) as deepseek_total,
                   sum(
                       case
                           when cr.model_name like 'openrouter-mimo-v2.5%' then cr.cost_amount
                           else 0
                       end
                   ) as mimo_total,
                   sum(
                       case
                           -- A batch's cost_source can be a combined
                           -- "deepseek_estimate+gemini_estimate" when the main
                           -- grouping call ran on deepseek but verification still
                           -- ran on gemini (or vice versa isn't possible today) -
                           -- one script_run's cost_amount can't be split by
                           -- provider after the fact, so a mixed row counts
                           -- toward deepseek_total only, not both, to avoid
                           -- double-counting the same dollar in two buckets.
                           when cr.cost_source like '%deepseek%' then 0
                           when cr.model_name like 'openrouter-mimo-v2.5%' then 0
                           when cr.cost_source like '%gemini%' then cr.cost_amount
                           when cr.cost_source = ''
                             and (cr.model_name like 'gemini%' or cr.script_name in ('grouping', 'carry_confidence', 'grouping_repair'))
                           then cr.cost_amount
                           else 0
                       end
                   ) as gemini_total,
                   sum(
                       case
                           when cr.cost_source like '%runpod%'
                             or cr.model_name = 'runpod_runner'
                           then cr.cost_amount
                           else 0
                       end
                   ) as runpod_total
            from cost_rows cr
            left join {location_table} l on l.{location_id_column} = cr.location_id
            group by cr.cost_currency, cr.location_id, l.{location_name_column}
            order by cr.location_id is null, cr.location_id asc
            """
        )
    )
    rows = _fetch_all_dicts(result)
    month = rows[0]["month"] if rows else current_month
    currency = rows[0]["cost_currency"] if rows else "USD"
    locations: list[dict[str, Any]] = []
    total = 0.0
    gemini_total = 0.0
    deepseek_total = 0.0
    mimo_total = 0.0
    runpod_total = 0.0
    for row in rows:
        row_total = float(row.get("total") or 0)
        row_gemini_total = float(row.get("gemini_total") or 0)
        row_deepseek_total = float(row.get("deepseek_total") or 0)
        row_mimo_total = float(row.get("mimo_total") or 0)
        row_runpod_total = float(row.get("runpod_total") or 0)
        total += row_total
        gemini_total += row_gemini_total
        deepseek_total += row_deepseek_total
        mimo_total += row_mimo_total
        runpod_total += row_runpod_total
        if row.get("location_id") is None:
            continue
        locations.append(
            {
                "location_id": int(row["location_id"]),
                "location_name": row.get("location_name") or f"Location {row['location_id']}",
                "total": row_total,
                "gemini_total": row_gemini_total,
                "deepseek_total": row_deepseek_total,
                "mimo_total": row_mimo_total,
                "runpod_total": row_runpod_total,
            }
        )

    return {
        "month": str(month or ""),
        "currency": str(currency or "USD"),
        "total": total,
        "gemini_total": gemini_total,
        "deepseek_total": deepseek_total,
        "mimo_total": mimo_total,
        "runpod_total": runpod_total,
        "locations": locations,
    }


def revise_script_run(
    db: Session,
    script_run_id: int,
    *,
    status: str,
    stdout_log: str,
    stderr_log: str,
) -> None:
    script_run_table = _table("script_run")
    db.execute(
        text(
            f"""
            update {script_run_table}
            set status = :status,
                stdout_log = :stdout_log,
                stderr_log = :stderr_log,
                finished_at = now()
            where id = :script_run_id
            """
        ),
        {
            "script_run_id": script_run_id,
            "status": status,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
        },
    )
    db.commit()


def assign_script_run_runner_job(
    db: Session,
    script_run_id: int,
    *,
    runner_job_id: str,
    runner_payload: Mapping[str, Any] | None = None,
) -> None:
    script_run_table = _table("script_run")
    db.execute(
        text(
            f"""
            update {script_run_table}
            set runner_job_id = :runner_job_id,
                runner_payload = coalesce(:runner_payload, runner_payload)
            where id = :script_run_id
            """
        ),
        {
            "script_run_id": script_run_id,
            "runner_job_id": runner_job_id,
            "runner_payload": _json_dumps(runner_payload) if runner_payload is not None else None,
        },
    )
    db.commit()


def get_script_run(db: Session, script_run_id: int) -> dict[str, Any]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where id = :script_run_id
            limit 1
            """
        ),
        {"script_run_id": script_run_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("runner_payload"), str):
        try:
            row["runner_payload"] = json.loads(row["runner_payload"])
        except json.JSONDecodeError:
            pass
    return row


def get_script_run_by_runner_job_id(db: Session, runner_job_id: str) -> dict[str, Any]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where runner_job_id = :runner_job_id
            limit 1
            """
        ),
        {"runner_job_id": runner_job_id},
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("runner_payload"), str):
        try:
            row["runner_payload"] = json.loads(row["runner_payload"])
        except json.JSONDecodeError:
            pass
    return row


def get_latest_script_run_for_session(
    db: Session,
    session_id: int,
    *,
    script_name: str | None = None,
) -> dict[str, Any]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    filters = ["session_id = :session_id"]
    params: dict[str, Any] = {"session_id": session_id}
    if script_name:
        filters.append("script_name = :script_name")
        params["script_name"] = script_name
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where {" and ".join(filters)}
            order by id desc
            limit 1
            """
        ),
        params,
    )
    row = _fetch_one_dict(result)
    if isinstance(row.get("runner_payload"), str):
        try:
            row["runner_payload"] = json.loads(row["runner_payload"])
        except json.JSONDecodeError:
            pass
    return row


def get_latest_script_run_for_trigger(
    db: Session,
    trigger_id: int,
    *,
    script_name: str | None = None,
) -> dict[str, Any]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    filters = ["trigger_id = :trigger_id"]
    params: dict[str, Any] = {"trigger_id": trigger_id}
    if script_name:
        filters.append("script_name = :script_name")
        params["script_name"] = script_name
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where {" and ".join(filters)}
            order by id desc
            limit 1
            """
        ),
        params,
    )
    row = result.mappings().first()
    if row is None:
        return {}
    payload = dict(row)
    if isinstance(payload.get("runner_payload"), str):
        try:
            payload["runner_payload"] = json.loads(payload["runner_payload"])
        except json.JSONDecodeError:
            pass
    return payload


def get_latest_script_run_for_video_asset(db: Session, video_asset_id: int) -> dict[str, Any]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where cast(json_unquote(json_extract(runner_payload, '$.video_asset_id')) as unsigned) = :video_asset_id
            order by id desc
            limit 1
            """
        ),
        {"video_asset_id": video_asset_id},
    )
    row = result.mappings().first()
    if row is None:
        return {}
    payload = dict(row)
    if isinstance(payload.get("runner_payload"), str):
        try:
            payload["runner_payload"] = json.loads(payload["runner_payload"])
        except json.JSONDecodeError:
            pass
    return payload


def has_active_remote_analysis_script_run(
    db: Session,
    *,
    script_names: list[str] | None = None,
) -> bool:
    script_run_table = _table("script_run")
    normalized_names = [
        str(script_name).strip().lower()
        for script_name in (script_names or [])
        if str(script_name or "").strip()
    ]
    params: dict[str, Any] = {}
    name_clause = "script_name in ('entry', 'kiosk', 'kiosk_match', 'grouping')"
    if normalized_names:
        placeholders: list[str] = []
        for index, script_name in enumerate(normalized_names):
            key = f"script_name_{index}"
            placeholders.append(f":{key}")
            params[key] = script_name
        name_clause = f"lower(script_name) in ({', '.join(placeholders)})"
    result = db.execute(
        text(
            f"""
            select 1
            from {script_run_table}
            where {name_clause}
              and status = 'running'
              and runner_job_id is not null
            limit 1
            """
        ),
        params,
    )
    return result.first() is not None


def list_running_remote_analysis_script_runs(db: Session) -> list[dict[str, Any]]:
    script_run_table = _table("script_run")
    cost_select = _script_run_cost_select(db, script_run_table)
    result = db.execute(
        text(
            f"""
            select id, session_id, trigger_id, script_name, model_name, runner_job_id, runner_payload,
                   status, command, stdout_log, stderr_log, {cost_select}, started_at, finished_at
            from {script_run_table}
            where script_name in ('entry', 'kiosk', 'kiosk_match', 'grouping')
              and status = 'running'
              and runner_job_id is not null
            order by id asc
            """
        )
    )
    rows = _fetch_all_dicts(result)
    for row in rows:
        if isinstance(row.get("runner_payload"), str):
            try:
                row["runner_payload"] = json.loads(row["runner_payload"])
            except json.JSONDecodeError:
                pass
    return rows


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
) -> int:
    script_run_id = create_script_run_started(
        db,
        session_id=session_id,
        trigger_id=trigger_id,
        script_name=script_name,
        model_name=model_name,
        status="running",
        command=command,
    )
    finish_script_run(
        db,
        script_run_id,
        status=status,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    return script_run_id


def finalize_session_result(
    db: Session,
    *,
    session_id: int,
    kiosk_total_items: int,
    actual_items_brought: int | None = None,
    tolerance: int = 1,
    extra_result_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session_table = _table("session")
    transaction_total_items = get_transaction_total_items(db, session_id)
    actual_items = kiosk_total_items if actual_items_brought is None else actual_items_brought
    comparison_items = actual_items if actual_items_brought is not None else kiosk_total_items
    difference = comparison_items - transaction_total_items

    if actual_items_brought is not None:
        if abs(difference) <= max(0, int(tolerance)):
            status = "not_detected"
        else:
            status = "detected"
    elif comparison_items == 0 or comparison_items < transaction_total_items:
        status = "need_review"
    elif abs(difference) <= max(0, int(tolerance)):
        status = "not_detected"
    else:
        status = "detected"

    result_summary = {
        "kiosk_total_items": kiosk_total_items,
        "transaction_total_items": transaction_total_items,
        "actual_items_brought": actual_items,
        "difference": difference,
        "comparison_items": comparison_items,
        "tolerance": max(0, int(tolerance)),
        "decision": status,
        "manual_review_completed": actual_items_brought is not None,
    }
    if extra_result_summary:
        result_summary.update(dict(extra_result_summary))

    db.execute(
        text(
            f"""
            update {session_table}
            set total_item_brought = :kiosk_total_items,
                actual_items_brought = :actual_items_brought,
                transaction_total_items = :transaction_total_items,
                status = :status,
                result_summary = :result_summary,
                issue_reason = null
            where id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "kiosk_total_items": kiosk_total_items,
            "actual_items_brought": actual_items,
            "transaction_total_items": transaction_total_items,
            "status": status,
            "result_summary": _json_dumps(result_summary),
        },
    )
    db.commit()

    if status == "detected":
        try:
            session = get_session(db, session_id)
            create_thief_alert_if_missing(
                db,
                location_id=int(session["location_id"]),
                method="TDS System",
                detail=f"Session {session_id}",
            )
        except Exception:
            # Alert creation should not block session result finalization.
            pass

    return {
        "session_id": session_id,
        "status": status,
        "kiosk_total_items": kiosk_total_items,
        "transaction_total_items": transaction_total_items,
        "actual_items_brought": actual_items,
        "result_summary": result_summary,
    }
