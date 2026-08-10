import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def _fetch_one_dict(result) -> dict[str, Any]:
    row = result.mappings().first()
    if row is None:
        raise ValueError("Expected a row but query returned nothing.")
    return dict(row)


def _fetch_all_dicts(result) -> list[dict[str, Any]]:
    return [dict(row) for row in result.mappings().all()]


def _merge_metadata(base: Any, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(base) if isinstance(base, Mapping) else {}
    if extra:
        payload.update(dict(extra))
    return payload


def create_active_gallery_record(
    db: Session,
    *,
    location_id: int,
    session_id: int | None,
    session_customer_id: int,
    person_id: int | None,
    image_url: str | None = None,
    image_kind: str = "reid_view",
    embedding_osnet: list[float] | None = None,
    embedding_fashion: list[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            insert into tds_active_gallery (
                location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                embedding_osnet, embedding_fashion, metadata
            )
            values (
                :location_id, :session_id, :session_customer_id, :person_id, :image_url, :image_kind,
                cast(:embedding_osnet as jsonb), cast(:embedding_fashion as jsonb), cast(:metadata as jsonb)
            )
            returning id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                      embedding_osnet, embedding_fashion, metadata, created_at, updated_at
            """
        ),
        {
            "location_id": location_id,
            "session_id": session_id,
            "session_customer_id": session_customer_id,
            "person_id": person_id,
            "image_url": image_url,
            "image_kind": image_kind,
            "embedding_osnet": json.dumps(embedding_osnet) if embedding_osnet is not None else None,
            "embedding_fashion": json.dumps(embedding_fashion) if embedding_fashion is not None else None,
            "metadata": json.dumps(dict(metadata)) if metadata is not None else None,
        },
    )
    db.commit()
    return _fetch_one_dict(result)


def create_customer_gallery_record(
    db: Session,
    *,
    location_id: int,
    session_id: int,
    person_id: int,
    session_customer_id: int | None = None,
    image_url: str | None = None,
    image_public_url: str | None = None,
    image_kind: str = "reid_view",
    embedding_osnet: list[float] | None = None,
    embedding_fashion: list[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            insert into tds_customer_gallery (
                location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                embedding_osnet, embedding_fashion, metadata
            )
            values (
                :location_id, :session_id, :session_customer_id, :person_id, :image_url, :image_public_url, :image_kind,
                cast(:embedding_osnet as jsonb), cast(:embedding_fashion as jsonb), cast(:metadata as jsonb)
            )
            returning id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                      embedding_osnet, embedding_fashion, metadata, created_at
            """
        ),
        {
            "location_id": location_id,
            "session_id": session_id,
            "session_customer_id": session_customer_id,
            "person_id": person_id,
            "image_url": image_url,
            "image_public_url": image_public_url,
            "image_kind": image_kind,
            "embedding_osnet": json.dumps(embedding_osnet) if embedding_osnet is not None else None,
            "embedding_fashion": json.dumps(embedding_fashion) if embedding_fashion is not None else None,
            "metadata": json.dumps(dict(metadata)) if metadata is not None else None,
        },
    )
    db.commit()
    return _fetch_one_dict(result)


def get_customer_gallery_record(db: Session, gallery_id: int) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from tds_customer_gallery
            where id = :gallery_id
            """
        ),
        {"gallery_id": gallery_id},
    )
    return _fetch_one_dict(result)


def get_active_gallery(
    db: Session,
    *,
    location_id: int,
    session_customer_id: int,
    active_gallery_id: int,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at, updated_at
            from tds_active_gallery
            where location_id = :location_id and session_customer_id = :session_customer_id and id = :active_gallery_id
            """
        ),
        {"location_id": location_id, "session_customer_id": session_customer_id, "active_gallery_id": active_gallery_id},
    )
    return _fetch_one_dict(result)


def get_active_gallery_record(db: Session, gallery_id: int) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at, updated_at
            from tds_active_gallery
            where id = :gallery_id
            """
        ),
        {"gallery_id": gallery_id},
    )
    return _fetch_one_dict(result)


def list_customer_gallery_records(
    db: Session,
    *,
    session_id: int,
) -> list[dict[str, Any]]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from tds_customer_gallery
            where session_id = :session_id
            order by id asc
            """
        ),
        {"session_id": session_id},
    )
    return _fetch_all_dicts(result)


def list_all_customer_gallery_records(
    db: Session,
    *,
    location_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if location_id is None:
        result = db.execute(
            text(
                """
                select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                       embedding_osnet, embedding_fashion, metadata, created_at
                from tds_customer_gallery
                order by created_at desc, id desc
                limit :limit
                """
            ),
            {"limit": limit},
        )
        return _fetch_all_dicts(result)

    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from tds_customer_gallery
            where location_id = :location_id
            order by created_at desc, id desc
            limit :limit
            """
        ),
        {"location_id": location_id, "limit": limit},
    )
    return _fetch_all_dicts(result)


def list_customer_gallery_records_for_session_customer(
    db: Session,
    *,
    session_customer_id: int,
) -> list[dict[str, Any]]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from tds_customer_gallery
            where session_customer_id = :session_customer_id
            order by id asc
            """
        ),
        {"session_customer_id": session_customer_id},
    )
    return _fetch_all_dicts(result)


def list_customer_gallery_records_by_ids(
    db: Session,
    *,
    gallery_ids: list[int],
) -> list[dict[str, Any]]:
    if not gallery_ids:
        return []
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_public_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from tds_customer_gallery
            where id = any(:gallery_ids)
            order by id asc
            """
        ),
        {"gallery_ids": gallery_ids},
    )
    return _fetch_all_dicts(result)


def delete_customer_gallery_records_for_session_customer(
    db: Session,
    *,
    session_customer_id: int,
) -> None:
    db.execute(
        text(
            """
            delete from tds_customer_gallery
            where session_customer_id = :session_customer_id
            """
        ),
        {"session_customer_id": session_customer_id},
    )
    db.commit()


def list_active_gallery_records(
    db: Session,
    *,
    location_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if location_id is None:
        result = db.execute(
            text(
                """
                select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                       embedding_osnet, embedding_fashion, metadata, created_at, updated_at
                from tds_active_gallery
                order by updated_at desc, id desc
                limit :limit
                """
            ),
            {"limit": limit},
        )
        return _fetch_all_dicts(result)

    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at, updated_at
            from tds_active_gallery
            where location_id = :location_id
            order by updated_at desc, id desc
            limit :limit
            """
        ),
        {"location_id": location_id, "limit": limit},
    )
    return _fetch_all_dicts(result)


def create_history_gallery_record(
    db: Session,
    *,
    active_gallery_id: int | None = None,
    location_id: int,
    session_id: int | None,
    session_customer_id: int | None,
    person_id: int | None,
    image_url: str | None = None,
    image_kind: str = "reid_view",
    embedding_osnet: list[float] | None = None,
    embedding_fashion: list[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    archived_reason: str | None = None,
    created_at: Any | None = None,
    updated_at: Any | None = None,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            insert into tds_history_gallery (
                active_gallery_id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                embedding_osnet, embedding_fashion, metadata, archived_reason, created_at, updated_at
            )
            values (
                :active_gallery_id, :location_id, :session_id, :session_customer_id, :person_id, :image_url, :image_kind,
                cast(:embedding_osnet as jsonb), cast(:embedding_fashion as jsonb), cast(:metadata as jsonb), :archived_reason,
                :created_at, :updated_at
            )
            returning id, active_gallery_id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                      embedding_osnet, embedding_fashion, metadata, archived_reason, archived_at, created_at, updated_at
            """
        ),
        {
            "active_gallery_id": active_gallery_id,
            "location_id": location_id,
            "session_id": session_id,
            "session_customer_id": session_customer_id,
            "person_id": person_id,
            "image_url": image_url,
            "image_kind": image_kind,
            "embedding_osnet": json.dumps(embedding_osnet) if embedding_osnet is not None else None,
            "embedding_fashion": json.dumps(embedding_fashion) if embedding_fashion is not None else None,
            "metadata": json.dumps(dict(metadata)) if metadata is not None else None,
            "archived_reason": archived_reason,
            "created_at": created_at,
            "updated_at": updated_at,
        },
    )
    db.commit()
    return _fetch_one_dict(result)


def archive_active_gallery_by_aliases(
    db: Session,
    *,
    location_id: int,
    session_customer_ids: list[int] | None = None,
    person_ids: list[int] | None = None,
    archived_reason: str = "customer_exited",
    metadata_extra: Mapping[str, Any] | None = None,
) -> int:
    normalized_session_customer_ids = sorted(
        {int(value) for value in (session_customer_ids or []) if value is not None}
    )
    normalized_person_ids = sorted(
        {int(value) for value in (person_ids or []) if value is not None}
    )
    if not normalized_session_customer_ids and not normalized_person_ids:
        return 0

    clauses = []
    params: dict[str, Any] = {"location_id": location_id}
    if normalized_session_customer_ids:
        clauses.append("session_customer_id = any(:session_customer_ids)")
        params["session_customer_ids"] = normalized_session_customer_ids
    if normalized_person_ids:
        clauses.append("person_id = any(:person_ids)")
        params["person_ids"] = normalized_person_ids

    rows = _fetch_all_dicts(
        db.execute(
            text(
                f"""
                select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                       embedding_osnet, embedding_fashion, metadata, created_at, updated_at
                from tds_active_gallery
                where location_id = :location_id
                  and ({' or '.join(clauses)})
                """
            ),
            params,
        )
    )
    if not rows:
        return 0

    for row in rows:
        create_history_gallery_record(
            db,
            active_gallery_id=int(row["id"]) if row.get("id") is not None else None,
            location_id=int(row["location_id"]),
            session_id=int(row["session_id"]) if row.get("session_id") is not None else None,
            session_customer_id=int(row["session_customer_id"]) if row.get("session_customer_id") is not None else None,
            person_id=int(row["person_id"]) if row.get("person_id") is not None else None,
            image_url=row.get("image_url"),
            image_kind=str(row.get("image_kind") or "reid_view"),
            embedding_osnet=row.get("embedding_osnet"),
            embedding_fashion=row.get("embedding_fashion"),
            metadata=_merge_metadata(
                row.get("metadata"),
                {
                    "archived_reason": archived_reason,
                    **(dict(metadata_extra) if metadata_extra is not None else {}),
                },
            ),
            archived_reason=archived_reason,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    db.execute(
        text(
            f"""
            delete from tds_active_gallery
            where location_id = :location_id
              and ({' or '.join(clauses)})
            """
        ),
        params,
    )
    db.commit()
    return len(rows)


def delete_active_gallery(
    db: Session,
    *,
    location_id: int,
    session_customer_id: int,
) -> None:
    archive_active_gallery_by_aliases(
        db,
        location_id=location_id,
        session_customer_ids=[session_customer_id],
        archived_reason="manual_delete",
    )


def delete_active_gallery_by_aliases(
    db: Session,
    *,
    location_id: int,
    session_customer_ids: list[int] | None = None,
    person_ids: list[int] | None = None,
) -> None:
    archive_active_gallery_by_aliases(
        db,
        location_id=location_id,
        session_customer_ids=session_customer_ids,
        person_ids=person_ids,
        archived_reason="delete_by_aliases",
    )
