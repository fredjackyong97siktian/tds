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


def upsert_active_gallery(
    db: Session,
    *,
    location_id: int,
    session_id: int | None,
    session_customer_id: int,
    person_id: int | None,
    state_kind: str,
    state_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    db.execute(
        text(
            """
            insert into active_gallery (
                location_id, session_id, session_customer_id, person_id, state_kind, state_payload, metadata
            )
            values (
                :location_id, :session_id, :session_customer_id, :person_id, :state_kind, cast(:state_payload as jsonb), cast(:metadata as jsonb)
            )
            on conflict (location_id, state_kind, session_customer_id) do update
            set state_payload = excluded.state_payload,
                session_id = excluded.session_id,
                person_id = excluded.person_id,
                metadata = excluded.metadata,
                updated_at = now()
            """
        ),
        {
            "location_id": location_id,
            "session_id": session_id,
            "session_customer_id": session_customer_id,
            "person_id": person_id,
            "state_kind": state_kind,
            "state_payload": json.dumps(dict(state_payload)),
            "metadata": json.dumps(dict(metadata)) if metadata is not None else None,
        },
    )
    db.commit()


def create_customer_gallery_record(
    db: Session,
    *,
    location_id: int,
    session_id: int,
    person_id: int,
    session_customer_id: int | None = None,
    image_url: str | None = None,
    image_kind: str = "reid_view",
    embedding_osnet: list[float] | None = None,
    embedding_fashion: list[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            insert into customer_gallery (
                location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                embedding_osnet, embedding_fashion, metadata
            )
            values (
                :location_id, :session_id, :session_customer_id, :person_id, :image_url, :image_kind,
                cast(:embedding_osnet as jsonb), cast(:embedding_fashion as jsonb), cast(:metadata as jsonb)
            )
            returning id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                      embedding_osnet, embedding_fashion, metadata, created_at
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


def get_customer_gallery_record(db: Session, gallery_id: int) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from customer_gallery
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
    state_kind: str,
) -> dict[str, Any]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, state_kind, state_payload, metadata, created_at, updated_at
            from active_gallery
            where location_id = :location_id and session_customer_id = :session_customer_id and state_kind = :state_kind
            """
        ),
        {"location_id": location_id, "session_customer_id": session_customer_id, "state_kind": state_kind},
    )
    return _fetch_one_dict(result)


def upsert_and_get_active_gallery(
    db: Session,
    *,
    location_id: int,
    session_id: int | None,
    session_customer_id: int,
    person_id: int | None,
    state_kind: str,
    state_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    upsert_active_gallery(
        db,
        location_id=location_id,
        session_id=session_id,
        session_customer_id=session_customer_id,
        person_id=person_id,
        state_kind=state_kind,
        state_payload=state_payload,
        metadata=metadata,
    )
    return get_active_gallery(
        db,
        location_id=location_id,
        session_customer_id=session_customer_id,
        state_kind=state_kind,
    )


def list_customer_gallery_records(
    db: Session,
    *,
    session_id: int,
) -> list[dict[str, Any]]:
    result = db.execute(
        text(
            """
            select id, location_id, session_id, session_customer_id, person_id, image_url, image_kind,
                   embedding_osnet, embedding_fashion, metadata, created_at
            from customer_gallery
            where session_id = :session_id
            order by id asc
            """
        ),
        {"session_id": session_id},
    )
    return _fetch_all_dicts(result)


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
                select id, location_id, session_id, session_customer_id, person_id, state_kind,
                       state_payload, metadata, created_at, updated_at
                from active_gallery
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
            select id, location_id, session_id, session_customer_id, person_id, state_kind,
                   state_payload, metadata, created_at, updated_at
            from active_gallery
            where location_id = :location_id
            order by updated_at desc, id desc
            limit :limit
            """
        ),
        {"location_id": location_id, "limit": limit},
    )
    return _fetch_all_dicts(result)
