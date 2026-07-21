-- PostgreSQL schema for theft-detection gallery and embeddings
--
-- Target:
--   a new PostgreSQL database dedicated to theft-detection vision memory
--
-- Recommended:
--   CREATE DATABASE theft_detection_gallery;
--   CREATE USER theft_api_pg WITH PASSWORD 'change_me';
--   GRANT CONNECT ON DATABASE theft_detection_gallery TO theft_api_pg;
--
-- After connecting to that database:
--   GRANT USAGE, CREATE ON SCHEMA public TO theft_api_pg;
--   \i postgres_schema.sql
--
-- Notes:
-- - This schema intentionally does not use foreign keys to MySQL tables.
-- - Join keys are carried by value: session_id, session_customer_id, person_id.
-- - Embeddings are JSONB for now. If you later enable pgvector, you can add vector columns.

create extension if not exists "pgcrypto";

create table if not exists tds_customer_gallery (
    id bigserial primary key,
    location_id bigint not null,
    session_id bigint not null,
    session_customer_id bigint,
    person_id integer not null,
    image_url text,
    image_kind varchar(50) not null default 'reid_view',
    embedding_osnet jsonb,
    embedding_fashion jsonb,
    metadata jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_customer_gallery_session_person
    on tds_customer_gallery(session_id, person_id);

create index if not exists idx_customer_gallery_location_person
    on tds_customer_gallery(location_id, person_id);

create index if not exists idx_customer_gallery_session_customer
    on tds_customer_gallery(session_customer_id);

create index if not exists idx_customer_gallery_image_kind
    on tds_customer_gallery(image_kind);

create table if not exists tds_active_gallery (
    id bigserial primary key,
    location_id bigint not null,
    session_id bigint,
    session_customer_id bigint not null,
    person_id integer,
    image_url text,
    image_kind varchar(50) not null default 'reid_view',
    embedding_osnet jsonb,
    embedding_fashion jsonb,
    metadata jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_active_gallery_location_id
    on tds_active_gallery(location_id);

create index if not exists idx_active_gallery_session_id
    on tds_active_gallery(session_id);

create index if not exists idx_active_gallery_session_customer_id
    on tds_active_gallery(session_customer_id);

create index if not exists idx_active_gallery_person_id
    on tds_active_gallery(person_id);

create index if not exists idx_active_gallery_image_kind
    on tds_active_gallery(image_kind);

comment on column tds_customer_gallery.location_id is
    'Store/location partition key copied from MySQL so embeddings can be isolated by location.';

comment on column tds_customer_gallery.session_customer_id is
    'Value-copied MySQL session_customer.id for the exact detected person row that produced this gallery entry.';

comment on column tds_active_gallery.location_id is
    'Store/location partition key for the active gallery currently used in comparisons.';

comment on column tds_active_gallery.session_customer_id is
    'Value-copied MySQL session_customer.id for the active customer represented by this runtime-state row.';

comment on column tds_active_gallery.embedding_osnet is
    'OSNet embedding for one active ReID view that should be compared against the next video.';

comment on column tds_active_gallery.embedding_fashion is
    'Fashion embedding for one active ReID view that should be compared against the next video.';
