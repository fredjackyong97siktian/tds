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
    on customer_gallery(session_id, person_id);

create index if not exists idx_customer_gallery_location_person
    on customer_gallery(location_id, person_id);

create index if not exists idx_customer_gallery_session_customer
    on customer_gallery(session_customer_id);

create index if not exists idx_customer_gallery_image_kind
    on customer_gallery(image_kind);

create table if not exists tds_active_gallery (
    id bigserial primary key,
    location_id bigint not null,
    session_id bigint,
    session_customer_id bigint not null,
    person_id integer,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (location_id, session_customer_id)
);

create index if not exists idx_active_gallery_location_id
    on active_gallery(location_id);

create index if not exists idx_active_gallery_session_id
    on active_gallery(session_id);

create index if not exists idx_active_gallery_session_customer_id
    on active_gallery(session_customer_id);

create index if not exists idx_active_gallery_payload_gin
    on active_gallery using gin (state_payload);

comment on column customer_gallery.location_id is
    'Store/location partition key copied from MySQL so embeddings can be isolated by location.';

comment on column customer_gallery.session_customer_id is
    'Value-copied MySQL session_customer.id for the exact detected person row that produced this gallery entry.';

comment on column active_gallery.location_id is
    'Store/location partition key for the active gallery currently used in comparisons.';

comment on column active_gallery.session_customer_id is
    'Value-copied MySQL session_customer.id for the active customer represented by this runtime-state row.';

comment on column active_gallery.state_kind is
    'active_gallery keeps only currently active customer_gallery references for one customer at one location.';

comment on column active_gallery.state_payload is
    'Recommended shape: {"customer_gallery_ids":[5001,5002],"primary_gallery_entry_id":5001,"is_active":true} so the app can fetch embeddings from customer_gallery by id for this active customer.';

