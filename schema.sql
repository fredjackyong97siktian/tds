-- PostgreSQL schema for theft detection orchestration
-- Assumption:
-- - You will keep this inside your existing application database.
-- - "embedding_*" fields are stored as JSONB arrays for now.
--   If you later enable pgvector, you can replace them with VECTOR columns.

create extension if not exists "pgcrypto";

create table if not exists whitelist_entry (
    id bigserial primary key,
    method varchar(50) not null,
    entry_id varchar(255) not null,
    status varchar(30) not null default 'active',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists idx_whitelist_entry_method_entry_id
    on whitelist_entry(method, entry_id);

create table if not exists cctv (
    id bigserial primary key,
    location_id bigint not null,
    section varchar(50) not null,
    stream_name varchar(255),
    recorder_channel varchar(100),
    delayed_seconds integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists idx_cctv_location_section
    on cctv(location_id, section);

create table if not exists trigger_event (
    id bigserial primary key,
    location_id bigint not null,
    aqara_event_id varchar(255),
    trigger_source varchar(100) not null default 'aqara',
    trigger_time timestamptz not null,
    status varchar(30) not null default 'pending',
    whitelist_hit boolean not null default false,
    raw_payload jsonb,
    issue_reason text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint chk_trigger_status
        check (status in ('pending', 'video_pending', 'processing', 'done', 'issue', 'whitelisted'))
);

create index if not exists idx_trigger_event_location_time
    on trigger_event(location_id, trigger_time desc);

create table if not exists "session" (
    id bigserial primary key,
    trigger_id bigint not null references trigger_event(id) on delete restrict,
    location_id bigint not null,
    entry_video_url text,
    kiosk_video_url text,
    exit_video_url text,
    start_time timestamptz,
    kiosk_start_time timestamptz,
    kiosk_end_time timestamptz,
    end_time timestamptz,
    total_item_brought integer not null default 0,
    actual_items_brought integer not null default 0,
    transaction_total_items integer not null default 0,
    total_customer integer not null default 0,
    status varchar(30) not null default 'pending',
    result_summary jsonb,
    issue_reason text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint chk_session_status
        check (status in ('pending', 'processing_entry', 'processing_kiosk', 'detected', 'not_detected', 'need_review', 'issue', 'whitelisted', 'closed'))
);

create index if not exists idx_session_trigger_id
    on "session"(trigger_id);

create index if not exists idx_session_location_created
    on "session"(location_id, created_at desc);

create table if not exists session_customer (
    id bigserial primary key,
    session_id bigint not null references "session"(id) on delete cascade,
    person_id integer not null,
    enter_time timestamptz,
    kiosk_start_time timestamptz,
    leave_time timestamptz,
    match_status varchar(30) not null default 'tracked',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (session_id, person_id)
);

create table if not exists customer_gallery (
    id bigserial primary key,
    session_id bigint not null references "session"(id) on delete cascade,
    session_customer_id bigint references session_customer(id) on delete cascade,
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

create table if not exists session_transaction (
    id bigserial primary key,
    session_id bigint not null references "session"(id) on delete cascade,
    receipt_number varchar(255) not null,
    transaction_time timestamptz,
    total_items integer not null default 0,
    total_amount numeric(12, 2),
    raw_payload jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_session_transaction_session_id
    on session_transaction(session_id);

create table if not exists video_asset (
    id bigserial primary key,
    session_id bigint references "session"(id) on delete cascade,
    trigger_id bigint references trigger_event(id) on delete cascade,
    section varchar(30) not null,
    video_url text not null,
    file_path text,
    captured_start_time timestamptz,
    captured_end_time timestamptz,
    status varchar(30) not null default 'ready',
    metadata jsonb,
    created_at timestamptz not null default now(),
    constraint chk_video_asset_section
        check (section in ('entry', 'kiosk', 'exit')),
    constraint chk_video_asset_status
        check (status in ('pending', 'ready', 'issue'))
);

create table if not exists script_run (
    id bigserial primary key,
    session_id bigint references "session"(id) on delete cascade,
    trigger_id bigint references trigger_event(id) on delete cascade,
    script_name varchar(50) not null,
    status varchar(30) not null default 'pending',
    command text not null,
    stdout_log text,
    stderr_log text,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    constraint chk_script_run_name
        check (script_name in ('retrieve_video', 'entry', 'kiosk')),
    constraint chk_script_run_status
        check (status in ('pending', 'running', 'success', 'failed'))
);

create or replace function set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_whitelist_entry_updated_at on whitelist_entry;
create trigger trg_whitelist_entry_updated_at
before update on whitelist_entry
for each row execute function set_updated_at();

drop trigger if exists trg_cctv_updated_at on cctv;
create trigger trg_cctv_updated_at
before update on cctv
for each row execute function set_updated_at();

drop trigger if exists trg_trigger_event_updated_at on trigger_event;
create trigger trg_trigger_event_updated_at
before update on trigger_event
for each row execute function set_updated_at();

drop trigger if exists trg_session_updated_at on "session";
create trigger trg_session_updated_at
before update on "session"
for each row execute function set_updated_at();

drop trigger if exists trg_session_customer_updated_at on session_customer;
create trigger trg_session_customer_updated_at
before update on session_customer
for each row execute function set_updated_at();
