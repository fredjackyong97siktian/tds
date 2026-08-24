-- Add date/time-period scope to active and history galleries.
-- Apply this to the PostgreSQL gallery database used by tds_active_gallery.

alter table tds_active_gallery
    add column if not exists gallery_date date,
    add column if not exists period_code varchar(30);

create index if not exists idx_active_gallery_location_period_date
    on tds_active_gallery(location_id, gallery_date, period_code);

alter table tds_history_gallery
    add column if not exists gallery_date date,
    add column if not exists period_code varchar(30);

create index if not exists idx_history_gallery_location_period_date
    on tds_history_gallery(location_id, gallery_date, period_code);

comment on column tds_active_gallery.gallery_date is
    'Local trigger date used to scope active matching so only same-date embeddings are loaded.';

comment on column tds_active_gallery.period_code is
    'Configured time-period code used to scope active matching inside the same store and date.';

comment on column tds_history_gallery.gallery_date is
    'Copied local trigger date from tds_active_gallery for history/audit matching scope.';

comment on column tds_history_gallery.period_code is
    'Copied configured time-period code from tds_active_gallery for history/audit matching scope.';
