-- Adds 'missed' and 'avoid' statuses for tds_filter_grouping_batch:
-- 'missed' - a scheduled window's grace period expired before an automatic
--            batch was ever created (e.g. the worker was down, or dispatch
--            was silently blocked).
-- 'avoid'  - the window's time period simply isn't active/selected for that
--            location, so it was never meant to be processed at all.
-- A real batch row is now created for both cases instead of only logging a
-- warning (or, for 'avoid', not even that). Run this once against an
-- existing database.

ALTER TABLE sesamedb.tds_filter_grouping_batch
    DROP CHECK chk_filter_grouping_batch_status;

ALTER TABLE sesamedb.tds_filter_grouping_batch
    ADD CONSTRAINT chk_filter_grouping_batch_status
    CHECK (status IN (
        'pending',
        'dispatching',
        'running',
        'success',
        'failed',
        'issue',
        'missed',
        'avoid'
    ));
