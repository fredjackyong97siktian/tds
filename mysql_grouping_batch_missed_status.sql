-- Adds a 'missed' status for tds_filter_grouping_batch, used when a
-- scheduled window's grace period expires before an automatic batch was
-- ever created (e.g. the worker was down, or dispatch was silently
-- blocked) - a real batch row is now created for it instead of only
-- logging a warning. Run this once against an existing database.

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
        'missed'
    ));
