-- Run once before deploying Expand. Existing memberships are preserved.
-- Build the replacement index before dropping the old batch-prefix index.
ALTER TABLE sesamedb.tds_filter_grouping_item
    ADD COLUMN membership_key VARCHAR(100)
        GENERATED ALWAYS AS (COALESCE(group_key, '')) STORED,
    ADD UNIQUE KEY uq_filter_grouping_item_membership (batch_id, trigger_id, role, membership_key),
    DROP INDEX uq_filter_grouping_item_batch_trigger;
