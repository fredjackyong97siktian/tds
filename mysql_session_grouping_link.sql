-- Add direct grouping linkage to sessions.
-- One grouping batch/group can create multiple sessions; sessions default to NULL
-- until they are created by Layer 0 Deep Analysis.

ALTER TABLE sesamedb.tds_session
    ADD COLUMN grouping_batch_id BIGINT NULL AFTER issue_reason,
    ADD COLUMN grouping_group_key VARCHAR(64) NULL AFTER grouping_batch_id,
    ADD INDEX idx_session_grouping (grouping_batch_id, grouping_group_key);
