-- Add one direct grouping linkage to sessions.
-- grouping_id stores tds_filter_grouping_batch.id. Default is NULL.

SET @has_grouping_id := (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'sesamedb'
      AND table_name = 'tds_session'
      AND column_name = 'grouping_id'
);

SET @has_grouping_batch_id := (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'sesamedb'
      AND table_name = 'tds_session'
      AND column_name = 'grouping_batch_id'
);

SET @has_idx_session_grouping := (
    SELECT COUNT(*)
    FROM information_schema.statistics
    WHERE table_schema = 'sesamedb'
      AND table_name = 'tds_session'
      AND index_name = 'idx_session_grouping'
);

SET @sql := IF(
    @has_idx_session_grouping > 0,
    'ALTER TABLE sesamedb.tds_session DROP INDEX idx_session_grouping',
    'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql := IF(
    @has_grouping_id = 0 AND @has_grouping_batch_id > 0,
    'ALTER TABLE sesamedb.tds_session CHANGE COLUMN grouping_batch_id grouping_id BIGINT NULL',
    IF(
        @has_grouping_id = 0,
        'ALTER TABLE sesamedb.tds_session ADD COLUMN grouping_id BIGINT NULL AFTER issue_reason',
        'SELECT 1'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_grouping_group_key := (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = 'sesamedb'
      AND table_name = 'tds_session'
      AND column_name = 'grouping_group_key'
);

SET @sql := IF(
    @has_grouping_group_key > 0,
    'ALTER TABLE sesamedb.tds_session DROP COLUMN grouping_group_key',
    'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql := 'ALTER TABLE sesamedb.tds_session ADD INDEX idx_session_grouping (grouping_id)';
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
