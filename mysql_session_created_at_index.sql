-- list_sessions orders by (created_at desc, id desc) with no location_id filter,
-- but the only relevant existing index (idx_session_location_created) requires
-- location_id first and can't help this query. Without a matching index MySQL
-- has to filesort the entire tds_session table for every page of results,
-- which blows past sort_buffer_size once the table gets large enough
-- (error 1038, "Out of sort memory"). This index lets it satisfy the ORDER BY
-- directly via an index scan instead, independent of table size.

SET @has_idx_session_created_at := (
    SELECT COUNT(*)
    FROM information_schema.statistics
    WHERE table_schema = 'sesamedb'
      AND table_name = 'tds_session'
      AND index_name = 'idx_session_created_at'
);

SET @sql := IF(
    @has_idx_session_created_at = 0,
    'ALTER TABLE sesamedb.tds_session ADD INDEX idx_session_created_at (created_at, id)',
    'SELECT 1'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
