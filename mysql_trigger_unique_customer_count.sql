ALTER TABLE sesamedb.tds_trigger_event
    ADD COLUMN unique_customer_count SMALLINT DEFAULT NULL AFTER issue_reason,
    ADD COLUMN unique_customer_count_confidence FLOAT DEFAULT NULL AFTER unique_customer_count,
    ADD COLUMN unique_customer_count_source VARCHAR(20) DEFAULT NULL AFTER unique_customer_count_confidence;
