ALTER TABLE sesamedb.tds_trigger_event
    ADD COLUMN customer_group_type VARCHAR(20) DEFAULT NULL AFTER unique_customer_count_source,
    ADD COLUMN age_brackets JSON DEFAULT NULL AFTER customer_group_type;
