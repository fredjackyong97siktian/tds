ALTER TABLE sesamedb.tds_trigger_event
    ADD COLUMN appearance_description VARCHAR(255) DEFAULT NULL COMMENT 'Short, reusable description of this trigger''s primary actor (clothing/footwear/carried items) from the grouping model - carried forward into later stages/batches as a matching aid.' AFTER unique_customer_count_source,
    ADD COLUMN appearance_direction VARCHAR(10) DEFAULT NULL COMMENT 'Grouping model''s own entry/exit/unclear read for this trigger, independent of identity.' AFTER appearance_description,
    ADD COLUMN appearance_source VARCHAR(20) DEFAULT NULL COMMENT 'Which grouping stage produced appearance_description/appearance_direction: adjacent, direct, or repair.' AFTER appearance_direction,
    ADD COLUMN appearance_updated_at DATETIME DEFAULT NULL AFTER appearance_source;
