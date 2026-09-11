ALTER TABLE sesamedb.tds_trigger_frame_asset
    ADD COLUMN retry_count INT NOT NULL DEFAULT 0 AFTER error;
