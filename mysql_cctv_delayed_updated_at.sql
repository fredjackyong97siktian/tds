ALTER TABLE sesamedb.tds_cctv
    ADD COLUMN delayed_updated_at DATETIME DEFAULT NULL COMMENT 'When delayed_seconds was last recalibrated against the NVR''s own clock - refreshed at most once per calendar day.' AFTER delayed_seconds;
