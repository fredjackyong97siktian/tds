ALTER TABLE sesamedb.tds_trigger_frame
    ADD COLUMN is_best_for_verification TINYINT(1) NOT NULL DEFAULT 0 AFTER status;
