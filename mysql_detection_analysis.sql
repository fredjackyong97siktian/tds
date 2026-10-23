-- Apply before starting the Detection Analysis worker. Existing audit rows remain.
ALTER TABLE sesamedb.tds_script_run
    DROP CHECK chk_script_run_name,
    ADD CONSTRAINT chk_script_run_name CHECK (script_name IN (
        'retrieve_video', 'retrieve_frame', 'entry', 'kiosk', 'kiosk_match',
        'grouping', 'grouping_repair', 'carry_confidence', 'detection_analysis'
    ));
