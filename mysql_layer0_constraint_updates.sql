-- Layer 0 constraint updates for existing MySQL databases.
-- Run this after deploying the Layer 0 worker/filtering code if your tables
-- were created before grouping/theft-confidence and trigger-frame statuses.

ALTER TABLE sesamedb.tds_worker_control
    DROP CHECK chk_worker_control_name;

ALTER TABLE sesamedb.tds_worker_control
    ADD CONSTRAINT chk_worker_control_name
    CHECK (worker_name IN (
        'retrieval',
        'grouping',
        'theft_confidence_analysis',
        'entrance_analysis',
        'kiosk_analysis'
    ));

ALTER TABLE sesamedb.tds_script_run
    DROP CHECK chk_script_run_name;

ALTER TABLE sesamedb.tds_script_run
    ADD CONSTRAINT chk_script_run_name
    CHECK (script_name IN (
        'retrieve_video',
        'entry',
        'kiosk',
        'kiosk_match',
        'grouping'
    ));

ALTER TABLE sesamedb.tds_video_asset
    DROP CHECK chk_video_asset_status;

ALTER TABLE sesamedb.tds_video_asset
    ADD CONSTRAINT chk_video_asset_status
    CHECK (status IN (
        'not_retrieved',
        'retrieving',
        'frames_retrieved',
        '10_frames_retrieved',
        'full_video_not_retrieved',
        'ready',
        'processing',
        'processed',
        'deleted',
        'issue'
    ));
