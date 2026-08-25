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
        'grouping',
        'carry_confidence'
    ));

ALTER TABLE sesamedb.tds_script_run
    ADD COLUMN cost_amount DECIMAL(12,6) NULL,
    ADD COLUMN cost_currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    ADD COLUMN cost_source VARCHAR(50) NULL;

ALTER TABLE sesamedb.tds_video_asset
    DROP CHECK chk_video_asset_status;

ALTER TABLE sesamedb.tds_video_asset
    ADD CONSTRAINT chk_video_asset_status
    CHECK (status IN (
        'not_retrieved',
        'retrieving',
        'frames_retrieved',
        '10_frames_retrieved',
        'ready',
        'processing',
        'processed',
        'deleted',
        'issue'
    ));

ALTER TABLE sesamedb.tds_filter_grouping_batch
    DROP CHECK chk_filter_grouping_batch_status;

ALTER TABLE sesamedb.tds_filter_grouping_batch
    ADD CONSTRAINT chk_filter_grouping_batch_status
    CHECK (status IN (
        'pending',
        'dispatching',
        'running',
        'success',
        'failed',
        'issue'
    ));

CREATE TABLE IF NOT EXISTS sesamedb.tds_trigger_frame_asset (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    trigger_id BIGINT NOT NULL,
    location_id BIGINT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'not_retrieved',
    error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_trigger_frame_asset_trigger_id (trigger_id),
    KEY idx_trigger_frame_asset_location_status (location_id, status),
    KEY idx_trigger_frame_asset_status_start (status, start_time),
    CONSTRAINT fk_trigger_frame_asset_trigger
        FOREIGN KEY (trigger_id) REFERENCES tds_trigger_event(id) ON DELETE CASCADE,
    CONSTRAINT chk_trigger_frame_asset_status
        CHECK (status IN ('not_retrieved', 'retrieving', 'retrieved', 'processing', 'processed', 'deleted', 'issue'))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_trigger_frame (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    frame_asset_id BIGINT NOT NULL,
    trigger_id BIGINT NOT NULL,
    frame_index INT NOT NULL,
    sample_time DATETIME,
    image_url TEXT,
    status VARCHAR(30) NOT NULL DEFAULT 'ok',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_trigger_frame_asset_index (frame_asset_id, frame_index),
    KEY idx_trigger_frame_trigger_id (trigger_id),
    CONSTRAINT fk_trigger_frame_asset
        FOREIGN KEY (frame_asset_id) REFERENCES tds_trigger_frame_asset(id) ON DELETE CASCADE,
    CONSTRAINT fk_trigger_frame_trigger
        FOREIGN KEY (trigger_id) REFERENCES tds_trigger_event(id) ON DELETE CASCADE,
    CONSTRAINT chk_trigger_frame_status
        CHECK (status IN ('ok', 'failed', 'deleted'))
);
