-- MySQL schema for theft-detection business data
--
-- Target database:
--   sesamedb
--
-- Recommended:
--   Create a dedicated MySQL user for the FastAPI server with
--   SELECT, INSERT, UPDATE, and only the minimum extra privileges needed.
--
-- Example:
--   CREATE DATABASE IF NOT EXISTS sesamedb.tds_sesamedb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
--   CREATE USER 'theft_api'@'%' IDENTIFIED BY 'change_me';
--   GRANT SELECT, INSERT, UPDATE, DELETE ON sesamedb.tds_* TO 'theft_api'@'%';
--   FLUSH PRIVILEGES;
--
-- Then:
--   USE sesamedb;
--   SOURCE mysql_schema.sql;

CREATE TABLE IF NOT EXISTS sesamedb.tds_whitelist_entry (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    method VARCHAR(50) NOT NULL,
    entry_id VARCHAR(255) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_whitelist_entry_method_entry_id (method, entry_id)
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_cctv (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    location_id BIGINT NOT NULL,
    section VARCHAR(50) NOT NULL,
    stream_name VARCHAR(255),
    recorder_channel VARCHAR(100),
    delayed_seconds INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_cctv_location_section (location_id, section)
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_trigger_event (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    location_id BIGINT NOT NULL,
    aqara_event_id VARCHAR(255),
    trigger_source VARCHAR(100) NOT NULL DEFAULT 'aqara',
    trigger_time DATETIME NOT NULL,
    phone_entry_id BIGINT COMMENT 'Optional link to the phone-based entry record if this trigger was matched by phone.',
    credit_card_entry_id BIGINT COMMENT 'Optional link to the credit-card-based entry record if this trigger was matched by card.',
    entry_source_type VARCHAR(30) NOT NULL DEFAULT 'unknown' COMMENT 'How the door was opened or resolved: phone, credit_card, app, unknown.',
    entry_match_status VARCHAR(30) NOT NULL DEFAULT 'pending' COMMENT 'Whether entry identity resolution is pending, matched, unmatched, not_applicable, or ambiguous.',
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    whitelist_hit TINYINT(1) NOT NULL DEFAULT 0,
    raw_payload JSON,
    issue_reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_trigger_event_location_time (location_id, trigger_time),
    KEY idx_trigger_event_phone_entry_id (phone_entry_id),
    KEY idx_trigger_event_credit_card_entry_id (credit_card_entry_id),
    CONSTRAINT chk_trigger_status
        CHECK (status IN ('pending', 'video_pending', 'processing', 'done', 'issue', 'whitelisted')),
    CONSTRAINT chk_trigger_entry_source_type
        CHECK (entry_source_type IN ('phone', 'credit_card', 'app', 'unknown')),
    CONSTRAINT chk_trigger_entry_match_status
        CHECK (entry_match_status IN ('pending', 'matched', 'unmatched', 'not_applicable', 'ambiguous')),
    CONSTRAINT chk_trigger_entry_reference
        CHECK (
            (phone_entry_id IS NULL OR credit_card_entry_id IS NULL)
            AND (
                entry_source_type <> 'phone' OR phone_entry_id IS NOT NULL
            )
            AND (
                entry_source_type <> 'credit_card' OR credit_card_entry_id IS NOT NULL
            )
        )
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_session (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    entry_trigger_id BIGINT NOT NULL,
    exit_trigger_id BIGINT,
    location_id BIGINT NOT NULL,
    start_time DATETIME,
    end_time DATETIME,
    total_item_brought INT NOT NULL DEFAULT 0,
    actual_items_brought INT NOT NULL DEFAULT 0,
    transaction_total_items INT NOT NULL DEFAULT 0,
    total_customer INT NOT NULL DEFAULT 0,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    result_summary JSON,
    issue_reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_session_entry_trigger_id (entry_trigger_id),
    KEY idx_session_exit_trigger_id (exit_trigger_id),
    KEY idx_session_location_created (location_id, created_at),
    CONSTRAINT fk_session_entry_trigger
        FOREIGN KEY (entry_trigger_id) REFERENCES tds_trigger_event(id),
    CONSTRAINT fk_session_exit_trigger
        FOREIGN KEY (exit_trigger_id) REFERENCES tds_trigger_event(id),
    CONSTRAINT chk_session_status
        CHECK (status IN ('pending', 'processing_entry', 'processing_kiosk', 'detected', 'not_detected', 'need_review', 'issue', 'whitelisted', 'closed'))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_session_customer (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    person_id INT NOT NULL,
    merged_into_session_customer_id BIGINT COMMENT 'If this row is a duplicate ReID/person track within the same session, point to the canonical session_customer row.',
    enter_time DATETIME,
    kiosk_start_time DATETIME,
    leave_time DATETIME,
    match_status VARCHAR(30) NOT NULL DEFAULT 'tracked',
    merge_reason TEXT,
    merged_at DATETIME,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_session_customer (session_id, person_id),
    KEY idx_session_customer_merged_into (merged_into_session_customer_id),
    CONSTRAINT fk_session_customer_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE,
    CONSTRAINT fk_session_customer_merged_into
        FOREIGN KEY (merged_into_session_customer_id) REFERENCES tds_session_customer(id) ON DELETE SET NULL,
    CONSTRAINT chk_session_customer_match_status
        CHECK (match_status IN ('tracked', 'merged', 'resolved', 'issue'))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_session_transaction (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    receipt_number VARCHAR(255) NOT NULL,
    transaction_time DATETIME,
    total_items INT NOT NULL DEFAULT 0,
    total_amount DECIMAL(12, 2),
    raw_payload JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_session_transaction_session_id (session_id),
    CONSTRAINT fk_session_transaction_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_video_asset (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    trigger_id BIGINT,
    section VARCHAR(30) NOT NULL,
    sequence_no INT,
    video_url TEXT NOT NULL,
    file_path TEXT,
    captured_start_time DATETIME,
    captured_end_time DATETIME,
    retention_until DATETIME COMMENT 'Delete the stored video file/object and this row after this timestamp; use 3-day retention by default.',
    status VARCHAR(30) NOT NULL DEFAULT 'not_retrieved' COMMENT 'Video lifecycle: not_retrieved, retrieving, ready, deleted, or issue.',
    metadata JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_video_asset_trigger_id (trigger_id),
    KEY idx_video_asset_section_sequence (section, sequence_no),
    KEY idx_video_asset_retention_until (retention_until),
    CONSTRAINT fk_video_asset_trigger
        FOREIGN KEY (trigger_id) REFERENCES tds_trigger_event(id) ON DELETE CASCADE,
    CONSTRAINT chk_video_asset_section
        CHECK (section IN ('entrance', 'kiosk')),
    CONSTRAINT chk_video_asset_status
        CHECK (status IN ('not_retrieved', 'retrieving', 'ready', 'processing', 'processed', 'deleted', 'issue'))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_session_video_asset (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    video_asset_id BIGINT NOT NULL,
    section VARCHAR(30) NOT NULL,
    sequence_no INT,
    clip_start_time DATETIME,
    clip_end_time DATETIME,
    is_primary TINYINT(1) NOT NULL DEFAULT 0,
    metadata JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_session_video_asset_session_video (session_id, video_asset_id),
    KEY idx_session_video_asset_session_section_sequence (session_id, section, sequence_no),
    KEY idx_session_video_asset_video_asset_id (video_asset_id),
    CONSTRAINT fk_session_video_asset_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE,
    CONSTRAINT fk_session_video_asset_video_asset
        FOREIGN KEY (video_asset_id) REFERENCES tds_video_asset(id) ON DELETE CASCADE,
    CONSTRAINT chk_session_video_asset_section
        CHECK (section IN ('entrance', 'kiosk')),
    CONSTRAINT chk_session_video_asset_primary
        CHECK (is_primary IN (0, 1))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_script_run (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT,
    trigger_id BIGINT,
    script_name VARCHAR(50) NOT NULL,
    model_name VARCHAR(255),
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    command TEXT NOT NULL,
    stdout_log LONGTEXT,
    stderr_log LONGTEXT,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME,
    KEY idx_script_run_session_id (session_id),
    KEY idx_script_run_trigger_id (trigger_id),
    CONSTRAINT fk_script_run_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE,
    CONSTRAINT fk_script_run_trigger
        FOREIGN KEY (trigger_id) REFERENCES tds_trigger_event(id) ON DELETE CASCADE,
    CONSTRAINT chk_script_run_name
        CHECK (script_name IN ('retrieve_video', 'entry', 'kiosk')),
    CONSTRAINT chk_script_run_status
        CHECK (status IN ('pending', 'running', 'success', 'failed'))
);

CREATE TABLE IF NOT EXISTS sesamedb.tds_kiosk_video_result (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    session_video_asset_id BIGINT NOT NULL,
    script_run_id BIGINT,
    sequence_no INT NOT NULL,
    items_added_in_clip INT NOT NULL DEFAULT 0,
    cumulative_items_after_clip INT NOT NULL DEFAULT 0,
    analysis_status VARCHAR(30) NOT NULL DEFAULT 'pending',
    result_payload JSON,
    issue_reason TEXT,
    started_at DATETIME,
    finished_at DATETIME,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_kiosk_video_result_session_video_asset_id (session_video_asset_id),
    UNIQUE KEY uq_kiosk_video_result_session_sequence (session_id, sequence_no),
    KEY idx_kiosk_video_result_session_id (session_id),
    KEY idx_kiosk_video_result_session_video_asset_id (session_video_asset_id),
    KEY idx_kiosk_video_result_script_run_id (script_run_id),
    CONSTRAINT fk_kiosk_video_result_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE,
    CONSTRAINT fk_kiosk_video_result_session_video_asset
        FOREIGN KEY (session_video_asset_id) REFERENCES tds_session_video_asset(id) ON DELETE CASCADE,
    CONSTRAINT fk_kiosk_video_result_script_run
        FOREIGN KEY (script_run_id) REFERENCES tds_script_run(id) ON DELETE SET NULL,
    CONSTRAINT chk_kiosk_video_result_status
        CHECK (analysis_status IN ('pending', 'running', 'done', 'issue'))
);
