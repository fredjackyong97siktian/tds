CREATE TABLE IF NOT EXISTS sesamedb.tds_paid_transaction_assignment (
    location_id BIGINT NOT NULL,
    receipt_number VARCHAR(255) NOT NULL,
    payment_time_utc DATETIME(6) NOT NULL,
    entry_trigger_id BIGINT NULL,
    status VARCHAR(20) NOT NULL,
    method VARCHAR(30) NOT NULL,
    context_hash CHAR(64) NOT NULL,
    evidence JSON NOT NULL,
    script_run_id BIGINT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (location_id, receipt_number),
    KEY idx_paid_assignment_entry (entry_trigger_id),
    CONSTRAINT chk_paid_assignment_status CHECK (status IN ('assigned', 'none', 'uncertain', 'error'))
);
