CREATE TABLE IF NOT EXISTS sesamedb.tds_blacklist_entry (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    method VARCHAR(50) NOT NULL,
    entry_id VARCHAR(255) NOT NULL,
    criteria TEXT NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_blacklist_entry_method_entry_id (method, entry_id)
);
