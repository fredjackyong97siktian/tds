-- Theft Action (Beta): audit trail for a human-triggered Stripe charge or
-- WhatsApp message sent against a session whose kiosk analysis detected
-- theft. Every attempt gets its own row - this is both the audit log and
-- the idempotency guard (the app checks for an existing 'success' row of
-- the same action_type before allowing another charge attempt for the same
-- session). Deliberately NEVER triggered automatically - see
-- workflow_service.trigger_theft_action_charge/trigger_theft_action_whatsapp,
-- both of which require an explicit human confirmation from the dashboard.
CREATE TABLE IF NOT EXISTS sesamedb.tds_theft_action (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT NOT NULL,
    action_type VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    identity_type VARCHAR(20),
    identity_entry_id VARCHAR(64),
    amount DECIMAL(12,2),
    currency VARCHAR(10) DEFAULT 'MYR',
    line_items JSON,
    message_text TEXT,
    provider_reference VARCHAR(191),
    provider_response JSON,
    error_message TEXT,
    session_transaction_id BIGINT,
    triggered_by VARCHAR(191),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_theft_action_session_id (session_id),
    CONSTRAINT fk_theft_action_session
        FOREIGN KEY (session_id) REFERENCES tds_session(id) ON DELETE CASCADE,
    CONSTRAINT chk_theft_action_type
        CHECK (action_type IN ('stripe_charge', 'whatsapp_message')),
    CONSTRAINT chk_theft_action_status
        CHECK (status IN ('pending', 'success', 'failed')),
    CONSTRAINT chk_theft_action_identity_type
        CHECK (identity_type IS NULL OR identity_type IN ('credit_card', 'phone'))
);
