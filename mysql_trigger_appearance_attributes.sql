-- Adds structured appearance attributes alongside the existing free-text
-- appearance_description - used by grouping_text as a cheap, deterministic
-- pre-filter (comparing fields directly) before ever spending an API call
-- on a candidate that's already clearly implausible. Deliberately does NOT
-- include gender - it was noticed to be wrong often enough that using it as
-- a hard filter risks wrongly rejecting a real match. appearance_description
-- itself is untouched - repair's existing "known appearance" reuse keeps
-- working exactly as before.
ALTER TABLE sesamedb.tds_trigger_event
    ADD COLUMN appearance_attributes JSON DEFAULT NULL
        COMMENT 'Structured appearance breakdown (top_color, top_type, bottom_color, bottom_type, footwear_color, footwear_type, distinguishing_features) from the grouping model - deliberately excludes gender. Used by grouping_text as a cheap pre-filter before spending an API call on a candidate.'
    AFTER appearance_updated_at;
