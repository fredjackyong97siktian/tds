-- Gates motion_sensor-sourced triggers out of grouping entirely when no
-- person is visible in their frames (motion sensors can false-trigger on
-- carts, pets, shadows, lighting changes etc. - unlike front_door_sensor,
-- where the door physically opening is already a strong signal). Checked
-- once per trigger, right before it's linked into any grouping batch; a
-- failed API call leaves this NULL and the trigger is used anyway (fail
-- open - never let a flaky check cost a real customer).
ALTER TABLE sesamedb.tds_trigger_event
    ADD COLUMN motion_person_present TINYINT(1) DEFAULT NULL
        COMMENT 'For trigger_source=motion_sensor only: whether a human was confirmed present in this trigger''s frames. NULL = not yet checked (or the check failed - treated as present for grouping purposes).'
        AFTER appearance_attributes,
    ADD COLUMN motion_person_present_confidence FLOAT DEFAULT NULL
        AFTER motion_person_present,
    ADD COLUMN motion_person_checked_at DATETIME DEFAULT NULL
        AFTER motion_person_present_confidence;
