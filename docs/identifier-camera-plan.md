# Identifier Camera Configuration

The no-purchase analysis camera is section `identifier`, recorder channel `3`.
Configure it for each applicable location in CCTV setup; the code does not
create or change production camera rows automatically.

Exit Direction is selectable only for `identifier` and `kiosk` cameras.
Directions refer to movement towards the exit as seen in that camera image,
not the person's anatomical left/right. Options include screen directions,
towards camera and away from camera. No direction is inferred by default.
The value is saved per CCTV ID in the existing app-setting table under
`cctv_exit_direction:<id>`; no SQL migration is needed. Updating a camera
through an older client that omits this field preserves the existing value.

Entrance behavior is unchanged. Existing entrance grouping prompts currently
describe screen-right movement as entry and screen-left movement as exit.

Confirmed identifier sampling window for exit `11:00:00`:
`10:59:51.5` to `10:59:57`, twelve samples spaced by 0.5 seconds. Equivalently,
sample times are exit minus 8.5 seconds, plus `index * 0.5` for indices 0..11.
The identifier camera's own time correction must be applied for NVR playback;
it must not inherit the entrance camera's correction.

## No-Purchase Carry Analysis

When the existing confidence pipeline reaches the enabled carry factor and
the visit-window query returned no paid transactions, it always uses Jev,
independent of the regular paid-visit carry-model selector. Existing earlier
factor short-circuit rules and the carry-factor enabled setting still apply.
Paid visits do not retrieve identifier frames and keep their selected model.

One identifier sequence is captured for EACH exit trigger time. A group with
two exits receives 24 identifier images total. Original entrance entry/exit
frames are reused, not retrieved again. All reference and identifier images
are submitted in one Jev request, with chronological sequences, camera labels,
exit-trigger associations and the identifier camera's exit direction.

Each sequence uses one FFmpeg playback pass, a separate NVR/FFmpeg lock, the
identifier camera's own clock correction and no entrance-image crop. Fractional
sample offsets are preserved even though the NVR URL uses whole seconds.
Capture must return all 12 frames; partial, failed, missing-camera or missing-
direction cases remain insufficient evidence and do not silently clear the visit.

Jev compares original belongings with newly carried goods. The prompt includes
the orange-bottle/pink-bottle example. A pre-existing bag or yellow container
alone does not trigger this special no-purchase check; paid-visit carry policy
is unchanged. Suspicious means a carry-factor hit requiring L1, not confirmed
theft. Reasonable means no newly carried goods observed, not proof that nothing
concealed was taken. Insufficient evidence is not a hit or an all-clear.

Retrieval has one `retrieve_video` Script Run per exit with model
`dahua_rtsp_playback:identifier_frames`, linked to its parent carry run. The AI
run uses script name `carry_confidence`, model
`openrouter-jev-1.14_no_purchase_carry_confidence`, and records actual API cost,
responses and identifier-frame URLs in its output. Uploads are under each exit
trigger's existing media prefix so the normal cleaner/theft-owner protection
continues to apply. No new video_asset or trigger_frame_asset rows are created.
