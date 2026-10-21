# Paid Transaction Assignment

## Temporary Pause

Transaction Assignment now defaults to PAUSED. Deploy/restart the API and
theft-confidence worker for this change to take effect. To stop new jobs on
an older deployment immediately, pause Theft Confidence Analysis in Workers.
That older control does not cancel requests already in flight.

Settings -> Pipeline -> Transaction Assignment provides a pause/resume
control backed by `tds_app_setting.paid_transaction_assignment_enabled`.
Only the explicit string `true` enables it. No new SQL migration is needed
for this control. Grouping and existing L1 analysis continue while new
confidence scoring remains queued before assignment; the pause never treats
unassigned purchases as zero purchases or reverts to window-only ownership.

The updated confidence worker terminates its own active confidence child
processes when it sees the pause, including when all worker slots are full.
An external request already accepted by the provider cannot be unbilled by
stopping the local process. An old worker must be restarted to adopt this
behavior. Manual scoring and the assignment API path also check the pause.

Failed receipt decisions are retained and reused rather than automatically
resent, even if the candidate context changes. Resolving/retrying those
failed receipts now requires an explicit reset of their ownership record;
ordinary polling or grouping retries will not trigger another paid attempt.

## Deployment

Run `migrations/20261021_paid_transaction_assignment.sql` on MySQL before
deploying this backend. The new table is also included in `mysql_schema.sql`
for fresh installs. No existing rows or historical results are rewritten.
Restart the API/confidence worker after deployment. No new model setting is
required: ambiguous receipts use the existing OpenRouter key and Jev 1.14.

## Rules

- Only paid transactions participate in receipt ownership. Failed/pending
  transactions remain separate factor evidence, not paid purchases.
- Prefer `paymentAttemptAt` (UTC), converted to store-local time for comparison
  with existing trigger timestamps. Fall back explicitly to the POS
  `Formatted Timestamp` when payment time is null. Never use `createdAt`.
- A group is active at payment when entry <= payment <= its last exit.
  Known open entries are included, even from earlier successful batches.
- Exactly one active group: assign automatically. More than one: compare
  each receipt independently with Jev. A group may receive multiple receipts,
  but a receipt may have only one owner per location.
- For each ambiguous receipt, capture ten kiosk frames in a single FFmpeg
  playback, at one-second intervals from payment - 5s through payment + 4s.
  Apply the kiosk camera's own configured NVR delay; do not entrance-crop.
- Include each candidate's existing entry and exit images. In Edumetro,
  machine Left means image-right, and machine Right means image-left.
  This reversed mapping is enabled by location name containing Edumetro;
  no camera-side assumption is applied to other locations.
- Jev can select one candidate, none, or uncertain. Invalid answers, missing
  images, capture failures and confidence below the current provisional 0.5
  Choice threshold never silently assign a receipt. Preserve probabilities
  for review; this threshold is not yet calibrated on store footage.
- Receipt decisions are serialized per location and cached across groups and
  retries. Identical candidate identities/timestamps reuse the recorded
  decision without another API call. If a previously assigned receipt's
  candidate context changes, retain the original ownership record, flag the
  new audit for review, and do not count that receipt in the new score. Do
  not silently steal a receipt from an already-scored visit.
- Missing SQL migration or failed transaction queries stop scoring instead of
  pretending there were no purchases. Image/API failures are visible as
  unresolved assignment decisions.

## Downstream And Audit

Both batch and single-group confidence paths consume only assigned receipts.
Zero assigned receipts route to the existing identifier-camera no-purchase
carry check when enabled and reached under factor short-circuit rules.
Uncertain ownership does not prove no purchase or theft.

Session creation persists assigned receipts. Confidence-linked kiosk retries
and manual transaction candidate lists consult that ownership, not the raw
visit-window receipt pool. Older results without an assignment audit retain
their legacy behavior.

Confidence Result detail has a Transaction Assignment section showing counts,
payment times, active groups, machine mapping, selected owner, errors, cached
decisions, kiosk frames and Jev probabilities. Script Runs can be filtered by
Transaction Assignment. Every Jev API attempt and its reported cost are
recorded by the existing logged-call/cost pipeline; capture jobs have their
own retrieve_video runs. Deterministic timing assignments make no AI call.

Kiosk probe media uses existing trigger storage prefixes. The cleaner also
protects shared probe aliases when their assigned owner becomes a theft case.

## Verification

Unit tests cover the three timing scenarios, one-receipt ambiguity, multiple
receipts, open visits, UTC conversion, Edumetro mapping, decision caching,
ownership conflicts, session retry guards and ten-frame kiosk capture.
No live database migration, production NVR capture or paid Jev request was
performed as part of local verification.
