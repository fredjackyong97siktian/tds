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

Failed receipt decisions are retained rather than automatically resent, even
if the candidate context changes. If corrected timing proves there are zero
or one active groups, the failure can be resolved deterministically without
another API request (the prior Script Run ID remains in the audit). Still-
ambiguous failures require an explicit retry; ordinary polling or grouping
retries will not trigger another paid attempt.

To resume, enable Transaction Assignment in Settings -> Pipeline. This also
resumes the Theft Confidence Analysis worker. Open the affected Transaction
Assignment Script Run and click Retry Transaction Assignment. This queues only
that receipt's affected confidence scores in its original batch, preserving
result IDs, grouping memberships and script history. The worker reuses a saved
Jev answer first. If none exists, the click authorizes one new paid attempt;
the durable attempt barrier is restored before making that request. No new
database migration is required for this retry control.

The Grouping page also has a Transaction Assignments panel showing the latest
20 unresolved receipts directly from the ownership ledger. Refresh checks the
latest state without loading Script Run logs. Retry is available in that panel
when processing is enabled; queued receipts are labeled to avoid repeated
clicks. Retry API failures return a JSON diagnostic (including database error
codes when available), and the browser also handles plain-text proxy errors.
Retry queue flags use JSON booleans; the worker accepts both boolean `true`
and legacy numeric `1` so older queued retries are not silently skipped.

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
- A completed group is active at payment when entry <= payment <= its last
  exit. Groups whose actual exit precedes the payment are never candidates.
  Both an entry and at least one exit must exist in the group.
- Open entries, unknowns and standalone identities are excluded, even if they
  occurred recently. They become eligible only after a group is formed.
  The current group must exist in its successful batch's saved grouping output
  or committed memberships; the assignment code cannot invent a group from
  a supplied trigger ID. This does not change grouping carry-forward rules.
- Queries use actual payment bounds rather than the whole batch period.
  A visit with no paid receipts skips candidate lookup entirely.
- Exactly one active group: assign automatically. More than one: compare
  each receipt independently with Jev. A group may receive multiple receipts,
  but a receipt may have only one owner per location.
- For each ambiguous receipt, capture ten kiosk frames in a single FFmpeg
  playback, at one-second intervals from payment - 5s through payment + 4s.
  Apply the kiosk camera's own configured NVR delay; do not entrance-crop.
- Before starting kiosk capture, wait for the same NVR/FFmpeg slot used by
  video and trigger-frame retrieval jobs. Lock checks use non-blocking
  `GET_LOCK(..., 0)` calls separated by short Python sleeps, never a long
  query exceeding PyMySQL's 30-second read timeout. Receipt captures wait
  up to `transaction_assignment_retrieval_wait_seconds` (default 900 seconds).
  A busy-slot deadline stops before capture or Jev and records a retryable
  error rather than assuming no purchase. Retrieval logs record actual wait
  duration. Restart the API, retrieval and confidence workers after deployment;
  all must use the shared lock implementation. No database migration needed.
- Include each candidate's existing entry and exit images. In Edumetro,
  machine Left means image-right, and machine Right means image-left.
  Machine names such as `Edumetro-right-1`, `Edumetro-left-2`, and `RIGHT_1`
  are recognized by their left/right token. Ambiguous names containing both
  directions are not guessed.
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
- The location-lock holder refreshes its DB read transaction before loading
  cached ownership, so a snapshot opened before waiting on the lock cannot
  hide another worker's just-committed result.
- Before any paid visual request, commit an error-state attempt reservation
  in the existing ownership table. This is an incomplete-attempt barrier,
  not a theft or no-purchase finding. If that write fails, no API call starts.
  If the child dies mid-request, automatic resending is blocked rather than
  risking another charge for a request whose outcome is unknown.
- After a valid API answer, checkpoint its decision on the Script Run while
  it is still running. Only mark the run successful AFTER ownership commits.
  A failed ownership write is shown as a persistence failure, not success.
  Subsequent work can recover this checkpoint or a legacy successful run by
  location and receipt without sending the images again. Existing failed API
  responses are not mistaken for recoverable successful answers.
- Successful `none` and `uncertain` decisions also remain cached if another
  batch changes the candidate set; flag the changed context for review instead
  of automatically paying for a new answer. A trigger ID is not the receipt's
  idempotency key: separate receipts may legitimately share the same trigger.
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
Regression coverage also includes 531 stale open entries, per-payment age
limits, current-batch stale snapshots, real long completed visits, automatic
assignment without Jev, and full kiosk machine names.
No live database migration, production NVR capture or paid Jev request was
performed as part of local verification.
