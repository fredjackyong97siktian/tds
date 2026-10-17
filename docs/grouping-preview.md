# Grouping Preview

`POST /api/v1/filtering/grouping-batches/preview` is read-only. It does not
create batches, send AI requests, download images, or change trigger status.

Request modes:

- `{"mode":"now"}`: same per-location ready-frame selection and one-hour
  window before the newest trigger as Run Now.
- `{"mode":"range","location_id":2,"start_time":"2026-10-17T12:00","end_time":"2026-10-17T13:00"}`:
  explicit trigger-clock range, start inclusive and end exclusive.
- `{"mode":"retry","batch_id":10}`: current linked trigger/frame snapshot.

Adjacent uses the same planning function as execution: one identity entry
with frames, paired with up to two later non-identity candidates within 30
minutes. Direct is a baseline count of whole-trigger chunks assuming no
matches in earlier stages. It is not an exact total of API attempts.

Wait estimates use the median elapsed seconds per entry/chunk from at most
20 recent successful script runs per stage, matching the current provider
label. Historical elapsed times include verification, ordinary retries and
request spacing. Until timing samples exist, missing stages use an explicitly
labelled rough 60-second-per-unit allowance. The displayed range is a heuristic,
not a statistical confidence interval or a maximum timeout.

Text/repair, queue waiting, and subsequent theft-confidence analysis are not
included. Carry-forward entries can increase chunk count; earlier matches can
reduce it. No capacity is reserved, so eligibility may change before confirmation.
Uncached motion-sensor person checks are not run during preview, and may change
the final selection. Retry uses current linked snapshots, which execution may
refresh from newly available frames.

The UI requests this preview before Run Now, Run Range or Retry, then waits
for Confirm & Run before invoking the existing execution endpoint. No schema
change is required. Scheduled workers keep their existing automatic behavior.
