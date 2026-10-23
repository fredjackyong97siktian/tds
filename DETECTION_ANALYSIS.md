# Detection Analysis

The new L1 path defaults to one MiMo-V2.6-Flash request over a maximum of 40 source
images and transaction records. It replaces automatic entry/kiosk video analysis
for newly completed, flagged confidence groups. It does not change grouping or
L0 carry-factor providers. Legacy entry/kiosk and Jev no-transaction analysis
remain paused.

## Deployment

1. `mysql_detection_analysis.sql` is optional. If the database rejects the new
   script name specifically under `chk_script_run_name`, logging falls back to
   the already-allowed `carry_confidence` type with a `_detection_analysis` model
   suffix. The UI, filters, and worker activity still label it Detection Analysis.
   No paid request is sent until a Script Run record exists. Other database
   errors are not silently retried. Applying the SQL permits the dedicated
   database script type; retain any extra custom names in your own constraint.
2. Configure the existing `THEFT_API_OPENROUTER_API_KEY` in the server environment.
   The local development `openrouter.txt` is NOT used by production code and
   must not be committed, copied into an image, or exposed to the browser.
3. Rebuild the API, retrieval worker, and new detection worker:
   `docker compose up -d --build theft-api theft-retrieval-worker theft-detection-analysis-worker`.
4. Deploy the corresponding frontend for the session result and retry control.

## Model Selection

Use Settings > Pipeline Settings > Detection Analysis Model. The choice is
stored in `tds_app_setting` under `detection_analysis_model`; no new table or
column is needed. It is separate from grouping, kiosk, and L0 carry providers.
The dropdown includes existing supported OpenRouter choices and a custom
OpenRouter model ID input. Custom models must accept images and JSON output;
the selected model's availability, limits, and price are controlled by OpenRouter.
No paid test is sent when saving the setting.

The worker reads the saved model just before each new paid attempt and records
it on that attempt. Running calls and saved answers are unchanged. Retrying a
failed persistence step still recovers its original answer/model before any new
paid call; explicitly retrying an inconclusive completed result uses the new
selection and creates a new Script Run. There is no detection model environment
override. Default remains `xiaomi/mimo-v2.6-flash` when no selection was saved.

## Flow

- Every completed group still preserves entry, all exit, and kiosk evidence.
- A group without a factor hit creates no paid Detection Analysis request.
- A hit creates/links its session only after required videos are retrieved.
- Detection is queued in `tds_session.result_summary.detection_analysis`.
- The dedicated worker waits for resolved paid-receipt ownership, assembles
  source evidence, and makes one streamed request. It does not wait on or occupy
  the NVR retrieval lock. It respects the existing `entrance_analysis` pause control.
- The model sees receipt items, quantities, amounts, and actual payment statuses,
  but NOT the factor verdict, previous AI findings, or a known theft label.
- MiMo `suspicious` maps to session `detected`, `reasonable` to `not_detected`,
  and `insufficient_evidence` to `need_review`. No confidence cutoff changes its
  valid decision. These are model findings, not proof of criminal intent.
- Each paid attempt has its own `detection_analysis` Script Run. Actual returned
  OpenRouter cost and raw response are persisted before applying the decision.
  Failed or interrupted attempts do not automatically make more paid requests.
- A raw-response journal on the mounted session volume additionally protects
  answers and returned cost during database outages. It contains model output,
  not credentials or image base64. Recovery restores the database audit before
  applying that saved answer.
- POST `/api/v1/sessions/{id}/detection-analysis/retry` queues work, not inline
  HTTP inference. A saved complete answer is recovered first after persistence
  failure; explicitly retrying an inconclusive completed result creates a new
  billed run, retaining the old run and cost. Resolved results cannot be resent
  accidentally by repeatedly clicking retry.

## Scope And Recovery

Existing completed/paused confidence results are not mass-enqueued on deployment.
Only newly finishing flagged groups enter this queue; historical analysis and
known-theft results remain unchanged. A worker restart recovers a saved answer
or marks an interrupted attempt as failed for explicit review/retry.

Sampling is evidence, not continuous-video proof. Missing exit references, hidden
goods, clock offsets, sparse footage, and uncertain receipts can leave a case
inconclusive. The session detail shows the model's reason and limitations.
No automatic Stripe charge, blacklist, or other theft action is performed.

## Issues Page

Unresolved Detection Analysis appears in Action > Issues: completed results with
`insufficient_evidence`, failed detection attempts, and their explicitly requested
queued/running retries. Initial routine jobs and resolved detected/not-detected
results are excluded. The list is paginated and includes session, confidence
result, and Script Run links. Retry uses the same background-only endpoint and
billing/recovery rules as session detail. Listing/refreshing issues never calls
the paid model. It does not list unrelated failed Script Runs.
