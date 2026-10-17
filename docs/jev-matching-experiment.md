# Jev 1.14 Matching Experiment

This is an opt-in adjacent-matching test, NOT a replacement of the production
grouping provider. Scheduled/manual grouping, MiMo and kiosk analysis remain
unchanged. No grouping batches, confidence results, frame statuses or sessions
are modified. Only the experiment's script_run and cost/logs are recorded.

## Run On The API Server

Use the same Python environment and working directory as tds_api. Configure
`THEFT_API_OPENROUTER_API_KEY` in its environment or `.env`, or set
`OPENROUTER_API_KEY`. A TypeSafe key does not authenticate with OpenRouter.
Do not paste credentials into commands or commit them.

Validate trigger selection first (no AI calls, downloads or DB writes):

```sh
python -m app.workers.try_jev_grouping --entry 82 --candidates 83 --dry-run
```

Send one paid request:

```sh
python -m app.workers.try_jev_grouping --entry 82 --candidates 83 --frames 5
```

82/83 are examples, not an automatically selected test. Choose a known correct
pair with retained frames. Add a second candidate ID to test a distractor.
Use a known non-match too, to test whether Jev correctly chooses none.

Each test accepts one confirmed identity-bearing entry and one or two later
same-location candidate exits within 30 minutes. It sends the first five usable
frames per trigger, or fewer with `--frames`. Frame order and timestamps are
preserved. Only the entry/trigger IDs and frame data are sent, not phone numbers,
fingerprints or transactions. Images are downloaded locally, rotated upright,
resized to a maximum 1024px side and padded if needed to meet the stricter 280px
upstream minimum. They are sent as JPEG data URLs, not remote URL text.

Jev receives a Choice with candidate IDs plus none, and an independent Noul
same-customer/exiting check per candidate. These questions run in ONE API request.
The experiment calls `https://openrouter.ai/api/alpha/decisions` with
`typesafe/jev-1.14`, not `/chat/completions`.

The output contains a proposed match/no_match/uncertain result, raw probabilities,
API time, total elapsed time, reported USD cost, and the script_run ID. A match
requires choice confidence and same-customer probability of at least 0.85 and
low probability for the other candidate. These are provisional, uncalibrated
thresholds, not evidence of production accuracy. Conflicting answers do not form
a group. Jev does not generate explanations, bag descriptions or the full
metadata production grouping currently expects.

Look in Monitoring > Script Runs for model `openrouter-jev-1.14_matching_test`.
It uses the existing `grouping` script type, with `test_only=true` and no batch ID,
so a successful test cannot trigger theft-confidence analysis or session creation.
Actual returned `usage.cost` is recorded before interpreting the answer. Missing
cost stays unknown instead of being invented. The request log omits image base64
to avoid large duplicate payloads. A failed test is not retried automatically.

## Sources

- https://openrouter.ai/docs/guides/community/jev
- https://openrouter.ai/docs/guides/community/multimodal-decisions
- https://docs.typesafe.ai/images

Vision is experimental. Validate known positives, negatives, bystanders,
similar clothing, changed bags and poor visibility before switching providers.
