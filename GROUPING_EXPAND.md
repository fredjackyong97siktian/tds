# Expand Grouping

Pipeline: adjacent -> optional text -> direct -> Expand -> optional repair -> confidence.

Expand uses Jev 1.14 and exactly the existing Jev adjacent request/result
adapter. It considers identity-bearing entries without an entry group after
direct. The evidence pool is saved before adjacent consumes its matches, so
candidate triggers may have an identity and/or already belong to another
group. Existing groups are not removed or replaced.

## Time Window

For each seed entry, candidates must satisfy:

```
entry_time < candidate_time <= entry_time + configured_duration
candidate_time < running_batch.window_end
```

The default duration is 20 minutes. A 22:23 entry in a batch ending at 22:30
only searches triggers before 22:30, not through 22:43. No candidates means
no API call or Expand Script Run. Triggers without usable frames, whitelisted
triggers, and known no-person/retrieval issues remain excluded as in grouping.
Expand also checks the live `tds_trigger_event.motion_person_present` flag in
one bulk lookup and excludes confirmed `0`/false results from both seeds and
candidates, even if a stale batch snapshot still contains their frames.
Unchecked (`NULL`) and uncertain results remain eligible; no presence API call
is made just to repeat a cached non-customer verdict. Excluded IDs are included
in Expand diagnostics.

Edit the duration (1-120 whole minutes) in Settings -> Pipeline -> Expand
Matching Duration. The single saved value is `tds_app_setting` key
`grouping_expand_duration_minutes`; there is no competing environment override.
The next grouping run reads the saved value.

## Matching And Audit

Expand compares each entry against chronological pairs of candidates using
the same prompt as adjacent, including the multi-person and exiting-direction
rules. A choice of candidate 1, candidate 2, both, or none is handled exactly
as in Jev adjacent, without a confidence cutoff. It stops at the first match
for that entry. A comparison failure stops that entry's scan, preserves it
unresolved, and is logged rather than guessed around.

Matches use source `jev_grouping_expand` and retain their selected-option
probabilities, shared trigger IDs and window cutoff. They join the same batch
summary and normal confidence-analysis queue. Carry analysis already anchors
its comparison to the grouped entry customer's identity, not every person in
the shared exit frames. This is still visual matching, not guaranteed identity;
test overlap examples against real footage before relying on the outcome.

Script Runs -> Grouping Expand shows a separate
`openrouter-jev-1.14_grouping_expand` run. Every API attempt and its actual
reported cost use the existing logging pipeline. Planned attempts are an
upper bound (ceil(candidate_count / 2) per entry), since an early match stops
the scan. Actual attempts and results are updated during execution.

## Required MySQL Migration

Apply `migrations/20261021_grouping_shared_memberships.sql` once with an
administrator account, BEFORE deploying/restarting the backend. It replaces
the unique `(batch_id, trigger_id)` index with a unique membership index on
`(batch_id, trigger_id, role, coalesce(group_key, ''))`. The generated column
does not require additional application-side writes or permissions. Existing
rows remain intact; the application refuses to run Expand without this column
to avoid overwriting an existing membership.

The fresh-install schema includes the replacement index. No live database was
migrated during development. Historical batches do not run Expand unless
retried. Adjacent/direct model selection, paid assignment and kiosk analysis
are not switched to different models by this feature.
