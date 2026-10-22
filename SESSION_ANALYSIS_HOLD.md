# Session Analysis Hold

Session analysis is temporarily paused for every group by `PAUSED = True` in
`app/services/session_analysis.py`. This is a code-level hold, independent of
the entrance worker's dashboard toggle. No database migration is required.

- Grouping, paid-receipt assignment, and L0 confidence scoring continue.
- L1 decisions still depend on enabled factor hits; not every group is flagged.
- Entry, kiosk, and exit video retrieval continues for every completed group.
- Once videos are ready, the media plan is complete with
  `session_analysis_status: paused`. It does not create a Session or enable L1.
- The entrance worker does not claim new analysis jobs. Manual entry and kiosk
  analysis dispatches are blocked, too.
- No-transaction checks are deferred out of L0 Carry Item Signal. They do not
  retrieve identifier frames or call Jev during confidence scoring.
- On a future explicit resume, only a group requiring L1 and having no assigned
  paid receipts can run the no-transaction check in its Session flow, regardless
  of which factor hit. Its Script Run records the Session ID and API cost.

Deploy the backend and restart the API, confidence worker, retrieval worker, and
analysis worker for this hold to take effect. Already-running remote jobs are
not cancelled. Existing Sessions, costs, and evidence are not deleted.

To resume later, deliberately change `PAUSED` to `False` and restart those
processes. `finish_pending` also picks up previously completed media plans marked
paused. Safe groups retain their videos without creating an analysis Session.
