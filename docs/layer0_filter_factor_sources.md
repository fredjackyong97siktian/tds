# Layer 0 Filter Factor Sources

This file records the agreed source of truth for each Layer 0 filter factor.

| Factor | Detail Needed | Correct Source |
|---|---|---|
| Long stay + low purchase | Entry time | MySQL `tds_trigger_event.trigger_time` |
| Long stay + low purchase | Exit time | MySQL `tds_trigger_event.trigger_time` |
| Long stay + low purchase | Duration between entry and exit | Calculated from entry/exit trigger time |
| Long stay + low purchase | Paid transaction count | MySQL `transaction.status = paid`, group by `receiptNumber` |
| Long stay + low purchase | Total transaction value | MySQL `transaction.status = paid`, group by `receiptNumber` |
| Long stay + low purchase | Total transaction quantity | MySQL `transactionDetail.quantity`, group by `receiptNumber` |
| Transaction issue + low purchase | Failed/pending transaction followed by a low final paid receipt | MySQL transaction status, `createdAt`, `receiptNumber`, and `transactionDetail.quantity` |
| Multiple transaction issues | Multiple non-paid transactions within a short period of time between entry and exit | MySQL `transaction.status != paid`, `transaction.createdAt`, entry trigger time, and exit trigger time |
| Multiple minus button alert | Alert method | MySQL `tds_thief_alert.method = Kiosk` |
| Multiple minus button alert | Alert detail | MySQL `tds_thief_alert.detail`, contains minus-button signal |
| Multiple minus button alert | Alert timestamp | MySQL `tds_thief_alert.createdAt` |
| Multiple minus button alert | Alert location | MySQL `tds_thief_alert.locationId` |
| Carry item signal | Carry score | Gemini result |
| Carry item signal | Before yellow bag flag | Gemini result |
| Carry item signal | After yellow bag flag | Gemini result |
| Unusual group size | Total customer count in group | RunPod grouping result |
| Customer credibility history | Customer identity | `phone_number_id` / `credit_card_id` from trigger |
| Customer credibility history | Previous sessions | MySQL `tds_session`, linked by phone/card identity |
| Customer credibility history | Previous detection result | MySQL `tds_session.status` / `tds_kiosk_video_result` |
| Analysis cost | Estimated script cost | MySQL `tds_script_run.estimated_cost` |
| Analysis cost | Cost currency | MySQL `tds_script_run.cost_currency` |

## Implemented Decision Rules

The current Layer 0 confidence logic uses an any-hit rule: if any enabled factor hits, the grouped triggers are promoted to full-video retrieval for deeper analysis.

| Factor | Hit Condition |
|---|---|
| Long stay + low purchase | Duration between entry and exit is at least `THEFT_API_FILTER_LONG_STAY_SECONDS` and the paid receipt count/value/quantity is low. |
| Transaction issue + low purchase | At least one failed/pending transaction happened before the final paid receipt, and the final paid receipt quantity/value is low. |
| Multiple transaction issues | At least two non-paid transactions happened within `THEFT_API_FILTER_TRANSACTION_ISSUE_SHORT_PERIOD_SECONDS` between entry and exit. |
| Multiple minus button alert | A MySQL `tds_thief_alert` row exists in the entry/exit window where `method = Kiosk` and `detail` contains `minus`. |
| Carry item signal | Gemini carry score is at least `THEFT_API_FILTER_CARRY_SCORE_THRESHOLD`, or yellow bag is absent before and present after. |
| Unusual group size | RunPod grouping `total_customer` is greater than `THEFT_API_FILTER_UNUSUAL_GROUP_SIZE`. |
| Customer credibility history | Visible in settings but intentionally disabled until identity history is implemented. |
