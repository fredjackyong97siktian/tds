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
