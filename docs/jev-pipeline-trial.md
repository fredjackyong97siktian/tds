# Jev Adjacent And Carry Trial

## Scope And Rollback

Adjacent matching and carry analysis now default to Jev 1.14 via OpenRouter.
Both have independent runtime app settings, `adjacent_provider` and
`carry_provider`, accepting `jev` or `inherit`. No schema migration is needed.
The existing `grouping_provider` still controls direct, repair, generic
verification and the legacy carry path. Kiosk analysis is unchanged.

In TDS Vision > Pipeline Settings, choose **Use Grouping Model (previous
behavior)** for either stage to revert it. The existing selected grouping
provider (for example MiMo) is retained; changing these overrides does not
overwrite it. A grouping batch pins the adjacent override for that stage.
An already running carry request is not cancelled by changing the setting.

The API/worker environment needs `THEFT_API_OPENROUTER_API_KEY` or
`OPENROUTER_API_KEY`. No key is committed or copied from chat into config.
Missing credentials or failed Jev calls never silently fall back to another
paid model. Adjacent fails visibly; carry is recorded as unclear/failed.

## Adjacent

One Choice per independent entry, using its ordered frames and up to two
later candidates. Outputs: candidate_1, candidate_2, both (only with two
candidates), or none. Both means both candidate sequences show the matched
customer leaving. Any visible matching customer in a multi-person sequence
may count; presence of unrelated people alone is insufficient.

Both exits are attached to ONE group and marked consumed. An exit cannot be
claimed by another adjacent group. If either exit of a both answer is already
consumed, reject the whole proposed match rather than silently changing it.
The normal grouping persistence and confidence pipeline consume the complete
exit array. A retry checkpoint cannot be reused across a provider switch.

Jev's trial minimum Choice confidence is 0.5, separate from the legacy
MiMo similarity cutoff. It is a provisional screening threshold, not a
calibrated accuracy claim. Raw selected option, confidence and full option
probabilities are kept in the group and Script Run for evaluation.

Jev does not fabricate carry descriptions, appearance prose or best-frame
flags. Existing fail-open behavior keeps available frames for remaining direct
checks; no presence response means no person-absence exclusion. Generic
verification can consequently use more frames than with MiMo's curated picks.

## Carry

One Jev request sends stored entry frames, frames from EVERY grouped exit,
and receipt details. It does not start with MiMo's bag descriptions and does
not automatically retrieve kiosk/exit probe frames on an unclear Jev result.
Uses the current grouping frame-per-trigger cap and JPEG preparation from the
matching experiments (maximum 1024px side, minimum 280px padded dimensions).

Two independent Choices share this evidence: carry status and reason code.
Reason codes supply fixed explanatory labels, not invented visual prose.
If they conflict, the reason cannot override the status and a generic status
message is used. Low Choice confidence becomes insufficient_evidence while
preserving the raw answer. Missing images and API/validation failures are
also unclear, never silently cleared or treated as confirmed theft.

Status mapping in BOTH scheduled and manual confidence scoring:

| Status | Carry factor hit |
| --- | --- |
| suspicious | true |
| reasonable | false |
| insufficient_evidence | false; displayed as unclear |

The Jev path does not apply the legacy insufficient-evidence + low-purchase
escalation. Reverting to inherit restores the legacy rules along with its
text-first call and possible probe-image re-examination.

Suspicious criteria mirror the existing policy: unexplained newly carried
goods, folded/scrunched bags visibly becoming fuller, and a visibly carried
yellow bag/container belonging to the customer. Yellow clothing/background
objects, a pre-existing bag, or no receipt alone are not evidence of theft.
Receipts currently come from the visit time window, not verified customer
ownership. This limitation is explicitly sent to Jev; if ownership is needed
to judge and uncertain, it must choose insufficient_evidence. Bag contents
and exact unseen item counts must not be guessed. A suspicious decision is
a screening flag for L1, not a theft confirmation.

Other enabled factors still run in the established cheap-first order and can
short-circuit before carry, so there is no Jev carry charge for those cases.

## Monitoring

Script names stay `grouping` and `carry_confidence`, avoiding constraint
changes. Model labels identify Jev adjacent and carry separately. Requests,
responses/errors, image URLs and probabilities are logged without duplicate
base64 images. OpenRouter's actual `usage.cost` is recorded before typed
response validation; missing cost remains unknown. No new price table is used.

Test known matches/non-matches, duplicate exits, similar clothing, changed
bags and unclear receipt attribution before judging production accuracy.

## Motion-Sensor Presence

Pipeline Settings includes a separate Sensor Customer Presence selector:
Jev 1.14 or Use Grouping Model. The default remains Use Grouping Model;
selecting Jev here does not change direct grouping, adjacent matching,
carry analysis, or kiosk analysis.

Each uncached motion-sensor presence check creates a Script Run before the
API call. To remain compatible with existing MySQL script-name constraints,
it uses script_name `grouping`, a model label ending in `_person_presence`,
and runner_payload.stage `person_presence`. Trigger ID, per-attempt responses,
errors, and reported/estimated provider costs use the existing monitoring path.
An API failure is recorded as failed and keeps the trigger eligible. Jev
unclear/low-confidence decisions also keep it eligible without caching an
absence. Existing cached presence decisions are not automatically rechecked.
