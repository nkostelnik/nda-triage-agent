# Eval results (20261010-223149)

Labels: **confirmed**. Mode: classification only. 14 synthetic NDAs, 126 rule verdicts.

Labels not yet confirmed by the attorney: 14_discloser_with_carveout.

| Metric | Result |
|---|---|
| Overall verdict accuracy | 100% (14/14) |
| Overall false GREENs | 0 |
| Rule verdict accuracy | 100% |
| Rule false GREENs (dangerous) | 0 |
| Rule under-calls (RED as YELLOW) | 0 |
| Rule over-escalations | 0 |
| Posture cross-check accuracy | 100% |
| Unverified quotes | 0 |
| Rules marked needs_review | 0 |
| API calls / estimated cost | 140 / $0.50 |

## Per rule

| Rule | Match | False GREEN | Under-call | Over-escalation |
|---|---|---|---|---|
| mutuality | 14 | 0 | 0 | 0 |
| term | 14 | 0 | 0 | 0 |
| survival | 14 | 0 | 0 | 0 |
| non_solicit | 14 | 0 | 0 | 0 |
| non_compete | 14 | 0 | 0 | 0 |
| ip_assignment | 14 | 0 | 0 | 0 |
| residuals | 14 | 0 | 0 | 0 |
| indemnification | 14 | 0 | 0 | 0 |
| ci_definition | 14 | 0 | 0 | 0 |

## Per case

| Case | Expected | Predicted | Rule misses |
|---|---|---|---|
| 01_clean_mutual | GREEN | GREEN | none |
| 02_one_way_recipient | GREEN | GREEN | none |
| 03_discloser_short_terms | RED | RED | none |
| 04_buried_no_hire | RED | RED | none |
| 05_disguised_non_compete | RED | RED | none |
| 06_fake_mutual | RED | RED | none |
| 07_residuals_recipient | GREEN | GREEN | none |
| 08_residuals_mutual | RED | RED | none |
| 09_indemnity_and_improvements | RED | RED | none |
| 10_evergreen_no_exit | RED | RED | none |
| 11_missing_exclusion | YELLOW | YELLOW | none |
| 12_standstill | RED | RED | none |
| 13_scary_but_green | GREEN | GREEN | none |
| 14_discloser_with_carveout | YELLOW | YELLOW | none |
