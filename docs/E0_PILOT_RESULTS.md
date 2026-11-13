# E0 pilot results (data of 13 Nov 2026, code v0.1.2)

Exploratory pilot (prereg/E0_PREREG.md section 1): it decides GO/STOP and plans the main study; it supports no paper claim.

## Run
- 5 VMs (shared i9-12900H host), 15:49–22:18 (6.5 h); 126/126 planned cells, 0 failed, 0 infeasible; freeze hashes match.
- Bundle `crossleak_v0.1.2_20261113-221812_all-e0.zip` (RESULT.json status ok; all shard exit codes 0).

## Pre-registered verdicts
| Dataset | Gain (c2base, ell <= 72) | Channel (k in {2, 4}) |
|---|---|---|
| ELD | YES (mixed, ell = 24: +7.5 % to +9.7 %, CI > 0) | YES |
| LifeSnaps | WEAK (max +3.4 %) | YES |

**Overall: GO.** Invariance OK (max change 4.2e-6).

## Independent re-analysis (separate code: 5-fold CV ridge/logistic instead of GCV/CV-AUC, own bootstrap)
| Cell | delta r report / re-analysis | delta AUC report / re-analysis |
|---|---|---|
| ELD c2base k=2 noise | 0.324 / 0.324 | 0.252 / 0.249 |
| ELD c2base k=2 short | 0.258 / 0.232 | 0.208 / 0.212 |
| ELD c2base k=2 honest | 0.137 / 0.112 | 0.164 / 0.162 |
| ELD c2base k=4 noise | 0.033 / 0.039 | 0.074 / 0.073 |
| LifeSnaps c2base k=2 noise | 0.256 / 0.256 | 0.206 / 0.202 |
| LifeSnaps c2base k=2 short | 0.261 / 0.259 | 0.192 / 0.215 |
| LifeSnaps c2base k=2 honest | 0.094 / 0.102 | 0.131 / 0.132 |
| LifeSnaps c2small k=4 short | 0.051 / 0.048 | 0.002 / 0.011 |
Gain cells recomputed from per-item arrays match to three decimals. No verdict changes.

## Findings that shape the main study
1. **Gain is narrow.** Cross-learning helps a cold-start user with one day of history among established co-tenants
   (ELD c2base mixed ell = 24: +7.5 % (k = 4) to +9.7 % (k = 64); 85–91 % of targets improve), but recovers only 17–22 %
   of the cold-start penalty (full-context WQL 0.055 vs 0.100 isolated). With >= 3 days of history it is neutral or
   **harmful** (ELD allshort ell = 72: -5 % to -7.5 %, CI < 0). LifeSnaps: +2 % to +3.4 % throughout.
   The small model gains less (ELD mixed ell = 24: +1.5 % to +3.5 %).
2. **Leakage is strong in pairs.** At k = 2 the attacker's own forecast reveals the co-tenant's previous day:
   ELD AUC 0.71–0.80 (control 0.52–0.56), anomaly r 0.32–0.58 (control 0.11–0.15); LifeSnaps AUC 0.64–0.72.
   Uninformative probes (noise, 24 h) leak more than an honest 14-day probe, as predicted.
3. **Per-victim leakage falls fast with group size.** At k = 4 a single-member linear attacker reaches AUC <= 0.63,
   delta r <= 0.11; swap influence drops roughly as 1/(k-1) (ELD c2base short: 0.028, 0.020, 0.007 at k = 2, 4, 16).
4. **Tension:** gain grows with k while per-victim leakage shrinks. Large groups therefore look like a free defence,
   unless the adversary controls most of the group (sybil / n-1 flooding: AutoGluon forms batches of 256 consecutive
   items). This is now the decisive experiment of the main study.
5. **Cost.** Under 5-VM contention: ~45–70 ms/series (small), ~180–230 ms/series (base). Decoder fitting, not
   forecasting, dominated the decode cells (median 44 min small, 69 min base); in the main study features are stored
   and decoders are fitted in the analysis stage.

## Pilot-based adjustments proposed for the main plan
- CL1 scoped to "one day of history"; report the harm at >= 3 days and the LifeSnaps magnitude honestly.
- CL3/CL4 thresholds per k; add the sybil attacker (m = k-1 and m = k/2 for k in {4, 16, 64, 256}) and a nonlinear
  decoder; group-level inference for k > 2 single-member attackers.
- Defences: isolation, public-context (D1), large groups (D2, expected to fail against sybils), DP-Prototype (D4), and
  a history-length rule (cross-learning only for <= 1 day of history).
- Edge-cost experiment (E7) on one active VM only.
