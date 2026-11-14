# Results of the v0.3 confirmatory study (V3_PREREG; report of 2026-11-14)

Release `release/v0.3.0`, commit f8f51b8 (clean checkout on every machine). Plan frozen before any v0.3 data
(`prereg/V3_FREEZE.sha256`: **match**; it also re-lists the v0.2 MAIN_FREEZE hashes, all unchanged). Bundle
`crossleak_v0.3.0_20261114-185432_all-v3.zip`; its `RESULT.json` lists the sha256 of the five machine artifacts. Every
number below comes from `crossleak main report` with `configs/v3.yaml` (the v0.3 report, `analysis/v3_report.py`) or
from the scripts in `docs/reanalysis/` (independent re-analysis and two post hoc analyses, labelled).

## Runs
- 79 / 79 cells ok; 0 failed, 0 infeasible, 0 missing; 0 results under another configuration. Smoke run COMPLETE.

- Main phase 12:10–18:54 on five VMs (6.7 h; plan 4.5 h). No cost phase (cost was measured in v0.2). Merged report 6.3 min.

## Verdicts (pre-registered rules, `prereg/V3_PREREG.md` section 4)
| Rule | Verdict | Estimate (c2base unless stated) |
|---|---|---|
| E1 ELD: D1 (k = 16) ≡ cross-learning (k = 16) | **EQUIVALENT** | Δgain +0.06 pp, 90 % CI [−0.40, +0.56] |
| E1 ELD: D4 (P = 8, ε = 1) ≡ cross-learning | **EQUIVALENT** (close to the margin) | +0.63 pp [−0.54, **+1.97**] (margin ±2) |
| E1 ELD: D4 ≡ D1 | **EQUIVALENT** | +0.56 pp [−0.52, +1.82] |
| E1 LifeSnaps: D1 (k = 4) ≡ cross-learning (k = 8) | **EQUIVALENT** | −0.40 pp [−1.18, +0.37] |
| E1 LifeSnaps: D4 ≡ cross-learning | NOT_EQUIVALENT (D4 **better**) | +2.02 pp [+0.20, +3.61] |
| E2 ELD, 3 days, k = 4: no benefit | **EQUIVALENT_NO_BENEFIT** | gain +0.51 % [−0.19, +1.47] |
| E2 ELD, 3 days, k = 16: no benefit | BENEFIT_OR_HARM | gain +1.15 % [+0.32, +2.12] (a small benefit) |
| **S1 the stronger sybil attacker fails at k = 256** (c2small, ELD) | **REFUTED** | AUC L1 0.597 [0.542, 0.650] vs control 0.500; ΔAUC +0.097 [+0.035, +0.161]; Δr1 +0.127 [+0.085, +0.170] |

Readings:
- **E1.** The cold-start benefit does not need private co-tenants: public context is equivalent to cross-learning on
  both datasets (TOST, 600 targets), and DP-Prototype at ε = 1 is equivalent on ELD. The D4-vs-cross-learning upper bound
  on ELD (1.97 pp) is close to the margin; it is stable across three bootstrap seeds and 5,000 resamples (1.95–1.96 pp,
  independent re-analysis). On LifeSnaps D4 is better than cross-learning; see the post hoc analysis below for why.
- **E2.** With three days of history the benefit is at most about one point (k = 4: equivalent to zero; k = 16: +1.15 %).
  This replaces the mislabelled CL1b of v0.2.
- **S1 (refuted).** With residual features and 1,500–2,000 training groups, an attacker that fills a batch of 256 with
  copies of its probe still infers the victim's previous-day evening routine on ELD (AUC 0.60) and reconstructs its
  profile beyond the population (Δr1 +0.13). This supersedes the v0.2 CL5 = NO, which used 800 training groups and raw
  features; v0.2's verdict is still reported as computed. On LifeSnaps the label-level leak at k = 256 is not detected
  (ΔAUC +0.025 [−0.038, +0.105]) but the profile leak is (Δr1 +0.072 [+0.043, +0.112]).

## Stronger sybil attacker across k (residual features; 95 % CI by victim)
| Dataset | Model | Configuration | k | n_train | AUC L1 | control | ΔAUC L1 | Δr1 |
|---|---|---|---|---|---|---|---|---|
| ELD | c2base | single (pair) | 2 | 1,500 | 0.724 | 0.515 | +0.209 [0.150, 0.263] | +0.222 |
| ELD | c2base | sybil, copy | 16 | 2,000 | 0.684 | 0.533 | +0.152 [0.095, 0.210] | +0.217 |
| ELD | c2small | single (pair) | 2 | 1,500 | 0.788 | 0.501 | +0.287 [0.226, 0.349] | +0.380 |
| ELD | c2small | sybil, copy | 16 | 2,000 | 0.690 | 0.470 | +0.220 [0.167, 0.276] | +0.251 |
| ELD | c2small | sybil, copy | 64 | 2,000 | 0.602 | 0.548 | +0.054 [−0.017, 0.123] | +0.118 [0.078, 0.160] |
| ELD | c2small | sybil, copy | 256 | 1,500 | **0.597** | 0.500 | **+0.097 [0.035, 0.161]** | **+0.127 [0.085, 0.170]** |
| LifeSnaps | c2base | single (pair) | 2 | 1,500 | 0.669 | 0.536 | +0.133 [0.091, 0.190] | +0.214 |
| LifeSnaps | c2small | sybil, copy | 16 | 2,000 | 0.610 | 0.533 | +0.077 [−0.013, 0.139] | +0.160 |
| LifeSnaps | c2small | sybil, copy | 64 | 2,000 | 0.605 | 0.497 | +0.108 [0.015, 0.182] | +0.108 |
| LifeSnaps | c2small | sybil, copy | 256 | 1,500 | 0.537 | 0.512 | +0.025 [−0.038, 0.105] | +0.072 [0.043, 0.112] |

The v0.2-style attacker (raw features, linear) on the same v0.3 data also detects the k = 256 ELD leak, just
(ΔAUC +0.071 [+0.001, +0.138]), so the larger training set alone is enough to change the v0.2 verdict; the residual
features add the profile leak (Δr1 +0.127 vs +0.003). Channel ratio vs the pair at k = 256 (ELD): single member 0.012
(c2small) / 0.018 (c2base); sybil copies 0.159 / 0.122.

## Utility replication (fresh seed, 600 targets; gain vs isolated, 95 % CI)
ELD, c2base, 1 day: cross-learning k = 4 +8.2 %, k = 16 +9.7 % [7.1, 12.2]; D1 k = 16 +9.7 %; D4 ε = 1 +10.3 %, ε = 4
+11.4 % (v0.2: 10.7 / 10.8 / 11.2 / 11.6 %). LifeSnaps, c2base: cross-learning +1.0 / +1.4 %, D1 +1.0 %, D4 ε = 1 +3.4 %;
c2small: +3.7 / +4.3 %, D1 +4.2 %, D4 +3.7 %. Pair leakage replicates (ELD AUC 0.724, LifeSnaps 0.669).

## Post hoc analyses (labelled; not pre-registered)
1. **Seasonal naive** (repeat the last 24 h; its WQL equals its WAPE), paired on the same targets
   (`docs/reanalysis/snaive_baseline.py`). ELD with one day of history: seasonal naive beats isolated Chronos-2 by
   +14.6 % [6.9, 23.3] in v0.3 (+3.3 % [−11.3, +15.2] in v0.2). No Chronos-2 configuration beats it significantly: on
   c2base cross-learning k = 16 minus seasonal naive is −4.9 pp [−11.5, +0.8] in v0.3 and +7.4 pp [−3.2, +20.4] in v0.2;
   on c2small every configuration is significantly worse in v0.3 (−10 to −11 pp). With three days of history Chronos-2
   beats seasonal naive by ≈ 30 pp; on LifeSnaps Chronos-2 beats it by 27–36 pp at one day. Reading: the ELD cold-start
   benefit is relative to isolated Chronos-2, which is itself not better than repeating yesterday for a day-one client.
2. **D4 on LifeSnaps uses the public centres** (`docs/reanalysis/d4_fallback_share.py`, releases recomputed exactly).
   Prototypes on the public-centre fallback: ELD 15.6 % (ε = 1) / 10.2 % (ε = 4) with a median of 229 service users per
   release; LifeSnaps 51.2 % / 46.4 % with a median of 9. The LifeSnaps D4 gain therefore comes mostly from the public
   centres (a public prior), not from private data.

## Independent re-analysis (`docs/reanalysis/v3_reanalysis_v0.3.0.py`)
Own code (no project analysis functions): TOST with its own bootstrap (three seeds × 5,000 resamples), labels recomputed
from the raw panels (agreement with the pipeline 100 % ELD, 95 % LifeSnaps), own logistic and ridge decoders with
5-fold CV. Same verdicts everywhere; ELD k = 256 sybil ΔAUC +0.102 [+0.034, +0.167] and Δr1 +0.128 (pipeline +0.097,
+0.127).

## Deviations and limitations
- None from the frozen plan. The prereg text of S1 mentions "90 %/95 %"; the report uses 95 % bootstrap CIs, under which
  S1 is refuted (lower bound +0.035 > 0); a 90 % interval would only widen the margin.
- Targets come from 49 cold-start users on ELD and 13 on LifeSnaps; the LifeSnaps equivalence rests on 13 clusters.
- The sybil experiment uses copy fillers and one read-out probe; the k = 256 leak is shown on c2small and ELD only.

## What the paper may claim (combined with docs/MAIN_RESULTS.md)
- May: default cross-learning leaks a co-tenant's previous-day routine and profile (pairs, both datasets, both models,
  replicated); a single member's leak vanishes by k = 4–16, but an attacker that fills the batch keeps it at k = 16
  (both datasets) and at the default batch size of 256 on ELD (small model, AUC 0.60); the channel and the defaults exist
  in the shipped AutoGluon predictor (docs/AGCHECK.md); public context is equivalent to cross-learning (2 pp margin) on both
  datasets and DP-Prototype at ε = 1 per day is equivalent on ELD; with three days of history the benefit is ≤ ≈ 1 point.
- May not: that cross-learning (or D1/D4) beats a seasonal-naive forecast for day-one ELD clients; that D4's LifeSnaps
  advantage comes from private data; a k = 256 leak on LifeSnaps at the label level or on the base model.
