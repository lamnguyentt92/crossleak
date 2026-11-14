# Results of the v0.4 causal-defence study (V4_PREREG; report of 2026-11-14)

Release `release/v0.4.0`, commit b8b19fa (clean checkout on every machine; `git describe --dirty` = b8b19fa). Plan
frozen before any v0.4 data (`prereg/V4_FREEZE.sha256`: **match** for all 13 frozen files, including the unchanged main
driver and analysis). Bundle `crossleak_v0.4.0_20261114-223724_all-v4.zip` (sha256 `d80b7be2d0c9db06…5ca1df1`, 538,731
bytes); its `RESULT.json` lists the sha256 of the five machine artifacts. Every number below comes from
`crossleak main report` with `configs/v4.yaml` (`analysis/v4_report.py`) or from
`docs/reanalysis/v4_reanalysis_v0.4.0.py` (independent re-analysis). No post hoc analysis was run on these data.

## Runs
- 9 / 9 cells ok; 0 failed, 0 infeasible, 0 missing, 0 configuration mismatches. Smoke run (`v4_smoke`) COMPLETE
  before the main run (bundle `crossleak_v0.4.0_20261114-212858_all-smoke.zip`).
- Main phase 21:28:59–22:37:10 on five VMs (1.1 h; plan about 1 h). Merged report 0.2 min.
- Operations: the monitoring dashboard (`scripts/watch.sh` without `--config`) stopped with a `NameError`; the campaign
  was not affected. Fixed in release v0.4.2 (dashboard only).

## Samples
| Dataset | Model | Targets | Cold-start users | Distinct (user, day) targets | Feasible origin days |
|---|---|---|---|---|---|
| ELD | base | 600 | 49 | 589 | 336 |
| ELD | small | 600 | 49 | 592 | 336 |
| LifeSnaps | base | 600 | **6** | 172 | 64 |
| LifeSnaps | small | 600 | **6** | 173 | 64 |

The origin rule of the plan (section 2: days on which the target, its co-tenants, past public windows at least 56 days
earlier and, on LifeSnaps, past centres all exist) leaves only 6 of the LifeSnaps cold-start users with an origin. The
plan's section 3 expected 13 users (the v0.3 count). The procedure followed the plan; the smaller sample is an
observation, logged in DECISIONS.md (Deviations log) and stated in the paper. Targets are drawn with replacement from
the feasible (user, day) pairs, so the LifeSnaps intervals rest on 6 user clusters.

## Verdicts (pre-registered rules, `prereg/V4_PREREG.md` section 3; base model, one day of history; TOST, 90 % CI, margin ±2 pp)
| Rule | Pair | Verdict | Δgain (pp) [90 % CI] | v0.3 counterpart |
|---|---|---|---|---|
| E1c-1 ELD | D1 past-only (k = 16) vs cross-learning (k = 16) | **EQUIVALENT** | +0.21 [−0.16, +0.56] | D1 vs XL: +0.06 [−0.40, +0.56], EQUIVALENT |
| E1c-2 ELD | D4 (P = 8, ε = 1) vs cross-learning (k = 16) | **NOT_EQUIVALENT** | −1.40 [−2.34, −0.39] | +0.63 [−0.54, +1.97], EQUIVALENT |
| E1c-3 ELD | D4 (ε = 1) vs D1 past-only | **NOT_EQUIVALENT** | −1.61 [−2.56, −0.59] | +0.56 [−0.52, +1.82], EQUIVALENT |
| E1c-4 LifeSnaps | D1 past-only (k = 4) vs cross-learning (k = 8) | **EQUIVALENT** | −0.74 [−1.60, −0.10] | −0.40 [−1.18, +0.37], EQUIVALENT |
| E1c-5 LifeSnaps | D4 past centres (ε = 1) vs cross-learning (k = 8) | NOT_EQUIVALENT | −0.42 [−2.31, +0.87] | D4: +2.02 [+0.20, +3.61], NOT_EQUIVALENT (better) |

## Estimands (section 4; no hypothesis)
Effect of restricting public data to the past, paired on the same targets (base model, 90 % CI):

| Dataset | Causal − original | Δgain (pp) [90 % CI] |
|---|---|---|
| ELD | D1 past-only − D1 (k = 16) | +0.37 [−0.31, +1.26] |
| LifeSnaps | D1 past-only − D1 (k = 4) | 0.00 [−0.31, +0.23] |
| LifeSnaps | D4 past centres − D4 (ε = 1) | −0.14 [−0.26, +0.05] |

Gains over isolated forecasting (%, 95 % CI):

| Configuration | ELD base | ELD small | LifeSnaps base | LifeSnaps small |
|---|---|---|---|---|
| XL k = 4 | 6.4 [4.2, 8.7] | 3.0 [1.7, 4.6] | 1.4 [−0.0, 3.4] | 2.5 [1.5, 3.8] |
| XL k = 16 / 8 | 8.1 [5.2, 10.9] | 1.9 [0.1, 4.0] | 2.2 [0.4, 5.2] | 3.1 [2.0, 4.7] |
| D1 (k = 16 / 4) | 7.9 [4.3, 11.3] | 1.5 [−0.4, 3.5] | 1.5 [0.2, 3.4] | 2.2 [1.3, 3.8] |
| D1 past-only | 8.3 [5.4, 11.2] | 1.4 [−0.4, 3.4] | 1.5 [0.2, 3.4] | 2.2 [1.1, 3.6] |
| D4 ε = 1 | 6.7 [3.8, 9.9] | 1.7 [−0.0, 3.7] | 1.9 [−0.0, 3.9] | 1.3 [−0.5, 3.9] |
| D4 ε = 4 | 7.4 [4.2, 10.7] | 2.4 [0.6, 4.4] | 2.0 [0.4, 4.0] | 1.5 [0.0, 3.9] |
| D4 past centres ε = 1 | – | – | 1.8 [0.1, 3.7] | 1.2 [−0.4, 3.7] |
| D4 past centres ε = 4 | – | – | 2.0 [0.3, 3.8] | 1.7 [0.1, 3.9] |
| Full history | 55.0 [49.5, 60.0] | 50.9 [43.2, 58.3] | 15.9 [9.6, 24.8] | 13.1 [7.3, 20.6] |
| Seasonal naive | 5.5 [−4.4, 15.7] | 6.8 [−0.9, 15.6] | −34.2 [−48.5, −23.4] | −40.7 [−54.5, −27.4] |

## Readings (interpretation fixed in `prereg/V4_PREREG.md` section 5)
- **E1c-1 EQUIVALENT.** Public context restricted to windows at least 56 days *before* the origin is equivalent to
  cross-learning on ELD (base model, one day of history). The paper states this. Restricting the windows changed the gain
  by +0.37 pp [−0.31, +1.26] on the same targets: the non-causal windows of v0.2/v0.3 did not inflate D1.
- **E1c-2/3 NOT_EQUIVALENT.** The v0.3 ELD result "D4 ≡ cross-learning" did not replicate on fresh targets. The ELD D4
  configuration is identical in v0.3 and v0.4 (centres from 2013), so the two campaigns differ only in their targets and
  origins (v0.4 origins need past public windows). Per section 5 the D4 claim is narrowed to the measured difference:
  D4 at ε = 1 keeps most of the cross-learning gain on ELD (6.7 % vs 8.1 % in v0.4); its difference to cross-learning lies
  within −2.34 to +1.97 pp across the two campaigns' 90 % intervals; equivalence within 2 pp is **not** claimed. The main
  study's CL6 rule (D4 keeps ≥ 50 % of the gain) is unaffected.
- **E1c-4 EQUIVALENT, E1c-5 NOT_EQUIVALENT** — reported as computed, with the caveat of section 3, now stronger: 6
  cold-start users. Past-only public context is within the margin but below cross-learning (interval below zero), and
  the cross-learning gain itself (2.2 %) is about the size of the margin. The v0.3 LifeSnaps D4 advantage (+2.02 pp) did not reappear (−0.42 pp, interval wide); restricting
  the centres to the past changed the gain by −0.14 pp [−0.26, +0.05] on the same targets, and D4 with the original
  centres was not better either (gain 1.9 % vs 2.2 % for cross-learning), so the restriction is not the reason.
- The v0.3 E1 verdicts remain reported; for a deployable (causal) defence the v0.4 verdicts take precedence.

## Independent re-analysis
`docs/reanalysis/v4_reanalysis_v0.4.0.py` (own pooling of the per-target arrays, own user-cluster bootstrap, 5,000
resamples, three seeds; does not import `crossleak`) → `docs/reanalysis/v4_reanalysis_v0.4.0.out`. Point estimates
identical to the report; 90 % bounds within 0.07 pp of the report's (bootstrap noise); all five verdicts agree under all three seeds.

## Evaluation type
Utility replay of real recorded series through the public models offline (`real_system` data, as in v0.2/v0.3); the
grouping service is the project's own code.

## Scope of the conclusions
- May: public context (also past-only) is equivalent to cross-learning within 2 pp on ELD in two campaigns (base model,
  one day of history), and on LifeSnaps in two campaigns with few users (13 and 6) and a margin about as large as the gain;
  restricting public data to the past had no measurable cost; DP-Prototype at ε = 1 keeps most of the gain on ELD,
  within −2.34 to +1.97 pp of cross-learning across two campaigns.
- May not: DP-Prototype is equivalent to cross-learning (it was in one campaign, not in the other); DP-Prototype is
  better than cross-learning on LifeSnaps.
