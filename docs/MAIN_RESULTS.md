# Results of the main study (MAIN_PREREG v0.2.0; report of 2026-11-14)

> Update (v0.3, `docs/V3_RESULTS.md`): CL5 = NO below is superseded by the v0.3 confirmatory test S1 (a stronger
> sybil attacker leaks at k = 256 on ELD), and CL1b is replaced by the v0.3 equivalence test E2. A post hoc
> seasonal-naive comparison (`docs/reanalysis/snaive_baseline.py`) qualifies CL1 on ELD. The v0.2 verdicts below are
> unchanged and reported as computed.

Release `release/v0.2.0`, commit b96cdcb (clean checkout on every machine). Plan frozen 2026-11-13T16:59Z
(`prereg/MAIN_FREEZE.sha256`: **match**; E0 freeze: match). Bundle `crossleak_v0.2.0_20261114-094533_all-main.zip`;
its `RESULT.json` lists the sha256 of the five machine artifacts. Every number below comes from
`crossleak main report` (linear decoders in the bundle; `--decoders full` re-run offline) or from the independent
re-analysis in `docs/reanalysis/`.

## Runs
- 168 / 168 cells ok; 0 failed, 0 infeasible, 0 missing; 0 results under another configuration. Invalid-cell rule
  (≤ 10 %): met.
- Main phase 00:27–09:34 on five VMs (9.1 h; plan 6.8 h: c2small cells took 1.35×, c2base 1.15× the estimate); cost
  phase 11 min on VM-A alone; merged report 12 min.

## Verdicts (pre-registered rules, section 5 of the plan)
| Rule | ELD | LifeSnaps | Estimate (c2base unless stated; 95 % CI) |
|---|---|---|---|
| CL1 gain, 1 day of history | **YES** | WEAK | ELD +9.0 % (k = 4) [7.2, 10.6], +10.7 % (k = 16), +11.2 % (k = 64) [8.7, 13.4]; LifeSnaps +1.4 % (k = 4), +2.5 % (k = 8) [1.0, 4.3] |
| CL1b no benefit, 3 days | BENEFIT* | BENEFIT | ELD −1.4 % [−5.8, 2.2], −0.3 % [−4.4, 3.2]; LifeSnaps +2.1 % [0.9, 3.4], +2.7 % [1.4, 4.4] |
| CL2 channel ≫ noise, invariance | **YES** | **YES** | influence 0.030 / 0.040 vs invariance error ≤ 5.3e-6 |
| CL3 L1 inference, pair | **YES** | **YES** | AUC 0.717 [0.671, 0.761] (ctrl 0.508), ΔAUC 0.209 [0.158, 0.258]; LS 0.722 [0.682, 0.754], ΔAUC 0.198 |
| CL3-k4 | NOT_DETECTED | PERSISTS | ΔAUC 0.038 [−0.005, 0.081]; LS 0.023 [0.003, 0.036] |
| CL4 profile, pair | **YES** | **YES** | Δr1 0.224 [0.179, 0.268]; LS 0.230 [0.204, 0.268] |
| CL5 sybils defeat large groups (decisive) | **NO** | **NO** | see below |
| CL5-base sybil, k = 16 | PERSISTS | PERSISTS | ΔAUC 0.196 [0.143, 0.248], AUC 0.690; LS ΔAUC 0.149, AUC 0.671 |
| CL6 D4 keeps ≥ 50 % at ε ≤ 4 | **YES at ε = 1** | NO | ELD P = 8, ε = 1: gain +11.2 % [8.9, 13.2], retained 1.05 [0.97, 1.16]; LS CIs include 0 |
| CL8 1,000 users ≤ 15 min | **YES** | – | 469 s (P = 8, c2base, one 4-vCPU VM) |
| CL9 linkage N = 10 | **YES** | NO | top-1 0.167 [0.136, 0.198] (chance 0.10); LS 0.166 [0.086, 0.226] |
| DP audit | CONSISTENT | CONSISTENT | no AUC above the ceiling |

\* CL1b: the rule says NO_BENEFIT only if every k has an adjusted upper bound below 2 %. For ELD the point estimates are
≤ 0 but the bounds are 2.2–3.2 %, so the label "BENEFIT" here means **equivalence not established**, not a benefit shown.
For LifeSnaps a small benefit at 3 days is shown (CI > 0). The label wording of this rule was a design flaw of the plan;
the verdict is reported as computed, with this reading.

### CL5 in detail (the decisive experiment)
| Group | Single member (one attacker) | Sybil, copy fillers (attacker holds k − 1) |
|---|---|---|
| Channel ratio vs pair, c2small ELD | k = 16: 0.17, k = 64: 0.042, k = 256: **0.010** [0.007, 0.013] | k = 16: 0.45, k = 64: 0.27, k = 256: **0.27** [0.16, 0.36] |
| Channel ratio, c2base ELD | k = 256: 0.013 | k = 256: 0.073 [0.048, 0.152] |
| AUC L1, c2base ELD | k = 4: 0.491, k = 16: 0.496 | k = 4: 0.728, k = 16: **0.690** |
| AUC L1, c2small ELD | k = 16: 0.545 | k = 16: 0.687, k = 64: 0.584 (ΔAUC 0.075 [0.036, 0.112]), k = 256: **0.533** |
| AUC L1, c2small LifeSnaps | – | k = 16: 0.640, k = 64: 0.558, k = 256: **0.557** (ΔAUC 0.024 [−0.024, 0.066]) |

Rule part (a), dilution of a single member, holds for ELD (ratio 0.010 at k = 256) but not for LifeSnaps, where the
largest feasible single-member group is k = 8 (ratio 0.31). Part (b) fails on both datasets: at k = 256 the sybil
attacker's AUC is 0.533 / 0.557 (< 0.60). The ELD ΔAUC of 0.122 at k = 256 is inflated by a control decoder at 0.410;
below-chance controls (0.41–0.45) occur in some ELD cells, where the 2013 → 2014 calendar shift misleads the
no-channel decoder, so absolute AUCs are the safer reading. Other fillers (estimands): `sparse` was weaker than `copy`
(k = 16 AUC 0.608; k = 256 0.512), contrary to the synthetic mechanism check; `noise` fillers gave no leakage
(0.51 / 0.50); half sybils (m = k/2) gave 0.555 (k = 16) and 0.537 (k = 64).

Reading: a group of 16–64 does not protect against an attacker that fills it; at the default batch size of 256 our
attacker (one read-out probe, linear decoder, 800 training groups) recovers almost nothing, although its probe's
output still carries 27 % (small) / 7 % (base) of the pair-level victim influence.

### Defences on the same targets (ELD, c2base, 1 day of history; CL6, CL7)
| Configuration | Gain | Retained vs cross-learning k = 16 |
|---|---|---|
| Cross-learning k = 16 (private co-tenants) | +10.7 % [8.3, 12.7] | 1 |
| D1 public context k = 16 (public users, same weekday, ≥ 56 days away) | +10.8 % [8.4, 13.0] | 1.02 [0.95, 1.09] |
| D4 DP-Prototype P = 8, ε = 0.5 / 1 / 4 / ∞ per day | +10.0 / +11.2 / +11.6 / +12.7 % | 0.93 / 1.05 / 1.09 / 1.19 |
| D4 P = 4, ε = 1 | +10.9 % | 1.03 [0.94, 1.15] |

Paired differences (independent re-analysis): D4 (P = 8, ε = 1) − D1: +0.3 points [−0.5, +1.3]; D1 − cross-learning:
+0.2 [−0.5, +0.9]; D4 ε = 4 − D1: +0.8 [−0.1, +1.8]. The cold-start benefit does not need private co-tenants: stale
public series or DP prototypes at ε = 1 give the same gain (estimands; no equivalence hypothesis was pre-registered).
LifeSnaps (13 cold-start users, ~6 established users a day): D1 +1.7 % [0.5, 3.0]; D4 CIs include 0.
c2small: cross-learning +4.7 % (k = 4); D4 retains 0.66–0.97 at ε = 1–4.

### Other estimands
- Probes (pairs): ELD c2base `noise` AUC 0.778, `short` 0.717, `honest` 0.622; the small model leaks more (ELD short 0.812).
- Day-level events, pairs, c2base: L5 (idiosyncratic) ΔAUC ELD 0.064 [0.011, 0.114], LifeSnaps 0.113 [0.067, 0.186];
  L4 (self-relative) ELD −0.008 (n.s.), LifeSnaps 0.136 [0.100, 0.184]. L3 (low-activity day) AUC 0.664 (ctrl 0.613).
- Linkage N = 50 (ELD, pairs): top-1 0.035 c2base, 0.098 c2small (chance 0.02).
- Nonlinear decoders (gradient boosting, MLP) were weaker than the linear ones wherever the linear decoder detected
  leakage (e.g. ELD pair 0.532 vs 0.717); they were marginally higher in 3 of 36 configurations, all without meaningful
  leakage (AUC ≤ 0.56). At these sample sizes the linear decoder is the strongest attacker tested. (Corrected on
  2026-11-14: the first version said "everywhere"; full-decoder report in `docs/reanalysis/main_full_decoders/`.)
- DP audit (adversary knowing every other user, sees the release): ELD AUC L1 0.504 / 0.530 / 0.590 at ε = 0.5 / 1 / 4
  (ceilings 0.73 / 0.88 / 1.00); 0.934 without noise (the 14-day mean profile reveals the stable routine).
- Cost (one VM, others idle, 4 threads), ms per user p50: c2base isolated 49 (batched) / 93 (single request); cross-learning
  43–57; D4 P = 4 249, P = 8 469 (≈ (P + 1) × isolated); c2small isolated 10.5, D4 P = 8 110. Release for 1,000 users 0.07 s.

## Independent re-analysis (`docs/reanalysis/`)
Own code (no project analysis functions): labels recomputed from the raw panels (agreement with the pipeline 100 %
ELD, 95–96 % LifeSnaps, where the validity rule was approximated), logistic and ridge decoders with 5-fold CV, own
bootstrap. Differences on the decisive cells: AUC ≤ 0.025, ΔAUC ≤ 0.03, Δr1 ≤ 0.05; gains and channel ratios identical.
No verdict changes.

## Deviations and limitations
- None from the frozen plan. Notes: the CL1b label (above); below-chance controls in some ELD cells (above).
- `peak_rss_mb` of the c2small cost cell is not meaningful: both models ran in one process (base first).
- Sybils were read out through one probe; the k = 256 decoders had 800 training groups (k = 2: 2,000). A stronger
  attacker (more groups, several read-out probes) was not tested.

## Evaluation types
Forecasting, attacks and utility: recorded real series replayed offline through the public models (claims within the
two datasets, the two models and the stated history lengths); cost: one VM as an emulated edge node.

## Scope of the conclusions
- May: one-day cold-start benefit on ELD (+9–11 %), small on LifeSnaps; leakage of a co-tenant's previous-day routine and
  profile in pairs on both datasets (AUC ≈ 0.72), above-chance re-identification on ELD; single-member leakage vanishes
  by k = 4–16; sybils that fill the group restore it at k = 16 on both models and datasets and partly at k = 64 (small
  model); D1 and D4 (ε = 1 per day, user-level DP) keep the full ELD benefit; D4 serves 1,000 users in 7.8 min.
- May not: that large groups fail against sybils at k = 256 (CL5 = NO); a benefit or its absence at 3 days on ELD;
  D4 utility on LifeSnaps; equivalence of D1, D4 and cross-learning as a tested hypothesis (estimand only); anything about
  AutoGluon's own batching beyond the k-variate equivalence verified in E0.
