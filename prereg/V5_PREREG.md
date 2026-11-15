# v0.5 study pre-registration (CrossLeak v0.5.0)

Frozen by `prereg/V5_FREEZE.sha256` before any v0.5 data exist. Approved by the author on 2026-11-15 (DECISIONS.md
D-036). Requested in the review of the MONET manuscript (round r3).

## 1. Why
The review accepted the reported verdicts but asked for evidence on four points:
- **M3** whether public context helps *because* its days are of the same kind (same weekday, same domain), or because
  any extra input helps (E5a);
- **M8** an end-to-end serving measurement instead of the estimated cost, including D1 (E5b);
- **M2** whether the weak k = 256 sybil leak of the confirmatory study (one attacker-training realisation, S1) is
  stable over independent realisations, and how a group size near the pretraining range (k = 32) behaves (E5c);
- **M1** how often and at what cost an attacker reaches a victim's batch under explicit grouping policies, and what
  it extracts at an occupancy short of 255 of 256 (E5d).
Four further analyses use only existing data and are labelled post hoc (`docs/reanalysis/posthoc_r3.py`, run before
this freeze): P2 re-bootstrapped by unique user, a ridge baseline, a point-forecast-only attacker and exact Gaussian
audit ceilings. They are not part of this plan.

## 2. Design
Driver `v5` (`src/crossleak/experiments/v5.py`): the frozen main driver, unchanged (its hash is part of this freeze),
patched at run time. Configuration `configs/v5.yaml`: seed 20261117 (no sample repeats E0, v0.2, v0.3 or v0.4); same
user pools (pool seed 20261112), with the cold-start subset of the victim pool redrawn by the campaign seed, as in
every study; δ = 1e-5; 2,000 bootstrap resamples by user or victim.

**E5a public-context ablation.** Base model, one day of history (ℓ = 24 h), 600 cold-start targets per dataset.
On the same targets (paired): `iso`, `full`, `xl<k>` and four public contexts of k − 1 windows (k = 16 ELD, 4
LifeSnaps), each a 14-day window of a distinct public user:
- `d1p` same weekday as the origin, at least 56 days before it (as v0.4);
- `d1x` a *different* weekday, at least 56 days before;
- `d1r` any weekday, at least 56 days before (random historical windows);
- `d1o` public users of the *other* dataset, any day (other domain; distinct user-days, since LifeSnaps has 9
  public users).
Origins: test-split days on which a target, k − 1 established users and enough past windows on the same and on other
weekdays exist.

**E5b serving run.** One idle machine (4 threads), base model, ELD. One night (a test origin d): 1,000 new clients
(last 24 hours of target users), served mode by mode: seasonal naive, a ridge baseline (`crossleak/baselines.py`,
fitted beforehand, fit time reported separately), D0, D4 (P = 8, ε = 1, release on night d included), D1 (`d1p`,
k = 16) and XL (k = 16 established co-tenants). Forecasts in calls of at most 256 series (AutoGluon's batch). Per mode:
total wall time from group construction (including the release) to the last forecast, per-call p50/p95, client
completion p50/p95 (from the mode's start), throughput, and the process's peak RSS during the mode with its increase
over the mode's start (VmHWM reset per mode; memory kept from earlier modes counts). Excluded from the timed modes:
model load, the ridge fit, the one-off public k-means and one untimed warm-up call of each mode's call shape.

**E5c sybil realisations.** The confirmatory study's stronger attacker (residual features + calendar, linear decoders,
`v3_report._residual_attacker`, unchanged), ELD, small model, copy fillers, k = 256 with 1,500 / 500 training / test
groups (as S1) and k = 32 with 2,000 / 600, five realisations each (`rep` 1–5). Each realisation draws its own
training groups, test groups and victims.

**E5d placement.** (i) Emulation without a model (`run_placement`): a nightly table forecast in consecutive blocks
of 256 items under four policies: `ordered` (items in registration order; the attacker registers m meters in one
burst at a random time), `informed` (registration order; the attacker sees the ID counter and, whenever an honest
registration opens a block, registers min(255, remaining budget) meters right after it), `informed_pad` (as informed,
but it first registers meters up to the next block boundary, once, if its budget allows), `hashed` (items in random-ID
order); N0 ∈ {1,000; 10,000;
100,000} established clients, λ ∈ {1, 10, 100} honest registrations per day, m ∈ {1, 16, 64, 255, 1,000, 10,000},
30 days, 200 runs; and the paper's service policy (each new client grouped with k − 1 random established clients,
k ∈ {2, 16, 256}). Outputs, for two victim populations (the new clients of the horizon and all honest clients):
probability of co-location, attacker share of the victim's block, victims isolated (255 of a full block of 256, and
alone with attacker meters in any block, including a partial last block), and meters used. (ii) Half sybils (128 of 256 members, 127 honest others), same attacker and sizes as S1, one
realisation.

Planning estimate (v0.3 per-series costs): E5a ≈ 3.5, E5c ≈ 42.6, E5d ≈ 7.3, E5b ≈ 1.2 machine-hours; about
10.7 hours per machine for the main phase on five machines, then about 1.2 hours of cost phase on one machine (about 12
hours in all). A deadline (`--until`) must leave room for the cost phase, or E5b is skipped.

## 3. Rules (fixed before the data)
**E5a** (base model; paired gain difference a − b in percentage points of gain over isolated forecasting, 90 % CI by
user; margin ±1 point), in this order: A_BETTER if the CI lies above 0 (A_BETTER_SMALL if it also lies inside
(−1, +1)); B_BETTER / B_BETTER_SMALL likewise below 0; NO_DIFFERENCE if inside (−1, +1) and not above or below 0;
otherwise INCONCLUSIVE.
| ID | Dataset | a | b | Question |
|---|---|---|---|---|
| A1 | ELD | d1p_16 | d1x_16 | does the same weekday help? |
| A2 | ELD | d1p_16 | d1r_16 | does choosing the weekday beat random historical windows? |
| A3 | ELD | d1p_16 | d1o_16 | does the same domain help? |
| A5 | ELD | d1p_16 | xl16 | replication of E1c-1 on fresh samples (estimand; the ±2 rule of v0.4 is also reported) |
| A1L–A3L, A5L | LifeSnaps | d1p_4 | d1x_4, d1r_4, d1o_4, xl4 | as above, small-sample caveat of v0.4 |
A4 / A4L: gain of d1o over isolated forecasting, 90 % CI: CONTEXT_HELPS if above 0, else NO_GAIN_SHOWN.

**E5c** per k (ΔAUC of label L1 against the no-channel control, 95 % CI by victim, per realisation): STABLE_LEAK if
at least 4 of the 5 planned realisations have a lower bound above 0; UNSTABLE if 1–3; NOT_REPLICATED if none. A
realisation without a result counts as not positive. Every planned verdict without a result is NO_DATA. The mean, SD and
range of AUC and ΔAUC over realisations are estimands.

**E5d** (ii): LEAKS if the ΔAUC 95 % CI lies above 0, else NOT_DETECTED. (i) has no rule (estimands).

**E5b**: MEETS_15MIN if D4's measured end-to-end time for 1,000 new clients is at most 15 minutes (the main study's CL8
bound, there checked by an estimate); all other numbers are estimands.

## 4. Interpretation fixed in advance
- **E5a.** A1 and A3 A_BETTER: the paper may say that the same weekday and the same domain contribute to the help of
  public context; A_BETTER_SMALL: that they contribute, by less than one point. B_BETTER_SMALL is read as B_BETTER.
  "Not A_BETTER" below means neither A_BETTER nor A_BETTER_SMALL. A2 is reported without an interpretation rule.
  A1 NO_DIFFERENCE or B_BETTER: the paper drops "same weekday" from its explanation. A3 NO_DIFFERENCE
  or B_BETTER, or A4 CONTEXT_HELPS with A3 not A_BETTER: the paper states that unrelated context helps as much and
  that the help is not shown to come from days of the same kind; the key insight keeps only "the help does not require
  private co-tenants". INCONCLUSIVE: reported as such; the explanation stays a premise. A5 is reported next to E1c-1.
- **E5b.** The measured times replace the estimate in the paper (the estimate stays in Table 1 as the main study's
  verdict). EXCEEDS_15MIN is reported as such.
- **E5c.** STABLE_LEAK at k = 256: the paper reports the distribution and keeps the conditional k = 256 claim.
  UNSTABLE: the claim becomes "in some realisations". NOT_REPLICATED: the paper reports that S1's leak did not
  replicate and removes the k = 256 leak from the abstract and contributions. The S1 verdict of v0.3 stays reported.
  k = 32 is reported with the same rule.
- **E5d.** The placement estimands state, per policy, the meters an attacker needs to isolate one victim and whether
  it can choose that victim. LEAKS at half occupancy: the paper states that half of a default batch suffices;
  NOT_DETECTED: the paper states that no leak was detected at half occupancy.
- All earlier verdicts remain reported unchanged.

## 5. Invalid cells, stopping, deviations
As in the main study: a failing cell is retried once; more than 10 % invalid or infeasible cells make the study
NOT_INTERPRETABLE; a configuration mismatch is flagged CONFIG_MISMATCH. The serving run needs an idle machine (cost
phase, shard 0). Any deviation is logged in DECISIONS.md and reported in the paper.

## 6. Frozen files
`prereg/V5_PREREG.md`, `configs/v5.yaml`, `src/crossleak/experiments/v5.py`, `src/crossleak/analysis/v5_report.py`,
`src/crossleak/baselines.py`, `src/crossleak/experiments/registry.py`, and, unchanged since their own freezes,
`src/crossleak/experiments/main.py`, `src/crossleak/analysis/main_report.py`, `src/crossleak/analysis/v3_report.py`,
`src/crossleak/defenses/dp_prototype.py`, `src/crossleak/metrics.py`, `src/crossleak/model.py`,
`src/crossleak/attack.py`, `src/crossleak/decode.py`, `src/crossleak/config.py`, `src/crossleak/experiments/v4.py`
(whose snapshot of the main driver v5 uses), `src/crossleak/data/panel.py`, `src/crossleak/data/prepare.py`.
`registry.py` changed after the v0.4 freeze only to dispatch the `v5` driver; its V4 hash is superseded (DECISIONS.md).
