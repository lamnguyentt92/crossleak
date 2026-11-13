# Main study pre-registration (CrossLeak v0.2.0)

Frozen by `prereg/MAIN_FREEZE.sha256` before any main-study data exist. The E0 pilot (`docs/E0_PILOT_RESULTS.md`), the
E0 re-analysis with this plan's labels (`docs/pilots/e0_recalibration.json`) and the D4 design pilot on ELD 2013
(`docs/pilots/D4_DESIGN.md`) set the design and the thresholds below; none of them supports a claim. Every main-study
sample is drawn with a new seed (20261114). Constants in section 5 are the constants of `src/crossleak/analysis/main_report.py`.

## 1. Question
Chronos-2 cross-learning forecasts a group of series jointly. We test (i) when it helps a cold-start user, (ii) how much
a group member learns about another member's previous day, (iii) whether large groups protect against an attacker that
controls many members (sybils), and (iv) whether DP-Prototype cross-learning (D4) keeps the benefit with a formal
user-level guarantee, at an acceptable cost on a CPU edge node.

## 2. Setting
- Data: ELD (UCI, 327 clients kept, hourly, 2013–2014) and LifeSnaps (Fitbit, hourly heart rate, 57 users kept). Day-ahead
  forecasting: origin 00:00, context 336 h, horizon 24 h. Validity rules, pools (victim 50 % / aux 35 % / public 15 %, pool
  seed 20261112) and splits as in E0. ELD: attacker training on 2013, evaluation on 2014 (subject- and time-disjoint);
  LifeSnaps: subject-disjoint only.
- Models: `amazon/chronos-2` (c2base, primary) and `autogluon/chronos-2-small` (c2small). A group of k series is one
  k-variate task (identical to `cross_learning=True` on those k series; verified in E0).
- Service population (utility cells): a fixed cold-start subset of the victim pool (30 % ELD, 50 % LifeSnaps; seeded)
  provides the targets; the other victim-pool users plus the aux pool are the established users (co-tenants, prototype
  contributors). Public users never belong to the service.
- Evaluation type: `real_system` recorded data with offline inference of the public models (forecasting, attacks,
  utility); `emulation` of an edge node by one 4-vCPU VM (cost). No LLM judge, no human rater.

## 3. Threat model and attacker
Honest service. The attacker controls m members of a group of size k, chooses their series and reads only the forecast
of its read-out probe (lower bound: other members' outputs are ignored). It knows the public model and has auxiliary
data of other users (aux pool). It never sees a victim's data. Configurations:
- `single` (m = 1); the other k − 2 members are honest users.
- `sybil` (m = k − 1): the victim is the only honest member. Fillers (the m − 1 non-read-out members):
  `copy` (identical copies of the probe; primary: look like ordinary new users), `sparse` (one reading in the last hour;
  assumes the service accepts such series), `noise` (independent 24-h noise series).
- `half` (m = k / 2, copy fillers) with k/2 − 1 honest others.
- Probes: `short` (the last 24 h of an aux user; primary), `noise`, `honest` (14 days of an aux user).
The attacker fits its decoders on groups it simulates itself (victims and others from the aux ∪ public pools, training
split) and is evaluated on victim-pool users (test split). Decoders: linear (ridge for profiles, logistic with 3-fold CV
for labels; primary) and nonlinear (gradient-boosted trees, one-hidden-layer MLP; estimand). Features: the probe's
forecast in model space (9 deciles × 24 h), the isolated forecast of the probe, calendar one-hots. Controls: the
no-channel decoder (isolated forecast + calendar only) and the permuted-label decoder.

## 4. Measures
- Labels of the victim's previous day (24 h before the origin):
  - **L1 (primary)** evening level (mean of the z-profile over 18–23 h) above the median of all users valid that day
    (population-wide shocks cannot predict it);
  - L2 evening level above the attacker's training median (the E0 label);
  - L3 daily mean below the 25th percentile of the victim's own previous 13 daily means (instance-norm units);
  - L4 evening level above the median of the victim's own previous 13 days (self-relative event);
  - L5 the L4 residual above the median residual of all users that day (idiosyncratic event).
  The attacker's *training* labels L1 and L5 use the same statistics over the users it can access (aux and public
  pools) that day, so that no training label depends on victim-pool data; test labels use all users. A sample where a
  label is undefined (fewer than 7 previous days, or no reference that day) is excluded from that label only.
- **R1 (primary)** within-day anomaly correlation corr(ŷ − m_d, y − m_d), m_d = mean z-profile of all users that day;
  R2 the E0 anomaly correlation (training mean subtracted).
- A1 linkage: the reconstructed previous-day profile ranks the true co-tenant among N ∈ {10, 50} victim-pool candidates,
  scored by correlation with each candidate's old weekday profile (same weekday, 3–10 weeks before the origin). Top-1,
  top-5; chance 1/N, 5/N.
- Channel: median swap influence (mean |Δ| of the probe's model-space output when the victim is replaced by another
  victim), and its ratio to the single-member k = 2 value of the same dataset, model and probe.
- Utility: WQL (deciles) of the cold-start target; gain = 1 − mean WQL(config) / mean WQL(isolated); retained = gain of a
  defence / gain of cross-learning with the reference group size (k = 16 ELD, k = 8 LifeSnaps), paired on targets.
  Configurations on the same targets: isolated (D0), full context (reference), cross-learning (XL) k ∈ {4, 16, 64}
  (ELD) / {4, 8} (LifeSnaps), public context D1 (k − 1 public users' windows of the same weekday, ≥ 56 days away),
  DP-Prototype D4 (P ∈ {4, 8}, ε ∈ {0.5, 1, 2, 4, 8, ∞} per day, δ = 1e-5, `dayfold` representation, clip √24, 90 % of
  ρ on sums). History ℓ = 24 h (all configurations) and ℓ = 72 h (D0, XL).
- DP audit: an adversary that knows every other user and sees the release observes the victim's clipped representation
  plus the release noise; logistic decoder trained on aux users (training split), tested on victims; compared with the
  attribute-inference ceiling of (2ε, (1 + e^ε)δ)-indistinguishability.
- Cost: per-user latency (p50/p95) and per-user time in batches, isolated / XL (k ∈ {4, 16, 64, 256}) / D4 (P ∈ {4, 8}),
  release time for 1,000 users, peak RSS; one VM with the other four idle; 4 threads.

## 5. Confirmatory rules (per dataset unless stated; α = 0.05; 95 % percentile cluster bootstrap, 1,000 resamples, by
victim for attacks and by user for utility)
| ID | Claim | Rule (estimates on the test split) | Constant |
|---|---|---|---|
| CL1 | With one day of history (ℓ = 24 h), cross-learning lowers the cold-start WQL (c2base) | YES: some k has gain ≥ 0.05 and Bonferroni-adjusted (α/|k|) CI lower bound > 0; WEAK: some k has adjusted lower bound > 0; else NO | GAIN_YES 0.05 |
| CL1b | With three days of history (ℓ = 72 h) cross-learning brings no benefit (c2base) | NO_BENEFIT: every k has adjusted CI upper bound < 0.02; else BENEFIT | 0.02 |
| CL2 | The channel is far above numerical noise and exactly invariant to positive affine transforms of others | YES: for both models, invariance max change < 1e-3 and median swap influence (single, k = 2, short) ≥ 100 × that change | 1e-3, 100 |
| CL3 | A single member of a pair infers L1 (c2base, short probe, k = 2) | YES: AUC ≥ 0.65 and ΔAUC vs no-channel CI lower bound > 0 | 0.65 |
| CL3-k4 | Leakage persists at k = 4 (c2base, short) | PERSISTS: ΔAUC L1 lower bound > 0; else NOT_DETECTED | — |
| CL4 | A single member of a pair reconstructs the victim's previous-day profile beyond the population (c2base, short, k = 2) | YES: Δr1 (channel − no-channel) ≥ 0.10 with lower bound > 0 | 0.10 |
| CL5 | Large groups do not protect against sybils (decisive experiment) | YES: (a) single-member channel ratio at the largest k (256 ELD, 8 LifeSnaps; c2small) has upper CI < 0.25 and (b) sybil-copy at k = 256 (c2small) has AUC L1 ≥ 0.60 with ΔAUC lower bound > 0; SYBIL_ONLY: (b) only; else NO | 0.25, 0.60 |
| CL5-base | Sybil leakage on the primary model (c2base, copy, k = 16) | PERSISTS: ΔAUC L1 lower bound > 0 | — |
| CL6 | D4 keeps ≥ 50 % of the cross-learning gain at ε ≤ 4 per day (c2base, ℓ = 24, P = 8) | YES_eps<e>: the smallest ε ∈ {1, 2, 4} with retained ≥ 0.50 and gain lower bound > 0 at α/3 (Bonferroni over the three ε); else NO. ELD is the primary dataset; LifeSnaps is reported with the same rule | 0.50 |
| CL8 | D4 serves 1,000 users per day within 15 min on one 4-vCPU VM (c2base, P = 8) | YES: 1,000 × per-user p50 in batches + release time ≤ 900 s | 900 s |
| CL9 | A1 linkage at k = 2 (c2base, short, N = 10) | YES: top-1 ≥ 0.15 with lower bound > 0.10 (chance) | 0.15 |
| DPAUDIT | Sanity of the DP implementation | CONSISTENT: no audit AUC above the ceiling + 0.03; a VIOLATION is a bug and stops the analysis | 0.03 |

The claims are made per dataset; no claim pools the datasets. Families corrected for multiplicity: the k values of CL1
and CL1b, the ε values of CL6.

## 6. Estimands reported without a hypothesis
Gains of every configuration and ℓ; D1 versus D4 (retained gain of each); the history rule (cross-learn only at
ℓ ≤ 24 h: its gain at ℓ = 24 h and the harm it avoids at ℓ = 72 h); all attack configurations and labels L2–L5 with
the linear and nonlinear decoders; sybil `sparse` and `noise` fillers; `half` sybils; k = 4/16 single members; probe
types; c2small versus c2base; channel ratios of every configuration; DP-audit curves; cost tables. Exploratory analyses
added after the data are labelled post hoc.

## 7. Design and sample sizes
Attack configurations (train/test samples): k = 2: 2,000/600; k = 4 and 16: 1,500/600; k = 64: 1,000/500; k = 256:
800/400. Channel pairs: 200 (k ≤ 16), 60 (k = 64), 40 (k = 256). Utility: 300 targets per dataset × model × ℓ.
DP audit: 3,000/2,000. Both models see identical samples (the seed excludes the model). Large cells are split into parts
of at most ~25 min; parts have their own seeds and are concatenated by the analysis.
Power (pilot-based): E0 re-analysis with L1 and R1 (c2base, k = 2, short): ΔAUC 0.216 [0.159, 0.272] (ELD) and 0.224
(LifeSnaps), AUC 0.757 / 0.700, Δr1 ≈ 0.28; A1 top-1 (N = 10) 0.19 / 0.185 (chance 0.10) → power for CL3, CL4 > 0.99,
CL9 ≈ 0.95. CL1: E0 gain +7.5 % to +9.7 % with n = 200; the main study has 300 targets from 49 cold-start users (ELD)
→ adequate. CL6: design pilot (2013, 120 targets) retained 1.10 / 1.01 at ε = 4 / 1 (P = 8). CL5 has no pilot (only a
synthetic mechanism check: copy fillers keep 0.4–1.2 of the k = 2 signal up to k = 256); its outcome is uncertain.

## 8. Invalid cells, exclusions, stopping
A cell that raises an error is retried once; an infeasible cell (no admissible sample) is reported. If failed +
infeasible cells exceed 10 % of the planned cells, the study is NOT_INTERPRETABLE. Cells not run (time budget) make every
verdict PARTIAL_ until they are run with the same frozen plan. Every result records the hash of the configuration it was
produced under; any result produced under another configuration makes the study CONFIG_MISMATCH. Samples are excluded only by the validity rules of E0 and
the label-definition rule of section 4. No interim look changes the plan; re-running a cell with another seed is not
allowed.

## 9. Predictions and interpretation of every outcome
| Rule | Prediction | If YES / as predicted | Otherwise |
|---|---|---|---|
| CL1 | ELD YES, LifeSnaps WEAK | Benefit exists but is narrow (cold start) | ELD NO: reframe as "default leakage without benefit"; the D4 claim moves to a secondary result |
| CL1b | NO_BENEFIT | Supports the history rule | BENEFIT: report; the history rule is not recommended |
| CL2 | YES | Channel exists exactly as modelled | NO: implementation or model change; stop and investigate |
| CL3 / CL4 | YES (both datasets) | Leakage of the victim's routine and profile in pairs | NO: report as a negative result; the paper narrows to the estimands |
| CL5 | ELD YES; LifeSnaps uncertain | Large groups are no defence; sybil filtering or D4 needed | SYBIL_ONLY: sybils leak but dilution is not established; NO: groups of 256 with duplicate filtering suffice against the tested sybils (report sparse/noise estimands) |
| CL6 | ELD YES at ε ≤ 4; LifeSnaps NO (≈ 6 established users a day) | D4 recommended where enough users exist | NO: recommend isolation or D1 |
| CL8 | YES | Practical on CPU edge | NO: report the cost honestly |
| CL9 | YES (weak) | Co-tenant re-identification above chance | NO: drop the linkage claim |

## 10. Deviations
Any change after the freeze (code that affects numbers, configuration, rules) is logged in `DECISIONS.md` with its reason
before the affected cells run, and the report marks the affected results. Bug fixes that do not change any number may be
applied with a log entry. The E0 frozen files stay unchanged.

## 11. Frozen files
`prereg/MAIN_PREREG.md`, `configs/main.yaml`, `src/crossleak/experiments/main.py`, `src/crossleak/analysis/main_report.py`,
`src/crossleak/attack.py`, `src/crossleak/defenses/dp_prototype.py`, `src/crossleak/decode.py`, `src/crossleak/metrics.py`,
`src/crossleak/model.py`, `src/crossleak/data/panel.py`, `src/crossleak/data/prepare.py`.
