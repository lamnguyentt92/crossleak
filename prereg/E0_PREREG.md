# Pre-registration E0: pilot and killer experiment (CrossLeak v0.1.0)

Frozen before the first E0 run. File hashes: `prereg/E0_FREEZE.sha256`. Any change after the
freeze is logged in `DECISIONS.md` (deviations) and makes the report print `MODIFIED`.

## 1. Status and purpose
E0 is the pilot and the killer experiment of the CrossLeak idea specification (11 Nov 2026, section 8).
It answers one question: **is the CrossLeak study worth running?** Its data are **exploratory**: they decide
GO / REFRAME / EXTEND / STOP and give variances for the power analysis of the main plan (E1-E7, `v0.2.x`).
No paper claim rests on E0 data; the main experiments are re-run with fresh seeds under their own frozen plan.

## 2. Questions
- **Q1 (gain).** Does Chronos-2 cross-learning reduce day-ahead forecast error of a cold-start user (short history)
  compared with forecasting that user alone?
- **Q2 (channel).** Can a participant who controls one member of a cross-learning group decode information about
  another member's previous day from its *own* forecast, beyond what it can predict without the channel?
- **Q3 (invariance).** Is the attacker's forecast invariant to positive affine transforms of a victim's series
  (expected exactly, from per-series standardisation)? Sanity check, not a decision input.
- **Q4 (cost).** CPU latency per series by group size on the operator's machines. Descriptive.

## 3. Design
- **Data.** ELD (UCI ElectricityLoadDiagrams 2011-2014, CC BY 4.0): hourly mean kW, 2013-2014, clients with
  <= 1 % zero hours (327 of 370). LifeSnaps v4 (Zenodo 7229547, CC BY 4.0): hourly mean heart rate (bpm) from the
  hourly CSV, users with >= 672 observed hours (57 of 71).
- **Windows.** Day-ahead: origin at 00:00, context 336 h, horizon 24 h. Valid (user, origin): ELD - context and
  horizon fully observed, <= 5 % zero hours in context, positive horizon sum; LifeSnaps - >= 50 % of context,
  >= 20 of the last 24 h and >= 20 of 24 horizon hours observed. The last 24 h must have positive variance.
- **Pools.** Users split once (seed 20261112) into victim 50 % / aux 35 % / public 15 %. The attacker's training data
  come only from the aux pool; test victims only from the victim pool (subject-disjoint).
- **Days.** ELD: attacker training origins 2013-01-15..2013-12-20, test origins 2014-01-15..2014-12-20 (time-disjoint),
  excluding origins whose label day or horizon contains a DST change day. LifeSnaps: all valid days for both
  (subject-disjoint only; deviation from time-disjointness accepted because of the short study period).
- **Models.** `amazon/chronos-2` (c2base, 120M) and `autogluon/chronos-2-small` (c2small, 28M),
  chronos-forecasting 2.3.2, CPU. A group of k users is passed as one k-variate task; `crossleak doctor` verifies
  that this equals `cross_learning=True` on the same k series.
- **Gain cells** (Q1). Scenarios: *mixed* (one target with ell hours + k-1 co-tenants with 336 h; 200 targets) and
  *allshort* (all k members with ell hours; >= 200 targets). ell in {24, 72, 168}; k in {4, 16, 64} (ELD),
  {4, 8} (LifeSnaps). Isolated baseline: the same target context forecast alone. Metric per item: WQL over the
  deciles 0.1..0.9. **Gain = 1 - mean WQL_xl / mean WQL_iso**, 95 % percentile CI by bootstrap over target users
  (1,000 resamples). Also reported: MASE, seasonal-naive WQL, full-context WQL (mixed).
- **Decode cells** (Q2). k in {2, 4}; probes: *honest* (attacker's own 336 h), *short* (attacker's last 24 h),
  *noise* (24 standard-normal values). 2,000 training groups (all aux users, training days) and 600 test groups
  (attacker from aux, victim and others from the victim pool, test days).
  - Attacker features: its own decile forecast mapped back to model space (arcsinh((q - loc)/scale) with its own
    context statistics), the isolated forecast of the same probe (computable with the public model) and calendar
    one-hots. **Control** attacker: isolated forecast + calendar only.
  - Targets: victim's previous-day profile (z-scored 24 h) and a binary label "evening level (18-23 h) above the
    training median".
  - Decoders: ridge (alpha by GCV over 1e-2..1e7) and L2 logistic regression (C by 5-fold CV), features standardised on training data.
  - Profile accuracy per test sample is the **anomaly correlation**: Pearson r between predicted and true profile
    after subtracting the training-set mean profile (the shared daily shape is not leakage). Raw r is secondary.
  - **delta_r** = mean over test samples of (anomaly r with channel - anomaly r of control); **delta_AUC** =
    AUC(channel) - AUC(control). 95 % CIs by bootstrap over victims (1,000 resamples). Permuted-label control reported.
- **Channel cells** (descriptive). Swap-one influence on the attacker's model-space forecast, k in {2, 4, 16}
  (ELD) / {2, 4, 8} (LifeSnaps), 200 pairs per cell.
- **Invariance** (Q3): k = 4, short probe, 50 groups x 4 transforms. **Speed** (Q4): k in {1, 4, 16, 64, 256}.

## 4. Decision rules (fixed; implemented in `src/crossleak/analysis/e0_report.py`)
- **GAIN per dataset** (primary model c2base, ell <= 72, any scenario and k):
  YES if some cell has gain >= 0.05 and CI lower bound > 0; WEAK if some cell has gain >= 0.02 and lower bound > 0;
  otherwise NO.
- **CHANNEL per dataset** (decode cells with k in {2, 4}, any probe, either model):
  YES if some cell has CI lower bound > 0.05 for delta_r or delta_AUC; AMBIGUOUS if some lower bound > 0;
  otherwise NO.
- **Invariance**: OK if the attacker's maximum absolute change < 1e-3 (model space) for every transform; otherwise
  WARN (investigate; not a decision input).
- **Validity**: if more than 10 % of planned cells failed (after one automatic retry), E0 is NOT_INTERPRETABLE.
  Cells that are infeasible by design (not enough valid users) are listed and not counted as failures.

## 5. Interpretation of every outcome (fixed in advance)
| Overall | Condition | Action |
|---|---|---|
| GO | CHANNEL YES on some dataset and GAIN YES on some dataset | Main plan as specified (attacks A1-A3, defences D0-D4) |
| GO_REFRAME | CHANNEL YES, no GAIN YES | Author decides; default framing "leakage enabled by default for little benefit"; D4 deprioritised |
| WARN_EXTEND | best CHANNEL is AMBIGUOUS | Author decides on E0-extended (optimised probe, larger n) before any main run |
| STOP | CHANNEL NO everywhere | Stop CrossLeak; record in the idea registry; pivot per the spec, section 9 |
| NOT_INTERPRETABLE | > 10 % failed cells | Fix, re-run E0 with the same plan |

A GAIN result alone never makes E0 a GO: without a channel there is no paper.

## 6. Deviations and stopping
- Each cell is retried once automatically on the next run; the operator script stops starting new cells after the
  time budget and resumes on the next call.
- Any deviation (data, rule, code after the freeze) is written to `DECISIONS.md` with date and reason before analysis.
- Results are analysed only after all cells are done or reported as not run.

## 7. Frozen files
`prereg/E0_PREREG.md`, `configs/e0.yaml`, `src/crossleak/experiments/e0.py`, `src/crossleak/analysis/e0_report.py`,
`src/crossleak/decode.py`, `src/crossleak/metrics.py`, `src/crossleak/data/panel.py`, `src/crossleak/data/prepare.py`,
`src/crossleak/model.py`.
