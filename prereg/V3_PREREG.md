# v0.3 confirmatory study pre-registration (CrossLeak v0.3.0)

Frozen by `prereg/V3_FREEZE.sha256` before any v0.3 data exist. It reuses the frozen main driver
(`src/crossleak/experiments/main.py`) and the frozen main analysis helpers (`src/crossleak/analysis/main_report.py`,
`attack.py`, `defenses/dp_prototype.py`, `decode.py`, `metrics.py`, `model.py`, `data/panel.py`, `data/prepare.py`) with
their v0.2 hashes unchanged; the only new analysis code is `src/crossleak/analysis/v3_report.py`. A fresh seed
(20261115) means no v0.3 sample repeats a v0.2 (20261114) or E0 (20261112) sample. The three goals below follow from the
v0.2 result (`docs/MAIN_RESULTS.md`): v0.2 showed the defences reach the same point estimates but did not test
equivalence; it showed single-member leakage vanishing with group size but left open whether a stronger sybil attacker
would do better; and reviewers will ask whether the channel exists in the shipped library.

## 1. Goal 1 — Defence equivalence (confirmatory)
Claim **E1**: at one day of history, on the primary model (c2base), the cold-start gain of public-context forecasting
(D1) and of DP-Prototype at ε = 1 per day (D4, P = 8) is **equivalent** to the gain of cross-learning with private
co-tenants, within a margin of 2 percentage points of gain.
- Test: TOST on the paired gain difference gain(a) − gain(b), bootstrapped by user (2,000 resamples). EQUIVALENT iff the
  90 % CI ⊂ (−0.02, +0.02). Pairs (ELD): D1_16 vs xl16; D4_P8_eps1 vs xl16; D4_P8_eps1 vs D1_16. LifeSnaps: D1_4 vs
  xl8; D4_P8_eps1 vs xl8 (reported; LifeSnaps has ≈ 6 established users/day, so equivalence may be undetermined).
- Claim **E2** (no benefit at 3 days): the cross-learning gain on ELD at ℓ = 72 h is equivalent to 0 within ±0.02
  (same TOST). This replaces the mislabelled CL1b of v0.2.
- Sample: 600 cold-start targets per dataset × model × ℓ (half the CI width of v0.2's 300). Power: v0.2's paired
  differences were ≤ 0.8 pp with a 95 % half-width ≈ 1.0 pp at n = 300; at n = 600 the 90 % half-width ≈ 0.6 pp, inside
  the 2 pp margin, so equivalence is well powered when the true difference is near 0.
- Interpretation: EQUIVALENT on ELD ⇒ the paper states the cold-start benefit needs neither private co-tenants nor a
  privacy loss beyond ε = 1. NOT_EQUIVALENT ⇒ report the measured difference; the "no private data needed" claim is
  softened to "comparable point estimates".

## 2. Goal 2 — Stronger sybil attacker (confirmatory + estimand)
The v0.2 attacker used raw forecast features and ≤ 800 training groups at k = 256. The v0.3 attacker is strengthened in
three pre-registered ways, all computable by the real adversary (it controls its probe and fillers and can query its
probe alone): (i) **residual features** f_probe(group) − f_probe(probe alone), which cancel the probe's own content;
(ii) **more training groups** (2,000 at k = 16/64, 1,500 at k = 256 vs 600–800 before); (iii) the profile-evening
read-out for L1. Filler = copies of the probe (the primary v0.2 sybil).
- Claim **S1** (confirmatory, the decisive question restated): with this stronger attacker, per-victim leakage at the
  default batch size still fails — AUC L1 at k = 256 (c2small, ELD) has a ΔAUC-vs-control 90 % CI whose upper bound is
  below 0.05. If instead ΔAUC lower bound > 0 at k = 256, S1 is refuted and the paper reports that large groups do
  **not** protect even against a single read-out probe.
- Estimands (no hypothesis): AUC and ΔAUC at k ∈ {16, 64, 256} for ELD (c2small; c2base at k = 16) and LifeSnaps
  (c2small); the profile Δr1; L5 (idiosyncratic-event) leakage. These quantify how far the stronger attacker gets at
  each k.
- Unit of analysis: victim; cluster bootstrap by victim, 2,000 resamples. Anchor: single-member k = 2 (same cells).

## 3. Goal 3 — Real-system check (AutoGluon), off-cluster
Claim **R1**: the leakage channel reproduces through AutoGluon's `TimeSeriesPredictor` with the Chronos-2 preset and
its shipped defaults, and the library accepts the adversarial inputs the attack needs.
- Procedure (`scripts/agcheck.py`, `docs/AGCHECK.md`): build real ELD co-tenant groups, forecast them through
  `TimeSeriesPredictor(prediction_length=24).fit(..., {"Chronos2": {...}})` with `batch_size = k` so each consecutive
  block of items is one cross-learning batch; read item 0 (the probe); fit a logistic decoder on the probe's forecast.
  Record the model's effective `cross_learning`/`batch_size`, acceptance of a 24 h series and of duplicate series, the
  group vs isolated-control AUC, the residual AUC, and the victim-swap channel.
- Success: `cross_learning` default is True; the library accepts a 24 h and a duplicate series; the residual AUC exceeds
  the isolated control by a clear margin on real data.
- This runs in a separate virtualenv with `autogluon.timeseries`; it is **not** part of the sharded overnight campaign
  (`agcheck.run_on_cluster: false`). Its results are produced off-cluster and committed to `docs/AGCHECK.md`; the
  evaluation-type ceiling for R1 is `real_system` (the shipped library on recorded data).

## 4. Confirmatory rules (summary)
| ID | Rule | Verdict key |
|---|---|---|
| E1 | 90 % CI of gain(a)−gain(b) ⊂ (−0.02, 0.02), each ELD pair | EQ_eld_* |
| E2 | 90 % CI of ELD ℓ=72 cross-learning gain ⊂ (−0.02, 0.02) | NOBEN_eld_* |
| S1 | ΔAUC-vs-control at k=256 (c2small, ELD): 90 %/95 % CI upper bound < 0.05 ⇒ attacker fails | SYB2_eld_c2small_k256 |
| R1 | AutoGluon defaults + acceptance + residual AUC ≫ control | docs/AGCHECK.md |

## 5. Invalid runs, exclusions, deviations
Same as the main plan: a cell errors → retried once; infeasible cells reported; > 10 % invalid ⇒ NOT_INTERPRETABLE;
every result carries its config hash and a mismatch ⇒ CONFIG_MISMATCH; samples excluded only by the validity and
label-definition rules. The frozen main and E0 files are untouched. `dpaudit`, `invariance` and `cost` cells emitted by
the frozen driver are kept small and are a sanity check only (CL2 and the DP audit were settled in v0.2). Any post-freeze
code change is logged in `DECISIONS.md` before the affected cells run.

## 6. Frozen files
`prereg/V3_PREREG.md`, `configs/v3.yaml`, `src/crossleak/analysis/v3_report.py`, plus every file already in
`prereg/MAIN_FREEZE.sha256` at its v0.2 hash (listed again here so one `sha256sum -c` covers the whole v0.3 analysis
path). `scripts/agcheck.py` is listed for provenance of the off-cluster check.
