# v0.4 causal-defence study pre-registration (CrossLeak v0.4.0)

Frozen by `prereg/V4_FREEZE.sha256` before any v0.4 data exist. Approved by the author on 2026-11-14 (DECISIONS.md D-033).

## 1. Why
The pre-submission review (round r1, finding B3-E1) found that the frozen main driver (`experiments/main.py`) took the
public-context (D1) windows at least 56 days away from the forecast origin in *either* direction, and fitted the
LifeSnaps DP-Prototype (D4) public centres on public users from any day (the ELD centres come from 2013 only). A
deployment can use only past data. The v0.3 equivalence results (E1) are therefore results for a non-causal emulation.
This study repeats them with public data restricted to the past, on fresh samples, and measures the effect of the
restriction on the same targets.

## 2. Design
- Driver `v4` (`src/crossleak/experiments/v4.py`): the frozen main driver, unchanged (its hash is part of this freeze),
  patched at run time to (a) run utility cells only, (b) add on the same cold-start targets
  - `d1p_<k>`: k − 1 windows of distinct public users on the same weekday, at least 56 days **before** the origin;
  - `d4p_P8_eps<e>`: D4 whose public centres are fitted by k-means only on public users' 14-day contexts that end
    before the origin (up to 40 days sampled, at least 2P = 16 representations; LifeSnaps only, since the ELD centres
    already come from 2013);
  next to the original `d1_<k>` and `d4_P8_eps<e>`.
- Origins: the test-split days on which a target, the k − 1 established co-tenants, the past public windows and (on
  LifeSnaps) the past centres all exist. The same origins serve every configuration, so all comparisons are paired.
- Configuration `configs/v4.yaml`: seed 20261116 (no sample repeats E0, v0.2 or v0.3); same user pools (pool seed
  20261112) and the same cold-start users; one day of history (ℓ = 24 h); both models; 600 targets per dataset × model;
  XL k ∈ {4, 16} (ELD), {4, 8} (LifeSnaps); D1/D1p k = 16 (ELD), 4 (LifeSnaps); D4 P = 8, ε ∈ {1, 4} per daily release,
  δ = 1e-5; 2,000 bootstrap resamples by user.

## 3. Confirmatory rules (base model, ℓ = 24 h; TOST on the paired gain difference; EQUIVALENT iff the 90 % CI lies in
(−0.02, +0.02); verdict keys EQC_*)
| ID | Dataset | a | b |
|---|---|---|---|
| E1c-1 | ELD | d1p_16 | xl16 |
| E1c-2 | ELD | d4_P8_eps1 | xl16 |
| E1c-3 | ELD | d4_P8_eps1 | d1p_16 |
| E1c-4 | LifeSnaps | d1p_4 | xl8 |
| E1c-5 | LifeSnaps | d4p_P8_eps1 | xl8 |

Margin: 2 percentage points of gain, the same as v0.3 (fixed before the v0.3 data from the main study's variance).
Power: v0.3 90 % half-widths at 600 targets were 0.5–1.3 points on ELD; the causal windows are expected to add little
variance, so equivalence is well powered on ELD when the true difference is near 0. LifeSnaps has 13 cold-start users
and a cross-learning gain smaller than the margin; its rules are reported as computed and carry little weight.

## 4. Estimands (no hypothesis)
- Effect of causality, paired, 90 % CI: d1p_16 − d1_16 (ELD), d1p_4 − d1_4 (LifeSnaps), d4p − d4 at ε = 1 (LifeSnaps).
- Gains over isolated forecasting (95 % CI) of every configuration and both models; seasonal naive.

## 5. Interpretation fixed in advance
- E1c-1 EQUIVALENT: the paper states that public context restricted to past data is equivalent to cross-learning on
  ELD (base model, one day of history). NOT_EQUIVALENT: the paper reports the measured difference as the result for a
  causal deployment and describes the v0.3 E1 result as a non-causal emulation.
- E1c-2/3: replication of the v0.3 ELD D4 result on fresh samples (D4 was already causal on ELD); NOT_EQUIVALENT is
  reported as such and the D4 claim narrowed to the measured difference.
- E1c-4/5: reported as computed with the caveat of section 3.
- The v0.3 E1 verdicts remain reported. For statements about a deployable defence, the v0.4 verdicts take precedence.

## 6. Invalid cells, stopping, deviations
Same rules as the main study: a failing cell is retried once; more than 10 % invalid or infeasible cells make the study
NOT_INTERPRETABLE; a configuration mismatch is flagged CONFIG_MISMATCH. Any deviation is logged in DECISIONS.md and
reported in the paper.

## 7. Frozen files
`prereg/V4_PREREG.md`, `configs/v4.yaml`, `src/crossleak/experiments/v4.py`, `src/crossleak/analysis/v4_report.py`,
`src/crossleak/experiments/registry.py`, and, unchanged since their own freezes, `src/crossleak/experiments/main.py`,
`src/crossleak/analysis/main_report.py`, `src/crossleak/analysis/v3_report.py`, `src/crossleak/defenses/dp_prototype.py`,
`src/crossleak/metrics.py`, `src/crossleak/model.py`, `src/crossleak/data/panel.py`, `src/crossleak/data/prepare.py`.
