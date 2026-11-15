# Changelog

## v0.5.0
v0.5 study, requested in the review of the manuscript (round r3): public-context ablation (E5a), end-to-end serving run
(E5b), sybil attack over five independent realisations at k = 256 and k = 32 (E5c), half sybils at k = 256 and a
batch-placement emulation (E5d); plan frozen by `prereg/V5_FREEZE.sha256` before any v0.5 data
(`src/crossleak/experiments/v5.py`, `src/crossleak/analysis/v5_report.py`, `src/crossleak/baselines.py`,
`configs/v5.yaml`). Post hoc analyses on the deposited bundles: `docs/reanalysis/posthoc_r3.py` (unique-user pooled D4
comparison, ridge baseline, point-forecast-only attacker, exact Gaussian audit ceilings). The frozen main driver and
analysis files are unchanged. This branch runs the v0.5 campaign; the paper release stays on `main`.

## v1.0.1
Data deposit DOI; post hoc analyses after the final pre-submission review (docs/reanalysis/posthoc_r2.py, summary in docs/V4_RESULTS.md); paper asset generator updated (post hoc numbers, LifeSnaps seasonal-naive range over three studies). Experiment and analysis code unchanged.

## v1.0.0
Release for the paper: results documents of all campaigns (`docs/*_RESULTS.md`), independent re-analyses (`docs/reanalysis/`), generator of every number, table and figure of the paper (`paper/make_assets.py`), data-deposit builder (`scripts/make_deposit.py`, `deposit/`), frozen-file check (`scripts/check_provenance.py`), dashboard fix. Experiment and analysis code unchanged.

## v0.4.0
Causal-defence study: D1 public windows and LifeSnaps D4 centres restricted to data from before the forecast origin, next to the original configurations on the same targets; the frozen main driver is patched at run time only; plan frozen by `prereg/V4_FREEZE.sha256`. Known issue: `scripts/watch.sh` without `--config` stops with a NameError (dashboard only; fixed in v1.0.0).

## v0.3.0
Confirmatory study: equivalence of D1 and D4 to cross-learning (TOST), stronger sybil attacker (residual features, larger training sets), real-system check through AutoGluon (`scripts/agcheck*`); the frozen main driver and analysis are reused unchanged; plan frozen by `prereg/V3_FREEZE.sha256`.

## v0.2.0
Main study: sybil attackers (copy, sparse, noise fillers; half sybils), non-linear decoders, linkage (A1), within-day labels, DP-Prototype defence (D4) with its zCDP accountant, edge-cost phase on one machine, live dashboard; plan frozen by `prereg/MAIN_FREEZE.sha256`.

## v0.1.2
Infrastructure and the E0 pilot: data preparation (ELD, LifeSnaps), Chronos-2 wrapper with k-variate groups, probes, attack decoders, D0/D1/D2 defences, resumable campaign driver, five-machine cluster controller that always writes one result bundle, E0 plan frozen by `prereg/E0_FREEZE.sha256`.
