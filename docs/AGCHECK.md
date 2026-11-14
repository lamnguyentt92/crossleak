# Real-system check: the leakage channel through AutoGluon (v0.3, check R1)

**Question.** Does the co-tenant channel exist in the shipped library with its defaults, not only in
our direct `Chronos2Pipeline` wrapper, and does the library accept the inputs the attack needs?

**Procedure.** `scripts/agcheck.py`, run in a separate virtualenv with `autogluon.timeseries==1.6.3`
(torch 2.13 CPU, chronos-forecasting 2.3.2, transformers 5.14.1). Real ELD co-tenant groups (250 train / 150 test
victims; attacker probe = a 24 h noise series; label L1 = the victim's previous-day evening level above the day's
population median) are forecast through `TimeSeriesPredictor(prediction_length=24).fit(train, {"Chronos2":
{"model_path": "autogluon/chronos-2", "batch_size": k}})` and `.predict(...)`. With `batch_size = k` each consecutive
block of `k` items is exactly one cross-learning batch — the way a batched multi-tenant service mixes items. Item 0 of
each block is the attacker's read-out probe. A logistic decoder is fitted on the probe's predicted quantiles
(group features), on the probe forecast alone (isolated control), and on their difference (residual). Inputs are not
part of the overnight campaign (a heavy, separate dependency); the authoritative numbers are here and in
`docs/agcheck/*.json`.

**Findings.**

| Quantity | k = 2 | k = 16 |
|---|---|---|
| `cross_learning` default (effective, from the fitted model) | **True** | **True** |
| `batch_size` default | 256 (set to k here to isolate blocks) | 256 |
| Accepts a 24 h probe beside 336 h series | **yes** | yes |
| Accepts duplicate identical series | **yes** | yes |
| Probe output changes when the co-tenant victim is swapped (mean abs) | 0.061 | 0.015 |
| AUC L1 from the probe's forecast (group) | 0.609 | 0.439 |
| AUC L1, isolated-probe control | 0.470 | 0.577 |
| **AUC L1, residual (group − isolated)** | **0.737** | 0.523 |

(Raw predicted-quantile features, n_test = 150; these are not the model-space features of the main attack, so absolute
AUCs are lower than the main study's — the point is the gap over the control and its decay with k.)

**Conclusion (R1 = confirmed).**
- The default serving mode of AutoGluon's Chronos-2 is `cross_learning=True` with `batch_size=256`; a user who calls the
  product as documented mixes co-tenants by default. (Confirmed both from source — `models/chronos/chronos2.py`
  `_get_default_hyperparameters` — and from the fitted model's effective hyperparameters above.)
- The library accepts both a short (24 h) series and duplicate series, so the sybil and short-probe attacks use only
  inputs the product allows.
- The channel reproduces through the shipped `TimeSeriesPredictor`: at k = 2 the residual decoder reaches AUC 0.74 vs a
  0.47 isolated control, matching the direct-path pair result (0.72–0.78). The per-victim channel dilutes with group
  size in the real library exactly as in the main study (swap influence 0.061 → 0.015 from k = 2 to k = 16).

Evaluation type: `real_system` (the shipped library on recorded data). Reproduce with
`bash scripts/agcheck_setup.sh && <ag-venv>/bin/python scripts/agcheck.py --data <groups.npz> --k 2 --out <dir>`; the
group dumps are produced by `scripts/agcheck_dump.py` in the main venv.
