# CrossLeak experiment data (deposit version {{DATA_VERSION}})

**CrossLeak: co-tenant leakage of in-context cross-learning in time-series foundation models.**
Result bundles of the four pre-registered experiment campaigns, the code that produced each bundle, and the
analysis code at release {{CODE_RELEASE}}.

- Creator: Tran Thanh Lam Nguyen, Faculty of Information Technology, HCMC University of Technology and Engineering
  (HCMUTE), Ho Chi Minh City, Vietnam.
- Code repository: https://github.com/lamnguyentt92/crossleak (release branches `release/vX.Y.Z`).
- Licences: experiment outputs, reports and logs: CC BY 4.0 (`LICENSE-DATA.txt`); code archives: MIT
  (`LICENSE-CODE.txt`). Parts of the outputs are derived from two public datasets that are themselves under CC BY 4.0
  (see "Source data").
- Deposit assembled on {{DATE}} by `scripts/make_deposit.py` (release {{CODE_RELEASE}}, commit {{CODE_COMMIT}}).

## What was studied

Chronos-2 can forecast several series jointly ("cross-learning", group attention). When a forecasting service puts
series of different users into one group, a group member's forecast can depend on the other members' histories. The
campaigns measure (1) the forecasting benefit of cross-learning for users with little history, (2) what a group member
(or an attacker controlling several members) can infer about another member's previous day, and (3) defences: a
public-context group (D1), large groups (D2) and a differentially private prototype context (D4, "DP-Prototype"),
including a causal variant whose public data come only from before the forecast origin (v0.4).
Models: `amazon/chronos-2` and `autogluon/chronos-2-small` (chronos-forecasting 2.3.2), run on CPU.

Every campaign followed a plan frozen by hash before its first run. The plans, freeze files, results documents and independent
re-analysis scripts are in the code archive of {{CODE_RELEASE}} (`prereg/`, `docs/`).
Evaluation types: forecasting, attacks and utility replay real recorded series through the public
models offline (`real_system` data); the grouping service is the project's own code, with a separate check of the
same channel through AutoGluon `TimeSeriesPredictor` (`docs/AGCHECK.md`); edge cost is measured on one virtual machine
(`emulation` of an edge node).

## Files

{{FILES_TABLE}}

`SHA256SUMS` lists the sha256 of every other file; check with `sha256sum -c SHA256SUMS`.
`DEPOSIT_MANIFEST.json` records how the deposit was assembled (where each bundle was found, the git commit of each code
archive, warnings).

The four bundles are byte-identical to the bundles analysed in the results documents (same sha256 as in
`deposit/bundles.json` of the code archive). They were not edited for publication.
Each bundle's `controller_state/git_state.txt` records the commit of the development repository that produced it;
the release listed below contains the same code, and `scripts/check_provenance.py` (code archive of the latest
release) confirms that every file frozen by the campaign's plan is byte-identical to the archived code.

| Bundle | Campaign | Plan | Results | Produced by |
|---|---|---|---|---|
{{BUNDLES_TABLE}}

## Bundle layout

Each bundle is one zip written by the cluster controller:

```
<bundle>/
  RESULT.json              status, timings, exit code of each shard, sha256 of the machine artifacts
  merged_report/           report merged over all machines (Markdown report, summary JSON, CSV tables)
  machines/<artifact>.zip  one zip per machine (shard), containing <artifact>/ with:
    INDEX.json             file list with sha256
    configs/ prereg/       configuration and frozen plans used by the run
    env/                   software and hardware versions
    data_manifest/         source-data checksums and preparation reports (user counts, filters)
    logs/                  every command log of that machine
    reports/               per-machine report
    runs/<experiment>/<cell>/
      result.json          cell configuration, summary statistics, timing
      arrays.npz           per-sample arrays (below)
      DONE                 completion marker
  controller_state/        doctor.json, environment.json, git_state.txt, testbed.json
  controller_logs/ controller_out/ remote_out/   controller and remote process logs
```

Cell directory names are `<kind>__<dataset>__<model>__<key>-<value>...`, e.g.
`attack__eld__c2base__fam-sybil__filler-copy__k-16__n-236__part-3__probe-short__role-train`
(`c2base` = `amazon/chronos-2`, `c2small` = `autogluon/chronos-2-small`; `k` = group size; `part` = slice of a large
cell; `role` = attacker training or test split). Directories under `runs/*_smoke/` are smoke tests run before the
campaign and are not used by any verdict.

## Arrays (`arrays.npz`, numpy `savez_compressed`)

Integer user indices (`user`, `victim`, `attacker`, `pools_*`) index the users of the prepared panel of each dataset
(order reproducible with `crossleak data`); `day` indexes the days of the evaluation split (`days` in the `ref` cell of
the same dataset gives the dates).

| Cell kind | Arrays | Meaning |
|---|---|---|
| `util` (main, v3), `gain` (e0) | `wql_<method>`, `mase_<method>`, `user`, `day` | Weighted quantile loss (and MASE in e0) of the next-24-h forecast per target user-day: `iso` the target's short history alone, `full` the target's full history (reference), `xl<k>` short history in a cross-learning group of k with established users, `d1_<k>` group with public-context windows, `d4_P<P>_eps<ε>` group with DP-Prototype context, `snaive` repeat of the last 24 h |
| `attack` (main, v3), `decode` (e0) | `fxl`, `fiso` (n × 216), `cal` (n × 19), `Y` (n × 24), `dm`, `ev14` (n × 14), `victim`, `attacker`, `day` | Attacker's probe forecast inside the group (`fxl`) and alone (`fiso`): 24 hours × 9 quantiles in model space; calendar features; the victim's last 24 h z-normalised (`Y`, the inference target); the victim's daily means and evening scores over 14 days (label inputs). In e0 (`decode`) the arrays `fxl`, `fiso`, `cal`, `Y`, `victim`, `attacker`, `day` carry `_tr` / `_te` suffixes (training and test samples) |
| `channel` | `i_swap`, `i_group`, `spread`, `attacker`, `victim`, `victim_alt`, `day` | Influence of the group on the attacker's forecast: change when the victim is swapped for another user, change from isolated to grouped, forecast spread |
| `invariance` | `probe_change`, `victim_equiv_err` | For four positive affine changes of the victim's series: largest change of the attacker's forecast, and relative error of the victim's forecast against the transformed original forecast |
| `ref` | `ref_mean`, `ref_even_med(_att)`, `ref_resid_med(_att)`, `n_valid`, `cand`, `days`, `weekday`, `pools_*` | Per-day population references used for labels (`_att`: computed only from attacker-accessible users), candidate profiles from older history (`cand`, users × days × 24, z-normalised) for linkage, and the victim / auxiliary / public / target user pools |
| `dpaudit` | `day_te`, `victim_te` | Test samples of the DP-Prototype attribute audit (statistics in `result.json`) |

`result.json` of each cell holds the summary statistics, configuration and timings; the decoders, labels and
verdicts are computed by the report code from these arrays.

## Source data

- **ELD:** Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C58C86. Licence CC BY 4.0. Used: 2013–2014, 15-min kW averaged to hourly values, clients
  with too many zero readings removed.
- **LifeSnaps:** Yfantidou, S. et al. LifeSnaps dataset, version 4. Zenodo. https://doi.org/10.5281/zenodo.7229547.
  Licence CC BY 4.0. Data descriptor: Yfantidou, S. et al. (2022). LifeSnaps, a 4-month multi-modal dataset capturing
  unobtrusive snapshots of our lives in the wild. Scientific Data 9, 663. https://doi.org/10.1038/s41597-022-01764-x.
  Used: the hourly Fitbit CSV (heart rate, steps).

Changes made to the source data: resampling to hourly values, per-user filtering, z-normalisation of daily profiles,
model forecasts and summary statistics. The `ref` cells contain averages of older normalised hourly daily profiles of all
prepared users, and `attack`/`decode` cells contain normalised 24-h profiles of sampled users; these are derived excerpts of the source
datasets and remain subject to their CC BY 4.0 terms (credit the original creators as above).

Both sources were published as anonymised open data. This deposit adds no new personal data and makes no attempt to
re-identify anyone; user indices are positions in the prepared panel, not the source identifiers.

## Reproducing the reports

1. Unzip the code archive of the release that produced the bundle (or check out its branch), then
   `python3 -m venv .venv && . .venv/bin/activate && pip install -e .` (see its `README.md`).
2. Main study and confirmatory study:
   `crossleak main report --config configs/<main|v3>.yaml --from-zip <bundle>.zip --decoders full --jobs 4 --out <dir>`
   (`--decoders full` also fits the non-linear decoders, which takes longer). The E0 report is regenerated with
   `crossleak e0 report --roots <unpacked machine runs directories>`.
3. Independent re-analysis scripts: `docs/reanalysis/` of {{CODE_RELEASE}} (each script's header gives its command;
   some read the prepared source data, which `crossleak data` downloads and prepares).
4. `sha256sum -c prereg/<PLAN>_FREEZE.sha256` in the code archive checks that the analysis files match the frozen plan.

## Notes

- The controller logs and `controller_state/testbed.json` contain the private (RFC 1918) addresses of the isolated
  testbed machines; no credentials, tokens or personal contact data are stored in the bundles.
- The E0 bundle was produced from a working copy whose `git describe` reports `-dirty`; release v0.1.2 ignores
  file-mode changes, and `controller_state/git_state.txt` in the bundle records the state.
- Negative and inconclusive results are reported in the results documents together with the positive ones.

## How to cite

Cite this deposit by its DOI (shown on the Zenodo record), the code repository, and the two source datasets above.
