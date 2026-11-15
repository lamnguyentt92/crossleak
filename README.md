# CrossLeak

Code, frozen analysis plans and analysis scripts for the paper *Cross-talk between tenants: co-tenant privacy leakage of
Chronos-2 cross-learning and private alternatives for edge forecasting*.

In cross-learning mode, the time-series foundation model Chronos-2 forecasts each series of a batch while attending to
the other series. A service that forecasts many users' series in one batch therefore lets one user's forecast depend on
the others' data. This repository measures that channel (attacks by a co-tenant and by sybils that fill the batch),
the cold-start benefit of cross-learning, and private alternatives: public context (D1), large groups (D2) and a
differentially private prototype context (D4, DP-Prototype). Data: ElectricityLoadDiagrams20112014 (ELD) and LifeSnaps
(Fitbit heart rate). Models: `amazon/chronos-2` and `autogluon/chronos-2-small`, on CPU.

## Campaigns

Each campaign followed a plan frozen by SHA-256 before its first run; its result bundle is deposited unchanged.

| Campaign | Release branch | Plan | Config | Results | Bundle |
|---|---|---|---|---|---|
| E0 pilot | `release/v0.1.2` | `prereg/E0_PREREG.md` | `configs/e0.yaml` | `docs/E0_PILOT_RESULTS.md` | `crossleak_v0.1.2_20261113-221812_all-e0.zip` |
| Main study | `release/v0.2.0` | `prereg/MAIN_PREREG.md` | `configs/main.yaml` | `docs/MAIN_RESULTS.md` | `crossleak_v0.2.0_20261114-094533_all-main.zip` |
| Confirmatory study | `release/v0.3.0` | `prereg/V3_PREREG.md` | `configs/v3.yaml` | `docs/V3_RESULTS.md` | `crossleak_v0.3.0_20261114-185432_all-v3.zip` |
| Causal-defence study | `release/v0.4.0` | `prereg/V4_PREREG.md` | `configs/v4.yaml` | `docs/V4_RESULTS.md` | `crossleak_v0.4.0_20261114-223724_all-v4.zip` |
| v0.5 study (requested in review) | `release/v0.5.0` | `prereg/V5_PREREG.md` | `configs/v5.yaml` | pending | `crossleak_v0.5.0_<datetime>_all-v5.zip` |

The bundles (per-cell results, per-sample arrays, merged reports, frozen plans, logs) are at Zenodo,
[doi:10.5281/zenodo.24224346](https://doi.org/10.5281/zenodo.24224346). Their sha256 values are in `deposit/bundles.json`. Each bundle's
`controller_state/git_state.txt` records the commit of the development repository that produced it; the release
branch above contains the same code (see "Check the frozen files" below). Decisions and deviations: `DECISIONS.md`.
Design, threat model and the DP proof: `docs/DESIGN.md`, `docs/THEORY.md`. AutoGluon check: `docs/AGCHECK.md`.

## Install

Linux, Python ≥ 3.10, CPU only.
```bash
git clone https://github.com/lamnguyentt92/crossleak.git crossleak && cd crossleak
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]" matplotlib
python -m pytest -q
```
(`bash scripts/setup_vm.sh` does the same and also downloads both models and runs the conformance checks.)

## Reproduce the reported results from the bundles (CPU, no model download)

Download the four bundles from Zenodo into `bundles/`, then:
```bash
cd bundles && sha256sum -c <(python3 -c "import json; [print(b['sha256'], b['name']) for b in json.load(open('../deposit/bundles.json'))['bundles']]") && cd ..

# check the frozen files: every plan, config and analysis file recorded in a bundle is byte-identical here
python scripts/check_provenance.py bundles/*.zip

# re-run the pre-registered analyses (verdicts, estimands, tables) from the per-sample outputs
crossleak main report --config configs/v4.yaml --from-zip bundles/crossleak_v0.4.0_*_all-v4.zip --out reports/v4
crossleak main report --config configs/v3.yaml --from-zip bundles/crossleak_v0.3.0_*_all-v3.zip --jobs 4 --out reports/v3
crossleak main report --config configs/main.yaml --from-zip bundles/crossleak_v0.2.0_*_all-main.zip --decoders full --jobs 4 --out reports/main

# every number, table and figure of the paper (paper/generated/)
python paper/make_assets.py --main-zip bundles/crossleak_v0.2.0_*_all-main.zip \
    --v3-zip bundles/crossleak_v0.3.0_*_all-v3.zip --v4-zip bundles/crossleak_v0.4.0_*_all-v4.zip

# independent re-analysis of the causal-defence study (standard library + numpy only)
python docs/reanalysis/v4_reanalysis_v0.4.0.py bundles/crossleak_v0.4.0_*_all-v4.zip

# post hoc analyses reported in the paper (seasonal naive, D4 sensitivity, margins, simultaneous intervals)
python docs/reanalysis/posthoc_r2.py --v3 bundles/crossleak_v0.3.0_*_all-v3.zip --v4 bundles/crossleak_v0.4.0_*_all-v4.zip
```
Verdicts and estimands are recomputed from the per-sample outputs. With a different scikit-learn version, decoder AUCs
can differ in the third decimal (for example 0.604 instead of 0.605); the versions used are recorded in each bundle
(`controller_state/environment.json`). The E0 pilot's merged report is inside its bundle (`merged_report/`). The other re-analysis and post hoc scripts in
`docs/reanalysis/` print their usage at the top of each file; their outputs are stored next to them (`*.out`).

## Re-run a campaign

Use the release branch of the campaign (its README has the exact commands). One machine:
```bash
git clone -b release/v0.4.0 https://github.com/lamnguyentt92/crossleak.git ~/crossleak-v0.4.0 && cd ~/crossleak-v0.4.0
bash scripts/setup_vm.sh                                             # virtualenv, tests, models, conformance checks
CROSSLEAK_CONFIG=configs/v4_smoke.yaml bash scripts/run_campaign.sh 0 1   # smoke run
CROSSLEAK_CONFIG=configs/v4.yaml bash scripts/run_campaign.sh 0 1         # the campaign (resumable)
```
Five machines: `CROSSLEAK_CONFIG=<config> bash scripts/run_campaign.sh <i> 5` on machine `i = 0..4`, then
`crossleak main report --config <config> --from-zip <the five shard zips>` on the machine that collects them; or
`bash scripts/cluster.sh run --config <config>` from one controller with root SSH access to the other machines. Wall-clock time on five 4-vCPU virtual machines: E0
6.5 h, main study 9.3 h, confirmatory study 6.7 h, causal-defence study 1.1 h. Every run writes one zip to
`~/crossleak-artifacts/`.

## v0.5 study (this branch)

Plan `prereg/V5_PREREG.md`, frozen by `prereg/V5_FREEZE.sha256` before any v0.5 data. From a controller with root SSH
access to the five machines:
```bash
git clone -b release/v0.5.0 https://github.com/lamnguyentt92/crossleak.git ~/crossleak-v0.5.0 && cd ~/crossleak-v0.5.0
bash scripts/cluster.sh run --config configs/v5_smoke.yaml        # smoke run on all machines
nohup bash scripts/cluster.sh run --config configs/v5.yaml > ~/crossleak-v0.5.0-cluster.out 2>&1 < /dev/null &
bash scripts/watch.sh --config configs/v5.yaml                    # dashboard; the run ends with one all-v5 zip
crossleak main report --config configs/v5.yaml --from-zip ~/crossleak-artifacts/crossleak_v0.5.0_*_all-v5.zip --jobs 4 --out reports/v5
```
About 55 machine-hours: a main phase of about 10.7 h on five 4-vCPU machines, then the serving run (E5b, cost phase)
of about 1.2 h on one idle machine. A `--until` deadline must leave room for the cost phase. Post hoc analyses on the
deposited bundles (needs `crossleak data all`):
```bash
python docs/reanalysis/posthoc_r3.py --main bundles/crossleak_v0.2.0_*_all-main.zip \
    --v3 bundles/crossleak_v0.3.0_*_all-v3.zip --v4 bundles/crossleak_v0.4.0_*_all-v4.zip
```

## Layout

```
src/crossleak/    data preparation, model wrapper, attacks, defences (defenses/dp_prototype.py), drivers, reports
configs/          one configuration per campaign (and a smoke variant)
prereg/           frozen plans and their SHA-256 freeze files
scripts/          operator scripts (setup, campaign, cluster, dashboard), AutoGluon check, deposit builder, provenance check
docs/             design, DP proof, results of each campaign, pilots, independent re-analyses
paper/            make_assets.py: numbers, tables and figures of the paper from the bundles
deposit/          data card, licences and metadata of the Zenodo deposit (built by scripts/make_deposit.py)
tests/            unit, property and negative-control tests
```

## Licences and data

Code: MIT (`LICENSE`). Experiment outputs: CC BY 4.0. Source data, both CC BY 4.0: ELD (UCI Machine Learning
Repository, doi:10.24432/C58C86) and LifeSnaps (Zenodo, doi:10.5281/zenodo.7229547); they are downloaded by
`crossleak data all` and are not redistributed here. Citation: `CITATION.cff`.
