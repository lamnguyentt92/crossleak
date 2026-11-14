# CrossLeak v0.3.0: confirmatory study

This release produced the result bundle `crossleak_v0.3.0_20261114-185432_all-v3.zip` (confirmatory study, plan `prereg/V3_PREREG.md`,
frozen by the matching `*_FREEZE.sha256`). The `main` branch of https://github.com/lamnguyentt92/crossleak has the documentation, the results of every
campaign, the re-analyses, the paper's asset generator and the data-deposit builder.

Requirements: Linux, Python ≥ 3.10, CPU only; internet access to download the two public datasets and the two
Chronos-2 models on first use. The campaign took 6.7 h of wall-clock time on five 4-vCPU virtual machines.

## Set up (each machine)
```bash
git clone -b release/v0.3.0 https://github.com/lamnguyentt92/crossleak.git ~/crossleak-v0.3.0 && cd ~/crossleak-v0.3.0
bash scripts/setup_vm.sh            # own virtualenv, unit tests, model download and conformance checks
```

## Run on one machine
```bash
CROSSLEAK_CONFIG=configs/v3_smoke.yaml bash scripts/run_campaign.sh 0 1    # smoke run
crossleak main plan --config configs/v3.yaml --nshards 1                         # planned cells and estimated hours
CROSSLEAK_CONFIG=configs/v3.yaml bash scripts/run_campaign.sh 0 1         # the campaign on one machine (resumable)
CROSSLEAK_CONFIG=configs/v3.yaml bash scripts/status.sh 0 1              # progress and errors
```

## Run on five machines
```bash
# on machine i = 0..4 (each writes one shard zip to ~/crossleak-artifacts/):
CROSSLEAK_CONFIG=configs/v3.yaml bash scripts/run_campaign.sh $i 5
# then copy the five shard zips to one machine and merge them into one report:
crossleak main report --config configs/v3.yaml --from-zip ~/crossleak-artifacts/crossleak_v0.3.0_*_v3-s*of5.zip --out reports/v3
```
or, from one controller with root SSH access to the other four machines:
```bash
bash scripts/cluster.sh check
bash scripts/cluster.sh run --config configs/v3_smoke.yaml
nohup bash scripts/cluster.sh run --config configs/v3.yaml > ~/crossleak-v0.3.0-cluster.out 2>&1 < /dev/null &
bash scripts/watch.sh --config configs/v3.yaml                                      # read-only dashboard
```
Every run writes one zip to `~/crossleak-artifacts/` (with `-error` in the name if a stage failed). Re-running the same
command resumes and skips finished cells.

The AutoGluon real-system check runs off-cluster in its own virtualenv: `bash scripts/agcheck_setup.sh`, then the
commands in `docs/AGCHECK.md`.
