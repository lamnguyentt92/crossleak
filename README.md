# CrossLeak v0.4.0: causal-defence study

This release produced the result bundle `crossleak_v0.4.0_20261114-223724_all-v4.zip` (causal-defence study, plan `prereg/V4_PREREG.md`,
frozen by the matching `*_FREEZE.sha256`). The `main` branch of https://github.com/lamnguyentt92/crossleak has the documentation, the results of every
campaign, the re-analyses, the paper's asset generator and the data-deposit builder.

Requirements: Linux, Python ≥ 3.10, CPU only; internet access to download the two public datasets and the two
Chronos-2 models on first use. The campaign took 1.1 h of wall-clock time on five 4-vCPU virtual machines.

## Set up (each machine)
```bash
git clone -b release/v0.4.0 https://github.com/lamnguyentt92/crossleak.git ~/crossleak-v0.4.0 && cd ~/crossleak-v0.4.0
bash scripts/setup_vm.sh            # own virtualenv, unit tests, model download and conformance checks
```

## Run on one machine
```bash
CROSSLEAK_CONFIG=configs/v4_smoke.yaml bash scripts/run_campaign.sh 0 1    # smoke run
crossleak main plan --config configs/v4.yaml --nshards 1                         # planned cells and estimated hours
CROSSLEAK_CONFIG=configs/v4.yaml bash scripts/run_campaign.sh 0 1         # the campaign on one machine (resumable)
CROSSLEAK_CONFIG=configs/v4.yaml bash scripts/status.sh 0 1              # progress and errors
```

## Run on five machines
```bash
# on machine i = 0..4 (each writes one shard zip to ~/crossleak-artifacts/):
CROSSLEAK_CONFIG=configs/v4.yaml bash scripts/run_campaign.sh $i 5
# then copy the five shard zips to one machine and merge them into one report:
crossleak main report --config configs/v4.yaml --from-zip ~/crossleak-artifacts/crossleak_v0.4.0_*_v4-s*of5.zip --out reports/v4
```
or, from one controller with root SSH access to the other four machines:
```bash
bash scripts/cluster.sh check
bash scripts/cluster.sh run --config configs/v4_smoke.yaml
nohup bash scripts/cluster.sh run --config configs/v4.yaml > ~/crossleak-v0.4.0-cluster.out 2>&1 < /dev/null &
bash scripts/watch.sh --config configs/v4.yaml                                      # read-only dashboard
```
Every run writes one zip to `~/crossleak-artifacts/` (with `-error` in the name if a stage failed). Re-running the same
command resumes and skips finished cells.
