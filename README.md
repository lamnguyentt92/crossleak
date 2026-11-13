# CrossLeak v0.1.2: E0 pilot

This release produced the result bundle `crossleak_v0.1.2_20261113-221812_all-e0.zip` (E0 pilot, plan `prereg/E0_PREREG.md`,
frozen by the matching `*_FREEZE.sha256`). The `main` branch of https://github.com/lamnguyentt92/crossleak has the documentation, the results of every
campaign, the re-analyses, the paper's asset generator and the data-deposit builder.

Requirements: Linux, Python ≥ 3.10, CPU only; internet access to download the two public datasets and the two
Chronos-2 models on first use. The campaign took 6.5 h of wall-clock time on five 4-vCPU virtual machines.

## Set up (each machine)
```bash
git clone -b release/v0.1.2 https://github.com/lamnguyentt92/crossleak.git ~/crossleak-v0.1.2 && cd ~/crossleak-v0.1.2
bash scripts/setup_vm.sh            # own virtualenv, unit tests, model download and conformance checks
```

## Run on one machine
```bash
CROSSLEAK_CONFIG=configs/e0_smoke.yaml bash scripts/run_e0.sh 0 1    # smoke run, a few minutes
bash scripts/run_e0.sh 0 1                                        # the pilot on one machine (resumable)
bash scripts/status.sh 0 1                                        # progress and errors
crossleak e0 report                                               # reports/e0/E0_REPORT.md
```

## Run on five machines
```bash
# on machine i = 0..4:
bash scripts/run_e0.sh $i 5
# then, on the machine that collects the five shard zips in ~/crossleak-artifacts:
bash scripts/bundle_artifacts.sh
```
or, from one controller with root SSH access to the other four machines:
```bash
bash scripts/cluster.sh init        # reads the testbed description (or: init --vm-a IP --vm-b IP --vm-d IP --vm-e IP)
bash scripts/cluster.sh run --smoke
nohup bash scripts/cluster.sh run > ~/crossleak-v0.1.2-cluster.out 2>&1 < /dev/null &
bash scripts/cluster.sh status
```
Every run writes one zip to `~/crossleak-artifacts/` (with `-error` in the name if a stage failed). Re-running the same
command resumes and skips finished cells.
