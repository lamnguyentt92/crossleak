# Design (v0.1.0 scope: E0)

## 1. System model
An edge forecasting service runs Chronos-2 for N users (households' meters, wearable users). For a cold-start user it
forms a group of k series and predicts them jointly (cross-learning); each user receives only its own forecast.
In code, one group is one k-variate task; tasks never interact (doctor checks 1 and 6).

## 2. Threat model
Honest server. The adversary controls m >= 1 members of a group (E0: m = 1), chooses their series (probe), sees only
their forecasts, knows the public weights, and has auxiliary data of *other* users. Goals: infer properties of, or
reconstruct, a co-member's previous day. Out of scope: malicious server, timing side channels, training-data leakage.

## 3. Properties of the real component the design relies on
Verified by `crossleak doctor` on each machine before E0 (`runs/doctor.json`):
group = cross-learning; swap changes outputs; NaN padding works; invariance to positive affine transforms of others;
determinism; task independence. If any fails, E0 must not run.

## 4. Components
| Module | Role |
|---|---|
| `data/fetch.py` | Download with retries; sha256/md5 verification; `raw/MANIFEST.json` |
| `data/prepare.py` | ELD: 15-min kW to hourly mean (interval-end timestamps), 2013-2014, zero-fraction filter. LifeSnaps: hourly bpm panel, attribute table |
| `data/panel.py` | Panel (U x T), window validity rules, user pools, day masks, sampler of distinct valid users per origin |
| `model.py` | Chronos-2 wrapper: groups to deciles; exact inverse of the per-series instance norm (model space) |
| `metrics.py` | WQL, MASE, AUC, row-wise Pearson, cluster bootstrap |
| `decode.py` | Ridge/logistic attacker, no-channel control, permuted-label control, anomaly correlation |
| `experiments/e0.py` | Cells (gain, channel, decode, invariance, speed), seeds, shard assignment, resumable driver |
| `analysis/e0_report.py` | Tables and pre-registered verdicts; freeze-hash check; partial-coverage flag |
| `artifact.py` | One zip with everything needed for analysis |
| `doctor.py` | Environment snapshot and conformance checks |

## 5. Data flow and storage
`~/crossleak-data/raw` (immutable, verified) → `~/crossleak-data/processed/*_p1.parquet` (versioned processing) →
`<checkout>/runs/<experiment>/<cell_id>/{result.json, arrays.npz, DONE|FAILED}` → `reports/<experiment>/` →
`~/crossleak-artifacts/crossleak_v<version>_<datetime>_<host>_<suffix>.zip`.
Each version has its own checkout and `.venv`; nothing is shared across versions except immutable inputs.

## 6. Reproducibility
- Pinned: chronos-forecasting 2.3.2; model IDs in the config; full `pip freeze` in every artifact.
- Seeds: `seed` in the config; every cell's seed = crc32(sample key) XOR seed. The sample key excludes the model, so both
  models see identical groups (paired comparison).
- Determinism: CPU inference is deterministic (doctor check 5).
- Hash manifest: `prereg/E0_FREEZE.sha256` covers the plan, the config and every file the verdicts depend on.

## 7. Baselines (E0)
Isolated forecast (cross-learning off) for gain; full-context isolated forecast and seasonal naive as references;
no-channel attacker (isolated probe forecast + calendar) and permuted-label attacker for decodability.
Main-plan baselines (v0.2): public-context cross-learning (D1), larger groups (D2), output noise / subgroup
ensembling (D3), DP-Prototype (D4).

## 8. Failure handling and safety
A failing cell writes `FAILED` (traceback) and the campaign continues; the next run retries it once. A time budget
stops starting new cells. The code only writes inside its checkout, `~/crossleak-data` and `~/crossleak-artifacts`,
and only deletes its own `.part` download files and its own `FAILED` markers.

## 9. Cluster operation (v0.1.1)
The controller (VM-C, root) holds the testbed file (`~/crossleak-testbed.json`, vm_a..vm_e, ssh_user; imported from
EviCon). Shard i runs on the i-th role (a..e). The controller distributes the code of its own commit (`git archive`),
writes `RELEASE_COMMIT` there, runs `setup_vm.sh` on every machine, copies the verified raw data, starts
`run_e0.sh i 5` with setsid/nohup (pid in `runs/RUN.pid`), polls `DONE`/`FAILED` counts and `RUN_EXIT.json`, then
fetches each artifact and writes one bundle with a report merged over all shards. Remote commands touch only
`~/crossleak-v<version>`, `~/crossleak-data`, `~/crossleak-artifacts`; `stop` signals only the recorded process group.
`CROSSLEAK_CLUSTER_FAKE=<dir>` simulates the five machines as directories for testing.
