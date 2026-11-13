# Decisions

Design and analysis decisions with their reasons, and the deviation log referred to by the frozen plans in `prereg/`.
Format: ID — date — decision — reason.

- **D-001** — 2026-11-12 — Target venue: Mobile Networks and Applications (MONET), journal article. — Scope of the topical collection on time-series intelligence at the mobile edge.
- **D-002** — 2026-11-12 — Day-ahead setting: origin 00:00, context 336 h, horizon 24 h, hourly data. — Standard energy and wearable daily forecasting; aligns the victim's label day with the calendar.
- **D-003** — 2026-11-12 — A cross-learning group is passed as one k-variate task. — Mathematically identical to `cross_learning=True` on the same k series (verified, max difference 0.0) and lets many independent groups share one batch.
- **D-004** — 2026-11-12 — E0 uses ELD and LifeSnaps (bpm); PMData deferred to v0.2 (public pool for D1). — E0 needs neither the public pool nor PMData.
- **D-005** — 2026-11-12 — User pools victim / aux / public = 50 / 35 / 15 %. ELD is subject- and time-disjoint (attacker training 2013, test 2014); LifeSnaps is subject-disjoint only. — LifeSnaps covers only ~8 months with sparse wear; a time split would leave too few valid users per day. Deviation from the design document's "time-disjoint" wording, recorded here.
- **D-006** — 2026-11-12 — LifeSnaps group sizes capped at 8 (gain k ∈ {4, 8}, channel k ∈ {2, 4, 8}). — At most 12 valid victims on any day (measured).
- **D-007** — 2026-11-12 — The attacker's features include the isolated forecast of its own probe. — It is computable with the public model; leaving it out would understate the attacker and confound the comparison with the control.
- **D-008** — 2026-11-12 — Profile reconstruction is scored by the anomaly correlation (training mean profile subtracted); raw r is secondary. Ridge alpha grid widened to 1e-2..1e7. — Smoke test (n = 120, before the freeze, not under the E0 plan) showed raw r ≈ 0.87 for a constant mean-profile prediction (ceiling effect) and alpha at the grid edge. The change is methodological and was made before any E0 data.
- **D-009** — 2026-11-12 — Code licence MIT.
- **D-010** — 2026-11-12 — Single author; commits are authored by Lam Nguyen.
- **D-011** — 2026-11-12 — Pilot (E0) data are exploratory and support no paper claim. — Pre-registration principle; the main plan re-runs with new seeds.
- **D-012** — 2026-11-12 — v0.1.1 adds a cluster controller (`crossleak cluster`, `scripts/cluster.sh`): one command on the controller sets up all five machines, copies data, runs one shard per machine, monitors, collects and bundles one zip. It reads the testbed description (`testbed.json`) and uses root SSH keys. E0 runs under v0.1.1; none of the files frozen by `prereg/E0_FREEZE.sha256` changed (the operator script now also writes `RUN_EXIT.json` and holds a lock). — One-command overnight runs on the five-machine testbed.
- **D-013** — 2026-11-12 — Releases are published as branches `release/vX.Y.Z`.
- **D-014** — 2026-11-13 — v0.1.2: (a) the controller's "uncommitted changes" check ignores file-mode changes (on the controller every non-script file had its exec bit set, content unchanged, which blocked the run); (b) `cluster run` always writes ONE zip, on success or on failure at any stage (machine artifacts, merged report, controller logs and console output, remote console output, git/environment state, RESULT.json); local setup failures also produce a zip. E0 frozen files unchanged. — Every run, successful or not, must leave one complete record.
- **D-015** — 2026-11-13 — v0.1.3 adds a read-only live dashboard; the E0 run already started with v0.1.2 continues unchanged. Known limitation: CPU cost in E0 is measured while all five VMs share one physical i9-12900H (≈57 ms/series for the small model vs ≈6 ms on an unloaded machine), so it is descriptive only; the edge-cost experiment of the main plan (E7) must run with a single active VM.

## Deviations log
None so far.
