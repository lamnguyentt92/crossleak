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
- **D-016** — 2026-11-13 — Main-study plan approved: CL1 scoped to one day of history; per-k thresholds; sybil attacker
  (m = k − 1 and k/2) and a nonlinear decoder; A1 linkage; defences D0, D1, D2, D4 and the history rule; E7 cost on one
  idle VM; decoders fitted in the analysis stage; LifeSnaps kept as the second dataset. — E0 = GO (docs/E0_PILOT_RESULTS.md).
- **D-017** — 2026-11-13 — Sybil filler designs from a synthetic mechanism check (before any main data): fillers
  are never inert; all-NaN / one-reading fillers keep or amplify the victim's influence up to k = 256 (1.0–2.1 × k = 2),
  copies of the probe keep 0.4–1.2 ×, noise probes dilute it (base: 0.41 → 0.02). Primary filler `copy` (series that look
  like ordinary new users; no acceptance assumption); `sparse` = strongest (assumes one-reading series are accepted);
  `noise` = natural dilution. — Decisive check of whether large groups can be defeated by an attacker.
- **D-018** — 2026-11-13 — D4 representation `dayfold` (24-h profile, clip √24) and primary P = 8, chosen by the design
  pilot on ELD 2013 with a rule fixed before running (largest gain at ε = 4, c2base): dayfold P = 8 retained 1.10 / 1.01 of
  the cross-learning gain at ε = 4 / 1 (docs/pilots/D4_DESIGN.md). The small model gained nothing from co-tenants there.
- **D-019** — 2026-11-13 — Primary A2 label L1 is *within-day* (above the day's population median) and the profile metric
  R1 subtracts the day's population mean, so population-wide information (holidays, weather) cannot count as individual
  leakage. Thresholds from the E0 re-analysis with these definitions (docs/pilots/e0_recalibration.json): CL3 AUC ≥ 0.65,
  CL4 Δr1 ≥ 0.10, CL9 top-1 ≥ 0.15. Labels L4 (self-relative evening) and L5 (idiosyncratic: self- and population-relative)
  added as estimands after the smoke DP audit showed that a user's 14-day mean profile alone predicts L1 (AUC 0.94):
  L1 mostly reflects a stable routine, L4/L5 isolate day-level events. — Before any main data.
- **D-020** — 2026-11-13 — Cost-driven scope: decoders for k ≥ 64 only on c2small (k = 256 sybil-copy and sparse);
  c2base decoders up to k = 16 and channel up to k = 256; single-member probes at k = 4 short only; n_train 1,500 for
  k = 4/16, 1,000 for 64, 800 for 256. Estimated 34 machine-hours (6.8 h per VM). — Fits one overnight run.
- **D-021** — 2026-11-13 — CL8 restated before any data as an absolute feasibility criterion (1,000 users per day
  within 15 min, base model, P = 8). D4 forecasts each user with its own P + 1 series, so a "< 10 % overhead per series
  over cross-learning" criterion cannot hold by construction.
- **D-022** — 2026-11-13 — Utility targets from a fixed cold-start subset of the victim pool (30 % ELD, 50 % LifeSnaps);
  established service users = rest of the victim pool + aux pool; D1 uses public users' windows of the same weekday at
  least 56 days away (a public dataset is not collected on the same dates; LifeSnaps has ~1 public user a day). PMData
  not used (D-004 deferred it; LifeSnaps' own public pool suffices for this design).
- **D-023** — 2026-11-13 — v0.2.0 runs the main plan on the same five VMs with `cluster.sh run` (default config now
  `configs/main.yaml`); the cost phase runs on VM-A after the other shards finish; E0 frozen files unchanged and E0 still
  runnable with `--config configs/e0.yaml`.
- **D-024** — 2026-11-13 — Before the freeze: the attacker's training labels L1/L5 use day references computed only
  over users it can access (aux and public pools); test labels use all users. Without this, LifeSnaps training labels
  (no time split) would depend on victim-pool data of the same days. The report also refuses to mix results produced
  under another configuration (CONFIG_MISMATCH). Main plan frozen at 2026-11-13T16:59Z (`prereg/MAIN_FREEZE.sha256`).
- **D-025** — 2026-11-14 — A v0.3 confirmatory study before publishing the final dataset, with three goals:
  (1) equivalence of D1 and D4(eps=1) to cross-learning (TOST, margin 2 pp, 600 targets); (2) a stronger sybil attacker
  (residual features, larger training sets) at k in {16,64,256} to confirm the v0.2 CL5 = NO is not an artefact of a weak
  attacker; (3) a real-system check through AutoGluon.
- **D-026** — 2026-11-14 — v0.3 reuses the frozen main driver and main analysis unchanged (no edit to any MAIN_FREEZE
  file; verified intact), adds only analysis/v3_report.py and configs/v3.yaml; the `v3` driver routes through the main
  runners. The frozen driver always emits ref/dpaudit/invariance/cost cells, so v0.3 sizes dpaudit/invariance down and
  sets cost models empty rather than editing the driver. — Keeps v0.2 reproducible.
- **D-027** — 2026-11-14 — The AutoGluon real-system check runs off-cluster (separate venv; scripts/agcheck*.{py,sh}),
  not in the overnight campaign (`agcheck.run_on_cluster: false`), because autogluon.timeseries is a heavy dependency
  that could break the pinned chronos-forecasting environment on the testbed. Results on real ELD data are in
  docs/AGCHECK.md: cross_learning=True is the shipped default, the library accepts short and duplicate series, and the
  channel reproduces (residual AUC 0.74 vs 0.47 control at k=2, diluting with k).
- **D-028** — 2026-11-14 — v0.3 main plan frozen (prereg/V3_FREEZE.sha256) before any v0.3 data; it also lists the v0.2
  MAIN_FREEZE hashes so one check covers the whole analysis path.
- **D-029** — 2026-11-14 — Licences: code MIT; derived data CC BY 4.0 (as ELD and LifeSnaps).
- **D-030** — 2026-11-14 — v0.3 read: E1 equivalent (ELD ×3, LifeSnaps D1), LifeSnaps D4 better than cross-learning; E2 mixed
  (k = 4 equivalent to zero, k = 16 +1.15 %); S1 refuted (the batch-filling sybil leaks at k = 256 on ELD), which supersedes
  v0.2 CL5 = NO (both reported). Two post hoc analyses, labelled: seasonal naive is not beaten on ELD day-one clients; D4
  on LifeSnaps runs mostly on public centres (51 % of prototypes).
- **D-031** — 2026-11-14 — Data deposit: the analysed bundles are published byte-identical (their sha256 match the
  analysed files; private testbed addresses in controller logs are not redacted, no credentials are present); outputs
  under CC BY 4.0 with credit to ELD and LifeSnaps (both CC BY 4.0), code archives under MIT; data version 1.0.
  `AutogluonModels/` (tool output of scripts/agcheck.py) is left out of the code archives.
- **D-032** — 2026-11-14 — Paper: every number, table and figure is generated by `paper/make_assets.py` from the
  bundles; references checked against primary records (arXiv, Crossref, DataCite).
- **D-033** — 2026-11-14 — A pre-submission check found that the frozen main driver took the D1 public windows at least
  56 days away from the origin in *either* direction and fitted the LifeSnaps D4 centres on public users from any day,
  whereas a deployment can use only past data. A causal-defence study v0.4.0 (prereg/V4_PREREG.md) was planned and
  approved to repeat the equivalence tests with public data from before the origin. Further experiments that were
  considered (base model at k = 256, several training seeds, output rounding as a defence) were not run; they are stated
  as limitations.
- **D-034** — 2026-11-14 — v0.4 read (docs/V4_RESULTS.md). Per prereg/V4_PREREG.md section 5: E1c-1 EQUIVALENT, so
  past-only public context is stated as equivalent to cross-learning on ELD; E1c-2/3 NOT_EQUIVALENT, so the v0.3 ELD D4
  equivalence did not replicate and the D4 claim is narrowed to "keeps most of the gain; −2.34 to +1.97 pp across the
  two campaigns"; E1c-4/5 reported with the caveat of few users (6 LifeSnaps users). The v0.3 verdicts remain reported.
  Independent re-analysis agrees (docs/reanalysis/v4_reanalysis_v0.4.0.*). No post hoc analysis on the v0.4 data.
- **D-035** — 2026-11-14 — A further pre-submission review asked for (1) seasonal naive on the causal-study targets,
  (2) the sensitivity of the D4 result to the sample, (3) narrower equivalence margins and (4) simultaneous intervals.
  These were run post hoc on the deposited bundles (docs/reanalysis/posthoc_r2.*, summary in docs/V4_RESULTS.md).
  Not run: replication of the k = 256 attack with other attacker-training seeds, an end-to-end timing run for 1,000
  users and a service-level test of batch placement; the paper states the k = 256 result as conditional on holding 255
  of the 256 slots, the cost as an estimate from measured median times, and these limits. Data deposit:
  doi:10.5281/zenodo.24224346.

## v0.5 study (2026-11-15)
- **D-036** — The author approved, after the round-r3 review of the manuscript, the work needed for the numbers it
  asks for: four post hoc analyses on the deposited bundles (pooled D4 comparison with unique-user clusters; a ridge
  baseline; a point-forecast-only attacker; exact Gaussian audit ceilings) and the v0.5 study (`prereg/V5_PREREG.md`).
- **D-037** — Before the freeze, the v0.5 code was reviewed independently and re-checked after the fixes: the E5c rule
  counts the five *planned* realisations (a missing one is not positive; NO_DATA verdicts; PARTIAL); the informed
  placement attacker pads once to a block boundary and then uses min(255, remaining budget) meters per victim; E5a
  labels *_SMALL and their reading; E5b timing and memory definitions; two victim populations in the placement
  emulation; `attack.py`, `decode.py`, `config.py` and `v4.py` added to the frozen list. `registry.py` changed after the
  v0.4 freeze only to dispatch the v5 driver; its V4 hash is superseded and the v0.4 results are unaffected. The ridge
  baseline uses distinct training windows, a user-disjoint held-out quarter, five training seeds and, on LifeSnaps,
  only windows before each origin; earlier runs with windows drawn with replacement are superseded and not used.
  The placement emulation reports two informed attackers: one that waits for an honest registration to open a block
  and one that first pads to a block boundary; the plan was re-frozen before any v0.5 data.

## Deviations log
- 2026-11-14, before any v0.4 data — erratum to prereg/V4_PREREG.md section 2 (the frozen file is left unchanged): it
  says v0.4 uses "the same user pools ... and the same cold-start users". The pools (victim/aux/public, pool seed
  20261112) are the same, but the cold-start subset of the victim pool is drawn with the campaign seed
  (`experiments/main.py`, `DS.__init__`), so v0.4 (seed 20261116) has different cold-start users from v0.3, as v0.3 had
  from v0.2 (overlap 13 of 49 ELD users and 7 of 14 LifeSnaps users between v0.2 and v0.3). The confirmatory rules are
  unaffected: all comparisons are paired within v0.4. Found during the pre-submission check; the paper describes the
  redraw (Sect. 6).
- 2026-11-14, observation on the v0.4 data (no procedural deviation) — prereg/V4_PREREG.md section 3 expected 13
  LifeSnaps cold-start users. The origin rule of section 2 (past public windows at least 56 days before the origin and,
  on LifeSnaps, past centres) leaves only 6 of them with an origin (64 feasible days; 172 distinct user-days among the
  600 targets, drawn with replacement). The rules were applied as frozen; the paper states the 6 users (Sects. 6, 8).
