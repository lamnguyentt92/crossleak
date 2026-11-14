"""v0.3 confirmatory report: defence equivalence (TOST), a stronger sybil attacker (residual features and more
training groups), and a pointer to the off-cluster AutoGluon real-system check.

Reuses the frozen main-study analysis (`crossleak.analysis.main_report`) for loading, the utility table, the linear
attacker and the channel; it adds only the v0.3 pre-registered rules (`prereg/V3_PREREG.md`). Writes
reports/v3/: V3_REPORT.md, v3_summary.json, equivalence.csv, sybil.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossleak.analysis import main_report as M
from crossleak.config import TAG

log = logging.getLogger("crossleak.v3_report")

PRIMARY_MODEL = "c2base"


def _equivalence(results: dict, cfg: dict, n_boot: int) -> dict:
    """TOST on paired gain differences: a 90% CI of gain(a) - gain(b) within +/- margin means EQUIVALENT."""
    eqc = cfg["equivalence"]
    margin = float(eqc["margin"])
    model = eqc.get("model", PRIMARY_MODEL)
    groups: dict = defaultdict(list)
    for r in results.values():
        if r["exp"] == "util" and r["status"] == "ok" and r["model"] == model:
            groups[(r["dataset"], int(r["params"]["ell"]))].append(r)

    def arrays_for(ds, ell):
        parts = sorted(groups.get((ds, ell), []), key=lambda r: r["params"]["part"])
        if not parts:
            return None, None
        arrs = [M.arrays(r) for r in parts]
        W = {k[4:]: np.concatenate([a[k] for a in arrs]) for k in arrs[0] if k.startswith("wql_")}
        users = np.concatenate([a["user"] for a in arrs])
        return W, users

    out = {"margin": margin, "model": model, "pairs": [], "no_benefit": []}
    for spec in eqc["pairs"]:
        W, users = arrays_for(spec["dataset"], spec["ell"])
        row = {"dataset": spec["dataset"], "ell": spec["ell"], "a": spec["a"], "b": spec["b"]}
        if W is None or spec["a"] not in W or spec["b"] not in W or "iso" not in W:
            row["verdict"] = "NO_DATA"
            out["pairs"].append(row)
            continue
        iso, wa, wb = W["iso"], W[spec["a"]], W[spec["b"]]
        ok = np.isfinite(iso) & np.isfinite(wa) & np.isfinite(wb)
        uu, iso, wa, wb = users[ok], iso[ok], wa[ok], wb[ok]

        def diff(i):  # gain(a) - gain(b), paired on the same targets
            return (1 - wa[i].mean() / iso[i].mean()) - (1 - wb[i].mean() / iso[i].mean())

        ci = M._boot(diff, uu, n_boot, 31, alpha=0.10)  # 90% CI == TOST at alpha 0.05
        row.update({"delta": ci["est"], "lo90": ci["lo"], "hi90": ci["hi"],
                    "verdict": "EQUIVALENT" if (ci["lo"] > -margin and ci["hi"] < margin) else "NOT_EQUIVALENT"})
        out["pairs"].append(row)
    for spec in eqc.get("no_benefit", []):
        W, users = arrays_for(spec["dataset"], spec["ell"])
        row = {"dataset": spec["dataset"], "ell": spec["ell"], "config": spec["config"]}
        if W is None or spec["config"] not in W:
            row["verdict"] = "NO_DATA"
            out["no_benefit"].append(row)
            continue
        iso, w = W["iso"], W[spec["config"]]
        ok = np.isfinite(iso) & np.isfinite(w)
        uu, iso, w = users[ok], iso[ok], w[ok]
        ci = M._boot(lambda i: 1 - w[i].mean() / iso[i].mean(), uu, n_boot, 32, alpha=0.10)
        row.update({"gain": ci["est"], "lo90": ci["lo"], "hi90": ci["hi"],
                    "verdict": "EQUIVALENT_NO_BENEFIT" if (ci["lo"] > -margin and ci["hi"] < margin) else "BENEFIT_OR_HARM"})
        out["no_benefit"].append(row)
    return out


def _residual_attacker(results: dict, n_boot: int, n_jobs: int) -> dict:
    """The stronger sybil attacker: residual features (fxl - fiso) + calendar, with the v0.3 (larger) training sets."""
    from crossleak import attack as A
    from crossleak.decode import evening_score
    from crossleak.metrics import auc

    fams: dict = defaultdict(lambda: {"train": [], "test": []})
    for r in results.values():
        if r["exp"] == "attack" and r["status"] == "ok":
            fams[r["family_key"]][r["params"]["role"]].append(r)
    refs: dict = {}
    jobs = []
    for key, rr in sorted(fams.items()):
        if not rr["train"] or not rr["test"]:
            continue
        ds = rr["test"][0]["dataset"]
        if ds not in refs:
            refs[ds] = M._ref(results, ds)
        if refs[ds] is None:
            continue
        jobs.append((key, sorted(rr["train"], key=lambda r: r["params"]["part"]), sorted(rr["test"], key=lambda r: r["params"]["part"]), ds))

    def one(job):
        key, tr, te, ds = job
        T = {k: np.concatenate([M.arrays(r)[k] for r in tr]) for k in ("fxl", "fiso", "cal", "Y", "dm", "ev14", "victim", "day")}
        E = {k: np.concatenate([M.arrays(r)[k] for r in te]) for k in ("fxl", "fiso", "cal", "Y", "dm", "ev14", "victim", "day")}
        ref = refs[ds]
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        Xtr = np.hstack([f32(T["fxl"]) - f32(T["fiso"]), f32(T["cal"])])
        Xte = np.hstack([f32(E["fxl"]) - f32(E["fiso"]), f32(E["cal"])])
        Ctr = np.hstack([f32(T["fiso"]), f32(T["cal"])])
        Cte = np.hstack([f32(E["fiso"]), f32(E["cal"])])
        thr = float(np.median(evening_score(T["Y"])))
        Ltr = A.labels(T["Y"], T["dm"], T["day"], ref["ref_even_med_att"], thr, T["ev14"], ref["ref_resid_med_att"])
        Lte = A.labels(E["Y"], E["dm"], E["day"], ref["ref_even_med"], thr, E["ev14"], ref["ref_resid_med"])
        dx = A.linear_decoder(Xtr, T["Y"], {k_: Ltr[k_] for k_ in ("L1", "L5")}, Xte, 73)
        dc = A.linear_decoder(Ctr, T["Y"], {k_: Ltr[k_] for k_ in ("L1", "L5")}, Cte, 73)
        vic = E["victim"]
        res = {"dataset": ds, "model": te[0]["model"], "fam": te[0]["params"]["fam"], "filler": te[0]["params"]["filler"],
               "k": int(te[0]["params"]["k"]), "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
        for lab in ("L1", "L5"):
            y = Lte[lab]
            ok = y >= 0
            if len(np.unique(y[ok])) < 2 or lab not in dx["scores"]:
                continue
            sx, sc = dx["scores"][lab][ok], dc["scores"][lab][ok]
            yy, vv = y[ok], vic[ok]
            res[f"auc_{lab}"] = M._boot(lambda i: auc(yy[i], sx[i]), vv, n_boot, 74)
            res[f"auc_{lab}_ctrl"] = auc(yy, sc)
            res[f"delta_auc_{lab}"] = M._boot(lambda i: auc(yy[i], sx[i]) - auc(yy[i], sc[i]), vv, n_boot, 75)
        rm = ref["ref_mean"][E["day"]]
        r1x = A.within_day_r(dx["yhat"], E["Y"], rm)
        r1c = A.within_day_r(dc["yhat"], E["Y"], rm)
        ok = np.isfinite(r1x) & np.isfinite(r1c)
        res["r1_median"] = float(np.nanmedian(r1x))
        res["r1_median_ctrl"] = float(np.nanmedian(r1c))
        res["delta_r1"] = M._boot(lambda i: float(np.mean(r1x[ok][i] - r1c[ok][i])), vic[ok], n_boot, 76)
        return key, res

    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed

        pairs = Parallel(n_jobs=n_jobs)(delayed(one)(j) for j in jobs)
    else:
        pairs = [one(j) for j in jobs]
    return dict(pairs)


def _verdicts(cfg: dict, equiv: dict, resid: dict, linear: dict, coverage: dict) -> dict:
    V: dict = {}
    for row in equiv["pairs"]:
        V[f"EQ_{row['dataset']}_{row['a']}_vs_{row['b']}"] = row["verdict"]
    for row in equiv["no_benefit"]:
        V[f"NOBEN_{row['dataset']}_{row['config']}"] = row["verdict"]
    # stronger-attacker verdicts: does the residual attacker beat chance at each k? and does it still fail at 256?
    for key, r in resid.items():
        if "auc_L1" not in r:
            continue
        tag = f"SYB2_{r['dataset']}_{r['model']}_k{r['k']}"
        lo = r["delta_auc_L1"]["lo"]
        V[tag] = ("LEAKS" if lo > 0 else "NOT_DETECTED") + f"_auc{r['auc_L1']['est']:.2f}"
    if coverage.get("config_mismatch", 0) > 0:
        V["overall"] = "CONFIG_MISMATCH"
    elif coverage["invalid_share"] > M.INVALID_MAX:
        V["overall"] = "NOT_INTERPRETABLE"
    else:
        V["overall"] = "COMPLETE" if coverage["missing"] == 0 else "PARTIAL"
    return V


def build_report(roots: list[Path], cfg: dict, decoders: str = "linear", n_jobs: int = 1, out_dir: Path | None = None) -> Path:
    from crossleak.config import reports_dir
    from crossleak.experiments.main import build_cells

    t0 = time.time()
    exp = cfg["experiment"]
    results, failed = M.load_results(roots)
    planned = {c.cell_id for c in build_cells(cfg)}
    ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
    infeasible = sorted(k for k, r in results.items() if r.get("status") == "infeasible")
    missing = sorted(planned - set(results))
    mismatch = sorted(k for k, r in results.items() if cfg.get("_sha256") and r.get("config_sha256") != cfg["_sha256"])
    coverage = {"planned": len(planned), "ok": len(ok), "infeasible": len(infeasible), "failed": len(failed),
                "missing": len(missing), "invalid_share": (len(failed) + len(infeasible)) / max(1, len(planned)),
                "config_mismatch": len(mismatch), "infeasible_cells": infeasible, "failed_cells": [f["cell_id"] for f in failed],
                "missing_cells": missing[:200]}
    n_boot = int(cfg["bootstrap"])
    util = M.analyse_utility(results, cfg, n_boot)
    equiv = _equivalence(results, cfg, n_boot)
    chan = M.analyse_channel(results, n_boot)
    linear = M.analyse_attacks(results, "linear", n_boot, n_jobs)
    resid = _residual_attacker(results, n_boot, n_jobs)
    V = _verdicts(cfg, equiv, resid, linear, coverage)
    agcheck = _load_agcheck(roots)
    summary = {"version": TAG, "experiment": exp, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "freeze_v3": M.check_freeze("V3_FREEZE.sha256"), "coverage": coverage, "verdicts": V,
               "equivalence": equiv, "utility": util, "channel": chan, "attacks_linear": linear,
               "attacks_residual": resid, "agcheck": agcheck, "runtime_min": round((time.time() - t0) / 60, 1)}
    out = out_dir or (reports_dir() / exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v3_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    _csvs(out, equiv, resid)
    (out / "V3_REPORT.md").write_text(_markdown(summary, cfg))
    log.info("v3 report written to %s (%.1f min)", out, (time.time() - t0) / 60)
    return out / "V3_REPORT.md"


def _load_agcheck(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for f in sorted(Path(root).rglob("agcheck__*/result.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:  # noqa: BLE001
                pass
    return out


def _csvs(out: Path, equiv: dict, resid: dict) -> None:
    with open(out / "equivalence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "dataset", "ell", "a", "b", "estimate", "lo90", "hi90", "margin", "verdict"])
        for r in equiv["pairs"]:
            w.writerow(["pair", r["dataset"], r["ell"], r["a"], r["b"], r.get("delta"), r.get("lo90"), r.get("hi90"), equiv["margin"], r["verdict"]])
        for r in equiv["no_benefit"]:
            w.writerow(["no_benefit", r["dataset"], r["ell"], r["config"], "", r.get("gain"), r.get("lo90"), r.get("hi90"), equiv["margin"], r["verdict"]])
    with open(out / "sybil.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "fam", "filler", "k", "n_train", "auc_L1", "lo", "hi", "ctrl", "delta_auc_L1", "auc_L5", "delta_auc_L5", "r1_median", "delta_r1"])
        for r in resid.values():
            g = lambda k, f="est": (r.get(k) or {}).get(f) if isinstance(r.get(k), dict) else r.get(k)  # noqa: E731
            w.writerow([r["dataset"], r["model"], r["fam"], r["filler"], r["k"], r["n_train"], g("auc_L1"), g("auc_L1", "lo"), g("auc_L1", "hi"),
                        r.get("auc_L1_ctrl"), g("delta_auc_L1"), g("auc_L5"), g("delta_auc_L5"), r.get("r1_median"), g("delta_r1")])


def _markdown(s: dict, cfg: dict) -> str:
    _f, _ci = M._f, M._ci
    L = [f"# CrossLeak v0.3 confirmatory report ({s['version']}, {s['created']})", "",
         f"Freeze (v3): **{s['freeze_v3']['status']}**. Cells: {s['coverage']['ok']} ok, {s['coverage']['failed']} failed, "
         f"{s['coverage']['missing']} missing of {s['coverage']['planned']} (invalid share {s['coverage']['invalid_share']:.3f}).", "",
         "## Verdicts", "", "| Rule | Verdict |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["verdicts"].items()]
    L += ["", f"## Defence equivalence (TOST, margin {s['equivalence']['margin']:.2f} gain, model {s['equivalence']['model']}; "
          "EQUIVALENT if the 90% CI of the paired gain difference is within +/- margin)", "",
          "| Dataset | history | a | b | gain(a)-gain(b) | 90% CI | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in s["equivalence"]["pairs"]:
        ci = f"[{_f(r.get('lo90'))}, {_f(r.get('hi90'))}]" if "lo90" in r else "–"
        L.append(f"| {r['dataset']} | {r['ell']} h | {r['a']} | {r['b']} | {_f(r.get('delta'))} | {ci} | {r['verdict']} |")
    L += ["", "**No benefit at 3 days of history (equivalence of the cross-learning gain to 0):**", "",
          "| Dataset | config | gain | 90% CI | Verdict |", "|---|---|---|---|---|"]
    for r in s["equivalence"]["no_benefit"]:
        ci = f"[{_f(r.get('lo90'))}, {_f(r.get('hi90'))}]" if "lo90" in r else "–"
        L.append(f"| {r['dataset']} | {r['config']} | {_f(r.get('gain'))} | {ci} | {r['verdict']} |")
    L += ["", "## Stronger sybil attacker (residual features fxl - fiso; larger training sets; L1 within-day evening)", "",
          "| Dataset | Model | Family | k | n_train | AUC L1 | ctrl | ΔAUC L1 | AUC L5 | Δr1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(s["attacks_residual"].values(), key=lambda r: (r["dataset"], r["model"], r["fam"], r["k"])):
        L.append(f"| {r['dataset']} | {r['model']} | {r['fam']}/{r['filler']} | {r['k']} | {r['n_train']} | {_ci(r.get('auc_L1'))} | "
                 f"{_f(r.get('auc_L1_ctrl'))} | {_ci(r.get('delta_auc_L1'))} | {_f((r.get('auc_L5') or {}).get('est'))} | {_ci(r.get('delta_r1'))} |")
    L += ["", "## Utility (gain vs isolated; same table as the main study, fresh seed, 600 targets)", "",
          "| Dataset | Model | history | Config | Gain | Retained |", "|---|---|---|---|---|---|"]
    for ds, mm in s["utility"].items():
        for m, ee in mm.items():
            for ell, r in sorted(ee.items()):
                for c, v in r["configs"].items():
                    ret = next((x for k, x in v.items() if k.startswith("retained_vs")), None)
                    L.append(f"| {ds} | {m} | {ell} h | {c} | {_ci(v['gain'])} | {_ci(ret, 2)} |")
    if s["attacks_linear"]:
        L += ["", "## Channel ratios and v0.2-style linear attacker (anchor)", "", "| Config | Influence | Ratio |", "|---|---|---|"]
        for k, v in s["channel"].items():
            L.append(f"| {k} | {_ci(v['i_swap_median'], 4)} | {_ci(v.get('ratio_to_single_k2'), 3)} |")
    if s["agcheck"]:
        L += ["", "## AutoGluon real-system check (off-cluster; see docs/AGCHECK.md)", "",
              "| Dataset | k | cross_learning | AUC group | AUC iso ctrl | AUC residual | channel swap |", "|---|---|---|---|---|---|---|"]
        for a in s["agcheck"]:
            hp = a.get("default_hyperparameters", {})
            L.append(f"| {a['dataset']} | {a['k']} | {hp.get('cross_learning')} | {_f(a.get('auc_L1_group'))} | {_f(a.get('auc_L1_iso_control'))} | "
                     f"{_f(a.get('auc_L1_residual'))} | {_f(a.get('channel_mean_abs_probe_change_victim_swap'), 4)} |")
    if s["coverage"]["failed_cells"] or s["coverage"]["infeasible_cells"]:
        L += ["", "## Invalid cells", ""] + [f"- infeasible: {c}" for c in s["coverage"]["infeasible_cells"]] + [f"- failed: {c}" for c in s["coverage"]["failed_cells"]]
    L += ["", f"Analysis runtime: {s['runtime_min']} min."]
    return "\n".join(L) + "\n"
