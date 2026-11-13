"""E0 report: tables and the pre-registered go/no-go verdicts (prereg/E0_PREREG.md).

Reads result.json files from one or more run directories (e.g. unpacked artifacts from
several machines) and writes reports/<experiment>/: E0_REPORT.md, e0_summary.json and CSVs.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from crossleak.config import TAG, repo_root, reports_dir, sha256_file

log = logging.getLogger("crossleak.report")

# Pre-registered thresholds (prereg/E0_PREREG.md section 4). Do not edit after the freeze.
GAIN_YES = 0.05
GAIN_WEAK = 0.02
GAIN_MAX_ELL = 72
GAIN_PRIMARY_MODEL = "c2base"
CHANNEL_YES = 0.05
CHANNEL_K = (2, 4)
INVARIANCE_TOL = 1e-3
INVALID_MAX = 0.10


def check_freeze() -> dict:
    path = repo_root() / "prereg" / "E0_FREEZE.sha256"
    if not path.exists():
        return {"status": "missing", "files": {}}
    files, ok = {}, True
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        f = repo_root() / rel.strip()
        actual = sha256_file(f) if f.exists() else None
        files[rel.strip()] = {"expected": digest, "actual": actual, "match": actual == digest}
        ok &= actual == digest
    return {"status": "match" if ok else "MODIFIED", "files": files}


def load_results(roots: list[Path]) -> tuple[list[dict], list[dict]]:
    results, failed = [], []
    for root in roots:
        for f in sorted(Path(root).rglob("result.json")):
            r = json.loads(f.read_text())
            r["_path"] = str(f)
            results.append(r)
        for f in sorted(Path(root).rglob("FAILED")):
            info = json.loads(f.read_text())
            info["cell_id"] = f.parent.name
            info["_path"] = str(f)
            failed.append(info)
    return results, failed


def _fmt(x, nd=3):
    if x is None:
        return "–"
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(b: dict, nd=3) -> str:
    return f"{_fmt(b.get('est'), nd)} [{_fmt(b.get('lo'), nd)}, {_fmt(b.get('hi'), nd)}]"


def verdicts(results: list[dict], failed: list[dict], n_planned: int | None) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    infeasible = [r["cell_id"] for r in results if r.get("status") == "infeasible"]
    datasets = sorted({r["dataset"] for r in ok if r["exp"] in ("gain", "decode")})
    out: dict = {"datasets": {}, "infeasible_cells": infeasible}
    for ds in datasets:
        g_primary = [r for r in ok if r["exp"] == "gain" and r["dataset"] == ds and r["model"] == GAIN_PRIMARY_MODEL and r["params"]["ell"] <= GAIN_MAX_ELL]
        gain = "NO_DATA"
        if g_primary:
            if any(r["summary"]["gain"]["est"] >= GAIN_YES and r["summary"]["gain"]["lo"] > 0 for r in g_primary):
                gain = "YES"
            elif any(r["summary"]["gain"]["est"] >= GAIN_WEAK and r["summary"]["gain"]["lo"] > 0 for r in g_primary):
                gain = "WEAK"
            else:
                gain = "NO"
        dec = [r for r in ok if r["exp"] == "decode" and r["dataset"] == ds and r["params"]["k"] in CHANNEL_K]
        channel = "NO_DATA"
        if dec:
            los = [_nanmax(r["summary"]["delta_r"]["lo"], r["summary"]["delta_auc"]["lo"]) for r in dec]
            if any(x > CHANNEL_YES for x in los):
                channel = "YES"
            elif any(x > 0 for x in los):
                channel = "AMBIGUOUS"
            else:
                channel = "NO"
        out["datasets"][ds] = {"gain": gain, "channel": channel}
    inv = [r for r in ok if r["exp"] == "invariance"]
    out["invariance"] = "NO_DATA" if not inv else ("OK" if all(r["summary"]["probe_max_abs_change"] < INVARIANCE_TOL for r in inv) else "WARN")
    found = {r["cell_id"] for r in results} | {f["cell_id"] for f in failed}
    total = n_planned if n_planned else len(found)
    n_failed = len({f["cell_id"] for f in failed} - {r["cell_id"] for r in results})
    out["cells_found"], out["cells_planned"] = len(found), total
    out["invalid_fraction"] = n_failed / max(total, 1)
    ch = [v["channel"] for v in out["datasets"].values()]
    ga = [v["gain"] for v in out["datasets"].values()]
    if out["invalid_fraction"] > INVALID_MAX:
        overall = "NOT_INTERPRETABLE"
    elif not ch or all(c == "NO_DATA" for c in ch):
        overall = "INCOMPLETE"
    elif any(c == "YES" for c in ch):
        overall = "GO" if any(g == "YES" for g in ga) else "GO_REFRAME"
    elif any(c == "AMBIGUOUS" for c in ch):
        overall = "WARN_EXTEND"
    else:
        overall = "STOP"
    if overall not in ("NOT_INTERPRETABLE",) and len(found) < total:
        overall = f"PARTIAL_{overall}"  # verdict provisional: not all planned cells are in the merged results
    out["overall"] = overall
    return out


def _nanmax(*xs: float) -> float:
    vals = [x for x in xs if x is not None and x == x]
    return max(vals) if vals else float("-inf")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_report(roots: list[Path], experiment: str = "e0", n_planned: int | None = None) -> Path:
    results, failed = load_results(roots)
    ok = [r for r in results if r.get("status") == "ok"]
    out_dir = reports_dir() / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = check_freeze()
    v = verdicts(results, failed, n_planned)

    gain_rows = [
        {
            "dataset": r["dataset"], "model": r["model"], "scen": r["params"]["scen"], "ell": r["params"]["ell"], "k": r["params"]["k"],
            "n": r["summary"]["n_targets"], "users": r["summary"]["n_users"], "wql_iso": r["summary"]["wql_iso_mean"], "wql_xl": r["summary"]["wql_xl_mean"],
            "wql_full": r["summary"]["wql_full_mean"], "wql_snaive": r["summary"]["wql_snaive_mean"], "gain": r["summary"]["gain"]["est"],
            "gain_lo": r["summary"]["gain"]["lo"], "gain_hi": r["summary"]["gain"]["hi"], "share_improved": r["summary"]["share_items_improved"],
            "mase_iso": r["summary"]["mase_iso_mean"], "mase_xl": r["summary"]["mase_xl_mean"], "host": r["host"],
        }
        for r in ok if r["exp"] == "gain"
    ]
    gain_rows.sort(key=lambda x: (x["dataset"], x["model"], x["scen"], x["ell"], x["k"]))
    dec_rows = [
        {
            "dataset": r["dataset"], "model": r["model"], "k": r["params"]["k"], "probe": r["params"]["probe"], "n_test": r["summary"]["n_test"],
            "victims": r["summary"]["n_test_victims"], "r_ctrl": r["summary"]["r_median_ctrl"], "r_xl": r["summary"]["r_median_xl"], "r_perm": r["summary"]["r_median_perm"],
            "delta_r": r["summary"]["delta_r"]["est"], "delta_r_lo": r["summary"]["delta_r"]["lo"], "delta_r_hi": r["summary"]["delta_r"]["hi"],
            "r_raw_ctrl": r["summary"].get("r_raw_median_ctrl"), "r_raw_xl": r["summary"].get("r_raw_median_xl"),
            "auc_ctrl": r["summary"]["auc_ctrl"], "auc_xl": r["summary"]["auc_xl"]["est"], "auc_perm": r["summary"]["auc_perm"],
            "delta_auc": r["summary"]["delta_auc"]["est"], "delta_auc_lo": r["summary"]["delta_auc"]["lo"], "delta_auc_hi": r["summary"]["delta_auc"]["hi"], "host": r["host"],
        }
        for r in ok if r["exp"] == "decode"
    ]
    dec_rows.sort(key=lambda x: (x["dataset"], x["model"], x["k"], x["probe"]))
    ch_rows = [
        {
            "dataset": r["dataset"], "model": r["model"], "k": r["params"]["k"], "probe": r["params"]["probe"], "n": r["summary"]["n_pairs"],
            "i_swap_median": r["summary"]["i_swap_median"], "i_swap_q90": r["summary"]["i_swap_q90"], "i_group_median": r["summary"]["i_group_median"],
            "spread_median": r["summary"]["spread_median"], "swap_to_spread": r["summary"]["swap_to_spread_median"], "host": r["host"],
        }
        for r in ok if r["exp"] == "channel"
    ]
    ch_rows.sort(key=lambda x: (x["dataset"], x["model"], x["k"], x["probe"]))
    inv_rows = [{"dataset": r["dataset"], "model": r["model"], **r["summary"], "host": r["host"]} for r in ok if r["exp"] == "invariance"]
    speed_rows = [{"model": r["model"], "host": r["host"], "threads": r["summary"]["threads"], **row} for r in ok if r["exp"] == "speed" for row in r["summary"]["rows"]]

    write_csv(out_dir / "gain.csv", gain_rows)
    write_csv(out_dir / "decode.csv", dec_rows)
    write_csv(out_dir / "channel.csv", ch_rows)
    write_csv(out_dir / "invariance.csv", inv_rows)
    write_csv(out_dir / "speed.csv", speed_rows)
    summary = {"version": TAG, "roots": [str(r) for r in roots], "n_results": len(results), "n_ok": len(ok), "n_failed_cells": len({f["cell_id"] for f in failed}), "freeze": freeze, "verdicts": v}
    (out_dir / "e0_summary.json").write_text(json.dumps(summary, indent=2))

    L = [f"# E0 report ({TAG})", "", f"- Coverage: {v['cells_found']} of {v['cells_planned']} planned cells in {', '.join(str(r) for r in roots)}" + (" (**partial: verdict provisional; merge all machines' runs**)" if v['cells_found'] < v['cells_planned'] else ""), f"- Result files: {len(results)} (ok {len(ok)}, infeasible {len(v['infeasible_cells'])}); failed cells: {summary['n_failed_cells']}", f"- Pre-registration freeze: **{freeze['status']}**", ""]
    L += ["## Verdicts (pre-registered rules, prereg/E0_PREREG.md)", "", "| Dataset | Gain (c2base, ell<=72) | Channel (decode, k in {2,4}) |", "|---|---|---|"]
    for ds, vv in v["datasets"].items():
        L.append(f"| {ds} | {vv['gain']} | {vv['channel']} |")
    L += ["", f"- Invariance check: {v['invariance']}", f"- Invalid-cell fraction: {v['invalid_fraction']:.3f} (limit {INVALID_MAX})", f"- **Overall: {v['overall']}**", ""]
    L += ["## Gain of cross-learning (1 - mean WQL_xl / mean WQL_iso, 95% cluster-bootstrap CI)", "", "| ds | model | scen | ell | k | n | WQL iso | WQL xl | WQL full ctx | WQL s-naive | gain [CI] | share improved |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in gain_rows:
        L.append(f"| {r['dataset']} | {r['model']} | {r['scen']} | {r['ell']} | {r['k']} | {r['n']} | {_fmt(r['wql_iso'])} | {_fmt(r['wql_xl'])} | {_fmt(r['wql_full'])} | {_fmt(r['wql_snaive'])} | {_fmt(r['gain'])} [{_fmt(r['gain_lo'])}, {_fmt(r['gain_hi'])}] | {_fmt(r['share_improved'], 2)} |")
    L += ["", "## Decodability (attacker with channel vs no-channel control; r = anomaly correlation, raw r in decode.csv)", "", "| ds | model | k | probe | n test | victims | r ctrl | r xl | r perm | delta r [CI] | AUC ctrl | AUC xl | AUC perm | delta AUC [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in dec_rows:
        L.append(f"| {r['dataset']} | {r['model']} | {r['k']} | {r['probe']} | {r['n_test']} | {r['victims']} | {_fmt(r['r_ctrl'])} | {_fmt(r['r_xl'])} | {_fmt(r['r_perm'])} | {_fmt(r['delta_r'])} [{_fmt(r['delta_r_lo'])}, {_fmt(r['delta_r_hi'])}] | {_fmt(r['auc_ctrl'])} | {_fmt(r['auc_xl'])} | {_fmt(r['auc_perm'])} | {_fmt(r['delta_auc'])} [{_fmt(r['delta_auc_lo'])}, {_fmt(r['delta_auc_hi'])}] |")
    L += ["", "## Channel influence (model output space)", "", "| ds | model | k | probe | n | swap median | swap q90 | group effect median | spread median | swap/spread |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ch_rows:
        L.append(f"| {r['dataset']} | {r['model']} | {r['k']} | {r['probe']} | {r['n']} | {_fmt(r['i_swap_median'], 4)} | {_fmt(r['i_swap_q90'], 4)} | {_fmt(r['i_group_median'], 4)} | {_fmt(r['spread_median'], 3)} | {_fmt(r['swap_to_spread'], 3)} |")
    L += ["", "## Invariance to victim affine transforms", "", "| ds | model | probe max abs change | victim equivariance max rel err |", "|---|---|---|---|"]
    for r in inv_rows:
        L.append(f"| {r['dataset']} | {r['model']} | {r['probe_max_abs_change']:.2e} | {r['victim_equivariance_max_rel_err']:.2e} |")
    L += ["", "## CPU cost (seconds per series, p50)", "", "| host | model | threads | k | ms/series |", "|---|---|---|---|---|"]
    for r in speed_rows:
        L.append(f"| {r['host']} | {r['model']} | {r['threads']} | {r['k']} | {r['sec_per_series_p50'] * 1000:.2f} |")
    if failed:
        L += ["", "## Failed cells", ""] + [f"- {f['cell_id']} (attempts {f.get('attempts')}): {f.get('error')}" for f in failed]
    if v["infeasible_cells"]:
        L += ["", "## Infeasible cells (design limit, not failures)", ""] + [f"- {c}" for c in v["infeasible_cells"]]
    L += ["", "Pilot data: exploratory; used only for the go/no-go decision and power planning (prereg section 1)."]
    path = out_dir / "E0_REPORT.md"
    path.write_text("\n".join(L) + "\n")
    log.info("report written: %s (overall %s)", path, v["overall"])
    return path
