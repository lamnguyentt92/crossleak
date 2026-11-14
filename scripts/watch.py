#!/usr/bin/env python3
"""Live, read-only dashboard of a cluster campaign (run on the controller).

Shows for every machine: state, cells done / planned, share of the work done, the cell running now and for how
long, and a rough finish time; exits when the controller has written the bundle (prints its path).
It only reads files over SSH and writes nothing anywhere, so it is safe next to a running campaign.

Run it with the Python of the version being watched (its crossleak package gives the planned work):
    bash scripts/watch.sh            # watch this version's run
    bash scripts/watch.sh 0.1.2      # watch the run of another installed version
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROLES = ("a", "b", "c", "d", "e")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new"]
SEC_PER_UNIT_DEFAULT = 0.06  # measured on the five VMs during the smoke run (all busy)
SPEED_CELL_DEFAULT = {"c2small": 600.0, "c2base": 1800.0}

REMOTE = r'''
import glob, json, os, re, time
d = os.path.expanduser(__DIR__)
exp = __EXP__
out = {"ok": os.path.isdir(d), "now": time.strftime("%Y-%m-%d %H:%M:%S")}
if out["ok"]:
    rd = os.path.join(d, "runs", exp)
    plans = sorted(glob.glob(os.path.join(rd, "plan_*shard*.json")), key=os.path.getmtime)
    cells = []
    for pf in plans:
        for c in json.load(open(pf))["cells"]:
            if c not in cells:
                cells.append(c)
    out["plan"] = cells
    out["phase"] = json.load(open(plans[-1])).get("phase", "main") if plans else None
    done = {}
    for f in glob.glob(os.path.join(rd, "*", "DONE")):
        c = os.path.dirname(f)
        try:
            dur = json.load(open(os.path.join(c, "result.json"))).get("duration_sec")
        except Exception:
            dur = None
        done[os.path.basename(c)] = dur
    out["done"] = done
    out["failed"] = [os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(rd, "*", "FAILED"))]
    alive = False
    try:
        os.kill(int(open(os.path.join(d, "runs", "RUN.pid")).read().strip()), 0)
        alive = True
    except Exception:
        pass
    out["alive"] = alive
    try:
        out["exit"] = json.load(open(os.path.join(rd, "RUN_EXIT.json")))
    except Exception:
        out["exit"] = None
    logs = sorted(glob.glob(os.path.join(d, "logs", "crossleak_*-run-shard*")), key=os.path.getmtime)
    cur, first = None, None
    if logs:
        for line in open(logs[-1], errors="replace"):
            m = re.search(r"^(\S+ \S+),\d+ .*crossleak\.(?:e0|main): cell start (\S+)", line)
            if m:
                cur = [m.group(2), m.group(1)]
                first = first or m.group(1)
            elif cur and ("cell done " + cur[0] in line or "cell FAILED " + cur[0] in line):
                cur = None
    out["current"], out["first_start"] = cur, first
print(json.dumps(out))
'''


def own_ips() -> set[str]:
    ips = {"127.0.0.1"}
    try:
        ips |= set(subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split())
    except Exception:  # noqa: BLE001
        pass
    return ips


def hosts() -> list[tuple[str, str, str, bool]]:
    fake = os.environ.get("CROSSLEAK_CLUSTER_FAKE")
    if fake:
        return [(r, f"fake-{r}", "fake", False) for r in ROLES]
    p = Path(os.environ.get("CROSSLEAK_TESTBED", Path.home() / "crossleak-testbed.json"))
    cfg = json.loads(p.read_text())
    mine = own_ips()
    return [(r, cfg[f"vm_{r}"], cfg.get("ssh_user", "root"), cfg[f"vm_{r}"] in mine) for r in ROLES]


def probe(role: str, ip: str, user: str, local: bool, version: str, exp: str) -> dict:
    code = REMOTE.replace("__DIR__", repr(f"~/crossleak-v{version}")).replace("__EXP__", repr(exp))
    cmd = f"echo {base64.b64encode(code.encode()).decode()} | base64 -d | python3 -"
    fake = os.environ.get("CROSSLEAK_CLUSTER_FAKE")
    try:
        if fake:
            env = {**os.environ, "HOME": str(Path(fake) / role)}
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30, env=env)
        elif local:
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
        else:
            r = subprocess.run([*SSH, f"{user}@{ip}", "bash -c " + shlex.quote(cmd)], capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:80]}


def costs(config: str) -> tuple[dict[str, tuple[str, str, float]], float]:
    """cell id -> (exp, model, estimated cost) from the watched version's own planner, and the default seconds per
    cost unit (main-study costs are seconds; E0 costs are relative units)."""
    from crossleak.config import load_config

    cfg = load_config(config)
    try:
        from crossleak.experiments.registry import cell_costs, driver_name

        return cell_costs(cfg), (1.0 if driver_name(cfg) in ("main", "v3", "v4") else SEC_PER_UNIT_DEFAULT)
    except ImportError:  # versions before 0.2 have only the E0 driver
        from crossleak.experiments.e0 import build_cells

        return {c.cell_id: (c.exp, c.model, c.cost(cfg)) for c in build_cells(cfg)}, SEC_PER_UNIT_DEFAULT


def fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


def parse_ts(s: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") if s else None
    except ValueError:
        return None


def shard_eta(st: dict, cost: dict, sec_per_unit: float, speed_est: dict) -> tuple[float, float, float]:
    """(share of work done, seconds remaining, total work in seconds) for one machine."""
    plan = st.get("plan") or []
    done = st.get("done") or {}
    total = rem = 0.0
    for cid in plan:
        exp, model, c = cost.get(cid, ("?", "c2small", 0.0))
        t = speed_est.get(model, 600.0) if exp == "speed" else c * sec_per_unit
        total += t
        if cid not in done:
            rem += t
    cur = st.get("current")
    if cur and cur[0] in cost:
        now, since = parse_ts(st.get("now")), parse_ts(cur[1])
        if now and since:
            exp, model, c = cost[cur[0]]
            t = speed_est.get(model, 600.0) if exp == "speed" else c * sec_per_unit
            rem -= min(t, (now - since).total_seconds())
    return (1 - rem / total) if total else 0.0, max(rem, 0.0), total


def detect_config(root: str) -> str:
    """The running experiment's config: newest plan file under <root>/runs (root is crossleak-v<version>)."""
    plans = sorted(glob.glob(os.path.join(root, "runs", "*", "plan_*shard*.json")), key=os.path.getmtime)
    if plans:
        return f"configs/{os.path.basename(os.path.dirname(plans[-1]))}.yaml"
    return "configs/main.yaml" if Path(root, "configs/main.yaml").exists() else "configs/e0.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--config", default=None, help="default: configs/main.yaml (versions before 0.2: configs/e0.yaml)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--out", default=None, help="controller console file (default ~/crossleak-v<version>-cluster.out)")
    a = ap.parse_args()
    import yaml

    config = a.config or detect_config(".")
    if a.smoke and not config.endswith("_smoke.yaml"):
        config = config.replace(".yaml", "_smoke.yaml")
    exp = yaml.safe_load(open(config))["experiment"]
    cost, spu_default = costs(config)
    out_file = Path(a.out or Path.home() / f"crossleak-v{a.version}-cluster.out")
    hs = hosts()
    while True:
        with ThreadPoolExecutor(len(hs)) as ex:
            sts = list(ex.map(lambda h: probe(*h, a.version, exp), hs))
        # calibrate speed from finished cells anywhere in the cluster
        units = secs = 0.0
        speed_done: dict[str, list[float]] = {}
        for st in sts:
            for cid, dur in (st.get("done") or {}).items():
                if cid in cost and dur:
                    e, m, c = cost[cid]
                    if e == "speed":
                        speed_done.setdefault(m, []).append(dur)
                    elif c > 0:
                        units += c
                        secs += dur
        spu = secs / units if units > 0 else spu_default
        speed_est = {m: sorted(v)[len(v) // 2] for m, v in speed_done.items()}
        speed_est = {**SPEED_CELL_DEFAULT, **speed_est}
        now = dt.datetime.now()
        lines = [f"CrossLeak v{a.version} {exp} — {now:%a %d/%m %H:%M:%S}   (refresh {a.interval}s, Ctrl-C to quit; read-only)", ""]
        lines.append(f"{'VM':5} {'state':9} {'cells':>7} {'work':>5} {'fail':>4}  {'running now':52} {'for':>8}  {'ETA':>5}")
        etas, all_done, any_alive, tot_done, tot_plan = [], True, False, 0, 0
        for (role, *_), st in zip(hs, sts):
            if not st.get("ok"):
                lines.append(f"{'vm-' + role:5} {'no-dir' if 'error' not in st else 'ERROR':9}  {st.get('error', 'version not installed')}")
                all_done = False
                continue
            plan, done = st.get("plan") or [], st.get("done") or {}
            nd = sum(1 for c in plan if c in done)
            tot_done += nd
            tot_plan += len(plan)
            state = "running" if st.get("alive") else ("finished" if st.get("exit") else ("waiting" if not plan else "stopped"))
            if st.get("exit") and st["exit"].get("rc") not in (0, None):
                state = f"exit={st['exit']['rc']}"
            any_alive |= bool(st.get("alive"))
            all_done &= bool(st.get("exit")) and not st.get("alive")
            frac, rem, _ = shard_eta(st, cost, spu, speed_est)
            cur = st.get("current")
            since = parse_ts(cur[1]) if cur else None
            nowr = parse_ts(st.get("now")) or now
            running = cur[0] if cur else "-"
            dur = fmt_dur((nowr - since).total_seconds()) if since else ""
            eta = "" if not st.get("alive") else (now + dt.timedelta(seconds=rem)).strftime("%H:%M")
            if st.get("alive"):
                etas.append(rem)
            lines.append(f"{'vm-' + role:5} {state:9} {nd:>3}/{len(plan):<3} {frac * 100:>4.0f}% {len(st.get('failed') or []):>4}  {running[:52]:52} {dur:>8}  {eta:>5}")
        lines.append("")
        if etas:
            finish = now + dt.timedelta(seconds=max(etas))
            lines.append(f"Overall: {tot_done}/{tot_plan} cells. Expected finish of the last machine: ~{finish:%H:%M} "
                         f"(rough; actual/estimated time so far = {spu / spu_default:.2f}).")
        phases = {st.get("phase") for st in sts if st.get("alive")}
        if "cost" in phases:
            lines.append("Cost phase: one machine measures latency alone (the others are idle on purpose).")
        sent = None
        if out_file.exists():
            for ln in out_file.read_text(errors="replace").splitlines():
                if "SEND THIS FILE:" in ln:
                    sent = ln.split("SEND THIS FILE:", 1)[1].strip()
        if sent:
            lines.append(f"FINISHED. Send this file: {sent}")
        elif all_done and not any_alive:
            lines.append("All machines finished; the controller is collecting artifacts and writing the zip ...")
        if not a.once and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print("\n".join(lines), flush=True)
        if a.once or sent:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
