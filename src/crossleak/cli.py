"""Command line interface: crossleak <command> ..."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from crossleak.config import TAG, load_config, runs_dir

log = logging.getLogger("crossleak.cli")


def _cmd_doctor(a) -> int:
    from crossleak.doctor import run_doctor

    return 0 if run_doctor(load_config(a.config), quick=a.quick) else 1


def _cmd_data(a) -> int:
    from crossleak.data.fetch import fetch
    from crossleak.data.prepare import PREPARERS

    names = ["eld", "lifesnaps"] if a.dataset == "all" else [a.dataset]
    for n in names:
        if a.action in ("fetch", "all"):
            fetch(n)
        if a.action in ("prepare", "all"):
            PREPARERS[n](force=a.force)
    return 0


def _cmd_e0(a) -> int:
    from crossleak.experiments.e0 import assign_shards, build_cells, cell_status, run_campaign

    cfg = load_config(a.config)
    if a.action == "plan":
        cells = build_cells(cfg)
        assign = assign_shards(cells, cfg, a.nshards)
        per = {s: [] for s in range(a.nshards)}
        for c in cells:
            for s in assign[c.cell_id]:
                per[s].append(c)
        for s, cs in per.items():
            cost = sum(c.cost(cfg) for c in cs)
            print(f"shard {s}: {len(cs)} cells, relative cost {cost:,.0f}")
        print(f"total cells: {len(cells)}")
        return 0
    if a.action == "run":
        only = a.only.split(",") if a.only else None
        counts = run_campaign(cfg, a.shard, a.nshards, a.budget_min, only)
        return 0 if counts["failed"] == 0 else 2
    if a.action == "status":
        cells = build_cells(cfg)
        assign = assign_shards(cells, cfg, a.nshards)
        mine = [c for c in cells if a.shard in assign[c.cell_id]]
        st = {"done": 0, "failed": 0, "todo": 0}
        for c in mine:
            st[cell_status(cfg, c)] += 1
        print(f"{TAG} {cfg['experiment']} shard {a.shard}/{a.nshards}: {st} of {len(mine)} cells")
        for c in mine:
            if cell_status(cfg, c) == "failed":
                info = json.loads((runs_dir() / cfg["experiment"] / c.cell_id / "FAILED").read_text())
                print(f"  FAILED {c.cell_id}: {info.get('error')}")
        return 0
    if a.action == "report":
        from pathlib import Path

        from crossleak.analysis.e0_report import build_report

        roots = [Path(r) for r in a.roots] if a.roots else [runs_dir() / cfg["experiment"]]
        n_planned = len({c.cell_id for c in build_cells(cfg)})
        path = build_report(roots, cfg["experiment"], n_planned)
        print(path.read_text())
        return 0
    return 1


def _cmd_cluster(a) -> int:
    from crossleak import cluster as C

    config = "configs/e0_smoke.yaml" if a.smoke else a.config
    if a.action == "init":
        vm = {r: getattr(a, f"vm_{r}") for r in "abcde"}
        return C.cmd_init(vm if any(vm.values()) else None, a.ssh_user, a.evicon)
    if a.action == "check":
        return C.cmd_check()
    if a.action == "setup":
        return C.cmd_setup(force=a.force, only=a.only.split(",") if a.only else None)
    if a.action == "data":
        return C.cmd_data()
    if a.action in ("run", "all"):
        poll = a.poll or (30 if a.smoke else 120)
        return C.cmd_run(config, a.until, poll, a.skip_setup, getattr(a, "_logfile", None))
    if a.action == "status":
        return C.cmd_status(config)
    if a.action == "stop":
        return C.cmd_stop(config)
    if a.action == "collect":
        hosts = C.load_hosts()
        zips = C.collect(hosts, config, len(hosts))
        C.cmd_bundle(config, len(hosts))
        return 0 if len(zips) == len(hosts) else 1
    if a.action == "bundle":
        C.cmd_bundle(config)
        return 0
    return 1


def _cmd_artifact(a) -> int:
    from crossleak.artifact import make_artifact

    make_artifact(a.suffix)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crossleak", description=f"CrossLeak {TAG}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment report and conformance checks")
    d.add_argument("--config", default="configs/e0.yaml")
    d.add_argument("--quick", action="store_true", help="skip model download and conformance checks")
    d.set_defaults(fn=_cmd_doctor)

    dd = sub.add_parser("data", help="download and prepare datasets")
    dd.add_argument("action", choices=["fetch", "prepare", "all"])
    dd.add_argument("--dataset", default="all", choices=["all", "eld", "lifesnaps"])
    dd.add_argument("--force", action="store_true", help="re-run preparation")
    dd.set_defaults(fn=_cmd_data)

    e = sub.add_parser("e0", help="pilot / killer experiment")
    e.add_argument("action", choices=["plan", "run", "status", "report"])
    e.add_argument("--config", default="configs/e0.yaml")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--nshards", type=int, default=1)
    e.add_argument("--budget-min", type=float, default=None, help="stop starting new cells after this many minutes")
    e.add_argument("--only", default=None, help="comma-separated experiment kinds, e.g. decode,gain")
    e.add_argument("--roots", nargs="*", help="report: run directories to merge (default: this machine)")
    e.set_defaults(fn=_cmd_e0)

    ar = sub.add_parser("artifact", help="bundle results for analysis into one zip")
    ar.add_argument("--suffix", default=None)
    ar.set_defaults(fn=_cmd_artifact)

    c = sub.add_parser("cluster", help="run on all testbed machines from this one (controller)")
    c.add_argument("action", choices=["init", "check", "setup", "data", "run", "all", "status", "stop", "collect", "bundle"])
    c.add_argument("--config", default="configs/e0.yaml")
    c.add_argument("--smoke", action="store_true", help="use configs/e0_smoke.yaml")
    c.add_argument("--until", default=None, help="HH:MM Vietnam time: shards stop starting new cells about 15 min before")
    c.add_argument("--poll", type=int, default=None, help="seconds between progress checks (default 120, smoke 30)")
    c.add_argument("--skip-setup", action="store_true")
    c.add_argument("--force", action="store_true", help="setup: redistribute code and rebuild venvs")
    c.add_argument("--only", default=None, help="setup: comma-separated roles, e.g. a,b")
    for r in "abcde":
        c.add_argument(f"--vm-{r}", dest=f"vm_{r}", default=None)
    c.add_argument("--ssh-user", default="root")
    c.add_argument("--evicon", default=None, help="init: path of an EviCon testbed.json")
    c.set_defaults(fn=_cmd_cluster)

    a = p.parse_args(argv)
    from crossleak.logutil import setup_logging

    name = a.cmd if a.cmd not in ("e0", "cluster") else f"{a.cmd}-{a.action}"
    if a.cmd == "e0" and a.action == "run":
        name += f"-shard{a.shard}of{a.nshards}"
    if a.cmd == "cluster" and a.smoke:
        name += "-smoke"
    a._logfile = setup_logging(name)
    log.info("command: %s", " ".join(sys.argv))
    try:
        return int(a.fn(a))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            log.error("%s", exc.code)
            return 1
        return int(exc.code or 0)
    except Exception:
        log.exception("command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
