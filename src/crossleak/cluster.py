"""Run a campaign on all five testbed machines from one of them (the controller, normally VM-C).

Same conventions as the EviCon testbed: the controller runs as root, the five addresses live in a
testbed file (vm_a ... vm_e, ssh_user), and the controller reaches root on the other machines with its
SSH key (BatchMode, no passwords). ``init`` can import EviCon's testbed.json and reuse its keys.

One command does everything (resumable; run it again after an interruption):
    crossleak cluster all [--smoke] [--until HH:MM]
= check -> setup (code + venv + doctor on every machine) -> data (controller downloads once, copies to the
others over the LAN) -> launch one shard per machine -> monitor -> collect every machine's artifact ->
one bundle zip with a merged report.

Paths on every machine: ~/crossleak-v<version> (code, .venv, runs, logs), ~/crossleak-data (shared inputs),
~/crossleak-artifacts (zips). Only these are written; ``stop`` kills only the process group this tool started.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from crossleak.config import TAG, VERSION, artifacts_dir, hostname, load_config, logs_dir, repo_root

log = logging.getLogger("crossleak.cluster")

ROLES = ("a", "b", "c", "d", "e")
VN = dt.timezone(dt.timedelta(hours=7))
DIR = f"crossleak-v{VERSION}"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]


# --------------------------------------------------------------------------- testbed


def testbed_path() -> Path:
    return Path(os.environ.get("CROSSLEAK_TESTBED", Path.home() / "crossleak-testbed.json")).expanduser()


def fake_root() -> Path | None:
    """Test mode: every 'machine' is a directory on this host (CROSSLEAK_CLUSTER_FAKE=<dir>)."""
    v = os.environ.get("CROSSLEAK_CLUSTER_FAKE")
    return Path(v) if v else None


def own_ips() -> set[str]:
    ips = {"127.0.0.1"}
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10).stdout
        ips |= set(out.split())
    except Exception:  # noqa: BLE001
        pass
    return ips


def validate_testbed(cfg: dict, own: set[str] | None = None) -> None:
    seen: dict[str, str] = {}
    for r in ROLES:
        v = str(cfg.get(f"vm_{r}", ""))
        if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", v) or any(int(x) > 255 for x in v.split(".")):
            raise SystemExit(f"testbed: vm_{r} must be an IPv4 address, not {v!r}")
        if v in seen:
            raise SystemExit(f"testbed: vm_{r} and {seen[v]} have the same address {v}")
        seen[v] = f"vm_{r}"
    if own is not None and not (set(seen) & own):
        raise SystemExit(f"testbed: none of the five addresses is an address of this machine ({', '.join(sorted(own - {'127.0.0.1'}))}); run on a testbed machine or write the testbed again")


def evicon_candidates() -> list[Path]:
    pats = ["/root/EVICON_TOIT_*/testbed.json", str(Path.home() / "EVICON_TOIT_*/testbed.json"), str(Path.home() / "evicon*/testbed.json")]
    found: set[Path] = set()
    import glob

    for p in pats:
        found |= {Path(x) for x in glob.glob(p)}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


@dataclass
class Host:
    role: str
    ip: str
    user: str
    local: bool

    @property
    def shard(self) -> int:
        return ROLES.index(self.role)

    @property
    def name(self) -> str:
        return f"vm-{self.role}"


def load_hosts() -> list[Host]:
    fr = fake_root()
    if fr:
        return [Host(r, f"10.255.0.{i + 1}", "fake", r == "c") for i, r in enumerate(ROLES)]
    p = testbed_path()
    if not p.exists():
        raise SystemExit(f"{p} missing: run `bash scripts/cluster.sh init` first")
    cfg = json.loads(p.read_text())
    own = own_ips()
    validate_testbed(cfg, own)
    user = cfg.get("ssh_user", "root")
    return [Host(r, cfg[f"vm_{r}"], user, cfg[f"vm_{r}"] in own) for r in ROLES]


# --------------------------------------------------------------------------- transport


def home_of(h: Host) -> Path | None:
    fr = fake_root()
    if fr:
        return fr / h.role
    return Path.home() if h.local else None


def dir_of(h: Host) -> str:
    """Code directory of this version on host h (relative to its home, or absolute for the controller)."""
    if h.local and not fake_root():
        return str(repo_root())
    return DIR


def _prefix(h: Host) -> str:
    fr = fake_root()
    if not fr:
        return ""
    home = fr / h.role
    d = home / DIR
    return (
        f"export HOME={shlex.quote(str(home))} CROSSLEAK_RUNS={shlex.quote(str(d / 'runs'))} CROSSLEAK_LOGS={shlex.quote(str(d / 'logs'))} "
        f"CROSSLEAK_REPORTS={shlex.quote(str(d / 'reports'))} CROSSLEAK_DATA={shlex.quote(str(home / 'crossleak-data'))} "
        f"CROSSLEAK_ARTIFACTS={shlex.quote(str(home / 'crossleak-artifacts'))}; "
    )


def run(h: Host, cmd: str, timeout: float = 600) -> subprocess.CompletedProcess:
    full = _prefix(h) + cmd
    home = home_of(h)
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)
        return subprocess.run(["bash", "-c", full], cwd=home, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(["ssh", *SSH_OPTS, f"{h.user}@{h.ip}", "bash -c " + shlex.quote(full)], capture_output=True, text=True, timeout=timeout)


def put(h: Host, src: Path, rel_dst: str, timeout: float = 1800) -> None:
    home = home_of(h)
    if home is not None:
        dst = home / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return
    run(h, f"mkdir -p {shlex.quote(str(Path(rel_dst).parent))}")
    r = subprocess.run(["scp", "-q", *SSH_OPTS, str(src), f"{h.user}@{h.ip}:{rel_dst}.part"], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp to {h.name} failed: {r.stderr.strip()}")
    run(h, f"mv -f {shlex.quote(rel_dst + '.part')} {shlex.quote(rel_dst)}")


def get(h: Host, remote_path: str, dst: Path, timeout: float = 1800) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    home = home_of(h)
    if home is not None:
        src = Path(remote_path) if Path(remote_path).is_absolute() else home / remote_path
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst
    r = subprocess.run(["scp", "-q", *SSH_OPTS, f"{h.user}@{h.ip}:{remote_path}", str(dst)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scp from {h.name} failed: {r.stderr.strip()}")
    return dst


def parallel(hosts: list[Host], fn) -> dict[str, object]:
    with ThreadPoolExecutor(len(hosts)) as ex:
        futs = {h.role: ex.submit(fn, h) for h in hosts}
    out = {}
    for r, f in futs.items():
        try:
            out[r] = f.result()
        except Exception as exc:  # noqa: BLE001
            out[r] = exc
    return out


# --------------------------------------------------------------------------- init / check


def cmd_init(vm: dict[str, str] | None, ssh_user: str, evicon: str | None) -> int:
    p = testbed_path()
    if vm and all(vm.get(r) for r in ("a", "b", "d", "e")):
        cfg = {f"vm_{r}": vm[r] for r in ROLES if vm.get(r)}
        if "vm_c" not in cfg:
            ips = sorted(own_ips() - {"127.0.0.1"})
            if not ips:
                raise SystemExit("cannot detect this machine's address; give --vm-c")
            cfg["vm_c"] = ips[0]
        cfg["ssh_user"] = ssh_user
        src = "command line"
    else:
        cands = [Path(evicon)] if evicon else evicon_candidates()
        if not cands:
            raise SystemExit("no EviCon testbed.json found; give the addresses: bash scripts/cluster.sh init --vm-a IP --vm-b IP --vm-d IP --vm-e IP")
        e = json.loads(cands[0].read_text())
        cfg = {f"vm_{r}": e[f"vm_{r}"] for r in ROLES}
        cfg["ssh_user"] = e.get("ssh_user", "root")
        src = str(cands[0])
    validate_testbed(cfg, own_ips())
    if p.exists():
        bak = p.with_name(p.name + ".bak-" + time.strftime("%Y%m%dT%H%M%S"))
        shutil.copy2(p, bak)
        log.info("previous testbed kept as %s", bak)
    p.write_text(json.dumps(cfg, indent=2) + "\n")
    log.info("wrote %s from %s: %s", p, src, cfg)
    return cmd_check()


def cmd_check() -> int:
    hosts = load_hosts()
    probe = (
        'h=$(hostname); . /etc/os-release 2>/dev/null; os="${ID:-?}-${VERSION_ID:-?}"; c=$(nproc); '
        'm=$(awk "/MemTotal/ {printf \\"%.1f\\", \\$2/1048576}" /proc/meminfo); dsk=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9); '
        'py=$(python3 -c "import sys; print(sys.version.split()[0])" 2>/dev/null || echo none); '
        'venv=$(python3 -c "import ensurepip, venv" 2>/dev/null && echo yes || echo NO); '
        'gh=$(curl -sS -m 8 -o /dev/null -w "%{http_code}" https://github.com 2>/dev/null || echo no); '
        'hf=$(curl -sS -m 8 -o /dev/null -w "%{http_code}" https://huggingface.co 2>/dev/null || echo no); '
        f'ok=$(cat {DIR}/.setup_ok 2>/dev/null || echo -); '
        'echo "$h|$os|$c|$m|$dsk|$py|$venv|$gh|$hf|$ok"'
    )
    res = parallel(hosts, lambda h: run(h, probe, timeout=60))
    rc = 0
    lines = [f"{'VM':5} {'address':16} {'hostname':14} {'OS':14} {'CPU':>3} {'RAM':>5} {'disk':>5} {'python':8} {'venv':4} {'github':6} {'hf':4} {'setup':8} notes"]
    for h in hosts:
        r = res[h.role]
        if isinstance(r, Exception) or r.returncode != 0 or "|" not in r.stdout:
            err = r if isinstance(r, Exception) else (r.stderr.strip().splitlines() or ["?"])[-1]
            lines.append(f"{h.name:5} {h.ip:16} NO SSH as {h.user} ({err}); key setup: EviCon `bash deploy/setup/testbed_init.sh keys` or ssh-copy-id {h.user}@{h.ip}")
            rc = 1
            continue
        hn, os_, c, m, dsk, py, venv, gh_, hf, ok = r.stdout.strip().splitlines()[-1].split("|")
        notes = []
        if py == "none" or tuple(int(x) for x in py.split(".")[:2]) < (3, 10):
            notes.append("python >= 3.10 needed")
            rc = 1
        if venv != "yes":
            notes.append("python3-venv missing (setup installs it as root)")
        if dsk.isdigit() and int(dsk) < 5:
            notes.append("disk < 5 GB")
            rc = 1
        if gh_ not in ("200", "301", "302") or hf not in ("200", "301", "302"):
            notes.append("no HTTPS to github/huggingface")
        lines.append(f"{h.name:5} {h.ip:16} {hn[:14]:14} {os_[:14]:14} {c:>3} {m:>5} {dsk:>5} {py:8} {venv:4} {gh_:6} {hf:4} {ok:8} {'; '.join(notes) or 'ok'}" + ("  <- controller" if h.local else ""))
    log.info("check:\n%s", "\n".join(lines))
    return rc


# --------------------------------------------------------------------------- setup and data


def _local_commit() -> str:
    r = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else (repo_root() / "RELEASE_COMMIT").read_text().strip() if (repo_root() / "RELEASE_COMMIT").exists() else "unknown"


def content_changes(root: Path) -> str:
    """Tracked files whose *content* differs from HEAD. File-mode changes (e.g. after `chmod -R +x`) are ignored."""
    r = subprocess.run(["git", "-c", "core.fileMode=false", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git status failed in {root}: {r.stderr.strip()} (the controller's code must be a git clone of the release branch)")
    return r.stdout.rstrip()


def git_state(root: Path) -> str:
    parts = []
    for args in (["rev-parse", "HEAD"], ["describe", "--always", "--dirty"], ["branch", "-vv"], ["status", "--porcelain"], ["-c", "core.fileMode=false", "status", "--porcelain", "--untracked-files=no"], ["diff", "--stat"], ["diff", "--summary"]):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        parts.append(f"$ git {' '.join(args)}\n{r.stdout}{r.stderr}")
    return "\n".join(parts)


def _code_tarball() -> Path:
    root = repo_root()
    dirty = content_changes(root)
    if dirty:
        log.warning("controller checkout differs from its commit in content:\n%s", dirty)
        if not os.environ.get("CROSSLEAK_ALLOW_DIRTY"):
            raise SystemExit(f"the controller's checkout has uncommitted content changes:\n{dirty}\nrefusing to distribute it (restore with `git checkout -- .`, or set CROSSLEAK_ALLOW_DIRTY=1)")
    out = Path(tempfile.gettempdir()) / f"crossleak-{TAG}-code.tar.gz"
    r = subprocess.run(["git", "-C", str(root), "archive", "--format=tar.gz", "-o", str(out), "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git archive failed: {r.stderr}")
    return out


def setup_host(h: Host, tarball: Path | None, commit: str, force: bool, stamp: str) -> str:
    lines = []
    if not h.local or fake_root():
        have = run(h, f"cat {DIR}/RELEASE_COMMIT 2>/dev/null").stdout.strip()
        if have != commit or force:
            put(h, tarball, f"{DIR}/.code.tar.gz")
            r = run(h, f"cd {DIR} && tar -xzf .code.tar.gz && rm -f .code.tar.gz && echo {commit} > RELEASE_COMMIT && echo extracted")
            lines.append(f"code: {r.stdout.strip() or r.stderr.strip()}")
            if r.returncode != 0:
                raise RuntimeError(f"extract failed on {h.name}: {r.stderr}")
        else:
            lines.append("code: up to date")
    d = dir_of(h)
    r = run(h, f"cd {shlex.quote(d)} && bash scripts/setup_vm.sh {'--force' if force else ''}", timeout=3600)
    path = logs_dir() / f"crossleak_{TAG}_cluster-setup-{h.name}_{stamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n--- stdout\n" + r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0 or "[setup] PASS" not in r.stdout:
        raise RuntimeError(f"setup FAILED on {h.name}; log {path}")
    return "PASS"


def cmd_setup(force: bool = False, only: list[str] | None = None) -> int:
    hosts = [h for h in load_hosts() if not only or h.role in only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = _local_commit()
    needs_code = [h for h in hosts if not h.local or fake_root()]
    tarball = _code_tarball() if needs_code else None
    log.info("setup %s on %s (commit %s)", TAG, ", ".join(h.name for h in hosts), commit[:12])
    res = parallel(hosts, lambda h: setup_host(h, tarball, commit, force, stamp))
    rc = 0
    for h in hosts:
        r = res[h.role]
        ok = r == "PASS"
        rc |= 0 if ok else 1
        log.info("setup %s: %s", h.name, "PASS" if ok else f"FAIL ({r})")
    return rc


def cmd_data() -> int:
    from crossleak.data.fetch import SOURCES, fetch, raw_path

    hosts = load_hosts()
    ctrl = next(h for h in hosts if h.local)
    infos = {n: fetch(n) for n in SOURCES}  # on the controller (verified)
    if fake_root():
        for n in SOURCES:
            put(ctrl, raw_path(n), f"crossleak-data/raw/{raw_path(n).name}")

    def copy(h: Host) -> str:
        if h.local and not fake_root():
            return "controller"
        done = []
        for n, info in infos.items():
            rel = f"crossleak-data/raw/{raw_path(n).name}"
            size = run(h, f"stat -c %s {rel} 2>/dev/null || echo 0").stdout.strip()
            if size != str(info["bytes"]):
                put(h, raw_path(n), rel)
                done.append(f"{n} copied")
            else:
                done.append(f"{n} present")
        return ", ".join(done)

    res = parallel(hosts, copy)
    rc = 0
    for h in hosts:
        r = res[h.role]
        if isinstance(r, Exception):
            rc = 1
        log.info("data %s: %s", h.name, r)
    return rc


# --------------------------------------------------------------------------- run, monitor, collect


def budget_until(until: str | None, now: dt.datetime | None = None, margin_min: int = 15) -> tuple[int | None, float | None]:
    """--until HH:MM (Vietnam time, next occurrence within 24 h) -> (budget minutes for the shards, deadline epoch)."""
    if not until:
        return None, None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", until)
    if not m:
        raise SystemExit("--until needs HH:MM")
    now = now or dt.datetime.now(VN)
    target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    minutes = (target - now).total_seconds() / 60.0
    return max(5, int(minutes - margin_min)), target.timestamp()


def experiment_of(config: str) -> str:
    return load_config(config)["experiment"]


def planned_per_shard(config: str, n: int) -> dict[int, int]:
    from crossleak.experiments.registry import planned_per_shard as pps

    return pps(load_config(config), n)


def out_file(exp: str, shard: int) -> str:
    return f"$HOME/crossleak-v{VERSION}-{exp}-s{shard}.out"


def launch(h: Host, config: str, n: int, budget: int | None, phase: str = "main") -> str:
    exp = experiment_of(config)
    d = shlex.quote(dir_of(h))
    env = f"CROSSLEAK_CONFIG={shlex.quote(config)}"
    b = str(budget) if budget else '""'
    outf = out_file(exp, h.shard) if phase == "main" else out_file(exp, h.shard).replace(".out", f"-{phase}.out")
    cmd = (
        f"cd {d} && mkdir -p runs/{exp} && "
        f'if [ -f runs/RUN.pid ] && kill -0 "$(cat runs/RUN.pid)" 2>/dev/null; then echo "ALREADY_RUNNING $(cat runs/RUN.pid)"; '
        f"else rm -f runs/{exp}/RUN_EXIT.json; {env} setsid nohup bash scripts/run_campaign.sh {h.shard} {n} {b} {phase} > {outf} 2>&1 < /dev/null & "
        f'echo $! > runs/RUN.pid; echo "STARTED $!"; fi'
    )
    r = run(h, cmd, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()


def probe_status(h: Host, exp: str) -> dict:
    d = shlex.quote(dir_of(h))
    cmd = (
        f"cd {d} 2>/dev/null || {{ echo 'NODIR'; exit 0; }}; pid=$(cat runs/RUN.pid 2>/dev/null); alive=0; "
        '[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1; '
        f"dn=$(ls runs/{exp}/*/DONE 2>/dev/null | wc -l); fl=$(ls runs/{exp}/*/FAILED 2>/dev/null | wc -l); "
        f"ex=$(cat runs/{exp}/RUN_EXIT.json 2>/dev/null | tr -d '\\n'); "
        "lf=$(ls -t logs/crossleak_*-run-shard* 2>/dev/null | head -n 1); "
        "last=$([ -n \"$lf\" ] && grep -hE 'cell start|cell done|FAILED|ERROR|budget|campaign finished' \"$lf\" | tail -n 1 | cut -c1-150); "
        'echo "$alive|$dn|$fl|$ex|$last"'
    )
    r = run(h, cmd, timeout=60)
    line = (r.stdout.strip().splitlines() or ["?"])[-1]
    if line == "NODIR" or "|" not in line:
        return {"reachable": r.returncode == 0, "alive": False, "done": 0, "failed": 0, "exit": None, "last": line}
    alive, dn, fl, ex, last = line.split("|", 4)
    return {"reachable": True, "alive": alive == "1", "done": int(dn or 0), "failed": int(fl or 0), "exit": json.loads(ex) if ex.strip() else None, "last": last}


def status_table(hosts: list[Host], exp: str, planned: dict[int, int]) -> tuple[str, dict]:
    res = parallel(hosts, lambda h: probe_status(h, exp))
    rows = [f"{'VM':5} {'shard':5} {'state':10} {'done':>9} {'failed':>6}  last"]
    for h in hosts:
        s = res[h.role]
        if isinstance(s, Exception):
            rows.append(f"{h.name:5} {h.shard:5} {'UNREACHABLE':10} {'':>9} {'':>6}  {s}")
            continue
        state = "running" if s["alive"] else ("finished" if s["exit"] else "stopped")
        if s["exit"] and s["exit"].get("rc") not in (0, None):
            state = f"exit={s['exit'].get('rc')}"
        rows.append(f"{h.name:5} {h.shard:5} {state:10} {s['done']:>4}/{planned.get(h.shard, 0):<4} {s['failed']:>6}  {s['last'][-110:]}")
    return "\n".join(rows), res


def cmd_status(config: str) -> int:
    hosts = load_hosts()
    exp = experiment_of(config)
    table, _ = status_table(hosts, exp, planned_per_shard(config, len(hosts)))
    log.info("status %s %s at %s (UTC+7)\n%s", TAG, exp, f"{dt.datetime.now(VN):%a %d/%m %H:%M}", table)
    return 0


def cmd_watch(smoke: bool, interval: int | None, once: bool) -> int:
    """Live dashboard (read-only): scripts/watch.py with this version."""
    args = [sys.executable, str(repo_root() / "scripts" / "watch.py"), "--version", VERSION]
    if smoke:
        args.append("--smoke")
    if interval:
        args += ["--interval", str(interval)]
    if once:
        args.append("--once")
    return subprocess.call(args)


def cmd_stop(config: str) -> int:
    hosts = load_hosts()

    def stop(h: Host) -> str:
        d = shlex.quote(dir_of(h))
        r = run(h, f'cd {d} && pid=$(cat runs/RUN.pid 2>/dev/null) && kill -0 "$pid" 2>/dev/null && kill -TERM -- -"$pid" && echo "stopped $pid" || echo "not running"', timeout=60)
        return r.stdout.strip()

    for h, r in zip(hosts, parallel(hosts, stop).values()):
        log.info("stop %s: %s", h.name, r)
    return 0


def collect(hosts: list[Host], config: str, n: int) -> list[Path]:
    exp = experiment_of(config)
    tag = "smoke" if "smoke" in exp else exp
    dest = artifacts_dir()
    dest.mkdir(parents=True, exist_ok=True)

    def one(h: Host) -> Path:
        s = probe_status(h, exp)
        path = (s["exit"] or {}).get("artifact")
        if not path:
            d = shlex.quote(dir_of(h))
            r = run(h, f"cd {d} && bash scripts/make_artifact.sh {tag}-s{h.shard}of{n}-partial", timeout=900)
            m = re.search(r"^ARTIFACT (\S+)", r.stdout, re.M)
            if not m:
                raise RuntimeError(f"no artifact from {h.name}: {r.stderr[-300:]}")
            path = m.group(1)
        local = dest / Path(path).name
        if h.local and not fake_root():
            return Path(path)
        return get(h, path, local)

    res = parallel(hosts, one)
    out = []
    for h in hosts:
        r = res[h.role]
        if isinstance(r, Exception):
            log.error("collect %s: %s", h.name, r)
        else:
            log.info("collect %s: %s", h.name, r)
            out.append(r)
    return out


def stdout_file() -> Path | None:
    """The file this process's stdout is redirected to (e.g. the nohup .out file), if any."""
    try:
        p = Path(os.readlink("/proc/self/fd/1"))
        return p if p.is_file() else None
    except OSError:
        return None


def fetch_outs(hosts: list[Host], exp: str, dest: Path) -> list[Path]:
    """Best effort: each remote machine's run_campaign.sh console output ($HOME/crossleak-v<version>-<exp>-s<i>[-cost].out)."""
    got = []
    for h in hosts:
        if h.local and not fake_root():
            continue
        for name in (f"crossleak-v{VERSION}-{exp}-s{h.shard}.out", f"crossleak-v{VERSION}-{exp}-s{h.shard}-cost.out"):
            try:
                r = run(h, f"test -f {name} && echo yes || echo no", timeout=30)
                if r.stdout.strip().endswith("yes"):
                    got.append(get(h, name, dest / f"{h.name}_{name}", timeout=120))
            except Exception as exc:  # noqa: BLE001
                log.warning("could not fetch console output %s of %s: %s", name, h.name, exc)
    return got


def controller_extras(exp: str, out_file_path: Path | None) -> list[tuple[Path, str]]:
    """Controller-side files that go into every bundle (success or failure)."""
    items: list[tuple[Path, str]] = []
    for f in sorted(logs_dir().glob(f"crossleak_{TAG}_*")):
        items.append((f, f"controller_logs/{f.name}"))
    outs = set(Path.home().glob(f"crossleak-v{VERSION}-*.out"))
    if out_file_path:
        outs.add(out_file_path)
    for f in sorted(outs):
        if f.is_file():
            items.append((f, f"controller_out/{f.name}"))
    for f, arc in ((testbed_path(), "controller_state/testbed.json"), (repo_root() / "runs" / "doctor.json", "controller_state/doctor.json")):
        if f.is_file():
            items.append((f, arc))
    return items


def make_bundle(zips: list[Path], tag: str, config: str, result: dict, extras: list[tuple[Path, str]], texts: dict[str, str] | None = None) -> Path:
    """ONE zip with: every machine's artifact, a merged report, controller logs and console output, remote console
    output, controller state and RESULT.json. Name: crossleak_<version>_<datetime>_all-<tag>[-error].zip"""
    from crossleak.experiments.registry import merged_report

    exp = experiment_of(config)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ok = result.get("status") in ("ok", "manual")
    name = f"crossleak_{TAG}_{stamp}_all-{tag}" + ("" if ok else "-error")
    out = artifacts_dir() / f"{name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        roots = []
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(tmp / "x", [m for m in zf.namelist() if f"/runs/{exp}/" in m])
            except Exception as exc:  # noqa: BLE001
                log.warning("cannot read %s: %s", z, exc)
        for top in sorted((tmp / "x").glob("*")) if (tmp / "x").exists() else []:
            if (top / "runs" / exp).exists():
                roots.append(top / "runs" / exp)
        report = None
        if roots:
            try:
                jobs = max(1, min(4, os.cpu_count() or 1))
                t0 = time.time()
                report, result["merged_verdict"] = merged_report(roots, load_config(config), tmp / "merged" / exp, decoders=os.environ.get("CROSSLEAK_BUNDLE_DECODERS", "linear"), n_jobs=jobs)
                log.info("merged report in %.1f min", (time.time() - t0) / 60)
            except Exception as exc:  # noqa: BLE001
                log.exception("merged report failed: %s", exc)
        result["artifacts"] = []
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for z in zips:
                zf.write(z, f"{name}/machines/{z.name}", compress_type=zipfile.ZIP_STORED)
                result["artifacts"].append({"file": z.name, "bytes": z.stat().st_size, "sha256": _sha256(z)})
            if report:
                for f in sorted((tmp / "merged" / exp).glob("*")):
                    zf.write(f, f"{name}/merged_report/{f.name}")
            seen = set()
            for f, arc in extras:
                if arc not in seen and Path(f).is_file():
                    zf.write(f, f"{name}/{arc}")
                    seen.add(arc)
            for arc, text in (texts or {}).items():
                zf.writestr(f"{name}/{arc}", text)
            result.update({"bundle": out.name, "version": TAG, "controller": hostname(), "created": stamp, "experiment": exp})
            zf.writestr(f"{name}/RESULT.json", json.dumps(result, indent=2, default=str))
    digest = _sha256(out)
    print(f"BUNDLE {out} {out.stat().st_size / 1e6:.1f}MB sha256={digest}", flush=True)
    log.info("bundle (%s): %s sha256=%s", result.get("status"), out, digest)
    return out


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latest_per_shard(paths: list[Path], tag: str, n: int) -> list[Path]:
    """Newest artifact of each shard: crossleak_<TAG>_<stamp>_<host>_<tag>-s<i>of<n>[-partial].zip (each one is cumulative)."""
    best: dict[int, Path] = {}
    pat = re.compile(rf"^crossleak_{re.escape(TAG)}_(\d{{8}}-\d{{6}})_.+_{re.escape(tag)}-s(\d+)of{n}(-partial)?\.zip$")
    for p in paths:
        m = pat.match(p.name)
        if not m:
            continue
        s = int(m.group(2))
        if s not in best or m.group(1) > pat.match(best[s].name).group(1):
            best[s] = p
    return [best[s] for s in sorted(best)]


def _snapshot_texts() -> dict[str, str]:
    texts = {"controller_state/git_state.txt": git_state(repo_root())}
    try:
        from crossleak.artifact import snapshot

        texts["controller_state/environment.json"] = json.dumps(snapshot(), indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        texts["controller_state/environment.json"] = json.dumps({"error": repr(exc)})
    return texts


def cmd_bundle(config: str, n: int = 5) -> Path:
    """Manual: bundle whatever exists now (newest artifact per shard, logs, console output) into one zip."""
    exp = experiment_of(config)
    tag = "smoke" if "smoke" in exp else exp
    zips = latest_per_shard(sorted(artifacts_dir().glob(f"crossleak_{TAG}_*.zip")), tag, n)
    extras = controller_extras(exp, stdout_file())
    try:
        hosts = load_hosts()
        with tempfile.TemporaryDirectory() as tmp:
            outs = fetch_outs(hosts, exp, Path(tmp))
            extras += [(f, f"remote_out/{f.name}") for f in outs]
            return make_bundle(zips, tag, config, {"status": "manual", "shard_artifacts_found": len(zips)}, extras, _snapshot_texts())
    except SystemExit as exc:
        return make_bundle(zips, tag, config, {"status": "manual", "note": str(exc.code)}, extras, _snapshot_texts())


def _sigterm(signum, frame):  # noqa: ARG001
    raise SystemExit("terminated by SIGTERM")


def cmd_run(config: str, until: str | None, poll: int, skip_setup: bool, logfile: Path | None) -> int:
    """check -> setup -> data -> launch -> monitor -> collect -> ONE zip. The zip is written whatever happens."""
    import signal

    signal.signal(signal.SIGTERM, _sigterm)
    out_path = stdout_file()
    exp = experiment_of(config)
    tag = "smoke" if "smoke" in exp else exp
    result: dict = {"status": "error", "stage": "start", "error": None, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": config}
    hosts: list[Host] = []
    launched = False
    res: dict = {}
    try:
        hosts = load_hosts()
        n = len(hosts)
        budget, deadline = budget_until(until)
        log.info("cluster run %s: %s on %s; budget per shard %s min", TAG, exp, ", ".join(f"{h.name}=shard{h.shard}" for h in hosts), budget or "none")
        result["stage"] = "check"
        if cmd_check() != 0:
            raise SystemExit("check failed: fix SSH/python/disk first (table above)")
        if not skip_setup:
            result["stage"] = "setup"
            if cmd_setup() != 0:
                raise SystemExit(f"setup failed on some machine: see controller_logs/crossleak_{TAG}_cluster-setup-*.log")
        result["stage"] = "data"
        if cmd_data() != 0:
            raise SystemExit("data distribution failed")
        result["stage"] = "launch"
        planned = planned_per_shard(config, n)
        launches = parallel(hosts, lambda h: launch(h, config, n, budget))
        launched = True
        result["launch"] = {h.name: str(launches[h.role]) for h in hosts}
        for h in hosts:
            log.info("launch %s (shard %d): %s", h.name, h.shard, launches[h.role])
        result["stage"] = "monitor"
        t0 = time.time()
        while True:
            time.sleep(poll)
            table, res = status_table(hosts, exp, planned)
            log.info("progress after %.0f min:\n%s", (time.time() - t0) / 60, table)
            alive = [h for h in hosts if not isinstance(res[h.role], Exception) and res[h.role]["alive"]]
            if not alive:
                break
            if deadline and time.time() > deadline + 600:
                log.warning("deadline passed with %s still running; collecting what exists", ", ".join(h.name for h in alive))
                break
        exits = {h.name: (res[h.role]["exit"] or {}).get("rc") if not isinstance(res[h.role], Exception) else "unreachable" for h in hosts}
        result["shard_exit_codes"] = exits
        log.info("main phase finished; shard exit codes: %s", exits)
        from crossleak.experiments.registry import has_cost_phase

        if has_cost_phase(load_config(config)):
            if deadline and time.time() > deadline:
                log.warning("deadline passed: cost phase skipped")
                result["cost_phase"] = "skipped (deadline)"
            else:
                result["stage"] = "cost"
                h0 = next(h for h in hosts if h.shard == 0)
                log.info("cost phase on %s alone (the other machines are idle now)", h0.name)
                result["cost_launch"] = launch(h0, config, n, None, phase="cost")
                t1 = time.time()
                while True:
                    time.sleep(min(poll, 60))
                    s0 = probe_status(h0, exp)
                    if not s0["alive"]:
                        break
                    if time.time() - t1 > 3 * 3600:
                        log.warning("cost phase still running after 3 h; collecting what exists")
                        break
                result["cost_phase_exit"] = (s0.get("exit") or {}).get("rc")
                log.info("cost phase finished (exit %s) in %.1f min", result["cost_phase_exit"], (time.time() - t1) / 60)
        result["stage"] = "done"
        result["status"] = "ok" if all(v == 0 for v in exits.values()) and result.get("cost_phase_exit", 0) in (0, None) else "shard_errors"
        log.info("cluster run finished: %s", result["status"])
    except SystemExit as exc:
        result["error"] = str(exc.code)
        log.error("cluster run stopped at stage %s: %s", result["stage"], exc.code)
    except BaseException as exc:  # noqa: BLE001 - always bundle, even on Ctrl-C
        result["error"] = repr(exc)
        log.exception("cluster run failed at stage %s", result["stage"])
    finally:
        result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        zips: list[Path] = []
        extras: list[tuple[Path, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            if hosts and launched:
                try:
                    got = collect(hosts, config, len(hosts))
                    zips = latest_per_shard(got, tag, len(hosts)) or got
                except BaseException as exc:  # noqa: BLE001
                    log.exception("collect failed: %s", exc)
                try:
                    extras += [(f, f"remote_out/{f.name}") for f in fetch_outs(hosts, exp, Path(tmp))]
                except BaseException as exc:  # noqa: BLE001
                    log.warning("fetching console outputs failed: %s", exc)
            result["shard_artifacts_collected"] = len(zips)
            try:
                path = make_bundle(zips, tag, config, result, controller_extras(exp, out_path) + extras, _snapshot_texts())
                log.info("SEND THIS FILE: %s", path)
            except BaseException as exc:  # noqa: BLE001
                log.exception("could not write the bundle: %s", exc)
    return 0 if result["status"] == "ok" else 1
