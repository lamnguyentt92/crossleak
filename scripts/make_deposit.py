#!/usr/bin/env python3
"""Assemble the Zenodo data deposit of CrossLeak (standard library only; no virtual environment needed).

On the controller that holds the result bundles (VM-C):
    python3 scripts/make_deposit.py
    python3 scripts/make_deposit.py --artifacts ~/crossleak-artifacts --out ~/crossleak-deposit --bundle /path/x.zip

Steps:
  1. find the analysed bundles listed in deposit/bundles.json by size and sha256 (file names may differ);
  2. copy them unchanged into <out>/crossleak-deposit_v<version>_<datetime>/upload/ and re-check the sha256;
  3. archive the code of each release that produced a bundle, and of this release (git archive of each commit);
  4. write README_DATA.md (tables generated from the files), LICENSE-DATA.txt, LICENSE-CODE.txt, CITATION.cff,
     zenodo_metadata.json, DEPOSIT_MANIFEST.json and SHA256SUMS;
  5. success or error, pack the log and every output except the large bundles into
     <out>/crossleak_v<version>_<datetime>_deposit-check.zip (the file to send back for checking).
The last line printed starts with "DEPOSIT OK", "DEPOSIT INCOMPLETE" or "DEPOSIT FAILED".
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_VERSION = "1.0"
# Tool output committed by mistake in release v0.3.0 (AutoGluon working directory of scripts/agcheck.py, pickled
# inputs); untracked from v0.3.1 on. Left out of the code archives: it is neither source nor a result.
ARCHIVE_EXCLUDE = ("AutogluonModels",)
log = logging.getLogger("crossleak.deposit")


class DepositError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_version(repo: Path) -> str:
    text = (repo / "src" / "crossleak" / "__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        raise DepositError("cannot read __version__ from src/crossleak/__init__.py")
    return m.group(1)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def has_commit(repo: Path, commit: str) -> bool:
    return git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0


def load_expected(repo: Path) -> list[dict]:
    data = json.loads((repo / "deposit" / "bundles.json").read_text())
    bundles = data["bundles"]
    for b in bundles:
        for key in ("name", "sha256", "bytes", "experiment", "release", "commit"):
            if key not in b:
                raise DepositError(f"deposit/bundles.json: entry {b.get('name')} lacks '{key}'")
    return bundles


def candidate_files(dirs: list[Path], extra: list[Path]) -> list[Path]:
    seen, out = set(), []
    paths = list(extra)
    for d in dirs:
        if d.is_dir():
            paths += sorted(d.glob("*.zip")) + sorted(d.glob("*/*.zip"))
    for p in paths:
        p = p.expanduser()
        if not p.is_file():
            continue
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_bundles(expected: list[dict], candidates: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for c in candidates:
        size = c.stat().st_size
        todo = [e for e in expected if e["bytes"] == size and e["name"] not in found]
        if not todo:
            continue
        h = sha256(c)
        for e in todo:
            if h == e["sha256"]:
                found[e["name"]] = c
                log.info("found %s at %s", e["name"], c)
    return found


def archive_code(repo: Path, commit: str, release: str, branch: str | None, dest: Path, warnings: list[str]) -> str | None:
    if not has_commit(repo, commit) and branch:
        log.info("commit %s not present; fetching %s from origin", commit[:7], branch)
        r = git(repo, "fetch", "--quiet", "origin", branch)
        if r.returncode != 0:
            log.warning("git fetch %s failed: %s", branch, r.stderr.strip())
    if not has_commit(repo, commit):
        msg = f"code of {release} (commit {commit[:7]}) not archived: commit not in this clone; it stays on GitHub branch {branch}"
        log.warning(msg)
        warnings.append(msg)
        return None
    name = f"crossleak-code-{release}-{commit[:7]}.zip"
    excludes = [f":(exclude){p}" for p in ARCHIVE_EXCLUDE]
    r = git(repo, "archive", "--format=zip", f"--prefix=crossleak-{release}/", "-o", str(dest / name), commit, "--", ".", *excludes)
    if r.returncode != 0:
        raise DepositError(f"git archive {commit[:7]} failed: {r.stderr.strip()}")
    log.info("archived code %s (%s) -> %s", release, commit[:7], name)
    return name


def file_description(name: str, bundles: list[dict], code: dict[str, dict]) -> str:
    for b in bundles:
        if b["name"] == name:
            return f"Result bundle: {b['title']} (`{b['experiment']}`), produced by release {b['release']}"
    if name in code:
        c = code[name]
        return f"Source code of release {c['release']} (commit {c['commit'][:7]}; MIT)"
    return {
        "README_DATA.md": "This description",
        "LICENSE-DATA.txt": "Licence of the experiment outputs (CC BY 4.0) and third-party credits",
        "LICENSE-CODE.txt": "Licence of the code archives (MIT)",
        "CITATION.cff": "Citation metadata of the software",
        "zenodo_metadata.json": "Record metadata (Zenodo deposit format)",
        "DEPOSIT_MANIFEST.json": "How this deposit was assembled",
        "SHA256SUMS": "sha256 of every other file",
    }.get(name, "")


def render_readme(template: str, *, files: list[str], upload: Path, bundles: list[dict], code: dict[str, dict],
                  release: str, commit: str, date: str) -> str:
    rows = ["| File | Bytes | Content |", "|---|---:|---|"]
    for name in files:
        p = upload / name
        size = f"{p.stat().st_size:,}" if p.exists() and name not in ("README_DATA.md", "SHA256SUMS", "DEPOSIT_MANIFEST.json") else "—"
        rows.append(f"| `{name}` | {size} | {file_description(name, bundles, code)} |")
    brows = []
    for b in bundles:
        rec = b.get("recorded_commit", b["commit"])
        made = f"{b['release']} (`{b['branch']}`, {b['commit'][:7]}"
        made += f"; recorded in the bundle as {rec[:7]})" if rec != b["commit"] else ")"
        brows.append(f"| `{b['name']}` | {b['title']} | `{b.get('prereg', '')}` | `{b.get('results_doc', '')}` | {made} |")
    out = template
    for key, val in {"{{FILES_TABLE}}": "\n".join(rows), "{{BUNDLES_TABLE}}": "\n".join(brows),
                     "{{DATA_VERSION}}": DATA_VERSION, "{{CODE_RELEASE}}": release, "{{CODE_COMMIT}}": commit[:7],
                     "{{DATE}}": date}.items():
        out = out.replace(key, val)
    left = re.findall(r"\{\{[A-Z_]+\}\}", out)
    if left:
        raise DepositError(f"README_DATA.md: unfilled placeholders {left}")
    return out


def write_check_zip(dep: Path, out_zip: Path, skip: set[str]) -> None:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dep.rglob("*")):
            if p.is_file() and p.name not in skip:
                zf.write(p, p.relative_to(dep.parent))


def build(args: argparse.Namespace, dep: Path, version: str, stamp: str, state: dict) -> None:
    repo = Path(args.repo).resolve()
    upload = dep / "upload"
    upload.mkdir(parents=True)
    warnings: list[str] = state["warnings"]

    head = git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise DepositError(f"{repo} is not a git clone: {head.stderr.strip()}")
    head = head.stdout.strip()
    state["head"] = head
    if git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        msg = "working tree has uncommitted changes; the code archive of this release uses the committed HEAD"
        log.warning(msg)
        warnings.append(msg)

    expected = load_expected(repo)
    dirs = [Path(d).expanduser() for d in args.artifacts]
    cands = candidate_files(dirs, [Path(b) for b in args.bundle])
    log.info("checking %d zip files in %s%s", len(cands), ", ".join(map(str, dirs)),
             f" plus {len(args.bundle)} given with --bundle" if args.bundle else "")
    found = find_bundles(expected, cands)
    missing = [e["name"] for e in expected if e["name"] not in found]
    state["missing"] = missing
    if missing:
        msg = "bundles not found (copy them next to the others or pass --bundle PATH): " + ", ".join(missing)
        if not args.allow_missing:
            raise DepositError(msg)
        log.warning(msg)
        warnings.append(msg)

    bundles_used = [e for e in expected if e["name"] in found]
    for e in bundles_used:
        dst = upload / e["name"]
        log.info("copying %s (%s bytes)", e["name"], f"{e['bytes']:,}")
        shutil.copyfile(found[e["name"]], dst)
        if sha256(dst) != e["sha256"]:
            raise DepositError(f"sha256 of the copy of {e['name']} differs from the source")
        state["bundles"].append({"name": e["name"], "sha256": e["sha256"], "bytes": e["bytes"], "source": str(found[e["name"]])})

    release = f"v{version}"
    commits: list[tuple[str, str, str | None]] = []
    for e in expected:
        if (e["commit"], e["release"]) not in [(c, r) for c, r, _ in commits]:
            commits.append((e["commit"], e["release"], e.get("branch")))
    if head not in [c for c, _, _ in commits]:
        commits.append((head, release, None))
    code: dict[str, dict] = {}
    for commit, rel, branch in commits:
        name = archive_code(repo, commit, rel, branch, upload, warnings)
        if name:
            code[name] = {"release": rel, "commit": commit}
            state["code"].append({"file": name, "release": rel, "commit": commit})

    shutil.copyfile(repo / "LICENSE", upload / "LICENSE-CODE.txt")
    shutil.copyfile(repo / "deposit" / "LICENSE-DATA.txt", upload / "LICENSE-DATA.txt")
    shutil.copyfile(repo / "CITATION.cff", upload / "CITATION.cff")
    meta = json.loads((repo / "deposit" / "zenodo_dataset.json").read_text())
    meta["metadata"]["version"] = DATA_VERSION
    meta["metadata"]["publication_date"] = state["date"]
    (upload / "zenodo_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    files = [e["name"] for e in bundles_used] + sorted(code) + [
        "README_DATA.md", "LICENSE-DATA.txt", "LICENSE-CODE.txt", "CITATION.cff", "zenodo_metadata.json",
        "DEPOSIT_MANIFEST.json", "SHA256SUMS"]
    readme = render_readme((repo / "deposit" / "README_DATA.md").read_text(), files=files, upload=upload,
                           bundles=bundles_used, code=code, release=release, commit=head, date=state["date"])
    (upload / "README_DATA.md").write_text(readme)

    state["status"] = "incomplete" if missing else "ok"
    manifest = {k: state[k] for k in ("created", "date", "version", "data_version", "head", "host", "python",
                                       "bundles", "missing", "code", "warnings", "status")}
    (upload / "DEPOSIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sums = [f"{sha256(upload / n)}  {n}" for n in files if n != "SHA256SUMS"]
    (upload / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    state["upload_files"] = files
    log.info("upload folder: %s (%d files, %s bytes)", upload, len(files),
             f"{sum((upload / n).stat().st_size for n in files):,}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", action="append", default=None,
                    help="directory with the result bundles (repeatable; default ~/crossleak-artifacts)")
    ap.add_argument("--bundle", action="append", default=[], help="path of a bundle zip (repeatable)")
    ap.add_argument("--out", default="~/crossleak-deposit", help="output directory (default ~/crossleak-deposit)")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    ap.add_argument("--allow-missing", action="store_true", help="build the deposit even if a bundle is not found")
    args = ap.parse_args(argv)
    if args.artifacts is None:
        args.artifacts = ["~/crossleak-artifacts"]

    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    try:
        version = package_version(Path(args.repo))
    except Exception:  # noqa: BLE001 - keep going so the log still ends up in the zip
        version = "unknown"
    dep = out / f"crossleak-deposit_v{version}_{stamp}"
    dep.mkdir()
    logfile = dep / f"make_deposit_{stamp}.log"
    handlers = [logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)]
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    state = {"created": now.isoformat(timespec="seconds"), "date": now.date().isoformat(), "version": version,
             "data_version": DATA_VERSION, "head": None, "host": platform.node(), "python": platform.python_version(),
             "bundles": [], "missing": [], "code": [], "warnings": [], "status": "failed", "error": None,
             "upload_files": []}
    log.info("CrossLeak deposit, release v%s, data version %s, output %s", version, DATA_VERSION, dep)
    try:
        build(args, dep, version, stamp, state)
    except Exception as exc:  # noqa: BLE001 - every failure is logged and bundled
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("deposit failed: %s", state["error"])
        log.error("%s", traceback.format_exc())
    finally:
        (dep / "RESULT.json").write_text(json.dumps(state, indent=2) + "\n")
        check = out / f"crossleak_v{version}_{stamp}_deposit-check.zip"
        for h in handlers:
            h.flush()
        skip = {b["name"] for b in state["bundles"]}
        write_check_zip(dep, check, skip)
        for h in handlers:
            log.removeHandler(h)
            h.close()
    if state["status"] == "ok":
        print(f"DEPOSIT OK: upload every file in {dep / 'upload'} ; send back {check}")
        return 0
    if state["status"] == "incomplete":
        print(f"DEPOSIT INCOMPLETE (missing: {', '.join(state['missing'])}): {dep / 'upload'} ; send back {check}")
        return 2
    print(f"DEPOSIT FAILED: {state['error']} ; send back {check}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
