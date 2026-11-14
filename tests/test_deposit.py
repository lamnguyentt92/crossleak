"""Deposit assembly (scripts/make_deposit.py) on a throw-away git repository and fake bundles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_deposit", ROOT / "scripts" / "make_deposit.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fake_zip(path: Path, payload: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("RESULT.json", json.dumps({"payload": payload}))


@pytest.fixture()
def setup(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src" / "crossleak").mkdir(parents=True)
    (repo / "deposit").mkdir()
    shutil.copy(ROOT / "src" / "crossleak" / "__init__.py", repo / "src" / "crossleak" / "__init__.py")
    for f in ("LICENSE", "CITATION.cff"):
        shutil.copy(ROOT / f, repo / f)
    for f in ("README_DATA.md", "LICENSE-DATA.txt", "zenodo_dataset.json"):
        shutil.copy(ROOT / "deposit" / f, repo / "deposit" / f)
    (repo / "AutogluonModels" / "ag-1").mkdir(parents=True)
    (repo / "AutogluonModels" / "ag-1" / "train.pkl").write_bytes(b"tool output")
    art = tmp_path / "artifacts"
    art.mkdir()
    _fake_zip(art / "bundle_a.zip", "a")
    _fake_zip(art / "renamed_b.zip", "b")
    _fake_zip(art / "unrelated.zip", "c")
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, env=GIT_ENV)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    def entry(name: str, src: Path, exp: str) -> dict:
        return {"name": name, "sha256": _sha(src), "bytes": src.stat().st_size, "experiment": exp, "title": exp,
                "release": "v0.0.1", "branch": "release/v0.0.1", "commit": head}

    bundles = [entry("crossleak_A_all-x.zip", art / "bundle_a.zip", "x"),
               entry("crossleak_B_all-y.zip", art / "renamed_b.zip", "y")]
    (repo / "deposit" / "bundles.json").write_text(json.dumps({"bundles": bundles}))
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "bundles"], check=True, env=GIT_ENV)
    return repo, art, tmp_path / "out", bundles


def _check_zip(out: Path) -> zipfile.ZipFile:
    zips = list(out.glob("*_deposit-check.zip"))
    assert len(zips) == 1
    return zipfile.ZipFile(zips[0])


def test_deposit_ok(setup):
    repo, art, out, bundles = setup
    rc = md.main(["--repo", str(repo), "--artifacts", str(art), "--out", str(out)])
    assert rc == 0
    dep = next(out.glob("crossleak-deposit_v*"))
    up = dep / "upload"
    for b in bundles:  # copied unchanged under the canonical name
        assert _sha(up / b["name"]) == b["sha256"]
    sums = dict(reversed(line.split("  ", 1)) for line in (up / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {p.name for p in up.iterdir()} - {"SHA256SUMS"}
    for name, h in sums.items():
        assert _sha(up / name) == h
    readme = (up / "README_DATA.md").read_text()
    assert "{{" not in readme and "crossleak_A_all-x.zip" in readme
    meta = json.loads((up / "zenodo_metadata.json").read_text())["metadata"]
    assert meta["license"] == "cc-by-4.0" and meta["version"] == md.DATA_VERSION
    code = [p for p in up.glob("crossleak-code-*.zip")]
    members = zipfile.ZipFile(code[0]).namelist()
    assert code and members[0].startswith("crossleak-")
    assert any(m.endswith("src/crossleak/__init__.py") for m in members)
    assert not any("AutogluonModels" in m for m in members)  # tool output left out of the code archive
    names = _check_zip(out).namelist()
    assert not any(n.endswith(("crossleak_A_all-x.zip", "crossleak_B_all-y.zip")) for n in names)  # bundles left out
    assert any(n.endswith("RESULT.json") for n in names) and any(n.endswith(".log") for n in names)


def test_deposit_missing_bundle_fails_but_bundles_log(setup):
    repo, art, out, _ = setup
    (art / "renamed_b.zip").unlink()
    rc = md.main(["--repo", str(repo), "--artifacts", str(art), "--out", str(out)])
    assert rc == 1
    zf = _check_zip(out)
    log = next(n for n in zf.namelist() if n.endswith(".log"))
    assert "bundles not found" in zf.read(log).decode()
    res = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith("RESULT.json"))))
    assert res["status"] == "failed"


def test_deposit_allow_missing_and_explicit_bundle(setup, tmp_path):
    repo, art, out, _ = setup
    moved = tmp_path / "elsewhere.zip"
    shutil.move(art / "renamed_b.zip", moved)
    assert md.main(["--repo", str(repo), "--artifacts", str(art), "--out", str(out / "1"), "--allow-missing"]) == 2
    assert md.main(["--repo", str(repo), "--artifacts", str(art), "--out", str(out / "2"), "--bundle", str(moved)]) == 0


def test_real_bundle_list_is_consistent():
    data = json.loads((ROOT / "deposit" / "bundles.json").read_text())["bundles"]
    assert len(data) == 3 and len({b["sha256"] for b in data}) == 3
    for b in data:
        assert len(b["sha256"]) == 64 and len(b["commit"]) == 40 and b["bytes"] > 0
        assert b["name"].startswith(f"crossleak_{b['release']}_")
