"""Cluster controller: pure functions (no SSH)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from crossleak import cluster as C
from crossleak.config import TAG


def test_validate_testbed():
    ok = {f"vm_{r}": f"192.168.1.{i + 10}" for i, r in enumerate("abcde")}
    C.validate_testbed(ok, {"192.168.1.12"})
    with pytest.raises(SystemExit):
        C.validate_testbed({**ok, "vm_b": ok["vm_a"]}, None)
    with pytest.raises(SystemExit):
        C.validate_testbed({**ok, "vm_c": "vm-c"}, None)
    with pytest.raises(SystemExit):
        C.validate_testbed(ok, {"10.0.0.1"})  # not one of this machine's addresses


def test_budget_until_next_occurrence():
    now = dt.datetime(2026, 11, 12, 22, 0, tzinfo=C.VN)
    b, deadline = C.budget_until("06:30", now)
    assert b == 8 * 60 + 30 - 15
    assert deadline == dt.datetime(2026, 11, 13, 6, 30, tzinfo=C.VN).timestamp()
    b2, _ = C.budget_until("22:10", now)
    assert b2 == 5  # floor
    assert C.budget_until(None) == (None, None)
    with pytest.raises(SystemExit):
        C.budget_until("6h", now)


def test_latest_per_shard_picks_newest_and_ignores_others():
    names = [
        f"crossleak_{TAG}_20261112-100000_vm-a_e0-s0of5.zip",
        f"crossleak_{TAG}_20261112-110000_vm-a_e0-s0of5-partial.zip",
        f"crossleak_{TAG}_20261112-090000_vm-b_e0-s1of5.zip",
        f"crossleak_{TAG}_20261112-120000_vm-b_smoke-s1of5.zip",
        "crossleak_v0.0.9_20261112-130000_vm-c_e0-s2of5.zip",
        f"crossleak_{TAG}_20261112-130000_vm-c_e0-s2of1.zip",
    ]
    got = [p.name for p in C.latest_per_shard([Path(n) for n in names], "e0", 5)]
    assert got == [names[1], names[2]]


def test_init_imports_evicon_testbed(tmp_path, monkeypatch):
    ev = tmp_path / "EVICON_TOIT_088" / "testbed.json"
    ev.parent.mkdir()
    cfg = {f"vm_{r}": f"10.1.1.{i + 1}" for i, r in enumerate("abcde")}
    ev.write_text(json.dumps({**cfg, "ssh_user": "root", "remote_root": "/root/evicon"}))
    out = tmp_path / "tb.json"
    monkeypatch.setenv("CROSSLEAK_TESTBED", str(out))
    monkeypatch.setattr(C, "own_ips", lambda: {"127.0.0.1", "10.1.1.3"})
    monkeypatch.setattr(C, "cmd_check", lambda: 0)
    assert C.cmd_init(None, "root", str(ev)) == 0
    written = json.loads(out.read_text())
    assert written == {**cfg, "ssh_user": "root"}
    hosts = C.load_hosts()
    assert [h.shard for h in hosts] == [0, 1, 2, 3, 4]
    assert [h.local for h in hosts] == [False, False, True, False, False]


def test_planned_per_shard_counts_speed_everywhere():
    p = C.planned_per_shard("configs/e0.yaml", 5)
    assert sum(p.values()) == 126 - 2 + 2 * 5  # speed cells (one per model) run on every shard


def test_content_changes_ignores_file_mode(tmp_path):
    import os
    import subprocess

    r = tmp_path / "repo"
    r.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    run("add", "a.py")
    run("commit", "-q", "-m", "c")
    os.chmod(r / "a.py", 0o755)  # what `chmod -R +x` does
    assert C.content_changes(r) == ""
    (r / "a.py").write_text("x = 2\n")
    assert "a.py" in C.content_changes(r)
