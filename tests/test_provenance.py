import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_provenance", ROOT / "scripts" / "check_provenance.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def _bundle(tmp_path: Path, files: dict[str, str]) -> Path:
    z = tmp_path / "crossleak_vX_all-test.zip"
    summary = {"freeze_test": {"status": "match", "files": {p: {"expected": h, "actual": h, "match": True}
                                                           for p, h in files.items()}}}
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("crossleak_vX_all-test/merged_report/test_summary.json", json.dumps(summary))
    return z


def test_matching_and_changed_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "prereg").mkdir(parents=True)
    plan = repo / "prereg" / "PLAN.md"
    plan.write_text("frozen plan\n")
    good = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert cp.check(_bundle(tmp_path, {"prereg/PLAN.md": good}), repo)
    plan.write_text("edited plan\n")  # negative control: an edited frozen file must be detected
    assert not cp.check(_bundle(tmp_path, {"prereg/PLAN.md": good}), repo)
    assert not cp.check(_bundle(tmp_path, {"prereg/MISSING.md": good}), repo)


def test_frozen_files_of_this_checkout():
    """Every file listed by a freeze file in prereg/ exists here (the freeze files themselves are the record)."""
    for f in (ROOT / "prereg").glob("*_FREEZE.sha256"):
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                path = line.split()[-1]
                assert (ROOT / path).is_file(), (f.name, path)
