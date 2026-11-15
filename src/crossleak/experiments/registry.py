"""Which campaign driver a configuration uses (``driver:`` key; E0 configs have none and use the E0 driver)."""

from __future__ import annotations

from pathlib import Path


def driver_name(cfg: dict) -> str:
    return str(cfg.get("driver", "e0"))


MAIN_DRIVERS = ("main", "v3", "v4", "v5")  # v3-v5 reuse the frozen main driver; v4 and v5 patch it at run time


def install_driver(cfg: dict) -> None:
    """Drivers that extend the frozen main driver patch it at run time (its source stays unchanged)."""
    from crossleak.experiments import main as _m  # noqa: F401  (import first so that v4 patches a loaded module)

    from crossleak.experiments import v4 as _v4
    from crossleak.experiments import v5 as _v5

    _v5.uninstall()
    _v4.uninstall()
    if driver_name(cfg) == "v4":
        _v4.install()
    elif driver_name(cfg) == "v5":
        _v5.install()


def build_cells(cfg: dict):
    if driver_name(cfg) in MAIN_DRIVERS:
        install_driver(cfg)
        from crossleak.experiments.main import build_cells as b
    else:
        from crossleak.experiments.e0 import build_cells as b
    return b(cfg)


def assign_shards(cells, cfg: dict, nshards: int) -> dict:
    if driver_name(cfg) in MAIN_DRIVERS:
        from crossleak.experiments.main import assign_shards as a
    else:
        from crossleak.experiments.e0 import assign_shards as a
    return a(cells, cfg, nshards)


def has_cost_phase(cfg: dict) -> bool:
    return driver_name(cfg) in MAIN_DRIVERS and any(c.exp == "cost" for c in build_cells(cfg))


def planned_per_shard(cfg: dict, n: int) -> dict[int, int]:
    cells = build_cells(cfg)
    a = assign_shards(cells, cfg, n)
    out = {s: 0 for s in range(n)}
    for c in cells:
        for s in a[c.cell_id]:
            out[s] += 1
    return out


def cell_costs(cfg: dict) -> dict[str, tuple[str, str, float]]:
    """cell id -> (exp, model, estimated cost); main costs are seconds, E0 costs are relative units."""
    return {c.cell_id: (c.exp, c.model, c.cost(cfg)) for c in build_cells(cfg)}


def merged_report(roots: list[Path], cfg: dict, out_dir: Path, decoders: str = "linear", n_jobs: int = 1) -> tuple[Path, str]:
    """Report over several machines' run directories -> (report path, overall verdict)."""
    import json
    import os

    if driver_name(cfg) == "v3":
        from crossleak.analysis.v3_report import build_report

        path = build_report(roots, cfg, decoders=decoders, n_jobs=n_jobs, out_dir=out_dir)
        return path, json.loads((out_dir / "v3_summary.json").read_text())["verdicts"]["overall"]
    if driver_name(cfg) == "v4":
        from crossleak.analysis.v4_report import build_report

        install_driver(cfg)
        path = build_report(roots, cfg, decoders=decoders, n_jobs=n_jobs, out_dir=out_dir)
        return path, json.loads((out_dir / "v4_summary.json").read_text())["verdicts"]["overall"]
    if driver_name(cfg) == "v5":
        from crossleak.analysis.v5_report import build_report

        install_driver(cfg)
        path = build_report(roots, cfg, decoders=decoders, n_jobs=n_jobs, out_dir=out_dir)
        return path, json.loads((out_dir / "v5_summary.json").read_text())["verdicts"]["overall"]
    if driver_name(cfg) == "main":
        from crossleak.analysis.main_report import build_report

        path = build_report(roots, cfg, decoders=decoders, n_jobs=n_jobs, out_dir=out_dir)
        return path, json.loads((out_dir / "main_summary.json").read_text())["verdicts"]["overall"]
    from crossleak.analysis.e0_report import build_report as e0_build

    old = os.environ.get("CROSSLEAK_REPORTS")
    os.environ["CROSSLEAK_REPORTS"] = str(out_dir.parent)
    try:
        exp = cfg["experiment"]
        path = e0_build(roots, exp, len({c.cell_id for c in build_cells(cfg)}))
        return path, json.loads((out_dir.parent / exp / "e0_summary.json").read_text())["verdicts"]["overall"]
    finally:
        if old is None:
            os.environ.pop("CROSSLEAK_REPORTS", None)
        else:
            os.environ["CROSSLEAK_REPORTS"] = old
