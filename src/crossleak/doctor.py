"""Environment report and conformance checks against the real Chronos-2 pipeline."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time

import numpy as np

from crossleak.config import TAG, data_dir, hostname, runs_dir

log = logging.getLogger("crossleak.doctor")


def environment() -> dict:
    info = {"version": TAG, "host": hostname(), "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for mod in ("numpy", "pandas", "torch", "transformers", "chronos", "sklearn", "scipy", "pyarrow"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            info[f"{mod}_version"] = f"MISSING ({exc})"
    try:
        import importlib.metadata as md

        info["chronos_forecasting_version"] = md.version("chronos-forecasting")
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo") as fh:
            info["mem_total_kb"] = int(fh.readline().split()[1])
        with open("/proc/cpuinfo") as fh:
            names = [ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")]
        info["cpu_model"] = names[0] if names else "?"
    except OSError:
        pass
    st = os.statvfs(str(data_dir().parent if data_dir().parent.exists() else "/"))
    info["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    return info


def conformance(model_ids: dict[str, str]) -> dict:
    """The six properties the experiment design relies on (see docs/DESIGN.md section 3)."""
    from crossleak.model import Forecaster, pad_left

    rng = np.random.default_rng(0)
    t = np.arange(336)
    rows = [np.sin(2 * np.pi * (t + ph) / 24) + 0.1 * rng.standard_normal(336) for ph in (0, 3, 6, 9)]
    X = np.stack(rows).astype(np.float32)
    checks: dict = {}
    for tag, mid in model_ids.items():
        fc = Forecaster(mid, horizon=24)
        pipe = fc.pipeline
        mv = pipe.predict([X], prediction_length=24)[0].numpy()
        cl = np.stack([o.numpy()[0] for o in pipe.predict([X[i] for i in range(4)], prediction_length=24, cross_learning=True)])
        iso = np.stack([o.numpy()[0] for o in pipe.predict([X[i] for i in range(4)], prediction_length=24, cross_learning=False)])
        X2 = X.copy()
        X2[3] = np.sin(2 * np.pi * (t + 15) / 24)
        mv2 = pipe.predict([X2], prediction_length=24)[0].numpy()
        Xn = X.copy()
        Xn[0] = pad_left(X[0, -24:], 336)
        on = pipe.predict([Xn], prediction_length=24)[0].numpy()
        X3 = X.copy()
        X3[3] = 5 * X[3] + 100
        mv3 = pipe.predict([X3], prediction_length=24)[0].numpy()
        rep = pipe.predict([X], prediction_length=24)[0].numpy()
        mvb = pipe.predict([X, X2, X3], prediction_length=24)[0].numpy()
        c = {
            "group_equals_cross_learning_max_abs": float(np.abs(mv - cl).max()),
            "group_differs_from_isolated_max_abs": float(np.abs(mv - iso).max()),
            "member_swap_changes_member0_mean_abs": float(np.abs(mv2[0] - mv[0]).mean()),
            "nan_padded_output_finite": bool(np.isfinite(on).all()),
            "affine_transform_other_member_max_abs": float(np.abs(mv3[0] - mv[0]).max()),
            "repeat_max_abs": float(np.abs(rep - mv).max()),
            "tasks_independent_in_batch_max_abs": float(np.abs(mvb - mv).max()),
            "arcsinh": fc.arcsinh,
        }
        c["pass"] = bool(
            c["group_equals_cross_learning_max_abs"] < 1e-4
            and c["group_differs_from_isolated_max_abs"] > 1e-3
            and c["member_swap_changes_member0_mean_abs"] > 1e-5
            and c["nan_padded_output_finite"]
            and c["affine_transform_other_member_max_abs"] < 1e-3
            and c["repeat_max_abs"] < 1e-5
            and c["tasks_independent_in_batch_max_abs"] < 1e-4
        )
        checks[tag] = c
        log.info("conformance %s (%s): %s", tag, mid, c)
    return checks


def run_doctor(cfg: dict, quick: bool = False) -> bool:
    env = environment()
    log.info("environment: %s", env)
    report = {"environment": env}
    ok = True
    if not quick:
        report["conformance"] = conformance(cfg["models"])
        ok = all(v["pass"] for v in report["conformance"].values())
    report["pass"] = ok
    out = runs_dir() / "doctor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("doctor %s -> %s", "PASS" if ok else "FAIL", out)
    return ok
