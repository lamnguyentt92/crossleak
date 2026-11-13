"""Turn raw archives into hourly wide panels (time x user), saved as parquet.

Processing versions are part of the file names (``_p1``) so that a new preprocessing
never overwrites what an older code version on the same machine relies on.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from crossleak.config import data_dir
from crossleak.data.fetch import raw_path

log = logging.getLogger("crossleak.prepare")

PROC = "p1"
ELD_START, ELD_END = "2013-01-01 00:00", "2014-12-31 23:00"
ELD_MAX_ZERO_FRAC = 0.01
LS_MIN_OBS_HOURS = 24 * 28  # at least four weeks of observed heart rate


def processed_path(name: str) -> Path:
    return data_dir() / "processed" / f"{name}_hourly_{PROC}.parquet"


def report_path(name: str) -> Path:
    return data_dir() / "processed" / f"{name}_hourly_{PROC}.report.json"


def prepare_eld(force: bool = False) -> dict:
    out = processed_path("eld")
    if out.exists() and not force:
        log.info("ELD panel already prepared: %s", out)
        return json.loads(report_path("eld").read_text())
    src = raw_path("eld")
    log.info("parsing ELD from %s (this takes a few minutes)", src)
    with zipfile.ZipFile(src) as zf, zf.open("LD2011_2014.txt") as fh:
        df = pd.read_csv(fh, sep=";", decimal=",", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index)
    df = df.astype("float32")
    n_total = df.shape[1]
    # Timestamps mark the end of each 15-min interval: hour h covers (h:00, h+1:00].
    hourly = df.resample("1h", closed="right", label="left").mean()
    hourly = hourly.loc[ELD_START:ELD_END]
    zero_frac = (hourly == 0).mean(axis=0)
    finite = hourly.notna().all(axis=0)
    keep = (zero_frac <= ELD_MAX_ZERO_FRAC) & finite
    hourly = hourly.loc[:, keep].astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out)
    report = {
        "dataset": "eld",
        "processing": PROC,
        "clients_total": int(n_total),
        "clients_kept": int(hourly.shape[1]),
        "hours": int(hourly.shape[0]),
        "start": str(hourly.index[0]),
        "end": str(hourly.index[-1]),
        "rule": f"hourly mean of 15-min kW; keep clients with zero fraction <= {ELD_MAX_ZERO_FRAC} in {ELD_START}..{ELD_END}",
    }
    report_path("eld").write_text(json.dumps(report, indent=2))
    log.info("ELD prepared: %s", report)
    return report


def prepare_lifesnaps(force: bool = False) -> dict:
    out = processed_path("lifesnaps")
    if out.exists() and not force:
        log.info("LifeSnaps panel already prepared: %s", out)
        return json.loads(report_path("lifesnaps").read_text())
    src = raw_path("lifesnaps")
    member = "rais_anonymized/csv_rais_anonymized/hourly_fitbit_sema_df_unprocessed.csv"
    log.info("parsing LifeSnaps hourly CSV from %s", src)
    with zipfile.ZipFile(src) as zf, zf.open(member) as fh:
        df = pd.read_csv(fh, usecols=["id", "date", "hour", "bpm", "steps", "age", "gender", "bmi"], low_memory=False)
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
    bpm = df.pivot_table(index="ts", columns="id", values="bpm", aggfunc="mean")
    full_index = pd.date_range(bpm.index.min().floor("D"), bpm.index.max().ceil("D"), freq="1h", inclusive="left")
    bpm = bpm.reindex(full_index).astype("float32")
    obs = bpm.notna().sum(axis=0)
    keep = obs >= LS_MIN_OBS_HOURS
    bpm = bpm.loc[:, keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    bpm.columns = [str(c) for c in bpm.columns]
    bpm.to_parquet(out)
    attrs = (
        df.groupby("id")[["age", "gender", "bmi"]]
        .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None)
        .reset_index()
    )
    attrs["id"] = attrs["id"].astype(str)
    attrs[attrs["id"].isin(bpm.columns)].to_csv(data_dir() / "processed" / f"lifesnaps_attributes_{PROC}.csv", index=False)
    report = {
        "dataset": "lifesnaps",
        "processing": PROC,
        "users_total": int(df["id"].nunique()),
        "users_kept": int(bpm.shape[1]),
        "hours": int(bpm.shape[0]),
        "start": str(bpm.index[0]),
        "end": str(bpm.index[-1]),
        "observed_fraction": float(np.isfinite(bpm.to_numpy()).mean()),
        "rule": f"hourly mean bpm; keep users with >= {LS_MIN_OBS_HOURS} observed hours",
    }
    report_path("lifesnaps").write_text(json.dumps(report, indent=2))
    log.info("LifeSnaps prepared: %s", report)
    return report


PREPARERS = {"eld": prepare_eld, "lifesnaps": prepare_lifesnaps}
