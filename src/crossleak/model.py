"""Chronos-2 wrapper.

A *group* is a 2-D array (k, T): the k rows share one group ID and therefore exchange
information through group attention. This is exactly Chronos-2's cross-learning mode
restricted to one group (verified by ``crossleak doctor``). Shorter members are
left-padded with NaN, which Chronos-2 treats as missing.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

log = logging.getLogger("crossleak.model")

DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EPS = 1e-5  # InstanceNorm eps used by Chronos-2


def pad_left(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= length:
        return x[-length:]
    out = np.full(length, np.nan, dtype=np.float32)
    out[length - len(x) :] = x
    return out


def instance_loc_scale(x: np.ndarray) -> tuple[float, float]:
    """Same statistics as Chronos-2 InstanceNorm on one context row (NaN ignored)."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).any():
        return 0.0, 1.0
    loc = float(np.nanmean(x))
    scale = float(np.sqrt(np.nanmean((x - loc) ** 2)))
    if scale == 0.0:
        scale = EPS
    return loc, scale


def to_model_space(q: np.ndarray, context_row: np.ndarray, arcsinh: bool = True) -> np.ndarray:
    """Map de-normalised forecasts of one row back to the model's internal output space.

    Chronos-2 outputs sinh(f) * scale + loc with (loc, scale) of the row's own context.
    The owner of the row knows its context, so this inversion is available to an attacker.
    """
    loc, scale = instance_loc_scale(context_row)
    z = (np.asarray(q, dtype=np.float64) - loc) / scale
    return (np.arcsinh(z) if arcsinh else z).astype(np.float32)


class Forecaster:
    def __init__(self, model_id: str, threads: int | None = None, horizon: int = 24):
        import torch
        from chronos import BaseChronosPipeline

        self.threads = int(threads or os.cpu_count() or 1)
        torch.set_num_threads(self.threads)
        self.model_id = model_id
        self.horizon = horizon
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
        qs = list(self.pipeline.quantiles)
        self.quantiles = qs
        self.decile_idx = [int(np.argmin([abs(q - d) for q in qs])) for d in DECILES]
        for i, d in zip(self.decile_idx, DECILES):
            if abs(qs[i] - d) > 1e-6:
                raise RuntimeError(f"model {model_id} lacks quantile {d}")
        self.arcsinh = bool(getattr(self.pipeline.model.chronos_config, "use_arcsinh", False))
        log.info("loaded %s (%d quantiles, arcsinh=%s, %d threads)", model_id, len(qs), self.arcsinh, self.threads)

    def model_space(self, q: np.ndarray, context_row: np.ndarray) -> np.ndarray:
        return to_model_space(q, context_row, arcsinh=self.arcsinh)

    def predict_groups(self, groups: Sequence[np.ndarray], max_series_per_call: int = 4096) -> list[np.ndarray]:
        """Forecast every group independently (groups never mix with each other).

        Returns one array (k, 9, H) of deciles per group, in original units.
        """
        out: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        n_series = 0

        def flush() -> None:
            nonlocal batch, n_series
            if not batch:
                return
            kmax = max(g.shape[0] for g in batch)
            preds = self.pipeline.predict(
                [np.asarray(g, dtype=np.float32) for g in batch],
                prediction_length=self.horizon,
                batch_size=max(256, kmax),
                cross_learning=False,
            )
            for p in preds:
                out.append(p.numpy()[:, self.decile_idx, :].astype(np.float32))
            batch, n_series = [], 0

        for g in groups:
            g = np.asarray(g, dtype=np.float32)
            if g.ndim == 1:
                g = g[None, :]
            batch.append(g)
            n_series += g.shape[0]
            if n_series >= max_series_per_call:
                flush()
        flush()
        return out
