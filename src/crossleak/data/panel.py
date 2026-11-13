"""Hourly panel access, window validity, user pools and day splits."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crossleak.config import stable_seed
from crossleak.data.prepare import processed_path

log = logging.getLogger("crossleak.panel")

POOLS = ("victim", "aux", "public")


@dataclass
class Panel:
    name: str
    values: np.ndarray  # (U, T) float32, NaN = missing
    index: pd.DatetimeIndex  # hourly timestamps, length T
    users: list[str]

    @property
    def n_users(self) -> int:
        return self.values.shape[0]


def load_panel(name: str) -> Panel:
    df = pd.read_parquet(processed_path(name))
    return Panel(name=name, values=df.to_numpy(dtype=np.float32).T.copy(), index=pd.DatetimeIndex(df.index), users=list(map(str, df.columns)))


def _window_counts(mask: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Number of True in mask[:, s:s+length] for every s in starts. mask: (U, T)."""
    c = np.concatenate([np.zeros((mask.shape[0], 1), dtype=np.int64), np.cumsum(mask, axis=1)], axis=1)
    return c[:, starts + length] - c[:, starts]


@dataclass
class Validity:
    """Valid (user, origin) pairs. Origins are midnight positions in the panel index."""

    origins: np.ndarray  # (D,) positions p: context [p-C, p), horizon [p, p+H)
    days: pd.DatetimeIndex  # (D,) timestamps of origins
    valid: np.ndarray  # (U, D) bool
    context: int
    horizon: int


def compute_validity(panel: Panel, context: int, horizon: int) -> Validity:
    """Dataset-specific window rules, fixed in the E0 pre-registration."""
    T = panel.values.shape[1]
    pos = np.where(panel.index.hour == 0)[0]
    pos = pos[(pos - context >= 0) & (pos + horizon <= T)]
    v = panel.values
    obs = np.isfinite(v)
    ctx_obs = _window_counts(obs, pos - context, context)
    last_obs = _window_counts(obs, pos - 24, 24)
    hor_obs = _window_counts(obs, pos, horizon)
    # standard deviation of the last 24 h (label window) must be positive
    last = np.stack([v[:, p - 24 : p] for p in pos], axis=1)  # (U, D, 24)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        last_std = np.nanstd(last, axis=2)
    if panel.name == "eld":
        zeros = (v == 0) & obs
        ctx_zero = _window_counts(zeros, pos - context, context)
        hor = np.stack([v[:, p : p + horizon] for p in pos], axis=1)
        valid = (ctx_obs == context) & (hor_obs == horizon) & (ctx_zero <= 0.05 * context)
        valid &= np.nansum(hor, axis=2) > 0
    else:
        valid = (ctx_obs >= 0.5 * context) & (last_obs >= 20) & (hor_obs >= 20)
    valid &= np.nan_to_num(last_std, nan=0.0) > 0
    return Validity(origins=pos, days=panel.index[pos], valid=valid, context=context, horizon=horizon)


def make_pools(panel: Panel, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Disjoint user pools from a seeded permutation (victim / aux / public)."""
    rng = np.random.default_rng(stable_seed("pools", panel.name, base=seed))
    perm = rng.permutation(panel.n_users)
    n_v = int(round(fractions["victim"] * panel.n_users))
    n_a = int(round(fractions["aux"] * panel.n_users))
    return {"victim": np.sort(perm[:n_v]), "aux": np.sort(perm[n_v : n_v + n_a]), "public": np.sort(perm[n_v + n_a :])}


def day_mask(validity: Validity, date_range: list[str] | None, exclude: list[str] | None) -> np.ndarray:
    days = validity.days
    m = np.ones(len(days), dtype=bool)
    if date_range:
        m &= (days >= pd.Timestamp(date_range[0])) & (days <= pd.Timestamp(date_range[1]))
    if exclude:
        ex = pd.DatetimeIndex([pd.Timestamp(x) for x in exclude])
        m &= ~days.normalize().isin(ex)
    return m


@dataclass
class Sampler:
    """Draws origins and distinct users that are valid at that origin."""

    panel: Panel
    validity: Validity
    pools: dict[str, np.ndarray]
    day_ok: np.ndarray  # (D,) allowed origins for this split
    rng: np.random.Generator
    _by_pool: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, users in self.pools.items():
            vu = self.validity.valid[users]  # (n_pool, D)
            self._by_pool[name] = [users[vu[:, d]] for d in range(vu.shape[1])]

    def feasible_days(self, need: dict[str, int]) -> np.ndarray:
        ok = self.day_ok.copy()
        for pool, n in need.items():
            ok &= np.array([len(x) >= n for x in self._by_pool[pool]])
        return np.where(ok)[0]

    def draw(self, need: dict[str, int], days: np.ndarray | None = None) -> tuple[int, dict[str, np.ndarray]]:
        """Return (day index, {pool: distinct users}) satisfying need. Raises if infeasible."""
        if days is None:
            days = self.feasible_days(need)
        if len(days) == 0:
            raise RuntimeError(f"no feasible origin for {need}")
        d = int(self.rng.choice(days))
        picked = {pool: self.rng.choice(self._by_pool[pool][d], size=n, replace=False) for pool, n in need.items()}
        return d, picked

    def context(self, u: int, d: int) -> np.ndarray:
        p = self.validity.origins[d]
        return self.panel.values[u, p - self.validity.context : p].astype(np.float32)

    def horizon(self, u: int, d: int) -> np.ndarray:
        p = self.validity.origins[d]
        return self.panel.values[u, p : p + self.validity.horizon].astype(np.float32)
