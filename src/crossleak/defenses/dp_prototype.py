"""D4: DP-Prototype cross-learning.

Once per period (here: once per day, at the forecast origin) the service releases P prototypes built from its
established users' recent data under user-level differential privacy:

1. every user u normalises its own context (zero mean, unit RMS over observed hours; missing hours -> 0) and maps it
   to a representation r_u (``REPRESENTATIONS``), then clips it to L2 norm <= C;
2. u is assigned to the nearest of P *public* centres (fitted on public users only), a function of u's own data;
3. per cluster p: noisy sum S_p = sum r_u + N(0, (sigma_s C)^2 I) and noisy count n_p = |p| + N(0, sigma_n^2);
   prototype pi_p = S_p / n_p if n_p >= 1, otherwise the public centre c_p (post-processing);
4. every user is then forecast in the group {own series} + {pi_1..pi_P}; no user's series meets another user's.

Privacy (add/remove one user, one release): clusters are disjoint, so adding a user changes one sum by a vector of
norm <= C and one count by 1. The Gaussian mechanism with noise std sigma*Delta is 1/(2 sigma^2)-zCDP, so the release
is rho-zCDP with rho = 1/(2 sigma_s^2) + 1/(2 sigma_n^2), hence (rho + 2 sqrt(rho ln(1/delta)), delta)-DP
(Bun & Steinke 2016, Prop. 1.3). Every forecast returned to anyone that day is post-processing of the release and of
the receiver's own data, so it inherits the guarantee with respect to every other user. Releases on T days compose to
T*rho.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from crossleak.model import pad_left

REPRESENTATIONS = ("full", "weekfold", "dayfold", "lastweek")


def normalise(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit RMS over observed values; missing values become 0 (the mean)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    out = np.zeros_like(x)
    if m.sum() < 2:
        return out
    mu = x[m].mean()
    sd = np.sqrt(((x[m] - mu) ** 2).mean())
    if sd > 0:
        out[m] = (x[m] - mu) / sd
    return out


def _fold(z: np.ndarray, obs: np.ndarray, period: int) -> np.ndarray:
    n = len(z) // period
    zz = z[-n * period :].reshape(n, period)
    oo = obs[-n * period :].reshape(n, period)
    cnt = oo.sum(axis=0)
    s = (zz * oo).sum(axis=0)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def represent(context: np.ndarray, kind: str) -> np.ndarray:
    """Representation of one user's own context (length C, NaN = missing)."""
    x = np.asarray(context, dtype=np.float64)
    obs = np.isfinite(x)
    z = normalise(x)
    if kind == "full":
        return z
    if kind == "weekfold":
        return _fold(z, obs, 168)
    if kind == "dayfold":
        return _fold(z, obs, 24)
    if kind == "lastweek":
        return z[-168:]
    raise ValueError(kind)


def dim_of(kind: str, context_length: int) -> int:
    return {"full": context_length, "weekfold": 168, "dayfold": 24, "lastweek": 168}[kind]


def default_clip(kind: str, context_length: int) -> float:
    """C = sqrt(dim): a normalised series has RMS 1, so its representation has norm <= sqrt(dim)."""
    return math.sqrt(dim_of(kind, context_length))


def to_series(r: np.ndarray, kind: str, context_length: int) -> np.ndarray:
    """Turn a (noisy) representation into a co-tenant series of length context_length."""
    r = np.asarray(r, dtype=np.float32)
    if kind == "full":
        return r[-context_length:]
    if kind in ("weekfold", "dayfold"):
        reps = int(math.ceil(context_length / len(r)))
        return np.tile(r, reps)[-context_length:]
    if kind == "lastweek":
        return pad_left(r, context_length)
    raise ValueError(kind)


def clip_l2(R: np.ndarray, C: float) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    n = np.linalg.norm(R, axis=1, keepdims=True)
    return R * np.minimum(1.0, C / np.maximum(n, 1e-12))


# --------------------------------------------------------------------------- accounting


def rho_to_eps(rho: float, delta: float) -> float:
    if rho == 0:
        return 0.0
    if math.isinf(rho):
        return math.inf
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def eps_to_rho(eps: float, delta: float) -> float:
    """Largest rho with rho_to_eps(rho, delta) <= eps (closed form)."""
    if math.isinf(eps):
        return math.inf
    L = math.log(1.0 / delta)
    x = -math.sqrt(L) + math.sqrt(L + eps)
    return x * x


@dataclass(frozen=True)
class NoiseScales:
    eps: float
    delta: float
    rho: float
    sigma_sum: float  # noise std of each sum coordinate = sigma_sum * C
    sigma_count: float  # noise std of each count

    @property
    def eps_check(self) -> float:
        return rho_to_eps(1.0 / (2 * self.sigma_sum**2) + 1.0 / (2 * self.sigma_count**2), self.delta) if self.sigma_sum > 0 else math.inf


def noise_scales(eps: float, delta: float, sum_share: float = 0.9) -> NoiseScales:
    if math.isinf(eps):
        return NoiseScales(eps, delta, math.inf, 0.0, 0.0)
    rho = eps_to_rho(eps, delta)
    s = math.sqrt(1.0 / (2.0 * sum_share * rho))
    n = math.sqrt(1.0 / (2.0 * (1.0 - sum_share) * rho))
    return NoiseScales(eps, delta, rho, s, n)


def dp_auc_ceiling(eps: float, delta: float, grid: int = 20001) -> float:
    """Largest ROC AUC of any test between two (eps, delta)-indistinguishable distributions.

    TPR <= min(1, e^eps FPR + delta, 1 - e^-eps (1 - delta - FPR)) (Kairouz et al. 2015)."""
    if math.isinf(eps):
        return 1.0
    a = np.linspace(0.0, 1.0, grid)
    b = np.minimum.reduce([np.ones_like(a), math.exp(eps) * a + delta, 1.0 - math.exp(-eps) * (1.0 - delta - a)])
    b = np.clip(b, a, 1.0)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(b, a))


def attribute_ceiling(eps: float, delta: float) -> float:
    """Ceiling for inferring an attribute of one user's record: replacing a record = remove + add, so the
    neighbouring pair is (2 eps, (1 + e^eps) delta)-indistinguishable."""
    if math.isinf(eps):
        return 1.0
    return dp_auc_ceiling(2.0 * eps, min(1.0, (1.0 + math.exp(eps)) * delta))


# --------------------------------------------------------------------------- clustering and release


def fit_centres(R_public: np.ndarray, P: int, seed: int) -> np.ndarray:
    """k-means centres on public representations only (no privacy cost)."""
    from sklearn.cluster import KMeans

    R_public = np.asarray(R_public, dtype=np.float64)
    if len(R_public) < P:
        raise ValueError(f"need >= {P} public representations, got {len(R_public)}")
    km = KMeans(n_clusters=P, n_init=10, random_state=seed).fit(R_public)
    return km.cluster_centers_


def assign(R: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d = ((np.asarray(R)[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


@dataclass
class Release:
    prototypes: np.ndarray  # (P, dim)
    counts: np.ndarray  # (P,) true counts (for diagnostics only; never returned to users)
    noisy_counts: np.ndarray  # (P,)
    fallback: np.ndarray  # (P,) bool: public centre used
    scales: NoiseScales


def release(R: np.ndarray, centres: np.ndarray, eps: float, delta: float, C: float, rng: np.random.Generator, sum_share: float = 0.9) -> Release:
    """One DP release of P prototypes from the users' representations R (n, dim)."""
    R = clip_l2(np.asarray(R, dtype=np.float64), C) if len(R) else np.zeros((0, centres.shape[1]))
    lab = assign(R, centres) if len(R) else np.zeros(0, dtype=int)
    sc = noise_scales(eps, delta, sum_share)
    P, dim = centres.shape
    protos = np.zeros((P, dim))
    counts = np.zeros(P)
    noisy = np.zeros(P)
    fb = np.zeros(P, dtype=bool)
    for p in range(P):
        members = R[lab == p]
        s = members.sum(axis=0) if len(members) else np.zeros(dim)
        n = float(len(members))
        if sc.sigma_sum > 0:
            s = s + rng.normal(0.0, sc.sigma_sum * C, size=dim)
            n_noisy = n + rng.normal(0.0, sc.sigma_count)
        else:
            n_noisy = n
        counts[p], noisy[p] = n, n_noisy
        if n_noisy >= 1.0:
            protos[p] = s / n_noisy
        else:
            protos[p] = centres[p]
            fb[p] = True
    return Release(protos, counts, noisy, fb, sc)
