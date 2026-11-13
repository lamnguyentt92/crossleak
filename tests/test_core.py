"""Unit, property and negative-control tests that need no model weights or data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crossleak.config import load_config, stable_seed
from crossleak.data.panel import Panel, Sampler, compute_validity, day_mask, make_pools
from crossleak.decode import decode_eval, evening_score, zprofile
from crossleak.experiments.e0 import assign_shards, build_cells
from crossleak.metrics import auc, cluster_bootstrap, mase_item, rowwise_pearson, wql_item, wql_point
from crossleak.model import instance_loc_scale, pad_left, to_model_space

DEC = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


# ---------------------------------------------------------------- metrics


def test_wql_perfect_and_point_equivalence():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert wql_item(y, np.tile(y, (9, 1))) == 0.0
    point = y + np.array([1.0, -1.0, 0.5, 0.0])
    # for symmetric deciles the mean of 2*pinball equals |e|
    assert wql_point(y, point) == pytest.approx(np.abs(y - point).sum() / np.abs(y).sum())


def test_wql_ignores_nan_targets():
    y = np.array([1.0, np.nan, 3.0])
    q = np.tile(np.array([1.0, 5.0, 3.0]), (9, 1))
    assert wql_item(y, q) == 0.0


def test_wql_hand_example():
    y = np.array([10.0])
    q = np.full((9, 1), 8.0)  # every quantile under-forecasts by 2
    expected = np.mean([2 * qq * 2.0 for qq in DEC]) / 10.0
    assert wql_item(y, q) == pytest.approx(expected)


def test_mase_seasonal_scale():
    hist = np.tile(np.arange(24, dtype=float), 3) + np.repeat([0.0, 1.0, 2.0], 24)  # seasonal diff = 1
    y = np.arange(24, dtype=float)
    assert mase_item(y, y + 2.0, hist) == pytest.approx(2.0)


def test_auc_and_pearson():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    a = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    r = rowwise_pearson(a, np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]))
    assert r[0] == pytest.approx(1.0) and np.isnan(r[1])


def test_cluster_bootstrap_reproducible_and_covers():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.0, 400)
    cl = np.repeat(np.arange(40), 10)
    b1 = cluster_bootstrap(lambda i: x[i].mean(), cl, 300, 5)
    b2 = cluster_bootstrap(lambda i: x[i].mean(), cl, 300, 5)
    assert b1 == b2
    assert b1["lo"] < b1["est"] < b1["hi"]
    assert b1["est"] == pytest.approx(x.mean())
    assert 0.05 < b1["hi"] - b1["lo"] < 0.5


# ---------------------------------------------------------------- model helpers


def test_pad_left():
    out = pad_left(np.arange(3), 5)
    assert np.isnan(out[:2]).all() and (out[2:] == np.arange(3)).all()
    assert (pad_left(np.arange(7), 5) == np.arange(2, 7)).all()


@pytest.mark.parametrize("arcsinh", [True, False])
def test_to_model_space_inverts_instance_norm(arcsinh):
    rng = np.random.default_rng(0)
    ctx = pad_left(rng.normal(50, 7, 30), 336)
    f = rng.normal(0, 1, (9, 24))
    loc, scale = instance_loc_scale(ctx)
    q = (np.sinh(f) if arcsinh else f) * scale + loc
    assert np.allclose(to_model_space(q, ctx, arcsinh=arcsinh), f, atol=1e-4)


# ---------------------------------------------------------------- panel and sampling


def _toy_panel(U=12, days=40, seed=0):
    idx = pd.date_range("2020-01-01", periods=24 * days, freq="1h")
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    v = 10 + np.sin(2 * np.pi * (t[None, :] + rng.integers(0, 24, (U, 1))) / 24) + 0.1 * rng.standard_normal((U, len(idx)))
    v = v.astype(np.float32)
    v[3, 24 * 20 : 24 * 25] = np.nan  # an outage for user 3
    return Panel("lifesnaps", v, idx, [f"u{i}" for i in range(U)])


def test_validity_rules():
    P = _toy_panel()
    V = compute_validity(P, context=336, horizon=24)
    assert (P.index[V.origins].hour == 0).all()
    assert V.origins.min() >= 336 and V.origins.max() + 24 <= P.values.shape[1]
    d_out = np.where(V.days == pd.Timestamp("2020-01-23"))[0][0]  # label day inside the outage
    assert not V.valid[3, d_out]
    assert V.valid[0].all()


def test_pools_disjoint_and_deterministic():
    P = _toy_panel(U=40)
    a = make_pools(P, {"victim": 0.5, "aux": 0.35, "public": 0.15}, 11)
    b = make_pools(P, {"victim": 0.5, "aux": 0.35, "public": 0.15}, 11)
    assert all((a[k] == b[k]).all() for k in a)
    allu = np.concatenate(list(a.values()))
    assert len(allu) == 40 and len(np.unique(allu)) == 40


def test_sampler_respects_validity_and_distinctness():
    P = _toy_panel(U=20)
    V = compute_validity(P, 336, 24)
    pools = make_pools(P, {"victim": 0.5, "aux": 0.35, "public": 0.15}, 3)
    S = Sampler(P, V, pools, day_mask(V, None, None), np.random.default_rng(0))
    for _ in range(50):
        d, pk = S.draw({"aux": 1, "victim": 4})
        us = np.concatenate([pk["aux"], pk["victim"]])
        assert len(np.unique(us)) == 5
        assert V.valid[us, d].all()
        assert set(pk["aux"]) <= set(pools["aux"]) and set(pk["victim"]) <= set(pools["victim"])


def test_day_mask_exclusions():
    P = _toy_panel()
    V = compute_validity(P, 336, 24)
    m = day_mask(V, ["2020-01-20", "2020-01-30"], ["2020-01-25"])
    assert not m[V.days == pd.Timestamp("2020-01-25")].any()
    assert m.sum() == 10


# ---------------------------------------------------------------- cells


def test_cells_and_shards_cover_everything_once():
    cfg = load_config("configs/e0.yaml")
    cells = build_cells(cfg)
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    for n in (1, 3, 5):
        assign = assign_shards(cells, cfg, n)
        for c in cells:
            if c.exp == "speed":
                assert assign[c.cell_id] == list(range(n))
            else:
                assert len(assign[c.cell_id]) == 1 and 0 <= assign[c.cell_id][0] < n


def test_sample_key_independent_of_model():
    cfg = load_config("configs/e0.yaml")
    cells = [c for c in build_cells(cfg) if c.exp == "decode"]
    by_key = {}
    for c in cells:
        by_key.setdefault(c.sample_key, set()).add(c.model)
    assert all(len(v) == len(cfg["models"]) for v in by_key.values())
    assert stable_seed("a", base=1) == stable_seed("a", base=1) != stable_seed("b", base=1)


# ---------------------------------------------------------------- decoder negative/positive controls


def _synthetic_decode(leak: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_tr, n_te, dim = 600, 300, 40
    base = np.sin(2 * np.pi * np.arange(24) / 24)

    def make(n):
        Y = np.stack([zprofile(base * rng.uniform(0.5, 1.5) + rng.normal(0, 1, 24)) for _ in range(n)])
        W = rng.normal(0, 1, (24, dim))
        W[:] = np.random.default_rng(99).normal(0, 1, (24, dim))
        fxl = leak * Y @ W + rng.normal(0, 1, (n, dim))
        fiso = rng.normal(0, 1, (n, dim))
        cal = np.zeros((n, 19))
        return np.hstack([fxl, fiso, cal]), np.hstack([fiso, cal]), Y

    Xa, Ca, Ya = make(n_tr)
    Xb, Cb, Yb = make(n_te)
    clusters = np.repeat(np.arange(n_te // 5), 5)
    return decode_eval(Xa, Ca, Ya, evening_score(Ya), Xb, Cb, Yb, evening_score(Yb), clusters, seed, 200)


def test_negative_control_no_leak_is_not_detected():
    res = _synthetic_decode(leak=0.0)
    assert res["delta_r"]["lo"] <= 0.05 and res["delta_auc"]["lo"] <= 0.05
    assert abs(res["auc_ctrl"] - 0.5) < 0.1


def test_planted_leak_is_detected():
    res = _synthetic_decode(leak=0.5)
    assert res["delta_r"]["lo"] > 0.05
    assert res["delta_auc"]["lo"] > 0.05
    # with the shared shape removed, permuted labels and the no-channel control carry no information
    assert abs(res["r_median_perm"]) < 0.15 and abs(res["r_median_ctrl"]) < 0.15
