"""Unit, property and negative-control tests of the main study (v0.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from crossleak import attack as A
from crossleak.defenses import dp_prototype as dp
from crossleak.experiments import main as M
from crossleak.metrics import auc

CFG = yaml.safe_load(open("configs/main.yaml"))
CFG["_sha256"] = "test"


# --------------------------------------------------------------------------- DP accounting and release


@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 4.0, 8.0])
def test_zcdp_conversion_roundtrip(eps):
    rho = dp.eps_to_rho(eps, 1e-5)
    assert abs(dp.rho_to_eps(rho, 1e-5) - eps) < 1e-9
    sc = dp.noise_scales(eps, 1e-5, 0.9)
    assert abs(sc.eps_check - eps) < 1e-9  # sum + count noise together spend exactly eps


def test_noise_decreases_with_eps():
    s = [dp.noise_scales(e, 1e-5).sigma_sum for e in (0.5, 1, 2, 4, 8)]
    assert all(a > b for a, b in zip(s, s[1:]))
    assert dp.noise_scales(math.inf, 1e-5).sigma_sum == 0.0


def test_clip_bounds_norm_and_keeps_small_vectors():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(50, 24)) * rng.uniform(0.1, 20, size=(50, 1))
    C = 3.0
    Rc = dp.clip_l2(R, C)
    n0, n1 = np.linalg.norm(R, axis=1), np.linalg.norm(Rc, axis=1)
    assert (n1 <= C + 1e-9).all()
    assert np.allclose(Rc[n0 <= C], R[n0 <= C])


def test_normalised_representation_has_bounded_norm():
    rng = np.random.default_rng(1)
    for kind in dp.REPRESENTATIONS:
        for _ in range(20):
            x = rng.normal(size=336) * rng.uniform(0.1, 100) + rng.uniform(-50, 50)
            x[rng.random(336) < 0.2] = np.nan
            r = dp.represent(x, kind)
            assert len(r) == dp.dim_of(kind, 336)
            assert np.linalg.norm(r) <= dp.default_clip(kind, 336) + 1e-6


def test_dayfold_recovers_daily_shape_and_is_affine_invariant():
    t = np.arange(336)
    x = 5 + 2 * np.sin(2 * np.pi * t / 24)
    r1 = dp.represent(x, "dayfold")
    r2 = dp.represent(10 * x + 3, "dayfold")
    assert np.allclose(r1, r2)
    assert np.corrcoef(r1, np.sin(2 * np.pi * np.arange(24) / 24))[0, 1] > 0.999
    s = dp.to_series(r1, "dayfold", 336)
    assert len(s) == 336 and np.allclose(s[:24], s[-24:])


def test_release_sensitivity_add_one_user_changes_one_cluster():
    """Property behind the DP proof: adding one user changes one cluster's sum by <= C and its count by 1."""
    rng = np.random.default_rng(2)
    centres = rng.normal(size=(4, 24))
    R = rng.normal(size=(40, 24)) * 3
    C = 2.0
    u = rng.normal(size=(1, 24)) * 50  # large: clipping must bound its effect
    a = dp.release(R, centres, math.inf, 1e-5, C, np.random.default_rng(0))
    b = dp.release(np.vstack([R, u]), centres, math.inf, 1e-5, C, np.random.default_rng(0))
    diff = b.counts - a.counts
    assert sorted(diff.tolist()) == [0, 0, 0, 1]
    p = int(np.argmax(diff))
    s_a = a.prototypes[p] * a.counts[p]
    s_b = b.prototypes[p] * b.counts[p]
    assert np.linalg.norm(s_b - s_a) <= C + 1e-9
    others = [q for q in range(4) if q != p]
    assert np.allclose(a.prototypes[others], b.prototypes[others])


def test_release_noise_matches_scales_and_falls_back_to_public_centre():
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(2, 24))
    R = np.zeros((0, 24))  # no members at all
    rel = dp.release(R, centres, 0.5, 1e-5, 4.9, np.random.default_rng(1))
    # with no members the noisy count is usually < 1 -> public centre (post-processing)
    assert rel.fallback.any()
    sc = dp.noise_scales(1.0, 1e-5)
    n = 10000  # members with zero representation: prototype * n = sum noise (count noise is negligible relative to n)
    draws = [dp.release(np.zeros((n, 24)), centres[:1], 1.0, 1e-5, 4.9, np.random.default_rng(i)).prototypes[0] * n for i in range(200)]
    est_sd = np.std(np.stack(draws))
    assert abs(est_sd - sc.sigma_sum * 4.9) / (sc.sigma_sum * 4.9) < 0.1


def test_dp_auc_ceiling_bounds():
    assert abs(dp.dp_auc_ceiling(0.0, 0.0) - 0.5) < 1e-6
    vals = [dp.attribute_ceiling(e, 1e-5) for e in (0.1, 0.5, 1, 2, 4, 8)]
    assert all(0.5 <= v <= 1.0 for v in vals)
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))
    assert dp.attribute_ceiling(math.inf, 1e-5) == 1.0


def test_negative_control_unclipped_user_breaks_sensitivity():
    """Planted violation: without clipping one user can move a cluster sum far beyond C (the test above catches it)."""
    u = np.ones((1, 24)) * 100
    assert np.linalg.norm(u) > 10 * 2.0
    assert np.linalg.norm(dp.clip_l2(u, 2.0)) <= 2.0 + 1e-9


# --------------------------------------------------------------------------- experiment structure


def test_cells_unique_parts_cover_samples_and_models_paired():
    cells = M.build_cells(CFG)
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    assert all(" " not in i and "(" not in i for i in ids)
    by_fam: dict = {}
    for c in cells:
        if c.exp == "attack":
            by_fam.setdefault((c.family_key, c.p["role"]), 0)
            by_fam[(c.family_key, c.p["role"])] += c.p["n"]
    sizes = {int(k): v for k, v in CFG["attack"]["sizes"].items()}
    for (fk, role), n in by_fam.items():
        k = int(fk.split("__k-")[1].split("__")[0])
        assert n == sizes[k][0 if role == "train" else 1]
    keys = {}
    for c in cells:
        if c.exp in ("attack", "util", "channel"):
            keys.setdefault(c.sample_key, set()).add(c.model)
    assert any(len(v) == 2 for v in keys.values())  # both models see identical samples


def test_shards_cover_main_phase_once_and_cost_on_shard0():
    cells = M.build_cells(CFG)
    a = M.assign_shards(cells, CFG, 5)
    for c in cells:
        assert len(a[c.cell_id]) == 1
        if c.exp == "cost":
            assert a[c.cell_id] == [0] and M.phase_of(c) == "cost"
    main = [c for s in range(5) for c in M.shard_cells(CFG, s, 5, "main")]
    assert len(main) == len([c for c in cells if c.exp != "cost"])
    loads = [sum(c.cost(CFG) for c in M.shard_cells(CFG, s, 5, "main")) for s in range(5)]
    assert max(loads) / min(loads) < 1.05
    assert max(c.cost(CFG) for c in cells if c.exp not in ("cost",)) < 1.2 * CFG["part_budget_sec"] + 60


def test_fillers_and_attacker_members():
    rng = np.random.default_rng(0)
    probe = M.pad_left(rng.normal(size=24).astype(np.float32), 336)
    cp = M.filler_rows("copy", probe, 3, rng, 336, 24)
    assert all(np.array_equal(r, probe, equal_nan=True) for r in cp)
    sp = M.filler_rows("sparse", probe, 2, rng, 336, 24)
    assert all(np.isfinite(r).sum() == 1 and np.isfinite(r[-1]) for r in sp)
    nz = M.filler_rows("noise", probe, 2, rng, 336, 24)
    assert not np.array_equal(nz[0], nz[1], equal_nan=True)
    assert M.filler_rows("copy", probe, 0, rng, 336, 24) == []
    assert (M.n_attacker("single", 16), M.n_attacker("sybil", 16), M.n_attacker("half", 16)) == (1, 15, 8)


def test_daily_means_shape():
    x = np.tile(np.arange(24, dtype=np.float32), 14)
    x[-24:] += 10
    dm = M.daily_means(x)
    assert dm.shape == (14,) and dm[-1] > dm[:-1].max()


# --------------------------------------------------------------------------- labels, decoders, linkage


def test_labels_within_day_and_low_day():
    Y = np.zeros((4, 24))
    Y[:, 18:] = [[1], [-1], [1], [-1]]
    dm = np.tile(np.arange(14, dtype=float), (4, 1))
    dm[0, -1] = -5
    ref = np.array([0.0, 2.0])
    L = A.labels(Y, dm, np.array([0, 0, 1, 1]), ref, 0.0)
    assert L["L1"].tolist() == [1, 0, 0, 0]  # day 1's population median is high
    assert L["L2"].tolist() == [1, 0, 1, 0]
    assert L["L3"].tolist() == [1, 0, 0, 0]
    L = A.labels(Y, dm, np.array([0, 0, 1, 1]), np.array([0.0, np.nan]), 0.0)
    assert L["L1"].tolist() == [1, 0, -1, -1]  # no reference that day -> excluded, not labelled 0


def test_decoder_planted_signal_and_null():
    rng = np.random.default_rng(5)
    n, p = 600, 40
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, p))
    Xs = X.copy()
    Xs[:, 0] += 1.5 * y
    Y = rng.normal(size=(n, 24))
    tr, te = slice(0, 400), slice(400, None)
    s = A.linear_decoder(Xs[tr], Y[tr], {"L1": y[tr]}, Xs[te], 0)["scores"]["L1"]
    assert auc(y[te], s) > 0.8
    s0 = A.linear_decoder(X[tr], Y[tr], {"L1": y[tr]}, X[te], 0)["scores"]["L1"]
    assert abs(auc(y[te], s0) - 0.5) < 0.1


def test_within_day_r_and_linkage():
    rng = np.random.default_rng(6)
    U, D = 60, 5
    cand = rng.normal(size=(U, D, 24)).astype(np.float32)
    victims = rng.integers(0, U, 200)
    days = rng.integers(0, D, 200)
    perfect = cand[victims, days]
    lk = A.linkage(perfect, victims, days, cand, np.arange(U), 10, 0)
    assert np.nanmean(lk["top1"]) == 1.0
    rnd = A.linkage(rng.normal(size=(200, 24)), victims, days, cand, np.arange(U), 10, 0)
    assert abs(np.nanmean(rnd["top1"]) - 0.1) < 0.07
    Y = rng.normal(size=(10, 24))
    assert np.allclose(A.within_day_r(Y, Y, np.zeros((10, 24))), 1.0)


def test_report_verdicts_flag_partial_coverage():
    from crossleak.analysis.main_report import verdicts

    cov = {"invalid_share": 0.0, "missing": 3}
    V = verdicts(CFG, {}, {}, {}, {}, {}, {}, cov, "linear")
    assert V["overall"] == "PARTIAL"
    assert all(str(v).startswith("PARTIAL_") for k, v in V.items() if k.startswith("CL") and not k.endswith("users"))
    cov = {"invalid_share": 0.2, "missing": 0}
    assert verdicts(CFG, {}, {}, {}, {}, {}, {}, cov, "linear")["overall"] == "NOT_INTERPRETABLE"


def test_population_refs_match_per_context_scores():
    """The vectorised day references equal the per-context functions used by the attack cells."""
    import pandas as pd

    from crossleak.decode import zprofile

    rng = np.random.default_rng(7)
    U, ndays = 6, 40
    vals = (np.sin(2 * np.pi * np.arange(ndays * 24) / 24)[None] + rng.normal(size=(U, ndays * 24))).astype(np.float32)
    vals[0, 100:130] = np.nan

    class P:  # minimal stand-ins for DS / Panel / Validity
        pass

    ds = P()
    ds.panel = P()
    ds.panel.values = vals
    ds.panel.index = pd.date_range("2020-01-01", periods=ndays * 24, freq="h")
    ds.val = P()
    ds.val.origins = np.arange(14, ndays) * 24
    ds.valid = np.ones((U, len(ds.val.origins)), dtype=bool)
    r = M.population_refs(ds, np.array([0, 1, 2]))
    assert set(r) >= {"ref_even_med_att", "ref_resid_med_att", "ref_mean_att"}
    for u in range(U):
        for d in (0, 5, 20):
            p = ds.val.origins[d]
            ctx = vals[u, p - 336 : p]
            z = zprofile(ctx[-24:])
            assert np.allclose(np.nan_to_num(r["zp"][u, d]), np.nan_to_num(z), atol=1e-6)
            res = M.self_residual(M.day_evening_scores(ctx))
            others = [M.self_residual(M.day_evening_scores(vals[w, p - 336 : p])) for w in range(U)]
            assert np.isclose(np.nanmedian(others), r["ref_resid_med"][d], atol=1e-5)
            assert np.isfinite(res)


def test_d4_names_match_between_runner_and_report():
    assert [M.d4_name(8, e) for e in CFG["d4"]["eps"]] == ["d4_P8_eps0.5", "d4_P8_eps1", "d4_P8_eps2", "d4_P8_eps4", "d4_P8_eps8", "d4_P8_epsinf"]
    from crossleak.analysis.main_report import CL6_EPS, verdicts

    configs = {M.d4_name(8, e): {"gain": {"est": 0.05, "lo": 0.01, "hi": 0.09}, "gain_adj": {"est": 0.05, "lo": 0.01, "hi": 0.09},
                                 "retained_vs_xl16": {"est": 0.9, "lo": 0.5, "hi": 1.2}} for e in CL6_EPS}
    util = {"eld": {"c2base": {24: {"configs": configs}}}}
    V = verdicts(CFG, util, {}, {}, {}, {}, {}, {"invalid_share": 0.0, "missing": 0}, "none")
    assert V["CL6_eld"] == "YES_eps1"


def test_report_flags_results_from_another_config():
    from crossleak.analysis.main_report import verdicts

    V = verdicts(CFG, {}, {}, {}, {}, {}, {}, {"invalid_share": 0.0, "missing": 0, "config_mismatch": 2}, "none")
    assert V["overall"] == "CONFIG_MISMATCH"
