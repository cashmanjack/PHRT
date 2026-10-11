"""Fast CPU tests (pytest tests/). GPU paths are exercised when torch.cuda is available."""
import os, sys, json, subprocess
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from phrt import datasets as DS, neighbors as N, intrinsic_dim as ID, subspace_filter as SF, profile_ph as PP


def cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------ datasets
@pytest.mark.parametrize("name", sorted(DS.REGISTRY))
def test_datasets_shape_and_determinism(name):
    X = DS.get(name, 300, seed=3)
    assert X.ndim == 2 and len(X) == 300 and np.isfinite(X).all()
    assert np.array_equal(X, DS.get(name, 300, seed=3))
    assert X.shape[1] == int(name.rsplit("_R", 1)[1])


def test_o3_is_orthogonal():
    Q = DS.get("o3_R9", 50).reshape(50, 3, 3)
    assert np.allclose(Q @ Q.transpose(0, 2, 1), np.eye(3), atol=1e-10)


def test_file_loader(tmp_path):
    X = np.random.default_rng(0).random((40, 3))
    p = tmp_path / "pc.txt"
    np.savetxt(p, X, delimiter=",")
    assert np.allclose(DS.get(f"file:{p}"), X)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.savetxt(tmp_path / "dm.txt", D)
    with pytest.raises(ValueError):
        DS.get(f"file:{tmp_path / 'dm.txt'}")


# ------------------------------------------------------------------ neighbors
@pytest.mark.parametrize("name", ["cube_R3", "o3_R9", "gmm_R128"])
def test_edge_methods_agree(name):
    X = DS.get(name, 700, seed=1)
    r = N.radius_for_degree(X, 12)
    P0, d0 = N.edges_kdtree(X, r)
    P1, d1 = N.edges_brute(X, r, chunk=97)
    assert np.array_equal(P0, P1) and np.allclose(d0, d1)
    full = np.linalg.norm(X[:, None] - X[None], axis=-1)
    iu = np.triu_indices(len(X), 1)
    assert len(P0) == int((full[iu] <= r).sum())
    if cuda():
        P2, _ = N.edges_torch(X, r)
        assert np.array_equal(P0, P2)
    else:
        P2, _ = N.edges_torch(X, r, device="cpu", chunk=128)
        assert np.array_equal(P0, P2)


def test_radius_for_degree():
    X = DS.get("cube_R3", 4000)
    r = N.radius_for_degree(X, 16)
    P, _ = N.edges_kdtree(X, r)
    assert 12 < 2 * len(P) / len(X) < 20


# ------------------------------------------------------------------ intrinsic dimension
@pytest.mark.parametrize("name,dim,tol", [("sphere2_R50", 2, 0.4), ("torus_R3", 2, 0.4), ("cube_R3", 3, 0.5),
                                          ("gauss_R16", 16, 5.0)])
def test_intrinsic_dimension(name, dim, tol):
    X = DS.get(name, 4000)
    assert abs(ID.twonn(X) - dim) < tol, ID.twonn(X)
    assert abs(ID.mle(X) - dim) < tol * 1.5, ID.mle(X)


def test_pca_energy_linear_span():
    e = ID.pca_energy(DS.get("sphere2_R50", 2000))
    assert e["top3"] > 0.99


# ------------------------------------------------------------------ subspace filter
@pytest.mark.parametrize("name", ["klein_R4", "o3_R9", "gmm_R128"])
def test_filter_is_lossless_and_monotone(name):
    X = DS.get(name, 500, seed=2)
    r = N.radius_for_degree(X, 10)
    c = SF.filter_counts(X, r, chunk=64)
    P, _ = N.edges_kdtree(X, r)
    assert c["true_edges"] == len(P)
    s, p = c["slice_k"], c["pyth_k"]
    assert all(a >= b for a, b in zip(s, s[1:])) and all(a >= b for a, b in zip(p, p[1:]))
    assert all(si >= pi >= c["true_edges"] for si, pi in zip(s, p)), "a lossless filter can never drop an edge"
    assert p[-1] == c["true_edges"], "all slices together = the full distance"
    if cuda():
        g = SF.filter_counts(X, r, chunk=64, device="cuda")
        assert g["slice_k"] == s and g["pyth_k"] == p and g["true_edges"] == c["true_edges"]


def test_filter_exact_for_linear_3d_data():
    X = DS.get("sphere2_R50", 600)
    X = X - X.mean(0)
    r = N.radius_for_degree(X, 10)
    c = SF.filter_counts(X, r)
    assert c["rt_candidates_per_edge"] < 1.6     # data is (almost) in a 3D subspace: the top slice nearly suffices


def test_filter_blows_up_in_high_dim():
    X = DS.get("gauss_R16", 1500)
    c = SF.filter_counts(X, N.radius_for_degree(X, 10))
    assert c["rt_candidates_per_edge"] > 10      # the RAGRT failure mode, reproduced on toy data


# ------------------------------------------------------------------ Ripser profile
def test_profile_circle_has_one_long_h1_bar():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    prof = PP.profile_rips(X, 0.5, maxdim=1, methods=("kdtree", "brute"), repeats=1, with_ripserpp=False)
    assert prof["edges"] > 0 and prof["amdahl_ceiling"] >= 1.0 and 0 < prof["nbr_fraction"] < 1
    from ripser import ripser
    res, _ = PP.ripser_sparse(len(X), *N.edges_kdtree(X, 0.5), 1)
    h1 = res["dgms"][1]
    life = np.where(np.isinf(h1[:, 1]), np.inf, h1[:, 1] - h1[:, 0])
    assert (life > 0.3).sum() == 1


def test_sparse_matches_dense_ripser_below_threshold():
    from ripser import ripser
    X = DS.get("torus_R3", 300)
    r = N.radius_for_degree(X, 20)
    sp, _ = PP.ripser_sparse(len(X), *N.edges_kdtree(X, r), 1)
    de = ripser(X, maxdim=1, thresh=r)
    for a, b in zip(sp["dgms"], de["dgms"]):
        a = a[np.argsort(a[:, 0] + 1e-9 * np.nan_to_num(a[:, 1], posinf=1e9))]
        b = b[np.argsort(b[:, 0] + 1e-9 * np.nan_to_num(b[:, 1], posinf=1e9))]
        assert len(a) == len(b) and np.allclose(a, b, atol=1e-6)


def test_killtest_cli_smoke(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "killtest.py"), "--tag", "t", "--n", "600",
           "--datasets", "sphere2_R3,o3_R9", "--degrees", "8", "--repeats", "1", "--out", str(tmp_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    R = json.load(open(tmp_path / "t" / "killtest.json"))
    assert set(R["datasets"]) == {"sphere2_R3", "o3_R9"}
    assert R["datasets"]["sphere2_R3"]["configs"][0]["filter"] is None
    assert R["datasets"]["o3_R9"]["configs"][0]["filter"]["rt_candidates_per_edge"] >= 1
    assert "Decision rule" in open(tmp_path / "t" / "report.md").read()
