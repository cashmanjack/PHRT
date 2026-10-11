"""
How many dimensions does the data really have? If the intrinsic dimension is above ~3, no 3D
projection preserves neighborhoods, and an RT radius query on a projection returns far more
candidates than true neighbors (the RAGRT lesson).

  twonn(X)            Facco et al. 2017 (TwoNN): ratio of 2nd to 1st neighbor distance.
  mle(X, k)           Levina-Bickel 2004 with the MacKay-Ghahramani average of inverses.
  pca_energy(X)       share of variance in the top 3, 6, 9, 12 principal components.
"""
import numpy as np
from scipy.spatial import cKDTree


def _knn(X, k, sample=None, seed=0):
    rng = np.random.default_rng(seed)
    S = X if sample is None or sample >= len(X) else X[rng.choice(len(X), sample, replace=False)]
    d, _ = cKDTree(X).query(S, k=k + 1)
    return d[:, 1:]                              # drop self


def twonn(X, discard=0.1, sample=5000):
    """Facco et al.: fit -log(1 - F(mu)) = d log(mu) through the origin on the empirical CDF of
    mu = r2 / r1, dropping the largest `discard` fraction of mu (least reliable)."""
    d = _knn(X, 2, sample)
    ok = (d[:, 0] > 0) & np.isfinite(d[:, 1])
    mu = np.sort(d[ok, 1] / d[ok, 0])
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = slice(0, int(n * (1 - discard)))
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    return float(np.dot(x, y) / np.dot(x, x))


def mle(X, k=20, sample=5000):
    d = _knn(X, k, sample)
    d = d[(d > 0).all(1)]
    logs = np.log(d[:, -1:] / d[:, :-1])          # log(T_k / T_j), j < k
    inv = logs.sum(1) / (k - 2)                   # unbiased form: (k - 2) in the estimator
    return float(1.0 / np.mean(inv))


def pca_energy(X, ks=(3, 6, 9, 12)):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    e = np.cumsum(s) / s.sum()
    return {f"top{k}": float(e[min(k, len(e)) - 1]) for k in ks}
