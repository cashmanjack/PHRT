"""
Point clouds for the kill test. Synthetic sets are deterministic (seeded) and cover the two
regimes that matter for RT cores: natively low-dimensional (ambient D <= 3, where an RT radius
query is exact) and higher-dimensional (where RT can only see a 3D projection).

get(name, n, seed) -> float64 [n, D]. Names: see REGISTRY, or "file:<path>" for a text/CSV/npy
point cloud (one point per row), e.g. the Ripser / PH-roadmap benchmark files fetched by
scripts/fetch_benchmarks.py.
"""
import os
import numpy as np


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def sphere2_R3(n, rng, noise=0.01):
    return _unit(rng.standard_normal((n, 3))) + noise * rng.standard_normal((n, 3))


def torus_R3(n, rng, R=2.0, r=0.7, noise=0.01):
    u, v = rng.uniform(0, 2 * np.pi, (2, n))
    X = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
    return X + noise * rng.standard_normal(X.shape)


def swissroll_R3(n, rng, noise=0.02):
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 10 * rng.random(n)
    X = np.stack([t * np.cos(t), h, t * np.sin(t)], 1) / 10
    return X + noise * rng.standard_normal(X.shape)


def cube_R3(n, rng):
    return rng.random((n, 3))


def clusters_R3(n, rng, k=12, spread=0.05):
    c = rng.random((k, 3))
    return c[rng.integers(0, k, n)] + spread * rng.standard_normal((n, 3))


def klein_R4(n, rng, noise=0.01):
    """Figure-8 Klein bottle in R^4 (intrinsic dim 2)."""
    u, v = rng.uniform(0, 2 * np.pi, (2, n))
    a = 2.0
    w = a + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)
    X = np.stack([w * np.cos(u), w * np.sin(u), np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v),
                  np.cos(v)], 1)
    return X + noise * rng.standard_normal(X.shape)


def o3_R9(n, rng):
    """Random 3x3 orthogonal matrices as points in R^9 (Ripser's o3 benchmark; intrinsic dim 3)."""
    Q, R = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    Q = Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]
    return Q.reshape(n, 9)


def _random_rotation(D, rng):
    Q, R = np.linalg.qr(rng.standard_normal((D, D)))
    return Q * np.sign(np.diag(R))


def sphere2_R50(n, rng, noise=0.001):
    """A 2-sphere rotated into R^50 (intrinsic 2, linear span 3: PCA recovers it exactly)."""
    X = np.zeros((n, 50))
    X[:, :3] = _unit(rng.standard_normal((n, 3)))
    return X @ _random_rotation(50, rng).T + noise * rng.standard_normal((n, 50))


def curved5_R64(n, rng, noise=0.01):
    """A curved 5-dimensional manifold in R^64 (random Fourier features of a 5D latent)."""
    z = rng.random((n, 5)) * 2 * np.pi
    W = rng.standard_normal((5, 32))
    b = rng.uniform(0, 2 * np.pi, 32)
    X = np.concatenate([np.cos(z @ W * 0.3 + b), np.sin(z @ W * 0.3 + b)], 1) / np.sqrt(32)
    return X + noise * rng.standard_normal(X.shape)


def gauss_R16(n, rng):
    return rng.standard_normal((n, 16))


def gmm_R128(n, rng, k=64, spread=0.35):
    """Unit vectors around 64 random directions in R^128: a stand-in for learned embeddings."""
    c = _unit(rng.standard_normal((k, 128)))
    return _unit(c[rng.integers(0, k, n)] + spread * rng.standard_normal((n, 128)) / np.sqrt(128) * 4)


REGISTRY = {
    "sphere2_R3": sphere2_R3, "torus_R3": torus_R3, "swissroll_R3": swissroll_R3, "cube_R3": cube_R3,
    "clusters_R3": clusters_R3, "klein_R4": klein_R4, "o3_R9": o3_R9, "sphere2_R50": sphere2_R50,
    "curved5_R64": curved5_R64, "gauss_R16": gauss_R16, "gmm_R128": gmm_R128,
}
LOW_DIM = ["sphere2_R3", "torus_R3", "swissroll_R3", "cube_R3", "clusters_R3"]
HIGH_DIM = ["klein_R4", "o3_R9", "sphere2_R50", "curved5_R64", "gauss_R16", "gmm_R128"]


def load_file(path):
    if path.endswith(".npy"):
        X = np.load(path)
    else:
        with open(path) as f:
            rows = [l.replace(",", " ").split() for l in f if l.strip() and not l.lstrip().startswith("#")]
        width = max(len(r) for r in rows)
        rows = [r for r in rows if len(r) == width]
        X = np.array(rows, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == X.shape[1] and np.allclose(X, X.T) and np.allclose(np.diag(X), 0):
        raise ValueError(f"{path} looks like a distance matrix, not a point cloud")
    return X.astype(np.float64)


def get(name, n=None, seed=0):
    rng = np.random.default_rng(seed)
    if name.startswith("file:"):
        X = load_file(name[5:])
        if n is not None and n < len(X):
            X = X[np.sort(rng.choice(len(X), n, replace=False))]
        return X
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name}; known: {sorted(REGISTRY)} or file:<path>")
    return REGISTRY[name](n or 2000, rng).astype(np.float64)
