# PHRT: persistent homology on RT cores (kill test first)

Question: can RT cores speed up Vietoris-Rips persistent homology? An RT core answers one kind
of query fast: which primitives does a ray (or a small sphere) touch in 3D. In Rips PH that maps
to exactly one phase, enumerating the edges with d(x_i, x_j) <= r. Before building anything,
this repo measures whether that phase is worth accelerating and whether it can be done exactly
when the data is not 3D.

## What the kill test measures (`scripts/killtest.py`)

| measurement | why it matters |
|---|---|
| Amdahl ceiling = (best non-RT neighbor search + Ripser reduction) / reduction | the speedup if RT made neighbor search free; below ~1.5x there is no paper |
| neighbor search: k-d tree (CPU) and brute force on the GPU (torch) | RT has to beat the best of these, not a naive loop |
| intrinsic dimension (TwoNN, MLE) and PCA energy | above ~3 intrinsic dims no 3D projection keeps neighborhoods |
| RT candidates per true edge on the top-3 PCA slice (lossless subspace filter) | what an RT query on a 3D projection would return; everything above 1 is wasted work verified on the SMs (RAGRT's failure mode) |

Thresholds are set per dataset for target mean degrees (sparse Rips). Ripser++ (GPU) is timed
when `ripserplusplus` is importable; a faster reduction raises the ceiling, so check both.

Decision rule: build RT-PH only for datasets with ceiling >= 1.5x AND (native <= 3D OR <= 4 RT
candidates per true edge).

## Run

```
pip install -r requirements.txt            # numpy scipy ripser pytest (+ torch for the GPU path)
python3 -m pytest -q tests                  # 30 tests, ~20 s on CPU
python3 scripts/killtest.py --tag smoke --n 2000 --datasets sphere2_R3,o3_R9 --degrees 8,32
python3 scripts/killtest.py --tag full --n 20000 --degrees 8,16,32,64 --maxdim 1
python3 scripts/killtest.py --tag full_h2 --n 5000 --degrees 8,16 --maxdim 2
python3 scripts/fetch_benchmarks.py        # Ripser examples + PH-roadmap point clouds into data/
python3 scripts/killtest.py --tag bench --datasets file:data/ripser/<file> --n 5000
```
Results: `results/<tag>/report.md` and `killtest.json`.

## Layout

- `phrt/datasets.py`: seeded synthetic clouds (native 3D: sphere, torus, Swiss roll, cube, clusters;
  higher-D: Klein bottle R^4, O(3) in R^9, sphere in R^50, curved 5-manifold in R^64, Gaussian R^16,
  embedding-like mixture R^128) and a point-cloud file loader.
- `phrt/neighbors.py`: exact radius edges (k-d tree, numpy brute force, torch GPU), radius for a target degree.
- `phrt/intrinsic_dim.py`: TwoNN, Levina-Bickel MLE, PCA energy.
- `phrt/subspace_filter.py`: PCA slices, lossless slice filters, candidate counts.
- `phrt/profile_ph.py`: phase timing and the Amdahl ceiling.
