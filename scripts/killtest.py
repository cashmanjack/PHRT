"""
PHRT kill test: should persistent homology go on RT cores at all?

For each dataset: intrinsic dimension (TwoNN, MLE), PCA energy, then for each target average
degree (sparse Rips thresholds): the Amdahl ceiling of making neighbor search free, and the
size of the candidate set an RT radius query on the top-3D PCA slice would return relative to
the true edges (lossless subspace filter). Writes results/<tag>/killtest.json and report.md.

  python3 scripts/killtest.py --tag smoke --n 2000 --datasets sphere2_R3,o3_R9 --degrees 8,32
  python3 scripts/killtest.py --tag full  --n 20000           # all built-in datasets
  python3 scripts/killtest.py --tag bench --datasets file:data/ripser/o3_4096.txt --n 4096

Decision rule (stated in the report): RT-PH is worth building only for datasets where
  (a) the Amdahl ceiling vs the best non-RT neighbor search is >= 1.5x, and
  (b) the data is natively <= 3D, or the top-slice RT candidates are <= 4 per true edge.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phrt import datasets as DS, neighbors as N, intrinsic_dim as ID, subspace_filter as SF, profile_ph as PP


def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--datasets", default=",".join(DS.LOW_DIM + DS.HIGH_DIM))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--degrees", default="8,16,32,64")
    ap.add_argument("--maxdim", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--filter_n", type=int, default=8000, help="subsample for the O(n^2) filter counts")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    args = ap.parse_args()
    dev = "cuda" if has_cuda() else None
    methods = ("kdtree", "torch") if dev else ("kdtree", "brute")
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    R = {"args": vars(args), "device": dev or "cpu", "ripserpp": PP.ripserpp_available(), "datasets": {}}
    for name in args.datasets.split(","):
        X = DS.get(name, args.n, seed=0)
        Y = X if len(X) <= args.filter_n else X[np.random.default_rng(1).choice(len(X), args.filter_n, replace=False)]
        info = {"n": int(len(X)), "D": int(X.shape[1]), "twonn": ID.twonn(X), "mle": ID.mle(X),
                "pca": ID.pca_energy(X), "configs": []}
        print(f"== {name}: n={len(X)} D={X.shape[1]} TwoNN {info['twonn']:.2f} MLE {info['mle']:.2f} "
              f"PCA top3 {info['pca']['top3']:.2f}", flush=True)
        for k in [int(x) for x in args.degrees.split(",")]:
            r = N.radius_for_degree(X, k)
            t0 = time.time()
            prof = PP.profile_rips(X, r, args.maxdim, methods=methods, device=dev or "cpu", repeats=args.repeats)
            filt = SF.filter_counts(Y, r, device=dev) if X.shape[1] > 3 else None
            cfg = {"degree_target": k, **prof, "filter": filt}
            info["configs"].append(cfg)
            f = f"  RT cand/edge {filt['rt_candidates_per_edge']:7.1f}" if filt else "  (native <= 3D: RT query is exact)"
            print(f"  deg {k:>3}: edges {prof['edges']:>10,}  nbr {prof['t_nbr_best']*1e3:8.1f} ms  "
                  f"reduce {prof['t_reduce']*1e3:9.1f} ms  ceiling {prof['amdahl_ceiling']:.2f}x{f}  ({time.time()-t0:.0f} s)", flush=True)
        R["datasets"][name] = info
        json.dump(R, open(os.path.join(out_dir, "killtest.json"), "w"), indent=1)
    write_report(R, out_dir)


def write_report(R, out_dir):
    L = [f"# PHRT kill test ({R['args']['tag']})", "",
         f"- device {R['device']}, maxdim {R['args']['maxdim']}, Ripser++ timed: {R['ripserpp']}",
         "- ceiling = (best non-RT neighbor search + reduction) / reduction: the speedup if neighbor search were free",
         "- RT cand/edge = pairs an RT radius query on the top-3 PCA slice returns per true edge (lossless filter; "
         "only for D > 3)", "",
         "| dataset | D | TwoNN | MLE | PCA top3 | target deg | edges | nbr ms | reduce ms | nbr share | ceiling | RT cand/edge | pass |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, info in R["datasets"].items():
        for c in info["configs"]:
            f = c["filter"]
            cand = f"{f['rt_candidates_per_edge']:.1f}" if f else "exact (<= 3D)"
            ok = c["amdahl_ceiling"] >= 1.5 and (f is None or f["rt_candidates_per_edge"] <= 4)
            L.append(f"| {name} | {info['D']} | {info['twonn']:.1f} | {info['mle']:.1f} | {info['pca']['top3']:.2f} | "
                     f"{c['degree_target']} | {c['edges']:,} | {c['t_nbr_best']*1e3:.1f} | {c['t_reduce']*1e3:.1f} | "
                     f"{c['nbr_fraction']:.1%} | {c['amdahl_ceiling']:.2f}x | {cand} | {'yes' if ok else 'no'} |")
    L += ["", "Decision rule: build RT-PH only for rows that pass, i.e. ceiling >= 1.5x AND (native <= 3D OR <= 4 RT "
          "candidates per true edge). If no dataset you care about passes, stop."]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_dir}/report.md")


if __name__ == "__main__":
    main()
