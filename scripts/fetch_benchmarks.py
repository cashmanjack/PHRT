"""
Download standard PH benchmark point clouds into data/ via the GitHub API (no git needed):
Ripser's examples and the PH-roadmap data sets (Otter et al. 2017). Lists each repo's tree,
keeps text/CSV files under the given path up to --max_mb, and skips files whose names say they
are distance matrices.

  python3 scripts/fetch_benchmarks.py            # then: --datasets file:data/<repo>/<file>
"""
import os, sys, json, argparse, urllib.request

SOURCES = [("Ripser/ripser", "master", "examples"), ("n-otter/PH-roadmap", "master", "data_sets")]


def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "phrt"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--max_mb", type=float, default=50)
    args = ap.parse_args()
    for repo, branch, prefix in SOURCES:
        try:
            tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")["tree"]
        except Exception as e:
            print(f"!! {repo}: {e}"); continue
        files = [t for t in tree if t["type"] == "blob" and t["path"].startswith(prefix)
                 and t["path"].lower().endswith((".txt", ".csv", ".xyz")) and t.get("size", 0) <= args.max_mb * 2**20
                 and not any(w in t["path"].lower() for w in ("distance", "_dm", "lower_", "upper_"))]
        print(f"{repo}: {len(files)} candidate point-cloud files under {prefix}/")
        for t in files:
            dst = os.path.join(args.out, repo.split("/")[1], os.path.basename(t["path"]))
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{t['path']}"
            try:
                urllib.request.urlretrieve(url, dst)
                print(f"  {dst} ({t.get('size', 0) / 1e6:.1f} MB)")
            except Exception as e:
                print(f"  !! {url}: {e}")


if __name__ == "__main__":
    main()
