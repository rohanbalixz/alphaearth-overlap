"""E1 (reproduction of Ma et al. 2026, Table 6), E3 (132-pair transfer matrix and target oracle) and
E4 (overlap measures) for every input arm.

Writes results/e1_region_transfer.csv, results/pairs.csv (one row per arm x ordered pair) and
results/environment.txt. Run: python -m aeoverlap.pipeline
"""
from __future__ import annotations

import hashlib
import os
import platform
import time
from importlib.metadata import version

import pandas as pd

from .data import EAST, RESULTS, SOURCE_CSV, STATES, WEST, load, state_frames
from .overlap import full_space, knn_cover, rivals, yield_range_cover, yield_subspace
from .transfer import oracle, pair_matrix, region_transfer


def _environment(started: float, finished: float) -> str:
    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]
    sha = hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest()
    return "\n".join([
        f"python {platform.python_version()} on {platform.platform()}",
        f"cpu_count {os.cpu_count()}",
        *[f"{p} {version(p)}" for p in pkgs],
        f"source_csv_sha256 {sha}",
        f"runtime_seconds {finished - started:.0f}",
        f"finished_utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(finished))}",
    ]) + "\n"


def main() -> None:
    t0 = time.time()
    df, arms = load()
    frames = state_frames(df)
    RESULTS.mkdir(exist_ok=True)
    print(f"[data] {len(df)} county-years, arms: {list(arms)}")

    reg = region_transfer(df, arms, EAST, WEST)
    reg.to_csv(RESULTS / "e1_region_transfer.csv", index=False)
    print("[E1] East<->West (Ma et al. 2026, Table 6, county corn RF: AEF 0.02/0.51, RS 0.66/0.60)\n",
          reg[["arm", "direction", "R2", "R2_level", "r2_corr", "rmse"]].round(3).to_string(index=False),
          f"\n  ({time.time() - t0:.0f}s)", flush=True)

    desc = pd.DataFrame([dict(src=s, tgt=t, yield_range_cover=yield_range_cover(frames[s], frames[t]),
                              n_src=len(frames[s]), n_tgt=len(frames[t]), **rivals(frames[s], frames[t]))
                         for s in STATES for t in STATES if s != t])

    out = []
    for arm, cols in arms.items():
        orc = oracle(frames, cols)
        tr = pair_matrix(frames, cols)
        tr["R2_oracle"] = tr.tgt.map(orc)
        ov = []
        for s in STATES:
            for t in STATES:
                if s == t:
                    continue
                S, T = frames[s], frames[t]
                f, u = full_space(S, T, cols), yield_subspace(S, T, cols)
                ov.append(dict(src=s, tgt=t, full_auc=f["auc"], full_cover=f["cover"],
                               full_share2s=f["share2s"], sub_auc=u["auc"], sub_cover=u["cover"],
                               sub_share2s=u["share2s"], knn_cover=knn_cover(S, T, cols)))
        m = tr.merge(pd.DataFrame(ov), on=["src", "tgt"]).merge(desc, on=["src", "tgt"])
        m.insert(0, "arm", arm)
        out.append(m)
        print(f"[E3/E4] {arm}: done ({time.time() - t0:.0f}s); oracle R2 by state:",
              {k: round(v, 2) for k, v in orc.items()}, flush=True)

    pairs = pd.concat(out, ignore_index=True)
    pairs["loss_total"] = pairs.R2_oracle - pairs.R2
    pairs["loss_level"] = pairs.R2_level - pairs.R2
    pairs["loss_rank"] = pairs.R2_oracle - pairs.R2_level
    pairs.to_csv(RESULTS / "pairs.csv", index=False)
    (RESULTS / "environment.txt").write_text(_environment(t0, time.time()))
    print(f"[done] wrote results/pairs.csv ({len(pairs)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
