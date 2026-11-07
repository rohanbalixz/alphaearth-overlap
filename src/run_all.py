"""E1 (reproduction), E3 (132-pair transfer matrix + oracle) and E4 (overlap) for every arm.
Writes results/e1_region_transfer.csv and results/pairs.csv (one row per arm x ordered pair)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import EAST, RESULTS, STATES, WEST, load, state_frames
from overlap import full_space, knn_cover, rivals, yield_range_cover, yield_subspace
from transfer import oracle, pair_matrix, region_transfer


def main():
    t0 = time.time()
    df, arms = load()
    frames = state_frames(df)
    RESULTS.mkdir(exist_ok=True)
    print(f"[data] {len(df)} county-years, arms: {list(arms)}")

    reg = region_transfer(df, arms, EAST, WEST)
    reg.to_csv(RESULTS / "e1_region_transfer.csv", index=False)
    print("[E1] East<->West (paper Table 6 county corn RF: AEF 0.02/0.51, RS 0.66/0.60)\n",
          reg[["arm", "direction", "R2", "R2_level", "r2_corr", "rmse"]].round(3).to_string(index=False),
          f"\n  ({time.time() - t0:.0f}s)")

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
              {k: round(v, 2) for k, v in orc.items()})

    pairs = pd.concat(out, ignore_index=True)
    pairs["loss_total"] = pairs.R2_oracle - pairs.R2
    pairs["loss_level"] = pairs.R2_level - pairs.R2
    pairs["loss_rank"] = pairs.R2_oracle - pairs.R2_level
    pairs.to_csv(RESULTS / "pairs.csv", index=False)
    print(f"[done] wrote results/pairs.csv ({len(pairs)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
