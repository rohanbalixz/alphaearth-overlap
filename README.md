# Overlap and transfer of AlphaEarth crop-yield models across Corn Belt states

A small, pre-registered test built on the county corn data shared by
Ma, Shen, Swatantran & Lobell (2026), *Harvesting AlphaEarth*, IJAEOG
([repo](https://github.com/yuchima8/Harvest_AlphaEarth)).

**Question.** Their Table 6 shows AlphaEarth (AEF) yield models transferring poorly between the
eastern and western Corn Belt, while Landsat (RS) models hold up, and their Figure 8 shows the two
regions' embeddings barely overlapping. With 12 states there are 132 ordered transfer pairs, not
two. That is enough to ask whether overlap, measured without any target yields, predicts how much
a model loses when moved, and whether thinner overlap is why AEF loses more.

The predictions, measures and tests were fixed in [PREREGISTRATION.md](PREREGISTRATION.md) before
any pair-level transfer loss was computed.

## Run

```bash
./get_data.sh                                   # downloads the paper's shared CSV into data/
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_units.py            # synthetic checks
.venv/bin/python src/run_all.py                 # E1 reproduction, E3 132-pair matrix, E4 overlap
.venv/bin/python src/run_tests.py               # pre-registered tests -> results/summary.md
.venv/bin/python src/make_figure.py             # results/figure.png
```

RS arm: `gee/export_landsat_harmonics.py` (Earth Engine) and then `src/rs_features.py`. Once
`data/rs_features.csv` exists, the scripts above add the `RS` and `RS-noclim` arms automatically.
**Gate:** the RS arm must reproduce Table 6 (East->West R2 ~0.66, West->East ~0.60) before its
results are used.

## Results (AlphaEarth, AlphaEarth + climate, climate only; the RS arm is pending the Earth Engine export)

![figure](results/figure.png)

**Reproduction (E1).** County corn, random forest, East<->West ecoregions:

| Input | East->West R2 | West->East R2 | East->West R2 with the target's mean yield matched |
|---|---|---|---|
| AlphaEarth | 0.07 (paper 0.02) | 0.51 (paper 0.51) | 0.39 |
| AlphaEarth + climate | 0.06 | 0.52 | 0.38 |
| Climate only | -0.08 | -0.39 | 0.13 |

Adding the RS baseline's climate inputs does not change AlphaEarth's transfer, so the RS
advantage in the paper's Table 6 comes from the Landsat features. Most of the East->West failure
is a level error: on average the eastern model over-predicts western yields by 23 bu/ac.

**Across all 132 state pairs (E3, E4).**

- In AlphaEarth space, states are islands. Domain AUC is >= 0.99 for 130 of 132 ordered pairs
  (climate only: 58 of 132). The planned "share of target inside [0.05, 0.95]" statistic is
  therefore near zero for almost every pair.
- Transfers between states usually fail. AlphaEarth's median transfer R2 is -0.02, and 68 of the
  132 pairs are negative, against a median of 0.74 for a model trained in the target state.
  Matching the target's mean yield raises the median to 0.24. About 43% of the loss is a level
  error; the rest is lost ranking.

**Pre-registered tests (E5, E6), AlphaEarth arm.** QAP permutation p-values (9,999 permutations
of state labels):

| | Prediction | Result |
|---|---|---|
| P2 (primary) | Yield-subspace overlap is negatively related to ranking loss | **Supported.** rho = -0.27, p = 0.0045 |
| P3 | That relation survives GDD distance, geographic distance and state fixed effects | **Not supported.** Partial coefficient -0.16, p = 0.09. It does beat GDD distance (rho = 0.12, p = 0.24). |
| P4 | Overlap relates more to the ranking part than to the level part | **Supported.** abs(rho) 0.27 vs 0.18 |
| P5 | Lower yield-range coverage explains which direction has the larger level loss | **Not supported.** 32/66 pairs, 48% |
| P1, P6 | AlphaEarth overlap is thinner than Landsat overlap, and that gap tracks the loss gap | Pending the RS export |

**The main caveat (exploratory).** The plain distance between states predicts ranking loss as
well as any overlap measure does (rho = 0.41, p = 0.002). In AlphaEarth space, overlap and
geography move together (rho = -0.35 between subspace overlap and distance). That is what you
would expect if the embeddings encode location. The full-space measures (one-sided coverage
rho = -0.42, kNN coverage rho = -0.40) also outperform the pre-registered subspace measure.
They are non-zero mainly for neighbouring states, though, so they largely measure adjacency.
Climate only: P2 is not supported (rho = -0.16, p = 0.17).

**What this does and does not show.** Overlap computed without any target yields does carry
information about which AlphaEarth transfers will lose most. With these inputs, though, it can't
be separated from how far apart the states are. The RS arm is the test that could separate them:
if Landsat features overlap across distant but agronomically similar states, overlap and geography
come apart there.

All numbers: [results/summary.md](results/summary.md), [results/tests.json](results/tests.json),
and per-pair rows in [results/pairs.csv](results/pairs.csv) (the table view of the figure).
