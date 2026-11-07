# Overlap and cross-region transfer of satellite crop-yield models

**A pre-registered test with AlphaEarth embeddings across 12 U.S. Corn Belt states**

Geospatial foundation-model embeddings such as AlphaEarth Foundations (AEF) match traditional
remote-sensing features for county corn-yield prediction when trained locally, but transfer much
worse across regions (Ma, Shen, Swatantran & Lobell, 2026). This repository asks whether that
failure can be anticipated without any yields from the target region. The measure is overlap: how
well the training region's data cover the target region in feature space, which is the positivity
condition of causal inference.

The analysis uses the authors' shared data: 6,325 county-years, 12 states, 2017–2024. It
reproduces their East↔West result, trains a model in every state and tests it in every other
(132 transfers), and evaluates predictions committed in a
[pre-registration](PREREGISTRATION.md) before any transfer loss was computed.

![Summary figure](results/figure.png)

## Key findings

1. **In AlphaEarth space, every state is an island.** A cross-validated domain classifier
   separates the two states with AUC ≥ 0.99 for 130 of 132 ordered pairs. With climate
   covariates alone, it does so for 58 of 132. Strict overlap in the full embedding space
   essentially never holds between states.
2. **Most cross-state transfers fail.** The median AEF transfer R² is −0.02, and 68 of 132 are
   negative, against 0.74 for a model trained in the target state. About 43% of the loss is a
   shift in level that the target's mean yield would correct; the rest is lost ranking skill.
3. **Overlap measured without target yields predicts the ranking loss** (pre-registered primary
   test: Spearman ρ = −0.27, QAP permutation p = 0.0045). It does so better than the difference
   in growing degree days between states (ρ = 0.12, p = 0.24).
4. **But it adds little beyond geography.** Distance between states predicts the ranking loss at
   least as well (ρ = 0.41, p = 0.002). The partial association controlling for distance, GDD
   difference and state fixed effects is not significant (p = 0.09). In AEF space, overlap and
   geography move together.
5. **Climate inputs do not explain the Landsat baseline's advantage.** Adding the RS baseline's
   ten climate covariates to AEF leaves East↔West transfer unchanged (0.06 / 0.52 vs 0.07 /
   0.51). The advantage in the source study's Table 6 therefore lies in the Landsat features.

### East↔West reproduction (E1)

Random forest, 200 trees, mean of 5 seeds.

| Input | East→West R² | West→East R² | East→West R², target mean matched |
|---|---|---|---|
| AlphaEarth | 0.07 (paper: 0.02) | 0.51 (paper: 0.51) | 0.39 |
| AlphaEarth + climate | 0.06 | 0.52 | 0.38 |
| Climate only | −0.08 | −0.39 | 0.13 |

### Pre-registered predictions (AlphaEarth arm)

| | Prediction | Result |
|---|---|---|
| **P2** (primary) | Yield-subspace overlap is negatively associated with ranking loss | **Supported.** ρ = −0.27, p = 0.0045 |
| P3 | The association survives distance, GDD difference and state fixed effects | **Not supported.** Partial coefficient −0.16, p = 0.09 |
| P4 | Overlap relates more to the ranking part of the loss than to the level part | **Supported.** abs(ρ) 0.27 vs 0.18 |
| P5 | Lower yield-range coverage explains which direction has the larger level loss | **Not supported.** 32 of 66 pairs |
| P1, P6 | AEF overlap is thinner than Landsat overlap, and the gap tracks the loss gap | **Not tested.** See [Landsat arm](#landsat-arm) |

All numbers are in [results/summary.md](results/summary.md) and
[results/tests.json](results/tests.json). Per-pair rows are in
[results/pairs.csv](results/pairs.csv), documented in [results/README.md](results/README.md).

## Reproducing the results

Requires Python ≥ 3.9 and 15–25 minutes on an 11-core laptop.

```bash
make setup      # .venv with the pinned environment (requirements.txt) + this package
make data       # source table, pinned to upstream commit 2ec9163, SHA-256 verified
make test       # unit tests on synthetic data with known answers
make results    # E1 reproduction, E3 132-pair transfer matrix, E4 overlap measures
make evaluate   # E5/E6 pre-registered tests -> results/tests.json, results/summary.md
make figure     # results/figure.png
```

All randomness is seeded. A full rerun from this package matched every committed value to within
3 × 10⁻¹⁴, which is floating-point rounding from parallel summation. `results/environment.txt` records the
versions, hardware and runtime of the committed run.

## Repository layout

```
├── PREREGISTRATION.md        predictions, measures and tests, committed before results (+ Amendment 1)
├── docs/METHODS.md           full specification, deviations from the source study, references
├── src/aeoverlap/
│   ├── data.py               loading, input arms, study constants
│   ├── transfer.py           random forest, oracle, loss decomposition
│   ├── overlap.py            domain-classifier, subspace and kNN overlap; rival predictors
│   ├── inference.py          QAP permutation and DSP partial tests for dyadic data
│   ├── pipeline.py           E1, E3, E4
│   ├── evaluate.py           E5, E6 (pre-registered tests)
│   ├── figures.py            summary figure
│   └── rs_features.py        Landsat harmonic coefficients -> phenology features
├── gee/                      Earth Engine export for the Landsat arm (see gee/README.md)
├── tests/                    pytest suite (synthetic data; runs in CI)
├── scripts/get_data.sh       pinned, checksum-verified data download
└── results/                  committed outputs of the run above
```

## Data

The source table was shared by Ma et al. (2026) at
[github.com/yuchima8/Harvest_AlphaEarth](https://github.com/yuchima8/Harvest_AlphaEarth).
It contains:

- NASS county corn yields;
- AlphaEarth annual embeddings averaged over CDL corn pixels;
- monthly ERA5-Land GDD and precipitation;
- county boundaries.

The upstream repository has no license file, so the data are **not redistributed** here.
`make data` downloads them from a pinned commit and verifies the checksum.

## Methods in brief

- **Loss decomposition.** Each transfer's loss, relative to a model trained in the target state,
  is split into a level part, removable by matching the target's mean yield, and a ranking part.
- **Overlap.** Coverage is the share of target county-years that the source covers. It is
  estimated with cross-fitted, class-balanced domain classifiers that never see target yields,
  using folds that hold out whole counties.
- **Primary measure.** Coverage in a three-component PLS subspace of yield-relevant directions,
  fitted on the source state. Overlap in all 64 dimensions is saturated, consistent with D'Amour
  et al. (2021).
- **Inference.** The 132 pairs share states, so inference uses QAP permutation of state labels
  and double-semi-partialling for partial effects.

Full details are in [docs/METHODS.md](docs/METHODS.md).

## Landsat arm

The comparison that could separate overlap from geography is the Landsat (RS) arm. If Landsat
features overlap across distant but agronomically similar states, the two come apart there.

The arm is implemented in [`gee/`](gee/README.md) and passed a one-county check on Earth Engine.
A single-year, 12-state export, however, consumed Earth Engine compute at roughly 25 EECU-hours
per minute, so the eight-year export does not fit the free noncommercial tier. P1 and P6 are
therefore untested. That cost is itself consistent with the source study's point (Sec. 5.1) that
embeddings are far cheaper to obtain than Landsat time-series features.

## Limitations

- **Small effective sample.** Twelve states give 132 dependent pairs. QAP accounts for the
  dependence but leaves power modest.
- **Model dependence.** The random forest, the source study's model, cannot predict outside its
  training-label range. This inflates level losses for low-yield targets, and the results are
  conditional on that choice.
- **Confounding with geography.** In this design, overlap and distance are correlated (ρ = −0.35),
  so the design cannot separate their effects.
- **Choice of primary measure.** K = 3 for the PLS subspace was chosen after the probe showed
  full-space overlap to be saturated, and before any transfer loss was computed. This is disclosed
  in the pre-registration.
- **Scope.** One crop, county scale, the United States, and years shared between source and
  target, so year-level weather shocks are common to both.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the source data:

> Ma, Y., Shen, Y., Swatantran, A., Lobell, D.B. (2026). Harvesting AlphaEarth: Benchmarking the
> geospatial foundation model for agricultural downstream tasks. *International Journal of
> Applied Earth Observation and Geoinformation*, 105258.
> https://doi.org/10.1016/j.jag.2026.105258

## License

The code and derived results are under the MIT license; see [LICENSE](LICENSE). The source data
belong to their authors. This project is independent and is not affiliated with or endorsed by
the authors of the source study.
