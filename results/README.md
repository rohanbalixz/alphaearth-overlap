# Results

Every file here is produced by the Makefile targets and contains derived metrics only; the source
data are not included.

**Naming convention.**
- `e1`–`e6` are the pre-registered experiments ([PREREGISTRATION.md](../PREREGISTRATION.md)).
- `f1`–`f3` are exploratory follow-ups, designed after the pre-registered results were known.
- `fig1` and `fig2` are figures.
- Each run folder holds its own `run.log` and `environment.txt` (versions, hardware, source-data
  SHA-256, runtime).

```
results/
├── preregistered/                     make preregistered evaluate
│   ├── e1_region_transfer.csv         E1: East<->West reproduction of Ma et al. (2026), Table 6
│   ├── e3_e4_pairs.csv                E3/E4: transfer metrics and overlap, 3 arms x 132 ordered pairs
│   ├── e5_e6_tests.json               E5/E6: pre-registered tests and secondary correlations
│   ├── e5_e6_summary.md               readable summary of e5_e6_tests.json
│   ├── environment.txt
│   └── run.log
├── exploratory/                       make exploratory
│   ├── f1_pairs.csv                   F1: ranking loss on covered vs uncovered county-years, per pair
│   ├── f1_county_years.csv.gz         F1: per county-year errors for every AEF transfer
│   ├── f1_irrigation_within_pairs.csv F1: within-pair irrigation-residual correlations
│   ├── f2_ridge_e1_region_transfer.csv F2: East<->West with ridge regression
│   ├── f2_ridge_pairs.csv             F2: the 132-pair matrix with ridge regression
│   ├── f3_variance_shares.csv         F3: between-state / between-county / location-surface variance shares
│   ├── f1_f3_summary.json             all F1-F3 statistics
│   ├── environment.txt
│   └── run.log
└── figures/                           make figures
    ├── fig1_preregistered.png
    └── fig2_exploratory.png
```

## `preregistered/e3_e4_pairs.csv`

One row per arm × ordered state pair. All metrics are means over five random-forest seeds.

| Column | Meaning | Units |
|---|---|---|
| `arm` | input arm: `AEF`, `AEF+clim`, `clim` (and `RS`, `RS-noclim` when available) | - |
| `src`, `tgt` | source (training) and target (test) state | USPS code |
| `R2` | transfer R² on the target, against the target mean | - |
| `R2_level` | R² after shifting predictions to the target's mean yield | - |
| `r2_corr` | squared Pearson correlation of predictions and yields | - |
| `rmse`, `bias` | root-mean-square error; mean prediction minus mean yield | bu/ac |
| `R2_oracle` | R² of a model trained inside the target (county-held-out CV) | - |
| `loss_total`, `loss_level`, `loss_rank` | `R2_oracle - R2`, `R2_level - R2`, `R2_oracle - R2_level` | R² points |
| `full_auc`, `full_cover`, `full_share2s` | domain-classifier AUC; one-sided coverage; two-sided [0.05, 0.95] share (all features) | - |
| `sub_auc`, `sub_cover`, `sub_share2s` | the same in the source-fitted PLS(3) yield subspace (`sub_cover` is the primary measure) | - |
| `knn_cover` | nearest-neighbour coverage of the target by the source | share |
| `yield_range_cover` | share of target yields inside the source's 5th–95th percentile (uses target yields; P5 only) | share |
| `gdd_dist` | absolute difference in mean May–Sep growing degree days | °C-days |
| `geo_km` | distance between state centroids | km |
| `n_src`, `n_tgt`, `log_n_src` | county-years in source and target; log source size | - |

## Exploratory files

- **`f1_pairs.csv`**
  - `cover`: identical to `sub_cover`.
  - `loss_rank`: from seed-averaged predictions; rank correlation with the pre-registered value
    above 0.9999.
  - `loss_rank_covered`, `loss_rank_uncovered`: mean excess error on covered and on uncovered
    county-years.
  - `loss_on_covered`: share of the pair's ranking loss that falls on covered county-years.
- **`f1_county_years.csv.gz`**
  - `excess`: excess squared error of the level-matched transfer over the target's own model,
    in units of the target's yield variance.
  - `resid`: level-matched transfer residual, y − p.
  - `resid_oracle`: the target model's residual.
  - `covered`, `irr_share`: coverage flag; share of county land irrigated (USGS 2015).
- **`f1_irrigation_within_pairs.csv`**: within-pair Spearman correlations of `irr_share` with
  - `resid`, the raw residual;
  - `resid - resid_oracle`, the transfer-specific error (labelled `relative`);
  - `resid_oracle`, the target model's own residual.
- **`f2_ridge_pairs.csv`**: the same columns as `e3_e4_pairs.csv` for ridge regression, plus
  `below_src_min`, the share of predictions below the source's lowest yield.
- **`f3_variance_shares.csv`**: for each AEF band, climate variable and yield, the share of
  variance that is
  - between states;
  - between counties (time-invariant);
  - on a cubic surface in longitude and latitude.

Definitions are in [docs/methods.md](../docs/methods.md).
