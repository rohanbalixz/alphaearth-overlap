# Results files

Everything here is produced by `make results evaluate figure` (see the top-level README). The files
are derived metrics only; the source data are not included.

| File | Contents |
|---|---|
| `figure.png` | Summary figure (overlap distributions, overlap vs ranking loss, rival predictors) |
| `e1_region_transfer.csv` | E1: East<->West reproduction of Ma et al. (2026), Table 6, per arm |
| `pairs.csv` | E3/E4: one row per arm x ordered state pair (3 arms x 132 pairs); the table view of the figure |
| `tests.json` | E5/E6: every pre-registered test and the secondary correlations, with p-values |
| `summary.md` | Human-readable summary of `tests.json` |
| `environment.txt` | Python and package versions, hardware, source-data SHA-256, runtime |
| `pipeline.log` | Console log of the pipeline run |

## `pairs.csv` columns

| Column | Meaning | Units |
|---|---|---|
| `arm` | input arm: `AEF`, `AEF+clim`, `clim` (and `RS`, `RS-noclim` when available) | - |
| `src`, `tgt` | source (training) and target (test) state | USPS code |
| `R2` | transfer R2 on the target, against the target mean | - |
| `R2_level` | R2 after shifting predictions to the target's mean yield | - |
| `r2_corr` | squared Pearson correlation of predictions and yields | - |
| `rmse`, `bias` | root-mean-square error; mean prediction minus mean yield | bu/ac |
| `R2_oracle` | R2 of a model trained inside the target (county-held-out CV) | - |
| `loss_total`, `loss_level`, `loss_rank` | `R2_oracle - R2`, `R2_level - R2`, `R2_oracle - R2_level` | R2 points |
| `full_auc`, `full_cover`, `full_share2s` | domain-classifier AUC; one-sided coverage; two-sided [0.05, 0.95] share (all features) | - |
| `sub_auc`, `sub_cover`, `sub_share2s` | the same in the source-fitted PLS(3) yield subspace; `sub_cover` is the primary measure | - |
| `knn_cover` | nearest-neighbour coverage of the target by the source | share |
| `yield_range_cover` | share of target yields inside the source's 5th-95th percentile (uses target yields; P5 only) | share |
| `gdd_dist` | absolute difference in mean May-Sep growing degree days | degree C-days |
| `geo_km` | distance between state centroids | km |
| `n_src`, `n_tgt`, `log_n_src` | county-years in source and target; log source size | - |

All metrics are means over five random-forest seeds. Definitions are in
[docs/METHODS.md](../docs/METHODS.md).
