# Methods

This document gives the full specification behind the results in the README. Predictions, the
primary measure and the tests were fixed in [PREREGISTRATION.md](../PREREGISTRATION.md) before any
pair-level transfer loss was computed (commit `474b4cd`).

## 1. Data

- **Source.** County-level corn table shared with Ma, Shen, Swatantran & Lobell (2026)
  ([repository](https://github.com/yuchima8/Harvest_AlphaEarth), file
  `code/US_FM_Corn_Data_shared.csv`, upstream commit `2ec9163`, SHA-256 `310f4982…7e33`).
  `scripts/get_data.sh` fetches and verifies it. It is not redistributed.
- **Units.** One row per county-year: 6,325 county-years, 12 states, 2017–2024, between 302
  (ND) and 712 (IA) per state.
- **Outcome.** NASS survey corn grain yield (bu/ac).
- **Features used.**
  - AlphaEarth Foundations annual embeddings `A00`–`A63`, averaged over CDL corn pixels in each
    county by the source authors.
  - Monthly growing degree days `GDD_5`–`GDD_9` and precipitation `ppt_5`–`ppt_9` (May–Sep,
    ERA5-Land), as used in the source study's RS baseline.
  - County polygons, used only to compute centroids for the distance predictor.

## 2. Input arms

| Arm | Features | Purpose |
|---|---|---|
| `AEF` | 64 embedding bands | the source study's AEF model |
| `AEF+clim` | 64 + 10 climate | tests whether the RS baseline's climate inputs explain its transfer advantage |
| `clim` | 10 climate | a reanalysis-only reference |
| `RS` | 80 Landsat harmonic/phenology features + 10 climate | the source study's RS baseline (not run, §8) |
| `RS-noclim` | 80 Landsat features | the input-provenance contrast with AEF (not run, §8) |

## 3. Model and transfer protocol

- **Model.** Random forest regression with 200 trees (as in the source study), scikit-learn
  defaults otherwise. Five seeds (0–4); metrics are averaged over seeds.
- **E1, reproduction.** Train on all county-years of one ecoregion and test on the other: East =
  IL, IN, MI, OH, WI; West = IA, KS, MN, MO, ND, NE, SD.
- **E3, pair matrix.** For each ordered pair (s, t) of the 12 states (132 pairs), train on all
  county-years of s and test on all county-years of t.
- **Oracle.** For each target t, a random forest trained inside t, evaluated by 5-fold
  cross-validation that holds out whole counties (`GroupKFold` by FIPS). Like the transferred
  model, it predicts unseen places in seen years.

## 4. Outcomes: decomposing the transfer loss

For target t with observed yields y and transferred predictions p:

- $R^2 = 1 - \sum (y - p)^2 / \sum (y - \bar y)^2$, the source study's Eq. 9.
- $R^2_{\text{level}}$ is $R^2$ after shifting the predictions so that their mean equals the
  target's mean yield: $p' = p - \bar p + \bar y$. The target mean is the kind of aggregate NASS
  publishes, so this is the part of the loss that an aggregate-statistics correction, in the
  spirit of Kluger, Wang & Lobell (2021), could remove.

The loss then splits into three quantities:

| Quantity | Definition | Meaning |
|---|---|---|
| `loss_total` | $R^2_{\text{oracle}} - R^2$ | the whole loss |
| `loss_level` | $R^2_{\text{level}} - R^2$ | removable with the target's mean yield |
| `loss_rank` | $R^2_{\text{oracle}} - R^2_{\text{level}}$ | not removable that way; **primary outcome** |

## 5. Overlap measures (no target yields)

Every propensity below is out-of-fold. It comes from L2-penalized logistic regression
(C = 1, standardized inputs, balanced class weights) under 5-fold cross-fitting that holds out whole
counties. The model estimates P(source | x) over the pooled source and target county-years.

| Measure | Definition |
|---|---|
| `full_auc` | domain-classifier AUC on all features of the arm |
| `full_cover` | share of target county-years with P(source \| x) ≥ 0.05 (one-sided) |
| `full_share2s` | share with P(source \| x) in [0.05, 0.95], the two-sided trimming rule of Kluger, Di Tommaso & Lobell (2025) |
| **`sub_cover`** (primary) | as `full_cover`, but x is replaced by z, the first K = 3 partial-least-squares components fitted on the **source** state's standardized features against yield |
| `sub_auc` | domain-classifier AUC on z |
| `knn_cover` | share of target points whose nearest source point (standardized on the source) is within the 95th percentile of the source's own nearest-neighbour distances, with neighbours taken from a different county |

**Why one-sided coverage.** For transfer, only target points that the source never covers are a
positivity problem. Target points that look like the source are not. This is the transfer analogue
of overlap for the effect on the treated, not the symmetric overlap of the average effect.

**Why a yield-relevant subspace.** Overlap in a high-dimensional covariate space fails almost
surely (D'Amour et al., 2021). What matters for prediction is overlap in the directions the outcome
depends on, the idea behind the prognostic score (Hansen, 2008). The feasibility probe showed
full-space AEF overlap to be saturated. That is why `sub_cover` was chosen as the primary measure,
before any transfer loss was computed.

**Explanatory only.** `yield_range_cover` is the share of target yields inside the source's
5th–95th percentile range. It uses target yields and serves only to test P5.

Domain-classifier measures of distribution shift have a long history in domain adaptation (e.g.
Ben-David et al., 2010). The contribution here is their use as positivity diagnostics, tested
against realized transfer loss under a pre-registered design.

## 6. Rival predictors (label-free)

- `gdd_dist`: absolute difference in mean May–Sep GDD between the two states. Two Shifts reports
  that random-forest transfer between US states "was high when the two states had similar growing
  degree days but performed poorly when the growing degree days differed by a lot" (Kluger, Wang &
  Lobell, 2021, describing Wang, Azzari & Lobell, 2019).
- `geo_km`: great-circle distance between state centroids, each the mean of county centroids.
- `log_n_src`: log of the source sample size.

## 7. Inference

The 132 ordered pairs share states, since each state appears in 22 pairs, so the pairs are not
independent.

- **Correlations.** Spearman ρ, with p-values from QAP permutation (Krackhardt, 1988). One random
  relabelling of the 12 states is applied to the rows and columns of the predictor matrix at once,
  with 9,999 permutations for the pre-registered tests and 1,999 for secondary correlations.
- **Partial effect (P3).** Rank regression of `loss_rank` on `sub_cover`, controlling for
  `gdd_dist`, `geo_km`, and source and target fixed effects. Inference uses double-semi-partialling
  QAP (Dekker, Krackhardt & Snijders, 2007).
- **Asymmetry (P5).** For each of the 66 unordered pairs, the test asks whether the direction with
  lower `yield_range_cover` has the larger `loss_level`. A one-sided sign test is reported, with the
  caveat that pairs share states.
- **Validation.** The unit tests check the procedures on synthetic data with known answers: a
  planted relation is detected, and the false-positive rate under the null is near 5%.

## 8. Landsat arm (implemented, not run)

`gee/export_landsat_harmonics.py` rebuilds the source study's RS features on Earth Engine:

- **Inputs.** Landsat 7/8/9 Collection 2 Level-2 surface reflectance, with a QA mask (cloud,
  dilated cloud, shadow, snow, water) and a saturation mask, and a CDL corn mask.
- **Bands.** Six bands plus NDVI and GCVI.
- **Fit.** A second-order harmonic regression. `aeoverlap.rs_features` then derives peak, b30,
  a30, b30_int and a30_int from each county's fitted curve, for 10 × 8 = 80 features.

Documented deviations from the source study:

1. Harmonics are fitted per pixel and then averaged over corn pixels. The source study averages
   per date and then fits. OLS coefficients are linear in y, so the two agree exactly when pixels
   share observation dates.
2. The season window is Apr 1 – Oct 31, because the source study does not state one.
3. GCVI = NIR/Green − 1. The constant offset does not affect a random forest.

A lower-compute variant (`--lite`: GCVI and NDVI at 120 m, Amendment 1) passed a one-county check.
The fitted GCVI peaked on day 210, with about 20 clear observations per pixel. A full-year,
12-state export, however, used about 25 EECU-hours per minute and more than 136 EECU-hours within
6 minutes without measurable progress. At that rate the eight years do not fit the 150 EECU-hours
per month of Earth Engine's free noncommercial tier, so P1 and P6 remain untested.

## 9. Software and runtime

- **Environment.** Python 3.9.6 with the pinned versions in `requirements.txt`. The full pipeline
  takes 15–25 minutes on an 11-core laptop; see `results/environment.txt`.
- **Determinism.** All randomness is seeded: random-forest seeds 0–4, fixed permutation seeds and
  deterministic `GroupKFold`.

## References

- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., Vaughan, J.W. (2010). A theory
  of learning from different domains. *Machine Learning*, 79, 151–175.
- Crump, R.K., Hotz, V.J., Imbens, G.W., Mitnik, O.A. (2009). Dealing with limited overlap in
  estimation of average treatment effects. *Biometrika*, 96(1), 187–199.
- D'Amour, A., Ding, P., Feller, A., Lei, L., Sekhon, J. (2021). Overlap in observational studies
  with high-dimensional covariates. *Journal of Econometrics*, 221(2), 644–654.
- Dekker, D., Krackhardt, D., Snijders, T.A.B. (2007). Sensitivity of MRQAP tests to collinearity
  and autocorrelation conditions. *Psychometrika*, 72(4), 563–581.
- Hansen, B.B. (2008). The prognostic analogue of the propensity score. *Biometrika*, 95(2),
  481–488.
- Kluger, D.M., Di Tommaso, S., Lobell, D.B. (2025). Precrop payoffs: causal machine learning
  reveals large but variable yield benefits of crop rotation in major breadbaskets.
  *Environmental Research Letters*.
- Kluger, D.M., Wang, S., Lobell, D.B. (2021). Two shifts for crop mapping: Leveraging aggregate
  crop statistics to improve satellite-based maps in new regions. *Remote Sensing of
  Environment*, 262, 112488. https://doi.org/10.1016/j.rse.2021.112488
- Krackhardt, D. (1988). Predicting with networks: Nonparametric multiple regression analysis of
  dyadic data. *Social Networks*, 10(4), 359–381.
- Li, F., Morgan, K.L., Zaslavsky, A.M. (2018). Balancing covariates via propensity score
  weighting. *Journal of the American Statistical Association*, 113(521), 390–400.
- Ma, Y., Shen, Y., Swatantran, A., Lobell, D.B. (2026). Harvesting AlphaEarth: Benchmarking the
  geospatial foundation model for agricultural downstream tasks. *International Journal of Applied
  Earth Observation and Geoinformation*, 105258. https://doi.org/10.1016/j.jag.2026.105258
- Wang, S., Azzari, G., Lobell, D.B. (2019). Crop type mapping without field-level labels: Random
  forest transfer and unsupervised clustering techniques. *Remote Sensing of Environment*, 222,
  303–317. https://doi.org/10.1016/j.rse.2018.12.026
