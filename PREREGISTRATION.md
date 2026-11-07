# Pre-registration

Written and committed 2026-11-07, **before** any state-pair transfer loss was computed. The git
history is the timestamp. Whatever the results, they are reported against this file.

## Question

Across the 12 Corn Belt states in Ma, Shen, Swatantran & Lobell (2026), can we predict, without
any target-state yields, how much a county corn-yield model loses when it is moved from one state
to another? And does lack of overlap explain why AlphaEarth (AEF) embeddings transfer worse than
Landsat (RS) features?

## Already seen (feasibility probe, same day)

Disclosed so the reader knows what informed the design:

1. East<->West reproduction with AEF + RF-200 (3 seeds): R2 0.06 East->West, 0.50 West->East
   (paper Table 6: 0.02, 0.51). Adding the paper's monthly climate inputs to AEF: 0.05 / 0.51.
   Climate alone: -0.08 / -0.38.
2. Shifting predictions to the target's mean yield moved AEF East->West from 0.06 to 0.38 and
   West->East from 0.50 to 0.51.
3. **Full-space AEF overlap is saturated.** Domain AUC >= 0.99 for 65 of 66 unordered state
   pairs (median 1.000, also at C = 0.01). The originally planned statistic (share of target
   inside [0.05, 0.95]) is therefore near zero for almost every pair and cannot rank pairs.
4. Overlap in a 3-component PLS subspace fitted on the source varies across pairs (AEF AUC
   0.59-1.00; median one-sided coverage 0.60). **This is why the primary measure below was
   chosen.** Its relation to transfer loss was not looked at.

## Design (fixed)

- Data: the 6,325 county-years shared in the paper's repo (12 states, 2017-2024).
- Arms: `AEF` (A00-A63), `AEF+clim` (+ GDD and PPT, May-Sep, as in the paper's RS baseline),
  `clim` (those 10 only). When the Landsat export is available: `RS` (80 harmonic features +
  climate, the paper's baseline) and `RS-noclim` (80 only). All arms use the same county-years.
- Model: RandomForestRegressor, 200 trees, 5 seeds, metrics averaged over seeds.
- Transfer: train on all years of state s, test on all years of state t, for all 132 ordered
  pairs.
- Oracle: RF inside t, 5-fold CV holding out whole counties.
- Outcomes:
  - `loss_total = R2_oracle - R2`
  - `loss_level = R2_level - R2`
  - `loss_rank = R2_oracle - R2_level` (**primary**)
  - `R2_level` shifts predictions so their mean equals the target's mean yield.
- **Primary overlap measure: `sub_cover`.** The share of target county-years with cross-fitted
  (whole-county folds), class-balanced propensity P(source | z) >= 0.05, where z is the 3
  PLS components fitted on the source state's standardized features against yield.
- Secondary measures: full-space domain AUC, full-space one-sided coverage, the two-sided share
  in [0.05, 0.95], kNN coverage, subspace AUC.
- Rival predictors (label-free): |difference in mean May-Sep GDD| (cf. Wang, Azzari & Lobell
  2019), distance between state centroids, log source sample size.

## Predictions

- **P2 (primary):** In the AEF arm, Spearman(`sub_cover`, `loss_rank`) < 0 with QAP permutation
  p < 0.05 (9,999 permutations of state labels).
- **P3:** In the AEF arm, `sub_cover` keeps a negative partial association with `loss_rank` after
  controlling for GDD distance and geographic distance, with source and target fixed effects
  (DSP-QAP p < 0.05). Also |rho(`sub_cover`)| > |rho(GDD distance)|.
- **P4:** |rho(`sub_cover`, `loss_rank`)| > |rho(`sub_cover`, `loss_level`)|. Overlap should
  speak to the part of the loss an aggregate correction cannot fix.
- **P5 (asymmetry):** Over the 66 unordered pairs, in the AEF arm, the direction whose target
  has lower yield-range coverage (share of target yields inside the source's 5th-95th percentile)
  has the larger `loss_level` in at least 2/3 of pairs. This uses target yields, so it is an
  explanation, not a diagnostic.
- **P1 (pending the RS export):** Full-space and subspace overlap are lower for AEF than for
  RS-noclim in at least 2/3 of the 132 pairs.
- **P6 (pending the RS export):** Across pairs, (AEF minus RS-noclim) in `loss_rank` correlates
  negatively with (AEF minus RS-noclim) in `sub_cover`.

The same tests are also run for `AEF+clim` and `clim` as secondary results, without
multiplicity claims.

## What would count against

- P2 fails if rho >= 0 or p >= 0.05. That is then the headline, stated plainly.
- No measure, threshold, K or loss definition will be changed after seeing the results. Any
  extra analysis will be labelled exploratory.

## Amendment 1 (2026-11-07, before any Landsat data exists)

Reason: compute limits on Earth Engine's free noncommercial tier, and the Nov 11 deadline. If
the full 80-feature Landsat export cannot finish in time, a lower-compute Landsat arm may be
used instead.

- `--lite` export: GCVI and NDVI only, the same 10 features each (20 in total), county means at
  120 m instead of 30 m. The QA and CDL corn masks, season window, model, overlap measures and
  tests are unchanged. GCVI and NDVI were chosen because the paper's feature-importance analysis
  (Sec. 5.2.2, Fig. 9) finds GCVI more informative than NDVI and the raw bands.
- Labelling: the lite arm is reported as "Landsat GCVI+NDVI (lite)", never as the paper's RS
  baseline.
- Gate: both East->West and West->East R2 must be within 0.15 of the paper's RS values
  (>= 0.51 and >= 0.45). If the gate fails, P1 and P6 are reported as not tested.
- P1 and P6 apply unchanged, with `RS-noclim` = the lite features without climate.
