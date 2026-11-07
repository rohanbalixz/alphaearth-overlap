"""aeoverlap: does feature-space overlap predict when crop-yield models transfer across regions?

A pre-registered test on the county corn data of Ma, Shen, Swatantran & Lobell (2026),
comparing AlphaEarth embeddings, climate covariates and (optionally) Landsat harmonic features.

Pipeline entry points (see the Makefile):
    python -m aeoverlap.pipeline     # E1 reproduction, E3 transfer matrix, E4 overlap
    python -m aeoverlap.evaluate     # pre-registered tests (E5, E6)
    python -m aeoverlap.figures      # results/figure.png
    python -m aeoverlap.rs_features  # Landsat export -> 80 (or 20) RS features
"""

__version__ = "0.1.0"
