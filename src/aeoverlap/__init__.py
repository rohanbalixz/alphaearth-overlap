"""aeoverlap: does feature-space overlap predict when crop-yield models transfer across regions?

A pre-registered test on the county corn data of Ma, Shen, Swatantran & Lobell (2026),
comparing AlphaEarth embeddings, climate covariates and (optionally) Landsat harmonic features.

Entry points (see the Makefile). e1-e6 are pre-registered, f1-f3 exploratory:
    python -m aeoverlap.run_preregistered       # E1 reproduction, E3 transfer matrix, E4 overlap
    python -m aeoverlap.evaluate_preregistered  # E5, E6 pre-registered tests
    python -m aeoverlap.run_exploratory         # F1 concept shift + irrigation, F2 ridge, F3 geography
    python -m aeoverlap.figures                 # fig1_preregistered.png, fig2_exploratory.png
    python -m aeoverlap.rs_features             # Landsat export -> 80 (or 20) RS features
"""

__version__ = "0.1.0"
