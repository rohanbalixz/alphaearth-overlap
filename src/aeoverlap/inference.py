"""Inference for 132 ordered state pairs.

Pairs share states (each state appears in 22 pairs), so ordinary p-values are anti-conservative.
We use QAP permutation: relabel the 12 states with one random permutation applied to the rows and
columns of the predictor matrix at once (Krackhardt 1988), and double-semi-partialling (DSP;
Dekker, Krackhardt & Snijders 2007) for partial effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import STATES

IDX = {s: i for i, s in enumerate(STATES)}


def _mat(pairs, col):
    M = np.full((len(STATES), len(STATES)), np.nan)
    for s, t, v in zip(pairs.src, pairs.tgt, pairs[col]):
        M[IDX[s], IDX[t]] = v
    return M


def _vec(M, pairs):
    return np.array([M[IDX[s], IDX[t]] for s, t in zip(pairs.src, pairs.tgt)])


def _rankmat(pairs, col):
    r = pairs.assign(_r=rankdata(pairs[col]))
    return _mat(r, "_r")


def qap_spearman(pairs, x, y, n_perm=9999, seed=0) -> tuple[float, float]:
    """Spearman rho between pair-level x and y with a two-sided QAP permutation p-value."""
    rng = np.random.default_rng(seed)
    Rx, ry = _rankmat(pairs, x), rankdata(pairs[y])
    rho = np.corrcoef(_vec(Rx, pairs), ry)[0, 1]
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(len(STATES))
        null[b] = np.corrcoef(_vec(Rx[np.ix_(p, p)], pairs), ry)[0, 1]
    pval = (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + n_perm)
    return float(rho), float(pval)


def _design(pairs, cols, fe):
    Z = [np.ones(len(pairs))] + [rankdata(pairs[c]) for c in cols]
    if fe:
        for side in ("src", "tgt"):
            D = pd.get_dummies(pairs[side], drop_first=True, dtype=float).values
            Z += list(D.T)
    return np.column_stack(Z)


def _tstat(yv, xv, Z):
    X = np.column_stack([xv, Z])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = len(yv) - np.linalg.matrix_rank(X)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    return beta[0], beta[0] / np.sqrt(cov[0, 0])


def dsp_partial(pairs, x, y, controls, fe=True, n_perm=9999, seed=0) -> tuple[float, float, float]:
    """Partial effect of rank(x) on rank(y) given rank(controls), plus source and target fixed
    effects. Returns the standardized coefficient (z-scored ranks), its t, and the DSP-QAP p-value."""
    rng = np.random.default_rng(seed)
    Z = _design(pairs, controls, fe)
    yv = rankdata(pairs[y])
    yv = (yv - yv.mean()) / yv.std()
    xv = rankdata(pairs[x])
    xv = (xv - xv.mean()) / xv.std()
    bx, *_ = np.linalg.lstsq(Z, xv, rcond=None)
    xres = xv - Z @ bx                                   # semi-partial x on the controls
    coef, t = _tstat(yv, xres, Z)
    Xr = _mat(pairs.assign(_x=xres), "_x")
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(len(STATES))
        null[b] = _tstat(yv, _vec(Xr[np.ix_(p, p)], pairs), Z)[1]
    pval = (1 + np.sum(np.abs(null) >= abs(t))) / (1 + n_perm)
    return float(coef), float(t), float(pval)
