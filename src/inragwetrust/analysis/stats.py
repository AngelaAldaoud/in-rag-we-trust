"""Statistical tools for Block F (analysis plan sections 3-4; D52, D253, D260, D261)."""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from inragwetrust.evaluation.agreement import BOOTSTRAP, SEED, wilson

__all__ = ["BOOTSTRAP", "SEED", "wilson", "holm", "mcnemar_exact", "cluster_bootstrap",
           "fit_gee", "odds_ratio"]

MAX_ABS_COEF = 15.0


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values; a non-estimable test (NaN) counts with p = 1 and stays NaN (D261)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = multipletests(np.where(np.isnan(p), 1.0, p), method="holm")[1]
    return np.where(np.isnan(p), np.nan, adjusted)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts."""
    if b + c == 0:
        return 1.0
    return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)


def cluster_bootstrap(num, den, n_boot: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of sum(num)/sum(den), resampling clusters (questions) with replacement."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _problem(res, caught) -> str | None:
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return "no convergence"
    if not (np.all(np.isfinite(res.params)) and np.all(np.isfinite(res.bse))):
        return "non-finite estimates"
    if np.max(np.abs(res.params)) > MAX_ABS_COEF:
        return "degenerate estimates"
    return None


def fit_gee(formula: str, data, group: str = "question_id"):
    """Logistic GEE clustered by question, robust SE; exchangeable first, independence fallback (D253)."""
    last = ""
    for name, cov in (("exchangeable", sm.cov_struct.Exchangeable), ("independence", sm.cov_struct.Independence)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = smf.gee(formula, group, data=data, family=sm.families.Binomial(),
                              cov_struct=cov()).fit()
            except Exception as e:  # numerical failure counts as a failed fit
                last = f"{name}: {type(e).__name__}"
                continue
        problem = _problem(res, caught)
        if problem is None:
            return res, name, "" if name == "exchangeable" else f"fallback to independence (D253); {last}"
        last = f"{name}: {problem}"
    return None, None, f"not estimable (D253); {last}"


def odds_ratio(res, term: str) -> tuple[float, float, float, float]:
    """Odds ratio of one term with its robust 95% CI and p-value."""
    lo, hi = res.conf_int().loc[term]
    return (float(np.exp(res.params[term])), float(np.exp(lo)), float(np.exp(hi)),
            float(res.pvalues[term]))