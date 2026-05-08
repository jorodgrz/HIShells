"""Smoke tests for ``hishells.catalog_bayes``.

Each test fits one model factory on a 50-hole synthetic subset and
asserts that NUTS converged: R-hat ≤ 1.05 and ESS ≥ 100 on the
*global* parameters of each model. We deliberately don't test the
*recovery* of the synthetic generative parameters (that would be
brittle for 50 observations); we test that the sampler runs cleanly
and the InferenceData has the groups the notebook driver expects.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")
pm = pytest.importorskip("pymc")

from hishells.catalog_bayes import (
    fit_axial_best,
    fit_chevalier,
    fit_count_rate,
    fit_truncated_diameter,
    fit_type_fractions,
    hash_dataframe,
)

# Keep PyMC's own progress bars and ArviZ's future-warnings out of the
# pytest log; they're irrelevant to whether the sampler converged.
_FAST_KW = {
    "draws": 500,
    "tune": 500,
    "chains": 2,
    "cores": 1,
    "progressbar": False,
}


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """50 synthetic holes across 4 galaxies + matching galaxies table.

    The structure mirrors what :mod:`hishells.catalog` produces from
    the on-disk B11 tables (column names, dtypes, value ranges) so the
    Bayes factories can be exercised without touching ``Data/``.
    """

    rng = np.random.default_rng(0)
    galaxies = ["NGC_A", "NGC_B", "NGC_C", "NGC_D"]
    n_per = [13, 12, 13, 12]
    rows = []
    for g, n_g, mu in zip(galaxies, n_per, [6.0, 6.4, 5.7, 6.1]):
        # Mostly type-1 with a sprinkle of type-2 and type-3, mirroring
        # the B11 dataset's heavy type-1 majority.
        types = rng.choice([1, 2, 3], size=n_g, p=[0.6, 0.25, 0.15])
        d_pc = np.exp(rng.normal(mu, 0.4, size=n_g))
        v_exp = np.where(types == 1, np.nan, rng.uniform(5.0, 20.0, size=n_g))
        # Chevalier-ish: log_E = -7 + 1.27 log d + 0.77 log v + N(0, 0.3)
        with np.errstate(invalid="ignore"):
            log_E = -7.0 + 1.27 * np.log(d_pc) + 0.77 * np.log(np.where(np.isnan(v_exp), 1.0, v_exp))
        log_E += rng.normal(0.0, 0.3, size=n_g)
        # Axial ratio: type-3 a touch more circular than type-1/2.
        ar_mean = np.where(types == 3, 0.78, np.where(types == 2, 0.74, 0.72))
        axial = np.clip(rng.normal(ar_mean, 0.12), 0.05, 0.99)
        for j in range(n_g):
            rows.append(
                {
                    "galaxy_id": g,
                    "hole_idx": j + 1,
                    "hole_type": int(types[j]),
                    "diameter_pc": float(d_pc[j]),
                    "vexp_kms": float(v_exp[j]) if not np.isnan(v_exp[j]) else np.nan,
                    "log_E_J43": float(log_E[j]),
                    "axial_ratio": float(axial[j]),
                }
            )
    holes = pd.DataFrame(rows)

    galaxies_df = pd.DataFrame(
        {
            "galaxy_id": galaxies,
            "distance_mpc": [4.0, 8.0, 6.0, 10.0],
            "log_sfr": [-1.2, -0.9, -1.4, -0.6],
            "MHI_1e8Msun": [3.0, 6.5, 1.8, 9.0],
            "resolution_pc": [120.0, 200.0, 150.0, 250.0],
        }
    )
    return holes, galaxies_df


def _max_rhat(idata, var_names):
    summ = az.summary(idata, var_names=var_names)
    return float(summ["r_hat"].max())


def _min_ess(idata, var_names):
    summ = az.summary(idata, var_names=var_names)
    return float(summ["ess_bulk"].min())


def test_fit_type_fractions(synthetic):
    holes, _ = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_type_fractions(holes, draws=500, seed=0, sample_kwargs=_FAST_KW)
    assert "posterior" in idata.groups()
    assert _max_rhat(idata, ["alpha"]) <= 1.05
    assert _min_ess(idata, ["alpha"]) >= 100


def test_fit_truncated_diameter(synthetic):
    holes, galaxies = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_truncated_diameter(
            holes, galaxies, draws=500, seed=0, sample_kwargs=_FAST_KW
        )
    assert "posterior" in idata.groups()
    assert _max_rhat(idata, ["mu_pop", "tau_mu", "tau_sigma"]) <= 1.05
    assert _min_ess(idata, ["mu_pop", "tau_mu", "tau_sigma"]) >= 100


def test_fit_chevalier(synthetic):
    holes, _ = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_chevalier(
            holes, types=(2, 3), draws=500, seed=0, sample_kwargs=_FAST_KW
        )
    assert {"posterior", "log_likelihood", "posterior_predictive"} <= set(idata.groups())
    assert _max_rhat(idata, ["alpha", "beta", "gamma", "sigma"]) <= 1.05
    assert _min_ess(idata, ["alpha", "beta", "gamma", "sigma"]) >= 100


def test_fit_count_rate(synthetic):
    holes, galaxies = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_count_rate(
            holes, galaxies, draws=500, seed=0, sample_kwargs=_FAST_KW
        )
    assert {"posterior", "log_likelihood"} <= set(idata.groups())
    assert _max_rhat(idata, ["beta_0", "beta_SFR", "beta_MHI", "beta_D", "alpha_nb"]) <= 1.05
    assert _min_ess(idata, ["beta_0", "beta_SFR", "beta_MHI", "beta_D", "alpha_nb"]) >= 100


def test_fit_axial_best(synthetic):
    holes, _ = synthetic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = fit_axial_best(holes, draws=500, seed=0, sample_kwargs=_FAST_KW)
    assert "posterior" in idata.groups()
    assert _max_rhat(idata, ["mu_t", "sigma_t", "diff_3_2", "diff_3_1"]) <= 1.05
    assert _min_ess(idata, ["mu_t", "sigma_t", "diff_3_2", "diff_3_1"]) >= 100


def test_hash_dataframe_stable(synthetic):
    """The cache key must not depend on Python object identity."""

    holes, galaxies = synthetic
    h1 = hash_dataframe(holes[["galaxy_id", "hole_type"]], 500, 0)
    h2 = hash_dataframe(holes[["galaxy_id", "hole_type"]], 500, 0)
    h3 = hash_dataframe(holes[["galaxy_id", "hole_type"]], 500, 1)
    assert h1 == h2, "hash must be deterministic across invocations"
    assert h1 != h3, "hash must change when inputs change"
