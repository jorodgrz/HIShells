"""Bayesian model factories over the B11 catalog (plan §0–§12 deep-dive).

Five PyMC models that consume the parsed ``B11Catalog`` (see
:mod:`hishells.catalog`) and return ``arviz.InferenceData``. Each model
ties back to a specific decision in ``plan.md``:

* :func:`fit_type_fractions` – hierarchical Dirichlet–Multinomial of the
  per-galaxy hole-type fractions. Quantifies whether the v1 default
  type-{2,3} filter (plan §2.4) starves any LOGO fold of positives.
* :func:`fit_truncated_diameter` – beam-truncated hierarchical
  log-normal of ``diameter_pc``. Recovers the *untruncated* per-galaxy
  size law and the cross-galaxy heterogeneity ``tau_mu`` that bounds
  the LOGO generalisation gap (plan §6, §12).
* :func:`fit_chevalier` – Bayesian linear regression
  ``log_E_J43 ~ alpha + beta · log(diameter_pc) + gamma · log(vexp_kms)``
  with optional partial-pooling intercept by galaxy. Posterior
  predictive check on ``log_E_J43`` is the canonical sanity test for
  ``hishells.catalog`` parsing (plan §11 step 6).
* :func:`fit_count_rate` – Negative-Binomial GLM of per-galaxy hole
  counts on ``log_sfr``, ``log_MHI``, ``log(distance)``. ``log_lik`` is
  written to the trace so callers can run ``arviz.loo`` to flag
  Pareto-k outlier galaxies (plan §6 anomalous-fold callouts).
* :func:`fit_axial_best` – BEST-style Student-t comparison of
  ``axial_ratio`` across hole types. Verifies the §2.4 corollary that
  type-3 (textbook expanding) shells are more circular than types 1/2.

Every fit returns an ``arviz.InferenceData`` containing ``posterior``,
``observed_data``, ``log_likelihood`` (where supported), and where
relevant ``posterior_predictive``. Cells in
``notebooks/01_explore_catalog.ipynb`` use :func:`cached_fit` to
checkpoint these traces under ``results/bayes/`` so re-running the
notebook is instant when the input data hash is unchanged.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # PyMC + ArviZ live in environment.yml
    import arviz as az
    import pymc as pm
except Exception as exc:  # pragma: no cover - guarded so non-Bayes code paths stay importable
    az = None  # type: ignore[assignment]
    pm = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Shared sampling defaults
# ---------------------------------------------------------------------------


# NUTS defaults shared across the five fits. ``cores=2`` keeps PyMC's
# default 4 chains but pins them to 2 OS processes, which avoids the
# pickling overhead that dominates for these small models. Override via
# ``sample_kwargs`` if you need to retune for a specific notebook run.
_DEFAULT_SAMPLE_KWARGS: dict = dict(
    draws=2000,
    tune=1000,
    chains=4,
    cores=2,
    target_accept=0.9,
    progressbar=False,
    return_inferencedata=True,
)


def _require_pymc() -> None:
    """Raise a friendly error if PyMC import failed at module load."""

    if pm is None or az is None:
        raise ImportError(
            "PyMC / ArviZ are required for hishells.catalog_bayes. "
            "Install them via environment.yml (conda env update -f "
            "HIShells/environment.yml) or `pip install pymc arviz`."
        ) from _IMPORT_ERROR


def _merged_sample_kwargs(seed: int, draws: int, overrides: dict | None) -> dict:
    """Combine the NUTS defaults with caller overrides."""

    out = dict(_DEFAULT_SAMPLE_KWARGS)
    out["draws"] = draws
    out["random_seed"] = seed
    if overrides:
        out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# Cache layer (used by 01_explore_catalog.ipynb so cells skip resampling
# when the input data has not changed)
# ---------------------------------------------------------------------------


def hash_dataframe(*frames_and_seeds: object) -> str:
    """Stable hash of (DataFrames, scalars) tuple for the cache key.

    DataFrames are reduced to ``(columns, dtypes, raw bytes)`` per
    column so two pandas instances with identical contents produce the
    same hash even when they differ in row index or block layout.
    Object-dtype columns (e.g. ``galaxy_id`` strings) are hashed via
    their UTF-8 representation; ``arr.tobytes()`` on an object array
    serialises Python pointers rather than the string contents, which
    is non-deterministic across invocations.
    """

    def _is_str_like(dtype) -> bool:
        # Catches both legacy ``object`` columns of Python strings and
        # the modern ``pd.StringDtype`` introduced in pandas 1.0+. Both
        # have ``dtype.kind == 'O'``; numeric/bool dtypes do not.
        return dtype.kind == "O" or pd.api.types.is_string_dtype(dtype)

    h = hashlib.sha1()
    for obj in frames_and_seeds:
        if isinstance(obj, pd.DataFrame):
            h.update(",".join(map(str, obj.columns)).encode())
            h.update(",".join(str(d) for d in obj.dtypes).encode())
            for col in obj.columns:
                series = obj[col]
                if _is_str_like(series.dtype):
                    h.update(("\x00".join(map(str, series.tolist()))).encode())
                else:
                    h.update(np.ascontiguousarray(series.to_numpy()).tobytes())
        elif isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                h.update(("\x00".join(map(str, obj.tolist()))).encode())
            else:
                h.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, (tuple, list)):
            h.update(("\x00".join(map(str, obj))).encode())
        else:
            h.update(repr(obj).encode())
    return h.hexdigest()[:16]


def cached_fit(
    cache_dir: str | Path,
    name: str,
    key: str,
    fit_fn: Callable[[], "az.InferenceData"],
    *,
    refresh: bool = False,
) -> "az.InferenceData":
    """Run ``fit_fn`` with a NetCDF round-trip cache.

    Parameters
    ----------
    cache_dir
        Directory under which the cached trace lives.
    name
        Stable name of the model (e.g. ``"type_fractions"``).
    key
        Hash of the inputs; computed by :func:`hash_dataframe`. The cached
        file is named ``{name}-{key}.nc``.
    fit_fn
        Zero-argument callable returning an ``arviz.InferenceData``.
    refresh
        If ``True``, ignore any existing cache file and re-fit.

    Returns
    -------
    arviz.InferenceData
    """

    _require_pymc()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}-{key}.nc"
    if path.exists() and not refresh:
        try:
            return az.from_netcdf(path)
        except (OSError, KeyError, ValueError) as exc:
            # Cache files in iCloud-backed workspaces occasionally end up
            # truncated; fall back to a fresh fit rather than failing the
            # whole notebook run.
            warnings.warn(
                f"cached_fit: NetCDF at {path} unreadable ({exc!r}); refitting."
            )
            try:
                path.unlink()
            except OSError:
                pass
    idata = fit_fn()
    idata.to_netcdf(path)
    return idata


# ---------------------------------------------------------------------------
# Helpers used by multiple models
# ---------------------------------------------------------------------------


def _galaxy_index(holes: pd.DataFrame, galaxy_ids: Sequence[str]) -> np.ndarray:
    """Map each row of ``holes`` to the integer index of its galaxy."""

    lookup = {g: i for i, g in enumerate(galaxy_ids)}
    return holes["galaxy_id"].map(lookup).to_numpy(dtype=np.int64)


def _resolve_galaxies(
    holes: pd.DataFrame,
    galaxy_ids: Iterable[str] | None,
) -> list[str]:
    """Pick the galaxy_id ordering used as the ``galaxy`` model coord."""

    if galaxy_ids is None:
        return sorted(holes["galaxy_id"].unique().tolist())
    return list(galaxy_ids)


# ---------------------------------------------------------------------------
# Model 1: hierarchical Dirichlet-Multinomial of hole-type fractions
# ---------------------------------------------------------------------------


def fit_type_fractions(
    holes: pd.DataFrame,
    *,
    galaxy_ids: Iterable[str] | None = None,
    draws: int = 2000,
    seed: int = 0,
    sample_kwargs: dict | None = None,
) -> "az.InferenceData":
    """Hierarchical Dirichlet-Multinomial of per-galaxy hole-type fractions.

    Model
    -----
    For each galaxy ``g`` and hole type ``t`` in ``{1, 2, 3}``:

    * ``alpha_t ~ HalfNormal(2)`` (population concentration; learned)
    * ``pi_g ~ Dirichlet(alpha)`` (per-galaxy fractions)
    * ``counts_g ~ Multinomial(N_g, pi_g)``

    The deterministic ``pi_23 = pi_g[2] + pi_g[3]`` is the per-galaxy
    fraction of B11 holes that the v1 default §2.4 filter retains.

    Returns
    -------
    arviz.InferenceData
        ``posterior`` includes ``alpha``, ``pi`` (galaxy × type), and
        ``pi_23`` (per-galaxy type-{2,3} fraction).
    """

    _require_pymc()
    galaxies = _resolve_galaxies(holes, galaxy_ids)
    types = (1, 2, 3)
    counts = (
        holes.groupby(["galaxy_id", "hole_type"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(galaxies, fill_value=0).reindex(columns=types, fill_value=0)
    count_matrix = counts.to_numpy(dtype=np.int64)
    n_per_galaxy = count_matrix.sum(axis=1)

    coords = {"galaxy": galaxies, "type": list(types)}
    with pm.Model(coords=coords):
        alpha = pm.HalfNormal("alpha", sigma=2.0, dims="type")
        pi = pm.Dirichlet("pi", a=alpha, dims=("galaxy", "type"))
        pm.Multinomial(
            "counts",
            n=n_per_galaxy,
            p=pi,
            observed=count_matrix,
            dims=("galaxy", "type"),
        )
        pm.Deterministic("pi_23", pi[:, 1] + pi[:, 2], dims="galaxy")
        idata = pm.sample(**_merged_sample_kwargs(seed, draws, sample_kwargs))
    return idata


# ---------------------------------------------------------------------------
# Model 2: beam-truncated hierarchical lognormal of diameter_pc
# ---------------------------------------------------------------------------


def fit_truncated_diameter(
    holes: pd.DataFrame,
    galaxies: pd.DataFrame,
    *,
    galaxy_ids: Iterable[str] | None = None,
    k: float = 0.5,
    draws: int = 2000,
    seed: int = 0,
    sample_kwargs: dict | None = None,
) -> "az.InferenceData":
    """Beam-truncated hierarchical log-normal of B11 hole diameters.

    Model
    -----
    For each galaxy ``g`` and hole observation ``i``, with the per-galaxy
    THINGS resolution ``R_g = resolution_pc[g]`` from B11 Table 2:

    * ``mu_pop ~ Normal(6, 1)``  (~ ln(400 pc))
    * ``tau_mu ~ HalfNormal(1)``
    * ``tau_sigma ~ HalfNormal(1)``
    * ``mu_g ~ Normal(mu_pop, tau_mu)`` (non-centered)
    * ``sigma_g ~ HalfNormal(tau_sigma)`` (non-centered)
    * ``log(diameter_{g,i}) ~ TruncatedNormal(mu_g, sigma_g, lower=ln(k · R_g))``

    The truncation correction is essential because B11 cannot resolve
    holes smaller than ~beam, so the *observed* lower tail is censored.

    The detection-floor multiplier ``k`` is exposed as a hyperparameter
    rather than a free parameter. The plan specified ``k ~ HalfNormal(1)``
    but data-driven inference of ``k`` proves degenerate: with a shared
    truncation across galaxies the posterior pins to the support
    boundary at ``min_g(d_min_g / R_g)``, which causes ~85% of NUTS
    transitions to diverge. Fixing ``k`` to the half-beam default (0.5)
    gives a clean truncation correction; sensitivity is checked by
    sweeping ``k ∈ {0.25, 0.5, 1.0}`` in the notebook.

    Returns
    -------
    arviz.InferenceData
        ``posterior`` includes ``mu_pop``, ``tau_mu``, ``tau_sigma``,
        ``mu_g`` (per-galaxy untruncated mean log-diameter), and
        ``sigma_g``. ``constant_data`` records the value of ``k`` and
        the per-galaxy ``log_floor``.
    """

    _require_pymc()
    galaxies_ord = _resolve_galaxies(holes, galaxy_ids)
    res_lookup = (
        galaxies.set_index("galaxy_id")["resolution_pc"]
        .reindex(galaxies_ord)
    )
    if res_lookup.isna().any():
        missing = res_lookup[res_lookup.isna()].index.tolist()
        raise ValueError(
            f"Missing resolution_pc for galaxies {missing!r}; "
            "load_galaxies(table2.dat) must include all rows."
        )

    sub = holes.dropna(subset=["diameter_pc"]).copy()
    sub = sub[sub["galaxy_id"].isin(galaxies_ord)].reset_index(drop=True)
    log_d = np.log(sub["diameter_pc"].to_numpy(dtype=np.float64))
    g_idx = _galaxy_index(sub, galaxies_ord)
    log_res = np.log(res_lookup.to_numpy(dtype=np.float64))

    if k <= 0:
        raise ValueError(f"k must be positive (got {k!r}); see docstring.")
    floor_per_galaxy = np.log(k) + log_res
    log_floor_obs = floor_per_galaxy[g_idx]

    coords = {"galaxy": galaxies_ord, "obs": np.arange(len(sub))}
    with pm.Model(coords=coords):
        mu_pop = pm.Normal("mu_pop", 6.0, 1.0)
        tau_mu = pm.HalfNormal("tau_mu", 1.0)
        tau_sigma = pm.HalfNormal("tau_sigma", 1.0)

        # Non-centered parameterisation avoids the funnel geometry that
        # makes the centered ``mu_g ~ Normal(mu_pop, tau_mu)`` form
        # diverge on small ``tau_mu``.
        mu_g_z = pm.Normal("mu_g_z", 0.0, 1.0, dims="galaxy")
        mu_g = pm.Deterministic("mu_g", mu_pop + tau_mu * mu_g_z, dims="galaxy")
        sigma_g_z = pm.HalfNormal("sigma_g_z", 1.0, dims="galaxy")
        sigma_g = pm.Deterministic(
            "sigma_g", tau_sigma * sigma_g_z, dims="galaxy"
        )

        # Hyperparameter k is logged into constant_data so a notebook
        # reader can recover the truncation point from the trace.
        pm.Data("k", k)
        pm.Data("log_floor", floor_per_galaxy, dims="galaxy")

        pm.TruncatedNormal(
            "log_d",
            mu=mu_g[g_idx],
            sigma=sigma_g[g_idx],
            lower=log_floor_obs,
            observed=log_d,
            dims="obs",
        )
        kwargs = _merged_sample_kwargs(seed, draws, sample_kwargs)
        kwargs["target_accept"] = max(kwargs.get("target_accept", 0.95), 0.95)
        idata = pm.sample(**kwargs)
    return idata


# ---------------------------------------------------------------------------
# Model 3: Bayesian Chevalier-style energy regression
# ---------------------------------------------------------------------------


def fit_chevalier(
    holes: pd.DataFrame,
    *,
    types: Sequence[int] = (2, 3),
    galaxy_ids: Iterable[str] | None = None,
    draws: int = 2000,
    seed: int = 0,
    sample_pp: bool = True,
    sample_kwargs: dict | None = None,
) -> "az.InferenceData":
    """Chevalier-style Bayesian regression of B11 hole energies.

    Model
    -----
    Restricted to the requested ``types`` (default {2, 3}, where vexp is
    measured), with ``log_d = ln(diameter_pc)`` and ``log_v = ln(vexp_kms)``:

    * ``alpha ~ Normal(0, 5)``
    * ``beta ~ Normal(2, 1)``  (Chevalier 1974 prior on size exponent)
    * ``gamma ~ Normal(2, 1)``  (Chevalier 1974 prior on velocity exponent)
    * ``tau_galaxy ~ HalfNormal(0.5)``
    * ``alpha_g ~ Normal(0, tau_galaxy)``  (per-galaxy intercept offset)
    * ``sigma ~ HalfNormal(1)``
    * ``log_E ~ Normal(alpha + alpha_g + beta · log_d + gamma · log_v, sigma)``

    A clean posterior predictive on ``log_E_J43`` is the canonical
    sanity check for the ``hishells.catalog`` parsing (plan §11 step 6).

    Returns
    -------
    arviz.InferenceData
        ``posterior`` plus ``log_likelihood`` and (when ``sample_pp``)
        ``posterior_predictive``.
    """

    _require_pymc()
    galaxies_ord = _resolve_galaxies(holes, galaxy_ids)
    sub = holes[holes["hole_type"].isin(set(types))].copy()
    sub = sub.dropna(subset=["log_E_J43", "diameter_pc", "vexp_kms"])
    sub = sub[sub["vexp_kms"] > 0].copy()
    sub = sub[sub["galaxy_id"].isin(galaxies_ord)].reset_index(drop=True)

    log_d = np.log(sub["diameter_pc"].to_numpy(dtype=np.float64))
    log_v = np.log(sub["vexp_kms"].to_numpy(dtype=np.float64))
    log_E = sub["log_E_J43"].to_numpy(dtype=np.float64)
    g_idx = _galaxy_index(sub, galaxies_ord)

    coords = {"galaxy": galaxies_ord, "obs": np.arange(len(sub))}
    with pm.Model(coords=coords):
        alpha = pm.Normal("alpha", 0.0, 5.0)
        beta = pm.Normal("beta", 2.0, 1.0)
        gamma = pm.Normal("gamma", 2.0, 1.0)
        tau_galaxy = pm.HalfNormal("tau_galaxy", 0.5)
        alpha_g = pm.Normal("alpha_g", 0.0, tau_galaxy, dims="galaxy")
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = alpha + alpha_g[g_idx] + beta * log_d + gamma * log_v
        pm.Normal("log_E", mu=mu, sigma=sigma, observed=log_E, dims="obs")

        kwargs = _merged_sample_kwargs(seed, draws, sample_kwargs)
        kwargs["idata_kwargs"] = {"log_likelihood": True}
        idata = pm.sample(**kwargs)
        if sample_pp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm.sample_posterior_predictive(
                    idata,
                    extend_inferencedata=True,
                    random_seed=seed,
                    progressbar=False,
                )
    return idata


# ---------------------------------------------------------------------------
# Model 4: NegBinomial rate model on per-galaxy hole counts
# ---------------------------------------------------------------------------


def fit_count_rate(
    holes: pd.DataFrame,
    galaxies: pd.DataFrame,
    *,
    galaxy_ids: Iterable[str] | None = None,
    draws: int = 2000,
    seed: int = 0,
    sample_kwargs: dict | None = None,
) -> "az.InferenceData":
    """Negative-Binomial GLM of per-galaxy B11 hole counts.

    Model
    -----
    For each galaxy ``g``:

    * ``beta_0 ~ Normal(3, 2)``  (~ log(20) = baseline 20 holes/galaxy)
    * ``beta_SFR, beta_MHI, beta_D ~ Normal(0, 1)``
    * ``alpha_nb ~ Exponential(1)`` (NB dispersion)
    * ``log(mu_g) = beta_0 + beta_SFR · z_SFR + beta_MHI · z_MHI + beta_D · z_D``
    * ``n_holes_g ~ NegativeBinomial(mu_g, alpha_nb)``

    Predictors are *standardised* (z-scored across galaxies) so the
    priors on the slopes are scale-free. The ``log_likelihood`` group is
    included so callers can run ``az.loo(idata, pointwise=True)`` to flag
    Pareto-k > 0.7 outlier galaxies (plan §6 anomalous-fold callouts).

    Returns
    -------
    arviz.InferenceData
        ``posterior`` plus ``log_likelihood`` and ``observed_data``.
        ``constant_data`` carries the standardised predictors so the
        notebook can compute posterior expected counts ``E[mu_g]`` after
        the fact.
    """

    _require_pymc()
    galaxies_ord = _resolve_galaxies(holes, galaxy_ids)
    counts = (
        holes.groupby("galaxy_id").size().reindex(galaxies_ord, fill_value=0)
    )
    n_obs = counts.to_numpy(dtype=np.int64)

    gal = galaxies.set_index("galaxy_id").reindex(galaxies_ord)
    log_sfr = gal["log_sfr"].to_numpy(dtype=np.float64)
    # MHI is in 1e8 Msun units in B11 Table 2
    log_mhi = np.log(gal["MHI_1e8Msun"].to_numpy(dtype=np.float64))
    log_dist = np.log(gal["distance_mpc"].to_numpy(dtype=np.float64))

    if np.any(np.isnan(log_sfr)) or np.any(np.isnan(log_mhi)) or np.any(np.isnan(log_dist)):
        raise ValueError(
            "fit_count_rate: NaNs in log_sfr / MHI_1e8Msun / distance_mpc; "
            "check Data/J_AJ_141_23/table2.dat parsing in hishells.catalog."
        )

    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std(ddof=0)

    z_sfr = _z(log_sfr)
    z_mhi = _z(log_mhi)
    z_d = _z(log_dist)

    coords = {"galaxy": galaxies_ord}
    with pm.Model(coords=coords):
        beta_0 = pm.Normal("beta_0", 3.0, 2.0)
        beta_SFR = pm.Normal("beta_SFR", 0.0, 1.0)
        beta_MHI = pm.Normal("beta_MHI", 0.0, 1.0)
        beta_D = pm.Normal("beta_D", 0.0, 1.0)
        alpha_nb = pm.Exponential("alpha_nb", 1.0)

        z_sfr_d = pm.Data("z_sfr", z_sfr, dims="galaxy")
        z_mhi_d = pm.Data("z_mhi", z_mhi, dims="galaxy")
        z_d_d = pm.Data("z_dist", z_d, dims="galaxy")

        log_mu = beta_0 + beta_SFR * z_sfr_d + beta_MHI * z_mhi_d + beta_D * z_d_d
        mu = pm.Deterministic("mu", pm.math.exp(log_mu), dims="galaxy")

        pm.NegativeBinomial(
            "n_holes",
            mu=mu,
            alpha=alpha_nb,
            observed=n_obs,
            dims="galaxy",
        )

        kwargs = _merged_sample_kwargs(seed, draws, sample_kwargs)
        kwargs["idata_kwargs"] = {"log_likelihood": True}
        idata = pm.sample(**kwargs)
    return idata


# ---------------------------------------------------------------------------
# Model 5: BEST-style StudentT comparison of axial_ratio across hole types
# ---------------------------------------------------------------------------


def fit_axial_best(
    holes: pd.DataFrame,
    *,
    draws: int = 2000,
    seed: int = 0,
    sample_kwargs: dict | None = None,
) -> "az.InferenceData":
    """Kruschke (2013) BEST comparison of axial_ratio across hole types.

    Model
    -----
    For ``y_i = axial_ratio_i`` and ``t_i ∈ {1, 2, 3}``:

    * ``nu - 1 ~ Exponential(1/29)`` (Kruschke's recommended
      ``Exponential(1/29) + 1`` for the StudentT degrees of freedom)
    * ``mu_t ~ Normal(0.7, 0.3)``  (axial_ratio is bounded in [0, 1])
    * ``sigma_t ~ HalfNormal(0.3)``
    * ``y_i ~ StudentT(nu, mu_{t_i}, sigma_{t_i})``

    Deterministics ``diff_3_2``, ``diff_3_1``, ``diff_2_1`` give the
    posterior of pairwise mean differences directly.

    Returns
    -------
    arviz.InferenceData
        ``posterior`` plus ``log_likelihood`` and ``observed_data``.
    """

    _require_pymc()
    sub = holes.dropna(subset=["axial_ratio", "hole_type"]).copy()
    sub = sub[sub["hole_type"].isin([1, 2, 3])].reset_index(drop=True)
    types = (1, 2, 3)
    type_idx = sub["hole_type"].map({t: i for i, t in enumerate(types)}).to_numpy(
        dtype=np.int64
    )
    y = sub["axial_ratio"].to_numpy(dtype=np.float64)

    coords = {"type": list(types), "obs": np.arange(len(sub))}
    with pm.Model(coords=coords):
        nu_minus_1 = pm.Exponential("nu_minus_1", 1.0 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_1 + 1.0)
        mu_t = pm.Normal("mu_t", 0.7, 0.3, dims="type")
        sigma_t = pm.HalfNormal("sigma_t", 0.3, dims="type")
        pm.Deterministic("diff_3_2", mu_t[2] - mu_t[1])
        pm.Deterministic("diff_3_1", mu_t[2] - mu_t[0])
        pm.Deterministic("diff_2_1", mu_t[1] - mu_t[0])
        pm.StudentT(
            "y",
            nu=nu,
            mu=mu_t[type_idx],
            sigma=sigma_t[type_idx],
            observed=y,
            dims="obs",
        )
        kwargs = _merged_sample_kwargs(seed, draws, sample_kwargs)
        kwargs["idata_kwargs"] = {"log_likelihood": True}
        idata = pm.sample(**kwargs)
    return idata


# ---------------------------------------------------------------------------
# Convenience wrapper for the notebook driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesFits:
    """Bundle of the five InferenceData traces produced by the notebook.

    Returned by :func:`fit_all` so the notebook driver and tests can
    pass a single object around instead of five named locals.
    """

    type_fractions: "az.InferenceData"
    truncated_diameter: "az.InferenceData"
    chevalier: "az.InferenceData"
    count_rate: "az.InferenceData"
    axial_best: "az.InferenceData"


def fit_all(
    holes: pd.DataFrame,
    galaxies: pd.DataFrame,
    *,
    galaxy_ids: Iterable[str] | None = None,
    draws: int = 2000,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> BayesFits:
    """Fit all five models and (optionally) cache them under ``cache_dir``.

    The caller-facing ``cache_dir`` keys are stable hashes of the input
    DataFrames + ``draws`` + ``seed``, so re-running with identical data
    is a NetCDF round-trip rather than a re-fit.
    """

    _require_pymc()
    galaxies_ord = list(_resolve_galaxies(holes, galaxy_ids))

    def _fit(name: str, fn: Callable[[], "az.InferenceData"], *inputs: object) -> "az.InferenceData":
        if cache_dir is None:
            return fn()
        key = hash_dataframe(*inputs, draws, seed, tuple(galaxies_ord))
        return cached_fit(cache_dir, name, key, fn, refresh=refresh)

    fits = BayesFits(
        type_fractions=_fit(
            "type_fractions",
            lambda: fit_type_fractions(
                holes, galaxy_ids=galaxies_ord, draws=draws, seed=seed
            ),
            holes[["galaxy_id", "hole_type"]],
        ),
        truncated_diameter=_fit(
            "truncated_diameter",
            lambda: fit_truncated_diameter(
                holes, galaxies, galaxy_ids=galaxies_ord, draws=draws, seed=seed
            ),
            holes[["galaxy_id", "diameter_pc"]],
            galaxies[["galaxy_id", "resolution_pc"]],
        ),
        chevalier=_fit(
            "chevalier",
            lambda: fit_chevalier(
                holes, galaxy_ids=galaxies_ord, draws=draws, seed=seed
            ),
            holes[["galaxy_id", "hole_type", "diameter_pc", "vexp_kms", "log_E_J43"]],
        ),
        count_rate=_fit(
            "count_rate",
            lambda: fit_count_rate(
                holes, galaxies, galaxy_ids=galaxies_ord, draws=draws, seed=seed
            ),
            holes[["galaxy_id"]],
            galaxies[["galaxy_id", "log_sfr", "MHI_1e8Msun", "distance_mpc"]],
        ),
        axial_best=_fit(
            "axial_best",
            lambda: fit_axial_best(holes, draws=draws, seed=seed),
            holes[["axial_ratio", "hole_type"]],
        ),
    )
    return fits
