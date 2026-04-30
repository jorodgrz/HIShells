"""CASI-2D baseline shim (plan §9 Row 2, §11 step 11).

The CASI ("Convolutional Approach to Shell Identification") code from
Van Oort et al. 2019 is hosted at
``https://gitlab.com/casi-project/casi-2d`` and is *not* pip-installable.
To use it as a baseline, clone the repo somewhere on disk and point the
``CASI_HOME`` environment variable at it::

    git clone https://gitlab.com/casi-project/casi-2d.git ~/casi-2d
    export CASI_HOME=~/casi-2d
    # follow casi-2d/README.md for the model weights download

This module wraps that checkout in a tiny subprocess shim:

1. :func:`run_casi_on_mom0` calls ``python casi_predict.py --input
   <mom0.fits> --output <out.fits>`` (the entry point in the CASI
   distribution; we expose ``--cli`` if the upstream filename
   changes), and returns the per-pixel score map as a 2-D numpy array
   plus the matching :class:`astropy.wcs.WCS`.
2. :func:`score_table` runs CASI on each galaxy's moment-0 once,
   caches the result, and bilinear-samples the score map at every row
   in the input window table -- matching the
   ``score_table(table, cubes) -> (scores, labels)`` interface of the
   sister baselines :mod:`hishells.baselines.mtb` and
   :mod:`hishells.baselines.trivial`.

If ``CASI_HOME`` is unset (or the entry-point script is missing) every
public function raises :class:`CASINotInstalledError` with the install
instructions inlined. Down-stream code (``scripts/run_baseline.py``)
catches that and writes ``notes='CASI unavailable'`` to ``ablations.csv``
so the LOGO sweep doesn't crash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from ..cubes import Cube, moment0
from ..data import CubeStore


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


_INSTALL_INSTRUCTIONS = (
    "CASI-2D is not installed. To enable this baseline, clone the upstream\n"
    "  https://gitlab.com/casi-project/casi-2d\n"
    "checkout, follow its README to download the trained weights, then set\n"
    "  export CASI_HOME=/path/to/casi-2d\n"
    "and re-run scripts/run_baseline.py.\n"
)


class CASINotInstalledError(RuntimeError):
    """Raised when ``CASI_HOME`` is unset or the CASI entry-point is missing."""

    def __init__(self, detail: str = ""):
        msg = _INSTALL_INSTRUCTIONS
        if detail:
            msg += f"\nDetail: {detail}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Locating the CASI entry-point
# ---------------------------------------------------------------------------


def _casi_home() -> Path:
    home = os.environ.get("CASI_HOME")
    if not home:
        raise CASINotInstalledError("CASI_HOME environment variable is not set")
    p = Path(home).expanduser()
    if not p.exists():
        raise CASINotInstalledError(f"CASI_HOME={p} does not exist")
    return p


def _casi_cli_script(home: Path | None = None) -> Path:
    """Locate the CASI prediction CLI inside ``CASI_HOME``.

    Tries ``casi_predict.py`` first (the script name in the public
    distribution as of writing); falls back to scanning for any
    ``*predict*.py`` at the top level. Override via the ``--cli`` flag
    of :func:`run_casi_on_mom0` if your fork uses a different name.
    """

    home = home or _casi_home()
    candidates = [
        home / "casi_predict.py",
        home / "scripts" / "casi_predict.py",
        home / "predict.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    glob_hits = list(home.glob("*predict*.py")) + list(home.glob("scripts/*predict*.py"))
    if glob_hits:
        return glob_hits[0]
    raise CASINotInstalledError(
        f"could not find a predict CLI under CASI_HOME={home}; "
        "tried casi_predict.py, scripts/casi_predict.py, predict.py"
    )


# ---------------------------------------------------------------------------
# Run CASI on one moment-0 map
# ---------------------------------------------------------------------------


def run_casi_on_mom0(
    mom0_path: str | Path,
    *,
    output_path: str | Path | None = None,
    cli: str | Path | None = None,
    extra_args: list[str] | None = None,
    timeout_s: int = 1800,
    python: str = "python",
) -> tuple[np.ndarray, WCS]:
    """Invoke CASI on a single moment-0 FITS and return its score map.

    Parameters
    ----------
    mom0_path
        Path to the input moment-0 FITS (single-HDU, 2-D image).
    output_path
        Where to write the CASI score map. If ``None``, writes to a
        temporary file that is deleted before this function returns.
    cli
        Override path to the CASI entry-point script. Defaults to
        :func:`_casi_cli_script`.
    extra_args
        Forwarded to the subprocess after ``--input`` / ``--output``.
        Use this to pass model-checkpoint paths, batch sizes, etc.
    timeout_s
        Wall-clock subprocess timeout. CASI inference is ~1-2 min/MOM0
        on a CPU; 30 min is conservative.
    python
        Interpreter to invoke CASI with. Defaults to whatever ``python``
        resolves to on ``PATH``; set to e.g. ``/path/casi/.venv/bin/python``
        if CASI lives in its own environment.

    Returns
    -------
    score_map : np.ndarray
        2-D float32 array of CASI scores in [0, 1].
    wcs : astropy.wcs.WCS
        Celestial WCS read from the output FITS header. Should match
        the input MOM0 WCS modulo any cropping CASI does internally.
    """

    home = _casi_home()
    cli_path = Path(cli) if cli else _casi_cli_script(home)
    if not cli_path.exists():
        raise CASINotInstalledError(f"CASI CLI not found at {cli_path}")

    mom0_path = Path(mom0_path)
    cleanup_output = False
    if output_path is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="casi_out_"))
        output_path = tmpdir / "score.fits"
        cleanup_output = True
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        str(cli_path),
        "--input",
        str(mom0_path),
        "--output",
        str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(home))
        proc = subprocess.run(
            cmd,
            cwd=str(home),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"CASI returned exit={proc.returncode}\nstderr:\n{proc.stderr.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError(
                f"CASI completed but did not produce {output_path}\nstdout:\n{proc.stdout.strip()}"
            )

        with fits.open(output_path) as hdul:
            data = np.asarray(hdul[0].data).astype(np.float32)
            wcs = WCS(hdul[0].header).celestial

        if data.ndim == 3:
            data = data[0]
        elif data.ndim == 4:
            data = data[0, 0]
        return data, wcs
    finally:
        if cleanup_output and output_path.exists():
            shutil.rmtree(output_path.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-galaxy convenience
# ---------------------------------------------------------------------------


def _ensure_mom0(cube: Cube, mom0_path: Path) -> Path:
    """Materialise a moment-0 FITS for ``cube`` if ``mom0_path`` doesn't exist."""

    if mom0_path.exists():
        return mom0_path
    m0 = moment0(cube)
    header = cube.wcs2d.to_header()
    header["BUNIT"] = "Jy/beam km/s"
    header["CUBE"] = cube.path
    fits.writeto(mom0_path, m0.astype(np.float32), header, overwrite=True)
    return mom0_path


def score_table(
    table: pd.DataFrame,
    cubes: CubeStore,
    *,
    mom0_dir: str | Path | None = None,
    cache: dict[str, np.ndarray] | None = None,
    cli: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every row in ``table`` with CASI and return ``(scores, labels)``.

    Parameters
    ----------
    table
        Window table from :func:`hishells.windows.build_window_table`.
    cubes
        :class:`CubeStore` providing the matching FITS cubes.
    mom0_dir
        Directory to cache per-galaxy moment-0 FITS. Defaults to
        ``<cube_dir>/mom0/`` next to the source cubes.
    cache
        Optional ``{galaxy_id: score_map}`` cache so repeated calls in
        a LOGO sweep don't re-invoke CASI on the same MOM0.
    cli, extra_args
        Forwarded to :func:`run_casi_on_mom0`.

    The function pre-runs CASI once per galaxy in ``table["galaxy_id"]``,
    then bilinear-samples the resulting score map at every row's
    ``(ra_deg, dec_deg)``. Score maps that don't cover a particular
    sightline produce a score of 0.

    Raises :class:`CASINotInstalledError` on the first call if CASI
    isn't installed -- callers should wrap the call in try/except and
    emit a placeholder ablation row.
    """

    cache = cache if cache is not None else {}
    mom0_root = (
        Path(mom0_dir)
        if mom0_dir is not None
        else (Path(cubes.cube_dir) / "mom0" if hasattr(cubes, "cube_dir") else Path("results/mom0"))
    )
    mom0_root.mkdir(parents=True, exist_ok=True)

    scores = np.zeros(len(table), dtype=np.float64)
    labels = table["label"].values.astype(np.int64)
    if len(table) == 0:
        return scores, labels

    galaxies = sorted(set(str(g) for g in table["galaxy_id"].values))
    wcs_by_galaxy: dict[str, WCS] = {}

    for gid in galaxies:
        if gid in cache:
            continue
        cube = cubes(gid)
        mom0_path = _ensure_mom0(cube, mom0_root / f"{gid}_mom0.fits")
        score_map, wcs = run_casi_on_mom0(mom0_path, cli=cli, extra_args=extra_args)
        cache[gid] = score_map
        wcs_by_galaxy[gid] = wcs

    for i, (_, row) in enumerate(table.iterrows()):
        gid = str(row["galaxy_id"])
        score_map = cache[gid]
        wcs = wcs_by_galaxy.get(gid)
        if wcs is None:
            # Cache hit from a previous call -- recover WCS from the
            # corresponding cube's celestial WCS as a fallback.
            wcs = cubes(gid).wcs2d
            wcs_by_galaxy[gid] = wcs
        x_pix, y_pix = wcs.wcs_world2pix(
            np.array([row["ra_deg"]]), np.array([row["dec_deg"]]), 0
        )
        coords = np.array([[float(y_pix[0])], [float(x_pix[0])]])
        try:
            scores[i] = float(
                map_coordinates(score_map, coords, order=1, mode="constant", cval=0.0)[0]
            )
        except Exception:
            scores[i] = 0.0
    return scores, labels
