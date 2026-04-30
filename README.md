# HIShells

A deep-learning pipeline that detects HI shells (a.k.a. holes / bubbles) in
21 cm radio data cubes from the THINGS survey. It ingests a FITS cube, runs a
2D CNN over position-velocity (p-v) cuts, and outputs a table of shell
candidates with Monte Carlo dropout confidence estimates.

HI shells are roughly spherical cavities in the neutral interstellar medium
carved out by stellar winds, expanding HII regions, and supernovae, key
tracers of stellar feedback. They have historically been catalogued by eye, a
slow and subjective process. This project tests whether a CNN trained on a
published, hand-built catalog can reproduce human-level detections directly
from raw cubes.

## Data

- **Survey:** THINGS (The HI Nearby Galaxy Survey; Walter et al. 2008,
  AJ 136, 2563). 34 nearby galaxies observed in the 21 cm line with the VLA.
  Cubes are downloadable from <https://www2.mpia-hd.mpg.de/THINGS/Data.html>.
- **Labels:** Bagetakos et al. 2011, AJ 141, 23 — *"The fine-scale structure
  of the neutral interstellar medium in nearby galaxies"*. ~1046 catalogued
  HI holes across 20 THINGS galaxies, with central RA/Dec, heliocentric
  velocity, diameter, expansion velocity, position angle, and axial ratio.
  The CDS bundle is unpacked under `Data/J_AJ_141_23/` (see its `ReadMe`).

The THINGS cubes and the CDS catalog are excluded from the repo via
`.gitignore`; download them locally before running the pipeline.

To pull the THINGS cubes from the MPIA mirror, use the bundled scraper.
It scrapes the index page and hands the URL list to `aria2c`, which
opens several parallel TCP connections per file to defeat MPIA's
per-connection throttling (~100 kB/s from the US):

```bash
python scripts/fetch_things.py --catalog-only          # 19 NA cubes (~20 GB)
python scripts/fetch_things.py                         # all 33 NA cubes (~30 GB)
python scripts/fetch_things.py --product MOM0          # moment-0 maps (small)
python scripts/fetch_things.py --weighting RO          # robust-weighted instead of natural
python scripts/fetch_things.py --galaxies NGC_2403     # explicit subset
python scripts/fetch_things.py --connections 8 --jobs 2  # tune aria2 parallelism
python scripts/fetch_things.py --dry-run               # list URLs only
```

`aria2c` is pinned by `environment.yml`; if you're not using the conda
env, install it manually with `conda install -c conda-forge aria2` or
`brew install aria2`. Files land in `Data/THINGS/` by default; partial
downloads resume across runs (`--continue=true`) and completed files are
not re-fetched (`--allow-overwrite=false`). IC 2574 is in the Bagetakos
catalog but is not served from the MPIA public page, so `--catalog-only`
yields 19 of the 20 catalog galaxies and prints a warning for the
missing one.

## Approach

- Extract fixed-size, scale-normalized **p–v cut windows** along candidate
  (RA, Dec, PA) sightlines through each cube. Expanding shells produce a
  characteristic ellipse signature in p–v space — a much cleaner feature for
  a CNN than a single-channel position–position image.
- Train a 2D CNN to classify each window as shell vs. non-shell.
- Apply **Monte Carlo dropout** at inference (Gal & Ghahramani 2016) to get a
  per-detection confidence estimate from many stochastic forward passes.
- Evaluate with **leave-one-galaxy-out cross-validation** so the model never
  sees the test galaxy's noise, beam, or rotation curve during training.

## Quick start

Create and activate the environment, then install the `hishells` package in
editable mode so notebooks and scripts can `import hishells`:

```bash
conda env create -f environment.yml
conda activate hishells
pip install -e .
```

(`environment.yml` already pins `pip: -e .`, so a future
`conda env update -f environment.yml --prune` will keep the editable install
in sync.)

Verify the install — this should print `hishells env ok` with no import errors:

```bash
python -c "import torch, astropy, spectral_cube, numpy, sklearn, hishells; print('hishells env ok')"
```

Optionally register the env as a Jupyter kernel:

```bash
python -m ipykernel install --user --name hishells --display-name "Python (hishells)"
```

To update an existing env after editing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Build status (plan §11 gates)

| Gate | Module(s) | Verification |
| --- | --- | --- |
| 1 | `hishells/catalog.py`, `notebooks/01_explore_catalog.ipynb` | `pytest tests/test_catalog.py` |
| 2 | `hishells/cubes.py`, `notebooks/02_cube_sanity.ipynb` | notebook 02 alignment overlay |
| 3 | `hishells/pvcut.py`, `notebooks/03_pvcut_examples.ipynb` | `pytest tests/test_pvcut.py` |
| 4 | `hishells/{windows,augment,data}.py`, `notebooks/04_window_inspection.ipynb` | `pytest tests/test_augment.py tests/test_data.py` |
| 5 | `hishells/baselines/{trivial,mtb}.py` | `pytest tests/test_mtb.py` |
| 6 | `hishells/{model,loss}.py` | `pytest tests/test_model.py` |
| 7 | `hishells/{train,eval}.py`, `scripts/train_logo.py`, `notebooks/05_training_diagnostics.ipynb` | `python scripts/train_logo.py --name v1_baseline --limit-folds 1` |
| 8 | `hishells/predict.py`, `notebooks/{06_failure_analysis,07_mc_dropout_calibration}.ipynb` | notebook 07 reliability diagram |
| 9 | `hishells/candidates.py` | `pytest tests/test_candidates.py` |
| 10 | `scripts/{predict_galaxy,eval_logo}.py` | `python scripts/eval_logo.py` after a sweep |
| 11 | `hishells/baselines/casi.py`, `scripts/run_baseline.py`, `notebooks/08_baseline_comparison.ipynb` | `python scripts/run_baseline.py --baseline trivial --name trivial_v1` |

CASI-2D is *not* pip-installable. To enable that baseline, clone
<https://gitlab.com/casi-project/casi-2d>, follow its README to grab the
trained weights, then `export CASI_HOME=/path/to/casi-2d`. Without it the
`casi` rows in `results/ablations.csv` are written with `notes='CASI
unavailable'` and the rest of the LOGO sweep continues.

## Run a full LOGO sweep

```bash
# 1. Train one CNN per held-out galaxy (~hours on CPU; minutes on a GPU/MPS).
python scripts/train_logo.py --name v1_baseline

# 2. Score the classical baselines under the same LOGO geometry.
python scripts/run_baseline.py --baseline trivial --name trivial_v1
python scripts/run_baseline.py --baseline mtb     --name mtb_v1
# CASI is optional; only runs if CASI_HOME is set.
python scripts/run_baseline.py --baseline casi    --name casi_v1

# 3. Aggregate, bootstrap CIs, and write per-ablation summary + figures.
python scripts/eval_logo.py

# 4. Per-galaxy MC-dropout inference for downstream analysis.
python scripts/predict_galaxy.py \
    --cube Data/THINGS/NGC_2403_NA_CUBE_THINGS.FITS \
    --checkpoint results/checkpoints/v1_baseline/NGC_2403.pt \
    --out results/per_galaxy/NGC_2403.fits
```

Results land in `results/`:

- `ablations.csv` — one row per `(name, fold)` with the §6.1 14-column schema.
- `summary.csv` — bootstrapped mean ± 95% CI per ablation (from `eval_logo.py`).
- `checkpoints/<name>/<galaxy>.pt` — per-fold model weights.
- `figures/` — per-ablation diagnostic PNGs.
- `per_galaxy/<galaxy>.fits` — MC-dropout candidate tables (§7 schema).
