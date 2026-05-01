# HIShells

A deep-learning pipeline for detecting HI shells (cavities in the neutral
interstellar medium) in 21 cm radio data cubes from the THINGS survey. It
ingests a FITS cube, runs a 2D CNN over position–velocity (p–v) cuts, and
emits a table of shell candidates with Monte Carlo dropout confidences.

HI shells are catalogued by hand in published surveys, which is slow and
subjective. This project tests whether a CNN trained on one such catalog can
reproduce human-level detections directly from raw cubes.

## Data

- **Cubes:** THINGS (Walter et al. 2008, AJ 136, 2563) — 34 nearby galaxies
  in 21 cm with the VLA. <https://www2.mpia-hd.mpg.de/THINGS/Data.html>
- **Labels:** Bagetakos et al. 2011, AJ 141, 23 — ~1046 HI holes across 20
  THINGS galaxies. CDS bundle unpacks to `Data/J_AJ_141_23/`.

Use the bundled scraper to pull cubes from the MPIA mirror via `aria2c`

## Commands:

```bash
python scripts/fetch_things.py --catalog-only          # 19 NA cubes (~20 GB)
python scripts/fetch_things.py                         # all 33 NA cubes (~30 GB)
python scripts/fetch_things.py --product MOM0          # moment-0 maps
python scripts/fetch_things.py --weighting RO          # robust instead of natural
python scripts/fetch_things.py --galaxies NGC_2403     # explicit subset
python scripts/fetch_things.py --connections 8 --jobs 2
python scripts/fetch_things.py --dry-run
```

`aria2c` is pinned in `environment.yml`; otherwise install via
`conda install -c conda-forge aria2` or `brew install aria2`. Files land in
`Data/THINGS/`, partial downloads resume, and completed files are skipped.
IC 2574 is in the Bagetakos catalog but not on the MPIA page, so
`--catalog-only` returns 19 of 20 galaxies and warns about the missing one.

## Approach

- Extract fixed-size, scale-normalized **p–v cut windows** along candidate
  (RA, Dec, PA) sightlines. Expanding shells leave a clean ellipse signature
  in p–v space — a sharper feature than any single position–position image.
- Train a 2D CNN to classify each window as shell vs. non-shell.
- Apply **Monte Carlo dropout** at inference (Gal & Ghahramani 2016) for
  per-detection confidence from many stochastic forward passes.
- Evaluate with **leave-one-galaxy-out (LOGO) cross-validation** so the
  model never sees the test galaxy's noise, beam, or rotation curve.

## Quick start

```bash
conda env create -f environment.yml
conda activate hishells
pip install -e .
```

Verify:

```bash
python -c "import torch, astropy, spectral_cube, numpy, sklearn, hishells; print('hishells env ok')"
```

Optional Jupyter kernel:

```bash
python -m ipykernel install --user --name hishells --display-name "Python (hishells)"
```

## Run Order

```bash
# 1. Train one CNN per held-out galaxy (hours on CPU, minutes on GPU/MPS).
python scripts/train_logo.py --name v1_baseline

# 2. Score classical baselines under the same LOGO geometry.
python scripts/run_baseline.py --baseline trivial --name trivial_v1
python scripts/run_baseline.py --baseline mtb     --name mtb_v1
python scripts/run_baseline.py --baseline casi    --name casi_v1   # optional

# 3. Aggregate, bootstrap CIs, write summaries and figures.
python scripts/eval_logo.py

# 4. Per-galaxy MC-dropout inference.
python scripts/predict_galaxy.py \
    --cube Data/THINGS/NGC_2403_NA_CUBE_THINGS.FITS \
    --checkpoint results/checkpoints/v1_baseline/NGC_2403.pt \
    --out results/per_galaxy/NGC_2403.fits
```

