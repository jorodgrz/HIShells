# HIShells

A deep-learning pipeline that detects HI shells (a.k.a. holes / bubbles) in
21 cm radio data cubes from the THINGS survey. It ingests a FITS cube, runs a
2D CNN over position–velocity (p–v) cuts, and outputs a table of shell
candidates with Monte Carlo dropout confidence estimates.

HI shells are roughly spherical cavities in the neutral interstellar medium
carved out by stellar winds, expanding HII regions, and supernovae — key
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

## Quick start (conda)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate hishells
```

Verify the install — this should print `hishells env ok` with no import errors:

```bash
python -c "import torch, astropy, spectral_cube, numpy, sklearn; print('hishells env ok')"
```

Optionally register the env as a Jupyter kernel:

```bash
python -m ipykernel install --user --name hishells --display-name "Python (hishells)"
```

To update an existing env after editing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Repository layout

```
HIShells/
├── Data/                # THINGS cubes + Bagetakos catalog (gitignored)
│   └── J_AJ_141_23/     # CDS bundle: ReadMe, table2.dat, table7.dat
├── Papers/              # Background PDFs (gitignored)
├── environment.yml      # conda environment spec
└── README.md
```

## References

PDFs live in `Papers/` (gitignored, so not part of a fresh clone).

| File | Role in this project |
| --- | --- |
| `Bagetakos_2011_AJ_141_23.pdf` | Source catalog; defines the labels. Bagetakos, Brinks, Walter, de Blok, Usero, Leroy, Rich, Kennicutt 2011, AJ 141, 23. |
| `CASI- A CONVOLUTIONAL NEURAL NETWORK APPROACH FOR SHELL IDENTIFICATION.pdf` | Closest prior work — CNN-based shell identification on molecular-line cubes. Methodological benchmark we want to beat in the p–v domain. |
| `Automatic Shell Detection in CGPS Data.pdf` | Classical (non-ML) shell-finding baseline on the Canadian Galactic Plane Survey; informs our non-ML floor. |
| `WALLABY Pilot Survey- HI source-finding with a machine learning framework.pdf` | ML source-finding precedent on HI cubes; supports the choice of CNNs over classical filtering. |
| `H i shells in the outer Milky Way.pdf` | Observational reference for shell morphology and expansion-velocity ranges; used to sanity-check augmentation parameters. |
| `Evidence for supernova feedback sustaining gas turbulence in nearby star-forming galaxies.pdf` | Science motivation for cataloging shells at scale: links shell populations to feedback-driven turbulence. |

> Bibliographic details for the last four entries are intentionally light —
> verify and fill in the full author lists / journal references before citing
> in any write-up.
