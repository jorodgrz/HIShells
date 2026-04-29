# HIShells

Code for a machine learning model to detect HI shells in FITS data cubes.

## Quick start (conda)

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate hishells

# 2. (Optional) register as a Jupyter kernel
python -m ipykernel install --user --name hishells --display-name "Python (hishells)"

# 3. Inspect a sample cube
python python.py
```

If you'd rather update an existing env after editing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Data

Large FITS cubes and PDF references are not tracked in git (see `.gitignore`).
Place data files locally under `Data/` — for example:

```
Data/
├── DDO154_NA_CUBE_THINGS.fits
└── ...
```

The THINGS survey cubes can be downloaded from
<https://www2.mpia-hd.mpg.de/THINGS/Data.html>.
