# HI Shell Segmentation (U-Net) – Multi-Galaxy Scaffold

This repository organizes a repeatable pipeline for generating PV slices, labeling from Bagetakos Table 7,
building training datasets, and training a U-Net to segment HI shells across many galaxies.

## Quick Start

```bash
# choose a galaxy id (see `projects/`):
make GAL=ngc_2403 pv
make GAL=ngc_2403 label
make GAL=ngc_2403 dataset
make GAL=ngc_2403 train
Configs live in projects/<GAL>/cfg/.

Outputs land under projects/<GAL>/pv, labels, dataset, runs, models.

You can swap galaxies via GAL=<gal_id> without changing code.
