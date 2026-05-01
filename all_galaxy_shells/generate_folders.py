#!/usr/bin/env python3
# scaffold_hi_shell_project.py
"""
Scaffold a multi-galaxy HI-shell project (PV generation, labeling, dataset build, training).
Creates a consistent directory layout, a master Makefile, and per-galaxy config templates.

Usage:
  python scaffold_hi_shell_project.py --root . [--overwrite]
  python scaffold_hi_shell_project.py --root . --only ngc_2403 ngc_628

Notes:
- GAL IDs are slugified (e.g., "NGC 2403" -> "ngc_2403", "Holmberg II" -> "holmberg_ii").
- You can safely re-run with --overwrite to refresh templates.
"""

import argparse
import json
import os
from pathlib import Path
import textwrap
import yaml
import re
from datetime import datetime

# --------- Galaxy catalog from your message ---------
RAW_GALAXIES = [
    # name, alt_name, RA, Dec, type, dist_mpc, incl_deg, pa_deg, extra1, extra2, extra3
    ("NGC 628",     "M 74",      "01 36 41.8", "+15 47 00.0", "SA(s)c",     7.3,  7,  20, 38.0,  0.08,  1.99, 218),
    ("NGC 2366",    "DDO 42",    "07 28 53.4", "+69 12 51.1", "IB(s)m",     3.4, 64,  40,  6.5, -1.02,  1.64, 107),
    ("NGC 2403",    "",          "07 36 51.1", "+65 36 02.9", "SAB(s)cd",   3.2, 63, 124, 25.8, -0.07,  2.20,  87),
    ("Holmberg II", "DDO 50",    "08 19 05.0", "+70 43 12.0", "Im",         3.4, 41, 177,  5.9, -1.13,  1.82, 107),
    ("DDO 53",      "",          "08 34 07.2", "+66 10 54.0", "Im",         3.6, 31, 132,  0.6, -2.10,  0.89, 103),
    ("NGC 2841",    "",          "09 22 02.6", "+50 58 35.4", "SA(r)b",    14.1, 74, 153, 85.6, -0.70,  1.80, 405),
    ("Holmberg I",  "DDO 63",    "09 40 32.3", "+71 10 56.0", "IAB(s)m",    3.8, 12,  50,  1.4, -2.23,  1.52, 128),
    ("NGC 2976",    "",          "09 47 15.3", "+67 55 00.0", "SAc",        3.6, 65, 335,  1.4, -0.98,  1.90,  87),
    ("NGC 3031",    "M 81",      "09 55 33.1", "+69 03 54.7", "SA(s)ab",    3.6, 59, 330, 36.3,  0.03,  2.33, 132),
    ("NGC 3184",    "",          "10 18 17.0", "+41 25 28.0", "SAB(rs)cd", 11.1, 16, 179, 30.6,  0.16,  1.87, 281),
    ("IC 2574",     "DDO 81",    "10 28 27.7", "+68 24 59.4", "SAB(s)m",    4.0, 53,  55, 14.7, -0.93,  2.11, 111),
    ("NGC 3521",    "",          "11 05 48.6", "-00 02 09.2", "SAB(rs)bc", 10.7, 73, 340, 80.2,  0.52,  1.92, 376),
    ("NGC 3627",    "M 66",      "11 20 15.0", "+12 59 29.6", "SAB(s)b",    9.3, 62, 173,  8.2,  0.39,  2.01, 249),
    ("NGC 4214",    "",          "12 15 39.2", "+36 19 37.0", "IAB(s)m",    2.9, 44,  65,  4.1, -1.28,  1.83,  98),
    ("NGC 4449",    "",          "12 28 11.9", "+44 05 40.0", "IBm",        4.2, 60, 230, 11.0, -0.33,  1.67, 267),
    ("NGC 4736",    "M 94",      "12 50 53.0", "+41 07 13.2", "(R)SA(r)ab", 4.7, 41, 296,  4.0, -0.37,  1.89, 130),
    ("DDO 154",     "NGC 4789A", "12 54 05.9", "+27 09 09.9", "IB(s)m",     4.3, 66, 230,  3.6, -2.44,  1.29, 147),
    ("NGC 5194",    "M 51",      "13 29 52.7", "+47 11 43.0", "SA(s)bc",    8.0, 42, 172, 25.4,  0.78,  1.89, 221),
    ("NGC 6946",    "",          "20 34 52.2", "+60 09 14.4", "SAB(rs)cd",  5.9, 33, 243, 41.5,  0.68,  2.06, 135),
    ("NGC 7793",    "",          "23 57 49.7", "-32 35 27.9", "SA(s)d",     3.9, 50, 290,  8.9, -0.29,  2.02, 142),
]

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def make_gal_record(row):
    (name, alt, ra, dec, morph, dist, incl, pa, *extras) = row
    gal_id = slugify(name)
    rec = {
        "galaxy_id": gal_id,
        "name": name,
        "aliases": [a for a in [alt] if a],
        "ra_hms": ra,
        "dec_dms": dec,
        "morphology": morph,
        "distance_mpc": float(dist),
        "inclination_deg": int(incl),
        "position_angle_deg": int(pa),
        "extras": list(extras),  # preserved but not interpreted
    }
    return rec

GALAXIES = [make_gal_record(r) for r in RAW_GALAXIES]

# --------- Templates ---------

TOP_GITIGNORE = """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.python-version
.venv/
env/
venv/
.ipynb_checkpoints/

# Artifacts
runs/
models/
*.ckpt
*.pth
*.h5
*.log

# Data caches
*.npz
*.npy
*.parquet

# OS
.DS_Store
"""

TOP_README = """# HI Shell Segmentation (U-Net) – Multi-Galaxy Scaffold

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
"""

MASTER_MAKEFILE = r"""

Master Makefile — use GAL=<gal_id>
Example:
make GAL=ngc_2403 pv
make GAL=ngc_2403 label
make GAL=ngc_2403 dataset
make GAL=ngc_2403 train
GAL ?= ngc_2403

Core paths
PROJ_DIR := projects/$(GAL)
CFG_DIR := $(PROJ_DIR)/cfg

PV & Labels
pv:
python -m src.pv.make_pv --config $(CFG_DIR)/pv.yaml

label:
python -m src.pv.label_pv --config $(CFG_DIR)/label.yaml

Dataset build (patching, splits)
dataset:
python -m src.data.build_dataset --config $(CFG_DIR)/dataset.yaml

Train U-Net
train:
python -m src.train.train --config $(CFG_DIR)/train.yaml

Optional evaluation/visualization target
eval:
python -m src.eval.evaluate --config $(CFG_DIR)/train.yaml

Housekeeping
clean:
rm -rf $(PROJ_DIR)/pv/* $(PROJ_DIR)/labels/* $(PROJ_DIR)/dataset/*

Convenience
show:
@echo "Using GAL=$(GAL)";
echo "Configs:"; ls -1 $(CFG_DIR)
"""

PER_GAL_README = """# {name} ({gal_id})

Home for configs and artifacts for {name}.

Typical flow
bash
Copy code
make GAL={gal_id} pv
make GAL={gal_id} label
make GAL={gal_id} dataset
make GAL={gal_id} train
"""

PV_YAML = """# PV generation config for {name}
galaxy:
id: {gal_id}
name: {name}
ra_hms: "{ra_hms}"
dec_dms: "{dec_dms}"
distance_mpc: {distance_mpc}
inclination_deg: {inclination_deg}
position_angle_deg: {position_angle_deg}

inputs:
cube_fits: "DATA/{gal_id}/cube.fits" # <--- put your data cube here
wcs_fallback: null

pv:

Example: radial and azimuthal sampling, or custom cuts you already implement
You can enrich these with your current, working PV code parameters.
n_cuts: 64
cut_length_arcmin: 30.0
cut_width_arcsec: 30.0
velocity_smooth_channels: 3

outputs:
pv_dir: "projects/{gal_id}/pv"
figs_dir: "projects/{gal_id}/figs"
overwrite: false
"""

LABEL_YAML = """# Labeling config for {name} (Bagetakos Table 7)
galaxy:
id: {gal_id}
name: {name}

labels:
bagetakos_table7_path: "DATA/bagetakos_table7.dat" # global file

If you keep per-galaxy filtered CSVs, point here:
per_gal_csv: null

pv:
pv_dir: "projects/{gal_id}/pv"

How you map Table 7 (RA/Dec, sizes, PA) to PV domain is handled by your existing code.
Include any matching tolerances you use here:
match:
max_sep_arcsec: 10.0
use_wcs: true

outputs:
label_dir: "projects/{gal_id}/labels"
figs_dir: "projects/{gal_id}/figs"
overwrite: false
"""

DATASET_YAML = """# Dataset build config for {name}
galaxy:
id: {gal_id}
name: {name}

data:
pv_root: "projects/{gal_id}/pv"
label_root: "projects/{gal_id}/labels"
out_root: "projects/{gal_id}/dataset"

patching:
patch_pos: 512 # spatial pixels
patch_vel: 96 # spectral channels
vel_channels: 128 # full PV vel depth to consider
pos_fraction: 0.5 # class-balance sampler (pos/neg mix)

splits:
train_fraction: 0.8
val_fraction: 0.1
test_fraction: 0.1
seed: 42

outputs:
overwrite: false
"""

TRAIN_YAML = """# Training config (U-Net) for {name}
galaxy:
id: {gal_id}
name: {name}

dataset:
root: "projects/{gal_id}/dataset"
batch_size: 8
num_workers: 4
augment:
flips: true
rotations: true
noise_std: 0.0

model:
type: "unet"
in_channels: 1
out_channels: 1
base_channels: 32
depth: 4

train:
epochs: 100
lr: 1e-3
optimizer: "adam"
loss: "bce_dice"
metrics: ["dice", "iou"]

outputs:
runs_dir: "projects/{gal_id}/runs"
models_dir: "projects/{gal_id}/models"
log_every: 50
save_every: 1
overwrite: false
"""

SRC_PLACEHOLDER = """# This is a placeholder to pin the expected module layout.

Implement your working code in these modules:
- src/pv/make_pv.py (reads cfg/pv.yaml)
- src/pv/label_pv.py (reads cfg/label.yaml)
- src/data/build_dataset.py (reads cfg/dataset.yaml)
- src/train/train.py (reads cfg/train.yaml)
"""

EVAL_PLACEHOLDER = """# src/eval/evaluate.py

Optional: implement evaluation/visualization for trained models.
if name == "main":
import argparse
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
args = p.parse_args()
print("Eval placeholder. Read:", args.config)
"""

def write_text(path: Path, content: str, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(content, encoding="utf-8")

def write_yaml(path: Path, obj, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def create_top_level(root: Path, overwrite: bool):
    write_text(root / ".gitignore", TOP_GITIGNORE.strip() + "\n", overwrite)
    write_text(root / "README.md", TOP_README, overwrite)
    write_text(root / "Makefile", MASTER_MAKEFILE.strip() + "\n", overwrite)

    # expected src layout (placeholders so imports resolve)
    write_text(root / "src/README.md", SRC_PLACEHOLDER, overwrite)
    for sub in ["pv", "data", "train", "eval"]:
        (root / f"src/{sub}").mkdir(parents=True, exist_ok=True)
        init = root / f"src/{sub}/__init__.py"       # ← moved inside loop
        write_text(init, "", overwrite)

    write_text(root / "src/eval/evaluate.py", EVAL_PLACEHOLDER, overwrite)

    # global DATA dir for cubes & table7
    (root / "DATA").mkdir(parents=True, exist_ok=True)
    # encourage per-galaxy data subfolders
    for g in GALAXIES:
        (root / f"DATA/{g['galaxy_id']}").mkdir(parents=True, exist_ok=True)


def create_per_galaxy(root: Path, gal, overwrite: bool):
    gal_id = gal["galaxy_id"]
    gdir = root / f"projects/{gal_id}"

    # folders
    for sub in ["cfg", "pv", "labels", "dataset", "runs", "models", "figs", "notes"]:
        (gdir / sub).mkdir(parents=True, exist_ok=True)

    # meta.yaml
    write_yaml(
        gdir / "meta.yaml",
        {"generated": datetime.utcnow().isoformat() + "Z", "galaxy": gal},
        overwrite,
    )

    # configs
    write_text(gdir / "cfg/pv.yaml", PV_YAML.format(**gal), overwrite)
    write_text(gdir / "cfg/label.yaml", LABEL_YAML.format(**gal), overwrite)
    write_text(gdir / "cfg/dataset.yaml", DATASET_YAML.format(**gal), overwrite)
    write_text(gdir / "cfg/train.yaml", TRAIN_YAML.format(**gal), overwrite)

    # README
    write_text(gdir / "README.md", PER_GAL_README.format(name=gal["name"], gal_id=gal_id), overwrite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root to scaffold")
    ap.add_argument("--only", nargs="*", help="Subset of galaxy IDs (e.g., ngc_2403 ngc_628)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    create_top_level(root, args.overwrite)

    selected = GALAXIES
    if args.only:
        want = set(slugify(x) for x in args.only)
        selected = [g for g in GALAXIES if g["galaxy_id"] in want]
        missing = want - {g["galaxy_id"] for g in selected}
        if missing:
            print("Warning: unknown galaxy IDs:", ", ".join(sorted(missing)))

    for gal in selected:
        create_per_galaxy(root, gal, args.overwrite)

    print(f"Scaffold complete at: {root}")
    print("Galaxies:")
    for g in selected:
        print(f" - {g['galaxy_id']} ({g['name']})")


if __name__ == "__main__":   # ← fixed guard
    main()