#!/usr/bin/env python3
# --- repo-root bootstrap ---
import os, sys
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f"[bootstrap] ROOT={ROOT}")

# --- real work ---
import argparse, random
from src.utils.io import load_yaml

def parse_family(p: Path) -> str:
    name = p.stem.lower()
    if name.startswith("spoke_"): return "spoke"
    if name.startswith("ring_"):  return "ring"
    if "major" in name or "minor" in name or name.startswith("axis_"): return "axis"
    return "unknown"

def write_manifest(paths, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in sorted(paths):
            f.write(p.name + "\n")

def main(cfg_path, seed):
    # prefer resolved config if available
    cfg = load_yaml("data/_resolved_config.yaml") if Path("data/_resolved_config.yaml").exists() else load_yaml(cfg_path)
    pv_dir   = Path(cfg["output_root"]) / "pv"
    lab_dir  = Path(cfg["output_root"]) / "labels"
    assert pv_dir.exists(), f"Missing {pv_dir}. Run scripts/make_pv.sh first."
    assert lab_dir.exists(), f"Missing {lab_dir}. Run labeling first."

    # 1) only base PV arrays (exclude *_posxy.npy)
    candidates = [p for p in pv_dir.glob("*.npy") if not p.name.endswith("_posxy.npy")]
    # 2) keep only those that have a matching label file
    files = [p for p in candidates if (lab_dir / p.name).exists()]
    dropped = set(candidates) - set(files)

    print(f"[gen_splits] pv candidates={len(candidates)}, with_labels={len(files)}, dropped_no_label={len(dropped)}")
    if not files:
        raise SystemExit("[gen_splits] No PV+label pairs found. Did labeling succeed?")

    random.seed(seed)
    fam2files = {}
    for f in files:
        fam = parse_family(f)
        fam2files.setdefault(fam, []).append(f)

    train_frac = float(cfg["train"]["splits"]["train_frac"])
    val_frac   = float(cfg["train"]["splits"]["val_frac"])
    test_frac  = float(cfg["train"]["splits"]["test_frac"])
    assert abs((train_frac+val_frac+test_frac)-1.0) < 1e-6, "Split fractions must sum to 1."

    train, val, test = [], [], []
    for fam, lst in fam2files.items():
        if not lst: continue
        random.shuffle(lst)
        n = len(lst)
        nt = int(n * train_frac)
        nv = int(n * val_frac)
        train += lst[:nt]
        val   += lst[nt:nt+nv]
        test  += lst[nt+nv:]

    splits_dir = Path(cfg["output_root"]) / "splits"
    write_manifest(train, splits_dir/"train_manifest.txt")
    write_manifest(val,   splits_dir/"val_manifest.txt")
    write_manifest(test,  splits_dir/"test_manifest.txt")

    print(f"[gen_splits] wrote {len(train)} train, {len(val)} val, {len(test)} test to {splits_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    main(args.config, args.seed)