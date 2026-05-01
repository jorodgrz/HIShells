#!/usr/bin/env python3
# path-safe bootstrap
import os, sys
from pathlib import Path
THIS = Path(__file__).resolve(); ROOT = THIS.parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from collections import Counter

def main():
    pv = Path("data/pv"); lab = Path("data/labels")
    pv_base = {p.name for p in pv.glob("*.npy") if not p.name.endswith("_posxy.npy")}
    lab_base= {p.name for p in lab.glob("*.npy")}
    missing_lab = sorted(pv_base - lab_base)
    missing_pv  = sorted(lab_base - pv_base)
    print(f"PV arrays: {len(pv_base)}  Label arrays: {len(lab_base)}")
    print("Missing labels for PV:", missing_lab[:10], ("... (+more)" if len(missing_lab)>10 else ""))
    print("Labels without PV:", missing_pv[:10], ("... (+more)" if len(missing_pv)>10 else ""))
    if missing_lab:
        exit(1)

if __name__ == "__main__":
    main()