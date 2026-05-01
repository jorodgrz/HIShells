#!/usr/bin/env python3
import os, sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
pv = Path("data/pv")
files = sorted(pv.glob("*.npy"))
meta = {p.stem for p in pv.glob("*.json")}
k = Counter()
bad = []
for f in files:
    if f.name.endswith("_posxy.npy"):
        k["posxy"] += 1
        continue
    k["pv_arrays"] += 1
    if f.stem not in meta:
        bad.append(f.name)
print("PV arrays:", k["pv_arrays"], "posxy helpers:", k["posxy"])
print("PV arrays missing meta.json:", bad if bad else "none")