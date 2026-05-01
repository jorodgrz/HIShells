#!/usr/bin/env python3
# --- repo-root bootstrap (works no matter where you run from) ---
import os, sys, json
from pathlib import Path

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]  # repo root = parent of scripts/
os.chdir(ROOT)          # make relative paths (pv_config.yaml, data/) work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print(f"[bootstrap] ROOT={ROOT}")
print(f"[bootstrap] in PYTHONPATH? {ROOT in list(map(Path, map(Path, sys.path)))}")

# --- real work starts here ---
from src.utils.io import load_yaml
from src.utils.catalog import load_catalogs

def main():
    cfg_path = "data/_resolved_config.yaml" if Path("data/_resolved_config.yaml").exists() else "pv_config.yaml"
    cfg = load_yaml(cfg_path)
    df = load_catalogs(cfg)

    # Basic summary
    rep = {
        "n_rows": int(len(df)),
        "n_with_vexp": int(df["vexp_kms"].notna().sum()),
        "null_counts": {k:int(v) for k,v in df[["ra_deg","dec_deg","major_arcsec","minor_arcsec","pa_deg","vexp_kms"]].isna().sum().items()},
    }
    print(json.dumps(rep, indent=2))

    # Peek a few rows for your target galaxy (e.g., NGC 2403)
    target = "NGC 2403"
    dfg = df[df["galaxy"].str.fullmatch(target, case=False, na=False)]
    print(f"\n== Sample rows for {target} ==")
    print(dfg.head(10).to_string(index=False))

    Path("data/qa").mkdir(parents=True, exist_ok=True)
    with open("data/qa/catalog_report.json","w") as f:
        json.dump(rep, f, indent=2)
    print("Wrote data/qa/catalog_report.json")

if __name__ == "__main__":
    main()