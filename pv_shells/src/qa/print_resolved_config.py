import argparse, json
from src.utils.config import resolve_config

def main(cfg_path, sets):
    cfg = resolve_config(cfg_path, sets, write_resolved=True)
    print(json.dumps({"hash": cfg["_meta"]["_hash"], "galaxy": cfg["galaxy"]}, indent=2))
    print("[print_resolved_config] OK")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. model.base_filters=48")
    args = ap.parse_args()
    main(args.config, args.set)