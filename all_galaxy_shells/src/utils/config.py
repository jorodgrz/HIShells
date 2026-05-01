# src/utils/config.py
from __future__ import annotations
from pathlib import Path
import hashlib, json
from .io import load_yaml, write_yaml

def _hash_obj(obj) -> str:
    # stable-ish hash of config (ignore _meta)
    canonical = json.dumps({k:v for k,v in obj.items() if k != "_meta"}, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

def resolve_config(cfg_path: str | Path, write_resolved: bool = False):
    """
    Load a YAML config, add a _meta hash, optionally write a resolved copy next to it.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = load_yaml(cfg_path)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["_hash"] = _hash_obj(cfg)
    if write_resolved:
        out = cfg_path.with_name(cfg_path.stem + "._resolved.yaml")
        write_yaml(out, cfg, overwrite=True)
    return cfg