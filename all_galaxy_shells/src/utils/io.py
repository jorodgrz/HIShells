# src/utils/io.py
from __future__ import annotations
from pathlib import Path
import json
import yaml

def load_yaml(path: str | Path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(path: str | Path, obj, overwrite: bool = True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        return
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def dumps_json(obj, path: str | Path, overwrite: bool = True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        return
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)