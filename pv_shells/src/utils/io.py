# src/utils/io.py
from __future__ import annotations
from pathlib import Path
import json
import yaml
from typing import Any, Dict

__all__ = ["load_yaml", "save_yaml", "dump_json", "dumps_json", "read_text", "write_text"]

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Read a YAML file and return a dict (returns {} if empty)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}

def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    """Write a dict to YAML (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def dump_json(obj: Any, path: str | Path) -> None:
    """Write JSON with pretty indent (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)

# backward-compat alias (older code calls dumps_json)
dumps_json = dump_json

def read_text(path: str | Path) -> str:
    return Path(path).read_text()

def write_text(text: str, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

if __name__ == "__main__":
    # quick self-test
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        yml = Path(td)/"x.yaml"
        jsonp = Path(td)/"x.json"
        save_yaml({"a": 1}, yml)
        assert load_yaml(yml)["a"] == 1
        dump_json({"b": 2}, jsonp)
        assert json.loads(jsonp.read_text())["b"] == 2
        print("[io] self-test OK")