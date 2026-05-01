#!/usr/bin/env python3
"""
Batch-update per-galaxy pv.yaml configs (shell-axis cuts + empty-region negatives),
with optional cleanup of legacy keys and migration of cube.path -> cube_path.

Features
- Set pv.shell_axes.* (enabled, allowed_types, label_table_path, PA convention, scales, etc.)
- Set pv.negatives.* (enabled, n_per_shell, mask percentiles, min sep, length, orientation, etc.)
- Deep-merge a patch YAML first, then apply CLI overrides
- Filter by galaxy IDs
- Dry-run preview of changes
- Purge legacy keys: pv.grid, pv.axes, pv.mode, pv.majmin
- Migrate cube.path (nested) -> cube_path (top level)

Usage examples
  python tools/update_pv_cfgs.py --purge-grid --purge-axes --purge-mode --purge-majmin \
      --migrate-cube-path --shell-enabled 1 --neg-enabled 1
  python tools/update_pv_cfgs.py --only holmberg_i --shell-label-path raw/Holmberg_I/bagetakos_table7.dat
  python tools/update_pv_cfgs.py --patch patch.yaml --dry-run
"""

import argparse
from pathlib import Path
import sys
import time
import copy
import yaml

ROOT = Path(__file__).resolve().parents[1]  # repo root


# --------------------------- YAML utils ---------------------------

def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(p: Path, obj):
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def deep_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def deep_set(d, path, value):
    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

def deep_delete(d, path):
    cur = d
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)

def deep_merge(a, b):
    """Deep-merge dict b into a (in place)."""
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


# --------------------------- CLI logic ---------------------------

def find_pv_yaml_files(root: Path, only_ids: set[str] | None):
    base = root / "projects"
    if not base.exists():
        return []
    out = []
    for gal_dir in sorted(base.iterdir()):
        if not gal_dir.is_dir():
            continue
        gal_id = gal_dir.name
        if only_ids and gal_id not in only_ids:
            continue
        pv_yaml = gal_dir / "cfg" / "pv.yaml"
        if pv_yaml.exists():
            out.append((gal_id, pv_yaml))
    return out

def migrate_cube_path(cfg, enable: bool, report_changes: list):
    """
    If enabled and cfg has cube.path (nested), move to top-level cube_path.
    Leaves the old cube.path removed.
    """
    if not enable or not isinstance(cfg, dict):
        return
    nested = deep_get(cfg, ["cube", "path"], None)
    top = cfg.get("cube_path")
    if nested and not top:
        # move
        cfg["cube_path"] = nested
        deep_delete(cfg, ["cube", "path"])
        # drop empty 'cube' if now empty
        if isinstance(cfg.get("cube"), dict) and not cfg["cube"]:
            cfg.pop("cube", None)
        report_changes.append(("cube_path", None, nested))
    elif nested and top and nested != top:
        # prefer existing top-level; remove nested
        deep_delete(cfg, ["cube", "path"])
        if isinstance(cfg.get("cube"), dict) and not cfg["cube"]:
            cfg.pop("cube", None)
        report_changes.append(("cube.path (removed)", nested, None))

def apply_cli_overrides(cfg, args):
    # --- pv.shell_axes.* ---
    if args.shell_enabled is not None:
        deep_set(cfg, ["pv", "shell_axes", "enabled"], bool(args.shell_enabled))
    if args.shell_label_path is not None:
        deep_set(cfg, ["pv", "shell_axes", "label_table_path"], args.shell_label_path)
    if args.shell_pa_convention is not None:
        deep_set(cfg, ["pv", "shell_axes", "pa_convention"], args.shell_pa_convention)
    if args.shell_allowed_types is not None:
        deep_set(cfg, ["pv", "shell_axes", "allowed_types"], [int(t) for t in args.shell_allowed_types])
    if args.shell_fallback_gal_pa is not None:
        deep_set(cfg, ["pv", "shell_axes", "fallback_to_gal_pa_if_missing"], bool(args.shell_fallback_gal_pa))
    if args.shell_len_major is not None:
        deep_set(cfg, ["pv", "shell_axes", "length_scale_major"], float(args.shell_len_major))
    if args.shell_len_minor is not None:
        deep_set(cfg, ["pv", "shell_axes", "length_scale_minor"], float(args.shell_len_minor))
    if args.shell_slit_width_pix is not None:
        deep_set(cfg, ["pv", "shell_axes", "slit_width_pix"], int(args.shell_slit_width_pix))
    if args.shell_pos_step_pix is not None:
        deep_set(cfg, ["pv", "shell_axes", "pos_step_pix"], float(args.shell_pos_step_pix))
    if args.shell_name_prefix is not None:
        deep_set(cfg, ["pv", "shell_axes", "name_prefix"], args.shell_name_prefix)

    # --- pv.negatives.* ---
    if args.neg_enabled is not None:
        deep_set(cfg, ["pv", "negatives", "enabled"], bool(args.neg_enabled))
    if args.neg_n_per_shell is not None:
        deep_set(cfg, ["pv", "negatives", "n_per_shell"], int(args.neg_n_per_shell))
    if args.neg_galaxy_pct is not None:
        deep_set(cfg, ["pv", "negatives", "galaxy_mask_percentile"], float(args.neg_galaxy_pct))
    if args.neg_empty_pct is not None:
        deep_set(cfg, ["pv", "negatives", "empty_percentile"], float(args.neg_empty_pct))
    if args.neg_min_sep_as is not None:
        deep_set(cfg, ["pv", "negatives", "min_sep_from_shell_arcsec"], float(args.neg_min_sep_as))
    if args.neg_length_as is not None:
        deep_set(cfg, ["pv", "negatives", "length_arcsec"], float(args.neg_length_as))
    if args.neg_orientation is not None:
        deep_set(cfg, ["pv", "negatives", "orientation"], args.neg_orientation)
    if args.neg_slit_width_pix is not None:
        deep_set(cfg, ["pv", "negatives", "slit_width_pix"], int(args.neg_slit_width_pix))
    if args.neg_pos_step_pix is not None:
        deep_set(cfg, ["pv", "negatives", "pos_step_pix"], float(args.neg_pos_step_pix))
    if args.neg_seed is not None:
        deep_set(cfg, ["pv", "negatives", "seed"], int(args.neg_seed))

    # --- Optional: purge legacy PV keys ---
    if args.purge_grid:
        deep_delete(cfg, ["pv", "grid"])
    if args.purge_axes:
        deep_delete(cfg, ["pv", "axes"])
    if args.purge_mode:
        deep_delete(cfg, ["pv", "mode"])
    if args.purge_majmin:
        deep_delete(cfg, ["pv", "majmin"])

def main():
    ap = argparse.ArgumentParser(description="Batch-update pv.yaml for shell-axis cuts + negatives, with legacy cleanup.")
    ap.add_argument("--root", default=str(ROOT), help="Repo root (default: repo root)")
    ap.add_argument("--only", nargs="*", help="Subset of galaxy IDs (e.g., ngc_2403 ngc_628)")
    ap.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    ap.add_argument("--patch", type=str, help="YAML patch file to deep-merge before CLI overrides")

    # cleanup flags
    ap.add_argument("--purge-grid", action="store_true", help="Remove pv.grid block from each pv.yaml")
    ap.add_argument("--purge-axes", action="store_true", help="Remove pv.axes block from each pv.yaml")
    ap.add_argument("--purge-mode", action="store_true", help="Remove pv.mode from each pv.yaml")
    ap.add_argument("--purge-majmin", action="store_true", help="Remove pv.majmin from each pv.yaml")

    # migrate cube.path -> cube_path
    ap.add_argument("--migrate-cube-path", action="store_true", help="Move cube.path to top-level cube_path and delete old key")

    # pv.shell_axes.*
    ap.add_argument("--shell-enabled", type=int, choices=[0,1], help="0/1")
    ap.add_argument("--shell-label-path", type=str)
    ap.add_argument("--shell-pa-convention", choices=["astro", "image"])
    ap.add_argument("--shell-allowed-types", nargs="*", type=int, help="e.g., 2 3")
    ap.add_argument("--shell-fallback-gal-pa", type=int, choices=[0,1], help="use galaxy PA if shell PA missing (0/1)")
    ap.add_argument("--shell-len-major", type=float, help="length_scale_major")
    ap.add_argument("--shell-len-minor", type=float, help="length_scale_minor")
    ap.add_argument("--shell-slit-width-pix", type=int)
    ap.add_argument("--shell-pos-step-pix", type=float)
    ap.add_argument("--shell-name-prefix", type=str)

    # pv.negatives.*
    ap.add_argument("--neg-enabled", type=int, choices=[0,1], help="0/1")
    ap.add_argument("--neg-n-per-shell", type=int)
    ap.add_argument("--neg-galaxy-pct", type=float, help="galaxy_mask_percentile")
    ap.add_argument("--neg-empty-pct", type=float, help="empty_percentile")
    ap.add_argument("--neg-min-sep-as", type=float, help="min_sep_from_shell_arcsec")
    ap.add_argument("--neg-length-as", type=float, help="length_arcsec")
    ap.add_argument("--neg-orientation", choices=["random","galaxy_axes"])
    ap.add_argument("--neg-slit-width-pix", type=int)
    ap.add_argument("--neg-pos-step-pix", type=float)
    ap.add_argument("--neg-seed", type=int)

    args = ap.parse_args()

    root = Path(args.root).resolve()
    only_ids = set(args.only) if args.only else None

    pv_files = find_pv_yaml_files(root, only_ids)
    if not pv_files:
        print("No pv.yaml files found. Check --root or project layout.")
        sys.exit(1)

    patch_obj = None
    if args.patch:
        patch_path = Path(args.patch).resolve()
        if not patch_path.exists():
            print(f"Patch file not found: {patch_path}")
            sys.exit(1)
        patch_obj = load_yaml(patch_path)
        if not isinstance(patch_obj, dict):
            print("Patch must be a YAML mapping (dict).")
            sys.exit(1)

    tstamp = time.strftime("%Y%m%d-%H%M%S")
    changed = 0

    for gal_id, yml in pv_files:
        cfg = load_yaml(yml)
        before = copy.deepcopy(cfg)

        # Collect migration notes for dry-run printing
        migration_notes = []

        # Apply patch first, then explicit CLI overrides, then migrations/purges
        if patch_obj:
            deep_merge(cfg, patch_obj)

        apply_cli_overrides(cfg, args)

        if args.migrate_cube_path:
            migrate_cube_path(cfg, True, migration_notes)

        if cfg == before and not migration_notes:
            print(f"[=] {gal_id}: no changes")
            continue

        print(f"[~] {gal_id}: updating {yml.relative_to(root)}")
        if args.dry_run:
            # Summarize important deltas
            keys_to_show = [
                # cube path & migrations
                ("cube_path", ["cube_path"]),
                # shell_axes
                ("pv.shell_axes.enabled", ["pv","shell_axes","enabled"]),
                ("pv.shell_axes.label_table_path", ["pv","shell_axes","label_table_path"]),
                ("pv.shell_axes.pa_convention", ["pv","shell_axes","pa_convention"]),
                ("pv.shell_axes.allowed_types", ["pv","shell_axes","allowed_types"]),
                ("pv.shell_axes.fallback_to_gal_pa_if_missing", ["pv","shell_axes","fallback_to_gal_pa_if_missing"]),
                ("pv.shell_axes.length_scale_major", ["pv","shell_axes","length_scale_major"]),
                ("pv.shell_axes.length_scale_minor", ["pv","shell_axes","length_scale_minor"]),
                ("pv.shell_axes.slit_width_pix", ["pv","shell_axes","slit_width_pix"]),
                ("pv.shell_axes.pos_step_pix", ["pv","shell_axes","pos_step_pix"]),
                ("pv.shell_axes.name_prefix", ["pv","shell_axes","name_prefix"]),
                # negatives
                ("pv.negatives.enabled", ["pv","negatives","enabled"]),
                ("pv.negatives.n_per_shell", ["pv","negatives","n_per_shell"]),
                ("pv.negatives.galaxy_mask_percentile", ["pv","negatives","galaxy_mask_percentile"]),
                ("pv.negatives.empty_percentile", ["pv","negatives","empty_percentile"]),
                ("pv.negatives.min_sep_from_shell_arcsec", ["pv","negatives","min_sep_from_shell_arcsec"]),
                ("pv.negatives.length_arcsec", ["pv","negatives","length_arcsec"]),
                ("pv.negatives.orientation", ["pv","negatives","orientation"]),
                ("pv.negatives.slit_width_pix", ["pv","negatives","slit_width_pix"]),
                ("pv.negatives.pos_step_pix", ["pv","negatives","pos_step_pix"]),
                ("pv.negatives.seed", ["pv","negatives","seed"]),
                # legacy presence (to see if purge worked)
                ("pv.grid (present?)", ["pv","grid"]),
                ("pv.axes (present?)", ["pv","axes"]),
                ("pv.mode (present?)", ["pv","mode"]),
                ("pv.majmin (present?)", ["pv","majmin"]),
            ]
            for label, path in keys_to_show:
                b = deep_get(before, path, None)
                a = deep_get(cfg, path, None)
                if b != a:
                    print(f"    - {label}: {b} -> {a}")
            # Also print migration notes, if any
            for lab, b, a in migration_notes:
                print(f"    - {lab}: {b} -> {a}")
        else:
            # backup + write
            bak = yml.with_suffix(f".yaml.bak.{tstamp}")
            bak.write_text(yml.read_text(encoding="utf-8"), encoding="utf-8")
            dump_yaml(yml, cfg)
            changed += 1

    if not args.dry_run:
        print(f"Done. Updated {changed} file(s). Backups: '*.yaml.bak.{tstamp}' alongside originals.")


if __name__ == "__main__":
    main()