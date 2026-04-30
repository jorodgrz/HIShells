"""Download THINGS HI data products from the MPIA mirror.

The page at https://www2.mpia-hd.mpg.de/THINGS/Data.html is a flat HTML
index of 264 FITS files: 33 galaxies x 2 weightings (NA = natural,
RO = robust) x 4 products (CUBE, MOM0, MOM1, MOM2). This script scrapes
that index and downloads the requested subset with a per-file progress
bar, HTTP-range resume across runs, and a galaxy filter for the
20 galaxies that appear in the Bagetakos+2011 hole catalog
(Data/J_AJ_141_23/) -- i.e. the ones we have labels for.

Examples
--------
    # All 33 NA cubes (~30 GB), into Data/THINGS/
    python scripts/fetch_things.py

    # Only the 19 catalog galaxies' NA cubes (~20 GB)
    python scripts/fetch_things.py --catalog-only

    # Moment-0 maps for a fast sanity check (~tens of MB total)
    python scripts/fetch_things.py --product MOM0

    # Robust-weighted cubes instead of natural
    python scripts/fetch_things.py --weighting RO

    # See what would be downloaded without fetching anything
    python scripts/fetch_things.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm.auto import tqdm

BASE = "https://www2.mpia-hd.mpg.de/THINGS"
INDEX_URL = f"{BASE}/Data.html"

# Galaxies that appear in Bagetakos+2011 (table2.dat in J_AJ_141_23). These
# are the 20 galaxies we have ground-truth HI hole labels for. Names match
# the THINGS filename convention (underscored, no spaces, "Holmberg I/II"
# -> "HO_I"/"HO_II"). IC_2574 is in the catalog but is NOT served from the
# MPIA public mirror, so list_galaxies() will not return it; main() warns.
CATALOG_20 = frozenset(
    {
        "NGC_628",
        "NGC_2366",
        "NGC_2403",
        "HO_II",
        "DDO53",
        "NGC_2841",
        "HO_I",
        "NGC_2976",
        "NGC_3031",
        "NGC_3184",
        "IC_2574",
        "NGC_3521",
        "NGC_3627",
        "NGC_4214",
        "NGC_4449",
        "NGC_4736",
        "DDO154",
        "NGC_5194",
        "NGC_6946",
        "NGC_7793",
    }
)

WEIGHTINGS = ("NA", "RO")
PRODUCTS = ("CUBE", "MOM0", "MOM1", "MOM2")
USER_AGENT = "HIShells-fetch_things/1.0 (+https://github.com/)"
CHUNK = 1 << 20  # 1 MiB


def list_galaxies(weighting: str, product: str, *, timeout: float = 30.0) -> list[str]:
    """Scrape the THINGS index and return galaxy stems for one (weighting, product)."""
    r = requests.get(INDEX_URL, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    pat = re.compile(
        rf"Data_files/([A-Z0-9_]+)_{weighting}_{product}_THINGS\.FITS"
    )
    return sorted(set(pat.findall(r.text)))


def remote_size(url: str, *, timeout: float = 30.0) -> int | None:
    """Return Content-Length for ``url``, or None if the server doesn't expose it."""
    r = requests.head(
        url, headers={"User-Agent": USER_AGENT}, allow_redirects=True, timeout=timeout
    )
    if not r.ok:
        return None
    cl = r.headers.get("Content-Length")
    return int(cl) if cl is not None else None


def download(url: str, dst: Path, *, timeout: float = 60.0) -> None:
    """Stream ``url`` to ``dst`` with a tqdm bar; resume if a partial file exists."""
    headers = {"User-Agent": USER_AGENT}
    have = dst.stat().st_size if dst.exists() else 0
    mode = "wb"

    # If the local file is already the full remote size, skip without reopening.
    total_remote = remote_size(url)
    if total_remote is not None and have == total_remote:
        tqdm.write(f"== {dst.name}  already complete ({have:,} B), skipping")
        return
    if have:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        # 416 = the byte range is past EOF, i.e. we already have all of it.
        if r.status_code == 416:
            tqdm.write(f"== {dst.name}  already complete, skipping")
            return
        r.raise_for_status()

        # Total bytes is the part-we-have plus whatever this response will deliver.
        delivered = int(r.headers.get("Content-Length", 0))
        total = have + delivered if delivered else total_remote

        with open(dst, mode) as f, tqdm(
            total=total,
            initial=have,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dst.name,
            leave=False,
        ) as bar:
            for buf in r.iter_content(CHUNK):
                if not buf:
                    continue
                f.write(buf)
                bar.update(len(buf))


def build_targets(
    galaxies: list[str], weighting: str, product: str
) -> list[tuple[str, str]]:
    """Turn galaxy stems into (filename, url) pairs."""
    out = []
    for g in galaxies:
        fn = f"{g}_{weighting}_{product}_THINGS.FITS"
        out.append((fn, f"{BASE}/Data_files/{fn}"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download THINGS HI data products from MPIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("Data/THINGS"),
        help="Output directory (created if missing).",
    )
    ap.add_argument(
        "--weighting",
        choices=WEIGHTINGS,
        default="NA",
        help="NA = natural-weighted (matches Bagetakos+2011), RO = robust-weighted.",
    )
    ap.add_argument(
        "--product",
        choices=PRODUCTS,
        default="CUBE",
        help="CUBE = full p-p-v cube; MOM0/1/2 = collapsed moment maps.",
    )
    ap.add_argument(
        "--catalog-only",
        action="store_true",
        help="Restrict to the 20 galaxies that appear in Bagetakos+2011.",
    )
    ap.add_argument(
        "--galaxies",
        nargs="*",
        metavar="STEM",
        help="Explicit galaxy stems (e.g. NGC_2403 DDO154). Overrides --catalog-only.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the URLs that would be fetched without downloading them.",
    )
    args = ap.parse_args(argv)

    available = list_galaxies(args.weighting, args.product)
    if args.galaxies:
        wanted = set(args.galaxies)
        unknown = wanted - set(available)
        if unknown:
            print(
                f"!! unknown galaxy stems for {args.weighting}/{args.product}: "
                f"{sorted(unknown)}",
                file=sys.stderr,
            )
        galaxies = [g for g in available if g in wanted]
    elif args.catalog_only:
        galaxies = [g for g in available if g in CATALOG_20]
        missing = CATALOG_20 - set(available)
        if missing:
            print(
                f"!! catalog galaxies missing from THINGS public release: "
                f"{sorted(missing)}",
                file=sys.stderr,
            )
    else:
        galaxies = available

    targets = build_targets(galaxies, args.weighting, args.product)
    print(
        f"-- {len(targets)} {args.weighting}/{args.product} files "
        f"-> {args.out.resolve()}"
    )

    if args.dry_run:
        for _, url in targets:
            print(url)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for fn, url in tqdm(targets, desc="overall", unit="file"):
        try:
            download(url, args.out / fn)
        except (requests.RequestException, OSError) as exc:
            tqdm.write(f"!! {fn}: {exc}")
            failures.append((fn, str(exc)))

    if failures:
        print(f"\nDone with {len(failures)} failure(s):", file=sys.stderr)
        for fn, msg in failures:
            print(f"  {fn}: {msg}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
