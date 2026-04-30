"""Download THINGS HI data products from the MPIA mirror via aria2c.

The page at https://www2.mpia-hd.mpg.de/THINGS/Data.html is a flat HTML
index of 264 FITS files: 33 galaxies x 2 weightings (NA/RO) x 4 products
(CUBE, MOM0, MOM1, MOM2). This script scrapes that index and hands the
resulting URL list to ``aria2c`` for the actual transfer.

Why aria2c? MPIA caps per-TCP-connection throughput (~100 kB/s from the
US). aria2c opens several parallel connections per file and downloads a
few files concurrently, which in practice gets 10-20x aggregate speed
while still being a polite citizen to an academic mirror.

aria2c must be on PATH. Install via conda (preferred, pinned by
environment.yml):

    conda install -c conda-forge aria2

or Homebrew:

    brew install aria2

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
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://www2.mpia-hd.mpg.de/THINGS"
INDEX_URL = f"{BASE}/Data.html"

# Galaxies that appear in Bagetakos+2011 (table2.dat in J_AJ_141_23). These
# are the 20 galaxies we have ground-truth HI hole labels for. IC_2574 is
# in the catalog but is NOT served from the MPIA public mirror, so
# list_galaxies() will not return it; main() warns.
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
USER_AGENT = "HIShells-fetch_things/2.0 (+https://github.com/)"


def list_galaxies(weighting: str, product: str, *, timeout: float = 30.0) -> list[str]:
    """Scrape the THINGS index and return galaxy stems for one (weighting, product)."""
    req = Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="replace")
    pat = re.compile(
        rf"Data_files/([A-Z0-9_]+)_{weighting}_{product}_THINGS\.FITS"
    )
    return sorted(set(pat.findall(html)))


def build_urls(galaxies: list[str], weighting: str, product: str) -> list[str]:
    """Turn galaxy stems into fully-qualified MPIA URLs."""
    return [
        f"{BASE}/Data_files/{g}_{weighting}_{product}_THINGS.FITS"
        for g in galaxies
    ]


def run_aria2(
    urls: list[str],
    out_dir: Path,
    *,
    connections_per_file: int,
    concurrent_files: int,
) -> int:
    """Invoke aria2c on ``urls``; return its exit code.

    URLs are passed on stdin so we don't have to manage a temp file.
    ``--continue=true`` makes partial files resume across runs, and
    ``--allow-overwrite=false`` protects completed files from being
    re-fetched if you re-run the script.
    """
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        print(
            "!! aria2c not found on PATH. Install with "
            "`conda install -c conda-forge aria2` or `brew install aria2`.",
            file=sys.stderr,
        )
        return 127

    cmd = [
        aria2,
        "--input-file=-",
        f"--dir={out_dir}",
        f"--max-connection-per-server={connections_per_file}",
        f"--split={connections_per_file}",
        "--min-split-size=1M",
        f"--max-concurrent-downloads={concurrent_files}",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--file-allocation=none",
        "--summary-interval=10",
        "--console-log-level=warn",
        f"--user-agent={USER_AGENT}",
    ]
    proc = subprocess.run(cmd, input="\n".join(urls).encode(), check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download THINGS HI data products from MPIA via aria2c.",
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
        "--connections",
        type=int,
        default=8,
        help="aria2 parallel connections per file (-x / --split).",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="aria2 files downloaded concurrently (-j).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the URLs that would be fetched without invoking aria2c.",
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

    urls = build_urls(galaxies, args.weighting, args.product)
    print(
        f"-- {len(urls)} {args.weighting}/{args.product} files "
        f"-> {args.out.resolve()}"
    )

    if args.dry_run:
        for u in urls:
            print(u)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    return run_aria2(
        urls,
        args.out,
        connections_per_file=args.connections,
        concurrent_files=args.jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
