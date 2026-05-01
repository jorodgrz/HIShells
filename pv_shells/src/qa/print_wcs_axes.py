#!/usr/bin/env python3
import json
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from src.utils.io import load_yaml
from src.utils.wcs_tools import _ctype_list

def main():
    cfg = load_yaml("data/_resolved_config.yaml") if Path("data/_resolved_config.yaml").exists() else load_yaml("pv_config.yaml")
    hdr = fits.getheader(cfg["cube_path"])
    w = WCS(hdr)
    info = {
        "NAXIS": hdr.get("NAXIS"),
        "CTYPES": _ctype_list(hdr),
        "CUNITS": [str(hdr.get(f"CUNIT{i}", "")) for i in range(1, hdr.get("NAXIS")+1)],
        "CRVAL":  [hdr.get(f"CRVAL{i}") for i in range(1, hdr.get("NAXIS")+1)],
        "CDELT":  [hdr.get(f"CDELT{i}") for i in range(1, hdr.get("NAXIS")+1)],
        "CRPIX":  [hdr.get(f"CRPIX{i}") for i in range(1, hdr.get("NAXIS")+1)],
        "world_axis_types": w.world_axis_physical_types,
    }
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()