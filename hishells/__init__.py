"""HIShells: deep-learning HI shell detection on THINGS cubes.

The package follows the build order in ``plan.md`` §11. See
``plan.md`` for the methodology and ``README.md`` for usage.
"""

from __future__ import annotations

__version__ = "0.1.0"

# THINGS beam FWHM (arcsec). Used as the DBSCAN eps in
# ``hishells.candidates.enumerate_candidates`` (plan §1.3) and as the
# default beam scale wherever code needs an angular resolution prior.
# Walter et al. 2008 quote a natural-weighted FWHM of 11"-13" for most
# THINGS cubes; 6" is the robust-weighted value (and is what the plan
# explicitly cites). The value lives here so the rest of the package
# imports it from one place.
THINGS_BEAM_ARCSEC = 6.0
