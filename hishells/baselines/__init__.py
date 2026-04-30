"""Classical / off-the-shelf baselines for plan \u00a79 Rows 1-3.

* :mod:`hishells.baselines.trivial` -- flux-deficit threshold (\u00a79 Row 3).
* :mod:`hishells.baselines.mtb` -- Mashchenko-Thilker-Braun template
  matching (\u00a79 Row 1).
* :mod:`hishells.baselines.casi` -- subprocess wrapper around the
  public CASI-2D code (\u00a79 Row 2). Requires ``CASI_HOME`` to be set;
  raises with install instructions otherwise.
"""

# Submodules are imported on demand to avoid pulling in optional
# dependencies (e.g. ``casi`` requires the user to set ``CASI_HOME``).
__all__ = ["mtb", "trivial", "casi"]
