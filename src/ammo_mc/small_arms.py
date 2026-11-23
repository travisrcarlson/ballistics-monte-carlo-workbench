"""Backward-compatible facade re-exporting `small_arms_pkg`."""

from __future__ import annotations

from ammo_mc.small_arms_pkg import *  # noqa: F401,F403
from ammo_mc.small_arms_pkg import _convex_hull_2d  # noqa: F401
from ammo_mc.small_arms_pkg import __all__  # noqa: F401
