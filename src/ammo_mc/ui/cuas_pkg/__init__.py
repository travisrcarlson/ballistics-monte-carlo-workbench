"""Modular Counter-UAS & Multi-Barrel Small Arms UI package."""

from __future__ import annotations

from ammo_mc.ui.cuas_pkg.helpers import (
    _render_cuas_profile_sync_bar,
    _render_effective_range_tab,
    _render_stage1_hardware_cad_tab,
)
from ammo_mc.ui.cuas_pkg.stage1_array import render_stage1_small_arms
from ammo_mc.ui.cuas_pkg.stage2_engagement import render_stage2_engagement
from ammo_mc.ui.cuas_pkg.stage3_tactical import render_stage3_tactical
from ammo_mc.ui.cuas_pkg.stage4_adversarial import render_stage4_adversarial
from ammo_mc.ui.cuas_pkg.stage4_calibration import render_stage4_calibration
from ammo_mc.ui.cuas_pkg.stage4_trade_studies import render_stage4_trade_studies

__all__ = [
    "_render_cuas_profile_sync_bar",
    "_render_effective_range_tab",
    "_render_stage1_hardware_cad_tab",
    "render_stage1_small_arms",
    "render_stage2_engagement",
    "render_stage3_tactical",
    "render_stage4_adversarial",
    "render_stage4_calibration",
    "render_stage4_trade_studies",
]
