"""Backward-compatible facade re-exporting `vis_3d_pkg` and C-UAS visualizers."""

from __future__ import annotations

from ammo_mc.vis_3d_pkg import *  # noqa: F401,F403
from ammo_mc.vis_3d_pkg import _compute_terrain_z  # noqa: F401
from ammo_mc.vis_3d_pkg import __all__ as _VIS_3D_PKG_ALL

from ammo_mc.vis_cuas import (  # noqa: F401
    create_2d_cuas_fallout_safety_map_figure,
    create_3d_cuas_fallout_footprint_figure,
    create_3d_swarm_hemisphere_figure,
    create_airburst_standoff_cone_figure,
    create_antidrone_optimizer_pareto_figure,
    create_cuas_payload_range_comparison_figure,
    create_fcs_tracking_and_evasion_figure,
    create_swarm_doctrine_tradeoff_figure,
    create_turret_recoil_dynamics_figure,
)
from ammo_mc.vis_cuas_mission import (  # noqa: F401
    create_3d_animated_intercept_figure,
    create_kill_chain_timeline_figure,
    create_logistics_thermal_and_cost_figure,
    create_multiturret_site_map_figure,
    create_terrain_dem_site_3d_figure,
)
from ammo_mc.vis_cuas_analytics import (  # noqa: F401
    create_ab_comparator_radar_and_bars_figure,
    create_layered_defense_figure,
    create_real_map_site_defense_figure,
    create_sensitivity_tornado_figure,
)

__all__ = list(_VIS_3D_PKG_ALL) + [
    "create_2d_cuas_fallout_safety_map_figure",
    "create_3d_cuas_fallout_footprint_figure",
    "create_3d_swarm_hemisphere_figure",
    "create_airburst_standoff_cone_figure",
    "create_antidrone_optimizer_pareto_figure",
    "create_cuas_payload_range_comparison_figure",
    "create_fcs_tracking_and_evasion_figure",
    "create_swarm_doctrine_tradeoff_figure",
    "create_turret_recoil_dynamics_figure",
    "create_3d_animated_intercept_figure",
    "create_kill_chain_timeline_figure",
    "create_logistics_thermal_and_cost_figure",
    "create_multiturret_site_map_figure",
    "create_terrain_dem_site_3d_figure",
    "create_ab_comparator_radar_and_bars_figure",
    "create_layered_defense_figure",
    "create_real_map_site_defense_figure",
    "create_sensitivity_tornado_figure",
]
