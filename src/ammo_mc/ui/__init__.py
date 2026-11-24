"""Modular Streamlit UI workbench package for ammo_mc."""

from ammo_mc.ui.common import (
    _AUDIT_SUBVIEWS,
    _CUAS_STAGES,
    _CUAS_STAGE_TOOLS,
    _PHYSICS_SUBVIEWS,
    _SIM_SUBVIEWS,
    _WORKBENCH_OPTIONS,
    _navigate_to,
    init_from_query_params,
    is_expert_mode,
    render_focus_mode_toggle,
    render_header,
    require_password,
    sync_query_params,
)
from ammo_mc.ui.cuas import tab_cuas_and_small_arms
from ammo_mc.ui.firing_tables import tab_firing_tables
from ammo_mc.ui.home import tab_home
from ammo_mc.ui.iatg import tab_iatg
from ammo_mc.ui.literature import tab_literature
from ammo_mc.ui.mcdrag import tab_mcdrag
from ammo_mc.ui.physics_labs import tab_calculators
from ammo_mc.ui.sim_3d import tab_3d_simulation
from ammo_mc.ui.simulator import tab_simulator
from ammo_mc.ui.validation import tab_validation

__all__ = [
    "_AUDIT_SUBVIEWS",
    "_CUAS_STAGES",
    "_CUAS_STAGE_TOOLS",
    "_PHYSICS_SUBVIEWS",
    "_SIM_SUBVIEWS",
    "_WORKBENCH_OPTIONS",
    "_navigate_to",
    "init_from_query_params",
    "is_expert_mode",
    "render_focus_mode_toggle",
    "render_header",
    "require_password",
    "sync_query_params",
    "tab_3d_simulation",
    "tab_calculators",
    "tab_cuas_and_small_arms",
    "tab_firing_tables",
    "tab_home",
    "tab_iatg",
    "tab_literature",
    "tab_mcdrag",
    "tab_simulator",
    "tab_validation",
]
