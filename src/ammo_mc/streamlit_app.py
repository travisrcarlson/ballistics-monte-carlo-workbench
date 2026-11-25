"""Interactive Streamlit Engineering Workbench entrypoint & router for ammo_mc."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ammo_mc import __version__  # noqa: E402
from ammo_mc.ui.common import (  # noqa: E402,F401
    _AUDIT_SUBVIEWS,
    _PHYSICS_SUBVIEWS,
    _SIM_SUBVIEWS,
    _WORKBENCH_OPTIONS,
    _configure_page_and_styles,
    _navigate_to,
    _set_active_workbench,
    build_deeplink_query,
    init_from_query_params,
    render_digital_twin_sidebar_control,
    render_focus_mode_toggle,
    render_header,
    render_physics_fidelity_sidebar_control,
    require_password,
    sync_query_params,
)
from ammo_mc.ui.cuas import tab_cuas_and_small_arms  # noqa: E402
from ammo_mc.ui.firing_tables import tab_firing_tables  # noqa: E402
from ammo_mc.ui.home import tab_home  # noqa: E402
from ammo_mc.ui.iatg import tab_iatg  # noqa: E402
from ammo_mc.ui.literature import tab_literature  # noqa: E402
from ammo_mc.ui.mcdrag import tab_mcdrag  # noqa: E402
from ammo_mc.ui.physics_labs import tab_calculators  # noqa: E402
from ammo_mc.ui.sim_3d import tab_3d_simulation  # noqa: E402
from ammo_mc.ui.simulator import tab_simulator  # noqa: E402
from ammo_mc.ui.validation import tab_validation  # noqa: E402


def _render_subview_bar(options: list[str], key: str, label: str) -> str:
    """Render a compact horizontal sub-view selector inside a consolidated domain."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return str(
        st.radio(
            label,
            options,
            horizontal=True,
            label_visibility="collapsed",
            key=key,
        )
    )


def main() -> None:
    """Render the password-gated, focus-mode-aware engineering workbench."""
    _configure_page_and_styles()
    if not require_password():
        return
    init_from_query_params()
    render_header()

    with st.sidebar:
        render_focus_mode_toggle()
        render_digital_twin_sidebar_control()
        render_physics_fidelity_sidebar_control()
        st.divider()

    if st.session_state.get("active_workbench") not in _WORKBENCH_OPTIONS:
        st.session_state["active_workbench"] = _WORKBENCH_OPTIONS[0]

    with st.container(border=True):
        active_wb = st.radio(
            "Select Active Workbench",
            _WORKBENCH_OPTIONS,
            horizontal=True,
            label_visibility="collapsed",
            key="active_workbench",
        )

    if active_wb == "Home":
        tab_home()
    elif active_wb == "C-UAS & Multi-Barrel":
        tab_cuas_and_small_arms()
    elif active_wb == "Ballistics & 3D Sim":
        sub = _render_subview_bar(_SIM_SUBVIEWS, "sim_active_subview", "Simulation View")
        if sub == _SIM_SUBVIEWS[0]:
            tab_simulator()
        elif sub == _SIM_SUBVIEWS[1]:
            tab_3d_simulation()
        else:
            tab_firing_tables()
    elif active_wb == "Physics & Aero Labs":
        sub = _render_subview_bar(_PHYSICS_SUBVIEWS, "physics_active_subview", "Physics Lab View")
        if sub == _PHYSICS_SUBVIEWS[0]:
            tab_calculators()
        else:
            tab_mcdrag()
    elif active_wb == "Safety & IATG":
        tab_iatg()
    elif active_wb == "Audit & Sources":
        sub = _render_subview_bar(_AUDIT_SUBVIEWS, "audit_active_subview", "Audit & Sources View")
        if sub == _AUDIT_SUBVIEWS[0]:
            tab_validation()
        else:
            tab_literature()

    sync_query_params()

    with st.sidebar:
        st.divider()
        from ammo_mc.data import get_kb
        _kb_meta = get_kb()
        st.caption(f"ammo-mc v{__version__} • {len(_kb_meta.sources)} Sources • {len(_kb_meta.formulas)} Formulas")
        st.caption(f"🔗 Shareable view: `{build_deeplink_query()}`")
        if not (active_wb == "Audit & Sources" and st.session_state.get("audit_active_subview") == _AUDIT_SUBVIEWS[1]):
            st.button(
                "Browse Scientific Sources",
                on_click=_navigate_to,
                args=("literature",),
                width="stretch",
                key="sidebar_jump_literature_btn",
            )
        if st.button("🔒 Lock Workbench", width="stretch", key="lock_workbench_btn"):
            st.session_state["authenticated"] = False
            st.rerun()


if __name__ == "__main__":
    main()
