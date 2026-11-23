"""Interactive Streamlit Engineering Workbench entrypoint & router for ammo_mc."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ammo_mc.ui.common import (  # noqa: E402,F401
    _WORKBENCH_OPTIONS,
    _configure_page_and_styles,
    build_deeplink_query,
    init_from_query_params,
    render_header,
    require_password,
    sync_query_params,
)
from ammo_mc.ui.cuas import tab_cuas_and_small_arms  # noqa: E402
from ammo_mc.ui.firing_tables import tab_firing_tables  # noqa: E402
from ammo_mc.ui.iatg import tab_iatg  # noqa: E402
from ammo_mc.ui.mcdrag import tab_mcdrag  # noqa: E402
from ammo_mc.ui.physics_labs import tab_calculators  # noqa: E402
from ammo_mc.ui.sim_3d import tab_3d_simulation  # noqa: E402
from ammo_mc.ui.simulator import tab_simulator  # noqa: E402
from ammo_mc.ui.validation import tab_validation  # noqa: E402


def main() -> None:
    """Render the password-gated, contextual-sidebar engineering workbench."""
    _configure_page_and_styles()
    if not require_password():
        return
    init_from_query_params()
    render_header()

    with st.container(border=True):
        active_wb = st.radio(
            "Select Active Workbench",
            _WORKBENCH_OPTIONS,
            horizontal=True,
            label_visibility="collapsed",
            key="active_workbench",
        )

    if active_wb == "🎯 Monte Carlo Simulator":
        tab_simulator()
    elif active_wb == "🛸 C-UAS & Small Arms":
        tab_cuas_and_small_arms()
    elif active_wb == "🌐 3D Simulations":
        tab_3d_simulation()
    elif active_wb == "🚀 MCDRAG & 3D CAD":
        tab_mcdrag()
    elif active_wb == "🔬 Physics & Ballistics Labs":
        tab_calculators()
    elif active_wb == "🏛️ UNODA IATG Safety":
        tab_iatg()
    elif active_wb == "📋 Firing Tables & Met":
        tab_firing_tables()
    elif active_wb == "✅ Validation Suite":
        tab_validation()

    sync_query_params()

    with st.sidebar:
        st.divider()
        st.caption(f"🔗 Shareable view: `{build_deeplink_query()}`")
        if st.button("🔒 Lock Workbench", width="stretch", key="lock_workbench_btn"):
            st.session_state["authenticated"] = False
            st.rerun()


if __name__ == "__main__":
    main()
