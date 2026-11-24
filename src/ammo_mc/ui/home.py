"""Task-first Quick-Start Home launcher workbench."""

from __future__ import annotations

import streamlit as st

from ammo_mc import __version__
from ammo_mc.data import get_kb
from ammo_mc.presets import list_presets
from ammo_mc.ui.common import (
    _navigate_to,
    _render_sidebar_header,
    is_expert_mode,
)


def tab_home() -> None:
    """Render a clean task-first Quick-Start launcher linking every engineering workflow."""
    kb = get_kb()
    presets = list_presets()
    expert = is_expert_mode()

    with st.sidebar:
        _render_sidebar_header(
            "Quick-Start Home",
            f"v{__version__} • {'Expert Mode' if expert else 'Standard Mode'}",
        )
        st.caption(
            "Choose a workflow card on the canvas or switch modes above. "
            "**Standard Mode** highlights the essential controls; toggle **Expert Mode** to expose full error budgets, "
            "FMEA toggles, aero overrides, and diagnostic plots."
        )
        st.button(
            "Open Multi-Shotgun C-UAS Array",
            type="primary",
            on_click=_navigate_to,
            args=("cuas", None, "array"),
            width="stretch",
            key="home_sb_cuas_btn",
        )
        st.button(
            "Open Monte Carlo Simulator",
            on_click=_navigate_to,
            args=("sim",),
            width="stretch",
            key="home_sb_sim_btn",
        )
        st.button(
            "Open Safety & IATG Tools",
            on_click=_navigate_to,
            args=("iatg",),
            width="stretch",
            key="home_sb_iatg_btn",
        )

    st.subheader("What would you like to analyze today?")
    st.caption(
        "Pick an engineering workflow below. All models run locally using the same underlying physics engine in both Standard and Expert modes."
    )

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        with st.container(border=True):
            st.markdown("### 1. Drone & Multi-Barrel Shotgun Defense")
            st.markdown(
                "Evaluate multi-shotgun turret layouts against quadcopters and fixed-wing UAVs, inspect 3D aspect-angle $P_{\\text{kill}}$, "
                "map elevated fallout safety zones (SDZ), and compare candidate architectures A/B."
            )
            b1, b2 = st.columns(2)
            b1.button(
                "Open Multi-Shotgun Array",
                type="primary",
                on_click=_navigate_to,
                args=("cuas", None, "array"),
                width="stretch",
                key="home_card_array",
            )
            b2.button(
                "Fallout Safety Zone (SDZ)",
                on_click=_navigate_to,
                args=("cuas", None, "sdz"),
                width="stretch",
                key="home_card_sdz",
            )
            b3, b4 = st.columns(2)
            b3.button(
                "Closed-Loop FCS & Evasion",
                on_click=_navigate_to,
                args=("cuas", None, "fcs"),
                width="stretch",
                key="home_card_fcs",
            )
            b4.button(
                "A/B Turret Comparator",
                on_click=_navigate_to,
                args=("cuas", None, "comparator"),
                width="stretch",
                key="home_card_comp",
            )
            b5, b6 = st.columns(2)
            b5.button(
                "Unified 3D Tactical Raid",
                on_click=_navigate_to,
                args=("cuas", None, "raid"),
                width="stretch",
                key="home_card_raid",
            )
            b6.button(
                "🤖 Adversarial ML Optimizer",
                on_click=_navigate_to,
                args=("cuas", None, "adversarial"),
                width="stretch",
                key="home_card_adversarial",
            )

    with row1_c2:
        with st.container(border=True):
            st.markdown("### 2. Ballistics, Dispersion & 3D Trajectories")
            st.markdown(
                "Simulate 6-stage stochastic flight (interior thermodynamics, 3-DoF / STANAG 4355 MPM, fuze & terminal effects), "
                "visualize 3D trajectory bundles over terrain, and generate tabular firing tables."
            )
            s1, s2 = st.columns(2)
            s1.button(
                "Open Monte Carlo Simulator",
                type="primary",
                on_click=_navigate_to,
                args=("sim",),
                width="stretch",
                key="home_card_sim",
            )
            s2.button(
                "3D Trajectory Lab",
                on_click=_navigate_to,
                args=("3d",),
                width="stretch",
                key="home_card_3d",
            )
            st.button(
                "Firing Tables & STANAG 4082 Met",
                on_click=_navigate_to,
                args=("firing",),
                width="stretch",
                key="home_card_ft",
            )

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        with st.container(border=True):
            st.markdown("### 3. Explosives Safety, Blast & Terminal Physics")
            st.markdown(
                "Compute IATG Quantity-Distance envelopes, Kingery–Bulmash blast & Mach-stem enhancement, masonry/glazing limits, "
                "concrete penetration, long-rod armor defeat, and McCoy MCDRAG aeroprediction."
            )
            p1, p2 = st.columns(2)
            p1.button(
                "Open Safety & IATG",
                type="primary",
                on_click=_navigate_to,
                args=("iatg",),
                width="stretch",
                key="home_card_iatg",
            )
            p2.button(
                "Physics & Terminal Labs",
                on_click=_navigate_to,
                args=("physics",),
                width="stretch",
                key="home_card_phys",
            )
            st.button(
                "MCDRAG & 3D Projectile CAD",
                on_click=_navigate_to,
                args=("mcdrag",),
                width="stretch",
                key="home_card_mcdrag",
            )

    with row2_c2:
        with st.container(border=True):
            st.markdown("### 4. Verification, Certificates & Sources")
            st.markdown(
                "Audit the engine against 22 workbook cases and 18 analytical physics benchmarks (40 total checks), "
                "export standalone HTML/Markdown audit certificates, and inspect all linked literature citations."
            )
            a1, a2 = st.columns(2)
            a1.button(
                "Open V&V Scorecard",
                type="primary",
                on_click=_navigate_to,
                args=("validation",),
                width="stretch",
                key="home_card_val",
            )
            a2.button(
                "Browse Scientific Sources",
                on_click=_navigate_to,
                args=("literature",),
                width="stretch",
                key="home_card_lit",
            )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Munition Presets", f"{len(presets)}", delta="Notional Reference Set")
    m2.metric("Governed Formulas", f"{len(kb.formulas)}", delta="Registry Verified")
    m3.metric("Primary Literature", f"{len(kb.sources)}", delta="100% Linked Provenance")
    m4.metric("V&V Benchmarks", "40", delta="22 Workbook + 18 Physics")
