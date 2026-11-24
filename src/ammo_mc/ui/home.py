"""Task-first Quick-Start Home launcher workbench."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import streamlit as st

from ammo_mc import __version__
from ammo_mc.briefing_export import generate_cuas_array_spec_sheet_pdf
from ammo_mc.cuas_hardware import MOUNT_CLASS_SPECS, export_muzzle_plate_dxf
from ammo_mc.data import get_kb
from ammo_mc.presets import list_presets
from ammo_mc.small_arms import generate_multishotgun_array_layout, get_drone_target_specs, simulate_anti_drone_multishotgun_mc
from ammo_mc.ui.common import (
    _cached_turret_sizing_wizard,
    _navigate_to,
    _render_sidebar_header,
    is_expert_mode,
)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_wizard_exports(
    cfg_json: str,
    drone_model: str,
    speed_mps: float,
    range_m: float,
) -> tuple[bytes, str]:
    cfg = json.loads(cfg_json)
    sim = simulate_anti_drone_multishotgun_mc(
        layout_type=str(cfg.get("layout_type", "circular")),
        n_barrels=int(cfg.get("n_barrels", 6)),
        pointing_mode=str(cfg.get("pointing_mode", "convergent")),
        splay_angle_mrad=float(cfg.get("splay_angle_mrad", 0.0)),
        firing_mode=str(cfg.get("firing_mode", "balanced_pair_ripple")),
        ripple_delay_ms=float(cfg.get("ripple_delay_ms", 8.0)),
        ammunition_preset=str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")),
        default_choke=str(cfg.get("choke", "Modified")),
        target_distance_m=float(range_m),
        drone_model=drone_model,
        drone_velocity_mps=float(speed_mps),
        n_volleys=12,
        seed=42,
        inner_ammunition_preset=cfg.get("inner_ammunition_preset") or None,
        inner_choke=cfg.get("inner_choke") or None,
    )
    pdf_bytes = generate_cuas_array_spec_sheet_pdf(
        array_config=cfg,
        sim_result=sim,
        scenario_name=f"Quick-Size Spec: {cfg.get('name', 'C-UAS Turret')}",
    )
    barrels = generate_multishotgun_array_layout(
        layout_type=str(cfg.get("layout_type", "circular")),
        n_barrels=int(cfg.get("n_barrels", 6)),
        mount_radius_m=0.20,
        default_choke=str(cfg.get("choke", "Modified")),
        default_ammunition_preset=str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")),
    )
    dxf_str = export_muzzle_plate_dxf(
        barrels=barrels,
        layout_type=str(cfg.get("layout_type", "circular")),
        title=str(cfg.get("name", "C-UAS Muzzle Plate")),
        splay_angle_mrad=float(cfg.get("splay_angle_mrad", 0.0)),
        pointing_mode=str(cfg.get("pointing_mode", "convergent")),
    )
    return pdf_bytes, dxf_str


def _load_wizard_pick_to_stage1(cfg: dict[str, Any], drone_model: str, speed_mps: float, range_m: float) -> None:
    st.session_state["cuas_layout"] = str(cfg.get("layout_type", "circular"))
    st.session_state["cuas_n_barrels"] = int(cfg.get("n_barrels", 6))
    st.session_state["cuas_ammo"] = str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))
    st.session_state["cuas_choke"] = str(cfg.get("choke", "Modified"))
    st.session_state["cuas_pointing"] = str(cfg.get("pointing_mode", "convergent"))
    st.session_state["cuas_splay"] = float(cfg.get("splay_angle_deg", round(math.degrees(float(cfg.get("splay_angle_mrad", 0.0)) / 1000.0), 2)))
    st.session_state["cuas_fire_mode"] = str(cfg.get("firing_mode", "balanced_pair_ripple"))
    st.session_state["cuas_ripple_dt"] = float(cfg.get("ripple_delay_ms", 8.0))
    st.session_state["_active_cuas_profile"] = {
        "label": str(cfg.get("name", "Wizard Turret")),
        "layout_type": str(cfg.get("layout_type", "circular")),
        "n_barrels": int(cfg.get("n_barrels", 6)),
        "ammunition_preset": str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")),
        "choke": str(cfg.get("choke", "Modified")),
        "pointing_mode": str(cfg.get("pointing_mode", "convergent")),
        "splay_angle_mrad": float(cfg.get("splay_angle_mrad", 0.0)),
        "firing_mode": str(cfg.get("firing_mode", "balanced_pair_ripple")),
        "ripple_delay_ms": float(cfg.get("ripple_delay_ms", 8.0)),
        "drone_model": drone_model,
        "target_distance_m": float(range_m),
        "drone_speed_mps": float(speed_mps),
        "hybrid_mode": bool(cfg.get("hybrid_mode", False)),
        "inner_ammunition_preset": str(cfg.get("inner_ammunition_preset", "")),
        "inner_choke": str(cfg.get("inner_choke", "")),
    }
    _navigate_to("cuas", None, "array")


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
        "Pick an engineering workflow below or use the 3-Click Turret Sizing Wizard to size a C-UAS platform in seconds."
    )

    drone_specs = get_drone_target_specs()
    with st.container(border=True):
        st.markdown("### 🧭 3-Click C-UAS Turret Sizing Wizard")
        wc1, wc2, wc3, wc4 = st.columns([1.25, 0.9, 1.0, 1.15])
        with wc1:
            wiz_drone = st.selectbox(
                "1. Target Drone Threat",
                list(drone_specs.keys()),
                format_func=lambda k: drone_specs[k]["name"],
                index=0,
                key="home_wiz_drone",
            )
        with wc2:
            wiz_speed = st.slider("Threat Speed (m/s)", 12.0, 45.0, 25.0, 1.0, key="home_wiz_spd")
        with wc3:
            wiz_range = st.slider("2. Required Range (m)", 20.0, 75.0, 45.0, 5.0, key="home_wiz_rng")
        with wc4:
            wiz_mount = st.selectbox(
                "3. Platform / Mount Limit",
                list(MOUNT_CLASS_SPECS.keys()),
                format_func=lambda k: MOUNT_CLASS_SPECS[k]["name"],
                index=1,
                key="home_wiz_mount",
            )

        wiz_res = _cached_turret_sizing_wizard(
            drone_model=str(wiz_drone),
            threat_speed_mps=float(wiz_speed),
            required_range_m=float(wiz_range),
            mount_class=str(wiz_mount),
            budget_priority="balanced",
        )
        top_3 = wiz_res.get("top_3", [])
        if not wiz_res.get("has_feasible", False):
            st.warning(
                f"No archetype met strict {wiz_res['constraints']['mount_name']} weight/recoil limits — showing closest options."
            )
        if top_3:
            wiz_df = pd.DataFrame(
                [
                    {
                        "Rank": f"#{i+1}",
                        "Recommended Turret": r["name"],
                        f"Pk @ {wiz_range:.0f}m (%)": r["pk_at_range_pct"],
                        "R80 / R50 (m)": f"{r['r80_m']:.0f} / {r['r50_m']:.0f}",
                        "Total Mass (kg)": r["mass_kg"],
                        "Recoil (kN)": r["peak_recoil_kn"],
                        "$/Kill": f"${r['cost_per_kill_usd']:.2f}",
                        "Fits Mount": "Yes" if r["mount_feasible"] else "Over Limit",
                    }
                    for i, r in enumerate(top_3)
                ]
            )
            st.dataframe(wiz_df, width="stretch", hide_index=True)
            best_cfg = top_3[0]["config"]
            pdf_b, dxf_s = _cached_wizard_exports(
                json.dumps(best_cfg, sort_keys=True),
                str(wiz_drone),
                float(wiz_speed),
                float(wiz_range),
            )
            wb1, wb2, wb3 = st.columns(3)
            wb1.button(
                f"⚡ Load #1 ({best_cfg['n_barrels']}B {best_cfg['layout_type'].title()}) into Stage 1",
                type="primary",
                on_click=_load_wizard_pick_to_stage1,
                args=(best_cfg, str(wiz_drone), float(wiz_speed), float(wiz_range)),
                width="stretch",
                key="home_wiz_load_btn",
            )
            wb2.download_button(
                "📑 Download #1 Spec Sheet (.pdf)",
                data=pdf_b,
                file_name="cuas_wizard_top1_spec.pdf",
                mime="application/pdf",
                width="stretch",
                key="home_wiz_pdf_btn",
            )
            wb3.download_button(
                "📐 Download #1 Muzzle CAD (.dxf)",
                data=dxf_s.encode("utf-8"),
                file_name="cuas_wizard_muzzle_plate.dxf",
                mime="application/dxf",
                width="stretch",
                key="home_wiz_dxf_btn",
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
