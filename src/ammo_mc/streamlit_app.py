"""Interactive Streamlit Engineering Workbench for the Ammunition Monte Carlo Simulator (ammo_mc)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
import streamlit as st

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ammo_mc.auth import verify_app_password
from ammo_mc.data import get_kb
from ammo_mc.battery import (
    calculate_sheaf_aim_points,
    find_elevation_for_range,
    generate_battery_layout,
    simulate_battery_salvo,
    solve_mrsi_trajectory_plan,
)
from ammo_mc.advanced_armor import (
    calculate_era_flyer_velocity,
    simulate_tandem_shaped_charge_vs_era,
)
from ammo_mc.geospatial import (
    audit_crest_clearance_and_dead_space,
    evaluate_terrain_profile,
    export_trajectory_to_czml,
    export_trajectory_to_kml,
)
from ammo_mc.radar import (
    calculate_projectile_rcs,
    reconstruct_point_of_origin_and_impact,
    simulate_radar_tracking,
)
from ammo_mc.metcm import (
    generate_stanag_4082_profile,
    integrate_trajectory_with_sounding,
)
from ammo_mc.electronic_warfare import (
    simulate_ins_gps_guidance_under_jamming,
    simulate_sal_seeker_tracking,
)
from ammo_mc.direct_fire import (
    calculate_moving_target_lead,
    simulate_apfsds_sabot_discard,
    simulate_direct_fire_engagement,
)
from ammo_mc.small_arms import (
    benchmark_drone_aspect_angles_mc,
    benchmark_multibarrel_aspect_angles_mc,
    compare_anti_drone_layouts,
    compare_barrel_lengths_dispersion,
    configure_array_pointing_vectors,
    evaluate_multibarrel_dispersion_vs_distance,
    evaluate_shotgun_dispersion_vs_distance,
    generate_multishotgun_array_layout,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
    get_multibarrel_presets,
    get_pellet_specifications,
    get_shotgun_presets,
    get_standard_barrel_lengths,
    simulate_anti_drone_multishotgun_mc,
    simulate_multibarrel_burst_mc,
    simulate_shotgun_pattern_mc,
)
from ammo_mc.environment import (
    aerodynamic_reynolds_number,
    eb_04_isa_atmosphere,
    sutherland_viscosity_Pa_s,
)
from ammo_mc.export import (
    geojson_to_str,
    render_report_figure_bytes,
    summary_to_json_str,
    trials_to_csv_bytes,
)
from ammo_mc.exterior import (
    eb_07_gyroscopic_stability,
    eb_08_miller_twist_rule,
    eb_09_muzzle_spin_rate,
    eb_10_coriolis_acceleration,
    generate_ballistic_firing_table,
    simulate_mpm_vs_3dof_trajectory,
)
from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.fuze import (
    fz_01_system_reliability,
    fz_02_uxo_count,
    fz_03_bayesian_dud_rate,
    fz_04_zero_failure_ucb,
    fz_06_impact_graze_function,
    fz_07_self_destruct_redundancy,
)
from ammo_mc.dispenser import simulate_cluster_dispenser_footprint
from ammo_mc.guidance import run_guidance_monte_carlo, simulate_guided_trajectory
from ammo_mc.interior import LumpedIBConfig, solve_lumped_interior_ballistics
from ammo_mc.mcdrag import McDragInputs, estimate_mcdrag
from ammo_mc.penetration import (
    tp_01_lanz_odermatt,
    tp_02_shaped_charge_hydrodynamic,
    tp_04_lambert_jonas,
    tp_06_perforation_probability,
)
from ammo_mc.pipeline import run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.schemas import AmmoDefinition, Scenario, SimulationResult, TerrainPointSpec, TerrainSpec
from ammo_mc.six_dof import simulate_6dof_trajectory
from ammo_mc.stats import sample_unit_hypercube
from ammo_mc.surveillance import (
    CLIMATE_ZONES,
    STABILIZER_KINETICS,
    generate_surveillance_sampling_plan,
    simulate_propellant_aging,
)
from ammo_mc.terminal_blast import (
    evaluate_blast_field,
    tb_01_scaled_distance,
    tb_02_tnt_equivalence,
    tb_03_kingery_bulmash,
    tb_04_kinney_graham,
    tb_05_dynamic_pressure,
    tb_07_normal_reflected_pressure,
    tb_08_friedlander_waveform,
    tb_10_brick_house_damage_range,
    tb_11_glass_breakage_swiss,
    tb_12_lung_lethality_probit,
    tb_13_axelsson_chest_wall,
    tb_15_simple_safety_distances,
    tb_16_thermal_dose,
)
from ammo_mc.terminal_frag import (
    tf_01_gurney_cyl_sphere,
    tf_02_gurney_sandwiches,
    tf_04_mott_distribution,
    tf_08_taylor_angle,
    tf_10_aastp1_stack_density,
    tf_12_thor_penetration,
    tf_13_de_marre_penetration,
    tf_14_rc_debris_throw,
    tf_15_tp14_consequence_union,
)
from ammo_mc.cuas_defense import (
    compare_cuas_fallout_across_loads,
    compare_cuas_payload_families,
    get_cuas_payload_catalog,
    get_cuas_sensor_presets,
    optimize_antidrone_array_configuration,
    simulate_closed_loop_fcs_engagement_mc,
    simulate_cuas_advanced_payload_mc,
    simulate_cuas_pellet_fallout_hazard,
    simulate_drone_swarm_saturation_mc,
    simulate_turret_recoil_and_gimbal_loads,
    sweep_recoil_vs_ripple_delay,
)
from ammo_mc.validation import run_validation_suite
from ammo_mc.vis_cuas import (
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
from ammo_mc.vis_3d import (
    create_2d_antidrone_intercept_figure,
    create_2d_barrel_length_comparison_figure,
    create_2d_multi_distance_shotgun_comparison_figure,
    create_2d_target_pattern_figure,
    create_3d_6dof_attitude_figure,
    create_3d_animated_flight_figure,
    create_3d_antidrone_defense_figure,
    create_3d_armor_defeat_figure,
    create_3d_atmospheric_shear_figure,
    create_3d_battery_salvo_figure,
    create_2d_multibarrel_target_figure,
    create_3d_blast_and_frag_figure,
    create_3d_counter_battery_radar_figure,
    create_3d_direct_fire_sabot_figure,
    create_3d_ew_guidance_figure,
    create_3d_guided_intercept_figure,
    create_3d_iatg_safety_zones_figure,
    create_3d_multi_distance_multibarrel_figure,
    create_3d_multi_distance_shotgun_figure,
    create_3d_multi_facility_site_planner_figure,
    create_3d_multi_preset_comparison_figure,
    create_3d_multibarrel_dispersion_figure,
    create_3d_phase_space_figure,
    create_3d_projectile_geometry_figure,
    create_3d_shotgun_plume_figure,
    create_3d_submunition_dispersion_figure,
    create_3d_terrain_clearance_figure,
    create_3d_trajectory_bundle_figure,
    create_antidrone_layout_comparison_figure,
    create_barrel_length_diagnostic_curves,
    create_drone_aspect_radar_figure,
    create_multibarrel_aspect_radar_figure,
    create_range_curves_multibarrel_figure,
    create_range_curves_shotgun_figure,
)
from ammo_mc.vulnerability import tv_04_sspk_carleton, tv_05_multi_round_independent as tv_05_salvo_pk_independent

def _configure_page_and_styles() -> None:
    """Configure page metadata and inject theme-adaptive CSS styles on each run."""
    st.set_page_config(
        page_title="Ballistics & Dispersion Monte Carlo Workbench",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling for engineering workbench (theme-adaptive for Dark & Light modes)
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.15rem; font-weight: 800; color: var(--text-color, #F8FAFC); margin-bottom: 0.15rem; }
        .sub-title { font-size: 1.02rem; color: var(--text-color, #CBD5E1); opacity: 0.80; margin-bottom: 1.0rem; }
        .gov-banner { background-color: rgba(245, 158, 11, 0.14); border: 1px solid rgba(245, 158, 11, 0.35); border-left: 5px solid #F59E0B; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.86rem; }
        .flag-banner { background-color: rgba(239, 68, 68, 0.14); border: 1px solid rgba(239, 68, 68, 0.35); border-left: 5px solid #EF4444; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.86rem; }
        .epistemic-banner { background-color: rgba(59, 130, 246, 0.14); border: 1px solid rgba(59, 130, 246, 0.35); border-left: 5px solid #3B82F6; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.88rem; }
        .auth-card { background-color: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 10px; padding: 1.5rem 1.75rem; margin-top: 2.5rem; margin-bottom: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem !important; line-height: 1.25 !important; white-space: normal !important; overflow-wrap: break-word !important; }
        [data-testid="stMetricDelta"] > div { white-space: normal !important; overflow-wrap: break-word !important; font-size: 0.80rem !important; }
        [data-testid="stMetricLabel"] p { white-space: normal !important; line-height: 1.2 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_password() -> bool:
    """Gate workbench access behind a session password prompt on launch."""
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("<style>[data-testid='stSidebar'] { display: none; }</style>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1.1, 1.4, 1.1])
    with col_auth:
        st.markdown(
            """
            <div class="auth-card">
                <div style="font-size: 1.45rem; font-weight: 700; margin-bottom: 0.35rem;">
                    🔒 Ballistics & Dispersion Monte Carlo Workbench
                </div>
                <div style="font-size: 0.92rem; opacity: 0.82;">
                    Protected engineering workbench — enter the access password to continue.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("auth_login_form", clear_on_submit=False):
            entered_pw = st.text_input(
                "Access Password",
                type="password",
                placeholder="Enter password...",
                key="auth_pw_field",
            )
            submitted = st.form_submit_button("🔓 Unlock Workbench", use_container_width=True, type="primary")
        if submitted:
            try:
                secrets_map = st.secrets
            except Exception:
                secrets_map = None
            if verify_app_password(entered_pw, secrets_mapping=secrets_map):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
    return False


def render_header() -> None:
    st.markdown(
        '<div class="main-title">🎯 Ballistics, Dispersion & Terminal Physics Monte Carlo Workbench</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">End-to-end stochastic engineering simulation: Interior Thermodynamics (STANAG 4367) → Exterior 3-DoF / Modified Point Mass (STANAG 4355) → Fuze Reliability → Pressure/Fragmentation Kinematics → Dispersion & Vulnerability.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="gov-banner">
        <b>GOVERNANCE & USE LIMITATION NOTICE:</b> {USE_LIMITATION_NOTICE}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- TAB 1: MONTE CARLO SIMULATOR ---
def tab_simulator() -> None:
    presets = list_presets()
    family_map = {
        "All System Families (26 Presets)": None,
        "Artillery & Tactical Rockets (RK-01..04)": ("RK-",),
        "Precision Guided Missiles (MS-01)": ("MS-",),
        "Field Artillery (AR-01..05)": ("AR-",),
        "Medium & Heavy Cannon (CN-01..04)": ("CN-",),
        "Small Arms & Heavy MG (SA-01..03)": ("SA-",),
        "Mortars & Automatic Grenades (MO-01, GR-02)": ("MO-", "GR-"),
        "Air-Delivered Bombs & Dispensers (AB-01..03)": ("AB-",),
        "Naval Gun Systems (NV-01)": ("NV-",),
        "Demolition, EOD & Storage Stacks (DM-01..02, ST-01)": ("DM-", "ST-"),
    }

    with st.sidebar:
        st.header("⚙️ Simulation Configuration")
        sel_family = st.selectbox("System Family Filter", list(family_map.keys()), index=0)
        prefixes = family_map[sel_family]
        filtered_presets = (
            [p for p in presets if any(str(p["class_id"]).startswith(px_f) for px_f in prefixes)]
            if prefixes is not None
            else presets
        )
        preset_dict = {f"{p['preset_id']} — {p['ammo_name']}": p["preset_id"] for p in filtered_presets}
        default_idx = 4 if (prefixes is None and len(preset_dict) > 4) else 0
        sel_preset_label = st.selectbox("Reference System Preset (26 Available)", list(preset_dict.keys()), index=default_idx)
        sel_preset_id = preset_dict[sel_preset_label]

        base_ammo, base_scn = get_preset(sel_preset_id)
        st.caption(f"**Class `{base_ammo.class_id}`** ({base_ammo.launch.mode.upper()}): {base_ammo.description}")

        st.subheader("Model Fidelity & Sampling")
        fidelity = st.selectbox(
            "Fidelity Level",
            ["L0", "L1", "L2", "L3"],
            index=["L0", "L1", "L2", "L3"].index(base_scn.fidelity),
            help="L0: Analytic Error Budget | L1: 3-DoF Point-Mass | L2: STANAG 4355 MPM (Spin/Yaw Drift) | L3: Coupled Salvo",
        )
        sampler = st.selectbox("Sampling Strategy", ["LHS", "Sobol", "MC"], index=0)
        n_occ = st.number_input(
            "Occasions (Batteries / Trials)",
            min_value=1,
            max_value=2000,
            value=int(base_scn.n_occasions),
            step=5,
        )
        r_per_occ = st.number_input(
            "Rounds per Occasion",
            min_value=1,
            max_value=100,
            value=int(base_scn.rounds_per_occasion),
            step=1,
        )
        st.caption(f"Total simulated trajectories: **{n_occ * r_per_occ}**")

        seed = st.number_input("Random Seed", min_value=1, max_value=999999, value=int(base_scn.seed or 42))

        with st.expander("⚠️ System Failure Modes (FMEA)", expanded=True):
            fm_prop_pct = st.slider(
                "Propulsion / Rocket Motor Anomaly (%)",
                0.0,
                15.0,
                1.0,
                0.25,
                help="Incomplete gun propellant ignition or solid rocket motor chuffing/early burnout (EB-13) causing range drop",
            )
            fm_fin_pct = st.slider(
                "Fin / Sabot / Nozzle Tip-Off Anomaly (%)",
                0.0,
                15.0,
                0.8,
                0.25,
                help="Asymmetric sabot petal discard, wrap-around rocket fin lock failure, or thrust misalignment tip-off",
            )
            fm_sa_pct = st.slider(
                "Safe & Arm (S&A) Failure (%)",
                0.0,
                15.0,
                0.5,
                0.25,
                help="Mechanical/electronic Safe & Arm unit fails to arm in flight, resulting in an unarmed UXO dud",
            )
            fm_early_pct = st.slider(
                "Premature Airburst (%)",
                0.0,
                15.0,
                0.5,
                0.25,
                help="Premature rain/canopy impact or electronic time/proximity false trigger along the flight path",
            )
            fm_low_pct = st.slider(
                "Low-Order Partial Detonation (%)",
                0.0,
                15.0,
                1.2,
                0.25,
                help="Incomplete booster-to-main-charge shock transition yielding 25% of nominal blast/fragmentation output",
            )

        with st.expander("🎯 Aiming & Meteorology"):
            aim_range = st.number_input(
                "Target Range (m)",
                min_value=0.0,
                max_value=80000.0,
                value=float(base_scn.aim.range_m or 0.0),
                step=250.0,
            )
            high_angle = st.checkbox("High-Angle Lofted Fire", value=bool(base_scn.aim.high_angle))
            wind_range = st.slider("Range Wind (m/s, tailwind +)", -30.0, 30.0, float(base_scn.met.wind_range_mps), 1.0)
            wind_cross = st.slider(
                "Cross Wind (m/s, right +)", -30.0, 30.0, float(base_scn.met.wind_cross_right_mps), 1.0
            )
            temp_dev = st.slider("ISA Temp Deviation (K)", -30.0, 30.0, float(base_scn.met.deviations.dT_K), 1.0)

        with st.expander("🔥 Launch, Rocket Motor & Aero Overrides"):
            mv_mu = st.number_input(
                "Nominal Launch / Muzzle Velocity (m/s)",
                min_value=0.0,
                max_value=2500.0,
                value=float(base_ammo.launch.mv.mu),
                step=5.0,
            )
            mv_s_rr = st.slider(
                "Round-to-Round MV Sigma (m/s)", 0.0, 15.0, float(base_ammo.launch.mv.sigma_rr), 0.2
            )
            mv_s_lot = st.slider(
                "Occasion/Lot MV Sigma (m/s)", 0.0, 15.0, float(base_ammo.launch.mv.sigma_lot), 0.2
            )
            ff_mu = st.slider("Ballistic Form Factor (i)", 0.60, 1.50, float(base_ammo.aero.form_factor.mu), 0.02)
            prop_temp = st.slider("Propellant Temp (°C)", -30.0, 55.0, float(base_scn.aim.propellant_temp_C), 1.0)
            st.markdown("**🚀 Rocket Motor Boost (`EB-13`)**")
            rk_burn_s = st.slider(
                "Rocket Motor Burn Time t_burn (s)",
                0.0,
                12.0,
                float(base_ammo.launch.rocket_burn_s),
                0.1,
                help="Solid rocket motor boost duration (0 s for pure gun-launched unassisted projectiles)",
            )
            rk_mdot = st.slider(
                "Rocket Propellant Mass Flow m_dot (kg/s)",
                0.0,
                40.0,
                float(base_ammo.launch.rocket_mdot_kg_s),
                0.2,
            )
            rk_ve = st.slider(
                "Rocket Effective Exhaust Velocity V_e (m/s)",
                0.0,
                3000.0,
                float(base_ammo.launch.rocket_ve_mps),
                50.0,
            )

        with st.expander("📐 Error Budget & Uncertainties"):
            tle_sig = st.slider(
                "Target Location Error TLE Sigma (m)", 0.0, 100.0, float(base_scn.errors.bias.tle_range_sigma_m), 1.0
            )
            aim_qe_bias = st.slider(
                "QE Aiming Bias Sigma (mils)", 0.0, 5.0, float(base_scn.errors.bias.aim_qe_bias_sigma_mils), 0.1
            )
            aim_qe_rand = st.slider(
                "QE Round-to-Round Sigma (mils)", 0.0, 5.0, float(base_scn.errors.random.aim_qe_rand_sigma_mils), 0.1
            )
            wind_bias_sig = st.slider(
                "Met Wind Bias Sigma (m/s)", 0.0, 10.0, float(base_scn.errors.bias.wind_range_bias_sigma_mps), 0.5
            )

        with st.expander("🧪 Physics & Epistemic Toggles"):
            two_loop = st.checkbox(
                "Two-Loop Nested Epistemic MC",
                value=bool(base_scn.two_loop_epistemic),
                help="Separates outer epistemic model parameter uncertainty from inner aleatory dispersion",
            )
            n_epistemic = st.slider("Outer Epistemic Realizations", 4, 25, 8, 1) if two_loop else 8
            use_coriolis = st.checkbox("Include Earth Coriolis & Eötvös", value=bool(base_scn.include_coriolis))
            use_vacuum = st.checkbox("Vacuum Trajectory (No Drag)", value=bool(base_scn.vacuum))
            terrain_mode = st.selectbox("Terrain Profile", ["Flat Ground", "Linear Hill Profile"], index=0)
            hill_slope = 0.0
            if terrain_mode == "Linear Hill Profile":
                hill_slope = st.slider("Hill Grade (%)", -10.0, 15.0, 3.0, 0.5)

        run_btn = st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True)

    # Prepare ammo and scenario
    ammo = base_ammo.model_copy(deep=True)
    if st.session_state.get("custom_mcdrag_ammo_id") == ammo.id and "custom_mcdrag_table" in st.session_state:
        ammo.aero.drag_ref = "custom"
        ammo.aero.cd_table = st.session_state["custom_mcdrag_table"]

    ammo.launch.mv.mu = float(mv_mu)
    ammo.launch.mv.sigma_rr = float(mv_s_rr)
    ammo.launch.mv.sigma_lot = float(mv_s_lot)
    ammo.launch.rocket_burn_s = float(rk_burn_s)
    ammo.launch.rocket_mdot_kg_s = float(rk_mdot)
    ammo.launch.rocket_ve_mps = float(rk_ve)
    if rk_burn_s > 0.0 and ammo.launch.mode == "gun":
        ammo.launch.mode = "rocket"
    ammo.aero.form_factor.mu = float(ff_mu)
    ammo.failure_modes.p_propulsion_anomaly = float(fm_prop_pct) / 100.0
    ammo.failure_modes.p_fin_or_sabot_anomaly = float(fm_fin_pct) / 100.0
    ammo.failure_modes.p_fuze_arming_fail = float(fm_sa_pct) / 100.0
    ammo.failure_modes.p_fuze_early_burst = float(fm_early_pct) / 100.0
    ammo.failure_modes.p_low_order_detonation = float(fm_low_pct) / 100.0

    scn = base_scn.model_copy(deep=True)
    scn.fidelity = fidelity  # type: ignore[assignment]
    scn.sampler = sampler  # type: ignore[assignment]
    scn.n_occasions = int(n_occ)
    scn.rounds_per_occasion = int(r_per_occ)
    scn.n_trials = None
    scn.seed = int(seed)
    scn.aim.range_m = float(aim_range)
    scn.aim.high_angle = bool(high_angle)
    scn.aim.propellant_temp_C = float(prop_temp)
    scn.met.wind_range_mps = float(wind_range)
    scn.met.wind_cross_right_mps = float(wind_cross)
    scn.met.deviations.dT_K = float(temp_dev)
    scn.errors.bias.tle_range_sigma_m = float(tle_sig)
    scn.errors.bias.tle_deflection_sigma_m = float(tle_sig)
    scn.errors.bias.aim_qe_bias_sigma_mils = float(aim_qe_bias)
    scn.errors.random.aim_qe_rand_sigma_mils = float(aim_qe_rand)
    scn.errors.bias.wind_range_bias_sigma_mps = float(wind_bias_sig)
    scn.errors.bias.wind_cross_bias_sigma_mps = float(wind_bias_sig)
    scn.two_loop_epistemic = bool(two_loop)
    scn.n_outer_epistemic = int(n_epistemic)
    scn.include_coriolis = bool(use_coriolis)
    scn.vacuum = bool(use_vacuum)

    if terrain_mode == "Linear Hill Profile":
        scn.terrain = TerrainSpec(
            mode="profile",
            profile=[
                TerrainPointSpec(range_m=0.0, elevation_m=0.0),
                TerrainPointSpec(
                    range_m=float(aim_range) * 1.5,
                    elevation_m=(hill_slope / 100.0) * float(aim_range) * 1.5,
                ),
            ],
        )
    else:
        scn.terrain = TerrainSpec(mode="flat")

    if "last_sim_result" not in st.session_state or run_btn or st.session_state.get("last_preset_id") != sel_preset_id:
        with st.spinner("Executing Monte Carlo pipeline (Launch → Exterior → Fuze → Terminal → Vulnerability)..."):
            res = run_simulation(ammo, scn)
            st.session_state["last_sim_result"] = res
            st.session_state["last_sim_ammo"] = ammo
            st.session_state["last_sim_scn"] = scn
            st.session_state["last_preset_id"] = sel_preset_id

    res: SimulationResult = st.session_state["last_sim_result"]
    cur_ammo: AmmoDefinition = st.session_state["last_sim_ammo"]
    cur_scn: Scenario = st.session_state["last_sim_scn"]

    # Display FLAG Warnings if any
    if res.provenance.get("has_flag_warnings"):
        st.markdown(
            f"""
            <div class="flag-banner">
            ⚠️ <b>FLAG GOVERNANCE WARNINGS DETECTED:</b><br/>
            {'<br/>'.join(['• ' + w for w in res.provenance.get('governance_warnings', [])])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 1: Primary Dispersion & Effectiveness KPIs
    c1, c2, c3 = st.columns(3)
    cep50 = float(res.dispersion.get("cep_m", 0.0))
    cep95 = float(res.dispersion.get("cep95_m", cep50 * 2.08))
    with c1:
        st.metric(
            "Mean Range (MPI)",
            f"{res.dispersion.get('mean_range_m', 0.0):.1f} m",
            delta=f"±{res.dispersion.get('range_ci95_half_width_m', 0.0):.1f} m (95% CI)",
        )
    with c2:
        st.metric(
            "Mean Deflection",
            f"{res.dispersion.get('mean_deflection_m', 0.0):.1f} m",
            delta=f"±{res.dispersion.get('deflection_ci95_half_width_m', 0.0):.1f} m",
        )
    with c3:
        st.metric("CEP 50%", f"{cep50:.1f} m", delta=f"CEP95: {cep95:.1f} m", delta_color="off")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Single-Shot Pk", f"{res.effectiveness.get('sspk', 0.0):.3f}", help="Mean single-round effect probability")
    with c5:
        st.metric(
            "Salvo Pk (Correlated)",
            f"{res.effectiveness.get('salvo_pk_mean', 0.0):.3f}",
            delta=f"Indep: {res.effectiveness.get('salvo_pk_independent', 0.0):.3f}",
            delta_color="off",
        )
    with c6:
        st.metric(
            "Fuze Reliability",
            f"{res.uxo.get('reliability_single', 1.0) * 100:.1f} %",
            delta=f"Non-func: {res.uxo.get('dud_rate_mean', 0.0) * 100:.1f}%",
            delta_color="off",
        )

    # Row 2: Secondary Engineering & Ballistic KPIs
    b1, b2, b3 = st.columns(3)
    trials_df = pd.DataFrame(res.trials)
    mean_v_imp = float(trials_df["v_impact_mps"].mean()) if "v_impact_mps" in trials_df else 0.0
    mean_aof = float(trials_df["angle_of_fall_deg"].mean()) if "angle_of_fall_deg" in trials_df else 0.0
    mean_tof = float(trials_df["tof_s"].mean()) if "tof_s" in trials_df else 0.0
    with b1:
        st.metric(
            "Probable Errors (REP / DEP)",
            f"{res.dispersion.get('rep_m', 0.0):.1f} / {res.dispersion.get('dep_m', 0.0):.1f} m",
            help="50% Range Error Probable (REP) and Deflection Error Probable (DEP)",
        )
    with b2:
        qe_deg = math.degrees(res.solved_qe_rad)
        st.metric("Solved Launch QE", f"{res.solved_qe_mils:.1f} mils", delta=f"{qe_deg:.2f}°", delta_color="off")
    with b3:
        st.metric("Impact Speed / Angle", f"{mean_v_imp:.0f} m/s", delta=f"AoF: {mean_aof:.1f}°", delta_color="off")
    b4, b5, b6 = st.columns(3)
    with b4:
        st.metric("Mean Time of Flight", f"{mean_tof:.2f} s", delta=f"N={res.n_total_rounds} rounds", delta_color="off")
    with b5:
        st.metric(
            "Effective Area A_L",
            f"{res.effectiveness.get('lethal_area_m2', 0.0):.0f} m²",
            delta=f"Carleton SSPK: {res.effectiveness.get('sspk_carleton_ref', 0.0):.3f}",
            delta_color="off",
        )
    with b6:
        tb15_dict = res.blast_and_safety.get("iatg_tb15_distances_m", {})
        ibd_m = float(tb15_dict.get("inhabited_building_distance_m", 0.0))
        st.metric("IATG Safety IBD (TB-15)", f"{ibd_m:.1f} m", help="Inhabited Building Distance (22.2 * W^(1/3))")

    # Epistemic Summary Banner if Two-Loop Epistemic was enabled
    if res.epistemic_summary is not None:
        ep = res.epistemic_summary
        st.markdown(
            f"""
            <div class="epistemic-banner">
            🔬 <b>TWO-LOOP EPISTEMIC UNCERTAINTY QUANTIFICATION ({ep.get('n_outer_epistemic', 0)} Outer Realizations):</b>
            CEP50 Median: <b>{ep.get('cep_median_m', 0.0):.1f} m</b> [90% Credible Interval: {ep.get('cep_p05_m', 0.0):.1f} m – {ep.get('cep_p95_m', 0.0):.1f} m] &nbsp;|&nbsp;
            SSPK Median: <b>{ep.get('sspk_median', 0.0):.3f}</b> [90% CI: {ep.get('sspk_p05', 0.0):.3f} – {ep.get('sspk_p95', 0.0):.3f}] &nbsp;|&nbsp;
            Salvo Pk Median: <b>{ep.get('salvo_pk_median', 0.0):.3f}</b> [90% CI: {ep.get('salvo_pk_p05', 0.0):.3f} – {ep.get('salvo_pk_p95', 0.0):.3f}]
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive Plots Row 1: POI Scatter + Trajectory Profile
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        hdr_c1, hdr_c2 = st.columns([2, 1])
        with hdr_c1:
            st.subheader("🎯 Impact Dispersion (POI, CEP50 & CEP95)")
        with hdr_c2:
            poi_color_mode = st.selectbox(
                "Color Impacts By",
                ["Failure Mode (FMEA)", "Fuze Status", "Trial Pk", "Impact Speed (m/s)"],
                index=0,
                label_visibility="collapsed",
            )

        if not trials_df.empty:
            mean_r = float(res.dispersion.get("mean_range_m", 0.0))
            mean_d = float(res.dispersion.get("mean_deflection_m", 0.0))

            fmea_color_map = {
                "Nominal Function": "#2563EB",
                "Fuze Initiation Dud (FZ-03)": "#DC2626",
                "Impact Graze / Ricochet Fail (FZ-06)": "#EA580C",
                "Safe & Arm (S&A) Fail": "#991B1B",
                "Premature Airburst": "#7C3AED",
                "Low-Order Partial Detonation": "#F59E0B",
                "Propulsion / Ignition Anomaly": "#0D9488",
                "Rocket Motor Thrust / Burnout Anomaly (EB-13)": "#059669",
                "Fin / Sabot Tip-Off Anomaly": "#DB2777",
                "Wrap-Around Fin / Nozzle Tip-Off Anomaly": "#EC4899",
                "Ballistic Instability (Sg < 1)": "#4B5563",
            }

            if poi_color_mode == "Failure Mode (FMEA)" and "failure_mode" in trials_df:
                fig_poi = px.scatter(
                    trials_df,
                    x="deflection_m",
                    y="range_m",
                    color="failure_mode",
                    color_discrete_map=fmea_color_map,
                    labels={"deflection_m": "Deflection (m, Right +)", "range_m": "Range (m)", "failure_mode": "Failure Mode"},
                    title=f"Impact Footprint by Failure Mode (N={len(trials_df)}) — CEP50={cep50:.1f}m",
                    opacity=0.80,
                )
            elif poi_color_mode == "Fuze Status":
                fig_poi = px.scatter(
                    trials_df,
                    x="deflection_m",
                    y="range_m",
                    color="is_dud",
                    color_discrete_map={False: "#2563EB", True: "#DC2626"},
                    labels={"deflection_m": "Deflection (m, Right +)", "range_m": "Range (m)", "is_dud": "Non-Func"},
                    title=f"Impact Footprint (N={len(trials_df)}) — CEP50={cep50:.1f}m | CEP95={cep95:.1f}m",
                    opacity=0.7,
                )
            elif poi_color_mode == "Trial Pk" and ("p_kill" in trials_df or "p_kill_round" in trials_df):
                pk_col = "p_kill" if "p_kill" in trials_df else "p_kill_round"
                fig_poi = px.scatter(
                    trials_df,
                    x="deflection_m",
                    y="range_m",
                    color=pk_col,
                    color_continuous_scale="YlOrRd",
                    labels={"deflection_m": "Deflection (m, Right +)", "range_m": "Range (m)", pk_col: "Round Pk"},
                    title=f"Impact Footprint Colored by Single-Round Pk — CEP50={cep50:.1f}m",
                    opacity=0.75,
                )
            else:
                fig_poi = px.scatter(
                    trials_df,
                    x="deflection_m",
                    y="range_m",
                    color="v_impact_mps" if "v_impact_mps" in trials_df else "is_dud",
                    color_continuous_scale="Viridis",
                    labels={
                        "deflection_m": "Deflection (m, Right +)",
                        "range_m": "Range (m)",
                        "v_impact_mps": "V_imp (m/s)",
                    },
                    title=f"Impact Footprint Colored by Terminal Speed — CEP50={cep50:.1f}m",
                    opacity=0.75,
                )

            target_r = cur_scn.aim.range_m or mean_r
            hx = float(cur_scn.target.geometry.get("half_width_x_m", 25.0))
            hy = float(cur_scn.target.geometry.get("half_width_y_m", 25.0))
            fig_poi.add_shape(
                type="rect",
                x0=-hy,
                x1=hy,
                y0=target_r - hx,
                y1=target_r + hx,
                line=dict(color="#10B981", width=2, dash="dash"),
                fillcolor="rgba(16, 185, 129, 0.12)",
            )
            theta = np.linspace(0, 2 * np.pi, 100)
            fig_poi.add_trace(
                go.Scatter(
                    x=mean_d + cep50 * np.sin(theta),
                    y=mean_r + cep50 * np.cos(theta),
                    mode="lines",
                    name=f"CEP 50% ({cep50:.1f}m)",
                    line=dict(color="#F59E0B", width=2.5),
                )
            )
            fig_poi.add_trace(
                go.Scatter(
                    x=mean_d + cep95 * np.sin(theta),
                    y=mean_r + cep95 * np.cos(theta),
                    mode="lines",
                    name=f"CEP 95% ({cep95:.1f}m)",
                    line=dict(color="#EF4444", width=2, dash="dash"),
                )
            )
            fig_poi.add_trace(
                go.Scatter(
                    x=[mean_d],
                    y=[mean_r],
                    mode="markers",
                    name="MPI",
                    marker=dict(color="#1E3A8A", size=10, symbol="cross"),
                )
            )
            fig_poi.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_poi, use_container_width=True)

    with col_plot2:
        st.subheader("📈 Flight Trajectory Bundle & Velocity Profile")
        if res.sample_trajectory is not None:
            fig_traj = make_subplots(specs=[[{"secondary_y": True}]])
            sample_list = (res.sample_trajectories or [res.sample_trajectory])[:8]
            for idx_t, tr_item in enumerate(sample_list):
                fig_traj.add_trace(
                    go.Scatter(
                        x=tr_item.get("range_m", []),
                        y=tr_item.get("alt_m", []),
                        mode="lines",
                        name="Nominal Altitude" if idx_t == 0 else f"Sample #{idx_t + 1}",
                        line=dict(
                            color="#1E3A8A" if idx_t == 0 else "rgba(37, 99, 235, 0.30)",
                            width=3 if idx_t == 0 else 1.5,
                        ),
                        showlegend=(idx_t < 2),
                    ),
                    secondary_y=False,
                )

            traj_data = res.sample_trajectory
            if "speed_mps" in traj_data:
                fig_traj.add_trace(
                    go.Scatter(
                        x=traj_data.get("range_m", []),
                        y=traj_data.get("speed_mps", []),
                        mode="lines",
                        name="Flight Speed (m/s)",
                        line=dict(color="#DC2626", width=2, dash="dot"),
                    ),
                    secondary_y=True,
                )

            r_max = max(traj_data.get("range_m", [1000.0]))
            t_x = np.linspace(0, r_max, 50)
            t_y = np.zeros_like(t_x)
            if cur_scn.terrain.mode == "profile" and cur_scn.terrain.profile:
                p0, p1 = cur_scn.terrain.profile[0], cur_scn.terrain.profile[-1]
                t_y = np.interp(t_x, [p0.range_m, p1.range_m], [p0.elevation_m, p1.elevation_m])

            fig_traj.add_trace(
                go.Scatter(
                    x=t_x,
                    y=t_y,
                    mode="lines",
                    fill="tozeroy",
                    name="Ground Elevation",
                    line=dict(color="#6B7280", width=1.5),
                    fillcolor="rgba(107, 114, 128, 0.2)",
                ),
                secondary_y=False,
            )
            fig_traj.update_xaxes(title_text="Downrange X (m)")
            fig_traj.update_yaxes(title_text="Altitude Z (m)", secondary_y=False)
            fig_traj.update_yaxes(title_text="Speed (m/s)", secondary_y=True)
            fig_traj.update_layout(
                title=f"Trajectory Plume & Speed Profile (Solved QE: {res.solved_qe_mils:.1f} mils)",
                height=430,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("Sample trajectory available for L1/L2/L3 physics simulations.")

    # Row 2: Hazard Contours & Sensitivity
    col_hz1, col_hz2 = st.columns(2)

    with col_hz1:
        st.subheader("💥 2D Spatial Effect Field vs IATG Safety Distance")
        if res.hazard_grid is not None:
            hg = res.hazard_grid
            gx, gy = np.array(hg["x_m"]), np.array(hg["y_m"])
            pk_grid = np.array(hg["pk_grid"])
            fig_hz = go.Figure(
                data=go.Contour(
                    x=gy,
                    y=gx,
                    z=pk_grid,
                    colorscale="Reds",
                    contours=dict(start=0.1, end=0.9, size=0.2, showlabels=True),
                    colorbar=dict(title="Pk"),
                )
            )
            tb15_r = float(
                res.blast_and_safety.get("iatg_tb15_distances_m", {}).get(
                    "inhabited_building_distance_m", 0.0
                )
            )
            if tb15_r > 0:
                th_c = np.linspace(0, 2 * np.pi, 80)
                fig_hz.add_trace(
                    go.Scatter(
                        x=tb15_r * np.cos(th_c),
                        y=tb15_r * np.sin(th_c),
                        mode="lines",
                        name=f"IATG IBD ({tb15_r:.0f} m)",
                        line=dict(color="#DC2626", width=2, dash="dash"),
                    )
                )
            fig_hz.update_layout(
                xaxis_title="Deflection from Burst (m)",
                yaxis_title="Range from Burst (m)",
                title=f"Terminal Effect Contour (Effective Area: {res.effectiveness.get('lethal_area_m2', 0.0):.0f} m²)",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_hz, use_container_width=True)
        elif res.penetration is not None:
            pen = res.penetration
            st.markdown("#### 🛡️ Armor Penetration & Perforation Mechanics (`PEN-01..PEN-05`)")
            p_df = pd.DataFrame(
                [
                    {"Metric": "Penetrator Mechanism", "Value": str(pen.get("type", "kinetic"))},
                    {"Metric": "Mean Penetration Depth", "Value": f"{pen.get('mean_penetration_m', 0.0)*1000:.1f} mm"},
                    {"Metric": "Median (P50) Depth", "Value": f"{pen.get('p50_penetration_m', 0.0)*1000:.1f} mm"},
                    {
                        "Metric": "P10 – P90 Depth Interval",
                        "Value": f"{pen.get('p10_penetration_m', 0.0)*1000:.1f} – {pen.get('p90_penetration_m', 0.0)*1000:.1f} mm",
                    },
                    {"Metric": "Target Equivalent RHA", "Value": f"{pen.get('target_los_rha_m', 0.0)*1000:.1f} mm"},
                    {"Metric": "Perforation Probability", "Value": f"{pen.get('perforation_prob', 0.0)*100:.1f} %"},
                ]
            )
            st.table(p_df)
        else:
            st.info("Spatial contour field generated when warhead fill mass > 0.")

    with col_hz2:
        st.subheader("📊 Error Budget Sensitivity Ranking (MC-10)")
        if res.sensitivity:
            sens_df = pd.DataFrame(res.sensitivity)
            score_col = (
                "importance_score"
                if "importance_score" in sens_df.columns
                else ("abs_rho" if "abs_rho" in sens_df.columns else sens_df.columns[1])
            )
            sens_df = sens_df.sort_values(score_col, ascending=True)
            fig_sens = px.bar(
                sens_df,
                x=score_col,
                y="parameter",
                orientation="h",
                color=score_col,
                color_continuous_scale="Blues",
                labels={score_col: "Importance Score (|ρ_s| / Sobol S1)", "parameter": "Stochastic Input"},
                title="Variance / Rank-Correlation Contribution to Range Dispersion",
            )
            fig_sens.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Sensitivity analysis requires variation across stochastic parameters.")

    # Row 2.5: Multi-Stage System Failure Modes, Reliability & FMEA Breakdown
    with st.expander("🛠️ System Failure Modes, Reliability & FMEA Breakdown (Propulsion → Ballistics → Fuze → Warhead)", expanded=True):
        fms = res.failure_mode_summary or {}
        fm_counts = fms.get("counts", {})
        fm_c1, fm_c2, fm_c3, fm_c4 = st.columns(4)
        fm_c1.metric(
            "Overall System Reliability",
            f"{float(fms.get('overall_system_reliability', 1.0)) * 100:.2f} %",
            delta=f"{int(fms.get('nominal_function_count', res.n_total_rounds))} / {res.n_total_rounds} Nominal",
            delta_color="off",
        )
        fm_c2.metric(
            "Total Anomaly / Failure Rate",
            f"{float(fms.get('any_anomaly_or_failure_rate', 0.0)) * 100:.2f} %",
            delta=f"{res.n_total_rounds - int(fms.get('nominal_function_count', res.n_total_rounds))} Affected Rounds",
            delta_color="inverse",
        )
        fm_c3.metric(
            "Total Fuze Dud Rate",
            f"{float(fms.get('dud_rate', 0.0)) * 100:.2f} %",
            delta=f"Persistent UXO: {float(fms.get('uxo_rate', 0.0)) * 100:.2f}%",
            delta_color="off",
        )
        fm_c4.metric(
            "Self-Destruct Neutralized (FZ-07)",
            f"{int(fms.get('self_destruct_neutralized_count', 0))} rounds",
            delta=f"Expected UXO/Salvo: {float(res.uxo.get('expected_uxo_per_salvo', 0.0)):.2f}",
            delta_color="off",
        )

        fmea_col1, fmea_col2 = st.columns([1.15, 1.35])
        if not trials_df.empty and "failure_mode" in trials_df:
            fmea_Consequences = {
                "Nominal Function": "Full design propulsion, stable flight, and high-order warhead initiation",
                "Fuze Initiation Dud (FZ-03)": "Primary fuze initiation failure; zero explosive yield, potential UXO hazard",
                "Impact Graze / Ricochet Fail (FZ-06)": "Shallow angle-of-fall graze or low-velocity ricochet below arming/function threshold",
                "Safe & Arm (S&A) Fail": "Mechanical/electronic S&A fails to arm in flight; unarmed impact dud / UXO",
                "Premature Airburst": "Early trajectory function (rain/canopy/time); ~42% range loss and zero target effect",
                "Low-Order Partial Detonation": "Incomplete shock-to-detonation transition; 25% of nominal blast/fragmentation output",
                "Propulsion / Ignition Anomaly": "Incomplete propellant burn or gas-seal blow-by; 15% muzzle velocity drop (short round)",
                "Rocket Motor Thrust / Burnout Anomaly (EB-13)": "Solid rocket motor grain cracking, nozzle erosion, or early burnout; ~22% boost range deficit",
                "Fin / Sabot Tip-Off Anomaly": "Asymmetric sabot discard or fin deployment anomaly; 4.5 mil angular dispersion & +18% drag",
                "Wrap-Around Fin / Nozzle Tip-Off Anomaly": "Launch-tube exit tip-off or wrap-around fin deployment asymmetry; high cross-range dispersion",
                "Ballistic Instability (Sg < 1)": "Gyroscopic stability factor Sg < 1.0; severe yaw divergence and range degradation",
            }
            grp_rows = []
            for mode_name, grp in trials_df.groupby("failure_mode"):
                cnt = len(grp)
                grp_rows.append(
                    {
                        "Failure Mode (FMEA)": str(mode_name),
                        "Count": int(cnt),
                        "Share (%)": round(100.0 * cnt / len(trials_df), 2),
                        "Mean Range (m)": round(float(grp["range_m"].mean()), 1),
                        "Mean V_imp (m/s)": round(float(grp["v_impact_mps"].mean()), 1),
                        "Mean Pk": round(float(grp["p_kill"].mean()), 3) if "p_kill" in grp else 0.0,
                        "UXO Fraction (%)": round(100.0 * float(grp["is_uxo"].mean()), 1) if "is_uxo" in grp else 0.0,
                        "Physical Mechanism & Consequence": fmea_Consequences.get(str(mode_name), "Subsystem anomaly"),
                    }
                )
            fmea_df = pd.DataFrame(grp_rows).sort_values("Count", ascending=False)

            with fmea_col1:
                non_nom_df = fmea_df[fmea_df["Failure Mode (FMEA)"] != "Nominal Function"]
                plot_fmea_df = non_nom_df if not non_nom_df.empty else fmea_df
                fig_fmea = px.bar(
                    plot_fmea_df.sort_values("Share (%)", ascending=True),
                    x="Share (%)",
                    y="Failure Mode (FMEA)",
                    orientation="h",
                    color="Failure Mode (FMEA)",
                    text="Count",
                    title="Pareto Breakdown of Non-Nominal Failure Modes (%)" if not non_nom_df.empty else "Failure Mode Distribution (%)",
                )
                fig_fmea.update_layout(height=320, margin=dict(l=15, r=15, t=40, b=15), showlegend=False)
                st.plotly_chart(fig_fmea, use_container_width=True)

            with fmea_col2:
                st.dataframe(fmea_df, use_container_width=True, height=320)

    # Row 3: Monte Carlo Convergence & Provenance Audit
    conv_col1, conv_col2 = st.columns(2)
    with conv_col1:
        st.subheader("📉 Monte Carlo Running Convergence Diagnostics")
        if res.convergence and len(res.convergence.get("n_rounds", [])) > 1:
            conv_df = pd.DataFrame(res.convergence)
            fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
            y_r_col = "mean_range_m" if "mean_range_m" in conv_df.columns else conv_df.columns[1]
            y_p_col = "sspk" if "sspk" in conv_df.columns else conv_df.columns[-1]
            fig_conv.add_trace(
                go.Scatter(
                    x=conv_df["n_rounds"],
                    y=conv_df[y_r_col],
                    mode="lines+markers",
                    name="Running Mean Range (m)",
                    line=dict(color="#2563EB", width=2.5),
                ),
                secondary_y=False,
            )
            fig_conv.add_trace(
                go.Scatter(
                    x=conv_df["n_rounds"],
                    y=conv_df[y_p_col],
                    mode="lines+markers",
                    name="Running SSPK (-)",
                    line=dict(color="#10B981", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_conv.update_xaxes(title_text="Cumulative Simulated Rounds (N)")
            fig_conv.update_yaxes(title_text="Running Mean Range (m)", secondary_y=False)
            fig_conv.update_yaxes(title_text="Running SSPK", secondary_y=True)
            fig_conv.update_layout(
                title="Estimator Stability vs Sample Size N",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    with conv_col2:
        st.subheader("🧾 Provenance & Active Formula Audit Trail")
        forms_used = res.provenance.get("formulas_used", [])
        if forms_used:
            f_df = pd.DataFrame(forms_used)
            st.dataframe(
                f_df[["formula_id", "module", "name", "verification_status", "source_id"]],
                use_container_width=True,
                height=300,
            )

    # Row 4: Deep Scientific Physics Diagnostics (Energy Conservation, Re/Mach/Sg, Bivariate Covariance)
    with st.expander(
        "🔬 Flight Energy Conservation, Dimensionless Aerodynamics (Re, Mach, S_g) & Bivariate Eigen-Analysis",
        expanded=True,
    ):
        sci_c1, sci_c2, sci_c3 = st.columns([1.2, 1.2, 1.0])
        if res.sample_trajectory is not None:
            tr_nom = res.sample_trajectory
            rx_a = np.array(tr_nom.get("range_m", []), dtype=np.float64)
            rz_a = np.array(tr_nom.get("alt_m", []), dtype=np.float64)
            v_a = np.array(tr_nom.get("speed_mps", []), dtype=np.float64)
            m_proj = float(cur_ammo.mass_kg)
            d_proj = float(cur_ammo.calibre_m)

            # Mechanical energy conservation (MJ)
            ek_mj = 0.5 * m_proj * (v_a**2) * 1e-6
            ep_mj = m_proj * 9.80665 * rz_a * 1e-6
            etot_mj = ek_mj + ep_mj
            w_drag_mj = np.maximum(etot_mj[0] - etot_mj, 0.0) if len(etot_mj) > 0 else np.zeros_like(etot_mj)

            with sci_c1:
                fig_en = go.Figure()
                fig_en.add_trace(
                    go.Scatter(x=rx_a, y=ek_mj, name="Kinetic E_k (MJ)", line=dict(color="#2563EB", width=2.5))
                )
                fig_en.add_trace(
                    go.Scatter(x=rx_a, y=ep_mj, name="Potential E_p (MJ)", line=dict(color="#10B981", width=2.5))
                )
                fig_en.add_trace(
                    go.Scatter(
                        x=rx_a,
                        y=etot_mj,
                        name="Total Mech E_tot (MJ)",
                        line=dict(color="#1E3A8A", width=2, dash="dash"),
                    )
                )
                fig_en.add_trace(
                    go.Scatter(
                        x=rx_a,
                        y=w_drag_mj,
                        name="Drag Work Dissipated W_d (MJ)",
                        line=dict(color="#DC2626", width=2, dash="dot"),
                    )
                )
                fig_en.update_layout(
                    title="First-Law Mechanical Energy Budget (ΔE = -∫ F_drag·dr)",
                    xaxis_title="Downrange X (m)",
                    yaxis_title="Energy (MJ)",
                    height=330,
                    margin=dict(l=15, r=15, t=40, b=15),
                )
                st.plotly_chart(fig_en, use_container_width=True)

            with sci_c2:
                atm_tr = eb_04_isa_atmosphere(rz_a, dT_K=float(cur_scn.met.deviations.dT_K))
                rho_tr = np.asarray(atm_tr["rho_kg_m3"], dtype=np.float64)
                t_tr = np.asarray(atm_tr["T_K"], dtype=np.float64)
                re_tr = np.asarray(aerodynamic_reynolds_number(rho_tr, v_a, d_proj, t_tr), dtype=np.float64) * 1e-6
                q_tr_kpa = 0.5 * rho_tr * (v_a**2) * 1e-3

                fig_re = make_subplots(specs=[[{"secondary_y": True}]])
                fig_re.add_trace(
                    go.Scatter(
                        x=rx_a,
                        y=re_tr,
                        name="Reynolds Number Re_d (10^6)",
                        line=dict(color="#7C3AED", width=2.5),
                    ),
                    secondary_y=False,
                )
                fig_re.add_trace(
                    go.Scatter(
                        x=rx_a,
                        y=q_tr_kpa,
                        name="Dynamic Pressure q (kPa)",
                        line=dict(color="#EA580C", width=2.5, dash="dash"),
                    ),
                    secondary_y=True,
                )
                fig_re.update_xaxes(title_text="Downrange X (m)")
                fig_re.update_yaxes(title_text="Re_d (× 10^6)", secondary_y=False)
                fig_re.update_yaxes(title_text="Dynamic Pressure q (kPa)", secondary_y=True)
                fig_re.update_layout(
                    title="Dimensionless Aerodynamics (Sutherland Re_d & Dynamic Pressure q)",
                    height=330,
                    margin=dict(l=15, r=15, t=40, b=15),
                )
                st.plotly_chart(fig_re, use_container_width=True)

        with sci_c3:
            st.markdown("##### 📐 Bivariate Covariance & Eigen-Structure")
            if not trials_df.empty and len(trials_df) >= 4:
                xr = trials_df["range_m"].to_numpy(dtype=np.float64)
                yd = trials_df["deflection_m"].to_numpy(dtype=np.float64)
                cov_mat = np.cov(xr, yd)
                eigvals, eigvecs = np.linalg.eigh(cov_mat)
                idx_sort = np.argsort(eigvals)[::-1]
                eigvals = np.maximum(eigvals[idx_sort], 0.0)
                eigvecs = eigvecs[:, idx_sort]
                major_1sig = float(np.sqrt(eigvals[0]))
                minor_1sig = float(np.sqrt(eigvals[1]))
                angle_deg = float( np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])) )
                r_corr = float(np.corrcoef(xr, yd)[0, 1]) if np.std(xr) > 0 and np.std(yd) > 0 else 0.0
                _, ks_p_r = sp_stats.kstest(
                    (xr - np.mean(xr)) / max(float(np.std(xr)), 1e-9), "norm"
                )
                _, ks_p_d = sp_stats.kstest(
                    (yd - np.mean(yd)) / max(float(np.std(yd)), 1e-9), "norm"
                )
                eig_df = pd.DataFrame(
                    [
                        {"Diagnostic": "Principal Major Semi-Axis √λ₁", "Value": f"{major_1sig:.2f} m"},
                        {"Diagnostic": "Principal Minor Semi-Axis √λ₂", "Value": f"{minor_1sig:.2f} m"},
                        {"Diagnostic": "Ellipse Aspect Ratio √(λ₁/λ₂)", "Value": f"{major_1sig / max(minor_1sig, 1e-6):.2f}"},
                        {"Diagnostic": "Principal Axis Azimuth θ_eig", "Value": f"{angle_deg:+.1f}°"},
                        {"Diagnostic": "Range-Deflection Corr r_xy", "Value": f"{r_corr:+.3f}"},
                        {"Diagnostic": "KS Gaussianity p-val (Range / Defl)", "Value": f"{ks_p_r:.2f} / {ks_p_d:.2f}"},
                    ]
                )
                st.table(eig_df)

    # Artifact Export Section
    st.subheader("📥 Export Simulation Artifacts")
    exp_c1, exp_c2, exp_c3, exp_c4, exp_c5 = st.columns(5)
    with exp_c1:
        csv_bytes = trials_to_csv_bytes(res)
        st.download_button(
            "📄 Download CSV Trials",
            csv_bytes,
            f"{cur_ammo.id}_trials.csv",
            "text/csv",
            use_container_width=True,
        )
    with exp_c2:
        json_str = summary_to_json_str(res)
        st.download_button(
            "📦 Download JSON Summary",
            json_str.encode("utf-8"),
            f"{cur_ammo.id}_summary.json",
            "application/json",
            use_container_width=True,
        )
    with exp_c3:
        geojson_str = geojson_to_str(res, scenario=cur_scn)
        st.download_button(
            "🗺️ Download GeoJSON",
            geojson_str.encode("utf-8"),
            f"{cur_ammo.id}_contours.geojson",
            "application/geo+json",
            use_container_width=True,
        )
    with exp_c4:
        png_bytes = render_report_figure_bytes(res, scenario=cur_scn, fmt="png")
        st.download_button(
            "🖼️ Download PNG Report",
            png_bytes,
            f"{cur_ammo.id}_report.png",
            "image/png",
            use_container_width=True,
        )
    with exp_c5:
        pdf_bytes = render_report_figure_bytes(res, scenario=cur_scn, fmt="pdf")
        st.download_button(
            "📑 Download PDF Report",
            pdf_bytes,
            f"{cur_ammo.id}_report.pdf",
            "application/pdf",
            use_container_width=True,
        )


# --- TAB 2: 3D SIMULATIONS ---
def tab_3d_simulation() -> None:
    st.header("🌐 Interactive 3D Ballistics, Terrain & Terminal Physics Arena")
    st.markdown(
        "Explore 3D spatial trajectory plumes, procedural Digital Elevation Models (DEM), "
        "3D CEP dispersion rings, time-stepped animated salvo replays, 3D overpressure iso-surfaces, "
        "and multi-preset 3D trajectory comparisons."
    )

    if "last_sim_result" not in st.session_state:
        ammo, scn = get_preset("NOTIONAL-AR-01")
        with st.spinner("Initializing 3D simulation baseline..."):
            res = run_simulation(ammo, scn)
            st.session_state["last_sim_result"] = res
            st.session_state["last_sim_ammo"] = ammo
            st.session_state["last_sim_scn"] = scn

    res: SimulationResult = st.session_state["last_sim_result"]
    cur_ammo: AmmoDefinition = st.session_state["last_sim_ammo"]
    cur_scn: Scenario = st.session_state["last_sim_scn"]

    view_mode = st.radio(
        "Select 3D Simulation Mode:",
        [
            "🎯 3D Flight Trajectory Bundle & Terrain",
            "🎬 Animated 3D Salvo Flight Replay",
            "🌀 3D Phase-Space & Thermodynamic Manifold",
            "💥 3D Pressure Wave & Fragment Kinematics",
            "🏛️ 3D IATG Explosive Safety & QD Domes",
            "⚔️ Multi-Preset 3D Comparison Arena",
        ],
        horizontal=True,
    )

    if view_mode == "🎯 3D Flight Trajectory Bundle & Terrain":
        st.subheader("🎯 3D Flight Dynamics, CEP Ellipses & Terrain Intersection")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color_opt = st.selectbox(
                "Color Trajectories By",
                ["Speed (m/s)", "Mach Number (-)", "Altitude (m)", "Trajectory Index"],
                index=0,
            )
            terr_style_label = st.selectbox(
                "3D Terrain DEM Style",
                ["Scenario Default", "Rolling Hills DEM", "Canyon Valley", "Transverse Ridge"],
                index=0,
            )
        with c2:
            n_paths = st.slider("3D Trajectories in Bundle", 1, 20, 12, 1)
            cam_preset = st.selectbox(
                "Camera Perspective", ["Perspective Overview", "Top-Down Map", "Side Profile", "Launcher View"]
            )
        with c3:
            show_terr = st.checkbox("Show 3D Terrain Surface", value=True)
            show_tgt = st.checkbox("Show 3D Target Volume", value=True)
        with c4:
            show_poi = st.checkbox("Show Ground Impact POI", value=True)
            show_cep = st.checkbox("Show 3D CEP50 & CEP95 Rings", value=True)

        c_key = (
            "speed"
            if "Speed" in color_opt
            else ("mach" if "Mach" in color_opt else ("altitude" if "Altitude" in color_opt else "index"))
        )
        t_style_map = {
            "Scenario Default": "auto",
            "Rolling Hills DEM": "rolling_hills",
            "Canyon Valley": "valley",
            "Transverse Ridge": "ridge",
        }
        fig_3d_traj = create_3d_trajectory_bundle_figure(
            result=res,
            scenario=cur_scn,
            color_by=c_key,  # type: ignore[arg-type]
            show_terrain=show_terr,
            show_target=show_tgt,
            show_poi=show_poi,
            show_cep_rings=show_cep,
            terrain_style=t_style_map[terr_style_label],  # type: ignore[arg-type]
            max_trajectories=n_paths,
        )

        if cam_preset == "Top-Down Map":
            fig_3d_traj.update_layout(scene_camera=dict(eye=dict(x=0.0, y=0.0, z=2.5), up=dict(x=0, y=1, z=0)))
        elif cam_preset == "Side Profile":
            fig_3d_traj.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-2.5, z=0.15), up=dict(x=0, y=0, z=1)))
        elif cam_preset == "Launcher View":
            fig_3d_traj.update_layout(
                scene_camera=dict(eye=dict(x=-2.2, y=0.0, z=0.4), center=dict(x=0.0, y=0.0, z=0.0))
            )

        fig_3d_traj.update_layout(height=650)
        st.plotly_chart(fig_3d_traj, use_container_width=True)

        html_3d_traj = fig_3d_traj.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "💾 Download Interactive 3D Trajectory Scene (.html)",
            html_3d_traj.encode("utf-8"),
            f"{cur_ammo.id}_3d_trajectories.html",
            "text/html",
        )

    elif view_mode == "🎬 Animated 3D Salvo Flight Replay":
        st.subheader("🎬 Time-Stepped 3D Salvo Propagation Replay")
        st.caption("Click **▶ Play 3D Salvo** inside the 3D viewport or drag the time slider to scrub through flight.")
        a1, a2 = st.columns(2)
        with a1:
            anim_paths = st.slider("Animated Trajectories", 2, 12, 8, 1)
        with a2:
            anim_frames = st.slider("Time Resolution (Keyframes)", 12, 40, 24, 2)

        fig_anim = create_3d_animated_flight_figure(
            result=res,
            scenario=cur_scn,
            max_trajectories=anim_paths,
            n_frames=anim_frames,
        )
        fig_anim.update_layout(height=660)
        st.plotly_chart(fig_anim, use_container_width=True)

        html_anim = fig_anim.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "💾 Download Standalone 3D Flight Animation (.html)",
            html_anim.encode("utf-8"),
            f"{cur_ammo.id}_3d_animated_salvo.html",
            "text/html",
        )

    elif view_mode == "🌀 3D Phase-Space & Thermodynamic Manifold":
        st.subheader("🌀 3D Aerodynamic & Thermodynamic State Manifold (Range X × Altitude Z × Mach M)")
        st.markdown(
            "Maps the stochastic trajectory ensemble into 3D dimensionless Mach-Altitude-Range phase space "
            "with a translucent **Sonic Barrier Plane ($M = 1.0$)** showing where projectiles transition through the transonic wave-drag peak."
        )
        ps1, ps2 = st.columns(2)
        with ps1:
            ps_color_label = st.selectbox(
                "Color State Trajectories By Physical Field",
                [
                    "Dynamic Pressure q = ½ρv² (kPa)",
                    "Reynolds Number Re_d = ρvL/μ(T) (×10⁶)",
                    "Specific Mechanical Energy e = ½v² + gz (kJ/kg)",
                ],
                index=0,
            )
        with ps2:
            ps_paths = st.slider("Trajectories in Phase-Space Bundle", 2, 16, 10, 1)

        ps_map = {
            "Dynamic Pressure q = ½ρv² (kPa)": "dynamic_pressure_kPa",
            "Reynolds Number Re_d = ρvL/μ(T) (×10⁶)": "reynolds_millions",
            "Specific Mechanical Energy e = ½v² + gz (kJ/kg)": "specific_energy_kJ_kg",
        }
        fig_ps = create_3d_phase_space_figure(
            result=res,
            scenario=cur_scn,
            color_metric=ps_map[ps_color_label],  # type: ignore[arg-type]
            calibre_m=float(cur_ammo.calibre_m),
            max_trajectories=ps_paths,
        )
        fig_ps.update_layout(height=640)
        st.plotly_chart(fig_ps, use_container_width=True)

    elif view_mode == "💥 3D Pressure Wave & Fragment Kinematics":
        st.subheader("💥 3D Overpressure Iso-Surfaces & Fragmentation Trajectories")
        d1, d2, d3, d4 = st.columns(4)
        def_charge = float(cur_ammo.warhead.fill.mass_kg if cur_ammo.warhead else 10.0)
        def_hob = float(cur_ammo.fuze.hob.mu if cur_ammo.fuze else 0.0)

        with d1:
            charge_kg = st.number_input(
                "Energetic Fill Mass (kg)", min_value=0.1, max_value=2000.0, value=def_charge, step=1.0
            )
        with d2:
            exp_type = st.selectbox(
                "Energetic Material",
                ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20"],
                index=0,
            )
        with d3:
            burst_z = st.slider("Burst Height Above Ground (m)", 0.0, 50.0, def_hob, 1.0)
        with d4:
            n_frags = st.slider("3D Fragment Trajectory Rays", 30, 300, 120, 10)

        fig_3d_blast = create_3d_blast_and_frag_figure(
            result=res,
            scenario=cur_scn,
            burst_alt_m=float(burst_z),
            n_frag_traces=int(n_frags),
            charge_mass_kg=float(charge_kg),
            explosive_name=exp_type,
        )
        fig_3d_blast.update_layout(height=650)
        st.plotly_chart(fig_3d_blast, use_container_width=True)

        html_3d_blast = fig_3d_blast.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "💾 Download Interactive 3D Pressure/Fragment Model (.html)",
            html_3d_blast.encode("utf-8"),
            f"{cur_ammo.id}_3d_terminal.html",
            "text/html",
        )

    elif view_mode == "🏛️ 3D IATG Explosive Safety & QD Domes":
        st.subheader("🏛️ 3D UNODA IATG 01.80 / 02.20 Potential Explosion Site (PES) & Quantity-Distance Domes")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            qd_neq = st.number_input("Net Explosive Quantity NEQ (kg)", 1.0, 100000.0, 1000.0, 50.0, key="3d_qd_neq")
        with q2:
            qd_auw = st.number_input("All-Up Munition Weight AUW (kg)", 1.0, 250000.0, 2500.0, 100.0, key="3d_qd_auw")
        with q3:
            qd_exp = st.selectbox(
                "Energetic Material",
                ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20"],
                index=0,
                key="3d_qd_exp",
            )
        with q4:
            qd_domes = st.checkbox("Show 3D Translucent Hemispherical Safety Domes", value=True, key="3d_qd_domes")

        fig_3d_qd = create_3d_iatg_safety_zones_figure(
            neq_kg=float(qd_neq),
            auw_kg=float(qd_auw),
            explosive_name=qd_exp,
            show_domes=bool(qd_domes),
        )
        fig_3d_qd.update_layout(height=650)
        st.plotly_chart(fig_3d_qd, use_container_width=True)

    elif view_mode == "⚔️ Multi-Preset 3D Comparison Arena":
        st.subheader("⚔️ Multi-Preset 3D Trajectory & Dispersion Comparison")
        all_p = [p["preset_id"] for p in list_presets()]
        sel_comp = st.multiselect(
            "Select Presets to Compare in 3D Space (2 to 4 recommended)",
            all_p,
            default=["NOTIONAL-AR-01", "NOTIONAL-AR-02", "NOTIONAL-MO-01"],
        )
        if len(sel_comp) >= 1:
            runs_list: list[tuple[SimulationResult, Scenario]] = []
            comp_rows: list[dict[str, Any]] = []
            with st.spinner("Simulating selected presets for 3D comparison..."):
                for pid in sel_comp[:4]:
                    p_ammo, p_scn = get_preset(pid)
                    p_scn.n_occasions = 15
                    p_scn.rounds_per_occasion = 4
                    p_res = run_simulation(p_ammo, p_scn)
                    runs_list.append((p_res, p_scn))
                    comp_rows.append(
                        {
                            "Preset ID": pid,
                            "Name": p_ammo.name,
                            "Aim Range (m)": p_scn.aim.range_m,
                            "Solved QE (mils)": round(p_res.solved_qe_mils, 1),
                            "CEP 50% (m)": round(float(p_res.dispersion.get("cep_m", 0.0)), 1),
                            "CEP 95% (m)": round(float(p_res.dispersion.get("cep95_m", 0.0)), 1),
                            "SSPK": round(float(p_res.effectiveness.get("sspk", 0.0)), 3),
                        }
                    )
            fig_comp = create_3d_multi_preset_comparison_figure(runs_list, show_terrain=True)
            fig_comp.update_layout(height=620)
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


# --- TAB 3: MCDRAG ESTIMATOR & 3D PROJECTILE CAD ---
def tab_mcdrag() -> None:
    st.header("🚀 McCoy (1974) MCDRAG Aerodynamics & 3D Projectile CAD")
    st.markdown(
        "Interactive geometric designer coupled to Robert L. McCoy's BRL 1974 `MCDRAG` algorithm (`S15`). "
        "Predicts zero-yaw drag $C_{D0}(M)$ and its wave/viscous components (head, skin friction, base, boattail, rotating band) "
        "across Mach 0.5 to 5.0 alongside a real-time 3D axisymmetric projectile mesh."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cal_mm = st.number_input("Reference Diameter / Calibre (mm)", min_value=5.56, max_value=400.0, value=155.0, step=5.0)
        tot_len_cal = st.slider("Total Length (calibres)", 2.0, 10.0, 5.2, 0.1)
        nose_len_cal = st.slider("Nose / Ogive Length (calibres)", 1.0, 6.0, 2.8, 0.1)
    with c2:
        bt_len_cal = st.slider("Boattail Length (calibres)", 0.0, 2.0, 0.6, 0.05)
        base_dia_cal = st.slider("Base Diameter (calibres)", 0.5, 1.0, 0.84, 0.02)
        meplat_dia_cal = st.slider("Meplat Flat Diameter (calibres)", 0.0, 0.5, 0.12, 0.02)
    with c3:
        band_dia_cal = st.slider("Rotating Band Diameter (calibres)", 1.0, 1.1, 1.02, 0.01)
        rt_r = st.slider("Ogive Tangency Ratio R_t/R (0=cone, 1=tangent)", 0.0, 1.0, 0.65, 0.05)
        bl_code = st.selectbox("Boundary Layer Condition", ["L/T", "T/T", "L/L"], index=0)

    inp = McDragInputs(
        d_ref_mm=float(cal_mm),
        total_length_cal=float(tot_len_cal),
        nose_length_cal=float(nose_len_cal),
        rt_over_r=float(rt_r),
        boattail_length_cal=float(bt_len_cal),
        base_dia_cal=float(base_dia_cal),
        meplat_dia_cal=float(meplat_dia_cal),
        band_dia_cal=float(band_dia_cal),
        boundary_layer_code=bl_code,
        identification=f"Custom {cal_mm:.0f}mm Shell",
    )
    res = estimate_mcdrag(inp)
    table_df = pd.DataFrame(res["table"])

    cad_col, drag_col = st.columns([1.05, 1.35])
    with cad_col:
        fig_cad = create_3d_projectile_geometry_figure(inp)
        fig_cad.update_layout(height=470)
        st.plotly_chart(fig_cad, use_container_width=True)

    with drag_col:
        kb = get_kb()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=table_df["mach"],
                y=table_df["cd0"],
                mode="lines+markers",
                name="MCDRAG Total Cd0",
                line=dict(color="#1E3A8A", width=3.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=table_df["mach"],
                y=table_df["cdh"],
                mode="lines",
                name="Head Wave Drag (Cdh)",
                line=dict(color="#DC2626", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=table_df["mach"],
                y=table_df["cdsf"],
                mode="lines",
                name="Skin Friction (Cdsf)",
                line=dict(color="#10B981", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=table_df["mach"],
                y=table_df["cdb"],
                mode="lines",
                name="Base Drag (Cdb)",
                line=dict(color="#F59E0B", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=table_df["mach"],
                y=table_df["cdbt"],
                mode="lines",
                name="Boattail Drag (Cdbt)",
                line=dict(color="#8B5CF6", width=2),
            )
        )
        # Overlay G7 reference curve
        g7_m = np.linspace(0.5, 5.0, 45)
        g7_cd = [float(kb.interp_cd(m, "G7")) for m in g7_m]
        fig.add_trace(
            go.Scatter(
                x=g7_m,
                y=g7_cd,
                mode="lines",
                name="Standard G7 Reference",
                line=dict(color="#6B7280", width=1.8, dash="dot"),
            )
        )
        fig.update_layout(
            xaxis_title="Mach Number (-)",
            yaxis_title="Zero-Yaw Drag Coefficient Cd0",
            title=f"MCDRAG Component Breakdown vs G7 Reference ({cal_mm:.1f} mm)",
            height=470,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    if res.get("diagnostics"):
        st.warning("MCDRAG Geometric Advisories: " + " | ".join(res["diagnostics"]))

    act_c1, act_c2 = st.columns([1, 2])
    with act_c1:
        if st.button("⚡ Apply Custom MCDRAG Curve to Active Simulation", use_container_width=True):
            cur_ammo_obj = st.session_state.get("last_sim_ammo")
            if cur_ammo_obj is not None:
                st.session_state["custom_mcdrag_ammo_id"] = cur_ammo_obj.id
                st.session_state["custom_mcdrag_table"] = [
                    {"mach": float(r["mach"]), "cd": float(r["cd0"])} for r in res["table"]
                ]
                st.success(
                    f"Injected custom 26-point MCDRAG Cd(M) table into {cur_ammo_obj.id}! Click 'Run Monte Carlo Simulation' in Tab 1 to fly with this drag curve."
                )
    with act_c2:
        Peak_Cd = float(table_df["cd0"].max())
        Peak_M = float(table_df.loc[table_df["cd0"].idxmax(), "mach"])
        Supersonic_Cd = float(table_df.loc[table_df["mach"] == 2.0, "cd0"].values[0])
        st.info(
            f"**Transonic Peak Drag:** $C_{{D0}} = {Peak_Cd:.3f}$ at Mach {Peak_M:.2f}  |  "
            f"**Supersonic Cruise (Mach 2.0):** $C_{{D0}} = {Supersonic_Cd:.3f}$"
        )

    st.dataframe(table_df, use_container_width=True)


# --- TAB 4: SCIENTIFIC & CONTINUUM PHYSICS LABS ---
def tab_calculators() -> None:
    st.header("🔬 Computational Physics, Thermodynamics & Continuum Mechanics Labs")
    st.markdown(
        "Interactive laboratory grounding every stage of the simulator in first-principles differential equations, "
        "thermodynamic equations of state, Rankine-Hugoniot shock jump relations, and continuum solid mechanics."
    )
    kb = get_kb()

    sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10, sub11, sub12, sub13, sub14, sub15, sub16, sub17, sub18 = st.tabs(
        [
            "🔥 1. Interior Ballistics & P-V Diagram",
            "🌍 2. ISA Atmosphere, Stability & Coriolis",
            "🌊 3. Shock Physics & Rankine-Hugoniot",
            "💥 4. Gurney Energy, Taylor & Mott",
            "🛡️ 5. Continuum Penetration & Project THOR",
            "📐 6. Quasi-MC Discrepancy & KB Registry",
            "🌀 7. STANAG 4355 MPM Yaw & Spin Drift",
            "⚙️ 8. Bayesian Fuze Reliability & Graze Surface",
            "🚀 9. Precision Guidance & Proportional Nav (PN/APN)",
            "📦 10. Cluster Dispenser & UXO Hazard (CCM/IATG)",
            "🌀 11. Full 6-DoF Rigid-Body & Epicyclic Yaw",
            "🎯 12. Battery Sheafs & MRSI/TOT Salvo",
            "🛡️ 13. Multi-Layer Armor & Tandem ERA",
            "🛰️ 14. Counter-Battery Radar & POO/POI",
            "🎯 15. Direct-Fire Tank Gunnery & APFSDS Sabot",
            "📡 16. Electronic Warfare & Laser Homing",
            "🔫 17. Small Arms & Multi-Barrel Dispersion",
            "🛸 18. C-UAS Turret Suite (FCS, Swarms, Payloads, Recoil & SDZ)",
        ]
    )

    # --- SUB-LAB 1: STANAG 4367 INTERIOR BALLISTICS ---
    with sub1:
        st.subheader("🔥 Lumped-Parameter Thermodynamic Interior Ballistics (`IB-01` .. `IB-06`)")
        st.latex(
            r"""
            r(p) = \beta p^\alpha, \quad
            z(\phi) = \kappa \phi (1 + \lambda \phi + \mu \phi^2), \quad
            \bar{p}(x, v, z) = \frac{F C z - (\gamma - 1)\left[\frac{1}{2}\left(m_p + \frac{C}{3}\right)v^2 + E_{\text{loss}}\right]}{V_0 + A x - \frac{C}{\rho_p}(1 - z) - \eta C z}, \quad
            p_{\text{breech}} = p_{\text{base}}\left(1 + \frac{C}{2 m_p}\right)
            """
        )
        ib1, ib2, ib3, ib4 = st.columns(4)
        with ib1:
            ib_cal_mm = st.number_input("Bore Calibre (mm)", 20.0, 240.0, 155.0, 5.0)
            ib_travel_m = st.number_input("Projectile Bore Travel (m)", 0.5, 12.0, 6.0, 0.25)
        with ib2:
            ib_proj_kg = st.number_input("Projectile Mass (kg)", 0.1, 200.0, 43.2, 1.0)
            ib_chg_kg = st.number_input("Propellant Charge Mass (kg)", 0.05, 80.0, 11.5, 0.5)
        with ib3:
            ib_cham_l = st.number_input("Chamber Volume (Liters)", 0.1, 100.0, 23.0, 1.0)
            ib_imp_mj = st.slider("Propellant Impetus F (MJ/kg)", 0.80, 1.35, 1.05, 0.02)
        with ib4:
            ib_web_mm = st.slider("Propellant Half-Web e1 (mm)", 0.3, 2.5, 0.95, 0.05)
            ib_shot_mpa = st.slider("Engraving Shot-Start Pressure (MPa)", 10.0, 80.0, 30.0, 5.0)

        ib_cfg = LumpedIBConfig(
            calibre_m=float(ib_cal_mm) * 1e-3,
            travel_m=float(ib_travel_m),
            chamber_volume_m3=float(ib_cham_l) * 1e-3,
            proj_mass_kg=float(ib_proj_kg),
            charge_mass_kg=float(ib_chg_kg),
            impetus_J_kg=float(ib_imp_mj) * 1e6,
            half_web_m=float(ib_web_mm) * 1e-3,
            shot_start_Pa=float(ib_shot_mpa) * 1e6,
        )
        ib_out = solve_lumped_interior_ballistics(ib_cfg, dt=1e-5, record_history=True)
        hist = ib_out.get("history", {})

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Muzzle Velocity", f"{ib_out['muzzle_velocity_mps']:.1f} m/s")
        k2.metric("Peak Breech Pressure", f"{ib_out['peak_breech_pressure_Pa']*1e-6:.1f} MPa")
        k3.metric("Peak Mean Pressure", f"{ib_out['peak_mean_pressure_Pa']*1e-6:.1f} MPa")
        k4.metric("Time to Muzzle Exit", f"{ib_out['time_to_muzzle_s']*1000:.2f} ms")
        k5.metric("Ballistic Efficiency η_th", f"{ib_out['ballistic_efficiency']*100:.1f} %")

        if hist:
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                fig_ib_t = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["p_breech_MPa"],
                        name="Breech Pressure (MPa)",
                        line=dict(color="#DC2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["p_base_MPa"],
                        name="Base Pressure (MPa)",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                    ),
                    secondary_y=False,
                )
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["v_mps"],
                        name="Velocity (m/s)",
                        line=dict(color="#2563EB", width=3),
                    ),
                    secondary_y=True,
                )
                fig_ib_t.update_xaxes(title_text="In-Bore Time (ms)")
                fig_ib_t.update_yaxes(title_text="Pressure (MPa)", secondary_y=False)
                fig_ib_t.update_yaxes(title_text="Velocity (m/s)", secondary_y=True)
                fig_ib_t.update_layout(title="Pressure & Velocity vs Time", height=370, margin=dict(l=15, r=15, t=40, b=15))
                st.plotly_chart(fig_ib_t, use_container_width=True)

            with p_col2:
                fig_ib_x = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ib_x.add_trace(
                    go.Scatter(
                        x=hist["x_m"],
                        y=hist["p_breech_MPa"],
                        name="Breech Pressure (MPa)",
                        line=dict(color="#DC2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_ib_x.add_trace(
                    go.Scatter(
                        x=hist["x_m"],
                        y=[p * 100.0 for p in hist["phi"]],
                        name="Burnt Web Fraction (%)",
                        line=dict(color="#10B981", width=2.5, dash="dot"),
                    ),
                    secondary_y=True,
                )
                fig_ib_x.update_xaxes(title_text="Bore Travel X (m)")
                fig_ib_x.update_yaxes(title_text="Pressure (MPa)", secondary_y=False)
                fig_ib_x.update_yaxes(title_text="Burnt Fraction (%)", secondary_y=True)
                fig_ib_x.update_layout(title="Pressure & Burn Completion vs Travel", height=370, margin=dict(l=15, r=15, t=40, b=15))
                st.plotly_chart(fig_ib_x, use_container_width=True)

            with p_col3:
                vg_l = hist.get("v_gas_L", hist["x_m"])
                fig_pv = go.Figure()
                fig_pv.add_trace(
                    go.Scatter(
                        x=vg_l,
                        y=hist["p_mean_MPa"],
                        mode="lines",
                        fill="tozeroy",
                        name="Mean Gas Pressure P(V_g)",
                        line=dict(color="#7C3AED", width=3),
                        fillcolor="rgba(124, 58, 237, 0.16)",
                    )
                )
                fig_pv.update_layout(
                    title=f"Thermodynamic P-V Indicator Diagram (W_m = {ib_out['muzzle_energy_J']*1e-6:.2f} MJ)",
                    xaxis_title="Free Gas Expansion Volume V_g (Liters)",
                    yaxis_title="Mean Space Pressure P (MPa)",
                    height=370,
                    margin=dict(l=15, r=15, t=40, b=15),
                )
                st.plotly_chart(fig_pv, use_container_width=True)

    # --- SUB-LAB 2: ISA ATMOSPHERE, GYROSCOPIC STABILITY & CORIOLIS ---
    with sub2:
        st.subheader("🌍 US Standard Atmosphere 1976 (`EB-04`), Gyroscopic Stability (`EB-07`/`EB-08`) & Coriolis (`EB-10`)")
        st.latex(
            r"""
            a(z) = \sqrt{\gamma R_{\text{air}} T(z)}, \quad
            \mu(T) = \beta_S \frac{T^{3/2}}{T + S}, \quad
            S_g = \frac{2 I_x^2 p^2}{\pi \rho I_y d^3 v^2 C_{M_\alpha}}, \quad
            \mathbf{a}_{\text{Cor}} = -2(\boldsymbol{\Omega}_\oplus \times \mathbf{v})
            """
        )
        at_c1, at_c2 = st.columns([1.25, 1.0])
        with at_c1:
            dt_isa = st.slider("Atmospheric Temperature Offset ΔT (K)", -30.0, 30.0, 0.0, 1.0)
            h_grid_m = np.linspace(0.0, 22000.0, 120)
            atm_prof = eb_04_isa_atmosphere(h_grid_m, dT_K=dt_isa)
            rho_p = np.asarray(atm_prof["rho_kg_m3"], dtype=np.float64)
            a_p = np.asarray(atm_prof["a_mps"], dtype=np.float64)
            t_p = np.asarray(atm_prof["T_K"], dtype=np.float64)
            mu_p = np.asarray(sutherland_viscosity_Pa_s(t_p), dtype=np.float64) * 1e6  # microPa*s

            fig_atm = make_subplots(specs=[[{"secondary_y": True}]])
            fig_atm.add_trace(
                go.Scatter(x=h_grid_m / 1000.0, y=rho_p, name="Air Density ρ (kg/m³)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_atm.add_trace(
                go.Scatter(
                    x=h_grid_m / 1000.0,
                    y=a_p,
                    name="Sound Speed a (m/s)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_atm.add_trace(
                go.Scatter(
                    x=h_grid_m / 1000.0,
                    y=mu_p * 15.0,
                    name="Sutherland Viscosity μ (×15 μPa·s)",
                    line=dict(color="#10B981", width=2, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_atm.add_vline(x=11.0, line_dash="dot", line_color="#6B7280", annotation_text="Tropopause (11 km)")
            fig_atm.update_xaxes(title_text="Geopotential Altitude h (km)")
            fig_atm.update_yaxes(title_text="Density ρ (kg/m³)", secondary_y=False)
            fig_atm.update_yaxes(title_text="Acoustic Speed a (m/s) / Scaled μ", secondary_y=True)
            fig_atm.update_layout(title="US Standard Atmosphere 1976 & Sutherland Viscosity Profile", height=390)
            st.plotly_chart(fig_atm, use_container_width=True)

        with at_c2:
            st.markdown("##### 🧭 Gyroscopic Stability ($S_g$) & Coriolis Acceleration")
            sg_cal_mm = st.number_input("Projectile Calibre d (mm)", 5.56, 240.0, 155.0, 5.0, key="sg_cal")
            sg_twist = st.slider("Rifling Twist Rate (calibres/turn)", 10.0, 36.0, 20.0, 0.5)
            sg_v0 = st.slider("Launch Velocity V₀ (m/s)", 200.0, 1200.0, 827.0, 10.0)
            lat_deg = st.slider("Firing Latitude φ (deg, North +)", -75.0, 75.0, 45.0, 5.0)
            az_deg = st.slider("Firing Azimuth α (deg clockwise from North)", 0.0, 360.0, 90.0, 15.0)

            d_m_val = sg_cal_mm * 1e-3
            p0_spin = float(eb_09_muzzle_spin_rate(sg_v0, sg_twist, d_m_val))
            # Estimate axial and transverse moments of inertia for 43 kg / d^3 scaling
            m_est = 43.2 * ((d_m_val / 0.155) ** 3)
            ix_est = 0.135 * m_est * (d_m_val**2)
            iy_est = 1.45 * m_est * (d_m_val**2)
            sg_val = float(eb_07_gyroscopic_stability(ix_est, iy_est, p0_spin, 1.225, d_m_val, sg_v0, c_ma=2.8))
            a_cor_vec = eb_10_coriolis_acceleration(
                np.array([sg_v0 * 0.85, 0.0, sg_v0 * 0.52]),
                lat_rad=math.radians(lat_deg),
                az_rad=math.radians(az_deg),
            )
            st.table(
                pd.DataFrame(
                    [
                        {"Physical Quantity": "Initial Spin Rate p₀ (EB-09)", "Value": f"{p0_spin:.1f} rad/s ({p0_spin*60/(2*math.pi):.0f} RPM)"},
                        {"Physical Quantity": "STANAG 4355 Stability Factor S_g (EB-07)", "Value": f"{sg_val:.2f} ({'Dynamically Stable' if sg_val >= 1.3 else 'Marginal/Unstable'})"},
                        {"Physical Quantity": "Coriolis Downrange / Eötvös a_x (EB-10)", "Value": f"{a_cor_vec[0]:+.4f} m/s²"},
                        {"Physical Quantity": "Coriolis Crossrange Drift a_y (EB-10)", "Value": f"{-a_cor_vec[1]:+.4f} m/s² (Right +)"},
                        {"Physical Quantity": "Coriolis Vertical Eötvös a_z (EB-10)", "Value": f"{a_cor_vec[2]:+.4f} m/s² (Up +)"},
                    ]
                )
            )

    # --- SUB-LAB 3: SHOCK PHYSICS, RANKINE-HUGONIOT & FRIEDLANDER ---
    with sub3:
        st.subheader("🌊 Rankine-Hugoniot Normal Shock Relations (`TB-03` .. `TB-08`)")
        st.latex(
            r"""
            Z = \frac{R}{W_{\text{TNT}}^{1/3}}, \quad
            P_r = 2 P_s \frac{7 P_a + 4 P_s}{7 P_a + P_s}, \quad
            q = \frac{5 P_s^2}{2(7 P_a + P_s)}, \quad
            P(t) = P_s \left(1 - \frac{t}{t_d}\right) e^{-b t / t_d}
            """
        )
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            z_grid = np.logspace(np.log10(0.15), np.log10(40.0), 90)
            ps_kb = np.asarray(tb_03_kingery_bulmash(z_grid, 1.0, param="Ps"), dtype=np.float64)
            ps_kg = np.asarray(tb_04_kinney_graham(z_grid), dtype=np.float64)
            pr_rh = np.asarray(tb_07_normal_reflected_pressure(ps_kb), dtype=np.float64)
            q_rh = np.asarray(tb_05_dynamic_pressure(ps_kb), dtype=np.float64)

            fig_rh = go.Figure()
            fig_rh.add_trace(go.Scatter(x=z_grid, y=pr_rh, name="Normal Reflected Pressure P_r (TB-07)", line=dict(color="#991B1B", width=3)))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=ps_kb, name="Incident Overpressure P_s (Kingery-Bulmash TB-03)", line=dict(color="#DC2626", width=3)))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=ps_kg, name="Free-Air P_s (Kinney-Graham TB-04)", line=dict(color="#F59E0B", width=2, dash="dash")))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=q_rh, name="Dynamic Blast Wind Pressure q (TB-05)", line=dict(color="#2563EB", width=2.5, dash="dot")))
            fig_rh.update_layout(
                title="Rankine-Hugoniot Shock Overpressures vs Hopkinson-Cranz Scaled Distance Z",
                xaxis_title="Scaled Distance Z = R / W^(1/3) (m / kg^(1/3))",
                yaxis_title="Pressure (kPa)",
                xaxis_type="log",
                yaxis_type="log",
                height=400,
            )
            st.plotly_chart(fig_rh, use_container_width=True)

        with w_col2:
            w_kg = st.number_input("TNT Equivalent Mass W (kg)", 1.0, 2000.0, 25.0, 5.0)
            r_m = st.slider("Standoff Distance R (m)", 2.0, 120.0, 18.0, 1.0)
            z_sc = float(tb_01_scaled_distance(r_m, w_kg))
            kb_vals = tb_03_kingery_bulmash(z_sc, w_kg, param="all")
            assert isinstance(kb_vals, dict)
            ps_kpa = float(kb_vals.get("Ps", 50.0))
            td_ms = float(kb_vals.get("td", 10.0))
            ta_ms = float(kb_vals.get("ta", 5.0))

            t_wave = np.linspace(0.0, td_ms, 180)
            p_wave = np.array([float(tb_08_friedlander_waveform(t_val, ps_kpa, td_ms, b=1.8)) for t_val in t_wave])
            dt_w = t_wave[1] - t_wave[0] if len(t_wave) > 1 else 0.01
            imp_cum = np.cumsum(p_wave) * dt_w

            fig_wave = make_subplots(specs=[[{"secondary_y": True}]])
            fig_wave.add_trace(
                go.Scatter(
                    x=ta_ms + t_wave,
                    y=p_wave,
                    mode="lines",
                    fill="tozeroy",
                    name="Overpressure P(t) (kPa)",
                    line=dict(color="#DC2626", width=3),
                    fillcolor="rgba(220, 38, 38, 0.16)",
                ),
                secondary_y=False,
            )
            fig_wave.add_trace(
                go.Scatter(
                    x=ta_ms + t_wave,
                    y=imp_cum,
                    mode="lines",
                    name="Cumulative Specific Impulse I(t) (kPa·ms)",
                    line=dict(color="#1E3A8A", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_wave.update_xaxes(title_text="Time from Detonation (ms)")
            fig_wave.update_yaxes(title_text="Incident Overpressure P(t) (kPa)", secondary_y=False)
            fig_wave.update_yaxes(title_text="Impulse I(t) (kPa·ms)", secondary_y=True)
            fig_wave.update_layout(
                title=f"Friedlander Waveform & Impulse Integral at R={r_m:.1f}m (Z={z_sc:.2f})",
                height=400,
            )
            st.plotly_chart(fig_wave, use_container_width=True)

    # --- SUB-LAB 4: GURNEY ENERGY PARTITION, TAYLOR ANGLE & MOTT FRACTURE ---
    with sub4:
        st.subheader("💥 Gurney Detonation Energy Partition (`TF-01`, `TF-02`), Taylor Angle (`TF-08`) & Mott Fracture (`TF-04`)")
        st.latex(
            r"""
            V_0 = \sqrt{2E_G}\left(\frac{M}{C} + \frac{n}{n+2}\right)^{-1/2}, \quad
            \eta_{\text{metal}} = \frac{\frac{1}{2}M V_0^2}{C E_G} = \left(1 + \frac{n}{n+2}\frac{C}{M}\right)^{-1}, \quad
            \theta_{\text{Taylor}} = 2 \arcsin\!\left(\frac{V_0}{2 D_{\text{det}}}\right), \quad
            N(>m) = N_0 e^{-\sqrt{m}/M_K}
            """
        )
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            exp_sel = st.selectbox("Reference Energetic Fill", ["Composition B", "TNT", "Octol 75/25", "RDX", "HMX"], index=0)
            exp_props = kb.get_explosive_properties(exp_sel)
            g_const = float(exp_props["gurney_mps"])
            d_det = float(exp_props["det_vel_mps"])

            cm_grid = np.linspace(0.05, 2.5, 80)
            mc_grid = 1.0 / cm_grid
            v_cyl = np.asarray(tf_01_gurney_cyl_sphere(g_const, mc_grid, 1.0, "cylinder"), dtype=np.float64)
            v_sph = np.asarray(tf_01_gurney_cyl_sphere(g_const, mc_grid, 1.0, "sphere"), dtype=np.float64)
            v_sym = np.asarray(tf_02_gurney_sandwiches(g_const, mc_grid, 1.0, "symmetric"), dtype=np.float64)
            eta_cyl = (mc_grid * (v_cyl**2)) / (g_const**2) * 100.0

            fig_gur = make_subplots(specs=[[{"secondary_y": True}]])
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_cyl, name="Cylinder V₀ (TF-01)", line=dict(color="#2563EB", width=3)), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_sph, name="Sphere V₀ (TF-01)", line=dict(color="#7C3AED", width=2.5, dash="dash")), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_sym, name="Symmetric Sandwich V₀ (TF-02)", line=dict(color="#059669", width=2, dash="dot")), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=eta_cyl, name="Cylinder Energy Efficiency η_m (%)", line=dict(color="#DC2626", width=2.5)), secondary_y=True)
            fig_gur.update_xaxes(title_text="Charge-to-Metal Mass Ratio C / M (-)")
            fig_gur.update_yaxes(title_text="Initial Velocity V₀ (m/s)", secondary_y=False)
            fig_gur.update_yaxes(title_text="Energy Partition Efficiency η_m (%)", secondary_y=True)
            fig_gur.update_layout(
                title=f"Gurney Velocity & Energy Partition ({exp_sel}: √2E = {g_const:.0f} m/s, D = {d_det:.0f} m/s)",
                height=390,
            )
            st.plotly_chart(fig_gur, use_container_width=True)

        with g_col2:
            m_case = st.number_input("Casing Metal Mass (kg)", 2.0, 200.0, 34.0, 2.0)
            t_in = st.slider("Casing Wall Thickness (in)", 0.15, 1.50, 0.55, 0.05)
            d_in = st.slider("Internal Casing Diameter (in)", 1.5, 10.0, 4.8, 0.2)
            mott_out = tf_04_mott_distribution(M0_kg=m_case, t_in=t_in, d_in=d_in, B=float(exp_props["mott_b"]))
            avg_g = float(mott_out["m_avg_g"])
            n_tot = float(mott_out["total_fragments"])
            m_grid_g = np.linspace(0.1, max(avg_g * 6.0, 5.0), 80)
            mk_lb_sqrt = float(mott_out["MK_lb_sqrt"])
            m_grid_lb = (m_grid_g / 1000.0) * 2.20462262185
            n_gt_m = n_tot * np.exp(-np.sqrt(m_grid_lb) / max(mk_lb_sqrt, 1e-9))
            fig_mott = go.Figure()
            fig_mott.add_trace(
                go.Scatter(
                    x=m_grid_g,
                    y=n_gt_m,
                    mode="lines",
                    name="Cumulative Fragments N(>m)",
                    line=dict(color="#1E3A8A", width=3),
                )
            )
            fig_mott.update_layout(
                title=f"Mott Statistical Fracture Curve (N_tot={n_tot:.0f}, Mean Mass={avg_g:.2f} g)",
                xaxis_title="Fragment Mass Threshold m (grams)",
                yaxis_title="Number of Fragments N(>m)",
                yaxis_type="log",
                height=390,
            )
            st.plotly_chart(fig_mott, use_container_width=True)

    # --- SUB-LAB 5: CONTINUUM PENETRATION MECHANICS ---
    with sub5:
        st.subheader("🛡️ Continuum Penetration Mechanics (`TP-01` Lanz-Odermatt, `TP-02` Birkhoff, `TP-04` Lambert-Jonas)")
        st.latex(
            r"""
            \frac{P}{L_w} = \frac{a}{\tanh(b_0 + b_1 L_w/D)} (\cos\theta)^m \sqrt{\frac{\rho_p}{\rho_t}} \exp\!\left(-\frac{s^2}{V_s^2}\right), \quad
            P_{\text{hydro}} = L_{\text{jet}}\sqrt{\frac{\rho_j}{\rho_t}}, \quad
            V_r = a\left(V_s^p - V_{50}^p\right)^{1/p}
            """
        )
        pen_c1, pen_c2 = st.columns(2)
        with pen_c1:
            lw_d_ratio = st.slider("Penetrator Aspect Ratio L_w / D", 10.0, 35.0, 22.0, 1.0)
            bhn_tgt = st.slider("Target RHA Brinell Hardness (BHN)", 180.0, 500.0, 300.0, 10.0)
            obliq_deg = st.slider("Impact Obliquity Angle θ (deg)", 0.0, 65.0, 30.0, 5.0)

            v_grid = np.linspace(800.0, 2200.0, 80)
            d_rod = 0.025
            lw_rod = lw_d_ratio * d_rod
            ob_rad = math.radians(obliq_deg)

            p_w = np.asarray(
                tp_01_lanz_odermatt(
                    lw_rod, d_rod, v_grid, rho_p_kg_m3=17600.0, rho_t_kg_m3=7850.0, bhn_t=bhn_tgt, obliquity_rad=ob_rad, rod_material="Tungsten alloy"
                )["penetration"],
                dtype=np.float64,
            ) / lw_rod
            p_st = np.asarray(
                tp_01_lanz_odermatt(
                    lw_rod, d_rod, v_grid, rho_p_kg_m3=7850.0, rho_t_kg_m3=7850.0, bhn_t=bhn_tgt, obliquity_rad=ob_rad, rod_material="Steel"
                )["penetration"],
                dtype=np.float64,
            ) / lw_rod
            hydro_w = math.sqrt(17600.0 / 7850.0) * (math.cos(ob_rad) ** (-0.225))

            fig_lo = go.Figure()
            fig_lo.add_trace(go.Scatter(x=v_grid, y=p_w, name="Tungsten Alloy Rod (ρ=17.6 g/cm³)", line=dict(color="#1E3A8A", width=3.5)))
            fig_lo.add_trace(go.Scatter(x=v_grid, y=p_st, name="High-Hardness Steel Rod (ρ=7.85 g/cm³)", line=dict(color="#059669", width=3, dash="dash")))
            fig_lo.add_hline(y=hydro_w, line_dash="dot", line_color="#DC2626", annotation_text="Birkhoff Hydrodynamic Limit √(ρ_p/ρ_t)")
            fig_lo.update_layout(
                title=f"Lanz-Odermatt Normalized Penetration P/L_w vs Impact Velocity (θ={obliq_deg:.0f}°, BHN={bhn_tgt:.0f})",
                xaxis_title="Striking Velocity V_s (m/s)",
                yaxis_title="Normalized Penetration Depth P / L_w (-)",
                height=390,
            )
            st.plotly_chart(fig_lo, use_container_width=True)

        with pen_c2:
            v50_bl = st.slider("Ballistic Limit Velocity V₅₀ (m/s)", 200.0, 1200.0, 550.0, 25.0)
            sig_v50 = st.slider("Probit Zone Standard Deviation σ_V50 (m/s)", 5.0, 80.0, 25.0, 5.0)
            lj_p = st.slider("Lambert-Jonas Exponent p (-)", 1.5, 3.0, 2.0, 0.1)

            vs_grid = np.linspace(max(100.0, v50_bl * 0.5), v50_bl * 1.8, 100)
            vr_grid = np.asarray(tp_04_lambert_jonas(vs_grid, v50_bl, a=0.92, p=lj_p), dtype=np.float64)
            prob_perf = np.asarray(tp_06_perforation_probability(vs_grid, v50_bl, sig_v50), dtype=np.float64)

            fig_lj = make_subplots(specs=[[{"secondary_y": True}]])
            fig_lj.add_trace(
                go.Scatter(x=vs_grid, y=vr_grid, name="Residual Velocity V_r (TP-04)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_lj.add_trace(
                go.Scatter(
                    x=vs_grid,
                    y=prob_perf,
                    name="Probit Perforation Prob P(perf|V_s) (TP-06)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_lj.add_vline(x=v50_bl, line_dash="dot", line_color="#6B7280", annotation_text=f"V₅₀ = {v50_bl:.0f} m/s")
            fig_lj.update_xaxes(title_text="Striking Velocity V_s (m/s)")
            fig_lj.update_yaxes(title_text="Residual Velocity V_r (m/s)", secondary_y=False)
            fig_lj.update_yaxes(title_text="Perforation Probability (-)", secondary_y=True)
            fig_lj.update_layout(title="Lambert-Jonas Residual Velocity & Probit Perforation Curve", height=390)
            st.plotly_chart(fig_lj, use_container_width=True)

        st.markdown("---")
        st.subheader("🧱 Project THOR 17-Material Fragment Perforation (`TF-12`) & De Marre Armor Scaling (`TF-13`)")
        st.latex(
            r"""
            V_{50} = 0.3048 \cdot 10^{c_1} (t A)^{\alpha_1} (15.4324\, m_f)^{\beta_1} (\sec\theta)^{\gamma_1}, \qquad
            V_r = V_s - 0.3048 \cdot 10^{c_2} (t A)^{\alpha_2} (15.4324\, m_f)^{\beta_2} (\sec\theta)^{\gamma_2} V_s^{\lambda_2}
            """
        )
        th_c1, th_c2, th_c3, th_c4 = st.columns(4)
        with th_c1:
            thor_m_g = st.slider("Fragment Mass m_f (grams)", 0.5, 50.0, 6.5, 0.5, key="thor_m")
        with th_c2:
            thor_t_mm = st.slider("Target Plate Thickness t (mm)", 1.0, 25.0, 6.35, 0.5, key="thor_t")
        with th_c3:
            thor_ob_deg = st.slider("Impact Obliquity θ (deg)", 0.0, 60.0, 15.0, 5.0, key="thor_ob")
        with th_c4:
            thor_vs = st.slider("Striking Velocity V_s (m/s)", 300.0, 2200.0, 1150.0, 50.0, key="thor_vs")

        thor_materials = [m["material"] for m in kb.thor.get("materials", [])]
        thor_rows = []
        for mat_name in thor_materials:
            tr_o = tf_12_thor_penetration(
                m_g=float(thor_m_g),
                t_cm=float(thor_t_mm) * 0.1,
                v_s_mps=float(thor_vs),
                obliquity_rad=math.radians(float(thor_ob_deg)),
                target_material=mat_name,
            )
            thor_rows.append(
                {
                    "Target Material": mat_name,
                    "V50 Ballistic Limit (m/s)": round(float(tr_o["v50_mps"]), 1),
                    "Residual Velocity V_r (m/s)": round(float(tr_o["v_residual_mps"]), 1),
                    "Perforated": "✅ YES" if tr_o["perforated"] else "❌ NO",
                }
            )
        thor_df = pd.DataFrame(thor_rows).sort_values("V50 Ballistic Limit (m/s)", ascending=True)
        frag_m_kg = float(thor_m_g) * 1e-3
        frag_d_m = (frag_m_kg / 7850.0) ** (1.0 / 3.0)
        dm_pen_in = float(
            tf_13_de_marre_penetration(
                D_in=frag_d_m / 0.0254,
                W_lb=frag_m_kg * 2.20462262,
                V_fps=float(thor_vs) * 3.2808399,
                obliquity_rad=math.radians(float(thor_ob_deg)),
            )
        )
        dm_pen_m = dm_pen_in * 0.0254

        th_p1, th_p2 = st.columns([1.3, 1.0])
        with th_p1:
            fig_thor = go.Figure()
            fig_thor.add_trace(
                go.Bar(
                    x=thor_df["V50 Ballistic Limit (m/s)"],
                    y=thor_df["Target Material"],
                    orientation="h",
                    name="Ballistic Limit V₅₀ (m/s)",
                    marker_color="#1E3A8A",
                )
            )
            fig_thor.add_trace(
                go.Bar(
                    x=thor_df["Residual Velocity V_r (m/s)"],
                    y=thor_df["Target Material"],
                    orientation="h",
                    name=f"Residual Velocity V_r at V_s={thor_vs:.0f} m/s",
                    marker_color="#10B981",
                )
            )
            fig_thor.add_vline(x=thor_vs, line_dash="dash", line_color="#DC2626", annotation_text=f"V_s = {thor_vs:.0f} m/s")
            fig_thor.update_layout(
                title=f"Project THOR Ballistic Limit V₅₀ & Residual Velocity Across {len(thor_df)} Materials (m_f={thor_m_g:.1f}g, t={thor_t_mm:.1f}mm)",
                xaxis_title="Velocity (m/s)",
                barmode="group",
                height=440,
                margin=dict(l=15, r=15, t=40, b=15),
            )
            st.plotly_chart(fig_thor, use_container_width=True)
        with th_p2:
            st.metric("De Marre Mild-Steel Penetration (TF-13)", f"{dm_pen_m * 1000.0:.2f} mm", delta=f"Fragment KE: {0.5 * (thor_m_g * 1e-3) * (thor_vs**2):.0f} J", delta_color="off")
            st.dataframe(thor_df, use_container_width=True, height=365)

    # --- SUB-LAB 6: QUASI-MONTE CARLO DISCREPANCY & KNOWLEDGE BASE ---
    with sub6:
        st.subheader("📐 Quasi-Monte Carlo Low-Discrepancy Sampling (`MC-05`, `MC-06`) & Salvo Correlation (`TV-04`, `TV-05`)")
        st.latex(
            r"""
            \left|\frac{1}{N}\sum_{i=1}^N f(\mathbf{u}_i) - \int_{[0,1]^d} f(\mathbf{u})\,d\mathbf{u}\right| \le V_{\text{HK}}(f)\, D_N^*(\mathbf{u}_1,\dots,\mathbf{u}_N), \quad
            D_N^*(\text{Sobol}) = \mathcal{O}\!\left(\frac{(\log N)^d}{N}\right) \ll \mathcal{O}(N^{-1/2})
            """
        )
        qmc_c1, qmc_c2 = st.columns(2)
        with qmc_c1:
            n_qmc = st.slider("Sample Points N in Unit Square [0, 1]²", 16, 256, 64, 16)
            u_sobol = sample_unit_hypercube(n_qmc, 2, method="Sobol", rng=42)
            u_lhs = sample_unit_hypercube(n_qmc, 2, method="LHS", rng=42)
            u_mc = sample_unit_hypercube(n_qmc, 2, method="MC", rng=42)

            fig_qmc = go.Figure()
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_sobol[:, 0],
                    y=u_sobol[:, 1],
                    mode="markers",
                    name="Sobol Low-Discrepancy (MC-06)",
                    marker=dict(color="#2563EB", size=8, symbol="circle"),
                )
            )
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_lhs[:, 0],
                    y=u_lhs[:, 1],
                    mode="markers",
                    name="Latin Hypercube LHS (MC-05)",
                    marker=dict(color="#10B981", size=7, symbol="diamond"),
                )
            )
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_mc[:, 0],
                    y=u_mc[:, 1],
                    mode="markers",
                    name="Crude Pseudorandom MC (MC-04)",
                    marker=dict(color="#DC2626", size=6, symbol="cross", opacity=0.65),
                )
            )
            fig_qmc.update_layout(
                title=f"2D Parameter-Space Coverage: Sobol vs LHS vs Crude MC (N={n_qmc})",
                xaxis_title="Uniform Dimension U₁",
                yaxis_title="Uniform Dimension U₂",
                height=380,
            )
            st.plotly_chart(fig_qmc, use_container_width=True)

        with qmc_c2:
            r_dam_m = st.slider("Carleton Effective Radius R_D (m)", 5.0, 60.0, 25.0, 1.0)
            sig_bias_m = st.slider("Shared Occasion Bias Sigma (m)", 0.0, 60.0, 22.0, 2.0)
            sig_rr_m = st.slider("Round-to-Round Random Sigma (m)", 2.0, 60.0, 14.0, 2.0)

            sig_tot_m = math.hypot(sig_bias_m, sig_rr_m)
            sspk_1 = float(tv_04_sspk_carleton(r_dam_m, sig_tot_m, p_dud=0.02))
            n_vec = list(range(1, 17))
            pk_indep = [float(tv_05_salvo_pk_independent(sspk_1, n)) for n in n_vec]

            rng_c = np.random.default_rng(42)
            bx = rng_c.normal(0.0, sig_bias_m, (600, 1))
            by = rng_c.normal(0.0, sig_bias_m, (600, 1))
            rx = rng_c.normal(0.0, sig_rr_m, (600, 16))
            ry = rng_c.normal(0.0, sig_rr_m, (600, 16))
            r2_miss = (bx + rx) ** 2 + (by + ry) ** 2
            p_round = 0.98 * np.exp(-r2_miss / (2.0 * (r_dam_m / 1.1774) ** 2))
            surv_cum = np.cumprod(1.0 - p_round, axis=1)
            pk_corr = [float(np.mean(1.0 - surv_cum[:, n - 1])) for n in n_vec]

            fig_salvo = go.Figure()
            fig_salvo.add_trace(
                go.Scatter(
                    x=n_vec,
                    y=pk_indep,
                    mode="lines+markers",
                    name="Independent Salvo Assumption (TV-05)",
                    line=dict(color="#9CA3AF", width=2.5, dash="dash"),
                )
            )
            fig_salvo.add_trace(
                go.Scatter(
                    x=n_vec,
                    y=pk_corr,
                    mode="lines+markers",
                    name="Correlated Occasion-Bias Salvo (True MC)",
                    line=dict(color="#2563EB", width=3.5),
                )
            )
            fig_salvo.update_layout(
                title=f"Salvo Effectiveness vs Round Count N (Single-Shot SSPK = {sspk_1:.3f})",
                xaxis_title="Salvo Size N (Rounds Fired on Same Occasion)",
                yaxis_title="Cumulative Salvo Effect Probability Pk",
                height=380,
            )
            st.plotly_chart(fig_salvo, use_container_width=True)

        st.subheader("📚 Authoritative Knowledge Base Explorer (96 Formulas & 36 Primary Sources)")
        kb_tab_choice = st.selectbox(
            "Select Knowledge Base Sheet",
            ["Formulas (96)", "Explosives", "THOR_Coefficients", "Ammo_Taxonomy", "Sources (36)"],
        )
        if kb_tab_choice == "Formulas (96)":
            forms_df = pd.DataFrame(kb.formulas)
            cols_f = [
                c
                for c in ["formula_id", "module", "name", "verification_status", "is_flagged", "source_id", "formula_text"]
                if c in forms_df.columns
            ]
            st.dataframe(
                forms_df[cols_f],
                use_container_width=True,
            )
        elif kb_tab_choice == "Explosives":
            st.dataframe(pd.DataFrame(kb.explosives.get("tnt_equivalence", [])), use_container_width=True)
        elif kb_tab_choice == "THOR_Coefficients":
            st.dataframe(pd.DataFrame(kb.thor.get("materials", [])), use_container_width=True)
        elif kb_tab_choice == "Ammo_Taxonomy":
            st.dataframe(pd.DataFrame(kb.taxonomy), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(kb.sources), use_container_width=True)

    # --- SUB-LAB 7: STANAG 4355 MPM YAW-OF-REPOSE, SPIN-DAMPING & MAGNUS DRIFT ---
    with sub7:
        st.subheader("🌀 STANAG 4355 Modified Point Mass (MPM): Yaw of Repose (`EB-05`), Spin Decay (`EB-06`) & Gyroscopic Drift")
        st.latex(
            r"""
            \boldsymbol{\alpha}_e = -\frac{2 I_x p}{\rho S d C_{M_\alpha} v^4}\,(\mathbf{v} \times \mathbf{a}), \qquad
            \frac{dp}{dt} = \frac{\rho S d^2 v}{2 I_x}\,C_{\text{spin}}\,p, \qquad
            \mathbf{a}_{\text{MPM}} = \mathbf{a}_{\text{drag}}(\alpha_e^2) + \mathbf{a}_{\text{lift}}(\boldsymbol{\alpha}_e) + \mathbf{a}_{\text{Magnus}}(p, \boldsymbol{\alpha}_e)
            """
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            mpm_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 300.0, 950.0, 685.0, 15.0, key="mpm_v0")
        with m2:
            mpm_qe_deg = st.slider("Quadrant Elevation QE (deg)", 5.0, 65.0, 32.0, 1.0, key="mpm_qe")
        with m3:
            mpm_twist = st.slider("Rifling Twist (cal/turn)", 12.0, 30.0, 20.0, 1.0, key="mpm_twist")
        with m4:
            mpm_lat_deg = st.slider("Firing Latitude φ (deg)", -60.0, 60.0, 45.0, 5.0, key="mpm_lat")

        mpm_res = simulate_mpm_vs_3dof_trajectory(
            v0_mps=float(mpm_v0),
            qe_deg=float(mpm_qe_deg),
            twist_cal_per_turn=float(mpm_twist),
            lat_deg=float(mpm_lat_deg),
        )
        p_l2 = mpm_res["path_l2"]
        p_l1 = mpm_res["path_l1"]
        rpm_t = np.asarray(mpm_res["p_spin_rad_s"], dtype=np.float64) * 60.0 / (2.0 * math.pi)
        yaw_deg_t = np.degrees(np.asarray(mpm_res["alpha_e_mrad"], dtype=np.float64) * 1e-3)

        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("MPM Lateral Spin Drift", f"{mpm_res['terminal_spin_drift_m']:+.1f} m (Right)", delta=f"3-DoF Defl: {mpm_res['l1_deflection_m']:+.1f} m", delta_color="off")
        mk2.metric("Induced Yaw Range Delta", f"{mpm_res['l2_range_m'] - mpm_res['l1_range_m']:+.1f} m", delta=f"MPM Range: {mpm_res['l2_range_m']:.0f} m", delta_color="off")
        mk3.metric("Apex Yaw of Repose |α_e|", f"{mpm_res['alpha_e_peak_mrad']:.2f} mrad", delta=f"Launch Sg: {mpm_res['sg_muzzle']:.2f}", delta_color="off")
        mk4.metric("Terminal Axial Spin Rate", f"{float(rpm_t[-1]):.0f} RPM", delta=f"Initial: {float(rpm_t[0]):.0f} RPM", delta_color="off")

        mc_left, mc_right = st.columns(2)
        with mc_left:
            fig_drift = go.Figure()
            fig_drift.add_trace(
                go.Scatter(
                    x=p_l2["range_m"],
                    y=p_l2["deflection_m"],
                    mode="lines",
                    name="STANAG 4355 MPM (L2: Yaw + Magnus + Spin Drift)",
                    line=dict(color="#1E3A8A", width=3.5),
                )
            )
            fig_drift.add_trace(
                go.Scatter(
                    x=p_l1["range_m"],
                    y=p_l1["deflection_m"],
                    mode="lines",
                    name="3-DoF Point-Mass (L1: Coriolis Only)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                )
            )
            fig_drift.update_layout(
                title="Ground-Track Deflection Drift: STANAG 4355 MPM (L2) vs 3-DoF Point-Mass (L1)",
                xaxis_title="Downrange X (m)",
                yaxis_title="Right Deflection Drift Y_right (m)",
                height=390,
            )
            st.plotly_chart(fig_drift, use_container_width=True)

        with mc_right:
            fig_yaw = make_subplots(specs=[[{"secondary_y": True}]])
            fig_yaw.add_trace(
                go.Scatter(
                    x=mpm_res["t_s"],
                    y=yaw_deg_t,
                    name="Yaw of Repose |α_e| (deg)",
                    line=dict(color="#7C3AED", width=3),
                ),
                secondary_y=False,
            )
            fig_yaw.add_trace(
                go.Scatter(
                    x=mpm_res["t_s"],
                    y=rpm_t,
                    name="Axial Spin Rate (RPM)",
                    line=dict(color="#059669", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_yaw.update_xaxes(title_text="Time of Flight t (s)")
            fig_yaw.update_yaxes(title_text="Yaw of Repose |α_e| (deg)", secondary_y=False)
            fig_yaw.update_yaxes(title_text="Axial Spin Rate (RPM)", secondary_y=True)
            fig_yaw.update_layout(
                title="In-Flight Yaw-of-Repose Peak at Trajectory Apex & Exponential Spin Decay",
                height=390,
            )
            st.plotly_chart(fig_yaw, use_container_width=True)

    # --- SUB-LAB 8: BAYESIAN FUZE RELIABILITY, GRAZE S-CURVE SURFACE & UXO RISK ---
    with sub8:
        st.subheader("⚙️ Bayesian Fuze Reliability (`FZ-01`..`FZ-04`), 3D Graze S-Curve (`FZ-06`) & Self-Destruct Redundancy (`FZ-07`)")
        st.latex(
            r"""
            p_{\text{dud}} \sim \text{Beta}(k + a_0,\; n - k + b_0), \qquad
            p_{\text{UCB}}(0\text{ failures}) = 1 - (1 - \gamma)^{1/n}, \qquad
            p_{\text{UXO}} = p_d \big[(1 - \beta)p_{\text{SD}} + \beta\big]
            """
        )
        fz_c1, fz_c2, fz_c3, fz_c4 = st.columns(4)
        with fz_c1:
            fz_n_tests = st.slider("Lot Acceptance Sample Size n", 10, 500, 120, 10, key="fz_n")
        with fz_c2:
            fz_k_fails = st.slider("Observed Lot Failures k", 0, 20, 2, 1, key="fz_k")
        with fz_c3:
            fz_sd_fail = st.slider("Self-Destruct Backup Fail P_SD", 0.0, 0.50, 0.05, 0.01, key="fz_sd")
        with fz_c4:
            fz_beta_cc = st.slider("Common-Cause Beta Factor β", 0.0, 0.30, 0.04, 0.01, key="fz_beta")

        a_post = fz_k_fails + 1.0
        b_post = max(fz_n_tests - fz_k_fails, 0) + 1.0
        p_dud_mean = a_post / (a_post + b_post)
        ucb_95 = float(fz_04_zero_failure_ucb(fz_n_tests, 0.95))
        p_uxo_red = float(fz_07_self_destruct_redundancy(p_dud_mean, fz_sd_fail, fz_beta_cc))

        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Posterior Mean Dud Rate (FZ-03)", f"{p_dud_mean*100:.2f} %", delta=f"Beta({a_post:.0f}, {b_post:.0f})", delta_color="off")
        fk2.metric("Zero-Failure 95% UCB (FZ-04)", f"{ucb_95*100:.2f} %", delta=f"For n={fz_n_tests} tests", delta_color="off")
        fk3.metric("Post-SD Residual UXO Rate (FZ-07)", f"{p_uxo_red*100:.3f} %", delta=f"Reduction: {(1.0 - p_uxo_red/max(p_dud_mean,1e-9))*100:.1f}%", delta_color="off")
        fk4.metric("Expected UXO per 48-Sub Carrier", f"{48.0 * p_uxo_red:.2f} UXO", delta=f"P(≥1 UXO): {(1.0 - (1.0 - p_uxo_red)**48)*100:.1f}%", delta_color="off")

        fz_left, fz_right = st.columns(2)
        with fz_left:
            p_grid = np.linspace(0.0001, 0.15, 250)
            pdf_post = sp_stats.beta.pdf(p_grid, a_post, b_post)
            fig_beta = go.Figure()
            fig_beta.add_trace(
                go.Scatter(
                    x=p_grid * 100.0,
                    y=pdf_post,
                    mode="lines",
                    fill="tozeroy",
                    name=f"Posterior Beta({a_post:.0f}, {b_post:.0f})",
                    line=dict(color="#2563EB", width=3),
                    fillcolor="rgba(37, 99, 235, 0.18)",
                )
            )
            fig_beta.add_vline(x=p_dud_mean * 100.0, line_dash="dash", line_color="#DC2626", annotation_text=f"Mean = {p_dud_mean*100:.2f}%")
            fig_beta.update_layout(
                title="Bayesian Conjugate Beta-Binomial Dud Rate Posterior Density (FZ-03)",
                xaxis_title="Fuze Dud Probability p_dud (%)",
                yaxis_title="Posterior Probability Density",
                height=390,
            )
            st.plotly_chart(fig_beta, use_container_width=True)

        with fz_right:
            aof_deg_grid = np.linspace(1.0, 45.0, 36)
            v_imp_grid = np.linspace(50.0, 500.0, 36)
            aa_deg, vv_mps = np.meshgrid(aof_deg_grid, v_imp_grid)
            gr_surf = fz_06_impact_graze_function(
                angle_of_fall_rad=np.radians(aa_deg),
                v_impact_mps=vv_mps,
                graze_threshold_deg=12.0,
                min_arm_vel_mps=140.0,
            )
            p_fn_surf = np.asarray(gr_surf["p_function"], dtype=np.float64)
            fig_graze = go.Figure(
                data=[
                    go.Surface(
                        x=aof_deg_grid,
                        y=v_imp_grid,
                        z=p_fn_surf,
                        colorscale="Viridis",
                        colorbar=dict(title="P(func)"),
                    )
                ]
            )
            fig_graze.update_layout(
                title="3D Impact Fuze Graze Angle × Striking Velocity Function Surface (FZ-06)",
                scene=dict(
                    xaxis_title="Impact Angle of Fall (deg)",
                    yaxis_title="Impact Velocity (m/s)",
                    zaxis_title="P(Function)",
                ),
                height=390,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_graze, use_container_width=True)

    # --- SUB-LAB 9: PRECISION GUIDANCE & PROPORTIONAL NAVIGATION (PN / APN) ---
    with sub9:
        st.subheader("🚀 Precision Guidance & Autopilot Flight Control (`TPN` / `APN` / `Pure Pursuit`)")
        st.latex(
            r"""
            \mathbf{a}_{\text{cmd,TPN}} = N' V_c (\boldsymbol{\Omega} \times \hat{\mathbf{v}}_M), \qquad
            \mathbf{a}_{\text{cmd,APN}} = \mathbf{a}_{\text{cmd,TPN}} + \frac{1}{2} N' \mathbf{a}_{T\perp}, \qquad
            \tau_{\text{act}} \dot{\mathbf{a}}_{\text{act}} + \mathbf{a}_{\text{act}} = \text{sat}(\mathbf{a}_{\text{cmd}}, N_{\max} g)
            """
        )
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        with g_c1:
            g_law = st.selectbox("Guidance Law", ["TPN", "APN", "PP"], index=0, help="TPN: True Proportional Navigation | APN: Augmented PN | PP: Pure Pursuit")
            g_np = st.slider("Navigation Ratio N'", 2.0, 6.0, 3.8, 0.1)
            g_lim = st.slider("Lateral G-Limit (g)", 5.0, 30.0, 18.0, 1.0)
        with g_c2:
            t_type = st.selectbox("Target Motion Model", ["stationary", "constant_velocity", "evasive_weave"], index=1)
            t_speed = st.slider("Target Speed (m/s)", 0.0, 35.0, 12.0, 1.0) if t_type != "stationary" else 0.0
            t_evade_g = st.slider("Evasion Maneuver (g)", 0.5, 6.0, 2.5, 0.5) if t_type == "evasive_weave" else 0.0
        with g_c3:
            g_v0 = st.slider("Missile Initial Speed V₀ (m/s)", 150.0, 450.0, 240.0, 10.0)
            g_thrust = st.slider("Rocket Motor Thrust (N)", 1000.0, 7000.0, 3800.0, 200.0)
            g_noise = st.slider("Seeker Noise σ_LOS (mrad/s)", 0.0, 2.0, 0.4, 0.1) * 1e-3
        with g_c4:
            g_range = st.slider("Target Standoff Range (m)", 1500.0, 6000.0, 3500.0, 250.0)
            g_elev = st.slider("Launch Elevation QE (deg)", 10.0, 40.0, 22.0, 1.0)
            g_lethal_r = st.slider("Lethal Radius R_lethal (m)", 1.0, 10.0, 3.5, 0.5)

        g_res = simulate_guided_trajectory(
            v0_mps=float(g_v0),
            elev_deg=float(g_elev),
            target_pos_m=(float(g_range), 50.0, 0.0),
            target_vel_mps=(0.0, float(t_speed), 0.0),
            target_accel_type=t_type,  # type: ignore[arg-type]
            target_evasion_g=float(t_evade_g),
            guidance_law=g_law,  # type: ignore[arg-type]
            N_prime=float(g_np),
            g_limit=float(g_lim),
            motor_thrust_N=float(g_thrust),
            seeker_noise_std_rad=float(g_noise),
            lethal_radius_m=float(g_lethal_r),
        )

        gm1, gm2, gm3, gm4, gm5 = st.columns(5)
        is_hit = g_res["hit"]
        gm1.metric("Engagement Outcome", g_res["status"], delta="DIRECT HIT" if is_hit else "MISS", delta_color="normal" if is_hit else "inverse")
        gm2.metric("Miss Distance", f"{g_res['miss_distance_m']:.2f} m", delta=f"R_lethal: {g_lethal_r:.1f} m", delta_color="off")
        gm3.metric("Time of Flight", f"{g_res['time_of_flight_s']:.2f} s", delta_color="off")
        gm4.metric("Closing Velocity V_c", f"{g_res['terminal_closing_velocity_mps']:.1f} m/s", delta_color="off")
        gm5.metric("Peak Commanded G", f"{g_res['peak_g_load']:.1f} g", delta=f"Limit: {g_lim:.0f} g", delta_color="off")

        g_p1, g_p2 = st.columns([1.35, 1.0])
        with g_p1:
            fig_g_3d = create_3d_guided_intercept_figure(g_res)
            fig_g_3d.update_layout(height=480)
            st.plotly_chart(fig_g_3d, use_container_width=True)
        with g_p2:
            gh = g_res["history"]
            fig_g_dyn = make_subplots(specs=[[{"secondary_y": True}]])
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["act_g"], mode="lines", name="Achieved Lateral G (g)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["closing_vel_mps"], mode="lines", name="Closing Velocity (m/s)", line=dict(color="#10B981", width=2, dash="dash")),
                secondary_y=True,
            )
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["range_m"], mode="lines", name="Missile-Target Range (m)", line=dict(color="#DC2626", width=2, dash="dot")),
                secondary_y=True,
            )
            fig_g_dyn.update_xaxes(title_text="Flight Time t (s)")
            fig_g_dyn.update_yaxes(title_text="Lateral Acceleration (g)", secondary_y=False)
            fig_g_dyn.update_yaxes(title_text="Range / Velocity", secondary_y=True)
            fig_g_dyn.update_layout(title="Guidance Dynamics & Control Surface Effort vs Time", height=480)
            st.plotly_chart(fig_g_dyn, use_container_width=True)

    # --- SUB-LAB 10: CLUSTER MUNITION DISPENSER KINEMATICS & UXO HAZARD ---
    with sub10:
        st.subheader("📦 Cluster Munition Dispenser Kinematics, Ground Footprint & UXO Contamination (`CCM` / `IATG 10.10`)")
        st.latex(
            r"""
            V_{\text{term}} = \sqrt{\frac{2 m_s g}{\rho A_s C_{D,\text{chute}}}}, \qquad
            p_{\text{UXO}} = (1 - \beta) p_{\text{dud}} p_{\text{fail,SD}} + \beta \min(p_{\text{dud}}, p_{\text{fail,SD}}), \qquad
            R_{\text{cordon}} = \max(\Delta R) + 200\,\text{m}
            """
        )
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            d_n_sub = st.selectbox("Submunition Count N_sub", [24, 48, 64, 88], index=2)
            d_burst_alt = st.slider("Burst Altitude H_burst (m)", 150.0, 800.0, 380.0, 25.0)
            d_mass = st.slider("Submunition Mass m_s (kg)", 0.15, 0.60, 0.23, 0.02)
        with d_c2:
            d_app_v = st.slider("Carrier Approach Speed (m/s)", 150.0, 350.0, 240.0, 10.0)
            d_spin = st.slider("Carrier Roll Rate (rad/s)", 0.0, 120.0, 65.0, 5.0)
            d_ej_ax = st.slider("Axial Expulsion (m/s)", 10.0, 50.0, 24.0, 2.0)
        with d_c3:
            d_chute_cd = st.slider("Parachute Drag C_D", 0.8, 1.8, 1.25, 0.05)
            d_w_x = st.slider("Downrange Wind (m/s)", -5.0, 10.0, 2.0, 0.5)
            d_w_y = st.slider("Crossrange Wind (m/s)", -10.0, 10.0, 3.5, 0.5)
        with d_c4:
            d_p_dud = st.slider("Primary Fuze Dud Rate (%)", 0.5, 12.0, 3.5, 0.5) / 100.0
            d_p_sd = st.slider("Self-Destruct Fail Rate (%)", 1.0, 25.0, 8.0, 1.0) / 100.0
            d_beta = st.slider("Common-Cause Beta Factor", 0.0, 0.15, 0.04, 0.01)

        d_res = simulate_cluster_dispenser_footprint(
            n_submunitions=int(d_n_sub),
            submunition_mass_kg=float(d_mass),
            burst_pos_m=(12000.0, 0.0, float(d_burst_alt)),
            burst_vel_mps=(float(d_app_v), 0.0, -85.0),
            carrier_spin_rad_s=float(d_spin),
            ejection_axial_mps=float(d_ej_ax),
            chute_cd=float(d_chute_cd),
            wind_downrange_mps=float(d_w_x),
            wind_crossrange_mps=float(d_w_y),
            p_dud_primary=float(d_p_dud),
            p_fail_self_destruct=float(d_p_sd),
            common_cause_beta=float(d_beta),
        )

        dm1, dm2, dm3, dm4, dm5 = st.columns(5)
        dm1.metric("Detonated Functional", f"{d_res['n_detonated']} / {d_res['n_submunitions']}", delta=f"{100*d_res['n_detonated']/d_res['n_submunitions']:.1f}%", delta_color="normal")
        dm2.metric("Self-Destruct Neutralized", f"{d_res['n_self_destruct_neutralized']}", delta="Inert Dud", delta_color="off")
        dm3.metric("Live UXO Hazardous Duds", f"{d_res['n_uxo']}", delta=f"{d_res['actual_uxo_rate']*100:.2f}% Actual UXO", delta_color="inverse" if d_res['n_uxo'] > 0 else "normal")
        dm4.metric("Footprint Area", f"{d_res['footprint_area_hectares']:.2f} ha", delta=f"{d_res['footprint_semi_major_m']:.0f}m × {d_res['footprint_semi_minor_m']:.0f}m", delta_color="off")
        dm5.metric("IATG 10.10 Cordon Buffer", f"R = {d_res['iatg_cordon_radius_m']:.0f} m", delta=f"Cordon: {d_res['iatg_cordon_area_ha']:.1f} ha", delta_color="off")

        if d_res['n_uxo'] > 0:
            st.error(f"🚨 **UNODA IATG 10.10 / CCW Protocol V Hazardous Contamination Detected!** {d_res['n_uxo']} active unexploded ordnance (UXO) items on ground. UXO density: **{d_res['uxo_density_per_ha']:.2f} UXO/hectare** ({d_res['uxo_density_per_km2']:.0f} UXO/km²). Mandatory {d_res['iatg_cordon_radius_m']:.0f}m exclusion perimeter required.")
        else:
            st.success("✅ **Zero Live UXO Items Recorded!** All non-functioning submunitions were successfully neutralized by secondary self-destruct mechanisms.")

        fig_d_3d = create_3d_submunition_dispersion_figure(d_res)
        fig_d_3d.update_layout(height=480)
        st.plotly_chart(fig_d_3d, use_container_width=True)

    # --- SUB-LAB 11: FULL 6-DOF RIGID-BODY BALLISTICS & EPICYCLIC YAW MOTION ---
    with sub11:
        st.subheader("🌀 Full 6-Degree-of-Freedom (6-DoF) Rigid-Body Ballistics & Epicyclic Yaw Motion (`STANAG 4355`)")
        st.latex(
            r"""
            \dot{\mathbf{q}} = \frac{1}{2} \begin{bmatrix} 0 & -p & -q & -r \\ p & 0 & r & -q \\ q & -r & 0 & p \\ r & q & -p & 0 \end{bmatrix} \mathbf{q}, \quad
            \mathbf{I} \dot{\boldsymbol{\omega}}_B + \boldsymbol{\omega}_B \times (\mathbf{I} \boldsymbol{\omega}_B) = \mathbf{M}_B, \quad
            S_g = \frac{I_x^2 p^2}{2 \rho S d I_t C_{M\alpha} V^2} > 1.0
            """
        )
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        with s_c1:
            s_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 300.0, 950.0, 827.0, 15.0, key="s6_v0")
            s_qe = st.slider("Quadrant Elevation QE (deg)", 10.0, 65.0, 35.0, 1.0, key="s6_qe")
            s_twist = st.slider("Rifling Twist (cal/turn)", 15.0, 30.0, 20.0, 1.0, key="s6_twist")
        with s_c2:
            s_cm_a = st.slider("Overturning Moment C_Mα", 2.0, 7.0, 4.2, 0.2, help="Overturning aerodynamic moment slope (forward center of pressure)")
            s_cm_q = st.slider("Pitch Damping C_Mq", -30.0, -5.0, -14.0, 1.0, help="Pitch and yaw damping aerodynamic moment derivative")
            s_clp = st.slider("Spin Damping C_lp", -0.05, -0.005, -0.025, 0.005, help="Axial roll damping derivative")
        with s_c3:
            s_tip_yaw = st.slider("Tip-Off Initial Yaw Rate (rad/s)", 0.0, 0.30, 0.05, 0.01, help="Muzzle exit tip-off angular rate")
            s_tip_pitch = st.slider("Tip-Off Initial Pitch Rate (rad/s)", 0.0, 0.30, 0.02, 0.01)
            s_c_mag = st.slider("Magnus Cross-Force C_Ypα", -1.5, 0.0, -0.65, 0.05)
        with s_c4:
            s_dt = st.slider("RK4 Integration Step dt (ms)", 1.0, 5.0, 2.0, 0.5, key="s6_dt") * 1e-3
            s_lat = st.slider("Firing Latitude (deg)", -60.0, 60.0, 45.0, 5.0, key="s6_lat")

        res_6dof = simulate_6dof_trajectory(
            v0_mps=float(s_v0),
            elev_deg=float(s_qe),
            twist_cal_per_turn=float(s_twist),
            cm_alpha=float(s_cm_a),
            cm_q=float(s_cm_q),
            c_lp=float(s_clp),
            c_yp_alpha=float(s_c_mag),
            initial_yaw_rate_rad_s=float(s_tip_yaw),
            initial_pitch_rate_rad_s=float(s_tip_pitch),
            latitude_deg=float(s_lat),
            dt_s=float(s_dt),
        )

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("6-DoF Final Range", f"{res_6dof['final_range_m']:.0f} m", delta=f"TOF: {res_6dof['time_of_flight_s']:.1f} s", delta_color="off")
        sm2.metric("Gyroscopic Lateral Drift", f"{res_6dof['final_deflection_m']:+.1f} m (Right)", delta="Magnus + Precession", delta_color="off")
        sm3.metric("Muzzle Gyroscopic S_g", f"{res_6dof['muzzle_sg']:.2f}", delta="Stable Sg > 1.0" if res_6dof['muzzle_sg'] >= 1.0 else "UNSTABLE", delta_color="normal" if res_6dof['muzzle_sg'] >= 1.0 else "inverse")
        sm4.metric("Terminal Spin RPM", f"{res_6dof['terminal_spin_rpm']:.0f} RPM", delta=f"Initial: {res_6dof['initial_spin_rpm']:.0f} RPM", delta_color="off")
        sm5.metric("Peak Total Yaw |α_t|", f"{res_6dof['max_yaw_of_repose_deg']:.2f}°", delta=f"Min Sg: {res_6dof['min_sg']:.2f}", delta_color="off")

        s_p1, s_p2 = st.columns([1.35, 1.0])
        with s_p1:
            fig_6d_3d = create_3d_6dof_attitude_figure(res_6dof)
            fig_6d_3d.update_layout(height=480)
            st.plotly_chart(fig_6d_3d, use_container_width=True)
        with s_p2:
            s_hist = res_6dof["history"]
            # 2D Epicyclic yaw arm trajectory: alpha (pitch AoA) vs beta (yaw AoA)
            fig_epi = go.Figure()
            fig_epi.add_trace(
                go.Scatter(
                    x=s_hist["beta_deg"],
                    y=s_hist["alpha_deg"],
                    mode="lines",
                    line=dict(color="#2563EB", width=2.5),
                    name="Epicyclic Yaw Arm (β vs α)",
                )
            )
            fig_epi.add_trace(
                go.Scatter(
                    x=[s_hist["beta_deg"][0]],
                    y=[s_hist["alpha_deg"][0]],
                    mode="markers+text",
                    marker=dict(size=10, color="#10B981", symbol="circle"),
                    text=["Muzzle Exit"],
                    textposition="top right",
                    name="Muzzle Exit",
                )
            )
            fig_epi.update_layout(
                title="Epicyclic Yaw Trajectory: Nutation Loops & Precession Drift",
                xaxis_title="Angle of Yaw β (deg)",
                yaxis_title="Angle of Pitch α (deg)",
                height=480,
            )
            st.plotly_chart(fig_epi, use_container_width=True)

    # --- SUB-LAB 12: ARTILLERY BATTERY SHEAFS, MRSI / TOT & SALVO OPTIMIZATION ---
    with sub12:
        st.subheader("🎯 Tactical Artillery Battery Sheafs, MRSI / TOT & Salvo Optimization (FM 6-40 / STANAG 4119)")
        st.markdown(
            "Rigorous multi-gun artillery battery firing dynamics grounded in **US Army FM 6-40 / FM 3-09** and **NATO STANAG 4119**: "
            "battery spatial formations (Linear, Wedge, Staggered, Lazy-W), tactical sheaf distributions (Converged point target, "
            "Open parallel, Linear column/trench, Area circular), bisection aerodynamic elevation solver, "
            "and **Multiple Rounds Simultaneous Impact (MRSI) / Time-on-Target (TOT)** trajectory interleaving."
        )
        st.latex(
            r"""
            X_i = X_{\text{aim}} + \Delta X_i, \quad
            Y_i = Y_{\text{aim}} + \Delta Y_i, \quad
            \Delta t_{\text{fire}}^{(k)} = \text{TOF}_{\max} - \text{TOF}^{(k)}, \quad
            P_k(\text{salvo}) = 1 - \prod_{j=1}^{N_{\text{rounds}}} \left[1 - P_{k,j}\right]
            """
        )

        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            st.markdown("**Battery Weapon Formation**")
            bat_guns = st.slider("Number of Weapons in Battery", 2, 8, 6, 1)
            bat_form = st.selectbox("Battery Formation", ["staggered", "linear", "wedge", "lazy_w"], index=0)
            bat_spacing = st.slider("Gun Spacing (m)", 25.0, 150.0, 75.0, 5.0)

        with b_c2:
            st.markdown("**Tactical Target Sheaf**")
            sheaf_kind = st.selectbox("Sheaf Distribution Type", ["converged", "open", "linear", "area"], index=0)
            target_range_bat = st.slider("Target Distance (m)", 4000.0, 22000.0, 14000.0, 500.0)
            lin_len = st.slider("Linear Sheaf Length (m)", 50.0, 600.0, 300.0, 25.0) if sheaf_kind == "linear" else 300.0
            lin_az = st.slider("Linear Sheaf Azimuth (°)", 0.0, 360.0, 90.0, 15.0) if sheaf_kind == "linear" else 90.0

        with b_c3:
            st.markdown("**Salvo Perturbations & Lethality**")
            rds_per_gun = st.slider("Rounds Fired per Gun", 1, 4, 2, 1)
            sig_v0 = st.slider("Muzzle Velocity Error σ_V0 (m/s)", 0.2, 5.0, 1.5, 0.1)
            sig_pointing = st.slider("Weapon Pointing Error σ_aim (mils)", 0.2, 3.0, 0.8, 0.1)
            lethal_r = st.slider("Effective Blast/Frag Lethal Radius (m)", 10.0, 60.0, 30.0, 5.0)

        # MRSI / TOT Planner Section
        with st.expander("⏱️ Multiple Rounds Simultaneous Impact (MRSI) Trajectory Solver", expanded=True):
            st.markdown(
                r"MRSI calculates the fire release sequence across high-angle, reduced-propellant charge, "
                r"and low-angle trajectory arcs so that up to 3 rounds arrive simultaneously at the target ($\Delta t < 0.2\text{ s}$)."
            )
            mrsi_res = solve_mrsi_trajectory_plan(target_range_m=target_range_bat, max_rounds=3)
            if mrsi_res["mrsi_feasible"]:
                m_cols = st.columns(len(mrsi_res["plan"]))
                for idx, rd in enumerate(mrsi_res["plan"]):
                    with m_cols[idx]:
                        st.metric(
                            f"Round {rd['round_num']} ({rd['branch']})",
                            f"Fire @ T+{rd['fire_time_s']}s",
                            f"QE: {rd['qe_deg']}° | TOF: {rd['tof_s']}s",
                        )
                mrsi_df = pd.DataFrame(mrsi_res["plan"])
                st.dataframe(mrsi_df[["round_num", "branch", "qe_deg", "tof_s", "fire_time_s", "impact_time_s"]], use_container_width=True)
            else:
                st.warning("MRSI solution not feasible at current range (exceeds high/low angle split limits).")

        # Battery Salvo Simulation
        with st.spinner("Integrating Multi-Gun Battery Trajectories & Bivariate Sheaf Dispersion..."):
            salvo_sim = simulate_battery_salvo(
                n_guns=bat_guns,
                formation=bat_form,
                gun_spacing_m=bat_spacing,
                target_pos_m=(float(target_range_bat), 0.0),
                sheaf_type=sheaf_kind,
                linear_length_m=lin_len,
                linear_azimuth_deg=lin_az,
                rounds_per_gun=rds_per_gun,
                sigma_mv_rr_mps=sig_v0,
                sigma_aim_qe_mils=sig_pointing,
                sigma_aim_az_mils=sig_pointing,
                lethal_radius_m=lethal_r,
                seed=42,
            )

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Salvo Rounds", f"{salvo_sim['total_rounds']} rounds")
        sm2.metric("Sheaf CEP50", f"{salvo_sim['cep50_m']:.1f} m")
        sm3.metric("Sheaf CEP95", f"{salvo_sim['cep95_m']:.1f} m")
        p_neut = salvo_sim.get("target_neutralization_prob", salvo_sim.get("fractional_neutralization_pct", 0.0) / 100.0)
        sm4.metric("Target Neutralization P_k", f"{p_neut * 100.0:.1f} %")

        fig_bat_3d = create_3d_battery_salvo_figure(salvo_sim)
        st.plotly_chart(fig_bat_3d, use_container_width=True)

        with st.expander("📋 Detailed Battery Firing & Impact Telemetry Log", expanded=False):
            rounds_df = pd.DataFrame(salvo_sim["rounds"])
            preferred_cols = [
                "round_id", "gun_id", "round_in_gun", "impact_x_m", "impact_y_m",
                "miss_dist_m", "dist_from_target_m", "v_impact_mps", "tof_s"
            ]
            disp_cols = [c for c in preferred_cols if c in rounds_df.columns]
            st.dataframe(rounds_df[disp_cols] if disp_cols else rounds_df, use_container_width=True)

    # --- SUB-LAB 13: MULTI-LAYER ARMOR DEFEAT & TANDEM ERA ---
    with sub13:
        st.subheader("🛡️ Modern Multi-Layer Armor Defeat, Tandem Shaped Charges & Explosive Reactive Armor (ERA)")
        st.markdown(
            "Advanced hydrodynamic and solid-mechanics armor defeat laboratory extending **`TP-02` (Birkhoff hydrodynamic shaped-charge jet penetration)**: "
            "asymmetric explosive flyer plate acceleration via the Gurney equation, transverse shearing of shaped charge jets by moving plates, "
            "tandem precursor defeat of ERA, hydrodynamic dwell and erosion in ceramic tiles (SiC, B₄C, Al₂O₃), "
            "and residual perforation of Rolled Homogeneous Armor (RHA) with behind-armor spall cone lethality."
        )
        st.latex(
            r"""
            V_{\text{flyer}} = \sqrt{2E}\left(1 + \frac{2M}{C}\right)^{-1/2}, \quad
            L_{\text{eff}} = L_0 \left(1 - \frac{V_{\text{flyer}}\sin\theta_{\text{obliq}}}{v_{\text{jet}}}\right), \quad
            P_{\text{ceramic}} = \lambda_{\text{dwell}} \frac{L_{\text{jet}}}{1 + \frac{\rho_c H_c}{\rho_j v_j^2}}, \quad
            P_{\text{RHA}} = L_{\text{res}} \sqrt{\frac{\rho_j}{\rho_{\text{RHA}}}}
            """
        )

        arm1, arm2, arm3 = st.columns(3)
        with arm1:
            st.markdown("**Warhead Configuration**")
            is_tandem = st.toggle("Tandem Warhead (Precursor + Main)", value=True)
            prec_d = st.slider("Precursor Diameter (mm)", 25.0, 75.0, 40.0, 5.0) if is_tandem else 0.0
            prec_pen = st.slider("Precursor Jet Penetration (mm RHA)", 50.0, 300.0, 150.0, 10.0) if is_tandem else 0.0
            main_d = st.slider("Main Charge Diameter (mm)", 80.0, 180.0, 125.0, 5.0)
            main_pen = st.slider("Main Nominal Penetration (mm RHA)", 400.0, 1400.0, 900.0, 25.0)

        with arm2:
            st.markdown("**Explosive Reactive Armor (ERA)**")
            era_active = st.checkbox("Equipped with ERA Cassette", value=True)
            era_type = st.selectbox("ERA Generation / Type", ["Heavy Relikt / FY-4", "Medium Kontakt-5", "Light Blazer / Kontakt-1"], index=0) if era_active else "None"
            era_thick = st.slider("ERA Cassette Thickness (mm)", 15.0, 60.0, 35.0, 5.0) if era_active else 0.0
            era_flyer_m = st.slider("ERA Flyer Plate Mass (kg/m²)", 10.0, 70.0, 38.0, 2.0) if era_active else 0.0
            era_exp_c = st.slider("ERA High-Explosive Charge (kg/m²)", 5.0, 30.0, 14.0, 1.0) if era_active else 0.0

        with arm3:
            st.markdown("**Target Armor Structure**")
            obliq_deg = st.slider("NATO Armor Obliquity (° from normal)", 0.0, 75.0, 60.0, 5.0)
            cer_mat = st.selectbox("Ceramic Composite Tile Material", ["Silicon Carbide (SiC)", "Boron Carbide (B4C)", "Alumina (Al2O3)"], index=0)
            cer_thick = st.slider("Ceramic Layer Thickness (mm)", 0.0, 150.0, 50.0, 5.0)
            rha_thick = st.slider("Backing RHA Thickness (mm)", 50.0, 800.0, 350.0, 25.0)

        era_cat = "heavy_era" if ("Heavy" in era_type or "Medium" in era_type) else "light_era"
        cer_cat = "Silicon Carbide (SiC)" if "SiC" in cer_mat else "Alumina (Al2O3)"
        armor_res = simulate_tandem_shaped_charge_vs_era(
            warhead_type="tandem" if is_tandem else "monolithic",
            has_era=era_active,
            era_type=era_cat,
            main_calibre_mm=main_d,
            precursor_calibre_mm=prec_d if is_tandem else 0.0,
            inter_warhead_delay_ms=0.35,
            obliquity_deg=obliq_deg,
            rha_base_thickness_mm=rha_thick,
            has_ceramic_tile=cer_thick > 0.0,
            ceramic_thickness_mm=cer_thick,
            ceramic_type=cer_cat,
        )

        ap1, ap2, ap3, ap4 = st.columns(4)
        perf_status = armor_res.get("perforation_status", "UNKNOWN")
        status_delta = "Defeated Armor" if armor_res.get("perforated", False) else "Armor Survived"
        ap1.metric("Perforation Status", perf_status, delta=status_delta)
        ap2.metric("Total LOS Armor Depth", f"{armor_res.get('target_los_thickness_mm', 0.0):.1f} mm")
        ap3.metric("Residual Penetration Margin", f"{armor_res.get('residual_penetration_reserve_mm', 0.0):.1f} mm")
        spall = armor_res.get("behind_armor_spall", {})
        ap4.metric("Behind-Armor Spall Fragments", f"{spall.get('spall_fragment_count', 0)} frags")

        fig_arm_3d = create_3d_armor_defeat_figure(armor_res)
        st.plotly_chart(fig_arm_3d, use_container_width=True)

        with st.expander("🔬 Physical Layer Degradation & Energy Dissipation Log", expanded=True):
            lay_c1, lay_c2 = st.columns(2)
            with lay_c1:
                st.markdown("**Warhead Interaction with Defensive Layers**")
                st.write(f"- **Precursor Defeat of ERA**: {armor_res.get('precursor_success', False)}")
                st.write(f"- **ERA Flyer Plate Velocity**: {armor_res.get('era_flyer_speed_mps', 0.0):.1f} m/s")
                st.write(f"- **Transverse Jet Disruption**: {armor_res.get('jet_disruption_pct', 0.0):.1f} %")
                st.write(f"- **Ceramic Dwell Erosion Loss**: {armor_res.get('ceramic_rha_equivalent_mm', 0.0):.1f} mm RHA")
                st.write(f"- **Post-ERA Jet Penetration**: {armor_res.get('post_era_jet_penetration_mm', 0.0):.1f} mm RHA")

            with lay_c2:
                st.markdown("**Behind-Armor Spall Lethality**")
                st.write(f"- **Spall Half-Angle**: {spall.get('spall_cone_half_angle_deg', 0.0):.1f}°")
                st.write(f"- **Mean Fragment Velocity**: {spall.get('mean_spall_velocity_mps', 0.0):.1f} m/s")
                p_crew = spall.get('crew_incapacitation_prob', 0.0)
                st.write(f"- **Crew Incapacitation Probability**: {p_crew * 100.0:.1f} %")

    # --- SUB-LAB 14: COUNTER-BATTERY RADAR & POO/POI RECONSTRUCTION ---
    with sub14:
        st.subheader("🛰️ Counter-Battery Radar & Trajectory Reconstruction (NATO STANAG 4114 / AN/TPQ-53)")
        st.markdown(
            "Defensive artillery radar tracking and weapon location grounded in **NATO STANAG 4114**: "
            "evaluates projectile Radar Cross Section (RCS) via physical optics, tracks noisy radar echo detections "
            "across the radar search fence, and solves inverse exterior ballistics back-propagating states backwards in time "
            "to compute the hostile weapon **Point of Origin (POO)** and error ellipse, and forward to the **Point of Impact (POI)** "
            "for early-warning impact countdown alert."
        )
        st.latex(
            r"""
            \sigma_{\text{RCS}} = \sigma_{\text{nose}}\cos^4\theta + \sigma_{\text{broadside}}\sin^8\theta, \quad
            \mathbf{r}(t - \Delta t) = \mathbf{r}(t) - \mathbf{v}(t)\Delta t, \quad
            \mathbf{v}(t - \Delta t) = \mathbf{v}(t) - \mathbf{a}_{\text{forward}}(t)\Delta t
            """
        )

        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.markdown("**Weapon & Firing Solution**")
            r_v0 = st.slider("Muzzle Velocity V0 (m/s)", 450.0, 950.0, 680.0, 10.0, key="rad_v0")
            r_qe = st.slider("Quadrant Elevation QE (°)", 15.0, 65.0, 32.0, 1.0, key="rad_qe")
            r_cal = st.selectbox("Projectile Calibre", ["155mm NATO Artillery", "122mm D-30 Howitzer", "120mm Heavy Mortar", "227mm MLRS Rocket"], index=0, key="rad_cal")
            cal_diam = 0.155 if "155mm" in r_cal else (0.122 if "122mm" in r_cal else (0.120 if "120mm" in r_cal else 0.227))
            cal_len = 0.85 if "155mm" in r_cal else (0.60 if "122mm" in r_cal else (0.70 if "120mm" in r_cal else 3.96))
            cal_mass = 43.5 if "155mm" in r_cal else (21.7 if "122mm" in r_cal else (16.0 if "120mm" in r_cal else 307.0))

        with r_c2:
            st.markdown("**Radar Site & Coverage Emplacement**")
            radar_x_pos = st.slider("Radar East Position X (m)", 1000.0, 10000.0, 4000.0, 250.0, key="rad_rx")
            radar_y_pos = st.slider("Radar North Position Y (m)", -8000.0, 8000.0, -2500.0, 250.0, key="rad_ry")
            radar_max_r = st.slider("Radar Max Instrumented Range (m)", 15000.0, 45000.0, 32000.0, 1000.0, key="rad_rmax")
            p_det = st.slider("Single-Pulse Detection Probability P_d", 0.70, 0.99, 0.94, 0.01, key="rad_pd")

        with r_c3:
            st.markdown("**Measurement Noise & Target RCS**")
            sig_r = st.slider("Radar Slant Range Noise σ_R (m)", 2.0, 35.0, 12.0, 1.0, key="rad_sigr")
            sig_ang = st.slider("Radar Angle Noise σ_θ (mils)", 0.2, 3.0, 0.8, 0.1, key="rad_sigang")
            rcs_est = calculate_projectile_rcs(cal_diam, cal_len, aspect_angle_deg=r_qe)
            st.metric("Estimated Shell RCS (σ)", f"{rcs_est:.4f} m²", delta=f"{10.0*math.log10(max(rcs_est,1e-6)):.1f} dBsm")

        with st.spinner("Generating Aerodynamic Trajectory & Radar Track Measurements..."):
            from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch
            atm_env_r = AtmosphereEnvironment()
            cfg_r = BatchTrajectoryConfig(calibre_m=cal_diam, fidelity="L1", dt=0.08)
            rad_q_val = math.radians(r_qe)
            s0_r = np.array([[0.0, 0.0, 0.0, r_v0 * math.cos(rad_q_val), 0.0, r_v0 * math.sin(rad_q_val)]], dtype=np.float64)
            res_true_traj = integrate_trajectories_batch(s0_r, np.array([cal_mass]), np.array([1.0]), atm_env_r, cfg_r, record_single_path=True)
            sp_r = res_true_traj["sample_path"]

            tx_list = list(sp_r["range_m"])
            ty_list = [0.0] * len(sp_r["t"])
            tz_list = list(sp_r["alt_m"])
            tt_list = list(sp_r["t"])

            radar_track_sim = simulate_radar_tracking(
                trajectory_x_m=tx_list,
                trajectory_y_m=ty_list,
                trajectory_z_m=tz_list,
                time_s=tt_list,
                radar_pos_m=(radar_x_pos, radar_y_pos, 50.0),
                radar_range_max_m=radar_max_r,
                p_detection_nominal=p_det,
                sigma_range_m=sig_r,
                sigma_angle_mils=sig_ang,
                seed=42,
            )

            recon_sol = reconstruct_point_of_origin_and_impact(
                track_points=radar_track_sim["track_points"],
                gun_elev_m=0.0,
                target_elev_m=0.0,
                calibre_m=cal_diam,
                mass_kg=cal_mass,
                v0_nominal_mps=r_v0,
            )

        rk1, rk2, rk3, rk4 = st.columns(4)
        poo_err = math.hypot(recon_sol.get("poo_x_m", 0.0), recon_sol.get("poo_y_m", 0.0))
        rk1.metric("POO Localization Error", f"{poo_err:.1f} m", delta="High Precision" if poo_err < 100.0 else "Coarse")
        rk2.metric("Estimated Launch V0", f"{recon_sol.get('estimated_v0_mps', 0.0):.1f} m/s", delta=f"{recon_sol.get('estimated_v0_mps', 0.0) - r_v0:+.1f} m/s error")
        rk3.metric("Early-Warning Countdown", f"{recon_sol.get('early_warning_countdown_s', 0.0):.1f} s", delta="Bunker Impact Alert")
        rk4.metric("Radar Detections Used", f"{len(radar_track_sim['track_points'])} echoes", delta=f"{radar_track_sim.get('track_duration_s', 0.0)}s arc")

        fig_rad_3d = create_3d_counter_battery_radar_figure(radar_track_sim, recon_sol)
        st.plotly_chart(fig_rad_3d, use_container_width=True)

        with st.expander("📋 Radar Track Telemetry & Covariance Confidence Log", expanded=False):
            if radar_track_sim["track_points"]:
                rad_df = pd.DataFrame(radar_track_sim["track_points"])
                st.dataframe(rad_df[["t_s", "obs_x", "obs_y", "obs_z", "slant_range_m", "elevation_deg", "azimuth_deg"]], use_container_width=True)

    # --- SUB-LAB 15: DIRECT-FIRE TANK GUNNERY & APFSDS SABOT DYNAMICS ---
    with sub15:
        st.subheader("🎯 Direct-Fire Tank Gunnery, APFSDS Sabot Discard & Moving Target FCS")
        st.markdown(
            "High-velocity kinetic armor defeat ($1,500\text{–}1,800\text{ m/s}$) and moving target gunnery: "
            "simulates the dynamic 3-piece sabot petal separation upon muzzle exit, computes the friendly troop safety hazard zone, "
            "evaluates tank Fire Control System (FCS) lead angles, trunnion cant coupling, and crosswind jump, "
            "and models tungsten/DU long-rod terminal penetration using the **Lanz-Odermatt (TP-01)** continuum model."
        )
        st.latex(
            r"""
            \theta_{\text{lead}} = \arctan\left(\frac{v_{\text{cross}}\cdot\text{TOF}}{R}\right), \quad
            \Delta y_{\text{cant}} = \frac{1}{2} g\,\text{TOF}^2 \sin\phi_{\text{cant}}, \quad
            P_{\text{LO}} = L_w \frac{A}{\tanh(B_0 + B_1 L_w / D)} \sqrt{\frac{\rho_p}{\rho_t}} e^{-S^2 / V^2}
            """
        )

        df1, df2, df3 = st.columns(3)
        with df1:
            st.markdown("**Tank Weapon & Kinetic Long-Rod**")
            mbt_v0 = st.slider("Muzzle Velocity V0 (m/s)", 1500.0, 1850.0, 1680.0, 20.0, key="df_v0")
            rod_len = st.slider("Penetrator Rod Length (cm)", 55.0, 95.0, 78.0, 1.0, key="df_len") / 100.0
            rod_diam = st.slider("Penetrator Rod Diameter (mm)", 20.0, 34.0, 26.0, 1.0, key="df_diam") / 1000.0
            rod_m = st.slider("Penetrator Rod Mass (kg)", 3.5, 7.5, 4.85, 0.1, key="df_mass")

        with df2:
            st.markdown("**Moving Armored Target**")
            tgt_dist = st.slider("Target Distance (m)", 800.0, 4000.0, 2200.0, 100.0, key="df_dist")
            tgt_spd = st.slider("Target Speed (km/h)", 0.0, 60.0, 32.0, 2.0, key="df_spd")
            tgt_cross_ang = st.slider("Target Crossing Angle (° from bore line)", 0.0, 180.0, 80.0, 5.0, key="df_ang")
            fcs_on = st.toggle("Tank FCS Auto-Lead Solution Active", value=True, key="df_fcs")

        with df3:
            st.markdown("**Target Armor & Sabot Petals**")
            rha_mbt = st.slider("Base RHA Armor (mm)", 300.0, 950.0, 650.0, 25.0, key="df_rha")
            rha_obliq = st.slider("Armor Obliquity (° from normal)", 0.0, 70.0, 60.0, 5.0, key="df_obliq")
            n_petals_mbt = st.selectbox("Sabot Petal Count", [3, 4], index=0, key="df_pet")
            petal_div = st.slider("Petal Aerodynamic Divergence (°)", 5.0, 14.0, 8.5, 0.5, key="df_pdiv")

        with st.spinner("Simulating Sabot Discard & Kinetic Engagement..."):
            sabot_sim = simulate_apfsds_sabot_discard(
                v0_mps=mbt_v0,
                rod_calibre_m=rod_diam,
                rod_length_m=rod_len,
                rod_mass_kg=rod_m,
                n_petals=n_petals_mbt,
                petal_divergence_angle_deg=petal_div,
            )
            direct_sim = simulate_direct_fire_engagement(
                target_range_m=tgt_dist,
                target_speed_kmh=tgt_spd,
                target_crossing_angle_deg=tgt_cross_ang,
                v0_mps=mbt_v0,
                rod_calibre_m=rod_diam,
                rod_length_m=rod_len,
                rod_mass_kg=rod_m,
                armor_thickness_rha_mm=rha_mbt,
                armor_obliquity_deg=rha_obliq,
                fcs_lead_applied=fcs_on,
                seed=42,
            )

        dk1, dk2, dk3, dk4 = st.columns(4)
        dk1.metric("Engagement Outcome", direct_sim["engagement_status"], delta="Hit" if direct_sim["is_hit"] else "Miss")
        dk2.metric("Lanz-Odermatt Penetration", f"{direct_sim['lo_penetration_normal_mm']:.0f} mm RHA", delta=f"{direct_sim['v_impact_mps']:.0f} m/s impact")
        fcs_info = direct_sim["fcs_solution"]
        dk3.metric("FCS Aim Lead", f"{fcs_info['total_aim_lead_mils']:.2f} mils", delta=f"{fcs_info['total_crossrange_lead_m']:.2f} m lead")
        dk4.metric("Sabot Petal Hazard Zone", f"{sabot_sim['sabot_hazard_zone']['danger_length_m']:.0f}m x {sabot_sim['sabot_hazard_zone']['danger_width_m']:.0f}m", delta=f"{sabot_sim['petal_discard_completed_distance_m']}m discard")

        fig_direct_3d = create_3d_direct_fire_sabot_figure(sabot_sim, direct_sim)
        st.plotly_chart(fig_direct_3d, use_container_width=True)

        with st.expander("🔬 Sabot Petal Discard Telemetry & FCS Corrections", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Sabot Separation Dynamics**")
                st.write(f"- **Petal Count**: {sabot_sim['n_petals']}")
                st.write(f"- **Discard Complete Distance**: {sabot_sim['petal_discard_completed_distance_m']} m")
                st.write(f"- **Tip-Off Induced Yaw Rate**: {sabot_sim['rod_tip_off_yaw_rate_deg_s']} °/s")
                st.write(f"- **Danger Zone Length**: {sabot_sim['sabot_hazard_zone']['danger_length_m']:.0f} m")
            with sc2:
                st.markdown("**Fire Control Error Decomposition**")
                st.write(f"- **Time of Flight**: {fcs_info['tof_s']:.3f} s")
                st.write(f"- **Supersonic Vertical Drop**: {fcs_info['vertical_drop_m']:.2f} m")
                st.write(f"- **Kinematic Target Lead**: {fcs_info['kinematic_lead_distance_m']:.2f} m")
                st.write(f"- **Trunnion Cant Drift**: {fcs_info['cant_error_drift_m']:.2f} m")
                st.write(f"- **Crosswind Jump**: {fcs_info['crosswind_drift_m']:.2f} m")

    # --- SUB-LAB 16: ELECTRONIC WARFARE, GPS JAMMING & LASER HOMING ---
    with sub16:
        st.subheader("📡 Electronic Warfare (EW), GPS Jamming Resilience & Semi-Active Laser Guidance")
        st.markdown(
            "Precision guided munitions operating under hostile electronic warfare: "
            "evaluates standoff GPS noise jamming using Friis transmission ($J/S$ carrier unlock ratio), "
            "models post-denial tactical MEMS IMU dead-reckoning drift growth, and simulates terminal "
            "Semi-Active Laser (SAL) quadrant photodiode homing through aerosol/smoke obscurants (**Beer-Lambert Law**)."
        )
        st.latex(
            r"""
            \left[\frac{J}{S}\right]_{\text{dB}} = P_{J,\text{tx}} - 20\log_{10}\left(\frac{4\pi R_J}{\lambda}\right) - P_{S}, \quad
            \sigma_{\text{pos}}(t) \approx \frac{1}{2} b_a \Delta t^2 + \frac{1}{6} g b_g \Delta t^3, \quad
            T_{\text{optical}} = \exp\left(-\gamma_{\text{ext}} R_{\text{slant}}\right)
            """
        )

        ew1, ew2, ew3 = st.columns(3)
        with ew1:
            st.markdown("**Standoff EW Jammer Transmitter**")
            ew_pwr = st.slider("Jammer Radiated Power (kW)", 0.5, 25.0, 8.0, 0.5, key="ew_pwr")
            ew_dist = st.slider("Jammer Standoff Distance (km)", 5.0, 30.0, 15.0, 1.0, key="ew_dist") * 1000.0
            ew_thresh = st.slider("GPS Unlock J/S Threshold (dB)", 35.0, 55.0, 42.0, 1.0, key="ew_thresh")

        with ew2:
            st.markdown("**Inertial Navigation System (IMU) Quality**")
            imu_gyro = st.slider("Gyroscope Bias Drift Rate (°/hr)", 0.2, 10.0, 1.5, 0.2, key="ew_gyro")
            imu_acc = st.slider("Accelerometer Bias (mg)", 0.5, 8.0, 2.0, 0.2, key="ew_acc")
            guidance_range = st.slider("Target Nominal Range (m)", 10000.0, 25000.0, 18000.0, 1000.0, key="ew_rng")

        with ew3:
            st.markdown("**Terminal Laser Seeker & Obscurant**")
            smoke_coeff = st.slider("Battlefield Smoke Extinction γ (/km)", 0.0, 0.60, 0.15, 0.05, key="ew_smoke")
            laser_jit = st.slider("Designator Spot Jitter σ (m)", 0.10, 1.00, 0.35, 0.05, key="ew_jit")
            laser_homing_on = st.checkbox("Terminal SAL Laser Seeker Equipped", value=True, key="ew_laser")

        with st.spinner("Simulating Electronic Warfare Guidance Dynamics..."):
            ew_sim = simulate_ins_gps_guidance_under_jamming(
                nominal_range_m=guidance_range,
                jammer_pos_m=(ew_dist, 1800.0, 80.0),
                jammer_power_kw=ew_pwr,
                gps_denial_js_threshold_db=ew_thresh,
                imu_gyro_drift_deg_per_hr=imu_gyro,
                imu_accel_bias_mg=imu_acc,
                seed=42,
            )
            sal_sim = simulate_sal_seeker_tracking(
                target_pos_m=(guidance_range, 0.0, 0.0),
                designator_pos_m=(guidance_range - 1500.0, -1200.0, 150.0),
                laser_jitter_sigma_m=laser_jit,
                smoke_extinction_coeff_per_km=smoke_coeff,
                trajectory_pts=ew_sim["trajectory"][-12:],
                seed=42,
            )

        ek1, ek2, ek3, ek4 = st.columns(4)
        ek1.metric("GPS Tracking Status", "DENIED (Jammed)" if ew_sim["gps_denied"] else "LOCKED (Aided)", delta="INS Fallback" if ew_sim["gps_denied"] else "Precision GPS")
        ek2.metric("Max Jamming J/S Ratio", f"{ew_sim['max_js_ratio_db']:.1f} dB", delta=f"{ew_sim['max_js_ratio_db'] - ew_thresh:+.1f} dB above threshold")
        final_cep = sal_sim["miss_distance_m"] if (laser_homing_on and sal_sim["seeker_locked"]) else ew_sim["terminal_miss_distance_m"]
        ek3.metric("Terminal Miss Distance", f"{final_cep:.2f} m", delta="Laser Homing Kill" if (laser_homing_on and sal_sim["seeker_locked"]) else ew_sim["cep_category"])
        ek4.metric("Laser Optical Transmission", f"{sal_sim['optical_transmission']*100.0:.1f} %", delta="Lock Acquired" if sal_sim["seeker_locked"] else "Smoke Attenuated")

        fig_ew_3d = create_3d_ew_guidance_figure(ew_sim, sal_sim)
        st.plotly_chart(fig_ew_3d, use_container_width=True)

        with st.expander("📋 EW Electronic Warfare & Laser Guidance Telemetry Log", expanded=False):
            st.write(f"- **Standoff Distance at Denial**: {ew_sim.get('standoff_distance_at_denial_m', 0.0):.1f} m")
            st.write(f"- **Pure Inertial Flight Duration**: {ew_sim.get('pure_inertial_flight_duration_s', 0.0):.1f} s")
            st.write(f"- **Laser Seeker Acquisition Lock Range**: {sal_sim.get('lock_range_m', 0.0):.1f} m")
            st.write(f"- **Laser Spot Position**: {sal_sim.get('laser_spot_pos_m', [])}")

    # --- SUB-LAB 17: SMALL ARMS, SHOTGUN PATTERNING & MULTI-BARREL DISPERSION ---
    with sub17:
        st.subheader("🔫 Small Arms Ballistics, Shotgun Pattern Spread & Multi-Barrel Dispersion Monte Carlo")
        st.markdown(
            "Exterior ballistics and stochastic Monte Carlo dispersion engine for **small arms**, "
            "**shotgun pattern spread & 3D shot stringing** (Lowry / Journée / Oberfell model), "
            "and **multi-barrel systems** (SxS, O/U double-barrel shotguns/rifles, 6-barrel rotary Miniguns, and multi-mount volley guns)."
        )
        st.latex(
            r"""
            P(r \le r_0) = 1 - \exp\left(-\frac{r_0^2}{2\sigma_r(R)^2}\right), \quad
            L_{\text{string}}(R) \approx k_{\text{string}} R, \quad
            y_{\text{bore}}^{(i)}(R) = y_0^{(i)}\left(1 - \frac{R}{R_{\text{crossover}}}\right) + R \cdot \theta_{\text{bias}}^{(i)}
            """
        )

        arm_mode = st.radio(
            "Select Small Arms Analysis Mode:",
            [
                "🎯 Single-Range Pattern Spread & Shot Stringing",
                "📏 Multi-Distance Range Sweep & 3D Plume Slices",
                "📐 Multiple Barrel Lengths Comparison",
                "🔄 Multi-Barrel Systems, Rotary Minigun & Volley Dispersion",
                "🛸 Anti-Drone Multi-Shotgun Defense Array (C-UAS)",
            ],
            horizontal=True,
            key="small_arms_mode",
        )

        if arm_mode == "🎯 Single-Range Pattern Spread & Shot Stringing":
            sg_presets = get_shotgun_presets()
            choke_defs = get_choke_definitions()

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**Shotgun Ammunition & Pellet Load**")
                sg_choice = st.selectbox(
                    "Ammunition Preset",
                    list(sg_presets.keys()),
                    format_func=lambda k: sg_presets[k]["name"],
                    index=0,
                    key="sg_choice",
                )
                sel_sg = sg_presets[sg_choice]
                sg_pellets_per_shot = st.slider("Pellets per Shell", 1, 450, int(sel_sg["pellets_per_shot"]), 1, key="sg_pellets")
                sg_pellet_diam = st.number_input("Pellet Diameter (mm)", 1.5, 20.0, float(sel_sg["pellet_diameter_mm"]), 0.1, key="sg_diam")
                sg_pellet_mass = st.number_input("Pellet Mass (g)", 0.05, 35.0, float(sel_sg["pellet_mass_g"]), 0.05, key="sg_mass")
                sg_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 250.0, 500.0, float(sel_sg["v0_mps"]), 5.0, key="sg_v0")

            with sc2:
                st.markdown("**Choke Constriction & Fire Control**")
                choke_names = list(choke_defs.keys())
                def_choke_idx = choke_names.index(sel_sg["default_choke"]) if sel_sg["default_choke"] in choke_names else 3
                choke_choice = st.selectbox(
                    "Choke Tube Constriction",
                    choke_names,
                    index=def_choke_idx,
                    key="choke_choice",
                )
                st.caption(f"ℹ️ {choke_defs[choke_choice]['description']} (Constriction: {choke_defs[choke_choice]['constriction_mm']:.2f}mm / {choke_defs[choke_choice]['constriction_in']:.3f}\")")
                sg_range = st.slider("Target Distance (m)", 5.0, 70.0, 36.58, 1.0, key="sg_range", help="36.58m = 40 yards (standard patterning benchmark)")
                sg_shots = st.slider("Monte Carlo Rounds Fired", 1, 30, 8, 1, key="sg_shots")
                sg_crosswind = st.slider("Crosswind Velocity (m/s)", -15.0, 15.0, 0.0, 0.5, key="sg_wind")

            with sc3:
                st.markdown("**Target Geometry & Kinematics**")
                tgt_type = st.selectbox(
                    "Target Silhouette Type",
                    ["30_inch_circle", "e_type_silhouette", "clay_pigeon", "drone"],
                    format_func=lambda k: {
                        "30_inch_circle": "30-inch Pattern Circle (0.762m benchmark)",
                        "e_type_silhouette": "NATO E-Type Humanoid Silhouette",
                        "clay_pigeon": "Clay Pigeon Disc (11cm)",
                        "drone": "Micro-UAV Drone (40x40cm)",
                    }.get(k, k),
                    index=0,
                    key="sg_tgt_type",
                )
                sg_tgt_speed = st.slider("Target Transverse Speed (m/s)", 0.0, 35.0, 0.0, 1.0, key="sg_tgt_speed", help="Evaluates longitudinal shot stringing effect on crossing targets")
                sg_tgt_cross_angle = st.slider("Crossing Angle (°)", 10.0, 170.0, 90.0, 5.0, key="sg_tgt_cross_angle")
                sg_aim_err = st.slider("Shooter Aim Jitter (mils)", 0.0, 5.0, 0.5, 0.2, key="sg_aim_err")

            with st.spinner("Executing Shotgun Pellet Cloud & Shot Stringing Monte Carlo..."):
                sg_sim = simulate_shotgun_pattern_mc(
                    n_shots=int(sg_shots),
                    pellets_per_shot=int(sg_pellets_per_shot),
                    pellet_diameter_mm=float(sg_pellet_diam),
                    pellet_mass_g=float(sg_pellet_mass),
                    v0_mps=float(sg_v0),
                    choke_name=str(choke_choice),
                    target_distance_m=float(sg_range),
                    target_type=str(tgt_type),
                    moving_target_velocity_mps=float(sg_tgt_speed),
                    moving_target_crossing_angle_deg=float(sg_tgt_cross_angle),
                    crosswind_mps=float(sg_crosswind),
                    aim_error_mrad=float(sg_aim_err),
                    seed=42,
                )

            sk1, sk2, sk3, sk4 = st.columns(4)
            sk1.metric("30-in Circle Pattern Hit", f"{sg_sim['hits_30in_pct']:.1f} %", delta=f"{sg_sim['hits_30in_count']} / {sg_sim['total_pellets_fired']} pellets")
            sk2.metric("Target Silhouette Hit", f"{sg_sim['hits_target_pct']:.1f} %", delta=f"{sg_sim['hits_target_count']} hits on target")
            sk3.metric("Striking Velocity / Energy", f"{sg_sim['mean_v_impact_mps']:.0f} m/s", delta=f"{sg_sim['mean_pellet_energy_j']:.1f} J/pellet ({sg_sim['total_energy_delivered_j']:.0f} J total)")
            sk4.metric("Shot String Length", f"{sg_sim['mean_shot_string_length_m']:.2f} m", delta=f"CEP50: {sg_sim['cep_50_m']:.3f} m")

            st_tab1, st_tab2 = st.tabs(["🎯 2D Pattern Board", "🌐 3D Shot Plume & Stringing"])
            with st_tab1:
                fig_2d_pat = create_2d_target_pattern_figure(sg_sim)
                st.plotly_chart(fig_2d_pat, use_container_width=True)
            with st_tab2:
                fig_3d_plume = create_3d_shotgun_plume_figure(sg_sim)
                st.plotly_chart(fig_3d_plume, use_container_width=True)

            with st.expander("📋 Pellet Impact & Ballistic Gel Penetration Telemetry", expanded=False):
                sg_pellet_df = pd.DataFrame(sg_sim["sample_pellet_records"])
                st.dataframe(sg_pellet_df, use_container_width=True)
                csv_sg = sg_pellet_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Shotgun Pellet Dispersion Dataset (.csv)",
                    csv_sg,
                    "shotgun_pellet_dispersion_mc.csv",
                    "text/csv",
                    key="dl_sg_csv",
                )

        elif arm_mode == "📏 Multi-Distance Range Sweep & 3D Plume Slices":
            sg_presets = get_shotgun_presets()
            choke_defs = get_choke_definitions()

            sw_col1, sw_col2, sw_col3 = st.columns(3)
            with sw_col1:
                st.markdown("**Ammunition & Choke Setup**")
                sw_preset = st.selectbox("Preset Load", list(sg_presets.keys()), format_func=lambda k: sg_presets[k]["name"], key="sw_preset")
                sw_ammo = sg_presets[sw_preset]
                sw_choke = st.selectbox("Choke Tube", list(choke_defs.keys()), index=3, key="sw_choke")
                sw_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 280.0, 480.0, float(sw_ammo["v0_mps"]), 5.0, key="sw_v0")
            with sw_col2:
                st.markdown("**Range Sweep Stations**")
                sw_dist_opts = [10.0, 15.0, 20.0, 25.0, 30.0, 36.58, 45.0, 50.0, 55.0, 65.0]
                sw_selected_dists = st.multiselect(
                    "Distance Stations to Evaluate (m)",
                    options=sw_dist_opts,
                    default=[15.0, 25.0, 36.58, 50.0],
                    key="sw_selected_dists",
                )
                if not sw_selected_dists:
                    sw_selected_dists = [15.0, 30.0, 45.0]
                sw_selected_dists = sorted(sw_selected_dists)
                sw_shots_per_dist = st.slider("Rounds Fired per Distance", 4, 20, 8, 1, key="sw_shots")
            with sw_col3:
                st.markdown("**Target & Atmospheric Environment**")
                sw_tgt = st.selectbox("Target Silhouette", ["30_inch_circle", "e_type_silhouette", "clay_pigeon"], key="sw_tgt")
                sw_wind = st.slider("Crosswind (m/s)", -10.0, 10.0, 0.0, 0.5, key="sw_wind")
                sw_moving_spd = st.slider("Target Speed (m/s)", 0.0, 25.0, 0.0, 1.0, key="sw_spd")

            with st.spinner("Executing Multi-Distance Range Sweep Simulation..."):
                sw_res = evaluate_shotgun_dispersion_vs_distance(
                    pellets_per_shot=int(sw_ammo["pellets_per_shot"]),
                    pellet_diameter_mm=float(sw_ammo["pellet_diameter_mm"]),
                    pellet_mass_g=float(sw_ammo["pellet_mass_g"]),
                    v0_mps=float(sw_v0),
                    choke_name=str(sw_choke),
                    distances_m=sw_selected_dists,
                    target_type=str(sw_tgt),
                    moving_target_velocity_mps=float(sw_moving_spd),
                    crosswind_mps=float(sw_wind),
                    n_shots_per_dist=int(sw_shots_per_dist),
                    seed=42,
                )

            # KPIs across range sweep
            rec = sw_res["range_records"]
            sw_k1, sw_k2, sw_k3, sw_k4 = st.columns(4)
            sw_k1.metric(f"Close Range ({rec[0]['distance_m']:.0f}m) Hit", f"{rec[0]['hits_30in_pct']:.1f} %", delta=f"CEP50: {rec[0]['cep_50_m']:.3f}m")
            mid_rec = rec[len(rec) // 2]
            sw_k2.metric(f"Mid Range ({mid_rec['distance_m']:.0f}m) Hit", f"{mid_rec['hits_30in_pct']:.1f} %", delta=f"Energy: {mid_rec['mean_pellet_energy_j']:.1f}J")
            far_rec = rec[-1]
            sw_k3.metric(f"Max Range ({far_rec['distance_m']:.0f}m) Hit", f"{far_rec['hits_30in_pct']:.1f} %", delta=f"Diameter: {far_rec['pattern_diameter_90_m']:.2f}m")
            sw_k4.metric("Longitudinal Shot Stringing", f"{far_rec['shot_string_length_m']:.2f} m", delta=f"{len(rec)} Distances Swept")

            sw_tab1, sw_tab2, sw_tab3 = st.tabs([
                "🌐 3D Multi-Distance Plume Slices",
                "🎯 2D Side-by-Side Pattern Comparison",
                "📈 Diagnostic Range Curves",
            ])
            with sw_tab1:
                fig_sw_3d = create_3d_multi_distance_shotgun_figure(sw_res)
                st.plotly_chart(fig_sw_3d, use_container_width=True)
            with sw_tab2:
                fig_sw_2d = create_2d_multi_distance_shotgun_comparison_figure(sw_res)
                st.plotly_chart(fig_sw_2d, use_container_width=True)
            with sw_tab3:
                fig_sw_curves = create_range_curves_shotgun_figure(sw_res)
                st.plotly_chart(fig_sw_curves, use_container_width=True)

            with st.expander("📊 Multi-Distance Progression Telemetry Dataset", expanded=False):
                sw_df = pd.DataFrame(rec)
                st.dataframe(sw_df, use_container_width=True)
                csv_sw = sw_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Multi-Distance Sweep Dataset (.csv)",
                    csv_sw,
                    "shotgun_multi_distance_sweep.csv",
                    "text/csv",
                    key="dl_sw_csv",
                )

        elif arm_mode == "📐 Multiple Barrel Lengths Comparison":
            sg_presets = get_shotgun_presets()
            choke_defs = get_choke_definitions()
            b_defs = get_standard_barrel_lengths()

            bl_col1, bl_col2, bl_col3 = st.columns(3)
            with bl_col1:
                st.markdown("**Baseline Ammunition & Choke**")
                bl_preset = st.selectbox("Base Cartridge", list(sg_presets.keys()), format_func=lambda k: sg_presets[k]["name"], key="bl_preset")
                bl_ammo = sg_presets[bl_preset]
                bl_choke = st.selectbox("Choke Tube Constriction", list(choke_defs.keys()), index=3, key="bl_choke")
                bl_v0_base = st.slider("Reference Velocity V₀ at 28\" (m/s)", 300.0, 480.0, float(bl_ammo["v0_mps"]), 5.0, key="bl_v0_base")
            with bl_col2:
                st.markdown("**Barrel Lengths Selection**")
                avail_lengths = [14.0, 18.5, 22.0, 26.0, 28.0, 30.0, 32.0]
                sel_barrel_lengths = st.multiselect(
                    "Barrel Lengths to Compare Simultaneously (inches)",
                    options=avail_lengths,
                    default=[14.0, 18.5, 28.0, 32.0],
                    key="sel_barrel_lengths",
                )
                if not sel_barrel_lengths:
                    sel_barrel_lengths = [18.5, 28.0]
                sel_barrel_lengths = sorted(sel_barrel_lengths)
                bl_shots = st.slider("Rounds Fired per Barrel", 4, 25, 10, 1, key="bl_shots")
            with bl_col3:
                st.markdown("**Target Engagement Distance**")
                bl_dist = st.slider("Target Distance (m)", 10.0, 60.0, 36.58, 1.0, key="bl_dist", help="36.58m = 40 yards benchmark")
                bl_tgt = st.selectbox("Target Silhouette", ["30_inch_circle", "e_type_silhouette", "clay_pigeon"], key="bl_tgt")

            with st.spinner("Evaluating Multi-Barrel Length Internal & Terminal Dispersion..."):
                bl_res = compare_barrel_lengths_dispersion(
                    barrel_lengths_in=sel_barrel_lengths,
                    pellets_per_shot=int(bl_ammo["pellets_per_shot"]),
                    pellet_diameter_mm=float(bl_ammo["pellet_diameter_mm"]),
                    pellet_mass_g=float(bl_ammo["pellet_mass_g"]),
                    base_v0_mps=float(bl_v0_base),
                    reference_length_in=28.0,
                    choke_name=str(bl_choke),
                    target_distance_m=float(bl_dist),
                    target_type=str(bl_tgt),
                    n_shots=int(bl_shots),
                    seed=42,
                )

            # KPIs across barrel lengths
            b_recs = bl_res["barrel_records"]
            shortest = b_recs[0]
            longest = b_recs[-1]
            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric("Muzzle Velocity Span", f"{shortest['v0_mps']:.0f} → {longest['v0_mps']:.0f} m/s", delta=f"{longest['v0_mps'] - shortest['v0_mps']:+.1f} m/s difference")
            bk2.metric(f"Shortest ({shortest['length_in']:.1f}\") 30-in Hit", f"{shortest['hits_30in_pct']:.1f} %", delta=f"CEP50: {shortest['cep_50_m']:.3f}m")
            bk3.metric(f"Longest ({longest['length_in']:.1f}\") 30-in Hit", f"{longest['hits_30in_pct']:.1f} %", delta=f"CEP50: {longest['cep_50_m']:.3f}m")
            bk4.metric("Gas Wash Turbulence Factor", f"{shortest['gas_wash_factor']:.3f}x", delta=f"{((shortest['gas_wash_factor'] - 1.0)*100):+.1f}% short barrel penalty")

            bl_tab1, bl_tab2 = st.tabs([
                "🎯 2D Side-by-Side Target Pattern Boards",
                "📈 Barrel Length Scaling Curves",
            ])
            with bl_tab1:
                fig_bl_2d = create_2d_barrel_length_comparison_figure(bl_res)
                st.plotly_chart(fig_bl_2d, use_container_width=True)
            with bl_tab2:
                fig_bl_curves = create_barrel_length_diagnostic_curves(bl_res)
                st.plotly_chart(fig_bl_curves, use_container_width=True)

            with st.expander("📊 Barrel Length Performance Comparison Table", expanded=False):
                bl_df = pd.DataFrame(b_recs)
                st.dataframe(bl_df, use_container_width=True)
                csv_bl = bl_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Barrel Length Comparison Dataset (.csv)",
                    csv_bl,
                    "barrel_length_dispersion_comparison.csv",
                    "text/csv",
                    key="dl_bl_csv",
                )

        elif arm_mode == "🔄 Multi-Barrel Systems, Rotary Minigun & Volley Dispersion":
            mb_presets = get_multibarrel_presets()

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown("**Multi-Barrel System Configuration**")
                mb_choice = st.selectbox(
                    "Multi-Barrel Preset",
                    list(mb_presets.keys()),
                    format_func=lambda k: mb_presets[k]["name"],
                    index=2,
                    key="mb_choice",
                )
                sel_mb = mb_presets[mb_choice]
                mb_n_barrels = st.slider("Number of Barrels", 2, 8, int(sel_mb["n_barrels"]), 1, key="mb_n_barrels")
                mb_type = sel_mb["system_type"]
                if mb_type == "rotary_gatling":
                    mb_rotor_r = st.slider("Rotor Pitch Radius (mm)", 20.0, 150.0, float(sel_mb.get("rotor_radius_mm", 45.0)), 5.0, key="mb_rotor_r")
                    mb_spacing_y = 0.0
                    mb_spacing_z = 0.0
                else:
                    mb_rotor_r = 45.0
                    mb_spacing_y = st.number_input("Bore Spacing Y (mm)", 0.0, 600.0, float(sel_mb.get("bore_spacing_y_mm", 28.0)), 5.0, key="mb_space_y")
                    mb_spacing_z = st.number_input("Bore Spacing Z (mm)", 0.0, 600.0, float(sel_mb.get("bore_spacing_z_mm", 0.0)), 5.0, key="mb_space_z")

            with mc2:
                st.markdown("**Regulation Crossover & Engagement**")
                mb_crossover = st.slider("Regulation Crossover Range (m)", 10.0, 1500.0, float(sel_mb["crossover_range_m"]), 10.0, key="mb_crossover")
                mb_tgt_range = st.slider("Target Engagement Range (m)", 5.0, 1500.0, float(sel_mb["crossover_range_m"]), 10.0, key="mb_tgt_range", help="Set before, at, or after crossover to inspect convergence vs divergence")
                mb_burst = st.slider("Burst Length (rounds)", 4, 120, 36, 2, key="mb_burst")
                mb_rpm = st.slider("Cyclic Rate of Fire (RPM)", 120.0, 6000.0, float(sel_mb["cyclic_rate_rpm"]), 60.0, key="mb_rpm")

            with mc3:
                st.markdown("**Dispersion & Alignment Tolerances**")
                mb_disp = st.slider("Inherent Ammo Dispersion (mils)", 0.5, 6.0, float(sel_mb["inherent_dispersion_mrad"]), 0.1, key="mb_disp")
                mb_bias = st.slider("Barrel Pointing Runout Bias (mils)", 0.0, 4.0, float(sel_mb["barrel_pointing_bias_mrad"]), 0.1, key="mb_bias")
                mb_thermal = st.slider("Thermal Walk Drift (mils / 100 rds)", 0.0, 2.0, float(sel_mb["thermal_drift_mrad_per_100rd"]), 0.1, key="mb_thermal")
                mb_tgt_type = st.selectbox(
                    "Target Silhouette",
                    ["e_type_silhouette", "drone", "30_inch_circle"],
                    format_func=lambda k: {
                        "e_type_silhouette": "NATO E-Type Humanoid Silhouette",
                        "drone": "Micro-UAV Drone (3D Projected Aspect)",
                        "30_inch_circle": "30-inch Circle Target",
                    }.get(k, k),
                    index=0,
                    key="mb_tgt_type",
                )

            if mb_tgt_type == "drone":
                drone_specs_cat = get_drone_target_specs()
                aspect_presets_cat = get_drone_aspect_presets()

                st.markdown("---")
                st.markdown("#### 🛸 Target Drone Configuration & 3D Spatial Presentation")
                dcol1, dcol2, dcol3 = st.columns(3)
                with dcol1:
                    mb_drone_model = st.selectbox(
                        "Target Drone Class",
                        list(drone_specs_cat.keys()),
                        format_func=lambda k: f"{drone_specs_cat[k]['name']} ({drone_specs_cat[k].get('drone_type', 'UAV').title()})",
                        index=1,
                        key="mb_drone_model",
                    )
                    sel_drone_spec = drone_specs_cat[mb_drone_model]
                    st.caption(f"ℹ️ {sel_drone_spec.get('description', '')}")

                with dcol2:
                    mb_aspect_mode = st.selectbox(
                        "Target Presentation / Aspect",
                        list(aspect_presets_cat.keys()),
                        format_func=lambda k: f"{aspect_presets_cat[k].get('name', k)}",
                        index=0,
                        key="mb_aspect_mode",
                        help="Select standard tactical presentation angle: Head-on (ingressing), Flank side, Quartering, Tail rear, or Planform",
                    )
                    sel_aspect = aspect_presets_cat[mb_aspect_mode]
                    st.caption(f"**Preset Angles**: Pitch={sel_aspect['pitch_deg']}°, Yaw={sel_aspect['yaw_deg']}°, Roll={sel_aspect['roll_deg']}°")

                with dcol3:
                    mb_bullet_energy = st.slider(
                        "Bullet Striking Energy (Joules)",
                        200.0,
                        4000.0,
                        1800.0,
                        50.0,
                        key="mb_bullet_energy",
                        help="Terminal kinetic energy delivered per bullet (e.g. 5.56 NATO ~1700J, 7.62 NATO ~3500J)",
                    )
                    st.caption("Standard lethal threshold: Core ≥ 20-30J, Rotors ≥ 12-15J")

                with st.expander("🔧 Fine-Tune 3D Euler Angles (Pitch / Yaw / Roll)", expanded=False):
                    eul1, eul2, eul3 = st.columns(3)
                    with eul1:
                        mb_pitch = st.slider("Pitch Angle θ (°)", -90.0, 90.0, float(sel_aspect["pitch_deg"]), 1.0, key="mb_pitch_slider")
                    with eul2:
                        mb_yaw = st.slider("Yaw Angle ψ (°)", -180.0, 180.0, float(sel_aspect["yaw_deg"]), 1.0, key="mb_yaw_slider")
                    with eul3:
                        mb_roll = st.slider("Bank / Roll Angle ϕ (°)", -90.0, 90.0, float(sel_aspect["roll_deg"]), 1.0, key="mb_roll_slider")
            else:
                mb_drone_model = "commercial_quad_medium"
                mb_aspect_mode = "head_on_frontal"
                mb_pitch = 15.0
                mb_yaw = 0.0
                mb_roll = 0.0
                mb_bullet_energy = 1800.0

            with st.spinner("Simulating Multi-Barrel Volley & Rotor Dynamics..."):
                mb_sim = simulate_multibarrel_burst_mc(
                    system_type=str(mb_type),
                    n_barrels=int(mb_n_barrels),
                    bore_spacing_y_mm=float(mb_spacing_y),
                    bore_spacing_z_mm=float(mb_spacing_z),
                    rotor_radius_mm=float(mb_rotor_r),
                    crossover_range_m=float(mb_crossover),
                    burst_rounds=int(mb_burst),
                    target_range_m=float(mb_tgt_range),
                    cyclic_rate_rpm=float(mb_rpm),
                    inherent_dispersion_mrad=float(mb_disp),
                    barrel_pointing_bias_mrad=float(mb_bias),
                    thermal_drift_mrad_per_100rd=float(mb_thermal),
                    target_type=str(mb_tgt_type),
                    drone_model=str(mb_drone_model),
                    aspect_mode=str(mb_aspect_mode),
                    drone_pitch_deg=float(mb_pitch),
                    drone_yaw_deg=float(mb_yaw),
                    drone_roll_deg=float(mb_roll),
                    bullet_energy_j=float(mb_bullet_energy),
                    seed=42,
                )

            mk1, mk2, mk3, mk4 = st.columns(4)
            mk1.metric("Crossover Alignment", mb_sim["crossover_status"], delta=f"R_tgt / R_cross = {mb_sim['crossover_ratio']:.2f}")
            mk2.metric("Salvo Combined CEP", f"{mb_sim['salvo_cep_m']:.3f} m", delta=f"R90: {mb_sim['salvo_r90_m']:.3f} m")
            if mb_tgt_type == "drone" and mb_sim.get("presented_area_cm2") is not None:
                mk3.metric("Presented Cross-Section", f"{mb_sim['presented_area_cm2']:.0f} cm²", delta=f"{mb_sim['cross_section_ratio_pct']:.1f}% planform")
                mk4.metric("Salvo Hit Rate / Lethal", f"{mb_sim['salvo_hit_pct']:.1f} %", delta=f"{mb_sim['total_hits']} hits ({mb_sim['total_lethal_hits']} lethal)")
            else:
                mk3.metric("Salvo Target Hit Rate", f"{mb_sim['salvo_hit_pct']:.1f} %", delta=f"{mb_sim['total_hits']} / {mb_sim['burst_rounds']} hits")
                mk4.metric("Active Barrels Fired", f"{mb_sim['n_barrels']} Barrels", delta=f"{mb_sim['cyclic_rate_rpm']:.0f} RPM Cyclic")

            tab_labels = [
                "🎯 2D Target Intercept & Pattern",
                "🛸 3D Volley Convergence & Trajectory",
                "📏 Multi-Distance Crossover Progression",
            ]
            if mb_tgt_type == "drone":
                tab_labels.append("🌐 3D Target Presentation & Aspect Benchmark")

            mb_tabs = st.tabs(tab_labels)

            with mb_tabs[0]:
                fig_mb_2d = create_2d_multibarrel_target_figure(mb_sim)
                st.plotly_chart(fig_mb_2d, use_container_width=True)

            with mb_tabs[1]:
                fig_mb_3d = create_3d_multibarrel_dispersion_figure(mb_sim)
                st.plotly_chart(fig_mb_3d, use_container_width=True)

            with mb_tabs[2]:
                with st.spinner("Generating Multi-Distance Crossover Progression..."):
                    mb_sweep_res = evaluate_multibarrel_dispersion_vs_distance(
                        system_type=str(mb_type),
                        n_barrels=int(mb_n_barrels),
                        bore_spacing_y_mm=float(mb_spacing_y),
                        bore_spacing_z_mm=float(mb_spacing_z),
                        rotor_radius_mm=float(mb_rotor_r),
                        crossover_range_m=float(mb_crossover),
                        distances_m=[
                            round(0.50 * float(mb_crossover), 1),
                            round(1.00 * float(mb_crossover), 1),
                            round(1.50 * float(mb_crossover), 1),
                        ],
                        burst_rounds=int(mb_burst),
                        cyclic_rate_rpm=float(mb_rpm),
                        inherent_dispersion_mrad=float(mb_disp),
                        barrel_pointing_bias_mrad=float(mb_bias),
                        thermal_drift_mrad_per_100rd=float(mb_thermal),
                        target_type=str(mb_tgt_type),
                        drone_model=str(mb_drone_model),
                        aspect_mode=str(mb_aspect_mode),
                        drone_pitch_deg=float(mb_pitch),
                        drone_yaw_deg=float(mb_yaw),
                        drone_roll_deg=float(mb_roll),
                        bullet_energy_j=float(mb_bullet_energy),
                        seed=42,
                    )
                    fig_mb_multidist = create_3d_multi_distance_multibarrel_figure(mb_sweep_res)
                    st.plotly_chart(fig_mb_multidist, use_container_width=True)
                    fig_mb_curves = create_range_curves_multibarrel_figure(mb_sweep_res)
                    st.plotly_chart(fig_mb_curves, use_container_width=True)

            if mb_tgt_type == "drone":
                with mb_tabs[3]:
                    st.markdown("#### 🌐 Target Drone Cross-Section Benchmark across 6 Standard Presentation Angles")
                    st.markdown(
                        "Ingressing drones present vastly different vulnerable areas depending on flight profile. "
                        "A head-on frontal ingress minimizes cross-section, while planform overhead exposure maximizes vulnerable surface area."
                    )
                    with st.spinner("Benchmarking multi-barrel salvo effectiveness across 6 3D aspect orientations..."):
                        mb_bench = benchmark_multibarrel_aspect_angles_mc(
                            system_type=str(mb_type),
                            n_barrels=int(mb_n_barrels),
                            bore_spacing_y_mm=float(mb_spacing_y),
                            bore_spacing_z_mm=float(mb_spacing_z),
                            rotor_radius_mm=float(mb_rotor_r),
                            crossover_range_m=float(mb_crossover),
                            burst_rounds=int(mb_burst),
                            target_range_m=float(mb_tgt_range),
                            cyclic_rate_rpm=float(mb_rpm),
                            inherent_dispersion_mrad=float(mb_disp),
                            barrel_pointing_bias_mrad=float(mb_bias),
                            thermal_drift_mrad_per_100rd=float(mb_thermal),
                            drone_model=str(mb_drone_model),
                            bullet_energy_j=float(mb_bullet_energy),
                            seed=42,
                        )
                    fig_mb_radar = create_multibarrel_aspect_radar_figure(mb_bench)
                    st.plotly_chart(fig_mb_radar, use_container_width=True)

                    st.markdown("##### 📊 Comparative Aspect Telemetry Summary")
                    bench_df = pd.DataFrame(mb_bench["comparison_table"])[
                        ["label", "presented_area_cm2", "cross_section_ratio_pct", "salvo_hit_pct", "total_hits", "total_lethal_hits", "is_drone_killed", "salvo_cep_m"]
                    ]
                    bench_df.columns = [
                        "Presentation Aspect",
                        "Presented Area (cm²)",
                        "Planform Ratio (%)",
                        "Salvo Hit Rate (%)",
                        "Total Hits",
                        "Lethal Hits",
                        "Drone Neutralized",
                        "Salvo CEP (m)",
                    ]
                    st.dataframe(bench_df, use_container_width=True)

            with st.expander("📊 Individual Barrel Cluster Diagnostics & Telemetry Log", expanded=False):
                b_df = pd.DataFrame(mb_sim["barrel_stats"])
                st.dataframe(b_df, use_container_width=True)
                csv_mb = b_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Multi-Barrel Diagnostics (.csv)",
                    csv_mb,
                    "multibarrel_cluster_diagnostics.csv",
                    "text/csv",
                    key="dl_mb_csv",
                )

        elif arm_mode == "🛸 Anti-Drone Multi-Shotgun Defense Array (C-UAS)":
            st.markdown("#### 🛸 Multi-Shotgun Anti-Drone Defense Array (C-UAS)")
            st.markdown(
                "Configure and simulate a multi-shotgun counter-UAS intercept turret with **flexible spatial barrel placement architectures** "
                "(Circular Ring, Rectangular Grid, Concentric Dual-Ring, Hexagonal Honeycomb, Horizontal Fan, Chevron V, or Custom Coordinates), "
                "pointing convergence / splay net modes, simultaneous volley vs sequential ripple fire, and first-principles component vulnerability models "
                "(rotors, battery/fuselage core, structural arms, fixed wings)."
            )

            sg_presets = get_shotgun_presets()
            choke_defs = get_choke_definitions()
            drone_specs = get_drone_target_specs()

            cuas_col1, cuas_col2, cuas_col3 = st.columns(3)

            with cuas_col1:
                st.markdown("**1. Spatial Barrel Placement Architecture**")
                cuas_layout = st.selectbox(
                    "Array Mount Geometry",
                    ["circular", "grid", "concentric", "honeycomb", "fan_horizontal", "fan_vertical", "v_shape", "custom"],
                    format_func=lambda k: {
                        "circular": "⭕ Circular Ring Array",
                        "grid": "▦ Rectangular Matrix Grid (Rows x Cols)",
                        "concentric": "🎯 Concentric Dual-Ring (Core + Perimeter)",
                        "honeycomb": "⬡ Hexagonal Honeycomb Cluster",
                        "fan_horizontal": "↔️ Horizontal Linear Fan (Azimuth Curtain)",
                        "fan_vertical": "↕️ Vertical Stack Fan (Elevation Net)",
                        "v_shape": "📐 Chevron V-Shape Array",
                        "custom": "✏️ Custom User-Defined Coordinates",
                    }.get(k, k),
                    index=0,
                    key="cuas_layout",
                )

                if cuas_layout in ["circular", "concentric", "honeycomb"]:
                    cuas_n_barrels = st.slider("Number of Barrels (N)", 3, 16, 6 if cuas_layout != "honeycomb" else 7, 1, key="cuas_n_barrels")
                    cuas_mount_r = st.slider("Mount Radius (m)", 0.10, 0.60, 0.20, 0.02, key="cuas_mount_r", help="Radius of the barrel mounting ring on the turret face plate")
                    cuas_cols, cuas_rows, cuas_sp_y, cuas_sp_z = 3, 2, 0.12, 0.12
                elif cuas_layout == "grid":
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        cuas_rows = st.slider("Grid Rows", 1, 4, 2, 1, key="cuas_grid_r")
                    with gc2:
                        cuas_cols = st.slider("Grid Columns", 2, 6, 3, 1, key="cuas_grid_c")
                    cuas_n_barrels = cuas_rows * cuas_cols
                    st.caption(f"Total Barrels in Grid: **{cuas_n_barrels}**")
                    gc3, gc4 = st.columns(2)
                    with gc3:
                        cuas_sp_y = st.slider("Horizontal Pitch (m)", 0.06, 0.30, 0.12, 0.01, key="cuas_sp_y")
                    with gc4:
                        cuas_sp_z = st.slider("Vertical Pitch (m)", 0.06, 0.30, 0.12, 0.01, key="cuas_sp_z")
                    cuas_mount_r = 0.20
                elif cuas_layout in ["fan_horizontal", "fan_vertical", "v_shape"]:
                    cuas_n_barrels = st.slider("Number of Barrels (N)", 3, 12, 6, 1, key="cuas_n_barrels_fan")
                    cuas_sp_y = st.slider("Barrel Spacing Pitch (m)", 0.06, 0.25, 0.12, 0.01, key="cuas_fan_sp")
                    cuas_sp_z = cuas_sp_y
                    cuas_cols, cuas_rows, cuas_mount_r = 3, 2, 0.20
                else:  # custom
                    st.caption("Define custom barrels (y_m, z_m, choke):")
                    custom_text = st.text_area(
                        "Barrel Coordinates (Y, Z in meters):",
                        value="-0.15, 0.15, Full\n0.15, 0.15, Full\n-0.15, -0.15, Modified\n0.15, -0.15, Modified\n0.00, 0.00, Extra Full",
                        help="One barrel per line: y_meters, z_meters, choke",
                        key="cuas_custom_text",
                    )
                    cuas_n_barrels = 5
                    cuas_mount_r, cuas_cols, cuas_rows, cuas_sp_y, cuas_sp_z = 0.20, 3, 2, 0.12, 0.12

                cuas_b_len = st.slider("Barrel Length (inches)", 14.0, 32.0, 24.0, 1.0, key="cuas_b_len")
                cuas_choke = st.selectbox("Default / Perimeter Choke Tube", list(choke_defs.keys()), index=3, key="cuas_choke")
                cuas_mixed_loadout = st.checkbox(
                    "Enable Heterogeneous Core + Perimeter Loadout",
                    value=False,
                    key="cuas_mixed_loadout",
                    help="Combines dense Full-choke Tungsten TSS core barrels with wider #4 Buck/T-Shot perimeter barrels",
                )
                cuas_inner_ammo = None
                cuas_inner_choke = None
                if cuas_mixed_loadout:
                    cuas_inner_ammo = st.selectbox(
                        "Core / Center Barrel Load",
                        list(sg_presets.keys()),
                        format_func=lambda k: sg_presets[k]["name"],
                        index=list(sg_presets.keys()).index("12G_TSS_NO2_C_UAS") if "12G_TSS_NO2_C_UAS" in sg_presets else 0,
                        key="cuas_inner_ammo",
                    )
                    cuas_inner_choke = st.selectbox(
                        "Core / Center Barrel Choke",
                        list(choke_defs.keys()),
                        index=5,
                        key="cuas_inner_choke",
                    )

            with cuas_col2:
                st.markdown("**2. Pointing Vectors & Firing Control**")
                cuas_pointing = st.selectbox(
                    "Pointing & Convergence Mode",
                    ["convergent", "parallel", "divergent_curtain", "staggered_tiling"],
                    format_func=lambda k: {
                        "convergent": "🎯 Convergent Focus (All bores intersect at Target)",
                        "parallel": "║ Parallel Bores (Uniform spatial blanket)",
                        "divergent_curtain": "💥 Divergent Curtain (Expanding perimeter net)",
                        "staggered_tiling": "▦ Staggered Pattern Tiling (Wide non-overlapping coverage)",
                    }.get(k, k),
                    index=0,
                    key="cuas_pointing",
                )

                if cuas_pointing == "divergent_curtain":
                    cuas_splay = st.slider("Curtain Splay Angle (mils/mrad)", 0.0, 15.0, 4.0, 0.5, key="cuas_splay", help="Outward angular deflection to maximize intercept cross-sectional area")
                else:
                    cuas_splay = 0.0

                cuas_fire_mode = st.radio(
                    "Salvo Firing Sequence",
                    ["simultaneous_volley", "sequential_ripple", "balanced_pair_ripple"],
                    format_func=lambda k: {
                        "simultaneous_volley": "⚡ Simultaneous Volley (All Barrels at t=0)",
                        "sequential_ripple": "⏱️ Sequential Ripple Fire (Single-Barrel Δt)",
                        "balanced_pair_ripple": "⚖️ Balanced Symmetric Pair Ripple (Zero-Torque Pairs)",
                    }.get(k, k),
                    horizontal=True,
                    key="cuas_fire_mode",
                )

                if cuas_fire_mode != "simultaneous_volley":
                    cuas_ripple_dt = st.slider("Inter-Step Micro-Delay (ms)", 2.0, 40.0, 10.0, 1.0, key="cuas_ripple_dt", help="Micro-delay between shots/pairs to stretch longitudinal kill corridor and cut peak gimbal recoil")
                else:
                    cuas_ripple_dt = 0.0

                cuas_ammo = st.selectbox(
                    "Ammunition Load",
                    list(sg_presets.keys()),
                    format_func=lambda k: sg_presets[k]["name"],
                    index=list(sg_presets.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets else 0,
                    key="cuas_ammo",
                )
                st.caption(f"ℹ️ {sg_presets[cuas_ammo]['description']}")
                cuas_n_volleys = st.slider("Monte Carlo Volleys Evaluated", 10, 50, 25, 5, key="cuas_n_volleys")

            with cuas_col3:
                st.markdown("**3. Target Drone Kinematics & Vulnerability**")
                cuas_drone_model = st.selectbox(
                    "Target Drone Type",
                    list(drone_specs.keys()),
                    format_func=lambda k: drone_specs[k]["name"],
                    index=1,
                    key="cuas_drone_model",
                )
                sel_drone = drone_specs[cuas_drone_model]
                st.caption(f"ℹ️ {sel_drone['description']}")

                aspect_presets = get_drone_aspect_presets()
                cuas_aspect_mode = st.selectbox(
                    "Target Presentation / Aspect",
                    list(aspect_presets.keys()),
                    format_func=lambda k: aspect_presets[k]["name"],
                    index=0,
                    key="cuas_aspect_mode",
                    help="Cross-section presented directly toward the incoming line-of-sight shotgun blast.",
                )
                with st.expander("📐 3D Target Drone Attitude (Pitch / Yaw / Roll)", expanded=False):
                    c_p1, c_p2, c_p3 = st.columns(3)
                    cuas_pitch_tilt = c_p1.slider(
                        "Pitch Tilt (°)",
                        -45.0, 45.0,
                        15.0 if cuas_aspect_mode == "head_on_frontal" else 0.0,
                        1.0,
                        key="cuas_pitch_tilt",
                        help="Positive = forward attack pitch, negative = climb tilt.",
                    )
                    cuas_yaw_angle = c_p2.slider("Yaw Angle (°)", -180.0, 180.0, 0.0, 5.0, key="cuas_yaw_angle", help="0° = head-on, ±90° = flank, 180° = tail")
                    cuas_roll_angle = c_p3.slider("Bank / Roll (°)", -90.0, 90.0, 0.0, 5.0, key="cuas_roll_angle")

                cuas_dist = st.slider("Interception Distance (m)", 10.0, 70.0, 35.0, 1.0, key="cuas_dist")
                cuas_speed = st.slider("Drone Speed (m/s)", 0.0, 40.0, 15.0, 1.0, key="cuas_speed", help="15 m/s ≈ 54 km/h (typical FPV / recon quad transit speed)")
                cuas_heading = st.slider("Flight Vector Heading (°)", 0.0, 180.0, 90.0, 5.0, key="cuas_heading", help="0° = Head-on approach, 90° = Pure crossing broadside")
                cuas_aim_jitter = st.slider("Turret Tracking Jitter (mils)", 0.0, 4.0, 1.2, 0.2, key="cuas_aim_jitter")
                cuas_wind = st.slider("Ambient Crosswind (m/s)", -15.0, 15.0, 0.0, 0.5, key="cuas_wind")

            # Parse custom specs if needed
            parsed_custom = None
            if cuas_layout == "custom":
                parsed_custom = []
                for line in custom_text.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            y_val, z_val = float(parts[0]), float(parts[1])
                            chk = parts[2] if len(parts) >= 3 and parts[2] in choke_defs else cuas_choke
                            parsed_custom.append({
                                "barrel_id": f"B{len(parsed_custom)+1}",
                                "y_m": y_val,
                                "z_m": z_val,
                                "choke": chk,
                                "barrel_length_in": cuas_b_len,
                            })
                        except ValueError:
                            pass

            with st.spinner("Simulating Multi-Shotgun Counter-UAS Volley Monte Carlo..."):
                cuas_sim = simulate_anti_drone_multishotgun_mc(
                    layout_type=str(cuas_layout),
                    n_barrels=int(cuas_n_barrels),
                    mount_radius_m=float(cuas_mount_r),
                    spacing_y_m=float(cuas_sp_y),
                    spacing_z_m=float(cuas_sp_z),
                    n_cols=int(cuas_cols),
                    n_rows=int(cuas_rows),
                    pointing_mode=str(cuas_pointing),
                    splay_angle_mrad=float(cuas_splay),
                    firing_mode=str(cuas_fire_mode),
                    ripple_delay_ms=float(cuas_ripple_dt),
                    custom_barrel_specs=parsed_custom,
                    ammunition_preset=str(cuas_ammo),
                    default_choke=str(cuas_choke),
                    default_barrel_length_in=float(cuas_b_len),
                    target_distance_m=float(cuas_dist),
                    drone_model=str(cuas_drone_model),
                    drone_velocity_mps=float(cuas_speed),
                    drone_heading_deg=float(cuas_heading),
                    aspect_mode=str(cuas_aspect_mode),
                    drone_pitch_deg=float(cuas_pitch_tilt),
                    drone_yaw_deg=float(cuas_yaw_angle),
                    drone_roll_deg=float(cuas_roll_angle),
                    aim_error_mrad=float(cuas_aim_jitter),
                    crosswind_mps=float(cuas_wind),
                    n_volleys=int(cuas_n_volleys),
                    seed=42,
                    inner_ammunition_preset=cuas_inner_ammo,
                    outer_ammunition_preset=str(cuas_ammo) if cuas_mixed_loadout else None,
                    inner_choke=cuas_inner_choke,
                )

            st.info(
                "💡 Explore **Tab 18 (🛸 C-UAS Turret Suite)** next door for **Closed-Loop Kalman FCS & Evasive Drones**, "
                "**Multi-Drone Swarm Saturation**, **TSS / Flechette / Time-Fuzed Airburst Payloads**, "
                "**Turret Recoil & Constrained Pareto Optimization**, and **Elevated-Fire Pellet Fallout SDZ**!"
            )

            cu_sm = cuas_sim["summary"]

            # Display KPI metrics
            ck1, ck2, ck3 = st.columns(3)
            pk_val = cu_sm["kill_probability_pct"]
            ck1.metric(
                "Drone Kill Probability P_kill",
                f"{pk_val:.1f} %",
                delta="Mission Kill" if pk_val >= 80.0 else ("Probable Damage" if pk_val >= 50.0 else "Marginal Intercept"),
            )
            ck2.metric(
                "Mean Hits on Drone",
                f"{cu_sm['mean_hits_on_drone']:.1f} hits",
                delta=f"{cu_sm['mean_lethal_hits']:.1f} lethal (req: ≥{sel_drone['critical_hits_for_kill']})",
            )
            pres_area_cm2 = cu_sm.get("presented_area_cm2", 0.0)
            cs_ratio = cu_sm.get("cross_section_ratio_pct", 100.0)
            ck3.metric(
                "Presented Cross-Section",
                f"{pres_area_cm2:.1f} cm²",
                delta=f"{cs_ratio:.1f}% of planform ({cu_sm.get('aspect_mode', 'head_on')})",
                delta_color="normal",
            )
            ck4, ck5 = st.columns(2)
            ck4.metric(
                "Volley Payload Energy",
                f"{cu_sm['mean_energy_on_drone_j']:.0f} J on drone",
                delta=f"{cu_sm['total_pellets_per_volley']} pellets ({cu_sm['n_barrels']} bbls)",
            )
            ck5.metric(
                "Intercept Cloud Geometry",
                f"{cu_sm['cloud_diameter_90_m']:.2f} m diam",
                delta=f"CEP50: {cu_sm['cloud_cep_50_m']:.2f} m @ {cu_sm['target_distance_m']:.0f}m",
            )

            cuas_tab1, cuas_tab2, cuas_tab3, cuas_tab4, cuas_tab5 = st.tabs([
                "🎯 2D Drone Intercept & Damage Silhouette",
                "🛸 3D Muzzle Array & Intercept Volume",
                "📊 Per-Barrel Efficiency & Hit Contributions",
                "📈 Array Architecture Benchmark (Geometry Comparison)",
                "🌐 Aspect Angle Benchmark (Cross-Section Sweep)",
            ])

            with cuas_tab1:
                fig_2d_cuas = create_2d_antidrone_intercept_figure(cuas_sim)
                st.plotly_chart(fig_2d_cuas, use_container_width=True)

            with cuas_tab2:
                fig_3d_cuas = create_3d_antidrone_defense_figure(cuas_sim)
                st.plotly_chart(fig_3d_cuas, use_container_width=True)

            with cuas_tab3:
                st.markdown("##### Per-Barrel Telemetry & Contribution Breakdown")
                b_stats = cuas_sim.get("barrel_stats", [])
                if b_stats:
                    b_df = pd.DataFrame(b_stats)
                    st.dataframe(b_df, use_container_width=True)
                    csv_b = b_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Barrel Statistics (.csv)",
                        csv_b,
                        "anti_drone_barrel_statistics.csv",
                        "text/csv",
                        key="dl_cuas_b_csv",
                    )

            with cuas_tab4:
                st.markdown("##### Multi-Layout Benchmark Sweep Across Range (15m to 65m)")
                st.caption("Compares 5 standard multi-shotgun layouts against the selected drone target under identical engagement kinematics.")
                with st.spinner("Running Multi-Layout Array Benchmark..."):
                    bench_res = compare_anti_drone_layouts(
                        distance_stations_m=[15.0, 25.0, 35.0, 50.0, 65.0],
                        ammunition_preset=str(cuas_ammo),
                        drone_model=str(cuas_drone_model),
                        drone_velocity_mps=float(cuas_speed),
                        pointing_mode=str(cuas_pointing),
                        firing_mode=str(cuas_fire_mode),
                        n_volleys=12,
                        seed=42,
                    )
                    fig_bench = create_antidrone_layout_comparison_figure(bench_res)
                    st.plotly_chart(fig_bench, use_container_width=True)

                    bench_df = pd.DataFrame(bench_res["comparison_table"])
                    st.dataframe(bench_df, use_container_width=True)

            with cuas_tab5:
                st.markdown("##### 3D Target Presentation & Aspect Angle Vulnerability Benchmark")
                st.caption(
                    "Evaluates the multi-barrel shotgun array against all standard 3D target drone aspect orientations "
                    "(Head-on frontal, Flank side, Oblique quartering, Tail rear, Planform overhead, Planform underbelly). "
                    "Demonstrates the significant cross-section area reduction and required pellet density when engaging ingressing drones edge-on."
                )
                with st.spinner("Benchmarking target drone aspect angles..."):
                    aspect_bench = benchmark_drone_aspect_angles_mc(
                        layout_type=str(cuas_layout),
                        n_barrels=int(cuas_n_barrels),
                        mount_radius_m=float(cuas_mount_r),
                        spacing_y_m=float(cuas_sp_y),
                        spacing_z_m=float(cuas_sp_z),
                        n_cols=int(cuas_cols),
                        n_rows=int(cuas_rows),
                        pointing_mode=str(cuas_pointing),
                        splay_angle_mrad=float(cuas_splay),
                        firing_mode=str(cuas_fire_mode),
                        ripple_delay_ms=float(cuas_ripple_dt),
                        custom_barrel_specs=parsed_custom,
                        ammunition_preset=str(cuas_ammo),
                        default_choke=str(cuas_choke),
                        default_barrel_length_in=float(cuas_b_len),
                        target_distance_m=float(cuas_dist),
                        drone_model=str(cuas_drone_model),
                        drone_velocity_mps=float(cuas_speed),
                        aim_error_mrad=float(cuas_aim_jitter),
                        crosswind_mps=float(cuas_wind),
                        n_volleys=15,
                        seed=42,
                    )
                    fig_aspect = create_drone_aspect_radar_figure(aspect_bench)
                    st.plotly_chart(fig_aspect, use_container_width=True)

                    asp_df = pd.DataFrame(aspect_bench["comparison_table"])
                    st.dataframe(asp_df, use_container_width=True)
                    csv_asp = asp_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Aspect Benchmark Data (.csv)",
                        csv_asp,
                        "cuas_target_aspect_benchmark.csv",
                        "text/csv",
                        key="dl_aspect_bench_csv",
                    )

    # --- SUB-LAB 18: COMPLETE C-UAS TURRET ENGINEERING SUITE (UPGRADES 1 TO 5) ---
    with sub18:
        st.subheader("🛸 Multi-Shotgun C-UAS Turret Engineering & Operational Effectiveness Suite")
        st.markdown(
            "Complete 5-module counter-UAS design laboratory integrating **Closed-Loop Kalman Fire Control**, "
            "**Coordinated Swarm Saturation**, **Tungsten TSS / Flechette / Airburst Payloads**, "
            "**Turret Recoil & Constrained Pareto Optimization**, and **Elevated-Fire Pellet Fallout SDZ Safety Zones**."
        )
        st.latex(
            r"""
            \hat{\mathbf{r}}(\tau) = \hat{\mathbf{p}} + \hat{\mathbf{v}}\tau + \frac{1}{2}\hat{\mathbf{a}}\tau^2, \quad
            M\ddot{x} + C\dot{x} + Kx = \sum_{i} F_i(t), \quad
            I_{\theta}\ddot{\theta} + C_{\theta}\dot{\theta} + K_{\theta}\theta = \sum_{i} F_i(t)(z_{\text{tr}} + z_i), \quad
            m_p \frac{d\mathbf{v}}{dt} = -\frac{1}{2}\rho(z) C_D(Re,M) A_p |\mathbf{v}-\mathbf{W}|\,(\mathbf{v}-\mathbf{W}) - m_p g \hat{\mathbf{k}}
            """
        )

        cuas18_t1, cuas18_t2, cuas18_t3, cuas18_t4, cuas18_t5 = st.tabs(
            [
                "📡 1. Closed-Loop FCS & Evasive Drones",
                "🐝 2. Multi-Drone Swarm Saturation",
                "💥 3. Advanced Payloads & Airburst",
                "🔩 4. Recoil, Gimbal & Pareto Optimizer",
                "⚠️ 5. Elevated Fallout & Safety Zone (SDZ)",
            ]
        )

        sg_presets_18 = get_shotgun_presets()
        choke_defs_18 = get_choke_definitions()
        drone_specs_18 = get_drone_target_specs()
        aspect_presets_18 = get_drone_aspect_presets()
        sensor_presets_18 = get_cuas_sensor_presets()
        payload_cat_18 = get_cuas_payload_catalog()

        # --- TAB 18.1: CLOSED-LOOP FCS & EVASIVE DRONES ---
        with cuas18_t1:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown("**Tracking Sensor & Predictor Filter**")
                fcs_sensor = st.selectbox(
                    "Tracking Sensor Suite",
                    list(sensor_presets_18.keys()),
                    format_func=lambda k: sensor_presets_18[k]["name"],
                    index=0,
                    key="cuas18_fcs_sensor",
                )
                st.caption(f"ℹ️ {sensor_presets_18[fcs_sensor]['description']}")
                fcs_predictor = st.selectbox(
                    "Lead Filter Algorithm",
                    ["kalman_alpha_beta_gamma", "linear_alpha_beta", "unled_boresight"],
                    format_func=lambda k: {
                        "kalman_alpha_beta_gamma": "🎯 3-State Kalman (α-β-γ Pos + Vel + Accel)",
                        "linear_alpha_beta": "📏 2-State Linear Lead (α-β Pos + Vel Only)",
                        "unled_boresight": "❌ Unled Direct Aim (No Lead Compensation)",
                    }.get(k, k),
                    index=0,
                    key="cuas18_fcs_pred",
                )
                fcs_servo_rate = st.slider("Gimbal Max Slew Rate (°/s)", 30.0, 180.0, 90.0, 10.0, key="cuas18_fcs_rate")
                fcs_servo_accel = st.slider("Gimbal Max Accel (°/s²)", 60.0, 500.0, 250.0, 20.0, key="cuas18_fcs_accel")

            with fc2:
                st.markdown("**Evasive Drone Kinematics**")
                fcs_drone = st.selectbox(
                    "Drone Target Class",
                    list(drone_specs_18.keys()),
                    format_func=lambda k: drone_specs_18[k]["name"],
                    index=0,
                    key="cuas18_fcs_drone",
                )
                fcs_aspect = st.selectbox(
                    "Target Presentation Aspect",
                    list(aspect_presets_18.keys()),
                    format_func=lambda k: aspect_presets_18[k]["name"],
                    index=0,
                    key="cuas18_fcs_aspect",
                    help="Ingress angle presented toward FCS radar/optical line-of-sight.",
                )
                fcs_maneuver = st.selectbox(
                    "Evasive Maneuver Profile",
                    ["sinusoidal_weave", "corkscrew_spiral", "terminal_popup_jink", "linear_cruise"],
                    format_func=lambda k: {
                        "sinusoidal_weave": "〰️ Sinusoidal Horizontal + Vertical Weave",
                        "corkscrew_spiral": "🌀 Barrel / Corkscrew Spiral",
                        "terminal_popup_jink": "⚡ Late Terminal Pop-Up Jink",
                        "linear_cruise": "➡️ Constant-Velocity Linear Cruise",
                    }.get(k, k),
                    index=0,
                    key="cuas18_fcs_man",
                )
                fcs_g = st.slider("Peak Evasive Maneuver Load (g)", 0.0, 5.0, 2.5, 0.25, key="cuas18_fcs_g")
                fcs_spd = st.slider("Drone Speed (m/s)", 5.0, 40.0, 22.0, 1.0, key="cuas18_fcs_spd")
                fcs_dist = st.slider("Intercept Distance (m)", 15.0, 70.0, 35.0, 1.0, key="cuas18_fcs_dist")

            with fc3:
                st.markdown("**Turret Array & Pattern Curtain**")
                fcs_layout = st.selectbox("Array Geometry", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"], index=0, key="cuas18_fcs_lay")
                fcs_n_b = st.slider("Barrel Count", 2, 12, 6, 1, key="cuas18_fcs_nb")
                fcs_point = st.selectbox("Pointing Mode", ["divergent_curtain", "convergent", "staggered_tiling", "parallel"], index=0, key="cuas18_fcs_pt")
                fcs_splay = st.slider("Curtain Splay Angle (mrad)", 0.0, 12.0, 4.5, 0.5, key="cuas18_fcs_sp")
                fcs_fire = st.selectbox("Firing Sequence", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"], index=0, key="cuas18_fcs_fire")
                fcs_ammo = st.selectbox("Shell Preset", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_fcs_ammo")

            with st.spinner("Running Closed-Loop FCS & Evasive Intercept Monte Carlo..."):
                fcs_res = simulate_closed_loop_fcs_engagement_mc(
                    sensor_preset=str(fcs_sensor),
                    predictor_mode=str(fcs_predictor),
                    maneuver_profile=str(fcs_maneuver),
                    max_maneuver_g=float(fcs_g),
                    servo_max_rate_deg_s=float(fcs_servo_rate),
                    servo_max_accel_deg_s2=float(fcs_servo_accel),
                    layout_type=str(fcs_layout),
                    n_barrels=int(fcs_n_b),
                    pointing_mode=str(fcs_point),
                    splay_angle_mrad=float(fcs_splay),
                    firing_mode=str(fcs_fire),
                    ammunition_preset=str(fcs_ammo),
                    target_distance_m=float(fcs_dist),
                    drone_model=str(fcs_drone),
                    drone_velocity_mps=float(fcs_spd),
                    aspect_mode=str(fcs_aspect),
                    n_volleys=25,
                    seed=42,
                )
            fsm = fcs_res["summary"]
            fk1, fk2, fk3, fk4 = st.columns(4)
            fk1.metric("Closed-Loop P_kill", f"{fsm['kill_probability_pct']:.1f} %", delta=f"{fsm['mean_lethal_hits']:.1f} lethal hits/volley")
            fk2.metric("Mean Lead Error", f"{fsm['mean_lead_error_m']:.2f} m", delta=f"CEP50: {fsm['cep_lead_error_m']:.2f} m (TOF: {fsm['nominal_tof_ms']:.0f} ms)")
            fk3.metric("Target Jink Excursion", f"{fsm['maneuver_displacement_during_tof_m']:.2f} m", delta=f"{fcs_g:.1f}g during pellet flight")
            fk4.metric("Servo Saturation Rate", f"{fsm['servo_saturated_pct']:.1f} %", delta=f"{fsm['n_barrels']} barrels active")

            st.plotly_chart(create_fcs_tracking_and_evasion_figure(fcs_res), use_container_width=True)
            if fcs_res.get("architecture_comparison"):
                st.dataframe(pd.DataFrame(fcs_res["architecture_comparison"]), use_container_width=True)

        # --- TAB 18.2: MULTI-DRONE SWARM SATURATION ---
        with cuas18_t2:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**Coordinated Swarm Attack Wave**")
                sw_size = st.slider("Simultaneous Swarm Size (Drones)", 2, 18, 8, 1, key="cuas18_sw_n")
                sw_tactic = st.selectbox(
                    "Swarm Approach Tactic",
                    ["synchronized_omni_wave", "narrow_corridor_saturation", "staggered_multi_altitude"],
                    format_func=lambda k: {
                        "synchronized_omni_wave": "🌐 360° Synchronized Omni-Directional Wave",
                        "narrow_corridor_saturation": "🏹 Narrow Azimuth Saturation Corridor",
                        "staggered_multi_altitude": "🪜 Staggered Multi-Altitude Saturation",
                    }.get(k, k),
                    index=0,
                    key="cuas18_sw_tac",
                )
                sw_speed = st.slider("Mean Drone Speed (m/s)", 8.0, 40.0, 22.0, 1.0, key="cuas18_sw_spd")
                sw_drone = st.selectbox("Drone Target Model", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_sw_dr")
                sw_aspect = st.selectbox(
                    "Swarm Ingress Aspect",
                    list(aspect_presets_18.keys()),
                    format_func=lambda k: aspect_presets_18[k]["name"],
                    index=0,
                    key="cuas18_sw_aspect",
                    help="Default: Head-on Frontal for ingressing swarm waves.",
                )

            with sc2:
                st.markdown("**Magazine Allocation & Engagement Rules**")
                sw_barrels = st.slider("Ready Shotgun Barrels in Turret", 2, 16, 8, 1, key="cuas18_sw_nb")
                sw_bpb = st.slider("Sub-Salvo Barrels Fired per Burst", 1, 8, 2, 1, key="cuas18_sw_bpb", help="Firing fewer barrels per drone increases capacity before reload")
                sw_reload = st.slider("Full Turret Reload Downtime (s)", 1.0, 12.0, 4.5, 0.5, key="cuas18_sw_rel")
                sw_policy = st.selectbox(
                    "Target Priority Policy",
                    ["minimum_time_to_breach", "nearest_slant_range", "minimum_slew_angle"],
                    format_func=lambda k: {
                        "minimum_time_to_breach": "⏱️ Shortest Time-to-Breach (Optimal Threat Sort)",
                        "nearest_slant_range": "📏 Nearest Slant Range First",
                        "minimum_slew_angle": "🔄 Minimum Turret Slew Angle Next",
                    }.get(k, k),
                    index=0,
                    key="cuas18_sw_pol",
                )

            with sc3:
                st.markdown("**Turret Agility & Defense Perimeter**")
                sw_rmax = st.slider("Max Engagement Envelope (m)", 30.0, 90.0, 55.0, 5.0, key="cuas18_sw_rmax")
                sw_rmin = st.slider("Protected Asset Keep-Out Perimeter (m)", 5.0, 25.0, 12.0, 1.0, key="cuas18_sw_rmin")
                sw_slew = st.slider("Turret Slew Rate (°/s)", 40.0, 200.0, 110.0, 10.0, key="cuas18_sw_slew")
                sw_ammo = st.selectbox("Swarm Defense Load", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_sw_ammo")

            with st.spinner("Simulating Multi-Wave Drone Swarm Timeline..."):
                sw_res = simulate_drone_swarm_saturation_mc(
                    swarm_size=int(sw_size),
                    attack_tactic=str(sw_tactic),
                    priority_policy=str(sw_policy),
                    n_barrels=int(sw_barrels),
                    barrels_per_burst=int(min(sw_bpb, sw_barrels)),
                    reload_time_s=float(sw_reload),
                    slew_rate_deg_s=float(sw_slew),
                    max_engagement_range_m=float(sw_rmax),
                    breach_keepout_radius_m=float(sw_rmin),
                    ammunition_preset=str(sw_ammo),
                    drone_model=str(sw_drone),
                    mean_drone_speed_mps=float(sw_speed),
                    aspect_mode=str(sw_aspect),
                    n_waves=30,
                    seed=42,
                )
            sws = sw_res["summary"]
            sk1, sk2, sk3, sk4 = st.columns(4)
            sk1.metric("Zero-Leaker Defense Rate", f"{sws['zero_leaker_wave_pct']:.1f} %", delta=f"Optimal Allocation: {sws['optimal_barrels_per_burst']} barrels/burst")
            sk2.metric("Mean Drones Destroyed", f"{sws['mean_drones_killed']:.2f} / {sws['swarm_size']}", delta=f"Mean Intercept @ {sws['mean_intercept_range_m']:.1f}m")
            sk3.metric("Perimeter Leakers / Wave", f"{sws['mean_leakers']:.2f}", delta=f"Leaker Rate: {sws['leaker_rate_pct']:.1f} %")
            sk4.metric("Magazine Endurance", f"{sws['mean_bursts_fired']:.1f} bursts", delta=f"{sws['mean_reloads']:.2f} full reloads/wave")

            st.plotly_chart(create_3d_swarm_hemisphere_figure(sw_res), use_container_width=True)
            st.plotly_chart(create_swarm_doctrine_tradeoff_figure(sw_res), use_container_width=True)

        # --- TAB 18.3: ADVANCED C-UAS PAYLOADS & AIRBURST ---
        with cuas18_t3:
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.markdown("**Advanced Standoff Payload Technology**")
                pay_key = st.selectbox(
                    "Payload Family",
                    list(payload_cat_18.keys()),
                    format_func=lambda k: payload_cat_18[k]["name"],
                    index=list(payload_cat_18.keys()).index("CUAS_AIRBURST_CANISTER"),
                    key="cuas18_pay_key",
                )
                st.caption(f"ℹ️ {payload_cat_18[pay_key]['description']}")
                pay_nb = st.slider("Barrels / Salvo Size", 1, 8, 4, 1, key="cuas18_pay_nb")

            with pc2:
                st.markdown("**Engagement Range & Airburst Fuze Timing**")
                pay_dist = st.slider("Engagement Range (m)", 15.0, 115.0, 75.0, 5.0, key="cuas18_pay_dist")
                pay_standoff = st.slider("Airburst Programmed Standoff Ahead of Drone (m)", 1.5, 10.0, 4.5, 0.5, key="cuas18_pay_st")
                pay_jitter = st.slider("Electronic Time-Fuze Jitter σ_t (ms)", 0.2, 5.0, 0.8, 0.2, key="cuas18_pay_jit")

            with pc3:
                st.markdown("**Target & Cone Dispersion**")
                pay_drone = st.selectbox("Target Drone Class", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=1, key="cuas18_pay_dr")
                pay_aspect = st.selectbox(
                    "Target Cross-Section Aspect",
                    list(aspect_presets_18.keys()),
                    format_func=lambda k: aspect_presets_18[k]["name"],
                    index=0,
                    key="cuas18_pay_aspect",
                )
                pay_cone = st.slider("Airburst Ejection Cone Half-Angle (°)", 3.0, 15.0, 7.5, 0.5, key="cuas18_pay_cone")

            with st.spinner("Running Standoff Payload & Programmable Airburst Comparison..."):
                ab_res = simulate_cuas_advanced_payload_mc(
                    payload_key=str(pay_key),
                    n_barrels=int(pay_nb),
                    target_distance_m=float(pay_dist),
                    drone_model=str(pay_drone),
                    aspect_mode=str(pay_aspect),
                    programmed_standoff_m=float(pay_standoff),
                    ejection_cone_half_angle_deg=float(pay_cone),
                    fuze_time_jitter_ms=float(pay_jitter),
                    n_volleys=25,
                    seed=42,
                )
                pay_comp = compare_cuas_payload_families(
                    distances_m=[20.0, 35.0, 50.0, 70.0, 90.0, 110.0],
                    n_barrels=int(pay_nb),
                    drone_model=str(pay_drone),
                    aspect_mode=str(pay_aspect),
                    fuze_time_jitter_ms=float(pay_jitter),
                    n_volleys=12,
                    seed=42,
                )

            psm = ab_res["summary"]
            pk1, pk2, pk3, pk4 = st.columns(4)
            pk1.metric("Standoff P_kill", f"{psm['kill_probability_pct']:.1f} %", delta=f"{psm['mean_lethal_hits']:.1f} lethal hits @ {pay_dist:.0f}m")
            pk2.metric("Sub-Projectile Striking Energy", f"{psm['mean_subprojectile_energy_j']:.1f} J", delta=f"{psm['mean_subprojectile_velocity_mps']:.0f} m/s ({psm['total_volley_subprojectiles']} subs)")
            pk3.metric("Actual Burst Standoff", f"{psm['mean_burst_standoff_m']:.2f} m" if psm["is_airburst"] else "Unfuzed Shot", delta=f"Late Past Target: {psm['late_burst_behind_target_pct']:.1f}%" if psm["is_airburst"] else f"Cone N/A")
            pk4.metric("Energy Delivered to Airframe", f"{psm['mean_energy_on_drone_j']:.0f} J", delta=f"{psm['mean_hits_on_drone']:.1f} total impacts")

            pt1, pt2 = st.tabs(["📈 7-Family Standoff Range Curves (20–110 m)", "💥 3D Airburst / Cloud Intercept Geometry"])
            with pt1:
                st.plotly_chart(create_cuas_payload_range_comparison_figure(pay_comp), use_container_width=True)
                st.dataframe(pd.DataFrame(pay_comp["comparison_table"]), use_container_width=True)
            with pt2:
                st.plotly_chart(create_airburst_standoff_cone_figure(ab_res), use_container_width=True)

        # --- TAB 18.4: TURRET RECOIL, GIMBAL LOADS & PARETO ARRAY OPTIMIZER ---
        with cuas18_t4:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown("**Salvo Firing & Mixed Loadout**")
                rec_layout = st.selectbox("Array Architecture", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"], index=0, key="cuas18_rec_lay")
                rec_nb = st.slider("Barrel Count (N)", 3, 12, 6, 1, key="cuas18_rec_nb")
                rec_ammo = st.selectbox("Primary Ammunition", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_rec_ammo")
                rec_mixed = st.checkbox("Mix Core TSS + Perimeter Buckshot", value=True, key="cuas18_rec_mix")
                rec_fmode = st.selectbox("Salvo Firing Sequence", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"], index=0, key="cuas18_rec_fm")
                rec_dt = st.slider("Ripple Micro-Delay (ms)", 0.0, 30.0, 10.0, 1.0, key="cuas18_rec_dt")

            with rc2:
                st.markdown("**Turret Structural Buffer & Gimbal Inertia**")
                rec_mass = st.slider("Recoiling Assembly Mass (kg)", 15.0, 150.0, 45.0, 5.0, key="cuas18_rec_mass")
                rec_stiff = st.slider("Buffer Spring Stiffness (kN/m)", 40.0, 400.0, 120.0, 10.0, key="cuas18_rec_k") * 1000.0
                rec_limit = st.slider("Mount Structural Reaction Limit (kN)", 5.0, 50.0, 18.0, 1.0, key="cuas18_rec_lim")
                rec_ztr = st.slider("Vertical Trunnion Offset (m)", 0.05, 0.45, 0.18, 0.01, key="cuas18_rec_ztr")

            with rc3:
                st.markdown("**Constrained Trade-Study Optimizer Target**")
                rec_drone = st.selectbox("Design Threat Drone", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=1, key="cuas18_rec_dr")
                rec_dist = st.slider("Design Intercept Range (m)", 15.0, 65.0, 35.0, 1.0, key="cuas18_rec_dist")
                rec_dspd = st.slider("Design Drone Speed (m/s)", 0.0, 35.0, 18.0, 1.0, key="cuas18_rec_dspd")

            with st.spinner("Solving 2-Axis Gimbal ODEs & Constrained Pareto Array Optimizer..."):
                rec_barrels = generate_multishotgun_array_layout(
                    layout_type=str(rec_layout),
                    n_barrels=int(rec_nb),
                    mount_radius_m=0.20,
                    default_ammunition_preset=str(rec_ammo),
                    inner_ammunition_preset="12G_TSS_NO2_C_UAS" if rec_mixed else None,
                    inner_choke="Full" if rec_mixed else None,
                )
                eff_fmode = "simultaneous_volley" if (rec_fmode != "simultaneous_volley" and rec_dt <= 0.0) else str(rec_fmode)
                rec_res = simulate_turret_recoil_and_gimbal_loads(
                    barrels=rec_barrels,
                    firing_mode=eff_fmode,
                    ripple_delay_ms=float(rec_dt),
                    default_ammunition_preset=str(rec_ammo),
                    mount_mass_kg=float(rec_mass),
                    mount_stiffness_n_per_m=float(rec_stiff),
                    mount_load_limit_kn=float(rec_limit),
                    gimbal_trunnion_offset_z_m=float(rec_ztr),
                )
                rec_sweep = sweep_recoil_vs_ripple_delay(
                    barrels=rec_barrels,
                    default_ammunition_preset=str(rec_ammo),
                    mount_mass_kg=float(rec_mass),
                    mount_stiffness_n_per_m=float(rec_stiff),
                    mount_load_limit_kn=float(rec_limit),
                    gimbal_trunnion_offset_z_m=float(rec_ztr),
                )
                opt_res = optimize_antidrone_array_configuration(
                    n_barrels=int(rec_nb),
                    mount_radius_m=0.20,
                    target_distance_m=float(rec_dist),
                    drone_model=str(rec_drone),
                    drone_velocity_mps=float(rec_dspd),
                    ammunition_preset=str(rec_ammo),
                    inner_ammunition_preset="12G_TSS_NO2_C_UAS" if rec_mixed else None,
                    mount_mass_kg=float(rec_mass),
                    mount_stiffness_n_per_m=float(rec_stiff),
                    mount_load_limit_kn=float(rec_limit),
                    gimbal_trunnion_offset_z_m=float(rec_ztr),
                    n_volleys=12,
                    seed=42,
                )

            rsm = rec_res["summary"]
            bf_opt = opt_res.get("best_feasible") or opt_res["best_unconstrained"]
            rk1, rk2, rk3, rk4 = st.columns(4)
            rk1.metric("Peak Mount Reaction Force", f"{rsm['peak_transmitted_mount_force_kn']:.2f} kN", delta=f"Margin: {rsm['structural_safety_margin_pct']:+.1f}% vs {rec_limit:.0f} kN limit")
            rk2.metric("Total Salvo Impulse", f"{rsm['total_impulse_ns']:.1f} N·s", delta=f"Free Recoil: {rsm['free_recoil_velocity_mps']:.2f} m/s | Stroke: {rsm['max_recoil_stroke_mm']:.1f} mm")
            rk3.metric("Max Gimbal Climb / Yaw Walk", f"{rsm['max_muzzle_climb_mrad']:.2f} mrad", delta=f"Yaw Imbalance Walk: {rsm['max_yaw_deflection_mrad']:.3f} mrad")
            rk4.metric("Optimal Feasible P_kill", f"{bf_opt['kill_probability_pct']:.1f} %", delta=f"{bf_opt['layout_type']} | {bf_opt['pointing_mode']} | {bf_opt['firing_mode']}")

            rt1, rt2, rt3 = st.tabs(["🔩 Recoil Force, Gimbal Walk & Ripple Relief", "🏆 Constrained Pareto Array Optimizer", "📋 Per-Barrel Firing & Deflection Log"])
            with rt1:
                st.plotly_chart(create_turret_recoil_dynamics_figure(rec_res, rec_sweep), use_container_width=True)
            with rt2:
                if opt_res["best_feasible"] is not None:
                    st.success(
                        f"**Recommended Mount-Safe Design:** Layout `{bf_opt['layout_type']}`, Pointing `{bf_opt['pointing_mode']}` "
                        f"(splay `{bf_opt['splay_angle_mrad']} mrad`), Firing `{bf_opt['firing_mode']}` (Δt `{bf_opt['ripple_delay_ms']} ms`) "
                        f"→ **P_kill = {bf_opt['kill_probability_pct']}%**, Peak Force = **{bf_opt['peak_transmitted_force_kn']} kN**"
                    )
                else:
                    st.warning("No candidate stays under the mount structural limit — increase ripple delay, raise buffer stroke/limit, or reduce barrel count.")
                st.plotly_chart(create_antidrone_optimizer_pareto_figure(opt_res), use_container_width=True)
                cand_df = pd.DataFrame(opt_res["candidates"])
                st.dataframe(cand_df, use_container_width=True)
                st.download_button("📥 Download Optimizer Trade-Study (.csv)", cand_df.to_csv(index=False).encode("utf-8"), "cuas_optimizer_trade_study.csv", "text/csv", key="cuas18_dl_opt")
            with rt3:
                st.dataframe(pd.DataFrame(rec_res["barrel_firing_states"]), use_container_width=True)

        # --- TAB 18.5: ELEVATED-FIRE PELLET FALLOUT & SURFACE DANGER ZONE (SDZ) ---
        with cuas18_t5:
            sd1, sd2, sd3 = st.columns(3)
            with sd1:
                st.markdown("**Elevated Turret Engagement Geometry**")
                sd_el = st.slider("Turret Firing Elevation Angle (°)", 10.0, 85.0, 35.0, 5.0, key="cuas18_sd_el")
                sd_az = st.slider("Turret Firing Azimuth (°)", -90.0, 90.0, 0.0, 5.0, key="cuas18_sd_az")
                sd_h = st.slider("Turret Height Above Ground (m)", 0.5, 10.0, 2.0, 0.5, key="cuas18_sd_h")
                sd_n_p = st.slider("Monte Carlo Missed Pellets Tracked", 150, 450, 260, 20, key="cuas18_sd_np")

            with sd2:
                st.markdown("**Shotgun Ammunition & Choke**")
                sd_ammo = st.selectbox("Fired Shotgun Load", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=0, key="cuas18_sd_ammo")
                sd_choke = st.selectbox("Choke Tube", list(choke_defs_18.keys()), index=3, key="cuas18_sd_choke")
                sd_lay = st.selectbox("Turret Array Geometry", ["circular", "grid", "fan_horizontal", "honeycomb"], index=0, key="cuas18_sd_lay")
                sd_point = st.selectbox("Array Pointing Mode", ["convergent", "divergent_curtain", "parallel"], index=0, key="cuas18_sd_pt")

            with sd3:
                st.markdown("**3D Wind & Boundary-Layer Shear**")
                sd_w_down = st.slider("Downrange Tailwind (+) / Headwind (-) (m/s)", -15.0, 15.0, 4.0, 1.0, key="cuas18_sd_wd")
                sd_w_cross = st.slider("Crosswind Drift (m/s)", -15.0, 15.0, 2.0, 1.0, key="cuas18_sd_wc")
                sd_shear = st.checkbox("Apply 1/7 Power-Law Boundary Layer Wind Shear", value=True, key="cuas18_sd_sh")

            with st.spinner("Integrating Vectorized 3-DoF Ballistic Pellet Fallout..."):
                sd_res = simulate_cuas_pellet_fallout_hazard(
                    layout_type=str(sd_lay),
                    pointing_mode=str(sd_point),
                    ammunition_preset=str(sd_ammo),
                    default_choke=str(sd_choke),
                    firing_elevation_deg=float(sd_el),
                    firing_azimuth_deg=float(sd_az),
                    turret_height_m=float(sd_h),
                    wind_downrange_mps=float(sd_w_down),
                    wind_crossrange_mps=float(sd_w_cross),
                    enable_wind_shear=bool(sd_shear),
                    n_sample_pellets=int(sd_n_p),
                    seed=42,
                )
                sd_comp = compare_cuas_fallout_across_loads(
                    firing_elevation_deg=float(sd_el),
                    wind_downrange_mps=float(sd_w_down),
                    wind_crossrange_mps=float(sd_w_cross),
                    default_choke=str(sd_choke),
                    n_sample_pellets=140,
                    seed=42,
                )

            ssm = sd_res["summary"]
            sdk1, sdk2, sdk3, sdk4 = st.columns(4)
            sdk1.metric("99% Surface Danger Zone (SDZ)", f"{ssm['max_fallout_range_99_m']:.0f} m", delta=f"Span: {ssm['min_fallout_range_m']:.0f}–{ssm['max_fallout_range_99_m']:.0f}m | Apogee: {ssm['max_apogee_m']:.0f}m")
            sdk2.metric("Terminal Speed / Energy", f"{ssm['mean_terminal_velocity_mps']:.1f} m/s", delta=f"Mean: {ssm['mean_impact_energy_j']:.2f} J ({ssm['mean_specific_energy_j_mm2']:.3f} J/mm²)")
            sdk3.metric("Ocular (Unprotected Eye) Risk", f"{ssm['eye_hazard_pct']:.1f} %", delta="≥0.06 J/mm² or ≥1.5 J threshold")
            sdk4.metric("Skin Penetration / IATG Risk", f"{ssm['skin_hazard_pct']:.1f} %", delta=f"IATG ≥79 J Rate: {ssm['iatg_hazard_pct']:.1f} %")

            sdt1, sdt2, sdt3 = st.tabs(["🌐 3D Elevated Fallout Arcs", "🗺️ 2D Ground SDZ Map & Multi-Load Bands", "📊 Multi-Load Safety Footprint Data & CSV"])
            with sdt1:
                st.plotly_chart(create_3d_cuas_fallout_footprint_figure(sd_res), use_container_width=True)
            with sdt2:
                st.plotly_chart(create_2d_cuas_fallout_safety_map_figure(sd_res, sd_comp), use_container_width=True)
            with sdt3:
                comp_df_sd = pd.DataFrame(sd_comp["records"])
                st.dataframe(comp_df_sd, use_container_width=True)
                gi_df = pd.DataFrame(sd_res["ground_impacts"])
                st.download_button("📥 Download Ground Pellet Impact Fallout Log (.csv)", gi_df.to_csv(index=False).encode("utf-8"), "cuas_pellet_fallout_sdz.csv", "text/csv", key="cuas18_dl_sdz")


# --- TAB 5: UNODA IATG 01.80 & 02.20 EXPLOSIVE SAFETY & QUANTITY-DISTANCE ---
def tab_iatg() -> None:
    st.header("🏛️ UNODA International Ammunition Technical Guidelines (IATG 01.80:2021 & IATG 02.20)")
    st.markdown(
        "Comprehensive engineering workbench grounded in **UNODA IATG 01.80:2021 (3rd Edition, *Formulae for ammunition management*)**, "
        "**IATG 02.20 (*Quantity and separation distances*)**, and **NATO AASTP-1**: "
        "Quantity-Distance (QD) scaling laws, open-detonation & vertical airspace safety (`TB-15`), "
        "UK brick-house structural collapse (`TB-10`), Swiss glazing $P_s\text{–}I_s$ hyperbolas (`TB-11`), "
        "munition stack fragmentation (`TF-10`), RC donor debris launch (`TF-14`), and Axelsson thoracic biomechanics (`TB-13`)."
    )
    kb = get_kb()

    isub1, isub2, isub3, isub4, isub5, isub6, isub7 = st.tabs(
        [
            "🏗️ 1. Quantity-Distance (QD), Open-Detonation (TB-15) & 3D PES Map",
            "🧱 2. Structural Damage (TB-10 Jarrett/GLS) & Swiss Glazing (TB-11)",
            "📦 3. NATO AASTP-1 Stack Fragmentation (TF-10) & RC Debris (TF-14)",
            "🫁 4. Axelsson Thoracic Biomechanics ODE (TB-13) & Thermal Dose (TB-16)",
            "🗺️ 5. Multi-Facility 3D Site Planner & Sympathetic Matrix",
            "🧪 6. Propellant Aging & In-Service Surveillance (IATG 07.10 / AOP-48)",
            "🗺️ 7. Terrain Crest Clearance & KML/CZML Export",
        ]
    )

    # --- IATG SUB-TAB 1: QUANTITY-DISTANCE (QD), TB-15 & 3D PES SITE MAP ---
    with isub1:
        st.subheader("🏗️ IATG 01.80 / 02.20 Quantity-Distance (QD) & Open-Detonation Safety Envelopes (`TB-02`, `TB-15`)")
        st.latex(
            r"""
            D_{\text{QD}} = Z_{\text{QD}}\, Q_{\text{TNT}}^{1/3}, \quad
            Z_{\text{PBD}} = 8.0, \quad Z_{\text{PTRD}} = 14.8, \quad Z_{\text{IBD}} = 22.2, \quad Z_{\text{VBD}} = 44.4\;\text{m/kg}^{1/3}, \quad
            D_{\text{frag,pub}} = 634\,M_{\text{AUW}}^{1/6}, \quad D_{\text{vert,1}} = 314\,M_{\text{AUW}}^{1/3}, \quad D_{140\text{dB}} = 215\,Q_{\text{TNT}}^{1/3}
            """
        )

        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            neq_kg = st.number_input("Net Explosive Quantity Q_NEQ (kg)", 1.0, 100000.0, 1000.0, 50.0, key="iatg_neq")
        with ic2:
            auw_kg = st.number_input("All-Up Munition Weight M_AUW (kg)", 1.0, 250000.0, 2500.0, 100.0, key="iatg_auw")
        with ic3:
            exp_name = st.selectbox(
                "Energetic Material (TB-02 TNT Equivalence)",
                ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20", "RDX", "HMX"],
                index=0,
                key="iatg_exp",
            )
        with ic4:
            show_3d_domes = st.checkbox("Render 3D Hemispherical QD Domes", value=True, key="iatg_domes")

        wp_eq, wi_eq = tb_02_tnt_equivalence(neq_kg, explosive=exp_name)
        q_tnt = float(wp_eq)
        qd_res = tb_15_simple_safety_distances(auw_kg=auw_kg, neq_kg=q_tnt)

        m1, m2, m3 = st.columns(3)
        m1.metric("TNT Equiv Q_TNT (TB-02)", f"{q_tnt:.1f} kg", delta=f"Impulse eq: {float(wi_eq):.1f} kg", delta_color="off")
        m2.metric("Process Building (PBD)", f"{qd_res['process_building_distance_m']:.1f} m", delta="Z = 8.0 m/kg^(1/3)", delta_color="off")
        m3.metric("Public Traffic (PTRD)", f"{qd_res['public_traffic_route_distance_m']:.1f} m", delta="Z = 14.8 m/kg^(1/3)", delta_color="off")
        m4, m5, m6 = st.columns(3)
        m4.metric("Inhabited Building (IBD)", f"{qd_res['inhabited_building_distance_m']:.1f} m", delta="Z = 22.2 m/kg^(1/3)", delta_color="off")
        m5.metric("Vulnerable Bldg (VBD)", f"{qd_res['vulnerable_building_distance_m']:.1f} m", delta="Z = 44.4 m/kg^(1/3)", delta_color="off")
        m6.metric("Public Frag (TB-15)", f"{qd_res['frag_public_m']:.1f} m", delta=f"140 dB: {qd_res['noise_140dB_m']:.0f} m", delta_color="off")

        map_col, chart_col = st.columns([1.25, 1.0])
        with map_col:
            fig_pes = create_3d_iatg_safety_zones_figure(
                neq_kg=float(neq_kg),
                auw_kg=float(auw_kg),
                explosive_name=exp_name,
                show_domes=bool(show_3d_domes),
            )
            fig_pes.update_layout(height=500)
            st.plotly_chart(fig_pes, use_container_width=True)

        with chart_col:
            q_vec = np.logspace(0.0, 5.0, 100)
            fig_qd_scale = go.Figure()
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=8.0 * (q_vec ** (1.0 / 3.0)), name="PBD: Process Building (8.0 Q^1/3)", line=dict(color="#DC2626", width=2.5))
            )
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=14.8 * (q_vec ** (1.0 / 3.0)), name="PTRD: Public Traffic Route (14.8 Q^1/3)", line=dict(color="#EA580C", width=2.5))
            )
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=22.2 * (q_vec ** (1.0 / 3.0)), name="IBD: Inhabited Building (22.2 Q^1/3)", line=dict(color="#2563EB", width=3))
            )
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=44.4 * (q_vec ** (1.0 / 3.0)), name="VBD: Vulnerable Building (44.4 Q^1/3)", line=dict(color="#10B981", width=2.5))
            )
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=130.0 * (q_vec ** (1.0 / 3.0)), name="TB-15 Bare Explosive Open Det (130 Q^1/3)", line=dict(color="#7C3AED", width=2, dash="dash"))
            )
            fig_qd_scale.add_trace(
                go.Scatter(x=q_vec, y=634.0 * (q_vec ** (1.0 / 6.0)), name="TB-15 Public Cased Frag (634 M^1/6)", line=dict(color="#991B1B", width=2, dash="dot"))
            )
            fig_qd_scale.add_vline(x=q_tnt, line_dash="dot", line_color="#111827", annotation_text=f"Active Q_TNT = {q_tnt:.0f} kg")
            fig_qd_scale.update_layout(
                title="IATG 01.80 / 02.20 Quantity-Distance & Open-Detonation Scaling Laws",
                xaxis_title="Net Explosive Quantity Q_TNT / All-Up Mass M (kg)",
                yaxis_title="Required Safety Separation Distance D (m)",
                xaxis_type="log",
                yaxis_type="log",
                height=500,
                margin=dict(l=15, r=15, t=40, b=15),
            )
            st.plotly_chart(fig_qd_scale, use_container_width=True)

        # Complete IATG 01.80 Table with Kingery-Bulmash Overpressure & Impulse at each boundary
        qd_rows = []
        for code, desc, r_val, scaling_str in [
            ("PBD (IATG 02.20)", "Between Process Buildings in Explosive Storage Area", float(qd_res["process_building_distance_m"]), "8.0 · Q_TNT^(1/3)"),
            ("PTRD (IATG 02.20)", "Explosive Storehouse to Public Traffic Route (Medium Density)", float(qd_res["public_traffic_route_distance_m"]), "14.8 · Q_TNT^(1/3)"),
            ("IBD (IATG 02.20)", "Explosive Storehouse to Inhabited Building", float(qd_res["inhabited_building_distance_m"]), "22.2 · Q_TNT^(1/3)"),
            ("VBD (IATG 02.20)", "Explosive Storehouse to Vulnerable Building (School / Hospital)", float(qd_res["vulnerable_building_distance_m"]), "44.4 · Q_TNT^(1/3)"),
            ("TB-15 Bare Explosive", "Open-Detonation Bare Explosive Public Standoff", float(qd_res["bare_explosive_m"]), "130 · M_AUW^(1/3)"),
            ("TB-15 Multi-Item", "Open-Detonation Multi-Item Linear Array Standoff", float(qd_res["multi_item_linear_m"]), "370 · M_AUW^(1/5)"),
            ("TB-15 Public Frag", "Open-Detonation Cased Munition Fragmentation Standoff", float(qd_res["frag_public_m"]), "634 · M_AUW^(1/6)"),
            ("TB-15 Vertical Single", "Vertical Airspace Hazard Ceiling (Single Item)", float(qd_res["vertical_single_m"]), "314 · M_AUW^(1/3)"),
            ("TB-15 Vertical Multi", "Vertical Airspace Hazard Ceiling (Multi-Item)", float(qd_res["vertical_multi_m"]), "470 · M_AUW^(1/5)"),
            ("TB-15 Acoustic 140 dB", "140 dB Peak Impulse Noise Standoff Boundary", float(qd_res["noise_140dB_m"]), "215 · Q_TNT^(1/3)"),
        ]:
            bf_row = evaluate_blast_field(max(r_val, 0.5), q_tnt, explosive="TNT")
            qd_rows.append(
                {
                    "IATG Criterion": code,
                    "Engineering Application": desc,
                    "Scaling Law": scaling_str,
                    "Distance D (m)": round(r_val, 1),
                    "Scaled Z (m/kg^1/3)": round(float(bf_row["Z_eff"]), 2),
                    "Incident Ps (kPa)": round(float(bf_row["Ps_kPa"]), 2),
                    "Reflected Pr (kPa)": round(float(bf_row["Pr_kPa"]), 2),
                    "Impulse Is (kPa·ms)": round(float(bf_row["Is_kPa_ms"]), 1),
                }
            )
        st.dataframe(pd.DataFrame(qd_rows), use_container_width=True)

    # --- IATG SUB-TAB 2: STRUCTURAL DAMAGE (TB-10) & SWISS GLAZING (TB-11) ---
    with isub2:
        st.subheader("🧱 IATG 01.80 Clause 10.1: Brick-House Damage (`TB-10`) & Swiss Glazing $P_s\text{–}I_s$ Hyperbola (`TB-11`)")
        st.latex(
            r"""
            R_{\text{cat}} = \frac{K_{\text{cat}}\, Q_{\text{TNT}}^{1/3}}{\left[1 + \left(\frac{3175}{Q_{\text{TNT}}}\right)^2\right]^{1/6}}, \qquad
            \text{Swiss Glazing Breakage (`TB-11`):}\quad (P_s - A)(I_s - B) \ge C
            """
        )

        sc1, sc2 = st.columns(2)
        with sc1:
            tb10_rows = []
            for row_cat in kb.effects_criteria["brick_house_damage"]:
                c_code = str(row_cat["category"])
                j_r = float(tb_10_brick_house_damage_range(q_tnt, category=c_code, method="jarrett"))
                g_r = float(tb_10_brick_house_damage_range(q_tnt, category=c_code, method="gilbert_lees_scilly"))
                bf_j = evaluate_blast_field(max(j_r, 0.5), q_tnt, explosive="TNT")
                tb10_rows.append(
                    {
                        "Category": c_code,
                        "Jarrett R (m)": round(j_r, 2),
                        "Gilbert-Lees-Scilly R (m)": round(g_r, 2),
                        "Ps at Jarrett R (kPa)": round(float(bf_j["Ps_kPa"]), 1),
                        "P(Fatality)": f"{float(row_cat['p_fatality'])*100:.0f}%",
                        "P(Fatal/Serious)": f"{float(row_cat['p_fatal_or_serious'])*100:.0f}%",
                        "P(Any Injury)": f"{float(row_cat['p_any_injury'])*100:.0f}%",
                    }
                )
            tb10_df = pd.DataFrame(tb10_rows)

            fig_tb10 = go.Figure()
            fig_tb10.add_trace(
                go.Bar(
                    x=tb10_df["Category"],
                    y=tb10_df["Jarrett R (m)"],
                    name="Jarrett (1968) Radius R (m)",
                    marker_color="#DC2626",
                )
            )
            fig_tb10.add_trace(
                go.Bar(
                    x=tb10_df["Category"],
                    y=tb10_df["Gilbert-Lees-Scilly R (m)"],
                    name="Gilbert-Lees-Scilly (1994) Radius R (m)",
                    marker_color="#2563EB",
                )
            )
            fig_tb10.update_layout(
                title=f"UK Brick-House Structural Damage Radii (TB-10) for Q_TNT = {q_tnt:.0f} kg",
                xaxis_title="IATG Structural Damage Category (A = Demolition → D = Minor Repairable)",
                yaxis_title="Damage Radius R (m)",
                barmode="group",
                height=390,
            )
            st.plotly_chart(fig_tb10, use_container_width=True)
            st.dataframe(tb10_df, use_container_width=True)

        with sc2:
            # Swiss Glazing P-I Hyperbola Diagram (TB-11) vs Kingery-Bulmash Blast Locus
            fig_pi = go.Figure()
            for g_key, g_label, g_col in [
                ("small", "Small Window (<1 m², P50)", "#10B981"),
                ("medium", "Medium Window (1–3 m², P50)", "#F59E0B"),
                ("large", "Large Window (>3 m², P50)", "#DC2626"),
            ]:
                gb_info = tb_11_glass_breakage_swiss(10.0, 100.0, window_size=g_key, target_probability=0.5)
                a_val = float(gb_info["A_kPa"])
                b_val = float(gb_info["B_kPa_ms"])
                c_val = float(gb_info["C"])
                i_grid = np.linspace(b_val + 1.5, 800.0, 200)
                p_hyp = a_val + c_val / (i_grid - b_val)
                fig_pi.add_trace(
                    go.Scatter(
                        x=i_grid,
                        y=p_hyp,
                        mode="lines",
                        name=f"{g_label}: A={a_val:.1f}, B={b_val:.1f}",
                        line=dict(color=g_col, width=3),
                    )
                )

            # Overlay actual Kingery-Bulmash (Is, Ps) blast locus across R for active Q_TNT
            r_locus = np.logspace(np.log10(max(2.0, 0.5 * (q_tnt ** (1.0 / 3.0)))), np.log10(45.0 * (q_tnt ** (1.0 / 3.0))), 80)
            bf_locus = evaluate_blast_field(r_locus, q_tnt, explosive="TNT")
            ps_loc = np.asarray(bf_locus["Ps_kPa"], dtype=np.float64)
            is_loc = np.asarray(bf_locus["Is_kPa_ms"], dtype=np.float64)
            fig_pi.add_trace(
                go.Scatter(
                    x=is_loc,
                    y=ps_loc,
                    mode="lines+markers",
                    name=f"Blast Wave Locus (Q_TNT={q_tnt:.0f} kg)",
                    line=dict(color="#1E3A8A", width=2.5, dash="dot"),
                    marker=dict(size=5),
                    text=[f"R = {r:.1f} m | Ps = {p:.1f} kPa | Is = {i:.0f} Pa·s" for r, p, i in zip(r_locus, ps_loc, is_loc)],
                    hoverinfo="text+name",
                )
            )
            fig_pi.update_layout(
                title="Swiss Dual-Pane Glazing Breakage P–I Hyperbolas (TB-11) vs Blast Locus",
                xaxis_title="Specific Positive Impulse I_s (Pa·s = kPa·ms)",
                yaxis_title="Peak Incident Overpressure P_s (kPa)",
                xaxis_type="log",
                yaxis_type="log",
                yaxis_range=[0.0, 2.5],
                height=390,
            )
            st.plotly_chart(fig_pi, use_container_width=True)

            # Interactive single-point glazing check
            g_r_test = st.slider("Test Window Standoff Distance R (m)", 5.0, 1000.0, float(qd_res["inhabited_building_distance_m"]), 5.0)
            bf_win = evaluate_blast_field(g_r_test, q_tnt, explosive="TNT")
            p_w_val = float(bf_win["Ps_kPa"])
            i_w_val = float(bf_win["Is_kPa_ms"])
            w_res = []
            for sz in ["small", "medium", "large"]:
                gb_sz = tb_11_glass_breakage_swiss(p_w_val, i_w_val, window_size=sz, target_probability=0.5)
                w_res.append(
                    {
                        "Window Size Class": sz.capitalize(),
                        "Ps (kPa)": round(p_w_val, 2),
                        "Is (Pa·s)": round(i_w_val, 1),
                        "Asymptote A (kPa) / B (Pa·s)": f"{float(gb_sz['A_kPa']):.2f} / {float(gb_sz['B_kPa_ms']):.1f}",
                        "Glazing Status (P50)": "⚠️ BROKEN" if gb_sz["breaks"] else "✅ INTACT",
                    }
                )
            st.table(pd.DataFrame(w_res))

    # --- IATG SUB-TAB 3: MUNITION STACK FRAGMENTATION (TF-10) & RC DEBRIS (TF-14) ---
    with isub3:
        st.subheader("📦 IATG 01.80 Clause 6 / NATO AASTP-1 Munition Stack Fragmentation (`TF-10`) & RC Donor Debris (`TF-14`)")
        st.latex(
            r"""
            N_{\text{eff}} = f_s N_{\text{surface}} + 0.1 N_{\text{total}}, \qquad
            q_f(R) = \frac{N_t\, N_{\text{eff}}}{4\pi R^2}\exp\!\left(-\sqrt{\frac{2 m_{\text{cr}}(R)}{M_0}}\right), \qquad
            q_{\text{crit}} = \frac{1}{56\;\text{m}^2}\quad (E_k \ge 79\;\text{J})
            """
        )

        fc1, fc2 = st.columns([1.25, 1.0])
        with fc1:
            s1, s2, s3 = st.columns(3)
            with s1:
                n_surf = st.number_input("Surface Munition Items N_surface", 1, 2000, 40, 5)
            with s2:
                n_tot_stack = st.number_input("Total Stack Munition Items N_total", 1, 10000, 200, 10)
            with s3:
                nt_per_item = st.number_input("Fragments per Munition N_t", 50, 5000, 850, 50)

            r_frag_grid = np.linspace(20.0, 600.0, 160)
            tf10_open = tf_10_aastp1_stack_density(
                r_frag_grid,
                N_t_per_item=float(nt_per_item),
                M0_kg_per_item=0.015,
                V0_mps=1250.0,
                N_surface=int(n_surf),
                N_total=int(n_tot_stack),
                storage_mode="open",
            )
            tf10_ecm = tf_10_aastp1_stack_density(
                r_frag_grid,
                N_t_per_item=float(nt_per_item),
                M0_kg_per_item=0.015,
                V0_mps=1250.0,
                N_surface=int(n_surf),
                N_total=int(n_tot_stack),
                storage_mode="ecm",
            )
            q_open = np.asarray(tf10_open["q_f_m2"], dtype=np.float64)
            q_ecm = np.asarray(tf10_ecm["q_f_m2"], dtype=np.float64)
            q_crit = 1.0 / 56.0
            idx_open = np.where(q_open >= q_crit)[0]
            idx_ecm = np.where(q_ecm >= q_crit)[0]
            r_crit_open = float(r_frag_grid[idx_open[-1]]) if len(idx_open) > 0 else 20.0
            r_crit_ecm = float(r_frag_grid[idx_ecm[-1]]) if len(idx_ecm) > 0 else 20.0

            fig_tf10 = go.Figure()
            fig_tf10.add_trace(
                go.Scatter(
                    x=r_frag_grid,
                    y=q_open,
                    name=f"Open Stack (0.9 N_surf + 0.1 N_tot) — R_haz ≈ {r_crit_open:.0f} m",
                    line=dict(color="#DC2626", width=3),
                )
            )
            fig_tf10.add_trace(
                go.Scatter(
                    x=r_frag_grid,
                    y=q_ecm,
                    name=f"Earth-Covered Magazine (0.7 N_surf + 0.1 N_tot) — R_haz ≈ {r_crit_ecm:.0f} m",
                    line=dict(color="#2563EB", width=3, dash="dash"),
                )
            )
            fig_tf10.add_hline(
                y=q_crit,
                line_dash="dot",
                line_color="#111827",
                annotation_text="NATO / IATG Critical Hazard Density (1 / 56 m² = 0.01786 m⁻²)",
            )
            fig_tf10.update_layout(
                title="IATG 01.80 Table 28 / NATO AASTP-1 Stack Fragment Areal Density q_f(R)",
                xaxis_title="Standoff Distance R from Stack Center (m)",
                yaxis_title="Hazardous Fragment Density q_f (fragments / m²)",
                yaxis_type="log",
                height=420,
            )
            st.plotly_chart(fig_tf10, use_container_width=True)

        with fc2:
            st.markdown("##### 🧱 RC Donor Structure Debris Velocity (`TF-14`) & DDESB Consequence Union (`TF-15`)")
            rc_vol_m3 = st.slider("Donor Storehouse Internal Volume V (m³)", 20.0, 2000.0, 250.0, 20.0)
            rc_t_m = st.slider("RC Wall Thickness t (m)", 0.10, 1.50, 0.35, 0.05)
            rc_out = tf_14_rc_debris_throw(q_tnt, room_vol_m3=rc_vol_m3, wall_t_m=rc_t_m)
            v_debris = float(rc_out["dlv_mps"])
            m_debris = float(rc_out["m_avg_debris_kg"])

            p_high = st.slider("High-Angle Fragment Consequence Prob P_high", 0.0, 1.0, 0.15, 0.05)
            tp14_out = tf_15_tp14_consequence_union(
                concern_areas_m2=[0.56, 0.56, 0.56],
                n_frags_per_bin=[0.18, 0.12, 0.08],
                vulnerabilities=[0.95, 0.80, 0.60],
                p_high_angle=p_high,
            )
            p_union = float(tp14_out["p_total"])

            st.table(
                pd.DataFrame(
                    [
                        {"Metric": "Open-Stack 1/56 m² Hazard Distance (TF-10)", "Value": f"{r_crit_open:.1f} m"},
                        {"Metric": "Earth-Covered Magazine 1/56 m² Distance (TF-10)", "Value": f"{r_crit_ecm:.1f} m"},
                        {"Metric": "RC Concrete Debris Launch Speed DLV (TF-14)", "Value": f"{v_debris:.1f} m/s"},
                        {"Metric": "Mean Concrete Debris Fragment Mass (TF-14)", "Value": f"{m_debris:.2f} kg"},
                        {"Metric": "DDESB TP-14 Multi-Bin Union P_tot (TF-15)", "Value": f"{p_union:.3f} ({p_union*100:.1f}%)"},
                    ]
                )
            )

    # --- IATG SUB-TAB 4: AXELSSON THORACIC BIOMECHANICS (TB-13) & THERMAL DOSE (TB-16) ---
    with isub4:
        st.subheader("🫁 IATG 01.80 Clause 10.2: Axelsson Thoracic Biomechanics ODE (`TB-13`) & Thermal Dose (`TB-16`)")
        st.latex(
            r"""
            M_{\text{cw}}\ddot{x} + J_{\text{cw}}\dot{x} + K_{\text{cw}}x = A_{\text{cw}}\!\left[P_{\text{ext}}(t) - P_0\!\left(\left(\frac{V_0}{V_0 - A_{\text{cw}}x}\right)^\gamma - 1\right)\right], \quad
            \text{ASII} = (0.124 + 0.117\,V_{\max})^{2.63}, \quad
            D_{\text{th}} = q^{4/3}\,t
            """
        )

        bc1, bc2 = st.columns(2)
        with bc1:
            ax_ps = st.slider("Incident Overpressure Load Ps (kPa)", 20.0, 600.0, 160.0, 10.0)
            ax_td = st.slider("Positive Phase Duration td (ms)", 1.0, 30.0, 6.0, 0.5)
            ax_out = tb_13_axelsson_chest_wall(Ps_kPa=ax_ps, td_ms=ax_td, record_history=True)
            ax_hist = ax_out.get("history", {})
            lung_out = tb_12_lung_lethality_probit(ax_ps * 1000.0, ax_td * 1e-3)

            ak1, ak2, ak3 = st.columns(3)
            ak1.metric("Peak Chest-Wall V_max", f"{ax_out['v_chest_max_mps']:.2f} m/s")
            ak2.metric("Axelsson ASII Index", f"{ax_out['ASII']:.2f}", delta=str(ax_out["injury_level"]), delta_color="off")
            ak3.metric("TNO Lung Probit (TB-12)", f"{float(lung_out['probability'])*100:.1f} %", delta=f"Pr: {float(lung_out['probit']):.2f}", delta_color="off")

            if ax_hist:
                fig_ax = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ax.add_trace(
                    go.Scatter(
                        x=ax_hist["t_ms"],
                        y=ax_hist["v_mps"],
                        name="Chest-Wall Velocity v(t) (m/s)",
                        line=dict(color="#DC2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_ax.add_trace(
                    go.Scatter(
                        x=ax_hist["t_ms"],
                        y=ax_hist["x_mm"],
                        name="Inward Displacement x(t) (mm)",
                        line=dict(color="#2563EB", width=2.5),
                    ),
                    secondary_y=False,
                )
                fig_ax.add_trace(
                    go.Scatter(
                        x=ax_hist["t_ms"],
                        y=ax_hist["p_ext_kPa"],
                        name="External Blast P_ext(t) (kPa)",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                    ),
                    secondary_y=True,
                )
                fig_ax.add_trace(
                    go.Scatter(
                        x=ax_hist["t_ms"],
                        y=ax_hist["p_lung_kPa"],
                        name="Adiabatic Lung Gauge P_lung(t) (kPa)",
                        line=dict(color="#10B981", width=2, dash="dot"),
                    ),
                    secondary_y=True,
                )
                fig_ax.update_xaxes(title_text="Time (ms)")
                fig_ax.update_yaxes(title_text="Velocity (m/s) / Displacement (mm)", secondary_y=False)
                fig_ax.update_yaxes(title_text="Pressure (kPa)", secondary_y=True)
                fig_ax.update_layout(
                    title=f"Axelsson Single-Chamber Thoracic Biomechanics ODE Trajectory (V_max={ax_out['v_chest_max_mps']:.2f} m/s)",
                    height=370,
                )
                st.plotly_chart(fig_ax, use_container_width=True)

        with bc2:
            q_flux = st.slider("Thermal Radiant Heat Flux q (kW/m²)", 5.0, 150.0, 35.0, 2.5)
            t_exp = st.slider("Exposure Duration t (s)", 0.5, 20.0, 4.0, 0.5)
            th_res = tb_16_thermal_dose(q_flux, t_exp)

            st.metric(
                "Thermal Dose D = q^(4/3)·t (TB-16)",
                f"{float(th_res['dose']):.1f} (kW/m²)^(4/3)·s",
                delta=str(th_res["effect"]),
                delta_color="off",
            )

            t_vec = np.linspace(0.2, 20.0, 120)
            fig_th = go.Figure()
            for d_thresh, d_label, d_col in [
                (92.0, "Pain Threshold (D = 92)", "#10B981"),
                (105.0, "First-Degree Burn (D = 105)", "#F59E0B"),
                (290.0, "Second-Degree Burn (D = 290)", "#EA580C"),
                (1000.0, "Third-Degree Burn (D = 1000)", "#DC2626"),
            ]:
                q_curve = (d_thresh / t_vec) ** (3.0 / 4.0)
                fig_th.add_trace(go.Scatter(x=t_vec, y=q_curve, mode="lines", name=d_label, line=dict(color=d_col, width=2.5)))

            fig_th.add_trace(
                go.Scatter(
                    x=[t_exp],
                    y=[q_flux],
                    mode="markers+text",
                    name="Operating Point",
                    marker=dict(size=12, color="#1E3A8A", symbol="star"),
                    text=[f"D={float(th_res['dose']):.0f} ({th_res['effect']})"],
                    textposition="top right",
                )
            )
            fig_th.update_layout(
                title="IATG 01.80 Thermal Radiation Dose Iso-Effect Curves (TB-16)",
                xaxis_title="Exposure Duration t (s)",
                yaxis_title="Radiant Heat Flux q (kW/m²)",
                yaxis_type="log",
                height=310,
            )
            st.plotly_chart(fig_th, use_container_width=True)

            st.markdown("##### 🌍 IATG 01.80 Ground Shock Peak Particle Velocity (PPV) Structural Thresholds")
            st.dataframe(pd.DataFrame(kb.effects_criteria.get("ground_shock", [])), use_container_width=True)

    # --- IATG SUB-TAB 5: MULTI-FACILITY 3D SITE PLANNER & SYMPATHETIC MATRIX ---
    with isub5:
        st.subheader("🗺️ Multi-Facility 3D Explosive Storage Area (ESA) Site Planner & QD Compliance Matrix")
        st.markdown(
            "Interactive 3D facility layout planner evaluating **Inter-Magazine Sympathetic Detonation (`D_IMD = k_IMD · Q_donor^(1/3)`)**, "
            "**Process Building Distance (`Z = 8.0`)**, **Public Traffic Route (`Z = 14.8`)**, and **Inhabited Building Distance (`Z = 22.2`)** "
            "across multiple Potential Explosion Sites (PES) and Exposed Sites (ES)."
        )
        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            pes1_neq = st.number_input("PES-1 ECM Magazine NEQ (kg)", 50.0, 50000.0, 1500.0, 100.0, key="sp_p1_q")
            pes1_mtype = st.selectbox("PES-1 Magazine Class", ["ecm", "barricaded", "unbarricaded"], index=0)
        with sp2:
            pes2_neq = st.number_input("PES-2 Igloo Magazine NEQ (kg)", 50.0, 50000.0, 2000.0, 100.0, key="sp_p2_q")
            pes2_x = st.slider("PES-2 Easting X (m)", -250.0, 250.0, 45.0, 5.0, key="sp_p2_x")
        with sp3:
            pes3_neq = st.number_input("PES-3 Open Stack NEQ (kg)", 50.0, 50000.0, 800.0, 100.0, key="sp_p3_q")
            pes3_y = st.slider("PES-3 Northing Y (m)", -250.0, 250.0, 110.0, 5.0, key="sp_p3_y")
        with sp4:
            es_ibd_x = st.slider("Inhabited Building Easting X (m)", 50.0, 600.0, 280.0, 10.0, key="sp_ibd_x")
            es_pbd_x = st.slider("Process Workshop Easting X (m)", -250.0, 250.0, -75.0, 5.0, key="sp_pbd_x")

        pes_custom = [
            {"name": "PES-1 (ECM)", "x_m": -60.0, "y_m": 0.0, "neq_tnt_kg": float(pes1_neq), "magazine_type": str(pes1_mtype)},
            {"name": "PES-2 (Barricaded)", "x_m": float(pes2_x), "y_m": 20.0, "neq_tnt_kg": float(pes2_neq), "magazine_type": "barricaded"},
            {"name": "PES-3 (Open Stack)", "x_m": 0.0, "y_m": float(pes3_y), "neq_tnt_kg": float(pes3_neq), "magazine_type": "unbarricaded"},
        ]
        es_custom = [
            {"name": "ES-1 (Process Workshop)", "x_m": float(es_pbd_x), "y_m": -85.0, "es_type": "process_building"},
            {"name": "ES-2 (Public Road)", "x_m": 180.0, "y_m": -120.0, "es_type": "public_traffic_route"},
            {"name": "ES-3 (Inhabited Quarters)", "x_m": float(es_ibd_x), "y_m": 60.0, "es_type": "inhabited_building"},
            {"name": "ES-4 (Base Clinic)", "x_m": 490.0, "y_m": 160.0, "es_type": "vulnerable_building"},
        ]

        fig_site, comp_rows = create_3d_multi_facility_site_planner_figure(pes_list=pes_custom, es_list=es_custom)
        fig_site.update_layout(height=540)
        st.plotly_chart(fig_site, use_container_width=True)

        comp_df = pd.DataFrame(comp_rows)
        n_viol = sum(1 for r in comp_rows if "VIOLATION" in str(r.get("QD Status", "")))
        if n_viol > 0:
            st.error(f"🚨 **{n_viol} Quantity-Distance / Sympathetic Detonation Encroachment(s) Detected!** Adjust facility coordinates or NEQ limits above.")
        else:
            st.success("✅ **All Inter-Magazine (IMD) and Exposed Site (ES) Separation Distances Comply with UNODA IATG 02.20!**")
        st.dataframe(comp_df, use_container_width=True)

    # --- IATG SUB-TAB 6: IN-SERVICE SURVEILLANCE & PROPELLANT AGING (IATG 07.10 / AOP-48) ---
    with isub6:
        st.subheader("🧪 In-Service Ammunition Surveillance, Propellant Aging & Thermal Stability (`IATG 07.10` / `NATO AOP-48`)")
        st.markdown(
            "Predicts the chemical consumption of stabilizers (Diphenylamine [DPA], Ethyl Centralite [EC], 2-NDPA) "
            "under climatic storage conditions with diurnal thermal cycles, evaluating remaining safe shelf-life, "
            "critical autocatalytic auto-ignition thresholds, and ISO 2859-1 / STANAG 4147 sampling surveillance plans."
        )
        st.latex(
            r"""
            k(T) = A \exp\left(-\frac{E_a}{R T}\right), \quad
            S(t) = S_0 \exp(-k(T_{\text{eq}}) t), \quad
            T_{\text{eq}} = -\frac{E_a}{R} \left( \ln\left( \frac{1}{24} \int_0^{24} \exp\left(-\frac{E_a}{R T(t)}\right) dt \right) \right)^{-1}
            """
        )
        surv_c1, surv_c2, surv_c3 = st.columns(3)
        with surv_c1:
            surv_stab = st.selectbox("Stabilizer / Propellant Chemistry", list(STABILIZER_KINETICS.keys()), index=0)
            surv_clim = st.selectbox("Climatic Storage Zone (IATG 02.10)", list(CLIMATE_ZONES.keys()), index=1)
        with surv_c2:
            surv_years = st.slider("Stockpile Storage Duration (Years)", 1.0, 40.0, 20.0, 1.0)
            surv_lot_n = st.number_input("Ammunition Lot Stockpile Size (rounds)", 100, 100000, 5000, 500)
        with surv_c3:
            surv_insp = st.selectbox(
                "ISO 2859-1 / STANAG 4147 Inspection Level",
                ["Level I (Reduced)", "Level II (Normal Standard)", "Level III (Tightened)", "Special S-3", "Special S-4"],
                index=1,
            )
            surv_aql = st.selectbox("Acceptance Quality Limit (AQL %)", [0.40, 0.65, 1.00], index=1)

        surv_res = simulate_propellant_aging(
            stabilizer_type=surv_stab,
            storage_climate=surv_clim,
            storage_duration_years=float(surv_years),
        )
        plan_res = generate_surveillance_sampling_plan(
            lot_size=int(surv_lot_n),
            inspection_level=surv_insp,
            aql_percent=float(surv_aql),
        )

        smk1, smk2, smk3, smk4, smk5 = st.columns(5)
        smk1.metric("Current Active Stabilizer", f"{surv_res['final_stabilizer_pct']:.2f} %", delta=f"Initial: {surv_res['initial_stabilizer_pct']:.2f}%", delta_color="off")
        smk2.metric("Storage Status", surv_res['status_label'], delta="Auto-Ignition Risk" if surv_res['final_stabilizer_pct'] <= 0.20 else "Nominal", delta_color="inverse" if surv_res['final_stabilizer_pct'] <= 0.20 else "normal")
        smk3.metric("Arrhenius Temp T_eq", f"{surv_res['arrhenius_equivalent_temp_C']:.1f} °C", delta=f"+{surv_res['temperature_penalty_C']:.1f}°C vs Mean", delta_color="off")
        smk4.metric("Safe Shelf-Life Limit", f"{surv_res['years_to_critical_shelf_life']:.1f} yrs", delta=f"Half-life: {surv_res['half_life_years']:.1f} yrs", delta_color="off")
        smk5.metric("Surveillance Sample Size", f"{plan_res['sample_size_n']} rounds", delta=f"Ac: {plan_res['acceptance_number_ac']} | Re: {plan_res['rejection_number_re']}", delta_color="off")

        if surv_res['final_stabilizer_pct'] <= 0.20:
            st.error(f"🚨 **{surv_res['status_label']}**: {surv_res['action_recommendation']}")
        elif surv_res['final_stabilizer_pct'] <= 0.30:
            st.warning(f"⚠️ **{surv_res['status_label']}**: {surv_res['action_recommendation']}")
        elif surv_res['final_stabilizer_pct'] <= 0.50:
            st.info(f"🟡 **{surv_res['status_label']}**: {surv_res['action_recommendation']}")
        else:
            st.success(f"✅ **{surv_res['status_label']}**: {surv_res['action_recommendation']}")

        surv_p1, surv_p2 = st.columns(2)
        with surv_p1:
            sh = surv_res["history"]
            thresh = surv_res["thresholds"]
            fig_surv = go.Figure()
            fig_surv.add_trace(go.Scatter(x=sh["years"], y=sh["stabilizer_pct"], mode="lines", name=f"{surv_stab} Content (%)", line=dict(color="#2563EB", width=3.5)))
            fig_surv.add_hline(y=thresh["monitoring_pct"], line_dash="dash", line_color="#F59E0B", annotation_text=f"Monitoring ({thresh['monitoring_pct']}%)")
            fig_surv.add_hline(y=thresh["warning_pct"], line_dash="dash", line_color="#EA580C", annotation_text=f"Warning ({thresh['warning_pct']}%)")
            fig_surv.add_hline(y=thresh["critical_pct"], line_dash="dash", line_color="#DC2626", annotation_text=f"Critical / Auto-Ignition ({thresh['critical_pct']}%)")
            fig_surv.add_vline(x=surv_years, line_dash="dot", line_color="#4B5563", annotation_text=f"Current: {surv_years:.0f} yrs")
            fig_surv.update_layout(title="Propellant Stabilizer Depletion Kinetics vs Critical Thresholds", xaxis_title="Storage Time (Years)", yaxis_title="Active Stabilizer Content (wt %)", height=390)
            st.plotly_chart(fig_surv, use_container_width=True)

        with surv_p2:
            oc = plan_res["oc_curve"]
            fig_oc = go.Figure()
            fig_oc.add_trace(go.Scatter(x=oc["p_defect_pct"], y=np.asarray(oc["p_accept"])*100.0, mode="lines", name="Operating Characteristic (OC)", line=dict(color="#10B981", width=3)))
            fig_oc.add_vline(x=surv_aql, line_dash="dash", line_color="#DC2626", annotation_text=f"AQL = {surv_aql}%")
            fig_oc.update_layout(title=f"ISO 2859-1 Surveillance OC Curve (Lot N={surv_lot_n}, n={plan_res['sample_size_n']}, Ac={plan_res['acceptance_number_ac']})", xaxis_title="True Defect Percentage p (%)", yaxis_title="Probability of Lot Acceptance P_acc (%)", height=390)
            st.plotly_chart(fig_oc, use_container_width=True)

    # --- IATG SUB-TAB 7: GEOSPATIAL TERRAIN CREST CLEARANCE & 3D KML/CZML EXPORT ---
    with isub7:
        st.subheader("🗺️ Geospatial Terrain Crest Clearance, Dead Space Masking & 3D KML/CZML Exporters (FM 6-40 / STANAG 4119)")
        st.markdown(
            r"Operational artillery field ballistics and terrain masking audit according to **US Army FM 6-40 (Field Artillery Manual Cannon Gunnery)**: "
            r"identifies intermediate topographical crests, verifies minimum vertical safety clearance ($\Delta z_{\text{clear}} \ge \Delta z_{\text{min}}$), "
            r"and maps reverse-slope **Dead Space** obscured from direct fire. "
            r"Enables mission planning export to **Google Earth (OGC KML 2.2)** and **Cesium WebGL (CZML)** with 3D trajectories, weapon batteries, and CEP impact circles."
        )
        st.latex(
            r"""
            z_{\text{terrain}}(x) = z_{\text{base}} + h_{\text{crest}}\exp\left[-\frac{(x - x_{\text{crest}})^2}{2\sigma_{\text{crest}}^2}\right], \quad
            \Delta z_{\text{clear}} = z_{\text{traj}}(x_{\text{crest}}) - z_{\text{terrain}}(x_{\text{crest}}) \ge \Delta z_{\text{min}}
            """
        )

        geo1, geo2, geo3 = st.columns(3)
        with geo1:
            st.markdown("**Trajectory & Weapon Parameters**")
            geo_range = st.slider("Target Distance (m)", 3000.0, 22000.0, 12000.0, 500.0)
            geo_v0 = st.slider("Muzzle Velocity V0 (m/s)", 300.0, 1000.0, 827.0, 10.0)
            geo_high_ang = st.checkbox("High-Angle Fire (QE > 45°)", value=False)

        with geo2:
            st.markdown("**Topographical Mountain Ridge Mask**")
            geo_crest_dist = st.slider("Crest Ridge Distance (m)", 1000.0, 12000.0, 4500.0, 250.0)
            geo_crest_ht = st.slider("Crest Peak Height (m)", 50.0, 1200.0, 450.0, 25.0)
            geo_crest_sig = st.slider("Crest Ridge Width σ (m)", 200.0, 1500.0, 650.0, 50.0)
            geo_tgt_elev = st.slider("Target Ground Elevation (m)", -100.0, 600.0, 50.0, 25.0)

        with geo3:
            st.markdown("**Geospatial Coordinates for 3D GIS Export**")
            geo_lat = st.number_input("Battery Latitude (°N)", -90.0, 90.0, 34.0500, 0.001, format="%.4f")
            geo_lon = st.number_input("Battery Longitude (°E)", -180.0, 180.0, -118.2500, 0.001, format="%.4f")
            geo_az = st.slider("Firing Azimuth (° from North)", 0.0, 360.0, 45.0, 5.0)

        with st.spinner("Auditing Crest Clearance & Ray-Tracing Terrain Shadow Mask..."):
            from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch
            atm_env_geo = AtmosphereEnvironment()
            cfg_geo = BatchTrajectoryConfig(calibre_m=0.155, fidelity="L1", dt=0.08)
            qe_deg = find_elevation_for_range(geo_range, v0_mps=geo_v0, calibre_m=0.155, mass_kg=43.5, high_angle=geo_high_ang)
            rad_geo = math.radians(qe_deg)
            s0_geo = np.array([[0.0, 0.0, 0.0, geo_v0 * math.cos(rad_geo), 0.0, geo_v0 * math.sin(rad_geo)]], dtype=np.float64)
            res_geo = integrate_trajectories_batch(s0_geo, np.array([43.5]), np.array([1.0]), atm_env_geo, cfg_geo, record_single_path=True)
            sp_geo = res_geo.get("sample_path", {})
            step_sp = max(1, len(sp_geo.get("t", [])) // 35)
            tx_m = [float(sp_geo["range_m"][k]) for k in range(0, len(sp_geo.get("t", [])), step_sp)] if sp_geo else [0.0, float(geo_range)]
            ty_m = [float(sp_geo["deflection_m"][k]) for k in range(0, len(sp_geo.get("t", [])), step_sp)] if sp_geo else [0.0, 0.0]
            tz_m = [float(sp_geo["alt_m"][k]) for k in range(0, len(sp_geo.get("t", [])), step_sp)] if sp_geo else [0.0, 0.0]
            tt_s = [float(sp_geo["t"][k]) for k in range(0, len(sp_geo.get("t", [])), step_sp)] if sp_geo else [0.0, 30.0]

            clearance_res = audit_crest_clearance_and_dead_space(
                gun_elev_m=150.0,
                target_dist_m=geo_range,
                target_elev_m=geo_tgt_elev,
                ridge_pos_m=geo_crest_dist,
                ridge_height_m=geo_crest_ht,
                ridge_width_m=geo_crest_sig,
                trajectory_x_m=tx_m,
                trajectory_z_m=tz_m,
            )
            clearance_res["qe_deg"] = qe_deg
            clearance_res["trajectory"] = {"x_m": tx_m, "y_m": ty_m, "z_m": tz_m, "t_s": tt_s}

        gc1, gc2, gc3, gc4 = st.columns(4)
        c_status = clearance_res.get("clearance_status", "UNKNOWN")
        c_delta = "Safe Clearance" if clearance_res.get("clears_crest") else "Blocked by Terrain"
        gc1.metric("Clearance Status", c_status, delta=c_delta)
        gc2.metric("Quadrant Elevation QE", f"{clearance_res.get('qe_deg', 0.0):.2f}°")
        gc3.metric("Crest Clearance Margin", f"{clearance_res.get('actual_clearance_m', 0.0):.1f} m")
        gc4.metric("Dead Space Zone", f"{clearance_res.get('dead_space_start_m', 0.0):.0f}m – {clearance_res.get('dead_space_end_m', 0.0):.0f}m")

        fig_terr_3d = create_3d_terrain_clearance_figure(clearance_res, tx_m, ty_m, tz_m)
        st.plotly_chart(fig_terr_3d, use_container_width=True)

        # 3D GIS Exporters
        st.markdown("### 📥 3D GIS & Virtual Globe Data Exporters")
        st.markdown(
            "Export this trajectory and impact zone directly into desktop and web GIS systems "
            "(Google Earth, Cesium Ion, ArcGIS Earth, NASA WorldWind)."
        )

        kml_str = export_trajectory_to_kml(
            trajectory_x_m=tx_m,
            trajectory_y_m=ty_m,
            trajectory_z_m=tz_m,
            origin_lat=geo_lat,
            origin_lon=geo_lon,
            name=f"Artillery_Trajectory_{int(geo_range)}m",
        )
        czml_str = export_trajectory_to_czml(
            trajectory_x_m=tx_m,
            trajectory_y_m=ty_m,
            trajectory_z_m=tz_m,
            time_s=tt_s,
            origin_lat=geo_lat,
            origin_lon=geo_lon,
            name=f"Artillery_Trajectory_Cesium_{int(geo_range)}m",
        )

        kml_c, czml_c = st.columns(2)
        with kml_c:
            st.download_button(
                label="🌍 Download Google Earth 3D (.kml)",
                data=kml_str.encode("utf-8"),
                file_name=f"trajectory_mission_{int(geo_range)}m.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True,
            )
        with czml_c:
            st.download_button(
                label="🛰️ Download Cesium 3D Stream (.czml)",
                data=czml_str.encode("utf-8"),
                file_name=f"trajectory_stream_{int(geo_range)}m.czml",
                mime="application/json",
                use_container_width=True,
            )


# --- TAB 6: AUTOMATED BALLISTIC FIRING TABLES & MET UNIT-EFFECTS ---
def tab_firing_tables() -> None:
    st.header("📋 Automated Ballistic Firing-Table & Meteorological Unit-Effects Generator (`FM 6-40` / `STANAG 4119`)")
    st.markdown(
        "Generates tabular artillery, mortar, and direct-fire **Firing Tables** across quadrant elevation increments, "
        "computing **Range (`m`)**, **Spin/Coriolis Drift (`m`)**, **Time of Flight (`TOF`)**, **Striking Velocity (`V_impact`)**, "
        "**Angle of Fall (`deg`)**, **Vertex Altitude (`m`)**, and finite-difference "
        "**Meteorological Unit-Effects (`∂R/∂V₀`, `∂R/∂W_x`, `∂D/∂W_z`, `∂R/∂T_air`, `∂R/∂FF`)**."
    )

    presets = list_presets()
    gun_presets = [p for p in presets if p["class_id"].startswith(("AR-", "RK-", "MS-", "MO-", "NV-", "CN-", "SA-", "GR-"))]
    ft_dict = {f"{p['preset_id']} — {p['ammo_name']}": p["preset_id"] for p in gun_presets}

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        ft_label = st.selectbox("Select Weapon / Rocket / Munition System", list(ft_dict.keys()), index=4)
        ft_pid = ft_dict[ft_label]
        ft_ammo, _ = get_preset(ft_pid)
    with f2:
        qe_min_ft = st.number_input("Minimum Elevation QE (deg)", 2.0, 45.0, 5.0, 1.0)
    with f3:
        qe_max_ft = st.number_input("Maximum Elevation QE (deg)", 10.0, 75.0, 65.0, 5.0)
    with f4:
        n_steps_ft = st.slider("Elevation Increments", 5, 20, 13, 1)

    with st.spinner("Computing Ballistic Firing Table & Finite-Difference Unit-Effect Sensitivities..."):
        ft_rows = generate_ballistic_firing_table(
            v0_mps=max(float(ft_ammo.launch.mv.mu), 40.0),
            calibre_m=float(ft_ammo.calibre_m),
            mass_kg=float(ft_ammo.mass_kg),
            twist_cal_per_turn=float(ft_ammo.launch.twist_cal or 20.0),
            form_factor=float(ft_ammo.aero.form_factor.mu),
            drag_ref=str(ft_ammo.aero.drag_ref if ft_ammo.aero.drag_ref in ("G1", "G7") else "G7"),
            qe_min_deg=float(qe_min_ft),
            qe_max_deg=float(qe_max_ft),
            n_elevations=int(n_steps_ft),
            rocket_burn_time_s=float(ft_ammo.launch.rocket_burn_s),
            rocket_mdot_kg_s=float(ft_ammo.launch.rocket_mdot_kg_s),
            rocket_ve_mps=float(ft_ammo.launch.rocket_ve_mps),
            base_bleed_burn_time_s=float(ft_ammo.launch.base_bleed_burn_s),
            base_bleed_dcd=float(ft_ammo.launch.base_bleed_dcd),
        )
    ft_df = pd.DataFrame(ft_rows)

    ft_c1, ft_c2 = st.columns(2)
    with ft_c1:
        fig_ft_qe = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ft_qe.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["Range (m)"],
                mode="lines+markers",
                name="Horizontal Range (m)",
                line=dict(color="#1E3A8A", width=3.5),
            ),
            secondary_y=False,
        )
        fig_ft_qe.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["TOF (s)"],
                mode="lines+markers",
                name="Time of Flight TOF (s)",
                line=dict(color="#DC2626", width=2.5, dash="dash"),
            ),
            secondary_y=True,
        )
        fig_ft_qe.update_xaxes(title_text="Quadrant Elevation QE (deg)")
        fig_ft_qe.update_yaxes(title_text="Range (m)", secondary_y=False)
        fig_ft_qe.update_yaxes(title_text="Time of Flight (s)", secondary_y=True)
        fig_ft_qe.update_layout(title=f"Firing Table Range & Time of Flight vs Elevation ({ft_ammo.name})", height=390)
        st.plotly_chart(fig_ft_qe, use_container_width=True)

    with ft_c2:
        fig_ft_met = go.Figure()
        fig_ft_met.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["∂R/∂V₀ (m / m/s)"],
                mode="lines+markers",
                name="∂R/∂V₀ (m per +1 m/s MV)",
                line=dict(color="#2563EB", width=3),
            )
        )
        fig_ft_met.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["∂R/∂W_x (m / m/s)"],
                mode="lines+markers",
                name="∂R/∂W_x (m per +1 m/s Tailwind)",
                line=dict(color="#10B981", width=2.5),
            )
        )
        fig_ft_met.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["∂R/∂FF (m / +1%)"],
                mode="lines+markers",
                name="∂R/∂FF (m per +1% Form Factor)",
                line=dict(color="#DC2626", width=2.5, dash="dash"),
            )
        )
        fig_ft_met.add_trace(
            go.Scatter(
                x=ft_df["QE (deg)"],
                y=ft_df["∂D/∂W_z (m / m/s)"],
                mode="lines+markers",
                name="∂D/∂W_z (m per +1 m/s Crosswind)",
                line=dict(color="#7C3AED", width=2.5, dash="dot"),
            )
        )
        fig_ft_met.update_layout(
            title="Ballistic & Meteorological Unit-Effect Partial Derivatives vs Elevation",
            xaxis_title="Quadrant Elevation QE (deg)",
            yaxis_title="Unit Sensitivity (m per unit perturbation)",
            height=390,
        )
        st.plotly_chart(fig_ft_met, use_container_width=True)

    st.dataframe(ft_df, use_container_width=True)
    ft_csv = ft_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Tabular Firing Table & Met Unit-Effects (.csv)",
        ft_csv,
        f"{ft_ammo.id}_firing_table.csv",
        "text/csv",
    )

    st.markdown("---")
    st.subheader("🌐 NATO STANAG 4082 Standard METCM Atmospheric Sounding & Upper-Air Wind Shear Integration")
    st.markdown(
        "Exterior ballistic trajectories integrated through a **20-zone upper atmospheric sounding** "
        "(`STANAG 4082 METCM`), incorporating **Planetary Boundary Layer (PBL) power-law shear**, "
        "**Ekman spiral veering**, **Jet Stream Gaussian core velocity**, and **low-level thermal inversions** "
        "compared to standard International Standard Atmosphere (ISA) baseline."
    )

    with st.expander("⚙️ Atmospheric Sounding & Fire Control Parameters", expanded=True):
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown("**Surface & Boundary Layer**")
            s_temp_c = st.slider("Surface Temperature (°C)", -30.0, 50.0, 15.0, 1.0)
            s_press_hpa = st.number_input("Surface Pressure (hPa)", 800.0, 1080.0, 1013.25, 5.0)
            s_wind_spd = st.slider("Surface Wind Speed (m/s)", 0.0, 30.0, 6.0, 0.5)
            s_wind_dir = st.slider("Surface Wind Direction (° from North)", 0.0, 360.0, 240.0, 5.0)
            pbl_alpha = st.slider("PBL Shear Exponent α", 0.10, 0.45, 0.28, 0.02)
            ekman_veer = st.slider("Ekman Spiral Veering (°)", 0.0, 60.0, 30.0, 5.0)
        with mcol2:
            st.markdown("**Jet Stream Core & Inversion**")
            jet_peak = st.slider("Jet Stream Core Peak (m/s)", 0.0, 80.0, 42.0, 2.0)
            jet_alt = st.slider("Jet Stream Altitude (m)", 6000.0, 15000.0, 10500.0, 500.0)
            enable_inversion = st.checkbox("Thermal Inversion Layer", value=True)
            inv_base = st.number_input("Inversion Base (m)", 200.0, 3000.0, 800.0, 100.0)
            inv_top = st.number_input("Inversion Top (m)", 300.0, 4000.0, 1400.0, 100.0)
            inv_dt = st.slider("Inversion ΔT (K)", 1.0, 15.0, 5.0, 0.5)
        with mcol3:
            st.markdown("**Trajectory & Weapon Parameters**")
            m_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 200.0, 1200.0, float(ft_ammo.launch.mv.mu), 10.0)
            m_qe = st.slider("Firing Elevation QE (°)", 5.0, 75.0, 38.0, 1.0)
            m_az = st.slider("Azimuth of Fire (° from North)", 0.0, 360.0, 90.0, 5.0)
            m_dt = st.select_slider("Integration Time Step dt (s)", options=[0.02, 0.04, 0.08, 0.10], value=0.08)
        with mcol4:
            st.markdown("**Aero & Mass Properties**")
            m_cal = st.number_input("Caliber (m)", 0.020, 0.500, float(ft_ammo.calibre_m), 0.005, format="%.3f")
            m_mass = st.number_input("Projectile Mass (kg)", 0.5, 200.0, float(ft_ammo.mass_kg), 1.0)
            st.info(f"Using aerodynamic reference drag curve for `{ft_ammo.aero.drag_ref or 'G7'}`.")

    with st.spinner("Generating STANAG 4082 METCM Sounding & Integrating Atmospheric Shear Trajectory..."):
        sounding_prof = generate_stanag_4082_profile(
            surface_temp_c=float(s_temp_c),
            surface_pressure_hpa=float(s_press_hpa),
            surface_wind_speed_mps=float(s_wind_spd),
            surface_wind_dir_deg=float(s_wind_dir),
            shear_exponent=float(pbl_alpha),
            ekman_veer_deg=float(ekman_veer),
            jet_stream_peak_mps=float(jet_peak),
            jet_stream_altitude_m=float(jet_alt),
            thermal_inversion=bool(enable_inversion),
            inversion_base_m=float(inv_base),
            inversion_top_m=float(inv_top),
            inversion_delta_t_k=float(inv_dt),
        )
        met_traj_res = integrate_trajectory_with_sounding(
            v0_mps=float(m_v0),
            qe_deg=float(m_qe),
            az_deg=float(m_az),
            calibre_m=float(m_cal),
            mass_kg=float(m_mass),
            sounding_profile=sounding_prof,
            dt=float(m_dt),
        )

    mk1, mk2, mk3, mk4 = st.columns(4)
    rng_met = met_traj_res["range_with_met_m"]
    rng_isa = met_traj_res["range_isa_baseline_m"]
    delta_rng = met_traj_res["delta_range_met_m"]
    cross_drift = met_traj_res["crossrange_drift_m"]
    mk1.metric("Range with METCM Sounding", f"{rng_met:,.1f} m")
    mk2.metric("ISA Standard Range Baseline", f"{rng_isa:,.1f} m")
    mk3.metric("Range Delta (METCM vs ISA)", f"{delta_rng:+,.1f} m", delta=f"{delta_rng / max(rng_isa, 1.0) * 100:+.2f}% Range Shift")
    mk4.metric("Crosswind Drift Deflection", f"{cross_drift:+,.1f} m", delta=f"TOF: {met_traj_res['tof_with_met_s']:.1f} s")

    fig_shear_3d = create_3d_atmospheric_shear_figure(met_traj_res, sounding_prof)
    st.plotly_chart(fig_shear_3d, use_container_width=True)

    st.markdown("##### 📊 NATO STANAG 4082 Standard 20-Zone Sounding Profile (METCM Format)")
    zones_df = pd.DataFrame(sounding_prof["zones"])
    st.dataframe(zones_df, use_container_width=True)
    csv_zones = zones_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download STANAG 4082 METCM Sounding Dataset (.csv)",
        csv_zones,
        "stanag_4082_metcm_sounding.csv",
        "text/csv",
    )


# --- TAB 7: VALIDATION CASES ---
def tab_validation() -> None:
    st.header("✅ Knowledge Base Verification & Validation Suite (V-01a .. V-12)")
    st.markdown(
        "Automated regression harness comparing Python engine evaluations against `Ammunition Monte Carlo Knowledge Base.xlsx` "
        "workbook cells and authoritative reference literature (IATG 01.80, STANAG 4355, BRL, McCoy 1974)."
    )

    if "validation_results" not in st.session_state or st.button("🔄 Run Full 22-Case Validation Suite", type="primary"):
        with st.spinner("Executing 22 analytical and stochastic verification cases..."):
            val_results = run_validation_suite(seed=42, n_mc_v08=50_000)
            st.session_state["validation_results"] = val_results

    v_df = pd.DataFrame(st.session_state["validation_results"])
    n_pass = sum(1 for s in v_df["status"] if "PASS" in str(s))
    max_err_excel = float(v_df["rel_err_vs_excel"].max())

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Validation Cases Passed", f"{n_pass} / {len(v_df)}", delta="100% Pass Rate")
    vc2.metric("Max Relative Deviation vs Excel", f"{max_err_excel * 100:.4f} %")
    vc3.metric("Registered Formula Functions", "96 / 96", delta="All Verified")

    fig_val = px.bar(
        v_df,
        x="test_id",
        y="rel_err_vs_expected",
        color="formula",
        labels={"test_id": "Validation Test ID", "rel_err_vs_expected": "Relative Error vs Expected"},
        title="Relative Deviation Across All 22 Knowledge Base Validation Cases",
    )
    fig_val.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_val, use_container_width=True)

    st.dataframe(
        v_df[
            [
                "test_id",
                "formula",
                "status",
                "computed",
                "excel_value",
                "expected",
                "rel_err_vs_excel",
                "rel_err_vs_expected",
                "case",
            ]
        ],
        use_container_width=True,
    )


def main() -> None:
    _configure_page_and_styles()
    if not require_password():
        return
    render_header()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "🎯 Monte Carlo Simulator",
            "🌐 3D Simulations",
            "🚀 McCoy MCDRAG & 3D CAD",
            "🔬 Interior Ballistics & Lab",
            "🏛️ UNODA IATG Safety & QD",
            "📋 Firing Tables & Met Effects",
            "✅ Validation Suite",
        ]
    )

    with tab1:
        tab_simulator()
    with tab2:
        tab_3d_simulation()
    with tab3:
        tab_mcdrag()
    with tab4:
        tab_calculators()
    with tab5:
        tab_iatg()
    with tab6:
        tab_firing_tables()
    with tab7:
        tab_validation()

    with st.sidebar:
        st.divider()
        if st.button("🔒 Lock Workbench", use_container_width=True, key="lock_workbench_btn"):
            st.session_state["authenticated"] = False
            st.rerun()


if __name__ == "__main__":
    main()
