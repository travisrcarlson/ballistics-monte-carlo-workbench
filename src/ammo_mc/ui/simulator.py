"""Monte Carlo Simulator UI workbench."""


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
from ammo_mc.cuas_mission import (
    generate_3d_intercept_replay_data,
    generate_cuas_trade_study_html,
    get_cuas_ammo_economics_catalog,
    get_kill_chain_sensor_catalog,
    get_magazine_system_specs,
    get_site_defense_presets,
    serialize_cuas_scenario_config,
    simulate_cuas_kill_chain_mc,
    simulate_cuas_sustained_logistics_mc,
    simulate_multiturret_site_defense_mc,
    validate_cuas_scenario_config,
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
from ammo_mc.vis_cuas_mission import (
    create_3d_animated_intercept_figure,
    create_kill_chain_timeline_figure,
    create_logistics_thermal_and_cost_figure,
    create_multiturret_site_map_figure,
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
from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.scenario_library import ScenarioLibrary
from ammo_mc.vis_cuas_analytics import create_sensitivity_tornado_figure


from ammo_mc.ui.common import (
    _CUAS18_MODULES,
    _CUAS_SUITES,
    _IATG_TOOLS,
    _PHYSICS_LABS,
    _SMALL_ARMS_MODES,
    _WORKBENCH_OPTIONS,
    _cached_benchmark_drone_aspect,
    _cached_benchmark_multibarrel_aspect,
    _cached_compare_anti_drone_layouts,
    _cached_compare_cuas_fallout,
    _cached_compare_cuas_payloads,
    _cached_compare_ab,
    _cached_cuas_tornado,
    _cached_artillery_tornado,
    _cached_intercept_replay,
    _cached_kill_chain,
    _cached_layered_defense,
    _cached_optimize_antidrone_array,
    _cached_site_defense,
    _cached_sustained_logistics,
    _render_sidebar_header,
    _set_active_workbench,
    _set_cuas_module,
)



def _jump_to_calibration_lab() -> None:
    st.session_state["active_workbench"] = _WORKBENCH_OPTIONS[1]
    st.session_state["cuas_wb_suite"] = _CUAS_SUITES[0]
    st.session_state["cuas18_active_module"] = _CUAS18_MODULES[12]


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
        _render_sidebar_header("🎯 Monte Carlo Simulator", "Stochastic 6-stage exterior, fuze & terminal pipeline")
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
            has_cal = any(
                k in st.session_state
                for k in ("_calibrated_mv_mps", "_calibrated_mv_sd_mps", "_calibrated_form_factor")
            )
            use_cal_overrides = False
            if has_cal:
                cal_parts = []
                if "_calibrated_mv_mps" in st.session_state:
                    cal_parts.append(f"MV={st.session_state['_calibrated_mv_mps']:.1f}m/s")
                if "_calibrated_mv_sd_mps" in st.session_state:
                    cal_parts.append(f"SD={st.session_state['_calibrated_mv_sd_mps']:.2f}")
                if "_calibrated_form_factor" in st.session_state:
                    cal_parts.append(f"i={st.session_state['_calibrated_form_factor']:.3f}")
                use_cal_overrides = st.checkbox(
                    f"Apply Field-Calibrated Values ({', '.join(cal_parts)})",
                    value=True,
                    key="sim_use_cal_overrides",
                )
            st.button(
                "🧪 Open Field Calibration Lab",
                on_click=_jump_to_calibration_lab,
                width="stretch",
                key="sim_jump_cal_btn",
            )
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

        run_btn = st.button("🚀 Run Monte Carlo Simulation", type="primary", width="stretch")

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
    if use_cal_overrides:
        if "_calibrated_mv_mps" in st.session_state:
            ammo.launch.mv.mu = float(st.session_state["_calibrated_mv_mps"])
        if "_calibrated_mv_sd_mps" in st.session_state:
            ammo.launch.mv.sigma_rr = float(st.session_state["_calibrated_mv_sd_mps"])
        if "_calibrated_dv_dt" in st.session_state:
            ammo.launch.temp_coeff_mps_per_C = float(st.session_state["_calibrated_dv_dt"])
        if "_calibrated_form_factor" in st.session_state:
            ammo.aero.form_factor.mu = float(st.session_state["_calibrated_form_factor"])
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
            st.plotly_chart(fig_poi, width="stretch")

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
            st.plotly_chart(fig_traj, width="stretch")
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
            st.plotly_chart(fig_hz, width="stretch")
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
        st.subheader("📊 Error Budget Sensitivity & Bottleneck Ranking")
        sens_view_mode = st.radio(
            "Sensitivity View Mode",
            ["📈 Spearman / Sobol Ranking (MC-10)", "📊 ±% Parameter Tornado Bottleneck"],
            horizontal=True,
            label_visibility="collapsed",
            key="sim_sens_view_mode",
        )
        if sens_view_mode == "📈 Spearman / Sobol Ranking (MC-10)":
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
                st.plotly_chart(fig_sens, width="stretch")
            else:
                st.info("Sensitivity analysis requires variation across stochastic parameters.")
        else:
            tor_pct = st.slider("Parameter Perturbation Sweep (±%)", 5.0, 40.0, 20.0, 5.0, key="sim_tornado_pct")
            with st.spinner("Running ±% CEP50 Bottleneck Tornado Sweep..."):
                tor_res = _cached_artillery_tornado(
                    cur_ammo.model_dump_json(),
                    cur_scn.model_dump_json(),
                    perturbation_pct=float(tor_pct),
                    solved_qe_rad=float(res.solved_qe_rad) if res.solved_qe_rad > 0 else None,
                    seed=int(cur_scn.seed or 42),
                )
            tsm = tor_res["summary"]
            st.session_state["sim_tornado_last"] = tor_res
            st.info(
                f"**#1 CEP50 Bottleneck:** `{tsm['top_driver']}` (swing **{tsm['top_swing']:.2f} m**) — "
                f"{tor_res['top_bottlenecks'][0]['recommendation']}"
            )
            fig_tor = create_sensitivity_tornado_figure(tor_res, metric_label="CEP 50% (m)", higher_is_better=False)
            fig_tor.update_layout(height=390, margin=dict(l=180, r=20, t=48, b=35))
            st.plotly_chart(fig_tor, width="stretch")
            with st.expander("📋 Artillery Tornado Sensitivity Table & CSV", expanded=False):
                tor_df = pd.DataFrame(tor_res["rows"])
                st.dataframe(tor_df, width="stretch")
                st.download_button(
                    "📥 Download Artillery Tornado Sensitivity (.csv)",
                    tor_df.to_csv(index=False).encode("utf-8"),
                    f"{cur_ammo.id}_cep_tornado.csv",
                    "text/csv",
                    key="sim_dl_tornado_csv",
                )

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
                st.plotly_chart(fig_fmea, width="stretch")

            with fmea_col2:
                st.dataframe(fmea_df, width="stretch", height=320)

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
            st.plotly_chart(fig_conv, width="stretch")

    with conv_col2:
        st.subheader("🧾 Provenance & Active Formula Audit Trail")
        forms_used = res.provenance.get("formulas_used", [])
        if forms_used:
            f_df = pd.DataFrame(forms_used)
            st.dataframe(
                f_df[["formula_id", "module", "name", "verification_status", "source_id"]],
                width="stretch",
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
                st.plotly_chart(fig_en, width="stretch")

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
                st.plotly_chart(fig_re, width="stretch")

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
            width="stretch",
        )
    with exp_c2:
        json_str = summary_to_json_str(res)
        st.download_button(
            "📦 Download JSON Summary",
            json_str.encode("utf-8"),
            f"{cur_ammo.id}_summary.json",
            "application/json",
            width="stretch",
        )
    with exp_c3:
        geojson_str = geojson_to_str(res, scenario=cur_scn)
        st.download_button(
            "🗺️ Download GeoJSON",
            geojson_str.encode("utf-8"),
            f"{cur_ammo.id}_contours.geojson",
            "application/geo+json",
            width="stretch",
        )
    with exp_c4:
        png_bytes = render_report_figure_bytes(res, scenario=cur_scn, fmt="png")
        st.download_button(
            "🖼️ Download PNG Report",
            png_bytes,
            f"{cur_ammo.id}_report.png",
            "image/png",
            width="stretch",
        )
    with exp_c5:
        pdf_bytes = render_report_figure_bytes(res, scenario=cur_scn, fmt="pdf")
        st.download_button(
            "📑 Download PDF Report",
            pdf_bytes,
            f"{cur_ammo.id}_report.pdf",
            "application/pdf",
            width="stretch",
        )

    sim_summary_payload = {"dispersion": res.dispersion, "effectiveness": res.effectiveness}
    src_ids_used = [s["source_id"] for s in res.provenance.get("sources", []) if "source_id" in s]
    with st.expander("💾 Scenario Library & Executive Briefing Deck (.pdf / .pptx)", expanded=False):
        eb1, eb2 = st.columns(2)
        with eb1:
            st.download_button(
                "📑 Executive Multi-Page Briefing (.pdf)",
                generate_executive_briefing_pdf(
                    title=f"{cur_ammo.name} — Monte Carlo Engineering Briefing",
                    subtitle=f"{cur_scn.name} [{cur_scn.fidelity}, {cur_scn.sampler}]",
                    simulator_summary=sim_summary_payload,
                    tornado_result=st.session_state.get("sim_tornado_last"),
                    source_ids=src_ids_used or None,
                ),
                f"{cur_ammo.id}_executive_briefing.pdf",
                "application/pdf",
                width="stretch",
                key="sim_dl_exec_pdf",
            )
        with eb2:
            st.download_button(
                "📊 Executive 16:9 Slide Deck (.pptx)",
                generate_executive_briefing_pptx(
                    title=f"{cur_ammo.name} — Monte Carlo Engineering Briefing",
                    subtitle=f"{cur_scn.name} [{cur_scn.fidelity}, {cur_scn.sampler}]",
                    simulator_summary=sim_summary_payload,
                    tornado_result=st.session_state.get("sim_tornado_last"),
                    source_ids=src_ids_used or None,
                ),
                f"{cur_ammo.id}_executive_briefing.pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                width="stretch",
                key="sim_dl_exec_pptx",
            )

        sim_lib = ScenarioLibrary()
        sl1, sl2, sl3 = st.columns([1.4, 1.0, 0.8])
        with sl1:
            sim_sv_name = st.text_input("Snapshot Name", value=f"{cur_ammo.id} @ {cur_scn.aim.range_m:.0f}m", key="sim_lib_name")
        with sl2:
            sim_sv_tags = st.text_input("Tags", value=f"{cur_ammo.class_id},{cur_scn.fidelity}", key="sim_lib_tags")
        with sl3:
            st.write("")
            if st.button("💾 Save Run", type="primary", width="stretch", key="sim_lib_save"):
                rec = sim_lib.save_scenario(
                    name=sim_sv_name,
                    domain="ballistics",
                    config={
                        "preset_id": cur_ammo.id,
                        "fidelity": cur_scn.fidelity,
                        "range_m": cur_scn.aim.range_m,
                        "mv_mu": cur_ammo.launch.mv.mu,
                        "form_factor": cur_ammo.aero.form_factor.mu,
                    },
                    metrics={
                        "mean_range_m": round(float(res.dispersion.get("mean_range_m", 0.0)), 2),
                        "cep_m": round(float(res.dispersion.get("cep_m", 0.0)), 2),
                        "salvo_pk_mean": round(float(res.effectiveness.get("salvo_pk_mean", 0.0)), 4),
                    },
                    tags=[t.strip() for t in sim_sv_tags.split(",") if t.strip()],
                )
                st.success(f"Saved `{rec['name']}` ({rec['id']}).")

        sim_saved = sim_lib.list_scenarios(domain="ballistics")
        if sim_saved:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": r["id"],
                            "Name": r["name"],
                            "Updated": r["updated_at"],
                            "CEP50 (m)": r["metrics"].get("cep_m", "—"),
                            "Salvo Pk": r["metrics"].get("salvo_pk_mean", "—"),
                        }
                        for r in sim_saved
                    ]
                ),
                width="stretch",
            )
            if len(sim_saved) >= 2:
                id_map_s = {f"{r['name']} ({r['id']})": r["id"] for r in sim_saved}
                sd1, sd2 = st.columns(2)
                with sd1:
                    sa_pick = st.selectbox("Compare Baseline A", list(id_map_s.keys()), index=1, key="sim_lib_da")
                with sd2:
                    sb_pick = st.selectbox("Compare Candidate B", list(id_map_s.keys()), index=0, key="sim_lib_db")
                if id_map_s[sa_pick] != id_map_s[sb_pick]:
                    d_sim = sim_lib.diff_scenarios(id_map_s[sa_pick], id_map_s[sb_pick])
                    if d_sim["metric_diffs"]:
                        st.dataframe(pd.DataFrame(d_sim["metric_diffs"]), width="stretch")
