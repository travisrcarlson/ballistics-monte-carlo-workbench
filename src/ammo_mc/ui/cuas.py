"""Counter-UAS & Multi-Barrel Small Arms UI workbench."""


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
    create_terrain_dem_site_3d_figure,
)
from ammo_mc.terrain_dem import (
    TerrainDEMGrid,
    build_preset_terrain_dem,
    fetch_open_meteo_elevation_dem,
    get_terrain_dem_presets,
    parse_dem_csv,
    parse_esri_ascii_grid,
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
from ammo_mc.cuas_analytics import (
    build_cuas_site_geojson,
    export_cuas_site_kml,
    get_cuas_comparator_presets,
    get_outer_layer_catalog,
    serialize_site_layout_json,
    validate_site_layout_json,
)
from ammo_mc.vis_cuas_analytics import (
    create_ab_comparator_radar_and_bars_figure,
    create_layered_defense_figure,
    create_real_map_site_defense_figure,
    create_sensitivity_tornado_figure,
)
from ammo_mc.vis_cuas_raid import (
    create_raid_timeline_and_heat_figure,
    create_unified_tactical_raid_3d_figure,
)
from ammo_mc.calibration import (
    calibrate_chronograph_from_csv,
    calibrate_drag_form_factor_from_csv,
    calibrate_shotgun_pattern_from_csv,
    generate_sample_chronograph_csv,
    generate_sample_pattern_csv,
    generate_sample_range_drop_csv,
)
from ammo_mc.presets import list_presets
from ammo_mc.vis_calibration import (
    create_chronograph_calibration_figure,
    create_drag_calibration_figure,
    create_pattern_calibration_figure,
)
from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.scenario_library import ScenarioLibrary


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
    _cached_unified_raid,
    _render_sidebar_header,
    _set_active_workbench,
    _set_cuas_module,
)

_SITE_GEO_ANCHORS: dict[str, tuple[float, float]] = {
    "Desert Test Range (32.9000°N, -106.4000°W)": (32.9000, -106.4000),
    "Coastal Airfield (36.8960°N, -76.2990°W)": (36.8960, -76.2990),
    "European Urban Site (48.8566°N, 2.3522°E)": (48.8566, 2.3522),
}


def _on_site_anchor_change() -> None:
    anchor = st.session_state.get("cuas18_geo_anchor")
    if anchor in _SITE_GEO_ANCHORS:
        lat, lon = _SITE_GEO_ANCHORS[anchor]
        st.session_state["cuas18_map_lat"] = float(lat)
        st.session_state["cuas18_map_lon"] = float(lon)


def _sanitize_site_editor_tables(
    tur_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    nfz_df: pd.DataFrame,
    fallback_site: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sanitize st.data_editor rows for turrets, obstacles, and NFZs."""
    clean_t: list[dict[str, Any]] = []
    if tur_df is not None and not tur_df.empty:
        for idx, row in tur_df.iterrows():
            try:
                x = float(row["x_m"])
                y = float(row["y_m"])
                if math.isnan(x) or math.isnan(y):
                    continue
                z = float(row.get("z_m", 3.5))
                az = float(row.get("boresight_az_deg", 0.0))
                half = float(row.get("sector_half_deg", 140.0))
                nb = float(row.get("n_barrels", 6))
                clean_t.append(
                    {
                        "id": str(row.get("id") or f"T{len(clean_t) + 1}")[:24],
                        "x_m": float(np.clip(x, -500.0, 500.0)),
                        "y_m": float(np.clip(y, -500.0, 500.0)),
                        "z_m": float(np.clip(3.5 if math.isnan(z) else z, 0.5, 100.0)),
                        "boresight_az_deg": float((0.0 if math.isnan(az) else az) % 360.0),
                        "sector_half_deg": float(np.clip(140.0 if math.isnan(half) else half, 15.0, 180.0)),
                        "n_barrels": int(np.clip(int(6 if math.isnan(nb) else round(nb)), 1, 16)),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue
    if not clean_t:
        clean_t = [dict(t) for t in fallback_site["turrets"]]

    clean_o: list[dict[str, Any]] = []
    if obs_df is not None and not obs_df.empty:
        for _, row in obs_df.iterrows():
            try:
                x = float(row["x_m"])
                y = float(row["y_m"])
                w = float(row.get("width_m", 10.0))
                l_val = float(row.get("length_m", 10.0))
                h = float(row.get("height_m", 12.0))
                if any(math.isnan(v) for v in (x, y, w, l_val, h)):
                    continue
                clean_o.append(
                    {
                        "name": str(row.get("name") or f"Obstacle {len(clean_o) + 1}")[:40],
                        "x_m": float(np.clip(x, -500.0, 500.0)),
                        "y_m": float(np.clip(y, -500.0, 500.0)),
                        "width_m": float(np.clip(w, 1.0, 200.0)),
                        "length_m": float(np.clip(l_val, 1.0, 200.0)),
                        "height_m": float(np.clip(h, 1.0, 120.0)),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue

    clean_n: list[dict[str, Any]] = []
    if nfz_df is not None and not nfz_df.empty:
        for _, row in nfz_df.iterrows():
            try:
                x = float(row["x_m"])
                y = float(row["y_m"])
                r = float(row.get("radius_m", 25.0))
                if any(math.isnan(v) for v in (x, y, r)):
                    continue
                clean_n.append(
                    {
                        "name": str(row.get("name") or f"NFZ {len(clean_n) + 1}")[:40],
                        "x_m": float(np.clip(x, -1000.0, 1000.0)),
                        "y_m": float(np.clip(y, -1000.0, 1000.0)),
                        "radius_m": float(np.clip(r, 5.0, 300.0)),
                    }
                )
            except (TypeError, ValueError, KeyError):
                continue

    return clean_t, clean_o, clean_n



def tab_cuas_and_small_arms() -> None:
    st.header("🛸 Counter-UAS Turret Engineering & Multi-Barrel Small Arms Workbench")
    st.markdown(
        "Dedicated engineering environment for **Multi-Shotgun Counter-UAS Turrets** (Closed-Loop FCS, Swarm Saturation, "
        "Airburst/TSS Payloads, Gimbal Recoil Optimization, Fallout SDZ, Multi-Sensor Kill Chain, Custom Site/GIS Map & NFZ, "
        "3D Replay, Logistics/Reports, A/B Comparator & Layered Defense, and Sensitivity Tornado) and **Small Arms & Multi-Barrel Dispersion**."
    )

    with st.sidebar:
        _render_sidebar_header("🛸 C-UAS & Small Arms", "Counter-UAS turret engineering & multi-barrel ballistics")
        cuas_suite = st.radio(
            "Engineering Suite",
            _CUAS_SUITES,
            key="cuas_wb_suite",
        )
        st.divider()
        active_cuas_mod = _CUAS18_MODULES[0]
        arm_mode = _SMALL_ARMS_MODES[4]
        if cuas_suite == _CUAS_SUITES[0]:
            active_cuas_mod = st.radio(
                "Active C-UAS Module",
                _CUAS18_MODULES,
                key="cuas18_active_module",
            )
            with st.expander("📄 Trade-Study Report Readiness", expanded=False):
                _rc_items = [
                    ("Array Summary (Mode 5)", st.session_state.get("_report_array_summary") is not None),
                    ("3D Aspect Table (Mode 5)", st.session_state.get("_report_aspect_table") is not None),
                    ("Kill Chain (Module 6)", st.session_state.get("_report_kc_summary") is not None),
                    ("Fallout SDZ (Module 5)", st.session_state.get("_report_sdz_summary") is not None),
                ]
                for _lbl, _ready in _rc_items:
                    st.markdown(f"{'✅' if _ready else '○'} {_lbl}")
                if active_cuas_mod != _CUAS18_MODULES[8]:
                    st.button(
                        "📄 Jump to Module 9: Report Export",
                        on_click=_set_cuas_module,
                        args=(_CUAS18_MODULES[8],),
                        width="stretch",
                        key="cuas18_jump_report_btn",
                    )
        else:
            arm_mode = st.radio(
                "Small Arms Analysis Mode",
                _SMALL_ARMS_MODES,
                key="small_arms_mode",
            )
            if arm_mode == "🛸 Anti-Drone Multi-Shotgun Defense Array (C-UAS)":
                with st.expander("💾 Scenario Preset Save / Load (JSON)", expanded=True):
                    uploaded_cfg = st.file_uploader(
                        "Upload Saved C-UAS Scenario Preset (.json)",
                        type=["json"],
                        key="cuas_cfg_upload",
                    )
                    if uploaded_cfg is not None:
                        raw_bytes = uploaded_cfg.getvalue()
                        sig = f"{uploaded_cfg.name}:{len(raw_bytes)}:{hash(raw_bytes)}"
                        if st.session_state.get("_cuas_last_uploaded_sig") != sig:
                            ok_cfg, cleaned_cfg, msg_cfg = validate_cuas_scenario_config(
                                raw_bytes.decode("utf-8", errors="replace")
                            )
                            if ok_cfg:
                                _key_map = {
                                    "layout_type": "cuas_layout",
                                    "n_barrels": "cuas_n_barrels",
                                    "mount_radius_m": "cuas_mount_r",
                                    "spacing_y_m": "cuas_sp_y",
                                    "spacing_z_m": "cuas_sp_z",
                                    "barrel_length_in": "cuas_b_len",
                                    "choke": "cuas_choke",
                                    "pointing_mode": "cuas_pointing",
                                    "splay_angle_mrad": "cuas_splay",
                                    "firing_mode": "cuas_fire_mode",
                                    "ripple_delay_ms": "cuas_ripple_dt",
                                    "ammunition_preset": "cuas_ammo",
                                    "drone_model": "cuas_drone_model",
                                    "aspect_mode": "cuas_aspect_mode",
                                    "drone_pitch_deg": "cuas_pitch_tilt",
                                    "drone_yaw_deg": "cuas_yaw_angle",
                                    "drone_roll_deg": "cuas_roll_angle",
                                    "target_distance_m": "cuas_dist",
                                    "drone_speed_mps": "cuas_speed",
                                    "drone_heading_deg": "cuas_heading",
                                    "custom_coords_text": "cuas_custom_text",
                                }
                                for cfg_k, ss_k in _key_map.items():
                                    if cfg_k in cleaned_cfg:
                                        st.session_state[ss_k] = cleaned_cfg[cfg_k]
                                if "n_barrels" in cleaned_cfg:
                                    st.session_state["cuas_n_barrels_fan"] = cleaned_cfg["n_barrels"]
                                st.session_state["_cuas_last_uploaded_sig"] = sig
                                st.session_state["_cuas_upload_notice"] = msg_cfg
                                st.rerun()
                            else:
                                st.error(f"❌ Invalid scenario file: {msg_cfg}")
                    if st.session_state.get("_cuas_upload_notice"):
                        st.success(f"✅ {st.session_state['_cuas_upload_notice']}")

    # --- SUB-LAB 17: SMALL ARMS, SHOTGUN PATTERNING & MULTI-BARREL DISPERSION ---
    if cuas_suite == _CUAS_SUITES[1]:
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
                st.plotly_chart(fig_2d_pat, width="stretch")
            with st_tab2:
                fig_3d_plume = create_3d_shotgun_plume_figure(sg_sim)
                st.plotly_chart(fig_3d_plume, width="stretch")

            with st.expander("📋 Pellet Impact & Ballistic Gel Penetration Telemetry", expanded=False):
                sg_pellet_df = pd.DataFrame(sg_sim["sample_pellet_records"])
                st.dataframe(sg_pellet_df, width="stretch")
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
                st.plotly_chart(fig_sw_3d, width="stretch")
            with sw_tab2:
                fig_sw_2d = create_2d_multi_distance_shotgun_comparison_figure(sw_res)
                st.plotly_chart(fig_sw_2d, width="stretch")
            with sw_tab3:
                fig_sw_curves = create_range_curves_shotgun_figure(sw_res)
                st.plotly_chart(fig_sw_curves, width="stretch")

            with st.expander("📊 Multi-Distance Progression Telemetry Dataset", expanded=False):
                sw_df = pd.DataFrame(rec)
                st.dataframe(sw_df, width="stretch")
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
                st.plotly_chart(fig_bl_2d, width="stretch")
            with bl_tab2:
                fig_bl_curves = create_barrel_length_diagnostic_curves(bl_res)
                st.plotly_chart(fig_bl_curves, width="stretch")

            with st.expander("📊 Barrel Length Performance Comparison Table", expanded=False):
                bl_df = pd.DataFrame(b_recs)
                st.dataframe(bl_df, width="stretch")
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
                tab_labels.extend([
                    "🌐 3D Target Presentation & Aspect Benchmark",
                    "🎬 3D Animated Volley Replay",
                ])

            mb_tabs = st.tabs(tab_labels)

            with mb_tabs[0]:
                fig_mb_2d = create_2d_multibarrel_target_figure(mb_sim)
                st.plotly_chart(fig_mb_2d, width="stretch")

            with mb_tabs[1]:
                fig_mb_3d = create_3d_multibarrel_dispersion_figure(mb_sim)
                st.plotly_chart(fig_mb_3d, width="stretch")

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
                    st.plotly_chart(fig_mb_multidist, width="stretch")
                    fig_mb_curves = create_range_curves_multibarrel_figure(mb_sweep_res)
                    st.plotly_chart(fig_mb_curves, width="stretch")

            if mb_tgt_type == "drone":
                with mb_tabs[3]:
                    st.markdown("#### 🌐 Target Drone Cross-Section Benchmark across 6 Standard Presentation Angles")
                    st.markdown(
                        "Ingressing drones present vastly different vulnerable areas depending on flight profile. "
                        "A head-on frontal ingress minimizes cross-section, while planform overhead exposure maximizes vulnerable surface area."
                    )
                    with st.spinner("Benchmarking multi-barrel salvo effectiveness across 6 3D aspect orientations..."):
                        mb_bench = _cached_benchmark_multibarrel_aspect(
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
                    st.plotly_chart(fig_mb_radar, width="stretch")

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
                    st.dataframe(bench_df, width="stretch")

                with mb_tabs[4]:
                    st.markdown("#### 🎬 3D Time-Scrubbed Multi-Barrel Intercept Replay")
                    st.caption("Press ▶ Play or scrub the timeline slider to inspect 3D drone motion, ripple firing, and projectile impacts.")
                    mb_replay = _cached_intercept_replay(
                        mode="multibarrel_volley",
                        layout_type="circular",
                        n_barrels=int(mb_n_barrels),
                        target_distance_m=float(min(max(mb_tgt_range, 15.0), 120.0)),
                        drone_model=str(mb_drone_model),
                        aspect_mode=str(mb_aspect_mode),
                        drone_speed_mps=18.0,
                        maneuver_profile="sinusoidal_weave",
                        maneuver_g=1.8,
                        ammunition_preset="12G_TSS_NO2_C_UAS",
                        choke="Full",
                        firing_mode="sequential_ripple",
                        ripple_delay_ms=12.0,
                        n_frames=28,
                        max_visual_pellets=78,
                        seed=42,
                    )
                    st.plotly_chart(create_3d_animated_intercept_figure(mb_replay), width="stretch")

            with st.expander("📊 Individual Barrel Cluster Diagnostics & Telemetry Log", expanded=False):
                b_df = pd.DataFrame(mb_sim["barrel_stats"])
                st.dataframe(b_df, width="stretch")
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
            current_cuas_cfg = {
                "layout_type": str(cuas_layout),
                "n_barrels": int(cuas_n_barrels),
                "mount_radius_m": float(cuas_mount_r),
                "spacing_y_m": float(cuas_sp_y),
                "spacing_z_m": float(cuas_sp_z),
                "barrel_length_in": float(cuas_b_len),
                "choke": str(cuas_choke),
                "pointing_mode": str(cuas_pointing),
                "splay_angle_mrad": float(cuas_splay),
                "firing_mode": str(cuas_fire_mode),
                "ripple_delay_ms": float(cuas_ripple_dt),
                "ammunition_preset": str(cuas_ammo),
                "drone_model": str(cuas_drone_model),
                "aspect_mode": str(cuas_aspect_mode),
                "drone_pitch_deg": float(cuas_pitch_tilt),
                "drone_yaw_deg": float(cuas_yaw_angle),
                "drone_roll_deg": float(cuas_roll_angle),
                "target_distance_m": float(cuas_dist),
                "drone_speed_mps": float(cuas_speed),
                "drone_heading_deg": float(cuas_heading),
            }
            st.session_state["_report_cuas_config"] = current_cuas_cfg
            st.session_state["_report_array_summary"] = cu_sm

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
            ck4, ck5, ck6 = st.columns([1.2, 1.2, 1.0])
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
            with ck6:
                st.download_button(
                    "💾 Save Scenario Preset (.json)",
                    serialize_cuas_scenario_config(current_cuas_cfg).encode("utf-8"),
                    "cuas_turret_scenario.json",
                    "application/json",
                    key="dl_cuas_scenario_json",
                    width="stretch",
                )

            cuas_tab1, cuas_tab2, cuas_tab3, cuas_tab4, cuas_tab5, cuas_tab6 = st.tabs([
                "🎯 2D Drone Intercept & Damage Silhouette",
                "🛸 3D Muzzle Array & Intercept Volume",
                "📊 Per-Barrel Efficiency & Hit Contributions",
                "📈 Array Architecture Benchmark (Geometry Comparison)",
                "🌐 Aspect Angle Benchmark (Cross-Section Sweep)",
                "🎬 3D Animated Intercept Replay",
            ])

            with cuas_tab1:
                fig_2d_cuas = create_2d_antidrone_intercept_figure(cuas_sim)
                st.plotly_chart(fig_2d_cuas, width="stretch")

            with cuas_tab2:
                fig_3d_cuas = create_3d_antidrone_defense_figure(cuas_sim)
                st.plotly_chart(fig_3d_cuas, width="stretch")

            with cuas_tab3:
                st.markdown("##### Per-Barrel Telemetry & Contribution Breakdown")
                b_stats = cuas_sim.get("barrel_stats", [])
                if b_stats:
                    b_df = pd.DataFrame(b_stats)
                    st.dataframe(b_df, width="stretch")
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
                    bench_res = _cached_compare_anti_drone_layouts(
                        distance_stations_m=(15.0, 25.0, 35.0, 50.0, 65.0),
                        ammunition_preset=str(cuas_ammo),
                        drone_model=str(cuas_drone_model),
                        drone_velocity_mps=float(cuas_speed),
                        pointing_mode=str(cuas_pointing),
                        firing_mode=str(cuas_fire_mode),
                        n_volleys=12,
                        seed=42,
                    )
                    fig_bench = create_antidrone_layout_comparison_figure(bench_res)
                    st.plotly_chart(fig_bench, width="stretch")

                    bench_df = pd.DataFrame(bench_res["comparison_table"])
                    st.dataframe(bench_df, width="stretch")

            with cuas_tab5:
                st.markdown("##### 3D Target Presentation & Aspect Angle Vulnerability Benchmark")
                st.caption(
                    "Evaluates the multi-barrel shotgun array against all standard 3D target drone aspect orientations "
                    "(Head-on frontal, Flank side, Oblique quartering, Tail rear, Planform overhead, Planform underbelly). "
                    "Demonstrates the significant cross-section area reduction and required pellet density when engaging ingressing drones edge-on."
                )
                with st.spinner("Benchmarking target drone aspect angles..."):
                    aspect_bench = _cached_benchmark_drone_aspect(
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
                        custom_barrel_specs_json=json.dumps(parsed_custom) if parsed_custom else "",
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
                    st.session_state["_report_aspect_table"] = aspect_bench["comparison_table"]
                    fig_aspect = create_drone_aspect_radar_figure(aspect_bench)
                    st.plotly_chart(fig_aspect, width="stretch")

                    asp_df = pd.DataFrame(aspect_bench["comparison_table"])
                    st.dataframe(asp_df, width="stretch")
                    csv_asp = asp_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Aspect Benchmark Data (.csv)",
                        csv_asp,
                        "cuas_target_aspect_benchmark.csv",
                        "text/csv",
                        key="dl_aspect_bench_csv",
                    )

            with cuas_tab6:
                st.markdown("##### 🎬 Time-Scrubbed 3D Ripple Volley & Evasive Drone Intercept Replay")
                st.caption("Press ▶ Play or drag the timeline scrubber below to visualize longitudinal shot-stringing, ripple offset timing, and target hits.")
                cuas_replay = _cached_intercept_replay(
                    mode="shotgun_array",
                    layout_type=str(cuas_layout if cuas_layout != "custom" else "circular"),
                    n_barrels=int(cuas_n_barrels),
                    target_distance_m=float(cuas_dist),
                    drone_model=str(cuas_drone_model),
                    aspect_mode=str(cuas_aspect_mode),
                    drone_speed_mps=float(cuas_speed),
                    maneuver_profile="sinusoidal_weave",
                    maneuver_g=2.0,
                    ammunition_preset=str(cuas_ammo),
                    choke=str(cuas_choke),
                    firing_mode=str(cuas_fire_mode),
                    ripple_delay_ms=float(cuas_ripple_dt),
                    n_frames=30,
                    max_visual_pellets=84,
                    seed=42,
                )
                r_sm = cuas_replay["summary"]
                rk_a, rk_b, rk_c = st.columns(3)
                rk_a.metric("Longitudinal Shot String", f"{r_sm['shot_string_length_m']:.2f} m", delta=f"{r_sm['firing_mode']} (Δt={r_sm['ripple_delay_ms']:.0f}ms)")
                rk_b.metric("Tracked Impact Summary", f"{r_sm['final_hits']} hits ({r_sm['final_lethal_hits']} lethal)", delta=f"{r_sm['total_pellets_tracked']} sampled pellets")
                rk_c.metric("Replay Horizon", f"{r_sm['t_end_ms']:.0f} ms", delta="NEUTRALIZED" if r_sm["drone_neutralized"] else "TRACKING")
                st.plotly_chart(create_3d_animated_intercept_figure(cuas_replay), width="stretch")

    # --- SUB-LAB 18: COMPLETE C-UAS TURRET ENGINEERING SUITE (13 MODULES) ---
    else:
        st.subheader("🛸 Multi-Shotgun C-UAS Turret Engineering & Mission Effectiveness Suite")
        st.markdown(
            "Complete 13-module counter-UAS engineering laboratory integrating **Closed-Loop Kalman Fire Control**, "
            "**Coordinated Swarm Saturation**, **Tungsten TSS / Flechette / Airburst Payloads**, "
            "**Turret Recoil & Constrained Pareto Optimization**, **Elevated Fallout SDZ Safety Zones**, "
            "**Multi-Sensor Kill-Chain Timelines**, **Custom Site Builder, DEM Terrain Masking, Real-World Map & GeoJSON/KML Exports**, "
            "**3D Animated Intercept Replay**, **Sustained Logistics, Barrel Heat, $/Kill & PDF/PPTX Briefings**, "
            "**A/B Turret Comparator & Layered Defense**, **Sensitivity Tornado Bottleneck Analysis**, "
            "**Unified Dynamic 3D Cooperative Raid**, and **Field-Test Calibration & Residual Diagnostics**."
        )
        st.latex(
            r"""
            \hat{\mathbf{r}}(\tau) = \hat{\mathbf{p}} + \hat{\mathbf{v}}\tau + \frac{1}{2}\hat{\mathbf{a}}\tau^2, \quad
            M\ddot{x} + C\dot{x} + Kx = \sum_{i} F_i(t), \quad
            I_{\theta}\ddot{\theta} + C_{\theta}\dot{\theta} + K_{\theta}\theta = \sum_{i} F_i(t)(z_{\text{tr}} + z_i), \quad
            m_p \frac{d\mathbf{v}}{dt} = -\frac{1}{2}\rho(z) C_D(Re,M) A_p |\mathbf{v}-\mathbf{W}|\,(\mathbf{v}-\mathbf{W}) - m_p g \hat{\mathbf{k}}
            """
        )

        sg_presets_18 = get_shotgun_presets()
        choke_defs_18 = get_choke_definitions()
        drone_specs_18 = get_drone_target_specs()
        aspect_presets_18 = get_drone_aspect_presets()
        sensor_presets_18 = get_cuas_sensor_presets()
        payload_cat_18 = get_cuas_payload_catalog()

        # --- TAB 18.1: CLOSED-LOOP FCS & EVASIVE DRONES ---
        if active_cuas_mod == _CUAS18_MODULES[0]:
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

            st.plotly_chart(create_fcs_tracking_and_evasion_figure(fcs_res), width="stretch")
            if fcs_res.get("architecture_comparison"):
                st.dataframe(pd.DataFrame(fcs_res["architecture_comparison"]), width="stretch")

        # --- TAB 18.2: MULTI-DRONE SWARM SATURATION ---
        elif active_cuas_mod == _CUAS18_MODULES[1]:
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

            st.plotly_chart(create_3d_swarm_hemisphere_figure(sw_res), width="stretch")
            st.plotly_chart(create_swarm_doctrine_tradeoff_figure(sw_res), width="stretch")

        # --- TAB 18.3: ADVANCED C-UAS PAYLOADS & AIRBURST ---
        elif active_cuas_mod == _CUAS18_MODULES[2]:
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
                pay_comp = _cached_compare_cuas_payloads(
                    distances_m=(20.0, 35.0, 50.0, 70.0, 90.0, 110.0),
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
                st.plotly_chart(create_cuas_payload_range_comparison_figure(pay_comp), width="stretch")
                st.dataframe(pd.DataFrame(pay_comp["comparison_table"]), width="stretch")
            with pt2:
                st.plotly_chart(create_airburst_standoff_cone_figure(ab_res), width="stretch")

        # --- TAB 18.4: TURRET RECOIL, GIMBAL LOADS & PARETO ARRAY OPTIMIZER ---
        elif active_cuas_mod == _CUAS18_MODULES[3]:
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
                opt_res = _cached_optimize_antidrone_array(
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
                st.plotly_chart(create_turret_recoil_dynamics_figure(rec_res, rec_sweep), width="stretch")
            with rt2:
                if opt_res["best_feasible"] is not None:
                    st.success(
                        f"**Recommended Mount-Safe Design:** Layout `{bf_opt['layout_type']}`, Pointing `{bf_opt['pointing_mode']}` "
                        f"(splay `{bf_opt['splay_angle_mrad']} mrad`), Firing `{bf_opt['firing_mode']}` (Δt `{bf_opt['ripple_delay_ms']} ms`) "
                        f"→ **P_kill = {bf_opt['kill_probability_pct']}%**, Peak Force = **{bf_opt['peak_transmitted_force_kn']} kN**"
                    )
                else:
                    st.warning("No candidate stays under the mount structural limit — increase ripple delay, raise buffer stroke/limit, or reduce barrel count.")
                st.plotly_chart(create_antidrone_optimizer_pareto_figure(opt_res), width="stretch")
                cand_df = pd.DataFrame(opt_res["candidates"])
                st.dataframe(cand_df, width="stretch")
                st.download_button("📥 Download Optimizer Trade-Study (.csv)", cand_df.to_csv(index=False).encode("utf-8"), "cuas_optimizer_trade_study.csv", "text/csv", key="cuas18_dl_opt")
            with rt3:
                st.dataframe(pd.DataFrame(rec_res["barrel_firing_states"]), width="stretch")

        # --- TAB 18.5: ELEVATED-FIRE PELLET FALLOUT & SURFACE DANGER ZONE (SDZ) ---
        elif active_cuas_mod == _CUAS18_MODULES[4]:
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
                sd_comp = _cached_compare_cuas_fallout(
                    firing_elevation_deg=float(sd_el),
                    wind_downrange_mps=float(sd_w_down),
                    wind_crossrange_mps=float(sd_w_cross),
                    default_choke=str(sd_choke),
                    n_sample_pellets=140,
                    seed=42,
                )

            ssm = sd_res["summary"]
            st.session_state["_report_sdz_summary"] = ssm
            sdk1, sdk2, sdk3, sdk4 = st.columns(4)
            sdk1.metric("99% Surface Danger Zone (SDZ)", f"{ssm['max_fallout_range_99_m']:.0f} m", delta=f"Span: {ssm['min_fallout_range_m']:.0f}–{ssm['max_fallout_range_99_m']:.0f}m | Apogee: {ssm['max_apogee_m']:.0f}m")
            sdk2.metric("Terminal Speed / Energy", f"{ssm['mean_terminal_velocity_mps']:.1f} m/s", delta=f"Mean: {ssm['mean_impact_energy_j']:.2f} J ({ssm['mean_specific_energy_j_mm2']:.3f} J/mm²)")
            sdk3.metric("Ocular (Unprotected Eye) Risk", f"{ssm['eye_hazard_pct']:.1f} %", delta="≥0.06 J/mm² or ≥1.5 J threshold")
            sdk4.metric("Skin Penetration / IATG Risk", f"{ssm['skin_hazard_pct']:.1f} %", delta=f"IATG ≥79 J Rate: {ssm['iatg_hazard_pct']:.1f} %")

            sdt1, sdt2, sdt3 = st.tabs(["🌐 3D Elevated Fallout Arcs", "🗺️ 2D Ground SDZ Map & Multi-Load Bands", "📊 Multi-Load Safety Footprint Data & CSV"])
            with sdt1:
                st.plotly_chart(create_3d_cuas_fallout_footprint_figure(sd_res), width="stretch")
            with sdt2:
                st.plotly_chart(create_2d_cuas_fallout_safety_map_figure(sd_res, sd_comp), width="stretch")
            with sdt3:
                comp_df_sd = pd.DataFrame(sd_comp["records"])
                st.dataframe(comp_df_sd, width="stretch")
                gi_df = pd.DataFrame(sd_res["ground_impacts"])
                st.download_button("📥 Download Ground Pellet Impact Fallout Log (.csv)", gi_df.to_csv(index=False).encode("utf-8"), "cuas_pellet_fallout_sdz.csv", "text/csv", key="cuas18_dl_sdz")

        # --- TAB 18.6: MULTI-SENSOR DETECTION, CLASSIFICATION & KILL-CHAIN TIMELINE ---
        elif active_cuas_mod == _CUAS18_MODULES[5]:
            st.markdown("#### 🛰️ Multi-Sensor Detection, Human-in-the-Loop & Engagement Kill-Chain Timeline")
            sensor_cat = get_kill_chain_sensor_catalog()
            kc1, kc2, kc3 = st.columns(3)
            with kc1:
                st.markdown("**Threat & Environment**")
                kc_drone = st.selectbox("Threat Drone Class", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_kc_dr")
                kc_aspect = st.selectbox("Approach Aspect", list(aspect_presets_18.keys()), format_func=lambda k: aspect_presets_18[k]["name"], index=0, key="cuas18_kc_asp")
                kc_speed = st.slider("Threat Approach Speed (m/s)", 8.0, 50.0, 28.0, 1.0, key="cuas18_kc_spd")
                kc_weather = st.selectbox(
                    "Atmospheric Visibility / Weather",
                    ["clear_day", "overcast_dusk", "night_clear", "heavy_fog_rain"],
                    format_func=lambda k: {
                        "clear_day": "☀️ Clear Day",
                        "overcast_dusk": "⛅ Overcast / Dusk",
                        "night_clear": "🌙 Clear Night (LWIR)",
                        "heavy_fog_rain": "🌧️ Heavy Fog & Rain",
                    }.get(k, k),
                    index=0,
                    key="cuas18_kc_wx",
                )
                kc_wind = st.slider("Ambient Wind Noise (m/s)", 0.0, 20.0, 5.0, 1.0, key="cuas18_kc_wind")

            with kc2:
                st.markdown("**Sensor Suite & Dark-Drone Countermeasures**")
                kc_sensors = st.multiselect(
                    "Active Acquisition Modalities",
                    list(sensor_cat.keys()),
                    default=list(sensor_cat.keys()),
                    format_func=lambda k: sensor_cat[k]["short_label"],
                    key="cuas18_kc_sens",
                )
                kc_fiber = st.checkbox(
                    "Fiber-Optic / Autonomous Inertial Threat (RF Silent)",
                    value=False,
                    key="cuas18_kc_fiber",
                    help="Drops passive RF / C2 ESM range to 0 m.",
                )
                kc_ai_t = st.slider("AI Track Classification Delay (s)", 0.1, 2.5, 0.6, 0.1, key="cuas18_kc_ai")
                kc_hitl = st.checkbox("Require Human-in-the-Loop Release Auth", value=True, key="cuas18_kc_hitl")
                kc_auth_t = st.slider("Operator Authorization Latency (s)", 0.2, 4.0, 1.2, 0.1, key="cuas18_kc_auth", disabled=not kc_hitl)

            with kc3:
                st.markdown("**Turret Agility & Fire Envelope**")
                kc_slew_ang = st.slider("Initial Off-Boresight Slew Angle (°)", 0.0, 180.0, 60.0, 10.0, key="cuas18_kc_sang")
                kc_slew_rate = st.slider("Turret Slew Rate (°/s)", 30.0, 220.0, 120.0, 10.0, key="cuas18_kc_sr")
                kc_rmax = st.slider("Max Weapon Intercept Range (m)", 30.0, 100.0, 55.0, 5.0, key="cuas18_kc_rmax")
                kc_rmin = st.slider("Asset Keep-Out Breach Radius (m)", 5.0, 30.0, 15.0, 1.0, key="cuas18_kc_rmin")
                kc_ammo = st.selectbox("Shotgun Payload", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_kc_ammo")

            with st.spinner("Simulating Multi-Sensor Acquisition & Kill-Chain Timeline..."):
                kc_res = _cached_kill_chain(
                    drone_model=str(kc_drone),
                    aspect_mode=str(kc_aspect),
                    drone_speed_mps=float(kc_speed),
                    approach_altitude_m=22.0,
                    weather=str(kc_weather),
                    ambient_wind_mps=float(kc_wind),
                    fiber_optic_autonomous=bool(kc_fiber),
                    enabled_sensors=tuple(kc_sensors) if kc_sensors else ("micro_doppler_radar",),
                    ai_classify_s=float(kc_ai_t),
                    human_in_the_loop=bool(kc_hitl),
                    human_auth_s=float(kc_auth_t),
                    slew_angle_deg=float(kc_slew_ang),
                    turret_slew_rate_deg_s=float(kc_slew_rate),
                    max_weapon_range_m=float(kc_rmax),
                    keepout_breach_radius_m=float(kc_rmin),
                    n_barrels=6,
                    ammunition_preset=str(kc_ammo),
                    choke="Modified",
                    salvo_cycle_time_s=0.85,
                    n_trials=100,
                    seed=42,
                )

            kcs = kc_res["summary"]
            st.session_state["_report_kc_summary"] = kcs
            kk1, kk2, kk3 = st.columns(3)
            kk1.metric("Fused Mean Detection Range", f"{kcs['fused_detection_range_m']:.1f} m", delta=f"RCS: {kcs['presented_area_cm2']:.0f} cm²")
            kk2.metric("First-Shot Open-Fire Range", f"{kcs['first_shot_range_m']:.1f} m", delta=f"C2 + Slew Overhead: {kcs['t_overhead_s']:.2f} s")
            kk3.metric("Engagement Window Duration", f"{kcs['t_engagement_window_s']:.2f} s", delta=f"Mean Salvos: {kcs['mean_salvos_fired']:.2f}")
            kk4, kk5 = st.columns(2)
            kk4.metric("Cumulative Kill-Chain P_kill", f"{kcs['cumulative_pk_pct']:.1f} %", delta=f"Bottleneck: {kcs['limiting_bottleneck']}")
            kk5.metric("Perimeter Breach Probability", f"{kcs['breach_rate_pct']:.1f} %", delta=f"Keep-out: {kcs['keepout_breach_radius_m']:.0f} m")

            st.plotly_chart(create_kill_chain_timeline_figure(kc_res), width="stretch")
            with st.expander("📋 Sensor Modality Performance & Approach Speed Sensitivity", expanded=False):
                st.dataframe(pd.DataFrame(kc_res["modality_breakdown"]), width="stretch")
                st.dataframe(pd.DataFrame(kc_res["speed_sweep"]), width="stretch")

        # --- TAB 18.7: MULTI-TURRET SITE DEFENSE, CUSTOM BUILDER & GIS MAP ---
        elif active_cuas_mod == _CUAS18_MODULES[6]:
            st.markdown("#### 🏰 Multi-Turret Site Defense, Custom Layout Builder & Real-World GIS Map")
            site_cat = get_site_defense_presets()
            st1, st2, st3 = st.columns(3)
            with st1:
                site_source = st.radio(
                    "Layout Source",
                    ["Preset Facility", "Custom Site Layout"],
                    horizontal=True,
                    key="cuas18_site_source",
                )
                site_key = st.selectbox(
                    "Base Facility Preset",
                    list(site_cat.keys()),
                    format_func=lambda k: site_cat[k]["name"],
                    index=0,
                    key="cuas18_site_key",
                )
            with st2:
                site_alt = st.slider("Threat Approach Altitude (m)", 5.0, 60.0, 22.0, 1.0, key="cuas18_site_alt")
                site_spd = st.slider("Threat Raid Speed (m/s)", 8.0, 45.0, 22.0, 1.0, key="cuas18_site_spd")
            with st3:
                site_rmax = st.slider("Turret Engagement Range (m)", 30.0, 100.0, 60.0, 5.0, key="cuas18_site_rmax")
                site_sdz = st.slider("Elevated Fallout SDZ Safety Envelope (m)", 100.0, 500.0, 280.0, 20.0, key="cuas18_site_sdz")

            base_site = site_cat[site_key]
            if (
                "cuas18_custom_turrets_df" not in st.session_state
                or st.session_state.get("cuas18_site_seeded_from") != site_key
            ):
                st.session_state["cuas18_custom_turrets_df"] = pd.DataFrame(base_site["turrets"])
                st.session_state["cuas18_custom_obs_df"] = pd.DataFrame(base_site["obstacles"])
                st.session_state["cuas18_custom_nfz_df"] = pd.DataFrame(base_site["nfz_sectors"])
                st.session_state["cuas18_site_seeded_from"] = site_key

            custom_tur_json = ""
            custom_obs_json = ""
            custom_nfz_json = ""
            edited_turrets = base_site["turrets"]
            edited_obs = base_site["obstacles"]
            edited_nfz = base_site["nfz_sectors"]

            if site_source == "Custom Site Layout":
                with st.expander("✏️ Custom Site Builder (Turrets, 3D Obstacles, NFZs & Site JSON)", expanded=True):
                    sc_u1, sc_u2 = st.columns([2.0, 1.0])
                    with sc_u1:
                        up_site = st.file_uploader(
                            "Upload Saved Custom Site JSON (.json)",
                            type=["json"],
                            key="cuas18_site_json_upload",
                        )
                        if up_site is not None:
                            raw_b = up_site.getvalue()
                            s_sig = f"{up_site.name}:{len(raw_b)}:{hash(raw_b)}"
                            if st.session_state.get("_cuas_site_upload_sig") != s_sig:
                                ok_s, cln_s, msg_s = validate_site_layout_json(raw_b.decode("utf-8", errors="replace"))
                                if ok_s:
                                    st.session_state["cuas18_custom_turrets_df"] = pd.DataFrame(cln_s["turrets"])
                                    st.session_state["cuas18_custom_obs_df"] = pd.DataFrame(cln_s["obstacles"])
                                    st.session_state["cuas18_custom_nfz_df"] = pd.DataFrame(cln_s["nfz_zones"])
                                    st.session_state["cuas18_map_lat"] = cln_s["origin_lat"]
                                    st.session_state["cuas18_map_lon"] = cln_s["origin_lon"]
                                    st.session_state["cuas18_map_hdg"] = cln_s["heading_deg"]
                                    st.session_state["_cuas_site_upload_sig"] = s_sig
                                    st.session_state["_cuas_site_upload_msg"] = msg_s
                                    st.rerun()
                                else:
                                    st.error(f"❌ Invalid site layout JSON: {msg_s}")
                    with sc_u2:
                        if st.button("🔄 Reset Tables from Selected Preset", width="stretch", key="cuas18_reset_site_btn"):
                            st.session_state["cuas18_custom_turrets_df"] = pd.DataFrame(base_site["turrets"])
                            st.session_state["cuas18_custom_obs_df"] = pd.DataFrame(base_site["obstacles"])
                            st.session_state["cuas18_custom_nfz_df"] = pd.DataFrame(base_site["nfz_sectors"])
                            st.rerun()
                        if st.session_state.get("_cuas_site_upload_msg"):
                            st.success(st.session_state["_cuas_site_upload_msg"])

                    ed_t_tab, ed_o_tab, ed_n_tab = st.tabs(["🔫 Turrets", "🏢 Building Obstacles", "⛔ Friendly NFZs"])
                    with ed_t_tab:
                        tur_ed_df = st.data_editor(
                            st.session_state["cuas18_custom_turrets_df"],
                            num_rows="dynamic",
                            width="stretch",
                            key="cuas18_ed_turrets",
                        )
                    with ed_o_tab:
                        obs_ed_df = st.data_editor(
                            st.session_state["cuas18_custom_obs_df"],
                            num_rows="dynamic",
                            width="stretch",
                            key="cuas18_ed_obs",
                        )
                    with ed_n_tab:
                        nfz_ed_df = st.data_editor(
                            st.session_state["cuas18_custom_nfz_df"],
                            num_rows="dynamic",
                            width="stretch",
                            key="cuas18_ed_nfz",
                        )

                    edited_turrets, edited_obs, edited_nfz = _sanitize_site_editor_tables(
                        tur_ed_df, obs_ed_df, nfz_ed_df, base_site
                    )
                    custom_tur_json = json.dumps(edited_turrets)
                    custom_obs_json = json.dumps(edited_obs)
                    custom_nfz_json = json.dumps(edited_nfz)

            # Terrain / DEM Elevation Model Selection
            dem_presets = get_terrain_dem_presets()
            active_dem_obj: TerrainDEMGrid | None = None
            active_dem_json = ""
            with st.expander("⛰️ Terrain Elevation (DEM) LOS Masking & Slope Fallout", expanded=False):
                td1, td2 = st.columns([1.2, 1.3])
                with td1:
                    dem_mode = st.radio(
                        "Terrain Elevation Source",
                        ["Flat Datum (0 m)", "Tactical DEM Preset", "Upload CSV / ESRI .asc DEM", "Fetch Real Elevation (Open-Meteo)"],
                        horizontal=False,
                        key="cuas18_dem_mode",
                    )
                with td2:
                    cur_lat = float(st.session_state.get("cuas18_map_lat", 32.9000))
                    cur_lon = float(st.session_state.get("cuas18_map_lon", -106.4000))
                    if dem_mode == "Tactical DEM Preset":
                        dem_pk = st.selectbox(
                            "Select Terrain Profile",
                            [k for k in dem_presets.keys() if k != "flat_airfield"],
                            format_func=lambda k: dem_presets[k]["name"],
                            key="cuas18_dem_preset_key",
                        )
                        active_dem_obj = build_preset_terrain_dem(
                            dem_pk, origin_lat=cur_lat, origin_lon=cur_lon
                        )
                        st.caption(dem_presets[dem_pk]["description"])
                    elif dem_mode == "Upload CSV / ESRI .asc DEM":
                        up_dem = st.file_uploader(
                            "Upload Terrain DEM (.csv `x_m,y_m,z_m` or `lat,lon,elev_m` | ESRI `.asc`)",
                            type=["csv", "asc", "txt"],
                            key="cuas18_dem_upload",
                        )
                        if up_dem is not None:
                            try:
                                txt = up_dem.getvalue().decode("utf-8", errors="replace")
                                if up_dem.name.lower().endswith(".asc") or "ncols" in txt[:200].lower():
                                    active_dem_obj = parse_esri_ascii_grid(txt, origin_lat=cur_lat, origin_lon=cur_lon, name=up_dem.name)
                                else:
                                    active_dem_obj = parse_dem_csv(txt, origin_lat=cur_lat, origin_lon=cur_lon, name=up_dem.name)
                                st.success(f"Loaded {active_dem_obj.name} ({active_dem_obj.stats()['grid_shape']})")
                            except Exception as exc:
                                st.error(f"DEM parse error: {exc}")
                        else:
                            st.info("Upload a CSV or `.asc` DEM file, or choose a preset.")
                    elif dem_mode == "Fetch Real Elevation (Open-Meteo)":
                        st.caption(f"Site Anchor: {cur_lat:.4f}°, {cur_lon:.4f}° (adjust in the Real-World Map tab)")
                        if st.button("🌐 Fetch Live Copernicus/SRTM DEM Patch", width="stretch", key="cuas18_fetch_dem_btn"):
                            ok_d, live_d, msg_d = fetch_open_meteo_elevation_dem(cur_lat, cur_lon)
                            if ok_d and live_d is not None:
                                st.session_state["_cuas18_live_dem_json"] = live_d.to_json()
                                st.success(msg_d)
                            else:
                                st.warning(msg_d)
                        if st.session_state.get("_cuas18_live_dem_json"):
                            active_dem_obj = TerrainDEMGrid.from_dict(
                                json.loads(st.session_state["_cuas18_live_dem_json"])
                            )

                if active_dem_obj is not None:
                    active_dem_json = active_dem_obj.to_json()
                    d_st = active_dem_obj.stats()
                    st.caption(
                        f"DEM Active: **{active_dem_obj.name}** | Relief: **{d_st['relief_m']:.1f} m** "
                        f"({d_st['min_m']:.1f}..{d_st['max_m']:.1f} m) | Max Slope: **{d_st['max_slope_deg']:.1f}°**"
                    )

            with st.spinner("Evaluating 360° Line-of-Sight Coverage, DEM Masking, NFZ Conflicts & Raids..."):
                site_res = _cached_site_defense(
                    site_preset=str(site_key),
                    threat_altitude_m=float(site_alt),
                    threat_speed_mps=float(site_spd),
                    drone_model="fpv_quad_small",
                    weapon_max_range_m=float(site_rmax),
                    elevation_sdz_range_m=float(site_sdz),
                    custom_turrets_json=custom_tur_json,
                    custom_obstacles_json=custom_obs_json,
                    custom_nfz_json=custom_nfz_json,
                    terrain_dem_json=active_dem_json,
                    n_raid_trials=80,
                    seed=42,
                )
            st.session_state["_report_site_result"] = site_res
            st.session_state["_site_active_layout"] = {
                "site_preset": str(site_key),
                "turrets": edited_turrets,
                "obstacles": edited_obs,
                "nfz_zones": edited_nfz,
                "terrain_dem_json": active_dem_json,
            }

            sts = site_res["summary"]
            sk_1, sk_2, sk_3, sk_4 = st.columns(4)
            sk_1.metric("Perimeter Azimuth Coverage", f"{sts['perimeter_coverage_pct']:.1f} %", delta=f"{sts['n_turrets']} Active Turrets")
            sk_2.metric("Overlapping Mutual Support (≥2)", f"{sts['overlapping_coverage_pct']:.1f} %", delta=f"Blind Spots: {sts['blind_spot_pct']:.1f} %")
            sk_3.metric("NFZ SDZ Sector Conflicts", f"{sts['nfz_fallout_conflict_pct']:.1f} %", delta=f"Zero-Breach Raid Stop: {sts['site_zero_breach_pct']:.1f} %")
            if active_dem_obj is not None and sts.get("terrain_fallout_max_m") is not None:
                sk_4.metric(
                    "DEM Masked / Fallout Span",
                    f"{sts['terrain_masked_pct']:.1f}% masked",
                    delta=f"Fallout {sts['terrain_fallout_min_m']:.0f}–{sts['terrain_fallout_max_m']:.0f} m (flat {sts['flat_fallout_m']:.0f} m)",
                )
            else:
                sk_4.metric("Terrain Relief", "0.0 m (Flat)", delta="Enable DEM above to model ridge masking")

            site_v1, site_v2, site_v3 = st.tabs([
                "🗺️ Local Tactical Map & Workload",
                "⛰️ 3D DEM Terrain LOS & Slope Fallout",
                "🌍 Real-World Lat/Lon Map & GIS Export (.geojson / .kml)",
            ])
            with site_v1:
                st.plotly_chart(create_multiturret_site_map_figure(site_res), width="stretch")
                st.markdown("##### 📊 Per-Turret Engagement Workload & Kills")
                st.dataframe(pd.DataFrame(site_res["turret_workload"]), width="stretch")

            with site_v2:
                dem_for_plot = active_dem_obj or build_preset_terrain_dem(
                    "flat_airfield",
                    origin_lat=float(st.session_state.get("cuas18_map_lat", 32.9000)),
                    origin_lon=float(st.session_state.get("cuas18_map_lon", -106.4000)),
                )
                st.plotly_chart(create_terrain_dem_site_3d_figure(site_res, dem_for_plot), width="stretch")
                if site_res.get("terrain_fallout_profile"):
                    st.markdown("##### 📐 Terrain-Adjusted Elevated Fallout SDZ vs Azimuth (12-Gauge #4 Buck @ 30°)")
                    st.dataframe(pd.DataFrame(site_res["terrain_fallout_profile"]), width="stretch")
                else:
                    st.info("Select a Tactical DEM Preset, upload a CSV/.asc grid, or fetch Copernicus/SRTM elevation above to compute ridge-line masking and upslope/downslope Fallout SDZs.")

            with site_v3:
                g1, g2, g3, g4 = st.columns([1.4, 0.9, 0.9, 0.9])
                with g1:
                    st.selectbox(
                        "Quick Location Anchor",
                        list(_SITE_GEO_ANCHORS.keys()),
                        index=0,
                        key="cuas18_geo_anchor",
                        on_change=_on_site_anchor_change,
                    )
                with g2:
                    map_lat = st.number_input("Origin Latitude (°)", -85.0, 85.0, 32.9000, 0.005, format="%.4f", key="cuas18_map_lat")
                with g3:
                    map_lon = st.number_input("Origin Longitude (°)", -180.0, 180.0, -106.4000, 0.005, format="%.4f", key="cuas18_map_lon")
                with g4:
                    map_hdg = st.slider("Site Rotation (°)", 0.0, 360.0, 0.0, 5.0, key="cuas18_map_hdg")

                map_style = st.radio(
                    "Basemap Style",
                    ["open-street-map", "carto-darkmatter", "carto-positron"],
                    horizontal=True,
                    key="cuas18_map_style",
                )
                fig_real_map = create_real_map_site_defense_figure(
                    site_res=site_res,
                    origin_lat=float(map_lat),
                    origin_lon=float(map_lon),
                    sdz_radius_m=float(site_sdz),
                    heading_deg=float(map_hdg),
                    map_style=str(map_style),
                )
                st.plotly_chart(fig_real_map, width="stretch")

                geo_dict = build_cuas_site_geojson(
                    site_res=site_res,
                    origin_lat=float(map_lat),
                    origin_lon=float(map_lon),
                    sdz_radius_m=float(site_sdz),
                    heading_deg=float(map_hdg),
                )
                kml_doc = export_cuas_site_kml(
                    site_res=site_res,
                    origin_lat=float(map_lat),
                    origin_lon=float(map_lon),
                    sdz_radius_m=float(site_sdz),
                    heading_deg=float(map_hdg),
                    name=f"CUAS_Site_{site_key}",
                )
                site_layout_str = serialize_site_layout_json(
                    {
                        "origin_lat": float(map_lat),
                        "origin_lon": float(map_lon),
                        "heading_deg": float(map_hdg),
                        "turrets": edited_turrets,
                        "obstacles": edited_obs,
                        "nfz_zones": edited_nfz,
                    }
                )
                dl_g1, dl_g2, dl_g3 = st.columns(3)
                dl_g1.download_button(
                    "📥 Download GeoJSON (.geojson)",
                    json.dumps(geo_dict, indent=2).encode("utf-8"),
                    f"cuas_site_{site_key}.geojson",
                    "application/geo+json",
                    key="cuas18_dl_geojson",
                    width="stretch",
                )
                dl_g2.download_button(
                    "🌍 Download Google Earth 3D (.kml)",
                    kml_doc.encode("utf-8"),
                    f"cuas_site_{site_key}.kml",
                    "application/vnd.google-earth.kml+xml",
                    key="cuas18_dl_kml",
                    width="stretch",
                )
                dl_g3.download_button(
                    "💾 Save Site Layout (.json)",
                    site_layout_str.encode("utf-8"),
                    f"cuas_site_layout_{site_key}.json",
                    "application/json",
                    key="cuas18_dl_site_json",
                    width="stretch",
                )

        # --- TAB 18.8: 3D ANIMATED INTERCEPT REPLAY ---
        elif active_cuas_mod == _CUAS18_MODULES[7]:
            st.markdown("#### 🎬 3D Time-Scrubbed Intercept Replay (Evasive Drone + Shot-Stringing Pellets)")
            an1, an2, an3 = st.columns(3)
            with an1:
                an_lay = st.selectbox("Turret Layout", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"], index=0, key="cuas18_an_lay")
                an_nb = st.slider("Active Barrels", 2, 12, 6, 1, key="cuas18_an_nb")
                an_ammo = st.selectbox("Ammunition", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_an_ammo")
            with an2:
                an_fire = st.selectbox("Firing Mode", ["balanced_pair_ripple", "sequential_ripple", "simultaneous_volley"], index=0, key="cuas18_an_fire")
                an_dt = st.slider("Ripple Delay Δt (ms)", 0.0, 35.0, 15.0, 1.0, key="cuas18_an_dt")
                an_dist = st.slider("Nominal Intercept Distance (m)", 15.0, 65.0, 35.0, 5.0, key="cuas18_an_dist")
            with an3:
                an_drone = st.selectbox("Target Drone", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=1, key="cuas18_an_dr")
                an_aspect = st.selectbox("Presentation Aspect", list(aspect_presets_18.keys()), format_func=lambda k: aspect_presets_18[k]["name"], index=0, key="cuas18_an_asp")
                an_man = st.selectbox("Evasive Maneuver", ["sinusoidal_weave", "corkscrew_spiral", "terminal_popup_jink", "linear_cruise"], index=0, key="cuas18_an_man")

            replay_res = _cached_intercept_replay(
                mode="shotgun_array",
                layout_type=str(an_lay),
                n_barrels=int(an_nb),
                target_distance_m=float(an_dist),
                drone_model=str(an_drone),
                aspect_mode=str(an_aspect),
                drone_speed_mps=18.0,
                maneuver_profile=str(an_man),
                maneuver_g=2.0,
                ammunition_preset=str(an_ammo),
                choke="Modified",
                firing_mode=str(an_fire),
                ripple_delay_ms=float(an_dt),
                n_frames=30,
                max_visual_pellets=84,
                seed=42,
            )
            rsm18 = replay_res["summary"]
            rk18_1, rk18_2, rk18_3, rk18_4 = st.columns(4)
            rk18_1.metric("Longitudinal Shot String", f"{rsm18['shot_string_length_m']:.2f} m", delta=f"Ripple Δt = {rsm18['ripple_delay_ms']:.0f} ms")
            rk18_2.metric("Sampled Pellet Hits", f"{rsm18['final_hits']} / {rsm18['total_pellets_tracked']}", delta=f"{rsm18['final_lethal_hits']} lethal strikes")
            rk18_3.metric("Replay Window", f"{rsm18['t_end_ms']:.0f} ms", delta=f"@ {rsm18['target_distance_m']:.0f} m")
            rk18_4.metric("Outcome", "NEUTRALIZED" if rsm18["drone_neutralized"] else "SURVIVED", delta=rsm18["firing_mode"])
            st.plotly_chart(create_3d_animated_intercept_figure(replay_res), width="stretch")

        # --- TAB 18.9: SUSTAINED LOGISTICS, BARREL HEAT, $/KILL & TRADE-STUDY REPORT ---
        elif active_cuas_mod == _CUAS18_MODULES[8]:
            st.markdown("#### 🔥 Sustained Multi-Wave Logistics, Barrel Heating, $/Kill Economics & HTML Report Export")
            econ_cat = get_cuas_ammo_economics_catalog()
            mag_cat = get_magazine_system_specs()
            lg1, lg2, lg3 = st.columns(3)
            with lg1:
                lg_waves = st.slider("Incoming Attack Waves", 2, 12, 6, 1, key="cuas18_lg_wv")
                lg_dpw = st.slider("Drones per Wave", 1, 10, 4, 1, key="cuas18_lg_dpw")
                lg_int = st.slider("Inter-Wave Cooling Interval (s)", 15.0, 120.0, 45.0, 5.0, key="cuas18_lg_int")
            with lg2:
                lg_ammo = st.selectbox("Primary Ammunition Family", list(econ_cat.keys()), format_func=lambda k: econ_cat[k]["name"], index=0, key="cuas18_lg_ammo")
                lg_mag = st.selectbox("Magazine / Feed System", list(mag_cat.keys()), format_func=lambda k: mag_cat[k]["name"], index=1, key="cuas18_lg_mag")
                lg_bmass = st.slider("Single Barrel Steel Mass (kg)", 0.70, 2.50, 1.15, 0.05, key="cuas18_lg_bmass")
            with lg3:
                lg_drone = st.selectbox("Wave Threat Class", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_lg_dr")
                lg_dist = st.slider("Mean Intercept Range (m)", 15.0, 70.0, 40.0, 5.0, key="cuas18_lg_dist")
                lg_nb = st.slider("Turret Barrels", 2, 12, 6, 1, key="cuas18_lg_nb")

            with st.spinner("Simulating Multi-Wave Barrel Thermodynamics & Payload Cost-per-Kill..."):
                log_res = _cached_sustained_logistics(
                    n_waves=int(lg_waves),
                    drones_per_wave=int(lg_dpw),
                    wave_interval_s=float(lg_int),
                    n_barrels=int(lg_nb),
                    ammo_family=str(lg_ammo),
                    magazine_type=str(lg_mag),
                    barrel_mass_kg=float(lg_bmass),
                    ambient_temp_c=22.0,
                    target_range_m=float(lg_dist),
                    drone_model=str(lg_drone),
                    aspect_mode="head_on_frontal",
                    n_trials=60,
                    seed=42,
                )

            lsm = log_res["summary"]
            lk1, lk2, lk3 = st.columns(3)
            lk1.metric("Peak Barrel Temperature", f"{lsm['peak_barrel_temp_c']:.1f} °C", delta=lsm["cookoff_risk"])
            lk2.metric("Mean Cost per Kill ($/Kill)", f"${lsm['mean_cost_per_kill_usd']:.2f}", delta=f"{lsm['total_rounds_fired']:.0f} total rounds fired")
            lk3.metric("Wave Clearance Survival", f"{lsm['wave_survival_pct']:.1f} %", delta=f"{lsm['total_drones_killed']:.1f} kills")
            lk4, lk5 = st.columns(2)
            lk4.metric("Reload & Clear Downtime", f"{lsm['total_reload_time_s']:.1f} s", delta=f"Mean Stoppages: {lsm['mean_stoppages']:.2f}")
            lk5.metric("Feed Architecture", lsm["magazine_name"], delta=lsm["ammo_name"])

            st.plotly_chart(create_logistics_thermal_and_cost_figure(log_res), width="stretch")
            st.markdown("##### 💵 7-Payload Cost-per-Kill & Mass Efficiency Comparison")
            st.dataframe(pd.DataFrame(log_res["economics_comparison"]), width="stretch")

            st.markdown("---")
            st.markdown("##### 📄 One-Click Executive Trade-Study Report Export")
            rep_name = st.text_input("Report Title", value="Multi-Shotgun C-UAS Turret Engineering Trade Study", key="cuas18_rep_title")
            rep_cfg = st.session_state.get(
                "_report_cuas_config",
                {
                    "n_barrels": int(lg_nb),
                    "ammo_family": str(lg_ammo),
                    "magazine_type": str(lg_mag),
                    "target_range_m": float(lg_dist),
                    "drone_model": str(lg_drone),
                },
            )
            rep_html = generate_cuas_trade_study_html(
                scenario_name=str(rep_name),
                config=rep_cfg,
                array_summary=st.session_state.get("_report_array_summary"),
                aspect_table=st.session_state.get("_report_aspect_table"),
                kill_chain_summary=st.session_state.get("_report_kc_summary"),
                logistics_summary=lsm,
                economics_table=log_res["economics_comparison"],
                sdz_summary=st.session_state.get("_report_sdz_summary"),
            )
            rep_c1, rep_c2, rep_c3, rep_c4 = st.columns(4)
            with rep_c1:
                st.download_button(
                    "📥 Standalone HTML Report",
                    rep_html.encode("utf-8"),
                    "cuas_trade_study_report.html",
                    "text/html",
                    key="cuas18_dl_html_report",
                    width="stretch",
                )
            with rep_c2:
                pdf_deck = generate_executive_briefing_pdf(
                    title=str(rep_name),
                    subtitle=f"{lsm['magazine_name']} | {lsm['ammo_name']}",
                    ab_result=st.session_state.get("_report_ab_result"),
                    site_result=st.session_state.get("_report_site_result"),
                    tornado_result=st.session_state.get("_report_tornado_result"),
                    logistics_result=log_res,
                    raid_result=st.session_state.get("_report_raid_result"),
                )
                st.download_button(
                    "📑 Executive Briefing (.pdf)",
                    pdf_deck,
                    "cuas_executive_briefing.pdf",
                    "application/pdf",
                    key="cuas18_dl_pdf_briefing",
                    width="stretch",
                )
            with rep_c3:
                pptx_deck = generate_executive_briefing_pptx(
                    title=str(rep_name),
                    subtitle=f"{lsm['magazine_name']} | {lsm['ammo_name']}",
                    ab_result=st.session_state.get("_report_ab_result"),
                    site_result=st.session_state.get("_report_site_result"),
                    tornado_result=st.session_state.get("_report_tornado_result"),
                    logistics_result=log_res,
                    raid_result=st.session_state.get("_report_raid_result"),
                )
                st.download_button(
                    "📊 PowerPoint Deck (.pptx)",
                    pptx_deck,
                    "cuas_executive_briefing.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key="cuas18_dl_pptx_briefing",
                    width="stretch",
                )
            with rep_c4:
                st.download_button(
                    "💾 Preset Config (.json)",
                    serialize_cuas_scenario_config(rep_cfg).encode("utf-8"),
                    "cuas_trade_study_preset.json",
                    "application/json",
                    key="cuas18_dl_json_report",
                    width="stretch",
                )

            with st.expander("💾 Persistent Scenario Library (Save / Load / Compare Diff)", expanded=False):
                lib = ScenarioLibrary()
                sv_c1, sv_c2, sv_c3 = st.columns([1.4, 1.0, 0.8])
                with sv_c1:
                    scn_save_name = st.text_input(
                        "Scenario Snapshot Name",
                        value=f"C-UAS {lg_nb}Bbl {str(lg_ammo)[:12]}",
                        key="cuas18_lib_name",
                    )
                with sv_c2:
                    scn_save_tags = st.text_input("Tags (comma-separated)", value="cuas,logistics", key="cuas18_lib_tags")
                with sv_c3:
                    st.write("")
                    if st.button("💾 Save to Library", type="primary", width="stretch", key="cuas18_lib_save_btn"):
                        rec = lib.save_scenario(
                            name=scn_save_name,
                            domain="cuas",
                            config={**rep_cfg, "n_waves": int(lg_waves), "drones_per_wave": int(lg_dpw), "n_barrels": int(lg_nb), "ammo_family": str(lg_ammo)},
                            metrics=lsm,
                            tags=[t.strip() for t in scn_save_tags.split(",") if t.strip()],
                        )
                        st.success(f"Saved `{rec['name']}` (id `{rec['id']}`).")

                saved_list = lib.list_scenarios(domain="cuas")
                if saved_list:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "ID": r["id"],
                                    "Name": r["name"],
                                    "Updated": r["updated_at"],
                                    "Tags": ", ".join(r["tags"]),
                                    "$/Kill": r["metrics"].get("mean_cost_per_kill_usd", "—"),
                                    "Peak °C": r["metrics"].get("peak_barrel_temp_c", "—"),
                                }
                                for r in saved_list
                            ]
                        ),
                        width="stretch",
                    )
                    if len(saved_list) >= 2:
                        id_map = {f"{r['name']} ({r['id']})": r["id"] for r in saved_list}
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            pick_a = st.selectbox("Diff Baseline A", list(id_map.keys()), index=1, key="cuas18_lib_da")
                        with dc2:
                            pick_b = st.selectbox("Diff Candidate B", list(id_map.keys()), index=0, key="cuas18_lib_db")
                        if id_map[pick_a] != id_map[pick_b]:
                            diff_res = lib.diff_scenarios(id_map[pick_a], id_map[pick_b])
                            if diff_res["metric_diffs"]:
                                st.markdown("**Outcome Metric Delta (B - A)**")
                                st.dataframe(pd.DataFrame(diff_res["metric_diffs"][:10]), width="stretch")

        # --- TAB 18.10: A/B TURRET COMPARATOR & LAYERED C-UAS DEFENSE ---
        elif active_cuas_mod == _CUAS18_MODULES[9]:
            st.markdown("#### ⚔️ A/B/C Turret Architecture Comparator & Layered Counter-UAS Defense")
            comp_t1, comp_t2 = st.tabs([
                "⚔️ Side-by-Side A/B/C Turret Architecture Scorecard",
                "🛡️ Layered C-UAS Defense (Outer Soft/Hard-Kill + Inner Multi-Shotgun)",
            ])

            with comp_t1:
                comp_cat = get_cuas_comparator_presets()
                avail_map: dict[str, dict[str, Any]] = dict(comp_cat)
                if st.session_state.get("_report_cuas_config"):
                    saved_cfg = dict(st.session_state["_report_cuas_config"])
                    saved_cfg.setdefault("preset_key", "active_mode5_cfg")
                    saved_cfg["name"] = f"Active Mode 5 ({saved_cfg.get('n_barrels', 6)}-Bbl {saved_cfg.get('layout_type', 'circular')})"
                    avail_map["active_mode5_cfg"] = saved_cfg

                all_ckeys = list(avail_map.keys())
                def_sel = all_ckeys[:3] if len(all_ckeys) >= 3 else all_ckeys
                ab_c1, ab_c2, ab_c3 = st.columns(3)
                with ab_c1:
                    sel_ab_keys = st.multiselect(
                        "Turret Configurations to Compare (2–4)",
                        all_ckeys,
                        default=def_sel,
                        format_func=lambda k: avail_map[k]["name"],
                        key="cuas18_ab_sel",
                    )
                    ab_drone = st.selectbox(
                        "Benchmark Threat Class",
                        list(drone_specs_18.keys()),
                        format_func=lambda k: drone_specs_18[k]["name"],
                        index=0,
                        key="cuas18_ab_dr",
                    )
                with ab_c2:
                    ab_aspect = st.selectbox(
                        "Presentation Aspect",
                        list(aspect_presets_18.keys()),
                        format_func=lambda k: aspect_presets_18[k]["name"],
                        index=0,
                        key="cuas18_ab_asp",
                    )
                    ab_dist = st.slider("Nominal Intercept Distance (m)", 20.0, 65.0, 40.0, 5.0, key="cuas18_ab_dist")
                    ab_spd = st.slider("Threat Approach Speed (m/s)", 10.0, 45.0, 25.0, 1.0, key="cuas18_ab_spd")
                with ab_c3:
                    ab_wx = st.selectbox(
                        "Weather / Visibility",
                        ["clear_day", "overcast_dusk", "night_clear", "heavy_fog_rain"],
                        index=0,
                        key="cuas18_ab_wx",
                    )
                    ab_fiber = st.checkbox("Fiber-Optic RF-Silent Threat", value=False, key="cuas18_ab_fo")
                    ab_swarm = st.slider("Swarm Wave Benchmark Size", 3, 12, 6, 1, key="cuas18_ab_sw")

                chosen_cfgs = [avail_map[k] for k in sel_ab_keys[:4]]
                if len(chosen_cfgs) < 2:
                    st.info("Select at least 2 turret configurations above to run the side-by-side comparison.")
                else:
                    with st.spinner("Evaluating Multi-Module Mission Scorecard across Selected Turrets..."):
                        ab_res = _cached_compare_ab(
                            configs_json=json.dumps(chosen_cfgs, sort_keys=True),
                            drone_model=str(ab_drone),
                            aspect_mode=str(ab_aspect),
                            target_distance_m=float(ab_dist),
                            drone_speed_mps=float(ab_spd),
                            weather=str(ab_wx),
                            fiber_optic_autonomous=bool(ab_fiber),
                            n_swarm_drones=int(ab_swarm),
                            n_waves=4,
                            drones_per_wave=3,
                            seed=42,
                        )
                    ab_sm = ab_res["summary"]
                    ab_sc = ab_res["scorecard"]
                    st.session_state["_report_ab_result"] = ab_res
                    win_row = max(ab_sc, key=lambda r: r["mei_score"])
                    ak1, ak2, ak3 = st.columns(3)
                    ak1.metric("Top Ranked Architecture", win_row["short_name"], delta=f"MEI Score: {win_row['mei_score']:.1f} / 100")
                    ak2.metric("Winner Kill-Chain / Swarm Pk", f"{win_row['kill_chain_pk_pct']:.1f} %", delta=f"Swarm Survival: {win_row['swarm_survival_pct']:.1f} %")
                    ak3.metric("Winner Cost & Thermal Margin", f"${win_row['cost_per_kill_usd']:.2f} / kill", delta=f"Cook-off Margin: +{win_row['cookoff_margin_c']:.1f} °C")

                    st.plotly_chart(create_ab_comparator_radar_and_bars_figure(ab_res), width="stretch")

                    disp_rows = [
                        {
                            "Architecture": r["label"],
                            "MEI Score": r["mei_score"],
                            "Kill-Chain Pk (%)": r["kill_chain_pk_pct"],
                            "ΔPk vs A": f"{r['delta_pk_vs_a']:+.1f}%",
                            "Swarm Survival (%)": r["swarm_survival_pct"],
                            "Peak Recoil (kN)": r["peak_recoil_kn"],
                            "Max Fallout SDZ (m)": r["sdz_max_range_m"],
                            "Peak Temp (°C)": r["peak_barrel_temp_c"],
                            "$/Kill (USD)": r["cost_per_kill_usd"],
                            "Δ$/Kill vs A": f"{r['delta_cost_vs_a_usd']:+.2f}",
                        }
                        for r in ab_sc
                    ]
                    ab_df = pd.DataFrame(disp_rows)
                    st.dataframe(ab_df, width="stretch")
                    st.download_button(
                        "📥 Download A/B Architecture Scorecard (.csv)",
                        ab_df.to_csv(index=False).encode("utf-8"),
                        "cuas_ab_scorecard.csv",
                        "text/csv",
                        key="cuas18_dl_ab_csv",
                    )

            with comp_t2:
                outer_cat = get_outer_layer_catalog()
                econ_cat_l = get_cuas_ammo_economics_catalog()
                mag_cat_l = get_magazine_system_specs()
                ly1, ly2, ly3 = st.columns(3)
                with ly1:
                    ly_outer = st.selectbox(
                        "Outer Screening Layer",
                        list(outer_cat.keys()),
                        format_func=lambda k: outer_cat[k]["name"],
                        index=0,
                        key="cuas18_ly_outer",
                    )
                    st.caption(f"ℹ️ {outer_cat[ly_outer]['description']}")
                    ly_omag = st.number_input(
                        "Outer Magazine / Interceptor Capacity",
                        min_value=1,
                        max_value=9999,
                        value=int(outer_cat[ly_outer]["magazine_capacity"]),
                        step=1,
                        key="cuas18_ly_omag",
                    )
                with ly2:
                    ly_fo_pct = st.slider(
                        "Fiber-Optic / Autonomous Mix (%)",
                        0.0,
                        100.0,
                        25.0,
                        5.0,
                        key="cuas18_ly_fo",
                        help="Fraction of incoming swarm immune to RF/GNSS jamming.",
                    )
                    ly_wx = st.selectbox("Weather Regime", ["clear_day", "heavy_fog_rain"], index=0, key="cuas18_ly_wx")
                    ly_waves = st.slider("Attack Waves", 2, 10, 5, 1, key="cuas18_ly_waves")
                    ly_dpw = st.slider("Drones per Wave", 2, 12, 5, 1, key="cuas18_ly_dpw")
                with ly3:
                    ly_nb = st.slider("Inner Shotgun Barrels", 2, 12, 6, 1, key="cuas18_ly_nb")
                    ly_ammo = st.selectbox(
                        "Inner Shotgun Payload",
                        list(econ_cat_l.keys()),
                        format_func=lambda k: econ_cat_l[k]["name"],
                        index=0,
                        key="cuas18_ly_ammo",
                    )
                    ly_mag = st.selectbox(
                        "Inner Magazine System",
                        list(mag_cat_l.keys()),
                        format_func=lambda k: mag_cat_l[k]["name"],
                        index=1,
                        key="cuas18_ly_mag",
                    )

                with st.spinner("Simulating Paired Layered vs Inner-Only Swarm Defense..."):
                    layer_res = _cached_layered_defense(
                        outer_layer_id=str(ly_outer),
                        outer_magazine=int(ly_omag),
                        n_waves=int(ly_waves),
                        drones_per_wave=int(ly_dpw),
                        wave_interval_s=40.0,
                        fiber_optic_fraction=float(ly_fo_pct) / 100.0,
                        weather=str(ly_wx),
                        inner_n_barrels=int(ly_nb),
                        inner_ammo_family=str(ly_ammo),
                        inner_magazine_type=str(ly_mag),
                        target_range_m=40.0,
                        drone_model="fpv_quad_small",
                        n_trials=60,
                        seed=42,
                    )
                lsm2 = layer_res["summary"]
                lk_1, lk_2, lk_3 = st.columns(3)
                lk_1.metric("Layered Defense Survival", f"{lsm2['layered_survival_pct']:.1f} %", delta=f"{lsm2['survival_gain_pct']:+.1f}% vs Inner-Only")
                lk_2.metric("Inner Ammo Saved / Raid", f"{lsm2['inner_rounds_saved']:.1f} rds", delta=f"Temp Relief: -{lsm2['peak_temp_reduction_c']:.1f} °C")
                lk_3.metric("Blended Cost per Kill", f"${lsm2['blended_cost_per_kill_usd']:.2f}", delta=f"Outer Share: {lsm2['outer_kill_share_pct']:.1f}% | Inner-Only: ${lsm2['inner_only_cost_per_kill_usd']:.2f}")

                st.plotly_chart(create_layered_defense_figure(layer_res), width="stretch")
                st.dataframe(pd.DataFrame(layer_res["wave_timeline"]), width="stretch")

        # --- TAB 18.11: SENSITIVITY & BOTTLENECK TORNADO ---
        elif active_cuas_mod == _CUAS18_MODULES[10]:
            st.markdown("#### 📊 C-UAS Kill-Chain Sensitivity & Engineering Bottleneck Tornado")
            st.caption(
                "One-at-a-time ±% perturbation sweep across 8 physical, sensor, C2, and kinematic parameters "
                "identifying which bottlenecks dominate cumulative Kill-Chain P_kill."
            )
            sn1, sn2, sn3 = st.columns(3)
            with sn1:
                sn_drone = st.selectbox("Baseline Threat Drone", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_sn_dr")
                sn_aspect = st.selectbox("Presentation Aspect", list(aspect_presets_18.keys()), format_func=lambda k: aspect_presets_18[k]["name"], index=0, key="cuas18_sn_asp")
                sn_ammo = st.selectbox("Baseline Payload", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_sn_ammo")
            with sn2:
                sn_speed = st.slider("Threat Speed (m/s)", 10.0, 45.0, 28.0, 1.0, key="cuas18_sn_spd")
                sn_rmax = st.slider("Weapon Max Range (m)", 35.0, 85.0, 55.0, 5.0, key="cuas18_sn_rmax")
                sn_nb = st.slider("Turret Barrels", 3, 12, 6, 1, key="cuas18_sn_nb")
            with sn3:
                sn_pct = st.slider("Parameter Perturbation (±%)", 10.0, 35.0, 20.0, 5.0, key="cuas18_sn_pct")
                sn_wx = st.selectbox("Weather Regime", ["clear_day", "overcast_dusk", "night_clear", "heavy_fog_rain"], index=0, key="cuas18_sn_wx")
                sn_fo = st.checkbox("Fiber-Optic Dark Threat", value=False, key="cuas18_sn_fo")

            base_sn_cfg = {
                "drone_model": str(sn_drone),
                "aspect_mode": str(sn_aspect),
                "ammunition_preset": str(sn_ammo),
                "drone_speed_mps": float(sn_speed),
                "max_weapon_range_m": float(sn_rmax),
                "n_barrels": int(sn_nb),
                "weather": str(sn_wx),
                "fiber_optic_autonomous": bool(sn_fo),
            }
            with st.spinner("Running 17-Point ±% C-UAS Kill-Chain Sensitivity Tornado..."):
                cuas_tor = _cached_cuas_tornado(
                    baseline_cfg_json=json.dumps(base_sn_cfg, sort_keys=True),
                    perturbation_pct=float(sn_pct),
                    n_trials=75,
                    seed=42,
                )
            ct_sm = cuas_tor["summary"]
            st.session_state["_report_tornado_result"] = cuas_tor
            bn_top = cuas_tor["top_bottlenecks"]
            sk_t1, sk_t2, sk_t3 = st.columns(3)
            sk_t1.metric("Baseline Kill-Chain P_kill", f"{ct_sm['baseline_value']:.1f} %", delta=f"±{ct_sm['perturbation_pct']:.0f}% OAT Sweep")
            sk_t2.metric("#1 Limiting Bottleneck", ct_sm["top_driver"], delta=f"Pk Swing: {ct_sm['top_swing']:.1f} %")
            sk_t3.metric("#2 Secondary Driver", ct_sm["second_driver"], delta=f"Pk Swing: {bn_top[1]['swing']:.1f} %")

            st.info(
                f"**Actionable Priority 1 ({bn_top[0]['label']}):** {bn_top[0]['recommendation']}\n\n"
                f"**Priority 2 ({bn_top[1]['label']}):** {bn_top[1]['recommendation']}"
            )
            st.plotly_chart(create_sensitivity_tornado_figure(cuas_tor, metric_label="Kill-Chain P_kill (%)", higher_is_better=True), width="stretch")

            tor_c_df = pd.DataFrame(cuas_tor["rows"])
            st.dataframe(tor_c_df, width="stretch")
            st.download_button(
                "📥 Download C-UAS Tornado Sensitivity Data (.csv)",
                tor_c_df.to_csv(index=False).encode("utf-8"),
                "cuas_killchain_tornado.csv",
                "text/csv",
                key="cuas18_dl_tor_csv",
            )

        # --- TAB 18.12: UNIFIED DYNAMIC 3D TACTICAL RAID SIMULATOR ---
        elif active_cuas_mod == _CUAS18_MODULES[11]:
            st.markdown("#### ⚔️ Unified Dynamic 3D Cooperative Multi-Turret Tactical Raid")
            st.caption(
                "Time-stepped 3D engagement engine unifying Multi-Sensor Kill Chain (RF / Fiber-Optic immunity), "
                "Swarm Evasion, Cooperative Turret Weapon-Target Assignment (WTA), Building & DEM LOS Masking, "
                "Magazine Reloads, and Newton Barrel Heating."
            )
            site_cat_12 = get_site_defense_presets()
            dem_presets_12 = get_terrain_dem_presets()
            kc_sens_12 = get_kill_chain_sensor_catalog()
            mag_cat_12 = get_magazine_system_specs()

            saved_site = st.session_state.get("_site_active_layout")
            use_m7 = False
            if saved_site is not None:
                use_m7 = st.checkbox(
                    f"Use Active Site & DEM Layout Configured in Module 7 ({saved_site.get('site_preset', 'custom')})",
                    value=True,
                    key="cuas18_raid_use_m7",
                )

            rd1, rd2, rd3 = st.columns(3)
            with rd1:
                if use_m7 and saved_site is not None:
                    r_site = str(saved_site.get("site_preset", "fob_compound_4_turret"))
                    r_tur_json = json.dumps(saved_site.get("turrets", []))
                    r_obs_json = json.dumps(saved_site.get("obstacles", []))
                    r_nfz_json = json.dumps(saved_site.get("nfz_zones", []))
                    r_dem_json = str(saved_site.get("terrain_dem_json", ""))
                    r_dem_obj = TerrainDEMGrid.from_dict(json.loads(r_dem_json)) if r_dem_json else None
                else:
                    r_site = st.selectbox(
                        "Facility Layout",
                        list(site_cat_12.keys()),
                        format_func=lambda k: site_cat_12[k]["name"],
                        key="cuas18_raid_site",
                    )
                    r_dem_pk = st.selectbox(
                        "Terrain Elevation Profile",
                        list(dem_presets_12.keys()),
                        format_func=lambda k: dem_presets_12[k]["name"],
                        key="cuas18_raid_dem",
                    )
                    r_tur_json, r_obs_json, r_nfz_json = "", "", ""
                    r_dem_obj = None if r_dem_pk == "flat_airfield" else build_preset_terrain_dem(r_dem_pk)
                    r_dem_json = r_dem_obj.to_json() if r_dem_obj is not None else ""

                r_sensors = st.multiselect(
                    "Active Kill-Chain Sensors",
                    list(kc_sens_12.keys()),
                    default=["micro_doppler_radar", "eoir_thermal_optical", "rf_c2_df"],
                    format_func=lambda k: kc_sens_12[k]["short_label"],
                    key="cuas18_raid_sensors",
                )
                r_wx = st.selectbox(
                    "Weather Regime",
                    ["clear_day", "overcast_dusk", "night_clear", "heavy_fog_rain"],
                    key="cuas18_raid_wx",
                )

            with rd2:
                r_n_drones = st.slider("Inbound Swarm Size", 4, 24, 10, 1, key="cuas18_raid_nd")
                r_fo_pct = st.slider("Fiber-Optic (RF-Dark) Share (%)", 0, 100, 30, 5, key="cuas18_raid_fo")
                r_pattern = st.selectbox(
                    "Approach Vector Doctrine",
                    ["multi_vector_360", "pincer_dual_axis", "concentrated_axis"],
                    key="cuas18_raid_pat",
                )
                r_evasion = st.selectbox(
                    "Drone Evasion Maneuver",
                    ["jinking_weave", "pop_up_terminal", "straight_dive"],
                    key="cuas18_raid_ev",
                )

            with rd3:
                r_ammo = st.selectbox(
                    "Turret Ammunition Load",
                    list(sg_presets_18.keys()),
                    format_func=lambda k: sg_presets_18[k]["name"],
                    index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0,
                    key="cuas18_raid_ammo",
                )
                r_mag = st.selectbox(
                    "Magazine / Feed System",
                    list(mag_cat_12.keys()),
                    format_func=lambda k: mag_cat_12[k]["name"],
                    index=1,
                    key="cuas18_raid_mag",
                )
                r_speed = st.slider("Swarm Speed (m/s)", 10.0, 42.0, 22.0, 1.0, key="cuas18_raid_spd")
                r_rmax = st.slider("Turret Engagement Range (m)", 35.0, 95.0, 65.0, 5.0, key="cuas18_raid_rmax")

            with st.spinner("Running 30-Trial Cooperative 3D Multi-Turret Raid Simulation..."):
                raid_res = _cached_unified_raid(
                    site_preset=r_site,
                    custom_turrets_json=r_tur_json,
                    custom_obstacles_json=r_obs_json,
                    custom_nfz_json=r_nfz_json,
                    terrain_dem_json=r_dem_json,
                    sensors_tuple=tuple(r_sensors or ["micro_doppler_radar"]),
                    weather_condition=str(r_wx),
                    n_drones=int(r_n_drones),
                    fiber_optic_fraction=float(r_fo_pct) / 100.0,
                    drone_model="fpv_quad_small",
                    approach_pattern=str(r_pattern),
                    evasion_profile=str(r_evasion),
                    maneuver_g=1.6,
                    threat_speed_mps=float(r_speed),
                    threat_altitude_m=22.0,
                    stagger_window_s=5.0,
                    ammunition_preset=str(r_ammo),
                    choke="Modified",
                    magazine_type=str(r_mag),
                    weapon_max_range_m=float(r_rmax),
                    n_mc_trials=30,
                    seed=42,
                )
            st.session_state["_report_raid_result"] = raid_res

            rsm = raid_res["summary"]
            rk1, rk2, rk3, rk4 = st.columns(4)
            rk1.metric("Zero-Breach Raid Stop", f"{rsm['zero_breach_pct']:.1f} %", delta=f"Mean Leakers: {rsm['mean_leakers']:.2f}/{rsm['n_drones']}")
            rk2.metric("Mean Swarm Kill Rate", f"{rsm['mean_kill_rate_pct']:.1f} %", delta=f"Mean Intercept: {rsm['mean_intercept_range_m']:.1f} m")
            rk3.metric("Cost per Neutralized Drone", f"${rsm['cost_per_kill_usd']:.2f}", delta=f"{rsm['mean_rounds_expended']:.1f} rds/raid • Handoffs: {rsm['handoff_count_mean']:.2f}")
            rk4.metric("Peak Barrel Temp", f"{rsm['peak_barrel_temp_c']:.1f} °C", delta=f"Reload Downtime: {rsm['reload_downtime_pct']:.1f}% • Terrain: {rsm['terrain_source'][:20]}")

            rt1, rt2, rt3 = st.tabs([
                "🎮 3D Animated Cooperative Raid Arena",
                "📊 Barrel Thermal & Attrition Telemetry",
                "📋 Turret Workload & Event Log",
            ])
            with rt1:
                st.plotly_chart(
                    create_unified_tactical_raid_3d_figure(raid_res, terrain_dem=r_dem_obj),
                    width="stretch",
                )
            with rt2:
                st.plotly_chart(create_raid_timeline_and_heat_figure(raid_res), width="stretch")
            with rt3:
                st.markdown("##### 🔫 Cooperative Turret Statistics (MC Means)")
                st.dataframe(pd.DataFrame(raid_res["turret_stats"]), width="stretch")
                st.markdown("##### 📝 Representative Trial Tactical Event Feed")
                st.dataframe(pd.DataFrame(raid_res["event_log"]), width="stretch", height=300)

        # --- TAB 18.13: FIELD-TEST CALIBRATION & EMPIRICAL RESIDUALS ---
        elif active_cuas_mod == _CUAS18_MODULES[12]:
            st.markdown("#### 🧪 Field-Test Calibration & Empirical vs. Model Residual Diagnostics")
            st.caption(
                "Upload range CSV measurements (or load realistic synthetic field logs) to fit choke spread multipliers, "
                "chronograph MV & FM 6-40 temperature sensitivity, and exterior ballistic drag form factors via SciPy nonlinear least squares."
            )
            cal_t1, cal_t2, cal_t3 = st.tabs([
                "🎯 Shotgun Patterning-Board Fit",
                "📏 Chronograph MV & Temp Lot Fit",
                "🚀 Exterior Drag Form-Factor (i) Fit",
            ])

            with cal_t1:
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    pat_rng = st.number_input("Patterning Range (m)", 10.0, 100.0, 36.58, 1.0, key="cal_pat_rng")
                with pc2:
                    pat_choke = st.selectbox("Nominal Choke", list(get_choke_definitions().keys()), index=3, key="cal_pat_choke")
                with pc3:
                    pat_ammo = st.selectbox("Reference Load", list(sg_presets_18.keys()), index=0, key="cal_pat_ammo")
                pat_up = st.file_uploader("Upload Pellet Impact CSV (`y_m,z_m` or `y_cm,z_cm` or `y_in,z_in`)", type=["csv"], key="cal_pat_up")
                if st.button("📋 Load Sample Patterning-Board CSV", key="cal_pat_sample_btn"):
                    st.session_state["_cal_pat_csv"] = generate_sample_pattern_csv(range_m=float(pat_rng), nominal_choke=str(pat_choke))
                pat_csv = pat_up.getvalue().decode("utf-8", errors="replace") if pat_up is not None else st.session_state.get("_cal_pat_csv") or generate_sample_pattern_csv(range_m=float(pat_rng), nominal_choke=str(pat_choke))
                try:
                    pat_res = calibrate_shotgun_pattern_from_csv(
                        pat_csv,
                        range_m=float(pat_rng),
                        nominal_choke=str(pat_choke),
                        ammunition_preset=str(pat_ammo),
                    )
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Fitted Spread Multiplier", f"{pat_res['calibrated_spread_factor']:.2f}×", delta=f"Closest: {pat_res['closest_standard_choke']}")
                    pm2.metric("POI Bias (Y / Z)", f"{pat_res['poi_bias_y_mrad']:+.1f} / {pat_res['poi_bias_z_mrad']:+.1f} mrad", delta=f"40-yd 30\" Eff: {pat_res['fitted_40yd_30in_pct']:.1f}%")
                    pm3.metric("Fitted R50 / R90", f"{pat_res['r50_fit_m']:.2f} / {pat_res['r90_fit_m']:.2f} m", delta=f"σ_fit={pat_res['sigma_fit_m']:.3f} m")
                    pm4.metric("Rayleigh Goodness-of-Fit", f"R²={pat_res['r2_cdf']:.3f}", delta=f"KS p={pat_res['ks_pvalue']:.3f} (RMSE={pat_res['rmse_cdf']:.3f})")
                    st.plotly_chart(create_pattern_calibration_figure(pat_res), width="stretch")
                    if st.button("✅ Apply Calibrated Choke & Aim Bias to C-UAS Workbench", type="primary", key="cal_pat_apply_btn"):
                        st.session_state["cuas_choke"] = pat_res["closest_standard_choke"]
                        st.session_state["_calibrated_aim_bias_mrad"] = round(
                            math.hypot(pat_res["poi_bias_y_mrad"], pat_res["poi_bias_z_mrad"]), 2
                        )
                        st.success(
                            f"Applied closest choke `{pat_res['closest_standard_choke']}` & radial bias "
                            f"{st.session_state['_calibrated_aim_bias_mrad']:.2f} mrad to session!"
                        )
                except Exception as exc:
                    st.error(f"Pattern calibration error: {exc}")

            with cal_t2:
                ch1, ch2 = st.columns(2)
                with ch1:
                    ch_nom = st.number_input("Nominal MV Reference (m/s)", 100.0, 1800.0, 415.0, 5.0, key="cal_ch_nom")
                with ch2:
                    ch_tstd = st.number_input("Standard Propellant Temp (°C)", 0.0, 35.0, 21.1, 0.5, key="cal_ch_tstd")
                ch_up = st.file_uploader("Upload Chronograph CSV (`velocity_mps` or `velocity_fps`, optional `temp_c`)", type=["csv"], key="cal_ch_up")
                if st.button("📋 Load Sample Multi-Temperature Chronograph CSV", key="cal_ch_sample_btn"):
                    st.session_state["_cal_ch_csv"] = generate_sample_chronograph_csv()
                ch_csv = ch_up.getvalue().decode("utf-8", errors="replace") if ch_up is not None else st.session_state.get("_cal_ch_csv") or generate_sample_chronograph_csv()
                try:
                    ch_res = calibrate_chronograph_from_csv(ch_csv, nominal_mv_mps=float(ch_nom), std_temp_c=float(ch_tstd))
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("Fitted Mean MV @ T_std", f"{ch_res['v_std_fit_mps']:.2f} m/s", delta=f"Bias: {ch_res['mv_bias_mps']:+.2f} m/s")
                    cm2.metric("Round-to-Round Residual SD", f"{ch_res['residual_sd_mps']:.2f} m/s", delta=f"Raw SD: {ch_res['sd_mps']:.2f} | ES: {ch_res['extreme_spread_mps']:.1f}")
                    cm3.metric("FM 6-40 dMV/dT Slope", f"{ch_res['dv_dt_mps_per_c']:+.3f} m/s/°C" if ch_res["has_temp_fit"] else "Single Temp", delta=f"R²={ch_res['r2_temp_fit']:.3f}" if ch_res["has_temp_fit"] else "Add temp_c column")
                    cm4.metric("Chrono Sample Size", f"{ch_res['n_shots']} shots", delta=f"{ch_res['min_mv_mps']:.1f}–{ch_res['max_mv_mps']:.1f} m/s")
                    st.plotly_chart(create_chronograph_calibration_figure(ch_res), width="stretch")
                    if st.button("✅ Apply Calibrated MV & SD to Monte Carlo Simulator", type="primary", key="cal_ch_apply_btn"):
                        st.session_state["_calibrated_mv_mps"] = float(ch_res["v_std_fit_mps"])
                        st.session_state["_calibrated_mv_sd_mps"] = float(ch_res["residual_sd_mps"])
                        if ch_res["has_temp_fit"]:
                            st.session_state["_calibrated_dv_dt"] = float(ch_res["dv_dt_mps_per_c"])
                        st.success(
                            f"Saved calibrated MV={ch_res['v_std_fit_mps']:.1f} m/s, SD={ch_res['residual_sd_mps']:.2f} m/s for Monte Carlo Simulator!"
                        )
                except Exception as exc:
                    st.error(f"Chronograph calibration error: {exc}")

            with cal_t3:
                all_pr = [p["preset_id"] for p in list_presets()]
                sa_def_idx = all_pr.index("NOTIONAL-SA-01") if "NOTIONAL-SA-01" in all_pr else 0
                dc1, dc2 = st.columns([1.4, 1.0])
                with dc1:
                    dg_preset = st.selectbox("Reference Projectile Preset", all_pr, index=sa_def_idx, key="cal_dg_preset")
                with dc2:
                    dg_fit_mv = st.checkbox("Jointly Fit MV Scale & Drag Form Factor", value=False, key="cal_dg_fit_mv")
                dg_up = st.file_uploader("Upload Range Drop / Radar CSV (`range_m,drop_m[,velocity_mps]`)", type=["csv"], key="cal_dg_up")
                if st.button("📋 Load Sample Range Drop & Velocity CSV", key="cal_dg_sample_btn"):
                    st.session_state["_cal_dg_csv"] = generate_sample_range_drop_csv(preset_id=str(dg_preset))
                    st.session_state["_cal_dg_preset_seeded"] = str(dg_preset)
                if dg_up is not None:
                    dg_csv = dg_up.getvalue().decode("utf-8", errors="replace")
                else:
                    if st.session_state.get("_cal_dg_preset_seeded") != str(dg_preset) or "_cal_dg_csv" not in st.session_state:
                        st.session_state["_cal_dg_csv"] = generate_sample_range_drop_csv(preset_id=str(dg_preset))
                        st.session_state["_cal_dg_preset_seeded"] = str(dg_preset)
                    dg_csv = st.session_state["_cal_dg_csv"]
                try:
                    dg_res = calibrate_drag_form_factor_from_csv(dg_csv, preset_id=str(dg_preset), fit_mv=bool(dg_fit_mv))
                    dm1, dm2, dm3, dm4 = st.columns(4)
                    dm1.metric("Calibrated Form Factor (i)", f"{dg_res['calibrated_form_factor']:.4f}", delta=f"Nominal: {dg_res['nominal_form_factor']:.3f} ({dg_res['calibrated_form_factor_multiplier']:.3f}×)")
                    dm2.metric("Drop RMSE (Before -> After)", f"{(dg_res['rmse_drop_after_m'] or 0.0):.4f} m", delta=f"Before: {(dg_res['rmse_drop_before_m'] or 0.0):.4f} m")
                    dm3.metric("Velocity RMSE (After)", f"{(dg_res['rmse_vel_after_mps'] or 0.0):.2f} m/s" if dg_res["has_vel"] else "—", delta=f"Before: {(dg_res['rmse_vel_before_mps'] or 0.0):.2f} m/s" if dg_res["has_vel"] else "Drop-only fit")
                    dm4.metric("Fit R²", f"{dg_res['r2_fit']:.5f}", delta=f"{dg_res['n_stations']} stations ({dg_res['drag_ref']})")
                    st.plotly_chart(create_drag_calibration_figure(dg_res), width="stretch")
                    st.dataframe(pd.DataFrame(dg_res["stations"]), width="stretch")
                    if st.button("✅ Apply Calibrated Form Factor (i) to Monte Carlo Simulator", type="primary", key="cal_dg_apply_btn"):
                        st.session_state["_calibrated_form_factor"] = float(dg_res["calibrated_form_factor"])
                        st.session_state["_calibrated_preset_id"] = str(dg_preset)
                        if dg_fit_mv:
                            st.session_state["_calibrated_mv_mps"] = float(dg_res["calibrated_mv_mps"])
                        st.success(f"Applied calibrated form factor i={dg_res['calibrated_form_factor']:.4f} for `{dg_preset}`!")
                except Exception as exc:
                    st.error(f"Exterior drag calibration error: {exc}")
