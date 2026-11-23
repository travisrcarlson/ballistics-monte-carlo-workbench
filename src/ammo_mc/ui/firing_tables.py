"""Firing Tables & STANAG 4082 METCM UI workbench."""


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

    with st.sidebar:
        _render_sidebar_header("📋 Firing Tables & Met", "FM 6-40 / STANAG 4119 tables & STANAG 4082 METCM")
        with st.expander("📋 Firing Table Sweep", expanded=True):
            ft_label = st.selectbox("Select Weapon / Rocket / Munition System", list(ft_dict.keys()), index=4, key="ft_sys_sel")
            ft_pid = ft_dict[ft_label]
            ft_ammo, _ = get_preset(ft_pid)
            qe_min_ft = st.number_input("Minimum Elevation QE (deg)", 2.0, 45.0, 5.0, 1.0, key="ft_qe_min")
            qe_max_ft = st.number_input("Maximum Elevation QE (deg)", 10.0, 75.0, 65.0, 5.0, key="ft_qe_max")
            n_steps_ft = st.slider("Elevation Increments", 5, 20, 13, 1, key="ft_n_steps")

        with st.expander("🌐 STANAG 4082 METCM Sounding", expanded=False):
            st.markdown("**Surface & Boundary Layer**")
            s_temp_c = st.slider("Surface Temperature (°C)", -30.0, 50.0, 15.0, 1.0, key="met_s_temp")
            s_press_hpa = st.number_input("Surface Pressure (hPa)", 800.0, 1080.0, 1013.25, 5.0, key="met_s_press")
            s_wind_spd = st.slider("Surface Wind Speed (m/s)", 0.0, 30.0, 6.0, 0.5, key="met_s_wspd")
            s_wind_dir = st.slider("Surface Wind Direction (° from North)", 0.0, 360.0, 240.0, 5.0, key="met_s_wdir")
            pbl_alpha = st.slider("PBL Shear Exponent α", 0.10, 0.45, 0.28, 0.02, key="met_pbl_alpha")
            ekman_veer = st.slider("Ekman Spiral Veering (°)", 0.0, 60.0, 30.0, 5.0, key="met_ekman_veer")
            st.markdown("**Jet Stream Core & Inversion**")
            jet_peak = st.slider("Jet Stream Core Peak (m/s)", 0.0, 80.0, 42.0, 2.0, key="met_jet_peak")
            jet_alt = st.slider("Jet Stream Altitude (m)", 6000.0, 15000.0, 10500.0, 500.0, key="met_jet_alt")
            enable_inversion = st.checkbox("Thermal Inversion Layer", value=True, key="met_inv_on")
            inv_base = st.number_input("Inversion Base (m)", 200.0, 3000.0, 800.0, 100.0, key="met_inv_base")
            inv_top = st.number_input("Inversion Top (m)", 300.0, 4000.0, 1400.0, 100.0, key="met_inv_top")
            inv_dt = st.slider("Inversion ΔT (K)", 1.0, 15.0, 5.0, 0.5, key="met_inv_dt")
            st.markdown("**Trajectory, Aero & Mass Properties**")
            m_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 200.0, 1200.0, float(ft_ammo.launch.mv.mu), 10.0, key="met_v0")
            m_qe = st.slider("Firing Elevation QE (°)", 5.0, 75.0, 38.0, 1.0, key="met_qe")
            m_az = st.slider("Azimuth of Fire (° from North)", 0.0, 360.0, 90.0, 5.0, key="met_az")
            m_dt = st.select_slider("Integration Time Step dt (s)", options=[0.02, 0.04, 0.08, 0.10], value=0.08, key="met_dt")
            m_cal = st.number_input("Caliber (m)", 0.020, 0.500, float(ft_ammo.calibre_m), 0.005, format="%.3f", key="met_cal")
            m_mass = st.number_input("Projectile Mass (kg)", 0.5, 200.0, float(ft_ammo.mass_kg), 1.0, key="met_mass")
            st.info(f"Using aerodynamic reference drag curve for `{ft_ammo.aero.drag_ref or 'G7'}`.")

    qe_max_clamped = max(float(qe_max_ft), float(qe_min_ft) + 1.0)
    with st.spinner("Computing Ballistic Firing Table & Finite-Difference Unit-Effect Sensitivities..."):
        ft_rows = generate_ballistic_firing_table(
            v0_mps=max(float(ft_ammo.launch.mv.mu), 40.0),
            calibre_m=float(ft_ammo.calibre_m),
            mass_kg=float(ft_ammo.mass_kg),
            twist_cal_per_turn=float(ft_ammo.launch.twist_cal or 20.0),
            form_factor=float(ft_ammo.aero.form_factor.mu),
            drag_ref=str(ft_ammo.aero.drag_ref if ft_ammo.aero.drag_ref in ("G1", "G7") else "G7"),
            qe_min_deg=float(qe_min_ft),
            qe_max_deg=qe_max_clamped,
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
        st.plotly_chart(fig_ft_qe, width="stretch")

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
        st.plotly_chart(fig_ft_met, width="stretch")

    st.dataframe(ft_df, width="stretch")
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
        "compared to standard International Standard Atmosphere (ISA) baseline. Configure sounding inputs in the left sidebar."
    )

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
    st.plotly_chart(fig_shear_3d, width="stretch")

    st.markdown("##### 📊 NATO STANAG 4082 Standard 20-Zone Sounding Profile (METCM Format)")
    zones_df = pd.DataFrame(sounding_prof["zones"])
    st.dataframe(zones_df, width="stretch")
    csv_zones = zones_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download STANAG 4082 METCM Sounding Dataset (.csv)",
        csv_zones,
        "stanag_4082_metcm_sounding.csv",
        "text/csv",
    )
