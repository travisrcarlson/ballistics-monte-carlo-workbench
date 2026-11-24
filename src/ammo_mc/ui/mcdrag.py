"""McCoy MCDRAG & 3D Projectile CAD UI workbench."""


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

_MCDRAG_ARCHETYPES: dict[str, dict[str, Any]] = {
    "155mm M795 Artillery": {"cal": 155.0, "tot": 5.2, "nose": 2.8, "bt": 0.60, "base": 0.84, "meplat": 0.12, "band": 1.02, "rt": 0.65, "bl": "L/T"},
    "120mm Mortar/Shell": {"cal": 120.0, "tot": 4.4, "nose": 2.1, "bt": 0.70, "base": 0.78, "meplat": 0.16, "band": 1.00, "rt": 0.80, "bl": "L/T"},
    "30mm Autocannon": {"cal": 30.0, "tot": 4.8, "nose": 2.6, "bt": 0.45, "base": 0.86, "meplat": 0.10, "band": 1.02, "rt": 0.70, "bl": "T/T"},
    "5.56mm NATO Bullet": {"cal": 5.56, "tot": 4.1, "nose": 2.2, "bt": 0.55, "base": 0.82, "meplat": 0.08, "band": 1.00, "rt": 0.90, "bl": "L/L"},
}


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
    is_expert_mode,
)



def _apply_mcdrag_archetype() -> None:
    arch = _MCDRAG_ARCHETYPES[st.session_state["mcdrag_archetype_sel"]]
    st.session_state["mcdrag_cal"] = float(arch["cal"])
    st.session_state["mcdrag_tot"] = float(arch["tot"])
    st.session_state["mcdrag_nose"] = float(arch["nose"])
    st.session_state["mcdrag_bt"] = float(arch["bt"])
    st.session_state["mcdrag_base"] = float(arch["base"])
    st.session_state["mcdrag_meplat"] = float(arch["meplat"])
    st.session_state["mcdrag_band"] = float(arch["band"])
    st.session_state["mcdrag_rt"] = float(arch["rt"])
    st.session_state["mcdrag_bl"] = str(arch["bl"])

def tab_mcdrag() -> None:
    st.header("🚀 McCoy (1974) MCDRAG Aerodynamics & 3D Projectile CAD")
    st.markdown(
        "Interactive geometric designer coupled to Robert L. McCoy's BRL 1974 `MCDRAG` algorithm (`S15`). "
        "Predicts zero-yaw drag $C_{D0}(M)$ and its wave/viscous components (head, skin friction, base, boattail, rotating band) "
        "across Mach 0.5 to 5.0 alongside a real-time 3D axisymmetric projectile mesh."
    )

    expert = is_expert_mode()
    with st.sidebar:
        _render_sidebar_header("🚀 MCDRAG & 3D CAD", "McCoy (1974) axisymmetric drag & 3D shell designer")
        st.selectbox(
            "Projectile Archetype Preset",
            list(_MCDRAG_ARCHETYPES.keys()),
            index=0,
            key="mcdrag_archetype_sel",
            on_change=_apply_mcdrag_archetype,
        )
        st.divider()
        st.subheader("📐 Geometry & Boundary Layer")
        cal_mm = st.number_input("Reference Diameter / Calibre (mm)", min_value=5.56, max_value=400.0, value=155.0, step=5.0, key="mcdrag_cal")
        tot_len_cal = st.slider("Total Length (calibres)", 2.0, 10.0, 5.2, 0.1, key="mcdrag_tot")
        nose_len_cal = st.slider("Nose / Ogive Length (calibres)", 1.0, 6.0, 2.8, 0.1, key="mcdrag_nose")
        with st.expander("Advanced Ogive, Boattail & Boundary Layer", expanded=expert):
            bt_len_cal = st.slider("Boattail Length (calibres)", 0.0, 2.0, 0.6, 0.05, key="mcdrag_bt")
            base_dia_cal = st.slider("Base Diameter (calibres)", 0.5, 1.0, 0.84, 0.02, key="mcdrag_base")
            meplat_dia_cal = st.slider("Meplat Flat Diameter (calibres)", 0.0, 0.5, 0.12, 0.02, key="mcdrag_meplat")
            band_dia_cal = st.slider("Rotating Band Diameter (calibres)", 1.0, 1.1, 1.02, 0.01, key="mcdrag_band")
            rt_r = st.slider("Ogive Tangency Ratio R_t/R (0=cone, 1=tangent)", 0.0, 1.0, 0.65, 0.05, key="mcdrag_rt")
            bl_code = st.selectbox("Boundary Layer Condition", ["L/T", "T/T", "L/L"], index=0, key="mcdrag_bl")

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
        st.plotly_chart(fig_cad, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

    if res.get("diagnostics"):
        st.warning("MCDRAG Geometric Advisories: " + " | ".join(res["diagnostics"]))

    act_c1, act_c2 = st.columns([1, 2])
    with act_c1:
        if st.button("⚡ Apply Custom MCDRAG Curve to Active Simulation", width="stretch"):
            cur_ammo_obj = st.session_state.get("last_sim_ammo")
            if cur_ammo_obj is None:
                cur_ammo_obj, _ = get_preset("NOTIONAL-AR-01")
                st.session_state["last_sim_ammo"] = cur_ammo_obj
            st.session_state["custom_mcdrag_ammo_id"] = cur_ammo_obj.id
            st.session_state["custom_mcdrag_table"] = [
                {"mach": float(r["mach"]), "cd": float(r["cd0"])} for r in res["table"]
            ]
            st.success(
                f"Injected custom 26-point MCDRAG Cd(M) table into {cur_ammo_obj.id}! Switch to '🎯 Monte Carlo Simulator' and click 'Run Monte Carlo Simulation' to fly with this drag curve."
            )
    with act_c2:
        Peak_Cd = float(table_df["cd0"].max())
        Peak_M = float(table_df.loc[table_df["cd0"].idxmax(), "mach"])
        Supersonic_Cd = float(table_df.loc[table_df["mach"] == 2.0, "cd0"].values[0])
        st.info(
            f"**Transonic Peak Drag:** $C_{{D0}} = {Peak_Cd:.3f}$ at Mach {Peak_M:.2f}  |  "
            f"**Supersonic Cruise (Mach 2.0):** $C_{{D0}} = {Supersonic_Cd:.3f}$"
        )

    st.dataframe(table_df, width="stretch")
