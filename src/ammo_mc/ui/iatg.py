"""UNODA IATG 01.80 / 02.20 Safety UI workbench."""


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
    is_expert_mode,
)



def tab_iatg() -> None:
    st.header("🏛️ UNODA International Ammunition Technical Guidelines (IATG 01.80:2021 & IATG 02.20)")
    st.markdown(
        "Comprehensive engineering workbench grounded in **UNODA IATG 01.80:2021 (3rd Edition, *Formulae for ammunition management*)**, "
        "**IATG 02.20 (*Quantity and separation distances*)**, and **NATO AASTP-1**: "
        "Quantity-Distance (QD) scaling laws, open-detonation & vertical airspace safety (`TB-15`), "
        "UK brick-house structural collapse (`TB-10`), Swiss glazing $P_s\\text{–}I_s$ hyperbolas (`TB-11`), "
        "munition stack fragmentation (`TF-10`), RC donor debris launch (`TF-14`), and Axelsson thoracic biomechanics (`TB-13`)."
    )
    kb = get_kb()

    with st.sidebar:
        _render_sidebar_header("🏛️ UNODA IATG Safety", "IATG 01.80:2021, IATG 02.20 & NATO AASTP-1")
        active_iatg_tool = st.radio("Select IATG Safety Tool (1–7)", _IATG_TOOLS, key="iatg_active_tool")
        st.divider()
        st.subheader("💣 Explosive Source Baseline")
        neq_kg = st.number_input("Net Explosive Quantity Q_NEQ (kg)", 1.0, 100000.0, 1000.0, 50.0, key="iatg_neq")
        auw_kg = st.number_input("All-Up Munition Weight M_AUW (kg)", 1.0, 250000.0, 2500.0, 100.0, key="iatg_auw")
        exp_name = st.selectbox(
            "Energetic Material (TB-02 TNT Equivalence)",
            ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20", "RDX", "HMX"],
            index=0,
            key="iatg_exp",
        )
        if active_iatg_tool == _IATG_TOOLS[0]:
            show_3d_domes = st.checkbox("Render 3D Hemispherical QD Domes", value=True, key="iatg_domes")
        else:
            show_3d_domes = True
        with st.expander("📘 IATG QD Z-Factor Legend", expanded=False):
            st.markdown(
                "- **PBD**: `8.0 m/kg^(1/3)` (Process Building)\n"
                "- **PTRD**: `14.8 m/kg^(1/3)` (Public Traffic Route)\n"
                "- **IBD**: `22.2 m/kg^(1/3)` (Inhabited Building)\n"
                "- **VBD**: `44.4 m/kg^(1/3)` (Vulnerable Building)\n"
                "- **TB-15 Public Frag**: `634 · M_AUW^(1/6)`"
            )

    wp_eq, wi_eq = tb_02_tnt_equivalence(neq_kg, explosive=exp_name)
    q_tnt = float(wp_eq)
    qd_res = tb_15_simple_safety_distances(auw_kg=auw_kg, neq_kg=q_tnt)

    # --- IATG SUB-TAB 1: QUANTITY-DISTANCE (QD), TB-15 & 3D PES SITE MAP ---
    if active_iatg_tool == _IATG_TOOLS[0]:
        st.subheader("🏗️ IATG 01.80 / 02.20 Quantity-Distance (QD) & Open-Detonation Safety Envelopes (`TB-02`, `TB-15`)")
        with st.expander("Governing Equations (IATG 01.80 / 02.20)", expanded=is_expert_mode()):
            st.latex(
                r"""
                D_{\text{QD}} = Z_{\text{QD}}\, Q_{\text{TNT}}^{1/3}, \quad
                Z_{\text{PBD}} = 8.0, \quad Z_{\text{PTRD}} = 14.8, \quad Z_{\text{IBD}} = 22.2, \quad Z_{\text{VBD}} = 44.4\;\text{m/kg}^{1/3}, \quad
                D_{\text{frag,pub}} = 634\,M_{\text{AUW}}^{1/6}, \quad D_{\text{vert,1}} = 314\,M_{\text{AUW}}^{1/3}, \quad D_{140\text{dB}} = 215\,Q_{\text{TNT}}^{1/3}
                """
            )

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
            st.plotly_chart(fig_pes, width="stretch")

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
            st.plotly_chart(fig_qd_scale, width="stretch")

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
        st.dataframe(pd.DataFrame(qd_rows), width="stretch")

    # --- IATG SUB-TAB 2: STRUCTURAL DAMAGE (TB-10) & SWISS GLAZING (TB-11) ---
    elif active_iatg_tool == _IATG_TOOLS[1]:
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
            st.plotly_chart(fig_tb10, width="stretch")
            st.dataframe(tb10_df, width="stretch")

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
            st.plotly_chart(fig_pi, width="stretch")

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
    elif active_iatg_tool == _IATG_TOOLS[2]:
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
            st.plotly_chart(fig_tf10, width="stretch")

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
    elif active_iatg_tool == _IATG_TOOLS[3]:
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
            ak3.metric("Lung Lethality Probit (CPR 16E / TB-12)", f"{float(lung_out['probability'])*100:.1f} %", delta=f"Pr: {float(lung_out['probit']):.2f}", delta_color="off")

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
                st.plotly_chart(fig_ax, width="stretch")

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
            st.plotly_chart(fig_th, width="stretch")

            st.markdown("##### 🌍 IATG 01.80 Ground Shock Peak Particle Velocity (PPV) Structural Thresholds")
            st.dataframe(pd.DataFrame(kb.effects_criteria.get("ground_shock", [])), width="stretch")

    # --- IATG SUB-TAB 5: MULTI-FACILITY 3D SITE PLANNER & SYMPATHETIC MATRIX ---
    elif active_iatg_tool == _IATG_TOOLS[4]:
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
        st.plotly_chart(fig_site, width="stretch")

        comp_df = pd.DataFrame(comp_rows)
        n_viol = sum(1 for r in comp_rows if "VIOLATION" in str(r.get("QD Status", "")))
        if n_viol > 0:
            st.error(f"🚨 **{n_viol} Quantity-Distance / Sympathetic Detonation Encroachment(s) Detected!** Adjust facility coordinates or NEQ limits above.")
        else:
            st.success("✅ **All Inter-Magazine (IMD) and Exposed Site (ES) Separation Distances Comply with UNODA IATG 02.20!**")
        st.dataframe(comp_df, width="stretch")

    # --- IATG SUB-TAB 6: IN-SERVICE SURVEILLANCE & PROPELLANT AGING (IATG 07.10 / AOP-48) ---
    elif active_iatg_tool == _IATG_TOOLS[5]:
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
            st.plotly_chart(fig_surv, width="stretch")

        with surv_p2:
            oc = plan_res["oc_curve"]
            fig_oc = go.Figure()
            fig_oc.add_trace(go.Scatter(x=oc["p_defect_pct"], y=np.asarray(oc["p_accept"])*100.0, mode="lines", name="Operating Characteristic (OC)", line=dict(color="#10B981", width=3)))
            fig_oc.add_vline(x=surv_aql, line_dash="dash", line_color="#DC2626", annotation_text=f"AQL = {surv_aql}%")
            fig_oc.update_layout(title=f"ISO 2859-1 Surveillance OC Curve (Lot N={surv_lot_n}, n={plan_res['sample_size_n']}, Ac={plan_res['acceptance_number_ac']})", xaxis_title="True Defect Percentage p (%)", yaxis_title="Probability of Lot Acceptance P_acc (%)", height=390)
            st.plotly_chart(fig_oc, width="stretch")

    # --- IATG SUB-TAB 7: GEOSPATIAL TERRAIN CREST CLEARANCE & 3D KML/CZML EXPORT ---
    elif active_iatg_tool == _IATG_TOOLS[6]:
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
        st.plotly_chart(fig_terr_3d, width="stretch")

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
                width="stretch",
            )
        with czml_c:
            st.download_button(
                label="🛰️ Download Cesium 3D Stream (.czml)",
                data=czml_str.encode("utf-8"),
                file_name=f"trajectory_stream_{int(geo_range)}m.czml",
                mime="application/json",
                width="stretch",
            )
