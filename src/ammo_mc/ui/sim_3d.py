"""Interactive 3D Ballistics Simulations UI workbench."""


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

    _3d_modes = [
        "🎯 3D Flight Trajectory Bundle & Terrain",
        "🎬 Animated 3D Salvo Flight Replay",
        "🌀 3D Phase-Space & Thermodynamic Manifold",
        "💥 3D Pressure Wave & Fragment Kinematics",
        "🏛️ 3D IATG Explosive Safety & QD Domes",
        "⚔️ Multi-Preset 3D Comparison Arena",
    ]
    all_presets_3d = list_presets()
    p3d_map = {f"{p['preset_id']} — {p['ammo_name']}": p["preset_id"] for p in all_presets_3d}
    p3d_keys = list(p3d_map.keys())
    cur_p3d_idx = next((i for i, k in enumerate(p3d_keys) if p3d_map[k] == cur_ammo.id), 0)

    expert = is_expert_mode()
    with st.sidebar:
        _render_sidebar_header("🌐 3D Simulations", "Interactive WebGL trajectories, terrain & terminal scenes")
        view_mode = st.radio("3D Simulation Mode", _3d_modes, key="sim3d_view_mode")
        with st.expander("🎯 Active 3D Baseline Munition", expanded=expert):
            st.caption(f"**Loaded:** `{cur_ammo.id}` — {cur_ammo.name}")
            sel_3d_label = st.selectbox("Switch Baseline Preset", p3d_keys, index=cur_p3d_idx, key="sim3d_preset_sel")
            if st.button("🔄 Simulate & Load in 3D", width="stretch", key="sim3d_reload_btn"):
                new_pid = p3d_map[sel_3d_label]
                new_ammo, new_scn = get_preset(new_pid)
                with st.spinner(f"Simulating {new_pid} for 3D scene..."):
                    new_res = run_simulation(new_ammo, new_scn)
                    st.session_state["last_sim_result"] = new_res
                    st.session_state["last_sim_ammo"] = new_ammo
                    st.session_state["last_sim_scn"] = new_scn
                    st.session_state["last_preset_id"] = new_pid
                res = new_res
                cur_ammo = new_ammo
                cur_scn = new_scn

        st.divider()
        st.subheader("🎛️ Scene Controls")
        if view_mode == "🎯 3D Flight Trajectory Bundle & Terrain":
            color_opt = st.selectbox(
                "Color Trajectories By",
                ["Speed (m/s)", "Mach Number (-)", "Altitude (m)", "Trajectory Index"],
                index=0,
            )
            n_paths = st.slider("3D Trajectories in Bundle", 1, 20, 12, 1)
            with st.expander("Camera, Terrain Style & Display Layers", expanded=expert):
                terr_style_label = st.selectbox(
                    "3D Terrain DEM Style",
                    ["Scenario Default", "Rolling Hills DEM", "Canyon Valley", "Transverse Ridge"],
                    index=0,
                )
                cam_preset = st.selectbox(
                    "Camera Perspective", ["Perspective Overview", "Top-Down Map", "Side Profile", "Launcher View"]
                )
                show_terr = st.checkbox("Show 3D Terrain Surface", value=True)
                show_tgt = st.checkbox("Show 3D Target Volume", value=True)
                show_poi = st.checkbox("Show Ground Impact POI", value=True)
                show_cep = st.checkbox("Show 3D CEP50 & CEP95 Rings", value=True)
        elif view_mode == "🎬 Animated 3D Salvo Flight Replay":
            anim_paths = st.slider("Animated Trajectories", 2, 12, 8, 1)
            anim_frames = st.slider("Time Resolution (Keyframes)", 12, 40, 24, 2)
        elif view_mode == "🌀 3D Phase-Space & Thermodynamic Manifold":
            ps_color_label = st.selectbox(
                "Color State Trajectories By Physical Field",
                [
                    "Dynamic Pressure q = ½ρv² (kPa)",
                    "Reynolds Number Re_d = ρvL/μ(T) (×10⁶)",
                    "Specific Mechanical Energy e = ½v² + gz (kJ/kg)",
                ],
                index=0,
            )
            ps_paths = st.slider("Trajectories in Phase-Space Bundle", 2, 16, 10, 1)
        elif view_mode == "💥 3D Pressure Wave & Fragment Kinematics":
            def_charge = float(cur_ammo.warhead.fill.mass_kg if cur_ammo.warhead else 10.0)
            def_hob = float(cur_ammo.fuze.hob.mu if cur_ammo.fuze else 0.0)
            charge_kg = st.number_input(
                "Energetic Fill Mass (kg)", min_value=0.1, max_value=2000.0, value=def_charge, step=1.0
            )
            exp_type = st.selectbox(
                "Energetic Material",
                ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20"],
                index=0,
            )
            burst_z = st.slider("Burst Height Above Ground (m)", 0.0, 50.0, def_hob, 1.0)
            n_frags = st.slider("3D Fragment Trajectory Rays", 30, 300, 120, 10)
        elif view_mode == "🏛️ 3D IATG Explosive Safety & QD Domes":
            qd_neq = st.number_input("Net Explosive Quantity NEQ (kg)", 1.0, 100000.0, 1000.0, 50.0, key="3d_qd_neq")
            qd_auw = st.number_input("All-Up Munition Weight AUW (kg)", 1.0, 250000.0, 2500.0, 100.0, key="3d_qd_auw")
            qd_exp = st.selectbox(
                "Energetic Material",
                ["Composition B", "TNT", "Composition C4", "Octol 75/25", "H-6", "Tritonal 80/20"],
                index=0,
                key="3d_qd_exp",
            )
            qd_domes = st.checkbox("Show 3D Translucent Hemispherical Safety Domes", value=True, key="3d_qd_domes")
        elif view_mode == "⚔️ Multi-Preset 3D Comparison Arena":
            all_p = [p["preset_id"] for p in all_presets_3d]
            sel_comp = st.multiselect(
                "Select Presets to Compare in 3D Space (2 to 4 recommended)",
                all_p,
                default=["NOTIONAL-AR-01", "NOTIONAL-AR-02", "NOTIONAL-MO-01"],
                key="3d_comp_presets",
            )

    if view_mode == "🎯 3D Flight Trajectory Bundle & Terrain":
        st.subheader("🎯 3D Flight Dynamics, CEP Ellipses & Terrain Intersection")
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
        st.plotly_chart(fig_3d_traj, width="stretch")

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
        fig_anim = create_3d_animated_flight_figure(
            result=res,
            scenario=cur_scn,
            max_trajectories=anim_paths,
            n_frames=anim_frames,
        )
        fig_anim.update_layout(height=660)
        st.plotly_chart(fig_anim, width="stretch")

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
        st.plotly_chart(fig_ps, width="stretch")

    elif view_mode == "💥 3D Pressure Wave & Fragment Kinematics":
        st.subheader("💥 3D Overpressure Iso-Surfaces & Fragmentation Trajectories")
        fig_3d_blast = create_3d_blast_and_frag_figure(
            result=res,
            scenario=cur_scn,
            burst_alt_m=float(burst_z),
            n_frag_traces=int(n_frags),
            charge_mass_kg=float(charge_kg),
            explosive_name=exp_type,
        )
        fig_3d_blast.update_layout(height=650)
        st.plotly_chart(fig_3d_blast, width="stretch")

        html_3d_blast = fig_3d_blast.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "💾 Download Interactive 3D Pressure/Fragment Model (.html)",
            html_3d_blast.encode("utf-8"),
            f"{cur_ammo.id}_3d_terminal.html",
            "text/html",
        )

    elif view_mode == "🏛️ 3D IATG Explosive Safety & QD Domes":
        st.subheader("🏛️ 3D UNODA IATG 01.80 / 02.20 Potential Explosion Site (PES) & Quantity-Distance Domes")
        fig_3d_qd = create_3d_iatg_safety_zones_figure(
            neq_kg=float(qd_neq),
            auw_kg=float(qd_auw),
            explosive_name=qd_exp,
            show_domes=bool(qd_domes),
        )
        fig_3d_qd.update_layout(height=650)
        st.plotly_chart(fig_3d_qd, width="stretch")

    elif view_mode == "⚔️ Multi-Preset 3D Comparison Arena":
        st.subheader("⚔️ Multi-Preset 3D Trajectory & Dispersion Comparison")
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
            st.plotly_chart(fig_comp, width="stretch")
            st.dataframe(pd.DataFrame(comp_rows), width="stretch")
