"""Stage 1: Small Arms & Anti-Drone Multi-Shotgun Array Workbench UI."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ammo_mc.briefing_export import generate_cuas_array_spec_sheet_pdf
from ammo_mc.cuas_mission import (
    generate_cuas_array_spec_sheet_html,
    serialize_cuas_scenario_config,
)
from ammo_mc.small_arms import (
    PELLET_MATERIAL_DENSITIES_G_CM3,
    build_custom_drone_spec,
    build_custom_shell_spec,
    compare_barrel_lengths_dispersion,
    evaluate_multibarrel_dispersion_vs_distance,
    evaluate_shotgun_dispersion_vs_distance,
    evaluate_stl_pellet_impacts_batch,
    generate_sample_drone_stl_bytes,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
    get_multibarrel_presets,
    get_pellet_specifications,
    get_shotgun_presets,
    get_standard_barrel_lengths,
    parse_stl_mesh,
    project_stl_mesh_vulnerability,
    simulate_multibarrel_burst_mc,
    simulate_shotgun_pattern_mc,
)
from ammo_mc.ui.common import (
    _cached_antidrone_array_mc,
    _cached_benchmark_drone_aspect,
    _cached_benchmark_multibarrel_aspect,
    _cached_compare_anti_drone_layouts,
    _cached_intercept_replay,
    _navigate_to,
    get_active_calibration_profile,
    get_active_physics_fidelity,
)
from ammo_mc.vis_3d import (
    create_2d_antidrone_intercept_figure,
    create_2d_barrel_length_comparison_figure,
    create_2d_multi_distance_shotgun_comparison_figure,
    create_2d_multibarrel_target_figure,
    create_2d_target_pattern_figure,
    create_3d_antidrone_defense_figure,
    create_3d_multi_distance_multibarrel_figure,
    create_3d_multi_distance_shotgun_figure,
    create_3d_multibarrel_dispersion_figure,
    create_3d_pattern_vs_distance_figure,
    create_3d_shotgun_plume_figure,
    create_antidrone_layout_comparison_figure,
    create_barrel_length_diagnostic_curves,
    create_drone_aspect_radar_figure,
    create_multibarrel_aspect_radar_figure,
    create_range_curves_multibarrel_figure,
    create_range_curves_shotgun_figure,
)
from ammo_mc.vis_cuas_mission import create_3d_animated_intercept_figure
from ammo_mc.vis_cuas_vv import create_stl_vulnerability_figure
from ammo_mc.ui.cuas_pkg.helpers import (
    _apply_quincunx_mixed_preset,
    _render_effective_range_tab,
    _render_stage1_hardware_cad_tab,
    render_high_fidelity_active_banner,
)


def render_stage1_small_arms(arm_mode: str, expert: bool) -> None:
    if expert:
        with st.expander("Governing Pattern & Stringing Equations", expanded=False):
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
        st.subheader("Multi-Shotgun Anti-Drone Defense Array")
        render_high_fidelity_active_banner()
        st.caption(
            "Configure the multi-barrel geometry, ammunition load, and drone kinematics. "
            "Toggle Expert Mode or open Advanced Options for pointing modes, ripple firing schedules, 3D Euler angles, and crosswind."
        )

        sg_presets = get_shotgun_presets()
        choke_defs = get_choke_definitions()
        drone_specs = get_drone_target_specs()
        aspect_presets = get_drone_aspect_presets()

        st.button(
            "⚡ Load 5-Barrel Quincunx Mixed Spec Preset (4× ALDA 6.2mm Copper + 1× SPARTAN 4T Tungsten vs 14\" PG-7M FPV)",
            on_click=_apply_quincunx_mixed_preset,
            key="cuas_load_quincunx_btn",
            width="stretch",
        )

        col_arr, col_tgt = st.columns(2)
        with col_arr:
            st.markdown("**1. Shotgun Array & Ammunition**")
            cuas_layout = st.selectbox(
                "Array Mount Geometry",
                ["circular", "single", "square", "quincunx", "grid", "concentric", "honeycomb", "fan_horizontal", "fan_vertical", "v_shape", "custom"],
                format_func=lambda k: {
                    "single": "1-Barrel Single Shotgun (0, 0)",
                    "square": "4-Barrel Square Pattern (20–40 cm)",
                    "quincunx": "5-Barrel Quincunx (Center + 4 Corners, 20–40 cm)",
                    "circular": "Circular Ring Array",
                    "grid": "Rectangular Matrix Grid (Rows x Cols)",
                    "concentric": "Concentric Dual-Ring (Core + Perimeter)",
                    "honeycomb": "Hexagonal Honeycomb Cluster",
                    "fan_horizontal": "Horizontal Linear Fan (Azimuth Curtain)",
                    "fan_vertical": "Vertical Stack Fan (Elevation Net)",
                    "v_shape": "Chevron V-Shape Array",
                    "custom": "Custom User-Defined Coordinates",
                }.get(k, k),
                index=0,
                key="cuas_layout",
            )
            cuas_sq_cm = 30.0
            cuas_qn_cm = 30.0
            if cuas_layout == "single":
                cuas_n_barrels = 1
                st.caption("Single 12-gauge barrel centered at (0, 0).")
            elif cuas_layout == "square":
                cuas_n_barrels = 4
                cuas_sq_cm = st.slider("Square Barrel Spacing (cm)", 20.0, 40.0, 30.0, 2.0, key="cuas_sq_sp_cm")
            elif cuas_layout == "quincunx":
                cuas_n_barrels = 5
                cuas_qn_cm = st.slider("Quincunx Center-to-Corner Distance (cm)", 20.0, 40.0, 30.0, 2.0, key="cuas_qn_sp_cm")
            elif cuas_layout in ["circular", "concentric", "honeycomb"]:
                cuas_n_barrels = st.slider(
                    "Number of Barrels (N)",
                    3,
                    16,
                    6 if cuas_layout != "honeycomb" else 7,
                    1,
                    key="cuas_n_barrels",
                )
            elif cuas_layout == "grid":
                gc1, gc2 = st.columns(2)
                cuas_rows = gc1.slider("Grid Rows", 1, 4, 2, 1, key="cuas_grid_r")
                cuas_cols = gc2.slider("Grid Columns", 2, 6, 3, 1, key="cuas_grid_c")
                cuas_n_barrels = cuas_rows * cuas_cols
            elif cuas_layout in ["fan_horizontal", "fan_vertical", "v_shape"]:
                cuas_n_barrels = st.slider("Number of Barrels (N)", 3, 12, 6, 1, key="cuas_n_barrels_fan")
            else:
                custom_text = st.text_area(
                    "Barrel Coordinates (Y, Z in meters, choke):",
                    value="-0.15, 0.15, Full\n0.15, 0.15, Full\n-0.15, -0.15, Modified\n0.15, -0.15, Modified\n0.00, 0.00, Extra Full",
                    help="One barrel per line: y_meters, z_meters, choke",
                    key="cuas_custom_text",
                )
                cuas_n_barrels = 5

            cuas_ammo = st.selectbox(
                "Ammunition Load",
                list(sg_presets.keys()),
                format_func=lambda k: sg_presets[k]["name"],
                index=list(sg_presets.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets else 0,
                key="cuas_ammo",
            )
            cuas_pellet_ovr: int | None = None
            if sg_presets[cuas_ammo].get("pellet_count_user_defined"):
                cuas_pellet_ovr = int(
                    st.number_input(
                        "User-Defined Pellet Count per Shotshell",
                        min_value=1,
                        max_value=500,
                        value=int(sg_presets[cuas_ammo]["pellets_per_shot"]),
                        step=1,
                        key="cuas_pel_cnt",
                    )
                )
            cuas_choke = st.selectbox("Choke Tube", list(choke_defs.keys()), index=3, key="cuas_choke")
            if sg_presets[cuas_ammo].get("choke_restriction") == "Cylinder" and cuas_choke != "Cylinder":
                st.caption("ℹ️ Tungsten load restriction: **Cylinder** choke is enforced automatically during simulation.")

        with col_tgt:
            st.markdown("**2. Target Drone & Engagement Kinematics**")
            cuas_drone_model = st.selectbox(
                "Target Drone Type",
                list(drone_specs.keys()),
                format_func=lambda k: drone_specs[k]["name"],
                index=list(drone_specs.keys()).index("commercial_quad_medium") if "commercial_quad_medium" in drone_specs else 0,
                key="cuas_drone_model",
            )
            sel_drone = drone_specs[cuas_drone_model]
            cuas_dist = st.slider("Interception Distance (m) [25 / 50 / 75 m]", 10.0, 100.0, 35.0, 1.0, key="cuas_dist")
            cuas_speed = st.slider("Drone Speed (m/s) [0 = Stationary]", 0.0, 50.0, 15.0, 1.0, key="cuas_speed")
            cuas_aspect_mode = st.selectbox(
                "Target Presentation / Aspect",
                list(aspect_presets.keys()),
                format_func=lambda k: aspect_presets[k]["name"],
                index=0,
                key="cuas_aspect_mode",
                help="Cross-section presented toward the incoming shot cloud.",
            )

        with st.expander(
            "Custom Drone & Custom Shell Builder (Optional)",
            expanded=bool(st.session_state.get("cuas_cust_drone_on") or st.session_state.get("cuas_cust_shell_on")),
        ):
            b_cd, b_cs = st.columns(2)
            custom_drone_spec: dict[str, Any] | None = None
            custom_shell_spec: dict[str, Any] | None = None
            with b_cd:
                use_custom_drone = st.checkbox(
                    "Override Threat with Custom Drone Geometry",
                    value=False,
                    key="cuas_cust_drone_on",
                )
                if use_custom_drone:
                    cd_name = st.text_input("Custom Drone Name", value="Custom Interceptor Quad", key="cuas_cd_name")
                    cd_c1, cd_c2 = st.columns(2)
                    cd_type = cd_c1.selectbox(
                        "Airframe Architecture",
                        ["quadcopter", "hexacopter", "octocopter", "fixed_wing"],
                        key="cuas_cd_type",
                    )
                    cd_span = cd_c2.number_input("Rotor Span / Wingspan (m)", 0.20, 4.00, 0.50, 0.05, key="cuas_cd_span")
                    cd_fl = cd_c1.number_input("Fuselage Length (m)", 0.08, 2.50, 0.24, 0.02, key="cuas_cd_fl")
                    cd_fw = cd_c2.number_input("Fuselage Width (m)", 0.05, 1.00, 0.13, 0.01, key="cuas_cd_fw")
                    cd_fh = cd_c1.number_input("Fuselage Height (m)", 0.04, 0.80, 0.10, 0.01, key="cuas_cd_fh")
                    cd_rr = cd_c2.number_input("Rotor / Prop Radius (m)", 0.04, 0.60, 0.12, 0.01, key="cuas_cd_rr")
                    cd_erot = cd_c1.number_input("Rotor / Motor Kill Energy (J)", 2.0, 100.0, 10.0, 1.0, key="cuas_cd_erot")
                    cd_ecore = cd_c2.number_input("Core / Battery Kill Energy (J)", 5.0, 150.0, 24.0, 2.0, key="cuas_cd_ecore")
                    cd_crit = st.slider("Critical Hits Required for Kill", 1, 10, 3, 1, key="cuas_cd_crit")
                    cd_bat_top = cd_c1.checkbox("Battery Mounted on Top", value=True, key="cuas_cd_bat_top")
                    cd_pg7m = cd_c2.checkbox("Underslung PG-7M Warhead", value=True, key="cuas_cd_pg7m")
                    cd_bl_mm = cd_c1.number_input("Battery Length (mm)", 60.0, 300.0, 150.0, 5.0, key="cuas_cd_bl")
                    cd_bw_mm = cd_c2.number_input("Battery Width (mm)", 30.0, 200.0, 70.0, 5.0, key="cuas_cd_bw")
                    cd_bh_mm = cd_c1.number_input("Battery Height (mm)", 20.0, 150.0, 50.0, 2.0, key="cuas_cd_bh")
                    custom_drone_spec = build_custom_drone_spec(
                        name=cd_name,
                        drone_type=cd_type,
                        span_m=cd_span,
                        fuselage_l_m=cd_fl,
                        fuselage_w_m=cd_fw,
                        fuselage_h_m=cd_fh,
                        rotor_radius_m=cd_rr,
                        lethal_energy_rotor_j=cd_erot,
                        lethal_energy_core_j=cd_ecore,
                        critical_hits_for_kill=cd_crit,
                        battery_l_m=cd_bl_mm / 1000.0,
                        battery_w_m=cd_bw_mm / 1000.0,
                        battery_h_m=cd_bh_mm / 1000.0,
                        battery_on_top=bool(cd_bat_top),
                        has_warhead=bool(cd_pg7m),
                    )
                    sel_drone = custom_drone_spec
                    st.caption(f"Active custom target: **{sel_drone['name']}** ({sel_drone['drone_type']}, {sel_drone['span_m']:.2f} m)")

            with b_cs:
                use_custom_shell = st.checkbox(
                    "Override Load with Custom Shotgun Shell",
                    value=False,
                    key="cuas_cust_shell_on",
                )
                if use_custom_shell:
                    cs_name = st.text_input("Custom Shell Name", value="Custom 12G TSS Load", key="cuas_cs_name")
                    cs_c1, cs_c2 = st.columns(2)
                    cs_mat = cs_c1.selectbox(
                        "Pellet Material",
                        list(PELLET_MATERIAL_DENSITIES_G_CM3.keys()),
                        index=4,
                        key="cuas_cs_mat",
                    )
                    cs_diam = cs_c2.number_input("Pellet Diameter (mm)", 1.50, 10.00, 3.81, 0.05, key="cuas_cs_diam")
                    cs_n = cs_c1.number_input("Pellets per Shell", 1, 500, 70, 1, key="cuas_cs_n")
                    cs_v0 = cs_c2.number_input("Muzzle Velocity V0 (m/s)", 200.0, 750.0, 415.0, 5.0, key="cuas_cs_v0")
                    cs_cd = cs_c1.number_input("Effective Drag Cd", 0.12, 0.90, 0.44, 0.02, key="cuas_cs_cd", help="Use ~0.44 for spheres, ~0.18 for fin-stabilized darts")
                    cs_pat_on = cs_c2.checkbox("Override 40-yd Pattern %", value=False, key="cuas_cs_pat_on")
                    cs_price_eur = cs_c1.number_input("Unit Price (€ / shotshell)", 0.10, 100.0, 2.00, 0.10, key="cuas_cs_price")
                    cs_pat40 = (
                        st.slider("Measured 40-yd 30\" Pattern (%)", 25.0, 95.0, 75.0, 1.0, key="cuas_cs_pat40")
                        if cs_pat_on
                        else None
                    )
                    custom_shell_spec = build_custom_shell_spec(
                        name=cs_name,
                        pellet_material=cs_mat,
                        pellet_diameter_mm=cs_diam,
                        pellets_per_shot=int(cs_n),
                        v0_mps=cs_v0,
                        effective_cd=cs_cd,
                        default_choke=str(cuas_choke),
                        custom_pattern_40yd_30in_pct=(cs_pat40 / 100.0 if cs_pat40 is not None else None),
                        unit_price_eur=float(cs_price_eur),
                    )
                    st.caption(
                        f"Computed pellet mass: **{custom_shell_spec['pellet_mass_g']:.3f} g** | "
                        f"Total shot charge: **{custom_shell_spec['payload_mass_g']:.1f} g**"
                    )

        with st.expander("Advanced Array Geometry, Fire Control & 3D Attitude Options", expanded=expert):
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                st.markdown("**Mount Spacing & Mixed Loadout**")
                if cuas_layout in ["circular", "concentric", "honeycomb"]:
                    cuas_mount_r = st.slider("Mount Radius (m)", 0.10, 0.60, 0.20, 0.02, key="cuas_mount_r")
                    cuas_cols, cuas_rows, cuas_sp_y, cuas_sp_z = 3, 2, 0.12, 0.12
                elif cuas_layout == "square":
                    cuas_sp_y = cuas_sp_z = float(cuas_sq_cm) / 100.0
                    cuas_cols, cuas_rows, cuas_mount_r = 2, 2, 0.20
                elif cuas_layout == "quincunx":
                    cuas_mount_r = float(cuas_qn_cm) / 100.0
                    cuas_cols, cuas_rows, cuas_sp_y, cuas_sp_z = 3, 2, 0.12, 0.12
                elif cuas_layout == "grid":
                    cuas_sp_y = st.slider("Horizontal Pitch (m)", 0.06, 0.30, 0.12, 0.01, key="cuas_sp_y")
                    cuas_sp_z = st.slider("Vertical Pitch (m)", 0.06, 0.30, 0.12, 0.01, key="cuas_sp_z")
                    cuas_mount_r = 0.20
                elif cuas_layout in ["fan_horizontal", "fan_vertical", "v_shape"]:
                    cuas_sp_y = st.slider("Barrel Spacing Pitch (m)", 0.06, 0.25, 0.12, 0.01, key="cuas_fan_sp")
                    cuas_sp_z = cuas_sp_y
                    cuas_cols, cuas_rows, cuas_mount_r = 3, 2, 0.20
                else:
                    cuas_mount_r, cuas_cols, cuas_rows, cuas_sp_y, cuas_sp_z = 0.20, 3, 2, 0.12, 0.12

                cuas_b_len_preset = st.selectbox(
                    "Barrel Length Preset (cm / in)",
                    ["Custom Slider", "66 cm (26.0 in)", "76 cm (29.9 in)"],
                    index=0,
                    key="cuas_b_len_cm_preset",
                )
                cuas_b_len = st.slider("Barrel Length (inches)", 14.0, 32.0, 24.0, 1.0, key="cuas_b_len")
                if cuas_b_len_preset.startswith("66 cm"):
                    cuas_b_len = 66.0 / 2.54
                elif cuas_b_len_preset.startswith("76 cm"):
                    cuas_b_len = 76.0 / 2.54
                cuas_mixed_loadout = st.checkbox(
                    "Heterogeneous Core + Perimeter Loadout",
                    value=False,
                    key="cuas_mixed_loadout",
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
                    cuas_inner_choke = st.selectbox("Core / Center Choke", list(choke_defs.keys()), index=5, key="cuas_inner_choke")

            with adv_c2:
                st.markdown("**Pointing, Ripple Fire & Sampling**")
                cuas_pointing = st.selectbox(
                    "Pointing & Convergence Mode",
                    ["convergent", "parallel", "divergent_curtain", "staggered_tiling"],
                    format_func=lambda k: {
                        "convergent": "Convergent Focus",
                        "parallel": "Parallel Bores",
                        "divergent_curtain": "Divergent Curtain",
                        "staggered_tiling": "Staggered Pattern Tiling",
                    }.get(k, k),
                    index=0,
                    key="cuas_pointing",
                )
                cuas_splay = (
                    st.slider("Curtain Splay Angle (mrad)", 0.0, 15.0, 4.0, 0.5, key="cuas_splay")
                    if cuas_pointing == "divergent_curtain"
                    else 0.0
                )
                cuas_fire_mode = st.radio(
                    "Salvo Firing Sequence",
                    ["simultaneous_volley", "sequential_ripple", "balanced_pair_ripple"],
                    format_func=lambda k: {
                        "simultaneous_volley": "Simultaneous Volley",
                        "sequential_ripple": "Sequential Ripple",
                        "balanced_pair_ripple": "Balanced Symmetric Pair Ripple",
                    }.get(k, k),
                    key="cuas_fire_mode",
                )
                cuas_ripple_dt = (
                    st.slider("Inter-Step Micro-Delay (ms)", 2.0, 40.0, 10.0, 1.0, key="cuas_ripple_dt")
                    if cuas_fire_mode != "simultaneous_volley"
                    else 0.0
                )
                cuas_n_volleys = st.slider("Monte Carlo Volleys Evaluated", 10, 50, 25, 5, key="cuas_n_volleys")

            with adv_c3:
                st.markdown("**3D Drone Attitude & Wind**")
                cuas_pitch_tilt = st.slider(
                    "Pitch Tilt (°)",
                    -45.0,
                    45.0,
                    15.0 if cuas_aspect_mode == "head_on_frontal" else 0.0,
                    1.0,
                    key="cuas_pitch_tilt",
                )
                cuas_yaw_angle = st.slider("Yaw Angle (°)", -180.0, 180.0, 0.0, 5.0, key="cuas_yaw_angle")
                cuas_roll_angle = st.slider("Bank / Roll (°)", -90.0, 90.0, 0.0, 5.0, key="cuas_roll_angle")
                cuas_heading = st.slider("Flight Vector Heading (°)", 0.0, 180.0, 90.0, 5.0, key="cuas_heading")
                cuas_aim_jitter = st.slider("Turret Tracking Jitter (mrad)", 0.0, 4.0, 1.2, 0.2, key="cuas_aim_jitter")
                cuas_wind = st.slider("Ambient Crosswind (m/s)", -15.0, 15.0, 0.0, 0.5, key="cuas_wind")
                cuas_headwind = st.slider("Headwind (+) / Tailwind (-) (m/s)", -20.0, 20.0, 0.0, 1.0, key="cuas_headwind")
                cuas_temp_c = st.slider("Ambient Temperature (°C, Annex A)", -20.0, 45.0, 15.0, 1.0, key="cuas_temp_c")
                cuas_prop_type = st.selectbox("Propellant Temp Curve", ["double_base", "single_base"], key="cuas_prop_type")
                cuas_drag_model = st.selectbox(
                    "Sphere Drag Model",
                    ["compressible_re_m", "mach_only_smooth_sphere"],
                    format_func=lambda k: {
                        "compressible_re_m": "Compressible Re + Mach (Morrison/Bailey-Hiatt)",
                        "mach_only_smooth_sphere": "Compton (1996) Mach-Only Smooth Sphere",
                    }.get(k, k),
                    key="cuas_drag_model",
                )

        with st.expander("🔬 High-Fidelity Physics & Terminal Realism", expanded=False):
            hfc1, hfc2 = st.columns(2)
            with hfc1:
                cuas_hf_rotor_solidity = st.checkbox(
                    "Rotor Blade Swept Transit Kinematics (Solidity Ratio)",
                    value=False,
                    key="cuas_hf_rotor_solidity",
                    help="Calculates rotating blade sweep volume during pellet transit across the rotor disc plane.",
                )
                cuas_hf_v50_limit = st.checkbox(
                    "Sectional Density V50 Ballistic Limit & Recht-Ipson Residual Velocity",
                    value=False,
                    key="cuas_hf_v50_limit",
                    help="Evaluates component-specific armor profiles and Recht-Ipson ballistic limit perforation.",
                )
                cuas_hf_ray_shield = st.checkbox(
                    "3D Front-to-Back Component Ray Shielding",
                    value=False,
                    key="cuas_hf_ray_shield",
                    help="Shields rear components when pellets perforate or are stopped by front components.",
                )
                cuas_hf_oblique_ricochet = st.checkbox(
                    "Oblique Ricochet Cutoff V50(θ) & Grazing Deflection",
                    value=False,
                    key="cuas_hf_oblique_ricochet",
                    help="Models Recht-Ipson oblique armor perforation V50(θ) = V50 / cos^k(θ) and grazing ricochet.",
                )
            with hfc2:
                cuas_pattern_dist = st.selectbox(
                    "Shot Cloud Dispersion Profile",
                    ["gaussian", "core_flyer_mixture", "choke_annular_mixture"],
                    format_func=lambda k: {
                        "gaussian": "Standard Gaussian (Nominal)",
                        "core_flyer_mixture": "Core-Flyer Mixture (Heavy Peripheral Drag)",
                        "choke_annular_mixture": "Choke Annular Ring Mixture (Tight Chokes)",
                    }.get(k, k),
                    key="cuas_pattern_dist",
                )
                cuas_flyer_frac = st.slider(
                    "Flyer / Non-Gaussian Fraction",
                    0.05,
                    0.35,
                    0.12,
                    0.01,
                    key="cuas_flyer_frac",
                )
                cuas_shot_cup_sep_m = st.slider(
                    "Shot-Cup / Wad Separation Distance (m)",
                    0.0,
                    12.0,
                    0.0,
                    0.5,
                    key="cuas_shot_cup_sep_m",
                    help="Delays full aerodynamic radial expansion until wad/shot-cup separation at this range.",
                )

        _pf_global = get_active_physics_fidelity() or {}
        _hf_rotor = bool(cuas_hf_rotor_solidity or _pf_global.get("use_rotor_solidity", False))
        _hf_v50 = bool(cuas_hf_v50_limit or _pf_global.get("use_v50_ballistic_limit", False))
        _hf_ray = bool(cuas_hf_ray_shield or _pf_global.get("enable_ray_shielding", False))
        _hf_ricochet = bool(cuas_hf_oblique_ricochet or _pf_global.get("use_oblique_ricochet", False))
        _hf_pat_dist = str(
            cuas_pattern_dist
            if (cuas_pattern_dist != "gaussian" or not _pf_global)
            else _pf_global.get("pattern_distribution", "core_flyer_mixture")
        )
        _hf_flyer = float(cuas_flyer_frac)
        _hf_shot_cup = float(
            cuas_shot_cup_sep_m
            if (cuas_shot_cup_sep_m > 0.0 or not _pf_global)
            else _pf_global.get("shot_cup_separation_m", 2.5)
        )

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

        custom_barrel_json = json.dumps(parsed_custom, sort_keys=True) if parsed_custom else ""
        custom_drone_json = json.dumps(custom_drone_spec, sort_keys=True) if custom_drone_spec else ""
        custom_shell_json = json.dumps(custom_shell_spec, sort_keys=True) if custom_shell_spec else ""

        _cal_twin_s1 = get_active_calibration_profile()
        _cal_twin_s1_json = json.dumps(_cal_twin_s1, sort_keys=True) if _cal_twin_s1 else ""
        if _cal_twin_s1:
            st.info(
                f"🧬 **Calibrated Digital Twin Active:** spread={_cal_twin_s1.get('spread_factor', 1.0):.2f}x "
                f"(exp={_cal_twin_s1.get('spread_exponent', 1.15):.2f}) | MV={_cal_twin_s1.get('mv_scale', 1.0):.3f}x | "
                f"E_crit={_cal_twin_s1.get('drone_ecrit_scale', 1.0):.2f}x"
            )
        with st.spinner("Simulating Multi-Shotgun Counter-UAS Volley Monte Carlo..."):
            cuas_sim = _cached_antidrone_array_mc(
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
                custom_barrel_specs_json=custom_barrel_json,
                ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                default_choke=str(cuas_choke),
                default_barrel_length_in=float(cuas_b_len),
                target_distance_m=float(cuas_dist),
                drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
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
                inner_ammunition_preset=cuas_inner_ammo if not custom_shell_spec else None,
                outer_ammunition_preset=str(cuas_ammo) if (cuas_mixed_loadout and not custom_shell_spec) else None,
                inner_choke=cuas_inner_choke,
                custom_drone_spec_json=custom_drone_json,
                custom_shell_spec_json=custom_shell_json,
                calibration_profile_json=_cal_twin_s1_json,
                headwind_mps=float(cuas_headwind),
                ambient_temp_c=float(cuas_temp_c),
                propellant_type=str(cuas_prop_type),
                drag_model=str(cuas_drag_model),
                pellets_per_shot=cuas_pellet_ovr if not custom_shell_spec else None,
                use_rotor_solidity=_hf_rotor,
                use_v50_ballistic_limit=_hf_v50,
                enable_ray_shielding=_hf_ray,
                use_oblique_ricochet=_hf_ricochet,
                shot_cup_separation_m=_hf_shot_cup,
                pattern_distribution=_hf_pat_dist,
                flyer_fraction=_hf_flyer,
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
            "ammunition_preset": "CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
            "drone_model": "CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
            "aspect_mode": str(cuas_aspect_mode),
            "drone_pitch_deg": float(cuas_pitch_tilt),
            "drone_yaw_deg": float(cuas_yaw_angle),
            "drone_roll_deg": float(cuas_roll_angle),
            "target_distance_m": float(cuas_dist),
            "drone_speed_mps": float(cuas_speed),
            "drone_heading_deg": float(cuas_heading),
            "custom_drone_enabled": bool(custom_drone_spec is not None),
            "custom_shell_enabled": bool(custom_shell_spec is not None),
        }
        if custom_drone_spec is not None:
            current_cuas_cfg["custom_drone_spec"] = custom_drone_spec
        if custom_shell_spec is not None:
            current_cuas_cfg["custom_shell_spec"] = custom_shell_spec
        st.session_state["_report_cuas_config"] = current_cuas_cfg
        st.session_state["_report_array_summary"] = cu_sm
        st.session_state["_active_cuas_profile"] = {
            "layout_type": str(cuas_layout if cuas_layout in ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"] else "circular"),
            "n_barrels": int(cuas_n_barrels),
            "mount_radius_m": float(cuas_mount_r),
            "pointing_mode": str(cuas_pointing),
            "splay_angle_mrad": float(cuas_splay),
            "firing_mode": str(cuas_fire_mode),
            "ripple_delay_ms": float(cuas_ripple_dt),
            "choke": str(cuas_choke),
            "ammunition_preset": str(cuas_ammo),
            "hybrid_mode": bool(cuas_mixed_loadout and not custom_shell_spec),
            "inner_ammunition_preset": str(cuas_inner_ammo or "") if not custom_shell_spec else "",
            "inner_choke": str(cuas_inner_choke or "") if not custom_shell_spec else "",
            "drone_model": str(cuas_drone_model),
            "aspect_mode": str(cuas_aspect_mode),
            "target_distance_m": float(cuas_dist),
            "drone_speed_mps": float(cuas_speed),
            "label": f"{int(cuas_n_barrels)}-bbl {cuas_layout} | {sel_drone.get('name', cuas_drone_model)} @ {cuas_dist:.0f} m",
        }

        pk_val = cu_sm["kill_probability_pct"]
        pres_area_cm2 = cu_sm.get("presented_area_cm2", 0.0)
        cs_ratio = cu_sm.get("cross_section_ratio_pct", 100.0)

        if not expert:
            ck1, ck2, ck3, ck4 = st.columns(4)
            ck1.metric(
                "Kill Probability P_kill",
                f"{pk_val:.1f} %",
                delta="Mission Kill" if pk_val >= 80.0 else ("Probable Damage" if pk_val >= 50.0 else "Marginal Intercept"),
            )
            ck2.metric(
                "Mean Lethal Hits",
                f"{cu_sm['mean_lethal_hits']:.1f} hits",
                delta=f"{cu_sm['mean_hits_on_drone']:.1f} total (req: ≥{sel_drone['critical_hits_for_kill']})",
            )
            ck3.metric(
                "Presented Cross-Section",
                f"{pres_area_cm2:.1f} cm²",
                delta=f"{cs_ratio:.1f}% planform",
            )
            ck4.metric(
                "Intercept Cloud Diameter",
                f"{cu_sm['cloud_diameter_90_m']:.2f} m",
                delta=f"CEP50: {cu_sm['cloud_cep_50_m']:.2f} m @ {cu_sm['target_distance_m']:.0f} m",
            )
        else:
            ck1, ck2, ck3 = st.columns(3)
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
            ck3.metric(
                "Presented Cross-Section",
                f"{pres_area_cm2:.1f} cm²",
                delta=f"{cs_ratio:.1f}% of planform ({cu_sm.get('aspect_mode', 'head_on')})",
                delta_color="normal",
            )
            ck4, ck5, ck6 = st.columns(3)
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
            ck6.metric(
                "Striking Velocity",
                f"{cu_sm['mean_v_impact_nominal_mps']:.0f} m/s",
                delta=f"V0: {cu_sm['v0_nominal_mps']:.0f} m/s | {cu_sm['ammunition_name'][:22]}",
            )

        nx1, nx2, nx3 = st.columns(3)
        nx1.button("Next: Elevated Fallout SDZ", on_click=_navigate_to, args=("cuas", None, "sdz"), width="stretch", key="arr_next_sdz")
        nx2.button("Compare A/B Architectures", on_click=_navigate_to, args=("cuas", None, "comparator"), width="stretch", key="arr_next_comp")
        nx3.button("Optimize Recoil & Pareto", on_click=_navigate_to, args=("cuas", None, "recoil"), width="stretch", key="arr_next_recoil")

        sp1, sp2, sp3, sp4 = st.columns(4)
        sp1.metric("Hit Probability P_hit", f"{cu_sm.get('hit_probability_pct', pk_val):.1f} %", delta=f"{cu_sm.get('pattern_30in_pct', 0.0):.0f}% in 30\" / {cu_sm.get('pattern_1m2_pct', 0.0):.0f}% in 1m²")
        sp2.metric("Pellet Impulse & Energy", f"{cu_sm.get('nominal_pellet_impulse_ns', 0.0):.3f} N·s", delta=f"{cu_sm.get('nominal_pellet_energy_j', 0.0):.1f} J/pellet @ {cu_sm['target_distance_m']:.0f} m")
        sp3.metric("Volley Cost (€)", f"€{cu_sm.get('cost_per_volley_eur', 0.0):.2f}", delta=f"{cu_sm.get('total_pellets_per_volley', 0)} pellets ({cu_sm['n_barrels']} bbls)")
        sp4.metric("Temp / Gravity / String", f"{cu_sm.get('ambient_temp_c', 15.0):+.0f}°C | {cu_sm.get('nominal_gravity_drop_m', 0.0)*100:.1f} cm", delta=f"String: {cu_sm.get('shot_string_length_m', 0.0):.2f} m | HW: {cu_sm.get('headwind_mps', 0.0):+.0f} m/s")
        if _hf_ricochet or cu_sm.get("ricochet_count", 0) > 0 or _hf_shot_cup > 0.0:
            st.caption(
                f"🔬 **High-Fidelity Telemetry:** `{cu_sm.get('ricochet_count', 0)}` oblique ricochets | "
                f"`{cu_sm.get('blade_pass_through_count', 0)}` rotor sweep pass-throughs | "
                f"Shot-cup separation: `{cu_sm.get('shot_cup_separation_m', 0.0):.1f} m` (`{_hf_pat_dist}`)"
            )

        stage1_env_res: dict[str, Any] | None = None
        env_kwargs = dict(
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
            custom_barrel_specs_json=custom_barrel_json,
            ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
            default_choke=str(cuas_choke),
            default_barrel_length_in=float(cuas_b_len),
            drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
            drone_velocity_mps=float(cuas_speed),
            drone_heading_deg=float(cuas_heading),
            aspect_mode=str(cuas_aspect_mode),
            drone_pitch_deg=float(cuas_pitch_tilt),
            drone_yaw_deg=float(cuas_yaw_angle),
            drone_roll_deg=float(cuas_roll_angle),
            aim_error_mrad=float(cuas_aim_jitter),
            crosswind_mps=float(cuas_wind),
            inner_ammunition_preset=cuas_inner_ammo if not custom_shell_spec else None,
            outer_ammunition_preset=str(cuas_ammo) if (cuas_mixed_loadout and not custom_shell_spec) else None,
            inner_choke=cuas_inner_choke,
            custom_drone_spec_json=custom_drone_json,
            custom_shell_spec_json=custom_shell_json,
            sg_presets=sg_presets,
            headwind_mps=float(cuas_headwind),
            ambient_temp_c=float(cuas_temp_c),
            propellant_type=str(cuas_prop_type),
            drag_model=str(cuas_drag_model),
            pellets_per_shot=cuas_pellet_ovr if not custom_shell_spec else None,
            use_rotor_solidity=_hf_rotor,
            use_v50_ballistic_limit=_hf_v50,
            enable_ray_shielding=_hf_ray,
            use_oblique_ricochet=_hf_ricochet,
            shot_cup_separation_m=_hf_shot_cup,
            pattern_distribution=_hf_pat_dist,
            flyer_fraction=_hf_flyer,
            annex_c_sample={
                "n_pellets": cu_sm["total_pellets_per_volley"],
                "vulnerable_area_m2": cu_sm["presented_area_m2"],
                "theta_p_rad": max(0.004, cu_sm["cloud_cep_50_m"] / max(1.177 * cu_sm["target_distance_m"], 1.0)),
                "theta_aim_rad": float(cuas_aim_jitter) / 1000.0,
                "pk_given_hit": float(cu_sm.get("annex_c_analytical", {}).get("pk_given_hit", 0.65)),
            },
        )

        if not expert:
            st.session_state.setdefault("_report_aspect_table", [])
            s_tab1, s_tab2, s_tab_env, s_tab3, s_tab_hw = st.tabs(
                [
                    "2D Drone Intercept & Silhouette",
                    "3D Intercept Volume",
                    "Effective Range (R80) & Heatmap",
                    "Layout, Aspect & Barrel Breakdown",
                    "📐 Hardware SWaP-C, OTM & CAD",
                ]
            )
            with s_tab1:
                st.plotly_chart(create_2d_antidrone_intercept_figure(cuas_sim), width="stretch", key="cuas_s_2d_fig")
            with s_tab2:
                st.plotly_chart(create_3d_antidrone_defense_figure(cuas_sim), width="stretch", key="cuas_s_3d_fig")
                if st.checkbox("Overlay Multi-Distance Pattern Slices (25 / 50 / 75 m)", value=False, key="cuas_show_3d_multidist_std"):
                    md_slices = [
                        _cached_antidrone_array_mc(
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
                            custom_barrel_specs_json=custom_barrel_json,
                            ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                            default_choke=str(cuas_choke),
                            default_barrel_length_in=float(cuas_b_len),
                            target_distance_m=float(d_sl),
                            drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                            drone_velocity_mps=float(cuas_speed),
                            drone_heading_deg=float(cuas_heading),
                            aspect_mode=str(cuas_aspect_mode),
                            drone_pitch_deg=float(cuas_pitch_tilt),
                            drone_yaw_deg=float(cuas_yaw_angle),
                            drone_roll_deg=float(cuas_roll_angle),
                            aim_error_mrad=float(cuas_aim_jitter),
                            crosswind_mps=float(cuas_wind),
                            n_volleys=10,
                            seed=42,
                            inner_ammunition_preset=cuas_inner_ammo if not custom_shell_spec else None,
                            outer_ammunition_preset=str(cuas_ammo) if (cuas_mixed_loadout and not custom_shell_spec) else None,
                            inner_choke=cuas_inner_choke,
                            custom_drone_spec_json=custom_drone_json,
                            custom_shell_spec_json=custom_shell_json,
                            calibration_profile_json=_cal_twin_s1_json,
                            headwind_mps=float(cuas_headwind),
                            ambient_temp_c=float(cuas_temp_c),
                            propellant_type=str(cuas_prop_type),
                            drag_model=str(cuas_drag_model),
                            pellets_per_shot=cuas_pellet_ovr if not custom_shell_spec else None,
                            use_rotor_solidity=_hf_rotor,
                            use_v50_ballistic_limit=_hf_v50,
                            enable_ray_shielding=_hf_ray,
                            use_oblique_ricochet=_hf_ricochet,
                            shot_cup_separation_m=_hf_shot_cup,
                            pattern_distribution=_hf_pat_dist,
                            flyer_fraction=_hf_flyer,
                        )
                        for d_sl in (25.0, 50.0, 75.0)
                    ]
                    st.plotly_chart(create_3d_pattern_vs_distance_figure(md_slices), width="stretch", key="cuas_s_3d_md_fig")
            with s_tab_env:
                stage1_env_res = _render_effective_range_tab(**env_kwargs)
            with s_tab3:
                b_stats = cuas_sim.get("barrel_stats", [])
                if b_stats:
                    st.markdown("##### Per-Barrel Telemetry & Contribution Breakdown")
                    b_df = pd.DataFrame(b_stats)
                    st.dataframe(b_df, width="stretch")
                    st.download_button(
                        "Download Barrel Statistics (.csv)",
                        b_df.to_csv(index=False).encode("utf-8"),
                        "anti_drone_barrel_statistics.csv",
                        "text/csv",
                        key="dl_cuas_b_csv",
                    )
                with st.expander("Multi-Layout Range Benchmark (15 m to 65 m)", expanded=False):
                    with st.spinner("Running Multi-Layout Array Benchmark..."):
                        bench_res = _cached_compare_anti_drone_layouts(
                            distance_stations_m=(15.0, 25.0, 35.0, 50.0, 65.0),
                            ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                            drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                            drone_velocity_mps=float(cuas_speed),
                            pointing_mode=str(cuas_pointing),
                            firing_mode=str(cuas_fire_mode),
                            n_volleys=12,
                            seed=42,
                            custom_drone_spec_json=custom_drone_json,
                            custom_shell_spec_json=custom_shell_json,
                        )
                        st.plotly_chart(create_antidrone_layout_comparison_figure(bench_res), width="stretch")
                        st.dataframe(pd.DataFrame(bench_res["comparison_table"]), width="stretch")
                with st.expander("3D Target Aspect Angle Vulnerability Benchmark", expanded=False):
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
                            custom_barrel_specs_json=custom_barrel_json,
                            ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                            default_choke=str(cuas_choke),
                            default_barrel_length_in=float(cuas_b_len),
                            target_distance_m=float(cuas_dist),
                            drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                            drone_velocity_mps=float(cuas_speed),
                            aim_error_mrad=float(cuas_aim_jitter),
                            crosswind_mps=float(cuas_wind),
                            n_volleys=15,
                            seed=42,
                            custom_drone_spec_json=custom_drone_json,
                            custom_shell_spec_json=custom_shell_json,
                        )
                        st.session_state["_report_aspect_table"] = aspect_bench["comparison_table"]
                        st.session_state["_report_aspect_bench"] = aspect_bench
                        st.plotly_chart(create_drone_aspect_radar_figure(aspect_bench), width="stretch")
                        st.dataframe(pd.DataFrame(aspect_bench["comparison_table"]), width="stretch")
            with s_tab_hw:
                _render_stage1_hardware_cad_tab(
                    layout_type=str(cuas_layout),
                    n_barrels=int(cuas_n_barrels),
                    mount_radius_m=float(cuas_mount_r),
                    spacing_y_m=float(cuas_sp_y),
                    spacing_z_m=float(cuas_sp_z),
                    n_cols=int(cuas_cols),
                    n_rows=int(cuas_rows),
                    ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                    default_choke=str(cuas_choke),
                    default_barrel_length_in=float(cuas_b_len),
                    pointing_mode=str(cuas_pointing),
                    splay_angle_mrad=float(cuas_splay),
                    firing_mode=str(cuas_fire_mode),
                    ripple_delay_ms=float(cuas_ripple_dt),
                    drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                    crosswind_mps=float(cuas_wind),
                    hybrid_mode=bool(cuas_mixed_loadout and not custom_shell_spec),
                    custom_barrel_specs=parsed_custom,
                )
        else:
            cuas_tab1, cuas_tab2, cuas_tab_env, cuas_tab3, cuas_tab4, cuas_tab5, cuas_tab6, cuas_tab_hw = st.tabs([
                "🎯 2D Drone Intercept & Damage Silhouette",
                "🛸 3D Muzzle Array & Intercept Volume",
                "📏 Effective Range (R80) & Heatmap",
                "📊 Per-Barrel Efficiency & Hit Contributions",
                "📈 Array Architecture Benchmark (Geometry Comparison)",
                "🌐 Aspect Angle Benchmark (Cross-Section Sweep)",
                "🎬 3D Animated Intercept Replay",
                "📐 Hardware SWaP-C, OTM Stabilization & CAD",
            ])

            with cuas_tab1:
                fig_2d_cuas = create_2d_antidrone_intercept_figure(cuas_sim)
                st.plotly_chart(fig_2d_cuas, width="stretch", key="cuas_exp_2d_fig")

            with cuas_tab2:
                fig_3d_cuas = create_3d_antidrone_defense_figure(cuas_sim)
                st.plotly_chart(fig_3d_cuas, width="stretch", key="cuas_exp_3d_fig")
                if st.checkbox("Overlay Multi-Distance Pattern Slices (25 / 50 / 75 m)", value=False, key="cuas_show_3d_multidist_exp"):
                    md_slices_e = [
                        _cached_antidrone_array_mc(
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
                            custom_barrel_specs_json=custom_barrel_json,
                            ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                            default_choke=str(cuas_choke),
                            default_barrel_length_in=float(cuas_b_len),
                            target_distance_m=float(d_sl),
                            drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                            drone_velocity_mps=float(cuas_speed),
                            drone_heading_deg=float(cuas_heading),
                            aspect_mode=str(cuas_aspect_mode),
                            drone_pitch_deg=float(cuas_pitch_tilt),
                            drone_yaw_deg=float(cuas_yaw_angle),
                            drone_roll_deg=float(cuas_roll_angle),
                            aim_error_mrad=float(cuas_aim_jitter),
                            crosswind_mps=float(cuas_wind),
                            n_volleys=10,
                            seed=42,
                            inner_ammunition_preset=cuas_inner_ammo if not custom_shell_spec else None,
                            outer_ammunition_preset=str(cuas_ammo) if (cuas_mixed_loadout and not custom_shell_spec) else None,
                            inner_choke=cuas_inner_choke,
                            custom_drone_spec_json=custom_drone_json,
                            custom_shell_spec_json=custom_shell_json,
                            calibration_profile_json=_cal_twin_s1_json,
                            headwind_mps=float(cuas_headwind),
                            ambient_temp_c=float(cuas_temp_c),
                            propellant_type=str(cuas_prop_type),
                            drag_model=str(cuas_drag_model),
                            pellets_per_shot=cuas_pellet_ovr if not custom_shell_spec else None,
                            use_rotor_solidity=_hf_rotor,
                            use_v50_ballistic_limit=_hf_v50,
                            enable_ray_shielding=_hf_ray,
                            use_oblique_ricochet=_hf_ricochet,
                            shot_cup_separation_m=_hf_shot_cup,
                            pattern_distribution=_hf_pat_dist,
                            flyer_fraction=_hf_flyer,
                        )
                        for d_sl in (25.0, 50.0, 75.0)
                    ]
                    st.plotly_chart(create_3d_pattern_vs_distance_figure(md_slices_e), width="stretch", key="cuas_exp_3d_md_fig")

            with cuas_tab_env:
                stage1_env_res = _render_effective_range_tab(**env_kwargs)

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
                        ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                        drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                        drone_velocity_mps=float(cuas_speed),
                        pointing_mode=str(cuas_pointing),
                        firing_mode=str(cuas_fire_mode),
                        n_volleys=12,
                        seed=42,
                        custom_drone_spec_json=custom_drone_json,
                        custom_shell_spec_json=custom_shell_json,
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
                        custom_barrel_specs_json=custom_barrel_json,
                        ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                        default_choke=str(cuas_choke),
                        default_barrel_length_in=float(cuas_b_len),
                        target_distance_m=float(cuas_dist),
                        drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                        drone_velocity_mps=float(cuas_speed),
                        aim_error_mrad=float(cuas_aim_jitter),
                        crosswind_mps=float(cuas_wind),
                        n_volleys=15,
                        seed=42,
                        custom_drone_spec_json=custom_drone_json,
                        custom_shell_spec_json=custom_shell_json,
                    )
                    st.session_state["_report_aspect_table"] = aspect_bench["comparison_table"]
                    st.session_state["_report_aspect_bench"] = aspect_bench
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

            with cuas_tab_hw:
                _render_stage1_hardware_cad_tab(
                    layout_type=str(cuas_layout),
                    n_barrels=int(cuas_n_barrels),
                    mount_radius_m=float(cuas_mount_r),
                    spacing_y_m=float(cuas_sp_y),
                    spacing_z_m=float(cuas_sp_z),
                    n_cols=int(cuas_cols),
                    n_rows=int(cuas_rows),
                    ammunition_preset="CUSTOM_SHELL" if custom_shell_spec else str(cuas_ammo),
                    default_choke=str(cuas_choke),
                    default_barrel_length_in=float(cuas_b_len),
                    pointing_mode=str(cuas_pointing),
                    splay_angle_mrad=float(cuas_splay),
                    firing_mode=str(cuas_fire_mode),
                    ripple_delay_ms=float(cuas_ripple_dt),
                    drone_model="CUSTOM_DRONE" if custom_drone_spec else str(cuas_drone_model),
                    crosswind_mps=float(cuas_wind),
                    hybrid_mode=bool(cuas_mixed_loadout and not custom_shell_spec),
                    custom_barrel_specs=parsed_custom,
                )

        with st.expander("🧊 3D .STL Drone CAD Mesh Ray-Cast (Optional Watertight Mesh Import)", expanded=False):
            st.caption(
                "Upload a CAD-exported ASCII or binary `.stl` drone mesh (or load the built-in 3D Quadcopter STL) to project "
                "triangle surface normals along the active Pitch/Yaw/Roll attitude and ray-cast the active Stage 1 shotgun volley."
            )
            stl_c1, stl_c2 = st.columns([1.4, 1.0])
            with stl_c1:
                cuas_stl_up = st.file_uploader("Upload 3D Drone CAD Mesh (.stl)", type=["stl"], key="cuas_stl_up")
            with stl_c2:
                st.write("")
                if st.button("🛸 Load Built-In 3D Quadcopter .STL Mesh", key="cuas_stl_demo_btn", width="stretch"):
                    st.session_state["_cuas_stl_bytes"] = generate_sample_drone_stl_bytes("quadcopter")
            stl_bytes = cuas_stl_up.getvalue() if cuas_stl_up is not None else st.session_state.get("_cuas_stl_bytes")
            if stl_bytes:
                try:
                    stl_mesh = parse_stl_mesh(stl_bytes)
                    stl_proj = project_stl_mesh_vulnerability(
                        stl_mesh,
                        aspect_mode=str(cuas_aspect_mode),
                        drone_pitch_deg=float(cuas_pitch_tilt),
                        drone_yaw_deg=float(cuas_yaw_angle),
                        drone_roll_deg=float(cuas_roll_angle),
                        grid_res=44,
                    )
                    scat = cuas_sim.get("sample_scatter", {})
                    sy_stl = np.asarray(scat.get("y") or [0.0], dtype=np.float64)
                    sz_stl = np.asarray(scat.get("z") or [0.0], dtype=np.float64)
                    ammo_ref_stl = custom_shell_spec if custom_shell_spec else sg_presets.get(str(cuas_ammo), {})
                    m_kg_stl = float(ammo_ref_stl.get("pellet_mass_g", 1.35)) / 1000.0
                    d_m_stl = float(ammo_ref_stl.get("pellet_diameter_mm", 3.5)) / 1000.0
                    dens_stl = float(PELLET_MATERIAL_DENSITIES_G_CM3.get(str(ammo_ref_stl.get("pellet_material", "Hardened Lead")), 11.3))
                    v_arr_stl = np.full(len(sy_stl), float(cu_sm.get("mean_v_impact_nominal_mps", 280.0)), dtype=np.float64)
                    stl_eval = evaluate_stl_pellet_impacts_batch(
                        sy_stl,
                        sz_stl,
                        stl_proj,
                        pellet_velocity_arr=v_arr_stl,
                        pellet_mass_kg=m_kg_stl,
                        pellet_diameter_m=d_m_stl,
                        pellet_density_g_cm3=dens_stl,
                        use_v50_ballistic_limit=True,
                        use_oblique_ricochet=bool(_hf_ricochet),
                    )
                    sm_a, sm_b, sm_c, sm_d = st.columns(4)
                    sm_a.metric("CAD Mesh Triangles", f"{stl_mesh['n_triangles']}", delta=f"{stl_mesh['format'].upper()} STL")
                    sm_b.metric("STL Presented Area", f"{stl_proj['presented_area_cm2']:.1f} cm²", delta=f"{cuas_aspect_mode}")
                    sm_c.metric("Sample Volley CAD Hits", f"{stl_eval['n_hits']}", delta=f"{stl_eval['n_lethal']} lethal perforations")
                    sm_d.metric("Grazing Ricochets", f"{stl_eval['n_ricochets']}", delta="Recht-Ipson V50(θ)")
                    st.plotly_chart(create_stl_vulnerability_figure(stl_proj), width="stretch", key="cuas_stl_fig")
                except Exception as exc:
                    st.error(f"STL CAD ray-cast error: {exc}")

        st.divider()
        st.markdown("##### 1-Click Stage 1 Engineering Spec Sheet & Preset Export")
        ex1, ex2, ex3 = st.columns(3)
        spec_html = generate_cuas_array_spec_sheet_html(
            array_config=current_cuas_cfg,
            sim_result=cuas_sim,
            range_envelope=stage1_env_res,
            aspect_benchmark=st.session_state.get("_report_aspect_bench"),
        )
        spec_pdf = generate_cuas_array_spec_sheet_pdf(
            array_config=current_cuas_cfg,
            sim_result=cuas_sim,
            range_envelope=stage1_env_res,
            aspect_benchmark=st.session_state.get("_report_aspect_bench"),
        )
        ex1.download_button(
            "Download Array Spec Sheet (.html)",
            spec_html.encode("utf-8"),
            "cuas_array_spec_sheet.html",
            "text/html",
            key="dl_stage1_spec_html",
            width="stretch",
        )
        ex2.download_button(
            "Download Array Spec Sheet (.pdf)",
            spec_pdf,
            "cuas_array_spec_sheet.pdf",
            "application/pdf",
            key="dl_stage1_spec_pdf",
            width="stretch",
        )
        ex3.download_button(
            "Save Scenario Preset (.json)",
            serialize_cuas_scenario_config(current_cuas_cfg).encode("utf-8"),
            "cuas_turret_scenario.json",
            "application/json",
            key="dl_cuas_scenario_json",
            width="stretch",
        )


