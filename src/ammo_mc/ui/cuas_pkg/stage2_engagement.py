"""Stage 2: Closed-Loop FCS, Swarm Saturation, Payloads, and Recoil Workbench UI."""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ammo_mc.cuas_defense import (
    simulate_cuas_advanced_payload_mc,
    simulate_turret_recoil_and_gimbal_loads,
    sweep_recoil_vs_ripple_delay,
)
from ammo_mc.small_arms import generate_multishotgun_array_layout
from ammo_mc.ui.common import (
    _CUAS18_MODULES,
    _cached_closed_loop_fcs,
    _cached_compare_cuas_payloads,
    _cached_optimize_antidrone_array,
    _cached_swarm_saturation,
)
from ammo_mc.vis_cuas import (
    create_3d_swarm_hemisphere_figure,
    create_airburst_standoff_cone_figure,
    create_antidrone_optimizer_pareto_figure,
    create_cuas_payload_range_comparison_figure,
    create_fcs_tracking_and_evasion_figure,
    create_swarm_doctrine_tradeoff_figure,
    create_turret_recoil_dynamics_figure,
)
from ammo_mc.vis_cuas_vv import (
    create_dead_time_waterfall_figure,
    create_doctrine_comparator_figure,
)
from ammo_mc.ui.cuas_pkg.helpers import (
    _render_cuas_profile_sync_bar,
    render_assumptions_and_provenance_drawer,
    render_high_fidelity_active_banner,
    render_verification_and_counter_evidence,
)


def render_stage2_engagement(active_cuas_mod: str, expert: bool, sg_presets_18: dict[str, Any], choke_defs_18: dict[str, Any], drone_specs_18: dict[str, Any], aspect_presets_18: dict[str, Any], sensor_presets_18: dict[str, Any], payload_cat_18: dict[str, Any]) -> None:
    render_high_fidelity_active_banner()
    if active_cuas_mod == _CUAS18_MODULES[0]:
        _render_cuas_profile_sync_bar(
            "fcs",
            {
                "drone_model": ("cuas18_fcs_drone", list(drone_specs_18.keys())),
                "aspect_mode": ("cuas18_fcs_aspect", list(aspect_presets_18.keys())),
                "drone_speed_mps": ("cuas18_fcs_spd", (5.0, 40.0)),
                "target_distance_m": ("cuas18_fcs_dist", (15.0, 70.0)),
                "layout_type": ("cuas18_fcs_lay", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"]),
                "n_barrels": ("cuas18_fcs_nb", (2, 12)),
                "pointing_mode": ("cuas18_fcs_pt", ["divergent_curtain", "convergent", "staggered_tiling", "parallel"]),
                "splay_angle_mrad": ("cuas18_fcs_sp", (0.0, 12.0)),
                "firing_mode": ("cuas18_fcs_fire", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"]),
                "ammunition_preset": ("cuas18_fcs_ammo", list(sg_presets_18.keys())),
            },
        )
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

        with fc2:
            st.markdown("**Evasive Drone Kinematics**")
            fcs_drone = st.selectbox(
                "Drone Target Class",
                list(drone_specs_18.keys()),
                format_func=lambda k: drone_specs_18[k]["name"],
                index=0,
                key="cuas18_fcs_drone",
            )
            fcs_maneuver = st.selectbox(
                "Evasive Maneuver Profile",
                ["sinusoidal_weave", "corkscrew_spiral", "terminal_popup_jink", "singer_correlated", "linear_cruise"],
                format_func=lambda k: {
                    "sinusoidal_weave": "〰️ Sinusoidal Horizontal + Vertical Weave",
                    "corkscrew_spiral": "🌀 Barrel / Corkscrew Spiral",
                    "terminal_popup_jink": "⚡ Late Terminal Pop-Up Jink",
                    "singer_correlated": "🎲 Singer Exponentially Correlated Evasion (FC-F09)",
                    "linear_cruise": "➡️ Constant-Velocity Linear Cruise",
                }.get(k, k),
                index=0,
                key="cuas18_fcs_man",
            )
            fcs_spd = st.slider("Drone Speed (m/s)", 0.0, 50.0, 22.0, 1.0, key="cuas18_fcs_spd")

        with fc3:
            st.markdown("**Engagement Geometry & Loadout**")
            fcs_dist = st.slider("Intercept Distance (m)", 15.0, 100.0, 35.0, 1.0, key="cuas18_fcs_dist")
            fcs_ammo = st.selectbox("Shell Preset", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_fcs_ammo")

        with st.expander("Servo Limits, Array Pointing & Target Aspect", expanded=expert):
            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                fcs_servo_rate = st.slider("Gimbal Max Slew Rate (°/s)", 30.0, 180.0, 90.0, 10.0, key="cuas18_fcs_rate")
                fcs_servo_accel = st.slider("Gimbal Max Accel (°/s²)", 60.0, 500.0, 250.0, 20.0, key="cuas18_fcs_accel")
            with fa2:
                fcs_aspect = st.selectbox(
                    "Target Presentation Aspect",
                    list(aspect_presets_18.keys()),
                    format_func=lambda k: aspect_presets_18[k]["name"],
                    index=0,
                    key="cuas18_fcs_aspect",
                    help="Ingress angle presented toward FCS radar/optical line-of-sight.",
                )
                fcs_g = st.slider("Peak Evasive Maneuver Load (g)", 0.0, 5.0, 2.5, 0.25, key="cuas18_fcs_g")
            with fa3:
                fcs_layout = st.selectbox("Array Geometry", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"], index=0, key="cuas18_fcs_lay")
                fcs_n_b = st.slider("Barrel Count", 2, 12, 6, 1, key="cuas18_fcs_nb")
                fcs_point = st.selectbox("Pointing Mode", ["divergent_curtain", "convergent", "staggered_tiling", "parallel"], index=0, key="cuas18_fcs_pt")
                fcs_splay = st.slider("Curtain Splay Angle (mrad)", 0.0, 12.0, 4.5, 0.5, key="cuas18_fcs_sp")
                fcs_fire = st.selectbox("Firing Sequence", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"], index=0, key="cuas18_fcs_fire")
            fd1, fd2, fd3 = st.columns(3)
            with fd1:
                fcs_use_dead = st.checkbox("Use Explicit Serial Dead-Time Chain (~141 ms, FC-T04)", value=False, key="cuas18_fcs_dead")
                fcs_dead_preset = st.selectbox("Latency Pipeline Profile", ["eo_60hz_1stream", "eo_60hz_3stream", "radar_10hz_fused"], key="cuas18_fcs_dead_p")
            with fd2:
                fcs_heavy_atr = st.checkbox("Heavy-Tailed AI ATR Centroid Noise (FC-P12)", value=False, key="cuas18_fcs_atr")
                fcs_iter_lead = st.checkbox("Iterative 3D TOF Lead Solver (M08 / FC-F12)", value=False, key="cuas21_fcs_iter_lead")
            with fd3:
                fcs_rcs_nulls = st.checkbox("Aspect-Dependent Radar RCS Nulls (25-35 dB)", value=False, key="cuas18_fcs_rcs")
                fcs_tgt_el = st.slider("Target Elevation Angle (°)", 0.0, 65.0, 15.0, 5.0, key="cuas21_fcs_el")
                fcs_site_alt = st.slider("Site Altitude (m ASL, AT-F01)", 0.0, 3500.0, 0.0, 100.0, key="cuas21_fcs_alt")

        with st.spinner("Running Closed-Loop FCS & Evasive Intercept Monte Carlo..."):
            fcs_res = _cached_closed_loop_fcs(
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
                use_serial_dead_time_chain=bool(fcs_use_dead),
                dead_time_preset=str(fcs_dead_preset),
                use_heavy_tailed_atr=bool(fcs_heavy_atr),
                enable_aspect_rcs_nulls=bool(fcs_rcs_nulls),
                use_iterative_3d_lead_solver=bool(fcs_iter_lead),
                site_altitude_m=float(fcs_site_alt),
                target_elevation_deg=float(fcs_tgt_el),
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
        with st.expander("⏱️ Serial Dead-Time Chain Breakdown (FC-T04) & Counter-Evidence", expanded=bool(fcs_use_dead or fcs_iter_lead)):
            if fsm.get("iterative_lead_solution"):
                ils = fsm["iterative_lead_solution"]
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Iterative Converged TOF (FC-F12)", f"{ils['tof_s']*1000:.1f} ms", delta=f"{ils['iterations']} iters (res={ils['residual_s']:.1e} s)")
                ic2.metric("Ballistic Superelevation", f"{ils['superelevation_mrad']:.2f} mrad", delta=f"Drop: {ils['gravity_drop_m']*100:.1f} cm @ {ils['slant_range_m']:.1f} m")
                ic3.metric("Predicted Aim Az / El", f"{ils['aim_azimuth_deg']:.1f}° / {ils['aim_elevation_deg']:.1f}°", delta=f"Station P={ils['station_pressure_pa']:.0f} Pa")
            if fsm.get("dead_time_budget"):
                st.plotly_chart(create_dead_time_waterfall_figure(fsm["dead_time_budget"]), width="stretch", key="cuas18_fcs_dt_fig")
            from ammo_mc.cuas_rigor import get_contextual_counter_evidence

            render_verification_and_counter_evidence(
                fsm.get("verification_grade"),
                get_contextual_counter_evidence(stage="fcs", range_m=float(fcs_dist), n_barrels=int(fcs_n_b), splay_mrad=float(fcs_splay)),
                key_prefix="stage2_fcs",
            )

        with st.expander("🎯 Engagement Doctrine Lab (Simultaneous Salvo vs. Ripple vs. Wilkening SLS vs. Whistler SAS, M12)", expanded=False):
            from ammo_mc.cuas_v21_science import build_cuas_scenario_manifest
            from ammo_mc.ui.common import _cached_doctrine_comparator

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                doc_bda = st.slider("Shoot-Look-Shoot BDA Delay (s)", 0.10, 1.50, 0.40, 0.05, key="cuas21_doc_bda")
                doc_mag = st.slider("Ready Magazine Shells", 4, 24, 12, 2, key="cuas21_doc_mag")
            with dc2:
                doc_bias = st.slider("Shared Aim Bias sigma (mrad)", 0.5, 5.0, 2.0, 0.25, key="cuas21_doc_bias")
                doc_rho = st.slider("Burst AR(1) Correlation rho (HP-F36)", 0.0, 0.90, 0.65, 0.05, key="cuas21_doc_rho")
            with dc3:
                doc_class = st.selectbox(
                    "Target Armor Class",
                    ["fpv_cfrp", "plastic_shell"],
                    format_func=lambda k: "3 mm CFRP FPV (Wen 2000)" if k == "fpv_cfrp" else "Plastic-Shell Consumer Quad (5 J)",
                    key="cuas21_doc_cls",
                )
                doc_rstart = st.slider("Open-Fire Start Range (m)", 35.0, 80.0, 55.0, 5.0, key="cuas21_doc_rstart")

            doc_res = _cached_doctrine_comparator(
                ammo_preset=str(fcs_ammo),
                choke="Modified",
                drone_class=str(doc_class),
                n_barrels=int(fcs_n_b),
                barrels_per_burst=min(2, int(fcs_n_b)),
                magazine_shells=int(doc_mag),
                r_start_m=float(doc_rstart),
                drone_speed_mps=max(8.0, float(fcs_spd)),
                drone_altitude_m=15.0,
                bda_delay_s=float(doc_bda),
                aim_bias_sigma_mrad=float(doc_bias),
                ar1_rho=float(doc_rho),
                n_trials=1000,
                seed=42,
                site_altitude_m=float(fcs_site_alt),
            )
            dsm = doc_res["summary"]
            dk1, dk2, dk3 = st.columns(3)
            dk1.metric("Top Cumulative Pk Doctrine", dsm["best_pk_doctrine"][:28], delta=f"Pk={dsm['best_pk_pct']:.1f}% | Window={dsm['closing_window_s']:.2f}s")
            dk2.metric("Highest Ammo-Efficiency Doctrine", dsm["best_efficiency_doctrine"][:28], delta=f"{dsm['best_efficiency_score']:.1f}% Pk / round")
            dk3.metric("Wilkening Optimal 1st Wave (HP-F39)", f"{dsm['wilkening_first_wave_shots']} shot(s)", delta=f"SSPK={dsm['nominal_midwindow_sspk']:.3f} | Safe>={dsm['r_min_safe_m']:.1f}m")
            st.plotly_chart(create_doctrine_comparator_figure(doc_res), width="stretch", key="cuas21_doc_fig")
            st.dataframe(pd.DataFrame(doc_res["doctrines"]), width="stretch", hide_index=True)
            st.caption("Glazebrook & Washburn (2004) Gittins Multi-Target SLS Priority Example (`HP-S03` Eq. 14):")
            st.dataframe(pd.DataFrame(doc_res["gittins_ranking"]), width="stretch", hide_index=True)

            fcs_manifest = build_cuas_scenario_manifest(
                scenario_name=f"Stage 2 FCS {fcs_sensor}",
                seed=42,
                site_altitude_m=float(fcs_site_alt),
                array_config={"layout_type": fcs_layout, "n_barrels": fcs_n_b, "ammunition_preset": fcs_ammo, "aim_error_mrad": float(doc_bias)},
                target_config={"drone_model": fcs_drone, "drone_speed_mps": float(fcs_spd), "maneuver_profile": fcs_maneuver},
                fcs_config={"sensor": fcs_sensor, "predictor": fcs_predictor, "dead_time_s": float(fsm.get("dead_time_budget", {}).get("t_dead_s", 0.141)), "iterative_3d_lead": bool(fcs_iter_lead)},
            )
            render_assumptions_and_provenance_drawer(fcs_manifest, key_prefix="stage2_fcs")

    # --- TAB 18.2: MULTI-DRONE SWARM SATURATION ---
    elif active_cuas_mod == _CUAS18_MODULES[1]:
        _render_cuas_profile_sync_bar(
            "swarm",
            {
                "drone_speed_mps": ("cuas18_sw_spd", (8.0, 40.0)),
                "drone_model": ("cuas18_sw_dr", list(drone_specs_18.keys())),
                "aspect_mode": ("cuas18_sw_aspect", list(aspect_presets_18.keys())),
                "n_barrels": ("cuas18_sw_nb", (2, 16)),
                "ammunition_preset": ("cuas18_sw_ammo", list(sg_presets_18.keys())),
            },
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Coordinated Swarm Wave**")
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
        with sc2:
            st.markdown("**Threat Profile & Speed**")
            sw_drone = st.selectbox("Drone Target Model", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_sw_dr")
            sw_speed = st.slider("Mean Drone Speed (m/s)", 8.0, 40.0, 22.0, 1.0, key="cuas18_sw_spd")
        with sc3:
            st.markdown("**Ready Magazine & Ammo**")
            sw_barrels = st.slider("Ready Shotgun Barrels in Turret", 2, 16, 8, 1, key="cuas18_sw_nb")
            sw_bpb = st.slider("Sub-Salvo Barrels Fired per Burst", 1, 8, 2, 1, key="cuas18_sw_bpb", help="Firing fewer barrels per drone increases capacity before reload")
            sw_ammo = st.selectbox("Swarm Defense Load", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_sw_ammo")

        with st.expander("Engagement Doctrine, Reload & Perimeter Settings", expanded=expert):
            sa1, sa2, sa3 = st.columns(3)
            with sa1:
                sw_aspect = st.selectbox(
                    "Swarm Ingress Aspect",
                    list(aspect_presets_18.keys()),
                    format_func=lambda k: aspect_presets_18[k]["name"],
                    index=0,
                    key="cuas18_sw_aspect",
                    help="Default: Head-on Frontal for ingressing swarm waves.",
                )
                sw_reload = st.slider("Full Turret Reload Downtime (s)", 1.0, 12.0, 4.5, 0.5, key="cuas18_sw_rel")
            with sa2:
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
                sw_slew = st.slider("Turret Slew Rate (°/s)", 40.0, 200.0, 110.0, 10.0, key="cuas18_sw_slew")
            with sa3:
                sw_rmax = st.slider("Max Engagement Envelope (m)", 30.0, 90.0, 55.0, 5.0, key="cuas18_sw_rmax")
                sw_rmin = st.slider("Protected Asset Keep-Out Perimeter (m)", 5.0, 25.0, 12.0, 1.0, key="cuas18_sw_rmin")

        with st.spinner("Simulating Multi-Wave Drone Swarm Timeline..."):
            sw_res = _cached_swarm_saturation(
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
        _render_cuas_profile_sync_bar(
            "recoil",
            {
                "layout_type": ("cuas18_rec_lay", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"]),
                "n_barrels": ("cuas18_rec_nb", (3, 12)),
                "ammunition_preset": ("cuas18_rec_ammo", list(sg_presets_18.keys())),
                "firing_mode": ("cuas18_rec_fm", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"]),
                "ripple_delay_ms": ("cuas18_rec_dt", (0.0, 30.0)),
                "drone_model": ("cuas18_rec_dr", list(drone_specs_18.keys())),
                "target_distance_m": ("cuas18_rec_dist", (15.0, 65.0)),
                "drone_speed_mps": ("cuas18_rec_dspd", (0.0, 35.0)),
            },
        )
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Array Architecture & Loadout**")
            rec_layout = st.selectbox("Array Architecture", ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"], index=0, key="cuas18_rec_lay")
            rec_nb = st.slider("Barrel Count (N)", 3, 12, 6, 1, key="cuas18_rec_nb")
            rec_ammo = st.selectbox("Primary Ammunition", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_rec_ammo")

        with rc2:
            st.markdown("**Mount Reaction Limit**")
            rec_limit = st.slider("Mount Structural Reaction Limit (kN)", 5.0, 50.0, 18.0, 1.0, key="cuas18_rec_lim")
            rec_drone = st.selectbox("Design Threat Drone", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=1, key="cuas18_rec_dr")

        with rc3:
            st.markdown("**Design Intercept Envelope**")
            rec_dist = st.slider("Design Intercept Range (m)", 15.0, 65.0, 35.0, 1.0, key="cuas18_rec_dist")
            rec_dspd = st.slider("Design Drone Speed (m/s)", 0.0, 35.0, 18.0, 1.0, key="cuas18_rec_dspd")

        with st.expander("Buffer Spring, Gimbal Inertia & Firing Sequence", expanded=expert):
            ra1, ra2 = st.columns(2)
            with ra1:
                rec_mixed = st.checkbox("Mix Core TSS + Perimeter Buckshot", value=True, key="cuas18_rec_mix")
                rec_fmode = st.selectbox("Salvo Firing Sequence", ["balanced_pair_ripple", "simultaneous_volley", "sequential_ripple"], index=0, key="cuas18_rec_fm")
                rec_dt = st.slider("Ripple Micro-Delay (ms)", 0.0, 30.0, 10.0, 1.0, key="cuas18_rec_dt")
            with ra2:
                rec_mass = st.slider("Recoiling Assembly Mass (kg)", 15.0, 150.0, 45.0, 5.0, key="cuas18_rec_mass")
                rec_stiff = st.slider("Buffer Spring Stiffness (kN/m)", 40.0, 400.0, 120.0, 10.0, key="cuas18_rec_k") * 1000.0
                rec_ztr = st.slider("Vertical Trunnion Offset (m)", 0.05, 0.45, 0.18, 0.01, key="cuas18_rec_ztr")
                rec_gas_model = st.selectbox(
                    "Propellant Gas Recoil Model",
                    ["legacy_1275", "saami_proportional"],
                    format_func=lambda k: "SAAMI Proportional (fg=1.50 * V0 + 2.79g wad, AM-F03)" if k == "saami_proportional" else "Fixed Gas Jet Velocity (1275 m/s)",
                    key="cuas18_rec_gas",
                )
                rec_fg = st.slider("SAAMI Gas Factor f_g", 1.25, 1.50, 1.50, 0.05, key="cuas18_rec_fg") if rec_gas_model == "saami_proportional" else 1.50

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
                recoil_gas_model=str(rec_gas_model),
                saami_gas_factor_fg=float(rec_fg),
            )
            rec_sweep = sweep_recoil_vs_ripple_delay(
                barrels=rec_barrels,
                default_ammunition_preset=str(rec_ammo),
                mount_mass_kg=float(rec_mass),
                mount_stiffness_n_per_m=float(rec_stiff),
                mount_load_limit_kn=float(rec_limit),
                gimbal_trunnion_offset_z_m=float(rec_ztr),
                recoil_gas_model=str(rec_gas_model),
                saami_gas_factor_fg=float(rec_fg),
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


