"""Stage 3: SDZ Safety, Kill Chain, Site Defense, 3D Intercept Replay & Unified Raid UI."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from ammo_mc.cuas_analytics import (
    build_cuas_site_geojson,
    export_cuas_site_kml,
    serialize_site_layout_json,
    validate_site_layout_json,
)
from ammo_mc.cuas_mission import (
    get_kill_chain_sensor_catalog,
    get_magazine_system_specs,
    get_site_defense_presets,
)
from ammo_mc.terrain_dem import (
    TerrainDEMGrid,
    build_preset_terrain_dem,
    fetch_open_meteo_elevation_dem,
    get_terrain_dem_presets,
    parse_dem_csv,
    parse_esri_ascii_grid,
)
from ammo_mc.ui.common import (
    _CUAS18_MODULES,
    _cached_collateral_safety_envelope,
    _cached_compare_cuas_fallout,
    _cached_intercept_replay,
    _cached_kill_chain,
    _cached_pellet_fallout_sdz,
    _cached_site_defense,
    _cached_unified_raid,
    is_expert_mode,
)
from ammo_mc.vis_cuas import (
    create_2d_cuas_fallout_safety_map_figure,
    create_3d_cuas_fallout_footprint_figure,
)
from ammo_mc.vis_cuas_advanced import create_collateral_safety_envelope_figure
from ammo_mc.vis_cuas_analytics import create_real_map_site_defense_figure
from ammo_mc.vis_cuas_mission import (
    create_3d_animated_intercept_figure,
    create_kill_chain_timeline_figure,
    create_multiturret_site_map_figure,
    create_terrain_dem_site_3d_figure,
)
from ammo_mc.vis_cuas_raid import (
    create_raid_timeline_and_heat_figure,
    create_unified_tactical_raid_3d_figure,
)
from ammo_mc.ui.cuas_pkg.helpers import (
    _SITE_GEO_ANCHORS,
    _get_active_physics_fidelity_json,
    _get_cal_twin_json,
    _get_effective_cal_twin_json,
    _on_site_anchor_change,
    _render_cuas_profile_sync_bar,
    _sanitize_site_editor_tables,
    render_high_fidelity_active_banner,
    render_verification_and_counter_evidence,
)


def render_stage3_tactical(active_cuas_mod: str, expert: bool, sg_presets_18: dict[str, Any], choke_defs_18: dict[str, Any], drone_specs_18: dict[str, Any], aspect_presets_18: dict[str, Any], sensor_presets_18: dict[str, Any]) -> None:
    render_high_fidelity_active_banner()
    if active_cuas_mod == _CUAS18_MODULES[4]:
        _render_cuas_profile_sync_bar(
            "sdz",
            {
                "ammunition_preset": ("cuas18_sd_ammo", list(sg_presets_18.keys())),
                "choke": ("cuas18_sd_choke", list(choke_defs_18.keys())),
                "layout_type": ("cuas18_sd_lay", ["circular", "grid", "fan_horizontal", "honeycomb"]),
                "pointing_mode": ("cuas18_sd_pt", ["convergent", "divergent_curtain", "parallel"]),
            },
        )
        sd1, sd2 = st.columns(2)
        with sd1:
            st.markdown("**Elevated Turret Engagement Geometry**")
            sd_el = st.slider("Turret Firing Elevation Angle (°)", 10.0, 85.0, 35.0, 5.0, key="cuas18_sd_el")
            sd_az = st.slider("Turret Firing Azimuth (°)", -90.0, 90.0, 0.0, 5.0, key="cuas18_sd_az")

        with sd2:
            st.markdown("**Shotgun Ammunition & Choke**")
            sd_ammo = st.selectbox("Fired Shotgun Load", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=0, key="cuas18_sd_ammo")
            sd_choke = st.selectbox("Choke Tube", list(choke_defs_18.keys()), index=3, key="cuas18_sd_choke")

        with st.expander("Array Mount & Atmospheric Wind Shear Options", expanded=expert):
            sa1, sa2, sa3 = st.columns(3)
            with sa1:
                sd_h = st.slider("Turret Height Above Ground (m)", 0.5, 10.0, 2.0, 0.5, key="cuas18_sd_h")
                sd_n_p = st.slider("Monte Carlo Missed Pellets Tracked", 150, 450, 260, 20, key="cuas18_sd_np")
            with sa2:
                sd_lay = st.selectbox("Turret Array Geometry", ["circular", "grid", "fan_horizontal", "honeycomb"], index=0, key="cuas18_sd_lay")
                sd_point = st.selectbox("Array Pointing Mode", ["convergent", "divergent_curtain", "parallel"], index=0, key="cuas18_sd_pt")
            with sa3:
                sd_w_down = st.slider("Downrange Tailwind (+) / Headwind (-) (m/s)", -15.0, 15.0, 4.0, 1.0, key="cuas18_sd_wd")
                sd_w_cross = st.slider("Crosswind Drift (m/s)", -15.0, 15.0, 2.0, 1.0, key="cuas18_sd_wc")
                sd_shear = st.checkbox("Apply 1/7 Power-Law Boundary Layer Wind Shear", value=True, key="cuas18_sd_sh")

        with st.spinner("Integrating Vectorized 3-DoF Ballistic Pellet Fallout..."):
            sd_res = _cached_pellet_fallout_sdz(
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

        sdt1, sdt2, sdt3, sdt4 = st.tabs([
            "🌐 3D Elevated Fallout Arcs",
            "🗺️ 2D Ground SDZ Map & Multi-Load Bands",
            "📊 Multi-Load Safety Footprint Data & CSV",
            "🔊 Acoustic Blast, Danger Cone & Urban Collateral",
        ])
        with sdt1:
            st.plotly_chart(create_3d_cuas_fallout_footprint_figure(sd_res), width="stretch")
        with sdt2:
            st.plotly_chart(create_2d_cuas_fallout_safety_map_figure(sd_res, sd_comp), width="stretch")
        with sdt3:
            comp_df_sd = pd.DataFrame(sd_comp["records"])
            st.dataframe(comp_df_sd, width="stretch")
            gi_df = pd.DataFrame(sd_res["ground_impacts"])
            st.download_button("📥 Download Ground Pellet Impact Fallout Log (.csv)", gi_df.to_csv(index=False).encode("utf-8"), "cuas_pellet_fallout_sdz.csv", "text/csv", key="cuas18_dl_sdz")
        with sdt4:
            _s_prof = st.session_state.get("_active_cuas_profile") or {}
            cs1, cs2 = st.columns(2)
            with cs1:
                collat_env = st.selectbox(
                    "Deployment Environment",
                    ["urban_permissive", "urban_canyon"],
                    format_func=lambda k: {"urban_permissive": "Open / Suburban Base Perimeter", "urban_canyon": "Dense Urban Canyon (Reflective Facades)"}.get(k, k),
                    key="cuas18_collat_env",
                )
            with cs2:
                collat_bpb = st.slider("Burst Barrels Fired Together", 1, 6, min(6, int(_s_prof.get("barrels_per_burst", 2))), 1, key="cuas18_collat_bpb")
            collat_res = _cached_collateral_safety_envelope(
                n_barrels=int(_s_prof.get("n_barrels", 6)),
                barrels_per_burst=int(collat_bpb),
                ammunition_preset=str(sd_ammo),
                default_choke=str(sd_choke),
                default_barrel_length_in=24.0,
                firing_elevation_deg=float(sd_el),
                firing_azimuth_deg=float(sd_az),
                turret_height_m=float(sd_h),
                wind_downrange_mps=float(sd_w_down),
                wind_crossrange_mps=float(sd_w_cross),
                environment_type=str(collat_env),
                calibration_profile_json=_get_effective_cal_twin_json(),
                n_sample_pellets=int(sd_n_p),
                seed=42,
            )
            csm = collat_res["summary"]
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Deployment Verdict", csm["verdict"], delta=f"Min Safe El >= {csm['recommended_min_elevation_deg']:.0f}°")
            cm2.metric("140 / 165 dB Blast Standoff", f"{csm['acoustic_140db_downrange_m']:.0f} / {csm['acoustic_165db_downrange_m']:.0f} m", delta=f"1m Peak: {csm['muzzle_peak_1m_db']:.1f} dB")
            cm3.metric("Crew Rear 140 dB Standoff", f"{csm['crew_rear_140db_standoff_m']:.1f} m", delta=f"Direct Cone: {csm['direct_fire_cone_range_m']:.0f} m")
            cm4.metric("Falling Lethal (>15 J) Share", f"{csm['lethal_15j_pct']:.1f}%", delta=f"Injury (>=2 J): {csm['injury_2j_pct']:.1f}%")
            if csm["verdict"] == "RESTRICTED_ELEVATION_OR_AMMO":
                st.error(csm["verdict_note"])
            else:
                st.info(csm["verdict_note"])
            st.plotly_chart(create_collateral_safety_envelope_figure(collat_res), width="stretch", key="cuas18_sd_collat_fig")
            st.dataframe(pd.DataFrame(collat_res["tier_breakdown"]), width="stretch", hide_index=True)

            dual_h = collat_res.get("dual_criterion_hazards") or {}
            deb_h = collat_res.get("falling_drone_debris") or {}
            doc_ov = collat_res.get("doctrine_reference_overlays") or {}
            if dual_h or deb_h:
                with st.expander("⚖️ Dual RCC 321-23 / Kneubuehl Criteria, Falling Debris & Doctrine Overlays", expanded=False):
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.markdown("**Pellet Energy & Specific Energy Density Thresholds**")
                        st.dataframe(
                            pd.DataFrame([
                                {"Standard / Criterion": "RCC 321-23 Blunt Casualty (>=14.91 J)", "Share (%)": dual_h.get("rcc_321_23_blunt_casualty_14_91j_pct", 0.0)},
                                {"Standard / Criterion": "RCC 321-23 Fatality Standing (>=33.89 J)", "Share (%)": dual_h.get("rcc_321_23_fatality_standing_33_89j_pct", 0.0)},
                                {"Standard / Criterion": "RCC 321-23 Penetration Casualty (>=7.145 J/cm²)", "Share (%)": dual_h.get("rcc_321_23_penetration_casualty_7_145j_cm2_pct", 0.0)},
                                {"Standard / Criterion": "Kneubuehl Skin Penetration (>=10.0 J/cm²)", "Share (%)": dual_h.get("kneubuehl_skin_10j_cm2_pct", 0.0)},
                                {"Standard / Criterion": "Kneubuehl Ocular Hazard (>=3.0 J/cm²)", "Share (%)": dual_h.get("kneubuehl_eye_3j_cm2_pct", 0.0)},
                            ]),
                            width="stretch",
                            hide_index=True,
                        )
                    with d_c2:
                        st.markdown("**Falling Drone Debris (SE-F04) & Doctrine Reference Angles**")
                        st.metric("1.5 kg Drone Impact Energy (35 m)", f"{deb_h.get('impact_energy_j', 0.0):.1f} J", delta=deb_h.get("threshold_status", ""))
                        ar_ov = doc_ov.get("ar_385_63_table_5_1", {})
                        uk_ov = doc_ov.get("uk_jsp_403_vol2_ch2_table3", {})
                        st.caption(
                            f"**US Army AR 385-63 Table 5-1:** {ar_ov.get('class_note', '')} | "
                            f"**UK JSP 403 Vol 2 Ch 2 Table 3:** +{uk_ov.get('lateral_cone_mils', 80):.0f} mils lateral azimuth, "
                            f"+{uk_ov.get('vertical_cone_mils', 200):.0f} mils vertical elevation."
                        )
                        st.warning(deb_h.get("uxo_post_kill_notice", ""))
                    st.caption(collat_res.get("scope_notice", "Engineering scoping aid only."))

            from ammo_mc.cuas_rigor import get_contextual_counter_evidence, resolve_verification_grade

            render_verification_and_counter_evidence(
                resolve_verification_grade({"ballistic_fallout": "V1", "injury_thresholds": "V1", "post_kill_debris": "V2"}),
                get_contextual_counter_evidence(stage="sdz"),
                key_prefix="stage3_sdz",
            )

    # --- TAB 18.6: MULTI-SENSOR DETECTION, CLASSIFICATION & KILL-CHAIN TIMELINE ---
    elif active_cuas_mod == _CUAS18_MODULES[5]:
        st.markdown("#### 🛰️ Multi-Sensor Detection, Human-in-the-Loop & Engagement Kill-Chain Timeline")
        _render_cuas_profile_sync_bar(
            "kill_chain",
            {
                "drone_model": ("cuas18_kc_dr", list(drone_specs_18.keys())),
                "aspect_mode": ("cuas18_kc_asp", list(aspect_presets_18.keys())),
                "drone_speed_mps": ("cuas18_kc_spd", (8.0, 50.0)),
                "ammunition_preset": ("cuas18_kc_ammo", list(sg_presets_18.keys())),
            },
        )
        sensor_cat = get_kill_chain_sensor_catalog()
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.markdown("**Threat & Environment**")
            kc_drone = st.selectbox("Threat Drone Class", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cuas18_kc_dr")
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

        with kc2:
            st.markdown("**Sensor Suite & Release Gate**")
            kc_sensors = st.multiselect(
                "Active Acquisition Modalities",
                list(sensor_cat.keys()),
                default=list(sensor_cat.keys()),
                format_func=lambda k: sensor_cat[k]["short_label"],
                key="cuas18_kc_sens",
            )
            kc_hitl = st.checkbox("Require Human-in-the-Loop Release Auth", value=True, key="cuas18_kc_hitl")

        with kc3:
            st.markdown("**Turret Effector Loadout**")
            kc_ammo = st.selectbox("Shotgun Payload", list(sg_presets_18.keys()), format_func=lambda k: sg_presets_18[k]["name"], index=list(sg_presets_18.keys()).index("12G_NO4_BUCK_C_UAS") if "12G_NO4_BUCK_C_UAS" in sg_presets_18 else 0, key="cuas18_kc_ammo")

        with st.expander("C2 Delays, Turret Slew & Stealth Threat Options", expanded=expert):
            ka1, ka2, ka3 = st.columns(3)
            with ka1:
                kc_aspect = st.selectbox("Approach Aspect", list(aspect_presets_18.keys()), format_func=lambda k: aspect_presets_18[k]["name"], index=0, key="cuas18_kc_asp")
                kc_wind = st.slider("Ambient Wind Noise (m/s)", 0.0, 20.0, 5.0, 1.0, key="cuas18_kc_wind")
            with ka2:
                kc_fiber = st.checkbox(
                    "Fiber-Optic / Autonomous Inertial Threat (RF Silent)",
                    value=False,
                    key="cuas18_kc_fiber",
                    help="Drops passive RF / C2 ESM range to 0 m.",
                )
                kc_ai_t = st.slider("AI Track Classification Delay (s)", 0.1, 2.5, 0.6, 0.1, key="cuas18_kc_ai")
                kc_auth_t = st.slider("Operator Authorization Latency (s)", 0.2, 4.0, 1.2, 0.1, key="cuas18_kc_auth", disabled=not kc_hitl)
            with ka3:
                kc_slew_ang = st.slider("Initial Off-Boresight Slew Angle (°)", 0.0, 180.0, 60.0, 10.0, key="cuas18_kc_sang")
                kc_slew_rate = st.slider("Turret Slew Rate (°/s)", 30.0, 220.0, 120.0, 10.0, key="cuas18_kc_sr")
                kc_rmax = st.slider("Max Weapon Intercept Range (m)", 30.0, 100.0, 55.0, 5.0, key="cuas18_kc_rmax")
                kc_rmin = st.slider("Asset Keep-Out Breach Radius (m)", 5.0, 30.0, 15.0, 1.0, key="cuas18_kc_rmin")

        _prof_kc = st.session_state.get("_active_cuas_profile", {})
        kc_n_b = int(_prof_kc.get("n_barrels", 6)) if isinstance(_prof_kc, dict) else 6
        kc_choke_eff = str(_prof_kc.get("choke", "Modified")) if isinstance(_prof_kc, dict) else "Modified"

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
                n_barrels=kc_n_b,
                ammunition_preset=str(kc_ammo),
                choke=kc_choke_eff,
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
        handoff_bs = st.session_state.get("_adversarial_raid_handoff")
        use_handoff = False
        if handoff_bs is not None:
            hc1, hc2 = st.columns([3.0, 1.0])
            with hc1:
                use_handoff = st.checkbox(
                    f"🚀 Use Active Nash Battlespace Handoff ({handoff_bs.get('label', 'Evolved Site')})",
                    value=True,
                    key="cuas18_raid_use_handoff",
                )
            with hc2:
                if st.button("✖ Clear Handoff", key="cuas18_raid_clear_handoff", width="stretch"):
                    st.session_state.pop("_adversarial_raid_handoff", None)
                    st.rerun()

        use_m7 = False
        if saved_site is not None and not use_handoff:
            use_m7 = st.checkbox(
                f"Use Active Site & DEM Layout Configured in Module 7 ({saved_site.get('site_preset', 'custom')})",
                value=True,
                key="cuas18_raid_use_m7",
            )

        swarm_pkg_json = ""
        r_outer_layer = "none"
        rd1, rd2, rd3 = st.columns(3)
        with rd1:
            if use_handoff and handoff_bs is not None:
                r_site = str(handoff_bs.get("site_preset", "fob_compound_4_turret"))
                r_tur_json = json.dumps(handoff_bs.get("custom_turrets", []))
                r_obs_json, r_nfz_json = "", ""
                t_pk = str(handoff_bs.get("terrain_preset", "rolling_hills_fob"))
                r_dem_obj = None if t_pk == "flat_airfield" else build_preset_terrain_dem(t_pk)
                r_dem_json = r_dem_obj.to_json() if r_dem_obj is not None else ""
                swarm_pkg_json = json.dumps(handoff_bs.get("swarm_package", []))
                r_outer_layer = str(handoff_bs.get("outer_layer_type", "none"))
            elif use_m7 and saved_site is not None:
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
            r_n_drones = st.slider("Inbound Swarm Size", 4, 24, int(handoff_bs.get("n_drones", 10)) if (use_handoff and handoff_bs) else 10, 1, key="cuas18_raid_nd")
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
            ammo_keys_12 = list(sg_presets_18.keys())
            def_ammo_12 = str(handoff_bs.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")) if (use_handoff and handoff_bs) else "12G_NO4_BUCK_C_UAS"
            r_ammo = st.selectbox(
                "Turret Ammunition Load",
                ammo_keys_12,
                format_func=lambda k: sg_presets_18[k]["name"],
                index=ammo_keys_12.index(def_ammo_12) if def_ammo_12 in ammo_keys_12 else 0,
                key="cuas18_raid_ammo",
            )
            r_mag = st.selectbox(
                "Magazine / Feed System",
                list(mag_cat_12.keys()),
                format_func=lambda k: mag_cat_12[k]["name"],
                index=1,
                key="cuas18_raid_mag",
            )
            r_speed = st.slider("Swarm Speed (m/s)", 10.0, 42.0, float(handoff_bs.get("threat_speed_mps", 22.0)) if (use_handoff and handoff_bs) else 22.0, 1.0, key="cuas18_raid_spd")
            r_rmax = st.slider("Turret Engagement Range (m)", 35.0, 95.0, float(handoff_bs.get("weapon_max_range_m", 65.0)) if (use_handoff and handoff_bs) else 65.0, 5.0, key="cuas18_raid_rmax")
            if is_expert_mode():
                outer_opts_12 = ["none", "rf_gnss_jammer", "hpm_screen", "airburst_35mm"]
                r_outer_layer = st.selectbox(
                    "Outer Defense Layer",
                    outer_opts_12,
                    index=outer_opts_12.index(r_outer_layer) if r_outer_layer in outer_opts_12 else 0,
                    key="cuas18_raid_outer",
                )

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
                evasion_profile=str(handoff_bs.get("evasion_profile", r_evasion)) if (use_handoff and handoff_bs) else str(r_evasion),
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
                swarm_package_json=swarm_pkg_json,
                outer_layer_type=str(r_outer_layer),
                calibration_profile_json=_get_effective_cal_twin_json(),
                physics_fidelity_json=_get_active_physics_fidelity_json(),
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


