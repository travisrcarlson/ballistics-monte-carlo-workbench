"""Shared C-UAS UI helper functions, profile sync bars, and CAD hardware tabs."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ammo_mc.cuas_hardware import (
    CARRIER_OTM_PROFILES,
    GYRO_STABILIZATION_GRADES,
    estimate_turret_swap_c,
    evaluate_muzzle_plate_fea,
    export_muzzle_plate_dxf,
    export_muzzle_plate_stl,
    export_muzzle_plate_svg,
)
from ammo_mc.small_arms import (
    compute_annex_c_analytical_pk,
    generate_multishotgun_array_layout,
)
from ammo_mc.ui.common import (
    _cached_antidrone_range_envelope,
    _cached_antidrone_range_speed_heatmap,
    _cached_otm_stabilization,
    _navigate_to,
    get_active_calibration_profile,
    get_active_physics_fidelity,
    get_effective_calibration_profile,
)
from ammo_mc.vis_cuas import (
    create_antidrone_effective_range_figure,
    create_antidrone_range_speed_heatmap_figure,
)
from ammo_mc.vis_cuas_advanced import create_swap_c_and_otm_figure


def _get_cal_twin_json() -> str:
    twin = get_active_calibration_profile()
    return json.dumps(twin, sort_keys=True) if twin else ""


def _get_effective_cal_twin_json() -> str:
    twin = get_effective_calibration_profile()
    return json.dumps(twin, sort_keys=True) if twin else ""


def _get_active_physics_fidelity_json() -> str:
    pf = get_active_physics_fidelity()
    return json.dumps(pf, sort_keys=True) if pf else ""


def render_high_fidelity_active_banner() -> None:
    """Display compact status banner when global High-Fidelity Physics Mode is active."""
    pf = get_active_physics_fidelity()
    if pf:
        st.caption(
            "🔬 **Global High-Fidelity Physics Active:** Rotor solidity, $V_{50}(\\theta)$ oblique ricochet, "
            "ray shielding, shot-cup separation (`2.5 m`), and core-flyer dispersion synced across stages."
        )


def _apply_stage1_fix_callback(patch: dict[str, Any], label: str = "Calibrated Design Fix") -> None:
    """Apply a recommended design compensation patch to Stage 1 controls cleanly via callback."""
    if "layout_type" in patch:
        st.session_state["cuas_layout"] = str(patch["layout_type"])
    if "n_barrels" in patch:
        st.session_state["cuas_n_barrels"] = int(patch["n_barrels"])
    if "ammunition_preset" in patch:
        st.session_state["cuas_ammo"] = str(patch["ammunition_preset"])
    if "choke" in patch or "default_choke" in patch:
        st.session_state["cuas_choke"] = str(patch.get("choke", patch.get("default_choke", "Modified")))
    if "pointing_mode" in patch:
        st.session_state["cuas_pointing"] = str(patch["pointing_mode"])
    if "splay_angle_mrad" in patch:
        st.session_state["cuas_splay"] = float(patch["splay_angle_mrad"])
    if "firing_mode" in patch:
        st.session_state["cuas_fire_mode"] = str(patch["firing_mode"])
    if "hybrid_mode" in patch:
        st.session_state["cuas_mixed_loadout"] = bool(patch["hybrid_mode"])
    if patch.get("inner_ammunition_preset"):
        st.session_state["cuas_inner_ammo"] = str(patch["inner_ammunition_preset"])
    if patch.get("inner_choke"):
        st.session_state["cuas_inner_choke"] = str(patch["inner_choke"])
    prof = dict(st.session_state.get("_active_cuas_profile") or {})
    prof.update(patch)
    prof["label"] = label
    st.session_state["_active_cuas_profile"] = prof
    _navigate_to("cuas", None, "array")

_SITE_GEO_ANCHORS: dict[str, tuple[float, float]] = {
    "Desert Test Range (32.9000°N, -106.4000°W)": (32.9000, -106.4000),
    "Coastal Airfield (36.8960°N, -76.2990°W)": (36.8960, -76.2990),
    "European Urban Site (48.8566°N, 2.3522°E)": (48.8566, 2.3522),
}


def _apply_active_cuas_profile(
    field_map: dict[str, tuple[str, Any]],
    only_if_missing: bool = False,
) -> None:
    """Copy validated parameters from Stage 1 active array profile into downstream widget keys."""
    prof = st.session_state.get("_active_cuas_profile")
    if not isinstance(prof, dict):
        return
    for prof_key, (widget_key, bounds) in field_map.items():
        if prof_key not in prof:
            continue
        if only_if_missing and widget_key in st.session_state:
            continue
        val = prof[prof_key]
        if isinstance(bounds, (list, tuple)) and bounds and isinstance(bounds[0], str):
            if val in bounds:
                st.session_state[widget_key] = val
        elif isinstance(bounds, tuple) and len(bounds) == 2:
            lo, hi = bounds
            try:
                clamped = max(lo, min(hi, int(round(float(val))) if isinstance(lo, int) else float(val)))
                st.session_state[widget_key] = clamped
            except (TypeError, ValueError):
                pass


def _render_cuas_profile_sync_bar(
    prefix: str,
    field_map: dict[str, tuple[str, Any]],
) -> None:
    """Auto-seed unset downstream widgets once and render a 1-click Stage 1 sync bar."""
    prof = st.session_state.get("_active_cuas_profile")
    if not isinstance(prof, dict):
        return
    seed_flag = f"_cuas_prof_seeded_{prefix}"
    if not st.session_state.get(seed_flag):
        _apply_active_cuas_profile(field_map, only_if_missing=True)
        st.session_state[seed_flag] = True

    c_info, c_btn = st.columns([3.4, 1.3])
    c_info.info(f"Stage 1 Active Profile: **{prof.get('label', 'Configured Array')}**")
    c_btn.button(
        "Sync Stage 1 Array & Drone",
        key=f"cuas_sync_btn_{prefix}",
        on_click=_apply_active_cuas_profile,
        args=(field_map, False),
        width="stretch",
    )


def _render_effective_range_tab(
    *,
    layout_type: str,
    n_barrels: int,
    mount_radius_m: float,
    spacing_y_m: float,
    spacing_z_m: float,
    n_cols: int,
    n_rows: int,
    pointing_mode: str,
    splay_angle_mrad: float,
    firing_mode: str,
    ripple_delay_ms: float,
    custom_barrel_specs_json: str,
    ammunition_preset: str,
    default_choke: str,
    default_barrel_length_in: float,
    drone_model: str,
    drone_velocity_mps: float,
    drone_heading_deg: float,
    aspect_mode: str,
    drone_pitch_deg: float,
    drone_yaw_deg: float,
    drone_roll_deg: float,
    aim_error_mrad: float,
    crosswind_mps: float,
    inner_ammunition_preset: str | None,
    outer_ammunition_preset: str | None,
    inner_choke: str | None,
    custom_drone_spec_json: str,
    custom_shell_spec_json: str,
    sg_presets: dict[str, Any],
    headwind_mps: float = 0.0,
    ambient_temp_c: float = 15.0,
    propellant_type: str = "double_base",
    drag_model: str = "compressible_re_m",
    pellets_per_shot: int | None = None,
    use_rotor_solidity: bool = False,
    use_v50_ballistic_limit: bool = False,
    enable_ray_shielding: bool = False,
    use_oblique_ricochet: bool = False,
    shot_cup_separation_m: float = 0.0,
    pattern_distribution: str = "gaussian",
    flyer_fraction: float = 0.12,
    annex_c_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render Stage 1 Effective Range Curve (R80 / R50), Annex C table, and optional Speed x Range Heatmap."""
    st.markdown("##### Effective Engagement Envelope (P_kill vs Slant Range)")
    alt_candidates = [
        k
        for k in sg_presets.keys()
        if k
        in {
            "12G_NO4_BUCK_C_UAS",
            "12G_TSS_NO2_C_UAS",
            "12G_DUPLEX_TSS_BUCK_C_UAS",
            "12G_FLECHETTE_SABOT_C_UAS",
            "ALDA_BUCKSHOT_12_76_COPPER_32G",
            "ALDA_DOUBLE_00_HP_N2_0",
            "ALDA_0_80_HP_N2",
            "SPARTAN_43_HP_1276_SPHERO_3T",
            "SPARTAN_33_HP_SPHERO_4T",
        }
    ]
    comp_loads = st.multiselect(
        "Overlay Comparison Shell Loads (Optional)",
        alt_candidates,
        default=[],
        format_func=lambda k: sg_presets[k]["name"],
        key="cuas_env_comp_loads",
    )
    stations = (15.0, 25.0, 35.0, 50.0, 65.0, 75.0, 85.0)
    with st.spinner("Computing effective range envelope P_kill(R)..."):
        env_res = _cached_antidrone_range_envelope(
            distance_stations_m=stations,
            layout_type=layout_type,
            n_barrels=n_barrels,
            mount_radius_m=mount_radius_m,
            spacing_y_m=spacing_y_m,
            spacing_z_m=spacing_z_m,
            n_cols=n_cols,
            n_rows=n_rows,
            pointing_mode=pointing_mode,
            splay_angle_mrad=splay_angle_mrad,
            firing_mode=firing_mode,
            ripple_delay_ms=ripple_delay_ms,
            custom_barrel_specs_json=custom_barrel_specs_json,
            ammunition_preset=ammunition_preset,
            default_choke=default_choke,
            default_barrel_length_in=default_barrel_length_in,
            drone_model=drone_model,
            drone_velocity_mps=drone_velocity_mps,
            drone_heading_deg=drone_heading_deg,
            aspect_mode=aspect_mode,
            drone_pitch_deg=drone_pitch_deg,
            drone_yaw_deg=drone_yaw_deg,
            drone_roll_deg=drone_roll_deg,
            aim_error_mrad=aim_error_mrad,
            crosswind_mps=crosswind_mps,
            n_volleys=12,
            seed=42,
            inner_ammunition_preset=inner_ammunition_preset,
            outer_ammunition_preset=outer_ammunition_preset,
            inner_choke=inner_choke,
            custom_drone_spec_json=custom_drone_spec_json,
            custom_shell_spec_json=custom_shell_spec_json,
            compare_ammo_presets=tuple(comp_loads),
            calibration_profile_json=_get_cal_twin_json(),
            headwind_mps=headwind_mps,
            ambient_temp_c=ambient_temp_c,
            propellant_type=propellant_type,
            drag_model=drag_model,
            pellets_per_shot=pellets_per_shot,
            use_rotor_solidity=use_rotor_solidity,
            use_v50_ballistic_limit=use_v50_ballistic_limit,
            enable_ray_shielding=enable_ray_shielding,
            use_oblique_ricochet=use_oblique_ricochet,
            shot_cup_separation_m=shot_cup_separation_m,
            pattern_distribution=pattern_distribution,
            flyer_fraction=flyer_fraction,
        )
    st.session_state["_report_range_envelope"] = env_res
    esm = env_res["summary"]
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Effective Range R80 (P_kill >= 80%)", esm["r80_label"], delta="Doctrinal intercept ceiling")
    ec2.metric("Effective Range R50 (P_kill >= 50%)", esm["r50_label"], delta="50% damage / harass limit")
    ec3.metric("Peak Single-Volley P_kill", f"{esm['max_pk_pct']:.1f} %", delta=esm["ammunition_name"][:34])
    st.plotly_chart(create_antidrone_effective_range_figure(env_res), width="stretch", key="cuas_env_range_fig")
    st.dataframe(pd.DataFrame(env_res["records"]), width="stretch")

    if annex_c_sample:
        with st.expander("📐 Annex C Poisson-Gaussian Analytical Pk Benchmark (vs. Monte Carlo)", expanded=False):
            ac_rows = []
            for d_pt in (20.0, 30.0, 40.0, 50.0, 75.0, 100.0):
                ac_pt = compute_annex_c_analytical_pk(
                    distance_m=d_pt,
                    n_pellets=int(annex_c_sample["n_pellets"]),
                    vulnerable_area_m2=float(annex_c_sample["vulnerable_area_m2"]),
                    theta_p_rad=float(annex_c_sample["theta_p_rad"]),
                    theta_aim_rad=float(annex_c_sample["theta_aim_rad"]),
                    pk_given_hit=float(annex_c_sample["pk_given_hit"]),
                    n_shots=3,
                )
                ac_rows.append(
                    {
                        "Distance (m)": int(d_pt),
                        "Pattern σ_p (m)": ac_pt["sigma_p_m"],
                        "Aim σ_aim (m)": ac_pt["sigma_aim_m"],
                        "Expected Hits λ": ac_pt["expected_hits_lambda"],
                        "Single-Shot Pk,1 (%)": ac_pt["pk_1_pct"],
                        "3-Shot Burst Pk,3 (%)": ac_pt["pk_n_pct"],
                    }
                )
            st.dataframe(pd.DataFrame(ac_rows), width="stretch", hide_index=True)

    st.divider()
    show_hm = st.checkbox(
        "Compute 2D Speed x Range Heatmap (5x6 Grid)",
        value=False,
        key="cuas_show_heatmap",
        help="Sweeps 5 crossing speeds (5-50 m/s) across 6 slant ranges (15-75 m).",
    )
    if show_hm:
        with st.spinner("Computing 2D Speed x Slant-Range P_kill Heatmap..."):
            hm_res = _cached_antidrone_range_speed_heatmap(
                distance_stations_m=(15.0, 25.0, 35.0, 45.0, 60.0, 75.0),
                speed_stations_mps=(5.0, 15.0, 25.0, 38.0, 50.0),
                layout_type=layout_type,
                n_barrels=n_barrels,
                mount_radius_m=mount_radius_m,
                spacing_y_m=spacing_y_m,
                spacing_z_m=spacing_z_m,
                n_cols=n_cols,
                n_rows=n_rows,
                pointing_mode=pointing_mode,
                splay_angle_mrad=splay_angle_mrad,
                firing_mode=firing_mode,
                ripple_delay_ms=ripple_delay_ms,
                custom_barrel_specs_json=custom_barrel_specs_json,
                ammunition_preset=ammunition_preset,
                default_choke=default_choke,
                default_barrel_length_in=default_barrel_length_in,
                drone_model=drone_model,
                drone_heading_deg=drone_heading_deg,
                aspect_mode=aspect_mode,
                drone_pitch_deg=drone_pitch_deg,
                drone_yaw_deg=drone_yaw_deg,
                drone_roll_deg=drone_roll_deg,
                aim_error_mrad=aim_error_mrad,
                crosswind_mps=crosswind_mps,
                n_volleys=10,
                seed=42,
                inner_ammunition_preset=inner_ammunition_preset,
                outer_ammunition_preset=outer_ammunition_preset,
                inner_choke=inner_choke,
                custom_drone_spec_json=custom_drone_spec_json,
                custom_shell_spec_json=custom_shell_spec_json,
                calibration_profile_json=_get_cal_twin_json(),
                headwind_mps=headwind_mps,
                ambient_temp_c=ambient_temp_c,
                propellant_type=propellant_type,
                drag_model=drag_model,
                pellets_per_shot=pellets_per_shot,
                use_oblique_ricochet=use_oblique_ricochet,
                shot_cup_separation_m=shot_cup_separation_m,
            )
        st.plotly_chart(create_antidrone_range_speed_heatmap_figure(hm_res), width="stretch")
        st.dataframe(pd.DataFrame(hm_res["r80_by_speed_m"]), width="stretch")
    return env_res


def _render_stage1_hardware_cad_tab(
    layout_type: str,
    n_barrels: int,
    mount_radius_m: float,
    spacing_y_m: float,
    spacing_z_m: float,
    n_cols: int,
    n_rows: int,
    ammunition_preset: str,
    default_choke: str,
    default_barrel_length_in: float,
    pointing_mode: str,
    splay_angle_mrad: float,
    firing_mode: str,
    ripple_delay_ms: float,
    drone_model: str,
    crosswind_mps: float,
    hybrid_mode: bool,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
) -> None:
    """Render Stage 1 SWaP-C mass/power/BOM budget, OTM stabilization, bolt-shear FEA, and 2D/3D CAD."""
    cal_prof_hw = get_active_calibration_profile()
    cal_json_hw = json.dumps(cal_prof_hw, sort_keys=True) if cal_prof_hw else ""
    swap_res = estimate_turret_swap_c(
        n_barrels=int(n_barrels),
        barrel_length_in=float(default_barrel_length_in),
        layout_type=str(layout_type),
        mount_radius_m=float(mount_radius_m),
        spacing_y_m=float(spacing_y_m),
        spacing_z_m=float(spacing_z_m),
        ammunition_preset=str(ammunition_preset),
        firing_mode=str(firing_mode),
        ripple_delay_ms=float(ripple_delay_ms),
        hybrid_mode=bool(hybrid_mode),
        calibration_profile=cal_prof_hw,
    )
    sw = swap_res["summary"]
    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("Total Turret Mass", f"{sw['total_mass_kg']:.1f} kg", delta=sw["recommended_mount_class"][:28])
    hm2.metric("Peak Servo / Bus Power", f"{sw['peak_power_w']:.0f} W", delta=f"{sw['bus_current_24v_a']:.1f} A @ 24V | Torque {sw['peak_torque_nm']:.0f} Nm")
    hm3.metric("Peak Recoil Mount Load", f"{sw['peak_recoil_force_kn']:.2f} kN", delta=f"Inertia Az/El: {sw['inertia_az_kg_m2']:.2f} kg·m²")
    hm4.metric("Hardware BOM Estimate", f"${sw['unit_bom_usd']:,.0f}", delta=f"Standby: {sw['standby_power_w']:.0f} W")

    st.dataframe(pd.DataFrame(swap_res["compatibility_table"]), width="stretch", hide_index=True)

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        otm_carrier = st.selectbox(
            "Carrier Platform",
            list(CARRIER_OTM_PROFILES.keys()),
            format_func=lambda k: CARRIER_OTM_PROFILES[k]["name"],
            index=2,
            key="cuas_hw_carrier",
        )
    with oc2:
        otm_gyro = st.selectbox(
            "Gimbal Stabilization Grade",
            list(GYRO_STABILIZATION_GRADES.keys()),
            format_func=lambda k: GYRO_STABILIZATION_GRADES[k]["name"],
            index=1,
            key="cuas_hw_gyro",
        )
    with oc3:
        otm_wx = st.selectbox("Weather Regime", ["clear_day", "overcast_dusk", "night_clear", "heavy_fog_rain"], key="cuas_hw_wx")
    with oc4:
        otm_temp = st.slider("Ambient Propellant Temp (°C)", -20.0, 50.0, 21.1, 2.0, key="cuas_hw_temp")

    otm_cfg_json = json.dumps(
        {
            "n_barrels": int(n_barrels),
            "barrels_per_burst": min(int(n_barrels), 3),
            "ammunition_preset": str(ammunition_preset),
            "choke": str(default_choke),
        },
        sort_keys=True,
    )
    otm_res = _cached_otm_stabilization(
        turret_config_json=otm_cfg_json,
        carrier_profile=str(otm_carrier),
        gyro_grade=str(otm_gyro),
        weather=str(otm_wx),
        ambient_temp_c=float(otm_temp),
        crosswind_mps=float(crosswind_mps),
        drone_model=str(drone_model if drone_model != "CUSTOM_DRONE" else "fpv_quad_small"),
        calibration_profile_json=cal_json_hw,
    )
    st.caption(
        f"Active LOS Stabilization Jitter: **{otm_res['residual_los_jitter_mrad']:.2f} mrad** | "
        f"Static R80: **{otm_res['r80_static_m']:.1f} m** -> Active OTM R80: **{otm_res['r80_active_m']:.1f} m** "
        f"(-{otm_res['range_loss_pct']:.1f}% range reduction)"
    )
    st.plotly_chart(create_swap_c_and_otm_figure(swap_res, otm_res), width="stretch", key="cuas_swap_otm_fig")

    st.divider()
    st.markdown("##### 📐 Parametric Muzzle-Clamp Blueprint (`.dxf` / `.svg` / 3D `.stl`) & Bolt-Shear FEA")
    barrels_cad = generate_multishotgun_array_layout(
        layout_type=str(layout_type),
        n_barrels=int(n_barrels),
        mount_radius_m=float(mount_radius_m),
        spacing_y_m=float(spacing_y_m),
        spacing_z_m=float(spacing_z_m),
        n_cols=int(n_cols),
        n_rows=int(n_rows),
        default_choke=str(default_choke),
        default_barrel_length_in=float(default_barrel_length_in),
        custom_specs=custom_barrel_specs,
        default_ammunition_preset=str(ammunition_preset),
    )
    fea_c1, fea_c2, fea_c3 = st.columns(3)
    with fea_c1:
        plate_thk_mm = st.slider("Clamp Plate Thickness (mm)", 10.0, 40.0, 22.0, 2.0, key="cuas_fea_thk")
    with fea_c2:
        plate_mat = st.selectbox("Clamp Alloy", ["7075-T6", "6061-T6", "4140-Steel"], index=0, key="cuas_fea_mat")
    with fea_c3:
        bolt_grd = st.selectbox("8x M8 Fastener Grade", ["8.8", "10.9", "12.9"], index=0, key="cuas_fea_bolt")

    fea_res = evaluate_muzzle_plate_fea(
        barrels=barrels_cad,
        layout_type=str(layout_type),
        peak_recoil_force_kn=float(sw["peak_recoil_force_kn"]),
        plate_thickness_mm=float(plate_thk_mm),
        material=str(plate_mat),
        bolt_grade=str(bolt_grd),
    )
    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("Structural FEA Verdict", fea_res["verdict"], delta=f"Min SF: {fea_res['overall_safety_factor']:.2f}×")
    fm2.metric("M8 Bolt Stress SF", f"{fea_res['sf_bolt']:.2f}×", delta=f"{fea_res['bolt_von_mises_mpa']:.0f} MPa vM")
    fm3.metric("Plate Bearing/Bend SF", f"{fea_res['sf_plate']:.2f}×", delta=f"{fea_res['plate_combined_stress_mpa']:.0f} MPa")
    fm4.metric("Min Ligament Wall", f"{fea_res['min_ligament_mm']:.1f} mm", delta=f"Clamp Mass: {fea_res['clamp_mass_kg']:.2f} kg")
    if fea_res["verdict"] == "PASS":
        st.success(fea_res["note"])
    elif fea_res["verdict"] == "MARGINAL":
        st.warning(fea_res["note"])
    else:
        st.error(fea_res["note"])

    svg_txt = export_muzzle_plate_svg(
        barrels=barrels_cad,
        layout_type=str(layout_type),
        title=f"{int(n_barrels)}B {layout_type.title()} Muzzle Clamp",
        splay_angle_mrad=float(splay_angle_mrad),
        pointing_mode=str(pointing_mode),
    )
    dxf_txt = export_muzzle_plate_dxf(
        barrels=barrels_cad,
        layout_type=str(layout_type),
        title=f"{int(n_barrels)}B {layout_type.title()} Muzzle Clamp",
        splay_angle_mrad=float(splay_angle_mrad),
        pointing_mode=str(pointing_mode),
    )
    stl_txt = export_muzzle_plate_stl(
        barrels=barrels_cad,
        layout_type=str(layout_type),
        plate_thickness_mm=float(plate_thk_mm),
        title=f"cuas_{layout_type}_{int(n_barrels)}bbl_clamp",
    )
    st.markdown(svg_txt, unsafe_allow_html=True)
    cad_c1, cad_c2, cad_c3 = st.columns(3)
    cad_c1.download_button(
        "📥 2D Manufacturing Drawing (.dxf)",
        data=dxf_txt.encode("utf-8"),
        file_name=f"cuas_{layout_type}_{n_barrels}bbl_muzzle_plate.dxf",
        mime="application/dxf",
        width="stretch",
        key="cuas_stage1_dl_dxf",
    )
    cad_c2.download_button(
        "📥 2D Vector Blueprint (.svg)",
        data=svg_txt.encode("utf-8"),
        file_name=f"cuas_{layout_type}_{n_barrels}bbl_muzzle_plate.svg",
        mime="image/svg+xml",
        width="stretch",
        key="cuas_stage1_dl_svg",
    )
    cad_c3.download_button(
        "🧊 3D CAD Solid (.stl)",
        data=stl_txt.encode("utf-8"),
        file_name=f"cuas_{layout_type}_{n_barrels}bbl_muzzle_plate.stl",
        mime="model/stl",
        width="stretch",
        key="cuas_stage1_dl_stl",
    )


def _launch_battlespace_to_raid(payload: dict[str, Any]) -> None:
    """Store 3D Nash battlespace handoff payload and navigate directly into Raid Simulator."""
    st.session_state["_adversarial_raid_handoff"] = payload
    _navigate_to("cuas", None, "raid")


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



def _apply_quincunx_mixed_preset() -> None:
    """Configure Stage 1 for 5-barrel Quincunx mixed-loadout evaluation (4x ALDA 6.2mm + 1x SPARTAN 4T)."""
    st.session_state["cuas_layout"] = "quincunx"
    st.session_state["cuas_qn_sp_cm"] = 30.0
    st.session_state["cuas_ammo"] = "ALDA_BUCKSHOT_12_76_COPPER_32G"
    st.session_state["cuas_choke"] = "Modified"
    st.session_state["cuas_mixed_loadout"] = True
    st.session_state["cuas_inner_ammo"] = "SPARTAN_33_HP_SPHERO_4T"
    st.session_state["cuas_inner_choke"] = "Cylinder"
    st.session_state["cuas_drone_model"] = "fpv_14in_36cm_pg7m"
    st.session_state["cuas_dist"] = 50.0
    st.session_state["cuas_speed"] = 0.0
    st.session_state["cuas_cust_drone_on"] = False
    st.session_state["cuas_cust_shell_on"] = False

