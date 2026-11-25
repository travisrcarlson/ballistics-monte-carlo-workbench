"""Stage 4B: Closed-Loop Field Calibration, Digital Twin & Scientific V&V Workbench UI."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ammo_mc.briefing_export import generate_calibration_certificate_pdf
from ammo_mc.calibration import (
    calibrate_chronograph_from_csv,
    calibrate_drag_form_factor_from_csv,
    calibrate_shotgun_pattern_from_csv,
    generate_sample_chronograph_csv,
    generate_sample_pattern_csv,
    generate_sample_range_drop_csv,
)
from ammo_mc.cuas_calibration_twin import (
    CalibrationProfileStore,
    calibrate_hardware_bench_tests,
    calibrate_livefire_drone_lethality,
    calibrate_multirange_pattern_and_drag,
    calibrate_tracker_telemetry,
    detect_fiducial_board_corners,
    digitize_pattern_board_image,
    export_range_test_card_csv,
    generate_printable_fiducial_board_pdf,
    generate_sample_hardware_bench_tests,
    generate_sample_livefire_drone_trials,
    generate_sample_multirange_pattern_rows,
    generate_sample_tracker_test_rows,
    generate_skewed_fiducial_board_png,
    generate_skewed_pattern_board_png_with_corners,
    generate_synthetic_pattern_board_png,
    get_default_nominal_twin,
    merge_twin_updates,
)
from ammo_mc.cuas_defense import get_cuas_sensor_presets
from ammo_mc.cuas_hardware import MOUNT_CLASS_SPECS
from ammo_mc.cuas_scientific_vv import (
    build_reproducible_validation_bundle,
    calibrate_doppler_radar_drag,
    compute_epistemic_aleatory_uq_envelope,
    detect_and_parse_instrument_file,
    evaluate_sprt_live_sequence,
    export_doe_test_card_csv,
    generate_sample_blind_holdout_dataset,
    generate_sample_doppler_radar_csv,
    generate_sample_garmin_xero_csv,
    generate_sample_kestrel_link_csv,
    generate_sample_labradar_native_csv,
    generate_sample_shotstring_csv,
    generate_scientific_vv_dossier_html,
    generate_scientific_vv_dossier_markdown,
    parse_weather_and_shotstring_csv,
    plan_range_validation_doe,
    run_blind_holdout_validation,
    run_experimental_validation_benchmarks,
)
from ammo_mc.cuas_track_replay import (
    generate_sample_drone_track_csv,
    generate_sample_px4_flight_csv,
    parse_ardupilot_px4_flight_log,
)
from ammo_mc.presets import list_presets
from ammo_mc.small_arms import get_choke_definitions
from ammo_mc.ui.common import (
    _cached_active_test_plan,
    _cached_calibration_design_delta,
    _cached_recorded_track_replay,
)
from ammo_mc.vis_calibration import (
    create_chronograph_calibration_figure,
    create_drag_calibration_figure,
    create_pattern_calibration_figure,
)
from ammo_mc.vis_cuas_advanced import (
    create_active_test_planner_figure,
    create_calibration_reliability_and_drift_figure,
    create_recorded_track_replay_figure,
)
from ammo_mc.vis_cuas_vv import (
    create_blind_holdout_validation_figure,
    create_doppler_drag_fit_figure,
    create_epistemic_uq_envelope_figure,
    create_experimental_benchmarks_figure,
    create_pattern_board_corners_overlay_figure,
    create_sprt_stopping_boundaries_figure,
)
from ammo_mc.ui.cuas_pkg.helpers import (
    _apply_stage1_fix_callback,
    _get_active_physics_fidelity_json,
    _get_effective_cal_twin_json,
    render_high_fidelity_active_banner,
)


def render_stage4_calibration(expert: bool, sg_presets_18: dict[str, Any], drone_specs_18: dict[str, Any]) -> None:
    st.caption(
        "4-pillar closed-loop calibration & V&V suite: ingest native range instruments, photos, tracks, and shoot-downs — "
        "then sync into the global Calibrated Digital Twin and verify via ASME V&V 20 & live Wald SPRT."
    )
    render_high_fidelity_active_banner()
    cal_p1, cal_p2, cal_p3, cal_p4 = st.tabs([
        "🎯 1. Ballistics & Range Instruments",
        "💥 2. Targets, Flight Tracks & Hardware",
        "🧬 3. Digital Twin, Drift & Test Planner",
        "🔬 4. Scientific V&V, UQ & Live SPRT",
    ])
    with cal_p1:
        _render_pillar_ballistics(sg_presets_18)
    with cal_p2:
        _render_pillar_tracks_lethality(drone_specs_18)
    with cal_p3:
        _render_pillar_twin_manager(drone_specs_18)
    with cal_p4:
        _render_pillar_scientific_vv(sg_presets_18)


def _render_pillar_ballistics(sg_presets_18: dict[str, Any]) -> None:
    with st.expander("📥 Native Range-Instrument Auto-Import (LabRadar / Garmin Xero C1 / Kestrel LiNK / PX4)", expanded=False):
        st.caption(
            "Upload raw `.csv` / `.xlsx` exports directly from field range hardware — format auto-detection extracts "
            "chronograph MV, ambient air density/wind, Doppler velocity decay, or UAV telemetry into your Digital Twin."
        )
        cal_inst_up = st.file_uploader(
            "Upload Raw Range Instrument File (.csv / .xlsx)",
            type=["csv", "xlsx", "txt"],
            key="cal_inst_up",
        )
        is1, is2, is3, is4 = st.columns(4)
        if is1.button("📋 Sample: LabRadar CSV", key="cal_inst_s_labradar", width="stretch"):
            st.session_state["_cal_inst_payload"] = ("labradar_shot01.csv", generate_sample_labradar_native_csv().encode("utf-8"))
        if is2.button("📋 Sample: Garmin Xero C1", key="cal_inst_s_garmin", width="stretch"):
            st.session_state["_cal_inst_payload"] = ("garmin_xero_c1.csv", generate_sample_garmin_xero_csv().encode("utf-8"))
        if is3.button("📋 Sample: Kestrel LiNK", key="cal_inst_s_kestrel", width="stretch"):
            st.session_state["_cal_inst_payload"] = ("kestrel_link_weather.csv", generate_sample_kestrel_link_csv().encode("utf-8"))
        if is4.button("📋 Sample: PX4 Flight Log", key="cal_inst_s_px4", width="stretch"):
            st.session_state["_cal_inst_payload"] = ("px4_flight_log.csv", generate_sample_px4_flight_csv().encode("utf-8"))

        inst_entry = (cal_inst_up.name, cal_inst_up.getvalue()) if cal_inst_up is not None else st.session_state.get("_cal_inst_payload")
        if inst_entry is not None:
            inst_name, inst_bytes = inst_entry
            try:
                inst_res = detect_and_parse_instrument_file(inst_name, inst_bytes)
                fmt = inst_res["instrument_type"]
                parsed = inst_res["parsed"]
                im_1, im_2, im_3 = st.columns(3)
                im_1.metric("Detected Hardware Format", fmt.upper(), delta=inst_name)
                if parsed.get("mean_v0_mps") is not None:
                    im_2.metric("Mean MV", f"{parsed['mean_v0_mps']:.1f} m/s", delta=f"SD={parsed.get('sd_v0_mps', 0.0):.2f} m/s")
                elif parsed.get("air_density_kg_m3") is not None:
                    im_2.metric("Air Density (ρ)", f"{parsed['air_density_kg_m3']:.3f} kg/m³", delta=f"{parsed.get('ambient_temp_c', 15.0):.1f}°C")
                elif parsed.get("n_points") is not None:
                    im_2.metric("Track Samples", f"{parsed['n_points']}", delta="ENU Telemetry")
                patch = inst_res.get("twin_patch") or {}
                im_3.metric("Digital Twin Fields", f"{len(patch)} ready", delta=", ".join(list(patch.keys())[:2]) or "Track data")
                if parsed.get("rows_preview"):
                    st.dataframe(pd.DataFrame(parsed["rows_preview"]), width="stretch", hide_index=True)
                if st.button("✅ Apply Instrument Patch to Digital Twin", type="primary", key="cal_inst_apply_btn"):
                    if fmt == "px4_ardupilot" and (parsed.get("track_csv") or parsed.get("enu_csv")):
                        st.session_state["_cal_trk_csv"] = parsed.get("track_csv") or parsed.get("enu_csv")
                        st.success("Loaded PX4 ENU trajectory into Track Replay (Pillar 2)!")
                    elif patch:
                        st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                            st.session_state.get("_calibrated_digital_twin"),
                            patch,
                            domain=f"Instrument ({fmt})",
                            summary=f"Auto-imported {inst_name}",
                        )
                        st.session_state["_digital_twin_pending_enable"] = True
                        st.success(f"Applied {fmt.upper()} patch from `{inst_name}` to Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Instrument parse error: {exc}")

    cal_t1, cal_t2, cal_t3, cal_t8 = st.tabs([
        "🎯 Patterning (CSV / Multi-Range / Photo)",
        "📏 Chronograph MV & Temp",
        "🚀 Exterior Drag Form-Factor",
        "📡 Doppler Radar & Shot-String",
    ])

    with cal_t1:
        pat_mode = st.radio(
            "Pattern Input Mode",
            [
                "Single-Range CSV",
                "Multi-Range Spread Exponent & Pellet Drag",
                "Photo-to-Pattern Hole Digitizer (PNG / JPG)",
            ],
            horizontal=True,
            key="cal_pat_mode",
        )
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            pat_rng = st.number_input("Reference Range (m)", 10.0, 100.0, 36.58, 1.0, key="cal_pat_rng")
        with pc2:
            pat_choke = st.selectbox("Nominal Choke", list(get_choke_definitions().keys()), index=3, key="cal_pat_choke")
        with pc3:
            pat_ammo = st.selectbox("Reference Load", list(sg_presets_18.keys()), index=0, key="cal_pat_ammo")

        if pat_mode == "Single-Range CSV":
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
                st.plotly_chart(create_pattern_calibration_figure(pat_res), width="stretch", key="cal_pat_fig")
                if st.button("✅ Apply Patterning Fit & Bias to Digital Twin", type="primary", key="cal_pat_apply_btn"):
                    rad_bias = round(math.hypot(pat_res["poi_bias_y_mrad"], pat_res["poi_bias_z_mrad"]), 2)
                    st.session_state["cuas_choke"] = pat_res["closest_standard_choke"]
                    st.session_state["_calibrated_aim_bias_mrad"] = rad_bias
                    st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                        st.session_state.get("_calibrated_digital_twin"),
                        {
                            "spread_factor": float(pat_res["calibrated_spread_factor"]),
                            "spread_factor_se": round(float(pat_res["sigma_fit_se_m"] / max(pat_res["sigma_nominal_m"], 1e-4)), 3),
                            "poi_bias_y_mrad": float(pat_res["poi_bias_y_mrad"]),
                            "poi_bias_z_mrad": float(pat_res["poi_bias_z_mrad"]),
                            "aim_bias_mrad": rad_bias,
                        },
                        domain="Single-Range Pattern",
                        summary=f"Spread {pat_res['calibrated_spread_factor']:.2f}x, aim bias {rad_bias:.1f} mrad",
                    )
                    st.session_state["_digital_twin_pending_enable"] = True
                    st.success(f"Applied spread={pat_res['calibrated_spread_factor']:.2f}x & enabled Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Pattern calibration error: {exc}")

        elif pat_mode == "Multi-Range Spread Exponent & Pellet Drag":
            st.caption("Edit multi-distance range test measurements directly (or paste from Excel) to jointly fit spread factor, range growth exponent p, and downrange pellet Cd scale.")
            if "_cal_mr_rows" not in st.session_state or st.button("📋 Reset Multi-Range Test Table", key="cal_mr_reset_btn"):
                st.session_state["_cal_mr_rows"] = generate_sample_multirange_pattern_rows(
                    nominal_choke=str(pat_choke),
                    ammunition_preset=str(pat_ammo),
                )
            mr_df = st.data_editor(
                pd.DataFrame(st.session_state["_cal_mr_rows"]),
                num_rows="dynamic",
                width="stretch",
                key="cal_mr_editor",
            )
            try:
                mr_res = calibrate_multirange_pattern_and_drag(
                    mr_df.to_dict(orient="records"),
                    nominal_choke=str(pat_choke),
                    ammunition_preset=str(pat_ammo),
                )
                mm1, mm2, mm3, mm4 = st.columns(4)
                mm1.metric("Spread Factor (k_σ)", f"{mr_res['spread_factor']:.2f}×", delta=f"± {mr_res['spread_factor_se']:.2f} SE")
                mm2.metric("Range Growth Exponent (p)", f"{mr_res['spread_exponent']:.2f}", delta=f"Nominal 1.15 (± {mr_res['spread_exponent_se']:.2f})")
                mm3.metric("Pellet Drag Cd Scale", f"{mr_res['pellet_cd_scale']:.2f}×", delta=f"v RMSE: {mr_res['rmse_velocity_mps']} m/s" if mr_res["rmse_velocity_mps"] is not None else "No velocity column")
                mm4.metric("Sigma Fit R² / RMSE", f"R²={mr_res['r2_spread']:.4f}", delta=f"RMSE: {mr_res['rmse_spread_m']:.4f} m ({mr_res['n_stations']} ranges)")
                st.dataframe(pd.DataFrame(mr_res["stations"]), width="stretch", hide_index=True)
                if st.button("✅ Apply Multi-Range Spread, Exponent & Cd to Digital Twin", type="primary", key="cal_mr_apply_btn"):
                    st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                        st.session_state.get("_calibrated_digital_twin"),
                        mr_res["twin_patch"],
                        domain="Multi-Range Pattern & Drag",
                        summary=f"Spread {mr_res['spread_factor']:.2f}x, exp p={mr_res['spread_exponent']:.2f}, Cd {mr_res['pellet_cd_scale']:.2f}x",
                    )
                    st.session_state["_digital_twin_pending_enable"] = True
                    st.success("Applied multi-range spread factor, exponent & Cd scale to Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Multi-range calibration error: {exc}")

        else:
            ph1, ph2, ph3 = st.columns(3)
            with ph1:
                bd_w_m = st.number_input("Board Width (m)", 0.4, 3.0, 1.0, 0.1, key="cal_ph_bw")
            with ph2:
                bd_h_m = st.number_input("Board Height (m)", 0.4, 3.0, 1.0, 0.1, key="cal_ph_bh")
            with ph3:
                bd_sig = st.slider("Hole Threshold (σ MAD)", 1.6, 4.5, 2.6, 0.2, key="cal_ph_sig")
            ph_up = st.file_uploader("Upload Patterning Target Photo (.png / .jpg)", type=["png", "jpg", "jpeg"], key="cal_ph_up")

            sp_b1, sp_b2, sp_b3, sp_b4 = st.columns(4)
            with sp_b1:
                if st.button("📸 Load Built-In Orthogonal Photo", key="cal_ph_sample_btn", width="stretch"):
                    st.session_state["_cal_ph_bytes"] = generate_synthetic_pattern_board_png(
                        range_m=float(pat_rng),
                        nominal_choke=str(pat_choke),
                        board_width_m=float(bd_w_m),
                        board_height_m=float(bd_h_m),
                        seed=42,
                    )
                    st.session_state["cal_ph_use_homo"] = False
            with sp_b2:
                if st.button("📐 Load Skewed Photo with 4 Corner Points", key="cal_ph_skewed_sample_btn", width="stretch"):
                    sk_img, sk_c, _ = generate_skewed_pattern_board_png_with_corners(
                        range_m=float(pat_rng),
                        nominal_choke=str(pat_choke),
                        board_width_m=float(bd_w_m),
                        board_height_m=float(bd_h_m),
                        seed=42,
                    )
                    st.session_state["_cal_ph_bytes"] = sk_img
                    st.session_state["cal_ph_use_homo"] = True
                    st.session_state["cal_ph_c1x"] = float(sk_c[0][0])
                    st.session_state["cal_ph_c1y"] = float(sk_c[0][1])
                    st.session_state["cal_ph_c2x"] = float(sk_c[1][0])
                    st.session_state["cal_ph_c2y"] = float(sk_c[1][1])
                    st.session_state["cal_ph_c3x"] = float(sk_c[2][0])
                    st.session_state["cal_ph_c3y"] = float(sk_c[2][1])
                    st.session_state["cal_ph_c4x"] = float(sk_c[3][0])
                    st.session_state["cal_ph_c4y"] = float(sk_c[3][1])
            with sp_b3:
                if st.button("🎯 Load Skewed Fiducial Demo", key="cal_ph_fiducial_sample_btn", width="stretch"):
                    fk_img, fk_c, _ = generate_skewed_fiducial_board_png(
                        range_m=float(pat_rng),
                        nominal_choke=str(pat_choke),
                        board_width_m=float(bd_w_m),
                        board_height_m=float(bd_h_m),
                        seed=42,
                    )
                    st.session_state["_cal_ph_bytes"] = fk_img
                    st.session_state["cal_ph_auto_fiducial"] = True
                    st.session_state["cal_ph_use_homo"] = True
                    st.session_state["cal_ph_c1x"] = float(fk_c[0][0])
                    st.session_state["cal_ph_c1y"] = float(fk_c[0][1])
                    st.session_state["cal_ph_c2x"] = float(fk_c[1][0])
                    st.session_state["cal_ph_c2y"] = float(fk_c[1][1])
                    st.session_state["cal_ph_c3x"] = float(fk_c[2][0])
                    st.session_state["cal_ph_c3y"] = float(fk_c[2][1])
                    st.session_state["cal_ph_c4x"] = float(fk_c[3][0])
                    st.session_state["cal_ph_c4y"] = float(fk_c[3][1])
            with sp_b4:
                board_pdf_bytes = generate_printable_fiducial_board_pdf(
                    board_width_m=float(bd_w_m),
                    board_height_m=float(bd_h_m),
                )
                st.download_button(
                    "🖨️ Download Printable Fiducial Board (.pdf)",
                    data=board_pdf_bytes,
                    file_name="cuas_fiducial_pattern_board.pdf",
                    mime="application/pdf",
                    key="cal_ph_board_pdf_dl",
                    width="stretch",
                )

            st.session_state.setdefault("cal_ph_auto_fiducial", False)
            st.session_state.setdefault("cal_ph_use_homo", False)
            hf_c1, hf_c2 = st.columns(2)
            with hf_c1:
                auto_fid = st.checkbox("Auto-Detect 4 Printable Fiducial Targets", key="cal_ph_auto_fiducial")
            with hf_c2:
                use_homo = st.checkbox("Enable 4-Corner DLT Homography Rectification", key="cal_ph_use_homo")
            corner_pts = None
            if use_homo and not auto_fid:
                st.caption("Specify pixel coordinates of the 4 target-board corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left) to rectify camera skew.")
                hc1, hc2, hc3, hc4 = st.columns(4)
                with hc1:
                    c1_x = st.number_input("C1 (TL) X (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c1x", 120.0)), 1.0, key="cal_ph_c1x")
                    c1_y = st.number_input("C1 (TL) Y (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c1y", 110.0)), 1.0, key="cal_ph_c1y")
                with hc2:
                    c2_x = st.number_input("C2 (TR) X (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c2x", 710.0)), 1.0, key="cal_ph_c2x")
                    c2_y = st.number_input("C2 (TR) Y (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c2y", 140.0)), 1.0, key="cal_ph_c2y")
                with hc3:
                    c3_x = st.number_input("C3 (BR) X (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c3x", 730.0)), 1.0, key="cal_ph_c3x")
                    c3_y = st.number_input("C3 (BR) Y (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c3y", 730.0)), 1.0, key="cal_ph_c3y")
                with hc4:
                    c4_x = st.number_input("C4 (BL) X (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c4x", 90.0)), 1.0, key="cal_ph_c4x")
                    c4_y = st.number_input("C4 (BL) Y (px)", 0.0, 4000.0, float(st.session_state.get("cal_ph_c4y", 700.0)), 1.0, key="cal_ph_c4y")
                corner_pts = [(c1_x, c1_y), (c2_x, c2_y), (c3_x, c3_y), (c4_x, c4_y)]

            img_bytes = ph_up.getvalue() if ph_up is not None else st.session_state.get("_cal_ph_bytes") or generate_synthetic_pattern_board_png(
                range_m=float(pat_rng),
                nominal_choke=str(pat_choke),
                board_width_m=float(bd_w_m),
                board_height_m=float(bd_h_m),
                seed=42,
            )
            try:
                dig_res = digitize_pattern_board_image(
                    img_bytes,
                    board_width_m=float(bd_w_m),
                    board_height_m=float(bd_h_m),
                    range_m=float(pat_rng),
                    nominal_choke=str(pat_choke),
                    ammunition_preset=str(pat_ammo),
                    threshold_sigma=float(bd_sig),
                    corner_points_px=corner_pts,
                    auto_detect_fiducials=bool(auto_fid),
                )
                d_pat = dig_res["pattern_calibration"]
                dp1, dp2, dp3, dp4 = st.columns(4)
                dp1.metric("Detected Pellet Holes", f"{dig_res['n_detected_holes']}", delta=f"{dig_res['image_shape'][1]}×{dig_res['image_shape'][0]} px")
                dp2.metric("Fitted Spread Multiplier", f"{d_pat['calibrated_spread_factor']:.2f}×", delta=f"Closest: {d_pat['closest_standard_choke']}")
                dp3.metric("Fitted R50 / R90", f"{d_pat['r50_fit_m']:.2f} / {d_pat['r90_fit_m']:.2f} m", delta=f"POI Bias Y/Z: {d_pat['poi_bias_y_mrad']:+.1f}/{d_pat['poi_bias_z_mrad']:+.1f} mrad")
                dp4.metric("Rayleigh GOF R²", f"{d_pat['r2_cdf']:.3f}", delta=f"KS p={d_pat['ks_pvalue']:.3f}")
                im_col, ch_col = st.columns([0.85, 1.35])
                with im_col:
                    vis_corners = dig_res.get("rectification_corners_px") or corner_pts
                    st.plotly_chart(
                        create_pattern_board_corners_overlay_figure(
                            img_bytes,
                            corners_px=vis_corners,
                            detected_holes_px=dig_res.get("detected_holes_px"),
                            title=f"Analyzed Target Board ({dig_res['n_detected_holes']} holes)",
                        ),
                        width="stretch",
                        key="cal_ph_overlay_fig",
                    )
                with ch_col:
                    st.plotly_chart(create_pattern_calibration_figure(d_pat), width="stretch", key="cal_ph_fig")
                if st.button("✅ Apply Photo-Digitized Pattern to Digital Twin", type="primary", key="cal_ph_apply_btn"):
                    st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                        st.session_state.get("_calibrated_digital_twin"),
                        dig_res["twin_patch"],
                        domain="Photo Pattern Digitizer",
                        summary=f"{dig_res['n_detected_holes']} holes detected, spread {d_pat['calibrated_spread_factor']:.2f}x",
                    )
                    st.session_state["_digital_twin_pending_enable"] = True
                    st.success("Applied photo-digitized pattern fit to Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Photo digitizer error: {exc}")

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
            st.plotly_chart(create_chronograph_calibration_figure(ch_res), width="stretch", key="cal_ch_fig")
            if st.button("✅ Apply Calibrated MV & SD to Digital Twin", type="primary", key="cal_ch_apply_btn"):
                st.session_state["_calibrated_mv_mps"] = float(ch_res["v_std_fit_mps"])
                st.session_state["_calibrated_mv_sd_mps"] = float(ch_res["residual_sd_mps"])
                ch_patch: dict[str, Any] = {
                    "mv_scale": round(float(ch_res["v_std_fit_mps"]) / max(float(ch_nom), 1.0), 4),
                    "mv_sd_mps": float(ch_res["residual_sd_mps"]),
                }
                if ch_res["has_temp_fit"]:
                    st.session_state["_calibrated_dv_dt"] = float(ch_res["dv_dt_mps_per_c"])
                    ch_patch["dv_dt_mps_per_c"] = float(ch_res["dv_dt_mps_per_c"])
                st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                    st.session_state.get("_calibrated_digital_twin"),
                    ch_patch,
                    domain="Chronograph MV & Temp",
                    summary=f"MV scale {ch_patch['mv_scale']:.3f}x, SD {ch_patch['mv_sd_mps']:.2f} m/s",
                )
                st.session_state["_digital_twin_pending_enable"] = True
                st.success(f"Applied MV={ch_res['v_std_fit_mps']:.1f} m/s & enabled Calibrated Digital Twin!")
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
            st.plotly_chart(create_drag_calibration_figure(dg_res), width="stretch", key="cal_dg_fig")
            st.dataframe(pd.DataFrame(dg_res["stations"]), width="stretch")
            if st.button("✅ Apply Calibrated Form Factor (i) to Monte Carlo Simulator", type="primary", key="cal_dg_apply_btn"):
                st.session_state["_calibrated_form_factor"] = float(dg_res["calibrated_form_factor"])
                st.session_state["_calibrated_preset_id"] = str(dg_preset)
                if dg_fit_mv:
                    st.session_state["_calibrated_mv_mps"] = float(dg_res["calibrated_mv_mps"])
                st.success(f"Applied calibrated form factor i={dg_res['calibrated_form_factor']:.4f} for `{dg_preset}`!")
        except Exception as exc:
            st.error(f"Exterior drag calibration error: {exc}")

    with cal_t8:
        _render_doppler_shotstring_tab(sg_presets_18)


def _render_pillar_tracks_lethality(drone_specs_18: dict[str, Any]) -> None:
    cal_t4, cal_t5, cal_t6 = st.tabs([
        "🛰️ Recorded Track Replay",
        "💥 Live-Fire Drone Lethality",
        "🔧 Hardware Bench & Tracker",
    ])
    with cal_t4:
        st.markdown("##### 🛰️ Recorded Drone Flight-Track Engagement Replay (CSV / MAVLink / PX4 Log)")
        st.caption(
            "Replay your Stage 1 active turret (plus any applied Calibration Digital Twin) directly against measured local ENU (`t_s, x_m, y_m, z_m`), "
            "GPS/MAVLink (`lat_deg, lon_deg, alt_m`), or PX4/ArduPilot ULog CSV telemetry."
        )
        ts1, ts2, ts3, ts_px4, ts4 = st.columns([1.0, 1.0, 1.0, 1.0, 1.2])
        if ts1.button("📋 Sample: Evasive FPV Dive", key="cal_trk_s1", width="stretch"):
            st.session_state["_cal_trk_csv"] = generate_sample_drone_track_csv("evasive_fpv_dive")
        if ts2.button("📋 Sample: 3-Drone Pincer", key="cal_trk_s2", width="stretch"):
            st.session_state["_cal_trk_csv"] = generate_sample_drone_track_csv("multi_drone_pincer")
        if ts3.button("📋 Sample: Ridge Pop-Up", key="cal_trk_s3", width="stretch"):
            st.session_state["_cal_trk_csv"] = generate_sample_drone_track_csv("low_altitude_terrain_pop")
        if ts_px4.button("📋 Sample: PX4/ArduPilot Log", key="cal_trk_s_px4", width="stretch"):
            st.session_state["_cal_trk_csv"] = generate_sample_px4_flight_csv("evasive_fpv_dive")
        with ts4:
            trk_breach_r = st.slider("Keepout Breach Radius (m)", 8.0, 20.0, 12.0, 1.0, key="cal_trk_breach")

        trk_up = st.file_uploader("Upload Recorded Track CSV (`t_s,x_m,y_m,z_m` or PX4/ArduPilot ULog CSV)", type=["csv"], key="cal_trk_up")
        trk_csv_raw = trk_up.getvalue().decode("utf-8", errors="replace") if trk_up is not None else st.session_state.get("_cal_trk_csv") or generate_sample_drone_track_csv("evasive_fpv_dive")
        trk_header_low = trk_csv_raw.splitlines()[0].lower() if trk_csv_raw.strip() else ""
        if (
            ("timestamp" in trk_header_low and "x" in trk_header_low and "t_s" not in trk_header_low)
            or "timeus" in trk_header_low
            or "local_position" in trk_header_low
            or ("lat" in trk_header_low and "lon" in trk_header_low and "lat_deg" not in trk_header_low)
        ):
            px4_parsed = parse_ardupilot_px4_flight_log(trk_csv_raw)
            trk_csv = px4_parsed["track_csv"]
        else:
            trk_csv = trk_csv_raw
        trk_turret = dict(st.session_state.get("_active_cuas_profile") or {})
        trk_twin = dict(st.session_state.get("_calibrated_digital_twin") or {})
        if trk_twin:
            st.info(
                f"Calibrated Digital Twin Active: spread={trk_twin.get('spread_factor', 1.0):.2f}x, "
                f"MV_scale={trk_twin.get('mv_scale', 1.0):.3f}, aim_bias={trk_twin.get('aim_bias_mrad', 0.0):.1f} mrad"
            )
        try:
            trk_res = _cached_recorded_track_replay(
                csv_text=trk_csv,
                turret_config_json=json.dumps(trk_turret, sort_keys=True),
                calibration_profile_json=_get_effective_cal_twin_json() or (json.dumps(trk_twin, sort_keys=True) if trk_twin else ""),
                breach_radius_m=float(trk_breach_r),
                n_mc_trials=30,
                seed=42,
                physics_fidelity_json=_get_active_physics_fidelity_json(),
            )
            tsm = trk_res["summary"]
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Zero-Breach Stop Rate", f"{tsm['zero_breach_pct']:.1f} %", delta=f"{tsm['n_tracks']} recorded tracks")
            tm2.metric("Mean Track Kill Rate", f"{tsm['mean_kill_rate_pct']:.1f} %", delta=f"Intercept: {tsm['mean_intercept_range_m']:.1f} m")
            tm3.metric("Measured Peak Speed / G", f"{tsm['peak_speed_mps']:.1f} m/s", delta=f"Lateral: {tsm['peak_lat_g']:.2f} G")
            tm4.metric("Rounds / Wave Cost", f"{tsm['mean_rounds_expended']:.1f} rds", delta=f"${tsm['mean_ammo_cost_usd']:.2f} ({tsm['turret_name'][:20]})")
            st.plotly_chart(create_recorded_track_replay_figure(trk_res), width="stretch", key="cal_trk_fig")
            st.dataframe(pd.DataFrame(trk_res["tracks"]), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Track replay error: {exc}")

    with cal_t5:
        st.markdown("##### 💥 Live-Fire Drone Shoot-Down Bayesian Lethality Calibration")
        st.caption(
            "Log actual drone engagement shots (`range_m, ammunition_preset, choke, barrels_fired, drone_killed`) "
            "to compute MAP posterior estimates of drone vulnerable area (A_vuln) and critical pellet kill energy (E_crit)."
        )
        lf1, lf2 = st.columns([1.3, 1.0])
        with lf1:
            lf_drone = st.selectbox("Tested Target Airframe", list(drone_specs_18.keys()), format_func=lambda k: drone_specs_18[k]["name"], index=0, key="cal_lf_drone")
        with lf2:
            st.write("")
            if st.button("📋 Reset Sample Live-Fire Range Log", key="cal_lf_reset_btn", width="stretch"):
                st.session_state["_cal_lf_rows"] = generate_sample_livefire_drone_trials(drone_model=str(lf_drone))
        if "_cal_lf_rows" not in st.session_state:
            st.session_state["_cal_lf_rows"] = generate_sample_livefire_drone_trials(drone_model=str(lf_drone))
        lf_up = st.file_uploader("Optional Shoot-Down Log CSV", type=["csv"], key="cal_lf_up")
        if lf_up is not None:
            lf_src: list[dict[str, Any]] | str = lf_up.getvalue().decode("utf-8", errors="replace")
        else:
            lf_edited_df = st.data_editor(
                pd.DataFrame(st.session_state["_cal_lf_rows"]),
                num_rows="dynamic",
                width="stretch",
                key="cal_lf_editor",
            )
            lf_src = lf_edited_df.to_dict(orient="records")
        try:
            leth_res = calibrate_livefire_drone_lethality(
                lf_src,
                drone_model=str(lf_drone),
                base_twin=st.session_state.get("_calibrated_digital_twin"),
            )
            st.session_state["_cal_last_lethality"] = leth_res
            lm1, lm2, lm3, lm4 = st.columns(4)
            lm1.metric("Fitted Vulnerable Area (A_vuln)", f"{leth_res['drone_avuln_scale']:.2f}×", delta=f"95% CI {leth_res['drone_avuln_ci95'][0]:.2f}–{leth_res['drone_avuln_ci95'][1]:.2f}")
            lm2.metric("Fitted Critical Energy (E_crit)", f"{leth_res['drone_ecrit_scale']:.2f}×", delta=f"95% CI {leth_res['drone_ecrit_ci95'][0]:.2f}–{leth_res['drone_ecrit_ci95'][1]:.2f}")
            lm3.metric("Brier Score (Nominal -> Cal)", f"{leth_res['brier_after']:.4f}", delta=f"From {leth_res['brier_before']:.4f} ({leth_res['n_trials']} shots)")
            lm4.metric("Observed Shoot-Down Rate", f"{leth_res['observed_kill_rate_pct']:.1f}%", delta=f"Log-loss {leth_res['logloss_before']:.3f} -> {leth_res['logloss_after']:.3f}")
            cur_drift_preview = _cached_calibration_design_delta(
                turret_cfg_json=json.dumps(dict(st.session_state.get("_active_cuas_profile") or {}), sort_keys=True),
                calibration_twin_json=json.dumps(
                    merge_twin_updates(st.session_state.get("_calibrated_digital_twin"), leth_res["twin_patch"]),
                    sort_keys=True,
                ),
                drone_model=str(lf_drone),
            )
            st.plotly_chart(create_calibration_reliability_and_drift_figure(leth_res, cur_drift_preview), width="stretch", key="cal_lf_drift_fig")
            if st.button("✅ Apply Bayesian Lethality Fit to Digital Twin", type="primary", key="cal_lf_apply_btn"):
                st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                    st.session_state.get("_calibrated_digital_twin"),
                    leth_res["twin_patch"],
                    domain="Bayesian Live-Fire Lethality",
                    summary=f"A_vuln {leth_res['drone_avuln_scale']:.2f}x, E_crit {leth_res['drone_ecrit_scale']:.2f}x (Brier {leth_res['brier_after']:.3f})",
                )
                st.session_state["_digital_twin_pending_enable"] = True
                st.success("Applied Bayesian target vulnerability & lethal energy fit to Digital Twin!")
        except Exception as exc:
            st.error(f"Live-fire lethality calibration error: {exc}")

    with cal_t6:
        st.markdown("##### 🔧 Hardware Bench-Test & Sensor Tracker Recalibration")
        hb_col, trk_col = st.columns(2)
        prof_hb = dict(st.session_state.get("_active_cuas_profile") or {"n_barrels": 6})
        with hb_col:
            st.markdown("**Load-Cell Recoil, Servo Slew & Weighed Hardware Bench**")
            if st.button("📋 Load Sample Bench Telemetry", key="cal_hb_sample_btn"):
                samp_hb = generate_sample_hardware_bench_tests(int(prof_hb.get("n_barrels", 6)))
                st.session_state["cal_hb_recoil"] = float(np.mean(samp_hb["peak_recoil_kn_samples"]))
                st.session_state["cal_hb_settle"] = float(np.mean(samp_hb["settle_time_ms_samples"]))
                st.session_state["cal_hb_slew"] = float(np.mean(samp_hb["slew_rate_deg_s_samples"]))
                st.session_state["cal_hb_accel"] = float(np.mean(samp_hb["slew_accel_deg_s2_samples"]))
                st.session_state["cal_hb_mass"] = float(samp_hb["weighed_mass_kg"])
                st.session_state["cal_hb_pwr"] = float(samp_hb["measured_peak_power_w"])
                st.session_state["_cal_hb_zeros"] = samp_hb["barrel_zero_offsets_mrad"]
            hb_rec = st.number_input("Measured Peak Recoil (kN)", 0.5, 25.0, 5.12, 0.1, key="cal_hb_recoil")
            hb_settle = st.number_input("Measured Recoil Settle Time (ms)", 40.0, 1200.0, 215.0, 10.0, key="cal_hb_settle")
            hb_slew = st.number_input("Measured Max Slew Rate (deg/s)", 30.0, 320.0, 151.0, 5.0, key="cal_hb_slew")
            hb_accel = st.number_input("Measured Slew Accel (deg/s²)", 80.0, 1800.0, 585.0, 20.0, key="cal_hb_accel")
            hb_mass = st.number_input("Weighed Turret Mass (kg)", 10.0, 600.0, 63.4, 0.5, key="cal_hb_mass")
            hb_pwr = st.number_input("Measured Peak Bus Power (W)", 50.0, 5000.0, 468.0, 10.0, key="cal_hb_pwr")
            zeros_init = st.session_state.get("_cal_hb_zeros") or generate_sample_hardware_bench_tests(int(prof_hb.get("n_barrels", 6)))["barrel_zero_offsets_mrad"]
            zeros_df = st.data_editor(pd.DataFrame(zeros_init), num_rows="dynamic", width="stretch", key="cal_hb_zeros_ed")
            hb_res = calibrate_hardware_bench_tests(
                prof_hb,
                {
                    "peak_recoil_kn_samples": [float(hb_rec)],
                    "settle_time_ms_samples": [float(hb_settle)],
                    "slew_rate_deg_s_samples": [float(hb_slew)],
                    "slew_accel_deg_s2_samples": [float(hb_accel)],
                    "weighed_mass_kg": float(hb_mass),
                    "measured_peak_power_w": float(hb_pwr),
                    "barrel_zero_offsets_mrad": zeros_df.to_dict(orient="records"),
                },
            )
            st.dataframe(pd.DataFrame(hb_res["comparisons"]), width="stretch", hide_index=True)
            for fl in hb_res["out_of_spec_flags"]:
                st.warning(fl)
            if st.button("✅ Apply Hardware Bench Calibration to Digital Twin", type="primary", key="cal_hb_apply_btn", width="stretch"):
                st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                    st.session_state.get("_calibrated_digital_twin"),
                    hb_res["twin_patch"],
                    domain="Hardware Bench Calibration",
                    summary=f"Recoil {hb_res['twin_patch']['recoil_force_scale']:.2f}x, Settle {hb_settle:.0f} ms, Zero RMS {hb_res['twin_patch']['barrel_zero_scatter_mrad']:.2f} mrad",
                )
                st.session_state["_digital_twin_pending_enable"] = True
                st.success("Applied measured recoil, slew/settle & barrel zeroing to Calibrated Digital Twin!")

        with trk_col:
            st.markdown("**Sensor / Tracker Angular Error, Latency & OTM Jitter**")
            trk_sensor = st.selectbox("Reference Sensor Suite", list(get_cuas_sensor_presets().keys())[:3], key="cal_trk_sens")
            if "_cal_trktel_rows" not in st.session_state or st.button("📋 Reset Tracker Telemetry Log", key="cal_trktel_reset"):
                st.session_state["_cal_trktel_rows"] = generate_sample_tracker_test_rows(sensor_preset=str(trk_sensor))
            tt_df = st.data_editor(
                pd.DataFrame(st.session_state["_cal_trktel_rows"]),
                num_rows="dynamic",
                width="stretch",
                key="cal_trktel_ed",
            )
            try:
                tt_res = calibrate_tracker_telemetry(tt_df.to_dict(orient="records"), sensor_preset=str(trk_sensor))
                tm_a, tm_b, tm_c = st.columns(3)
                tm_a.metric("Measured RMS Jitter", f"{tt_res['measured_rms_mrad']:.2f} mrad", delta=f"Nom: {tt_res['nominal_rms_mrad']:.2f}")
                tm_b.metric("Mean / P95 Latency", f"{tt_res['measured_latency_ms']:.1f} ms", delta=f"P95: {tt_res['latency_p95_ms']:.1f} ms")
                tm_c.metric("Measured OTM Jitter", f"{tt_res['otm_jitter_mrad']:.2f} mrad", delta=f"{tt_res['n_samples']} samples")
                if st.button("✅ Apply Tracker & OTM Fit to Digital Twin", type="primary", key="cal_trktel_apply_btn", width="stretch"):
                    st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                        st.session_state.get("_calibrated_digital_twin"),
                        tt_res["twin_patch"],
                        domain="Tracker Telemetry",
                        summary=f"RMS {tt_res['measured_rms_mrad']:.2f} mrad, Latency {tt_res['measured_latency_ms']:.0f} ms",
                    )
                    st.session_state["_digital_twin_pending_enable"] = True
                    st.success("Applied tracker jitter & OTM measurements to Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Tracker telemetry calibration error: {exc}")

def _render_pillar_twin_manager(drone_specs_18: dict[str, Any]) -> None:
    st.markdown("##### 🧬 Persistent Digital Twin Manager, Design Drift & Active Test Planner")
    with st.container():
        active_twin = merge_twin_updates(None, st.session_state.get("_calibrated_digital_twin") or {})
        prof_s1 = dict(st.session_state.get("_active_cuas_profile") or {})

        mc1, mc2, mc3, mc4 = st.columns([1.1, 0.9, 0.9, 1.1])
        with mc1:
            twin_name = st.text_input("Profile Name", value="Proto-V2_Range_Campaign", key="cal_mgr_name")
        with mc2:
            twin_lot = st.text_input("Ammo Lot #", value=str(active_twin.get("lot_number", "LOT-2026-04")), key="cal_mgr_lot")
        with mc3:
            twin_serial = st.text_input("Barrel / Turret Serial", value=str(active_twin.get("barrel_serial", "SN-001")), key="cal_mgr_serial")
        with mc4:
            st.write("")
            if st.button("💾 Save Profile to SQLite Library", type="primary", key="cal_mgr_save_btn", width="stretch"):
                twin_to_save = dict(active_twin)
                twin_to_save["lot_number"] = twin_lot
                twin_to_save["barrel_serial"] = twin_serial
                saved_rec = CalibrationProfileStore().save_profile(
                    name=twin_name,
                    twin=twin_to_save,
                    notes=f"R80 drift analysis & empirical test twin ({len(twin_to_save.get('history', []))} calibration updates)",
                    tags=["cuas", "digital-twin", twin_lot],
                )
                st.session_state["_calibrated_digital_twin"] = saved_rec["twin"]
                st.session_state["_digital_twin_pending_enable"] = True
                st.success(f"Saved Digital Twin `{saved_rec['name']}` (id `{saved_rec['id']}`) to persistent SQLite store!")

        store = CalibrationProfileStore()
        all_saved_twins = store.list_profiles()
        if all_saved_twins:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": p["id"],
                            "Name": p["name"],
                            "Lot": p["lot_number"],
                            "Serial": p["barrel_serial"],
                            "Spread": f"{p['twin'].get('spread_factor', 1.0):.2f}x",
                            "E_crit": f"{p['twin'].get('drone_ecrit_scale', 1.0):.2f}x",
                            "Recoil": f"{p['twin'].get('recoil_force_scale', 1.0):.2f}x",
                            "Updated": p["updated_at"],
                        }
                        for p in all_saved_twins
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            if len(all_saved_twins) >= 2:
                with st.expander("🔍 Compare Two Saved Digital Twin Profiles (Before vs After Diff)", expanded=False):
                    p_id_map = {f"{p['name']} ({p['id']})": p for p in all_saved_twins}
                    p_labels = list(p_id_map.keys())
                    pd_c1, pd_c2 = st.columns(2)
                    with pd_c1:
                        twin_a_lbl = st.selectbox("Baseline Profile A", p_labels, index=min(1, len(p_labels) - 1), key="cal_diff_a")
                    with pd_c2:
                        twin_b_lbl = st.selectbox("Candidate Profile B", p_labels, index=0, key="cal_diff_b")
                    tw_a = p_id_map[twin_a_lbl]["twin"]
                    tw_b = p_id_map[twin_b_lbl]["twin"]
                    diff_fields = [
                        ("Spread Scale Factor", "spread_factor", "x"),
                        ("Spread Growth Exponent (p)", "spread_exponent", ""),
                        ("Muzzle Velocity Scale", "mv_scale", "x"),
                        ("Pellet Drag Cd Scale", "pellet_cd_scale", "x"),
                        ("Drone A_vuln Scale", "drone_avuln_scale", "x"),
                        ("Drone E_crit Scale", "drone_ecrit_scale", "x"),
                        ("Recoil Force Scale", "recoil_force_scale", "x"),
                        ("Recoil Settle Time", "recoil_settle_ms", "ms"),
                        ("Servo Slew Rate", "slew_rate_deg_s", "°/s"),
                        ("Tracker RMS Jitter", "tracker_rms_mrad", "mrad"),
                        ("Tracker Latency", "tracker_latency_ms", "ms"),
                    ]
                    diff_rows = []
                    for f_lbl, f_key, f_unit in diff_fields:
                        va = float(tw_a.get(f_key, 0.0) or 0.0)
                        vb = float(tw_b.get(f_key, 0.0) or 0.0)
                        diff_rows.append(
                            {
                                "Parameter": f_lbl,
                                "Profile A": f"{va:.3f} {f_unit}".strip(),
                                "Profile B": f"{vb:.3f} {f_unit}".strip(),
                                "Delta (B - A)": f"{vb - va:+.3f} {f_unit}".strip(),
                            }
                        )
                    st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)

        dc_a, dc_b, dc_c = st.columns(3)
        with dc_a:
            drift_drone = st.selectbox("Reference Threat Airframe", list(drone_specs_18.keys()), index=0, key="cal_drift_drone")
        with dc_b:
            drift_rng = st.slider("Required Intercept Range (m)", 20.0, 75.0, 45.0, 5.0, key="cal_drift_rng")
        with dc_c:
            drift_mnt = st.selectbox("Mount Class Constraint", list(MOUNT_CLASS_SPECS.keys()), index=1, key="cal_drift_mnt")

        drift_res = _cached_calibration_design_delta(
            turret_cfg_json=json.dumps(prof_s1, sort_keys=True),
            calibration_twin_json=json.dumps(active_twin, sort_keys=True),
            drone_model=str(drift_drone),
            required_range_m=float(drift_rng),
            mount_class=str(drift_mnt),
        )
        plan_res = _cached_active_test_plan(
            turret_cfg_json=json.dumps(prof_s1, sort_keys=True),
            calibration_twin_json=json.dumps(active_twin, sort_keys=True),
            drone_model=str(drift_drone),
            target_range_m=float(drift_rng),
        )

        dm_1, dm_2, dm_3, dm_4 = st.columns(4)
        dm_1.metric("Nominal -> Calibrated R80", f"{drift_res['calibrated']['r80_m']:.1f} m", delta=f"{drift_res['r80_drift_m']:+.1f} m drift")
        dm_2.metric(f"Pk @ {drift_rng:.0f} m (Cal)", f"{drift_res['calibrated']['pk_at_req_pct']:.1f}%", delta=f"{drift_res['pk_at_req_drift_pp']:+.1f} pp vs nominal")
        dm_3.metric("R80 95% Uncertainty", f"+/- {plan_res['r80_ci95_halfwidth_m']:.2f} m", delta="From posterior SEs")
        dm_4.metric("Calibrated Mass / Recoil", f"{drift_res['calibrated']['mass_kg']:.1f} kg", delta=f"{drift_res['calibrated']['peak_recoil_kn']:.2f} kN")
        st.info(f"**Model Drift Diagnosis:** {drift_res['diagnosis']}")

        st.plotly_chart(
            create_calibration_reliability_and_drift_figure(st.session_state.get("_cal_last_lethality"), drift_res),
            width="stretch",
            key="cal_mgr_drift_fig",
        )
        fixes = drift_res.get("recommended_fixes", [])
        if fixes:
            st.markdown("###### 🛠️ Top-3 Recommended Design Compensations Under Recalibrated Twin")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Rank": f"#{i+1}",
                            "Recommended Design Change": f["fix_name"],
                            "Calibrated R80 (m)": f["r80_m"],
                            "Gain (m)": f"{f['r80_gain_m']:+.1f}",
                            f"Pk @ {drift_rng:.0f}m (%)": f["pk_at_req_pct"],
                            "Mass / Recoil": f"{f['mass_kg']:.1f} kg / {f['peak_recoil_kn']:.2f} kN",
                            "Fits Mount": "Yes" if f["mount_feasible"] else "No",
                        }
                        for i, f in enumerate(fixes)
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            fb1, fb2 = st.columns(2)
            fb1.button(
                f"⚡ Apply Design Fix #1 ({fixes[0]['fix_name']}) to Stage 1",
                type="primary",
                on_click=_apply_stage1_fix_callback,
                args=(fixes[0]["stage1_patch"], fixes[0]["fix_name"]),
                width="stretch",
                key="cal_mgr_apply_fix1_btn",
            )
            cert_bytes = generate_calibration_certificate_pdf(
                twin=active_twin,
                drift_res=drift_res,
                test_plan=plan_res,
                title=f"Digital Twin Calibration Certificate — {twin_name}",
            )
            fb2.download_button(
                "📜 Download Calibration Certificate (.pdf)",
                data=cert_bytes,
                file_name="cuas_calibration_certificate.pdf",
                mime="application/pdf",
                width="stretch",
                key="cal_mgr_cert_dl",
            )

        st.divider()
        st.markdown("###### 🧪 Bayesian Active Test Planner — Highest-Value Next Range Tests")
        st.plotly_chart(create_active_test_planner_figure(plan_res), width="stretch", key="cal_mgr_plan_fig")
        st.dataframe(pd.DataFrame(plan_res["top_recommendations"]), width="stretch", hide_index=True)
        st.download_button(
            "📋 Download Printable Range Test Card (.csv)",
            data=export_range_test_card_csv(plan_res).encode("utf-8"),
            file_name="cuas_range_test_card.csv",
            mime="text/csv",
            key="cal_mgr_testcard_dl",
            width="stretch",
        )

def _render_doppler_shotstring_tab(sg_presets_18: dict[str, Any]) -> None:
    with st.container():
        st.markdown("##### 📡 Doppler Radar Pellet Drag Regression & Shot-String Analysis")
        st.caption(
            "Regress projectile velocity decay $v(R)$ from radar track CSVs to calibrate in-flight $C_d$ scale, "
            "and analyze temporal shot-string spatial elongation and moist-air density."
        )
        r_ref_ammo = str(st.session_state.get("cal_pat_ammo", list(sg_presets_18.keys())[0]))
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown("**Doppler Radar Track $v(R)$ Fit**")
            dop_up = st.file_uploader("Upload Doppler Radar Track CSV (`range_m, velocity_mps`)", type=["csv"], key="cal_dop_up")
            if st.button("📋 Load Sample Doppler Radar Track CSV", key="cal_dop_sample_btn"):
                st.session_state["_cal_dop_csv"] = generate_sample_doppler_radar_csv(ammunition_preset=r_ref_ammo)
            dop_csv = dop_up.getvalue().decode("utf-8", errors="replace") if dop_up is not None else st.session_state.get("_cal_dop_csv") or generate_sample_doppler_radar_csv(ammunition_preset=r_ref_ammo)
            try:
                dop_res = calibrate_doppler_radar_drag(dop_csv, ammunition_preset=r_ref_ammo)
                dm1, dm2, dm3, dm4 = st.columns(4)
                dm1.metric("Fitted V0", f"{dop_res['fitted_v0_mps']:.1f} m/s", delta=f"Nom: {dop_res['nominal_v0_mps']:.1f} m/s")
                dm2.metric("Drag Cd Scale", f"{dop_res['calibrated_cd_scale']:.2f}×", delta=f"± {dop_res['cd_scale_se']:.2f} SE")
                dm3.metric("Velocity RMSE", f"{dop_res['rmse_velocity_mps']:.2f} m/s", delta=f"{dop_res['n_track_points']} points")
                dm4.metric("Fit Status", "CONVERGED" if dop_res["success"] else "FAILED")
                st.plotly_chart(create_doppler_drag_fit_figure(dop_res), width="stretch", key="cal_dop_fig")
                if st.button("✅ Apply Doppler Drag Cd Scale to Digital Twin", type="primary", key="cal_dop_apply_btn"):
                    st.session_state["_calibrated_digital_twin"] = merge_twin_updates(
                        st.session_state.get("_calibrated_digital_twin"),
                        dop_res["twin_patch"],
                        domain="Doppler Radar Drag",
                        summary=f"Fitted Cd scale {dop_res['calibrated_cd_scale']:.2f}x (RMSE {dop_res['rmse_velocity_mps']:.2f} m/s)",
                    )
                    st.session_state["_digital_twin_pending_enable"] = True
                    st.success(f"Applied Cd scale {dop_res['calibrated_cd_scale']:.2f}x to Calibrated Digital Twin!")
            except Exception as exc:
                st.error(f"Doppler calibration error: {exc}")

        with r_c2:
            st.markdown("**Environmental Met & Acoustic Shot-String**")
            ss_up = st.file_uploader("Upload Acoustic / Optical Shot-String CSV (`arrival_time_ms` or `velocity_mps`)", type=["csv"], key="cal_ss_up")
            ss_rng = st.number_input("Shot-String Target Range (m)", 15.0, 100.0, 50.0, 5.0, key="cal_ss_rng")
            if st.button("📋 Load Sample Shot-String & Met CSV", key="cal_ss_sample_btn"):
                st.session_state["_cal_ss_csv"] = generate_sample_shotstring_csv(range_m=float(ss_rng))
            ss_csv = ss_up.getvalue().decode("utf-8", errors="replace") if ss_up is not None else st.session_state.get("_cal_ss_csv") or generate_sample_shotstring_csv(range_m=float(ss_rng))
            try:
                ss_res = parse_weather_and_shotstring_csv(ss_csv, range_m=float(ss_rng))
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Air Density (ρ)", f"{ss_res['air_density_kg_m3']:.3f} kg/m³", delta=f"{ss_res['ambient_temp_c']:.1f}°C, {ss_res['relative_humidity_pct']:.0f}% RH")
                sm2.metric("Shot-String Length", f"{ss_res['string_length_m']:.2f} m", delta=f"Duration: {ss_res['string_duration_ms']:.2f} ms")
                sm3.metric("Pellets Arrived", f"{ss_res['n_pellets_recorded']}", delta=f"Spread SD: {ss_res['arrival_time_sd_ms']:.2f} ms")
                sm4.metric("Sound Speed", f"{ss_res['speed_of_sound_mps']:.1f} m/s", delta=f"P_amb: {ss_res['ambient_pressure_hpa']:.0f} hPa")
                st.json({
                    "range_m": ss_res["range_m"],
                    "string_length_5_95_pct_m": ss_res["string_length_m"],
                    "string_duration_ms": ss_res["string_duration_ms"],
                    "air_density_kg_m3": ss_res["air_density_kg_m3"],
                    "virtual_temperature_k": ss_res["virtual_temp_k"],
                })
            except Exception as exc:
                st.error(f"Shot-string analysis error: {exc}")

def _render_pillar_scientific_vv(sg_presets_18: dict[str, Any]) -> None:
    r_ref_ammo = str(st.session_state.get("cal_pat_ammo", list(sg_presets_18.keys())[0]))
    st.markdown("##### 🔬 Scientific V&V, Uncertainty Quantification & Live Range-Day SPRT")
    st.caption(
        "Rigorous verification & validation tools conforming to ASME V&V 20 guidelines, Ferson-Oberkampf Area Validation Metrics (AVM), "
        "live Wald Sequential Probability Ratio Tests (SPRT), and Latin Hypercube epistemic uncertainty propagation."
    )
    with st.container():
        vv_tab1, vv_tab2, vv_tab3, vv_tab4, vv_tab5 = st.tabs([
            "🎯 Blind Holdout Cross-Validation",
            "📊 Epistemic + Aleatory UQ P-Boxes",
            "🎲 Pre-Test Range DOE & Wald SPRT",
            "🧪 Published C-UAS Benchmarks (E-01..E-06)",
            "📦 FAIR Validation Dossier & SHA-256 Bundle",
        ])

        with vv_tab1:
            st.markdown("###### Blind Out-of-Sample Holdout Cross-Validation")
            st.caption("Train the Digital Twin strictly on near-range stations and evaluate true predictive error on distant holdout stations.")
            ref_choke_val = str(st.session_state.get("cal_pat_choke", "Modified"))
            if st.button("🚀 Run Blind Holdout Cross-Validation", key="cal_vv_holdout_btn", type="primary"):
                sample_ds = generate_sample_blind_holdout_dataset()
                st.session_state["_cal_holdout_res"] = run_blind_holdout_validation(
                    sample_ds["pattern_rows"],
                    livefire_rows=sample_ds["livefire_rows"],
                    holdout_ranges_m=(65.0, 75.0),
                    nominal_choke=ref_choke_val,
                    ammunition_preset=r_ref_ammo,
                )
            if "_cal_holdout_res" in st.session_state:
                h_res = st.session_state["_cal_holdout_res"]
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Calibrated Holdout RMSE", f"{h_res['holdout_rmse_calibrated_m']:.3f} m", delta=f"{h_res['error_reduction_pct']:+.1f}% vs Nominal")
                h2.metric("Nominal Holdout RMSE", f"{h_res['holdout_rmse_nominal_m']:.3f} m", delta="Untuned model")
                asme = h_res.get("asme_vv20_sigma", {})
                h3.metric("ASME V&V 20 Error |E|", f"{asme.get('mean_comparison_error_e', 0.0):.3f} m", delta=f"u_val: {asme.get('u_val', 0.0):.3f} m")
                h4.metric("Area Validation Metric", f"{asme.get('area_validation_metric_avm', 0.0):.3f} m", delta=f"Passed: {asme.get('validated', False)}")
                st.plotly_chart(create_blind_holdout_validation_figure(h_res), width="stretch", key="cal_vv_holdout_fig")
                st.dataframe(pd.DataFrame(h_res["holdout_pattern_rows"]), width="stretch", hide_index=True)

        with vv_tab2:
            st.markdown("###### 95% Epistemic + Aleatory Uncertainty P-Box Envelope")
            st.caption("Propagate digital twin parameter estimation errors (epistemic uncertainty via Latin Hypercube) together with shot-cloud stochastic dispersion (aleatory uncertainty).")
            active_twin_val = st.session_state.get("_calibrated_digital_twin") or get_default_nominal_twin()
            if st.button("📊 Compute Epistemic UQ Envelope", key="cal_vv_uq_btn", type="primary"):
                with st.spinner("Propagating nested epistemic x aleatory Latin Hypercube samples..."):
                    st.session_state["_cal_uq_res"] = compute_epistemic_aleatory_uq_envelope(
                        twin_profile=active_twin_val,
                        distance_stations_m=(20.0, 35.0, 50.0, 65.0, 75.0),
                        n_epistemic_draws=10,
                        n_volleys_per_draw=8,
                        seed=42,
                    )
            if "_cal_uq_res" in st.session_state:
                uq_r = st.session_state["_cal_uq_res"]
                u1, u2, u3 = st.columns(3)
                u1.metric("R80 Median", f"{uq_r['r80_median_m']:.1f} m", delta=f"P05–P95: {uq_r['r80_p05_m']:.1f}–{uq_r['r80_p95_m']:.1f} m")
                u2.metric("Mean Band Width", f"{uq_r['mean_band_width_pp']:.1f} pp", delta="90% UQ Spread")
                u3.metric("Latin Hypercube Draws", f"{uq_r['n_epistemic_draws']} draws", delta="5D parameter space")
                st.plotly_chart(create_epistemic_uq_envelope_figure(uq_r), width="stretch", key="cal_vv_uq_fig")
                st.dataframe(pd.DataFrame(uq_r["band_rows"]), width="stretch", hide_index=True)

        with vv_tab3:
            st.markdown("###### Pre-Test Design of Experiments (DOE) & Interactive Live Range-Day Wald SPRT Companion")
            st.caption(
                "Plan optimal live-fire test matrix across ranges and log Kill / Miss shots live on the range "
                "to trigger exact Wald SPRT early stopping and exact Clopper-Pearson binomial confidence intervals."
            )
            doe_target_r = float(st.session_state.get("cal_drift_rng", 45.0))
            doe_p = plan_range_validation_doe(target_range_m=doe_target_r)

            st.session_state.setdefault("_cal_sprt_outcomes", [1, 1, 1, 0, 1, 1, 1, 1])
            sc_1, sc_2, sc_3, sc_4, sc_5 = st.columns(5)
            if sc_1.button("🎯 Record Kill (1)", key="cal_sprt_kill_btn", type="primary", width="stretch"):
                st.session_state["_cal_sprt_outcomes"] = [*st.session_state["_cal_sprt_outcomes"], 1]
            if sc_2.button("💨 Record Miss (0)", key="cal_sprt_miss_btn", width="stretch"):
                st.session_state["_cal_sprt_outcomes"] = [*st.session_state["_cal_sprt_outcomes"], 0]
            if sc_3.button("↩️ Undo Last Shot", key="cal_sprt_undo_btn", width="stretch"):
                if st.session_state["_cal_sprt_outcomes"]:
                    st.session_state["_cal_sprt_outcomes"] = st.session_state["_cal_sprt_outcomes"][:-1]
            if sc_4.button("🔄 Reset Sequence", key="cal_sprt_reset_btn", width="stretch"):
                st.session_state["_cal_sprt_outcomes"] = []

            live_sprt = evaluate_sprt_live_sequence(
                list(st.session_state["_cal_sprt_outcomes"]),
                sprt_p0=float(doe_p["sprt_p0"]),
                sprt_p1=float(doe_p["sprt_p1"]),
                alpha=0.05,
                power=0.80,
                range_m=doe_target_r,
                ammunition_preset=r_ref_ammo,
                choke=str(st.session_state.get("cal_pat_choke", "Modified")),
            )
            st.session_state["_cal_live_sprt_res"] = live_sprt
            if sc_5.button("💾 Commit SPRT Log to Live-Fire Table", key="cal_sprt_commit_btn", width="stretch"):
                cur_rows = list(st.session_state.get("_cal_lf_rows") or [])
                cur_rows.extend(live_sprt.get("livefire_rows", []))
                st.session_state["_cal_lf_rows"] = cur_rows
                st.success(f"Committed {live_sprt['n_rounds']} SPRT trial shots to Live-Fire Drone Lethality table!")

            sp_a, sp_b, sp_c, sp_d = st.columns(4)
            sp_a.metric("Live SPRT Decision", live_sprt["decision"], delta=f"{live_sprt['n_kills']}/{live_sprt['n_rounds']} kills")
            sp_b.metric("Observed Kill Rate", f"{live_sprt['observed_pk']*100.0:.1f}%", delta=f"CI95 [{live_sprt['ci95_low']*100:.1f}–{live_sprt['ci95_high']*100:.1f}%]")
            _cur_llr = float(live_sprt["trajectory"][-1]["llr"]) if live_sprt.get("trajectory") else 0.0
            sp_c.metric("Log-Likelihood Ratio (LLR)", f"{_cur_llr:.2f}", delta=f"Bounds [{live_sprt['b_bound_llr']:.2f}, {live_sprt['a_bound_llr']:.2f}]")
            sp_d.metric("Stopping Shot", f"Shot #{live_sprt['stopping_round']}" if live_sprt.get("stopping_round") else "In Progress", delta=f"Range: {doe_target_r:.0f} m")

            st.plotly_chart(create_sprt_stopping_boundaries_figure(doe_p, live_sprt=live_sprt), width="stretch", key="cal_vv_sprt_fig")
            st.dataframe(pd.DataFrame(doe_p["doe_matrix"]), width="stretch", hide_index=True)
            st.download_button(
                "📋 Download Range DOE Test Card (.csv)",
                data=export_doe_test_card_csv(doe_p).encode("utf-8"),
                file_name="cuas_range_doe_plan.csv",
                mime="text/csv",
                key="cal_vv_doe_dl",
                width="stretch",
            )

        with vv_tab4:
            st.markdown("###### Published Aeroballistic, Patterning, Shot-String, V50 & Rotor Kinematics Benchmarks")
            st.caption("Verify against published external benchmarks (E-01..E-06): Bailey & Hiatt (1972) sphere drag, Compton (1996) RK2 transonic decay, 40-yd choke pattern %, shot-string transit, Recht-Ipson V50 limit, and rotor solidity.")
            if "cuas_exp_benchmarks_s4" not in st.session_state or st.button("🔄 Run E-01..E-06 Benchmarks", key="cal_vv_exp_btn"):
                st.session_state["cuas_exp_benchmarks_s4"] = run_experimental_validation_benchmarks()
            e_b_res = st.session_state["cuas_exp_benchmarks_s4"]
            st.plotly_chart(create_experimental_benchmarks_figure(e_b_res), width="stretch", key="cal_vv_exp_fig")
            st.dataframe(pd.DataFrame(e_b_res["benchmarks"]), width="stretch", hide_index=True)

        with vv_tab5:
            st.markdown("###### FAIR Reproducible Validation Dossier & SHA-256 Bundle")
            st.caption("Compile a complete, cryptographically signed FAIR research artifact bundling digital twin parameters, benchmark reports, and calibration logs.")
            active_twin_bun = st.session_state.get("_calibrated_digital_twin") or get_default_nominal_twin()
            if st.button("📦 Build Reproducible Validation Bundle", key="cal_vv_bundle_btn", type="primary"):
                st.session_state["_cal_bundle"] = build_reproducible_validation_bundle(
                    twin=active_twin_bun,
                    livefire_dataset=st.session_state.get("_cal_last_lethality"),
                    blind_holdout_results=st.session_state.get("_cal_holdout_res"),
                    uq_results=st.session_state.get("_cal_uq_res"),
                    target_range_m=float(st.session_state.get("cal_drift_rng", 45.0)),
                )
            if "_cal_bundle" in st.session_state:
                bun = st.session_state["_cal_bundle"]
                st.success(f"Generated FAIR Validation Bundle! SHA-256 Digest: `{bun['manifest_sha256']}`")
                b_html = generate_scientific_vv_dossier_html(bun)
                b_md = generate_scientific_vv_dossier_markdown(bun)
                b_json = json.dumps(bun, indent=2)
                bd1, bd2, bd3 = st.columns(3)
                bd1.download_button("📜 Download HTML Dossier", b_html.encode("utf-8"), "cuas_scientific_validation_dossier.html", "text/html", width="stretch", key="cal_vv_html_dl")
                bd2.download_button("📝 Download Markdown Dossier", b_md.encode("utf-8"), "cuas_scientific_validation_dossier.md", "text/markdown", width="stretch", key="cal_vv_md_dl")
                bd3.download_button("🧾 Download JSON Bundle", b_json.encode("utf-8"), "cuas_validation_bundle.json", "application/json", width="stretch", key="cal_vv_json_dl")


