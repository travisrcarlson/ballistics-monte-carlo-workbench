"""Stage 4A: Sustained Logistics, Briefings, A/B Comparator, Layered Defense & Tornado UI."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.cuas_analytics import (
    get_cuas_comparator_presets,
    get_outer_layer_catalog,
)
from ammo_mc.cuas_mission import (
    generate_cuas_trade_study_html,
    get_cuas_ammo_economics_catalog,
    get_magazine_system_specs,
    serialize_cuas_scenario_config,
)
from ammo_mc.scenario_library import ScenarioLibrary
from ammo_mc.ui.common import (
    _CUAS18_MODULES,
    _cached_calibration_design_delta,
    _cached_compare_ab,
    _cached_cuas_tornado,
    _cached_layered_defense,
    _cached_sustained_logistics,
    get_active_calibration_profile,
)
from ammo_mc.vis_cuas_analytics import (
    create_ab_comparator_radar_and_bars_figure,
    create_layered_defense_figure,
    create_sensitivity_tornado_figure,
)
from ammo_mc.vis_cuas_mission import create_logistics_thermal_and_cost_figure
from ammo_mc.ui.cuas_pkg.helpers import render_verification_and_counter_evidence


def render_stage4_trade_studies(active_cuas_mod: str, expert: bool, sg_presets_18: dict[str, Any], drone_specs_18: dict[str, Any], aspect_presets_18: dict[str, Any]) -> None:
    if active_cuas_mod == _CUAS18_MODULES[8]:
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
        _rep_cal_twin = get_active_calibration_profile()
        _rep_cal_payload = None
        if _rep_cal_twin:
            _rep_drift = _cached_calibration_design_delta(
                turret_cfg_json=json.dumps(rep_cfg, sort_keys=True),
                calibration_twin_json=json.dumps(_rep_cal_twin, sort_keys=True),
                drone_model=str(lg_drone),
                required_range_m=float(lg_dist),
            )
            _rep_cal_payload = {"twin": _rep_cal_twin, "drift": _rep_drift}
        _vv_bench = st.session_state.get("cuas_exp_benchmarks_s4")
        _vv_hold = st.session_state.get("_cal_holdout_res")
        _vv_uq = st.session_state.get("_cal_uq_res")
        _vv_sprt = st.session_state.get("_cal_live_sprt_res")
        _rep_vv_payload: dict[str, Any] | None = None
        if any(x is not None for x in (_vv_bench, _vv_hold, _vv_uq, _vv_sprt)):
            _rep_vv_payload = {
                "experimental_benchmarks": _vv_bench,
                "blind_holdout": _vv_hold,
                "uq_envelope": _vv_uq,
                "live_sprt": _vv_sprt,
            }
        with rep_c2:
            pdf_deck = generate_executive_briefing_pdf(
                title=str(rep_name),
                subtitle=f"{lsm['magazine_name']} | {lsm['ammo_name']}",
                ab_result=st.session_state.get("_report_ab_result"),
                site_result=st.session_state.get("_report_site_result"),
                tornado_result=st.session_state.get("_report_tornado_result"),
                logistics_result=log_res,
                raid_result=st.session_state.get("_report_raid_result"),
                adversarial_result=st.session_state.get("_report_adversarial_result"),
                calibration_result=_rep_cal_payload,
                vv_result=_rep_vv_payload,
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
                adversarial_result=st.session_state.get("_report_adversarial_result"),
                calibration_result=_rep_cal_payload,
                vv_result=_rep_vv_payload,
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
                else:
                    st.caption("Save 2 or more scenario snapshots above to compare metrics & configuration diffs side-by-side.")

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
                ab_crn = st.checkbox("Use Common Random Numbers (CRN Paired Variance Reduction, MC-F09)", value=True, key="cuas18_ab_crn")

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
                        use_common_random_numbers=bool(ab_crn),
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
                        "Pk 95% Wilson CI": f"[{r.get('kill_chain_pk_ci95_pct', [0, 0])[0]:.1f}, {r.get('kill_chain_pk_ci95_pct', [0, 0])[1]:.1f}]%",
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
                from ammo_mc.cuas_rigor import get_contextual_counter_evidence, resolve_verification_grade

                render_verification_and_counter_evidence(
                    resolve_verification_grade({"paired_crn": "V1" if ab_crn else "V2", "kill_chain": "V3", "vulnerability": "V3"}),
                    get_contextual_counter_evidence(stage="comparator", range_m=float(ab_dist), n_mc_trials=45),
                    key_prefix="stage4_ab",
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


