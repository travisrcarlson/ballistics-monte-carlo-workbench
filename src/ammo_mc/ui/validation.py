"""Knowledge Base Validation Suite UI workbench."""


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

from ammo_mc.cuas_scientific_vv import run_experimental_validation_benchmarks
from ammo_mc.cuas_rigor import run_workbook_validation_cases
from ammo_mc.vis_cuas_vv import create_experimental_benchmarks_figure, create_workbook_vv_matrix_figure
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
from ammo_mc.validation import (
    generate_vv_certificate_html,
    generate_vv_certificate_markdown,
    run_extended_vv_benchmarks,
    run_validation_suite,
)
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



def tab_validation() -> None:
    st.header("✅ Verification & Validation Benchmark Dashboard & Audit Certificates")
    st.markdown(
        "Automated V&V audit harness combining the 22-case `Ammunition Monte Carlo Knowledge Base.xlsx` regression suite "
        "(`V-01a..V-12`), 22 independent physical, analytical, and standards benchmarks (`B-01..B-22`), and 53 C-UAS workbook "
        "oracles across 101 linked literature sources."
    )

    with st.sidebar:
        _render_sidebar_header("✅ Validation Suite", "44-check V&V harness & audit certificates")
        run_val_btn = st.button("🔄 Run Full 44-Check V&V Suite", type="primary", width="stretch", key="val_run_btn")
        val_n_mc = st.select_slider(
            "V-08 Monte Carlo Samples",
            options=[10_000, 25_000, 50_000, 100_000],
            value=50_000,
            key="val_n_mc",
        )

    if (
        "vv_benchmark_report" not in st.session_state
        or "validation_results" not in st.session_state
        or run_val_btn
        or st.session_state.get("_val_last_n_mc") != val_n_mc
    ):
        with st.spinner("Executing 22 workbook cases + 18 extended physics V&V benchmarks..."):
            report = run_extended_vv_benchmarks(seed=42, n_mc_v08=int(val_n_mc))
            st.session_state["vv_benchmark_report"] = report
            st.session_state["validation_results"] = report["workbook_cases"]
            st.session_state["_val_last_n_mc"] = int(val_n_mc)

    report = st.session_state["vv_benchmark_report"]
    summary = report["summary"]
    lit = summary["literature"]
    v_df = pd.DataFrame(report["workbook_cases"])
    b_df = pd.DataFrame(report["physics_benchmarks"])
    dom_df = pd.DataFrame(summary["domain_breakdown"])

    with st.sidebar:
        st.divider()
        st.subheader("🔎 Filters & Exports")
        status_filter = st.radio(
            "Filter by Status",
            ["All Cases", "PASS Only", "Non-PASS Only"],
            key="val_status_filter",
        )
        domain_opts = sorted(b_df["domain"].astype(str).unique().tolist())
        domain_filter = st.multiselect(
            "Filter Physics Benchmarks by Domain",
            domain_opts,
            default=[],
            key="val_domain_filter",
        )
        formula_opts = sorted(v_df["formula"].astype(str).unique().tolist())
        formula_filter = st.multiselect(
            "Filter Workbook by Formula ID",
            formula_opts,
            default=[],
            key="val_formula_filter",
        )
        cert_html = generate_vv_certificate_html(report)
        cert_md = generate_vv_certificate_markdown(report)
        st.download_button(
            "📜 Download Audit Certificate (.html)",
            cert_html.encode("utf-8"),
            "ammo_mc_vv_certificate.html",
            "text/html",
            key="val_dl_html_btn",
            width="stretch",
        )
        st.download_button(
            "📝 Download Audit Certificate (.md)",
            cert_md.encode("utf-8"),
            "ammo_mc_vv_certificate.md",
            "text/markdown",
            key="val_dl_md_btn",
            width="stretch",
        )
        st.download_button(
            "📥 Download Workbook Results (.csv)",
            v_df.to_csv(index=False).encode("utf-8"),
            "kb_validation_suite_results.csv",
            "text/csv",
            key="val_dl_csv_btn",
            width="stretch",
        )
        st.download_button(
            "🧾 Download Full V&V Payload (.json)",
            json.dumps(report, indent=2).encode("utf-8"),
            "ammo_mc_vv_report.json",
            "application/json",
            key="val_dl_json_btn",
            width="stretch",
        )

    filtered_b = b_df.copy()
    filtered_v = v_df.copy()
    if status_filter == "PASS Only":
        filtered_b = filtered_b[filtered_b["status"].astype(str).str.contains("PASS")]
        filtered_v = filtered_v[filtered_v["status"].astype(str).str.contains("PASS")]
    elif status_filter == "Non-PASS Only":
        filtered_b = filtered_b[~filtered_b["status"].astype(str).str.contains("PASS")]
        filtered_v = filtered_v[~filtered_v["status"].astype(str).str.contains("PASS")]
    if domain_filter:
        filtered_b = filtered_b[filtered_b["domain"].astype(str).isin(domain_filter)]
    if formula_filter:
        filtered_v = filtered_v[filtered_v["formula"].astype(str).isin(formula_filter)]

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric(
        "Extended Physics Benchmarks",
        f"{summary['physics_passed']} / {summary['physics_total']}",
        delta=f"{summary['overall_physics_pass_rate_pct']:.1f}% Verified",
    )
    kc2.metric(
        "Workbook Cases (Excel Match)",
        f"{summary['workbook_excel_passed']} / {summary['workbook_total']}",
        delta=f"{summary['workbook_source_passed']}/{summary['workbook_total']} Source Tol",
    )
    kc3.metric(
        "Literature & Formula Governance",
        f"{lit['total_formulas']} / {lit['total_sources']}",
        delta=f"{lit['linked_source_coverage_pct']:.0f}% Linked ({len(lit['flagged_formula_ids'])} Flagged)",
    )
    kc4.metric(
        "Total V&V Runtime",
        f"{report['elapsed_s']:.2f} s",
        delta=f"{summary['total_checks']} Total Checks",
    )

    tab_score, tab_phys, tab_wb, tab_cuas_exp, tab_wb_vv = st.tabs(
        [
            "📊 Executive Scorecard & Tolerance Margins",
            "🔬 Extended Physics Benchmarks (B-01..B-22)",
            "📋 Knowledge Base Workbook Suite (V-01a..V-12)",
            "🧪 Experimental C-UAS Benchmarks (E-01..E-06)",
            "📐 C-UAS Workbook V&V Matrix (53 Oracles)",
        ]
    )

    with tab_score:
        st.subheader("Domain Verification Scorecard")
        st.dataframe(dom_df, width="stretch")

        if filtered_b.empty:
            st.info("No extended physics benchmarks match the selected sidebar filters.")
        else:
            fig_margin = px.bar(
                filtered_b.sort_values("utilization_pct", ascending=True),
                x="utilization_pct",
                y="benchmark_id",
                color="domain",
                orientation="h",
                hover_data=["title", "computed", "reference", "rel_error", "tolerance", "status"],
                labels={
                    "utilization_pct": "Tolerance Utilization (% of Allowed Margin Used)",
                    "benchmark_id": "Benchmark ID",
                },
                title="Normalized Tolerance Utilization Across Physics Benchmarks (Lower = Tighter Match)",
            )
            fig_margin.add_vline(x=100.0, line_dash="dash", line_color="red", annotation_text="100% Limit")
            fig_margin.update_layout(height=480, margin=dict(l=20, r=20, t=44, b=20))
            st.plotly_chart(fig_margin, width="stretch")

        st.warning(
            f"**Formula Registry Governance Status:** `{', '.join(lit['flagged_formula_ids'])}` remain explicitly flagged "
            f"requiring secondary verification (workbook open issue: `{', '.join(summary['workbook_flagged_open_ids']) or 'None'}`)."
        )

    with tab_phys:
        st.subheader("18 Independent Physical, Analytical & Standards Benchmarks (B-01..B-18)")
        if filtered_b.empty:
            st.info("No extended physics benchmarks match the selected sidebar filters.")
        else:
            disp_b = filtered_b.copy()
            disp_b["source_ids"] = disp_b["source_ids"].apply(lambda s: ", ".join(s) if isinstance(s, list) else str(s))
            st.dataframe(
                disp_b[
                    [
                        "benchmark_id",
                        "domain",
                        "status",
                        "computed",
                        "reference",
                        "unit",
                        "rel_error",
                        "tolerance",
                        "utilization_pct",
                        "source_ids",
                        "title",
                        "notes",
                    ]
                ],
                width="stretch",
                height=520,
            )

    with tab_wb:
        st.subheader("22 Knowledge Base Workbook Regression Cases (V-01a..V-12)")
        if filtered_v.empty:
            st.info("No validation cases match the selected sidebar filters.")
        else:
            if len(filtered_v) != len(v_df):
                st.caption(f"Showing {len(filtered_v)} of {len(v_df)} validation cases based on sidebar filters.")
            fig_val = px.bar(
                filtered_v,
                x="test_id",
                y="rel_err_vs_expected",
                color="formula",
                labels={"test_id": "Validation Test ID", "rel_err_vs_expected": "Relative Error vs Expected"},
                title="Relative Deviation Across Knowledge Base Validation Cases",
            )
            fig_val.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_val, width="stretch")

            st.dataframe(
                filtered_v[
                    [
                        "test_id",
                        "formula",
                        "status",
                        "computed",
                        "excel_value",
                        "expected",
                        "rel_err_vs_excel",
                        "rel_err_vs_expected",
                        "case",
                    ]
                ],
                width="stretch",
            )

    with tab_cuas_exp:
        st.subheader("🧪 Published Aeroballistic, Patterning, Shot-String, V50 & Rotor Solidity Benchmarks (E-01..E-06)")
        st.caption(
            "Empirical cross-validation against published wind-tunnel sphere drag (Bailey & Hiatt 1972), "
            "RK2 transonic pellet integration (Compton 1996 UCL), 12-gauge 40-yd 30-inch choke patterning, "
            "temporal shot-string elongation, Recht-Ipson V50 limit perforation, and rotor blade solidity kinematics."
        )
        if "cuas_exp_benchmarks" not in st.session_state or st.button("🔄 Re-run Experimental Benchmarks", key="val_rerun_exp_btn"):
            with st.spinner("Evaluating Experimental C-UAS Benchmarks E-01..E-06..."):
                st.session_state["cuas_exp_benchmarks"] = run_experimental_validation_benchmarks()
        exp_res = st.session_state["cuas_exp_benchmarks"]
        st.plotly_chart(create_experimental_benchmarks_figure(exp_res), width="stretch", key="val_exp_bench_fig")
        st.dataframe(pd.DataFrame(exp_res["benchmarks"]), width="stretch", hide_index=True)

    with tab_wb_vv:
        st.subheader("📐 Deterministic C-UAS Workbook Oracles & Statistical Unit Tests")
        st.caption(
            "Reproduces Walsh (1976) / NOL / KPY / WB-F05 trajectory integration, WB-F01/F02 closed-form salvo brackets vs. "
            "quadrature (CK-A1..A6 & CK-C1..C12), Thomas-Gemmill, Whistler SAS, Wilkening, SAAMI recoil, Wen CFRP, "
            "and Wilson / Clopper-Pearson / Agresti-Coull / alpha-spending intervals."
        )
        if "cuas_wb_vv_matrix" not in st.session_state or st.button("🔄 Re-run C-UAS Workbook Matrix", key="val_rerun_wb_vv_btn"):
            with st.spinner("Running 53 C-UAS Workbook Regression Oracles..."):
                st.session_state["cuas_wb_vv_matrix"] = run_workbook_validation_cases()
        wb_vv = st.session_state["cuas_wb_vv_matrix"]
        st.plotly_chart(create_workbook_vv_matrix_figure(wb_vv), width="stretch", key="val_wb_vv_fig")
        wb_cases_df = pd.DataFrame(wb_vv["cases"])
        st.dataframe(wb_cases_df, width="stretch", hide_index=True)
        st.download_button(
            "📥 Download C-UAS Workbook V&V Matrix (.csv)",
            wb_cases_df.to_csv(index=False).encode("utf-8"),
            "cuas_workbook_vv_matrix.csv",
            "text/csv",
            key="val_dl_wb_vv_csv",
        )
