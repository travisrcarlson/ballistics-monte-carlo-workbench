"""First-Principles Physics & Ballistics Labs (Labs 1-16) UI workbench."""


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
    tno_eardrum_rupture_probit,
    tno_lung_lethality_probit,
    tno_thermal_lethality_probit,
)
from ammo_mc.terminal_concrete import (
    forrestal_frew_concrete_penetration,
    ndrc_concrete_perforation_and_scabbing,
)
from ammo_mc.advanced_armor import tate_alekseevskii_penetration
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
    _navigate_to,
    _render_sidebar_header,
    _set_active_workbench,
    _set_cuas_module,
    is_expert_mode,
)



def tab_calculators() -> None:
    st.header("Physics & Ballistics Engineering Labs")
    st.caption(
        "Focused first-principles labs for thermodynamics, exterior ballistics, terminal effects, 6-DoF/guidance, and the Knowledge Base."
    )
    kb = get_kb()

    _phys_groups: dict[str, list[str]] = {
        "1. Interior, Aero & Trajectory": [_PHYSICS_LABS[0], _PHYSICS_LABS[1], _PHYSICS_LABS[2], _PHYSICS_LABS[13]],
        "2. Blast, Frag, Armor & Concrete": [_PHYSICS_LABS[3], _PHYSICS_LABS[4], _PHYSICS_LABS[12], _PHYSICS_LABS[14]],
        "3. Guidance, 6-DoF, Fuze & Fire Control": [
            _PHYSICS_LABS[6],
            _PHYSICS_LABS[7],
            _PHYSICS_LABS[8],
            _PHYSICS_LABS[9],
            _PHYSICS_LABS[10],
            _PHYSICS_LABS[11],
            _PHYSICS_LABS[15],
        ],
        "4. QMC Statistics & Knowledge Base": [_PHYSICS_LABS[5]],
    }
    _group_names = list(_phys_groups.keys())

    with st.sidebar:
        _render_sidebar_header("Physics & Ballistics Labs", "4 discipline groups • 16 focused labs")
        if st.session_state.get("physics_lab_group") not in _group_names:
            st.session_state["physics_lab_group"] = _group_names[0]
        active_group = st.radio("Discipline Group", _group_names, key="physics_lab_group")
        group_labs = _phys_groups[active_group]
        if st.session_state.get("physics_active_lab") not in group_labs:
            st.session_state["physics_active_lab"] = group_labs[0]
        active_lab = st.radio("Select Lab", group_labs, key="physics_active_lab")
        st.divider()
        st.caption("Only the active lab runs on each interaction for fast responsiveness.")
        st.button(
            "Open C-UAS & Multi-Barrel Array",
            on_click=_navigate_to,
            args=("cuas", None, "array"),
            width="stretch",
            key="phys_jump_cuas_btn",
        )

    # --- SUB-LAB 1: STANAG 4367 INTERIOR BALLISTICS ---
    if active_lab == _PHYSICS_LABS[0]:
        st.subheader("🔥 Lumped-Parameter Thermodynamic Interior Ballistics (`IB-01` .. `IB-06`)")
        st.latex(
            r"""
            r(p) = \beta p^\alpha, \quad
            z(\phi) = \kappa \phi (1 + \lambda \phi + \mu \phi^2), \quad
            \bar{p}(x, v, z) = \frac{F C z - (\gamma - 1)\left[\frac{1}{2}\left(m_p + \frac{C}{3}\right)v^2 + E_{\text{loss}}\right]}{V_0 + A x - \frac{C}{\rho_p}(1 - z) - \eta C z}, \quad
            p_{\text{breech}} = p_{\text{base}}\left(1 + \frac{C}{2 m_p}\right)
            """
        )
        ib1, ib2, ib3, ib4 = st.columns(4)
        with ib1:
            ib_cal_mm = st.number_input("Bore Calibre (mm)", 20.0, 240.0, 155.0, 5.0)
            ib_travel_m = st.number_input("Projectile Bore Travel (m)", 0.5, 12.0, 6.0, 0.25)
        with ib2:
            ib_proj_kg = st.number_input("Projectile Mass (kg)", 0.1, 200.0, 43.2, 1.0)
            ib_chg_kg = st.number_input("Propellant Charge Mass (kg)", 0.05, 80.0, 11.5, 0.5)
        with ib3:
            ib_cham_l = st.number_input("Chamber Volume (Liters)", 0.1, 100.0, 23.0, 1.0)
            ib_imp_mj = st.slider("Propellant Impetus F (MJ/kg)", 0.80, 1.35, 1.05, 0.02)
        with ib4:
            ib_web_mm = st.slider("Propellant Half-Web e1 (mm)", 0.3, 2.5, 0.95, 0.05)
            ib_shot_mpa = st.slider("Engraving Shot-Start Pressure (MPa)", 10.0, 80.0, 30.0, 5.0)

        ib_cfg = LumpedIBConfig(
            calibre_m=float(ib_cal_mm) * 1e-3,
            travel_m=float(ib_travel_m),
            chamber_volume_m3=float(ib_cham_l) * 1e-3,
            proj_mass_kg=float(ib_proj_kg),
            charge_mass_kg=float(ib_chg_kg),
            impetus_J_kg=float(ib_imp_mj) * 1e6,
            half_web_m=float(ib_web_mm) * 1e-3,
            shot_start_Pa=float(ib_shot_mpa) * 1e6,
        )
        ib_out = solve_lumped_interior_ballistics(ib_cfg, dt=1e-5, record_history=True)
        hist = ib_out.get("history", {})

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Muzzle Velocity", f"{ib_out['muzzle_velocity_mps']:.1f} m/s")
        k2.metric("Peak Breech Pressure", f"{ib_out['peak_breech_pressure_Pa']*1e-6:.1f} MPa")
        k3.metric("Peak Mean Pressure", f"{ib_out['peak_mean_pressure_Pa']*1e-6:.1f} MPa")
        k4.metric("Time to Muzzle Exit", f"{ib_out['time_to_muzzle_s']*1000:.2f} ms")
        k5.metric("Ballistic Efficiency η_th", f"{ib_out['ballistic_efficiency']*100:.1f} %")

        if hist:
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                fig_ib_t = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["p_breech_MPa"],
                        name="Breech Pressure (MPa)",
                        line=dict(color="#DC2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["p_base_MPa"],
                        name="Base Pressure (MPa)",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                    ),
                    secondary_y=False,
                )
                fig_ib_t.add_trace(
                    go.Scatter(
                        x=hist["t_ms"],
                        y=hist["v_mps"],
                        name="Velocity (m/s)",
                        line=dict(color="#2563EB", width=3),
                    ),
                    secondary_y=True,
                )
                fig_ib_t.update_xaxes(title_text="In-Bore Time (ms)")
                fig_ib_t.update_yaxes(title_text="Pressure (MPa)", secondary_y=False)
                fig_ib_t.update_yaxes(title_text="Velocity (m/s)", secondary_y=True)
                fig_ib_t.update_layout(title="Pressure & Velocity vs Time", height=370, margin=dict(l=15, r=15, t=40, b=15))
                st.plotly_chart(fig_ib_t, width="stretch")

            with p_col2:
                fig_ib_x = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ib_x.add_trace(
                    go.Scatter(
                        x=hist["x_m"],
                        y=hist["p_breech_MPa"],
                        name="Breech Pressure (MPa)",
                        line=dict(color="#DC2626", width=3),
                    ),
                    secondary_y=False,
                )
                fig_ib_x.add_trace(
                    go.Scatter(
                        x=hist["x_m"],
                        y=[p * 100.0 for p in hist["phi"]],
                        name="Burnt Web Fraction (%)",
                        line=dict(color="#10B981", width=2.5, dash="dot"),
                    ),
                    secondary_y=True,
                )
                fig_ib_x.update_xaxes(title_text="Bore Travel X (m)")
                fig_ib_x.update_yaxes(title_text="Pressure (MPa)", secondary_y=False)
                fig_ib_x.update_yaxes(title_text="Burnt Fraction (%)", secondary_y=True)
                fig_ib_x.update_layout(title="Pressure & Burn Completion vs Travel", height=370, margin=dict(l=15, r=15, t=40, b=15))
                st.plotly_chart(fig_ib_x, width="stretch")

            with p_col3:
                vg_l = hist.get("v_gas_L", hist["x_m"])
                fig_pv = go.Figure()
                fig_pv.add_trace(
                    go.Scatter(
                        x=vg_l,
                        y=hist["p_mean_MPa"],
                        mode="lines",
                        fill="tozeroy",
                        name="Mean Gas Pressure P(V_g)",
                        line=dict(color="#7C3AED", width=3),
                        fillcolor="rgba(124, 58, 237, 0.16)",
                    )
                )
                fig_pv.update_layout(
                    title=f"Thermodynamic P-V Indicator Diagram (W_m = {ib_out['muzzle_energy_J']*1e-6:.2f} MJ)",
                    xaxis_title="Free Gas Expansion Volume V_g (Liters)",
                    yaxis_title="Mean Space Pressure P (MPa)",
                    height=370,
                    margin=dict(l=15, r=15, t=40, b=15),
                )
                st.plotly_chart(fig_pv, width="stretch")

    # --- SUB-LAB 2: ISA ATMOSPHERE, GYROSCOPIC STABILITY & CORIOLIS ---
    elif active_lab == _PHYSICS_LABS[1]:
        st.subheader("🌍 US Standard Atmosphere 1976 (`EB-04`), Gyroscopic Stability (`EB-07`/`EB-08`) & Coriolis (`EB-10`)")
        st.latex(
            r"""
            a(z) = \sqrt{\gamma R_{\text{air}} T(z)}, \quad
            \mu(T) = \beta_S \frac{T^{3/2}}{T + S}, \quad
            S_g = \frac{2 I_x^2 p^2}{\pi \rho I_y d^3 v^2 C_{M_\alpha}}, \quad
            \mathbf{a}_{\text{Cor}} = -2(\boldsymbol{\Omega}_\oplus \times \mathbf{v})
            """
        )
        at_c1, at_c2 = st.columns([1.25, 1.0])
        with at_c1:
            dt_isa = st.slider("Atmospheric Temperature Offset ΔT (K)", -30.0, 30.0, 0.0, 1.0)
            h_grid_m = np.linspace(0.0, 22000.0, 120)
            atm_prof = eb_04_isa_atmosphere(h_grid_m, dT_K=dt_isa)
            rho_p = np.asarray(atm_prof["rho_kg_m3"], dtype=np.float64)
            a_p = np.asarray(atm_prof["a_mps"], dtype=np.float64)
            t_p = np.asarray(atm_prof["T_K"], dtype=np.float64)
            mu_p = np.asarray(sutherland_viscosity_Pa_s(t_p), dtype=np.float64) * 1e6  # microPa*s

            fig_atm = make_subplots(specs=[[{"secondary_y": True}]])
            fig_atm.add_trace(
                go.Scatter(x=h_grid_m / 1000.0, y=rho_p, name="Air Density ρ (kg/m³)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_atm.add_trace(
                go.Scatter(
                    x=h_grid_m / 1000.0,
                    y=a_p,
                    name="Sound Speed a (m/s)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_atm.add_trace(
                go.Scatter(
                    x=h_grid_m / 1000.0,
                    y=mu_p * 15.0,
                    name="Sutherland Viscosity μ (×15 μPa·s)",
                    line=dict(color="#10B981", width=2, dash="dot"),
                ),
                secondary_y=True,
            )
            fig_atm.add_vline(x=11.0, line_dash="dot", line_color="#6B7280", annotation_text="Tropopause (11 km)")
            fig_atm.update_xaxes(title_text="Geopotential Altitude h (km)")
            fig_atm.update_yaxes(title_text="Density ρ (kg/m³)", secondary_y=False)
            fig_atm.update_yaxes(title_text="Acoustic Speed a (m/s) / Scaled μ", secondary_y=True)
            fig_atm.update_layout(title="US Standard Atmosphere 1976 & Sutherland Viscosity Profile", height=390)
            st.plotly_chart(fig_atm, width="stretch")

        with at_c2:
            st.markdown("##### 🧭 Gyroscopic Stability ($S_g$) & Coriolis Acceleration")
            sg_cal_mm = st.number_input("Projectile Calibre d (mm)", 5.56, 240.0, 155.0, 5.0, key="sg_cal")
            sg_twist = st.slider("Rifling Twist Rate (calibres/turn)", 10.0, 36.0, 20.0, 0.5)
            sg_v0 = st.slider("Launch Velocity V₀ (m/s)", 200.0, 1200.0, 827.0, 10.0)
            lat_deg = st.slider("Firing Latitude φ (deg, North +)", -75.0, 75.0, 45.0, 5.0)
            az_deg = st.slider("Firing Azimuth α (deg clockwise from North)", 0.0, 360.0, 90.0, 15.0)

            d_m_val = sg_cal_mm * 1e-3
            p0_spin = float(eb_09_muzzle_spin_rate(sg_v0, sg_twist, d_m_val))
            # Estimate axial and transverse moments of inertia for 43 kg / d^3 scaling
            m_est = 43.2 * ((d_m_val / 0.155) ** 3)
            ix_est = 0.135 * m_est * (d_m_val**2)
            iy_est = 1.45 * m_est * (d_m_val**2)
            sg_val = float(eb_07_gyroscopic_stability(ix_est, iy_est, p0_spin, 1.225, d_m_val, sg_v0, c_ma=2.8))
            a_cor_vec = eb_10_coriolis_acceleration(
                np.array([sg_v0 * 0.85, 0.0, sg_v0 * 0.52]),
                lat_rad=math.radians(lat_deg),
                az_rad=math.radians(az_deg),
            )
            st.table(
                pd.DataFrame(
                    [
                        {"Physical Quantity": "Initial Spin Rate p₀ (EB-09)", "Value": f"{p0_spin:.1f} rad/s ({p0_spin*60/(2*math.pi):.0f} RPM)"},
                        {"Physical Quantity": "STANAG 4355 Stability Factor S_g (EB-07)", "Value": f"{sg_val:.2f} ({'Dynamically Stable' if sg_val >= 1.3 else 'Marginal/Unstable'})"},
                        {"Physical Quantity": "Coriolis Downrange / Eötvös a_x (EB-10)", "Value": f"{a_cor_vec[0]:+.4f} m/s²"},
                        {"Physical Quantity": "Coriolis Crossrange Drift a_y (EB-10)", "Value": f"{-a_cor_vec[1]:+.4f} m/s² (Right +)"},
                        {"Physical Quantity": "Coriolis Vertical Eötvös a_z (EB-10)", "Value": f"{a_cor_vec[2]:+.4f} m/s² (Up +)"},
                    ]
                )
            )

    # --- SUB-LAB 3: SHOCK PHYSICS, RANKINE-HUGONIOT & FRIEDLANDER ---
    elif active_lab == _PHYSICS_LABS[2]:
        st.subheader("🌊 Rankine-Hugoniot Normal Shock Relations (`TB-03` .. `TB-08`)")
        st.latex(
            r"""
            Z = \frac{R}{W_{\text{TNT}}^{1/3}}, \quad
            P_r = 2 P_s \frac{7 P_a + 4 P_s}{7 P_a + P_s}, \quad
            q = \frac{5 P_s^2}{2(7 P_a + P_s)}, \quad
            P(t) = P_s \left(1 - \frac{t}{t_d}\right) e^{-b t / t_d}
            """
        )
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            z_grid = np.logspace(np.log10(0.15), np.log10(40.0), 90)
            ps_kb = np.asarray(tb_03_kingery_bulmash(z_grid, 1.0, param="Ps"), dtype=np.float64)
            ps_kg = np.asarray(tb_04_kinney_graham(z_grid), dtype=np.float64)
            pr_rh = np.asarray(tb_07_normal_reflected_pressure(ps_kb), dtype=np.float64)
            q_rh = np.asarray(tb_05_dynamic_pressure(ps_kb), dtype=np.float64)

            fig_rh = go.Figure()
            fig_rh.add_trace(go.Scatter(x=z_grid, y=pr_rh, name="Normal Reflected Pressure P_r (TB-07)", line=dict(color="#991B1B", width=3)))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=ps_kb, name="Incident Overpressure P_s (Kingery-Bulmash TB-03)", line=dict(color="#DC2626", width=3)))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=ps_kg, name="Free-Air P_s (Kinney-Graham TB-04)", line=dict(color="#F59E0B", width=2, dash="dash")))
            fig_rh.add_trace(go.Scatter(x=z_grid, y=q_rh, name="Dynamic Blast Wind Pressure q (TB-05)", line=dict(color="#2563EB", width=2.5, dash="dot")))
            fig_rh.update_layout(
                title="Rankine-Hugoniot Shock Overpressures vs Hopkinson-Cranz Scaled Distance Z",
                xaxis_title="Scaled Distance Z = R / W^(1/3) (m / kg^(1/3))",
                yaxis_title="Pressure (kPa)",
                xaxis_type="log",
                yaxis_type="log",
                height=400,
            )
            st.plotly_chart(fig_rh, width="stretch")

        with w_col2:
            w_kg = st.number_input("TNT Equivalent Mass W (kg)", 1.0, 2000.0, 25.0, 5.0)
            r_m = st.slider("Standoff Distance R (m)", 2.0, 120.0, 18.0, 1.0)
            z_sc = float(tb_01_scaled_distance(r_m, w_kg))
            kb_vals = tb_03_kingery_bulmash(z_sc, w_kg, param="all")
            assert isinstance(kb_vals, dict)
            ps_kpa = float(kb_vals.get("Ps", 50.0))
            td_ms = float(kb_vals.get("td", 10.0))
            ta_ms = float(kb_vals.get("ta", 5.0))

            t_wave = np.linspace(0.0, td_ms, 180)
            p_wave = np.array([float(tb_08_friedlander_waveform(t_val, ps_kpa, td_ms, b=1.8)) for t_val in t_wave])
            dt_w = t_wave[1] - t_wave[0] if len(t_wave) > 1 else 0.01
            imp_cum = np.cumsum(p_wave) * dt_w

            fig_wave = make_subplots(specs=[[{"secondary_y": True}]])
            fig_wave.add_trace(
                go.Scatter(
                    x=ta_ms + t_wave,
                    y=p_wave,
                    mode="lines",
                    fill="tozeroy",
                    name="Overpressure P(t) (kPa)",
                    line=dict(color="#DC2626", width=3),
                    fillcolor="rgba(220, 38, 38, 0.16)",
                ),
                secondary_y=False,
            )
            fig_wave.add_trace(
                go.Scatter(
                    x=ta_ms + t_wave,
                    y=imp_cum,
                    mode="lines",
                    name="Cumulative Specific Impulse I(t) (kPa·ms)",
                    line=dict(color="#1E3A8A", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_wave.update_xaxes(title_text="Time from Detonation (ms)")
            fig_wave.update_yaxes(title_text="Incident Overpressure P(t) (kPa)", secondary_y=False)
            fig_wave.update_yaxes(title_text="Impulse I(t) (kPa·ms)", secondary_y=True)
            fig_wave.update_layout(
                title=f"Friedlander Waveform & Impulse Integral at R={r_m:.1f}m (Z={z_sc:.2f})",
                height=400,
            )
            st.plotly_chart(fig_wave, width="stretch")

    # --- SUB-LAB 4: GURNEY ENERGY PARTITION, TAYLOR ANGLE & MOTT FRACTURE ---
    elif active_lab == _PHYSICS_LABS[3]:
        st.subheader("💥 Gurney Detonation Energy Partition (`TF-01`, `TF-02`), Taylor Angle (`TF-08`) & Mott Fracture (`TF-04`)")
        st.latex(
            r"""
            V_0 = \sqrt{2E_G}\left(\frac{M}{C} + \frac{n}{n+2}\right)^{-1/2}, \quad
            \eta_{\text{metal}} = \frac{\frac{1}{2}M V_0^2}{C E_G} = \left(1 + \frac{n}{n+2}\frac{C}{M}\right)^{-1}, \quad
            \theta_{\text{Taylor}} = 2 \arcsin\!\left(\frac{V_0}{2 D_{\text{det}}}\right), \quad
            N(>m) = N_0 e^{-\sqrt{m}/M_K}
            """
        )
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            exp_sel = st.selectbox("Reference Energetic Fill", ["Composition B", "TNT", "Octol 75/25", "RDX", "HMX"], index=0)
            exp_props = kb.get_explosive_properties(exp_sel)
            g_const = float(exp_props["gurney_mps"])
            d_det = float(exp_props["det_vel_mps"])

            cm_grid = np.linspace(0.05, 2.5, 80)
            mc_grid = 1.0 / cm_grid
            v_cyl = np.asarray(tf_01_gurney_cyl_sphere(g_const, mc_grid, 1.0, "cylinder"), dtype=np.float64)
            v_sph = np.asarray(tf_01_gurney_cyl_sphere(g_const, mc_grid, 1.0, "sphere"), dtype=np.float64)
            v_sym = np.asarray(tf_02_gurney_sandwiches(g_const, mc_grid, 1.0, "symmetric"), dtype=np.float64)
            eta_cyl = (mc_grid * (v_cyl**2)) / (g_const**2) * 100.0

            fig_gur = make_subplots(specs=[[{"secondary_y": True}]])
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_cyl, name="Cylinder V₀ (TF-01)", line=dict(color="#2563EB", width=3)), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_sph, name="Sphere V₀ (TF-01)", line=dict(color="#7C3AED", width=2.5, dash="dash")), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=v_sym, name="Symmetric Sandwich V₀ (TF-02)", line=dict(color="#059669", width=2, dash="dot")), secondary_y=False)
            fig_gur.add_trace(go.Scatter(x=cm_grid, y=eta_cyl, name="Cylinder Energy Efficiency η_m (%)", line=dict(color="#DC2626", width=2.5)), secondary_y=True)
            fig_gur.update_xaxes(title_text="Charge-to-Metal Mass Ratio C / M (-)")
            fig_gur.update_yaxes(title_text="Initial Velocity V₀ (m/s)", secondary_y=False)
            fig_gur.update_yaxes(title_text="Energy Partition Efficiency η_m (%)", secondary_y=True)
            fig_gur.update_layout(
                title=f"Gurney Velocity & Energy Partition ({exp_sel}: √2E = {g_const:.0f} m/s, D = {d_det:.0f} m/s)",
                height=390,
            )
            st.plotly_chart(fig_gur, width="stretch")

        with g_col2:
            m_case = st.number_input("Casing Metal Mass (kg)", 2.0, 200.0, 34.0, 2.0)
            t_in = st.slider("Casing Wall Thickness (in)", 0.15, 1.50, 0.55, 0.05)
            d_in = st.slider("Internal Casing Diameter (in)", 1.5, 10.0, 4.8, 0.2)
            mott_out = tf_04_mott_distribution(M0_kg=m_case, t_in=t_in, d_in=d_in, B=float(exp_props["mott_b"]))
            avg_g = float(mott_out["m_avg_g"])
            n_tot = float(mott_out["total_fragments"])
            m_grid_g = np.linspace(0.1, max(avg_g * 6.0, 5.0), 80)
            mk_lb_sqrt = float(mott_out["MK_lb_sqrt"])
            m_grid_lb = (m_grid_g / 1000.0) * 2.20462262185
            n_gt_m = n_tot * np.exp(-np.sqrt(m_grid_lb) / max(mk_lb_sqrt, 1e-9))
            fig_mott = go.Figure()
            fig_mott.add_trace(
                go.Scatter(
                    x=m_grid_g,
                    y=n_gt_m,
                    mode="lines",
                    name="Cumulative Fragments N(>m)",
                    line=dict(color="#1E3A8A", width=3),
                )
            )
            fig_mott.update_layout(
                title=f"Mott Statistical Fracture Curve (N_tot={n_tot:.0f}, Mean Mass={avg_g:.2f} g)",
                xaxis_title="Fragment Mass Threshold m (grams)",
                yaxis_title="Number of Fragments N(>m)",
                yaxis_type="log",
                height=390,
            )
            st.plotly_chart(fig_mott, width="stretch")

    # --- SUB-LAB 5: CONTINUUM PENETRATION MECHANICS ---
    elif active_lab == _PHYSICS_LABS[4]:
        st.subheader("🛡️ Continuum Penetration Mechanics (`TP-01` Lanz-Odermatt, `TP-02` Birkhoff, `TP-04` Lambert-Jonas)")
        st.latex(
            r"""
            \frac{P}{L_w} = \frac{a}{\tanh(b_0 + b_1 L_w/D)} (\cos\theta)^m \sqrt{\frac{\rho_p}{\rho_t}} \exp\!\left(-\frac{s^2}{V_s^2}\right), \quad
            P_{\text{hydro}} = L_{\text{jet}}\sqrt{\frac{\rho_j}{\rho_t}}, \quad
            V_r = a\left(V_s^p - V_{50}^p\right)^{1/p}
            """
        )
        pen_c1, pen_c2 = st.columns(2)
        with pen_c1:
            lw_d_ratio = st.slider("Penetrator Aspect Ratio L_w / D", 10.0, 35.0, 22.0, 1.0)
            bhn_tgt = st.slider("Target RHA Brinell Hardness (BHN)", 180.0, 500.0, 300.0, 10.0)
            obliq_deg = st.slider("Impact Obliquity Angle θ (deg)", 0.0, 65.0, 30.0, 5.0)

            v_grid = np.linspace(800.0, 2200.0, 80)
            d_rod = 0.025
            lw_rod = lw_d_ratio * d_rod
            ob_rad = math.radians(obliq_deg)

            p_w = np.asarray(
                tp_01_lanz_odermatt(
                    lw_rod, d_rod, v_grid, rho_p_kg_m3=17600.0, rho_t_kg_m3=7850.0, bhn_t=bhn_tgt, obliquity_rad=ob_rad, rod_material="Tungsten alloy"
                )["penetration"],
                dtype=np.float64,
            ) / lw_rod
            p_st = np.asarray(
                tp_01_lanz_odermatt(
                    lw_rod, d_rod, v_grid, rho_p_kg_m3=7850.0, rho_t_kg_m3=7850.0, bhn_t=bhn_tgt, obliquity_rad=ob_rad, rod_material="Steel"
                )["penetration"],
                dtype=np.float64,
            ) / lw_rod
            hydro_w = math.sqrt(17600.0 / 7850.0) * (math.cos(ob_rad) ** (-0.225))

            fig_lo = go.Figure()
            fig_lo.add_trace(go.Scatter(x=v_grid, y=p_w, name="Tungsten Alloy Rod (ρ=17.6 g/cm³)", line=dict(color="#1E3A8A", width=3.5)))
            fig_lo.add_trace(go.Scatter(x=v_grid, y=p_st, name="High-Hardness Steel Rod (ρ=7.85 g/cm³)", line=dict(color="#059669", width=3, dash="dash")))
            fig_lo.add_hline(y=hydro_w, line_dash="dot", line_color="#DC2626", annotation_text="Birkhoff Hydrodynamic Limit √(ρ_p/ρ_t)")
            fig_lo.update_layout(
                title=f"Lanz-Odermatt Normalized Penetration P/L_w vs Impact Velocity (θ={obliq_deg:.0f}°, BHN={bhn_tgt:.0f})",
                xaxis_title="Striking Velocity V_s (m/s)",
                yaxis_title="Normalized Penetration Depth P / L_w (-)",
                height=390,
            )
            st.plotly_chart(fig_lo, width="stretch")

        with pen_c2:
            v50_bl = st.slider("Ballistic Limit Velocity V₅₀ (m/s)", 200.0, 1200.0, 550.0, 25.0)
            sig_v50 = st.slider("Probit Zone Standard Deviation σ_V50 (m/s)", 5.0, 80.0, 25.0, 5.0)
            lj_p = st.slider("Lambert-Jonas Exponent p (-)", 1.5, 3.0, 2.0, 0.1)

            vs_grid = np.linspace(max(100.0, v50_bl * 0.5), v50_bl * 1.8, 100)
            vr_grid = np.asarray(tp_04_lambert_jonas(vs_grid, v50_bl, a=0.92, p=lj_p), dtype=np.float64)
            prob_perf = np.asarray(tp_06_perforation_probability(vs_grid, v50_bl, sig_v50), dtype=np.float64)

            fig_lj = make_subplots(specs=[[{"secondary_y": True}]])
            fig_lj.add_trace(
                go.Scatter(x=vs_grid, y=vr_grid, name="Residual Velocity V_r (TP-04)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_lj.add_trace(
                go.Scatter(
                    x=vs_grid,
                    y=prob_perf,
                    name="Probit Perforation Prob P(perf|V_s) (TP-06)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_lj.add_vline(x=v50_bl, line_dash="dot", line_color="#6B7280", annotation_text=f"V₅₀ = {v50_bl:.0f} m/s")
            fig_lj.update_xaxes(title_text="Striking Velocity V_s (m/s)")
            fig_lj.update_yaxes(title_text="Residual Velocity V_r (m/s)", secondary_y=False)
            fig_lj.update_yaxes(title_text="Perforation Probability (-)", secondary_y=True)
            fig_lj.update_layout(title="Lambert-Jonas Residual Velocity & Probit Perforation Curve", height=390)
            st.plotly_chart(fig_lj, width="stretch")

        st.markdown("---")
        st.subheader("🧱 Project THOR 17-Material Fragment Perforation (`TF-12`) & De Marre Armor Scaling (`TF-13`)")
        st.latex(
            r"""
            V_{50} = 0.3048 \cdot 10^{c_1} (t A)^{\alpha_1} (15.4324\, m_f)^{\beta_1} (\sec\theta)^{\gamma_1}, \qquad
            V_r = V_s - 0.3048 \cdot 10^{c_2} (t A)^{\alpha_2} (15.4324\, m_f)^{\beta_2} (\sec\theta)^{\gamma_2} V_s^{\lambda_2}
            """
        )
        th_c1, th_c2, th_c3, th_c4 = st.columns(4)
        with th_c1:
            thor_m_g = st.slider("Fragment Mass m_f (grams)", 0.5, 50.0, 6.5, 0.5, key="thor_m")
        with th_c2:
            thor_t_mm = st.slider("Target Plate Thickness t (mm)", 1.0, 25.0, 6.35, 0.5, key="thor_t")
        with th_c3:
            thor_ob_deg = st.slider("Impact Obliquity θ (deg)", 0.0, 60.0, 15.0, 5.0, key="thor_ob")
        with th_c4:
            thor_vs = st.slider("Striking Velocity V_s (m/s)", 300.0, 2200.0, 1150.0, 50.0, key="thor_vs")

        thor_materials = [m["material"] for m in kb.thor.get("materials", [])]
        thor_rows = []
        for mat_name in thor_materials:
            tr_o = tf_12_thor_penetration(
                m_g=float(thor_m_g),
                t_cm=float(thor_t_mm) * 0.1,
                v_s_mps=float(thor_vs),
                obliquity_rad=math.radians(float(thor_ob_deg)),
                target_material=mat_name,
            )
            thor_rows.append(
                {
                    "Target Material": mat_name,
                    "V50 Ballistic Limit (m/s)": round(float(tr_o["v50_mps"]), 1),
                    "Residual Velocity V_r (m/s)": round(float(tr_o["v_residual_mps"]), 1),
                    "Perforated": "✅ YES" if tr_o["perforated"] else "❌ NO",
                }
            )
        thor_df = pd.DataFrame(thor_rows).sort_values("V50 Ballistic Limit (m/s)", ascending=True)
        frag_m_kg = float(thor_m_g) * 1e-3
        frag_d_m = (frag_m_kg / 7850.0) ** (1.0 / 3.0)
        dm_pen_in = float(
            tf_13_de_marre_penetration(
                D_in=frag_d_m / 0.0254,
                W_lb=frag_m_kg * 2.20462262,
                V_fps=float(thor_vs) * 3.2808399,
                obliquity_rad=math.radians(float(thor_ob_deg)),
            )
        )
        dm_pen_m = dm_pen_in * 0.0254

        th_p1, th_p2 = st.columns([1.3, 1.0])
        with th_p1:
            fig_thor = go.Figure()
            fig_thor.add_trace(
                go.Bar(
                    x=thor_df["V50 Ballistic Limit (m/s)"],
                    y=thor_df["Target Material"],
                    orientation="h",
                    name="Ballistic Limit V₅₀ (m/s)",
                    marker_color="#1E3A8A",
                )
            )
            fig_thor.add_trace(
                go.Bar(
                    x=thor_df["Residual Velocity V_r (m/s)"],
                    y=thor_df["Target Material"],
                    orientation="h",
                    name=f"Residual Velocity V_r at V_s={thor_vs:.0f} m/s",
                    marker_color="#10B981",
                )
            )
            fig_thor.add_vline(x=thor_vs, line_dash="dash", line_color="#DC2626", annotation_text=f"V_s = {thor_vs:.0f} m/s")
            fig_thor.update_layout(
                title=f"Project THOR Ballistic Limit V₅₀ & Residual Velocity Across {len(thor_df)} Materials (m_f={thor_m_g:.1f}g, t={thor_t_mm:.1f}mm)",
                xaxis_title="Velocity (m/s)",
                barmode="group",
                height=440,
                margin=dict(l=15, r=15, t=40, b=15),
            )
            st.plotly_chart(fig_thor, width="stretch")
        with th_p2:
            st.metric("De Marre Mild-Steel Penetration (TF-13)", f"{dm_pen_m * 1000.0:.2f} mm", delta=f"Fragment KE: {0.5 * (thor_m_g * 1e-3) * (thor_vs**2):.0f} J", delta_color="off")
            st.dataframe(thor_df, width="stretch", height=365)

        st.divider()
        st.subheader("🧱 Concrete Penetration (Forrestal–Frew & Modified NDRC), Tate–Alekseevskii Long-Rod ODE & CPR 16E Probits")
        t_conc, t_tate, t_tno = st.tabs(
            [
                "🧱 Concrete & Slab Limits (S73)",
                "🎯 Tate–Alekseevskii ODE (S74)",
                "🫧 CPR 16E Blast & Thermal Probits (S78)",
            ]
        )
        with t_conc:
            cc1, cc2, cc3 = st.columns(3)
            c_mass = cc1.slider("Projectile mass (kg)", 0.5, 50.0, 10.0, 0.5, key="pl5_conc_mass")
            c_diam_mm = cc1.slider("Projectile diameter (mm)", 20.0, 203.0, 105.0, 5.0, key="pl5_conc_d")
            c_vel = cc2.slider("Impact velocity (m/s)", 200.0, 1600.0, 800.0, 25.0, key="pl5_conc_v")
            c_fc = cc2.slider("Concrete compressive strength f'c (MPa)", 20.0, 80.0, 35.0, 1.0, key="pl5_conc_fc")
            c_crh = cc3.slider("Ogive CRH (caliber radius head)", 1.0, 6.0, 3.0, 0.25, key="pl5_conc_crh")
            c_wall = cc3.slider("Target slab thickness (m)", 0.2, 3.0, 0.85, 0.05, key="pl5_conc_wall")

            ff_res = forrestal_frew_concrete_penetration(
                mass_kg=c_mass,
                diameter_m=c_diam_mm / 1000.0,
                velocity_mps=c_vel,
                crh_psi=c_crh,
                fc_mpa=c_fc,
            )
            ndrc_res = ndrc_concrete_perforation_and_scabbing(
                mass_kg=c_mass,
                diameter_m=c_diam_mm / 1000.0,
                velocity_mps=c_vel,
                fc_mpa=c_fc,
                wall_thickness_m=c_wall,
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Forrestal–Frew DoP", f"{ff_res['penetration_mm']:.0f} mm", f"S={ff_res['s_resistance_factor']:.2f}")
            mc2.metric("Modified NDRC DoP", f"{ndrc_res['penetration_mm']:.0f} mm", ff_res["regime"])
            mc3.metric(
                "Perforation Limit",
                f"{ndrc_res['perforation_limit_mm']:.0f} mm",
                "PERFORATES" if ndrc_res["perforates_wall"] else "STOPPED",
            )
            mc4.metric(
                "Scabbing Limit",
                f"{ndrc_res['scabbing_limit_mm']:.0f} mm",
                "SPALLS" if ndrc_res["spalls_back_face"] else "INTACT",
            )

            v_sweep_c = np.linspace(250.0, 1500.0, 40)
            ff_curve = [
                forrestal_frew_concrete_penetration(c_mass, c_diam_mm / 1000.0, float(v), crh_psi=c_crh, fc_mpa=c_fc)["penetration_mm"]
                for v in v_sweep_c
            ]
            ndrc_dop_c = [
                ndrc_concrete_perforation_and_scabbing(c_mass, c_diam_mm / 1000.0, float(v), fc_mpa=c_fc)["penetration_mm"]
                for v in v_sweep_c
            ]
            ndrc_perf_c = [
                ndrc_concrete_perforation_and_scabbing(c_mass, c_diam_mm / 1000.0, float(v), fc_mpa=c_fc)["perforation_limit_mm"]
                for v in v_sweep_c
            ]
            fig_conc = go.Figure()
            fig_conc.add_trace(go.Scatter(x=v_sweep_c, y=ff_curve, name="Forrestal–Frew DoP (mm)", line=dict(width=3)))
            fig_conc.add_trace(go.Scatter(x=v_sweep_c, y=ndrc_dop_c, name="Modified NDRC DoP (mm)", line=dict(dash="dash")))
            fig_conc.add_trace(go.Scatter(x=v_sweep_c, y=ndrc_perf_c, name="NDRC Perforation Limit (mm)", line=dict(dash="dot")))
            fig_conc.add_hline(y=c_wall * 1000.0, line_dash="dashdot", line_color="red", annotation_text=f"Slab {c_wall:.2f} m")
            fig_conc.update_layout(
                title="Concrete Penetration & Slab Perforation Threshold vs Impact Velocity",
                xaxis_title="Impact Velocity (m/s)",
                yaxis_title="Depth / Thickness (mm)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_conc, width="stretch")

        with t_tate:
            tc1, tc2 = st.columns(2)
            t_v = tc1.slider("Long-rod impact velocity (m/s)", 800.0, 2200.0, 1650.0, 25.0, key="pl5_tate_v")
            t_len = tc1.slider("Rod length L0 (m)", 0.10, 0.80, 0.50, 0.02, key="pl5_tate_l")
            t_rt = tc2.slider("Target resistance Rt (GPa)", 2.0, 10.0, 5.5, 0.25, key="pl5_tate_rt")
            t_yp = tc2.slider("Penetrator yield Yp (GPa)", 0.5, 3.5, 1.5, 0.1, key="pl5_tate_yp")

            tate_res = tate_alekseevskii_penetration(
                v_impact_mps=t_v,
                rod_length_m=t_len,
                y_p_gpa=t_yp,
                r_t_gpa=t_rt,
            )
            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Final Penetration", f"{tate_res['penetration_mm']:.1f} mm", f"P/L = {tate_res['penetration_over_length']:.2f}")
            tm2.metric("Hydrodynamic Limit P/L", f"{tate_res['hydrodynamic_limit_p_over_l']:.2f}")
            tm3.metric("Critical Velocity Vcrit", f"{tate_res['critical_velocity_mps']:.0f} m/s")
            tm4.metric("Residual Rod", f"{tate_res['residual_rod_length_mm']:.1f} mm", f"{tate_res['duration_us']:.0f} µs")

            hist_t = tate_res["history"]
            fig_tate = make_subplots(specs=[[{"secondary_y": True}]])
            fig_tate.add_trace(
                go.Scatter(x=hist_t["t_us"], y=hist_t["penetration_mm"], name="Penetration Depth (mm)"),
                secondary_y=False,
            )
            fig_tate.add_trace(
                go.Scatter(x=hist_t["t_us"], y=hist_t["rod_length_mm"], name="Residual Rod Length (mm)", line=dict(dash="dash")),
                secondary_y=False,
            )
            fig_tate.add_trace(
                go.Scatter(x=hist_t["t_us"], y=hist_t["tail_velocity_mps"], name="Tail Velocity (m/s)"),
                secondary_y=True,
            )
            fig_tate.add_trace(
                go.Scatter(x=hist_t["t_us"], y=hist_t["interface_velocity_mps"], name="Interface Velocity u (m/s)", line=dict(dash="dot")),
                secondary_y=True,
            )
            fig_tate.update_layout(
                title="Tate–Alekseevskii Erosion & Deceleration Time History (WHA Rod vs Steel)",
                xaxis_title="Time (µs)",
                template="plotly_white",
                height=350,
            )
            fig_tate.update_yaxes(title_text="Length / Depth (mm)", secondary_y=False)
            fig_tate.update_yaxes(title_text="Velocity (m/s)", secondary_y=True)
            st.plotly_chart(fig_tate, width="stretch")

        with t_tno:
            pc1, pc2 = st.columns(2)
            p_ear_kpa = pc1.slider("Incident overpressure Ps for eardrum (kPa)", 5.0, 200.0, 35.0, 2.5, key="pl5_tno_ear")
            p_lung_kpa = pc1.slider("Effective overpressure Peff for lung (kPa)", 100.0, 1200.0, 650.0, 25.0, key="pl5_tno_lung_p")
            i_lung_pas = pc2.slider("Blast impulse I for lung (Pa·s)", 100.0, 5000.0, 2400.0, 50.0, key="pl5_tno_lung_i")
            tdu_val = pc2.slider("Thermal dose TDU ((kW/m²)^4/3·s)", 100.0, 5000.0, 2400.0, 50.0, key="pl5_tno_tdu")

            ear_res = tno_eardrum_rupture_probit(p_ear_kpa * 1000.0)
            lung_res = tno_lung_lethality_probit(p_lung_kpa * 1000.0, i_lung_pas)
            therm_res = tno_thermal_lethality_probit(tdu_val)

            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Eardrum Rupture (CPR 16E)", f"{ear_res['probability'] * 100.0:.1f}%", f"Y = {ear_res['probit']:.2f}")
            pm2.metric("Lung Lethality (CPR 16E)", f"{lung_res['probability'] * 100.0:.1f}%", f"Y = {lung_res['probit']:.2f}")
            pm3.metric("Thermal Lethality (CPR 16E)", f"{therm_res['probability'] * 100.0:.1f}%", f"Y = {therm_res['probit']:.2f}")

            ps_grid = np.linspace(5.0, 200.0, 60)
            ear_probs = np.asarray(tno_eardrum_rupture_probit(ps_grid * 1000.0)["probability"]) * 100.0
            peff_grid = np.linspace(150.0, 1200.0, 60)
            lung_probs = np.asarray(tno_lung_lethality_probit(peff_grid * 1000.0, np.full_like(peff_grid, i_lung_pas))["probability"]) * 100.0
            fig_tno = go.Figure()
            fig_tno.add_trace(go.Scatter(x=ps_grid, y=ear_probs, name="Eardrum Rupture vs Ps (kPa)"))
            fig_tno.add_trace(go.Scatter(x=peff_grid, y=lung_probs, name=f"Lung Lethality vs Peff (I={i_lung_pas:.0f} Pa·s)"))
            fig_tno.update_layout(
                title="CPR 16E Injury & Lethality Probit Curves",
                xaxis_title="Overpressure (kPa)",
                yaxis_title="Response Probability (%)",
                template="plotly_white",
                height=340,
            )
            st.plotly_chart(fig_tno, width="stretch")

    # --- SUB-LAB 6: QUASI-MONTE CARLO DISCREPANCY & KNOWLEDGE BASE ---
    elif active_lab == _PHYSICS_LABS[5]:
        st.subheader("📐 Quasi-Monte Carlo Low-Discrepancy Sampling (`MC-05`, `MC-06`) & Salvo Correlation (`TV-04`, `TV-05`)")
        st.latex(
            r"""
            \left|\frac{1}{N}\sum_{i=1}^N f(\mathbf{u}_i) - \int_{[0,1]^d} f(\mathbf{u})\,d\mathbf{u}\right| \le V_{\text{HK}}(f)\, D_N^*(\mathbf{u}_1,\dots,\mathbf{u}_N), \quad
            D_N^*(\text{Sobol}) = \mathcal{O}\!\left(\frac{(\log N)^d}{N}\right) \ll \mathcal{O}(N^{-1/2})
            """
        )
        qmc_c1, qmc_c2 = st.columns(2)
        with qmc_c1:
            n_qmc = st.slider("Sample Points N in Unit Square [0, 1]²", 16, 256, 64, 16)
            u_sobol = sample_unit_hypercube(n_qmc, 2, method="Sobol", rng=42)
            u_lhs = sample_unit_hypercube(n_qmc, 2, method="LHS", rng=42)
            u_mc = sample_unit_hypercube(n_qmc, 2, method="MC", rng=42)

            fig_qmc = go.Figure()
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_sobol[:, 0],
                    y=u_sobol[:, 1],
                    mode="markers",
                    name="Sobol Low-Discrepancy (MC-06)",
                    marker=dict(color="#2563EB", size=8, symbol="circle"),
                )
            )
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_lhs[:, 0],
                    y=u_lhs[:, 1],
                    mode="markers",
                    name="Latin Hypercube LHS (MC-05)",
                    marker=dict(color="#10B981", size=7, symbol="diamond"),
                )
            )
            fig_qmc.add_trace(
                go.Scatter(
                    x=u_mc[:, 0],
                    y=u_mc[:, 1],
                    mode="markers",
                    name="Crude Pseudorandom MC (MC-04)",
                    marker=dict(color="#DC2626", size=6, symbol="cross", opacity=0.65),
                )
            )
            fig_qmc.update_layout(
                title=f"2D Parameter-Space Coverage: Sobol vs LHS vs Crude MC (N={n_qmc})",
                xaxis_title="Uniform Dimension U₁",
                yaxis_title="Uniform Dimension U₂",
                height=380,
            )
            st.plotly_chart(fig_qmc, width="stretch")

        with qmc_c2:
            r_dam_m = st.slider("Carleton Effective Radius R_D (m)", 5.0, 60.0, 25.0, 1.0)
            sig_bias_m = st.slider("Shared Occasion Bias Sigma (m)", 0.0, 60.0, 22.0, 2.0)
            sig_rr_m = st.slider("Round-to-Round Random Sigma (m)", 2.0, 60.0, 14.0, 2.0)

            sig_tot_m = math.hypot(sig_bias_m, sig_rr_m)
            sspk_1 = float(tv_04_sspk_carleton(r_dam_m, sig_tot_m, p_dud=0.02))
            n_vec = list(range(1, 17))
            pk_indep = [float(tv_05_salvo_pk_independent(sspk_1, n)) for n in n_vec]

            rng_c = np.random.default_rng(42)
            bx = rng_c.normal(0.0, sig_bias_m, (600, 1))
            by = rng_c.normal(0.0, sig_bias_m, (600, 1))
            rx = rng_c.normal(0.0, sig_rr_m, (600, 16))
            ry = rng_c.normal(0.0, sig_rr_m, (600, 16))
            r2_miss = (bx + rx) ** 2 + (by + ry) ** 2
            p_round = 0.98 * np.exp(-r2_miss / (2.0 * (r_dam_m / 1.1774) ** 2))
            surv_cum = np.cumprod(1.0 - p_round, axis=1)
            pk_corr = [float(np.mean(1.0 - surv_cum[:, n - 1])) for n in n_vec]

            fig_salvo = go.Figure()
            fig_salvo.add_trace(
                go.Scatter(
                    x=n_vec,
                    y=pk_indep,
                    mode="lines+markers",
                    name="Independent Salvo Assumption (TV-05)",
                    line=dict(color="#9CA3AF", width=2.5, dash="dash"),
                )
            )
            fig_salvo.add_trace(
                go.Scatter(
                    x=n_vec,
                    y=pk_corr,
                    mode="lines+markers",
                    name="Correlated Occasion-Bias Salvo (True MC)",
                    line=dict(color="#2563EB", width=3.5),
                )
            )
            fig_salvo.update_layout(
                title=f"Salvo Effectiveness vs Round Count N (Single-Shot SSPK = {sspk_1:.3f})",
                xaxis_title="Salvo Size N (Rounds Fired on Same Occasion)",
                yaxis_title="Cumulative Salvo Effect Probability Pk",
                height=380,
            )
            st.plotly_chart(fig_salvo, width="stretch")

        st.subheader(
            f"📚 Authoritative Knowledge Base Explorer ({len(kb.formulas)} Formulas & {len(kb.sources)} Primary Sources)"
        )
        kb_tab_choice = st.selectbox(
            "Select Knowledge Base Sheet",
            [
                f"Formulas ({len(kb.formulas)})",
                "Explosives",
                "THOR_Coefficients",
                "Ammo_Taxonomy",
                f"Sources ({len(kb.sources)})",
            ],
        )
        if kb_tab_choice.startswith("Formulas"):
            forms_df = pd.DataFrame(kb.formulas)
            cols_f = [
                c
                for c in ["formula_id", "module", "name", "verification_status", "is_flagged", "source_id", "formula_text"]
                if c in forms_df.columns
            ]
            st.dataframe(
                forms_df[cols_f],
                width="stretch",
            )
        elif kb_tab_choice == "Explosives":
            st.dataframe(pd.DataFrame(kb.explosives.get("tnt_equivalence", [])), width="stretch")
        elif kb_tab_choice == "THOR_Coefficients":
            st.dataframe(pd.DataFrame(kb.thor.get("materials", [])), width="stretch")
        elif kb_tab_choice == "Ammo_Taxonomy":
            st.dataframe(pd.DataFrame(kb.taxonomy), width="stretch")
        else:
            st.dataframe(pd.DataFrame(kb.sources), width="stretch")

    # --- SUB-LAB 7: STANAG 4355 MPM YAW-OF-REPOSE, SPIN-DAMPING & MAGNUS DRIFT ---
    elif active_lab == _PHYSICS_LABS[6]:
        st.subheader("🌀 STANAG 4355 Modified Point Mass (MPM): Yaw of Repose (`EB-05`), Spin Decay (`EB-06`) & Gyroscopic Drift")
        st.latex(
            r"""
            \boldsymbol{\alpha}_e = -\frac{2 I_x p}{\rho S d C_{M_\alpha} v^4}\,(\mathbf{v} \times \mathbf{a}), \qquad
            \frac{dp}{dt} = \frac{\rho S d^2 v}{2 I_x}\,C_{\text{spin}}\,p, \qquad
            \mathbf{a}_{\text{MPM}} = \mathbf{a}_{\text{drag}}(\alpha_e^2) + \mathbf{a}_{\text{lift}}(\boldsymbol{\alpha}_e) + \mathbf{a}_{\text{Magnus}}(p, \boldsymbol{\alpha}_e)
            """
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            mpm_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 300.0, 950.0, 685.0, 15.0, key="mpm_v0")
        with m2:
            mpm_qe_deg = st.slider("Quadrant Elevation QE (deg)", 5.0, 65.0, 32.0, 1.0, key="mpm_qe")
        with m3:
            mpm_twist = st.slider("Rifling Twist (cal/turn)", 12.0, 30.0, 20.0, 1.0, key="mpm_twist")
        with m4:
            mpm_lat_deg = st.slider("Firing Latitude φ (deg)", -60.0, 60.0, 45.0, 5.0, key="mpm_lat")

        mpm_res = simulate_mpm_vs_3dof_trajectory(
            v0_mps=float(mpm_v0),
            qe_deg=float(mpm_qe_deg),
            twist_cal_per_turn=float(mpm_twist),
            lat_deg=float(mpm_lat_deg),
        )
        p_l2 = mpm_res["path_l2"]
        p_l1 = mpm_res["path_l1"]
        rpm_t = np.asarray(mpm_res["p_spin_rad_s"], dtype=np.float64) * 60.0 / (2.0 * math.pi)
        yaw_deg_t = np.degrees(np.asarray(mpm_res["alpha_e_mrad"], dtype=np.float64) * 1e-3)

        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("MPM Lateral Spin Drift", f"{mpm_res['terminal_spin_drift_m']:+.1f} m (Right)", delta=f"3-DoF Defl: {mpm_res['l1_deflection_m']:+.1f} m", delta_color="off")
        mk2.metric("Induced Yaw Range Delta", f"{mpm_res['l2_range_m'] - mpm_res['l1_range_m']:+.1f} m", delta=f"MPM Range: {mpm_res['l2_range_m']:.0f} m", delta_color="off")
        mk3.metric("Apex Yaw of Repose |α_e|", f"{mpm_res['alpha_e_peak_mrad']:.2f} mrad", delta=f"Launch Sg: {mpm_res['sg_muzzle']:.2f}", delta_color="off")
        mk4.metric("Terminal Axial Spin Rate", f"{float(rpm_t[-1]):.0f} RPM", delta=f"Initial: {float(rpm_t[0]):.0f} RPM", delta_color="off")

        mc_left, mc_right = st.columns(2)
        with mc_left:
            fig_drift = go.Figure()
            fig_drift.add_trace(
                go.Scatter(
                    x=p_l2["range_m"],
                    y=p_l2["deflection_m"],
                    mode="lines",
                    name="STANAG 4355 MPM (L2: Yaw + Magnus + Spin Drift)",
                    line=dict(color="#1E3A8A", width=3.5),
                )
            )
            fig_drift.add_trace(
                go.Scatter(
                    x=p_l1["range_m"],
                    y=p_l1["deflection_m"],
                    mode="lines",
                    name="3-DoF Point-Mass (L1: Coriolis Only)",
                    line=dict(color="#DC2626", width=2.5, dash="dash"),
                )
            )
            fig_drift.update_layout(
                title="Ground-Track Deflection Drift: STANAG 4355 MPM (L2) vs 3-DoF Point-Mass (L1)",
                xaxis_title="Downrange X (m)",
                yaxis_title="Right Deflection Drift Y_right (m)",
                height=390,
            )
            st.plotly_chart(fig_drift, width="stretch")

        with mc_right:
            fig_yaw = make_subplots(specs=[[{"secondary_y": True}]])
            fig_yaw.add_trace(
                go.Scatter(
                    x=mpm_res["t_s"],
                    y=yaw_deg_t,
                    name="Yaw of Repose |α_e| (deg)",
                    line=dict(color="#7C3AED", width=3),
                ),
                secondary_y=False,
            )
            fig_yaw.add_trace(
                go.Scatter(
                    x=mpm_res["t_s"],
                    y=rpm_t,
                    name="Axial Spin Rate (RPM)",
                    line=dict(color="#059669", width=2.5, dash="dash"),
                ),
                secondary_y=True,
            )
            fig_yaw.update_xaxes(title_text="Time of Flight t (s)")
            fig_yaw.update_yaxes(title_text="Yaw of Repose |α_e| (deg)", secondary_y=False)
            fig_yaw.update_yaxes(title_text="Axial Spin Rate (RPM)", secondary_y=True)
            fig_yaw.update_layout(
                title="In-Flight Yaw-of-Repose Peak at Trajectory Apex & Exponential Spin Decay",
                height=390,
            )
            st.plotly_chart(fig_yaw, width="stretch")

    # --- SUB-LAB 8: BAYESIAN FUZE RELIABILITY, GRAZE S-CURVE SURFACE & UXO RISK ---
    elif active_lab == _PHYSICS_LABS[7]:
        st.subheader("⚙️ Bayesian Fuze Reliability (`FZ-01`..`FZ-04`), 3D Graze S-Curve (`FZ-06`) & Self-Destruct Redundancy (`FZ-07`)")
        st.latex(
            r"""
            p_{\text{dud}} \sim \text{Beta}(k + a_0,\; n - k + b_0), \qquad
            p_{\text{UCB}}(0\text{ failures}) = 1 - (1 - \gamma)^{1/n}, \qquad
            p_{\text{UXO}} = p_d \big[(1 - \beta)p_{\text{SD}} + \beta\big]
            """
        )
        fz_c1, fz_c2, fz_c3, fz_c4 = st.columns(4)
        with fz_c1:
            fz_n_tests = st.slider("Lot Acceptance Sample Size n", 10, 500, 120, 10, key="fz_n")
        with fz_c2:
            fz_k_fails = st.slider("Observed Lot Failures k", 0, 20, 2, 1, key="fz_k")
        with fz_c3:
            fz_sd_fail = st.slider("Self-Destruct Backup Fail P_SD", 0.0, 0.50, 0.05, 0.01, key="fz_sd")
        with fz_c4:
            fz_beta_cc = st.slider("Common-Cause Beta Factor β", 0.0, 0.30, 0.04, 0.01, key="fz_beta")

        a_post = fz_k_fails + 1.0
        b_post = max(fz_n_tests - fz_k_fails, 0) + 1.0
        p_dud_mean = a_post / (a_post + b_post)
        ucb_95 = float(fz_04_zero_failure_ucb(fz_n_tests, 0.95))
        p_uxo_red = float(fz_07_self_destruct_redundancy(p_dud_mean, fz_sd_fail, fz_beta_cc))

        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Posterior Mean Dud Rate (FZ-03)", f"{p_dud_mean*100:.2f} %", delta=f"Beta({a_post:.0f}, {b_post:.0f})", delta_color="off")
        fk2.metric("Zero-Failure 95% UCB (FZ-04)", f"{ucb_95*100:.2f} %", delta=f"For n={fz_n_tests} tests", delta_color="off")
        fk3.metric("Post-SD Residual UXO Rate (FZ-07)", f"{p_uxo_red*100:.3f} %", delta=f"Reduction: {(1.0 - p_uxo_red/max(p_dud_mean,1e-9))*100:.1f}%", delta_color="off")
        fk4.metric("Expected UXO per 48-Sub Carrier", f"{48.0 * p_uxo_red:.2f} UXO", delta=f"P(≥1 UXO): {(1.0 - (1.0 - p_uxo_red)**48)*100:.1f}%", delta_color="off")

        fz_left, fz_right = st.columns(2)
        with fz_left:
            p_grid = np.linspace(0.0001, 0.15, 250)
            pdf_post = sp_stats.beta.pdf(p_grid, a_post, b_post)
            fig_beta = go.Figure()
            fig_beta.add_trace(
                go.Scatter(
                    x=p_grid * 100.0,
                    y=pdf_post,
                    mode="lines",
                    fill="tozeroy",
                    name=f"Posterior Beta({a_post:.0f}, {b_post:.0f})",
                    line=dict(color="#2563EB", width=3),
                    fillcolor="rgba(37, 99, 235, 0.18)",
                )
            )
            fig_beta.add_vline(x=p_dud_mean * 100.0, line_dash="dash", line_color="#DC2626", annotation_text=f"Mean = {p_dud_mean*100:.2f}%")
            fig_beta.update_layout(
                title="Bayesian Conjugate Beta-Binomial Dud Rate Posterior Density (FZ-03)",
                xaxis_title="Fuze Dud Probability p_dud (%)",
                yaxis_title="Posterior Probability Density",
                height=390,
            )
            st.plotly_chart(fig_beta, width="stretch")

        with fz_right:
            aof_deg_grid = np.linspace(1.0, 45.0, 36)
            v_imp_grid = np.linspace(50.0, 500.0, 36)
            aa_deg, vv_mps = np.meshgrid(aof_deg_grid, v_imp_grid)
            gr_surf = fz_06_impact_graze_function(
                angle_of_fall_rad=np.radians(aa_deg),
                v_impact_mps=vv_mps,
                graze_threshold_deg=12.0,
                min_arm_vel_mps=140.0,
            )
            p_fn_surf = np.asarray(gr_surf["p_function"], dtype=np.float64)
            fig_graze = go.Figure(
                data=[
                    go.Surface(
                        x=aof_deg_grid,
                        y=v_imp_grid,
                        z=p_fn_surf,
                        colorscale="Viridis",
                        colorbar=dict(title="P(func)"),
                    )
                ]
            )
            fig_graze.update_layout(
                title="3D Impact Fuze Graze Angle × Striking Velocity Function Surface (FZ-06)",
                scene=dict(
                    xaxis_title="Impact Angle of Fall (deg)",
                    yaxis_title="Impact Velocity (m/s)",
                    zaxis_title="P(Function)",
                ),
                height=390,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_graze, width="stretch")

    # --- SUB-LAB 9: PRECISION GUIDANCE & PROPORTIONAL NAVIGATION (PN / APN) ---
    elif active_lab == _PHYSICS_LABS[8]:
        st.subheader("🚀 Precision Guidance & Autopilot Flight Control (`TPN` / `APN` / `Pure Pursuit`)")
        st.latex(
            r"""
            \mathbf{a}_{\text{cmd,TPN}} = N' V_c (\boldsymbol{\Omega} \times \hat{\mathbf{v}}_M), \qquad
            \mathbf{a}_{\text{cmd,APN}} = \mathbf{a}_{\text{cmd,TPN}} + \frac{1}{2} N' \mathbf{a}_{T\perp}, \qquad
            \tau_{\text{act}} \dot{\mathbf{a}}_{\text{act}} + \mathbf{a}_{\text{act}} = \text{sat}(\mathbf{a}_{\text{cmd}}, N_{\max} g)
            """
        )
        g_c1, g_c2, g_c3, g_c4 = st.columns(4)
        with g_c1:
            g_law = st.selectbox("Guidance Law", ["TPN", "APN", "PP"], index=0, help="TPN: True Proportional Navigation | APN: Augmented PN | PP: Pure Pursuit")
            g_np = st.slider("Navigation Ratio N'", 2.0, 6.0, 3.8, 0.1)
            g_lim = st.slider("Lateral G-Limit (g)", 5.0, 30.0, 18.0, 1.0)
        with g_c2:
            t_type = st.selectbox("Target Motion Model", ["stationary", "constant_velocity", "evasive_weave"], index=1)
            t_speed = st.slider("Target Speed (m/s)", 0.0, 35.0, 12.0, 1.0) if t_type != "stationary" else 0.0
            t_evade_g = st.slider("Evasion Maneuver (g)", 0.5, 6.0, 2.5, 0.5) if t_type == "evasive_weave" else 0.0
        with g_c3:
            g_v0 = st.slider("Missile Initial Speed V₀ (m/s)", 150.0, 450.0, 240.0, 10.0)
            g_thrust = st.slider("Rocket Motor Thrust (N)", 1000.0, 7000.0, 3800.0, 200.0)
            g_noise = st.slider("Seeker Noise σ_LOS (mrad/s)", 0.0, 2.0, 0.4, 0.1) * 1e-3
        with g_c4:
            g_range = st.slider("Target Standoff Range (m)", 1500.0, 6000.0, 3500.0, 250.0)
            g_elev = st.slider("Launch Elevation QE (deg)", 10.0, 40.0, 22.0, 1.0)
            g_lethal_r = st.slider("Lethal Radius R_lethal (m)", 1.0, 10.0, 3.5, 0.5)

        g_res = simulate_guided_trajectory(
            v0_mps=float(g_v0),
            elev_deg=float(g_elev),
            target_pos_m=(float(g_range), 50.0, 0.0),
            target_vel_mps=(0.0, float(t_speed), 0.0),
            target_accel_type=t_type,  # type: ignore[arg-type]
            target_evasion_g=float(t_evade_g),
            guidance_law=g_law,  # type: ignore[arg-type]
            N_prime=float(g_np),
            g_limit=float(g_lim),
            motor_thrust_N=float(g_thrust),
            seeker_noise_std_rad=float(g_noise),
            lethal_radius_m=float(g_lethal_r),
        )

        gm1, gm2, gm3, gm4, gm5 = st.columns(5)
        is_hit = g_res["hit"]
        gm1.metric("Engagement Outcome", g_res["status"], delta="DIRECT HIT" if is_hit else "MISS", delta_color="normal" if is_hit else "inverse")
        gm2.metric("Miss Distance", f"{g_res['miss_distance_m']:.2f} m", delta=f"R_lethal: {g_lethal_r:.1f} m", delta_color="off")
        gm3.metric("Time of Flight", f"{g_res['time_of_flight_s']:.2f} s", delta_color="off")
        gm4.metric("Closing Velocity V_c", f"{g_res['terminal_closing_velocity_mps']:.1f} m/s", delta_color="off")
        gm5.metric("Peak Commanded G", f"{g_res['peak_g_load']:.1f} g", delta=f"Limit: {g_lim:.0f} g", delta_color="off")

        g_p1, g_p2 = st.columns([1.35, 1.0])
        with g_p1:
            fig_g_3d = create_3d_guided_intercept_figure(g_res)
            fig_g_3d.update_layout(height=480)
            st.plotly_chart(fig_g_3d, width="stretch")
        with g_p2:
            gh = g_res["history"]
            fig_g_dyn = make_subplots(specs=[[{"secondary_y": True}]])
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["act_g"], mode="lines", name="Achieved Lateral G (g)", line=dict(color="#2563EB", width=3)),
                secondary_y=False,
            )
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["closing_vel_mps"], mode="lines", name="Closing Velocity (m/s)", line=dict(color="#10B981", width=2, dash="dash")),
                secondary_y=True,
            )
            fig_g_dyn.add_trace(
                go.Scatter(x=gh["t_s"], y=gh["range_m"], mode="lines", name="Missile-Target Range (m)", line=dict(color="#DC2626", width=2, dash="dot")),
                secondary_y=True,
            )
            fig_g_dyn.update_xaxes(title_text="Flight Time t (s)")
            fig_g_dyn.update_yaxes(title_text="Lateral Acceleration (g)", secondary_y=False)
            fig_g_dyn.update_yaxes(title_text="Range / Velocity", secondary_y=True)
            fig_g_dyn.update_layout(title="Guidance Dynamics & Control Surface Effort vs Time", height=480)
            st.plotly_chart(fig_g_dyn, width="stretch")

    # --- SUB-LAB 10: CLUSTER MUNITION DISPENSER KINEMATICS & UXO HAZARD ---
    elif active_lab == _PHYSICS_LABS[9]:
        st.subheader("📦 Cluster Munition Dispenser Kinematics, Ground Footprint & UXO Contamination (`CCM` / `IATG 10.10`)")
        st.latex(
            r"""
            V_{\text{term}} = \sqrt{\frac{2 m_s g}{\rho A_s C_{D,\text{chute}}}}, \qquad
            p_{\text{UXO}} = (1 - \beta) p_{\text{dud}} p_{\text{fail,SD}} + \beta \min(p_{\text{dud}}, p_{\text{fail,SD}}), \qquad
            R_{\text{cordon}} = \max(\Delta R) + 200\,\text{m}
            """
        )
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            d_n_sub = st.selectbox("Submunition Count N_sub", [24, 48, 64, 88], index=2)
            d_burst_alt = st.slider("Burst Altitude H_burst (m)", 150.0, 800.0, 380.0, 25.0)
            d_mass = st.slider("Submunition Mass m_s (kg)", 0.15, 0.60, 0.23, 0.02)
        with d_c2:
            d_app_v = st.slider("Carrier Approach Speed (m/s)", 150.0, 350.0, 240.0, 10.0)
            d_spin = st.slider("Carrier Roll Rate (rad/s)", 0.0, 120.0, 65.0, 5.0)
            d_ej_ax = st.slider("Axial Expulsion (m/s)", 10.0, 50.0, 24.0, 2.0)
        with d_c3:
            d_chute_cd = st.slider("Parachute Drag C_D", 0.8, 1.8, 1.25, 0.05)
            d_w_x = st.slider("Downrange Wind (m/s)", -5.0, 10.0, 2.0, 0.5)
            d_w_y = st.slider("Crossrange Wind (m/s)", -10.0, 10.0, 3.5, 0.5)
        with d_c4:
            d_p_dud = st.slider("Primary Fuze Dud Rate (%)", 0.5, 12.0, 3.5, 0.5) / 100.0
            d_p_sd = st.slider("Self-Destruct Fail Rate (%)", 1.0, 25.0, 8.0, 1.0) / 100.0
            d_beta = st.slider("Common-Cause Beta Factor", 0.0, 0.15, 0.04, 0.01)

        d_res = simulate_cluster_dispenser_footprint(
            n_submunitions=int(d_n_sub),
            submunition_mass_kg=float(d_mass),
            burst_pos_m=(12000.0, 0.0, float(d_burst_alt)),
            burst_vel_mps=(float(d_app_v), 0.0, -85.0),
            carrier_spin_rad_s=float(d_spin),
            ejection_axial_mps=float(d_ej_ax),
            chute_cd=float(d_chute_cd),
            wind_downrange_mps=float(d_w_x),
            wind_crossrange_mps=float(d_w_y),
            p_dud_primary=float(d_p_dud),
            p_fail_self_destruct=float(d_p_sd),
            common_cause_beta=float(d_beta),
        )

        dm1, dm2, dm3, dm4, dm5 = st.columns(5)
        dm1.metric("Detonated Functional", f"{d_res['n_detonated']} / {d_res['n_submunitions']}", delta=f"{100*d_res['n_detonated']/d_res['n_submunitions']:.1f}%", delta_color="normal")
        dm2.metric("Self-Destruct Neutralized", f"{d_res['n_self_destruct_neutralized']}", delta="Inert Dud", delta_color="off")
        dm3.metric("Live UXO Hazardous Duds", f"{d_res['n_uxo']}", delta=f"{d_res['actual_uxo_rate']*100:.2f}% Actual UXO", delta_color="inverse" if d_res['n_uxo'] > 0 else "normal")
        dm4.metric("Footprint Area", f"{d_res['footprint_area_hectares']:.2f} ha", delta=f"{d_res['footprint_semi_major_m']:.0f}m × {d_res['footprint_semi_minor_m']:.0f}m", delta_color="off")
        dm5.metric("IATG 10.10 Cordon Buffer", f"R = {d_res['iatg_cordon_radius_m']:.0f} m", delta=f"Cordon: {d_res['iatg_cordon_area_ha']:.1f} ha", delta_color="off")

        if d_res['n_uxo'] > 0:
            st.error(f"🚨 **UNODA IATG 10.10 / CCW Protocol V Hazardous Contamination Detected!** {d_res['n_uxo']} active unexploded ordnance (UXO) items on ground. UXO density: **{d_res['uxo_density_per_ha']:.2f} UXO/hectare** ({d_res['uxo_density_per_km2']:.0f} UXO/km²). Mandatory {d_res['iatg_cordon_radius_m']:.0f}m exclusion perimeter required.")
        else:
            st.success("✅ **Zero Live UXO Items Recorded!** All non-functioning submunitions were successfully neutralized by secondary self-destruct mechanisms.")

        fig_d_3d = create_3d_submunition_dispersion_figure(d_res)
        fig_d_3d.update_layout(height=480)
        st.plotly_chart(fig_d_3d, width="stretch")

    # --- SUB-LAB 11: FULL 6-DOF RIGID-BODY BALLISTICS & EPICYCLIC YAW MOTION ---
    elif active_lab == _PHYSICS_LABS[10]:
        st.subheader("🌀 Full 6-Degree-of-Freedom (6-DoF) Rigid-Body Ballistics & Epicyclic Yaw Motion (`STANAG 4355`)")
        st.latex(
            r"""
            \dot{\mathbf{q}} = \frac{1}{2} \begin{bmatrix} 0 & -p & -q & -r \\ p & 0 & r & -q \\ q & -r & 0 & p \\ r & q & -p & 0 \end{bmatrix} \mathbf{q}, \quad
            \mathbf{I} \dot{\boldsymbol{\omega}}_B + \boldsymbol{\omega}_B \times (\mathbf{I} \boldsymbol{\omega}_B) = \mathbf{M}_B, \quad
            S_g = \frac{I_x^2 p^2}{2 \rho S d I_t C_{M\alpha} V^2} > 1.0
            """
        )
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        with s_c1:
            s_v0 = st.slider("Muzzle Velocity V₀ (m/s)", 300.0, 950.0, 827.0, 15.0, key="s6_v0")
            s_qe = st.slider("Quadrant Elevation QE (deg)", 10.0, 65.0, 35.0, 1.0, key="s6_qe")
            s_twist = st.slider("Rifling Twist (cal/turn)", 15.0, 30.0, 20.0, 1.0, key="s6_twist")
        with s_c2:
            s_cm_a = st.slider("Overturning Moment C_Mα", 2.0, 7.0, 4.2, 0.2, help="Overturning aerodynamic moment slope (forward center of pressure)")
            s_cm_q = st.slider("Pitch Damping C_Mq", -30.0, -5.0, -14.0, 1.0, help="Pitch and yaw damping aerodynamic moment derivative")
            s_clp = st.slider("Spin Damping C_lp", -0.05, -0.005, -0.025, 0.005, help="Axial roll damping derivative")
        with s_c3:
            s_tip_yaw = st.slider("Tip-Off Initial Yaw Rate (rad/s)", 0.0, 0.30, 0.05, 0.01, help="Muzzle exit tip-off angular rate")
            s_tip_pitch = st.slider("Tip-Off Initial Pitch Rate (rad/s)", 0.0, 0.30, 0.02, 0.01)
            s_c_mag = st.slider("Magnus Cross-Force C_Ypα", -1.5, 0.0, -0.65, 0.05)
        with s_c4:
            s_dt = st.slider("RK4 Integration Step dt (ms)", 1.0, 5.0, 2.0, 0.5, key="s6_dt") * 1e-3
            s_lat = st.slider("Firing Latitude (deg)", -60.0, 60.0, 45.0, 5.0, key="s6_lat")

        res_6dof = simulate_6dof_trajectory(
            v0_mps=float(s_v0),
            elev_deg=float(s_qe),
            twist_cal_per_turn=float(s_twist),
            cm_alpha=float(s_cm_a),
            cm_q=float(s_cm_q),
            c_lp=float(s_clp),
            c_yp_alpha=float(s_c_mag),
            initial_yaw_rate_rad_s=float(s_tip_yaw),
            initial_pitch_rate_rad_s=float(s_tip_pitch),
            latitude_deg=float(s_lat),
            dt_s=float(s_dt),
        )

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("6-DoF Final Range", f"{res_6dof['final_range_m']:.0f} m", delta=f"TOF: {res_6dof['time_of_flight_s']:.1f} s", delta_color="off")
        sm2.metric("Gyroscopic Lateral Drift", f"{res_6dof['final_deflection_m']:+.1f} m (Right)", delta="Magnus + Precession", delta_color="off")
        sm3.metric("Muzzle Gyroscopic S_g", f"{res_6dof['muzzle_sg']:.2f}", delta="Stable Sg > 1.0" if res_6dof['muzzle_sg'] >= 1.0 else "UNSTABLE", delta_color="normal" if res_6dof['muzzle_sg'] >= 1.0 else "inverse")
        sm4.metric("Terminal Spin RPM", f"{res_6dof['terminal_spin_rpm']:.0f} RPM", delta=f"Initial: {res_6dof['initial_spin_rpm']:.0f} RPM", delta_color="off")
        sm5.metric("Peak Total Yaw |α_t|", f"{res_6dof['max_yaw_of_repose_deg']:.2f}°", delta=f"Min Sg: {res_6dof['min_sg']:.2f}", delta_color="off")

        s_p1, s_p2 = st.columns([1.35, 1.0])
        with s_p1:
            fig_6d_3d = create_3d_6dof_attitude_figure(res_6dof)
            fig_6d_3d.update_layout(height=480)
            st.plotly_chart(fig_6d_3d, width="stretch")
        with s_p2:
            s_hist = res_6dof["history"]
            # 2D Epicyclic yaw arm trajectory: alpha (pitch AoA) vs beta (yaw AoA)
            fig_epi = go.Figure()
            fig_epi.add_trace(
                go.Scatter(
                    x=s_hist["beta_deg"],
                    y=s_hist["alpha_deg"],
                    mode="lines",
                    line=dict(color="#2563EB", width=2.5),
                    name="Epicyclic Yaw Arm (β vs α)",
                )
            )
            fig_epi.add_trace(
                go.Scatter(
                    x=[s_hist["beta_deg"][0]],
                    y=[s_hist["alpha_deg"][0]],
                    mode="markers+text",
                    marker=dict(size=10, color="#10B981", symbol="circle"),
                    text=["Muzzle Exit"],
                    textposition="top right",
                    name="Muzzle Exit",
                )
            )
            fig_epi.update_layout(
                title="Epicyclic Yaw Trajectory: Nutation Loops & Precession Drift",
                xaxis_title="Angle of Yaw β (deg)",
                yaxis_title="Angle of Pitch α (deg)",
                height=480,
            )
            st.plotly_chart(fig_epi, width="stretch")

    # --- SUB-LAB 12: ARTILLERY BATTERY SHEAFS, MRSI / TOT & SALVO OPTIMIZATION ---
    elif active_lab == _PHYSICS_LABS[11]:
        st.subheader("🎯 Tactical Artillery Battery Sheafs, MRSI / TOT & Salvo Optimization (FM 6-40 / STANAG 4119)")
        st.markdown(
            "Rigorous multi-gun artillery battery firing dynamics grounded in **US Army FM 6-40 / FM 3-09** and **NATO STANAG 4119**: "
            "battery spatial formations (Linear, Wedge, Staggered, Lazy-W), tactical sheaf distributions (Converged point target, "
            "Open parallel, Linear column/trench, Area circular), bisection aerodynamic elevation solver, "
            "and **Multiple Rounds Simultaneous Impact (MRSI) / Time-on-Target (TOT)** trajectory interleaving."
        )
        st.latex(
            r"""
            X_i = X_{\text{aim}} + \Delta X_i, \quad
            Y_i = Y_{\text{aim}} + \Delta Y_i, \quad
            \Delta t_{\text{fire}}^{(k)} = \text{TOF}_{\max} - \text{TOF}^{(k)}, \quad
            P_k(\text{salvo}) = 1 - \prod_{j=1}^{N_{\text{rounds}}} \left[1 - P_{k,j}\right]
            """
        )

        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            st.markdown("**Battery Weapon Formation**")
            bat_guns = st.slider("Number of Weapons in Battery", 2, 8, 6, 1)
            bat_form = st.selectbox("Battery Formation", ["staggered", "linear", "wedge", "lazy_w"], index=0)
            bat_spacing = st.slider("Gun Spacing (m)", 25.0, 150.0, 75.0, 5.0)

        with b_c2:
            st.markdown("**Tactical Target Sheaf**")
            sheaf_kind = st.selectbox("Sheaf Distribution Type", ["converged", "open", "linear", "area"], index=0)
            target_range_bat = st.slider("Target Distance (m)", 4000.0, 22000.0, 14000.0, 500.0)
            lin_len = st.slider("Linear Sheaf Length (m)", 50.0, 600.0, 300.0, 25.0) if sheaf_kind == "linear" else 300.0
            lin_az = st.slider("Linear Sheaf Azimuth (°)", 0.0, 360.0, 90.0, 15.0) if sheaf_kind == "linear" else 90.0

        with b_c3:
            st.markdown("**Salvo Perturbations & Lethality**")
            rds_per_gun = st.slider("Rounds Fired per Gun", 1, 4, 2, 1)
            sig_v0 = st.slider("Muzzle Velocity Error σ_V0 (m/s)", 0.2, 5.0, 1.5, 0.1)
            sig_pointing = st.slider("Weapon Pointing Error σ_aim (mils)", 0.2, 3.0, 0.8, 0.1)
            lethal_r = st.slider("Effective Blast/Frag Lethal Radius (m)", 10.0, 60.0, 30.0, 5.0)

        # MRSI / TOT Planner Section
        with st.expander("⏱️ Multiple Rounds Simultaneous Impact (MRSI) Trajectory Solver", expanded=True):
            st.markdown(
                r"MRSI calculates the fire release sequence across high-angle, reduced-propellant charge, "
                r"and low-angle trajectory arcs so that up to 3 rounds arrive simultaneously at the target ($\Delta t < 0.2\text{ s}$)."
            )
            mrsi_res = solve_mrsi_trajectory_plan(target_range_m=target_range_bat, max_rounds=3)
            if mrsi_res["mrsi_feasible"]:
                m_cols = st.columns(len(mrsi_res["plan"]))
                for idx, rd in enumerate(mrsi_res["plan"]):
                    with m_cols[idx]:
                        st.metric(
                            f"Round {rd['round_num']} ({rd['branch']})",
                            f"Fire @ T+{rd['fire_time_s']}s",
                            f"QE: {rd['qe_deg']}° | TOF: {rd['tof_s']}s",
                        )
                mrsi_df = pd.DataFrame(mrsi_res["plan"])
                st.dataframe(mrsi_df[["round_num", "branch", "qe_deg", "tof_s", "fire_time_s", "impact_time_s"]], width="stretch")
            else:
                st.warning("MRSI solution not feasible at current range (exceeds high/low angle split limits).")

        # Battery Salvo Simulation
        with st.spinner("Integrating Multi-Gun Battery Trajectories & Bivariate Sheaf Dispersion..."):
            salvo_sim = simulate_battery_salvo(
                n_guns=bat_guns,
                formation=bat_form,
                gun_spacing_m=bat_spacing,
                target_pos_m=(float(target_range_bat), 0.0),
                sheaf_type=sheaf_kind,
                linear_length_m=lin_len,
                linear_azimuth_deg=lin_az,
                rounds_per_gun=rds_per_gun,
                sigma_mv_rr_mps=sig_v0,
                sigma_aim_qe_mils=sig_pointing,
                sigma_aim_az_mils=sig_pointing,
                lethal_radius_m=lethal_r,
                seed=42,
            )

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Salvo Rounds", f"{salvo_sim['total_rounds']} rounds")
        sm2.metric("Sheaf CEP50", f"{salvo_sim['cep50_m']:.1f} m")
        sm3.metric("Sheaf CEP95", f"{salvo_sim['cep95_m']:.1f} m")
        p_neut = salvo_sim.get("target_neutralization_prob", salvo_sim.get("fractional_neutralization_pct", 0.0) / 100.0)
        sm4.metric("Target Neutralization P_k", f"{p_neut * 100.0:.1f} %")

        fig_bat_3d = create_3d_battery_salvo_figure(salvo_sim)
        st.plotly_chart(fig_bat_3d, width="stretch")

        with st.expander("📋 Detailed Battery Firing & Impact Telemetry Log", expanded=False):
            rounds_df = pd.DataFrame(salvo_sim["rounds"])
            preferred_cols = [
                "round_id", "gun_id", "round_in_gun", "impact_x_m", "impact_y_m",
                "miss_dist_m", "dist_from_target_m", "v_impact_mps", "tof_s"
            ]
            disp_cols = [c for c in preferred_cols if c in rounds_df.columns]
            st.dataframe(rounds_df[disp_cols] if disp_cols else rounds_df, width="stretch")

    # --- SUB-LAB 13: MULTI-LAYER ARMOR DEFEAT & TANDEM ERA ---
    elif active_lab == _PHYSICS_LABS[12]:
        st.subheader("🛡️ Modern Multi-Layer Armor Defeat, Tandem Shaped Charges & Explosive Reactive Armor (ERA)")
        st.markdown(
            "Advanced hydrodynamic and solid-mechanics armor defeat laboratory extending **`TP-02` (Birkhoff hydrodynamic shaped-charge jet penetration)**: "
            "asymmetric explosive flyer plate acceleration via the Gurney equation, transverse shearing of shaped charge jets by moving plates, "
            "tandem precursor defeat of ERA, hydrodynamic dwell and erosion in ceramic tiles (SiC, B₄C, Al₂O₃), "
            "and residual perforation of Rolled Homogeneous Armor (RHA) with behind-armor spall cone lethality."
        )
        st.latex(
            r"""
            V_{\text{flyer}} = \sqrt{2E}\left(1 + \frac{2M}{C}\right)^{-1/2}, \quad
            L_{\text{eff}} = L_0 \left(1 - \frac{V_{\text{flyer}}\sin\theta_{\text{obliq}}}{v_{\text{jet}}}\right), \quad
            P_{\text{ceramic}} = \lambda_{\text{dwell}} \frac{L_{\text{jet}}}{1 + \frac{\rho_c H_c}{\rho_j v_j^2}}, \quad
            P_{\text{RHA}} = L_{\text{res}} \sqrt{\frac{\rho_j}{\rho_{\text{RHA}}}}
            """
        )

        arm1, arm2, arm3 = st.columns(3)
        with arm1:
            st.markdown("**Warhead Configuration**")
            is_tandem = st.toggle("Tandem Warhead (Precursor + Main)", value=True)
            prec_d = st.slider("Precursor Diameter (mm)", 25.0, 75.0, 40.0, 5.0) if is_tandem else 0.0
            prec_pen = st.slider("Precursor Jet Penetration (mm RHA)", 50.0, 300.0, 150.0, 10.0) if is_tandem else 0.0
            main_d = st.slider("Main Charge Diameter (mm)", 80.0, 180.0, 125.0, 5.0)
            main_pen = st.slider("Main Nominal Penetration (mm RHA)", 400.0, 1400.0, 900.0, 25.0)

        with arm2:
            st.markdown("**Explosive Reactive Armor (ERA)**")
            era_active = st.checkbox("Equipped with ERA Cassette", value=True)
            era_type = st.selectbox("ERA Generation / Type", ["Heavy Relikt / FY-4", "Medium Kontakt-5", "Light Blazer / Kontakt-1"], index=0) if era_active else "None"
            era_thick = st.slider("ERA Cassette Thickness (mm)", 15.0, 60.0, 35.0, 5.0) if era_active else 0.0
            era_flyer_m = st.slider("ERA Flyer Plate Mass (kg/m²)", 10.0, 70.0, 38.0, 2.0) if era_active else 0.0
            era_exp_c = st.slider("ERA High-Explosive Charge (kg/m²)", 5.0, 30.0, 14.0, 1.0) if era_active else 0.0

        with arm3:
            st.markdown("**Target Armor Structure**")
            obliq_deg = st.slider("NATO Armor Obliquity (° from normal)", 0.0, 75.0, 60.0, 5.0)
            cer_mat = st.selectbox("Ceramic Composite Tile Material", ["Silicon Carbide (SiC)", "Boron Carbide (B4C)", "Alumina (Al2O3)"], index=0)
            cer_thick = st.slider("Ceramic Layer Thickness (mm)", 0.0, 150.0, 50.0, 5.0)
            rha_thick = st.slider("Backing RHA Thickness (mm)", 50.0, 800.0, 350.0, 25.0)

        era_cat = "heavy_era" if ("Heavy" in era_type or "Medium" in era_type) else "light_era"
        cer_cat = "Silicon Carbide (SiC)" if "SiC" in cer_mat else "Alumina (Al2O3)"
        armor_res = simulate_tandem_shaped_charge_vs_era(
            warhead_type="tandem" if is_tandem else "monolithic",
            has_era=era_active,
            era_type=era_cat,
            main_calibre_mm=main_d,
            precursor_calibre_mm=prec_d if is_tandem else 0.0,
            inter_warhead_delay_ms=0.35,
            obliquity_deg=obliq_deg,
            rha_base_thickness_mm=rha_thick,
            has_ceramic_tile=cer_thick > 0.0,
            ceramic_thickness_mm=cer_thick,
            ceramic_type=cer_cat,
        )

        ap1, ap2, ap3, ap4 = st.columns(4)
        perf_status = armor_res.get("perforation_status", "UNKNOWN")
        status_delta = "Defeated Armor" if armor_res.get("perforated", False) else "Armor Survived"
        ap1.metric("Perforation Status", perf_status, delta=status_delta)
        ap2.metric("Total LOS Armor Depth", f"{armor_res.get('target_los_thickness_mm', 0.0):.1f} mm")
        ap3.metric("Residual Penetration Margin", f"{armor_res.get('residual_penetration_reserve_mm', 0.0):.1f} mm")
        spall = armor_res.get("behind_armor_spall", {})
        ap4.metric("Behind-Armor Spall Fragments", f"{spall.get('spall_fragment_count', 0)} frags")

        fig_arm_3d = create_3d_armor_defeat_figure(armor_res)
        st.plotly_chart(fig_arm_3d, width="stretch")

        with st.expander("🔬 Physical Layer Degradation & Energy Dissipation Log", expanded=True):
            lay_c1, lay_c2 = st.columns(2)
            with lay_c1:
                st.markdown("**Warhead Interaction with Defensive Layers**")
                st.write(f"- **Precursor Defeat of ERA**: {armor_res.get('precursor_success', False)}")
                st.write(f"- **ERA Flyer Plate Velocity**: {armor_res.get('era_flyer_speed_mps', 0.0):.1f} m/s")
                st.write(f"- **Transverse Jet Disruption**: {armor_res.get('jet_disruption_pct', 0.0):.1f} %")
                st.write(f"- **Ceramic Dwell Erosion Loss**: {armor_res.get('ceramic_rha_equivalent_mm', 0.0):.1f} mm RHA")
                st.write(f"- **Post-ERA Jet Penetration**: {armor_res.get('post_era_jet_penetration_mm', 0.0):.1f} mm RHA")

            with lay_c2:
                st.markdown("**Behind-Armor Spall Lethality**")
                st.write(f"- **Spall Half-Angle**: {spall.get('spall_cone_half_angle_deg', 0.0):.1f}°")
                st.write(f"- **Mean Fragment Velocity**: {spall.get('mean_spall_velocity_mps', 0.0):.1f} m/s")
                p_crew = spall.get('crew_incapacitation_prob', 0.0)
                st.write(f"- **Crew Incapacitation Probability**: {p_crew * 100.0:.1f} %")

    # --- SUB-LAB 14: COUNTER-BATTERY RADAR & POO/POI RECONSTRUCTION ---
    elif active_lab == _PHYSICS_LABS[13]:
        st.subheader("🛰️ Counter-Battery Radar & Trajectory Reconstruction (NATO STANAG 4114 / AN/TPQ-53)")
        st.markdown(
            "Defensive artillery radar tracking and weapon location grounded in **NATO STANAG 4114**: "
            "evaluates projectile Radar Cross Section (RCS) via physical optics, tracks noisy radar echo detections "
            "across the radar search fence, and solves inverse exterior ballistics back-propagating states backwards in time "
            "to compute the hostile weapon **Point of Origin (POO)** and error ellipse, and forward to the **Point of Impact (POI)** "
            "for early-warning impact countdown alert."
        )
        st.latex(
            r"""
            \sigma_{\text{RCS}} = \sigma_{\text{nose}}\cos^4\theta + \sigma_{\text{broadside}}\sin^8\theta, \quad
            \mathbf{r}(t - \Delta t) = \mathbf{r}(t) - \mathbf{v}(t)\Delta t, \quad
            \mathbf{v}(t - \Delta t) = \mathbf{v}(t) - \mathbf{a}_{\text{forward}}(t)\Delta t
            """
        )

        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.markdown("**Weapon & Firing Solution**")
            r_v0 = st.slider("Muzzle Velocity V0 (m/s)", 450.0, 950.0, 680.0, 10.0, key="rad_v0")
            r_qe = st.slider("Quadrant Elevation QE (°)", 15.0, 65.0, 32.0, 1.0, key="rad_qe")
            r_cal = st.selectbox("Projectile Calibre", ["155mm NATO Artillery", "122mm D-30 Howitzer", "120mm Heavy Mortar", "227mm MLRS Rocket"], index=0, key="rad_cal")
            cal_diam = 0.155 if "155mm" in r_cal else (0.122 if "122mm" in r_cal else (0.120 if "120mm" in r_cal else 0.227))
            cal_len = 0.85 if "155mm" in r_cal else (0.60 if "122mm" in r_cal else (0.70 if "120mm" in r_cal else 3.96))
            cal_mass = 43.5 if "155mm" in r_cal else (21.7 if "122mm" in r_cal else (16.0 if "120mm" in r_cal else 307.0))

        with r_c2:
            st.markdown("**Radar Site & Coverage Emplacement**")
            radar_x_pos = st.slider("Radar East Position X (m)", 1000.0, 10000.0, 4000.0, 250.0, key="rad_rx")
            radar_y_pos = st.slider("Radar North Position Y (m)", -8000.0, 8000.0, -2500.0, 250.0, key="rad_ry")
            radar_max_r = st.slider("Radar Max Instrumented Range (m)", 15000.0, 45000.0, 32000.0, 1000.0, key="rad_rmax")
            p_det = st.slider("Single-Pulse Detection Probability P_d", 0.70, 0.99, 0.94, 0.01, key="rad_pd")

        with r_c3:
            st.markdown("**Measurement Noise & Target RCS**")
            sig_r = st.slider("Radar Slant Range Noise σ_R (m)", 2.0, 35.0, 12.0, 1.0, key="rad_sigr")
            sig_ang = st.slider("Radar Angle Noise σ_θ (mils)", 0.2, 3.0, 0.8, 0.1, key="rad_sigang")
            rcs_est = calculate_projectile_rcs(cal_diam, cal_len, aspect_angle_deg=r_qe)
            st.metric("Estimated Shell RCS (σ)", f"{rcs_est:.4f} m²", delta=f"{10.0*math.log10(max(rcs_est,1e-6)):.1f} dBsm")

        with st.spinner("Generating Aerodynamic Trajectory & Radar Track Measurements..."):
            from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch
            atm_env_r = AtmosphereEnvironment()
            cfg_r = BatchTrajectoryConfig(calibre_m=cal_diam, fidelity="L1", dt=0.08)
            rad_q_val = math.radians(r_qe)
            s0_r = np.array([[0.0, 0.0, 0.0, r_v0 * math.cos(rad_q_val), 0.0, r_v0 * math.sin(rad_q_val)]], dtype=np.float64)
            res_true_traj = integrate_trajectories_batch(s0_r, np.array([cal_mass]), np.array([1.0]), atm_env_r, cfg_r, record_single_path=True)
            sp_r = res_true_traj["sample_path"]

            tx_list = list(sp_r["range_m"])
            ty_list = [0.0] * len(sp_r["t"])
            tz_list = list(sp_r["alt_m"])
            tt_list = list(sp_r["t"])

            radar_track_sim = simulate_radar_tracking(
                trajectory_x_m=tx_list,
                trajectory_y_m=ty_list,
                trajectory_z_m=tz_list,
                time_s=tt_list,
                radar_pos_m=(radar_x_pos, radar_y_pos, 50.0),
                radar_range_max_m=radar_max_r,
                p_detection_nominal=p_det,
                sigma_range_m=sig_r,
                sigma_angle_mils=sig_ang,
                seed=42,
            )

            recon_sol = reconstruct_point_of_origin_and_impact(
                track_points=radar_track_sim["track_points"],
                gun_elev_m=0.0,
                target_elev_m=0.0,
                calibre_m=cal_diam,
                mass_kg=cal_mass,
                v0_nominal_mps=r_v0,
            )

        rk1, rk2, rk3, rk4 = st.columns(4)
        poo_err = math.hypot(recon_sol.get("poo_x_m", 0.0), recon_sol.get("poo_y_m", 0.0))
        rk1.metric("POO Localization Error", f"{poo_err:.1f} m", delta="High Precision" if poo_err < 100.0 else "Coarse")
        rk2.metric("Estimated Launch V0", f"{recon_sol.get('estimated_v0_mps', 0.0):.1f} m/s", delta=f"{recon_sol.get('estimated_v0_mps', 0.0) - r_v0:+.1f} m/s error")
        rk3.metric("Early-Warning Countdown", f"{recon_sol.get('early_warning_countdown_s', 0.0):.1f} s", delta="Bunker Impact Alert")
        rk4.metric("Radar Detections Used", f"{len(radar_track_sim['track_points'])} echoes", delta=f"{radar_track_sim.get('track_duration_s', 0.0)}s arc")

        fig_rad_3d = create_3d_counter_battery_radar_figure(radar_track_sim, recon_sol)
        st.plotly_chart(fig_rad_3d, width="stretch")

        with st.expander("📋 Radar Track Telemetry & Covariance Confidence Log", expanded=False):
            if radar_track_sim["track_points"]:
                rad_df = pd.DataFrame(radar_track_sim["track_points"])
                st.dataframe(rad_df[["t_s", "obs_x", "obs_y", "obs_z", "slant_range_m", "elevation_deg", "azimuth_deg"]], width="stretch")

    # --- SUB-LAB 15: DIRECT-FIRE TANK GUNNERY & APFSDS SABOT DYNAMICS ---
    elif active_lab == _PHYSICS_LABS[14]:
        st.subheader("🎯 Direct-Fire Tank Gunnery, APFSDS Sabot Discard & Moving Target FCS")
        st.markdown(
            "High-velocity kinetic armor defeat ($1,500\text{–}1,800\text{ m/s}$) and moving target gunnery: "
            "simulates the dynamic 3-piece sabot petal separation upon muzzle exit, computes the friendly troop safety hazard zone, "
            "evaluates tank Fire Control System (FCS) lead angles, trunnion cant coupling, and crosswind jump, "
            "and models tungsten/DU long-rod terminal penetration using the **Lanz-Odermatt (TP-01)** continuum model."
        )
        st.latex(
            r"""
            \theta_{\text{lead}} = \arctan\left(\frac{v_{\text{cross}}\cdot\text{TOF}}{R}\right), \quad
            \Delta y_{\text{cant}} = \frac{1}{2} g\,\text{TOF}^2 \sin\phi_{\text{cant}}, \quad
            P_{\text{LO}} = L_w \frac{A}{\tanh(B_0 + B_1 L_w / D)} \sqrt{\frac{\rho_p}{\rho_t}} e^{-S^2 / V^2}
            """
        )

        df1, df2, df3 = st.columns(3)
        with df1:
            st.markdown("**Tank Weapon & Kinetic Long-Rod**")
            mbt_v0 = st.slider("Muzzle Velocity V0 (m/s)", 1500.0, 1850.0, 1680.0, 20.0, key="df_v0")
            rod_len = st.slider("Penetrator Rod Length (cm)", 55.0, 95.0, 78.0, 1.0, key="df_len") / 100.0
            rod_diam = st.slider("Penetrator Rod Diameter (mm)", 20.0, 34.0, 26.0, 1.0, key="df_diam") / 1000.0
            rod_m = st.slider("Penetrator Rod Mass (kg)", 3.5, 7.5, 4.85, 0.1, key="df_mass")

        with df2:
            st.markdown("**Moving Armored Target**")
            tgt_dist = st.slider("Target Distance (m)", 800.0, 4000.0, 2200.0, 100.0, key="df_dist")
            tgt_spd = st.slider("Target Speed (km/h)", 0.0, 60.0, 32.0, 2.0, key="df_spd")
            tgt_cross_ang = st.slider("Target Crossing Angle (° from bore line)", 0.0, 180.0, 80.0, 5.0, key="df_ang")
            fcs_on = st.toggle("Tank FCS Auto-Lead Solution Active", value=True, key="df_fcs")

        with df3:
            st.markdown("**Target Armor & Sabot Petals**")
            rha_mbt = st.slider("Base RHA Armor (mm)", 300.0, 950.0, 650.0, 25.0, key="df_rha")
            rha_obliq = st.slider("Armor Obliquity (° from normal)", 0.0, 70.0, 60.0, 5.0, key="df_obliq")
            n_petals_mbt = st.selectbox("Sabot Petal Count", [3, 4], index=0, key="df_pet")
            petal_div = st.slider("Petal Aerodynamic Divergence (°)", 5.0, 14.0, 8.5, 0.5, key="df_pdiv")

        with st.spinner("Simulating Sabot Discard & Kinetic Engagement..."):
            sabot_sim = simulate_apfsds_sabot_discard(
                v0_mps=mbt_v0,
                rod_calibre_m=rod_diam,
                rod_length_m=rod_len,
                rod_mass_kg=rod_m,
                n_petals=n_petals_mbt,
                petal_divergence_angle_deg=petal_div,
            )
            direct_sim = simulate_direct_fire_engagement(
                target_range_m=tgt_dist,
                target_speed_kmh=tgt_spd,
                target_crossing_angle_deg=tgt_cross_ang,
                v0_mps=mbt_v0,
                rod_calibre_m=rod_diam,
                rod_length_m=rod_len,
                rod_mass_kg=rod_m,
                armor_thickness_rha_mm=rha_mbt,
                armor_obliquity_deg=rha_obliq,
                fcs_lead_applied=fcs_on,
                seed=42,
            )

        dk1, dk2, dk3, dk4 = st.columns(4)
        dk1.metric("Engagement Outcome", direct_sim["engagement_status"], delta="Hit" if direct_sim["is_hit"] else "Miss")
        dk2.metric("Lanz-Odermatt Penetration", f"{direct_sim['lo_penetration_normal_mm']:.0f} mm RHA", delta=f"{direct_sim['v_impact_mps']:.0f} m/s impact")
        fcs_info = direct_sim["fcs_solution"]
        dk3.metric("FCS Aim Lead", f"{fcs_info['total_aim_lead_mils']:.2f} mils", delta=f"{fcs_info['total_crossrange_lead_m']:.2f} m lead")
        dk4.metric("Sabot Petal Hazard Zone", f"{sabot_sim['sabot_hazard_zone']['danger_length_m']:.0f}m x {sabot_sim['sabot_hazard_zone']['danger_width_m']:.0f}m", delta=f"{sabot_sim['petal_discard_completed_distance_m']}m discard")

        fig_direct_3d = create_3d_direct_fire_sabot_figure(sabot_sim, direct_sim)
        st.plotly_chart(fig_direct_3d, width="stretch")

        with st.expander("🔬 Sabot Petal Discard Telemetry & FCS Corrections", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Sabot Separation Dynamics**")
                st.write(f"- **Petal Count**: {sabot_sim['n_petals']}")
                st.write(f"- **Discard Complete Distance**: {sabot_sim['petal_discard_completed_distance_m']} m")
                st.write(f"- **Tip-Off Induced Yaw Rate**: {sabot_sim['rod_tip_off_yaw_rate_deg_s']} °/s")
                st.write(f"- **Danger Zone Length**: {sabot_sim['sabot_hazard_zone']['danger_length_m']:.0f} m")
            with sc2:
                st.markdown("**Fire Control Error Decomposition**")
                st.write(f"- **Time of Flight**: {fcs_info['tof_s']:.3f} s")
                st.write(f"- **Supersonic Vertical Drop**: {fcs_info['vertical_drop_m']:.2f} m")
                st.write(f"- **Kinematic Target Lead**: {fcs_info['kinematic_lead_distance_m']:.2f} m")
                st.write(f"- **Trunnion Cant Drift**: {fcs_info['cant_error_drift_m']:.2f} m")
                st.write(f"- **Crosswind Jump**: {fcs_info['crosswind_drift_m']:.2f} m")

    # --- SUB-LAB 16: ELECTRONIC WARFARE, GPS JAMMING & LASER HOMING ---
    elif active_lab == _PHYSICS_LABS[15]:
        st.subheader("📡 Electronic Warfare (EW), GPS Jamming Resilience & Semi-Active Laser Guidance")
        st.markdown(
            "Precision guided munitions operating under hostile electronic warfare: "
            "evaluates standoff GPS noise jamming using Friis transmission ($J/S$ carrier unlock ratio), "
            "models post-denial tactical MEMS IMU dead-reckoning drift growth, and simulates terminal "
            "Semi-Active Laser (SAL) quadrant photodiode homing through aerosol/smoke obscurants (**Beer-Lambert Law**)."
        )
        st.latex(
            r"""
            \left[\frac{J}{S}\right]_{\text{dB}} = P_{J,\text{tx}} - 20\log_{10}\left(\frac{4\pi R_J}{\lambda}\right) - P_{S}, \quad
            \sigma_{\text{pos}}(t) \approx \frac{1}{2} b_a \Delta t^2 + \frac{1}{6} g b_g \Delta t^3, \quad
            T_{\text{optical}} = \exp\left(-\gamma_{\text{ext}} R_{\text{slant}}\right)
            """
        )

        ew1, ew2, ew3 = st.columns(3)
        with ew1:
            st.markdown("**Standoff EW Jammer Transmitter**")
            ew_pwr = st.slider("Jammer Radiated Power (kW)", 0.5, 25.0, 8.0, 0.5, key="ew_pwr")
            ew_dist = st.slider("Jammer Standoff Distance (km)", 5.0, 30.0, 15.0, 1.0, key="ew_dist") * 1000.0
            ew_thresh = st.slider("GPS Unlock J/S Threshold (dB)", 35.0, 55.0, 42.0, 1.0, key="ew_thresh")

        with ew2:
            st.markdown("**Inertial Navigation System (IMU) Quality**")
            imu_gyro = st.slider("Gyroscope Bias Drift Rate (°/hr)", 0.2, 10.0, 1.5, 0.2, key="ew_gyro")
            imu_acc = st.slider("Accelerometer Bias (mg)", 0.5, 8.0, 2.0, 0.2, key="ew_acc")
            guidance_range = st.slider("Target Nominal Range (m)", 10000.0, 25000.0, 18000.0, 1000.0, key="ew_rng")

        with ew3:
            st.markdown("**Terminal Laser Seeker & Obscurant**")
            smoke_coeff = st.slider("Battlefield Smoke Extinction γ (/km)", 0.0, 0.60, 0.15, 0.05, key="ew_smoke")
            laser_jit = st.slider("Designator Spot Jitter σ (m)", 0.10, 1.00, 0.35, 0.05, key="ew_jit")
            laser_homing_on = st.checkbox("Terminal SAL Laser Seeker Equipped", value=True, key="ew_laser")

        with st.spinner("Simulating Electronic Warfare Guidance Dynamics..."):
            ew_sim = simulate_ins_gps_guidance_under_jamming(
                nominal_range_m=guidance_range,
                jammer_pos_m=(ew_dist, 1800.0, 80.0),
                jammer_power_kw=ew_pwr,
                gps_denial_js_threshold_db=ew_thresh,
                imu_gyro_drift_deg_per_hr=imu_gyro,
                imu_accel_bias_mg=imu_acc,
                seed=42,
            )
            sal_sim = simulate_sal_seeker_tracking(
                target_pos_m=(guidance_range, 0.0, 0.0),
                designator_pos_m=(guidance_range - 1500.0, -1200.0, 150.0),
                laser_jitter_sigma_m=laser_jit,
                smoke_extinction_coeff_per_km=smoke_coeff,
                trajectory_pts=ew_sim["trajectory"][-12:],
                seed=42,
            )

        ek1, ek2, ek3, ek4 = st.columns(4)
        ek1.metric("GPS Tracking Status", "DENIED (Jammed)" if ew_sim["gps_denied"] else "LOCKED (Aided)", delta="INS Fallback" if ew_sim["gps_denied"] else "Precision GPS")
        ek2.metric("Max Jamming J/S Ratio", f"{ew_sim['max_js_ratio_db']:.1f} dB", delta=f"{ew_sim['max_js_ratio_db'] - ew_thresh:+.1f} dB above threshold")
        final_cep = sal_sim["miss_distance_m"] if (laser_homing_on and sal_sim["seeker_locked"]) else ew_sim["terminal_miss_distance_m"]
        ek3.metric("Terminal Miss Distance", f"{final_cep:.2f} m", delta="Laser Homing Kill" if (laser_homing_on and sal_sim["seeker_locked"]) else ew_sim["cep_category"])
        ek4.metric("Laser Optical Transmission", f"{sal_sim['optical_transmission']*100.0:.1f} %", delta="Lock Acquired" if sal_sim["seeker_locked"] else "Smoke Attenuated")

        fig_ew_3d = create_3d_ew_guidance_figure(ew_sim, sal_sim)
        st.plotly_chart(fig_ew_3d, width="stretch")

        with st.expander("📋 EW Electronic Warfare & Laser Guidance Telemetry Log", expanded=False):
            st.write(f"- **Standoff Distance at Denial**: {ew_sim.get('standoff_distance_at_denial_m', 0.0):.1f} m")
            st.write(f"- **Pure Inertial Flight Duration**: {ew_sim.get('pure_inertial_flight_duration_s', 0.0):.1f} s")
            st.write(f"- **Laser Seeker Acquisition Lock Range**: {sal_sim.get('lock_range_m', 0.0):.1f} m")
            st.write(f"- **Laser Spot Position**: {sal_sim.get('laser_spot_pos_m', [])}")
