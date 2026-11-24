"""Test suite for the 4 scientific upgrades:
1. Precision Guidance, Autopilot Dynamics, and Proportional Navigation (PN / APN)
2. Cluster Munition Dispenser Kinematics and UXO Contamination Analysis (IATG 10.10 / CCM)
3. Full 6-Degree-of-Freedom (6-DoF) Rigid-Body Projectile Dynamics (STANAG 4355)
4. Ammunition In-Service Surveillance, Propellant Aging & Thermal Stability (IATG 07.10 / NATO AOP-48)
"""

import math
import numpy as np
import plotly.graph_objects as go
import pytest

from ammo_mc.guidance import (
    proportional_navigation_acceleration,
    simulate_guided_trajectory,
    run_guidance_monte_carlo,
)
from ammo_mc.dispenser import simulate_cluster_dispenser_footprint
from ammo_mc.six_dof import (
    simulate_6dof_trajectory,
    quaternion_to_dcm,
    euler_angles_to_quaternion,
)
from ammo_mc.surveillance import (
    calculate_arrhenius_equivalent_temp,
    calculate_arrhenius_rate,
    simulate_propellant_aging,
    generate_surveillance_sampling_plan,
)
from ammo_mc.vis_3d import (
    create_3d_guided_intercept_figure,
    create_3d_submunition_dispersion_figure,
    create_3d_6dof_attitude_figure,
)


def test_proportional_navigation_kinematics_and_intercept() -> None:
    # 1. Test commanded acceleration computation
    r_target = np.array([3000.0, 100.0, 0.0])
    v_target = np.array([0.0, 15.0, 0.0])
    r_missile = np.array([1000.0, 0.0, 300.0])
    v_missile = np.array([350.0, 0.0, 0.0])

    a_cmd_tpn, tel = proportional_navigation_acceleration(
        r_target=r_target,
        v_target=v_target,
        r_missile=r_missile,
        v_missile=v_missile,
        N_prime=3.5,
        law="TPN",
    )
    assert tel["range_m"] > 2000.0
    assert tel["closing_velocity_mps"] > 200.0
    assert tel["los_rate_rad_s"] > 0.0
    assert len(a_cmd_tpn) == 3

    # 2. Test closed-loop guided flight trajectory to direct hit
    res_direct = simulate_guided_trajectory(
        elev_deg=22.0,
        target_pos_m=(3500.0, 50.0, 0.0),
        target_vel_mps=(0.0, 0.0, 0.0),
        target_accel_type="stationary",
        guidance_law="TPN",
        N_prime=4.0,
        lethal_radius_m=3.5,
    )
    assert res_direct["hit"] is True
    assert res_direct["status"] == "DIRECT_HIT"
    assert res_direct["miss_distance_m"] <= 3.5
    assert res_direct["time_of_flight_s"] > 5.0
    assert res_direct["peak_g_load"] <= 18.0

    # 3. Test Guidance Monte Carlo
    mc_res = run_guidance_monte_carlo(n_trials=8, seed=42)
    assert mc_res["n_trials"] == 8
    assert mc_res["p_intercept"] >= 0.75
    assert mc_res["cep50_m"] < 5.0

    # 4. Test 3D Guided Intercept Figure
    fig = create_3d_guided_intercept_figure(res_direct)
    assert isinstance(fig, go.Figure)
    assert any("Guided Missile" in (t.name or "") for t in fig.data)


def test_cluster_munition_dispenser_kinematics_and_uxo_hazard() -> None:
    # 1. Test payload ejection and ground dispersion
    disp_res = simulate_cluster_dispenser_footprint(
        n_submunitions=48,
        submunition_mass_kg=0.23,
        burst_pos_m=(10000.0, 0.0, 350.0),
        burst_vel_mps=(220.0, 0.0, -70.0),
        carrier_spin_rad_s=50.0,
        wind_downrange_mps=3.0,
        wind_crossrange_mps=2.0,
        p_dud_primary=0.04,
        p_fail_self_destruct=0.08,
        common_cause_beta=0.03,
        seed=123,
    )

    assert disp_res["n_submunitions"] == 48
    assert (disp_res["n_detonated"] + disp_res["n_self_destruct_neutralized"] + disp_res["n_uxo"]) == 48
    assert disp_res["footprint_area_hectares"] > 0.1
    assert disp_res["footprint_semi_major_m"] > disp_res["footprint_semi_minor_m"]
    assert disp_res["iatg_cordon_radius_m"] > 200.0  # IATG minimum safety buffer
    assert disp_res["mean_terminal_velocity_mps"] > 15.0

    # 2. Test 3D Submunition Dispersion Figure
    fig = create_3d_submunition_dispersion_figure(disp_res)
    assert isinstance(fig, go.Figure)
    assert any("Carrier Approach" in (t.name or "") for t in fig.data)
    assert any("IATG 10.10" in (t.name or "") for t in fig.data)


def test_full_6dof_rigid_body_projectile_dynamics() -> None:
    # 1. Quaternion to DCM correctness
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    dcm_ident = quaternion_to_dcm(q_ident)
    assert np.allclose(dcm_ident, np.eye(3))

    q_rot = euler_angles_to_quaternion(pitch_rad=math.radians(30.0), yaw_rad=0.0)
    assert np.isclose(np.linalg.norm(q_rot), 1.0)

    # 2. Integrate 6-DoF trajectory for 155mm spin-stabilized projectile
    res_6dof = simulate_6dof_trajectory(
        v0_mps=827.0,
        elev_deg=35.0,
        calibre_m=0.155,
        mass_kg=43.5,
        twist_cal_per_turn=20.0,
        initial_pitch_rate_rad_s=0.01,
        initial_yaw_rate_rad_s=0.02,
        dt_s=0.002,
    )

    assert res_6dof["final_range_m"] > 15000.0
    assert res_6dof["final_deflection_m"] > 50.0  # Rightward gyroscopic spin drift
    assert res_6dof["time_of_flight_s"] > 45.0
    assert res_6dof["muzzle_sg"] > 1.2  # Gyroscopic stability S_g > 1.0
    assert res_6dof["terminal_spin_rpm"] > 5000.0  # Retains spin at impact

    # 3. Test 3D 6-DoF Attitude Figure
    fig = create_3d_6dof_attitude_figure(res_6dof)
    assert isinstance(fig, go.Figure)
    assert any("6-DoF Rigid-Body" in (t.name or "") for t in fig.data)
    assert any("Symmetry Axis" in (t.name or "") for t in fig.data)


def test_in_service_ammunition_surveillance_and_aging() -> None:
    # 1. Arrhenius equivalent temperature calculation
    t_eq = calculate_arrhenius_equivalent_temp(mean_temp_C=37.0, delta_temp_C=14.0, ea_J_mol=95000.0)
    assert t_eq > 37.0  # T_eq must be higher than mean due to exponential Arrhenius kinetics

    # 2. Temperate storage aging
    res_temp = simulate_propellant_aging(
        stabilizer_type="Diphenylamine (DPA, Single-Base)",
        storage_climate="Zone C1 (Temperate Depot, 21°C)",
        storage_duration_years=20.0,
    )
    assert res_temp["final_stabilizer_pct"] > 0.50
    assert "SAFE" in res_temp["status_label"]
    assert res_temp["years_to_critical_shelf_life"] > 40.0

    # 3. Desert harsh storage aging
    res_desert = simulate_propellant_aging(
        stabilizer_type="Diphenylamine (DPA, Single-Base)",
        storage_climate="Zone A1 (Extreme Hot Dry)",
        storage_duration_years=12.0,
    )
    assert res_desert["final_stabilizer_pct"] < 0.20
    assert "HAZARD" in res_desert["status_label"]
    assert res_desert["years_to_critical_shelf_life"] < 10.0

    # 4. ISO 2859-1 / STANAG 4147 sampling plan (Code M, n=200, AQL=0.65% -> Table 2-A Ac=3, Re=4)
    plan = generate_surveillance_sampling_plan(lot_size=5000, inspection_level="Level II (Normal Standard)", aql_percent=0.65)
    assert plan["sample_size_n"] == 200
    assert plan["acceptance_number_ac"] == 3
    assert plan["rejection_number_re"] == 4
    assert plan["producer_risk_alpha_pct"] < 6.0
    assert len(plan["oc_curve"]["p_accept"]) == 100
