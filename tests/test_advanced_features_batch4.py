"""Test suite for the four scientific and engineering upgrades:
1. Counter-Battery Radar & Trajectory Reconstruction (NATO STANAG 4114 / AN/TPQ-53)
2. Upper Atmospheric Soundings & Wind Shear (NATO STANAG 4082 METCM)
3. Electronic Warfare, GPS Jamming Resilience & Semi-Active Laser Guidance
4. Direct-Fire Tank Gunnery & APFSDS Sabot Discard Dynamics
"""

import math
import numpy as np
import plotly.graph_objects as go
import pytest

from ammo_mc.radar import (
    calculate_projectile_rcs,
    simulate_radar_tracking,
    reconstruct_point_of_origin_and_impact,
)
from ammo_mc.metcm import (
    generate_stanag_4082_profile,
    integrate_trajectory_with_sounding,
)
from ammo_mc.electronic_warfare import (
    simulate_sal_seeker_tracking,
    simulate_ins_gps_guidance_under_jamming,
)
from ammo_mc.direct_fire import (
    simulate_apfsds_sabot_discard,
    calculate_moving_target_lead,
    simulate_direct_fire_engagement,
)
from ammo_mc.vis_3d import (
    create_3d_counter_battery_radar_figure,
    create_3d_atmospheric_shear_figure,
    create_3d_ew_guidance_figure,
    create_3d_direct_fire_sabot_figure,
)


def test_counter_battery_radar_and_reconstruction() -> None:
    # 1. Test Projectile Radar Cross Section (RCS) model
    rcs_nose = calculate_projectile_rcs(calibre_m=0.155, length_m=0.85, aspect_angle_deg=5.0)
    rcs_broad = calculate_projectile_rcs(calibre_m=0.155, length_m=0.85, aspect_angle_deg=85.0)
    assert 0.003 <= rcs_nose < rcs_broad
    assert rcs_broad > 0.1

    # 2. Simulate projectile flight and radar track acquisition
    # Generate real aerodynamic trajectory using exterior ballistics
    from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch

    atm_env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=0.155, fidelity="L1", dt=0.08)
    s0 = np.array([[0.0, 0.0, 0.0, 650.0, 0.0, 320.0]], dtype=np.float64)
    res_traj = integrate_trajectories_batch(s0, np.array([43.5]), np.array([1.0]), atm_env, cfg, record_single_path=True)
    sp = res_traj["sample_path"]

    x_m = list(sp["range_m"])
    y_m = [0.0] * len(sp["t"])
    z_m = list(sp["alt_m"])
    t_s = list(sp["t"])

    radar_res = simulate_radar_tracking(
        trajectory_x_m=x_m,
        trajectory_y_m=y_m,
        trajectory_z_m=z_m,
        time_s=t_s,
        radar_pos_m=(4000.0, -2500.0, 50.0),
        p_detection_nominal=0.95,
        seed=42,
    )
    assert radar_res["total_detections"] >= 10
    assert len(radar_res["track_points"]) >= 10

    # 3. Test Inverse Ballistics Reconstruct Point of Origin (POO) & Impact (POI)
    recon = reconstruct_point_of_origin_and_impact(
        track_points=radar_res["track_points"],
        gun_elev_m=0.0,
        target_elev_m=0.0,
    )
    assert recon["success"] is True
    # POO should be very close to weapon launch point (0, 0)
    assert abs(recon["poo_x_m"] - 0.0) < 50.0
    assert abs(recon["poo_y_m"] - 0.0) < 50.0
    # POI should be close to true impact range
    true_range = float(res_traj["range_m"][0])
    assert abs(recon["poi_x_m"] - true_range) < 600.0
    assert recon["early_warning_countdown_s"] > 2.0
    assert recon["poo_error_ellipse"]["semi_major_m"] > 0.0

    # 4. Test 3D Radar Figure
    fig = create_3d_counter_battery_radar_figure(radar_res, recon)
    assert isinstance(fig, go.Figure)


def test_stanag_4082_metcm_wind_shear() -> None:
    # 1. Test profile generation across 20 altitude zones
    prof = generate_stanag_4082_profile(
        surface_temp_c=20.0,
        surface_pressure_hpa=1015.0,
        surface_wind_speed_mps=8.0,
        surface_wind_dir_deg=270.0,
        thermal_inversion=True,
    )
    zones = prof["zones"]
    assert len(zones) == 20
    assert zones[0]["altitude_m"] == 0
    assert zones[-1]["altitude_m"] == 20000
    # Temperature at 11,000m should be colder than surface
    assert zones[14]["temp_c"] < zones[0]["temp_c"]
    # Wind speed should increase with altitude into jet stream
    assert zones[10]["wind_speed_mps"] > zones[0]["wind_speed_mps"]

    # 2. Test trajectory integration through sheared atmosphere
    met_res = integrate_trajectory_with_sounding(
        v0_mps=827.0,
        qe_deg=35.0,
        az_deg=90.0,
        sounding_profile=prof,
    )
    assert met_res["range_with_met_m"] > 8000.0
    assert met_res["vertex_altitude_m"] > 1500.0
    assert abs(met_res["delta_range_met_m"]) > 0.0
    assert met_res["crossrange_drift_m"] > 0.0

    # 3. Test 3D Met Shear Figure
    fig = create_3d_atmospheric_shear_figure(met_res, prof)
    assert isinstance(fig, go.Figure)


def test_electronic_warfare_and_laser_guidance() -> None:
    # 1. Test Semi-Active Laser (SAL) quadrant seeker homing
    sal_res = simulate_sal_seeker_tracking(
        target_pos_m=(14000.0, 0.0, 0.0),
        laser_jitter_sigma_m=0.25,
        smoke_extinction_coeff_per_km=0.10,
        seed=42,
    )
    assert sal_res["seeker_locked"] is True
    assert sal_res["miss_distance_m"] < 2.0
    assert sal_res["optical_transmission"] > 0.1

    # 2. Test Standoff EW GPS Noise Jammer & Inertial Navigation Drift
    ew_res = simulate_ins_gps_guidance_under_jamming(
        nominal_range_m=18000.0,
        jammer_pos_m=(15000.0, 1500.0, 50.0),
        jammer_power_kw=10.0,
        gps_denial_js_threshold_db=40.0,
        imu_gyro_drift_deg_per_hr=2.0,
        seed=42,
    )
    assert ew_res["gps_denied"] is True
    assert ew_res["max_js_ratio_db"] > 40.0
    assert ew_res["standoff_distance_at_denial_m"] > 0.0
    assert ew_res["pure_inertial_flight_duration_s"] > 0.0
    assert len(ew_res["trajectory"]) > 10

    # 3. Test 3D EW Figure
    fig = create_3d_ew_guidance_figure(ew_res, sal_res)
    assert isinstance(fig, go.Figure)


def test_direct_fire_tank_gunnery_and_sabot_discard() -> None:
    # 1. Test APFSDS sabot petal separation dynamics
    sabot_res = simulate_apfsds_sabot_discard(
        v0_mps=1680.0,
        rod_calibre_m=0.026,
        rod_length_m=0.78,
        rod_mass_kg=4.85,
        sabot_mass_kg=3.15,
        n_petals=3,
        petal_divergence_angle_deg=8.5,
    )
    assert sabot_res["n_petals"] == 3
    assert len(sabot_res["petal_paths"]) == 3
    assert sabot_res["petal_discard_completed_distance_m"] < 60.0
    assert sabot_res["sabot_hazard_zone"]["danger_length_m"] > 100.0
    assert sabot_res["rod_tip_off_yaw_rate_deg_s"] > 0.0

    # 2. Test Moving Target Fire Control Lead
    fcs_lead = calculate_moving_target_lead(
        target_range_m=2400.0,
        target_speed_kmh=40.0,
        target_crossing_angle_deg=90.0,
        v0_mps=1680.0,
    )
    assert fcs_lead["tof_s"] < 2.0
    assert fcs_lead["kinematic_lead_distance_m"] > 10.0
    assert fcs_lead["total_aim_lead_mils"] > 4.0

    # 3. Test Complete Direct-Fire Engagement
    eng_res = simulate_direct_fire_engagement(
        target_range_m=2000.0,
        target_speed_kmh=25.0,
        target_crossing_angle_deg=80.0,
        v0_mps=1680.0,
        armor_thickness_rha_mm=550.0,
        armor_obliquity_deg=60.0,
        fcs_lead_applied=True,
    )
    assert eng_res["lo_penetration_normal_mm"] > 500.0
    assert eng_res["v_impact_mps"] > 1400.0
    assert "HIT" in eng_res["engagement_status"] or "DIRECT" in eng_res["engagement_status"]

    # 4. Test 3D Direct Fire Figure
    fig = create_3d_direct_fire_sabot_figure(sabot_res, eng_res)
    assert isinstance(fig, go.Figure)
