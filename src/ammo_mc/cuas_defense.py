"""Complete 5-Module Counter-UAS (C-UAS) Multi-Shotgun Turret Engineering Suite.

Implements first-principles physics, stochastic Monte Carlo engines, and multi-objective
optimization across five advanced C-UAS capabilities:
  1. Closed-Loop Fire Control (Radar / EO-IR / Kalman a-b-g), Servo Limits & Evasive Drones
  2. Multi-Drone Swarm Saturation & Sub-Salvo Magazine Allocation Doctrine
  3. Advanced C-UAS Payloads (Tungsten Super-Shot, Duplex, Flechettes & Time-Fuzed Airburst)
  4. Turret Recoil Impulse, 2-Axis Gimbal Shock Loads & Constrained Pareto Array Optimizer
  5. Elevated-Fire 3-DoF Pellet Fallout Ballistics & Friendly Surface Danger Zone (SDZ)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.aero_sphere import sphere_drag_compressible, sphere_drag_compressible_array
from ammo_mc.small_arms import (
    PELLET_MATERIAL_DENSITIES_G_CM3,
    compute_balanced_pair_fire_times,
    compute_component_v50_perforation,
    compute_drone_projected_geometry,
    compute_rotor_blade_strike_probability,
    configure_array_pointing_vectors,
    evaluate_drone_impact,
    generate_multishotgun_array_layout,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
    get_shotgun_presets,
    simulate_anti_drone_multishotgun_mc,
)

# -------------------------------------------------------------------------
# UPGRADE 1: CLOSED-LOOP FCS, SENSOR TRACKING & EVASIVE DRONE MANEUVERS
# -------------------------------------------------------------------------

CUAS_SENSOR_PRESETS: dict[str, dict[str, Any]] = {
    "EO_IR_Dual_Thermal_Tracker": {
        "name": "EO/IR Dual Optical-Thermal Video Tracker",
        "sigma_az_mrad": 0.45,
        "sigma_el_mrad": 0.45,
        "sigma_range_m": 1.20,
        "update_rate_hz": 60.0,
        "fcs_latency_ms": 28.0,
        "description": "High angular precision optical/LWIR tracker with laser rangefinder updates.",
    },
    "Ku_Band_Micro_Doppler_Radar": {
        "name": "Ku-Band AESA Micro-Doppler Drone Radar",
        "sigma_az_mrad": 1.60,
        "sigma_el_mrad": 1.60,
        "sigma_range_m": 0.35,
        "update_rate_hz": 40.0,
        "fcs_latency_ms": 35.0,
        "description": "All-weather 3D Ku-band radar with crisp range accuracy and moderate beamwidth jitter.",
    },
    "Sensor_Fused_Radar_EOIR": {
        "name": "Sensor-Fused Ku-Band Radar + EO/IR Tracker",
        "sigma_az_mrad": 0.40,
        "sigma_el_mrad": 0.40,
        "sigma_range_m": 0.25,
        "update_rate_hz": 60.0,
        "fcs_latency_ms": 24.0,
        "description": "Low-latency fused tracking combining EO angular sharpness with radar range lock.",
    },
    "Acoustic_Coarse_Cueing_Only": {
        "name": "Passive Acoustic Array Cueing (Degraded)",
        "sigma_az_mrad": 12.0,
        "sigma_el_mrad": 12.0,
        "sigma_range_m": 3.50,
        "update_rate_hz": 15.0,
        "fcs_latency_ms": 85.0,
        "description": "Wide-dispersion acoustic bearing cueing under optical/RF blackout conditions.",
    },
}


def get_cuas_sensor_presets() -> dict[str, dict[str, Any]]:
    """Return catalog of C-UAS fire-control tracking sensor suites."""
    return CUAS_SENSOR_PRESETS


def evaluate_evasive_drone_state(
    t_s: float | np.ndarray,
    target_distance_m: float,
    velocity_mps: float,
    heading_deg: float,
    maneuver_profile: str = "sinusoidal_weave",
    max_maneuver_g: float = 2.5,
    weave_freq_rad_s: float = 4.0,
    phase_rad: float = 0.0,
) -> dict[str, Any]:
    """Evaluate exact 3D drone position, velocity, and acceleration at time t_s (t=0 at gun fire)."""
    t_arr = np.asarray(t_s, dtype=np.float64)
    heading_rad = math.radians(heading_deg)
    # 90 deg = pure crossing (+Y), 0 deg = closing (-X)
    vx_base = -velocity_mps * math.cos(heading_rad)
    vy_base = velocity_mps * math.sin(heading_rad)
    vz_base = 0.0

    a_amp = max(float(max_maneuver_g), 0.0) * 9.80665
    w = max(float(weave_freq_rad_s), 0.5)

    if maneuver_profile == "linear_cruise":
        dy_m = np.zeros_like(t_arr)
        dz_m = np.zeros_like(t_arr)
        vy_m = np.zeros_like(t_arr)
        vz_m = np.zeros_like(t_arr)
        ay_m = np.zeros_like(t_arr)
        az_m = np.zeros_like(t_arr)
    elif maneuver_profile == "corkscrew_spiral":
        amp = a_amp / (w ** 2)
        dy_m = amp * (np.sin(w * t_arr + phase_rad) - np.sin(phase_rad))
        dz_m = amp * (np.cos(w * t_arr + phase_rad) - np.cos(phase_rad))
        vy_m = (a_amp / w) * np.cos(w * t_arr + phase_rad)
        vz_m = -(a_amp / w) * np.sin(w * t_arr + phase_rad)
        ay_m = -a_amp * np.sin(w * t_arr + phase_rad)
        az_m = -a_amp * np.cos(w * t_arr + phase_rad)
    elif maneuver_profile == "terminal_popup_jink":
        # Drone cruises until t_jink = -0.22 s, then initiates sharp combined lateral+vertical jink
        t_jink = -0.22
        dt_j = np.maximum(t_arr - t_jink, 0.0)
        mask_j = (t_arr > t_jink).astype(np.float64)
        ay_j = 0.75 * a_amp * math.cos(phase_rad)
        az_j = 0.65 * a_amp
        dy_m = 0.5 * ay_j * (dt_j ** 2)
        dz_m = 0.5 * az_j * (dt_j ** 2)
        vy_m = ay_j * dt_j
        vz_m = az_j * dt_j
        ay_m = ay_j * mask_j
        az_m = az_j * mask_j
    elif maneuver_profile == "singer_correlated":
        from ammo_mc.cuas_ballistics_v2 import generate_singer_evasive_trajectory

        rng_s = np.random.default_rng(int(abs(phase_rad) * 1000) + 7)
        s_traj = generate_singer_evasive_trajectory(
            t_arr if np.ndim(t_arr) > 0 else np.array([float(t_arr)]),
            target_distance_m=target_distance_m,
            velocity_mps=velocity_mps,
            heading_deg=heading_deg,
            sigma_a_mps2=max(1.0, a_amp),
            tau_m_s=1.2,
            rng=rng_s,
        )
        if np.ndim(t_arr) == 0:
            return {k: float(v[0]) for k, v in s_traj.items()}
        return s_traj
    else:  # "sinusoidal_weave"
        amp = a_amp / (w ** 2)
        dy_m = amp * (np.sin(w * t_arr + phase_rad) - np.sin(phase_rad))
        dz_m = 0.30 * amp * (np.cos(0.75 * w * t_arr + phase_rad) - np.cos(phase_rad))
        vy_m = (a_amp / w) * np.cos(w * t_arr + phase_rad)
        vz_m = -0.30 * 0.75 * (a_amp / w) * np.sin(0.75 * w * t_arr + phase_rad)
        ay_m = -a_amp * np.sin(w * t_arr + phase_rad)
        az_m = -0.30 * (0.75 ** 2) * a_amp * np.cos(0.75 * w * t_arr + phase_rad)

    x_pos = float(target_distance_m) + vx_base * t_arr
    y_pos = vy_base * t_arr + dy_m
    z_pos = vz_base * t_arr + dz_m
    vx_tot = np.full_like(t_arr, vx_base)
    vy_tot = vy_base + vy_m
    vz_tot = vz_base + vz_m

    return {
        "x": float(x_pos) if np.ndim(x_pos) == 0 else x_pos,
        "y": float(y_pos) if np.ndim(y_pos) == 0 else y_pos,
        "z": float(z_pos) if np.ndim(z_pos) == 0 else z_pos,
        "vx": float(vx_tot) if np.ndim(vx_tot) == 0 else vx_tot,
        "vy": float(vy_tot) if np.ndim(vy_tot) == 0 else vy_tot,
        "vz": float(vz_tot) if np.ndim(vz_tot) == 0 else vz_tot,
        "ay": float(ay_m) if np.ndim(ay_m) == 0 else ay_m,
        "az": float(az_m) if np.ndim(az_m) == 0 else az_m,
    }


def _run_3state_kalman_axis(
    z_meas: np.ndarray,
    dt: float,
    meas_sigma: float,
    accel_process_q: float = 40.0,
) -> tuple[float, float, float]:
    """Run a discrete 3-state [pos, vel, accel] Kalman filter on uniformly sampled measurements."""
    if len(z_meas) == 0:
        return 0.0, 0.0, 0.0
    if len(z_meas) == 1:
        return float(z_meas[0]), 0.0, 0.0

    F = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Continuous white-noise jerk process covariance Q
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    Q = accel_process_q * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    R_var = max(meas_sigma ** 2, 1e-6)
    v_init = (z_meas[min(2, len(z_meas) - 1)] - z_meas[0]) / max(min(2, len(z_meas) - 1) * dt, 1e-4)
    x_hat = np.array([z_meas[0], v_init, 0.0], dtype=np.float64)
    P = np.diag([R_var, 400.0, 900.0]).astype(np.float64)
    I3 = np.eye(3, dtype=np.float64)

    for k in range(1, len(z_meas)):
        x_hat = F @ x_hat
        P = F @ P @ F.T + Q
        innov = float(z_meas[k]) - float(x_hat[0])
        S = float(P[0, 0]) + R_var
        K = P[:, 0] / S
        x_hat = x_hat + K * innov
        P = (I3 - np.outer(K, np.array([1.0, 0.0, 0.0]))) @ P

    return float(x_hat[0]), float(x_hat[1]), float(x_hat[2])


def _run_2state_kalman_axis(
    z_meas: np.ndarray,
    dt: float,
    meas_sigma: float,
    accel_noise_q: float = 35.0,
) -> tuple[float, float]:
    """Run a discrete 2-state [pos, vel] Kalman filter for linear lead estimation."""
    if len(z_meas) == 0:
        return 0.0, 0.0
    if len(z_meas) == 1:
        return float(z_meas[0]), 0.0

    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    Q = accel_noise_q * np.array(
        [[0.25 * (dt ** 4), 0.5 * (dt ** 3)], [0.5 * (dt ** 3), dt ** 2]],
        dtype=np.float64,
    )
    R_var = max(meas_sigma ** 2, 1e-6)
    v_init = (z_meas[1] - z_meas[0]) / max(dt, 1e-4)
    x_hat = np.array([z_meas[0], v_init], dtype=np.float64)
    P = np.diag([R_var, 400.0]).astype(np.float64)
    I2 = np.eye(2, dtype=np.float64)

    for k in range(1, len(z_meas)):
        x_hat = F @ x_hat
        P = F @ P @ F.T + Q
        innov = float(z_meas[k]) - float(x_hat[0])
        S = float(P[0, 0]) + R_var
        K = P[:, 0] / S
        x_hat = x_hat + K * innov
        P = (I2 - np.outer(K, np.array([1.0, 0.0]))) @ P

    return float(x_hat[0]), float(x_hat[1])


def simulate_closed_loop_fcs_engagement_mc(
    sensor_preset: str = "EO_IR_Dual_Thermal_Tracker",
    predictor_mode: str = "kalman_alpha_beta_gamma",
    maneuver_profile: str = "sinusoidal_weave",
    max_maneuver_g: float = 2.5,
    weave_freq_rad_s: float = 4.0,
    servo_max_rate_deg_s: float = 90.0,
    servo_max_accel_deg_s2: float = 250.0,
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    pointing_mode: str = "divergent_curtain",
    splay_angle_mrad: float = 4.5,
    firing_mode: str = "balanced_pair_ripple",
    ripple_delay_ms: float = 8.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    target_distance_m: float = 35.0,
    drone_model: str = "fpv_quad_small",
    drone_velocity_mps: float = 22.0,
    drone_heading_deg: float = 90.0,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = 15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    n_volleys: int = 30,
    seed: int = 42,
    include_comparison: bool = True,
    use_serial_dead_time_chain: bool = False,
    dead_time_preset: str = "eo_60hz_1stream",
    dead_time_override_ms: float | None = None,
    use_heavy_tailed_atr: bool = False,
    enable_aspect_rcs_nulls: bool = False,
) -> dict[str, Any]:
    """Simulate closed-loop sensor tracking, Kalman lead prediction, servo limits & evasive drone intercept."""
    rng = np.random.default_rng(seed)
    sensor = CUAS_SENSOR_PRESETS.get(sensor_preset, CUAS_SENSOR_PRESETS["EO_IR_Dual_Thermal_Tracker"])
    presets = get_shotgun_presets()
    ammo = presets.get(ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])
    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    choke_defs = get_choke_definitions()

    aspect_geom = compute_drone_projected_geometry(
        drone_spec,
        aspect_mode=aspect_mode,
        drone_pitch_deg=drone_pitch_deg,
        drone_yaw_deg=drone_yaw_deg,
        drone_roll_deg=drone_roll_deg,
    )

    pellets_per_shot = int(ammo["pellets_per_shot"])
    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    v0_nom = float(ammo["v0_mps"]) + 2.2 * (default_barrel_length_in - 28.0)
    area_m2 = math.pi * 0.25 * (d_m ** 2)
    base_cd = float(ammo.get("effective_cd", 0.47))

    rho_air = 1.225
    kappa_nom = (rho_air * base_cd * area_m2) / (2.0 * max(m_kg, 1e-6))
    v_imp_nom = v0_nom * math.exp(-kappa_nom * target_distance_m)
    t_tof_nom = target_distance_m / max((v0_nom + v_imp_nom) / 2.0, 10.0)

    update_hz = float(sensor["update_rate_hz"])
    dt_track = 1.0 / update_hz
    from ammo_mc.cuas_ballistics_v2 import (
        compute_aspect_dependent_rcs_dbsm,
        compute_serial_dead_time_budget,
        sample_heavy_tailed_atr_error_mrad,
    )
    from ammo_mc.cuas_rigor import resolve_verification_grade

    ov_dt = {"edge_gpu_vision_ms": float(dead_time_override_ms)} if dead_time_override_ms is not None else None
    dead_budget = compute_serial_dead_time_budget(
        preset=dead_time_preset,
        overrides_ms=ov_dt,
        tof_s=t_tof_nom,
        a_unmod_mps2=max(0.0, float(max_maneuver_g) * 9.80665 * 0.5),
    )
    t_lat = float(dead_budget["t_dead_s"]) if use_serial_dead_time_chain else (float(sensor["fcs_latency_ms"]) / 1000.0)
    rcs_null_scale = 1.0
    if enable_aspect_rcs_nulls:
        rcs_db = compute_aspect_dependent_rcs_dbsm(-20.0, aspect_deg=float(drone_yaw_deg) if abs(float(drone_yaw_deg)) > 1e-3 else 35.0)
        rcs_null_scale = 1.0 + max(0.0, (-20.0 - rcs_db) / 20.0) * 1.4
    sigma_y_meas = target_distance_m * (float(sensor["sigma_az_mrad"]) / 1000.0) * rcs_null_scale
    sigma_z_meas = target_distance_m * (float(sensor["sigma_el_mrad"]) / 1000.0) * rcs_null_scale

    # Barrels setup
    barrels_raw = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        default_choke=default_choke,
        default_barrel_length_in=default_barrel_length_in,
        default_ammunition_preset=ammunition_preset,
    )
    barrels = configure_array_pointing_vectors(
        barrels_raw,
        pointing_mode=pointing_mode,
        intercept_range_m=target_distance_m,
        splay_angle_mrad=splay_angle_mrad,
    )
    fire_times_s = compute_balanced_pair_fire_times(barrels, firing_mode=firing_mode, ripple_delay_ms=ripple_delay_ms)

    # Choke dispersion
    c_info = choke_defs.get(default_choke, choke_defs["Modified"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_info["benchmark_40yd_30in_pct"])))
    theta_choke = sigma_ref / 36.58
    wash_turb = 1.0 + 0.35 * (max(0.0, 24.0 - default_barrel_length_in) / 24.0) ** 1.5
    sigma_radial = (0.01852 / 4.0 + theta_choke * target_distance_m * (target_distance_m / 36.58) ** 0.15) * wash_turb

    # Tracking window times from -0.80 s up to -t_lat
    t_track_grid = np.arange(-0.80, -t_lat + 1e-6, dt_track)
    servo_max_omega = math.radians(servo_max_rate_deg_s)
    servo_max_alpha = math.radians(servo_max_accel_deg_s2)

    kills = 0
    lead_errors_m: list[float] = []
    hits_counts: list[int] = []
    lethal_counts: list[int] = []
    servo_sat_count = 0
    crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 2))

    sample_track: dict[str, Any] = {}
    sample_pellets: dict[str, Any] = {"y_offset": [], "z_offset": [], "is_hit": [], "is_lethal": []}

    for v_idx in range(n_volleys):
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        true_track = evaluate_evasive_drone_state(
            t_track_grid, target_distance_m, drone_velocity_mps, drone_heading_deg, maneuver_profile, max_maneuver_g, weave_freq_rad_s, phase
        )
        if use_heavy_tailed_atr:
            y_atr_noise = target_distance_m * (sample_heavy_tailed_atr_error_mrad(rng, len(t_track_grid)) / 1000.0) * rcs_null_scale
            z_atr_noise = target_distance_m * (sample_heavy_tailed_atr_error_mrad(rng, len(t_track_grid)) / 1000.0) * rcs_null_scale
            y_meas = np.asarray(true_track["y"]) + y_atr_noise
            z_meas = np.asarray(true_track["z"]) + z_atr_noise
        else:
            y_meas = np.asarray(true_track["y"]) + rng.normal(0.0, sigma_y_meas, size=len(t_track_grid))
            z_meas = np.asarray(true_track["z"]) + rng.normal(0.0, sigma_z_meas, size=len(t_track_grid))

        # True state at nominal flight arrival time
        true_at_fire = evaluate_evasive_drone_state(0.0, target_distance_m, drone_velocity_mps, drone_heading_deg, maneuver_profile, max_maneuver_g, weave_freq_rad_s, phase)
        true_at_imp = evaluate_evasive_drone_state(t_tof_nom, target_distance_m, drone_velocity_mps, drone_heading_deg, maneuver_profile, max_maneuver_g, weave_freq_rad_s, phase)
        tau_pred = float(-t_track_grid[-1]) + t_tof_nom

        if predictor_mode == "unled_boresight":
            # Aims directly at last measured target angular position without lead
            y_aim = float(y_meas[-1])
            z_aim = float(z_meas[-1])
        elif predictor_mode == "linear_alpha_beta":
            py, vy = _run_2state_kalman_axis(y_meas, dt_track, sigma_y_meas)
            pz, vz = _run_2state_kalman_axis(z_meas, dt_track, sigma_z_meas)
            y_aim = py + vy * tau_pred
            z_aim = pz + vz * tau_pred
        else:  # "kalman_alpha_beta_gamma"
            py, vy, ay = _run_3state_kalman_axis(y_meas, dt_track, sigma_y_meas, accel_process_q=max(40.0, (max_maneuver_g * 9.81) ** 2))
            pz, vz, az = _run_3state_kalman_axis(z_meas, dt_track, sigma_z_meas, accel_process_q=max(20.0, (0.5 * max_maneuver_g * 9.81) ** 2))
            # Clamp unreasonable noisy acceleration estimates under degraded acoustic cueing
            a_cap = max(max_maneuver_g, 1.5) * 9.81 * 1.3
            ay = float(np.clip(ay, -a_cap, a_cap))
            az = float(np.clip(az, -a_cap, a_cap))
            y_aim = py + vy * tau_pred + 0.5 * ay * (tau_pred ** 2)
            z_aim = pz + vz * tau_pred + 0.5 * az * (tau_pred ** 2)

        # Servo rate/acceleration lag penalty if angular slew rate or angular accel exceeds gimbal capability
        req_omega = math.hypot(true_at_fire["vy"], true_at_fire["vz"]) / max(target_distance_m, 1.0)
        req_alpha = math.hypot(true_at_fire["ay"], true_at_fire["az"]) / max(target_distance_m, 1.0)
        servo_lag_m = 0.0
        if req_omega > servo_max_omega:
            servo_lag_m += target_distance_m * (req_omega - servo_max_omega) * max(t_lat, 0.04)
        if req_alpha > servo_max_alpha:
            servo_lag_m += 0.5 * target_distance_m * (req_alpha - servo_max_alpha) * (max(t_lat, 0.04) ** 2)
        if servo_lag_m > 0.0:
            servo_sat_count += 1
            # Lag opposes target velocity vector
            v_norm = max(math.hypot(true_at_fire["vy"], true_at_fire["vz"]), 1e-4)
            y_aim -= servo_lag_m * (true_at_fire["vy"] / v_norm)
            z_aim -= servo_lag_m * (true_at_fire["vz"] / v_norm)

        lead_err = math.hypot(y_aim - true_at_imp["y"], z_aim - true_at_imp["z"])
        lead_errors_m.append(lead_err)

        # Evaluate volley pellets against true drone position at each pellet's exact arrival time
        v_hits = 0
        v_lethal = 0
        v_fuselage_lethal = 0
        v_rotors_killed: set[str] = set()

        for b_idx, b in enumerate(barrels):
            y0, z0 = float(b["y_m"]), float(b["z_m"])
            y_bore = y_aim + y0 + target_distance_m * (float(b["aim_yaw_mrad"]) / 1000.0)
            z_bore = z_aim + z0 + target_distance_m * (float(b["aim_pitch_mrad"]) / 1000.0)
            t_fire = fire_times_s[b_idx]

            dy_arr = rng.normal(0.0, sigma_radial, size=pellets_per_shot)
            dz_arr = rng.normal(0.0, sigma_radial, size=pellets_per_shot)
            cd_arr = rng.uniform(0.92 * base_cd, 1.25 * base_cd, size=pellets_per_shot)
            kappa_arr = (rho_air * cd_arr * area_m2) / (2.0 * max(m_kg, 1e-6))
            v_imp_arr = v0_nom * np.exp(-kappa_arr * target_distance_m)
            t_arr_vec = t_fire + (target_distance_m / np.maximum(0.5 * (v0_nom + v_imp_arr), 10.0))

            drone_at_arr = evaluate_evasive_drone_state(
                t_arr_vec, target_distance_m, drone_velocity_mps, drone_heading_deg, maneuver_profile, max_maneuver_g, weave_freq_rad_s, phase
            )

            for p_i in range(pellets_per_shot):
                y_p = float(y_bore + dy_arr[p_i])
                z_p = float(z_bore + dz_arr[p_i])
                y_rel = y_p - float(drone_at_arr["y"][p_i])
                z_rel = z_p - float(drone_at_arr["z"][p_i])
                ke = 0.5 * m_kg * float(v_imp_arr[p_i] ** 2)
                hit_info = evaluate_drone_impact(y_rel, z_rel, drone_spec, ke, aspect_geometry=aspect_geom)
                if hit_info["is_hit"]:
                    v_hits += 1
                    if hit_info["is_lethal"]:
                        v_lethal += 1
                        if hit_info["component"] in ("fuselage_core", "battery_payload"):
                            v_fuselage_lethal += 1
                        elif hit_info["component"].startswith("rotor_"):
                            v_rotors_killed.add(hit_info["component"])
                        elif hit_info["component"].startswith("motor_"):
                            v_rotors_killed.add(hit_info["component"].replace("motor_", "rotor_"))
                if v_idx == 0:
                    sample_pellets["y_offset"].append(round(y_p - true_at_imp["y"], 4))
                    sample_pellets["z_offset"].append(round(z_p - true_at_imp["z"], 4))
                    sample_pellets["is_hit"].append(bool(hit_info["is_hit"]))
                    sample_pellets["is_lethal"].append(bool(hit_info["is_lethal"]))

        d_type = str(drone_spec.get("drone_type", "quadcopter"))
        req_rotors = 1 if d_type == "fixed_wing" else (2 if d_type == "quadcopter" else 3)
        if (v_lethal >= crit_hits_req) or (v_fuselage_lethal >= 1) or (len(v_rotors_killed) >= req_rotors):
            kills += 1
        hits_counts.append(v_hits)
        lethal_counts.append(v_lethal)

        if v_idx == 0:
            t_future = np.linspace(-0.60, t_tof_nom + 0.04, 50)
            fut_true = evaluate_evasive_drone_state(
                t_future, target_distance_m, drone_velocity_mps, drone_heading_deg, maneuver_profile, max_maneuver_g, weave_freq_rad_s, phase
            )
            sample_track = {
                "t_meas": [round(float(x), 3) for x in t_track_grid],
                "y_meas": [round(float(x), 3) for x in y_meas],
                "z_meas": [round(float(x), 3) for x in z_meas],
                "t_true": [round(float(x), 3) for x in t_future],
                "y_true": [round(float(x), 3) for x in fut_true["y"]],
                "z_true": [round(float(x), 3) for x in fut_true["z"]],
                "y_true_intercept": round(float(true_at_imp["y"]), 3),
                "z_true_intercept": round(float(true_at_imp["z"]), 3),
                "y_aim_predicted": round(float(y_aim), 3),
                "z_aim_predicted": round(float(z_aim), 3),
                "lead_error_sample_m": round(lead_err, 3),
            }

    # Nominal lateral maneuver excursion across TOF
    maneuver_disp = 0.5 * (max_maneuver_g * 9.80665) * (t_tof_nom ** 2) if maneuver_profile != "linear_cruise" else 0.0

    # Optional architectural benchmark comparing single barrel vs convergent vs splay+Kalman
    arch_comp: list[dict[str, Any]] = []
    if include_comparison:
        comp_cfgs = [
            ("Single-Barrel Full Choke", 1, "convergent", 0.0, "simultaneous_volley", "Full", "linear_alpha_beta"),
            ("Multi-Barrel Convergent (Linear)", max(2, n_barrels), "convergent", 0.0, "simultaneous_volley", default_choke, "linear_alpha_beta"),
            ("Multi-Barrel Splay + Kalman Lead", max(2, n_barrels), "divergent_curtain", max(splay_angle_mrad, 4.5), firing_mode, default_choke, "kalman_alpha_beta_gamma"),
        ]
        for label, nb, pm, sp, fm, chk, pred in comp_cfgs:
            c_res = simulate_closed_loop_fcs_engagement_mc(
                sensor_preset=sensor_preset,
                predictor_mode=pred,
                maneuver_profile=maneuver_profile,
                max_maneuver_g=max_maneuver_g,
                weave_freq_rad_s=weave_freq_rad_s,
                servo_max_rate_deg_s=servo_max_rate_deg_s,
                servo_max_accel_deg_s2=servo_max_accel_deg_s2,
                layout_type=layout_type if nb > 1 else "circular",
                n_barrels=nb,
                mount_radius_m=mount_radius_m if nb > 1 else 0.0,
                pointing_mode=pm,
                splay_angle_mrad=sp,
                firing_mode=fm,
                ripple_delay_ms=ripple_delay_ms,
                ammunition_preset=ammunition_preset,
                default_choke=chk,
                default_barrel_length_in=default_barrel_length_in,
                target_distance_m=target_distance_m,
                drone_model=drone_model,
                drone_velocity_mps=drone_velocity_mps,
                drone_heading_deg=drone_heading_deg,
                aspect_mode=aspect_mode,
                drone_pitch_deg=drone_pitch_deg,
                drone_yaw_deg=drone_yaw_deg,
                drone_roll_deg=drone_roll_deg,
                n_volleys=min(18, n_volleys),
                seed=seed + 101,
                include_comparison=False,
            )
            c_sm = c_res["summary"]
            arch_comp.append({
                "doctrine": label,
                "n_barrels": nb,
                "predictor": pred,
                "kill_probability_pct": c_sm["kill_probability_pct"],
                "mean_lead_error_m": c_sm["mean_lead_error_m"],
                "mean_lethal_hits": c_sm["mean_lethal_hits"],
            })

    return {
        "summary": {
            "kill_probability_pct": round((kills / max(1, n_volleys)) * 100.0, 1),
            "mean_lead_error_m": round(float(np.mean(lead_errors_m)), 3),
            "cep_lead_error_m": round(float(np.percentile(lead_errors_m, 50)), 3),
            "maneuver_displacement_during_tof_m": round(float(maneuver_disp), 3),
            "servo_saturated_pct": round((servo_sat_count / max(1, n_volleys)) * 100.0, 1),
            "mean_hits_on_drone": round(float(np.mean(hits_counts)), 1),
            "mean_lethal_hits": round(float(np.mean(lethal_counts)), 1),
            "nominal_tof_ms": round(t_tof_nom * 1000.0, 1),
            "sensor_name": sensor["name"],
            "predictor_mode": predictor_mode,
            "maneuver_profile": maneuver_profile,
            "target_distance_m": target_distance_m,
            "drone_name": drone_spec.get("name", drone_model),
            "aspect_mode": aspect_mode,
            "presented_area_cm2": round(float(aspect_geom.get("presented_area_cm2", 0.0)), 1),
            "planform_ratio_pct": round(float(aspect_geom.get("planform_ratio_pct", aspect_geom.get("cross_section_ratio_pct", 100.0))), 1),
            "cross_section_ratio_pct": round(float(aspect_geom.get("cross_section_ratio_pct", 100.0)), 1),
            "n_barrels": len(barrels),
            "use_serial_dead_time_chain": bool(use_serial_dead_time_chain),
            "dead_time_budget": dead_budget,
            "quadratic_maneuver_miss_m": dead_budget["quadratic_maneuver_miss_m"],
            "use_heavy_tailed_atr": bool(use_heavy_tailed_atr),
            "enable_aspect_rcs_nulls": bool(enable_aspect_rcs_nulls),
            "verification_grade": resolve_verification_grade(
                {
                    "sensor_noise": "V1" if use_heavy_tailed_atr else "V2",
                    "dead_time": "V1" if use_serial_dead_time_chain else "V3",
                    "maneuver_model": "V1" if maneuver_profile == "singer_correlated" else "V3",
                    "target_vulnerability": "V3",
                }
            ),
        },
        "drone_spec": drone_spec,
        "aspect_geometry": aspect_geom,
        "sample_track": sample_track,
        "sample_pellets": sample_pellets,
        "architecture_comparison": arch_comp,
    }


# -------------------------------------------------------------------------
# UPGRADE 2: MULTI-DRONE SWARM SATURATION & ENGAGEMENT TIMELINE
# -------------------------------------------------------------------------

def _estimate_burst_pk_analytics(
    slant_range_m: float,
    barrels_fired: int,
    pellets_per_shot: int,
    pellet_diameter_mm: float,
    pellet_mass_g: float,
    v0_mps: float,
    choke_name: str,
    drone_spec: dict[str, Any],
    base_cd: float = 0.47,
    aspect_mode: str = "head_on_frontal",
    aspect_geom: dict[str, Any] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    physics_fidelity: dict[str, Any] | None = None,
) -> float:
    """Analytical Poisson + lethal energy single-burst P_kill estimate at slant_range_m."""
    cp = dict(calibration_profile or {})
    pf = dict(physics_fidelity) if isinstance(physics_fidelity, dict) else (
        dict(cp["_physics_fidelity"]) if isinstance(cp.get("_physics_fidelity"), dict) else None
    )
    spread_factor = max(0.35, float(cp.get("spread_factor", 1.0) or 1.0))
    spread_exp = max(0.80, min(1.80, float(cp.get("spread_exponent", 1.15) or 1.15)))
    mv_scale = max(0.50, min(1.50, float(cp.get("mv_scale", 1.0) or 1.0)))
    cd_scale = max(0.40, min(2.20, float(cp.get("pellet_cd_scale", 1.0) or 1.0)))
    avuln_scale = max(0.25, min(2.50, float(cp.get("drone_avuln_scale", 1.0) or 1.0)))
    ecrit_scale = max(0.30, min(2.60, float(cp.get("drone_ecrit_scale", 1.0) or 1.0)))

    choke_defs = get_choke_definitions()
    c_info = choke_defs.get(choke_name, choke_defs["Modified"])
    p_bench = float(c_info["benchmark_40yd_30in_pct"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_bench))
    theta_choke = sigma_ref / 36.58
    R = max(slant_range_m, 5.0)
    cup_sep_m = float((pf or {}).get("shot_cup_separation_m", cp.get("shot_cup_separation_m", 0.0)) or 0.0)
    R_spread = max(0.5, R - min(cup_sep_m, 0.65 * R)) if cup_sep_m > 0.0 else R
    sigma_r = (0.01852 / 4.0 + theta_choke * R_spread * ((R_spread / 36.58) ** (spread_exp - 1.0))) * spread_factor

    # Pellet striking energy at range R
    rho_air = float(cp.get("air_density_kg_m3", 1.225) or 1.225) if pf else 1.225
    d_m = pellet_diameter_mm / 1000.0
    m_kg = pellet_mass_g / 1000.0
    area_m2 = math.pi * 0.25 * (d_m ** 2)
    kappa = (rho_air * base_cd * cd_scale * area_m2) / (2.0 * max(m_kg, 1e-6))
    v_imp = (v0_mps * mv_scale) * math.exp(-kappa * R)
    ke_j = 0.5 * m_kg * (v_imp ** 2)

    e_rotor = float(drone_spec.get("lethal_energy_rotor_j", 8.0)) * ecrit_scale
    if ke_j < e_rotor:
        return 0.0

    # Presented vulnerable area
    geom_eff = aspect_geom
    if geom_eff is None and aspect_mode:
        geom_eff = compute_drone_projected_geometry(drone_spec, aspect_mode=aspect_mode)

    if geom_eff is not None:
        area_vuln = float(geom_eff.get("presented_area_m2", 0.02))
    else:
        rotors = drone_spec.get("rotors", [])
        area_rotors = sum(math.pi * (float(r["radius_m"]) ** 2) for r in rotors)
        area_core = float(drone_spec.get("fuselage_w_m", 0.10)) * float(drone_spec.get("fuselage_h_m", 0.10))
        area_vuln = min(area_rotors + area_core, math.pi * (float(drone_spec.get("span_m", 0.4)) ** 2) * 0.65)
    area_vuln *= avuln_scale

    if pf and bool(pf.get("use_rotor_solidity", False)):
        rot_list = drone_spec.get("rotors", [])
        mean_r = float(np.mean([float(r.get("radius_m", 0.12)) for r in rot_list])) if rot_list else 0.12
        p_blade = compute_rotor_blade_strike_probability(
            radial_frac=0.55,
            n_blades=int(drone_spec.get("rotor_blade_count", 3)),
            rotor_radius_m=mean_r,
            rpm=float(drone_spec.get("rotor_rpm", 11500.0)),
            pellet_diameter_m=d_m,
            pellet_velocity_mps=v_imp,
        )
        if geom_eff is not None and float(geom_eff.get("presented_area_cm2", 0.0)) > 1e-3:
            rotor_share = min(0.85, float(geom_eff.get("rotors_area_cm2", 0.0)) * 0.75 / float(geom_eff["presented_area_cm2"]))
        else:
            rotor_share = 0.55
        area_vuln *= (1.0 - rotor_share) + rotor_share * max(0.28, p_blade)

    # Gaussian core hit probability inside vulnerable disk (nominal 1.2 mrad + empirical biases/jitters)
    r_eq_sq = area_vuln / math.pi
    nom_trk = float(cp.get("tracker_jitter_mrad")) if cp.get("tracker_jitter_mrad") is not None else 1.2
    aim_err_mrad = math.sqrt(
        (nom_trk ** 2)
        + (float(cp.get("aim_bias_mrad", 0.0) or 0.0) ** 2)
        + (float(cp.get("barrel_zero_scatter_mrad", 0.0) or 0.0) ** 2)
        + (float(cp.get("otm_jitter_mrad", 0.0) or 0.0) ** 2)
    )
    sigma_aim = (aim_err_mrad / 1000.0) * R
    sigma_eff_sq = (sigma_r ** 2) + (sigma_aim ** 2)
    p_pellet = 1.0 - math.exp(-r_eq_sq / max(2.0 * sigma_eff_sq, 1e-6))

    if pf and bool(pf.get("use_v50_ballistic_limit", False)):
        rho_pel = float(PELLET_MATERIAL_DENSITIES_G_CM3.get(str(pf.get("pellet_material", "Hardened Lead")), 11.3))
        obliq_deg = min(45.0, abs(float((geom_eff or {}).get("pitch_deg", 15.0))))
        v50_res = compute_component_v50_perforation(
            "fuselage_core",
            pellet_velocity_mps=v_imp,
            pellet_mass_kg=m_kg,
            pellet_diameter_m=d_m,
            pellet_density_g_cm3=rho_pel,
            obliquity_deg=obliq_deg,
            enable_ricochet=bool(pf.get("use_oblique_ricochet", False)),
        )
        energy_factor = 0.30 + 0.70 * float(v50_res["p_perforation"])
    else:
        # Transition smoothly between rotor-only lethality (~0.55) and full core lethality (1.0)
        e_core = float(drone_spec.get("lethal_energy_core_j", 20.0)) * ecrit_scale
        energy_factor = 0.55 + 0.45 * min(1.0, max(0.0, ke_j - e_rotor) / max(e_core - e_rotor, 1.0))

    flyer_frac = float(pf.get("flyer_fraction", 0.12)) if pf else 0.0
    pat_dist = str(pf.get("pattern_distribution", "gaussian")) if pf else "gaussian"
    flyer_mult = (1.0 - 0.55 * flyer_frac) if (pf and pat_dist != "gaussian") else 1.0
    shield_mult = 0.90 if (pf and bool(pf.get("enable_ray_shielding", False))) else 1.0

    lam_lethal = barrels_fired * pellets_per_shot * p_pellet * 0.78 * energy_factor * flyer_mult * shield_mult

    n_crit = int(drone_spec.get("critical_hits_for_kill", 3))
    # Poisson survival sum P(X >= n_crit)
    p_survive = 0.0
    term = math.exp(-lam_lethal)
    for k in range(n_crit):
        p_survive += term
        term *= lam_lethal / float(k + 1)
    return float(max(0.0, min(1.0, 1.0 - p_survive)))


def _trapezoidal_slew_time(angle_deg: float, max_rate_deg_s: float, max_accel_deg_s2: float) -> float:
    """Minimum-time trapezoidal/triangular angular slew time for a 2-axis turret."""
    d_th = abs(angle_deg)
    if d_th < 1e-3:
        return 0.0
    w = max(max_rate_deg_s, 1.0)
    a = max(max_accel_deg_s2, 1.0)
    d_ramp = (w ** 2) / a
    if d_th >= d_ramp:
        return (2.0 * w / a) + (d_th - d_ramp) / w
    return 2.0 * math.sqrt(d_th / a)


def simulate_drone_swarm_saturation_mc(
    swarm_size: int = 8,
    attack_tactic: str = "synchronized_omni_wave",
    priority_policy: str = "minimum_time_to_breach",
    n_barrels: int = 8,
    barrels_per_burst: int = 2,
    reload_time_s: float = 4.5,
    slew_rate_deg_s: float = 110.0,
    slew_accel_deg_s2: float = 300.0,
    settle_time_s: float = 0.18,
    max_engagement_range_m: float = 55.0,
    breach_keepout_radius_m: float = 12.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Modified",
    drone_model: str = "fpv_quad_small",
    mean_drone_speed_mps: float = 22.0,
    drone_velocity_mps: float | None = None,
    aspect_mode: str = "head_on_frontal",
    n_waves: int = 40,
    seed: int = 42,
    include_doctrine_sweep: bool = True,
) -> dict[str, Any]:
    """Event-driven Monte Carlo simulation of coordinated multi-drone swarm saturation vs a multi-barrel C-UAS turret."""
    rng = np.random.default_rng(seed)
    presets = get_shotgun_presets()
    ammo = presets.get(ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])
    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode=aspect_mode)
    eff_drone_speed = drone_velocity_mps if drone_velocity_mps is not None else mean_drone_speed_mps

    pellets_per_shot = int(ammo["pellets_per_shot"])
    d_mm = float(ammo["pellet_diameter_mm"])
    m_g = float(ammo["pellet_mass_g"])
    v0_mps = float(ammo["v0_mps"])
    base_cd = float(ammo.get("effective_cd", 0.47))

    zero_leaker_waves = 0
    killed_counts: list[int] = []
    leaker_counts: list[int] = []
    bursts_counts: list[int] = []
    reloads_counts: list[int] = []
    intercept_ranges: list[float] = []
    sample_wave_tracks: list[dict[str, Any]] = []

    for w_idx in range(n_waves):
        # Generate incoming swarm wave
        drones: list[dict[str, Any]] = []
        for d_i in range(swarm_size):
            if attack_tactic == "narrow_corridor_saturation":
                az_deg = float(rng.uniform(-22.0, 22.0))
                el_deg = float(rng.uniform(10.0, 28.0))
                r0 = float(rng.uniform(max_engagement_range_m + 5.0, max_engagement_range_m + 22.0))
            elif attack_tactic == "staggered_multi_altitude":
                az_deg = float(rng.uniform(-95.0, 95.0))
                el_deg = float(rng.choice([12.0, 32.0, 52.0]) + rng.uniform(-4.0, 4.0))
                r0 = float(rng.uniform(max_engagement_range_m + 8.0, max_engagement_range_m + 40.0))
            else:  # "synchronized_omni_wave"
                az_deg = float(rng.uniform(-145.0, 145.0))
                el_deg = float(rng.uniform(10.0, 48.0))
                r0 = float(rng.uniform(max_engagement_range_m + 5.0, max_engagement_range_m + 25.0))

            spd = float(max(6.0, rng.normal(eff_drone_speed, 0.12 * eff_drone_speed)))
            drones.append({
                "id": f"D{d_i+1}",
                "az_deg": az_deg,
                "el_deg": el_deg,
                "r0_m": r0,
                "speed_mps": spd,
                "status": "active",
                "end_r_m": r0,
                "end_t_s": 0.0,
                "bursts_absorbed": 0,
            })

        t_now = 0.0
        b_ready = int(n_barrels)
        turret_az = 0.0
        turret_el = 25.0
        wave_bursts = 0
        wave_reloads = 0

        # Event-driven loop
        while True:
            # Mark any drones that already crossed the keepout line at t_now
            for dr in drones:
                if dr["status"] == "active":
                    r_cur = dr["r0_m"] - dr["speed_mps"] * t_now
                    if r_cur <= breach_keepout_radius_m:
                        dr["status"] = "leaker_breach"
                        dr["end_r_m"] = float(breach_keepout_radius_m)
                        dr["end_t_s"] = round((dr["r0_m"] - breach_keepout_radius_m) / dr["speed_mps"], 3)

            active = [dr for dr in drones if dr["status"] == "active"]
            if not active:
                break

            # Handle empty turret reload before next engagement
            if b_ready <= 0:
                t_now += reload_time_s
                b_ready = int(n_barrels)
                wave_reloads += 1
                continue

            # Advance clock if no active drone has entered max_engagement_range_m yet
            in_range = [dr for dr in active if (dr["r0_m"] - dr["speed_mps"] * t_now) <= max_engagement_range_m]
            if not in_range:
                t_enter = min((dr["r0_m"] - max_engagement_range_m) / dr["speed_mps"] for dr in active)
                t_now = max(t_now, t_enter + 1e-4)
                in_range = [dr for dr in active if (dr["r0_m"] - dr["speed_mps"] * t_now) <= max_engagement_range_m]
                if not in_range:
                    continue

            # Select highest-priority target in range
            def _priority_key(dr: dict[str, Any]) -> float:
                r_cur = max(dr["r0_m"] - dr["speed_mps"] * t_now, 0.1)
                if priority_policy == "nearest_slant_range":
                    return r_cur
                if priority_policy == "minimum_slew_angle":
                    d_az = ((dr["az_deg"] - turret_az + 180.0) % 360.0) - 180.0
                    d_el = dr["el_deg"] - turret_el
                    return math.hypot(d_az, d_el)
                # "minimum_time_to_breach"
                return (r_cur - breach_keepout_radius_m) / max(dr["speed_mps"], 1.0)

            tgt = min(in_range, key=_priority_key)
            d_az = abs(((tgt["az_deg"] - turret_az + 180.0) % 360.0) - 180.0)
            d_el = abs(tgt["el_deg"] - turret_el)
            slew_angle = math.hypot(d_az, d_el)
            t_slew = _trapezoidal_slew_time(slew_angle, slew_rate_deg_s, slew_accel_deg_s2) + settle_time_s

            t_fire = t_now + t_slew
            turret_az = tgt["az_deg"]
            turret_el = tgt["el_deg"]

            r_fire = tgt["r0_m"] - tgt["speed_mps"] * t_fire
            if r_fire <= breach_keepout_radius_m:
                tgt["status"] = "leaker_breach"
                tgt["end_r_m"] = float(breach_keepout_radius_m)
                tgt["end_t_s"] = round((tgt["r0_m"] - breach_keepout_radius_m) / tgt["speed_mps"], 3)
                t_now = t_fire
                continue

            # Fire allocation
            n_fire = min(int(barrels_per_burst), b_ready)
            b_ready -= n_fire
            wave_bursts += 1
            tgt["bursts_absorbed"] += 1
            intercept_ranges.append(float(r_fire))

            p_kill_burst = _estimate_burst_pk_analytics(
                r_fire,
                n_fire,
                pellets_per_shot,
                d_mm,
                m_g,
                v0_mps,
                default_choke,
                drone_spec,
                base_cd=base_cd,
                aspect_geom=aspect_geom,
            )
            if float(rng.random()) <= p_kill_burst:
                tgt["status"] = "killed"
                tgt["end_r_m"] = round(float(r_fire), 2)
                tgt["end_t_s"] = round(float(t_fire), 3)

            # Small firing dwell time before re-evaluation
            t_now = t_fire + 0.08

        w_killed = sum(1 for d in drones if d["status"] == "killed")
        w_leakers = sum(1 for d in drones if d["status"] == "leaker_breach")
        killed_counts.append(w_killed)
        leaker_counts.append(w_leakers)
        bursts_counts.append(wave_bursts)
        reloads_counts.append(wave_reloads)
        if w_leakers == 0:
            zero_leaker_waves += 1

        if w_idx == 0:
            for d in drones:
                az_r = math.radians(d["az_deg"])
                el_r = math.radians(d["el_deg"])
                r0 = d["r0_m"]
                re = d["end_r_m"]
                sample_wave_tracks.append({
                    "id": d["id"],
                    "status": d["status"],
                    "x0": round(r0 * math.cos(el_r) * math.cos(az_r), 2),
                    "y0": round(r0 * math.cos(el_r) * math.sin(az_r), 2),
                    "z0": round(r0 * math.sin(el_r), 2),
                    "xe": round(re * math.cos(el_r) * math.cos(az_r), 2),
                    "ye": round(re * math.cos(el_r) * math.sin(az_r), 2),
                    "ze": round(re * math.sin(el_r), 2),
                    "end_r_m": re,
                    "end_t_s": d["end_t_s"],
                    "bursts_absorbed": d["bursts_absorbed"],
                })

    doctrine_sweep: list[dict[str, Any]] = []
    optimal_bpb = barrels_per_burst
    if include_doctrine_sweep:
        cand_bpb = sorted({1, 2, 3, 4, max(1, n_barrels // 2), int(n_barrels)})
        cand_bpb = [c for c in cand_bpb if 1 <= c <= n_barrels]
        for bpb in cand_bpb:
            sub = simulate_drone_swarm_saturation_mc(
                swarm_size=swarm_size,
                attack_tactic=attack_tactic,
                priority_policy=priority_policy,
                n_barrels=n_barrels,
                barrels_per_burst=bpb,
                reload_time_s=reload_time_s,
                slew_rate_deg_s=slew_rate_deg_s,
                slew_accel_deg_s2=slew_accel_deg_s2,
                settle_time_s=settle_time_s,
                max_engagement_range_m=max_engagement_range_m,
                breach_keepout_radius_m=breach_keepout_radius_m,
                ammunition_preset=ammunition_preset,
                default_choke=default_choke,
                drone_model=drone_model,
                mean_drone_speed_mps=eff_drone_speed,
                drone_velocity_mps=eff_drone_speed,
                aspect_mode=aspect_mode,
                n_waves=min(25, n_waves),
                seed=seed + 19,
                include_doctrine_sweep=False,
            )
            sm = sub["summary"]
            doctrine_sweep.append({
                "barrels_per_burst": bpb,
                "zero_leaker_wave_pct": sm["zero_leaker_wave_pct"],
                "mean_leakers": sm["mean_leakers"],
                "mean_drones_killed": sm["mean_drones_killed"],
                "mean_reloads": sm["mean_reloads"],
            })
        if doctrine_sweep:
            best_doc = max(doctrine_sweep, key=lambda r: (r["zero_leaker_wave_pct"], -r["mean_leakers"]))
            optimal_bpb = int(best_doc["barrels_per_burst"])

    mean_leakers = float(np.mean(leaker_counts))
    return {
        "summary": {
            "swarm_size": swarm_size,
            "zero_leaker_wave_pct": round((zero_leaker_waves / max(1, n_waves)) * 100.0, 1),
            "mean_drones_killed": round(float(np.mean(killed_counts)), 2),
            "mean_leakers": round(mean_leakers, 2),
            "leaker_rate_pct": round((mean_leakers / max(1, swarm_size)) * 100.0, 1),
            "mean_bursts_fired": round(float(np.mean(bursts_counts)), 1),
            "mean_reloads": round(float(np.mean(reloads_counts)), 2),
            "mean_intercept_range_m": round(float(np.mean(intercept_ranges)) if intercept_ranges else 0.0, 1),
            "optimal_barrels_per_burst": optimal_bpb,
            "max_engagement_range_m": max_engagement_range_m,
            "breach_keepout_radius_m": breach_keepout_radius_m,
            "attack_tactic": attack_tactic,
            "priority_policy": priority_policy,
            "aspect_mode": aspect_mode,
            "presented_area_cm2": round(float(aspect_geom["presented_area_cm2"]), 1),
            "n_barrels": n_barrels,
            "barrels_per_burst": barrels_per_burst,
        },
        "sample_wave_tracks": sample_wave_tracks,
        "doctrine_sweep": doctrine_sweep,
    }


# -------------------------------------------------------------------------
# UPGRADE 3: ADVANCED C-UAS PAYLOADS & PROGRAMMABLE AIRBURST CANISTER
# -------------------------------------------------------------------------

CUAS_PAYLOAD_CATALOG: dict[str, dict[str, Any]] = {
    "12G_LEAD_NO4_BUCK": {
        "name": "12G #4 Buckshot (Lead, 11.34 g/cm³)",
        "category": "buckshot",
        "density_kg_m3": 11340.0,
        "v0_mps": 415.0,
        "default_choke": "Modified",
        "sub_groups": [{"count": 27, "d_mm": 6.10, "m_g": 1.34, "cd": 0.47, "lethality_mult": 1.0}],
        "description": "Baseline lead #4 buckshot; strong close-in rotor destruction, sheds energy past 45 m.",
    },
    "12G_STEEL_T_SHOT": {
        "name": "12G T-Shot (Steel, 7.85 g/cm³)",
        "category": "steel_shot",
        "density_kg_m3": 7850.0,
        "v0_mps": 435.0,
        "default_choke": "Full",
        "sub_groups": [{"count": 52, "d_mm": 5.08, "m_g": 0.53, "cd": 0.47, "lethality_mult": 1.0}],
        "description": "High-count light steel shot; dense close pattern, rapid aerodynamic deceleration.",
    },
    "12G_BISMUTH_BB": {
        "name": "12G BB (Bismuth Alloy, 9.75 g/cm³)",
        "category": "bismuth",
        "density_kg_m3": 9750.0,
        "v0_mps": 420.0,
        "default_choke": "Modified",
        "sub_groups": [{"count": 62, "d_mm": 4.57, "m_g": 0.49, "cd": 0.46, "lethality_mult": 1.0}],
        "description": "Non-toxic medium-density bismuth waterfowl/C-UAS load bridging steel and lead.",
    },
    "12G_TSS_NO2": {
        "name": "12G #2 Tungsten Super-Shot TSS (18.0 g/cm³)",
        "category": "tungsten_tss",
        "density_kg_m3": 18000.0,
        "v0_mps": 410.0,
        "default_choke": "Full",
        "sub_groups": [{"count": 70, "d_mm": 3.81, "m_g": 0.52, "cd": 0.44, "lethality_mult": 1.10}],
        "description": "Ultra-dense tungsten alloy retaining lethal penetration and dense pattern beyond 75 m.",
    },
    "12G_DUPLEX_TSS_BUCK": {
        "name": "12G Stacked Duplex (12 Heavy Core + 45 TSS)",
        "category": "duplex",
        "density_kg_m3": 15000.0,
        "v0_mps": 412.0,
        "default_choke": "Modified",
        "sub_groups": [
            {"count": 12, "d_mm": 6.10, "m_g": 2.10, "cd": 0.45, "lethality_mult": 1.15},
            {"count": 45, "d_mm": 3.30, "m_g": 0.34, "cd": 0.45, "lethality_mult": 1.00},
        ],
        "description": "Dual-layer stacked shell pairing heavy core penetrators with high-density rotor shredders.",
    },
    "12G_FLECHETTE_PACK": {
        "name": "12G Saboted Steel Flechettes (20 Low-Drag Darts)",
        "category": "flechette",
        "density_kg_m3": 7850.0,
        "v0_mps": 445.0,
        "default_choke": "Cylinder",
        "sub_groups": [{"count": 20, "d_mm": 1.90, "m_g": 0.75, "cd": 0.18, "lethality_mult": 1.25}],
        "description": "Fin-stabilized needle darts with Cd ≈ 0.18 maintaining supersonic/high velocity > 90 m.",
    },
    "CUAS_AIRBURST_CANISTER": {
        "name": "Programmable Time-Fuzed Airburst Canister (85 W-Sub)",
        "category": "airburst",
        "density_kg_m3": 18000.0,
        "v0_mps": 425.0,
        "default_choke": "Cylinder",
        "carrier_mass_g": 38.0,
        "carrier_d_mm": 18.5,
        "carrier_cd": 0.28,
        "sub_groups": [{"count": 85, "d_mm": 3.50, "m_g": 0.40, "cd": 0.44, "lethality_mult": 1.15}],
        "description": "AHEAD-style muzzle-programmed time-fuze carrier ejecting 85 tungsten sub-projectiles ahead of target.",
    },
}


def get_cuas_payload_catalog() -> dict[str, dict[str, Any]]:
    """Return catalog of advanced C-UAS payload technologies."""
    return CUAS_PAYLOAD_CATALOG


def simulate_cuas_advanced_payload_mc(
    payload_key: str = "CUAS_AIRBURST_CANISTER",
    n_barrels: int = 4,
    target_distance_m: float = 75.0,
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    choke_override: str | None = None,
    programmed_standoff_m: float = 4.5,
    ejection_cone_half_angle_deg: float = 7.5,
    fuze_time_jitter_ms: float = 0.8,
    v0_meas_sigma_mps: float = 0.8,
    aim_error_mrad: float = 1.2,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = 15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    n_volleys: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate advanced C-UAS payloads (TSS, Duplex, Flechettes, Time-Fuzed Airburst) vs drones at extended range."""
    rng = np.random.default_rng(seed)
    payload = CUAS_PAYLOAD_CATALOG.get(payload_key, CUAS_PAYLOAD_CATALOG["12G_TSS_NO2"])
    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["commercial_quad_medium"])
    choke_defs = get_choke_definitions()

    aspect_geom = compute_drone_projected_geometry(
        drone_spec,
        aspect_mode=aspect_mode,
        drone_pitch_deg=drone_pitch_deg,
        drone_yaw_deg=drone_yaw_deg,
        drone_roll_deg=drone_roll_deg,
    )

    choke_name = choke_override or str(payload.get("default_choke", "Modified"))
    c_info = choke_defs.get(choke_name, choke_defs["Modified"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_info["benchmark_40yd_30in_pct"])))
    theta_choke = sigma_ref / 36.58

    v0_nom = float(payload["v0_mps"])
    is_airburst = payload["category"] == "airburst"
    rho_air = 1.225
    crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 3))
    aim_sigma_m = (target_distance_m * (aim_error_mrad / 1000.0)) / 2.0

    kills = 0
    hits_list: list[int] = []
    lethal_list: list[int] = []
    energy_list: list[float] = []
    vel_list: list[float] = []
    late_bursts = 0
    total_bursts = 0
    burst_standoffs: list[float] = []
    sample_burst_points: dict[str, list[Any]] = {"x": [], "y": [], "z": [], "status": []}
    sample_impacts: dict[str, list[Any]] = {"y": [], "z": [], "is_hit": [], "is_lethal": [], "energy_j": []}

    for v_idx in range(n_volleys):
        aim_err_y = float(rng.normal(0.0, aim_sigma_m))
        aim_err_z = float(rng.normal(0.0, aim_sigma_m))
        v_hits = 0
        v_lethal = 0
        v_energy = 0.0
        v_fuselage_lethal = 0
        v_rotors: set[str] = set()

        for b_i in range(n_barrels):
            if is_airburst:
                total_bursts += 1
                m_car = float(payload["carrier_mass_g"]) / 1000.0
                d_car = float(payload["carrier_d_mm"]) / 1000.0
                cd_car = float(payload["carrier_cd"])
                area_car = math.pi * 0.25 * (d_car ** 2)
                kappa_car = (rho_air * cd_car * area_car) / (2.0 * m_car)

                v0_true = v0_nom + float(rng.normal(0.0, 2.5))
                v0_meas = v0_true + float(rng.normal(0.0, v0_meas_sigma_mps))
                r_nom_burst = max(target_distance_m - programmed_standoff_m, 2.0)

                # Integrated distance X(t) = (1/k) * ln(1 + k*v0*t) -> invert for t_prog using v0_meas
                t_prog = (math.exp(kappa_car * r_nom_burst) - 1.0) / max(kappa_car * v0_meas, 1e-4)
                t_actual = max(0.005, t_prog + float(rng.normal(0.0, fuze_time_jitter_ms / 1000.0)))

                r_actual_burst = math.log(1.0 + kappa_car * v0_true * t_actual) / max(kappa_car, 1e-6)
                v_at_burst = v0_true / (1.0 + kappa_car * v0_true * t_actual)
                standoff_act = target_distance_m - r_actual_burst
                burst_standoffs.append(standoff_act)

                if v_idx == 0:
                    sample_burst_points["x"].append(round(r_actual_burst, 2))
                    sample_burst_points["y"].append(round(aim_err_y * (r_actual_burst / target_distance_m), 3))
                    sample_burst_points["z"].append(round(aim_err_z * (r_actual_burst / target_distance_m), 3))
                    sample_burst_points["status"].append("LATE_PAST_TARGET" if standoff_act <= 0.1 else "OPTIMAL_AIRBURST")

                if standoff_act <= 0.1:
                    late_bursts += 1
                    continue

                cone_rad = math.radians(ejection_cone_half_angle_deg)
                sigma_cone = max(0.04, (standoff_act * math.tan(cone_rad)) / 1.45)

                for grp in payload["sub_groups"]:
                    cnt = int(grp["count"])
                    d_m = float(grp["d_mm"]) / 1000.0
                    m_kg = float(grp["m_g"]) / 1000.0
                    cd_sub = float(grp["cd"])
                    mult = float(grp.get("lethality_mult", 1.0))
                    area_sub = math.pi * 0.25 * (d_m ** 2)
                    kappa_sub = (rho_air * cd_sub * area_sub) / (2.0 * m_kg)

                    dy_arr = rng.normal(0.0, sigma_cone, size=cnt)
                    dz_arr = rng.normal(0.0, sigma_cone, size=cnt)
                    # Axial + radial ejection velocity (~25 m/s boost)
                    v_eject = (v_at_burst + 22.0) * math.exp(-kappa_sub * standoff_act)
                    for k_i in range(cnt):
                        y_p = aim_err_y + float(dy_arr[k_i])
                        z_p = aim_err_z + float(dz_arr[k_i])
                        ke = 0.5 * m_kg * (v_eject ** 2) * mult
                        vel_list.append(v_eject)
                        hit_info = evaluate_drone_impact(y_p, z_p, drone_spec, ke, aspect_geometry=aspect_geom)
                        if hit_info["is_hit"]:
                            v_hits += 1
                            v_energy += 0.5 * m_kg * (v_eject ** 2)
                            if hit_info["is_lethal"]:
                                v_lethal += 1
                                if hit_info["component"] in ("fuselage_core", "battery_payload"):
                                    v_fuselage_lethal += 1
                                elif hit_info["component"].startswith("rotor_"):
                                    v_rotors.add(hit_info["component"])
                                elif hit_info["component"].startswith("motor_"):
                                    v_rotors.add(hit_info["component"].replace("motor_", "rotor_"))
                        if v_idx == 0 and b_i == 0:
                            sample_impacts["y"].append(round(y_p, 4))
                            sample_impacts["z"].append(round(z_p, 4))
                            sample_impacts["is_hit"].append(bool(hit_info["is_hit"]))
                            sample_impacts["is_lethal"].append(bool(hit_info["is_lethal"]))
                            sample_impacts["energy_j"].append(round(0.5 * m_kg * (v_eject ** 2), 2))
            else:
                # Non-airburst conventional / TSS / Duplex / Flechette shell
                sigma_r = 0.01852 / 4.0 + theta_choke * target_distance_m * (target_distance_m / 36.58) ** 0.15
                for grp in payload["sub_groups"]:
                    cnt = int(grp["count"])
                    d_m = float(grp["d_mm"]) / 1000.0
                    m_kg = float(grp["m_g"]) / 1000.0
                    cd_sub = float(grp["cd"])
                    mult = float(grp.get("lethality_mult", 1.0))
                    area_sub = math.pi * 0.25 * (d_m ** 2)

                    dy_arr = rng.normal(0.0, sigma_r, size=cnt)
                    dz_arr = rng.normal(0.0, sigma_r, size=cnt)
                    cd_arr = rng.uniform(0.92 * cd_sub, 1.20 * cd_sub, size=cnt)
                    kappa_arr = (rho_air * cd_arr * area_sub) / (2.0 * m_kg)
                    v_arr = v0_nom * np.exp(-kappa_arr * target_distance_m)
                    vel_list.extend(v_arr.tolist())

                    for k_i in range(cnt):
                        y_p = aim_err_y + float(dy_arr[k_i])
                        z_p = aim_err_z + float(dz_arr[k_i])
                        v_i = float(v_arr[k_i])
                        ke_raw = 0.5 * m_kg * (v_i ** 2)
                        hit_info = evaluate_drone_impact(y_p, z_p, drone_spec, ke_raw * mult, aspect_geometry=aspect_geom)
                        if hit_info["is_hit"]:
                            v_hits += 1
                            v_energy += ke_raw
                            if hit_info["is_lethal"]:
                                v_lethal += 1
                                if hit_info["component"] in ("fuselage_core", "battery_payload"):
                                    v_fuselage_lethal += 1
                                elif hit_info["component"].startswith("rotor_"):
                                    v_rotors.add(hit_info["component"])
                                elif hit_info["component"].startswith("motor_"):
                                    v_rotors.add(hit_info["component"].replace("motor_", "rotor_"))
                        if v_idx == 0 and b_i == 0:
                            sample_impacts["y"].append(round(y_p, 4))
                            sample_impacts["z"].append(round(z_p, 4))
                            sample_impacts["is_hit"].append(bool(hit_info["is_hit"]))
                            sample_impacts["is_lethal"].append(bool(hit_info["is_lethal"]))
                            sample_impacts["energy_j"].append(round(ke_raw, 2))

        d_type = str(drone_spec.get("drone_type", "quadcopter"))
        req_rotors = 1 if d_type == "fixed_wing" else (2 if d_type == "quadcopter" else 3)
        if (v_lethal >= crit_hits_req) or (v_fuselage_lethal >= 1) or (len(v_rotors) >= req_rotors):
            kills += 1
        hits_list.append(v_hits)
        lethal_list.append(v_lethal)
        energy_list.append(v_energy)

    total_subs_per_shell = sum(int(g["count"]) for g in payload["sub_groups"])
    sub_masses = [float(g["m_g"]) / 1000.0 for g in payload["sub_groups"] for _ in range(int(g["count"]))]
    mean_sub_v = float(np.mean(vel_list)) if vel_list else 0.0
    mean_sub_e = 0.5 * float(np.mean(sub_masses)) * (mean_sub_v ** 2) if vel_list else 0.0

    return {
        "summary": {
            "payload_key": payload_key,
            "payload_name": payload["name"],
            "category": payload["category"],
            "kill_probability_pct": round((kills / max(1, n_volleys)) * 100.0, 1),
            "mean_hits_on_drone": round(float(np.mean(hits_list)), 1),
            "mean_lethal_hits": round(float(np.mean(lethal_list)), 1),
            "mean_energy_on_drone_j": round(float(np.mean(energy_list)), 1),
            "mean_subprojectile_velocity_mps": round(mean_sub_v, 1),
            "mean_subprojectile_energy_j": round(mean_sub_e, 2),
            "subprojectiles_per_shell": total_subs_per_shell,
            "total_volley_subprojectiles": total_subs_per_shell * n_barrels,
            "target_distance_m": target_distance_m,
            "is_airburst": is_airburst,
            "mean_burst_standoff_m": round(float(np.mean(burst_standoffs)), 2) if burst_standoffs else 0.0,
            "late_burst_behind_target_pct": round((late_bursts / max(1, total_bursts)) * 100.0, 1) if total_bursts else 0.0,
            "ejection_cone_half_angle_deg": ejection_cone_half_angle_deg if is_airburst else 0.0,
            "aspect_mode": aspect_mode,
            "presented_area_cm2": round(float(aspect_geom.get("presented_area_cm2", 0.0)), 1),
            "planform_ratio_pct": round(float(aspect_geom.get("planform_ratio_pct", aspect_geom.get("cross_section_ratio_pct", 100.0))), 1),
            "cross_section_ratio_pct": round(float(aspect_geom.get("cross_section_ratio_pct", 100.0)), 1),
        },
        "drone_spec": drone_spec,
        "aspect_geometry": aspect_geom,
        "sample_burst_points": sample_burst_points,
        "sample_impacts": sample_impacts,
    }


def compare_cuas_payload_families(
    distances_m: list[float] | None = None,
    n_barrels: int = 4,
    drone_model: str = "commercial_quad_medium",
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = 15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    fuze_time_jitter_ms: float = 0.8,
    n_volleys: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark all 7 C-UAS payload families from 20m close range out to 110m standoff."""
    if distances_m is None:
        distances_m = [20.0, 35.0, 50.0, 70.0, 90.0, 110.0]

    curves: dict[str, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []

    for p_idx, (p_key, p_info) in enumerate(CUAS_PAYLOAD_CATALOG.items()):
        p_name = p_info["name"]
        curves[p_name] = {
            "payload_key": p_key,
            "distances_m": list(distances_m),
            "pk_pct": [],
            "sub_energy_j": [],
            "lethal_hits": [],
        }
        for d_m in distances_m:
            sim = simulate_cuas_advanced_payload_mc(
                payload_key=p_key,
                n_barrels=n_barrels,
                target_distance_m=float(d_m),
                drone_model=drone_model,
                aspect_mode=aspect_mode,
                drone_pitch_deg=drone_pitch_deg,
                drone_yaw_deg=drone_yaw_deg,
                drone_roll_deg=drone_roll_deg,
                fuze_time_jitter_ms=fuze_time_jitter_ms,
                n_volleys=n_volleys,
                seed=seed + p_idx * 13,
            )
            sm = sim["summary"]
            curves[p_name]["pk_pct"].append(sm["kill_probability_pct"])
            curves[p_name]["sub_energy_j"].append(sm["mean_subprojectile_energy_j"])
            curves[p_name]["lethal_hits"].append(sm["mean_lethal_hits"])
            table.append({
                "payload": p_name,
                "distance_m": d_m,
                "kill_probability_pct": sm["kill_probability_pct"],
                "mean_lethal_hits": sm["mean_lethal_hits"],
                "sub_velocity_mps": sm["mean_subprojectile_velocity_mps"],
                "sub_energy_j": sm["mean_subprojectile_energy_j"],
            })

    return {
        "distances_m": distances_m,
        "drone_model": drone_model,
        "aspect_mode": aspect_mode,
        "n_barrels": n_barrels,
        "curves": curves,
        "comparison_table": table,
    }


# -------------------------------------------------------------------------
# UPGRADE 4: TURRET RECOIL, GIMBAL SHOCK LOADS & PARETO ARRAY OPTIMIZER
# -------------------------------------------------------------------------

def simulate_turret_recoil_and_gimbal_loads(
    barrels: list[dict[str, Any]],
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
    default_ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    mount_mass_kg: float = 45.0,
    mount_stiffness_n_per_m: float = 120000.0,
    mount_damping_ratio: float = 0.35,
    mount_load_limit_kn: float = 18.0,
    gimbal_trunnion_offset_z_m: float = 0.18,
    gimbal_pitch_inertia_kg_m2: float = 3.2,
    gimbal_yaw_inertia_kg_m2: float = 3.5,
    gimbal_torsional_stiffness_nm_per_rad: float = 8500.0,
    gimbal_damping_ratio: float = 0.30,
    in_bore_duration_ms: float = 3.0,
    recoil_gas_model: str = "legacy_1275",
    saami_gas_factor_fg: float = 1.50,
) -> dict[str, Any]:
    """Solve 2nd-order buffered mount & 2-axis gimbal pitch/yaw ODEs via RK4 under salvo impulse pulses."""
    presets = get_shotgun_presets()
    fire_times_s = compute_balanced_pair_fire_times(barrels, firing_mode=firing_mode, ripple_delay_ms=ripple_delay_ms)
    tau_bore = max(in_bore_duration_ms / 1000.0, 0.0005)

    # Precompute per-barrel impulse & moment arms
    impulses_ns: list[float] = []
    moments_pitch_ns_m: list[float] = []
    moments_yaw_ns_m: list[float] = []
    for b in barrels:
        b_ammo_key = str(b.get("ammunition_preset", default_ammunition_preset))
        ammo = presets.get(b_ammo_key, presets["12G_NO4_BUCK_C_UAS"])
        m_shot_kg = (int(ammo["pellets_per_shot"]) * float(ammo["pellet_mass_g"])) / 1000.0
        m_wad_kg = float(ammo.get("wad_mass_g", 2.79)) / 1000.0
        m_prop_kg = float(ammo.get("propellant_mass_g", 2.2)) / 1000.0
        b_len = float(b.get("barrel_length_in", 24.0))
        v0_b = float(ammo["v0_mps"]) + 2.2 * (b_len - 28.0)

        if str(recoil_gas_model).lower().strip() == "saami_proportional":
            j_i = (m_shot_kg + m_wad_kg + m_prop_kg * max(1.0, float(saami_gas_factor_fg))) * v0_b
        else:
            j_i = (m_shot_kg + m_wad_kg) * v0_b + m_prop_kg * 1275.0
        impulses_ns.append(j_i)
        moments_pitch_ns_m.append(j_i * (gimbal_trunnion_offset_z_m + float(b["z_m"])))
        moments_yaw_ns_m.append(+j_i * float(b["y_m"]))

    total_impulse = float(sum(impulses_ns))
    free_recoil_vel = total_impulse / max(mount_mass_kg, 1.0)
    free_recoil_energy = 0.5 * max(mount_mass_kg, 1.0) * (free_recoil_vel ** 2)

    # Structural ODE constants
    M = max(mount_mass_kg, 1.0)
    Kx = max(mount_stiffness_n_per_m, 1000.0)
    Cx = 2.0 * mount_damping_ratio * math.sqrt(Kx * M)

    Ip = max(gimbal_pitch_inertia_kg_m2, 0.1)
    Iy = max(gimbal_yaw_inertia_kg_m2, 0.1)
    Kth = max(gimbal_torsional_stiffness_nm_per_rad, 100.0)
    Cp = 2.0 * gimbal_damping_ratio * math.sqrt(Kth * Ip)
    Cy = 2.0 * gimbal_damping_ratio * math.sqrt(Kth * Iy)

    t_end = (max(fire_times_s) if fire_times_s else 0.0) + tau_bore + 0.16
    dt = 0.0001
    n_steps = int(math.ceil(t_end / dt)) + 1

    # State vector: [x_r, v_r, theta, omega_p, psi, omega_y]
    state = np.zeros(6, dtype=np.float64)

    def _forcing(t_val: float) -> tuple[float, float, float]:
        f_tot = 0.0
        tp_tot = 0.0
        ty_tot = 0.0
        half_pi_over_tau = math.pi / tau_bore
        for i in range(len(barrels)):
            dt_i = t_val - fire_times_s[i]
            if 0.0 <= dt_i <= tau_bore:
                shape = (math.pi / (2.0 * tau_bore)) * math.sin(half_pi_over_tau * dt_i)
                f_tot += impulses_ns[i] * shape
                tp_tot += moments_pitch_ns_m[i] * shape
                ty_tot += moments_yaw_ns_m[i] * shape
        return f_tot, tp_tot, ty_tot

    def _deriv(s: np.ndarray, f_val: float, tp_val: float, ty_val: float) -> np.ndarray:
        x_r, v_r, th, wp, ps, wy = s
        ax = (f_val - Cx * v_r - Kx * x_r) / M
        ap = (tp_val - Cp * wp - Kth * th) / Ip
        ay = (ty_val - Cy * wy - Kth * ps) / Iy
        return np.array([v_r, ax, wp, ap, wy, ay], dtype=np.float64)

    t_vals = np.linspace(0.0, (n_steps - 1) * dt, n_steps)
    f_raw_arr = np.zeros(n_steps, dtype=np.float64)
    f_trans_arr = np.zeros(n_steps, dtype=np.float64)
    x_mm_arr = np.zeros(n_steps, dtype=np.float64)
    pitch_mrad_arr = np.zeros(n_steps, dtype=np.float64)
    yaw_mrad_arr = np.zeros(n_steps, dtype=np.float64)
    tp_max = 0.0
    ty_max = 0.0

    for k in range(n_steps):
        tk = float(t_vals[k])
        f0, tp0, ty0 = _forcing(tk)
        f_raw_arr[k] = f0 / 1000.0
        f_trans_arr[k] = (Cx * state[1] + Kx * state[0]) / 1000.0
        x_mm_arr[k] = state[0] * 1000.0
        pitch_mrad_arr[k] = state[2] * 1000.0
        yaw_mrad_arr[k] = state[4] * 1000.0
        tp_max = max(tp_max, abs(tp0))
        ty_max = max(ty_max, abs(ty0))

        if k < n_steps - 1:
            fm, tpm, tym = _forcing(tk + 0.5 * dt)
            f1, tp1, ty1 = _forcing(tk + dt)
            k1 = _deriv(state, f0, tp0, ty0)
            k2 = _deriv(state + 0.5 * dt * k1, fm, tpm, tym)
            k3 = _deriv(state + 0.5 * dt * k2, fm, tpm, tym)
            k4 = _deriv(state + dt * k3, f1, tp1, ty1)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Evaluate dynamic muzzle climb & yaw walk experienced by each barrel at shot exit (t_fire + tau_bore)
    barrel_firing_states: list[dict[str, Any]] = []
    for i, b in enumerate(barrels):
        t_exit = fire_times_s[i] + 0.85 * tau_bore
        idx_exit = min(n_steps - 1, max(0, int(round(t_exit / dt))))
        barrel_firing_states.append({
            "barrel_id": b["barrel_id"],
            "ammunition_preset": b.get("ammunition_preset", default_ammunition_preset),
            "y_m": b["y_m"],
            "z_m": b["z_m"],
            "fire_time_ms": round(fire_times_s[i] * 1000.0, 2),
            "impulse_ns": round(impulses_ns[i], 2),
            "pitch_deflection_at_fire_mrad": round(float(pitch_mrad_arr[idx_exit]), 3),
            "yaw_deflection_at_fire_mrad": round(float(yaw_mrad_arr[idx_exit]), 3),
        })

    # Downsample time history for fast plotting (~350 points)
    stride = max(1, n_steps // 350)
    peak_trans_kn = float(np.max(np.abs(f_trans_arr)))
    margin_pct = ((mount_load_limit_kn - peak_trans_kn) / max(mount_load_limit_kn, 0.1)) * 100.0

    from ammo_mc.cuas_ballistics_v2 import compute_saami_recoil_impulse

    ref_ammo = presets.get(default_ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])
    saami_ref = compute_saami_recoil_impulse(
        shot_mass_g=int(ref_ammo["pellets_per_shot"]) * float(ref_ammo["pellet_mass_g"]) * len(barrels),
        wad_mass_g=float(ref_ammo.get("wad_mass_g", 2.79)) * len(barrels),
        propellant_mass_g=float(ref_ammo.get("propellant_mass_g", 2.2)) * len(barrels),
        v0_mps=float(ref_ammo["v0_mps"]),
        gun_mass_kg=max(mount_mass_kg, 1.0),
        gas_factor_fg=float(saami_gas_factor_fg),
    )

    return {
        "summary": {
            "total_impulse_ns": round(total_impulse, 2),
            "free_recoil_velocity_mps": round(free_recoil_vel, 3),
            "free_recoil_energy_j": round(free_recoil_energy, 1),
            "peak_raw_breech_force_kn": round(float(np.max(f_raw_arr)), 2),
            "peak_transmitted_mount_force_kn": round(peak_trans_kn, 2),
            "mount_load_limit_kn": float(mount_load_limit_kn),
            "exceeds_mount_limit": bool(peak_trans_kn > mount_load_limit_kn),
            "structural_safety_margin_pct": round(margin_pct, 1),
            "max_recoil_stroke_mm": round(float(np.max(np.abs(x_mm_arr))), 2),
            "peak_gimbal_pitch_torque_nm": round(tp_max, 1),
            "peak_gimbal_yaw_torque_nm": round(ty_max, 1),
            "max_muzzle_climb_mrad": round(float(np.max(np.abs(pitch_mrad_arr))), 3),
            "max_yaw_deflection_mrad": round(float(np.max(np.abs(yaw_mrad_arr))), 3),
            "salvo_duration_ms": round((max(fire_times_s) if fire_times_s else 0.0) * 1000.0 + in_bore_duration_ms, 1),
            "firing_mode": firing_mode,
            "ripple_delay_ms": ripple_delay_ms,
            "recoil_gas_model": str(recoil_gas_model),
            "saami_gas_factor_fg": float(saami_gas_factor_fg),
            "saami_recoil_reference": saami_ref,
        },
        "time_history": {
            "t_ms": [round(float(x * 1000.0), 2) for x in t_vals[::stride]],
            "raw_breech_force_kn": [round(float(x), 3) for x in f_raw_arr[::stride]],
            "transmitted_force_kn": [round(float(x), 3) for x in f_trans_arr[::stride]],
            "recoil_stroke_mm": [round(float(x), 3) for x in x_mm_arr[::stride]],
            "pitch_climb_mrad": [round(float(x), 3) for x in pitch_mrad_arr[::stride]],
            "yaw_walk_mrad": [round(float(x), 3) for x in yaw_mrad_arr[::stride]],
        },
        "barrel_firing_states": barrel_firing_states,
    }


def sweep_recoil_vs_ripple_delay(
    barrels: list[dict[str, Any]],
    ripple_delays_ms: list[float] | None = None,
    default_ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    mount_mass_kg: float = 45.0,
    mount_stiffness_n_per_m: float = 120000.0,
    mount_damping_ratio: float = 0.35,
    mount_load_limit_kn: float = 18.0,
    gimbal_trunnion_offset_z_m: float = 0.18,
    gimbal_pitch_inertia_kg_m2: float = 3.2,
    gimbal_yaw_inertia_kg_m2: float = 3.5,
) -> dict[str, Any]:
    """Compare peak structural shock force and gimbal muzzle walk across ripple delays for Sequential vs Balanced Pair."""
    if ripple_delays_ms is None:
        ripple_delays_ms = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]

    seq_force: list[float] = []
    pair_force: list[float] = []
    seq_walk: list[float] = []
    pair_walk: list[float] = []

    for d_ms in ripple_delays_ms:
        fmode_seq = "simultaneous_volley" if d_ms <= 0.0 else "sequential_ripple"
        res_seq = simulate_turret_recoil_and_gimbal_loads(
            barrels=barrels,
            firing_mode=fmode_seq,
            ripple_delay_ms=d_ms,
            default_ammunition_preset=default_ammunition_preset,
            mount_mass_kg=mount_mass_kg,
            mount_stiffness_n_per_m=mount_stiffness_n_per_m,
            mount_damping_ratio=mount_damping_ratio,
            mount_load_limit_kn=mount_load_limit_kn,
            gimbal_trunnion_offset_z_m=gimbal_trunnion_offset_z_m,
            gimbal_pitch_inertia_kg_m2=gimbal_pitch_inertia_kg_m2,
            gimbal_yaw_inertia_kg_m2=gimbal_yaw_inertia_kg_m2,
        )
        seq_force.append(res_seq["summary"]["peak_transmitted_mount_force_kn"])
        seq_walk.append(max(res_seq["summary"]["max_muzzle_climb_mrad"], res_seq["summary"]["max_yaw_deflection_mrad"]))

        fmode_pair = "simultaneous_volley" if d_ms <= 0.0 else "balanced_pair_ripple"
        res_pair = simulate_turret_recoil_and_gimbal_loads(
            barrels=barrels,
            firing_mode=fmode_pair,
            ripple_delay_ms=d_ms,
            default_ammunition_preset=default_ammunition_preset,
            mount_mass_kg=mount_mass_kg,
            mount_stiffness_n_per_m=mount_stiffness_n_per_m,
            mount_damping_ratio=mount_damping_ratio,
            mount_load_limit_kn=mount_load_limit_kn,
            gimbal_trunnion_offset_z_m=gimbal_trunnion_offset_z_m,
            gimbal_pitch_inertia_kg_m2=gimbal_pitch_inertia_kg_m2,
            gimbal_yaw_inertia_kg_m2=gimbal_yaw_inertia_kg_m2,
        )
        pair_force.append(res_pair["summary"]["peak_transmitted_mount_force_kn"])
        pair_walk.append(max(res_pair["summary"]["max_muzzle_climb_mrad"], res_pair["summary"]["max_yaw_deflection_mrad"]))

    return {
        "delays_ms": list(ripple_delays_ms),
        "seq_peak_force_kn": seq_force,
        "pair_peak_force_kn": pair_force,
        "seq_max_walk_mrad": seq_walk,
        "pair_max_walk_mrad": pair_walk,
        "mount_load_limit_kn": float(mount_load_limit_kn),
    }


def optimize_antidrone_array_configuration(
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    target_distance_m: float = 35.0,
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 18.0,
    drone_heading_deg: float = 90.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    inner_ammunition_preset: str | None = None,
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    aim_error_mrad: float = 1.5,
    mount_mass_kg: float = 45.0,
    mount_stiffness_n_per_m: float = 120000.0,
    mount_load_limit_kn: float = 18.0,
    gimbal_trunnion_offset_z_m: float = 0.18,
    candidate_layouts: list[str] | None = None,
    n_volleys: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Constrained multi-objective optimizer searching array geometry, splay & ripple timing under mount shock limits."""
    layouts = candidate_layouts or ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"]
    pointing_grid = [("convergent", 0.0), ("divergent_curtain", 4.0), ("staggered_tiling", 0.0)]
    firing_grid = [("simultaneous_volley", 0.0), ("balanced_pair_ripple", 8.0), ("sequential_ripple", 12.0)]

    n_rows = 2
    n_cols = max(2, int(round(n_barrels / 2.0)))

    candidates: list[dict[str, Any]] = []
    idx_c = 0
    for l_type in layouts:
        b_raw = generate_multishotgun_array_layout(
            layout_type=l_type,
            n_barrels=n_barrels,
            mount_radius_m=mount_radius_m,
            n_cols=n_cols,
            n_rows=n_rows,
            default_choke=default_choke,
            default_barrel_length_in=default_barrel_length_in,
            default_ammunition_preset=ammunition_preset,
            inner_ammunition_preset=inner_ammunition_preset,
        )
        for p_mode, sp_ang in pointing_grid:
            for f_mode, r_dt in firing_grid:
                idx_c += 1
                rec = simulate_turret_recoil_and_gimbal_loads(
                    barrels=b_raw,
                    firing_mode=f_mode,
                    ripple_delay_ms=r_dt,
                    default_ammunition_preset=ammunition_preset,
                    mount_mass_kg=mount_mass_kg,
                    mount_stiffness_n_per_m=mount_stiffness_n_per_m,
                    mount_load_limit_kn=mount_load_limit_kn,
                    gimbal_trunnion_offset_z_m=gimbal_trunnion_offset_z_m,
                )
                r_sm = rec["summary"]
                p_offs = [b["pitch_deflection_at_fire_mrad"] for b in rec["barrel_firing_states"]]
                y_offs = [b["yaw_deflection_at_fire_mrad"] for b in rec["barrel_firing_states"]]

                sim = simulate_anti_drone_multishotgun_mc(
                    layout_type=l_type,
                    n_barrels=n_barrels,
                    mount_radius_m=mount_radius_m,
                    n_cols=n_cols,
                    n_rows=n_rows,
                    pointing_mode=p_mode,
                    splay_angle_mrad=sp_ang,
                    firing_mode=f_mode,
                    ripple_delay_ms=r_dt,
                    ammunition_preset=ammunition_preset,
                    inner_ammunition_preset=inner_ammunition_preset,
                    default_choke=default_choke,
                    default_barrel_length_in=default_barrel_length_in,
                    target_distance_m=target_distance_m,
                    drone_model=drone_model,
                    drone_velocity_mps=drone_velocity_mps,
                    drone_heading_deg=drone_heading_deg,
                    aim_error_mrad=aim_error_mrad,
                    n_volleys=n_volleys,
                    seed=seed + idx_c,
                    barrel_pitch_offsets_mrad=p_offs,
                    barrel_yaw_offsets_mrad=y_offs,
                )
                s_sm = sim["summary"]
                peak_f = float(r_sm["peak_transmitted_mount_force_kn"])
                feasible = bool(peak_f <= mount_load_limit_kn)

                candidates.append({
                    "config_id": f"{l_type}_{p_mode}_{f_mode}",
                    "layout_type": l_type,
                    "n_barrels": s_sm["n_barrels"],
                    "pointing_mode": p_mode,
                    "splay_angle_mrad": sp_ang,
                    "firing_mode": f_mode,
                    "ripple_delay_ms": r_dt,
                    "kill_probability_pct": s_sm["kill_probability_pct"],
                    "mean_lethal_hits": s_sm["mean_lethal_hits"],
                    "peak_transmitted_force_kn": peak_f,
                    "max_muzzle_climb_mrad": r_sm["max_muzzle_climb_mrad"],
                    "max_yaw_walk_mrad": r_sm["max_yaw_deflection_mrad"],
                    "feasible": feasible,
                })

    # Sort: Feasible first, highest P_kill, highest lethal hits, lowest peak force
    candidates.sort(
        key=lambda c: (not c["feasible"], -c["kill_probability_pct"], -c["mean_lethal_hits"], c["peak_transmitted_force_kn"])
    )

    feasible_cands = [c for c in candidates if c["feasible"]]
    best_feasible = feasible_cands[0] if feasible_cands else None
    best_unconstrained = max(
        candidates,
        key=lambda c: (c["kill_probability_pct"], c["mean_lethal_hits"], -c["peak_transmitted_force_kn"]),
    )

    # Non-dominated Pareto frontier (minimize peak_transmitted_force_kn, maximize kill_probability_pct)
    pareto_frontier: list[dict[str, Any]] = []
    sorted_by_force = sorted(candidates, key=lambda c: (c["peak_transmitted_force_kn"], -c["kill_probability_pct"]))
    best_pk_so_far = -1.0
    for c in sorted_by_force:
        if c["kill_probability_pct"] > best_pk_so_far:
            pareto_frontier.append(c)
            best_pk_so_far = c["kill_probability_pct"]

    return {
        "mount_load_limit_kn": float(mount_load_limit_kn),
        "best_feasible": best_feasible,
        "best_unconstrained": best_unconstrained,
        "pareto_frontier": pareto_frontier,
        "candidates": candidates,
    }


# -------------------------------------------------------------------------
# UPGRADE 5: ELEVATED-FIRE 3-DOF PELLET FALLOUT & SURFACE DANGER ZONE (SDZ)
# -------------------------------------------------------------------------

def _sphere_drag_coeff(Re: float | np.ndarray, Mach: float | np.ndarray) -> float | np.ndarray:
    """Delegate sphere drag Cd(Re, M) to canonical C1-continuous Morrison/Bailey-Hiatt/Loth model."""
    if np.ndim(Re) == 0 and np.ndim(Mach) == 0:
        return sphere_drag_compressible(float(Re), float(Mach))
    return sphere_drag_compressible_array(np.asarray(Re, dtype=np.float64), np.asarray(Mach, dtype=np.float64))


def compute_barrel_thermal_step_convective_radiative(
    temp_c: float,
    heat_input_j: float,
    dt_s: float,
    barrel_mass_kg: float = 1.4,
    cp_j_kg_k: float = 460.0,
    surface_area_m2: float = 0.045,
    h_conv_w_m2_k: float = 18.0,
    emissivity: float = 0.80,
    ambient_c: float = 20.0,
) -> dict[str, float]:
    """Integrate single-barrel thermal ODE with natural convection + Stefan-Boltzmann T^4 radiation.

    m * c_p * dT/dt = Q_in/dt - h_c * A_s * (T - T_inf) - eps * sigma_SB * A_s * (T_K^4 - T_inf,K^4)
    """
    sigma_sb = 5.670374419e-8
    cap_j_k = max(float(barrel_mass_kg) * float(cp_j_kg_k), 1.0)
    a_s = max(float(surface_area_m2), 1e-4)
    hc = max(float(h_conv_w_m2_k), 0.0)
    eps = max(0.0, min(1.0, float(emissivity)))
    t_inf_k = max(float(ambient_c) + 273.15, 50.0)
    t_k = max(float(temp_c) + 273.15, 50.0)
    dur = max(float(dt_s), 0.0)
    q_rate_w = float(heat_input_j) / dur if dur > 1e-9 else 0.0

    def _dT_dt(tk: float) -> float:
        p_conv = hc * a_s * (tk - t_inf_k)
        p_rad = eps * sigma_sb * a_s * ((tk ** 4) - (t_inf_k ** 4))
        return (q_rate_w - p_conv - p_rad) / cap_j_k

    if dur > 0.0:
        n_sub = max(1, int(math.ceil(dur / 0.25)))
        dt_sub = dur / n_sub
        for _ in range(n_sub):
            k1 = _dT_dt(t_k)
            k2 = _dT_dt(max(50.0, t_k + 0.5 * dt_sub * k1))
            k3 = _dT_dt(max(50.0, t_k + 0.5 * dt_sub * k2))
            k4 = _dT_dt(max(50.0, t_k + dt_sub * k3))
            t_k = max(50.0, t_k + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
    else:
        t_k = max(50.0, t_k + float(heat_input_j) / cap_j_k)

    conv_w = hc * a_s * (t_k - t_inf_k)
    rad_w = eps * sigma_sb * a_s * ((t_k ** 4) - (t_inf_k ** 4))
    new_temp_c = t_k - 273.15
    return {
        "temp_c": round(float(new_temp_c), 4),
        "convective_loss_w": round(float(conv_w), 4),
        "radiative_loss_w": round(float(rad_w), 4),
        "total_cooling_w": round(float(conv_w + rad_w), 4),
        "net_heat_gain_j": round(float((new_temp_c - float(temp_c)) * cap_j_k), 4),
    }


def simulate_cuas_pellet_fallout_hazard(
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    firing_elevation_deg: float = 35.0,
    firing_azimuth_deg: float = 0.0,
    turret_height_m: float = 2.0,
    intercept_slant_range_m: float = 35.0,
    wind_downrange_mps: float = 0.0,
    wind_crossrange_mps: float = 0.0,
    enable_wind_shear: bool = True,
    n_sample_pellets: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Vectorized 3-DoF spherical pellet fallout integrator computing ground Surface Danger Zone (SDZ) & hazard tiers."""
    rng = np.random.default_rng(seed)
    presets = get_shotgun_presets()
    choke_defs = get_choke_definitions()
    ammo = presets.get(ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])

    d_mm = float(ammo["pellet_diameter_mm"])
    d_m = d_mm / 1000.0
    m_g = float(ammo["pellet_mass_g"])
    m_kg = m_g / 1000.0
    area_m2 = math.pi * 0.25 * (d_m ** 2)
    area_mm2 = math.pi * 0.25 * (d_mm ** 2)
    vol_m3 = (math.pi / 6.0) * (d_m ** 3)
    rho_p = (m_kg / max(vol_m3, 1e-12))
    v0_nom = float(ammo["v0_mps"]) + 2.2 * (default_barrel_length_in - 28.0)
    eff_cd_override = float(ammo.get("effective_cd", 0.47))
    is_flechette = eff_cd_override < 0.25

    barrels_raw = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        default_choke=default_choke,
        default_barrel_length_in=default_barrel_length_in,
        default_ammunition_preset=ammunition_preset,
    )
    barrels = configure_array_pointing_vectors(
        barrels_raw,
        pointing_mode=pointing_mode,
        intercept_range_m=intercept_slant_range_m,
        splay_angle_mrad=splay_angle_mrad,
    )

    c_info = choke_defs.get(default_choke, choke_defs["Modified"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_info["benchmark_40yd_30in_pct"])))
    theta_choke = sigma_ref / 36.58

    N = int(n_sample_pellets)
    b_indices = rng.integers(0, len(barrels), size=N)

    pos = np.zeros((N, 3), dtype=np.float64)
    vel = np.zeros((N, 3), dtype=np.float64)
    el0_rad = math.radians(firing_elevation_deg)
    az0_rad = math.radians(firing_azimuth_deg)

    for i in range(N):
        b = barrels[int(b_indices[i])]
        by_local = float(b["y_m"])
        pos[i, 0] = -by_local * math.sin(az0_rad)
        pos[i, 1] = by_local * math.cos(az0_rad)
        pos[i, 2] = float(turret_height_m) + float(b["z_m"])

        # Spherical-coordinate tangent azimuth correction: d_az = cone_angle / cos(el)
        d_el = float(rng.normal(0.0, theta_choke))
        d_az = float(rng.normal(0.0, theta_choke / max(math.cos(el0_rad), 0.15)))
        el_i = el0_rad + (float(b["aim_pitch_mrad"]) / 1000.0) + d_el
        az_i = az0_rad + (float(b["aim_yaw_mrad"]) / (1000.0 * max(math.cos(el0_rad), 0.15))) + d_az
        v_i = v0_nom * (1.0 + float(rng.normal(0.0, 0.015)))

        vel[i, 0] = v_i * math.cos(el_i) * math.cos(az_i)
        vel[i, 1] = v_i * math.cos(el_i) * math.sin(az_i)
        vel[i, 2] = v_i * math.sin(el_i)

    # Deformation drag factor (soft lead pellets deform slightly under ~40k g setback, hard tungsten/steel stay spherical)
    if 10500.0 <= rho_p <= 12200.0:
        f_deform = rng.uniform(1.02, 1.24, size=N)
    else:
        f_deform = rng.uniform(1.00, 1.06, size=N)

    def _accel(p_arr: np.ndarray, v_arr: np.ndarray, f_def_arr: np.ndarray) -> np.ndarray:
        z_m = np.maximum(p_arr[:, 2], 0.0)
        temp_ratio = np.maximum(1.0 - 2.2558e-5 * z_m, 0.1)
        rho_air = 1.225 * (temp_ratio ** 4.2559)
        a_sound = 340.29 * np.sqrt(temp_ratio)

        shear = (np.maximum(z_m, 2.0) / 10.0) ** 0.143 if enable_wind_shear else np.ones_like(z_m)
        w_world_x = wind_downrange_mps * math.cos(az0_rad) - wind_crossrange_mps * math.sin(az0_rad)
        w_world_y = wind_downrange_mps * math.sin(az0_rad) + wind_crossrange_mps * math.cos(az0_rad)
        w_vec = np.column_stack([w_world_x * shear, w_world_y * shear, np.zeros_like(z_m)])
        v_rel = v_arr - w_vec
        v_mag = np.linalg.norm(v_rel, axis=1)
        Re = np.maximum(v_mag * d_m / 1.48e-5, 1.0)
        Mach = v_mag / a_sound
        cd_val = np.full_like(v_mag, eff_cd_override) if is_flechette else _sphere_drag_coeff(Re, Mach)
        cd_eff = cd_val * f_def_arr

        drag_factor = -(0.5 * rho_air * cd_eff * area_m2 / max(m_kg, 1e-6)) * v_mag
        a_out = drag_factor[:, None] * v_rel
        a_out[:, 2] -= 9.80665
        return a_out

    dt = 0.025
    max_steps = 4800  # 120 s max flight (allows steep 75+ deg low-drag flechettes to land)
    active = np.ones(N, dtype=bool)
    apogee = pos[:, 2].copy()
    t_impact = np.zeros(N, dtype=np.float64)
    x_impact = np.zeros(N, dtype=np.float64)
    y_impact = np.zeros(N, dtype=np.float64)
    v_impact = np.zeros(N, dtype=np.float64)

    # Sample ~35 trajectories for 3D arc visualization
    n_traj_plot = min(35, N)
    plot_stride = max(1, N // n_traj_plot)
    plot_ids = np.arange(0, N, plot_stride)[:n_traj_plot]
    traj_hist: dict[int, tuple[list[float], list[float], list[float]]] = {
        int(pi): ([round(float(pos[pi, 0]), 2)], [round(float(pos[pi, 1]), 2)], [round(float(pos[pi, 2]), 2)])
        for pi in plot_ids
    }

    for step in range(max_steps):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        p_act = pos[idx]
        v_act = vel[idx]
        f_def_act = f_deform[idx]

        k1_v = _accel(p_act, v_act, f_def_act)
        k1_p = v_act
        k2_v = _accel(p_act + 0.5 * dt * k1_p, v_act + 0.5 * dt * k1_v, f_def_act)
        k2_p = v_act + 0.5 * dt * k1_v
        k3_v = _accel(p_act + 0.5 * dt * k2_p, v_act + 0.5 * dt * k2_v, f_def_act)
        k3_p = v_act + 0.5 * dt * k2_v
        k4_v = _accel(p_act + dt * k3_p, v_act + dt * k3_v, f_def_act)
        k4_p = v_act + dt * k3_v

        p_next = p_act + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        v_next = v_act + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        apogee[idx] = np.maximum(apogee[idx], p_next[:, 2])

        # Record 3D polyline samples every 3 steps (~0.075 s)
        if step % 3 == 0:
            for pi_idx, global_i in enumerate(idx):
                gi = int(global_i)
                if gi in traj_hist:
                    traj_hist[gi][0].append(round(float(p_next[pi_idx, 0]), 2))
                    traj_hist[gi][1].append(round(float(p_next[pi_idx, 1]), 2))
                    traj_hist[gi][2].append(round(float(max(p_next[pi_idx, 2], 0.0)), 2))

        hit_ground = p_next[:, 2] <= 0.0
        if np.any(hit_ground):
            hg_local = np.where(hit_ground)[0]
            for li in hg_local:
                gi = int(idx[li])
                z_prev = float(p_act[li, 2])
                z_curr = float(p_next[li, 2])
                frac = z_prev / max(z_prev - z_curr, 1e-6)
                frac = max(0.0, min(1.0, frac))

                x_imp = float(p_act[li, 0] + frac * (p_next[li, 0] - p_act[li, 0]))
                y_imp = float(p_act[li, 1] + frac * (p_next[li, 1] - p_act[li, 1]))
                v_imp_vec = v_act[li] + frac * (v_next[li] - v_act[li])

                x_impact[gi] = x_imp
                y_impact[gi] = y_imp
                v_impact[gi] = float(np.linalg.norm(v_imp_vec))
                t_impact[gi] = (step + frac) * dt
                active[gi] = False

                if gi in traj_hist:
                    traj_hist[gi][0].append(round(x_imp, 2))
                    traj_hist[gi][1].append(round(y_imp, 2))
                    traj_hist[gi][2].append(0.0)

        still_up = idx[~hit_ground]
        if len(still_up) > 0:
            pos[still_up] = p_next[~hit_ground]
            vel[still_up] = v_next[~hit_ground]

    ranges_m = np.hypot(x_impact, y_impact)
    energies_j = 0.5 * m_kg * (v_impact ** 2)
    spec_energies = energies_j / max(area_mm2, 1e-6)

    hazard_classes: list[str] = []
    eye_cnt = 0
    skin_cnt = 0
    iatg_cnt = 0
    for ke, se in zip(energies_j, spec_energies):
        if ke >= 79.0:
            hazard_classes.append("iatg_lethal")
            iatg_cnt += 1
            skin_cnt += 1
            eye_cnt += 1
        elif ke >= 20.0:
            hazard_classes.append("incapacitating")
            skin_cnt += 1
            eye_cnt += 1
        elif se >= 0.10:
            hazard_classes.append("skin_penetration")
            skin_cnt += 1
            eye_cnt += 1
        elif se >= 0.06 or ke >= 1.5:
            hazard_classes.append("eye_hazard")
            eye_cnt += 1
        else:
            hazard_classes.append("safe_low_energy")

    # Footprint ellipse area estimate in downrange/crossrange frame rotated by firing_azimuth_deg
    u_dr_x, u_dr_y = math.cos(az0_rad), math.sin(az0_rad)
    u_cr_x, u_cr_y = -math.sin(az0_rad), math.cos(az0_rad)
    downrange_imp = x_impact * u_dr_x + y_impact * u_dr_y
    crossrange_imp = x_impact * u_cr_x + y_impact * u_cr_y
    x_span = max(float(np.percentile(downrange_imp, 99) - np.percentile(downrange_imp, 1)), 1.0)
    y_spread = max(float(np.percentile(crossrange_imp, 97.5) - np.percentile(crossrange_imp, 2.5)), 1.0)
    sdz_area_m2 = 0.25 * math.pi * x_span * y_spread

    trajectories_3d = [
        {
            "x": traj_hist[gi][0],
            "y": traj_hist[gi][1],
            "z": traj_hist[gi][2],
            "hazard_class": hazard_classes[gi],
        }
        for gi in sorted(traj_hist.keys())
    ]

    return {
        "summary": {
            "ammunition_preset": ammunition_preset,
            "ammunition_name": ammo["name"],
            "firing_elevation_deg": firing_elevation_deg,
            "firing_azimuth_deg": firing_azimuth_deg,
            "turret_height_m": turret_height_m,
            "pellet_density_kg_m3": round(rho_p, 0),
            "min_fallout_range_m": round(float(np.min(ranges_m)), 1),
            "mean_fallout_range_m": round(float(np.mean(ranges_m)), 1),
            "max_fallout_range_99_m": round(float(np.percentile(ranges_m, 99)), 1),
            "crossrange_spread_95_m": round(y_spread, 1),
            "sdz_area_m2": round(sdz_area_m2, 0),
            "mean_apogee_m": round(float(np.mean(apogee)), 1),
            "max_apogee_m": round(float(np.max(apogee)), 1),
            "mean_flight_time_s": round(float(np.mean(t_impact)), 2),
            "mean_terminal_velocity_mps": round(float(np.mean(v_impact)), 1),
            "mean_impact_energy_j": round(float(np.mean(energies_j)), 2),
            "max_impact_energy_j": round(float(np.max(energies_j)), 2),
            "mean_specific_energy_j_mm2": round(float(np.mean(spec_energies)), 3),
            "eye_hazard_pct": round((eye_cnt / N) * 100.0, 1),
            "skin_hazard_pct": round((skin_cnt / N) * 100.0, 1),
            "iatg_hazard_pct": round((iatg_cnt / N) * 100.0, 1),
        },
        "ground_impacts": {
            "x_m": [round(float(x), 2) for x in x_impact],
            "y_m": [round(float(x), 2) for x in y_impact],
            "range_m": [round(float(x), 2) for x in ranges_m],
            "v_impact_mps": [round(float(x), 1) for x in v_impact],
            "energy_j": [round(float(x), 2) for x in energies_j],
            "specific_energy_j_mm2": [round(float(x), 3) for x in spec_energies],
            "hazard_class": hazard_classes,
            "flight_time_s": [round(float(x), 2) for x in t_impact],
        },
        "trajectories_3d": trajectories_3d,
    }


def compare_cuas_fallout_across_loads(
    firing_elevation_deg: float = 35.0,
    wind_downrange_mps: float = 0.0,
    wind_crossrange_mps: float = 0.0,
    default_choke: str = "Modified",
    n_sample_pellets: int = 180,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare downrange fallout footprint & terminal injury risk across 5 shotgun C-UAS ammunition types."""
    loads = [
        "12G_00_BUCK_TACTICAL",
        "12G_NO4_BUCK_C_UAS",
        "12G_TSS_NO2_C_UAS",
        "12G_T_SHOT_STEEL_C_UAS",
        "12G_NO75_BIRD_TRAP",
    ]
    records: list[dict[str, Any]] = []
    for l_idx, key in enumerate(loads):
        res = simulate_cuas_pellet_fallout_hazard(
            ammunition_preset=key,
            default_choke=default_choke,
            firing_elevation_deg=firing_elevation_deg,
            wind_downrange_mps=wind_downrange_mps,
            wind_crossrange_mps=wind_crossrange_mps,
            n_sample_pellets=n_sample_pellets,
            seed=seed + l_idx * 7,
        )
        sm = res["summary"]
        records.append({
            "ammunition_preset": key,
            "ammunition_name": sm["ammunition_name"],
            "min_range_m": sm["min_fallout_range_m"],
            "mean_range_m": sm["mean_fallout_range_m"],
            "max_range_99_m": sm["max_fallout_range_99_m"],
            "max_apogee_m": sm["max_apogee_m"],
            "mean_terminal_v_mps": sm["mean_terminal_velocity_mps"],
            "mean_energy_j": sm["mean_impact_energy_j"],
            "eye_hazard_pct": sm["eye_hazard_pct"],
            "skin_hazard_pct": sm["skin_hazard_pct"],
            "iatg_hazard_pct": sm["iatg_hazard_pct"],
        })

    return {
        "firing_elevation_deg": firing_elevation_deg,
        "wind_downrange_mps": wind_downrange_mps,
        "records": records,
    }
