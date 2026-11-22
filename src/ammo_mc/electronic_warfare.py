"""Electronic Warfare (EW), GPS Jamming Resilience & Semi-Active Laser (SAL) Seeker Guidance.

Implements:
1. Semi-Active Laser (SAL) quadrant photodiode seeker kinematics and gimbal field-of-view limits.
2. Beer-Lambert atmospheric aerosol/smoke obscurant transmission and laser lock loss.
3. Standoff Electronic Warfare (EW) GPS noise jammer power model and Jamming-to-Signal (J/S) ratio.
4. GPS satellite denial event and subsequent MEMS IMU inertial navigation dead-reckoning drift growth.
5. Comparative terminal impact CEP: GPS-aided vs. Jammed INS dead-reckoning vs. Laser homing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def simulate_sal_seeker_tracking(
    target_pos_m: tuple[float, float, float] = (14000.0, 0.0, 0.0),
    designator_pos_m: tuple[float, float, float] = (12500.0, -1200.0, 150.0),
    laser_jitter_sigma_m: float = 0.35,
    smoke_extinction_coeff_per_km: float = 0.15,
    seeker_fov_half_angle_deg: float = 18.0,
    seeker_acquisition_range_m: float = 4500.0,
    trajectory_pts: list[list[float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates terminal Semi-Active Laser (SAL) quadrant photodiode seeker homing.

    Computes:
    - Line-of-sight laser return power using Beer-Lambert aerosol/smoke attenuation:
      T = exp(-gamma * R_slant)
    - Seeker acquisition lock range and field-of-view (FOV) angle checks.
    - Seeker steering corrections to guide projectile towards jittered laser spot.
    """
    rng = np.random.default_rng(seed)
    tx, ty, tz = target_pos_m
    dx, dy, dz = designator_pos_m

    # Line of sight slant range from designator to target
    r_des_tgt_km = math.hypot(tx - dx, ty - dy, tz - dz) / 1000.0
    trans_des = math.exp(-smoke_extinction_coeff_per_km * r_des_tgt_km)

    # If nominal trajectory not provided, synthesize typical terminal trajectory arc
    if trajectory_pts is None or len(trajectory_pts) < 5:
        trajectory_pts = [
            [tx - 3500.0 + i * 350.0, -40.0 + i * 4.0, 1800.0 - i * 180.0]
            for i in range(11)
        ]

    seeker_locked = False
    lock_point_idx = -1
    lock_slant_range_m = 0.0

    corrected_traj: list[list[float]] = []

    # Current jittered laser designation spot
    spot_x = tx + float(rng.normal(0.0, laser_jitter_sigma_m))
    spot_y = ty + float(rng.normal(0.0, laser_jitter_sigma_m))
    spot_z = tz

    for idx, pt in enumerate(trajectory_pts):
        px, py, pz = pt[0], pt[1], pt[2]
        dist_to_tgt = math.hypot(spot_x - px, spot_y - py, spot_z - pz)

        # Seeker look angle relative to velocity vector
        if idx < len(trajectory_pts) - 1:
            vx = trajectory_pts[idx + 1][0] - px
            vy = trajectory_pts[idx + 1][1] - py
            vz = trajectory_pts[idx + 1][2] - pz
        else:
            vx = spot_x - px
            vy = spot_y - py
            vz = spot_z - pz

        v_mag = max(math.hypot(vx, vy, vz), 1e-3)
        los_x = (spot_x - px) / max(dist_to_tgt, 1e-3)
        los_y = (spot_y - py) / max(dist_to_tgt, 1e-3)
        los_z = (spot_z - pz) / max(dist_to_tgt, 1e-3)

        cos_look = (vx * los_x + vy * los_y + vz * los_z) / v_mag
        look_angle_deg = math.degrees(math.acos(float(np.clip(cos_look, -1.0, 1.0))))

        # Check optical acquisition conditions
        r_proj_tgt_km = dist_to_tgt / 1000.0
        trans_proj = math.exp(-smoke_extinction_coeff_per_km * r_proj_tgt_km)
        total_optical_trans = trans_des * trans_proj

        can_acquire = (
            dist_to_tgt <= seeker_acquisition_range_m
            and look_angle_deg <= seeker_fov_half_angle_deg
            and total_optical_trans >= 0.05
        )

        if can_acquire and not seeker_locked:
            seeker_locked = True
            lock_point_idx = idx
            lock_slant_range_m = dist_to_tgt

        # Apply terminal homing pull when locked
        if seeker_locked and idx >= lock_point_idx:
            pull_factor = min(1.0, (idx - lock_point_idx + 1) / max(1, len(trajectory_pts) - lock_point_idx))
            c_px = px + (spot_x - px) * 0.45 * pull_factor
            c_py = py + (spot_y - py) * 0.45 * pull_factor
            corrected_traj.append([c_px, c_py, pz])
        else:
            corrected_traj.append([px, py, pz])

    # Final impact coordinates
    impact_x = corrected_traj[-1][0] if seeker_locked else trajectory_pts[-1][0]
    impact_y = corrected_traj[-1][1] if seeker_locked else trajectory_pts[-1][1]
    miss_dist_m = math.hypot(impact_x - tx, impact_y - ty)

    return {
        "seeker_locked": seeker_locked,
        "optical_transmission": round(float(trans_des), 3),
        "lock_range_m": round(lock_slant_range_m, 1),
        "lock_point_index": lock_point_idx,
        "laser_spot_pos_m": [round(spot_x, 2), round(spot_y, 2), round(spot_z, 2)],
        "miss_distance_m": round(miss_dist_m, 2),
        "terminal_guidance_active": seeker_locked,
        "trajectory": corrected_traj,
    }


def simulate_ins_gps_guidance_under_jamming(
    nominal_range_m: float = 18000.0,
    jammer_pos_m: tuple[float, float, float] = (15000.0, 1800.0, 80.0),
    jammer_power_kw: float = 8.0,
    jamming_carrier_ghz: float = 1.57542,  # GPS L1 band
    gps_denial_js_threshold_db: float = 42.0,
    imu_gyro_drift_deg_per_hr: float = 1.5,
    imu_accel_bias_mg: float = 2.0,
    trajectory_pts: list[list[float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates GPS-aided INS precision guidance under Electronic Warfare (EW) jamming.

    Models:
    - Free-space jamming power propagation and Jamming-to-Signal (J/S) ratio:
      J/S = P_j - L_j - P_s
    - GPS carrier tracking loop unlock when J/S exceeds threshold.
    - Post-denial pure inertial navigation dead-reckoning drift growth:
      delta_pos(t) ~ 0.5 * bias_acc * t_drift^2 + (1/6) * g * bias_gyro * t_drift^3.
    """
    rng = np.random.default_rng(seed)
    jx, jy, jz = jammer_pos_m

    # Synthesize nominal high-altitude guided glide trajectory if not provided
    if trajectory_pts is None or len(trajectory_pts) < 10:
        trajectory_pts = []
        n_steps = 30
        for i in range(n_steps):
            frac = i / (n_steps - 1)
            x = frac * nominal_range_m
            # Glide trajectory with high vertex
            z = max(0.0, 4800.0 * math.sin(math.pi * frac) + 1200.0 * (1.0 - frac))
            y = 0.0
            trajectory_pts.append([x, y, z])

    gps_denied = False
    denial_idx = -1
    denial_range_to_target_m = 0.0
    denial_time_s = 0.0

    actual_guided_traj: list[list[float]] = []
    js_ratios_db: list[float] = []

    # Constants
    lambda_gps = 0.299792458 / jamming_carrier_ghz  # m
    p_signal_nominal_dbw = -158.5  # Typical received GPS satellite power (dBW)
    dt_step = 2.0  # seconds per segment

    t_elapsed = 0.0
    drift_x = 0.0
    drift_y = 0.0

    for idx, pt in enumerate(trajectory_pts):
        px, py, pz = pt[0], pt[1], pt[2]
        dist_to_jammer = math.hypot(px - jx, py - jy, pz - jz)

        # Free space path loss to jammer
        # P_rx_jam = P_tx * (lambda / (4 * pi * R))^2
        path_loss_db = 20.0 * math.log10(max(dist_to_jammer, 10.0)) + 20.0 * math.log10(jamming_carrier_ghz) + 32.44
        p_tx_jam_dbw = 10.0 * math.log10(max(jammer_power_kw * 1000.0, 1.0))
        p_jam_rx_dbw = p_tx_jam_dbw - path_loss_db
        js_ratio_db = p_jam_rx_dbw - p_signal_nominal_dbw
        js_ratios_db.append(round(js_ratio_db, 1))

        if js_ratio_db >= gps_denial_js_threshold_db and not gps_denied:
            gps_denied = True
            denial_idx = idx
            denial_range_to_target_m = nominal_range_m - px
            denial_time_s = t_elapsed

        if gps_denied:
            # Inertial drift growth
            t_drift = t_elapsed - denial_time_s
            # Accel bias drift (converted to m/s^2)
            a_bias_mps2 = (imu_accel_bias_mg * 1e-3) * 9.80665
            # Gyro drift rate (converted to rad/s)
            g_drift_rad_s = math.radians(imu_gyro_drift_deg_per_hr / 3600.0)

            # Quadratic and cubic navigation drift
            pos_drift_mag = 0.5 * a_bias_mps2 * (t_drift**2) + (1.0 / 6.0) * 9.80665 * g_drift_rad_s * (t_drift**3)
            # Stochastic angle walk
            drift_x += float(rng.normal(0.0, 0.4 * pos_drift_mag * 0.1))
            drift_y += float(rng.normal(0.0, 0.7 * pos_drift_mag * 0.15))

            actual_guided_traj.append([px + drift_x, py + drift_y, pz])
        else:
            # GPS-corrected path (tight precision)
            gps_jitter_x = float(rng.normal(0.0, 1.2))
            gps_jitter_y = float(rng.normal(0.0, 1.2))
            actual_guided_traj.append([px + gps_jitter_x, py + gps_jitter_y, pz])

        t_elapsed += dt_step

    final_miss_dist_m = math.hypot(actual_guided_traj[-1][0] - nominal_range_m, actual_guided_traj[-1][1])

    return {
        "gps_denied": gps_denied,
        "max_js_ratio_db": round(float(np.max(js_ratios_db)), 1),
        "gps_denial_threshold_db": gps_denial_js_threshold_db,
        "denial_point_index": denial_idx,
        "standoff_distance_at_denial_m": round(denial_range_to_target_m, 1),
        "pure_inertial_flight_duration_s": round(max(0.0, t_elapsed - denial_time_s), 1),
        "terminal_miss_distance_m": round(final_miss_dist_m, 2),
        "cep_category": "Precision (<3m)" if final_miss_dist_m <= 3.0 else ("Near-Precision (<15m)" if final_miss_dist_m <= 15.0 else "Degraded Dead-Reckoning (>15m)"),
        "trajectory": actual_guided_traj,
        "js_ratios_db": js_ratios_db,
        "jammer_pos_m": list(jammer_pos_m),
    }
