"""Counter-Battery Radar & Trajectory Reconstruction (NATO STANAG 4114 / AN/TPQ-53).

Implements:
1. Physical optics projectile Radar Cross Section (RCS) model.
2. Weapon-locating radar detection volume (SNR vs range, beam elevation/azimuth search fence).
3. Noisy radar track-point measurement generator with range and angular jitter.
4. Inverse exterior ballistics state estimator back-propagating tracked projectile states
   through atmospheric drag equations to compute the enemy Point of Origin (POO) and error ellipse.
5. Forward trajectory extrapolation computing the predicted Point of Impact (POI)
   and early-warning impact countdown timer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def calculate_projectile_rcs(
    calibre_m: float = 0.155,
    length_m: float = 0.85,
    aspect_angle_deg: float = 25.0,
    frequency_ghz: float = 3.2,  # S-band / C-band artillery radar
) -> float:
    """Calculates effective Radar Cross Section (RCS, m^2) of an ogive-cylinder shell.

    Uses high-frequency physical optics (PO) formulation for a body of revolution:
    - Nose-on aspect: Tip diffraction and nose cone specular return (small RCS, ~0.005 - 0.02 m^2).
    - Broadside aspect: Specular cylinder return (peak RCS, ~0.2 - 0.8 m^2).
    - Base aspect: Flat base specular return (~0.05 - 0.15 m^2).
    """
    wavelength_m = 0.299792458 / max(frequency_ghz, 0.1)
    radius_m = 0.5 * calibre_m
    aspect_rad = math.radians(abs(aspect_angle_deg) % 180.0)

    # Cylinder broadside peak
    k = 2.0 * math.pi / wavelength_m
    rcs_broadside = (k * radius_m * (length_m**2)) / 2.0

    # Nose-on Rayleigh-Gans / cone tip return
    rcs_nose = (math.pi * (radius_m**2)) * (math.tan(math.radians(12.0)) ** 4)

    # Base return
    rcs_base = math.pi * (radius_m**2)

    # Smooth angular transition with sin(aspect) weighting
    sin_asp = math.sin(aspect_rad)
    cos_asp = math.cos(aspect_rad)

    rcs_val = rcs_nose * (cos_asp**4) + rcs_broadside * (sin_asp**8) + rcs_base * (cos_asp**2) * 0.25
    return float(np.clip(rcs_val, 0.003, 1.5))


def simulate_radar_tracking(
    trajectory_x_m: list[float],
    trajectory_y_m: list[float],
    trajectory_z_m: list[float],
    time_s: list[float],
    radar_pos_m: tuple[float, float, float] = (4000.0, -2500.0, 50.0),
    radar_range_max_m: float = 32000.0,
    scan_elevation_min_deg: float = 2.0,
    scan_elevation_max_deg: float = 40.0,
    p_detection_nominal: float = 0.94,
    sigma_range_m: float = 12.0,
    sigma_angle_mils: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates realistic detection and tracking of a projectile in flight by a weapon-locating radar.

    Filters projectile trajectory points through the radar detection fence (range and elevation gates),
    applies radar detection probability, and corrupts observed coordinates with Gaussian measurement noise.
    """
    rng = np.random.default_rng(seed)
    rx, ry, rz = radar_pos_m
    mils_to_rad = 0.05625 * (math.pi / 180.0)

    detected_track_points: list[dict[str, Any]] = []

    for i in range(len(time_s)):
        px = trajectory_x_m[i]
        py = trajectory_y_m[i]
        pz = trajectory_z_m[i]
        t = time_s[i]

        # Vector from radar to projectile
        dx = px - rx
        dy = py - ry
        dz = pz - rz
        r_dist = math.hypot(dx, dy, dz)
        horiz_dist = math.hypot(dx, dy)
        elev_deg = math.degrees(math.atan2(dz, max(horiz_dist, 1e-3)))
        az_deg = math.degrees(math.atan2(dy, dx))

        # Check radar coverage volume
        in_range = 1000.0 <= r_dist <= radar_range_max_m
        in_beam = scan_elevation_min_deg <= elev_deg <= scan_elevation_max_deg

        if in_range and in_beam:
            if rng.uniform(0.0, 1.0) <= p_detection_nominal:
                # Add range and angular noise
                r_noisy = r_dist + float(rng.normal(0.0, sigma_range_m))
                az_noisy_rad = math.radians(az_deg) + float(rng.normal(0.0, sigma_angle_mils * mils_to_rad))
                el_noisy_rad = math.radians(elev_deg) + float(rng.normal(0.0, sigma_angle_mils * mils_to_rad))

                # Reconstruct noisy 3D point in world Cartesian space
                obs_x = rx + r_noisy * math.cos(el_noisy_rad) * math.cos(az_noisy_rad)
                obs_y = ry + r_noisy * math.cos(el_noisy_rad) * math.sin(az_noisy_rad)
                obs_z = rz + r_noisy * math.sin(el_noisy_rad)

                detected_track_points.append({
                    "t_s": round(t, 3),
                    "true_x": round(px, 1),
                    "true_y": round(py, 1),
                    "true_z": round(pz, 1),
                    "obs_x": round(obs_x, 1),
                    "obs_y": round(obs_y, 1),
                    "obs_z": round(obs_z, 1),
                    "slant_range_m": round(r_noisy, 1),
                    "elevation_deg": round(math.degrees(el_noisy_rad), 2),
                    "azimuth_deg": round(math.degrees(az_noisy_rad), 2),
                })

    return {
        "radar_pos_m": list(radar_pos_m),
        "total_detections": len(detected_track_points),
        "track_duration_s": (
            round(detected_track_points[-1]["t_s"] - detected_track_points[0]["t_s"], 2)
            if len(detected_track_points) >= 2
            else 0.0
        ),
        "track_points": detected_track_points,
    }


def reconstruct_point_of_origin_and_impact(
    track_points: list[dict[str, Any]],
    gun_elev_m: float = 0.0,
    target_elev_m: float = 0.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    v0_nominal_mps: float = 827.0,
) -> dict[str, Any]:
    """Solves inverse exterior ballistics to estimate enemy Point of Origin (POO) and Point of Impact (POI).

    Fits a second-order polynomial state vector to noisy radar observations to extract
    instantaneous position and velocity at mid-track:
    x(t) = a0 + a1*t + a2*t^2, y(t) = b0 + b1*t + b2*t^2, z(t) = c0 + c1*t + c2*t^2.
    Then back-integrates backwards in time to ground level (z = gun_elev_m) to locate the firing weapon,
    and integrates forward to z = target_elev_m to calculate warning time and expected POI.
    """
    if len(track_points) < 4:
        return {
            "success": False,
            "error_msg": "Insufficient radar track points (minimum 4 required for ballistic arc reconstruction)",
            "poo_x_m": 0.0,
            "poo_y_m": 0.0,
            "poi_x_m": 0.0,
            "poi_y_m": 0.0,
        }

    t_arr = np.array([p["t_s"] for p in track_points], dtype=np.float64)
    x_arr = np.array([p["obs_x"] for p in track_points], dtype=np.float64)
    y_arr = np.array([p["obs_y"] for p in track_points], dtype=np.float64)
    z_arr = np.array([p["obs_z"] for p in track_points], dtype=np.float64)

    # Fit quadratic polynomials for smoothed state estimation
    poly_x = np.polyfit(t_arr, x_arr, 2)
    poly_y = np.polyfit(t_arr, y_arr, 2)
    poly_z = np.polyfit(t_arr, z_arr, 2)

    # Mid-track evaluation time
    t_mid = float(np.mean(t_arr))
    r_mid = np.array([np.polyval(poly_x, t_mid), np.polyval(poly_y, t_mid), np.polyval(poly_z, t_mid)])
    v_mid = np.array([
        2.0 * poly_x[0] * t_mid + poly_x[1],
        2.0 * poly_y[0] * t_mid + poly_y[1],
        2.0 * poly_z[0] * t_mid + poly_z[1],
    ])

    from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch

    atm_env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=calibre_m, fidelity="L1", dt=0.08)

    # 1. Back-integrate backwards in time to find Point of Origin (POO)
    dt_rev = 0.05
    t_curr = t_mid
    pos_curr = r_mid.copy()
    vel_curr = v_mid.copy()
    area = 0.25 * math.pi * (calibre_m**2)

    poo_traj_x = [float(pos_curr[0])]
    poo_traj_y = [float(pos_curr[1])]
    poo_traj_z = [float(pos_curr[2])]

    # Backward stepping: r(t - dt) = r(t) - v*dt, v(t - dt) = v(t) - a*dt
    while pos_curr[2] > gun_elev_m and t_curr > -5.0:
        v_mag = math.hypot(vel_curr[0], vel_curr[1], vel_curr[2])
        rho = 1.225 * math.exp(-max(pos_curr[2], 0.0) / 9300.0)
        drag_acc = 0.5 * rho * area * 0.28 / mass_kg * v_mag
        ax = -drag_acc * vel_curr[0]
        ay = -drag_acc * vel_curr[1]
        az = -drag_acc * vel_curr[2] - 9.80665

        pos_curr -= vel_curr * dt_rev
        vel_curr -= np.array([ax, ay, az]) * dt_rev
        t_curr -= dt_rev

        poo_traj_x.append(float(pos_curr[0]))
        poo_traj_y.append(float(pos_curr[1]))
        poo_traj_z.append(float(pos_curr[2]))

    poo_x = float(pos_curr[0])
    poo_y = float(pos_curr[1])
    launch_time_s = max(0.0, float(t_curr))
    muzzle_v_est = float(np.linalg.norm(vel_curr))

    # 2. Forward-integrate forward in time to find Point of Impact (POI)
    t_fwd = t_mid
    pos_fwd = r_mid.copy()
    vel_fwd = v_mid.copy()

    poi_traj_x = [float(pos_fwd[0])]
    poi_traj_y = [float(pos_fwd[1])]
    poi_traj_z = [float(pos_fwd[2])]

    while pos_fwd[2] > target_elev_m and t_fwd < 200.0:
        v_mag = math.hypot(vel_fwd[0], vel_fwd[1], vel_fwd[2])
        rho = 1.225 * math.exp(-max(pos_fwd[2], 0.0) / 9300.0)
        drag_acc = 0.5 * rho * area * 0.28 / mass_kg * v_mag
        ax = -drag_acc * vel_fwd[0]
        ay = -drag_acc * vel_fwd[1]
        az = -drag_acc * vel_fwd[2] - 9.80665

        pos_fwd += vel_fwd * dt_rev
        vel_fwd += np.array([ax, ay, az]) * dt_rev
        t_fwd += dt_rev

        poi_traj_x.append(float(pos_fwd[0]))
        poi_traj_y.append(float(pos_fwd[1]))
        poi_traj_z.append(float(pos_fwd[2]))

    poi_x = float(pos_fwd[0])
    poi_y = float(pos_fwd[1])
    impact_time_s = float(t_fwd)
    early_warning_time_s = max(0.0, impact_time_s - t_arr[-1])

    # 3. Error covariance estimates (POO / POI semi-major and semi-minor axes)
    # Residual RMSE from polynomial fit
    residuals_x = x_arr - np.polyval(poly_x, t_arr)
    residuals_y = y_arr - np.polyval(poly_y, t_arr)
    rmse = math.hypot(float(np.std(residuals_x)), float(np.std(residuals_y)))
    # Error propagation multiplier (extrapolation from track arc to ground)
    track_span_s = max(float(t_arr[-1] - t_arr[0]), 0.5)
    extrap_factor = max(1.0, (impact_time_s - launch_time_s) / track_span_s)

    poo_semi_major_m = round(max(25.0, rmse * extrap_factor * 1.8), 1)
    poo_semi_minor_m = round(max(15.0, rmse * extrap_factor * 0.9), 1)
    poi_semi_major_m = round(max(30.0, rmse * extrap_factor * 2.2), 1)
    poi_semi_minor_m = round(max(18.0, rmse * extrap_factor * 1.1), 1)

    return {
        "success": True,
        "n_track_points_used": len(track_points),
        "track_start_time_s": round(float(t_arr[0]), 2),
        "track_end_time_s": round(float(t_arr[-1]), 2),
        "poo_x_m": round(poo_x, 1),
        "poo_y_m": round(poo_y, 1),
        "poo_z_m": round(gun_elev_m, 1),
        "estimated_v0_mps": round(muzzle_v_est, 1),
        "estimated_launch_time_s": round(launch_time_s, 2),
        "poi_x_m": round(poi_x, 1),
        "poi_y_m": round(poi_y, 1),
        "poi_z_m": round(target_elev_m, 1),
        "predicted_impact_time_s": round(impact_time_s, 2),
        "early_warning_countdown_s": round(early_warning_time_s, 1),
        "poo_error_ellipse": {
            "semi_major_m": poo_semi_major_m,
            "semi_minor_m": poo_semi_minor_m,
            "azimuth_deg": round(math.degrees(math.atan2(v_mid[1], v_mid[0])), 1),
        },
        "poi_error_ellipse": {
            "semi_major_m": poi_semi_major_m,
            "semi_minor_m": poi_semi_minor_m,
            "azimuth_deg": round(math.degrees(math.atan2(v_mid[1], v_mid[0])), 1),
        },
        "reconstructed_trajectory": {
            "x_m": poo_traj_x[::-1] + poi_traj_x[1:],
            "y_m": poo_traj_y[::-1] + poi_traj_y[1:],
            "z_m": poo_traj_z[::-1] + poi_traj_z[1:],
        },
    }
