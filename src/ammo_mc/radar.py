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

    # Exact Physical Optics broadside cylinder peak sigma_cyl = k * a * L^2
    k = 2.0 * math.pi / wavelength_m
    rcs_broadside = k * radius_m * (length_m**2)

    # Nose-on cone tip return (Rayleigh / cone apex contribution)
    rcs_nose = (math.pi * (radius_m**2)) * (math.tan(math.radians(12.0)) ** 4)

    # Flat circular base specular return (tail-on aspect cos_asp < 0 only!)
    rcs_base = math.pi * (radius_m**2)

    sin_asp = math.sin(aspect_rad)
    cos_asp = math.cos(aspect_rad)
    nose_term = rcs_nose * (max(0.0, cos_asp) ** 4)
    base_term = rcs_base * (max(0.0, -cos_asp) ** 2) * 0.25

    rcs_val = nose_term + rcs_broadside * (sin_asp**8) + base_term
    return float(np.clip(rcs_val, 0.003, 2.5))


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

    # Fit local quadratic state windows at track start (for POO) and track end (for POI)
    # Using a single global quadratic across 40+ s of drag-dominated flight biases endpoint velocities.
    def _fit_local_state(
        t_sub: np.ndarray,
        x_sub: np.ndarray,
        y_sub: np.ndarray,
        z_sub: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        deg = 2 if len(t_sub) >= 4 else 1
        t_eval = float(np.mean(t_sub))
        t_rel = t_sub - t_eval
        px = np.polyfit(t_rel, x_sub, deg)
        py = np.polyfit(t_rel, y_sub, deg)
        pz = np.polyfit(t_rel, z_sub, deg)
        if deg == 2:
            r_eval = np.array([px[2], py[2], pz[2]], dtype=np.float64)
            v_eval = np.array([px[1], py[1], pz[1]], dtype=np.float64)
        else:
            r_eval = np.array([px[1], py[1], pz[1]], dtype=np.float64)
            v_eval = np.array([px[0], py[0], pz[0]], dtype=np.float64)
        return t_eval, r_eval, v_eval

    early_mask = (t_arr - t_arr[0]) <= 4.5
    if int(np.sum(early_mask)) < 6:
        k_e = min(len(t_arr), 10)
        early_slice = slice(0, k_e)
    else:
        early_slice = early_mask
    t_early, r_early, v_early = _fit_local_state(t_arr[early_slice], x_arr[early_slice], y_arr[early_slice], z_arr[early_slice])

    late_mask = (t_arr[-1] - t_arr) <= 4.5
    if int(np.sum(late_mask)) < 6:
        k_l = min(len(t_arr), 10)
        late_slice = slice(len(t_arr) - k_l, len(t_arr))
    else:
        late_slice = late_mask
    t_late, r_late, v_late = _fit_local_state(t_arr[late_slice], x_arr[late_slice], y_arr[late_slice], z_arr[late_slice])

    from ammo_mc.data import get_kb
    from ammo_mc.environment import eb_04_isa_atmosphere

    kb = get_kb()
    area = 0.25 * math.pi * (calibre_m**2)

    def ball_accel(pos_v: np.ndarray, vel_v: np.ndarray) -> np.ndarray:
        alt = max(float(pos_v[2]), 0.0)
        atm = eb_04_isa_atmosphere(h_m=alt)
        v_m = math.hypot(float(vel_v[0]), float(vel_v[1]), float(vel_v[2]))
        mach = v_m / max(float(atm["a_mps"]), 100.0)
        cd = float(kb.interp_cd(mach, drag_ref="G7"))
        drag_acc = 0.5 * float(atm["rho_kg_m3"]) * area * cd / max(mass_kg, 1e-3) * v_m
        return np.array([-drag_acc * vel_v[0], -drag_acc * vel_v[1], -drag_acc * vel_v[2] - 9.80665], dtype=np.float64)

    def rk4_step(pos_v: np.ndarray, vel_v: np.ndarray, dt_step: float) -> tuple[np.ndarray, np.ndarray]:
        a1 = ball_accel(pos_v, vel_v)
        p2, v2 = pos_v + 0.5 * dt_step * vel_v, vel_v + 0.5 * dt_step * a1
        a2 = ball_accel(p2, v2)
        p3, v3 = pos_v + 0.5 * dt_step * v2, vel_v + 0.5 * dt_step * a2
        a3 = ball_accel(p3, v3)
        p4, v4 = pos_v + dt_step * v3, vel_v + dt_step * a3
        a4 = ball_accel(p4, v4)
        p_n = pos_v + (dt_step / 6.0) * (vel_v + 2.0 * v2 + 2.0 * v3 + v4)
        v_n = vel_v + (dt_step / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        return p_n, v_n

    # 1. Back-integrate backwards in time from early window with RK4 (dt_rev < 0) to Point of Origin (POO)
    dt_step = 0.04
    t_curr = t_early
    pos_curr = r_early.copy()
    vel_curr = v_early.copy()

    poo_traj_x = [float(pos_curr[0])]
    poo_traj_y = [float(pos_curr[1])]
    poo_traj_z = [float(pos_curr[2])]

    while pos_curr[2] > gun_elev_m and t_curr > -5.0:
        pos_prev, vel_prev, t_prev = pos_curr.copy(), vel_curr.copy(), t_curr
        pos_n, vel_n = rk4_step(pos_curr, vel_curr, -dt_step)
        t_n = t_curr - dt_step
        if pos_n[2] <= gun_elev_m and pos_prev[2] > gun_elev_m and (pos_prev[2] - pos_n[2]) > 1e-12:
            frac = (pos_prev[2] - gun_elev_m) / (pos_prev[2] - pos_n[2])
            pos_curr = pos_prev + frac * (pos_n - pos_prev)
            vel_curr = vel_prev + frac * (vel_n - vel_prev)
            pos_curr[2] = gun_elev_m
            t_curr = t_prev - frac * dt_step
            poo_traj_x.append(float(pos_curr[0]))
            poo_traj_y.append(float(pos_curr[1]))
            poo_traj_z.append(float(gun_elev_m))
            break
        pos_curr, vel_curr, t_curr = pos_n, vel_n, t_n
        poo_traj_x.append(float(pos_curr[0]))
        poo_traj_y.append(float(pos_curr[1]))
        poo_traj_z.append(float(pos_curr[2]))

    poo_x = float(pos_curr[0])
    poo_y = float(pos_curr[1])
    launch_time_s = max(0.0, float(t_curr))
    muzzle_v_est = float(np.linalg.norm(vel_curr))

    # 2. Forward-integrate in time from late window with RK4 to Point of Impact (POI)
    t_fwd = t_late
    pos_fwd = r_late.copy()
    vel_fwd = v_late.copy()

    poi_traj_x = [float(pos_fwd[0])]
    poi_traj_y = [float(pos_fwd[1])]
    poi_traj_z = [float(pos_fwd[2])]

    while pos_fwd[2] > target_elev_m and t_fwd < 200.0:
        pos_prev, vel_prev, t_prev = pos_fwd.copy(), vel_fwd.copy(), t_fwd
        pos_n, vel_n = rk4_step(pos_fwd, vel_fwd, dt_step)
        t_n = t_fwd + dt_step
        if pos_n[2] <= target_elev_m and pos_prev[2] > target_elev_m and (pos_prev[2] - pos_n[2]) > 1e-12:
            frac = (pos_prev[2] - target_elev_m) / (pos_prev[2] - pos_n[2])
            pos_fwd = pos_prev + frac * (pos_n - pos_prev)
            vel_fwd = vel_prev + frac * (vel_n - vel_prev)
            pos_fwd[2] = target_elev_m
            t_fwd = t_prev + frac * dt_step
            poi_traj_x.append(float(pos_fwd[0]))
            poi_traj_y.append(float(pos_fwd[1]))
            poi_traj_z.append(float(target_elev_m))
            break
        pos_fwd, vel_fwd, t_fwd = pos_n, vel_n, t_n
        poi_traj_x.append(float(pos_fwd[0]))
        poi_traj_y.append(float(pos_fwd[1]))
        poi_traj_z.append(float(pos_fwd[2]))

    poi_x = float(pos_fwd[0])
    poi_y = float(pos_fwd[1])
    impact_time_s = float(t_fwd)
    early_warning_time_s = max(0.0, impact_time_s - t_arr[-1])

    # Smooth track bridge between t_early and t_late for full arc visualization & noise estimation
    deg_cov = min(4, max(1, len(t_arr) - 1))
    t_rel_all = t_arr - float(np.mean(t_arr))
    poly_x_cov = np.polyfit(t_rel_all, x_arr, deg_cov)
    poly_y_cov = np.polyfit(t_rel_all, y_arr, deg_cov)
    poly_z_cov = np.polyfit(t_rel_all, z_arr, deg_cov)
    if t_late > t_early + 0.05:
        t_bridge = np.linspace(t_early, t_late, 30)[1:-1] - float(np.mean(t_arr))
        mid_x = np.polyval(poly_x_cov, t_bridge).tolist()
        mid_y = np.polyval(poly_y_cov, t_bridge).tolist()
        mid_z = np.polyval(poly_z_cov, t_bridge).tolist()
    else:
        mid_x, mid_y, mid_z = [], [], []

    # 3. Error covariance estimates (POO / POI semi-major and semi-minor axes)
    residuals_x = x_arr - np.polyval(poly_x_cov, t_rel_all)
    residuals_y = y_arr - np.polyval(poly_y_cov, t_rel_all)
    rmse = math.hypot(float(np.std(residuals_x)), float(np.std(residuals_y)))
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
            "azimuth_deg": round(math.degrees(math.atan2(v_early[1], v_early[0])), 1),
        },
        "poi_error_ellipse": {
            "semi_major_m": poi_semi_major_m,
            "semi_minor_m": poi_semi_minor_m,
            "azimuth_deg": round(math.degrees(math.atan2(v_late[1], v_late[0])), 1),
        },
        "reconstructed_trajectory": {
            "x_m": poo_traj_x[::-1] + mid_x + poi_traj_x[1:],
            "y_m": poo_traj_y[::-1] + mid_y + poi_traj_y[1:],
            "z_m": poo_traj_z[::-1] + mid_z + poi_traj_z[1:],
        },
    }
