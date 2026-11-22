"""Precision Guidance, Autopilot Dynamics, and Proportional Navigation (PN / APN).

Implements:
- True Proportional Navigation (TPN) and Augmented Proportional Navigation (APN)
- Pure Pursuit (PP) baseline guidance
- Line-of-sight (LOS) kinematics, closing velocity, and seeker line-of-sight rate
- Seeker sensor noise, gimbal angle / Field of View (FoV) limits, and lock-loss conditions
- First-order autopilot / fin actuator lag and structural G-load saturation
- 3-DoF point-mass missile flight dynamics with thrust, drag, and gravity
- Stationary, moving, and evasive sinusoidal maneuvering targets
- Guidance Monte Carlo dispersion and Single-Shot Probability of Intercept (SSPI)
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np

from ammo_mc.environment import eb_04_isa_atmosphere


def proportional_navigation_acceleration(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_missile: np.ndarray,
    v_missile: np.ndarray,
    N_prime: float = 3.5,
    a_target: np.ndarray | None = None,
    law: Literal["TPN", "APN", "PP"] = "TPN",
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute commanded lateral acceleration vector (m/s^2) via TPN, APN, or Pure Pursuit.

    Args:
        r_target: 3D target position [x, y, z] in meters.
        v_target: 3D target velocity [vx, vy, vz] in m/s.
        r_missile: 3D missile position [x, y, z] in meters.
        v_missile: 3D missile velocity [vx, vy, vz] in m/s.
        N_prime: Effective navigation ratio (typically 3.0 to 5.0).
        a_target: Optional 3D target acceleration for Augmented PN (APN).
        law: Guidance law ('TPN', 'APN', 'PP').

    Returns:
        (a_cmd, telemetry_dict) where a_cmd is 3D acceleration vector (m/s^2).
    """
    R_vec = np.asarray(r_target, dtype=np.float64) - np.asarray(r_missile, dtype=np.float64)
    dist = float(np.linalg.norm(R_vec))
    if dist < 1e-6:
        return np.zeros(3, dtype=np.float64), {
            "range_m": 0.0,
            "closing_velocity_mps": 0.0,
            "los_rate_rad_s": 0.0,
            "look_angle_deg": 0.0,
        }

    R_hat = R_vec / dist
    V_rel = np.asarray(v_target, dtype=np.float64) - np.asarray(v_missile, dtype=np.float64)
    v_m_mag = float(np.linalg.norm(v_missile))
    v_m_hat = np.asarray(v_missile, dtype=np.float64) / max(v_m_mag, 1e-6)

    # Closing velocity V_c = - d(dist)/dt = - (R_vec . V_rel) / dist
    v_closing = -float(np.dot(R_vec, V_rel)) / dist

    # Line of sight rate vector Omega = (R_vec x V_rel) / (dist^2)
    omega_los = np.cross(R_vec, V_rel) / (dist**2)
    los_rate_mag = float(np.linalg.norm(omega_los))

    # Look angle (off-boresight angle between missile velocity and LOS)
    cos_look = float(np.clip(np.dot(v_m_hat, R_hat), -1.0, 1.0))
    look_angle_deg = math.degrees(math.acos(cos_look))

    telemetry = {
        "range_m": dist,
        "closing_velocity_mps": v_closing,
        "los_rate_rad_s": los_rate_mag,
        "look_angle_deg": look_angle_deg,
    }

    if law == "PP":
        # Pure Pursuit: missile velocity is driven toward target LOS vector
        tau_pp = 0.5
        a_cmd = (v_m_mag / tau_pp) * (R_hat - v_m_hat)
        return a_cmd, telemetry

    # True Proportional Navigation (TPN): a_cmd = N' * V_c * (Omega x v_m_hat)
    a_cmd_tpn = float(N_prime) * max(v_closing, 0.0) * np.cross(omega_los, v_m_hat)

    if law == "APN" and a_target is not None:
        # Augmented Proportional Navigation: adds 0.5 * N' * a_target_perp
        a_t_arr = np.asarray(a_target, dtype=np.float64)
        a_t_perp = a_t_arr - np.dot(a_t_arr, R_hat) * R_hat
        a_cmd = a_cmd_tpn + 0.5 * float(N_prime) * a_t_perp
    else:
        a_cmd = a_cmd_tpn

    return a_cmd, telemetry


def simulate_guided_trajectory(
    r0_m: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 10.0),
    v0_mps: float = 240.0,
    elev_deg: float = 22.0,
    azim_deg: float = 0.0,
    target_pos_m: tuple[float, float, float] | list[float] | np.ndarray = (3500.0, 50.0, 0.0),
    target_vel_mps: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 8.0, 0.0),
    target_accel_type: Literal["stationary", "constant_velocity", "evasive_weave"] = "constant_velocity",
    target_evasion_g: float = 2.0,
    target_evasion_freq_hz: float = 0.25,
    guidance_law: Literal["TPN", "APN", "PP"] = "TPN",
    N_prime: float = 3.5,
    g_limit: float = 18.0,
    tau_actuator_s: float = 0.08,
    seeker_fov_deg: float = 50.0,
    seeker_noise_std_rad: float = 0.0,
    motor_burn_time_s: float = 3.2,
    motor_thrust_N: float = 3800.0,
    mass_initial_kg: float = 28.0,
    mass_burnout_kg: float = 18.0,
    drag_cd: float = 0.38,
    calibre_m: float = 0.152,
    lethal_radius_m: float = 3.5,
    dt_s: float = 0.01,
    max_t_s: float = 40.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Integrates a closed-loop guided missile flight trajectory against a target.

    Models:
    - 3D flight translation with rocket motor thrust, aerodynamic drag, gravity, guidance lift.
    - Seeker line-of-sight tracking with noise and field-of-view limits.
    - Autopilot first-order actuator lag and G-saturation limit.
    - Stationary, moving, or sinusoidal evasive target dynamics.
    - Intercept detection (closest approach or direct hit within lethal radius).
    """
    rng = np.random.default_rng(seed)

    # Initial kinematics
    r_m = np.asarray(r0_m, dtype=np.float64).copy()
    theta_rad = math.radians(elev_deg)
    psi_rad = math.radians(azim_deg)
    v_dir = np.array([
        math.cos(theta_rad) * math.cos(psi_rad),
        math.cos(theta_rad) * math.sin(psi_rad),
        math.sin(theta_rad),
    ], dtype=np.float64)
    v_m = v_dir * float(v0_mps)

    r_t = np.asarray(target_pos_m, dtype=np.float64).copy()
    v_t = np.asarray(target_vel_mps, dtype=np.float64).copy()
    if target_accel_type == "stationary":
        v_t[:] = 0.0

    a_act = np.zeros(3, dtype=np.float64)
    area = 0.25 * math.pi * (calibre_m**2)
    mdot = (mass_initial_kg - mass_burnout_kg) / max(motor_burn_time_s, 1e-4) if motor_burn_time_s > 0 else 0.0

    # History buffers
    t_hist: list[float] = []
    r_m_hist: list[list[float]] = []
    r_t_hist: list[list[float]] = []
    v_m_hist: list[float] = []
    dist_hist: list[float] = []
    closing_vel_hist: list[float] = []
    look_angle_hist: list[float] = []
    a_cmd_g_hist: list[float] = []
    a_act_g_hist: list[float] = []

    t = 0.0
    min_dist = float("inf")
    intercept_status = "FLIGHT_TIMEOUT"
    lock_active = True

    while t <= max_t_s:
        dist = float(np.linalg.norm(r_t - r_m))
        v_mag = float(np.linalg.norm(v_m))
        v_hat = v_m / max(v_mag, 1e-6)

        if dist < min_dist:
            min_dist = dist

        # Target acceleration
        if target_accel_type == "evasive_weave":
            a_evade_mag = target_evasion_g * 9.80665
            a_t = np.array([
                0.0,
                a_evade_mag * math.sin(2.0 * math.pi * target_evasion_freq_hz * t),
                0.0,
            ], dtype=np.float64)
        else:
            a_t = np.zeros(3, dtype=np.float64)

        # Seeker LOS computation
        R_vec = r_t - r_m
        R_hat = R_vec / max(dist, 1e-6)
        cos_look = float(np.clip(np.dot(v_hat, R_hat), -1.0, 1.0))
        look_angle_deg = math.degrees(math.acos(cos_look))

        # Check seeker FoV limit
        if look_angle_deg > seeker_fov_deg:
            lock_active = False

        # Guidance command
        if lock_active and dist > lethal_radius_m:
            # Optional seeker noise perturbation
            noisy_r_t = r_t.copy()
            if seeker_noise_std_rad > 0.0:
                noise_vec = rng.normal(0.0, seeker_noise_std_rad * dist, size=3)
                noisy_r_t += noise_vec

            a_cmd_raw, tel = proportional_navigation_acceleration(
                r_target=noisy_r_t,
                v_target=v_t,
                r_missile=r_m,
                v_missile=v_m,
                N_prime=N_prime,
                a_target=a_t,
                law=guidance_law,
            )
            v_closing = tel["closing_velocity_mps"]

            # Command must be perpendicular to missile velocity
            a_cmd_perp = a_cmd_raw - np.dot(a_cmd_raw, v_hat) * v_hat

            # Actuator saturation limit
            max_accel = g_limit * 9.80665
            a_cmd_mag = float(np.linalg.norm(a_cmd_perp))
            if a_cmd_mag > max_accel:
                a_cmd = a_cmd_perp * (max_accel / a_cmd_mag)
            else:
                a_cmd = a_cmd_perp
        else:
            a_cmd = np.zeros(3, dtype=np.float64)
            V_rel = v_t - v_m
            v_closing = -float(np.dot(R_vec, V_rel)) / max(dist, 1e-6)

        # Actuator lag: d(a_act)/dt = (a_cmd - a_act) / tau
        tau = max(tau_actuator_s, 1e-4)
        a_act += (dt_s / tau) * (a_cmd - a_act)
        # Re-enforce saturation on achieved acceleration
        act_mag = float(np.linalg.norm(a_act))
        if act_mag > (g_limit * 9.80665):
            a_act *= (g_limit * 9.80665) / act_mag

        # Record history
        t_hist.append(t)
        r_m_hist.append(r_m.tolist())
        r_t_hist.append(r_t.tolist())
        v_m_hist.append(v_mag)
        dist_hist.append(dist)
        closing_vel_hist.append(v_closing)
        look_angle_hist.append(look_angle_deg)
        a_cmd_g_hist.append(float(np.linalg.norm(a_cmd)) / 9.80665)
        a_act_g_hist.append(float(np.linalg.norm(a_act)) / 9.80665)

        # Intercept termination checks
        if dist <= lethal_radius_m:
            intercept_status = "DIRECT_HIT"
            break
        if v_closing < 0.0 and t > 0.5:
            # Closest approach passed
            intercept_status = "CLOSEST_APPROACH"
            break
        if r_m[2] <= 0.0 and t > 0.2:
            intercept_status = "GROUND_IMPACT"
            break
        if not lock_active and t > 0.5:
            intercept_status = "SEEKER_LOCK_LOST"
            break

        # Physics integration
        # Atmosphere
        atm = eb_04_isa_atmosphere(h_m=max(0.0, float(r_m[2])))
        rho = float(atm["rho_kg_m3"])

        # Current mass & thrust
        if t <= motor_burn_time_s:
            cur_mass = mass_initial_kg - mdot * t
            f_thrust = v_hat * float(motor_thrust_N)
        else:
            cur_mass = mass_burnout_kg
            f_thrust = np.zeros(3, dtype=np.float64)

        # Aerodynamic drag
        f_drag = -0.5 * rho * (v_mag**2) * area * drag_cd * v_hat

        # Gravity
        f_grav = np.array([0.0, 0.0, -cur_mass * 9.80665], dtype=np.float64)

        # Total acceleration = a_aero_drag_thrust_grav + a_act
        a_ballistic = (f_thrust + f_drag + f_grav) / max(cur_mass, 1.0)
        a_total = a_ballistic + a_act

        # Update states
        r_m += v_m * dt_s + 0.5 * a_total * (dt_s**2)
        v_m += a_total * dt_s

        r_t += v_t * dt_s + 0.5 * a_t * (dt_s**2)
        v_t += a_t * dt_s

        t += dt_s

    # Metrics
    final_dist = dist_hist[-1] if dist_hist else min_dist
    peak_g = max(a_act_g_hist) if a_act_g_hist else 0.0
    mean_g = float(np.mean(a_act_g_hist)) if a_act_g_hist else 0.0

    return {
        "status": intercept_status,
        "hit": bool(final_dist <= lethal_radius_m),
        "miss_distance_m": float(final_dist),
        "min_distance_m": float(min_dist),
        "time_of_flight_s": float(t),
        "terminal_closing_velocity_mps": float(closing_vel_hist[-1]) if closing_vel_hist else 0.0,
        "peak_g_load": float(peak_g),
        "mean_g_load": float(mean_g),
        "lock_active_at_end": bool(lock_active),
        "final_missile_pos_m": r_m.tolist(),
        "final_target_pos_m": r_t.tolist(),
        "history": {
            "t_s": t_hist,
            "missile_pos": r_m_hist,
            "target_pos": r_t_hist,
            "missile_vel_mps": v_m_hist,
            "range_m": dist_hist,
            "closing_vel_mps": closing_vel_hist,
            "look_angle_deg": look_angle_hist,
            "cmd_g": a_cmd_g_hist,
            "act_g": a_act_g_hist,
        },
    }


def run_guidance_monte_carlo(
    n_trials: int = 50,
    r0_m: tuple[float, float, float] = (0.0, 0.0, 10.0),
    v0_mps: float = 240.0,
    elev_deg: float = 22.0,
    target_pos_m: tuple[float, float, float] = (3500.0, 50.0, 0.0),
    target_vel_mps: tuple[float, float, float] = (0.0, 8.0, 0.0),
    guidance_law: Literal["TPN", "APN", "PP"] = "TPN",
    N_prime: float = 3.5,
    g_limit: float = 18.0,
    seeker_noise_std_rad: float = 0.0004,
    lethal_radius_m: float = 3.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Executes a Monte Carlo dispersion study over guided missile engagements."""
    rng = np.random.default_rng(seed)
    miss_distances: list[float] = []
    hits = 0
    tofs: list[float] = []
    peak_gs: list[float] = []

    for i in range(n_trials):
        trial_seed = int(rng.integers(1, 1000000))
        # Add slight target position jitter
        t_pos_jit = (
            target_pos_m[0] + float(rng.normal(0.0, 10.0)),
            target_pos_m[1] + float(rng.normal(0.0, 10.0)),
            target_pos_m[2],
        )
        res = simulate_guided_trajectory(
            r0_m=r0_m,
            v0_mps=v0_mps + float(rng.normal(0.0, 3.0)),
            elev_deg=elev_deg,
            target_pos_m=t_pos_jit,
            target_vel_mps=target_vel_mps,
            guidance_law=guidance_law,
            N_prime=N_prime,
            g_limit=g_limit,
            seeker_noise_std_rad=seeker_noise_std_rad,
            lethal_radius_m=lethal_radius_m,
            seed=trial_seed,
        )
        miss_distances.append(res["miss_distance_m"])
        tofs.append(res["time_of_flight_s"])
        peak_gs.append(res["peak_g_load"])
        if res["hit"]:
            hits += 1

    misses = np.sort(np.asarray(miss_distances, dtype=np.float64))
    cep50 = float(np.percentile(misses, 50))
    cep95 = float(np.percentile(misses, 95))

    return {
        "n_trials": n_trials,
        "hits": hits,
        "p_intercept": float(hits / max(n_trials, 1)),
        "mean_miss_distance_m": float(np.mean(misses)),
        "std_miss_distance_m": float(np.std(misses)),
        "cep50_m": cep50,
        "cep95_m": cep95,
        "min_miss_distance_m": float(misses[0]),
        "max_miss_distance_m": float(misses[-1]),
        "mean_tof_s": float(np.mean(tofs)),
        "mean_peak_g": float(np.mean(peak_gs)),
        "miss_distances": misses.tolist(),
    }
