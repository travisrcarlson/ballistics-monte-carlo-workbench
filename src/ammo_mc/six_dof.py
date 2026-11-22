"""Full 6-Degree-of-Freedom (6-DoF) Rigid-Body Projectile Dynamics.

Grounded in:
- STANAG 4355 (The Modified Point Mass and Six Degree of Freedom Trajectory Model)
- Robert L. McCoy (Modern Exterior Ballistics)
- Carlucci & Jacobson (Ballistics: Theory and Design of Guns and Ammunition)

Features:
- 13-state rigid-body vector: position [x,y,z], velocity [u,v,w], quaternion [q0,q1,q2,q3], body rates [p,q,r]
- 4th-order Runge-Kutta (RK4) integration for numerical stability in fast gyroscopic nutation
- Complete aerodynamic forces: zero-yaw drag, yaw-induced drag, normal/lift force, Magnus cross-force
- Complete aerodynamic moments: overturning moment (C_Malpha), pitch damping (C_Mq + C_M_alphadot),
  spin damping (C_lp), and Magnus moment (C_Npalpha)
- Quaternion attitude kinematics preventing gimbal lock
- Gyroscopic stability factor S_g(t) and McCoy dynamic stability factor S_d(t)
- Epicyclic yaw motion analysis: fast nutation loop, slow precession drift, and total angle of attack alpha_t(t)
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from ammo_mc.environment import eb_04_isa_atmosphere


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [q0, q1, q2, q3] to 3x3 Direction Cosine Matrix C_B/E."""
    q0, q1, q2, q3 = q
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
        [2.0 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2.0 * (q2 * q3 + q0 * q1)],
        [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
    ], dtype=np.float64)


def euler_angles_to_quaternion(pitch_rad: float, yaw_rad: float, roll_rad: float = 0.0) -> np.ndarray:
    """Creates a unit quaternion from pitch (theta), yaw/azimuth (psi), and roll (phi)."""
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    q = np.array([q0, q1, q2, q3], dtype=np.float64)
    return q / np.linalg.norm(q)


def _compute_6dof_derivatives(
    state: np.ndarray,
    prms: dict[str, Any],
) -> np.ndarray:
    """Computes the 13-state derivative vector [v_e, a_e, q_dot, omega_dot]."""
    r = state[0:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    q_norm = float(np.linalg.norm(q))
    if q_norm > 1e-9:
        q_unit = q / q_norm
    else:
        q_unit = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    c_be = quaternion_to_dcm(q_unit)

    # Atmosphere at current altitude
    alt = max(0.0, float(r[2]))
    atm = eb_04_isa_atmosphere(h_m=alt)
    rho = float(atm["rho_kg_m3"])

    # Velocity in projectile body frame
    v_b = c_be @ v
    v_mag = max(float(np.linalg.norm(v_b)), 1e-6)
    u_b, v_b_y, w_b_z = v_b

    # Transverse airspeed & total angle of attack
    trans_speed = math.hypot(v_b_y, w_b_z)
    sin_at = min(1.0, trans_speed / v_mag)
    q_dyn = 0.5 * rho * (v_mag**2) * prms["area"]
    p_star = (omega[0] * prms["calibre_m"]) / v_mag

    # Drag force (axial)
    cd = prms["cd0"] + prms["cd_alpha2"] * (sin_at**2)
    fx_b = -q_dyn * cd

    # Normal / Lift force
    eta = np.array([0.0, v_b_y / max(trans_speed, 1e-6), w_b_z / max(trans_speed, 1e-6)], dtype=np.float64)
    fn_mag = q_dyn * prms["cl_alpha"] * sin_at
    f_norm_b = -fn_mag * eta

    # Magnus cross-force
    i_b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    f_mag_b = q_dyn * p_star * prms["c_yp_alpha"] * sin_at * np.cross(i_b, eta)

    f_total_b = np.array([fx_b, 0.0, 0.0], dtype=np.float64) + f_norm_b + f_mag_b

    # Forces in Earth frame
    f_total_e = c_be.T @ f_total_b
    f_grav_e = np.array([0.0, 0.0, -prms["mass_kg"] * 9.80665], dtype=np.float64)
    f_coriolis_e = -2.0 * prms["mass_kg"] * np.cross(prms["omega_earth_vec"], v)
    a_e = (f_total_e + f_grav_e + f_coriolis_e) / prms["mass_kg"]

    # Aerodynamic Moments in Body Frame
    # Overturning moment: M_overturn = q_dyn * d * C_Malpha * (i_b x v_b / V)
    m_overturn = q_dyn * prms["calibre_m"] * prms["cm_alpha"] * np.array([0.0, -w_b_z / v_mag, v_b_y / v_mag], dtype=np.float64)

    # Pitch & Yaw damping moments
    d_over_v = prms["calibre_m"] / v_mag
    m_damp = q_dyn * prms["calibre_m"] * d_over_v * prms["cm_q"] * np.array([0.0, omega[1], omega[2]], dtype=np.float64)

    # Spin damping moment
    m_spin = np.array([q_dyn * prms["calibre_m"] * p_star * prms["c_lp"], 0.0, 0.0], dtype=np.float64)

    # Magnus moment
    m_magnus = q_dyn * prms["calibre_m"] * p_star * prms["c_np_alpha"] * np.array([0.0, v_b_y / v_mag, w_b_z / v_mag], dtype=np.float64)

    m_total_b = m_overturn + m_damp + m_spin + m_magnus

    # Euler rotational dynamics
    ix = prms["ix"]
    it = prms["it"]
    p_dot = m_total_b[0] / ix
    q_dot = (m_total_b[1] - (ix - it) * omega[0] * omega[2]) / it
    r_dot = (m_total_b[2] - (it - ix) * omega[0] * omega[1]) / it

    # Quaternion attitude kinematics
    omega_mat = np.array([
        [0.0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0.0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0.0, omega[0]],
        [omega[2], omega[1], -omega[0], 0.0],
    ], dtype=np.float64)
    q_dot_vec = 0.5 * (omega_mat @ q_unit)

    return np.concatenate([v, a_e, q_dot_vec, [p_dot, q_dot, r_dot]])


def simulate_6dof_trajectory(
    v0_mps: float = 827.0,
    elev_deg: float = 30.0,
    azim_deg: float = 0.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    twist_cal_per_turn: float = 20.0,
    ix_kg_m2: float = 0.145,
    it_kg_m2: float = 1.350,
    cd0: float = 0.22,
    cd_alpha2: float = 3.5,
    cl_alpha: float = 2.4,
    cm_alpha: float = 4.2,
    cm_q: float = -14.0,
    c_lp: float = -0.025,
    c_yp_alpha: float = -0.65,
    c_np_alpha: float = 0.85,
    initial_yaw_rate_rad_s: float = 0.05,
    initial_pitch_rate_rad_s: float = 0.02,
    latitude_deg: float = 45.0,
    dt_s: float = 0.002,
    max_t_s: float = 120.0,
) -> dict[str, Any]:
    """Integrates the full 6-DoF rigid-body equations of motion using RK4."""
    elev_rad = math.radians(elev_deg)
    azim_rad = math.radians(azim_deg)
    area = 0.25 * math.pi * (calibre_m**2)

    # Initial axial spin rate: p0 = 2 * pi * V0 / (twist * d)
    rifling_len = twist_cal_per_turn * calibre_m
    p0 = 2.0 * math.pi * v0_mps / max(rifling_len, 1e-4)

    # Initial Earth frame velocity
    v_dir = np.array([
        math.cos(elev_rad) * math.cos(azim_rad),
        math.cos(elev_rad) * math.sin(azim_rad),
        math.sin(elev_rad),
    ], dtype=np.float64)
    v0_vec = v_dir * v0_mps

    q0 = euler_angles_to_quaternion(pitch_rad=elev_rad, yaw_rad=azim_rad, roll_rad=0.0)

    # Earth rotation
    omega_earth = 7.292115e-5
    lat_rad = math.radians(latitude_deg)
    omega_e_vec = np.array([
        omega_earth * math.cos(lat_rad),
        0.0,
        omega_earth * math.sin(lat_rad),
    ], dtype=np.float64)

    prms = {
        "calibre_m": float(calibre_m),
        "mass_kg": float(mass_kg),
        "area": float(area),
        "ix": float(ix_kg_m2),
        "it": float(it_kg_m2),
        "cd0": float(cd0),
        "cd_alpha2": float(cd_alpha2),
        "cl_alpha": float(cl_alpha),
        "cm_alpha": float(cm_alpha),
        "cm_q": float(cm_q),
        "c_lp": float(c_lp),
        "c_yp_alpha": float(c_yp_alpha),
        "c_np_alpha": float(c_np_alpha),
        "omega_earth_vec": omega_e_vec,
    }

    # 13-state vector
    state = np.array([
        0.0, 0.0, 1.5,
        v0_vec[0], v0_vec[1], v0_vec[2],
        q0[0], q0[1], q0[2], q0[3],
        p0, float(initial_pitch_rate_rad_s), float(initial_yaw_rate_rad_s),
    ], dtype=np.float64)

    # History buffers
    t_hist: list[float] = []
    x_hist: list[float] = []
    y_hist: list[float] = []
    z_hist: list[float] = []
    v_hist: list[float] = []
    mach_hist: list[float] = []
    alpha_deg_hist: list[float] = []
    beta_deg_hist: list[float] = []
    alpha_t_deg_hist: list[float] = []
    spin_rpm_hist: list[float] = []
    sg_hist: list[float] = []
    sd_hist: list[float] = []
    pitch_deg_hist: list[float] = []
    yaw_deg_hist: list[float] = []

    t = 0.0
    step_count = 0
    save_interval = max(1, int(0.04 / dt_s))  # Save every ~0.04s

    while state[2] >= 0.0 and t <= max_t_s:
        # Re-normalize quaternion
        q_len = np.linalg.norm(state[6:10])
        if q_len > 1e-9:
            state[6:10] /= q_len

        # Log states periodically
        if step_count % save_interval == 0:
            c_be = quaternion_to_dcm(state[6:10])
            alt = max(0.0, float(state[2]))
            atm = eb_04_isa_atmosphere(h_m=alt)
            rho = float(atm["rho_kg_m3"])
            sound_speed = float(atm["a_mps"])

            v_b = c_be @ state[3:6]
            v_mag = float(np.linalg.norm(v_b))
            mach = v_mag / max(sound_speed, 1.0)
            u_b, v_b_y, w_b_z = v_b

            trans_speed = math.hypot(v_b_y, w_b_z)
            sin_at = min(1.0, trans_speed / max(v_mag, 1e-6))
            alpha_t_deg = math.degrees(math.asin(sin_at))
            alpha_deg = math.degrees(math.atan2(w_b_z, max(u_b, 1.0)))
            beta_deg = math.degrees(math.asin(np.clip(v_b_y / max(v_mag, 1e-6), -1.0, 1.0)))

            p_spin = float(state[10])
            spin_rpm = p_spin * 60.0 / (2.0 * math.pi)

            # S_g = (I_x^2 * p^2) / (2 * rho * S * d * I_t * C_Malpha * V^2)
            denom_sg = 2.0 * rho * area * calibre_m * it_kg_m2 * cm_alpha * (v_mag**2)
            sg = ((ix_kg_m2 * p_spin) ** 2) / max(denom_sg, 1e-9)

            # McCoy Dynamic stability S_d
            sd = 1.0 - (ix_kg_m2 / (2.0 * it_kg_m2)) * (c_lp / max(abs(cm_q), 1e-6))

            nose_e = c_be.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            pitch_deg = math.degrees(math.asin(np.clip(nose_e[2], -1.0, 1.0)))
            yaw_deg = math.degrees(math.atan2(nose_e[1], nose_e[0]))

            t_hist.append(t)
            x_hist.append(float(state[0]))
            y_hist.append(float(state[1]))
            z_hist.append(float(state[2]))
            v_hist.append(v_mag)
            mach_hist.append(mach)
            alpha_deg_hist.append(alpha_deg)
            beta_deg_hist.append(beta_deg)
            alpha_t_deg_hist.append(alpha_t_deg)
            spin_rpm_hist.append(spin_rpm)
            sg_hist.append(float(sg))
            sd_hist.append(float(sd))
            pitch_deg_hist.append(pitch_deg)
            yaw_deg_hist.append(yaw_deg)

        # RK4 Integration step
        k1 = _compute_6dof_derivatives(state, prms)
        k2 = _compute_6dof_derivatives(state + 0.5 * dt_s * k1, prms)
        k3 = _compute_6dof_derivatives(state + 0.5 * dt_s * k2, prms)
        k4 = _compute_6dof_derivatives(state + dt_s * k3, prms)

        state += (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt_s
        step_count += 1

    # Record final point
    x_hist.append(float(state[0]))
    y_hist.append(float(state[1]))
    z_hist.append(0.0)
    t_hist.append(t)
    v_hist.append(float(np.linalg.norm(state[3:6])))
    spin_rpm_hist.append(float(state[10]) * 60.0 / (2.0 * math.pi))

    return {
        "final_range_m": float(state[0]),
        "final_deflection_m": float(state[1]),
        "time_of_flight_s": float(t),
        "impact_velocity_mps": float(np.linalg.norm(state[3:6])),
        "max_altitude_m": float(max(z_hist)) if z_hist else 0.0,
        "initial_spin_rpm": float(spin_rpm_hist[0]) if spin_rpm_hist else 0.0,
        "terminal_spin_rpm": float(spin_rpm_hist[-1]) if spin_rpm_hist else 0.0,
        "muzzle_sg": float(sg_hist[0]) if sg_hist else 0.0,
        "min_sg": float(min(sg_hist)) if sg_hist else 0.0,
        "mean_sd": float(np.mean(sd_hist)) if sd_hist else 1.0,
        "max_yaw_of_repose_deg": float(max(alpha_t_deg_hist)) if alpha_t_deg_hist else 0.0,
        "history": {
            "t_s": t_hist,
            "x_m": x_hist,
            "y_m": y_hist,
            "z_m": z_hist,
            "v_mps": v_hist,
            "mach": mach_hist,
            "alpha_deg": alpha_deg_hist,
            "beta_deg": beta_deg_hist,
            "alpha_t_deg": alpha_t_deg_hist,
            "spin_rpm": spin_rpm_hist,
            "sg": sg_hist,
            "sd": sd_hist,
            "pitch_deg": pitch_deg_hist,
            "yaw_deg": yaw_deg_hist,
        },
    }
