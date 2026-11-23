"""Exterior ballistics formulas (EB-01..EB-17) and vectorised 3-DOF / MPM trajectory integrators.

Coordinate frame for 3-D trajectory integration (right-handed RLU basis):
- axis 0 (`x`): horizontal downrange along nominal line of fire (m)
- axis 1 (`y_left`): horizontal crossrange to the LEFT of line of fire (m)
  -> Note: Gunner's deflection (+ RIGHT) is reported as `deflection_m = -y_left`!
- axis 2 (`z`): altitude UP above reference datum (m)

Because `(range, left, up)` is strictly orthonormal and right-handed (`x_hat x y_hat = z_hat`),
all vector cross products (`Omega x v` in EB-10 and `v x dv/dt` in STANAG 4355 MPM EB-05)
produce physically exact Coriolis and rightward gyroscopic spin drift!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from scipy.integrate import solve_ivp

from ammo_mc.data import get_kb
from ammo_mc.environment import (
    G0,
    AtmosphereEnvironment,
    eb_03_mach_sound_speed,
    eb_04_isa_atmosphere,
)
from ammo_mc.formulas_registry import record_formula_use, register_formula

OMEGA_EARTH_RAD_S = 7.2921e-5  # Verified S14 (ADA499519)


@register_formula("EB-01")
def eb_01_point_mass_eom(
    v_mps: np.ndarray,
    wind_mps: np.ndarray,
    rho_kg_m3: float | np.ndarray,
    d_m: float | np.ndarray,
    m_kg: float | np.ndarray,
    cd: float | np.ndarray,
    lat_rad: float = 0.0,
    az_rad: float = 0.0,
    include_coriolis: bool = True,
    g0: float = G0,
) -> np.ndarray:
    """EB-01: 3-DOF point-mass acceleration dv/dt = a_drag + g - 2*(Omega x v)."""
    v = np.asarray(v_mps, dtype=np.float64)
    w = np.asarray(wind_mps, dtype=np.float64)
    vr = v - w
    speed_rel = np.linalg.norm(vr, axis=-1, keepdims=True)
    area = 0.25 * np.pi * (np.asarray(d_m, dtype=np.float64) ** 2)
    k_drag = (
        np.asarray(rho_kg_m3, dtype=np.float64)[..., None]
        * area[..., None]
        * np.asarray(cd, dtype=np.float64)[..., None]
        / (2.0 * np.maximum(np.asarray(m_kg, dtype=np.float64)[..., None], 1e-9))
    )
    a_drag = -k_drag * speed_rel * vr
    g_vec = np.zeros_like(v)
    g_vec[..., 2] = -g0
    if include_coriolis:
        a_cor = eb_10_coriolis_acceleration(v, lat_rad=lat_rad, az_rad=az_rad)
    else:
        a_cor = 0.0
    return a_drag + g_vec + a_cor


@register_formula("EB-02")
def eb_02_drag_standard_bc(
    mach: float | np.ndarray,
    m_kg: float | np.ndarray,
    d_m: float | np.ndarray,
    i_form: float | np.ndarray = 1.0,
    drag_ref: str = "G7",
    rho_kg_m3: float | np.ndarray = 1.225,
    v_rel_mps: float | np.ndarray | None = None,
    custom_cd_table: list[dict[str, float]] | None = None,
) -> dict[str, float | np.ndarray]:
    """EB-02: Standard drag Cd = i * Cd_ref(M), ballistic coefficient BC = m/(i*d^2), and drag decel."""
    kb = get_kb()
    cd_ref = kb.interp_cd(mach, drag_ref=drag_ref, custom_table=custom_cd_table)
    i_arr = np.asarray(i_form, dtype=np.float64)
    d_arr = np.asarray(d_m, dtype=np.float64)
    m_arr = np.asarray(m_kg, dtype=np.float64)
    cd = i_arr * cd_ref
    bc_si = m_arr / np.maximum(i_arr * (d_arr**2), 1e-12)
    if v_rel_mps is not None:
        v_arr = np.asarray(v_rel_mps, dtype=np.float64)
        a_drag = np.asarray(rho_kg_m3, dtype=np.float64) * (v_arr**2) * np.pi * cd_ref / (8.0 * bc_si)
    else:
        a_drag = np.zeros_like(bc_si)
    if np.ndim(mach) == 0 and np.ndim(m_kg) == 0 and np.ndim(i_form) == 0:
        return {
            "cd": float(cd),
            "cd_ref": float(cd_ref),
            "bc_kg_m2": float(bc_si),
            "bc_lb_in2": float(bc_si / 703.06958),
            "a_drag_mps2": float(a_drag),
        }
    return {
        "cd": cd,
        "cd_ref": cd_ref,
        "bc_kg_m2": bc_si,
        "bc_lb_in2": bc_si / 703.06958,
        "a_drag_mps2": a_drag,
    }


@register_formula("EB-05")
def eb_05_mpm_acceleration(
    v_mps: np.ndarray,
    wind_mps: np.ndarray,
    rho_kg_m3: float | np.ndarray,
    d_m: float,
    m_kg: float | np.ndarray,
    ix_kg_m2: float,
    p_spin_rad_s: float | np.ndarray,
    cd0: float | np.ndarray,
    i_form: float | np.ndarray = 1.0,
    c_da2: float = 4.0,
    c_la: float = 2.2,
    c_la3: float = 0.0,
    c_ma: float = 2.8,
    c_mag: float = 0.25,
    q_d: float = 1.0,
    f_l: float = 1.0,
    q_m: float = 1.0,
    lat_rad: float = 0.0,
    az_rad: float = 0.0,
    g0: float = G0,
) -> tuple[np.ndarray, np.ndarray]:
    """EB-05: Modified Point Mass (STANAG 4355) acceleration with yaw of repose, lift & Magnus.

    Returns `(accel_mps2, alpha_e_vec_rad)`.
    """
    v = np.asarray(v_mps, dtype=np.float64)
    w = np.asarray(wind_mps, dtype=np.float64)
    vr = v - w
    v_mag = np.maximum(np.linalg.norm(vr, axis=-1, keepdims=True), 1.0)

    rho = np.asarray(rho_kg_m3, dtype=np.float64)[..., None]
    m_arr = np.maximum(np.asarray(m_kg, dtype=np.float64)[..., None], 1e-9)
    p_spin = np.asarray(p_spin_rad_s, dtype=np.float64)[..., None]
    i_arr = np.asarray(i_form, dtype=np.float64)[..., None]
    cd0_arr = np.asarray(cd0, dtype=np.float64)[..., None]

    g_vec = np.zeros_like(v)
    g_vec[..., 2] = -g0
    a_cor = eb_10_coriolis_acceleration(v, lat_rad=lat_rad, az_rad=az_rad)

    # First-order dv/dt for yaw-of-repose (note vr x vr = 0, so drag drops out of vr x dv/dt!)
    dv_dt_approx = g_vec + a_cor
    cross_v_dv = np.cross(vr, dv_dt_approx)
    denom_alpha = np.maximum(np.pi * rho * (d_m**3) * c_ma * (v_mag**4), 1e-12)
    alpha_e = -8.0 * ix_kg_m2 * p_spin * cross_v_dv / denom_alpha
    # Cap yaw of repose to physically reasonable regime (< 0.25 rad)
    alpha_mag = np.linalg.norm(alpha_e, axis=-1, keepdims=True)
    scale_cap = np.where(alpha_mag > 0.25, 0.25 / np.maximum(alpha_mag, 1e-12), 1.0)
    alpha_e = alpha_e * scale_cap
    alpha_sq = np.sum(alpha_e**2, axis=-1, keepdims=True)

    k_base = np.pi * rho * (d_m**2) / (8.0 * m_arr)
    df = -k_base * i_arr * (cd0_arr + c_da2 * ((q_d**2) * alpha_sq)) * v_mag * vr
    lf = k_base * f_l * (c_la + c_la3 * alpha_sq) * (v_mag**2) * alpha_e
    mf = -(np.pi * rho * (d_m**3) * q_m * p_spin * c_mag / (8.0 * m_arr)) * np.cross(alpha_e, vr)

    accel = df + lf + mf + g_vec + a_cor
    return accel, alpha_e


@register_formula("EB-06")
def eb_06_spin_damping(
    p_spin_rad_s: float | np.ndarray,
    v_rel_mag_mps: float | np.ndarray,
    rho_kg_m3: float | np.ndarray,
    d_m: float,
    ix_kg_m2: float,
    c_spin: float = -0.015,
) -> float | np.ndarray:
    """EB-06: Axial spin damping rate dp/dt = (pi * rho * d^4 * |v| * C_spin / (8 * Ix)) * p."""
    c_sp = -abs(c_spin) if c_spin > 0 else c_spin
    dp_dt = (
        np.pi
        * np.asarray(rho_kg_m3, dtype=np.float64)
        * (d_m**4)
        * np.asarray(v_rel_mag_mps, dtype=np.float64)
        * c_sp
        / (8.0 * max(ix_kg_m2, 1e-12))
    ) * np.asarray(p_spin_rad_s, dtype=np.float64)
    return float(dp_dt) if np.ndim(dp_dt) == 0 else dp_dt


@register_formula("EB-07")
def eb_07_gyroscopic_stability(
    ix_kg_m2: float,
    iy_kg_m2: float,
    p_spin_rad_s: float | np.ndarray,
    rho_kg_m3: float | np.ndarray,
    d_m: float,
    v_rel_mps: float | np.ndarray,
    c_ma: float = 2.8,
) -> float | np.ndarray:
    """EB-07: Gyroscopic stability factor Sg = 2 * Ix^2 * p^2 / (pi * rho * Iy * d^3 * v^2 * C_Ma)."""
    denom = (
        np.pi
        * np.maximum(np.asarray(rho_kg_m3, dtype=np.float64), 1e-9)
        * max(iy_kg_m2, 1e-12)
        * (d_m**3)
        * np.maximum(np.asarray(v_rel_mps, dtype=np.float64) ** 2, 1e-6)
        * max(c_ma, 1e-6)
    )
    sg = 2.0 * (ix_kg_m2**2) * (np.asarray(p_spin_rad_s, dtype=np.float64) ** 2) / denom
    return float(sg) if np.ndim(sg) == 0 else sg


@register_formula("EB-08")
def eb_08_miller_twist_rule(
    m_grains: float,
    d_in: float,
    length_in: float,
    twist_in_per_turn: float,
    v_fps: float = 2800.0,
    h_ft: float = 0.0,
) -> float:
    """EB-08: Miller gyroscopic stability estimate for small-arms bullets (validated by V-11)."""
    t_cal = twist_in_per_turn / d_in
    l_cal = length_in / d_in
    sg_base = 30.0 * m_grains / ((t_cal**2) * (d_in**3) * l_cal * (1.0 + l_cal**2))
    v_corr = (v_fps / 2800.0) ** (1.0 / 3.0)
    alt_corr = float(np.exp(3.158e-5 * h_ft)) if h_ft != 0.0 else 1.0
    return float(sg_base * v_corr * alt_corr)


@register_formula("EB-09")
def eb_09_muzzle_spin_rate(
    v0_mps: float | np.ndarray,
    twist_cal_per_turn: float,
    d_m: float,
) -> float | np.ndarray:
    """EB-09: Initial axial spin rate p0 = 2 * pi * V0 / (n_cal * d) (rad/s)."""
    if twist_cal_per_turn == 0.0 or d_m <= 0.0:
        z = np.zeros_like(np.asarray(v0_mps, dtype=np.float64))
        return float(z) if np.ndim(v0_mps) == 0 else z
    res = 2.0 * np.pi * np.asarray(v0_mps, dtype=np.float64) / (twist_cal_per_turn * d_m)
    return float(res) if np.ndim(v0_mps) == 0 else res


@register_formula("EB-10")
def eb_10_coriolis_acceleration(
    v_mps: np.ndarray,
    lat_rad: float = 0.0,
    az_rad: float = 0.0,
    omega_earth: float = OMEGA_EARTH_RAD_S,
) -> np.ndarray:
    """EB-10: Coriolis acceleration a_c = -2 * (Omega x v) in right-handed (range, left, up) frame."""
    om_vec = np.array(
        [
            omega_earth * np.cos(lat_rad) * np.cos(az_rad),
            omega_earth * np.cos(lat_rad) * np.sin(az_rad),
            omega_earth * np.sin(lat_rad),
        ],
        dtype=np.float64,
    )
    return -2.0 * np.cross(om_vec, np.asarray(v_mps, dtype=np.float64))


@register_formula("EB-11")
def eb_11_vacuum_range(
    v0_mps: float | np.ndarray,
    theta_rad: float | np.ndarray,
    g0: float = G0,
) -> float | np.ndarray:
    """EB-11: Flat-Earth vacuum ballistic range R_vac = V0^2 * sin(2*theta) / g0."""
    res = (np.asarray(v0_mps, dtype=np.float64) ** 2) * np.sin(2.0 * np.asarray(theta_rad, dtype=np.float64)) / g0
    return float(res) if np.ndim(v0_mps) == 0 and np.ndim(theta_rad) == 0 else res


@register_formula("EB-12")
def eb_12_base_bleed_drag(
    cd0: float | np.ndarray,
    mach: float | np.ndarray,
    burning: bool | np.ndarray = True,
    dcd_max: float = 0.06,
    mdot_kg_s: float = 0.0,
    rho_kg_m3: float | np.ndarray = 1.225,
    v_mps: float | np.ndarray = 500.0,
    a_base_m2: float = 0.01,
) -> float | np.ndarray:
    """EB-12: Base-bleed / tracer base-drag reduction Cd = Cd0 - dCd_bb(I, M) while burning."""
    m_arr = np.asarray(mach, dtype=np.float64)
    # Supersonic/transonic base-drag reduction profile scaled by injection parameter if mdot supplied
    if mdot_kg_s > 0.0 and a_base_m2 > 0.0:
        inj = mdot_kg_s / np.maximum(
            np.asarray(rho_kg_m3, dtype=np.float64) * np.asarray(v_mps, dtype=np.float64) * a_base_m2,
            1e-9,
        )
        inj_factor = np.clip(inj / 0.005, 0.0, 1.5)
    else:
        inj_factor = 1.0
    mach_factor = np.where(m_arr >= 0.9, np.exp(-0.35 * np.maximum(m_arr - 1.1, 0.0)), 0.3)
    dcd = dcd_max * mach_factor * inj_factor
    cd_red = np.maximum(0.02, np.asarray(cd0, dtype=np.float64) - np.where(burning, dcd, 0.0))
    return float(cd_red) if np.ndim(cd0) == 0 and np.ndim(mach) == 0 and np.ndim(burning) == 0 else cd_red


@register_formula("EB-13")
def eb_13_rocket_boost(
    u_hat: np.ndarray,
    drag_force_N: np.ndarray,
    m_kg: float | np.ndarray,
    mdot_kg_s: float | np.ndarray,
    v_exhaust_mps: float,
    p_exit_Pa: float = 101325.0,
    p_amb_Pa: float | np.ndarray = 101325.0,
    a_exit_m2: float = 0.0,
    m0_kg: float | None = None,
    mf_kg: float | None = None,
    g0: float = G0,
) -> dict[str, Any]:
    """EB-13: Rocket boost thrust T = mdot*v_e + (p_e - p_a)*A_e, EOM, and Tsiolkovsky dv_ideal."""
    md = np.asarray(mdot_kg_s, dtype=np.float64)
    thrust = md * v_exhaust_mps + (p_exit_Pa - np.asarray(p_amb_Pa, dtype=np.float64)) * a_exit_m2
    thrust = np.where(md > 0.0, thrust, 0.0)
    m_arr = np.maximum(np.asarray(m_kg, dtype=np.float64), 1e-9)
    g_vec = np.zeros_like(u_hat, dtype=np.float64)
    g_vec[..., 2] = -g0
    dv_dt = (thrust[..., None] * u_hat - drag_force_N) / m_arr[..., None] + g_vec
    isp_s = v_exhaust_mps / g0
    dv_ideal = float(v_exhaust_mps * np.log(m0_kg / mf_kg)) if (m0_kg and mf_kg and m0_kg > mf_kg) else 0.0
    return {
        "thrust_N": thrust,
        "dv_dt": dv_dt,
        "dm_dt": -md,
        "isp_s": isp_s,
        "dv_ideal_mps": dv_ideal,
    }


@register_formula("EB-14")
def eb_14_free_fall_bomb(
    release_alt_m: float | np.ndarray,
    tas_mps: float | np.ndarray,
    dive_angle_rad: float | np.ndarray = 0.0,
    eject_vel_down_mps: float | np.ndarray = 0.0,
    cross_vel_left_mps: float | np.ndarray = 0.0,
) -> np.ndarray:
    """EB-14: Aircraft bomb release initial state [x, y_left, z, vx, vy_left, vz]."""
    alt = np.asarray(release_alt_m, dtype=np.float64)
    tas = np.asarray(tas_mps, dtype=np.float64)
    dive = np.asarray(dive_angle_rad, dtype=np.float64)
    vej = np.asarray(eject_vel_down_mps, dtype=np.float64)
    vy = np.asarray(cross_vel_left_mps, dtype=np.float64)

    bshape = np.broadcast(alt, tas, dive, vej, vy).shape
    state = np.zeros(bshape + (6,), dtype=np.float64)
    state[..., 2] = alt
    state[..., 3] = tas * np.cos(dive)
    state[..., 4] = vy
    state[..., 5] = -tas * np.sin(dive) - vej
    return state


@register_formula("EB-15")
def eb_15_stability_failure_branch(
    sg: float | np.ndarray,
    cd_normal: float | np.ndarray,
    cd_tumble: float = 1.25,
) -> tuple[float | np.ndarray, bool | np.ndarray]:
    """EB-15: Stability & large-yaw failure branch (Sg < 1 -> tumble drag & unstable flag)."""
    sg_arr = np.asarray(sg, dtype=np.float64)
    unstable = sg_arr < 1.0
    cd_eff = np.where(unstable, np.maximum(cd_normal, cd_tumble), cd_normal)
    if np.ndim(sg) == 0 and np.ndim(cd_normal) == 0:
        return float(cd_eff), bool(unstable)
    return cd_eff, unstable


@register_formula("EB-16")
def eb_16_fragment_trajectory(
    m_frag_kg: np.ndarray | float,
    v0_mps: np.ndarray | float,
    elev_rad: np.ndarray | float,
    z0_m: float = 1.0,
    rho_m_kg_m3: float = 7850.0,
    shape_number_sn: float = 2.0,
    cd_scale: np.ndarray | float = 1.0,
    rho_air_sl: float = 1.225,
    dt: float = 0.01,
    max_time_s: float = 120.0,
) -> dict[str, np.ndarray | float]:
    """EB-16: 2-D natural fragment trajectory integration per IATG 01.80 5.5.1.

    Vectorised across an arbitrary number of fragments simultaneously.
    """
    kb = get_kb()
    m_arr = np.atleast_1d(np.asarray(m_frag_kg, dtype=np.float64))
    v0_arr = np.broadcast_to(np.asarray(v0_mps, dtype=np.float64), m_arr.shape)
    el_arr = np.broadcast_to(np.asarray(elev_rad, dtype=np.float64), m_arr.shape)
    cds_arr = np.broadcast_to(np.asarray(cd_scale, dtype=np.float64), m_arr.shape)

    n = m_arr.size
    x = np.zeros(n, dtype=np.float64)
    z = np.full(n, max(float(z0_m), 0.05), dtype=np.float64)
    u = v0_arr * np.cos(el_arr)
    v = v0_arr * np.sin(el_arr)
    t_arr = np.zeros(n, dtype=np.float64)

    prefactor = shape_number_sn / (2.0 * (rho_m_kg_m3 ** (2.0 / 3.0)) * np.maximum(m_arr, 1e-9) ** (1.0 / 3.0))

    def deriv(
        z_cur: np.ndarray,
        u_cur: np.ndarray,
        v_cur: np.ndarray,
        pref_cur: np.ndarray,
        cds_cur: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        speed = np.hypot(u_cur, v_cur)
        # Local ISA density & sound speed
        t_k = np.where(z_cur <= 11000.0, 288.15 - 0.0065 * np.maximum(z_cur, 0.0), 216.65)
        rho_a = rho_air_sl * np.power(np.maximum(t_k, 150.0) / 288.15, 4.25588)
        a_snd = np.sqrt(1.4 * 287.05287 * np.maximum(t_k, 150.0))
        mach = speed / a_snd
        cd_f = kb.interp_cd(mach, drag_ref="FRAGMENT") * cds_cur
        kap = pref_cur * rho_a * cd_f
        return -kap * speed * u_cur, -kap * speed * v_cur - G0

    active = np.ones(n, dtype=bool)
    x_imp = np.zeros(n, dtype=np.float64)
    v_imp = np.hypot(u, v)
    t_imp = np.zeros(n, dtype=np.float64)

    n_steps = int(max_time_s / dt)
    for _ in range(n_steps):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        za, ua, va, xa, ta = z[idx], u[idx], v[idx], x[idx], t_arr[idx]
        pref_a = prefactor[idx]
        cds_a = cds_arr[idx]

        du1, dv1 = deriv(za, ua, va, pref_a, cds_a)
        du2, dv2 = deriv(za + 0.5 * dt * va, ua + 0.5 * dt * du1, va + 0.5 * dt * dv1, pref_a, cds_a)
        du3, dv3 = deriv(za + 0.5 * dt * (va + 0.5 * dt * dv1), ua + 0.5 * dt * du2, va + 0.5 * dt * dv2, pref_a, cds_a)
        du4, dv4 = deriv(za + dt * (va + 0.5 * dt * dv2), ua + dt * du3, va + dt * dv3, pref_a, cds_a)

        x_next = xa + (dt / 6.0) * (ua + 2.0 * (ua + 0.5 * dt * du1) + 2.0 * (ua + 0.5 * dt * du2) + (ua + dt * du3))
        z_next = za + (dt / 6.0) * (va + 2.0 * (va + 0.5 * dt * dv1) + 2.0 * (va + 0.5 * dt * dv2) + (va + dt * dv3))
        u_next = ua + (dt / 6.0) * (du1 + 2.0 * du2 + 2.0 * du3 + du4)
        v_next = va + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        t_next = ta + dt

        hit = (z_next <= 0.0) & (t_next > 1e-6)
        if np.any(hit):
            h_idx = idx[hit]
            frac = za[hit] / np.maximum(za[hit] - z_next[hit], 1e-12)
            x_imp[h_idx] = xa[hit] + frac * (x_next[hit] - xa[hit])
            uh = ua[hit] + frac * (u_next[hit] - ua[hit])
            vh = va[hit] + frac * (v_next[hit] - va[hit])
            v_imp[h_idx] = np.hypot(uh, vh)
            t_imp[h_idx] = ta[hit] + frac * dt
            active[h_idx] = False

        still = ~hit
        if np.any(still):
            s_idx = idx[still]
            x[s_idx] = x_next[still]
            z[s_idx] = z_next[still]
            u[s_idx] = u_next[still]
            v[s_idx] = v_next[still]
            t_arr[s_idx] = t_next[still]

    # Any still airborne at max_time_s
    if np.any(active):
        x_imp[active] = x[active]
        v_imp[active] = np.hypot(u[active], v[active])
        t_imp[active] = t_arr[active]

    ke_imp = 0.5 * m_arr * (v_imp**2)
    if np.ndim(m_frag_kg) == 0 and np.ndim(v0_mps) == 0 and np.ndim(elev_rad) == 0:
        return {
            "range_m": float(x_imp[0]),
            "v_impact_mps": float(v_imp[0]),
            "ke_impact_J": float(ke_imp[0]),
            "tof_s": float(t_imp[0]),
        }
    return {
        "range_m": x_imp,
        "v_impact_mps": v_imp,
        "ke_impact_J": ke_imp,
        "tof_s": t_imp,
    }


@register_formula("EB-17")
def eb_17_fragment_velocity_decay(
    v0_mps: float | np.ndarray,
    R_m: float | np.ndarray,
    m_kg: float | np.ndarray,
    A_p_m2: float | np.ndarray,
    cd: float | np.ndarray = 1.29,
    rho_air_kg_m3: float | np.ndarray = 1.225,
) -> float | np.ndarray:
    """EB-17: Straight-line fragment velocity decay V(R) = V0 * exp(-Cd * rho_a * A_p * R / (2 * m))."""
    exponent = (
        -np.asarray(cd, dtype=np.float64)
        * np.asarray(rho_air_kg_m3, dtype=np.float64)
        * np.asarray(A_p_m2, dtype=np.float64)
        * np.asarray(R_m, dtype=np.float64)
        / np.maximum(2.0 * np.asarray(m_kg, dtype=np.float64), 1e-12)
    )
    res = np.asarray(v0_mps, dtype=np.float64) * np.exp(exponent)
    return float(res) if np.ndim(res) == 0 else res


# ---------------------------------------------------------------------------
# High-Performance Vectorised Trajectory Integrator (L1 Point-Mass & L2 MPM)
# ---------------------------------------------------------------------------

@dataclass
class BatchTrajectoryConfig:
    calibre_m: float
    drag_ref: str = "G7"
    custom_cd_table: list[dict[str, float]] | None = None
    fidelity: str = "L1"  # "L1" (point mass) or "L2" (STANAG 4355 MPM)
    ix_kg_m2: float = 0.0
    iy_kg_m2: float = 0.0
    twist_cal_per_turn: float = 0.0
    c_da2: float = 4.0
    c_la: float = 2.2
    c_ma: float = 2.8
    c_mag: float = 0.25
    c_spin: float = -0.015
    lat_rad: float = 0.0
    az_rad: float = 0.0
    include_coriolis: bool = True
    vacuum: bool = False
    # Base bleed / tracer (EB-12)
    base_bleed_burn_time_s: float = 0.0
    base_bleed_dcd: float = 0.0
    # Rocket boost (EB-13)
    rocket_burn_time_s: float = 0.0
    rocket_mdot_kg_s: float = 0.0
    rocket_ve_mps: float = 0.0
    # Numerics
    dt: float = 0.01
    max_time_s: float = 180.0
    integrator: str = "RK4"  # "RK4" or "RK45"
    rtol: float = 1e-6


def integrate_trajectories_batch(
    initial_states: np.ndarray,
    masses_kg: np.ndarray,
    form_factors: np.ndarray,
    env: AtmosphereEnvironment,
    cfg: BatchTrajectoryConfig,
    fuze_times_s: np.ndarray | None = None,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    record_single_path: bool = False,
    record_n_paths: int = 0,
    trial_wind_offset: np.ndarray | None = None,
    trial_dT_offset: np.ndarray | None = None,
) -> dict[str, Any]:
    """Simultaneously integrate `N` 3-D trajectories to ground impact or time-fuze burst.

    `initial_states`: shape `[N, 6]` -> `[x, y_left, z, vx, vy_left, vz]`.
    Returns impact/burst positions (`range_m`, `deflection_m` [+ right], `alt_m`),
    striking speeds, angles of fall, times of flight, and stability flags.
    """
    kb = get_kb()
    record_formula_use("MC-07")
    if not cfg.vacuum:
        record_formula_use("EB-03")
        record_formula_use("EB-04")
        record_formula_use("EB-02")
    if cfg.fidelity.upper() == "L2":
        record_formula_use("EB-05")
        record_formula_use("EB-06")
        record_formula_use("EB-07")
        record_formula_use("EB-09")
        record_formula_use("EB-15")
    else:
        record_formula_use("EB-01")
    if cfg.include_coriolis and not cfg.vacuum:
        record_formula_use("EB-10")
    if cfg.base_bleed_burn_time_s > 0.0:
        record_formula_use("EB-12")
    if cfg.rocket_burn_time_s > 0.0 and cfg.rocket_mdot_kg_s > 0.0:
        record_formula_use("EB-13")

    states0 = np.atleast_2d(np.asarray(initial_states, dtype=np.float64)).copy()
    n = states0.shape[0]
    m_cur = np.broadcast_to(np.asarray(masses_kg, dtype=np.float64), (n,)).copy()
    i_form = np.broadcast_to(np.asarray(form_factors, dtype=np.float64), (n,)).copy()
    w_off_all = (
        np.broadcast_to(np.asarray(trial_wind_offset, dtype=np.float64), (n, 3)).copy()
        if trial_wind_offset is not None
        else np.zeros((n, 3), dtype=np.float64)
    )
    dt_off_all = (
        np.broadcast_to(np.asarray(trial_dT_offset, dtype=np.float64), (n,)).copy()
        if trial_dT_offset is not None
        else np.zeros(n, dtype=np.float64)
    )

    is_mpm = (cfg.fidelity.upper() == "L2") and (cfg.ix_kg_m2 > 0.0) and (cfg.twist_cal_per_turn != 0.0)
    v0_mag = np.linalg.norm(states0[:, 3:6], axis=1)
    p_spin = (
        np.asarray(eb_09_muzzle_spin_rate(v0_mag, cfg.twist_cal_per_turn, cfg.calibre_m), dtype=np.float64)
        if is_mpm
        else np.zeros(n, dtype=np.float64)
    )

    area_ref = 0.25 * np.pi * (cfg.calibre_m**2)
    om_vec = np.array(
        [
            OMEGA_EARTH_RAD_S * np.cos(cfg.lat_rad) * np.cos(cfg.az_rad),
            OMEGA_EARTH_RAD_S * np.cos(cfg.lat_rad) * np.sin(cfg.az_rad),
            OMEGA_EARTH_RAD_S * np.sin(cfg.lat_rad),
        ],
        dtype=np.float64,
    )
    ox, oy, oz = float(om_vec[0]), float(om_vec[1]), float(om_vec[2])
    has_wind_bands = bool(env.wind_bands)
    const_wind = np.array(env.constant_wind_mps, dtype=np.float64)[None, :]
    has_rocket = cfg.rocket_burn_time_s > 0.0 and cfg.rocket_mdot_kg_s > 0.0

    def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
        out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
        out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return out

    def ground_z(x_r: np.ndarray, y_l: np.ndarray) -> np.ndarray:
        if terrain_fn is None:
            return np.zeros_like(x_r)
        return terrain_fn(x_r, -y_l)

    def compute_rhs(
        t_cur: np.ndarray | float,
        y_vec: np.ndarray,
        m_sub: np.ndarray,
        i_sub: np.ndarray,
        w_off_sub: np.ndarray,
        dt_off_sub: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute state derivative `[M, 6 or 7]` and instantaneous Sg `[M]`."""
        pos = y_vec[:, 0:3]
        vel = y_vec[:, 3:6]
        m_rows = vel.shape[0]
        m_sub_eff = (
            np.maximum(m_sub - cfg.rocket_mdot_kg_s * np.minimum(t_cur, cfg.rocket_burn_time_s), 0.1 * m_sub)
            if has_rocket
            else m_sub
        )

        if cfg.vacuum:
            dy = np.zeros_like(y_vec)
            dy[:, 0:3] = vel
            dy[:, 5] = -G0
            if cfg.include_coriolis:
                vx_v, vy_v, vz_v = vel[:, 0], vel[:, 1], vel[:, 2]
                dy[:, 3] -= 2.0 * (oy * vz_v - oz * vy_v)
                dy[:, 4] -= 2.0 * (oz * vx_v - ox * vz_v)
                dy[:, 5] -= 2.0 * (ox * vy_v - oy * vx_v)
            return dy, np.full(m_rows, 2.0)

        alt = pos[:, 2]
        t_std = np.where(alt <= 11000.0, 288.15 - 0.0065 * alt, 216.65)
        t_k = np.maximum(t_std + env.dT_K + dt_off_sub, 50.0)
        p_pa = np.where(
            alt <= 11000.0,
            env.p_sl_Pa * np.power(np.maximum(t_std, 10.0) / 288.15, 5.25588),
            env.p_sl_Pa * ((216.65 / 288.15) ** 5.25588) * np.exp(-0.000157688 * (alt - 11000.0)),
        )
        rho = p_pa / (287.05287 * t_k)
        a_snd = np.sqrt(1.4 * 287.05287 * t_k)

        wind = (env.wind_at_altitude(alt) + w_off_sub) if has_wind_bands else (const_wind + w_off_sub)
        vr = vel - wind
        vr0, vr1, vr2 = vr[:, 0], vr[:, 1], vr[:, 2]
        v_rel_mag = np.maximum(np.sqrt(vr0 * vr0 + vr1 * vr1 + vr2 * vr2), 1e-6)
        mach = v_rel_mag / a_snd

        cd_ref = kb.interp_cd(mach, drag_ref=cfg.drag_ref, custom_table=cfg.custom_cd_table)
        if cfg.base_bleed_burn_time_s > 0.0:
            burning = t_cur <= cfg.base_bleed_burn_time_s
            mach_fac = np.where(mach >= 0.9, np.exp(-0.35 * np.maximum(mach - 1.1, 0.0)), 0.3)
            cd_ref = np.maximum(0.02, cd_ref - np.where(burning, cfg.base_bleed_dcd * mach_fac, 0.0))

        dy = np.empty_like(y_vec)
        dy[:, 0:3] = vel

        if is_mpm:
            g_vec = np.zeros_like(vel)
            g_vec[:, 2] = -G0
            if cfg.include_coriolis:
                vx_c, vy_c, vz_c = vel[:, 0], vel[:, 1], vel[:, 2]
                a_cor = np.empty_like(vel)
                a_cor[:, 0] = -2.0 * (oy * vz_c - oz * vy_c)
                a_cor[:, 1] = -2.0 * (oz * vx_c - ox * vz_c)
                a_cor[:, 2] = -2.0 * (ox * vy_c - oy * vx_c)
            else:
                a_cor = np.zeros_like(vel)

            p_sp = y_vec[:, 6]
            sg = (
                2.0
                * (cfg.ix_kg_m2**2)
                * (p_sp**2)
                / np.maximum(np.pi * rho * max(cfg.iy_kg_m2, 1e-9) * (cfg.calibre_m**3) * (v_rel_mag**2) * cfg.c_ma, 1e-12)
            )
            cd0_eff = np.where(sg < 1.0, np.maximum(cd_ref, 1.25), cd_ref)
            dv_approx = g_vec + a_cor
            cross_v_dv = _cross3(vr, dv_approx)
            denom_a = np.maximum(np.pi * rho * (cfg.calibre_m**3) * cfg.c_ma * (v_rel_mag**4), 1e-12)[:, None]
            alpha_e = -8.0 * cfg.ix_kg_m2 * p_sp[:, None] * cross_v_dv / denom_a
            a_mag = np.sqrt(
                alpha_e[:, 0] * alpha_e[:, 0] + alpha_e[:, 1] * alpha_e[:, 1] + alpha_e[:, 2] * alpha_e[:, 2]
            )[:, None]
            alpha_e = np.where(a_mag > 0.25, alpha_e * (0.25 / np.maximum(a_mag, 1e-12)), alpha_e)
            alpha_sq = (
                alpha_e[:, 0] * alpha_e[:, 0] + alpha_e[:, 1] * alpha_e[:, 1] + alpha_e[:, 2] * alpha_e[:, 2]
            )[:, None]

            k_b = (np.pi * rho * (cfg.calibre_m**2) / (8.0 * m_sub_eff))[:, None]
            df = -k_b * i_sub[:, None] * (cd0_eff[:, None] + cfg.c_da2 * alpha_sq) * v_rel_mag[:, None] * vr
            lf = k_b * cfg.c_la * (v_rel_mag[:, None] ** 2) * alpha_e
            mf = -(
                (np.pi * rho * (cfg.calibre_m**3) * p_sp * cfg.c_mag / (8.0 * m_sub_eff))[:, None]
                * _cross3(alpha_e, vr)
            )
            dy[:, 3:6] = df + lf + mf + g_vec + a_cor
            c_sp_neg = -abs(cfg.c_spin) if cfg.c_spin > 0 else cfg.c_spin
            dy[:, 6] = (np.pi * rho * (cfg.calibre_m**4) * v_rel_mag * c_sp_neg / (8.0 * cfg.ix_kg_m2)) * p_sp
        else:
            sg = np.full(m_rows, 2.0)
            k_drag_v = (0.5 * area_ref) * (rho * i_sub * cd_ref / m_sub_eff) * v_rel_mag
            dy[:, 3] = -k_drag_v * vr0
            dy[:, 4] = -k_drag_v * vr1
            dy[:, 5] = -k_drag_v * vr2 - G0
            if cfg.include_coriolis:
                vx_c, vy_c, vz_c = vel[:, 0], vel[:, 1], vel[:, 2]
                dy[:, 3] -= 2.0 * (oy * vz_c - oz * vy_c)
                dy[:, 4] -= 2.0 * (oz * vx_c - ox * vz_c)
                dy[:, 5] -= 2.0 * (ox * vy_c - oy * vx_c)

        # Optional rocket thrust along velocity unit vector
        if has_rocket:
            thrust_on = t_cur <= cfg.rocket_burn_time_s
            if np.any(thrust_on):
                vx_t, vy_t, vz_t = vel[:, 0], vel[:, 1], vel[:, 2]
                v_mag_abs = np.maximum(np.sqrt(vx_t * vx_t + vy_t * vy_t + vz_t * vz_t), 1e-6)[:, None]
                u_hat = vel / v_mag_abs
                thrust_acc = np.where(
                    np.asarray(thrust_on)[..., None] if np.ndim(thrust_on) > 0 else thrust_on,
                    (cfg.rocket_mdot_kg_s * cfg.rocket_ve_mps / m_sub_eff)[:, None] * u_hat,
                    0.0,
                )
                dy[:, 3:6] += thrust_acc

        return dy, sg

    # Adaptive RK45 option per App_Spec R13
    if cfg.integrator.upper() == "RK45":
        end_states = np.zeros((n, 6), dtype=np.float64)
        tof = np.zeros(n, dtype=np.float64)
        unstable_any = np.zeros(n, dtype=bool)
        burst_by_fuze = np.zeros(n, dtype=bool)
        sample_paths: list[dict[str, np.ndarray]] = []
        n_rec = min(n, record_n_paths if record_n_paths > 0 else (1 if record_single_path else 0))

        for j in range(n):
            y0_j = np.append(states0[j], p_spin[j]) if is_mpm else states0[j]
            t_end_j = float(fuze_times_s[j]) if (fuze_times_s is not None and np.isfinite(fuze_times_s[j])) else cfg.max_time_s
            t_end_j = min(t_end_j, cfg.max_time_s)

            def ode_j(tj: float, yj: np.ndarray) -> np.ndarray:
                dy, sg_j = compute_rhs(
                    tj,
                    yj[None, :],
                    m_cur[j : j + 1],
                    i_form[j : j + 1],
                    w_off_all[j : j + 1],
                    dt_off_all[j : j + 1],
                )
                if sg_j[0] < 1.0:
                    unstable_any[j] = True
                return dy[0]

            def hit_ground(tj: float, yj: np.ndarray) -> float:
                if tj < 1e-4 and yj[5] >= 0.0:
                    return 1.0
                gz = float(ground_z(np.array([yj[0]]), np.array([yj[1]]))[0])
                return float(yj[2] - gz)

            hit_ground.terminal = True  # type: ignore[attr-defined]
            hit_ground.direction = -1.0  # type: ignore[attr-defined]

            sol = solve_ivp(
                ode_j,
                (0.0, t_end_j),
                y0_j,
                method="RK45",
                events=hit_ground,
                rtol=cfg.rtol,
                atol=1e-8,
                dense_output=(j < n_rec),
            )
            end_states[j] = sol.y[:6, -1]
            tof[j] = float(sol.t[-1])
            if fuze_times_s is not None and np.isfinite(fuze_times_s[j]) and len(sol.t_events[0]) == 0:
                burst_by_fuze[j] = True
            if j < n_rec:
                v_mag_j = np.linalg.norm(sol.y[3:6], axis=0)
                sample_paths.append({
                    "t": sol.t,
                    "range_m": sol.y[0],
                    "deflection_m": -sol.y[1],
                    "alt_m": sol.y[2],
                    "speed_mps": v_mag_j,
                })

        sample_path = sample_paths[0] if sample_paths else None
        v_end = end_states[:, 3:6]
        v_impact = np.linalg.norm(v_end, axis=1)
        horiz_speed = np.hypot(v_end[:, 0], v_end[:, 1])
        angle_of_fall_rad = np.arctan2(-v_end[:, 2], np.maximum(horiz_speed, 1e-9))
        return {
            "range_m": end_states[:, 0],
            "deflection_m": -end_states[:, 1],
            "alt_m": end_states[:, 2],
            "vx_mps": v_end[:, 0],
            "vy_right_mps": -v_end[:, 1],
            "vz_mps": v_end[:, 2],
            "v_impact_mps": v_impact,
            "angle_of_fall_rad": angle_of_fall_rad,
            "tof_s": tof,
            "unstable": unstable_any,
            "burst_in_air": burst_by_fuze,
            "sample_path": sample_path,
            "sample_paths": sample_paths,
        }

    # Vectorised fixed-step RK4 across all N trials simultaneously
    y = np.column_stack([states0, p_spin]) if is_mpm else states0.copy()
    t = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    end_states = states0.copy()
    tof = np.zeros(n, dtype=np.float64)
    unstable_any = np.zeros(n, dtype=bool)
    burst_by_fuze = np.zeros(n, dtype=bool)

    dt = float(cfg.dt)
    n_steps = int(np.ceil(cfg.max_time_s / dt))
    n_rec = min(n, record_n_paths if record_n_paths > 0 else (1 if record_single_path else 0))
    paths_t: list[list[float]] = [[0.0] for _ in range(n_rec)]
    paths_x: list[list[float]] = [[float(states0[k, 0])] for k in range(n_rec)]
    paths_d: list[list[float]] = [[float(-states0[k, 1])] for k in range(n_rec)]
    paths_z: list[list[float]] = [[float(states0[k, 2])] for k in range(n_rec)]
    paths_v: list[list[float]] = [[float(np.linalg.norm(states0[k, 3:6]))] for k in range(n_rec)]

    for _ in range(n_steps):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        ya = y[idx]
        ta = t[idx]
        ma = m_cur[idx]
        ia = i_form[idx]
        wa = w_off_all[idx]
        dta = dt_off_all[idx]

        k1, sg1 = compute_rhs(ta, ya, ma, ia, wa, dta)
        unstable_any[idx] |= sg1 < 1.0
        k2, _ = compute_rhs(ta + 0.5 * dt, ya + 0.5 * dt * k1, ma, ia, wa, dta)
        k3, _ = compute_rhs(ta + 0.5 * dt, ya + 0.5 * dt * k2, ma, ia, wa, dta)
        k4, _ = compute_rhs(ta + dt, ya + dt * k3, ma, ia, wa, dta)

        y_next = ya + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_next = ta + dt

        # Check time-fuze event first if configured
        fuze_trig = np.zeros(idx.size, dtype=bool)
        if fuze_times_s is not None:
            tf_sub = fuze_times_s[idx]
            fuze_trig = np.isfinite(tf_sub) & (t_next >= tf_sub)

        # Check ground impact event (after clearing launch pad or descending)
        gz_prev = ground_z(ya[:, 0], ya[:, 1])
        gz_next = ground_z(y_next[:, 0], y_next[:, 1])
        h_prev = ya[:, 2] - gz_prev
        h_next = y_next[:, 2] - gz_next
        ground_hit = (h_next <= 0.0) & ((t_next > 2.0 * dt) | (y_next[:, 5] < 0.0))

        # Resolve which event happened earlier within [ta, t_next]
        frac_ground = np.where(ground_hit, h_prev / np.maximum(h_prev - h_next, 1e-12), 2.0)
        frac_ground = np.clip(frac_ground, 0.0, 1.0)
        if fuze_times_s is not None:
            tf_sub = fuze_times_s[idx]
            frac_fuze = np.where(fuze_trig, (tf_sub - ta) / dt, 2.0)
            frac_fuze = np.clip(frac_fuze, 0.0, 1.0)
        else:
            frac_fuze = np.full(idx.size, 2.0)

        done_local = ground_hit | fuze_trig
        if np.any(done_local):
            d_pos = np.flatnonzero(done_local)
            g_idx = idx[d_pos]
            use_fuze = fuze_trig[d_pos] & ((~ground_hit[d_pos]) | (frac_fuze[d_pos] <= frac_ground[d_pos]))
            frac = np.where(use_fuze, frac_fuze[d_pos], frac_ground[d_pos])
            y_interp = ya[d_pos, :6] + frac[:, None] * (y_next[d_pos, :6] - ya[d_pos, :6])
            end_states[g_idx] = y_interp
            tof[g_idx] = ta[d_pos] + frac * dt
            burst_by_fuze[g_idx] = use_fuze
            active[g_idx] = False

        still = ~done_local
        if np.any(still):
            s_idx = idx[still]
            y[s_idx] = y_next[still]
            t[s_idx] = t_next[still]

        if n_rec > 0:
            for k in range(n_rec):
                if active[k]:
                    paths_t[k].append(float(t[k]))
                    paths_x[k].append(float(y[k, 0]))
                    paths_d[k].append(float(-y[k, 1]))
                    paths_z[k].append(float(y[k, 2]))
                    paths_v[k].append(float(np.linalg.norm(y[k, 3:6])))
                elif len(paths_t[k]) > 0 and paths_t[k][-1] < tof[k]:
                    paths_t[k].append(float(tof[k]))
                    paths_x[k].append(float(end_states[k, 0]))
                    paths_d[k].append(float(-end_states[k, 1]))
                    paths_z[k].append(float(end_states[k, 2]))
                    paths_v[k].append(float(np.linalg.norm(end_states[k, 3:6])))

    if np.any(active):
        end_states[active] = y[active, :6]
        tof[active] = t[active]

    v_end = end_states[:, 3:6]
    v_impact = np.linalg.norm(v_end, axis=1)
    horiz_speed = np.hypot(v_end[:, 0], v_end[:, 1])
    angle_of_fall_rad = np.arctan2(-v_end[:, 2], np.maximum(horiz_speed, 1e-9))

    sample_paths = [
        {
            "t": np.array(paths_t[k]),
            "range_m": np.array(paths_x[k]),
            "deflection_m": np.array(paths_d[k]),
            "alt_m": np.array(paths_z[k]),
            "speed_mps": np.array(paths_v[k]),
        }
        for k in range(n_rec)
    ]
    sample_path = sample_paths[0] if sample_paths else None

    return {
        "range_m": end_states[:, 0],
        "deflection_m": -end_states[:, 1],
        "alt_m": end_states[:, 2],
        "vx_mps": v_end[:, 0],
        "vy_right_mps": -v_end[:, 1],
        "vz_mps": v_end[:, 2],
        "v_impact_mps": v_impact,
        "angle_of_fall_rad": angle_of_fall_rad,
        "tof_s": tof,
        "unstable": unstable_any,
        "burst_in_air": burst_by_fuze,
        "sample_path": sample_path,
        "sample_paths": sample_paths,
    }


def solve_quadrant_elevation(
    target_range_m: float,
    mv_mps: float,
    mass_kg: float,
    env: AtmosphereEnvironment,
    cfg: BatchTrajectoryConfig,
    high_angle: bool = False,
    z0_m: float = 0.0,
) -> float:
    """Solve for launch Quadrant Elevation (radians) that achieves `target_range_m`."""
    # Sample 28 candidate elevations to bracket peak range cleanly
    thetas = np.linspace(0.002, 1.45, 28)
    states = np.zeros((len(thetas), 6), dtype=np.float64)
    states[:, 2] = z0_m
    states[:, 3] = mv_mps * np.cos(thetas)
    states[:, 5] = mv_mps * np.sin(thetas)

    fast_cfg = BatchTrajectoryConfig(
        calibre_m=cfg.calibre_m,
        drag_ref=cfg.drag_ref,
        custom_cd_table=cfg.custom_cd_table,
        fidelity=cfg.fidelity,
        ix_kg_m2=cfg.ix_kg_m2,
        iy_kg_m2=cfg.iy_kg_m2,
        twist_cal_per_turn=cfg.twist_cal_per_turn,
        lat_rad=cfg.lat_rad,
        az_rad=cfg.az_rad,
        include_coriolis=cfg.include_coriolis,
        vacuum=cfg.vacuum,
        base_bleed_burn_time_s=cfg.base_bleed_burn_time_s,
        base_bleed_dcd=cfg.base_bleed_dcd,
        rocket_burn_time_s=cfg.rocket_burn_time_s,
        rocket_mdot_kg_s=cfg.rocket_mdot_kg_s,
        rocket_ve_mps=cfg.rocket_ve_mps,
        dt=max(cfg.dt, 0.02),
        max_time_s=cfg.max_time_s,
        integrator="RK4",
    )
    res = integrate_trajectories_batch(states, np.full(len(thetas), mass_kg), np.ones(len(thetas)), env, fast_cfg)
    ranges = res["range_m"]
    idx_peak = int(np.argmax(ranges))
    if target_range_m >= ranges[idx_peak]:
        return float(thetas[idx_peak])

    if high_angle and idx_peak < len(thetas) - 1:
        sub_th = thetas[idx_peak:]
        sub_r = ranges[idx_peak:]
        # Monotonically decreasing with theta -> flip for interp
        qe_est = float(np.interp(target_range_m, sub_r[::-1], sub_th[::-1]))
    else:
        sub_th = thetas[: idx_peak + 1]
        sub_r = ranges[: idx_peak + 1]
        qe_est = float(np.interp(target_range_m, sub_r, sub_th))

    # Two secant/Newton refinement iterations at exact dt
    th0 = qe_est
    th1 = qe_est + (0.002 if not high_angle else -0.002)
    for _ in range(4):
        st_pair = np.zeros((2, 6), dtype=np.float64)
        st_pair[:, 2] = z0_m
        st_pair[0, 3], st_pair[0, 5] = mv_mps * np.cos(th0), mv_mps * np.sin(th0)
        st_pair[1, 3], st_pair[1, 5] = mv_mps * np.cos(th1), mv_mps * np.sin(th1)
        r_pair = integrate_trajectories_batch(st_pair, np.full(2, mass_kg), np.ones(2), env, cfg)["range_m"]
        err1 = float(r_pair[1] - target_range_m)
        if abs(err1) < 0.25:
            return float(th1)
        dr = float(r_pair[1] - r_pair[0])
        if abs(dr) < 1e-6:
            break
        th_next = float(np.clip(th1 - err1 * (th1 - th0) / dr, 0.0005, 1.50))
        th0, th1 = th1, th_next
    return float(th1)


def simulate_mpm_vs_3dof_trajectory(
    v0_mps: float = 827.0,
    qe_deg: float = 35.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.2,
    twist_cal_per_turn: float = 20.0,
    form_factor: float = 1.0,
    drag_ref: str = "G7",
    lat_deg: float = 45.0,
    az_deg: float = 90.0,
    include_coriolis: bool = True,
    c_ma: float = 2.8,
    c_la: float = 2.2,
    c_mag: float = 0.25,
    c_spin: float = -0.015,
) -> dict[str, Any]:
    """Compare L1 (3-DoF Point-Mass) vs L2 (STANAG 4355 Modified Point-Mass) trajectories.

    Returns full flight histories including yaw-of-repose magnitude |alpha_e| (mrad),
    axial spin decay p(t) (rad/s), gyroscopic stability factor Sg(t), and pure spin drift (m).
    """
    qe_rad = np.radians(qe_deg)
    lat_rad = np.radians(lat_deg)
    az_rad = np.radians(az_deg)
    ix = 0.135 * mass_kg * (calibre_m**2)
    iy = 1.45 * mass_kg * (calibre_m**2)

    state0 = np.array([[0.0, 0.0, 0.0, v0_mps * np.cos(qe_rad), 0.0, v0_mps * np.sin(qe_rad)]], dtype=np.float64)
    env = AtmosphereEnvironment()

    cfg_l1 = BatchTrajectoryConfig(
        calibre_m=calibre_m,
        drag_ref=drag_ref,
        fidelity="L1",
        lat_rad=lat_rad,
        az_rad=az_rad,
        include_coriolis=include_coriolis,
        dt=0.02,
    )
    cfg_l2 = BatchTrajectoryConfig(
        calibre_m=calibre_m,
        drag_ref=drag_ref,
        fidelity="L2",
        ix_kg_m2=ix,
        iy_kg_m2=iy,
        twist_cal_per_turn=twist_cal_per_turn,
        c_la=c_la,
        c_ma=c_ma,
        c_mag=c_mag,
        c_spin=c_spin,
        lat_rad=lat_rad,
        az_rad=az_rad,
        include_coriolis=include_coriolis,
        dt=0.02,
    )

    res_l1 = integrate_trajectories_batch(state0, np.array([mass_kg]), np.array([form_factor]), env, cfg_l1, record_single_path=True)
    res_l2 = integrate_trajectories_batch(state0, np.array([mass_kg]), np.array([form_factor]), env, cfg_l2, record_single_path=True)

    p1 = res_l1["sample_path"]
    p2 = res_l2["sample_path"]
    t2 = np.asarray(p2["t"], dtype=np.float64)
    r2 = np.asarray(p2["range_m"], dtype=np.float64)
    d2 = np.asarray(p2["deflection_m"], dtype=np.float64)
    z2 = np.asarray(p2["alt_m"], dtype=np.float64)
    v2 = np.asarray(p2["speed_mps"], dtype=np.float64)

    # Interpolate L1 deflection onto L2 range grid to isolate pure gyroscopic + Magnus spin drift
    r1 = np.asarray(p1["range_m"], dtype=np.float64)
    d1 = np.asarray(p1["deflection_m"], dtype=np.float64)
    d1_on_r2 = np.interp(r2, r1, d1)
    spin_drift_m = d2 - d1_on_r2

    # Reconstruct axial spin rate p(t), dynamic stability Sg(t), and yaw of repose |alpha_e|(t)
    atm = eb_04_isa_atmosphere(z2)
    rho = np.asarray(atm["rho_kg_m3"], dtype=np.float64)
    p0_spin = float(eb_09_muzzle_spin_rate(v0_mps, twist_cal_per_turn, calibre_m))
    dt_arr = np.diff(t2, prepend=0.0)
    c_sp_neg = -abs(c_spin) if c_spin > 0 else c_spin
    decay_rate = np.pi * rho * (calibre_m**4) * v2 * c_sp_neg / (8.0 * ix)
    p_spin_t = p0_spin * np.exp(np.cumsum(decay_rate * dt_arr))
    sg_t = np.asarray(eb_07_gyroscopic_stability(ix, iy, p_spin_t, rho, calibre_m, v2, c_ma=c_ma), dtype=np.float64)

    # Estimate local flight path angle gamma(t) for |v x g| = v * g * cos(gamma)
    dr = np.gradient(r2, np.maximum(t2, 1e-6))
    cos_gam = np.clip(dr / np.maximum(v2, 1.0), 0.05, 1.0)
    alpha_e_rad = (8.0 * ix * np.abs(p_spin_t) * G0 * cos_gam) / np.maximum(
        np.pi * rho * (calibre_m**3) * c_ma * (v2**3), 1e-9
    )
    alpha_e_mrad = np.minimum(alpha_e_rad, 0.25) * 1000.0

    return {
        "l1_range_m": float(res_l1["range_m"][0]),
        "l2_range_m": float(res_l2["range_m"][0]),
        "l1_deflection_m": float(res_l1["deflection_m"][0]),
        "l2_deflection_m": float(res_l2["deflection_m"][0]),
        "terminal_spin_drift_m": float(res_l2["deflection_m"][0] - res_l1["deflection_m"][0]),
        "tof_l2_s": float(res_l2["tof_s"][0]),
        "p0_spin_rad_s": p0_spin,
        "p_impact_rad_s": float(p_spin_t[-1]),
        "sg_muzzle": float(sg_t[0]),
        "sg_min": float(np.min(sg_t)),
        "alpha_e_peak_mrad": float(np.max(alpha_e_mrad)),
        "path_l1": p1,
        "path_l2": p2,
        "t_s": t2,
        "range_m": r2,
        "spin_drift_m": spin_drift_m,
        "p_spin_rad_s": p_spin_t,
        "sg_t": sg_t,
        "alpha_e_mrad": alpha_e_mrad,
    }


def generate_ballistic_firing_table(
    v0_mps: float = 827.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.2,
    twist_cal_per_turn: float = 20.0,
    form_factor: float = 1.0,
    drag_ref: str = "G7",
    fidelity: str = "L2",
    lat_deg: float = 45.0,
    az_deg: float = 90.0,
    qe_min_deg: float = 5.0,
    qe_max_deg: float = 65.0,
    n_elevations: int = 13,
    rocket_burn_time_s: float = 0.0,
    rocket_mdot_kg_s: float = 0.0,
    rocket_ve_mps: float = 0.0,
    base_bleed_burn_time_s: float = 0.0,
    base_bleed_dcd: float = 0.0,
) -> list[dict[str, float]]:
    """Generate a tabular Ballistic Firing Table (FM 6-40 / TC 3-09.81 / NATO S4 NABK) with Met Unit-Effects.

    Supports both gun-launched projectiles and solid-propellant rockets (EB-13 boost-coast).
    Uses a single vectorised batch integration of 6 * n_elevations trajectories to evaluate
    nominal ballistic trajectories and finite-difference unit partial derivatives:
    - dR / dV0 (m per +1 m/s muzzle/launch velocity)
    - dR / dWx (m per +1 m/s range tailwind)
    - dD / dWz (m per +1 m/s right crosswind)
    - dR / dT  (m per +1 K air temperature deviation)
    - dR / dFF (m per +1% ballistic form factor change)
    """
    qe_degs = np.linspace(qe_min_deg, qe_max_deg, n_elevations)
    qe_rads = np.radians(qe_degs)
    mils_per_deg = 6400.0 / 360.0

    # Stack 6 blocks of n_elevations states:
    # 0: Nominal
    # 1: V0 + 1.0 m/s
    # 2: Range tailwind Wx = +1.0 m/s
    # 3: Right crosswind Wz = +1.0 m/s (in RLU frame, y_left = -1.0 m/s)
    # 4: Air temp dT = +1.0 K
    # 5: Form factor FF * 1.01 (+1%)
    n_el = len(qe_rads)
    states_list = []
    masses_all = np.full(6 * n_el, mass_kg, dtype=np.float64)
    ff_all = np.full(6 * n_el, form_factor, dtype=np.float64)
    w_off_all = np.zeros((6 * n_el, 3), dtype=np.float64)
    dt_off_all = np.zeros(6 * n_el, dtype=np.float64)

    for blk in range(6):
        v_blk = v0_mps + (1.0 if blk == 1 else 0.0)
        st_b = np.zeros((n_el, 6), dtype=np.float64)
        st_b[:, 3] = v_blk * np.cos(qe_rads)
        st_b[:, 5] = v_blk * np.sin(qe_rads)
        states_list.append(st_b)

    states_all = np.vstack(states_list)
    w_off_all[2 * n_el : 3 * n_el, 0] = 1.0  # +1 m/s tailwind
    w_off_all[3 * n_el : 4 * n_el, 1] = -1.0  # +1 m/s crosswind from left (blowing right)
    dt_off_all[4 * n_el : 5 * n_el] = 1.0  # +1 K temperature
    ff_all[5 * n_el : 6 * n_el] = form_factor * 1.01  # +1% drag form factor

    ix = 0.135 * mass_kg * (calibre_m**2)
    iy = 1.45 * mass_kg * (calibre_m**2)
    eff_fidelity = "L1" if (rocket_burn_time_s > 0.0 or twist_cal_per_turn <= 0.0) else fidelity
    cfg = BatchTrajectoryConfig(
        calibre_m=calibre_m,
        drag_ref=drag_ref,
        fidelity=eff_fidelity,
        ix_kg_m2=ix,
        iy_kg_m2=iy,
        twist_cal_per_turn=twist_cal_per_turn,
        lat_rad=float(np.radians(lat_deg)),
        az_rad=float(np.radians(az_deg)),
        include_coriolis=True,
        rocket_burn_time_s=rocket_burn_time_s,
        rocket_mdot_kg_s=rocket_mdot_kg_s,
        rocket_ve_mps=rocket_ve_mps,
        base_bleed_burn_time_s=base_bleed_burn_time_s,
        base_bleed_dcd=base_bleed_dcd,
        dt=0.025,
    )
    env = AtmosphereEnvironment()
    out = integrate_trajectories_batch(
        states_all,
        masses_all,
        ff_all,
        env,
        cfg,
        record_n_paths=n_el,
        trial_wind_offset=w_off_all,
        trial_dT_offset=dt_off_all,
    )

    rng = out["range_m"].reshape(6, n_el)
    defl = out["deflection_m"].reshape(6, n_el)
    v_imp = out["v_impact_mps"][:n_el]
    aof_deg = np.degrees(out["angle_of_fall_rad"][:n_el])
    tof_s = out["tof_s"][:n_el]
    paths = out.get("sample_paths") or []

    rows: list[dict[str, float]] = []
    for k in range(n_el):
        max_ord = float(np.max(paths[k]["alt_m"])) if k < len(paths) and len(paths[k]["alt_m"]) > 0 else 0.0
        rows.append(
            {
                "QE (deg)": round(float(qe_degs[k]), 1),
                "QE (mils)": round(float(qe_degs[k] * mils_per_deg), 1),
                "Range (m)": round(float(rng[0, k]), 1),
                "Drift / Defl (m)": round(float(defl[0, k]), 1),
                "Vertex Alt (m)": round(max_ord, 1),
                "TOF (s)": round(float(tof_s[k]), 2),
                "V_impact (m/s)": round(float(v_imp[k]), 1),
                "Angle of Fall (deg)": round(float(aof_deg[k]), 1),
                "∂R/∂V₀ (m / m/s)": round(float(rng[1, k] - rng[0, k]), 2),
                "∂R/∂W_x (m / m/s)": round(float(rng[2, k] - rng[0, k]), 2),
                "∂D/∂W_z (m / m/s)": round(float(defl[3, k] - defl[0, k]), 2),
                "∂R/∂T_air (m / K)": round(float(rng[4, k] - rng[0, k]), 2),
                "∂R/∂FF (m / +1%)": round(float(rng[5, k] - rng[0, k]), 2),
            }
        )
    return rows
