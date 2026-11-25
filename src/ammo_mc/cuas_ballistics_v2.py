"""Deep C-UAS Exterior, Terminal, Recoil & Fire-Control Physics Kernels (v2.0.0).

Implements peer-reviewed sphere drag tables (Walsh 1976 NASA TN D-8120,
NOLTR 72-34, Compton 1996 lead deformation), CIPM-2007 moist air density
(Picard et al. 2008), SAAMI instrumental velocity back-correction and recoil gas
factor (fg = 1.50 / 1.25 + wad mass), Wen (2000) FRP composite perforation,
FAA shear-plugging V50 (with TE-C09 calibration-envelope caveat), Project THOR TR-47
metallic residual velocity/mass (with TE-C01 validity warnings), normal velocity
ogive P_eff(v), Markov redundant rotor/core kill trees, serial fire-control
dead-time chains, heavy-tailed ATR centroid noise, aspect-dependent RCS nulls,
and Singer correlated evasive maneuvers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr

G0_MPS2 = 9.80665
R_DRY_AIR_J_KG_K = 287.058

# -------------------------------------------------------------------------
# 1. AERODYNAMIC SPHERE DRAG TABLES & CIPM-2007 MOIST AIR DENSITY
# -------------------------------------------------------------------------

# Walsh (1976) NASA TN D-8120 Continuum Sphere Cd(M) (EB-T01)
WALSH_1976_MACH: np.ndarray = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0],
    dtype=np.float64,
)
WALSH_1976_CD: np.ndarray = np.array(
    [0.38, 0.39, 0.398, 0.41, 0.426, 0.443, 0.466, 0.5, 0.513, 0.54, 0.6, 0.71, 0.78, 0.82, 0.86, 0.89, 0.91, 0.92, 0.94, 0.94],
    dtype=np.float64,
)

# Krumins (1972) NOLTR 72-34 Table I Re=10000 Sphere Cd(M) (EB-T02)
NOL_1972_MACH: np.ndarray = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0],
    dtype=np.float64,
)
NOL_1972_CD_RE1E4: np.ndarray = np.array(
    [0.402, 0.41, 0.418, 0.429, 0.44, 0.455, 0.478, 0.523, 0.585, 0.71, 0.857, 0.927, 0.95, 0.959, 0.962, 0.971, 0.978],
    dtype=np.float64,
)

COMPTON_DEFORMATION_RATIO = 1.234  # EB-P04 trailing deformed lead / leading sphere ratio


def sphere_drag_walsh_1976(mach: float) -> float:
    """Walsh (1976) NASA TN D-8120 continuum sphere drag coefficient Cd(M)."""
    m = max(0.05, min(2.5, float(mach)))
    return float(np.interp(m, WALSH_1976_MACH, WALSH_1976_CD))


def sphere_drag_walsh_1976_array(mach_arr: np.ndarray) -> np.ndarray:
    """Vectorized Walsh (1976) NASA TN D-8120 continuum sphere Cd(M)."""
    m = np.clip(np.asarray(mach_arr, dtype=np.float64), 0.05, 2.5)
    return np.interp(m, WALSH_1976_MACH, WALSH_1976_CD)


def sphere_drag_nol_1972(mach: float) -> float:
    """Krumins (1972) NOLTR 72-34 Table I sphere Cd(M) at Re=10,000."""
    m = max(0.05, min(2.5, float(mach)))
    return float(np.interp(m, NOL_1972_MACH, NOL_1972_CD_RE1E4))


def sphere_drag_nol_1972_array(mach_arr: np.ndarray) -> np.ndarray:
    """Vectorized NOLTR 72-34 Table I sphere Cd(M)."""
    m = np.clip(np.asarray(mach_arr, dtype=np.float64), 0.05, 2.5)
    return np.interp(m, NOL_1972_MACH, NOL_1972_CD_RE1E4)


def sphere_drag_compton_1996(mach: float, trailing_lead: bool = False) -> float:
    """Compton (1996) piecewise sphere Cd(M) for undeformed leading vs deformed trailing lead."""
    m = max(0.0, float(mach))
    if m < 0.63:
        return 0.52 * (COMPTON_DEFORMATION_RATIO if trailing_lead else 1.0)
    slope = 1.01 if trailing_lead else 0.82
    return float(slope * m)


def sphere_drag_compton_1996_array(mach_arr: np.ndarray, trailing_lead: bool = False) -> np.ndarray:
    """Vectorized Compton (1996) leading/trailing sphere drag coefficient."""
    m = np.maximum(np.asarray(mach_arr, dtype=np.float64), 0.0)
    slope = 1.01 if trailing_lead else 0.82
    sub_val = 0.52 * (COMPTON_DEFORMATION_RATIO if trailing_lead else 1.0)
    return np.where(m < 0.63, sub_val, slope * m)


def cipm_2007_air_density(
    temp_c: float = 15.0,
    pressure_pa: float = 101325.0,
    rh_fraction: float = 0.0,
) -> dict[str, float]:
    """Compute moist-air density rho (kg/m^3) via Picard et al. (2008) CIPM-2007 (AT-F04)."""
    t = float(temp_c)
    p = max(1000.0, float(pressure_pa))
    h = max(0.0, min(1.0, float(rh_fraction)))
    t_k = t + 273.15

    # Saturation vapor pressure (Davis 1992 / CIPM-2007)
    p_sv = math.exp(
        1.2378847e-5 * (t_k ** 2)
        - 1.9121316e-2 * t_k
        + 33.93711047
        - 6.3431645e3 / t_k
    )
    # Enhancement factor f(p, t)
    f_enh = 1.00062 + 3.14e-8 * p + 5.6e-7 * (t ** 2)
    x_v = h * f_enh * p_sv / p

    # Compressibility factor Z
    p_over_t = p / t_k
    z_fac = (
        1.0
        - p_over_t
        * (
            1.58123e-6
            - 2.9331e-8 * t
            + 1.1043e-10 * (t ** 2)
            + (5.707e-6 - 2.051e-8 * t) * x_v
            + (1.9898e-4 - 2.376e-6 * t) * (x_v ** 2)
        )
        + (p_over_t ** 2) * (1.83e-11 - 0.765e-8 * (x_v ** 2))
    )

    rho_kg_m3 = 3.483740e-3 * (p / (z_fac * t_k)) * (1.0 - 0.3780 * x_v)
    a_mps = math.sqrt(1.4 * R_DRY_AIR_J_KG_K * t_k * (1.0 + 0.14 * x_v))
    return {
        "rho_kg_m3": float(rho_kg_m3),
        "a_mps": float(a_mps),
        "x_v": float(x_v),
        "z_factor": float(z_fac),
        "p_sv_pa": float(p_sv),
        "temp_k": float(t_k),
    }


def _eval_v2_drag_cd(
    v_rel_mps: float,
    a_sound_mps: float,
    drag_law: str = "walsh_1976",
    diam_m: float = 0.0033,
    rho_kg_m3: float = 1.225,
    mu_pa_s: float = 1.789e-5,
) -> float:
    mach = v_rel_mps / max(a_sound_mps, 100.0)
    law = str(drag_law).lower().strip()
    if law == "walsh_1976":
        return sphere_drag_walsh_1976(mach)
    if law == "nol_1972":
        return sphere_drag_nol_1972(mach)
    if law == "compton_1996":
        return sphere_drag_compton_1996(mach, trailing_lead=False)
    if law == "compton_trailing":
        return sphere_drag_compton_1996(mach, trailing_lead=True)
    # Fallback compressible sphere drag
    from ammo_mc.aero_sphere import sphere_drag_compressible

    re = max(100.0, v_rel_mps * diam_m * rho_kg_m3 / max(mu_pa_s, 1e-7))
    return float(sphere_drag_compressible(re, mach))


def integrate_pellet_trajectory_rk4(
    v0_mps: float,
    diam_mm: float,
    density_kg_m3: float,
    range_m: float,
    *,
    drag_law: str = "walsh_1976",
    k_d: float = 1.0,
    temp_c: float = 15.0,
    pressure_pa: float = 101325.0,
    rh_fraction: float = 0.0,
    headwind_mps: float = 0.0,
    n_steps: int = 80,
    near_muzzle_indep_m: float = 0.0,
    pellet_mass_g: float | None = None,
) -> dict[str, Any]:
    """Integrate 1D horizontal sphere deceleration over range using RK4 (`WB-F05` / `EB-T01`)."""
    d_m = max(1e-4, float(diam_mm) / 1000.0)
    rho_p = max(100.0, float(density_kg_m3))
    m_kg = (
        float(pellet_mass_g) / 1000.0
        if (pellet_mass_g is not None and float(pellet_mass_g) > 0.0)
        else rho_p * math.pi * (d_m ** 3) / 6.0
    )
    area_p = math.pi * 0.25 * (d_m ** 2)

    if rh_fraction > 0.0:
        atm = cipm_2007_air_density(temp_c=temp_c, pressure_pa=pressure_pa, rh_fraction=rh_fraction)
        rho_air = atm["rho_kg_m3"]
        a_sound = atm["a_mps"]
    else:
        # Ideal dry-air state aligned with Calc_SingleShot (287.053 J/(kg*K))
        t_k = max(180.0, float(temp_c) + 273.15)
        rho_air = max(1000.0, float(pressure_pa)) / (287.053 * t_k)
        a_sound = math.sqrt(1.4 * 287.053 * t_k)

    r_tot = max(0.0, float(range_m))
    v = max(1.0, float(v0_mps))
    tof = 0.0
    if r_tot <= 0.0:
        return {
            "pellet_mass_g": m_kg * 1000.0,
            "v_retained_mps": v,
            "tof_s": 0.0,
            "energy_j": 0.5 * m_kg * (v ** 2),
            "air_density_kg_m3": rho_air,
            "speed_of_sound_mps": a_sound,
            "mach_terminal": v / a_sound,
        }

    steps = max(4, int(n_steps))
    dx = r_tot / steps
    kd_eff = max(0.5, float(k_d))
    hw = float(headwind_mps)
    x_indep = max(0.0, float(near_muzzle_indep_m))

    def _deriv(x_val: float, v_val: float) -> tuple[float, float]:
        v_safe = max(1.0, v_val)
        v_rel = max(1.0, v_safe + hw)
        cd = _eval_v2_drag_cd(v_rel, a_sound, drag_law=drag_law, diam_m=d_m, rho_kg_m3=rho_air)
        cluster_fac = 0.75 if (x_indep > 0.0 and x_val < x_indep) else 1.0
        dv_dx = -(0.5 * rho_air * kd_eff * cluster_fac * cd * area_p * (v_rel ** 2)) / (m_kg * v_safe)
        dt_dx = 1.0 / v_safe
        return dv_dx, dt_dx

    x = 0.0
    for _ in range(steps):
        k1_v, k1_t = _deriv(x, v)
        k2_v, k2_t = _deriv(x + 0.5 * dx, v + 0.5 * dx * k1_v)
        k3_v, k3_t = _deriv(x + 0.5 * dx, v + 0.5 * dx * k2_v)
        k4_v, k4_t = _deriv(x + dx, v + dx * k3_v)
        v = max(1.0, v + (dx / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v))
        tof += (dx / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        x += dx

    return {
        "pellet_mass_g": float(m_kg * 1000.0),
        "v_retained_mps": float(v),
        "tof_s": float(tof),
        "energy_j": float(0.5 * m_kg * (v ** 2)),
        "air_density_kg_m3": float(rho_air),
        "speed_of_sound_mps": float(a_sound),
        "mach_terminal": float(v / a_sound),
    }


def integrate_pellet_trajectory_wb_f05(
    v0_mps: float,
    diam_mm: float,
    density_kg_m3: float,
    range_m: float,
    *,
    drag_law: str = "walsh_1976",
    k_d: float = 1.0,
    temp_c: float = 15.0,
    pressure_pa: float = 101325.0,
    rh_fraction: float = 0.0,
    dx_m: float = 1.0,
    pellet_mass_g: float | None = None,
) -> dict[str, float]:
    """Exact-per-step exponential integrator on a fixed dx grid (`WB-F05` / `EB-T01`)."""
    d_m = max(1e-4, float(diam_mm) / 1000.0)
    rho_p = max(100.0, float(density_kg_m3))
    m_kg = (
        float(pellet_mass_g) / 1000.0
        if (pellet_mass_g is not None and float(pellet_mass_g) > 0.0)
        else rho_p * math.pi * (d_m ** 3) / 6.0
    )
    if float(rh_fraction) > 0.0:
        atm = cipm_2007_air_density(temp_c=temp_c, pressure_pa=pressure_pa, rh_fraction=rh_fraction)
        rho_air = atm["rho_kg_m3"]
        a_sound = atm["a_mps"]
    else:
        t_k = max(180.0, float(temp_c) + 273.15)
        rho_air = max(1000.0, float(pressure_pa)) / (287.053 * t_k)
        a_sound = math.sqrt(1.4 * 287.053 * t_k)

    area_p = math.pi * 0.25 * (d_m ** 2)
    k_fac = (0.5 * rho_air * area_p) / m_kg if (pellet_mass_g is not None and float(pellet_mass_g) > 0.0) else (3.0 * rho_air) / (4.0 * rho_p * d_m)
    kd_eff = max(0.5, float(k_d))
    r_tot = max(0.0, float(range_m))
    v = max(1.0, float(v0_mps))
    tof = 0.0
    if r_tot > 0.0:
        step = max(0.05, float(dx_m))
        n_full = int(r_tot // step)
        rem = r_tot - n_full * step
        steps_list = [step] * n_full + ([rem] if rem > 1e-9 else [])
        for dx_cur in steps_list:
            cd = _eval_v2_drag_cd(v, a_sound, drag_law=drag_law, diam_m=d_m, rho_kg_m3=rho_air)
            q = dx_cur * k_fac * kd_eff * cd
            if q < 1e-12:
                tof += dx_cur / v
            else:
                tof += dx_cur * (math.exp(q) - 1.0) / (q * v)
            v *= math.exp(-q)

    return {
        "pellet_mass_g": float(m_kg * 1000.0),
        "v_retained_mps": float(v),
        "tof_s": float(tof),
        "energy_j": float(0.5 * m_kg * (v ** 2)),
        "air_density_kg_m3": float(rho_air),
        "speed_of_sound_mps": float(a_sound),
        "mach_terminal": float(v / a_sound),
    }


def correct_saami_instrumental_velocity(
    v_instr_mps: float,
    diam_mm: float,
    density_kg_m3: float,
    gauge_dist_ft: float = 3.0,
    *,
    drag_law: str = "walsh_1976",
    k_d: float = 1.0,
    temp_c: float = 15.0,
    pressure_pa: float = 101325.0,
) -> dict[str, float]:
    """Invert chrono velocity at SAAMI 3 ft (lead) or 8 ft (steel) back to true muzzle V0 (`AM-P10`)."""
    gauge_m = max(0.0, float(gauge_dist_ft)) * 0.3048
    v_tgt = max(50.0, float(v_instr_mps))
    if gauge_m <= 0.0:
        return {"v0_muzzle_mps": v_tgt, "v_instr_mps": v_tgt, "delta_v_mps": 0.0, "gauge_dist_m": 0.0}

    lo = v_tgt
    hi = v_tgt * 1.45
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        v_at_g = integrate_pellet_trajectory_rk4(
            mid,
            diam_mm=diam_mm,
            density_kg_m3=density_kg_m3,
            range_m=gauge_m,
            drag_law=drag_law,
            k_d=k_d,
            temp_c=temp_c,
            pressure_pa=pressure_pa,
            n_steps=16,
        )["v_retained_mps"]
        if v_at_g < v_tgt:
            lo = mid
        else:
            hi = mid
    v0_muzzle = 0.5 * (lo + hi)
    return {
        "v0_muzzle_mps": float(v0_muzzle),
        "v_instr_mps": float(v_tgt),
        "delta_v_mps": float(v0_muzzle - v_tgt),
        "gauge_dist_m": float(gauge_m),
    }


# -------------------------------------------------------------------------
# 2. TERMINAL PENETRATION, VELOCITY OGIVE, MARKOV KILL TREE & SAAMI RECOIL
# -------------------------------------------------------------------------

FAA_SHEAR_MODULI_MPA: dict[str, float] = {
    "2024-T351 Al": 276.0,
    "CFRP_Nomex_Sandwich": 251.0,
    "Ti-6Al-4V": 900.0,
    "SS_321": 1300.0,
}

# Project THOR TR-47 Table IV (TE-T01) & Table VI (TE-T02) regression constants (Imperial: fps, in, in^2, grains)
THOR_TR47_METALS: dict[str, dict[str, float]] = {
    "Magnesium alloys": {"c": 6.904, "alpha": 1.092, "beta": -1.170, "gamma": 1.050, "lam": -0.087, "c1": 6.349, "a1": 1.004, "b1": -1.076, "g1": 0.966},
    "2024T-3 Al": {"c": 7.047, "alpha": 1.029, "beta": -1.072, "gamma": 1.251, "lam": -0.139, "c1": 6.185, "a1": 0.903, "b1": -0.941, "g1": 1.098},
    "Titanium alloy": {"c": 6.292, "alpha": 1.103, "beta": -1.095, "gamma": 1.369, "lam": 0.167, "c1": 7.552, "a1": 1.325, "b1": -1.314, "g1": 1.643},
    "Cast iron": {"c": 4.840, "alpha": 1.042, "beta": -1.051, "gamma": 1.028, "lam": 0.523, "c1": 10.153, "a1": 2.186, "b1": -2.204, "g1": 2.156},
    "Face-hardened steel": {"c": 4.356, "alpha": 0.674, "beta": -0.791, "gamma": 0.989, "lam": 0.434, "c1": 7.694, "a1": 1.191, "b1": -1.397, "g1": 1.747},
    "Mild steel": {"c": 6.399, "alpha": 0.889, "beta": -0.945, "gamma": 1.262, "lam": 0.019, "c1": 6.523, "a1": 0.906, "b1": -0.963, "g1": 1.286},
    "Hard homogeneous steel": {"c": 6.475, "alpha": 0.889, "beta": -0.945, "gamma": 1.262, "lam": 0.019, "c1": 6.601, "a1": 0.906, "b1": -0.963, "g1": 1.286},
    "Copper": {"c": 2.785, "alpha": 0.678, "beta": -0.730, "gamma": 0.846, "lam": 0.802, "c1": 14.065, "a1": 3.476, "b1": -3.687, "g1": 4.270},
    "Lead": {"c": 1.999, "alpha": 0.499, "beta": -0.502, "gamma": 0.655, "lam": 0.818, "c1": 10.955, "a1": 2.735, "b1": -2.753, "g1": 3.590},
    "Dense tungsten alloy": {"c": 2.537, "alpha": 0.583, "beta": -0.603, "gamma": 0.865, "lam": 0.828, "c1": 14.773, "a1": 3.393, "b1": -3.510, "g1": 5.037},
}


def wen_2000_frp_ballistic_limit(
    pellet_mass_kg: float,
    pellet_diam_m: float,
    thickness_m: float,
    rho_target_kg_m3: float = 1550.0,
    sigma_e_pa: float = 260.0e6,
    beta: float = 1.5,
) -> dict[str, Any]:
    """Wen (2000) analytical ballistic limit velocity for FRP/CFRP laminates (`TE-F07`).

    Reference anchor: 3 mm CFRP plate (rho_t=1550 kg/m^3, sigma_e=260 MPa) struck by a 5 mm, 3.0 g
    steel projectile yields V_bl = 121.5 m/s (22.1 J); Kevlar29/Phenolic (TE-P06 sigma_e=194 MPa)
    yields V_bl = 104.9 m/s.
    """
    m = max(1e-6, float(pellet_mass_kg))
    d = max(1e-4, float(pellet_diam_m))
    t = max(1e-4, float(thickness_m))
    rho_t = max(200.0, float(rho_target_kg_m3))
    sig_e = max(1.0e6, float(sigma_e_pa))
    b_fac = max(0.5, float(beta))

    pref = (math.pi * b_fac * math.sqrt(rho_t * sig_e) * (d ** 2) * t) / (4.0 * m)
    inner = 1.0 + (8.0 * m) / max(math.pi * (b_fac ** 2) * rho_t * (d ** 2) * t, 1e-18)
    v_bl = pref * (1.0 + math.sqrt(inner))
    e_bl = 0.5 * m * (v_bl ** 2)
    return {
        "v_bl_mps": float(v_bl),
        "e_bl_j": float(e_bl),
        "model": "Wen (2000) FRP Laminate Analytical Model (TE-F07)",
        "verification_grade": "V1",
    }


def faa_shear_plugging_v50(
    pellet_mass_kg: float,
    pellet_diam_m: float,
    thickness_m: float,
    material: str = "2024-T351 Al",
    obliquity_deg: float = 0.0,
) -> dict[str, Any]:
    """FAA shear-plugging V50 equation (`TE-F04`/`TE-T03`) with explicit `TE-C09` caveat."""
    c_s_mpa = float(FAA_SHEAR_MODULI_MPA.get(material, FAA_SHEAR_MODULI_MPA["2024-T351 Al"]))
    c_s_pa = c_s_mpa * 1.0e6
    l_perim = math.pi * max(1e-4, float(pellet_diam_m))
    t = max(1e-4, float(thickness_m))
    m = max(1e-6, float(pellet_mass_kg))
    cos_t = max(0.15, math.cos(math.radians(max(0.0, min(80.0, float(obliquity_deg))))))

    v50 = math.sqrt((2.0 * l_perim * c_s_pa * (t ** 2)) / (m * (cos_t ** 2)))
    e50 = 0.5 * m * (v50 ** 2)
    return {
        "v50_mps": float(v50),
        "e50_j": float(e50),
        "cs_mpa": c_s_mpa,
        "material": material,
        "caveat_id": "TE-C09",
        "caveat_warning": (
            "TE-C09: FAA DOT/FAA/AR-03/37 shear-plugging constants Cs were fitted to 244-800 g "
            "turbofan blade fragments; extrapolation to sub-gram pellets carries V3/V4 uncertainty."
        ),
        "verification_grade": "V3",
    }


def thor_tr47_metallic_penetration(
    pellet_mass_g: float,
    pellet_diam_mm: float,
    v_impact_mps: float,
    thickness_mm: float,
    obliquity_deg: float = 0.0,
    material: str = "2024T-3 Al",
    pellet_material: str = "Steel",
) -> dict[str, Any]:
    """Project THOR TR-47 residual velocity and ballistic limit (`TE-F01`, `TE-F02`, `TE-C01`)."""
    consts = THOR_TR47_METALS.get(material, THOR_TR47_METALS["2024T-3 Al"])
    m_gr = max(0.5, float(pellet_mass_g) * 15.43235835)
    d_in = max(0.01, (float(pellet_diam_mm) / 1000.0) / 0.0254)
    h_in = max(0.005, (float(thickness_mm) / 1000.0) / 0.0254)
    area_in2 = math.pi * 0.25 * (d_in ** 2)
    sec_t = 1.0 / max(0.15, math.cos(math.radians(max(0.0, min(80.0, float(obliquity_deg))))))
    v_fps = max(10.0, float(v_impact_mps) / 0.3048)

    ha = h_in * area_in2
    v_lim_fps = (10.0 ** consts["c1"]) * (ha ** consts["a1"]) * (m_gr ** consts["b1"]) * (sec_t ** consts["g1"])
    dv_fps = (10.0 ** consts["c"]) * (ha ** consts["alpha"]) * (m_gr ** consts["beta"]) * (sec_t ** consts["gamma"]) * (v_fps ** consts["lam"])

    v_res_fps = max(0.0, v_fps - dv_fps)
    if v_fps <= v_lim_fps:
        v_res_fps = 0.0

    warnings: list[str] = []
    if "lead" in str(pellet_material).lower():
        warnings.append("TE-C01: Project THOR TR-47 assumes non-deforming steel fragments; soft lead splats and overpredicts penetration.")
    if "al" in material.lower() and float(v_impact_mps) < 366.0:
        warnings.append("TE-C01: Below THOR 2024T-3 Al calibrated velocity floor (1200 fps / 366 m/s).")
    if m_gr < 5.0:
        warnings.append(f"TE-C01: Pellet mass {m_gr:.1f} gr is below THOR 5-grain lower test envelope.")

    return {
        "v_limit_mps": float(v_lim_fps * 0.3048),
        "v_residual_mps": float(v_res_fps * 0.3048),
        "perforates": bool(v_res_fps > 0.0),
        "warnings": warnings,
        "material": material,
        "verification_grade": "V1" if not warnings else "V3",
    }


def velocity_ogive_peff(v_impact_mps: float, v50_mps: float, spread_s_v: float = 0.10) -> float:
    """Normal velocity-ogive perforation effectiveness P_eff(v) (`WB-F04`)."""
    v50 = max(1.0, float(v50_mps))
    sv = max(1e-4, float(spread_s_v))
    return float(ndtr((float(v_impact_mps) - v50) / (sv * v50)))


def velocity_ogive_peff_array(v_impact_arr: np.ndarray, v50_mps: float, spread_s_v: float = 0.10) -> np.ndarray:
    """Vectorized normal velocity-ogive perforation effectiveness (`WB-F04`)."""
    v50 = max(1.0, float(v50_mps))
    sv = max(1e-4, float(spread_s_v))
    v_arr = np.asarray(v_impact_arr, dtype=np.float64)
    return np.asarray(ndtr((v_arr - v50) / (sv * v50)), dtype=np.float64)


def evaluate_markov_redundant_kill_tree(
    n_rotors: int = 6,
    max_lost_rotors_survivable: int = 1,
    p_rotor_hit: float = 0.35,
    p_core_hit: float = 0.15,
    *,
    expected_hits_lambda: float | None = None,
    n_hits: int | None = None,
) -> dict[str, Any]:
    """Markov-chain redundant component kill tree (`TE-F16`) for multi-rotors / dual-battery UAS.

    Transient states `0..m` represent lost rotor count `<= max_lost_rotors_survivable`;
    final absorbing state `m+1` represents catastrophic loss (core kill or >m rotors disabled).
    """
    m_surv = max(0, min(int(n_rotors) - 1, int(max_lost_rotors_survivable)))
    n_states = m_surv + 2  # last index is KILLED
    t_mat = np.zeros((n_states, n_states), dtype=np.float64)

    p_r = max(0.0, min(1.0, float(p_rotor_hit)))
    p_c = max(0.0, min(1.0 - p_r, float(p_core_hit)))

    for k in range(m_surv + 1):
        frac_alive = max(0.0, (int(n_rotors) - k) / max(1, int(n_rotors)))
        p_loss = min(1.0 - p_c, p_r * frac_alive)
        p_stay = max(0.0, 1.0 - p_c - p_loss)
        t_mat[k, k] = p_stay
        if k < m_surv:
            t_mat[k + 1, k] = p_loss
            t_mat[m_surv + 1, k] = p_c
        else:
            t_mat[m_surv + 1, k] = p_c + p_loss
    t_mat[m_surv + 1, m_surv + 1] = 1.0

    s0 = np.zeros(n_states, dtype=np.float64)
    s0[0] = 1.0

    if expected_hits_lambda is not None:
        lam = max(0.0, float(expected_hits_lambda))
        q_mat = (t_mat - np.eye(n_states, dtype=np.float64)) * lam
        s_final = expm(q_mat) @ s0
        mode = "poisson_matrix_exp"
    else:
        nh = max(0, int(n_hits if n_hits is not None else 3))
        s_final = np.linalg.matrix_power(t_mat, nh) @ s0
        mode = "discrete_n_hits"

    return {
        "p_kill": float(np.clip(s_final[-1], 0.0, 1.0)),
        "state_probs": [round(float(x), 6) for x in s_final],
        "n_rotors": int(n_rotors),
        "max_lost_rotors_survivable": m_surv,
        "mode": mode,
        "verification_grade": "V3",
    }


def compute_saami_recoil_impulse(
    shot_mass_g: float,
    wad_mass_g: float = 2.79,
    propellant_mass_g: float = 2.164,
    v0_mps: float = 388.62,
    gun_mass_kg: float = 3.17514659,
    gas_factor_fg: float = 1.50,
) -> dict[str, float]:
    """Compute free-recoil impulse & energy via SAAMI proportional gas factor `f_g` (`AM-F03`/`AM-P18`)."""
    m_e_kg = (max(0.0, float(shot_mass_g)) + max(0.0, float(wad_mass_g))) / 1000.0
    m_p_kg = max(0.0, float(propellant_mass_g)) / 1000.0
    v0 = max(1.0, float(v0_mps))
    fg = max(1.0, float(gas_factor_fg))
    m_gun = max(0.5, float(gun_mass_kg))

    impulse_ns = (m_e_kg + m_p_kg * fg) * v0
    v_gun = impulse_ns / m_gun
    fre_j = 0.5 * m_gun * (v_gun ** 2)
    return {
        "impulse_ns": float(impulse_ns),
        "free_recoil_velocity_mps": float(v_gun),
        "free_recoil_energy_j": float(fre_j),
        "gas_factor_fg": float(fg),
        "ejecta_mass_g": float(m_e_kg * 1000.0),
    }


# -------------------------------------------------------------------------
# 3. FIRE-CONTROL SERIAL DEAD TIME, HEAVY-TAILED ATR, RCS & SINGER MODEL
# -------------------------------------------------------------------------

SERIAL_DEAD_TIME_PRESETS_MS: dict[str, dict[str, float]] = {
    "eo_60hz_1stream": {
        "sensor_half_frame_ms": 8.3,
        "edge_gpu_vision_ms": 83.9,   # p99 105.2 ms
        "tracker_comms_ms": 10.0,
        "servo_lag_ms": 15.0,
        "trigger_solenoid_ms": 15.0,
        "mechanical_lock_ms": 5.2,
        "primer_ignition_ms": 0.9,
        "barrel_transit_ms": 2.6,
    },
    "eo_60hz_3stream": {
        "sensor_half_frame_ms": 8.3,
        "edge_gpu_vision_ms": 129.7,  # p99 195.3 ms
        "tracker_comms_ms": 12.0,
        "servo_lag_ms": 15.0,
        "trigger_solenoid_ms": 15.0,
        "mechanical_lock_ms": 5.2,
        "primer_ignition_ms": 0.9,
        "barrel_transit_ms": 2.6,
    },
    "radar_10hz_fused": {
        "sensor_half_frame_ms": 50.0,
        "edge_gpu_vision_ms": 42.0,
        "tracker_comms_ms": 10.0,
        "servo_lag_ms": 15.0,
        "trigger_solenoid_ms": 15.0,
        "mechanical_lock_ms": 3.2,
        "primer_ignition_ms": 0.9,
        "barrel_transit_ms": 2.4,
    },
}


def compute_serial_dead_time_budget(
    preset: str = "eo_60hz_1stream",
    overrides_ms: dict[str, float] | None = None,
    tof_s: float = 0.14,
    a_unmod_mps2: float = 5.0,
) -> dict[str, Any]:
    """Sum serial fire-control latency chain (`FC-T04`, `FC-F15`) and quadratic miss (`FC-F10`)."""
    stages = dict(SERIAL_DEAD_TIME_PRESETS_MS.get(preset, SERIAL_DEAD_TIME_PRESETS_MS["eo_60hz_1stream"]))
    if overrides_ms:
        for k, v in overrides_ms.items():
            if k in stages:
                stages[k] = max(0.0, float(v))
    t_dead_ms = float(sum(stages.values()))
    t_dead_s = t_dead_ms / 1000.0
    t_total_s = t_dead_s + max(0.0, float(tof_s))
    miss_m = 0.5 * max(0.0, float(a_unmod_mps2)) * (t_total_s ** 2)
    return {
        "preset": preset,
        "stages_ms": stages,
        "t_dead_ms": round(t_dead_ms, 2),
        "t_dead_s": round(t_dead_s, 4),
        "tof_s": round(max(0.0, float(tof_s)), 4),
        "t_total_s": round(t_total_s, 4),
        "quadratic_maneuver_miss_m": round(miss_m, 4),
        "note": "Reported vision SD reflects between-trial spread; operational p99 reaches 105-195 ms.",
    }


def sample_heavy_tailed_atr_error_mrad(
    rng: Any,
    size: int,
    core_sigma_mrad: float = 0.835,
    tail_prob: float = 0.02,
    tail_max_mrad: float = 13.74,
    tail_min_mrad: float | None = None,
) -> np.ndarray:
    """Sample AI/EO ATR centroid noise (`FC-P12`: 0.835 mrad core + 2% outlier tail U(3, 13.74) mrad)."""
    n = max(1, int(size))
    sig_c = max(0.05, float(core_sigma_mrad))
    out = rng.normal(0.0, sig_c, size=n)
    tail_mask = rng.random(n) < max(0.0, min(0.5, float(tail_prob)))
    n_tail = int(np.sum(tail_mask))
    if n_tail > 0:
        signs = np.where(rng.random(n_tail) < 0.5, -1.0, 1.0)
        lo_tail = float(tail_min_mrad) if tail_min_mrad is not None else max(3.0, 3.5 * sig_c)
        hi_tail = max(lo_tail + 0.1, float(tail_max_mrad))
        out[tail_mask] = signs * rng.uniform(lo_tail, hi_tail, size=n_tail)
    return out


def compute_aspect_dependent_rcs_dbsm(
    base_rcs_dbsm: float = -20.0,
    aspect_deg: float = 0.0,
    null_depth_db: float = 28.0,
) -> float:
    """Evaluate small-UAS aspect-dependent radar cross-section with 25-35 dB nulls (`FC-P02`)."""
    depth = max(0.0, min(35.0, float(null_depth_db)))
    # Sharp specular/interference nulls near oblique rotor-hub angles (~37 deg, ~112 deg)
    ang = math.radians(float(aspect_deg))
    null_lobe = 0.5 * (1.0 - math.cos(4.0 * ang))
    sharp_null = (math.sin(3.0 * ang) ** 6)
    drop_db = depth * min(1.0, 0.35 * null_lobe + 0.85 * sharp_null)
    return float(base_rcs_dbsm - drop_db)


def generate_singer_evasive_trajectory(
    t_grid: np.ndarray,
    target_distance_m: float,
    velocity_mps: float,
    heading_deg: float,
    sigma_a_mps2: float = 15.0,
    tau_m_s: float = 1.2,
    rng: Any | None = None,
) -> dict[str, np.ndarray]:
    """Generate Singer (1970) exponentially correlated maneuver state along time grid (`FC-F09`)."""
    t_arr = np.asarray(t_grid, dtype=np.float64)
    n_pts = len(t_arr)
    gen = rng if rng is not None else np.random.default_rng(42)

    heading_rad = math.radians(float(heading_deg))
    vx_base = -float(velocity_mps) * math.cos(heading_rad)
    vy_base = float(velocity_mps) * math.sin(heading_rad)

    ay = np.zeros(n_pts, dtype=np.float64)
    az = np.zeros(n_pts, dtype=np.float64)
    vy_ev = np.zeros(n_pts, dtype=np.float64)
    vz_ev = np.zeros(n_pts, dtype=np.float64)
    dy = np.zeros(n_pts, dtype=np.float64)
    dz = np.zeros(n_pts, dtype=np.float64)

    tau = max(0.1, float(tau_m_s))
    sig_a = max(0.0, float(sigma_a_mps2))
    ay[0] = float(gen.normal(0.0, sig_a))
    az[0] = float(gen.normal(0.0, 0.5 * sig_a))

    for k in range(1, n_pts):
        dt = max(1e-4, float(t_arr[k] - t_arr[k - 1]))
        rho = math.exp(-dt / tau)
        q_std = sig_a * math.sqrt(max(0.0, 1.0 - rho * rho))
        ay[k] = rho * ay[k - 1] + float(gen.normal(0.0, q_std))
        az[k] = rho * az[k - 1] + float(gen.normal(0.0, 0.5 * q_std))
        vy_ev[k] = vy_ev[k - 1] + 0.5 * (ay[k - 1] + ay[k]) * dt
        vz_ev[k] = vz_ev[k - 1] + 0.5 * (az[k - 1] + az[k]) * dt
        dy[k] = dy[k - 1] + 0.5 * (vy_ev[k - 1] + vy_ev[k]) * dt
        dz[k] = dz[k - 1] + 0.5 * (vz_ev[k - 1] + vz_ev[k]) * dt

    # Center positional drift around t=0 if 0 falls inside grid
    if n_pts > 0 and t_arr[0] <= 0.0 <= t_arr[-1]:
        dy0 = float(np.interp(0.0, t_arr, dy))
        dz0 = float(np.interp(0.0, t_arr, dz))
        dy = dy - dy0
        dz = dz - dz0

    return {
        "x": float(target_distance_m) + vx_base * t_arr,
        "y": vy_base * t_arr + dy,
        "z": dz,
        "vx": np.full(n_pts, vx_base, dtype=np.float64),
        "vy": vy_base + vy_ev,
        "vz": vz_ev,
        "ay": ay,
        "az": az,
    }
