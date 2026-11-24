"""NATO STANAG 4082 Computer Met Message (METCM) & Upper Atmospheric Wind Shear.

Implements:
1. NATO STANAG 4082 standard altitude zone profile generator (0 to 20,000 m).
2. Boundary-layer wind shear (logarithmic/power-law) and Ekman spiral directional veering.
3. Jet-stream upper-troposphere shear layer (9,000 - 11,000 m).
4. Ballistic wind and density weighting factors along projectile trajectory arcs.
5. Numerical trajectory integration comparing standard uniform atmosphere vs. sheared sounding profile.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


STANAG_4082_STANDARD_ZONES = [
    0, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    11000, 12000, 14000, 16000, 18000, 20000
]


def generate_stanag_4082_profile(
    surface_temp_c: float = 15.0,
    surface_pressure_hpa: float = 1013.25,
    surface_wind_speed_mps: float = 5.0,
    surface_wind_dir_deg: float = 240.0,
    shear_exponent: float = 0.28,
    ekman_veer_deg: float = 30.0,
    jet_stream_peak_mps: float = 38.0,
    jet_stream_altitude_m: float = 10500.0,
    thermal_inversion: bool = False,
    inversion_base_m: float = 800.0,
    inversion_top_m: float = 1400.0,
    inversion_delta_t_k: float = 4.5,
) -> dict[str, Any]:
    """Generates a complete NATO STANAG 4082 METCM atmospheric sounding profile table.

    Computes:
    - Temperature (K and °C) with optional surface/boundary layer thermal inversion.
    - Barometric pressure (hPa) via hypsometric hydrostatic integration.
    - True air density (kg/m^3) and percent deviation from standard ISA.
    - Vector wind speed (m/s) with power-law boundary layer shear + upper jet stream gaussian peak.
    - Wind direction (deg) with Ekman spiral veering across the boundary layer.
    """
    R_spec = 287.05287  # J/(kg*K)
    g = 9.80665

    t_sfc_k = surface_temp_c + 273.15
    p_sfc_pa = surface_pressure_hpa * 100.0

    zones = []
    z_prev = 0.0
    p_prev = p_sfc_pa
    t_prev_k = t_sfc_k

    for z in STANAG_4082_STANDARD_ZONES:
        # 1. Temperature model
        if z <= 11000.0:
            t_base = t_sfc_k - 0.0065 * z
        else:
            t_base = t_sfc_k - 0.0065 * 11000.0  # Isothermal lower stratosphere

        # Add optional thermal inversion
        t_inv = 0.0
        if thermal_inversion and inversion_base_m <= z <= inversion_top_m:
            frac = (z - inversion_base_m) / max(inversion_top_m - inversion_base_m, 1.0)
            t_inv = inversion_delta_t_k * math.sin(math.pi * frac)

        t_curr_k = t_base + t_inv

        # 2. Hydrostatic Pressure integration across the layer [z_prev, z]
        if z == 0.0:
            p_curr_pa = p_sfc_pa
        else:
            t_layer_mean = 0.5 * (t_prev_k + t_curr_k)
            dz = z - z_prev
            p_curr_pa = p_prev * math.exp(-g * dz / (R_spec * t_layer_mean))

        p_prev = p_curr_pa
        z_prev = z
        t_prev_k = t_curr_k

        # 3. Density
        rho_curr = p_curr_pa / (R_spec * t_curr_k)
        # ISA standard density at altitude z
        if z <= 11000.0:
            t_isa = 288.15 - 0.0065 * z
            p_isa = 101325.0 * ((t_isa / 288.15) ** (g / (0.0065 * R_spec)))
        else:
            t_isa = 216.65
            p_11k = 101325.0 * ((216.65 / 288.15) ** (g / (0.0065 * R_spec)))
            p_isa = p_11k * math.exp(-g * (z - 11000.0) / (R_spec * 216.65))
        rho_isa = p_isa / (R_spec * t_isa)
        density_pct_isa = ((rho_curr / rho_isa) - 1.0) * 100.0

        # 4. Wind profile
        # Boundary layer power-law shear
        z_eff = max(z, 10.0)
        speed_bl = surface_wind_speed_mps * ((z_eff / 10.0) ** shear_exponent)
        # Jet stream Gaussian peak
        jet_bump = jet_stream_peak_mps * math.exp(-(((z - jet_stream_altitude_m) / 3200.0) ** 2))
        wind_speed_mps = min(speed_bl + jet_bump, 120.0)

        # Ekman directional veer (veers clockwise up to boundary layer top ~1800m)
        veer_fraction = min(1.0, z / 1800.0)
        wind_dir_deg = (surface_wind_dir_deg + ekman_veer_deg * math.sin(0.5 * math.pi * veer_fraction)) % 360.0

        # Cartesian components (meteorological convention: wind FROM direction)
        rad_dir = math.radians(wind_dir_deg)
        w_x = -wind_speed_mps * math.sin(rad_dir)
        w_y = -wind_speed_mps * math.cos(rad_dir)

        zones.append({
            "zone_index": len(zones),
            "altitude_m": z,
            "temp_c": round(t_curr_k - 273.15, 1),
            "temp_k": round(t_curr_k, 2),
            "pressure_hpa": round(p_curr_pa / 100.0, 1),
            "density_kg_m3": round(rho_curr, 4),
            "density_pct_isa": round(density_pct_isa, 1),
            "wind_speed_mps": round(wind_speed_mps, 1),
            "wind_direction_deg": round(wind_dir_deg, 1),
            "wind_vx_mps": round(w_x, 2),
            "wind_vy_mps": round(w_y, 2),
        })

    return {
        "surface_conditions": {
            "temp_c": surface_temp_c,
            "pressure_hpa": surface_pressure_hpa,
            "wind_speed_mps": surface_wind_speed_mps,
            "wind_dir_deg": surface_wind_dir_deg,
        },
        "jet_stream": {
            "peak_speed_mps": jet_stream_peak_mps,
            "altitude_m": jet_stream_altitude_m,
        },
        "thermal_inversion_active": thermal_inversion,
        "zones": zones,
    }


def integrate_trajectory_with_sounding(
    v0_mps: float = 827.0,
    qe_deg: float = 38.0,
    az_deg: float = 90.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    sounding_profile: dict[str, Any] | None = None,
    dt: float = 0.08,
) -> dict[str, Any]:
    r"""Integrates 3D exterior ballistics trajectory through a multi-layer sounding profile.

    Computes:
    - 3D flight path coordinates $(X, Y, Z)$ under local altitude-dependent winds, density, and speed of sound.
    - True horizontal range, lateral wind deflection, vertex altitude, and time of flight.
    - Trajectory baseline through uniform sea-level ISA standard atmosphere for comparison.
    - Net range error ($\Delta X_{\text{met}}$) and deflection drift ($\Delta Y_{\text{met}}$) caused by met effects.
    """
    if sounding_profile is None:
        sounding_profile = generate_stanag_4082_profile()

    zones = sounding_profile["zones"]
    altitudes = np.array([z["altitude_m"] for z in zones], dtype=np.float64)
    densities = np.array([z["density_kg_m3"] for z in zones], dtype=np.float64)
    temps_k = np.array([z["temp_k"] for z in zones], dtype=np.float64)
    wind_vx = np.array([z["wind_vx_mps"] for z in zones], dtype=np.float64)
    wind_vy = np.array([z["wind_vy_mps"] for z in zones], dtype=np.float64)

    area = 0.25 * math.pi * (calibre_m**2)
    gamma = 1.4
    R_gas = 287.05

    def get_atmo_at_z(z_val: float) -> tuple[float, float, float, float]:
        z_clamped = float(np.clip(z_val, 0.0, 20000.0))
        rho = float(np.interp(z_clamped, altitudes, densities))
        tk = float(np.interp(z_clamped, altitudes, temps_k))
        wx = float(np.interp(z_clamped, altitudes, wind_vx))
        wy = float(np.interp(z_clamped, altitudes, wind_vy))
        return rho, tk, wx, wy

    # 1. Integrate with Sheared Met Sounding via RK4 + G7 table + linear ground interpolation
    from ammo_mc.data import get_kb
    from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch

    kb = get_kb()
    rad_qe = math.radians(qe_deg)
    rad_az = math.radians(az_deg)

    def met_accel(z_val: float, vx_v: float, vy_v: float, vz_v: float) -> tuple[float, float, float]:
        rho, tk, wx, wy = get_atmo_at_z(z_val)
        vrx = vx_v - wx
        vry = vy_v - wy
        vrz = vz_v
        vr_mag = math.hypot(vrx, vry, vrz)
        a_sound = math.sqrt(gamma * R_gas * tk)
        mach = vr_mag / max(a_sound, 100.0)
        cd = float(kb.interp_cd(mach, drag_ref="G7"))
        # Consistent EB-02 G7 area scaling: (pi * d^2 / 8) * rho * cd / m * |v_rel|
        drag_acc = (math.pi * (calibre_m ** 2) / (8.0 * mass_kg)) * rho * cd * vr_mag
        ax = -drag_acc * vrx
        ay = -drag_acc * vry
        az = -drag_acc * vrz - 9.80665
        return ax, ay, az

    px, py, pz = 0.0, 0.0, 0.0
    vx = v0_mps * math.cos(rad_qe) * math.sin(rad_az)
    vy = v0_mps * math.cos(rad_qe) * math.cos(rad_az)
    vz = v0_mps * math.sin(rad_qe)
    t = 0.0

    traj_x, traj_y, traj_z, traj_t = [0.0], [0.0], [0.0], [0.0]

    while pz >= 0.0 and t < 180.0:
        px0, py0, pz0, vx0, vy0, vz0, t0 = px, py, pz, vx, vy, vz, t
        ax1, ay1, az1 = met_accel(pz0, vx0, vy0, vz0)
        ax2, ay2, az2 = met_accel(
            pz0 + 0.5 * dt * vz0,
            vx0 + 0.5 * dt * ax1,
            vy0 + 0.5 * dt * ay1,
            vz0 + 0.5 * dt * az1,
        )
        vx_m2 = vx0 + 0.5 * dt * ax1
        vy_m2 = vy0 + 0.5 * dt * ay1
        vz_m2 = vz0 + 0.5 * dt * az1
        ax3, ay3, az3 = met_accel(
            pz0 + 0.5 * dt * vz_m2,
            vx_m2,
            vy_m2,
            vz_m2,
        )
        vx_k3 = vx0 + 0.5 * dt * ax2
        vy_k3 = vy0 + 0.5 * dt * ay2
        vz_k3 = vz0 + 0.5 * dt * az2
        vx_k4 = vx0 + dt * ax3
        vy_k4 = vy0 + dt * ay3
        vz_k4 = vz0 + dt * az3
        ax4, ay4, az4 = met_accel(pz0 + dt * vz_k3, vx_k4, vy_k4, vz_k4)

        px_n = px0 + (dt / 6.0) * (vx0 + 2.0 * vx_m2 + 2.0 * vx_k3 + vx_k4)
        py_n = py0 + (dt / 6.0) * (vy0 + 2.0 * vy_m2 + 2.0 * vy_k3 + vy_k4)
        pz_n = pz0 + (dt / 6.0) * (vz0 + 2.0 * vz_m2 + 2.0 * vz_k3 + vz_k4)
        vx_n = vx0 + (dt / 6.0) * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
        vy_n = vy0 + (dt / 6.0) * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
        vz_n = vz0 + (dt / 6.0) * (az1 + 2.0 * az2 + 2.0 * az3 + az4)
        t_n = t0 + dt

        if pz_n < 0.0 and pz0 >= 0.0 and (pz0 - pz_n) > 1e-12:
            frac = pz0 / (pz0 - pz_n)
            px = px0 + frac * (px_n - px0)
            py = py0 + frac * (py_n - py0)
            pz = 0.0
            t = t0 + frac * dt
            traj_x.append(px)
            traj_y.append(py)
            traj_z.append(0.0)
            traj_t.append(t)
            break

        px, py, pz, vx, vy, vz, t = px_n, py_n, pz_n, vx_n, vy_n, vz_n, t_n
        traj_x.append(px)
        traj_y.append(py)
        traj_z.append(pz)
        traj_t.append(t)

    # 2. Integrate with Standard ISA Atmosphere (Zero Wind, no Coriolis so physics isolate MET effects cleanly)
    atm_std = AtmosphereEnvironment()
    cfg_std = BatchTrajectoryConfig(calibre_m=calibre_m, fidelity="L1", dt=dt, include_coriolis=False)
    s0_std = np.array([[0.0, 0.0, 0.0, v0_mps * math.cos(rad_qe), 0.0, v0_mps * math.sin(rad_qe)]], dtype=np.float64)
    res_std = integrate_trajectories_batch(s0_std, np.array([mass_kg]), np.array([1.0]), atm_std, cfg_std, record_single_path=True)

    std_sp = res_std.get("sample_path", {})
    std_range = float(res_std["range_m"][0])
    std_tof = float(res_std["tof_s"][0])

    # Convert standard trajectory to world X, Y
    std_traj_x = [float(std_sp["range_m"][k]) * math.sin(rad_az) for k in range(len(std_sp.get("t", [])))] if std_sp else [0.0, std_range * math.sin(rad_az)]
    std_traj_y = [float(std_sp["range_m"][k]) * math.cos(rad_az) for k in range(len(std_sp.get("t", [])))] if std_sp else [0.0, std_range * math.cos(rad_az)]
    std_traj_z = [float(std_sp["alt_m"][k]) for k in range(len(std_sp.get("t", [])))] if std_sp else [0.0, 0.0]

    # Downrange projection along firing azimuth and perpendicular crossrange drift
    along_range_m = traj_x[-1] * math.sin(rad_az) + traj_y[-1] * math.cos(rad_az)
    cross_perp_m = traj_x[-1] * math.cos(rad_az) - traj_y[-1] * math.sin(rad_az)
    delta_range_m = along_range_m - std_range

    # Downsample trajectory points to ~35 points for fast 3D WebGL rendering
    step = max(1, len(traj_t) // 35)
    step_std = max(1, len(std_traj_x) // 35)

    return {
        "range_with_met_m": round(along_range_m, 1),
        "range_isa_baseline_m": round(std_range, 1),
        "delta_range_met_m": round(delta_range_m, 1),
        "crossrange_drift_m": round(abs(cross_perp_m), 1),
        "crossrange_signed_m": round(cross_perp_m, 1),
        "tof_with_met_s": round(traj_t[-1], 2),
        "tof_isa_baseline_s": round(std_tof, 2),
        "vertex_altitude_m": round(float(np.max(traj_z)), 1),
        "trajectory_met": {
            "x_m": [round(traj_x[k], 1) for k in range(0, len(traj_t), step)],
            "y_m": [round(traj_y[k], 1) for k in range(0, len(traj_t), step)],
            "z_m": [round(traj_z[k], 1) for k in range(0, len(traj_t), step)],
            "t_s": [round(traj_t[k], 2) for k in range(0, len(traj_t), step)],
        },
        "trajectory_isa": {
            "x_m": [round(std_traj_x[k], 1) for k in range(0, len(std_traj_x), step_std)],
            "y_m": [round(std_traj_y[k], 1) for k in range(0, len(std_traj_y), step_std)],
            "z_m": [round(std_traj_z[k], 1) for k in range(0, len(std_traj_z), step_std)],
        },
    }


def compute_ballistic_met_weighting(
    sounding_profile: dict[str, Any],
    vertex_alt_m: float = 4000.0,
) -> dict[str, float]:
    """Compute NATO STANAG 4061 / 4082 parabolic ordinate-weighted ballistic wind, temp, and density."""
    zones = sounding_profile.get("zones", [])
    if not zones:
        return {"ballistic_wx_mps": 0.0, "ballistic_wy_mps": 0.0, "ballistic_temp_c": 15.0, "ballistic_density_pct_isa": 0.0}

    h_max = max(float(vertex_alt_m), 200.0)
    grid = np.linspace(0.0, h_max, 50)
    # Time spent at altitude z on a parabolic arc scales as (h_max - z)^(-1/2) -> weights concentrated near apogee
    weights = 1.0 / np.sqrt(np.maximum(1.0 - (grid / h_max), 0.04))
    weights /= float(np.sum(weights))

    alts = np.array([float(z["altitude_m"]) for z in zones], dtype=np.float64)
    wx = np.interp(grid, alts, [float(z["wind_vx_mps"]) for z in zones])
    wy = np.interp(grid, alts, [float(z["wind_vy_mps"]) for z in zones])
    tc = np.interp(grid, alts, [float(z["temp_c"]) for z in zones])
    dp = np.interp(grid, alts, [float(z["density_pct_isa"]) for z in zones])

    bwx = float(np.sum(weights * wx))
    bwy = float(np.sum(weights * wy))
    return {
        "ballistic_wx_mps": round(bwx, 2),
        "ballistic_wy_mps": round(bwy, 2),
        "ballistic_wind_speed_mps": round(math.hypot(bwx, bwy), 2),
        "ballistic_temp_c": round(float(np.sum(weights * tc)), 2),
        "ballistic_density_pct_isa": round(float(np.sum(weights * dp)), 2),
    }
