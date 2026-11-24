"""Atmosphere, meteorology, speed of sound, and altitude wind profile models.

Implements Formula IDs EB-03 and EB-04 (US Standard Atmosphere 1976 + deviations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from ammo_mc.formulas_registry import register_formula

R_GAS_DRY_AIR = 287.05287  # J/(kg K) per US Std Atm 1976 (S16)
GAMMA_AIR = 1.4
G0 = 9.80665  # m/s^2
P0_ISA_PA = 101325.0  # Pa
T0_ISA_K = 288.15  # K


@register_formula("EB-03")
def eb_03_mach_sound_speed(
    v: np.ndarray | float,
    T_K: np.ndarray | float,
    wind: np.ndarray | float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """EB-03: Speed of sound a = sqrt(gamma*R*T) and Mach number M = |v - w| / a.

    Supports both scalar speeds and 3-D velocity vectors [..., 3].
    Returns `(mach, sound_speed_mps)`.
    """
    t_arr = np.maximum(np.asarray(T_K, dtype=np.float64), 50.0)
    a = np.sqrt(GAMMA_AIR * R_GAS_DRY_AIR * t_arr)

    v_arr = np.asarray(v, dtype=np.float64)
    w_arr = np.asarray(wind, dtype=np.float64)
    v_rel = v_arr - w_arr
    is_vec = (v_rel.ndim >= 2 and v_rel.shape[-1] == 3) or (
        v_rel.ndim == 1
        and v_rel.shape == (3,)
        and np.ndim(T_K) == 0
        and (np.ndim(wind) == 0 or np.asarray(wind).shape == (3,))
    )
    if is_vec:
        speed_rel = np.linalg.norm(v_rel, axis=-1)
    else:
        speed_rel = np.abs(v_rel)

    if np.ndim(a) >= 2 and np.asarray(a).shape[-1] == 1 and speed_rel.ndim == np.ndim(a) - 1:
        a = np.squeeze(a, axis=-1)

    mach = speed_rel / a
    if np.ndim(v) == 0 and np.ndim(T_K) == 0 and np.ndim(wind) == 0:
        return float(mach), float(a)
    return mach, a


@register_formula("EB-04")
def eb_04_isa_atmosphere(
    h_m: float | np.ndarray,
    dT_K: float | np.ndarray = 0.0,
    p_sl_Pa: float | np.ndarray = P0_ISA_PA,
) -> dict[str, float | np.ndarray]:
    """EB-04: US Standard Atmosphere 1976 (0-51 km: troposphere + tropopause + upper stratosphere).

    Preserves exact V-09/V-10 values for h <= 20,000 m and adds US Std Atm 1976
    stratospheric lapse layers (20-32 km, 32-47 km, 47-51 km) for high-apogee trajectories.
    """
    h = np.asarray(h_m, dtype=np.float64)
    dt = np.asarray(dT_K, dtype=np.float64)
    p0 = np.asarray(p_sl_Pa, dtype=np.float64)

    t_std = np.where(
        h <= 11000.0,
        T0_ISA_K - 0.0065 * h,
        np.where(
            h <= 20000.0,
            216.65,
            np.where(
                h <= 32000.0,
                216.65 + 0.0010 * (h - 20000.0),
                np.where(
                    h <= 47000.0,
                    228.65 + 0.0028 * (h - 32000.0),
                    270.65,
                ),
            ),
        ),
    )
    t_k = np.maximum(t_std + dt, 50.0)

    p_11k = p0 * ((216.65 / T0_ISA_K) ** 5.25588)
    p_20k = p_11k * math.exp(-0.000157688 * 9000.0)
    exp_20_32 = -G0 / (R_GAS_DRY_AIR * 0.0010)
    p_32k = p_20k * ((228.65 / 216.65) ** exp_20_32)
    exp_32_47 = -G0 / (R_GAS_DRY_AIR * 0.0028)
    p_47k = p_32k * ((270.65 / 228.65) ** exp_32_47)

    p_pa = np.where(
        h <= 11000.0,
        p0 * np.power(np.maximum(T0_ISA_K - 0.0065 * h, 10.0) / T0_ISA_K, 5.25588),
        np.where(
            h <= 20000.0,
            p_11k * np.exp(-0.000157688 * (h - 11000.0)),
            np.where(
                h <= 32000.0,
                p_20k * np.power(np.maximum(216.65 + 0.0010 * (h - 20000.0), 10.0) / 216.65, exp_20_32),
                np.where(
                    h <= 47000.0,
                    p_32k * np.power(np.maximum(228.65 + 0.0028 * (h - 32000.0), 10.0) / 228.65, exp_32_47),
                    p_47k * np.exp(-G0 * (h - 47000.0) / (R_GAS_DRY_AIR * 270.65)),
                ),
            ),
        ),
    )
    rho = p_pa / (R_GAS_DRY_AIR * t_k)
    a = np.sqrt(GAMMA_AIR * R_GAS_DRY_AIR * t_k)

    if np.ndim(h_m) == 0 and np.ndim(dT_K) == 0 and np.ndim(p_sl_Pa) == 0:
        return {
            "T_K": float(t_k),
            "p_Pa": float(p_pa),
            "rho_kg_m3": float(rho),
            "a_mps": float(a),
        }
    return {
        "T_K": t_k,
        "p_Pa": p_pa,
        "rho_kg_m3": rho,
        "a_mps": a,
    }


def isa_atmosphere_extended(
    h_m: float,
    dT_K: float = 0.0,
    rh_pct: float = 0.0,
    p_sl_Pa: float = P0_ISA_PA,
) -> dict[str, float]:
    """High-fidelity US Std Atm 1976 profile with geopotential height, humidity/virtual T, & hydrostatic dT.

    Unlike the fast vectorized EB-04 table expression, this analytically integrates
    hydrostatic equilibrium dp/dH = -g0*p/(R*T(H)) when dT_K != 0 and applies Buck's
    saturation vapor pressure equation for moist-air virtual temperature T_v.
    """
    r_eff_m = 6356766.0  # Effective US Std Atm 1976 spherical Earth radius
    z = max(0.0, float(h_m))
    H = (r_eff_m * z) / (r_eff_m + z)  # Geopotential altitude

    layers = [
        (0.0, 11000.0, T0_ISA_K, -0.0065),
        (11000.0, 20000.0, 216.65, 0.0),
        (20000.0, 32000.0, 216.65, 0.0010),
        (32000.0, 47000.0, 228.65, 0.0028),
        (47000.0, 86000.0, 270.65, 0.0),
    ]
    p_cur = float(p_sl_Pa)
    t_cur = max(T0_ISA_K + float(dT_K), 50.0)
    for h0, h1, t_base0, lapse in layers:
        if H <= h0:
            break
        dh = min(H, h1) - h0
        t_start = max(t_base0 + float(dT_K), 50.0)
        t_cur = max(t_start + lapse * dh, 50.0)
        if abs(lapse) > 1e-12:
            p_cur *= (t_cur / t_start) ** (-G0 / (R_GAS_DRY_AIR * lapse))
        else:
            p_cur *= math.exp(-G0 * dh / (R_GAS_DRY_AIR * t_cur))

    rh = max(0.0, min(100.0, float(rh_pct)))
    t_c = t_cur - 273.15
    # Buck (1981) saturation vapor pressure [Pa]
    e_sat_pa = 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / max(257.14 + t_c, 1.0)))
    e_vapor = (rh / 100.0) * min(e_sat_pa, 0.5 * p_cur)
    q_spec = 0.62198 * e_vapor / max(p_cur - (1.0 - 0.62198) * e_vapor, 1.0)
    t_virtual_k = t_cur * (1.0 + 0.608 * q_spec)
    rho = p_cur / (R_GAS_DRY_AIR * t_virtual_k)
    gamma_moist = GAMMA_AIR * (1.0 - 0.05 * q_spec)
    a = math.sqrt(gamma_moist * R_GAS_DRY_AIR * t_virtual_k)
    return {
        "geometric_alt_m": z,
        "geopotential_alt_m": H,
        "T_K": float(t_cur),
        "T_virtual_K": float(t_virtual_k),
        "p_Pa": float(p_cur),
        "rho_kg_m3": float(rho),
        "a_mps": float(a),
        "specific_humidity_kg_kg": float(q_spec),
    }


@dataclass
class AtmosphereEnvironment:
    """Altitude-dependent atmosphere & wind vector evaluator for trajectory integration.

    Coordinate convention (local firing / ENU-compatible ballistic frame):
    - x: downrange / North-aligned horizontal axis (m)
    - y: crossrange / deflection (+ right / East) (m)
    - z: altitude Up (m)
    Wait: STANAG 4355 NUE uses North-Up-East or Range-Crossrange-Up; in our 3-D integrator
    we use `[x_range, y_deflection, z_alt]` consistently and document every axis clearly!
    """

    dT_K: float = 0.0
    p_sl_Pa: float = P0_ISA_PA
    # Wind bands: list of (alt_m, wind_range_mps, wind_cross_mps, wind_up_mps)
    wind_bands: list[tuple[float, float, float, float]] = field(default_factory=list)
    constant_wind_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.wind_bands:
            pts = sorted(self.wind_bands, key=lambda b: b[0])
            self._w_alt = np.array([p[0] for p in pts], dtype=np.float64)
            self._w_x = np.array([p[1] for p in pts], dtype=np.float64)
            self._w_y = np.array([p[2] for p in pts], dtype=np.float64)
            self._w_z = np.array([p[3] for p in pts], dtype=np.float64)
        else:
            self._w_alt = np.array([], dtype=np.float64)
            self._w_x = np.array([], dtype=np.float64)
            self._w_y = np.array([], dtype=np.float64)
            self._w_z = np.array([], dtype=np.float64)

    def wind_at_altitude(self, alt_m: float | np.ndarray) -> np.ndarray:
        """Return wind vector `[..., 3]` (range, crossrange, vertical) in m/s."""
        h = np.asarray(alt_m, dtype=np.float64)
        if self._w_alt.size == 0:
            w = np.broadcast_to(
                np.array(self.constant_wind_mps, dtype=np.float64),
                h.shape + (3,),
            ).copy()
            return w

        wx = np.interp(h, self._w_alt, self._w_x) + self.constant_wind_mps[0]
        wy = np.interp(h, self._w_alt, self._w_y) + self.constant_wind_mps[1]
        wz = np.interp(h, self._w_alt, self._w_z) + self.constant_wind_mps[2]
        w = np.stack([wx, wy, wz], axis=-1)
        return w


def sutherland_viscosity_Pa_s(T_K: float | np.ndarray) -> float | np.ndarray:
    """Sutherland's law for dynamic viscosity of air mu(T) in Pa*s (US Standard Atmosphere 1976, S16).

    mu(T) = beta_s * T^(3/2) / (T + S), where beta_s = 1.458e-6 kg/(m*s*K^(1/2)) and S = 110.4 K.
    """
    t = np.maximum(np.asarray(T_K, dtype=np.float64), 50.0)
    mu = 1.458e-6 * np.power(t, 1.5) / (t + 110.4)
    return float(mu) if np.ndim(T_K) == 0 else mu


def aerodynamic_reynolds_number(
    rho_kg_m3: float | np.ndarray,
    v_rel_mps: float | np.ndarray,
    length_m: float | np.ndarray,
    T_K: float | np.ndarray = T0_ISA_K,
) -> float | np.ndarray:
    """Compute dimensionless Reynolds number Re = rho * |v_rel| * L / mu(T)."""
    mu = sutherland_viscosity_Pa_s(T_K)
    re = (
        np.asarray(rho_kg_m3, dtype=np.float64)
        * np.abs(np.asarray(v_rel_mps, dtype=np.float64))
        * np.asarray(length_m, dtype=np.float64)
        / np.maximum(mu, 1e-12)
    )
    return float(re) if np.ndim(re) == 0 else re
