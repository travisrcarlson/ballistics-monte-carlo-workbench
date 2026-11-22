"""Atmosphere, meteorology, speed of sound, and altitude wind profile models.

Implements Formula IDs EB-03 and EB-04 (US Standard Atmosphere 1976 + deviations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    if v_rel.ndim >= 1 and v_rel.shape[-1] in (2, 3) and np.ndim(T_K) < v_rel.ndim:
        speed_rel = np.linalg.norm(v_rel, axis=-1)
    else:
        speed_rel = np.abs(v_rel)

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
    """EB-04: US Standard Atmosphere 1976 (troposphere <= 11 km & lower stratosphere 11-20+ km).

    Matches the exact live formulas on the `Atmosphere` sheet (validated by V-09 and V-10).
    """
    h = np.asarray(h_m, dtype=np.float64)
    dt = np.asarray(dT_K, dtype=np.float64)
    p0 = np.asarray(p_sl_Pa, dtype=np.float64)

    t_std = np.where(h <= 11000.0, T0_ISA_K - 0.0065 * h, 216.65)
    t_k = np.maximum(t_std + dt, 50.0)

    p_pa = np.where(
        h <= 11000.0,
        p0 * np.power(np.maximum(T0_ISA_K - 0.0065 * h, 10.0) / T0_ISA_K, 5.25588),
        p0 * ((216.65 / T0_ISA_K) ** 5.25588) * np.exp(-0.000157688 * (h - 11000.0)),
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
            return w[0] if np.ndim(alt_m) == 0 else w

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
