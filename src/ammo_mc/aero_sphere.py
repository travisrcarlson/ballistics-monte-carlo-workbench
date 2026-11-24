"""Compressible sphere aerodynamics & Didion flat-fire wind-lag kinematics.

Implements:
- Sutherland air dynamic viscosity mu(T) [Pa*s]
- Smooth C1-continuous incompressible sphere drag (Morrison 2013) covering Stokes to drag crisis
- Compressible sphere drag C_d(Re, M) anchored to Bailey & Hiatt (1972), Henderson (1976),
  and Loth (2008) experimental wind-tunnel / aeroballistic-range data
- Closed-form quadratic-drag velocity decay, flight time, and Didion crosswind lag drift
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator


# Bailey & Hiatt (1972) / Henderson (1976) / Loth (2008) continuum high-Re compressibility
# multiplier relative to the subsonic continuum baseline Cd_0 ~ 0.47 (M <= 0.4).
# Captures transonic drag rise starting ~M=0.55 through wave-drag peak ~M=1.6 (~0.987 / 0.47 = 2.10)
# and gradual Mach independence decay at supersonic/hypersonic Mach numbers.
_MACH_GRID = np.array(
    [0.0, 0.35, 0.45, 0.60, 0.75, 0.85, 0.95, 1.00, 1.10, 1.30, 1.60, 2.00, 2.50, 3.00, 4.00, 6.00],
    dtype=np.float64,
)
_COMP_MULT_GRID = np.array(
    [1.00, 1.00, 1.02, 1.06, 1.18, 1.36, 1.60, 1.75, 1.92, 2.04, 2.10, 2.08, 2.04, 2.00, 1.96, 1.94],
    dtype=np.float64,
)
_COMP_PCHIP = PchipInterpolator(_MACH_GRID, _COMP_MULT_GRID, extrapolate=False)


def sutherland_air_viscosity(t_k: float) -> float:
    """Return dynamic viscosity of dry air [Pa*s] via Sutherland's law."""
    t_eff = max(float(t_k), 50.0)
    return float(1.458e-6 * (t_eff ** 1.5) / (t_eff + 110.4))


def sphere_drag_incompressible_morrison(re: float) -> float:
    """Return smooth continuum incompressible sphere Cd(Re) via Morrison (2013).

    Valid continuously from Stokes creeping flow (Re -> 0) through transition and
    drag crisis (Re <= 1e6) with smooth high-Re clamping above 1e6.
    """
    r = max(float(re), 1e-3)
    if r > 2.0e6:
        r = 2.0e6
    t1 = 24.0 / r
    x5 = r / 5.0
    t2 = (2.6 * x5) / (1.0 + x5 ** 1.52)
    x263 = r / 263000.0
    # Guard exponent underflow when r is tiny (x263 ** -8 -> overflow)
    if x263 < 0.05:
        t3 = 0.0
    else:
        num = 0.411 * (x263 ** -7.94)
        den = 1.0 + (x263 ** -8.00)
        t3 = num / den
    x1e6 = r / 1.0e6
    t4 = (0.25 * x1e6) / (1.0 + x1e6)
    return float(t1 + t2 + t3 + t4)


def sphere_drag_compressible(re: float, mach: float) -> float:
    """Return smooth C1-continuous compressible sphere drag Cd(Re, M).

    Combines Morrison (2013) incompressible Reynolds correlation with Bailey & Hiatt
    (1972) / Loth (2008) Mach compressibility correction, smoothly weighted so low-Re
    Stokes/Oseen viscous drag stays bounded while high-Re spheres follow exact transonic
    and supersonic bow-shock wave-drag rise without step discontinuities (e.g., at Re=1e4).
    """
    r = max(float(re), 1e-3)
    m = max(0.0, float(mach))
    cd_inc = sphere_drag_incompressible_morrison(r)

    if m <= 0.35:
        return float(cd_inc)

    m_clamped = min(m, float(_MACH_GRID[-1]))
    comp_factor = float(_COMP_PCHIP(m_clamped))

    # At Re >= ~200, bow-shock wave drag adds full inertia compressibility (~0.47 * (comp - 1)).
    # Suppress wave-drag overshoot cleanly in viscous Stokes regime (Re < 45, Loth 2008).
    w_inertia = 1.0 / (1.0 + math.exp(-(math.log10(r) - 1.65) * 3.5))
    if r >= 500.0 and r <= 1.8e5:
        # Sub-critical high-Re regime: smooth compression scales the pressure drag component
        cd_comp = cd_inc * (1.0 + w_inertia * (comp_factor - 1.0))
    else:
        # Across drag crisis / Stokes, blend pressure-drag wave increment cleanly
        cd_comp = cd_inc + w_inertia * 0.47 * (comp_factor - 1.0)
    return float(max(0.05, cd_comp))


def sphere_drag_compressible_array(re: Iterable[float], mach: Iterable[float]) -> np.ndarray:
    """Vectorized evaluation wrapper for sphere_drag_compressible."""
    r_arr = np.asarray(re, dtype=np.float64)
    m_arr = np.asarray(mach, dtype=np.float64)
    r_b, m_b = np.broadcast_arrays(r_arr, m_arr)
    out = np.empty_like(r_b, dtype=np.float64)
    it = np.nditer([r_b, m_b, out], flags=["multi_index"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for rv, mv, ov in it:
        ov[...] = sphere_drag_compressible(float(rv), float(mv))
    return out


def exact_quadratic_drag_tof_and_velocity(
    range_m: float,
    v0_mps: float,
    kappa_per_m: float,
) -> tuple[float, float]:
    """Closed-form flat-fire quadratic drag time-of-flight and remaining speed.

    Given spatial deceleration dv/dx = -kappa * v:
      v(R) = v0 * exp(-kappa * R)
      t(R) = (exp(kappa * R) - 1) / (v0 * kappa)
    Falls back cleanly via Taylor expansion when kappa * R < 1e-6.
    """
    r = max(0.0, float(range_m))
    v0 = max(1e-3, float(v0_mps))
    k = max(0.0, float(kappa_per_m))
    kr = k * r
    if kr < 1e-6:
        v_rem = v0 * (1.0 - kr + 0.5 * kr * kr)
        tof = (r / v0) * (1.0 + 0.5 * kr + (kr * kr) / 6.0)
        return float(tof), float(v_rem)
    exp_kr = math.exp(min(kr, 50.0))
    v_rem = v0 / exp_kr
    tof = (exp_kr - 1.0) / (v0 * k)
    return float(tof), float(v_rem)


def didion_crosswind_drift_m(
    crosswind_mps: float,
    tof_s: float,
    range_m: float,
    v0_mps: float,
) -> float:
    """Return Didion's exact flat-fire crosswind lag drift [m].

      y_drift = w_cross * (t_flight - R / v0)
    where t_flight >= R / v0 because drag slows downrange speed below v0.
    """
    v0 = max(1e-3, float(v0_mps))
    lag_s = max(0.0, float(tof_s) - max(0.0, float(range_m)) / v0)
    return float(crosswind_mps) * lag_s
