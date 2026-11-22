"""Kinetic-energy and shaped-charge armour penetration module (TP-01 through TP-06)."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats as sp_stats

from ammo_mc.data import get_kb
from ammo_mc.formulas_registry import register_formula


@register_formula("TP-01")
def tp_01_lanz_odermatt(
    Lw_m: float | np.ndarray,
    D_m: float | np.ndarray,
    V_mps: float | np.ndarray,
    rho_p_kg_m3: float | np.ndarray = 17600.0,
    rho_t_kg_m3: float | np.ndarray = 7850.0,
    bhn_t: float | np.ndarray = 300.0,
    obliquity_rad: float | np.ndarray = 0.0,
    rod_material: str = "Tungsten alloy",
    mode: str = "semi-infinite",
    bhn_p: float = 600.0,
) -> dict[str, Any]:
    """TP-01 [FLAG]: Lanz-Odermatt APFSDS long-rod penetration depth P (same length unit as Lw_m).

    Matches `Calculators` R100..R108 (s^2 unit convention: BHN factor x 1e6 / rho_p).
    """
    coeffs = get_kb().get_lanz_odermatt_coeffs(rod_material, mode=mode)
    a = coeffs["a"]
    b0 = coeffs["b0"]
    b1 = coeffs["b1"]
    m_exp = coeffs["m"]
    c0 = coeffs["c0"]
    c1 = coeffs["c1"]

    lw = np.asarray(Lw_m, dtype=np.float64)
    d = np.maximum(np.asarray(D_m, dtype=np.float64), 1e-9)
    v = np.maximum(np.asarray(V_mps, dtype=np.float64), 1.0)
    rp = np.asarray(rho_p_kg_m3, dtype=np.float64)
    rt = np.asarray(rho_t_kg_m3, dtype=np.float64)
    bt = np.asarray(bhn_t, dtype=np.float64)
    cos_phi = np.maximum(np.cos(np.asarray(obliquity_rad, dtype=np.float64)), 0.05)

    if c1 is not None:
        s2 = (c0 + c1 * bt) * bt * 1e6 / rp
    else:
        # Steel (UHSS) rod variant
        k_a = coeffs["k_alpha"] or 0.3598
        n_b = coeffs["n_beta"] or -0.2342
        s2 = c0 * (bt**k_a) * (bhn_p**n_b) * 1e6 / rp

    pen = (
        lw
        * (a / np.tanh(b0 + b1 * (lw / d)))
        * (cos_phi**m_exp)
        * np.sqrt(rp / rt)
        * np.exp(-s2 / (v**2))
    )
    if np.ndim(pen) == 0:
        return {"penetration": float(pen), "s2_m2_s2": float(s2)}
    return {"penetration": pen, "s2_m2_s2": s2}


@register_formula("TP-02")
def tp_02_shaped_charge_hydrodynamic(
    L_jet_m: float | np.ndarray,
    rho_jet_kg_m3: float | np.ndarray = 8960.0,
    rho_target_kg_m3: float | np.ndarray = 7850.0,
) -> float | np.ndarray:
    """TP-02: Birkhoff hydrodynamic limit penetration P = L_jet * sqrt(rho_j / rho_t)."""
    p = np.asarray(L_jet_m, dtype=np.float64) * np.sqrt(
        np.asarray(rho_jet_kg_m3, dtype=np.float64) / np.maximum(np.asarray(rho_target_kg_m3, dtype=np.float64), 1.0)
    )
    return float(p) if np.ndim(p) == 0 else p


@register_formula("TP-03")
def tp_03_standoff_degradation(
    P_opt_m: float | np.ndarray,
    standoff_over_cd: float | np.ndarray,
    curve: list[tuple[float, float]] | None = None,
) -> float | np.ndarray:
    """TP-03: Shaped-charge standoff / jet-breakup degradation curve P(SO) = P_opt * g(SO/CD)."""
    if curve is None:
        # Generic unlined/copper-lined bell curve peaking near 3-5 charge diameters
        x_grid = np.array([0.0, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 20.0], dtype=np.float64)
        g_grid = np.array([0.45, 0.72, 0.92, 1.00, 0.98, 0.80, 0.55, 0.25], dtype=np.float64)
    else:
        x_grid = np.array([pt[0] for pt in curve], dtype=np.float64)
        g_grid = np.array([pt[1] for pt in curve], dtype=np.float64)
    so = np.asarray(standoff_over_cd, dtype=np.float64)
    g_val = np.interp(so, x_grid, g_grid)
    res = np.asarray(P_opt_m, dtype=np.float64) * g_val
    return float(res) if np.ndim(res) == 0 else res


@register_formula("TP-04")
def tp_04_lambert_jonas(
    v_s_mps: float | np.ndarray,
    v_bl_mps: float | np.ndarray,
    a: float = 1.0,
    p: float = 2.0,
) -> float | np.ndarray:
    """TP-04: Lambert-Jonas residual velocity V_r = a * (V_s^p - V_bl^p)^(1/p) for V_s > V_bl."""
    vs = np.asarray(v_s_mps, dtype=np.float64)
    vbl = np.asarray(v_bl_mps, dtype=np.float64)
    diff = np.maximum(0.0, (vs**p) - (vbl**p))
    vr = np.where(vs > vbl, a * np.power(diff, 1.0 / p), 0.0)
    return float(vr) if np.ndim(vr) == 0 else vr


@register_formula("TP-05")
def tp_05_los_thickness(
    t_plate_m: float | np.ndarray,
    obliquity_rad: float | np.ndarray,
) -> float | np.ndarray:
    """TP-05: Line-of-sight armour thickness t_LOS = t / cos(theta)."""
    cos_th = np.maximum(np.cos(np.asarray(obliquity_rad, dtype=np.float64)), 0.02)
    res = np.asarray(t_plate_m, dtype=np.float64) / cos_th
    return float(res) if np.ndim(res) == 0 else res


@register_formula("TP-06")
def tp_06_perforation_probability(
    v_striking_mps: float | np.ndarray,
    v50_mps: float | np.ndarray,
    sigma_v50_mps: float | np.ndarray = 25.0,
) -> float | np.ndarray:
    """TP-06: Probit probability of perforation P(perf | V) = Phi((V - V50) / sigma_V50)."""
    vs = np.asarray(v_striking_mps, dtype=np.float64)
    v50 = np.asarray(v50_mps, dtype=np.float64)
    sig = np.asarray(sigma_v50_mps, dtype=np.float64)
    res = np.where(sig <= 0.0, (vs >= v50).astype(np.float64), sp_stats.norm.cdf((vs - v50) / np.maximum(sig, 1e-9)))
    return float(res) if np.ndim(res) == 0 else res
