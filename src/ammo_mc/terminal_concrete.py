"""Concrete Penetration, Perforation, and Back-Face Scabbing Models.

Implements:
- Forrestal, Altman, Cargile & Hanchak (1994) / Frew et al. (1998) spherical cavity-expansion
  concrete penetration depth for ogive-nose rigid projectiles.
- Modified NDRC (Kennedy 1976 / TM 5-855-1) concrete depth, perforation thickness, and
  back-face spall/scabbing threshold thickness.
"""

from __future__ import annotations

import math
from typing import Any


def forrestal_frew_concrete_penetration(
    mass_kg: float,
    diameter_m: float,
    velocity_mps: float,
    crh_psi: float = 3.0,
    fc_mpa: float = 35.0,
    concrete_density_kg_m3: float = 2300.0,
) -> dict[str, Any]:
    """Compute rigid ogive-nose concrete penetration depth [m, mm] via Forrestal-Frew (1994/1998).

    Formulation:
      S  = 82.6 * (f'_c_MPa)^(-0.544)                 [Frew et al. 1998 empirical static resistance]
      N* = (8 * psi - 1) / (24 * psi^2)               [Ogive nose geometry factor, psi = CRH >= 0.5]
      Crater depth: 4*a = 2*d (surface crater stage)
      Transition velocity V1^2 = (M*Vs^2 - 4*pi*a^3*S*f'_c) / (M + 4*pi*a^3*N*rho)
      Deep tunnel (V1^2 > 0):
        P = 4*a + (M / (2*pi*a^2*N*rho)) * ln(1 + N*rho*V1^2 / (S*f'_c))
    """
    m = max(float(mass_kg), 1e-4)
    d = max(float(diameter_m), 1e-4)
    a = 0.5 * d
    vs = max(float(velocity_mps), 0.0)
    psi = max(float(crh_psi), 0.5)
    fc_m = max(float(fc_mpa), 5.0)
    fc_pa = fc_m * 1.0e6
    rho = max(float(concrete_density_kg_m3), 1000.0)

    n_star = (8.0 * psi - 1.0) / (24.0 * (psi ** 2))
    n_star = max(min(n_star, 1.0), 1e-3)
    s_coeff = 82.6 * (fc_m ** -0.544)
    r_static_pa = s_coeff * fc_pa

    # Energy threshold to traverse the initial 2*d (4*a) surface crater zone
    c_vol = 4.0 * math.pi * (a ** 3)
    numer_v1_sq = m * (vs ** 2) - c_vol * r_static_pa
    denom_v1_sq = m + c_vol * n_star * rho

    if numer_v1_sq > 0.0:
        v1_sq = numer_v1_sq / denom_v1_sq
        prefactor_m = m / (2.0 * math.pi * (a ** 2) * n_star * rho)
        p_m = (4.0 * a) + prefactor_m * math.log(1.0 + (n_star * rho * v1_sq) / r_static_pa)
        regime = "deep_tunnel"
    else:
        # Shallow surface cratering regime (P < 2*d), continuous at V1 = 0
        v_crater_sq = (c_vol * r_static_pa) / m
        p_m = (4.0 * a) * ((vs ** 2) / max(v_crater_sq, 1e-9))
        regime = "shallow_crater"

    return {
        "penetration_m": float(p_m),
        "penetration_mm": float(p_m * 1000.0),
        "penetration_calibers": float(p_m / d),
        "s_resistance_factor": float(s_coeff),
        "nose_factor_n_star": float(n_star),
        "static_cavity_stress_MPa": float(r_static_pa * 1e-6),
        "regime": regime,
        "crater_threshold_m": float(4.0 * a),
    }


def ndrc_concrete_perforation_and_scabbing(
    mass_kg: float,
    diameter_m: float,
    velocity_mps: float,
    fc_mpa: float = 35.0,
    nose_factor_n: float = 1.0,
    wall_thickness_m: float | None = None,
) -> dict[str, Any]:
    """Compute Modified NDRC (TM 5-855-1 / Kennedy 1976) concrete depth, perforation, & scabbing limits.

    Args:
        nose_factor_n: NDRC nose shape coefficient (0.72 flat, 0.84 blunt, 1.0 spherical, 1.14 sharp ogive).
    """
    m_lb = max(float(mass_kg), 1e-4) * 2.20462262185
    d_in = max(float(diameter_m), 1e-4) / 0.0254
    v_fps = max(float(velocity_mps), 0.0) * 3.280839895
    fc_psi = max(float(fc_mpa), 5.0) * 145.03773773
    n_nose = max(float(nose_factor_n), 0.5)

    g_ndrc = (180.0 * n_nose * m_lb / (math.sqrt(fc_psi) * (d_in ** 2.8))) * ((v_fps / 1000.0) ** 1.8)
    if g_ndrc <= 1.0:
        x_over_d = 2.0 * math.sqrt(max(g_ndrc, 0.0))
    else:
        x_over_d = g_ndrc + 1.0

    # Continuous Modified NDRC perforation thickness hp and back-face scabbing thickness hs
    if x_over_d <= 1.35:
        hp_over_d = 3.19 * x_over_d - 0.718 * (x_over_d ** 2)
    else:
        hp_over_d = 1.32 + 1.24 * x_over_d

    if x_over_d <= 0.65:
        hs_over_d = 7.91 * x_over_d - 5.06 * (x_over_d ** 2)
    else:
        hs_over_d = 2.12 + 1.36 * x_over_d

    x_m = x_over_d * float(diameter_m)
    hp_m = hp_over_d * float(diameter_m)
    hs_m = hs_over_d * float(diameter_m)

    out: dict[str, Any] = {
        "g_factor": float(g_ndrc),
        "penetration_m": float(x_m),
        "penetration_mm": float(x_m * 1000.0),
        "perforation_limit_m": float(hp_m),
        "perforation_limit_mm": float(hp_m * 1000.0),
        "scabbing_limit_m": float(hs_m),
        "scabbing_limit_mm": float(hs_m * 1000.0),
    }
    if wall_thickness_m is not None:
        w = max(float(wall_thickness_m), 1e-4)
        out["wall_thickness_m"] = w
        out["perforates_wall"] = bool(w <= hp_m)
        out["spalls_back_face"] = bool(w <= hs_m)
    return out
