"""Terminal fragmentation effects module (TF-01 through TF-15).

Implements Gurney velocity equations (cylinder, sphere, symmetric/asymmetric/open/tamped
sandwiches, axial end-effect correction), Mott fragment mass distribution & inverse-CDF
sampler, THOR presented area & perforation/residual velocity [FLAG], Taylor projection
angle, polar spray density, Poisson hit/kill, AASTP-1 stack critical density, 79 J
hazard criterion, De Marre naval armour penetration, RC debris throw, and TP-14 union.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ammo_mc.data import get_kb
from ammo_mc.formulas_registry import register_formula

LB_PER_KG = 2.2046226218487757
KG_PER_LB = 0.45359237
GRAINS_PER_GRAM = 15.4324
IN_PER_CM = 0.393701
IN2_PER_CM2 = 0.155
FPS_PER_MPS = 3.28084

SHAPE_FACTORS_CM2_G23: dict[str, float] = {
    "sphere": 0.3079,
    "cube": 0.3799,
    "random": 0.5199,
}


@register_formula("TF-01")
def tf_01_gurney_cyl_sphere(
    gurney_mps: float | np.ndarray,
    M_case_kg: float | np.ndarray,
    C_charge_kg: float | np.ndarray,
    geometry: str = "cylinder",
) -> float | np.ndarray:
    """TF-01: Gurney initial fragment velocity for cylinder (k=1/2) or sphere (k=3/5) (validated V-07)."""
    mc_ratio = np.asarray(M_case_kg, dtype=np.float64) / np.maximum(np.asarray(C_charge_kg, dtype=np.float64), 1e-12)
    k_geom = 0.6 if geometry.strip().lower().startswith("sph") else 0.5
    v = np.asarray(gurney_mps, dtype=np.float64) * np.power(mc_ratio + k_geom, -0.5)
    return float(v) if np.ndim(v) == 0 else v


@register_formula("TF-02")
def tf_02_gurney_sandwiches(
    gurney_mps: float | np.ndarray,
    M_flyer_kg: float | np.ndarray,
    C_charge_kg: float | np.ndarray,
    configuration: str = "symmetric",
    N_tamper_kg: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """TF-02: Gurney velocity for symmetric, asymmetric, open-faced, and infinitely tamped sandwiches."""
    G = np.asarray(gurney_mps, dtype=np.float64)
    mc = np.asarray(M_flyer_kg, dtype=np.float64) / np.maximum(np.asarray(C_charge_kg, dtype=np.float64), 1e-12)
    cfg = configuration.strip().lower()

    if cfg.startswith("sym"):
        v = G * np.power(2.0 * mc + 1.0 / 3.0, -0.5)
    elif cfg.startswith("asym"):
        nc = np.asarray(N_tamper_kg, dtype=np.float64) / np.maximum(np.asarray(C_charge_kg, dtype=np.float64), 1e-12)
        A = (1.0 + 2.0 * mc) / (1.0 + 2.0 * nc)
        v = G * np.power((1.0 + A**3) / (3.0 * (1.0 + A)) + (A**2) * nc + mc, -0.5)
    elif cfg.startswith("open"):
        v = G * np.power(((1.0 + (1.0 + 2.0 * mc) ** 3) / (6.0 * (1.0 + mc))) + mc, -0.5)
    elif "tamp" in cfg or cfg.startswith("inf"):
        v = G * np.power(mc + 1.0 / 3.0, -0.5)
    else:
        raise ValueError(f"Unknown sandwich configuration '{configuration}'.")
    return float(v) if np.ndim(v) == 0 else v


@register_formula("TF-03")
def tf_03_gurney_estimate_axial(
    D_det_mps: float | np.ndarray,
    x_m: float | np.ndarray | None = None,
    L_m: float | None = None,
    d_m: float | None = None,
    v_gurney_mps: float | np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """TF-03: Gurney constant estimate G ~ 0.338*D and end-initiated cylindrical axial correction."""
    g_est = 0.338 * np.asarray(D_det_mps, dtype=np.float64)
    if x_m is None or L_m is None or d_m is None:
        return {"gurney_est_mps": float(g_est) if np.ndim(g_est) == 0 else g_est, "axial_factor": 1.0}

    x_arr = np.asarray(x_m, dtype=np.float64)
    cx = (1.0 - np.exp(-2.3617 * np.maximum(x_arr, 0.0) / d_m)) * (
        1.0 - 0.28806 * np.exp(-4.603 * np.maximum(L_m - x_arr, 0.0) / d_m)
    )
    vg = g_est if v_gurney_mps is None else np.asarray(v_gurney_mps, dtype=np.float64)
    vx = vg * cx
    if np.ndim(vx) == 0:
        return {"gurney_est_mps": float(g_est), "axial_factor": float(cx), "v_axial_mps": float(vx)}
    return {"gurney_est_mps": g_est, "axial_factor": cx, "v_axial_mps": vx}


@register_formula("TF-04")
def tf_04_mott_distribution(
    M0_lb: float | None = None,
    t_in: float = 0.5,
    d_in: float = 5.0,
    B: float = 0.0779,
    m_thresh_lb: float | np.ndarray | None = None,
    *,
    M0_kg: float | None = None,
) -> dict[str, Any]:
    """TF-04: Mott natural fragmentation distribution (natively imperial per IATG 01.80 5.5.3)."""
    if M0_lb is None:
        if M0_kg is None:
            raise ValueError("Provide either `M0_lb` or `M0_kg`.")
        M0_lb = M0_kg * LB_PER_KG

    mk = float(B * (t_in ** (5.0 / 16.0)) * (d_in ** (1.0 / 3.0)) * (1.0 + t_in / d_in))
    nt = float(M0_lb / (2.0 * (mk**2)))
    m_avg_lb = float(2.0 * (mk**2))
    m_avg_kg = m_avg_lb * KG_PER_LB

    out: dict[str, Any] = {
        "MK_lb_sqrt": mk,
        "total_fragments": nt,
        "m_avg_lb": m_avg_lb,
        "m_avg_g": m_avg_kg * 1000.0,
        "m_avg_kg": m_avg_kg,
    }
    if m_thresh_lb is not None:
        mt = np.maximum(np.asarray(m_thresh_lb, dtype=np.float64), 0.0)
        n_gt = nt * np.exp(-np.sqrt(mt) / mk)
        out["n_heavier"] = float(n_gt) if np.ndim(m_thresh_lb) == 0 else n_gt
    return out


@register_formula("TF-05")
def tf_05_mott_sampler(
    MK_lb_sqrt: float,
    n_fragments: int = 1,
    rng: np.random.Generator | None = None,
    u: np.ndarray | None = None,
    unit: str = "kg",
) -> np.ndarray:
    """TF-05: Mott inverse-CDF individual fragment mass sampler m = (MK * ln(1/U))^2."""
    if u is not None:
        u_draws = np.clip(np.asarray(u, dtype=np.float64), 1e-15, 1.0)
    else:
        gen = rng if rng is not None else np.random.default_rng()
        u_draws = np.clip(gen.random(size=n_fragments), 1e-15, 1.0)
    m_lb = (MK_lb_sqrt * np.log(1.0 / u_draws)) ** 2
    u_low = unit.strip().lower()
    if u_low == "lb":
        return m_lb
    if u_low == "g":
        return m_lb * (KG_PER_LB * 1000.0)
    return m_lb * KG_PER_LB


@register_formula("TF-06")
def tf_06_fragment_presented_area(
    m_kg: float | np.ndarray | None = None,
    shape: str | float = "random",
    *,
    m_g: float | np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """TF-06: Average presented fragment area A = K * m^(2/3) (returns both cm^2, m^2, and in^2)."""
    if m_g is None:
        if m_kg is None:
            raise ValueError("Provide either `m_kg` or `m_g`.")
        mg = np.asarray(m_kg, dtype=np.float64) * 1000.0
    else:
        mg = np.asarray(m_g, dtype=np.float64)

    if isinstance(shape, (int, float)):
        k_val = float(shape)
    else:
        k_val = SHAPE_FACTORS_CM2_G23.get(shape.strip().lower(), 0.5199)

    a_cm2 = k_val * np.power(np.maximum(mg, 1e-12), 2.0 / 3.0)
    a_m2 = a_cm2 * 1e-4
    a_in2 = a_cm2 * IN2_PER_CM2
    if np.ndim(mg) == 0:
        return {"A_cm2": float(a_cm2), "A_m2": float(a_m2), "A_in2": float(a_in2), "K": k_val}
    return {"A_cm2": a_cm2, "A_m2": a_m2, "A_in2": a_in2, "K": k_val}


@register_formula("TF-07")
def tf_07_fragment_spray_density(
    n_zone: float | np.ndarray,
    R_m: float | np.ndarray,
    theta1_rad: float = 0.0,
    theta2_rad: float = np.pi,
) -> float | np.ndarray:
    """TF-07: Expected fragment areal density rho_f = N_zone / (Omega * R^2) in polar zone [theta1, theta2]."""
    omega = 2.0 * np.pi * abs(np.cos(theta1_rad) - np.cos(theta2_rad))
    omega = max(float(omega), 1e-9)
    rho_f = np.asarray(n_zone, dtype=np.float64) / (omega * np.maximum(np.asarray(R_m, dtype=np.float64) ** 2, 1e-6))
    return float(rho_f) if np.ndim(n_zone) == 0 and np.ndim(R_m) == 0 else rho_f


@register_formula("TF-08")
def tf_08_taylor_angle(
    v0_mps: float | np.ndarray,
    D_det_mps: float | np.ndarray,
) -> float | np.ndarray:
    """TF-08: Taylor fragment projection angle deflection sin(theta_T) = V0 / (2 * D) (rad)."""
    ratio = np.clip(
        np.asarray(v0_mps, dtype=np.float64) / np.maximum(2.0 * np.asarray(D_det_mps, dtype=np.float64), 1.0),
        -1.0,
        1.0,
    )
    th = np.arcsin(ratio)
    return float(th) if np.ndim(th) == 0 else th


@register_formula("TF-09")
def tf_09_poisson_hit_kill(
    rho_f_m2: float | np.ndarray,
    A_target_m2: float | np.ndarray = 0.56,
    pk_given_hit: float | np.ndarray = 1.0,
) -> dict[str, float | np.ndarray]:
    """TF-09: Poisson fragment hit & kill probabilities P_hit = 1 - exp(-rho_f * A_T)."""
    expected_hits = np.maximum(np.asarray(rho_f_m2, dtype=np.float64) * np.asarray(A_target_m2, dtype=np.float64), 0.0)
    p_hit = 1.0 - np.exp(-expected_hits)
    p_kill = 1.0 - np.exp(-expected_hits * np.clip(np.asarray(pk_given_hit, dtype=np.float64), 0.0, 1.0))
    if np.ndim(expected_hits) == 0 and np.ndim(pk_given_hit) == 0:
        return {"expected_hits": float(expected_hits), "p_hit": float(p_hit), "p_kill": float(p_kill)}
    return {"expected_hits": expected_hits, "p_hit": p_hit, "p_kill": p_kill}


@register_formula("TF-10")
def tf_10_aastp1_stack_density(
    R_m: float | np.ndarray,
    N_t_per_item: float,
    M0_kg_per_item: float,
    V0_mps: float,
    L_decay_m: float = 150.0,
    N_surface: int = 1,
    N_total: int = 1,
    storage_mode: str = "open",
    E_crit_J: float = 79.0,
    mean_frag_mass_kg: float | None = None,
) -> dict[str, Any]:
    """TF-10: AASTP-1 critical hazardous fragment density from munition stacks (IATG 01.80 Table 28)."""
    r_arr = np.maximum(np.asarray(R_m, dtype=np.float64), 1.0)
    n_eff = (0.7 * N_surface + 0.1 * N_total) if "ecm" in storage_mode.lower() else (0.9 * N_surface + 0.1 * N_total)
    q0_eff = (N_t_per_item / (4.0 * np.pi)) * max(n_eff, 1.0)

    v_i = V0_mps * np.exp(-r_arr / max(L_decay_m, 1.0))
    m_cr_low = 2.0 * E_crit_J / np.maximum(v_i**2, 1e-6)
    m_cr_high = (2.0 * E_crit_J / (9.81 * max(L_decay_m, 1.0))) ** 0.75
    m_f = np.minimum(m_cr_low, m_cr_high)

    # AASTP-1 / IATG Table 28 Mott tail uses mean fragment mass (2 * M_K^2 = M_case / N_t), NOT whole casing mass
    if mean_frag_mass_kg is not None and mean_frag_mass_kg > 0.0:
        m_avg = float(mean_frag_mass_kg)
    elif M0_kg_per_item > 0.5 and N_t_per_item > 10.0:
        m_avg = float(M0_kg_per_item) / float(N_t_per_item)
    else:
        m_avg = max(float(M0_kg_per_item), 1e-6)

    q_f = (q0_eff / (r_arr**2)) * np.exp(-np.sqrt(2.0 * m_f / max(m_avg, 1e-6)))
    exceeds_crit = q_f >= (1.0 / 56.0)
    if np.ndim(R_m) == 0:
        return {"q_f_m2": float(q_f), "exceeds_1_per_56m2": bool(exceeds_crit), "N_effective": float(n_eff)}
    return {"q_f_m2": q_f, "exceeds_1_per_56m2": exceeds_crit, "N_effective": float(n_eff)}


@register_formula("TF-11")
def tf_11_hazardous_fragment_ke(
    m_kg: float | np.ndarray,
    v_mps: float | np.ndarray,
    e_crit_J: float = 79.0,
) -> tuple[bool | np.ndarray, float | np.ndarray]:
    """TF-11: NATO/DDESB 79 J hazardous fragment kinetic energy criterion."""
    ke = 0.5 * np.asarray(m_kg, dtype=np.float64) * (np.asarray(v_mps, dtype=np.float64) ** 2)
    haz = ke >= e_crit_J
    if np.ndim(ke) == 0:
        return bool(haz), float(ke)
    return haz, ke


@register_formula("TF-12")
def tf_12_thor_penetration(
    m_g: float | np.ndarray,
    t_cm: float | np.ndarray,
    v_s_mps: float | np.ndarray,
    obliquity_rad: float | np.ndarray = 0.0,
    target_material: str = "RHA armor",
    shape: str | float = "random",
) -> dict[str, Any]:
    """TF-12 [FLAG]: Classic Project THOR fragment perforation, V50, residual velocity & residual mass.

    Matches the exact unit conversions in `Calculators` R89..R98 (validated against V-12 Excel value).
    """
    coeffs = get_kb().get_thor_material(target_material)
    c1, a1, b1, g1, l1 = coeffs["c1"], coeffs["alpha1"], coeffs["beta1"], coeffs["gamma1"], coeffs["lambda1"]
    c2, a2, b2, g2, l2 = coeffs["c2"], coeffs["alpha2"], coeffs["beta2"], coeffs["gamma2"], coeffs["lambda2"]

    mg_arr = np.asarray(m_g, dtype=np.float64)
    area_info = tf_06_fragment_presented_area(m_g=mg_arr, shape=shape)
    A_in2 = np.asarray(area_info["A_in2"], dtype=np.float64)
    t_in = np.asarray(t_cm, dtype=np.float64) * IN_PER_CM
    m_gr = mg_arr * GRAINS_PER_GRAM
    vs_fps = np.asarray(v_s_mps, dtype=np.float64) * FPS_PER_MPS
    sec_th = 1.0 / np.maximum(np.cos(np.asarray(obliquity_rad, dtype=np.float64)), 0.05)

    tA = np.maximum(t_in * A_in2, 1e-12)
    v_drop_fps = (10.0**c1) * (tA**a1) * (m_gr**b1) * (sec_th**g1) * (np.maximum(vs_fps, 1e-6) ** l1)
    vr_fps = np.maximum(0.0, vs_fps - v_drop_fps)
    vr_mps = vr_fps / FPS_PER_MPS

    v50_fps = np.power((10.0**c1) * (tA**a1) * (m_gr**b1) * (sec_th**g1), 1.0 / (1.0 - l1))
    v50_mps = v50_fps / FPS_PER_MPS

    m_drop_gr = (10.0**c2) * (tA**a2) * (m_gr**b2) * (sec_th**g2) * (np.maximum(vs_fps, 1e-6) ** l2)
    mr_g = np.where(vr_mps > 0.0, np.maximum(0.0, m_gr - m_drop_gr) / GRAINS_PER_GRAM, 0.0)
    perforated = vr_mps > 0.0

    if np.ndim(m_g) == 0 and np.ndim(t_cm) == 0 and np.ndim(v_s_mps) == 0:
        return {
            "v_residual_mps": float(vr_mps),
            "v50_mps": float(v50_mps),
            "m_residual_g": float(mr_g),
            "perforated": bool(perforated),
            "A_in2": float(A_in2),
            "v_drop_fps": float(v_drop_fps),
        }
    return {
        "v_residual_mps": vr_mps,
        "v50_mps": v50_mps,
        "m_residual_g": mr_g,
        "perforated": perforated,
        "A_in2": A_in2,
        "v_drop_fps": v_drop_fps,
    }


@register_formula("TF-13")
def tf_13_de_marre_penetration(
    D_in: float | np.ndarray,
    W_lb: float | np.ndarray,
    V_fps: float | np.ndarray,
    obliquity_rad: float | np.ndarray = 0.0,
    C_de_marre: float = 1.0,
    variant: str = "nickel_steel",
) -> float | np.ndarray:
    """TF-13: Historical De Marre armour penetration thickness T (inches) for cross-checking."""
    D = np.asarray(D_in, dtype=np.float64)
    W = np.asarray(W_lb, dtype=np.float64)
    V = np.asarray(V_fps, dtype=np.float64)
    cos_ob = np.maximum(np.cos(np.asarray(obliquity_rad, dtype=np.float64)), 0.0)
    if "wrought" in variant.lower():
        T_in = D * 0.00002778 * (D**0.1542) * (((W / (D**3)) * (V**2) * (cos_ob**2)) ** 0.7695)
    else:
        T_in = (
            D
            * 0.00005021
            * (D**0.07144)
            * (((W / (D**3)) * ((V / C_de_marre) ** 2) * (cos_ob**3)) ** 0.71429)
        )
    return float(T_in) if np.ndim(T_in) == 0 else T_in


@register_formula("TF-14")
def tf_14_rc_debris_throw(
    neq_kg: float | np.ndarray,
    room_vol_m3: float,
    wall_t_m: float,
    wall_rho_kg_m3: float = 2400.0,
    L_av_m: float = 0.15,
) -> dict[str, float | np.ndarray]:
    """TF-14: Reinforced-concrete debris launch velocity DLV and average debris mass."""
    dlv = 525.0 * np.sqrt(
        np.maximum(np.asarray(neq_kg, dtype=np.float64), 0.0)
        / max((room_vol_m3 ** (2.0 / 3.0)) * wall_t_m * wall_rho_kg_m3, 1e-9)
    )
    m_av_kg = 6.0 * wall_rho_kg_m3 * (L_av_m**3)
    if np.ndim(neq_kg) == 0:
        return {"dlv_mps": float(dlv), "m_avg_debris_kg": float(m_av_kg)}
    return {"dlv_mps": dlv, "m_avg_debris_kg": m_av_kg}


@register_formula("TF-15")
def tf_15_tp14_consequence_union(
    concern_areas_m2: list[float] | np.ndarray,
    n_frags_per_bin: list[float] | np.ndarray,
    vulnerabilities: list[float] | np.ndarray | float = 1.0,
    p_high_angle: float | None = None,
) -> dict[str, Any]:
    """TF-15: DDESB TP-14 fragment consequence probability by mass bin and independent union."""
    ca = np.asarray(concern_areas_m2, dtype=np.float64)
    n_bin = np.asarray(n_frags_per_bin, dtype=np.float64)
    v_bin = np.broadcast_to(np.asarray(vulnerabilities, dtype=np.float64), ca.shape)

    p_bins = np.clip(v_bin * (1.0 - np.exp(-np.maximum(ca * n_bin, 0.0))), 0.0, 1.0)
    p_low = float(1.0 - np.prod(1.0 - p_bins))
    if p_high_angle is not None:
        ph = float(np.clip(p_high_angle, 0.0, 1.0))
        p_tot = ph + p_low * (1.0 - ph)
    else:
        p_tot = p_low
    return {"p_bins": p_bins, "p_angle_union": p_low, "p_total": p_tot}
