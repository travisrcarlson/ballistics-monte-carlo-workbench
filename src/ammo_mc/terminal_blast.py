"""Terminal blast and thermal effects module (TB-01 through TB-16).

Implements Hopkinson-Cranz scaling, TNT equivalence, Kingery-Bulmash hemispherical surface
burst polynomials (7 parameters), Kinney-Graham far-field overpressure, dynamic & Rankine-Hugoniot
reflected pressures, Sachs altitude scaling, Friedlander waveform, Scilly brick-house damage,
Swiss glass-breakage P-I hyperbola, TNO lung injury probit [FLAG], Axelsson chest-wall ODE,
probit conversion, IATG 01.80 simple range safety distances, and thermal dose.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

from ammo_mc.data import get_kb
from ammo_mc.formulas_registry import register_formula

KB_PARAM_ALIASES: dict[str, str] = {
    "ps": "Ps",
    "incident_pressure": "Ps",
    "is": "Is",
    "incident_impulse": "Is",
    "pr": "Pr",
    "reflected_pressure": "Pr",
    "ir": "Ir",
    "reflected_impulse": "Ir",
    "ta": "ta",
    "arrival_time": "ta",
    "u": "U",
    "shock_velocity": "U",
    "td": "td",
    "positive_duration": "td",
}


@register_formula("TB-01")
def tb_01_scaled_distance(
    R_m: float | np.ndarray,
    W_kg: float | np.ndarray,
) -> float | np.ndarray:
    """TB-01: Hopkinson-Cranz scaled distance Z = R / W^(1/3) (m / kg^(1/3))."""
    w_cbrt = np.power(np.maximum(np.asarray(W_kg, dtype=np.float64), 1e-12), 1.0 / 3.0)
    z = np.asarray(R_m, dtype=np.float64) / w_cbrt
    return float(z) if np.ndim(R_m) == 0 and np.ndim(W_kg) == 0 else z


@register_formula("TB-02")
def tb_02_tnt_equivalence(
    W_kg: float | np.ndarray,
    explosive: str = "TNT",
    f_p: float | None = None,
    f_i: float | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """TB-02: Separate TNT-equivalent masses (W_TNT_p, W_TNT_i) for peak pressure and impulse."""
    if f_p is None or f_i is None:
        props = get_kb().get_explosive_properties(explosive)
        fp_val = props["f_tnt_p"] if f_p is None else f_p
        fi_val = props["f_tnt_i"] if f_i is None else f_i
    else:
        fp_val, fi_val = f_p, f_i
    w_arr = np.asarray(W_kg, dtype=np.float64)
    wp = w_arr * fp_val
    wi = w_arr * fi_val
    if np.ndim(W_kg) == 0:
        return float(wp), float(wi)
    return wp, wi


@register_formula("TB-03")
def tb_03_kingery_bulmash(
    Z: float | np.ndarray,
    W: float | np.ndarray = 1.0,
    param: str = "Ps",
) -> float | np.ndarray | dict[str, float | np.ndarray]:
    """TB-03: Kingery-Bulmash hemispherical surface-burst TNT blast parameters (metric).

    Valid for 0.0674 <= Z <= 40 m/kg^(1/3) (0.178 <= Z <= 40 for positive duration `td`).
    Signature matches `App_Spec` R7: `tb_03_kingery_bulmash(Z, W, param)`.
    Pass `param="all"` to compute all 7 parameters at once.
    """
    kb = get_kb()
    if param.strip().lower() == "all":
        return {
            pkey: tb_03_kingery_bulmash(Z, W, pkey)
            for pkey in ("Ps", "Is", "Pr", "Ir", "ta", "U", "td")
        }

    pkey = KB_PARAM_ALIASES.get(param.strip().lower(), param.strip())
    pieces = kb.kb_by_param.get(pkey)
    if not pieces:
        raise ValueError(f"Unknown KB parameter '{param}'. Expected one of {list(kb.kb_by_param.keys())} or 'all'.")

    z_arr = np.asarray(Z, dtype=np.float64)
    w_arr = np.asarray(W, dtype=np.float64)
    bshape = np.broadcast(z_arr, w_arr).shape
    z_b = np.broadcast_to(z_arr, bshape)
    w_cbrt = np.broadcast_to(np.power(np.maximum(w_arr, 1e-12), 1.0 / 3.0), bshape)

    out = np.zeros(bshape, dtype=np.float64)
    log_z = np.log10(np.maximum(z_b, 1e-12))

    for idx_p, piece in enumerate(pieces):
        z_min, z_max = piece["z_min"], piece["z_max"]
        if idx_p == 0:
            mask = (z_b >= z_min * (1.0 - 1e-9)) & (z_b <= z_max)
        else:
            mask = (z_b > z_min) & (z_b <= z_max)
        if not np.any(mask):
            continue
        u_val = piece["k0"] + piece["k1"] * log_z[mask]
        # polyval expects highest power first: C[::-1]
        log_y = np.polyval(piece["c"][::-1], u_val)
        y_val = (10.0**log_y) * piece["multiplier"]
        if piece["scale_by_w_cbrt"]:
            y_val = y_val * w_cbrt[mask]
        out[mask] = y_val

    if np.ndim(Z) == 0 and np.ndim(W) == 0:
        return float(out)
    return out


@register_formula("TB-04")
def tb_04_kinney_graham(
    Z: float | np.ndarray,
    Pa_kPa: float | np.ndarray = 101.325,
) -> float | np.ndarray:
    """TB-04: Kinney & Graham free-air incident peak overpressure Ps (kPa), valid out to Z ~ 500."""
    z = np.maximum(np.asarray(Z, dtype=np.float64), 1e-6)
    pa = np.asarray(Pa_kPa, dtype=np.float64)
    numer = 808.0 * (1.0 + (z / 4.5) ** 2)
    denom = np.sqrt((1.0 + (z / 0.048) ** 2) * (1.0 + (z / 0.32) ** 2) * (1.0 + (z / 1.35) ** 2))
    ps = pa * numer / denom
    return float(ps) if np.ndim(Z) == 0 and np.ndim(Pa_kPa) == 0 else ps


@register_formula("TB-05")
def tb_05_dynamic_pressure(
    Ps: float | np.ndarray,
    Pa: float | np.ndarray = 101.325,
) -> float | np.ndarray:
    """TB-05: Peak dynamic blast wind pressure q = 2.5 * Ps^2 / (7 * Pa + Ps) (same units as Ps, Pa)."""
    ps = np.asarray(Ps, dtype=np.float64)
    pa = np.asarray(Pa, dtype=np.float64)
    q = 2.5 * (ps**2) / np.maximum(7.0 * pa + ps, 1e-12)
    return float(q) if np.ndim(Ps) == 0 and np.ndim(Pa) == 0 else q


@register_formula("TB-06")
def tb_06_sachs_altitude_scaling(
    Pz_kPa: float | np.ndarray,
    Tz_K: float | np.ndarray,
    P0_kPa: float = 101.33,
    T0_K: float = 288.16,
) -> dict[str, float | np.ndarray]:
    """TB-06: Sachs ambient-pressure/temperature blast scaling factors (Sd, Sp, Si, St)."""
    pz = np.asarray(Pz_kPa, dtype=np.float64)
    tz = np.asarray(Tz_K, dtype=np.float64)
    sd = np.power(P0_kPa / pz, 1.0 / 3.0)
    sp = pz / P0_kPa
    si = np.power(pz / P0_kPa, 2.0 / 3.0) * np.sqrt(T0_K / tz)
    st = np.power(P0_kPa / pz, 1.0 / 3.0) * np.sqrt(T0_K / tz)
    if np.ndim(Pz_kPa) == 0 and np.ndim(Tz_K) == 0:
        return {"Sd": float(sd), "Sp": float(sp), "Si": float(si), "St": float(st)}
    return {"Sd": sd, "Sp": sp, "Si": si, "St": st}


@register_formula("TB-07")
def tb_07_normal_reflected_pressure(
    Ps: float | np.ndarray,
    Pa: float | np.ndarray = 101.325,
) -> float | np.ndarray:
    """TB-07: Normal reflected overpressure (Rankine-Hugoniot ideal gas): Pr = 2*Ps*(7*Pa + 4*Ps)/(7*Pa + Ps)."""
    ps = np.asarray(Ps, dtype=np.float64)
    pa = np.asarray(Pa, dtype=np.float64)
    pr = 2.0 * ps * (7.0 * pa + 4.0 * ps) / np.maximum(7.0 * pa + ps, 1e-12)
    return float(pr) if np.ndim(Ps) == 0 and np.ndim(Pa) == 0 else pr


def solve_friedlander_decay_b(Ps_kPa: float, td_ms: float, Is_kPa_ms: float) -> float:
    """Solve for Friedlander decay exponent `b` such that integral_0^{td} p(t) dt = Is."""
    if Ps_kPa <= 0.0 or td_ms <= 0.0 or Is_kPa_ms <= 0.0:
        return 1.0
    ratio = Is_kPa_ms / (Ps_kPa * td_ms)
    if ratio >= 0.499:
        return 0.01
    if ratio <= 0.02:
        return 50.0

    def res(b: float) -> float:
        return (1.0 / b) - (1.0 - np.exp(-b)) / (b * b) - ratio

    try:
        return float(brentq(res, 1e-3, 100.0))
    except ValueError:
        return 1.0


@register_formula("TB-08")
def tb_08_friedlander_waveform(
    t_ms: float | np.ndarray,
    Ps_kPa: float,
    td_ms: float,
    b: float | None = None,
    Is_kPa_ms: float | None = None,
) -> float | np.ndarray:
    """TB-08: Friedlander positive-phase overpressure history p(t) = Ps*(1 - t/td)*exp(-b*t/td)."""
    if b is None:
        b = solve_friedlander_decay_b(Ps_kPa, td_ms, Is_kPa_ms) if Is_kPa_ms is not None else 1.0
    t_arr = np.asarray(t_ms, dtype=np.float64)
    tau = t_arr / max(td_ms, 1e-9)
    p = np.where((tau >= 0.0) & (tau <= 1.0), Ps_kPa * (1.0 - tau) * np.exp(-b * tau), 0.0)
    return float(p) if np.ndim(t_ms) == 0 else p


@register_formula("TB-09")
def tb_09_surface_free_air_equivalence(
    W_kg: float | np.ndarray,
    mode: str = "surface_to_hemi",
    reflection_factor: float = 1.8,
) -> float | np.ndarray:
    """TB-09: Surface vs free-air burst effective charge conversion (1.8 ground reflection factor).

    - `free_air_to_hemi_kb`: equivalent hemispherical KB charge for a free-air burst -> `W / 1.8`
    - `surface_to_free_air`: equivalent free-air Kinney-Graham charge for a surface burst -> `1.8 * W`
    """
    w = np.asarray(W_kg, dtype=np.float64)
    m_low = mode.strip().lower()
    if m_low in ("free_air_to_hemi_kb", "free_to_hemi"):
        res = w / reflection_factor
    else:
        res = w * reflection_factor
    return float(res) if np.ndim(W_kg) == 0 else res


@register_formula("TB-10")
def tb_10_brick_house_damage_range(
    M_kg: float | np.ndarray,
    category: str = "CB",
    method: str = "gilbert_lees_scilly",
) -> float | np.ndarray:
    """TB-10: UK brick-house damage range R_x = K_x * M^(1/3) / (1 + (3175/M)^2)^(1/6)."""
    kb = get_kb()
    cat_up = category.strip().upper()
    k_val = None
    for row in kb.effects_criteria["brick_house_damage"]:
        if row["category"].upper() == cat_up:
            k_val = row["k_jarrett"] if "jarrett" in method.lower() else row["k_gilbert_lees_scilly"]
            break
    if k_val is None:
        raise ValueError(f"Unknown damage category '{category}'. Expected A, B, CB, CA, or D.")
    m = np.maximum(np.asarray(M_kg, dtype=np.float64), 1e-9)
    rx = k_val * np.power(m, 1.0 / 3.0) / np.power(1.0 + (3175.0 / m) ** 2, 1.0 / 6.0)
    return float(rx) if np.ndim(M_kg) == 0 else rx


@register_formula("TB-11")
def tb_11_glass_breakage_swiss(
    P_kPa: float | np.ndarray,
    I_kPa_ms: float | np.ndarray,
    window_size: str = "medium",
    target_probability: float = 0.5,
) -> dict[str, Any]:
    """TB-11: Swiss dual-pane annealed glass breakage P-I hyperbola (P - A)*(I - B) >= C."""
    kb = get_kb()
    ws_low = window_size.strip().lower()
    row = next(
        (r for r in kb.effects_criteria["glass_breakage"] if ws_low in r["key"] or ws_low in r["window_size"].lower()),
        kb.effects_criteria["glass_breakage"][1],
    )
    pr_target = 5.0 + float(sp_stats.norm.ppf(np.clip(target_probability, 1e-4, 1.0 - 1e-4)))
    A_thresh = float(np.exp((pr_target - row["c_a"]) / row["slope_a"]))
    B_thresh = float(np.exp((pr_target - row["c_b"]) / row["slope_b"]))
    C_val = float(np.exp(1.3 + 2.23 * np.log(A_thresh)))

    p_arr = np.asarray(P_kPa, dtype=np.float64)
    i_arr = np.asarray(I_kPa_ms, dtype=np.float64)
    breaks = (p_arr > A_thresh) & (i_arr > B_thresh) & ((p_arr - A_thresh) * (i_arr - B_thresh) >= C_val)
    if np.ndim(P_kPa) == 0 and np.ndim(I_kPa_ms) == 0:
        return {"breaks": bool(breaks), "A_kPa": A_thresh, "B_kPa_ms": B_thresh, "C": C_val}
    return {"breaks": breaks, "A_kPa": A_thresh, "B_kPa_ms": B_thresh, "C": C_val}


@register_formula("TB-12")
def tb_12_lung_lethality_probit(
    P_Pa: float | np.ndarray,
    t0_s: float | np.ndarray,
    p0_Pa: float = 101325.0,
    m_body_kg: float = 70.0,
) -> dict[str, float | np.ndarray]:
    """TB-12 [FLAG]: TNO Green Book primary blast lung lethality probit as printed in IATG 01.80 Table 50."""
    p_arr = np.maximum(np.asarray(P_Pa, dtype=np.float64), 1.0)
    t_arr = np.maximum(np.asarray(t0_s, dtype=np.float64), 1e-6)
    t_sc = t_arr * ((70.0 / m_body_kg) ** (1.0 / 3.0)) * np.sqrt(p0_Pa / 101300.0)
    ps_sc = p_arr / p0_Pa
    denom = 4.17 - 0.00164 * np.log(t_sc) / t_sc + 0.0161 / t_sc
    v_val = np.maximum(ps_sc / np.maximum(denom, 1e-9), 1e-12)
    pr = 5.0 + 5.7 * np.log(v_val)
    prob = tb_14_probit_to_probability(pr)
    if np.ndim(P_Pa) == 0 and np.ndim(t0_s) == 0:
        return {
            "t_scaled_s": float(t_sc),
            "V": float(v_val),
            "probit": float(pr),
            "probability": float(prob),
        }
    return {"t_scaled_s": t_sc, "V": v_val, "probit": pr, "probability": prob}


@register_formula("TB-13")
def tb_13_axelsson_chest_wall(
    Ps_kPa: float,
    td_ms: float,
    Is_kPa_ms: float | None = None,
    p0_Pa: float = 101325.0,
    dt_s: float = 2e-5,
    record_history: bool = False,
) -> dict[str, Any]:
    """TB-13: Axelsson single-point chest-wall dynamic model -> peak chest velocity V (m/s) & ASII."""
    M_cw = 2.03  # kg
    A_cw = 0.082  # m^2
    V0_lung = 0.00182  # m^3
    J_cw = 696.0  # N s / m
    K_cw = 989.0  # N / m
    gam_lung = 1.2

    b = solve_friedlander_decay_b(Ps_kPa, td_ms, Is_kPa_ms) if Is_kPa_ms is not None else 1.0
    td_s = max(td_ms * 1e-3, 1e-5)
    t_end = max(3.0 * td_s, 0.015)
    n_steps = min(20000, max(100, int(t_end / dt_s)))
    dt = t_end / n_steps

    x = 0.0
    v = 0.0
    v_max = 0.0

    hist_t_ms: list[float] = []
    hist_x_mm: list[float] = []
    hist_v_mps: list[float] = []
    hist_p_ext_kpa: list[float] = []
    hist_p_lung_kpa: list[float] = []
    stride = max(1, n_steps // 250)

    def eval_pressures(t_cur: float, x_cur: float) -> tuple[float, float]:
        p_over_pa = (
            Ps_kPa * 1000.0 * (1.0 - t_cur / td_s) * np.exp(-b * t_cur / td_s)
            if t_cur <= td_s
            else 0.0
        )
        vol = max(V0_lung - A_cw * x_cur, 0.15 * V0_lung)
        p_lung_gauge = p0_Pa * ((V0_lung / vol) ** gam_lung - 1.0)
        return float(p_over_pa), float(p_lung_gauge)

    def accel(t_cur: float, x_cur: float, v_cur: float) -> float:
        p_over_pa, p_lung_gauge = eval_pressures(t_cur, x_cur)
        return (A_cw * (p_over_pa - p_lung_gauge) - J_cw * v_cur - K_cw * x_cur) / M_cw

    t = 0.0
    for step_idx in range(n_steps):
        if record_history and (step_idx % stride == 0 or step_idx == n_steps - 1):
            p_ext_pa, p_lung_pa = eval_pressures(t, x)
            hist_t_ms.append(float(t * 1000.0))
            hist_x_mm.append(float(x * 1000.0))
            hist_v_mps.append(float(v))
            hist_p_ext_kpa.append(float(p_ext_pa * 1e-3))
            hist_p_lung_kpa.append(float(p_lung_pa * 1e-3))

        a1 = accel(t, x, v)
        x2, v2 = x + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = accel(t + 0.5 * dt, x2, v2)
        x3, v3 = x + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = accel(t + 0.5 * dt, x3, v3)
        x4, v4 = x + dt * v3, v + dt * a3
        a4 = accel(t + dt, x4, v4)
        x += (dt / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4)
        v += (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        if abs(v) > v_max:
            v_max = abs(v)
        t += dt

    asii = float((0.124 + 0.117 * v_max) ** 2.63)
    if v_max > 12.8 or asii > 3.6:
        level = ">50% lethality"
    elif v_max >= 7.5:
        level = "Moderate to extensive"
    elif v_max >= 3.6:
        level = "Trace to slight"
    else:
        level = "No injury"

    out: dict[str, Any] = {
        "v_chest_max_mps": float(v_max),
        "ASII": asii,
        "injury_level": level,
    }
    if record_history:
        out["history"] = {
            "t_ms": hist_t_ms,
            "x_mm": hist_x_mm,
            "v_mps": hist_v_mps,
            "p_ext_kPa": hist_p_ext_kpa,
            "p_lung_kPa": hist_p_lung_kpa,
        }
    return out


@register_formula("TB-14")
def tb_14_probit_to_probability(probit: float | np.ndarray) -> float | np.ndarray:
    """TB-14: Standard probit conversion P = Phi(Pr - 5) (validated by V-06)."""
    res = sp_stats.norm.cdf(np.asarray(probit, dtype=np.float64) - 5.0)
    return float(res) if np.ndim(probit) == 0 else res


@register_formula("TB-15")
def tb_15_simple_safety_distances(
    auw_kg: float | np.ndarray,
    neq_kg: float | np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """TB-15: IATG 01.80 / 02.20 Quantity-Distance (QD), open-detonation safety & 140 dB noise distances (m)."""
    auw = np.maximum(np.asarray(auw_kg, dtype=np.float64), 0.0)
    m_exp = np.maximum(np.asarray(auw_kg if neq_kg is None else neq_kg, dtype=np.float64), 0.0)
    q13 = np.power(m_exp, 1.0 / 3.0)
    res = {
        "process_building_distance_m": 8.0 * q13,
        "public_traffic_route_distance_m": 14.8 * q13,
        "inhabited_building_distance_m": 22.2 * q13,
        "vulnerable_building_distance_m": 44.4 * q13,
        "frag_public_m": 634.0 * np.power(auw, 1.0 / 6.0),
        "bare_explosive_m": 130.0 * np.power(auw, 1.0 / 3.0),
        "multi_item_linear_m": 370.0 * np.power(auw, 1.0 / 5.0),
        "vertical_single_m": 314.0 * np.power(auw, 1.0 / 3.0),
        "vertical_multi_m": 470.0 * np.power(auw, 1.0 / 5.0),
        "noise_140dB_m": 215.0 * q13,
    }
    if np.ndim(auw_kg) == 0 and (neq_kg is None or np.ndim(neq_kg) == 0):
        return {k: float(v) for k, v in res.items()}
    return res


@register_formula("TB-16")
def tb_16_thermal_dose(
    q_kw_m2: float | np.ndarray,
    t_exposure_s: float | np.ndarray,
) -> dict[str, Any]:
    """TB-16: Thermal radiation dose D = q^(4/3) * t and burn threshold classification."""
    dose = np.power(np.maximum(np.asarray(q_kw_m2, dtype=np.float64), 0.0), 4.0 / 3.0) * np.maximum(
        np.asarray(t_exposure_s, dtype=np.float64), 0.0
    )
    if np.ndim(dose) == 0:
        d_f = float(dose)
        if d_f >= 1000.0:
            eff = "Third-degree burn"
        elif d_f >= 290.0:
            eff = "Second-degree burn"
        elif d_f >= 105.0:
            eff = "First-degree burn"
        elif d_f >= 92.0:
            eff = "Pain threshold"
        else:
            eff = "Below pain threshold"
        return {"dose": d_f, "effect": eff}
    return {"dose": dose, "first_degree": dose >= 105.0, "second_degree": dose >= 290.0, "third_degree": dose >= 1000.0}


def evaluate_blast_field(
    R_m: float | np.ndarray,
    w_fill_kg: float | np.ndarray,
    explosive: str = "TNT",
    hob_m: float | np.ndarray = 0.0,
    alt_m: float | np.ndarray = 0.0,
    dT_K: float = 0.0,
    p_sl_Pa: float = 101325.0,
    f_p_override: float | None = None,
    f_i_override: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Full blast field pipeline per App_Spec R18:

    - Separate TNT equivalence for pressure & impulse (`TB-02`)
    - Surface vs free-air reflection (`TB-09`) when `hob_m > 2.0 * W^(1/3)`
    - Sachs altitude scaling (`TB-06`)
    - Kingery-Bulmash (`TB-03`) inside `0.0674 <= Z <= 40`
    - Kinney-Graham (`TB-04`) beyond `Z > 40` (with surface reflection factor)
    - Dynamic (`TB-05`) & Rankine-Hugoniot reflected (`TB-07`) pressures.
    """
    from ammo_mc.environment import eb_04_isa_atmosphere

    r_arr = np.maximum(np.asarray(R_m, dtype=np.float64), 0.05)
    wp, wi = tb_02_tnt_equivalence(w_fill_kg, explosive=explosive, f_p=f_p_override, f_i=f_i_override)
    wp_arr = np.asarray(wp, dtype=np.float64)
    wi_arr = np.asarray(wi, dtype=np.float64)
    hob_arr = np.asarray(hob_m, dtype=np.float64)

    # Slant distance from burst point at height hob_m
    slant_r = np.hypot(r_arr, hob_arr)
    is_free_air = hob_arr > 1.5 * np.power(np.maximum(wp_arr, 1e-6), 1.0 / 3.0)

    # Effective hemispherical TNT mass for KB (divide by 1.8 if free-air)
    wp_hemi = np.where(is_free_air, tb_09_surface_free_air_equivalence(wp_arr, "free_to_hemi"), wp_arr)
    wi_hemi = np.where(is_free_air, tb_09_surface_free_air_equivalence(wi_arr, "free_to_hemi"), wi_arr)

    atm = eb_04_isa_atmosphere(alt_m, dT_K=dT_K, p_sl_Pa=p_sl_Pa)
    pz_kpa = np.asarray(atm["p_Pa"], dtype=np.float64) / 1000.0
    tz_k = np.asarray(atm["T_K"], dtype=np.float64)
    sachs = tb_06_sachs_altitude_scaling(pz_kpa, tz_k)
    sd, sp, si, st = sachs["Sd"], sachs["Sp"], sachs["Si"], sachs["St"]

    zp = np.asarray(tb_01_scaled_distance(slant_r, wp_hemi), dtype=np.float64) / sd
    zi = np.asarray(tb_01_scaled_distance(slant_r, wi_hemi), dtype=np.float64) / sd

    zp_clamped = np.maximum(zp, 0.0675)
    zi_clamped = np.maximum(zi, 0.0675)

    # Inside Z <= 40: Kingery-Bulmash; beyond Z > 40: Kinney-Graham (using 1.8*wp_hemi free-air equiv)
    ps_kb = np.asarray(tb_03_kingery_bulmash(np.minimum(zp_clamped, 40.0), wp_hemi, "Ps"), dtype=np.float64)
    z_kg_free = slant_r / (np.power(np.maximum(wp_hemi * 1.8, 1e-9), 1.0 / 3.0) * sd)
    ps_kg = np.asarray(tb_04_kinney_graham(z_kg_free, Pa_kPa=101.325), dtype=np.float64)
    ps_sl = np.where(zp_clamped <= 40.0, ps_kb, ps_kg)
    ps_kpa = ps_sl * sp

    is_sl = np.asarray(tb_03_kingery_bulmash(np.minimum(zi_clamped, 40.0), wi_hemi, "Is"), dtype=np.float64)
    # Far-field acoustic impulse decay (~1/Z) beyond Z=40
    is_far = np.asarray(tb_03_kingery_bulmash(40.0, wi_hemi, "Is"), dtype=np.float64) * (40.0 / np.maximum(zi_clamped, 40.0))
    is_kpa_ms = np.where(zi_clamped <= 40.0, is_sl, is_far) * si

    td_sl = np.asarray(tb_03_kingery_bulmash(np.clip(zp_clamped, 0.18, 40.0), wp_hemi, "td"), dtype=np.float64)
    td_ms = td_sl * st

    q_kpa = np.asarray(tb_05_dynamic_pressure(ps_kpa, pz_kpa), dtype=np.float64)
    pr_kb = np.asarray(tb_03_kingery_bulmash(np.minimum(zp_clamped, 40.0), wp_hemi, "Pr"), dtype=np.float64) * sp
    pr_rh = np.asarray(tb_07_normal_reflected_pressure(ps_kpa, pz_kpa), dtype=np.float64)
    pr_kpa = np.where(zp_clamped <= 40.0, pr_kb, pr_rh)

    if np.ndim(R_m) == 0 and np.ndim(w_fill_kg) == 0:
        return {
            "Z_eff": float(zp),
            "Ps_kPa": float(ps_kpa),
            "Is_kPa_ms": float(is_kpa_ms),
            "Pr_kPa": float(pr_kpa),
            "q_kPa": float(q_kpa),
            "td_ms": float(td_ms),
        }
    return {
        "Z_eff": zp,
        "Ps_kPa": ps_kpa,
        "Is_kPa_ms": is_kpa_ms,
        "Pr_kPa": pr_kpa,
        "q_kPa": q_kpa,
        "td_ms": td_ms,
    }


# Aliases for convenience and backward compatibility
tb_01_hopkinson_cranz_scaled_distance = tb_01_scaled_distance
tb_03_kingery_bulmash_blast = tb_03_kingery_bulmash
