"""Interior ballistics and muzzle-velocity launch corrections (IB-01 through IB-08).

Includes both the fast firing-table correction path (IB-07/IB-08) and a full
STANAG 4367-style lumped-parameter thermodynamic interior ballistics ODE integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from ammo_mc.formulas_registry import register_formula


@register_formula("IB-01")
def ib_01_vieille_burning_law(
    p_Pa: float | np.ndarray,
    beta: float | np.ndarray,
    alpha: float | np.ndarray,
) -> float | np.ndarray:
    """IB-01: Vieille / Saint Robert linear burn rate r = beta * p^alpha (m/s)."""
    p_pos = np.maximum(np.asarray(p_Pa, dtype=np.float64), 0.0)
    res = np.asarray(beta, dtype=np.float64) * np.power(p_pos, np.asarray(alpha, dtype=np.float64))
    return float(res) if np.ndim(p_Pa) == 0 and np.ndim(beta) == 0 else res


@register_formula("IB-02")
def ib_02_form_function(
    phi: float | np.ndarray,
    r_mps: float | np.ndarray,
    e1_m: float,
    kappa: float = 1.0,
    lam: float = 0.0,
    mu: float = 0.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """IB-02: Propellant geometric form function z(phi) and web burn rate dphi/dt = r / e1."""
    phi_c = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    z = np.clip(kappa * phi_c * (1.0 + lam * phi_c + mu * (phi_c**2)), 0.0, 1.0)
    dphi_dt = np.where(phi_c < 1.0, np.asarray(r_mps, dtype=np.float64) / e1_m, 0.0)
    if np.ndim(phi) == 0 and np.ndim(r_mps) == 0:
        return float(z), float(dphi_dt)
    return z, dphi_dt


@register_formula("IB-03")
def ib_03_noble_abel_eos(
    C_kg: float | np.ndarray,
    z: float | np.ndarray,
    F_J_kg: float | np.ndarray,
    V_g_m3: float | np.ndarray,
    eta_m3_kg: float = 0.001,
) -> float | np.ndarray:
    """IB-03: Noble-Abel real-gas equation of state p = C * z * F / (V_g - eta * C * z)."""
    cz = np.asarray(C_kg, dtype=np.float64) * np.asarray(z, dtype=np.float64)
    free_vol = np.maximum(np.asarray(V_g_m3, dtype=np.float64) - eta_m3_kg * cz, 1e-9)
    p = cz * np.asarray(F_J_kg, dtype=np.float64) / free_vol
    return float(p) if np.ndim(p) == 0 else p


@register_formula("IB-04")
def ib_04_resal_energy(
    z: float | np.ndarray,
    x_m: float | np.ndarray,
    v_mps: float | np.ndarray,
    C_kg: float,
    F_J_kg: float,
    V0_m3: float,
    A_bore_m2: float,
    rho_p_kg_m3: float,
    eta_m3_kg: float,
    gamma: float,
    m_eff_kg: float,
    E_loss_J: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """IB-04: Lumped-parameter Resal energy equation for mean space pressure p (Pa)."""
    z_c = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    x_a = np.asarray(x_m, dtype=np.float64)
    v_a = np.asarray(v_mps, dtype=np.float64)
    denom = V0_m3 + A_bore_m2 * x_a - (C_kg / rho_p_kg_m3) * (1.0 - z_c) - eta_m3_kg * C_kg * z_c
    denom = np.maximum(denom, 1e-9)
    numer = F_J_kg * C_kg * z_c - (gamma - 1.0) * (0.5 * m_eff_kg * (v_a**2) + E_loss_J)
    p = np.maximum(numer / denom, 101325.0)
    return float(p) if np.ndim(z) == 0 and np.ndim(x_m) == 0 and np.ndim(v_mps) == 0 else p


@register_formula("IB-05")
def ib_05_projectile_eom_lagrange(
    p_mean_Pa: float | np.ndarray,
    A_bore_m2: float,
    m_p_kg: float,
    C_kg: float,
    p_res_Pa: float | np.ndarray = 0.0,
) -> dict[str, float | np.ndarray]:
    """IB-05: Projectile equation of motion with Lagrange pressure gradient."""
    pm = np.asarray(p_mean_Pa, dtype=np.float64)
    p_base = pm / (1.0 + C_kg / (3.0 * m_p_kg))
    p_breech = p_base * (1.0 + C_kg / (2.0 * m_p_kg))
    dv_dt = np.where(p_base > p_res_Pa, A_bore_m2 * (p_base - p_res_Pa) / m_p_kg, 0.0)
    if np.ndim(p_mean_Pa) == 0 and np.ndim(p_res_Pa) == 0:
        return {
            "dv_dt": float(dv_dt),
            "p_base_Pa": float(p_base),
            "p_breech_Pa": float(p_breech),
        }
    return {
        "dv_dt": dv_dt,
        "p_base_Pa": p_base,
        "p_breech_Pa": p_breech,
    }


@register_formula("IB-06")
def ib_06_muzzle_energy_efficiency(
    m_p_kg: float | np.ndarray,
    v_m_mps: float | np.ndarray,
    C_kg: float | np.ndarray,
    F_J_kg: float | np.ndarray,
    gamma: float = 1.25,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """IB-06: Muzzle kinetic energy E_m (J) and thermodynamic ballistic efficiency (-)."""
    em = 0.5 * np.asarray(m_p_kg, dtype=np.float64) * (np.asarray(v_m_mps, dtype=np.float64) ** 2)
    chem_energy = np.asarray(C_kg, dtype=np.float64) * np.asarray(F_J_kg, dtype=np.float64) / (gamma - 1.0)
    eff = em / np.maximum(chem_energy, 1e-9)
    if np.ndim(em) == 0 and np.ndim(eff) == 0:
        return float(em), float(eff)
    return em, eff


@register_formula("IB-07")
def ib_07_propellant_temp_correction(
    mv_std_mps: float | np.ndarray,
    t_prop_C: float | np.ndarray,
    k_T_mps_per_C: float = 0.0,
    t_std_C: float = 21.1,
) -> float | np.ndarray:
    """IB-07: Firing-table propellant temperature correction MV(T) = MV_std + k_T*(T_prop - 21.1 C)."""
    res = np.asarray(mv_std_mps, dtype=np.float64) + k_T_mps_per_C * (
        np.asarray(t_prop_C, dtype=np.float64) - t_std_C
    )
    return float(res) if np.ndim(mv_std_mps) == 0 and np.ndim(t_prop_C) == 0 else res


@register_formula("IB-08")
def ib_08_mv_error_budget(
    s_lot: float = 0.0,
    s_wear: float = 0.0,
    s_temp: float = 0.0,
    s_rr: float = 0.0,
) -> dict[str, float]:
    """IB-08: Muzzle-velocity total, bias (occasion), and random (round-to-round) sigmas."""
    sig_bias = float(np.sqrt(s_lot**2 + s_wear**2 + s_temp**2))
    sig_rr = float(abs(s_rr))
    sig_tot = float(np.sqrt(sig_bias**2 + sig_rr**2))
    return {
        "sigma_total_mps": sig_tot,
        "sigma_bias_mps": sig_bias,
        "sigma_rr_mps": sig_rr,
    }


@dataclass
class LumpedIBConfig:
    """STANAG 4367-type lumped-parameter interior ballistics configuration."""

    calibre_m: float
    travel_m: float
    chamber_volume_m3: float
    proj_mass_kg: float
    charge_mass_kg: float
    impetus_J_kg: float = 1.05e6
    covolume_m3_kg: float = 0.0010
    prop_density_kg_m3: float = 1600.0
    gamma: float = 1.24
    burn_beta: float = 1.2e-8
    burn_alpha: float = 0.85
    half_web_m: float = 0.0008
    kappa: float = 1.0
    lam: float = 0.0
    mu: float = 0.0
    shot_start_Pa: float = 30.0e6
    bore_resistance_Pa: float = 5.0e6
    heat_loss_fraction: float = 0.08


def solve_lumped_interior_ballistics(
    cfg: LumpedIBConfig,
    dt: float = 1e-5,
    max_time_s: float = 0.1,
    record_history: bool = False,
) -> dict[str, Any]:
    """Integrate IB-01..IB-05 ODE system from primer ignition to muzzle exit (x = travel_m)."""
    area = 0.25 * np.pi * (cfg.calibre_m**2)
    m_eff = cfg.proj_mass_kg + cfg.charge_mass_kg / 3.0

    # Start with small igniter burnt fraction so chamber builds pressure cleanly
    phi = 0.02
    x = 0.0
    v = 0.0
    t = 0.0
    started = False
    p_max_mean = 0.0
    p_max_breech = 0.0

    hist_t_ms: list[float] = []
    hist_x_m: list[float] = []
    hist_v_mps: list[float] = []
    hist_p_breech_MPa: list[float] = []
    hist_p_base_MPa: list[float] = []
    hist_p_mean_MPa: list[float] = []
    hist_phi: list[float] = []
    hist_z: list[float] = []
    hist_v_gas_L: list[float] = []

    def rhs(state: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        nonlocal started
        phi_s, x_s, v_s = float(state[0]), float(state[1]), float(state[2])
        z_s, _ = ib_02_form_function(phi_s, 0.0, cfg.half_web_m, cfg.kappa, cfg.lam, cfg.mu)
        e_kin = 0.5 * m_eff * (v_s**2)
        e_res = area * cfg.bore_resistance_Pa * max(x_s, 0.0)
        e_loss = cfg.heat_loss_fraction * e_kin + e_res
        p_mean = float(
            ib_04_resal_energy(
                z=z_s,
                x_m=x_s,
                v_mps=v_s,
                C_kg=cfg.charge_mass_kg,
                F_J_kg=cfg.impetus_J_kg,
                V0_m3=cfg.chamber_volume_m3,
                A_bore_m2=area,
                rho_p_kg_m3=cfg.prop_density_kg_m3,
                eta_m3_kg=cfg.covolume_m3_kg,
                gamma=cfg.gamma,
                m_eff_kg=m_eff,
                E_loss_J=e_loss,
            )
        )
        r_burn = float(ib_01_vieille_burning_law(p_mean, cfg.burn_beta, cfg.burn_alpha))
        _, dphi = ib_02_form_function(phi_s, r_burn, cfg.half_web_m, cfg.kappa, cfg.lam, cfg.mu)
        if started:
            eom = ib_05_projectile_eom_lagrange(
                p_mean, area, cfg.proj_mass_kg, cfg.charge_mass_kg, cfg.bore_resistance_Pa
            )
            p_base = float(eom["p_base_Pa"])
            p_breech = float(eom["p_breech_Pa"])
            dv = float(eom["dv_dt"])
            dx = v_s
        else:
            # Stationary projectile prior to engraving: uniform chamber gas pressure (p_breech = p_base = p_mean)
            p_base = p_mean
            p_breech = p_mean
            if p_base >= cfg.shot_start_Pa:
                eom = ib_05_projectile_eom_lagrange(
                    p_mean, area, cfg.proj_mass_kg, cfg.charge_mass_kg, cfg.bore_resistance_Pa
                )
                dv = float(eom["dv_dt"])
                dx = v_s
            else:
                dv = 0.0
                dx = 0.0
        return np.array([float(dphi), dx, dv], dtype=np.float64), p_mean, p_breech, p_base

    y = np.array([phi, x, v], dtype=np.float64)
    n_steps = int(max_time_s / dt)
    for step_i in range(n_steps):
        k1, pm1, pb1, pbase1 = rhs(y)
        if pb1 > p_max_breech:
            p_max_breech = pb1
            p_max_mean = pm1
        if not started and pbase1 >= cfg.shot_start_Pa:
            started = True
        if record_history and (step_i % 2 == 0 or step_i == 0):
            z_cur, _ = ib_02_form_function(float(y[0]), 0.0, cfg.half_web_m, cfg.kappa, cfg.lam, cfg.mu)
            vg_cur = cfg.chamber_volume_m3 + area * float(y[1]) - (cfg.charge_mass_kg / cfg.prop_density_kg_m3) * (1.0 - float(z_cur))
            hist_t_ms.append(float(t * 1000.0))
            hist_x_m.append(float(y[1]))
            hist_v_mps.append(float(y[2]))
            hist_p_breech_MPa.append(float(pb1 * 1e-6))
            hist_p_base_MPa.append(float(pbase1 * 1e-6))
            hist_p_mean_MPa.append(float(pm1 * 1e-6))
            hist_phi.append(float(y[0]))
            hist_z.append(float(z_cur))
            hist_v_gas_L.append(float(vg_cur * 1000.0))

        k2, _, _, _ = rhs(y + 0.5 * dt * k1)
        k3, _, _, _ = rhs(y + 0.5 * dt * k2)
        k4, _, _, _ = rhs(y + dt * k3)
        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y_next[0] = np.clip(y_next[0], 0.0, 1.0)
        if y_next[1] >= cfg.travel_m:
            frac = (cfg.travel_m - y[1]) / max(y_next[1] - y[1], 1e-12)
            v_muzzle = float(y[2] + frac * (y_next[2] - y[2]))
            t_muzzle = float(t + frac * dt)
            phi_exit = float(y[0] + frac * (y_next[0] - y[0]))
            if record_history:
                _, pm_end, pb_end, pbase_end = rhs(np.array([phi_exit, cfg.travel_m, v_muzzle]))
                z_end, _ = ib_02_form_function(phi_exit, 0.0, cfg.half_web_m, cfg.kappa, cfg.lam, cfg.mu)
                vg_end = cfg.chamber_volume_m3 + area * cfg.travel_m - (cfg.charge_mass_kg / cfg.prop_density_kg_m3) * (1.0 - float(z_end))
                hist_t_ms.append(float(t_muzzle * 1000.0))
                hist_x_m.append(float(cfg.travel_m))
                hist_v_mps.append(float(v_muzzle))
                hist_p_breech_MPa.append(float(pb_end * 1e-6))
                hist_p_base_MPa.append(float(pbase_end * 1e-6))
                hist_p_mean_MPa.append(float(pm_end * 1e-6))
                hist_phi.append(float(phi_exit))
                hist_z.append(float(z_end))
                hist_v_gas_L.append(float(vg_end * 1000.0))
            em, eff = ib_06_muzzle_energy_efficiency(
                cfg.proj_mass_kg, v_muzzle, cfg.charge_mass_kg, cfg.impetus_J_kg, cfg.gamma
            )
            out: dict[str, Any] = {
                "muzzle_velocity_mps": v_muzzle,
                "time_to_muzzle_s": t_muzzle,
                "peak_mean_pressure_Pa": p_max_mean,
                "peak_breech_pressure_Pa": p_max_breech,
                "burnt_fraction_exit": phi_exit,
                "muzzle_energy_J": float(em),
                "ballistic_efficiency": float(eff),
            }
            if record_history:
                out["history"] = {
                    "t_ms": hist_t_ms,
                    "x_m": hist_x_m,
                    "v_mps": hist_v_mps,
                    "p_breech_MPa": hist_p_breech_MPa,
                    "p_base_MPa": hist_p_base_MPa,
                    "p_mean_MPa": hist_p_mean_MPa,
                    "phi": hist_phi,
                    "z_mass_frac": hist_z,
                    "v_gas_L": hist_v_gas_L,
                }
            return out
        y = y_next
        t += dt

    em, eff = ib_06_muzzle_energy_efficiency(
        cfg.proj_mass_kg, float(y[2]), cfg.charge_mass_kg, cfg.impetus_J_kg, cfg.gamma
    )
    out = {
        "muzzle_velocity_mps": float(y[2]),
        "time_to_muzzle_s": float(t),
        "peak_mean_pressure_Pa": p_max_mean,
        "peak_breech_pressure_Pa": p_max_breech,
        "burnt_fraction_exit": float(y[0]),
        "muzzle_energy_J": float(em),
        "ballistic_efficiency": float(eff),
    }
    if record_history:
        out["history"] = {
            "t_ms": hist_t_ms,
            "x_m": hist_x_m,
            "v_mps": hist_v_mps,
            "p_breech_MPa": hist_p_breech_MPa,
            "p_base_MPa": hist_p_base_MPa,
            "p_mean_MPa": hist_p_mean_MPa,
            "phi": hist_phi,
            "z_mass_frac": hist_z,
            "v_gas_L": hist_v_gas_L,
        }
    return out


def le_duc_interior_ballistics(
    charge_mass_kg: float,
    proj_mass_kg: float,
    bore_diameter_m: float,
    barrel_travel_m: float,
    chamber_volume_m3: float,
    propellant_force_J_kg: float = 1.05e6,
    gamma: float = 1.24,
    characteristic_b_m: float | None = None,
) -> dict[str, float]:
    """Closed-form Le Duc hyperbolic interior-ballistics gun scoping model (v = a * x / (b + x)).

    Provides fast first-principles scaling for muzzle velocity, peak base/breech pressure,
    and position of peak pressure from charge-to-mass ratio, chamber volume, and barrel travel.
    """
    m_p = max(float(proj_mass_kg), 1e-4)
    C = max(float(charge_mass_kg), 1e-5)
    d = max(float(bore_diameter_m), 1e-4)
    A = 0.25 * float(np.pi) * (d ** 2)
    X = max(float(barrel_travel_m), 1e-3)
    V0 = max(float(chamber_volume_m3), 1e-8)

    # Effective energetic velocity scale from thermodynamic potential of charge + Lagrange inertia factor
    m_eff = m_p + C / 3.0
    v_inf = float(np.sqrt(2.0 * C * float(propellant_force_J_kg) / ((gamma - 1.0) * m_eff)))
    # Classic Le Duc scale b ~ beta_ch * V0 / A (empirical quick-burning chamber length scale)
    b = float(characteristic_b_m) if characteristic_b_m is not None else max(2.2 * (V0 / A), 0.15 * X)
    v_muzzle = v_inf * X / (b + X)
    x_pmax = 0.5 * b
    # Peak space-mean acceleration occurs at x = b / 2 -> p_base_max = m_p * a_max / A
    p_base_max = (m_p * (v_inf ** 2) * b / A) * (x_pmax / ((b + x_pmax) ** 3))
    p_breech_max = p_base_max * (1.0 + C / (2.0 * m_p))
    return {
        "v_asymptote_mps": float(v_inf),
        "le_duc_b_m": float(b),
        "muzzle_velocity_mps": float(v_muzzle),
        "x_peak_pressure_m": float(x_pmax),
        "peak_base_pressure_Pa": float(p_base_max),
        "peak_breech_pressure_Pa": float(p_breech_max),
    }
