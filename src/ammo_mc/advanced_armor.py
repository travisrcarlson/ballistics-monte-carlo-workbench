"""Modern Multi-Layer Armor Defeat, Tandem Shaped Charges, and ERA Mechanics.

Grounded in:
- Birkhoff / Allison-Vitali hydrodynamic shaped-charge jet penetration (TP-02 extension)
- Gurney flyer plate explosive dynamics for Explosive Reactive Armor (ERA)
- Tate-Alekseevskii hydrodynamic penetration model for eroded penetrators & ceramic dwell
- Mott-Grady behind-armor spall fragment cone and crew compartment vulnerability

Features:
- Tandem vs Monolithic shaped charge warheads (Precursor + Main charge with ms delay)
- Heavy ERA (Kontakt-5 / Relikt) and Light ERA flyer plate velocity, angle, and jet disruption
- Composite ceramic armor (Silicon Carbide SiC, Alumina Al2O3) with shock dwell phase
- High-hardness Rolled Homogeneous Armor (RHA) steel backing plate
- Behind-armor spall cone: fragment count, spall velocity, cone half-angle, and crew incapacitation Pk
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np


def calculate_era_flyer_velocity(
    c_exp_kg_m2: float = 4.5,
    m_plate_kg_m2: float = 18.0,
    gurney_vel_mps: float = 2400.0,
    configuration: Literal["asymmetric_tamped", "symmetric"] = "asymmetric_tamped",
) -> float:
    """Calculates ERA flyer plate velocity via exact Gurney sandwich equations (TF-02 / Cooper 1996).

    - ``asymmetric_tamped`` (infinitely heavy armor backing): V_flyer = sqrt(2E) * (M/C + 1/3)^(-1/2)
    - ``symmetric`` (dual equal plates): V_flyer = sqrt(2E) * (2*M/C + 1/3)^(-1/2)
    """
    ratio = max(m_plate_kg_m2 / max(c_exp_kg_m2, 1e-4), 0.05)
    if configuration == "symmetric":
        factor = math.sqrt(1.0 / (2.0 * ratio + (1.0 / 3.0)))
    else:
        factor = math.sqrt(1.0 / (ratio + (1.0 / 3.0)))
    return float(gurney_vel_mps * factor)


def simulate_tandem_shaped_charge_vs_era(
    warhead_type: Literal["monolithic", "tandem"] = "tandem",
    has_era: bool = True,
    era_type: Literal["light_era", "heavy_era"] = "heavy_era",
    main_calibre_mm: float = 135.0,
    precursor_calibre_mm: float = 55.0,
    inter_warhead_delay_ms: float = 0.35,
    obliquity_deg: float = 60.0,
    rha_base_thickness_mm: float = 450.0,
    has_ceramic_tile: bool = True,
    ceramic_thickness_mm: float = 60.0,
    ceramic_type: Literal["Silicon Carbide (SiC)", "Alumina (Al2O3)"] = "Silicon Carbide (SiC)",
) -> dict[str, Any]:
    """Simulates tandem or monolithic shaped-charge jet defeat against ERA and composite armor.

    Returns:
        Full penetration audit: precursor status, ERA flyer plate activation, main jet residual length,
        ceramic dwell erosion, total RHA penetration depth, perforation outcome, and behind-armor spall metrics.
    """
    ob_rad = math.radians(obliquity_deg)
    cos_ob = max(math.cos(ob_rad), 0.15)
    sec_ob = 1.0 / cos_ob

    # Base RHA line-of-sight (LOS) thickness
    t_rha_los = rha_base_thickness_mm * sec_ob
    t_cer_los = (ceramic_thickness_mm * sec_ob) if has_ceramic_tile else 0.0

    # 1. Main shaped-charge theoretical baseline hydrodynamic penetration (Birkhoff model)
    # Typical shaped charges penetrate 6.5 to 7.5 CD (charge diameters) in monolithic RHA
    cd_factor = 7.0
    p_main_uninhibited_mm = main_calibre_mm * cd_factor

    era_activated = False
    era_flyer_speed = 0.0
    jet_loss_pct = 0.0
    precursor_success = False

    if has_era:
        # Flyer plate parameters & Gurney speed
        c_exp = 3.5 if era_type == "light_era" else 6.0
        m_plate = 12.0 if era_type == "light_era" else 22.0
        era_flyer_speed = calculate_era_flyer_velocity(c_exp, m_plate, gurney_vel_mps=2450.0, configuration="asymmetric_tamped")

        # Held (1970/1993) flyer-jet cut scaling driven by era_flyer_speed, obliquity & plate areal mass
        sin_ob = abs(math.sin(ob_rad))
        v_flyer_transverse = era_flyer_speed * max(sin_ob, 0.25)
        held_cut_factor = min(0.82, 0.36 + 0.00038 * v_flyer_transverse * (1.0 + 0.015 * m_plate))

        if warhead_type == "tandem":
            # Precursor detonates ERA first
            precursor_success = True
            era_activated = True
            # Longer inter-warhead delay lets flyer plate travel clear before main jet arrival
            if inter_warhead_delay_ms >= 0.25:
                jet_loss_pct = min(14.0, 0.11 * held_cut_factor * 100.0)
            else:
                jet_loss_pct = min(45.0, 0.45 * held_cut_factor * 100.0)
        else:
            # Monolithic warhead: main jet impacts moving heavy flyer plates
            era_activated = True
            precursor_success = False
            era_gain = 1.0 if era_type == "heavy_era" else 0.65
            jet_loss_pct = float(np.clip(held_cut_factor * era_gain * 100.0, 25.0, 82.0))

    p_main_post_era_mm = p_main_uninhibited_mm * (1.0 - jet_loss_pct / 100.0)

    # 2. Ceramic dwell and erosion using true thickness efficiency E_t = E_m * (rho_cer / rho_RHA)
    ceramic_dwell_loss_mm = 0.0
    if has_ceramic_tile:
        # SiC: rho=3.15 g/cm^3, E_m ~ 2.4 -> E_t ~ 0.96; Al2O3: rho=3.90 g/cm^3, E_m ~ 1.7 -> E_t ~ 0.845
        if "SiC" in ceramic_type:
            em = 2.4
            rho_cer = 3.15
        else:
            em = 1.7
            rho_cer = 3.90
        et_ceramic = em * (rho_cer / 7.85)
        rha_equiv_ceramic = t_cer_los * et_ceramic
        ceramic_dwell_loss_mm = rha_equiv_ceramic
        p_post_ceramic_mm = max(0.0, p_main_post_era_mm - ceramic_dwell_loss_mm)
    else:
        p_post_ceramic_mm = p_main_post_era_mm

    # 3. Base RHA penetration
    p_final_in_rha_mm = min(p_post_ceramic_mm, t_rha_los)
    perforated = p_post_ceramic_mm > t_rha_los
    residual_reserve_mm = max(0.0, p_post_ceramic_mm - t_rha_los)

    # 4. Behind-Armor Spall Cone & Crew Vulnerability
    if perforated:
        # Excess kinetic/hydrodynamic energy creates spall spray
        excess_ratio = min(residual_reserve_mm / max(t_rha_los, 1.0), 1.0)
        n_spall = int(45 + 180 * excess_ratio)
        spall_vel_mps = float(450.0 + 800.0 * math.sqrt(excess_ratio))
        spall_half_angle_deg = float(24.0 + 12.0 * excess_ratio)
        # Target internal incapacitation probability (Mott-Carleton behind armor model)
        p_crew_incapacitation = float(np.clip(1.0 - math.exp(-0.022 * n_spall), 0.15, 0.99))
    else:
        n_spall = 0
        spall_vel_mps = 0.0
        spall_half_angle_deg = 0.0
        p_crew_incapacitation = 0.0

    return {
        "warhead_type": warhead_type,
        "has_era": has_era,
        "era_type": era_type,
        "precursor_success": precursor_success,
        "era_activated": era_activated,
        "era_flyer_speed_mps": round(era_flyer_speed, 1),
        "jet_disruption_pct": round(jet_loss_pct, 1),
        "uninhibited_jet_penetration_mm": round(p_main_uninhibited_mm, 1),
        "post_era_jet_penetration_mm": round(p_main_post_era_mm, 1),
        "ceramic_rha_equivalent_mm": round(ceramic_dwell_loss_mm, 1),
        "target_los_thickness_mm": round(t_rha_los + t_cer_los, 1),
        "base_rha_los_thickness_mm": round(t_rha_los, 1),
        "residual_penetration_reserve_mm": round(residual_reserve_mm, 1),
        "perforated": perforated,
        "perforation_status": "🔴 COMPLETE PERFORATION" if perforated else "🛡️ ARMOR DEFEATED PROJECTILE (STOPPED)",
        "behind_armor_spall": {
            "perforated": perforated,
            "spall_fragment_count": n_spall,
            "mean_spall_velocity_mps": round(spall_vel_mps, 1),
            "spall_cone_half_angle_deg": round(spall_half_angle_deg, 1),
            "crew_incapacitation_prob": round(p_crew_incapacitation, 3),
        },
    }


def tate_alekseevskii_penetration(
    v_impact_mps: float,
    rod_length_m: float,
    rho_p_kg_m3: float = 17600.0,
    rho_t_kg_m3: float = 7850.0,
    y_p_gpa: float = 1.5,
    r_t_gpa: float = 5.5,
    dt_s: float = 2e-6,
    max_time_s: float = 2e-3,
) -> dict[str, Any]:
    """Integrate the Tate (1967) / Alekseevskii 1D hydrodynamic-strength rod erosion ODE.

    Governing modified Bernoulli balance at the penetrator-target interface:
      0.5 * rho_p * (v - u)^2 + Y_p = 0.5 * rho_t * u^2 + R_t
    with:
      dv/dt = -Y_p / (rho_p * L),   dL/dt = -(v - u),   dP/dt = u.
    Includes rigid-body phase (u = v) when R_t >= Y_p and interface stress threshold stops erosion.
    """
    rho_p = max(float(rho_p_kg_m3), 500.0)
    rho_t = max(float(rho_t_kg_m3), 500.0)
    y_p = max(float(y_p_gpa), 0.05) * 1.0e9
    r_t = max(float(r_t_gpa), 0.1) * 1.0e9

    v = max(float(v_impact_mps), 0.0)
    L = max(float(rod_length_m), 1e-4)
    P = 0.0
    t = 0.0
    mu_sq = rho_t / rho_p
    diff_str = 2.0 * (r_t - y_p) / rho_p
    v_crit_hydro = math.sqrt(max(diff_str, 0.0)) if r_t >= y_p else 0.0
    v_rigid_stop = math.sqrt(2.0 * r_t / rho_p)

    hist_t_us: list[float] = [0.0]
    hist_P_mm: list[float] = [0.0]
    hist_L_mm: list[float] = [L * 1000.0]
    hist_v_mps: list[float] = [v]
    hist_u_mps: list[float] = []

    def step_rates(v_cur: float, l_cur: float) -> tuple[float, float, float, float]:
        if v_cur <= 1e-3 or l_cur <= 1e-5:
            return 0.0, 0.0, 0.0, 0.0
        if r_t >= y_p and v_cur <= v_crit_hydro:
            # Below hydrodynamic interface stagnation threshold (0.5*rho_p*v^2 + Y_p <= R_t): erosion/penetration stops
            return 0.0, 0.0, 0.0, 0.0
        if r_t < y_p and (0.5 * rho_t * (v_cur ** 2) + r_t) <= y_p:
            # Rigid-body penetration phase (penetrator stronger than target dynamic + static resistance)
            u_val = v_cur
            dv_dt = -(r_t + 0.5 * rho_t * (v_cur ** 2)) / (rho_p * l_cur)
            return dv_dt, 0.0, u_val, u_val
        if abs(1.0 - mu_sq) < 1e-6:
            u_val = max(0.0, (v_cur ** 2 - diff_str) / max(2.0 * v_cur, 1e-6))
        else:
            rad = max(mu_sq * (v_cur ** 2) + (1.0 - mu_sq) * diff_str, 0.0)
            u_val = max(0.0, (v_cur - math.sqrt(rad)) / (1.0 - mu_sq))
        if u_val > v_cur:
            u_val = v_cur
        dv_dt = -y_p / (rho_p * l_cur)
        dl_dt = -(v_cur - u_val)
        dp_dt = u_val
        return dv_dt, dl_dt, dp_dt, u_val

    n_max = int(max_time_s / dt_s)
    for k in range(n_max):
        dv1, dl1, dp1, u1 = step_rates(v, L)
        if k == 0:
            hist_u_mps.append(u1)
        if u1 <= 1e-3 or L <= 1e-4 or v <= 1e-3:
            break
        dv2, dl2, dp2, _ = step_rates(v + 0.5 * dt_s * dv1, max(L + 0.5 * dt_s * dl1, 1e-5))
        dv3, dl3, dp3, _ = step_rates(v + 0.5 * dt_s * dv2, max(L + 0.5 * dt_s * dl2, 1e-5))
        dv4, dl4, dp4, _ = step_rates(v + dt_s * dv3, max(L + dt_s * dl3, 1e-5))
        v = max(0.0, v + (dt_s / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4))
        L = max(0.0, L + (dt_s / 6.0) * (dl1 + 2.0 * dl2 + 2.0 * dl3 + dl4))
        P += (dt_s / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        t += dt_s
        if k % 5 == 0:
            hist_t_us.append(t * 1e6)
            hist_P_mm.append(P * 1000.0)
            hist_L_mm.append(L * 1000.0)
            hist_v_mps.append(v)
            hist_u_mps.append(u1)

    return {
        "penetration_m": float(P),
        "penetration_mm": float(P * 1000.0),
        "penetration_over_length": float(P / max(float(rod_length_m), 1e-6)),
        "residual_rod_length_m": float(L),
        "residual_rod_length_mm": float(L * 1000.0),
        "residual_velocity_mps": float(v),
        "hydrodynamic_limit_p_over_l": float(math.sqrt(rho_p / rho_t)),
        "critical_velocity_mps": float(v_crit_hydro),
        "rigid_stop_velocity_mps": float(v_rigid_stop),
        "duration_us": float(t * 1e6),
        "history": {
            "t_us": hist_t_us,
            "penetration_mm": hist_P_mm,
            "rod_length_mm": hist_L_mm,
            "tail_velocity_mps": hist_v_mps,
            "interface_velocity_mps": hist_u_mps,
        },
    }
