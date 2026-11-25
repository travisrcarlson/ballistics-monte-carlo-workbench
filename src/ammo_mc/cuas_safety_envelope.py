"""C-UAS Acoustic Muzzle-Blast, Friendly Danger-Cone & Collateral Safety Envelope Engine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.cuas_defense import simulate_cuas_pellet_fallout_hazard
from ammo_mc.small_arms import get_choke_definitions, get_shotgun_presets


def _acoustic_radius_for_db(
    target_db: float,
    l0_db_at_1m: float,
    theta_rel_rad: float,
    air_atten_db_per_m: float = 0.016,
) -> float:
    """Solve L(r, theta) = target_db for range r (m) under spherical spreading + directional lobe."""
    dir_gain = 6.5 * math.cos(theta_rel_rad) - 3.0 * (1.0 - math.cos(theta_rel_rad))
    l_eff_1m = l0_db_at_1m + dir_gain
    if l_eff_1m <= target_db:
        return 0.5
    # Bracket and bisection for 20*log10(r) + alpha*r = l_eff_1m - target_db
    rhs = l_eff_1m - target_db
    lo, hi = 0.5, 650.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        loss = 20.0 * math.log10(max(mid, 0.25)) + air_atten_db_per_m * mid
        if loss < rhs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate_cuas_collateral_safety_envelope(
    n_barrels: int = 6,
    barrels_per_burst: int = 2,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    firing_elevation_deg: float = 35.0,
    firing_azimuth_deg: float = 0.0,
    turret_height_m: float = 2.0,
    wind_downrange_mps: float = 0.0,
    wind_crossrange_mps: float = 0.0,
    environment_type: str = "urban_permissive",
    calibration_profile: dict[str, Any] | None = None,
    n_sample_pellets: int = 240,
    seed: int = 42,
    drone_mass_kg: float = 1.5,
    drone_altitude_m: float = 35.0,
    drone_frontal_area_m2: float = 0.06,
) -> dict[str, Any]:
    """Compute directional 12-ga muzzle-blast dB contours, friendly danger cone, & falling-pellet KE tiers."""
    cp = dict(calibration_profile or {})
    presets = get_shotgun_presets()
    ammo = presets.get(ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])
    choke_defs = get_choke_definitions()
    c_info = choke_defs.get(default_choke, choke_defs["Modified"])
    p_bench = float(c_info["benchmark_40yd_30in_pct"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_bench))
    spread_fac = max(0.35, float(cp.get("spread_factor", 1.0) or 1.0))
    mv_scale = max(0.5, min(1.5, float(cp.get("mv_scale", 1.0) or 1.0)))

    # 1. Directional Muzzle-Blast Acoustic Contours (175 dB, 165 dB, 140 dB peak SPL)
    bpb = max(1, min(int(n_barrels), int(barrels_per_burst)))
    b_len = max(14.0, min(32.0, float(default_barrel_length_in)))
    # Shorter barrels & higher MV raise muzzle exit overpressure; multi-barrel simultaneous/tight ripple adds power
    l0_1m = 168.5 + 0.35 * (24.0 - b_len) + 9.0 * math.log10(max(bpb, 1)) + 15.0 * math.log10(max(mv_scale, 0.8))
    if str(environment_type) == "urban_canyon":
        air_atten = 0.010  # Reverberant hard facades slow attenuation
    else:
        air_atten = 0.016

    az0_rad = math.radians(firing_azimuth_deg)
    el0_rad = math.radians(max(0.0, min(85.0, firing_elevation_deg)))
    thetas_rel = np.linspace(0.0, 2.0 * math.pi, 73, endpoint=True)
    db_contours: dict[str, dict[str, Any]] = {}
    for db_level, key_lbl in ((175.0, "db_175"), (165.0, "db_165"), (140.0, "db_140")):
        xs: list[float] = []
        ys: list[float] = []
        rads: list[float] = []
        for th in thetas_rel:
            r_db = _acoustic_radius_for_db(db_level, l0_1m, float(th), air_atten_db_per_m=air_atten)
            world_ang = az0_rad + float(th)
            xs.append(round(r_db * math.cos(world_ang), 2))
            ys.append(round(r_db * math.sin(world_ang), 2))
            rads.append(r_db)
        db_contours[key_lbl] = {
            "threshold_db": db_level,
            "x_m": xs,
            "y_m": ys,
            "max_downrange_m": round(rads[0], 1),
            "rear_radius_m": round(rads[len(rads) // 2], 1),
            "flank_radius_m": round(rads[len(rads) // 4], 1),
        }

    # 2. Friendly-Troop Direct-Fire Danger-Space Cone
    theta_3sig_deg = math.degrees((sigma_ref / 36.58) * spread_fac) * 3.0 + 3.5
    half_cone_deg = max(6.0, min(32.0, theta_3sig_deg))
    # Direct low-angle ground danger projection vs elevated sky shot
    if firing_elevation_deg < 12.0:
        direct_cone_len_m = 260.0 if float(ammo.get("pellet_mass_g", 1.34)) >= 1.5 else 210.0
    else:
        direct_cone_len_m = max(45.0, min(220.0, (float(turret_height_m) + 3.0) / max(math.tan(el0_rad), 0.12) + 85.0))
    c_angs = np.linspace(az0_rad - math.radians(half_cone_deg), az0_rad + math.radians(half_cone_deg), 19)
    cone_x = [0.0] + [round(direct_cone_len_m * math.cos(float(a)), 2) for a in c_angs] + [0.0]
    cone_y = [0.0] + [round(direct_cone_len_m * math.sin(float(a)), 2) for a in c_angs] + [0.0]

    # 3. Falling-Pellet Kinetic Energy Collateral Risk Footprint
    eff_bbl_len_for_mv = b_len + (mv_scale - 1.0) * float(ammo["v0_mps"]) / 2.2
    fallout = simulate_cuas_pellet_fallout_hazard(
        n_barrels=int(n_barrels),
        ammunition_preset=ammunition_preset,
        default_choke=default_choke,
        default_barrel_length_in=max(14.0, min(34.0, eff_bbl_len_for_mv)),
        firing_elevation_deg=firing_elevation_deg,
        firing_azimuth_deg=firing_azimuth_deg,
        turret_height_m=turret_height_m,
        wind_downrange_mps=wind_downrange_mps,
        wind_crossrange_mps=wind_crossrange_mps,
        n_sample_pellets=int(n_sample_pellets),
        seed=seed,
    )
    gi = fallout["ground_impacts"]
    e_arr = np.asarray(gi["energy_j"], dtype=np.float64)
    r_arr = np.asarray(gi["range_m"], dtype=np.float64)
    x_arr = np.asarray(gi["x_m"], dtype=np.float64)
    y_arr = np.asarray(gi["y_m"], dtype=np.float64)

    tiers: list[str] = []
    n_lethal = 0
    n_injury = 0
    n_glass = 0
    n_safe = 0
    for e_val in e_arr:
        if e_val >= 15.0:
            tiers.append("Lethal (>15 J)")
            n_lethal += 1
        elif e_val >= 2.0:
            tiers.append("Injury (2-15 J)")
            n_injury += 1
        elif e_val >= 0.5:
            tiers.append("Property/Glass (0.5-2 J)")
            n_glass += 1
        else:
            tiers.append("Safe (<0.5 J)")
            n_safe += 1

    n_tot = max(1, len(e_arr))
    lethal_pct = round(100.0 * n_lethal / n_tot, 1)
    injury_pct = round(100.0 * (n_lethal + n_injury) / n_tot, 1)

    # Dual-Criterion Energy + Sectional Energy Density (RCC 321-23 & Kneubuehl 2022)
    d_cm = max(0.05, float(ammo.get("pellet_diameter_mm", 6.10)) / 10.0)
    area_cm2 = math.pi * 0.25 * (d_cm ** 2)
    sed_arr = e_arr / max(area_cm2, 1e-6)
    rcc_blunt_cas_pct = round(float(np.mean(e_arr >= 14.91)) * 100.0, 1)
    rcc_prone_fat_pct = round(float(np.mean(e_arr >= 18.98)) * 100.0, 1)
    rcc_stand_fat_pct = round(float(np.mean(e_arr >= 33.89)) * 100.0, 1)
    rcc_pen_cas_pct = round(float(np.mean(sed_arr >= 7.145)) * 100.0, 1)
    kneubuehl_skin_pct = round(float(np.mean(sed_arr >= 10.0)) * 100.0, 1)
    kneubuehl_eye_pct = round(float(np.mean(sed_arr >= 3.0)) * 100.0, 1)

    # Falling Drone Debris Impact Energy (SE-F04) vs FAA Cat 2/3, ASTM, EASA
    m_dr = max(0.1, float(drone_mass_kg))
    h_dr = max(1.0, float(drone_altitude_m))
    a_dr = max(0.005, float(drone_frontal_area_m2))
    rho_sl = 1.225
    cd_dr = 0.30
    k_dr = (cd_dr * rho_sl * a_dr) / (2.0 * m_dr)
    v_term_mps = math.sqrt((2.0 * m_dr * 9.80665) / max(cd_dr * rho_sl * a_dr, 1e-6))
    v_imp_dr_mps = v_term_mps * math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * k_dr * h_dr)))
    e_imp_dr_j = 0.5 * m_dr * (v_imp_dr_mps ** 2)
    debris_tier = (
        "Exceeds EASA (>=80 J)" if e_imp_dr_j >= 80.0
        else ("Exceeds ASTM (>=73 J)" if e_imp_dr_j >= 73.0
              else ("Exceeds FAA Cat 3 (>=34 J)" if e_imp_dr_j >= 34.0
                    else ("Exceeds FAA Cat 2 (>=15 J)" if e_imp_dr_j >= 15.0 else "Below 15 J")))
    )

    shot_num_est = float(ammo.get("pellet_diameter_mm", 6.10))
    ar_dist_x_m = 1073.0 if shot_num_est > 2.45 else 275.0  # AR 385-63 Table 5-1: >7.5 shot -> 1073 m, 7.5-9 -> 275 m
    doctrine_reference_overlays = {
        "ar_385_63_table_5_1": {
            "dist_x_m": ar_dist_x_m,
            "class_note": "Shot > #7.5 / Buckshot / Slug (Dist X = 1,073 m)" if ar_dist_x_m > 500.0 else "Shot #7.5-#9 Trap/Skeet (Dist X = 275 m)",
        },
        "uk_jsp_403_vol2_ch2_table3": {
            "lateral_cone_mils": 80.0,
            "vertical_cone_mils": 200.0,
            "note": "UK JSP 403 Vol 2 Ch 2 Table 3 Cone of Fire: +80 mils lateral (azimuth), +200 mils vertical (elevation).",
        },
    }

    max_rng_99 = float(fallout["summary"]["max_fallout_range_99_m"])
    mean_imp_j = float(fallout["summary"]["mean_impact_energy_j"])
    r_140 = float(db_contours["db_140"]["max_downrange_m"])
    r_165 = float(db_contours["db_165"]["max_downrange_m"])
    r_rear_safe = round(max(12.0, float(db_contours["db_140"]["rear_radius_m"])), 1)

    if lethal_pct <= 1.0 and injury_pct <= 15.0 and firing_elevation_deg >= 25.0:
        verdict = "CLEARED_WITH_STANDOFF"
        verdict_note = (
            f"Safe for perimeter C-UAS above {firing_elevation_deg:.0f} deg el: falling lethal pellet fraction is {lethal_pct:.1f}% "
            f"(mean {mean_imp_j:.2f} J); maintain >={r_rear_safe:.0f} m unprotected crew rear standoff and {max_rng_99:.0f} m downrange buffer."
        )
    elif firing_elevation_deg < 20.0 or lethal_pct > 15.0:
        verdict = "RESTRICTED_ELEVATION_OR_AMMO"
        verdict_note = (
            f"HIGH COLLATERAL RISK: {lethal_pct:.1f}% lethal fallout (>15 J) / low {firing_elevation_deg:.0f} deg fire angle. "
            f"Raise minimum fire-inhibit elevation >= 28 deg or restrict urban overflight sectors to lighter shot (#4 Buck / Steel)."
        )
    else:
        verdict = "CLEARED_PERIMETER_CONTROLLED"
        verdict_note = (
            f"Controlled collateral envelope: {injury_pct:.1f}% injury-tier (>=2 J) pellets landing around {fallout['summary']['mean_fallout_range_m']:.0f} m; "
            f"clear unprotected personnel within {r_140:.0f} m downrange / {r_165:.0f} m single-hearing-protection zone."
        )

    return {
        "summary": {
            "verdict": verdict,
            "verdict_note": verdict_note,
            "environment_type": environment_type,
            "firing_elevation_deg": round(float(firing_elevation_deg), 1),
            "firing_azimuth_deg": round(float(firing_azimuth_deg), 1),
            "muzzle_peak_1m_db": round(l0_1m, 1),
            "acoustic_140db_downrange_m": db_contours["db_140"]["max_downrange_m"],
            "acoustic_165db_downrange_m": db_contours["db_165"]["max_downrange_m"],
            "acoustic_175db_downrange_m": db_contours["db_175"]["max_downrange_m"],
            "crew_rear_140db_standoff_m": r_rear_safe,
            "direct_fire_cone_range_m": round(direct_cone_len_m, 1),
            "direct_fire_half_angle_deg": round(half_cone_deg, 1),
            "fallout_99pct_range_m": round(max_rng_99, 1),
            "mean_impact_energy_j": round(mean_imp_j, 2),
            "lethal_15j_pct": lethal_pct,
            "injury_2j_pct": injury_pct,
            "glass_risk_pct": round(100.0 * n_glass / n_tot, 1),
            "safe_low_energy_pct": round(100.0 * n_safe / n_tot, 1),
            "recommended_min_elevation_deg": 28.0 if lethal_pct > 2.0 or firing_elevation_deg < 20.0 else 18.0,
        },
        "acoustic_contours": db_contours,
        "direct_fire_cone": {"x_m": cone_x, "y_m": cone_y},
        "fallout_impacts": {
            "x_m": [round(float(v), 2) for v in x_arr],
            "y_m": [round(float(v), 2) for v in y_arr],
            "range_m": [round(float(v), 2) for v in r_arr],
            "energy_j": [round(float(v), 2) for v in e_arr],
            "risk_tier": tiers,
        },
        "tier_breakdown": [
            {"tier": "Lethal (>15 J)", "count": n_lethal, "share_pct": lethal_pct},
            {"tier": "Injury (2-15 J)", "count": n_injury, "share_pct": round(100.0 * n_injury / n_tot, 1)},
            {"tier": "Property/Glass (0.5-2 J)", "count": n_glass, "share_pct": round(100.0 * n_glass / n_tot, 1)},
            {"tier": "Safe (<0.5 J)", "count": n_safe, "share_pct": round(100.0 * n_safe / n_tot, 1)},
        ],
        "dual_criterion_hazards": {
            "pellet_area_cm2": round(area_cm2, 4),
            "mean_specific_energy_j_cm2": round(float(np.mean(sed_arr)), 2),
            "rcc_321_23_blunt_casualty_14_91j_pct": rcc_blunt_cas_pct,
            "rcc_321_23_fatality_prone_18_98j_pct": rcc_prone_fat_pct,
            "rcc_321_23_fatality_standing_33_89j_pct": rcc_stand_fat_pct,
            "rcc_321_23_penetration_casualty_7_145j_cm2_pct": rcc_pen_cas_pct,
            "kneubuehl_skin_10j_cm2_pct": kneubuehl_skin_pct,
            "kneubuehl_eye_3j_cm2_pct": kneubuehl_eye_pct,
        },
        "falling_drone_debris": {
            "drone_mass_kg": m_dr,
            "drone_altitude_m": h_dr,
            "terminal_velocity_mps": round(v_term_mps, 2),
            "impact_velocity_mps": round(v_imp_dr_mps, 2),
            "impact_energy_j": round(e_imp_dr_j, 1),
            "threshold_status": debris_tier,
            "uxo_post_kill_notice": (
                "SE-C20 UXO caution: downed strike drones retain unexploded payload and thermal-runaway "
                "LiPo/Li-Ion risk downrange until EOD clearance."
            ),
        },
        "doctrine_reference_overlays": doctrine_reference_overlays,
        "scope_notice": (
            "Engineering analysis aid only; does not replace official Range Safety Officer approvals "
            "or governing standards (AR 385-63, UK JSP 403, RCC 321-23)."
        ),
    }
