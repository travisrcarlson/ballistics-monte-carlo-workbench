"""Direct-Fire Tank Gunnery, APFSDS Sabot Discard Dynamics & Moving Target FCS.

Implements:
1. Armor-Piercing Fin-Stabilized Discarding Sabot (APFSDS) ultra-high-speed flat ballistics (1,500 - 1,800 m/s).
2. Sabot petal aerodynamic discard separation, divergence scoops, and safety hazard footprint.
3. Tank Fire Control System (FCS) lead angle computation for maneuvering armored targets.
4. Gun trunnion cant angle and crosswind jump corrections.
5. High-velocity kinetic long-rod penetration using Lanz-Odermatt terminal mechanics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def simulate_apfsds_sabot_discard(
    v0_mps: float = 1680.0,
    rod_calibre_m: float = 0.026,
    rod_length_m: float = 0.78,
    rod_mass_kg: float = 4.85,
    sabot_mass_kg: float = 3.15,
    n_petals: int = 3,
    petal_divergence_angle_deg: float = 8.5,
    asymmetry_factor: float = 0.04,  # Differential release timing across petals
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates supersonic 3-piece or 4-piece sabot petal separation upon muzzle exit.

    Computes:
    - Petal radial divergence trajectories under supersonic air scoop pressure.
    - Asymmetric petal discard impulse inducing rod tip-off rate (deg/s).
    - Sabot ground impact hazard footprint (danger zone for friendly forward troops).
    """
    rng = np.random.default_rng(seed)
    petal_mass = sabot_mass_kg / max(n_petals, 1)

    # Petal discard begins 0.005s post muzzle and completes within ~0.035s (10 - 50m downrange)
    petal_discard_distance_m = round(v0_mps * 0.028, 1)

    # Asymmetric petal release impulse sets initial rod tip-off yaw rate
    tip_off_rate_deg_s = float(np.clip(rng.normal(1.2, 0.4) * (1.0 + asymmetry_factor * 10.0), 0.3, 4.5))

    # Generate 3D paths for individual sabot petals peeling away
    petal_paths: list[list[list[float]]] = []
    petal_hazard_impacts: list[dict[str, float]] = []

    for p_idx in range(n_petals):
        angle_rad = 2.0 * math.pi * (p_idx / n_petals)
        path = [[0.0, 0.0, 1.8]]  # Gun trunnion height ~1.8m

        px, py, pz = 0.0, 0.0, 1.8
        vx = v0_mps
        # Radial separation velocity from scoop lift
        v_rad = v0_mps * math.tan(math.radians(petal_divergence_angle_deg))
        vy = v_rad * math.cos(angle_rad)
        vz = v_rad * math.sin(angle_rad)

        dt = 0.02
        t = 0.0
        # High drag bluff body decelerates rapidly
        while pz > 0.0 and t < 4.0:
            v_mag = math.hypot(vx, vy, vz)
            # Petal drag coefficient Cd ~ 1.2, area ~ 0.015 m^2
            drag_f = 0.5 * 1.225 * 0.015 * 1.15 * (v_mag**2) / petal_mass
            ax = -drag_f * (vx / v_mag)
            ay = -drag_f * (vy / v_mag)
            az = -drag_f * (vz / v_mag) - 9.80665

            px += vx * dt
            py += vy * dt
            pz += vz * dt

            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            t += dt

            path.append([round(px, 1), round(py, 1), round(max(0.0, pz), 1)])

        petal_paths.append(path)
        petal_hazard_impacts.append({
            "petal_index": p_idx + 1,
            "impact_downrange_m": round(px, 1),
            "impact_crossrange_m": round(py, 1),
            "flight_time_s": round(t, 2),
        })

    max_hazard_dist = float(max(p["impact_downrange_m"] for p in petal_hazard_impacts))
    max_hazard_width = float(max(abs(p["impact_crossrange_m"]) for p in petal_hazard_impacts)) * 2.0

    return {
        "v0_mps": v0_mps,
        "n_petals": n_petals,
        "petal_discard_completed_distance_m": petal_discard_distance_m,
        "rod_tip_off_yaw_rate_deg_s": round(tip_off_rate_deg_s, 2),
        "sabot_hazard_zone": {
            "danger_length_m": round(max_hazard_dist, 0),
            "danger_width_m": round(max_hazard_width, 0),
        },
        "petal_paths": petal_paths,
        "petal_impacts": petal_hazard_impacts,
    }


def calculate_moving_target_lead(
    target_range_m: float = 2400.0,
    target_speed_kmh: float = 36.0,
    target_crossing_angle_deg: float = 90.0,  # 90 deg = pure cross target
    v0_mps: float = 1680.0,
    gun_cant_deg: float = 2.5,
    crosswind_mps: float = 6.0,
) -> dict[str, Any]:
    r"""Computes tank ballistic Fire Control System (FCS) lead solution for moving armor.

    Calculates:
    - Time of Flight (TOF) over supersonic flat trajectory.
    - Dynamic lead angle \theta_{\text{lead}} = \arctan(v_{\text{cross}} \cdot \text{TOF} / R).
    - Trunnion cant angle cross-coupling drift error.
    - Aerodynamic crosswind jump correction.
    """
    # Average projectile velocity across flight (supersonic rod loses ~75 m/s per km)
    v_avg = v0_mps - 38.0 * (target_range_m / 1000.0)
    tof_s = target_range_m / max(v_avg, 500.0)

    # Target crossing velocity component
    tgt_speed_mps = target_speed_kmh / 3.6
    cross_speed_mps = tgt_speed_mps * math.sin(math.radians(target_crossing_angle_deg))
    range_speed_mps = tgt_speed_mps * math.cos(math.radians(target_crossing_angle_deg))

    # Lead angle
    lead_dist_m = cross_speed_mps * tof_s
    lead_angle_mils = (lead_dist_m / target_range_m) * 1000.0

    # Trunnion cant cross-coupling: theta_elevation couples into azimuth
    # Approximate supersonic drop: drop = 0.5 * g * tof^2
    vert_drop_m = 0.5 * 9.80665 * (tof_s**2)
    cant_rad = math.radians(gun_cant_deg)
    cant_drift_m = vert_drop_m * math.sin(cant_rad)

    # Crosswind jump: drag drift
    crosswind_drift_m = 0.5 * crosswind_mps * (tof_s - target_range_m / v0_mps) * 12.0

    net_crossrange_correction_m = lead_dist_m + cant_drift_m + crosswind_drift_m
    fcs_azimuth_aim_mils = (net_crossrange_correction_m / target_range_m) * 1000.0

    return {
        "target_range_m": float(target_range_m),
        "target_speed_mps": round(tgt_speed_mps, 1),
        "crossing_velocity_mps": round(cross_speed_mps, 1),
        "tof_s": round(tof_s, 3),
        "vertical_drop_m": round(vert_drop_m, 2),
        "kinematic_lead_distance_m": round(lead_dist_m, 2),
        "cant_error_drift_m": round(cant_drift_m, 2),
        "crosswind_drift_m": round(crosswind_drift_m, 2),
        "total_aim_lead_mils": round(fcs_azimuth_aim_mils, 2),
        "total_crossrange_lead_m": round(net_crossrange_correction_m, 2),
    }


def simulate_direct_fire_engagement(
    target_range_m: float = 2200.0,
    target_speed_kmh: float = 32.0,
    target_crossing_angle_deg: float = 75.0,
    v0_mps: float = 1680.0,
    rod_calibre_m: float = 0.026,
    rod_length_m: float = 0.78,
    rod_mass_kg: float = 4.85,
    armor_thickness_rha_mm: float = 650.0,
    armor_obliquity_deg: float = 60.0,
    fcs_lead_applied: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates a complete direct-fire kinetic engagement against a moving main battle tank.

    Computes:
    - Moving target lead calculation.
    - Flat 3D trajectory integration.
    - Impact point coordinates relative to target center of mass.
    - Striking velocity and terminal Lanz-Odermatt penetration depth against armor.
    """
    rng = np.random.default_rng(seed)
    fcs_res = calculate_moving_target_lead(
        target_range_m=target_range_m,
        target_speed_kmh=target_speed_kmh,
        target_crossing_angle_deg=target_crossing_angle_deg,
        v0_mps=v0_mps,
    )

    tof = fcs_res["tof_s"]
    v_impact = v0_mps - 75.0 * (target_range_m / 1000.0)

    # Lead aiming error
    if fcs_lead_applied:
        # FCS laser rangefinder and auto-lead tracker error ~0.15 mils
        aim_err_x = float(rng.normal(0.0, 0.15 * target_range_m * 1e-3))
        aim_err_z = float(rng.normal(0.0, 0.12 * target_range_m * 1e-3))
    else:
        # Fired at instantaneous target location without lead -> major miss
        aim_err_x = -fcs_res["kinematic_lead_distance_m"]
        aim_err_z = -fcs_res["vertical_drop_m"]

    # Target box dimensions: MBT front turret/hull ~ 3.4m wide x 2.2m high
    mbt_half_width = 1.7
    mbt_half_height = 1.1

    is_hit = (abs(aim_err_x) <= mbt_half_width) and (abs(aim_err_z) <= mbt_half_height)

    # Lanz-Odermatt penetration depth calculation (TP-01)
    from ammo_mc.penetration import tp_01_lanz_odermatt
    # Tungsten alloy density 17,600 kg/m^3 vs steel RHA 7,850 kg/m^3
    lo_res = tp_01_lanz_odermatt(
        Lw_m=rod_length_m,
        D_m=rod_calibre_m,
        V_mps=v_impact,
        rho_p_kg_m3=17600.0,
        rho_t_kg_m3=7850.0,
        bhn_t=300.0,
        obliquity_rad=math.radians(armor_obliquity_deg),
    )
    lo_pen_normal_mm = float(lo_res["penetration"]) * 1000.0

    sec_ob = 1.0 / max(math.cos(math.radians(armor_obliquity_deg)), 0.2)
    effective_los_armor_mm = armor_thickness_rha_mm * sec_ob
    perforated = is_hit and (lo_pen_normal_mm >= effective_los_armor_mm)

    # Generate 3D trajectory points
    n_pts = 25
    traj_x = [float(i * target_range_m / (n_pts - 1)) for i in range(n_pts)]
    traj_y = [float(aim_err_x * (i / (n_pts - 1)) ** 1.2) for i in range(n_pts)]
    # Flat trajectory with slight parabolic drop
    traj_z = [float(1.8 - fcs_res["vertical_drop_m"] * ((i / (n_pts - 1)) ** 2) + aim_err_z * (i / (n_pts - 1))) for i in range(n_pts)]

    return {
        "target_range_m": float(target_range_m),
        "v0_mps": float(v0_mps),
        "v_impact_mps": round(v_impact, 1),
        "tof_s": round(tof, 3),
        "fcs_lead_applied": fcs_lead_applied,
        "impact_offset_x_m": round(aim_err_x, 2),
        "impact_offset_z_m": round(aim_err_z, 2),
        "is_hit": is_hit,
        "lo_penetration_normal_mm": round(lo_pen_normal_mm, 1),
        "target_los_armor_mm": round(effective_los_armor_mm, 1),
        "perforated": perforated,
        "engagement_status": (
            "🎯 DIRECT HIT & COMPLETE ARMOR PERFORATION (KILL)"
            if perforated
            else ("💥 HIT BUT ARMOR DEFEATED PENETRATOR (PARTIAL)" if is_hit else "💨 MISS (OFF-TARGET)")
        ),
        "fcs_solution": fcs_res,
        "trajectory": {
            "x_m": traj_x,
            "y_m": traj_y,
            "z_m": traj_z,
        },
    }
