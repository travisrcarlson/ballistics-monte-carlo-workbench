"""Cluster Munitions, Cargo Dispenser Kinematics, and UXO Contamination Analysis.

Grounded in:
- UNODA IATG 10.10:2021 (Demining and UXO Clearance Management)
- CCW Protocol V (Explosive Remnants of War)
- Convention on Cluster Munitions (CCM) Articles 1 & 2
- NATO AASTP-1 and US Army FM 6-40

Models:
- Carrier projectile airburst at apex or descending trajectory (H_burst, V_burst, spin)
- Explosive ejection mechanics (axial burster impulse, radial blast expulsion, centrifugal spin)
- Two-stage aerodynamic descent (initial dispersal tumble + retarded parachute/ribbon descent)
- Atmospheric wind drift integration during descent
- Fuze reliability, graze failure, and backup self-destruct redundancy (FZ-06, FZ-07)
- Submunition ground strike footprint geometry (dispersion ellipse, density contours)
- UXO spatial density (items/hectare, items/km^2) and IATG 10.10 clearance cordon buffer
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from ammo_mc.environment import eb_04_isa_atmosphere
from ammo_mc.fuze import fz_06_impact_graze_function, fz_07_self_destruct_redundancy


def simulate_cluster_dispenser_footprint(
    n_submunitions: int = 64,
    submunition_mass_kg: float = 0.23,
    submunition_calibre_m: float = 0.040,
    burst_pos_m: tuple[float, float, float] = (12000.0, 0.0, 380.0),
    burst_vel_mps: tuple[float, float, float] = (240.0, 0.0, -85.0),
    carrier_spin_rad_s: float = 65.0,
    carrier_radius_m: float = 0.065,
    ejection_axial_mps: float = 24.0,
    ejection_radial_mps: float = 16.0,
    chute_cd: float = 1.25,
    chute_deploy_delay_s: float = 0.45,
    wind_downrange_mps: float = 2.0,
    wind_crossrange_mps: float = 3.5,
    p_dud_primary: float = 0.035,
    p_fail_self_destruct: float = 0.08,
    common_cause_beta: float = 0.04,
    graze_threshold_deg: float = 14.0,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Simulates carrier payload ejection, aerodynamic descent, and ground UXO footprint.

    Args:
        n_submunitions: Number of submunitions packed in carrier (e.g. 48, 64, 88).
        submunition_mass_kg: Individual submunition mass in kg (e.g. 0.23 kg for DPICM).
        submunition_calibre_m: Submunition diameter in m (e.g. 0.040 m).
        burst_pos_m: 3D coordinates [X, Y, Z] of carrier payload expulsion in meters.
        burst_vel_mps: Carrier velocity vector [Vx, Vy, Vz] at expulsion in m/s.
        carrier_spin_rad_s: Carrier roll rate at burst in rad/s.
        carrier_radius_m: Radial offset of submunition bay from spin axis in meters.
        ejection_axial_mps: Mean expulsion velocity along carrier longitudinal axis.
        ejection_radial_mps: Mean explosive blast velocity outward radially.
        chute_cd: Parachute / ribbon retarder drag coefficient.
        chute_deploy_delay_s: Delay time from expulsion until full chute inflation.
        wind_downrange_mps: Atmospheric wind blowing in +X downrange direction (m/s).
        wind_crossrange_mps: Atmospheric wind blowing in +Y crossrange direction (m/s).
        p_dud_primary: Baseline primary impact fuze failure rate (0.0 to 1.0).
        p_fail_self_destruct: Backup self-destruct mechanism failure rate (0.0 to 1.0).
        common_cause_beta: Common-cause failure fraction between primary and backup.
        graze_threshold_deg: Graze angle threshold below which dud rate increases.
        seed: Random seed for stochastic dispersion and fuze draws.

    Returns:
        Dictionary containing submunition trajectories, ground impact coordinates,
        dud/UXO classification, footprint geometry, and IATG 10.10 clearance metrics.
    """
    rng = np.random.default_rng(seed)

    x_b, y_b, z_b = [float(v) for v in burst_pos_m]
    v_b = np.array(burst_vel_mps, dtype=np.float64)
    v_b_mag = float(np.linalg.norm(v_b))
    v_b_hat = v_b / max(v_b_mag, 1e-6)

    # Establish orthogonal basis {v_b_hat, n1_hat, n2_hat} perpendicular to carrier path
    arbitrary = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(arbitrary, v_b_hat)) > 0.95:
        arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n1_hat = np.cross(v_b_hat, arbitrary)
    n1_hat /= np.linalg.norm(n1_hat)
    n2_hat = np.cross(v_b_hat, n1_hat)

    # Cross-sectional area and atmospheric terminal velocity
    area_s = 0.25 * math.pi * (submunition_calibre_m**2)
    atm_ground = eb_04_isa_atmosphere(h_m=0.0)
    rho_0 = float(atm_ground["rho_kg_m3"])
    v_term_nom = math.sqrt(2.0 * submunition_mass_kg * 9.80665 / max(rho_0 * area_s * chute_cd, 1e-9))

    # Overall theoretical UXO rate via FZ-07
    p_uxo_nom = float(
        fz_07_self_destruct_redundancy(
            p_dud_primary=p_dud_primary,
            p_fail_sd=p_fail_self_destruct,
            common_cause_beta=common_cause_beta,
        )
    )

    impact_x: list[float] = []
    impact_y: list[float] = []
    impact_v: list[float] = []
    impact_aof_deg: list[float] = []
    tof_list: list[float] = []
    status_list: list[str] = []
    trajectories: list[list[list[float]]] = []

    # Tangential spin velocity component: V_tan = omega * r
    v_tan_mag = carrier_spin_rad_s * carrier_radius_m

    for i in range(n_submunitions):
        # Azimuthal ejection angle with spiral offset
        phi_eject = 2.0 * math.pi * (i / max(n_submunitions, 1)) + float(rng.normal(0.0, 0.15))
        cos_p = math.cos(phi_eject)
        sin_p = math.sin(phi_eject)

        # Stochastic ejection velocities
        v_ax = ejection_axial_mps + float(rng.normal(0.0, 2.5))
        v_rad = ejection_radial_mps + float(rng.normal(0.0, 2.0))
        v_tan = v_tan_mag + float(rng.normal(0.0, 1.5))

        # Initial submunition velocity at burst
        # Radial blast + tangential spin
        v_trans = (v_rad * cos_p - v_tan * sin_p) * n1_hat + (v_rad * sin_p + v_tan * cos_p) * n2_hat
        v_sub = v_b + v_ax * v_b_hat + v_trans

        # Initial submunition position at burst
        r_sub = np.array([x_b, y_b, z_b], dtype=np.float64) + carrier_radius_m * (cos_p * n1_hat + sin_p * n2_hat)

        # Integration loop for this submunition
        dt = 0.05
        t_sub = 0.0
        traj_sub: list[list[float]] = [r_sub.tolist()]

        while r_sub[2] > 0.0 and t_sub < 120.0:
            cur_z = max(0.0, float(r_sub[2]))
            cur_rho = float(eb_04_isa_atmosphere(h_m=cur_z)["rho_kg_m3"])

            # Wind vector with height profile & turbulence
            w_x = wind_downrange_mps * math.pow(max(cur_z, 1.0) / 10.0, 0.14) + float(rng.normal(0.0, 0.3))
            w_y = wind_crossrange_mps * math.pow(max(cur_z, 1.0) / 10.0, 0.14) + float(rng.normal(0.0, 0.3))
            wind_vec = np.array([w_x, w_y, 0.0], dtype=np.float64)

            # Velocity relative to air
            v_air = v_sub - wind_vec
            v_air_mag = float(np.linalg.norm(v_air))

            # Parachute drag model
            if t_sub < chute_deploy_delay_s:
                cur_cd = 0.42  # Tumbling body
            else:
                cur_cd = chute_cd

            f_drag = -0.5 * cur_rho * (v_air_mag**2) * area_s * cur_cd * (v_air / max(v_air_mag, 1e-6))
            f_grav = np.array([0.0, 0.0, -submunition_mass_kg * 9.80665], dtype=np.float64)

            a_sub = (f_drag + f_grav) / max(submunition_mass_kg, 1e-6)

            r_sub += v_sub * dt + 0.5 * a_sub * (dt**2)
            v_sub += a_sub * dt
            t_sub += dt

            # Sample every 0.25 s for visualization
            if int(t_sub / dt) % 5 == 0:
                traj_sub.append(r_sub.tolist())

        # Ground impact state
        r_sub[2] = 0.0
        traj_sub.append(r_sub.tolist())
        trajectories.append(traj_sub)

        impact_speed = float(np.linalg.norm(v_sub))
        # Angle of fall relative to ground plane (0 = horizontal graze, 90 = vertical)
        aof_deg = abs(math.degrees(math.atan2(abs(float(v_sub[2])), math.hypot(float(v_sub[0]), float(v_sub[1])))))

        # Graze function check via FZ-06
        graze_eval = fz_06_impact_graze_function(
            angle_of_fall_rad=math.radians(aof_deg),
            v_impact_mps=impact_speed,
            graze_threshold_deg=graze_threshold_deg,
            min_arm_vel_mps=15.0,
            rng=rng,
        )
        p_graze_func = float(graze_eval["p_function"])

        # Stochastic Fuze draw
        primary_functions = (rng.random() > p_dud_primary) and (rng.random() < p_graze_func)
        if primary_functions:
            status = "DETONATED"
        else:
            # Primary failed -> check backup self-destruct / neutralization
            sd_fails = (rng.random() < p_fail_self_destruct) or (rng.random() < common_cause_beta)
            if sd_fails:
                status = "UNEXPLODED_ORDNANCE_UXO"
            else:
                status = "SELF_DESTRUCT_NEUTRALIZED"

        impact_x.append(float(r_sub[0]))
        impact_y.append(float(r_sub[1]))
        impact_v.append(impact_speed)
        impact_aof_deg.append(aof_deg)
        tof_list.append(t_sub)
        status_list.append(status)

    arr_x = np.asarray(impact_x, dtype=np.float64)
    arr_y = np.asarray(impact_y, dtype=np.float64)

    # Footprint geometry
    mean_x = float(np.mean(arr_x))
    mean_y = float(np.mean(arr_y))
    dx = arr_x - mean_x
    dy = arr_y - mean_y

    # Principal dispersion axes (PCA)
    cov_mat = np.cov(dx, dy)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    # 2-sigma footprint semi-axes (covers ~86% of bivariate normal impacts)
    semi_major_m = float(2.0 * math.sqrt(max(eig_vals[1], 1.0)))
    semi_minor_m = float(2.0 * math.sqrt(max(eig_vals[0], 1.0)))
    footprint_area_m2 = math.pi * semi_major_m * semi_minor_m
    footprint_area_ha = footprint_area_m2 / 10000.0

    # Footprint ellipse orientation angle
    rot_angle_deg = math.degrees(math.atan2(eig_vecs[1, 1], eig_vecs[0, 1]))

    # UXO counts & densities
    n_detonated = status_list.count("DETONATED")
    n_sd = status_list.count("SELF_DESTRUCT_NEUTRALIZED")
    n_uxo = status_list.count("UNEXPLODED_ORDNANCE_UXO")
    uxo_rate_actual = n_uxo / max(n_submunitions, 1)

    uxo_density_per_ha = (n_uxo / max(footprint_area_ha, 0.01))
    uxo_density_per_km2 = uxo_density_per_ha * 100.0

    # UNODA IATG 10.10 Clearance Cordon Buffer (200 m minimum beyond outermost strike)
    max_radial_dist = float(np.max(np.hypot(dx, dy)))
    cordon_radius_m = max_radial_dist + 200.0
    cordon_area_ha = (math.pi * (cordon_radius_m**2)) / 10000.0

    return {
        "n_submunitions": n_submunitions,
        "n_detonated": n_detonated,
        "n_self_destruct_neutralized": n_sd,
        "n_uxo": n_uxo,
        "actual_uxo_rate": float(uxo_rate_actual),
        "nominal_uxo_rate_fz07": float(p_uxo_nom),
        "mean_impact_x_m": mean_x,
        "mean_impact_y_m": mean_y,
        "footprint_semi_major_m": semi_major_m,
        "footprint_semi_minor_m": semi_minor_m,
        "footprint_area_m2": float(footprint_area_m2),
        "footprint_area_hectares": float(footprint_area_ha),
        "footprint_rotation_deg": float(rot_angle_deg),
        "uxo_density_per_ha": float(uxo_density_per_ha),
        "uxo_density_per_km2": float(uxo_density_per_km2),
        "iatg_cordon_radius_m": float(cordon_radius_m),
        "iatg_cordon_area_ha": float(cordon_area_ha),
        "mean_terminal_velocity_mps": float(np.mean(impact_v)),
        "nominal_terminal_velocity_mps": float(v_term_nom),
        "mean_angle_of_fall_deg": float(np.mean(impact_aof_deg)),
        "mean_time_of_flight_s": float(np.mean(tof_list)),
        "submunitions": [
            {
                "id": i + 1,
                "impact_x_m": impact_x[i],
                "impact_y_m": impact_y[i],
                "impact_v_mps": impact_v[i],
                "angle_of_fall_deg": impact_aof_deg[i],
                "status": status_list[i],
                "is_uxo": bool(status_list[i] == "UNEXPLODED_ORDNANCE_UXO"),
            }
            for i in range(n_submunitions)
        ],
        "burst_point_m": list(burst_pos_m),
        "trajectories": trajectories,
    }
