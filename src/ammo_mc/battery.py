"""Artillery Battery Sheafs, Multiple Rounds Simultaneous Impact (MRSI/TOT), and Salvo Optimization.

Grounded in:
- US Army FM 6-40 (Tactics, Techniques, and Procedures for Field Artillery Manual Cannon Gunnery)
- US Army FM 3-09 (Field Artillery Operations and Fire Support)
- NATO STANAG 4119 (Artillery Firing Tables and System Software)

Models:
- Multi-gun battery layout (4 to 8 howitzers / 6 MLRS launchers) with staggered tactical spacing
- Sheaf geometry: Open Sheaf, Converged Sheaf, Linear Sheaf, and Area Polygon Sheaf
- Multiple Rounds Simultaneous Impact (MRSI) & Time-on-Target (TOT) inverse ballistic elevation solver
- Multi-gun simultaneous 3D trajectory integration with correlated occasion errors and aleatory dispersion
- Target neutralization & Weapon-Target Assignment (WTA) optimization via Carleton damage function
"""

from __future__ import annotations

import math
from typing import Any, Literal
import numpy as np

from ammo_mc.environment import eb_04_isa_atmosphere
from ammo_mc.exterior import BatchTrajectoryConfig, integrate_trajectories_batch
from ammo_mc.vulnerability import tv_04_sspk_carleton


def generate_battery_layout(
    n_guns: int = 6,
    formation: Literal["linear", "wedge", "staggered", "lazy_w"] = "staggered",
    gun_spacing_m: float = 75.0,
    center_pos_m: tuple[float, float] = (0.0, 0.0),
) -> list[dict[str, float]]:
    """Generates 2D gun positions [X, Y] for an artillery battery in meters."""
    cx, cy = center_pos_m
    guns: list[dict[str, float]] = []

    if formation == "linear":
        # Straight line along Y-axis (crossrange)
        y_start = cy - 0.5 * (n_guns - 1) * gun_spacing_m
        for i in range(n_guns):
            guns.append({"gun_id": i + 1, "x_m": cx, "y_m": y_start + i * gun_spacing_m, "z_m": 0.0})

    elif formation == "wedge":
        # V-shaped wedge pointing downrange +X
        mid = (n_guns - 1) / 2.0
        for i in range(n_guns):
            offset_y = (i - mid) * gun_spacing_m
            offset_x = -abs(i - mid) * 0.4 * gun_spacing_m
            guns.append({"gun_id": i + 1, "x_m": cx + offset_x, "y_m": cy + offset_y, "z_m": 0.0})

    elif formation == "lazy_w":
        # W-shaped dispersals to minimize counter-battery radar alignment
        w_offsets_x = [0.0, 35.0, -25.0, 20.0, -30.0, 15.0, -20.0, 25.0]
        y_start = cy - 0.5 * (n_guns - 1) * gun_spacing_m
        for i in range(n_guns):
            ox = w_offsets_x[i % len(w_offsets_x)]
            guns.append({"gun_id": i + 1, "x_m": cx + ox, "y_m": y_start + i * gun_spacing_m, "z_m": 0.0})

    else:  # staggered (default)
        # Staggered alternating offset (forward/back)
        y_start = cy - 0.5 * (n_guns - 1) * gun_spacing_m
        for i in range(n_guns):
            ox = 30.0 if (i % 2 == 1) else -30.0
            guns.append({"gun_id": i + 1, "x_m": cx + ox, "y_m": y_start + i * gun_spacing_m, "z_m": 0.0})

    return guns


def calculate_sheaf_aim_points(
    guns: list[dict[str, float]],
    target_pos_m: tuple[float, float] = (15000.0, 0.0),
    sheaf_type: Literal["converged", "open", "linear", "area"] = "converged",
    linear_length_m: float = 300.0,
    linear_azimuth_deg: float = 90.0,
    area_radius_m: float = 120.0,
) -> list[tuple[float, float]]:
    """Calculates intended mean point of impact (MPI) aim points (X_aim, Y_aim) for each gun."""
    tx, ty = target_pos_m
    n_guns = len(guns)
    aim_points: list[tuple[float, float]] = []

    if sheaf_type == "converged":
        # All weapons aim at the exact same target coordinates
        for _ in range(n_guns):
            aim_points.append((tx, ty))

    elif sheaf_type == "open":
        # Weapons lay parallel centered on the battery geometric centroid
        cx = sum(g["x_m"] for g in guns) / max(n_guns, 1)
        cy = sum(g["y_m"] for g in guns) / max(n_guns, 1)
        for g in guns:
            dx = g["x_m"] - cx
            dy = g["y_m"] - cy
            aim_points.append((tx + dx, ty + dy))

    elif sheaf_type == "linear":
        # Aim points evenly staggered along a line (e.g. trench or mechanized column)
        ang_rad = math.radians(linear_azimuth_deg)
        half_len = 0.5 * linear_length_m
        if n_guns > 1:
            step = linear_length_m / (n_guns - 1)
        else:
            step = 0.0
        for i in range(n_guns):
            dist_along = -half_len + i * step
            ax = tx + dist_along * math.cos(ang_rad)
            ay = ty + dist_along * math.sin(ang_rad)
            aim_points.append((ax, ay))

    else:  # area
        # Aim points distributed uniformly inside a circular area polygon
        for i in range(n_guns):
            frac = (i + 0.5) / n_guns
            rad = area_radius_m * math.sqrt(frac)
            theta = 2.0 * math.pi * (i / max(n_guns, 1))
            aim_points.append((tx + rad * math.cos(theta), ty + rad * math.sin(theta)))

    return aim_points


def find_elevation_for_range(
    target_range_m: float,
    v0_mps: float = 827.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    high_angle: bool = False,
    max_iter: int = 10,
) -> float:
    """Finds the required quadrant elevation QE (degrees) for a given ground range using bisection."""
    from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch

    atm_env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=calibre_m, fidelity="L1", dt=0.12)

    def eval_r(qe_deg: float) -> float:
        rad = math.radians(qe_deg)
        s0 = np.array([[0.0, 0.0, 0.0, v0_mps * math.cos(rad), 0.0, v0_mps * math.sin(rad)]], dtype=np.float64)
        res = integrate_trajectories_batch(s0, np.array([mass_kg]), np.array([1.0]), atm_env, cfg)
        return float(res["range_m"][0])

    # Vectorized sweep finds true drag-shifted max-range elevation (often 48-52 deg under G7 drag)
    sweep_qes = np.linspace(2.0, 80.0, 20)
    rads = np.radians(sweep_qes)
    s_sw = np.zeros((len(sweep_qes), 6), dtype=np.float64)
    s_sw[:, 3] = v0_mps * np.cos(rads)
    s_sw[:, 5] = v0_mps * np.sin(rads)
    r_sw = integrate_trajectories_batch(s_sw, np.full(len(sweep_qes), mass_kg), np.ones(len(sweep_qes)), atm_env, cfg)["range_m"]
    idx_peak = int(np.argmax(r_sw))
    qe_peak = float(sweep_qes[idx_peak])

    if not high_angle:
        q_low, q_high = 2.0, min(qe_peak + 1.5, 80.0)
        for _ in range(max_iter):
            q_mid = 0.5 * (q_low + q_high)
            r_mid = eval_r(q_mid)
            if abs(r_mid - target_range_m) < 15.0:
                return q_mid
            if r_mid < target_range_m:
                q_low = q_mid
            else:
                q_high = q_mid
        return 0.5 * (q_low + q_high)
    else:
        q_low, q_high = max(qe_peak - 1.5, 45.0), 82.0
        for _ in range(max_iter):
            q_mid = 0.5 * (q_low + q_high)
            r_mid = eval_r(q_mid)
            if abs(r_mid - target_range_m) < 15.0:
                return q_mid
            if r_mid > target_range_m:
                q_low = q_mid
            else:
                q_high = q_mid
        return 0.5 * (q_low + q_high)


def solve_mrsi_trajectory_plan(
    target_range_m: float = 12000.0,
    v0_mps: float = 827.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    max_rounds: int = 4,
    min_qe_deg: float = 5.0,
    max_qe_deg: float = 78.0,
) -> dict[str, Any]:
    """Solves for a Multiple Rounds Simultaneous Impact (MRSI) firing sequence."""
    from ammo_mc.exterior import AtmosphereEnvironment, BatchTrajectoryConfig, integrate_trajectories_batch

    atm_env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=calibre_m, fidelity="L1", dt=0.04)

    # Compute true aerodynamic max range for active charge
    probe_qes = np.linspace(35.0, 58.0, 12)
    probe_rads = np.radians(probe_qes)
    s_probe = np.zeros((len(probe_qes), 6), dtype=np.float64)
    s_probe[:, 3] = v0_mps * np.cos(probe_rads)
    s_probe[:, 5] = v0_mps * np.sin(probe_rads)
    true_max_range_m = float(np.max(integrate_trajectories_batch(s_probe, np.full(len(probe_qes), mass_kg), np.ones(len(probe_qes)), atm_env, cfg)["range_m"]))

    mrsi_rounds: list[dict[str, Any]] = []
    if target_range_m <= true_max_range_m:
        # 1. Low angle solution
        qe_low = find_elevation_for_range(target_range_m, v0_mps, calibre_m, mass_kg, high_angle=False)
        if min_qe_deg <= qe_low <= max_qe_deg:
            rad_low = math.radians(qe_low)
            s0_l = np.array([[0.0, 0.0, 0.0, v0_mps * math.cos(rad_low), 0.0, v0_mps * math.sin(rad_low)]], dtype=np.float64)
            res_l = integrate_trajectories_batch(s0_l, np.array([mass_kg]), np.array([1.0]), atm_env, cfg)
            tof_low = float(res_l["tof_s"][0])
            mrsi_rounds.append({"round_num": 1, "qe_deg": round(qe_low, 2), "tof_s": round(tof_low, 2), "branch": "Low Angle"})

        # 2. High angle solution
        qe_high = find_elevation_for_range(target_range_m, v0_mps, calibre_m, mass_kg, high_angle=True)
        if min_qe_deg <= qe_high <= max_qe_deg:
            rad_high = math.radians(qe_high)
            s0_h = np.array([[0.0, 0.0, 0.0, v0_mps * math.cos(rad_high), 0.0, v0_mps * math.sin(rad_high)]], dtype=np.float64)
            res_h = integrate_trajectories_batch(s0_h, np.array([mass_kg]), np.array([1.0]), atm_env, cfg)
            tof_high = float(res_h["tof_s"][0])
            if not mrsi_rounds or abs(tof_high - mrsi_rounds[0]["tof_s"]) > 8.0:
                mrsi_rounds.append({"round_num": len(mrsi_rounds) + 1, "qe_deg": round(qe_high, 2), "tof_s": round(tof_high, 2), "branch": "High Angle"})

    # 3. Intermediate angle with reduced Charge 4 velocity (check reachability!)
    if max_rounds >= 3 and len(mrsi_rounds) >= 2:
        v0_mid = v0_mps * 0.90
        s_p4 = np.zeros((len(probe_qes), 6), dtype=np.float64)
        s_p4[:, 3] = v0_mid * np.cos(probe_rads)
        s_p4[:, 5] = v0_mid * np.sin(probe_rads)
        chg4_max_r = float(np.max(integrate_trajectories_batch(s_p4, np.full(len(probe_qes), mass_kg), np.ones(len(probe_qes)), atm_env, cfg)["range_m"]))
        if target_range_m <= chg4_max_r:
            qe_mid = find_elevation_for_range(target_range_m, v0_mid, calibre_m, mass_kg, high_angle=False)
            if min_qe_deg <= qe_mid <= max_qe_deg:
                rad_mid = math.radians(qe_mid)
                s0_m = np.array([[0.0, 0.0, 0.0, v0_mid * math.cos(rad_mid), 0.0, v0_mid * math.sin(rad_mid)]], dtype=np.float64)
                res_m = integrate_trajectories_batch(s0_m, np.array([mass_kg]), np.array([1.0]), atm_env, cfg)
                tof_mid = float(res_m["tof_s"][0])
                mrsi_rounds.append({"round_num": 3, "qe_deg": round(qe_mid, 2), "tof_s": round(tof_mid, 2), "branch": "Charge 4 (Mid)"})

    mrsi_rounds.sort(key=lambda r: r["tof_s"], reverse=True)
    max_tof = mrsi_rounds[0]["tof_s"] if mrsi_rounds else 0.0

    for idx, rd in enumerate(mrsi_rounds):
        rd["round_num"] = idx + 1
        rd["fire_time_s"] = round(max_tof - rd["tof_s"], 2)
        rd["impact_time_s"] = round(max_tof, 2)

    return {
        "mrsi_feasible": len(mrsi_rounds) >= 2,
        "target_range_m": float(target_range_m),
        "max_range_m": round(true_max_range_m, 1),
        "rounds_planned": len(mrsi_rounds),
        "total_engagement_duration_s": max_tof,
        "firing_window_s": round(mrsi_rounds[-1]["fire_time_s"], 2) if mrsi_rounds else 0.0,
        "plan": mrsi_rounds,
    }


def simulate_battery_salvo(
    n_guns: int = 6,
    formation: Literal["linear", "wedge", "staggered", "lazy_w"] = "staggered",
    gun_spacing_m: float = 75.0,
    target_pos_m: tuple[float, float] = (15000.0, 0.0),
    sheaf_type: Literal["converged", "open", "linear", "area"] = "converged",
    linear_length_m: float = 300.0,
    linear_azimuth_deg: float = 90.0,
    rounds_per_gun: int = 2,
    v0_mps: float = 827.0,
    calibre_m: float = 0.155,
    mass_kg: float = 43.5,
    sigma_mv_rr_mps: float = 1.5,
    sigma_aim_qe_mils: float = 0.8,
    sigma_aim_az_mils: float = 0.8,
    lethal_radius_m: float = 30.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulates a complete tactical artillery battery salvo firing across a sheaf.

    Computes:
    - Weapon layout and aim points.
    - Stochastic 3D trajectory integration for every round fired by the battery.
    - Ground impact cluster coordinates $(X_{i,j}, Y_{i,j})$.
    - Bivariate dispersion (CEP50, CEP95) per gun and across the entire battery sheaf.
    - Fractional target area neutralization / lethality coverage.
    """
    rng = np.random.default_rng(seed)
    guns = generate_battery_layout(n_guns=n_guns, formation=formation, gun_spacing_m=gun_spacing_m)
    aim_pts = calculate_sheaf_aim_points(
        guns=guns,
        target_pos_m=target_pos_m,
        sheaf_type=sheaf_type,
        linear_length_m=linear_length_m,
        linear_azimuth_deg=linear_azimuth_deg,
    )

    from ammo_mc.exterior import AtmosphereEnvironment

    atm_env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=calibre_m, fidelity="L1", dt=0.08)

    total_rounds = n_guns * rounds_per_gun
    round_records: list[dict[str, Any]] = []
    impact_x_list: list[float] = []
    impact_y_list: list[float] = []
    trajectories: list[list[list[float]]] = []

    # Cache elevation solutions for guns at similar ranges
    qe_cache: dict[int, float] = {}

    for g_idx, gun in enumerate(guns):
        gx, gy = gun["x_m"], gun["y_m"]
        ax, ay = aim_pts[g_idx]

        # Range and bearing from this specific gun to its assigned sheaf aim point
        dx = ax - gx
        dy = ay - gy
        dist_gun = math.hypot(dx, dy)
        bearing_rad = math.atan2(dy, dx)

        # Accurate elevation angle for this gun's distance with drag (cached to nearest 20m)
        dist_key = int(round(dist_gun / 20.0) * 20)
        if dist_key not in qe_cache:
            qe_cache[dist_key] = find_elevation_for_range(dist_gun, v0_mps, calibre_m, mass_kg)
        qe_deg_val = qe_cache[dist_key]
        qe_rad = math.radians(qe_deg_val)

        for r_idx in range(rounds_per_gun):
            # Stochastic round perturbations
            v0_actual = v0_mps + float(rng.normal(0.0, sigma_mv_rr_mps))
            d_qe = float(rng.normal(0.0, math.radians(sigma_aim_qe_mils * 0.05625)))
            d_az = float(rng.normal(0.0, math.radians(sigma_aim_az_mils * 0.05625)))

            qe_actual = qe_rad + d_qe
            # Relative downrange and crossrange velocity components
            v_down = v0_actual * math.cos(qe_actual) * math.cos(d_az)
            v_cross_left = -v0_actual * math.cos(qe_actual) * math.sin(d_az)
            vz = v0_actual * math.sin(qe_actual)

            s0 = np.array([[0.0, 0.0, 0.0, v_down, v_cross_left, vz]], dtype=np.float64)
            res = integrate_trajectories_batch(s0, np.array([mass_kg]), np.array([1.0]), atm_env, cfg, record_single_path=True)

            sp = res.get("sample_path", {})
            if sp and "range_m" in sp:
                # Downsample trajectory to ~30 points for fast 3D WebGL rendering
                step_tr = max(1, len(sp["t"]) // 30)
                cos_b = math.cos(bearing_rad)
                sin_b = math.sin(bearing_rad)
                # deflection_m is +RIGHT -> rightward normal of bore heading (cos b, sin b) is (+sin b, -cos b)
                traj_pts = [
                    [
                        gx + float(sp["range_m"][k]) * cos_b + float(sp["deflection_m"][k]) * sin_b,
                        gy + float(sp["range_m"][k]) * sin_b - float(sp["deflection_m"][k]) * cos_b,
                        float(sp["alt_m"][k]),
                    ]
                    for k in range(0, len(sp["t"]), step_tr)
                ]
                trajectories.append(traj_pts)

            bore_range = float(res["range_m"][0])
            bore_defl = float(res["deflection_m"][0])
            imp_x = gx + bore_range * math.cos(bearing_rad) + bore_defl * math.sin(bearing_rad)
            imp_y = gy + bore_range * math.sin(bearing_rad) - bore_defl * math.cos(bearing_rad)
            tof = float(res["tof_s"][0])
            v_imp = float(res["v_impact_mps"][0])

            impact_x_list.append(imp_x)
            impact_y_list.append(imp_y)

            round_records.append({
                "round_id": len(round_records) + 1,
                "gun_id": gun["gun_id"],
                "round_in_gun": r_idx + 1,
                "aim_x_m": ax,
                "aim_y_m": ay,
                "impact_x_m": imp_x,
                "impact_y_m": imp_y,
                "miss_dist_m": math.hypot(imp_x - ax, imp_y - ay),
                "dist_from_target_m": math.hypot(imp_x - target_pos_m[0], imp_y - target_pos_m[1]),
                "tof_s": round(tof, 2),
                "v_impact_mps": round(v_imp, 1),
            })

    arr_ix = np.asarray(impact_x_list)
    arr_iy = np.asarray(impact_y_list)

    # Battery sheaf dispersion statistics
    mean_imp_x = float(np.mean(arr_ix))
    mean_imp_y = float(np.mean(arr_iy))
    radial_misses = np.hypot(arr_ix - target_pos_m[0], arr_iy - target_pos_m[1])
    radial_misses_sorted = np.sort(radial_misses)

    cep50 = float(np.percentile(radial_misses_sorted, 50))
    cep95 = float(np.percentile(radial_misses_sorted, 95))

    # Carleton lethality coverage: compute expected fractional neutralization over target grid
    grid_x = np.linspace(target_pos_m[0] - 250.0, target_pos_m[0] + 250.0, 35)
    grid_y = np.linspace(target_pos_m[1] - 250.0, target_pos_m[1] + 250.0, 35)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # Cumulative survival across all rounds via TV-01 Carleton damage function: P_kill = exp(-r^2 / R_C^2)
    surv_prob = np.ones_like(GX)
    r_c_sq = max(float(lethal_radius_m) ** 2, 1e-6)
    for ix, iy in zip(impact_x_list, impact_y_list):
        r2 = (GX - ix) ** 2 + (GY - iy) ** 2
        p_hit = np.exp(-r2 / r_c_sq)
        surv_prob *= (1.0 - p_hit)

    lethal_map = 1.0 - surv_prob
    fractional_neutralization = float(np.mean(lethal_map))

    return {
        "n_guns": n_guns,
        "rounds_per_gun": rounds_per_gun,
        "total_rounds": total_rounds,
        "formation": formation,
        "sheaf_type": sheaf_type,
        "target_pos_m": list(target_pos_m),
        "mean_impact_x_m": mean_imp_x,
        "mean_impact_y_m": mean_imp_y,
        "cep50_m": cep50,
        "cep95_m": cep95,
        "fractional_neutralization_pct": round(fractional_neutralization * 100.0, 1),
        "target_neutralization_prob": fractional_neutralization,
        "guns": guns,
        "aim_points": aim_pts,
        "rounds": round_records,
        "trajectories": trajectories,
        "lethality_grid": {
            "x_m": grid_x.tolist(),
            "y_m": grid_y.tolist(),
            "lethal_map": lethal_map.tolist(),
        },
    }
