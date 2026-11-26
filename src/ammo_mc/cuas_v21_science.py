"""v2.1.0 Scientific Engine: Nested L0 Epistemic UQ, Morris/Sobol, Doctrine Lab, Manifests & GP Pareto.

Implements:
  - Track 1: Nested L0 Epistemic outer-loop P5/P50/P95 bands (`MC-F21`), Morris screening (`MC-F19`),
    and Saltelli/Jansen Sobol global sensitivity (`MC-F18`).
  - Track 3: Interactive M12 Engagement Doctrine Lab (Simultaneous Salvo, Ripple Salvo, Wilkening SLS,
    Whistler Shoot-Adjust-Shoot, and Glazebrook-Washburn Gittins multi-target priority).
  - Track 5: Pydantic-validated Reproducible Scenario Manifest & Provenance bundle (JSON/YAML + SHA-256).
  - Track 6: Pure-SciPy Matern-5/2 Gaussian Process surrogate emulator with closed-form LOO-CV &
    multi-objective Pareto optimizer (`M17`).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field
from scipy import integrate, linalg, optimize, stats as sp_stats

from ammo_mc import __version__
from ammo_mc.cuas_ballistics_v2 import (
    _eval_v2_drag_cd,
    cipm_2007_air_density,
    compute_saami_recoil_impulse,
    sutherland_viscosity_pa_s,
    tropospheric_pressure_pa,
    velocity_ogive_peff,
    wen_2000_frp_ballistic_limit,
)
from ammo_mc.cuas_rigor import (
    compute_wilson_ci,
    gittins_sls_index,
    pk_salvo_brackets,
    pk_single_shot_closed_form,
    resolve_verification_grade,
    wilkening_ns,
)

# -------------------------------------------------------------------------
# 1. NESTED L0 EPISTEMIC UNCERTAINTY & MORRIS / SOBOL GLOBAL SENSITIVITY
# -------------------------------------------------------------------------

EPISTEMIC_PARAMETER_SPECS: list[dict[str, Any]] = [
    {
        "key": "drag_multiplier_kd",
        "label": "Drag Curve Factor k_d",
        "min": 0.95,
        "nominal": 1.00,
        "max": 1.30,
        "unit": "-",
        "grade": "V2",
        "source_id": "S71",
        "note": "Soft lead deformation / table uncertainty (+0 to +30% Cd).",
    },
    {
        "key": "pattern_sigma_scale",
        "label": "Pattern Spread Scale",
        "min": 0.80,
        "nominal": 1.00,
        "max": 1.25,
        "unit": "-",
        "grade": "V2",
        "source_id": "S70",
        "note": "Lot-to-lot & shot-to-shot pattern sigma dispersion multiplier.",
    },
    {
        "key": "vulnerable_area_scale",
        "label": "Vulnerable Area A_v Scale",
        "min": 0.60,
        "nominal": 1.00,
        "max": 1.40,
        "unit": "-",
        "grade": "V3",
        "source_id": "S72",
        "note": "Effective presented critical-component area multiplier.",
    },
    {
        "key": "aim_error_scale",
        "label": "Tracker / Aim Error Scale",
        "min": 0.65,
        "nominal": 1.00,
        "max": 1.50,
        "unit": "-",
        "grade": "V4",
        "source_id": "S85",
        "note": "Turret + ATR + predictor total angular/maneuver error multiplier.",
    },
    {
        "key": "v50_scale",
        "label": "Ballistic Limit V50 Scale",
        "min": 0.80,
        "nominal": 1.00,
        "max": 1.25,
        "unit": "-",
        "grade": "V3",
        "source_id": "S82",
        "note": "CFRP / shell V50 perforation threshold uncertainty.",
    },
    {
        "key": "dead_time_scale",
        "label": "FCS Dead-Time Scale",
        "min": 0.75,
        "nominal": 1.00,
        "max": 1.35,
        "unit": "-",
        "grade": "V4",
        "source_id": "S85",
        "note": "End-to-end latency variation around ~141 ms nominal.",
    },
]


def _resolve_ammo_and_target_baseline(
    ammo_preset: str,
    choke: str,
    drone_class: str,
) -> dict[str, Any]:
    from ammo_mc.small_arms_pkg.catalogs import CHOKE_DEFINITIONS, SHOTGUN_PRESETS, resolve_pellet_density_g_cm3

    preset = SHOTGUN_PRESETS.get(ammo_preset, SHOTGUN_PRESETS["12G_NO4_BUCK_C_UAS"])
    choke_info = CHOKE_DEFINITIONS.get(choke, CHOKE_DEFINITIONS["Modified"])
    p40 = float(choke_info["benchmark_40yd_30in_pct"])
    sigma_ang_rad = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(max(1e-6, 1.0 - p40)))

    d_mm = float(preset["pellet_diameter_mm"])
    m_g = float(preset["pellet_mass_g"])
    n_pellets = int(preset["pellets_per_shot"])
    v0 = float(preset["v0_mps"])
    rho_p = resolve_pellet_density_g_cm3(str(preset.get("material", "lead")), preset.get("material_density_g_cm3")) * 1000.0

    if "plastic" in str(drone_class).lower():
        a_v_m2 = 0.0080
        v50_mps = math.sqrt(2.0 * 5.0 / max(m_g / 1000.0, 1e-6))
    else:
        a_v_m2 = 0.0055
        wen = wen_2000_frp_ballistic_limit(m_g / 1000.0, d_mm / 1000.0, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
        v50_mps = float(wen["v_bl_mps"])

    return {
        "ammo_name": str(preset["name"]),
        "d_mm": d_mm,
        "m_g": m_g,
        "n_pellets": n_pellets,
        "v0_mps": v0,
        "rho_p_kg_m3": rho_p,
        "sigma_ang_rad": sigma_ang_rad,
        "a_v_m2": a_v_m2,
        "v50_mps": v50_mps,
    }


def _single_pass_rk4_stations(
    v0_mps: float,
    d_mm: float,
    rho_p_kg_m3: float,
    m_g: float,
    stations_m: tuple[float, ...],
    drag_law: str,
    k_d: float,
    temp_c: float,
    pressure_pa: float,
) -> list[tuple[float, float, float]]:
    """Integrate 1D horizontal sphere deceleration once across ascending range stations (fast RK4)."""
    d_m = max(1e-4, float(d_mm) / 1000.0)
    m_kg = max(1e-6, float(m_g) / 1000.0)
    area_p = math.pi * 0.25 * (d_m ** 2)
    t_k = max(180.0, float(temp_c) + 273.15)
    rho_air = max(1000.0, float(pressure_pa)) / (287.053 * t_k)
    a_sound = math.sqrt(1.4 * 287.053 * t_k)
    mu_air = sutherland_viscosity_pa_s(temp_c)
    kd_eff = max(0.5, float(k_d))

    v = max(1.0, float(v0_mps))
    tof = 0.0
    x = 0.0
    out: list[tuple[float, float, float]] = []

    def _deriv(v_val: float) -> tuple[float, float]:
        vs = max(1.0, v_val)
        cd = _eval_v2_drag_cd(vs, a_sound, drag_law=drag_law, diam_m=d_m, rho_kg_m3=rho_air, mu_pa_s=mu_air)
        return -(0.5 * rho_air * kd_eff * cd * area_p * vs) / m_kg, 1.0 / vs

    for r_tgt in sorted(stations_m):
        rt = max(x, float(r_tgt))
        span = rt - x
        if span > 1e-6:
            steps = max(4, int(math.ceil(span / 1.5)))
            dx = span / steps
            for _ in range(steps):
                k1_v, k1_t = _deriv(v)
                k2_v, k2_t = _deriv(v + 0.5 * dx * k1_v)
                k3_v, k3_t = _deriv(v + 0.5 * dx * k2_v)
                k4_v, k4_t = _deriv(v + dx * k3_v)
                v = max(1.0, v + (dx / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v))
                tof += (dx / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
            x = rt
        out.append((rt, float(v), float(tof)))
    return out


def _eval_epistemic_sample(
    base: dict[str, Any],
    stations_m: tuple[float, ...],
    *,
    n_barrels: int,
    aim_error_mrad: float,
    dead_time_s: float,
    a_unmod_mps2: float,
    drone_speed_mps: float,
    drone_altitude_m: float,
    warhead_blast_radius_m: float,
    pk_threshold: float,
    drag_law: str,
    station_p_pa: float,
    temp_c: float,
    kd_mult: float,
    sig_scale: float,
    av_scale: float,
    aim_scale: float,
    v50_scale: float,
    dead_scale: float,
) -> dict[str, Any]:
    traj_pts = _single_pass_rk4_stations(
        base["v0_mps"],
        base["d_mm"],
        base["rho_p_kg_m3"],
        base["m_g"],
        stations_m,
        drag_law=drag_law,
        k_d=kd_mult,
        temp_c=temp_c,
        pressure_pa=station_p_pa,
    )
    v50_eff = max(10.0, base["v50_mps"] * float(v50_scale))
    av_eff = max(1e-5, base["a_v_m2"] * float(av_scale))
    td_eff = max(0.01, float(dead_time_s) * float(dead_scale))
    aim_ang_eff = max(0.1, float(aim_error_mrad) * float(aim_scale))
    x_wreck = max(0.5, float(drone_speed_mps)) * math.sqrt(2.0 * max(1.0, float(drone_altitude_m)) / 9.80665)
    r_min_safe = x_wreck + max(0.0, float(warhead_blast_radius_m))

    pk_shared_list: list[float] = []
    pk_indep_list: list[float] = []
    pk_single_list: list[float] = []
    qualifying_r: list[float] = []
    for r_m, v_ret, tof_s in traj_pts:
        p_eff = velocity_ogive_peff(v_ret, v50_eff, spread_s_v=0.10)
        sig_p = max(1e-3, base["sigma_ang_rad"] * r_m * float(sig_scale))
        sig_a = math.hypot((aim_ang_eff / 1000.0) * r_m, 0.5 * float(a_unmod_mps2) * ((td_eff + tof_s) ** 2))
        lam0 = (base["n_pellets"] / (2.0 * math.pi * (sig_p ** 2))) * av_eff * p_eff
        br = pk_salvo_brackets(lam0, sig_p, sig_a, n_shots=n_barrels)
        pk_shared_list.append(br["pk_shared_lower"] * 100.0)
        pk_indep_list.append(br["pk_independent_upper"] * 100.0)
        pk_single_list.append(br["pk_single"] * 100.0)
        if br["pk_shared_lower"] >= float(pk_threshold):
            qualifying_r.append(r_m)

    r_max_shared = max(qualifying_r) if qualifying_r else 0.0
    usable_w = max(0.0, r_max_shared - r_min_safe)
    return {
        "pk_shared": pk_shared_list,
        "pk_indep": pk_indep_list,
        "pk_single": pk_single_list,
        "usable_window_m": usable_w,
        "r_max_shared_m": r_max_shared,
        "r_min_safe_m": r_min_safe,
    }


def run_cuas_l0_epistemic_envelope(
    *,
    ammo_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    drone_class: str = "fpv_cfrp",
    n_barrels: int = 4,
    aim_error_mrad: float = 2.0,
    dead_time_s: float = 0.141,
    a_unmod_mps2: float = 5.0,
    drone_speed_mps: float = 25.0,
    drone_altitude_m: float = 15.0,
    warhead_blast_radius_m: float = 8.0,
    pk_threshold: float = 0.50,
    drag_law: str = "walsh_1976",
    site_altitude_m: float = 0.0,
    temp_c: float = 15.0,
    range_stations_m: tuple[float, ...] = (15.0, 25.0, 35.0, 40.0, 50.0, 65.0, 75.0, 85.0),
    n_outer: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Nested L0 Epistemic outer-loop LHS propagation giving P5/P50/P95 Pk(R) & Window bounds (`MC-F21`)."""
    n_out = max(16, int(n_outer))
    stations = tuple(sorted(float(r) for r in range_stations_m))
    base = _resolve_ammo_and_target_baseline(ammo_preset, choke, drone_class)
    station_p_pa = tropospheric_pressure_pa(site_altitude_m)

    sampler = sp_stats.qmc.LatinHypercube(d=6, seed=int(seed))
    u_mat = sampler.random(n=n_out)
    lo = np.array([s["min"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)
    hi = np.array([s["max"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)
    theta_mat = lo + u_mat * (hi - lo)

    pk_sh_mat = np.zeros((n_out, len(stations)), dtype=np.float64)
    pk_in_mat = np.zeros((n_out, len(stations)), dtype=np.float64)
    win_arr = np.zeros(n_out, dtype=np.float64)
    rmax_arr = np.zeros(n_out, dtype=np.float64)

    for j in range(n_out):
        kd_m, sig_s, av_s, aim_s, v50_s, dead_s = [float(v) for v in theta_mat[j]]
        ev = _eval_epistemic_sample(
            base,
            stations,
            n_barrels=n_barrels,
            aim_error_mrad=aim_error_mrad,
            dead_time_s=dead_time_s,
            a_unmod_mps2=a_unmod_mps2,
            drone_speed_mps=drone_speed_mps,
            drone_altitude_m=drone_altitude_m,
            warhead_blast_radius_m=warhead_blast_radius_m,
            pk_threshold=pk_threshold,
            drag_law=drag_law,
            station_p_pa=station_p_pa,
            temp_c=temp_c,
            kd_mult=kd_m,
            sig_scale=sig_s,
            av_scale=av_s,
            aim_scale=aim_s,
            v50_scale=v50_s,
            dead_scale=dead_s,
        )
        pk_sh_mat[j, :] = ev["pk_shared"]
        pk_in_mat[j, :] = ev["pk_indep"]
        win_arr[j] = ev["usable_window_m"]
        rmax_arr[j] = ev["r_max_shared_m"]

    nom_ev = _eval_epistemic_sample(
        base,
        stations,
        n_barrels=n_barrels,
        aim_error_mrad=aim_error_mrad,
        dead_time_s=dead_time_s,
        a_unmod_mps2=a_unmod_mps2,
        drone_speed_mps=drone_speed_mps,
        drone_altitude_m=drone_altitude_m,
        warhead_blast_radius_m=warhead_blast_radius_m,
        pk_threshold=pk_threshold,
        drag_law=drag_law,
        station_p_pa=station_p_pa,
        temp_c=temp_c,
        kd_mult=1.0,
        sig_scale=1.0,
        av_scale=1.0,
        aim_scale=1.0,
        v50_scale=1.0,
        dead_scale=1.0,
    )

    curve_rows: list[dict[str, float]] = []
    for idx_r, r_val in enumerate(stations):
        col = pk_sh_mat[:, idx_r]
        col_in = pk_in_mat[:, idx_r]
        curve_rows.append({
            "range_m": float(r_val),
            "pk_shared_nominal_pct": round(float(nom_ev["pk_shared"][idx_r]), 2),
            "pk_shared_p5_pct": round(float(np.percentile(col, 5)), 2),
            "pk_shared_p50_pct": round(float(np.percentile(col, 50)), 2),
            "pk_shared_p95_pct": round(float(np.percentile(col, 95)), 2),
            "pk_indep_p50_pct": round(float(np.percentile(col_in, 50)), 2),
        })

    ref_idx = int(np.argmin(np.abs(np.array(stations) - 40.0)))
    ref_band = curve_rows[ref_idx]["pk_shared_p95_pct"] - curve_rows[ref_idx]["pk_shared_p5_pct"]
    win_p5, win_p50, win_p95 = np.percentile(win_arr, [5, 50, 95])

    return {
        "summary": {
            "n_outer_epistemic": n_out,
            "ammo_preset": ammo_preset,
            "ammo_name": base["ammo_name"],
            "drone_class": drone_class,
            "n_barrels": int(n_barrels),
            "r_min_safe_m": round(float(nom_ev["r_min_safe_m"]), 1),
            "usable_window_nominal_m": round(float(nom_ev["usable_window_m"]), 1),
            "usable_window_p5_m": round(float(win_p5), 1),
            "usable_window_p50_m": round(float(win_p50), 1),
            "usable_window_p95_m": round(float(win_p95), 1),
            "window_collapse_prob_pct": round(float(np.mean(win_arr <= 0.0) * 100.0), 1),
            "epistemic_band_40m_pct": round(float(ref_band), 1),
            "reference_range_m": float(stations[ref_idx]),
            "station_pressure_pa": round(float(station_p_pa), 1),
        },
        "curves": curve_rows,
        "window_samples_m": [round(float(x), 2) for x in win_arr],
        "parameter_specs": EPISTEMIC_PARAMETER_SPECS,
    }


def run_cuas_morris_screening(
    *,
    ammo_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    drone_class: str = "fpv_cfrp",
    n_barrels: int = 4,
    aim_error_mrad: float = 2.0,
    dead_time_s: float = 0.141,
    a_unmod_mps2: float = 5.0,
    drone_speed_mps: float = 25.0,
    target_range_m: float = 40.0,
    drag_law: str = "walsh_1976",
    site_altitude_m: float = 0.0,
    n_trajectories: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    """Morris elementary-effects global screening across 6 epistemic factors (`MC-F19`)."""
    rng = np.random.default_rng(int(seed))
    base = _resolve_ammo_and_target_baseline(ammo_preset, choke, drone_class)
    station_p_pa = tropospheric_pressure_pa(site_altitude_m)
    stations = (15.0, 25.0, 35.0, float(target_range_m), 55.0, 70.0)
    ref_pos = sorted(stations).index(float(target_range_m))

    d_dim = len(EPISTEMIC_PARAMETER_SPECS)
    p_levels = 4
    delta_u = p_levels / (2.0 * (p_levels - 1))  # 2/3 standard Morris step
    grid_vals = np.linspace(0.0, 1.0 - delta_u, p_levels - 2)
    lo = np.array([s["min"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)
    hi = np.array([s["max"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)

    def _eval_u(u_vec: np.ndarray) -> tuple[float, float]:
        th = lo + np.clip(u_vec, 0.0, 1.0) * (hi - lo)
        ev = _eval_epistemic_sample(
            base,
            tuple(sorted(stations)),
            n_barrels=n_barrels,
            aim_error_mrad=aim_error_mrad,
            dead_time_s=dead_time_s,
            a_unmod_mps2=a_unmod_mps2,
            drone_speed_mps=drone_speed_mps,
            drone_altitude_m=15.0,
            warhead_blast_radius_m=8.0,
            pk_threshold=0.50,
            drag_law=drag_law,
            station_p_pa=station_p_pa,
            temp_c=15.0,
            kd_mult=float(th[0]),
            sig_scale=float(th[1]),
            av_scale=float(th[2]),
            aim_scale=float(th[3]),
            v50_scale=float(th[4]),
            dead_scale=float(th[5]),
        )
        return float(ev["pk_shared"][ref_pos]), float(ev["usable_window_m"])

    ee_pk: list[list[float]] = [[] for _ in range(d_dim)]
    ee_win: list[list[float]] = [[] for _ in range(d_dim)]
    n_traj = max(8, int(n_trajectories))

    for _ in range(n_traj):
        u_cur = rng.choice(grid_vals, size=d_dim)
        signs = rng.choice([-1.0, 1.0], size=d_dim)
        # Keep perturbed point inside [0, 1]^d
        for j in range(d_dim):
            if u_cur[j] + signs[j] * delta_u > 1.0 or u_cur[j] + signs[j] * delta_u < 0.0:
                signs[j] = 1.0 if u_cur[j] + delta_u <= 1.0 else -1.0
        order = rng.permutation(d_dim)
        pk_cur, win_cur = _eval_u(u_cur)
        for j in order:
            u_next = u_cur.copy()
            step_j = signs[j] * delta_u
            if not (0.0 <= u_next[j] + step_j <= 1.0):
                step_j = -step_j
            u_next[j] += step_j
            pk_next, win_next = _eval_u(u_next)
            ee_pk[j].append((pk_next - pk_cur) / step_j)
            ee_win[j].append((win_next - win_cur) / step_j)
            u_cur, pk_cur, win_cur = u_next, pk_next, win_next

    rows: list[dict[str, Any]] = []
    for j, spec in enumerate(EPISTEMIC_PARAMETER_SPECS):
        arr_p = np.asarray(ee_pk[j], dtype=np.float64)
        arr_w = np.asarray(ee_win[j], dtype=np.float64)
        mu_star = float(np.mean(np.abs(arr_p)))
        sig_p = float(np.std(arr_p, ddof=1)) if len(arr_p) > 1 else 0.0
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "grade": spec["grade"],
            "source_id": spec["source_id"],
            "mu_star_pk_pct": round(mu_star, 2),
            "mu_signed_pk_pct": round(float(np.mean(arr_p)), 2),
            "sigma_pk_pct": round(sig_p, 2),
            "interaction_ratio": round(sig_p / max(mu_star, 1e-6), 2),
            "mu_star_window_m": round(float(np.mean(np.abs(arr_w))), 2),
        })
    rows.sort(key=lambda r: r["mu_star_pk_pct"], reverse=True)
    return {
        "summary": {
            "n_trajectories": n_traj,
            "target_range_m": float(target_range_m),
            "top_driver": rows[0]["label"],
            "top_mu_star_pk_pct": rows[0]["mu_star_pk_pct"],
            "second_driver": rows[1]["label"],
        },
        "factors": rows,
    }


def run_cuas_sobol_sensitivity(
    *,
    ammo_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    drone_class: str = "fpv_cfrp",
    n_barrels: int = 4,
    aim_error_mrad: float = 2.0,
    dead_time_s: float = 0.141,
    a_unmod_mps2: float = 5.0,
    drone_speed_mps: float = 25.0,
    target_range_m: float = 40.0,
    drag_law: str = "walsh_1976",
    site_altitude_m: float = 0.0,
    n_base: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Saltelli / Jansen Sobol first-order S1 & total-effect ST sensitivity indices (`MC-F18`)."""
    base = _resolve_ammo_and_target_baseline(ammo_preset, choke, drone_class)
    station_p_pa = tropospheric_pressure_pa(site_altitude_m)
    stations = tuple(sorted({15.0, 25.0, 35.0, float(target_range_m), 55.0, 70.0}))
    ref_pos = stations.index(float(target_range_m))

    d_dim = len(EPISTEMIC_PARAMETER_SPECS)
    lo = np.array([s["min"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)
    hi = np.array([s["max"] for s in EPISTEMIC_PARAMETER_SPECS], dtype=np.float64)

    # Power-of-two Sobol sequence of dimension 2*d -> split cleanly into independent A & B blocks
    m_pow = max(5, min(10, int(round(math.log2(max(32, int(n_base)))))))
    qmc = sp_stats.qmc.Sobol(d=2 * d_dim, scramble=True, seed=int(seed))
    u_full = qmc.random_base2(m=m_pow)
    n_pts = u_full.shape[0]
    a_u = u_full[:, :d_dim]
    b_u = u_full[:, d_dim:]

    def _eval_batch(u_mat: np.ndarray) -> np.ndarray:
        th_mat = lo + u_mat * (hi - lo)
        ys = np.empty(u_mat.shape[0], dtype=np.float64)
        for i_r in range(u_mat.shape[0]):
            th = th_mat[i_r]
            ev = _eval_epistemic_sample(
                base,
                stations,
                n_barrels=n_barrels,
                aim_error_mrad=aim_error_mrad,
                dead_time_s=dead_time_s,
                a_unmod_mps2=a_unmod_mps2,
                drone_speed_mps=drone_speed_mps,
                drone_altitude_m=15.0,
                warhead_blast_radius_m=8.0,
                pk_threshold=0.50,
                drag_law=drag_law,
                station_p_pa=station_p_pa,
                temp_c=15.0,
                kd_mult=float(th[0]),
                sig_scale=float(th[1]),
                av_scale=float(th[2]),
                aim_scale=float(th[3]),
                v50_scale=float(th[4]),
                dead_scale=float(th[5]),
            )
            ys[i_r] = float(ev["pk_shared"][ref_pos])
        return ys

    y_a = _eval_batch(a_u)
    y_b = _eval_batch(b_u)
    y_all = np.concatenate([y_a, y_b])
    v_tot = max(float(np.var(y_all, ddof=1)), 1e-9)

    indices: list[dict[str, Any]] = []
    for j, spec in enumerate(EPISTEMIC_PARAMETER_SPECS):
        ab_j = a_u.copy()
        ab_j[:, j] = b_u[:, j]
        y_ab_j = _eval_batch(ab_j)

        # Saltelli S1 estimator & Jansen ST estimator
        s1_raw = float(np.mean(y_b * (y_ab_j - y_a)) / v_tot)
        st_raw = float(np.mean((y_a - y_ab_j) ** 2) / (2.0 * v_tot))
        s1 = max(0.0, min(1.0, s1_raw))
        st = max(s1, min(1.0, st_raw))
        # Analytic standard error estimate across n_pts blocks
        s1_terms = y_b * (y_ab_j - y_a) / v_tot
        st_terms = ((y_a - y_ab_j) ** 2) / (2.0 * v_tot)
        indices.append({
            "key": spec["key"],
            "label": spec["label"],
            "grade": spec["grade"],
            "source_id": spec["source_id"],
            "S1": round(s1, 3),
            "ST": round(st, 3),
            "interaction_ST_minus_S1": round(max(0.0, st - s1), 3),
            "S1_ci95_half": round(1.96 * float(np.std(s1_terms, ddof=1) / math.sqrt(n_pts)), 3),
            "ST_ci95_half": round(1.96 * float(np.std(st_terms, ddof=1) / math.sqrt(n_pts)), 3),
            "priority_note": spec["note"],
        })

    indices.sort(key=lambda r: r["ST"], reverse=True)
    top = indices[0]
    return {
        "summary": {
            "n_base_samples": n_pts,
            "total_evaluations": n_pts * (d_dim + 2),
            "target_range_m": float(target_range_m),
            "mean_pk_pct": round(float(np.mean(y_all)), 2),
            "std_pk_pct": round(float(np.std(y_all, ddof=1)), 2),
            "top_sobol_driver": top["label"],
            "top_ST": top["ST"],
            "measurement_priority": (
                f"Highest epistemic variance driver is {top['label']} "
                f"(ST={top['ST']:.2f}, grade {top['grade']}, {top['source_id']})."
            ),
        },
        "indices": indices,
    }


# -------------------------------------------------------------------------
# 2. ENGAGEMENT DOCTRINE LAB (SALVO vs RIPPLE vs WILKENING SLS vs WHISTLER SAS)
# -------------------------------------------------------------------------

def compare_cuas_engagement_doctrines(
    *,
    ammo_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    drone_class: str = "fpv_cfrp",
    n_barrels: int = 4,
    barrels_per_burst: int = 2,
    magazine_shells: int = 12,
    r_start_m: float = 55.0,
    drone_speed_mps: float = 25.0,
    drone_altitude_m: float = 15.0,
    warhead_blast_radius_m: float = 8.0,
    salvo_interval_s: float = 0.45,
    bda_delay_s: float = 0.40,
    bda_false_neg_prob: float = 0.08,
    aim_bias_sigma_mrad: float = 2.0,
    aim_jitter_sigma_mrad: float = 0.8,
    ar1_rho: float = 0.65,
    obs_noise_m: float = 0.05,
    n_trials: int = 1200,
    seed: int = 42,
    drag_law: str = "walsh_1976",
    site_altitude_m: float = 0.0,
) -> dict[str, Any]:
    """Compare Simultaneous Salvo, Ripple Salvo, Wilkening SLS, & Whistler SAS over closing range (`M12`)."""
    base = _resolve_ammo_and_target_baseline(ammo_preset, choke, drone_class)
    station_p_pa = tropospheric_pressure_pa(site_altitude_m)
    v_dr = max(3.0, float(drone_speed_mps))
    x_wreck = v_dr * math.sqrt(2.0 * max(1.0, float(drone_altitude_m)) / 9.80665)
    r_min_safe = x_wreck + max(0.0, float(warhead_blast_radius_m))
    r_st = max(r_min_safe + 4.0, float(r_start_m))

    n_b = max(1, int(n_barrels))
    bpb = max(1, min(n_b, int(barrels_per_burst)))
    mag = max(n_b, int(magazine_shells))
    n_tr = max(200, int(n_trials))
    t_window = max(0.1, (r_st - r_min_safe) / v_dr)

    # Pre-build fine range -> (lam0, sigma_p) lookup across [r_min_safe, r_st]
    r_lut = np.linspace(max(2.0, r_min_safe), r_st + 2.0, 25)
    traj_lut = _single_pass_rk4_stations(
        base["v0_mps"],
        base["d_mm"],
        base["rho_p_kg_m3"],
        base["m_g"],
        tuple(float(x) for x in r_lut),
        drag_law=drag_law,
        k_d=1.0,
        temp_c=15.0,
        pressure_pa=station_p_pa,
    )
    lam_lut = np.empty(len(r_lut), dtype=np.float64)
    sigp_lut = np.empty(len(r_lut), dtype=np.float64)
    for idx_l, (rm, vr, _) in enumerate(traj_lut):
        pe = velocity_ogive_peff(vr, base["v50_mps"], spread_s_v=0.10)
        sp = max(0.02, base["sigma_ang_rad"] * rm)
        sigp_lut[idx_l] = sp
        lam_lut[idx_l] = (base["n_pellets"] / (2.0 * math.pi * (sp ** 2))) * base["a_v_m2"] * pe

    def _eval_r(r_val: float) -> tuple[float, float]:
        return float(np.interp(r_val, r_lut, lam_lut)), float(np.interp(r_val, r_lut, sigp_lut))

    # Nominal SSPK at mid-window for Wilkening wave-sizing (HP-F39)
    r_mid = 0.5 * (r_st + r_min_safe)
    lam_mid, sigp_mid = _eval_r(r_mid)
    sa_mid = (math.hypot(aim_bias_sigma_mrad, aim_jitter_sigma_mrad) / 1000.0) * r_mid
    k_sspk_nom = max(0.05, min(0.90, pk_single_shot_closed_form(lam_mid, sigp_mid, sa_mid)))
    w_s = 1
    for s_cand in range(1, min(n_b, 8) + 1):
        if wilkening_ns(s_cand, k_sspk_nom) <= float(mag):
            w_s = s_cand

    # Common random numbers across all 4 doctrines
    rng = np.random.default_rng(int(seed))
    bias_unit = rng.normal(0.0, float(aim_bias_sigma_mrad) / 1000.0, size=(n_tr, 2))
    ar_innov = rng.normal(0.0, 1.0, size=(n_tr, 24, 2))
    kill_u = rng.random(size=(n_tr, 24))
    bda_u = rng.random(size=(n_tr, 24))

    def _run_doctrine(name: str, label: str, fire_sched: list[tuple[float, int]], adjust_rule: bool = False) -> dict[str, Any]:
        alive = np.ones(n_tr, dtype=bool)
        kill_r = np.zeros(n_tr, dtype=np.float64)
        rounds_used = np.zeros(n_tr, dtype=np.float64)
        bias_corr = np.zeros((n_tr, 2), dtype=np.float64)  # mrad bias correction estimate

        shot_curve: list[dict[str, float]] = []
        rho_val = max(0.0, min(0.95, float(ar1_rho)))
        jit_rad = float(aim_jitter_sigma_mrad) / 1000.0
        cur_ar = ar_innov[:, 0, :] * jit_rad

        for step_k, (t_fire, n_sh) in enumerate(fire_sched):
            if step_k > 0:
                cur_ar = rho_val * cur_ar + math.sqrt(max(0.0, 1.0 - rho_val * rho_val)) * jit_rad * ar_innov[:, min(step_k, 23), :]
            r_fire = r_st - v_dr * t_fire
            if r_fire < r_min_safe or not np.any(alive):
                continue
            lam_1, sig_p = _eval_r(r_fire)

            # Fire only if trial still believes target is alive (or BDA false negative)
            will_shoot = alive.copy()
            if adjust_rule and step_k > 0:
                pass  # sequential adjust always tracks alive targets cleanly

            # Angular miss vector = systematic bias - learned correction + correlated jitter
            ang_err = bias_unit - bias_corr + cur_ar
            miss_m = ang_err * r_fire
            r2 = np.sum(miss_m * miss_m, axis=1)
            lam_shot = (float(n_sh) * lam_1) * np.exp(-r2 / (2.0 * (sig_p ** 2)))
            p_k_trial = 1.0 - np.exp(-lam_shot)

            rounds_used[will_shoot] += float(n_sh)
            killed_now = will_shoot & (kill_u[:, min(step_k, 23)] < p_k_trial)
            alive[killed_now] = False
            kill_r[killed_now] = r_fire

            if adjust_rule:
                # Whistler HP-F37 rule: update aimpoint bias correction using noisy angular miss observation
                obs_ang = ang_err + rng.normal(0.0, float(obs_noise_m) / max(r_fire, 1.0), size=(n_tr, 2))
                bias_corr = bias_corr + (obs_ang / float(step_k + 1))

            # Optional imperfect BDA extra wasted round check (handled implicitly by surviving state; log cumulative Pk)
            if float(bda_false_neg_prob) > 0.0 and "wilkening" in name:
                false_neg = (~alive) & (~killed_now) & (bda_u[:, min(step_k, 23)] < float(bda_false_neg_prob))
                rounds_used[false_neg] += float(min(n_sh, 1))

            shot_curve.append({
                "doctrine": label,
                "shot_idx": step_k + 1,
                "time_s": round(float(t_fire), 3),
                "range_m": round(float(r_fire), 2),
                "cum_pk_pct": round(float(np.mean(~alive) * 100.0), 2),
                "mean_rounds": round(float(np.mean(rounds_used)), 2),
            })

        n_killed = int(np.sum(~alive))
        pk_est = n_killed / n_tr
        ci_lo, ci_hi = compute_wilson_ci(n_killed, n_tr, 0.05)
        mean_rds = float(np.mean(rounds_used))
        rds_per_kill = mean_rds / max(pk_est, 1e-3)
        mean_kr = float(np.mean(kill_r[kill_r > 0.0])) if n_killed > 0 else 0.0
        eff_score = (pk_est * 100.0) / max(mean_rds, 1.0)
        return {
            "row": {
                "doctrine_id": name,
                "doctrine_label": label,
                "cum_pk_pct": round(pk_est * 100.0, 1),
                "wilson_ci95_pct": [round(ci_lo * 100.0, 1), round(ci_hi * 100.0, 1)],
                "leaker_rate_pct": round((1.0 - pk_est) * 100.0, 1),
                "mean_rounds_expended": round(mean_rds, 2),
                "mean_rounds_per_kill": round(rds_per_kill, 2),
                "mean_kill_range_m": round(mean_kr, 1),
                "n_opportunities": len(shot_curve),
                "efficiency_pk_per_round": round(eff_score, 1),
            },
            "curve": shot_curve,
        }

    # Build fire timelines constrained by window & magazine capacity
    def _build_sched(cadence_s: float, burst_sz: int, first_burst_sz: int | None = None) -> list[tuple[float, int]]:
        sched: list[tuple[float, int]] = []
        t_cur = 0.0
        rem = mag
        step_i = 0
        while t_cur <= t_window + 1e-9 and rem > 0 and len(sched) < 18:
            sh = min(rem, first_burst_sz if (step_i == 0 and first_burst_sz is not None) else burst_sz)
            sched.append((t_cur, int(sh)))
            rem -= sh
            t_cur += max(0.08, float(cadence_s))
            step_i += 1
        return sched

    sim_sched = _build_sched(salvo_interval_s, n_b)
    rip_cad = max(0.08, float(salvo_interval_s) / max(1, n_b // bpb))
    rip_sched = _build_sched(rip_cad, bpb)
    sls_cad = max(float(salvo_interval_s), float(bda_delay_s) + 0.12)
    sls_sched = _build_sched(sls_cad, max(1, bpb), first_burst_sz=w_s)
    sas_sched = _build_sched(max(0.18, float(bda_delay_s) + 0.08), 1)

    res_list = [
        _run_doctrine("simultaneous_salvo", "Simultaneous Full Salvo (Shared Bias)", sim_sched, adjust_rule=False),
        _run_doctrine("ripple_salvo", f"Staggered {bpb}-Bbl Ripple Salvo", rip_sched, adjust_rule=False),
        _run_doctrine("shoot_look_shoot_wilkening", f"Wilkening SLS (s1={w_s}, BDA={bda_delay_s:.2f}s)", sls_sched, adjust_rule=False),
        _run_doctrine("shoot_adjust_shoot_whistler", "Whistler Shoot-Adjust-Shoot (Bias Corr.)", sas_sched, adjust_rule=True),
    ]

    rows = [x["row"] for x in res_list]
    curves = [pt for x in res_list for pt in x["curve"]]
    best_pk = max(rows, key=lambda r: r["cum_pk_pct"])
    best_eff = max(rows, key=lambda r: r["efficiency_pk_per_round"])

    # Glazebrook-Washburn Gittins Priority Illustration across mixed raid targets
    gittins_demo = [
        {
            "target": "Striker FPV (High Value v=4.0, p_alive=0.20, q=0.50)",
            "v": 4.0,
            "p_alive": 0.20,
            "q_miss": 0.50,
            "myopic_score": round(0.20 * 4.0 * 0.50, 3),
            "gittins_index": round(gittins_sls_index(0.90, 0.50, 4.0, 0.20), 3),
        },
        {
            "target": "Nominal Striker (Confirmed v=1.0, p_alive=1.00, q=0.50)",
            "v": 1.0,
            "p_alive": 1.00,
            "q_miss": 0.50,
            "myopic_score": round(1.00 * 1.0 * 0.50, 3),
            "gittins_index": round(gittins_sls_index(0.90, 0.50, 1.0, 1.00), 3),
        },
        {
            "target": "Low-Value Decoy (v=0.4, p_alive=0.90, q=0.35)",
            "v": 0.4,
            "p_alive": 0.90,
            "q_miss": 0.35,
            "myopic_score": round(0.90 * 0.4 * 0.65, 3),
            "gittins_index": round(gittins_sls_index(0.90, 0.35, 0.4, 0.90), 3),
        },
    ]

    return {
        "summary": {
            "r_start_m": round(r_st, 1),
            "r_min_safe_m": round(r_min_safe, 1),
            "closing_window_s": round(t_window, 2),
            "wilkening_first_wave_shots": int(w_s),
            "nominal_midwindow_sspk": round(k_sspk_nom, 3),
            "best_pk_doctrine": best_pk["doctrine_label"],
            "best_pk_pct": best_pk["cum_pk_pct"],
            "best_efficiency_doctrine": best_eff["doctrine_label"],
            "best_efficiency_score": best_eff["efficiency_pk_per_round"],
        },
        "doctrines": rows,
        "closing_curves": curves,
        "gittins_ranking": gittins_demo,
    }


# -------------------------------------------------------------------------
# 3. REPRODUCIBLE PYDANTIC SCENARIO MANIFEST & PROVENANCE BUNDLE (M01/M16)
# -------------------------------------------------------------------------

class ParameterProvenanceItem(BaseModel):
    name: str
    value: Any
    unit: str = "-"
    grade: str = "V3"
    source_id: str = "S70"
    bounds: str = ""
    editable: bool = True
    note: str = ""


class CuasScenarioManifest(BaseModel):
    schema_version: str = "2.1.0"
    app_version: str = __version__
    scenario_name: str = "C-UAS Multi-Shotgun Engineering Scenario"
    seed: int = 42
    created_utc: str = ""
    site_altitude_m: float = 0.0
    weather: dict[str, Any] = Field(default_factory=dict)
    array_config: dict[str, Any] = Field(default_factory=dict)
    target_config: dict[str, Any] = Field(default_factory=dict)
    fcs_config: dict[str, Any] = Field(default_factory=dict)
    safety_config: dict[str, Any] = Field(default_factory=dict)
    active_formula_ids: list[str] = Field(default_factory=list)
    active_source_ids: list[str] = Field(default_factory=list)
    parameter_grades: list[ParameterProvenanceItem] = Field(default_factory=list)
    worst_verification_grade: str = "V3"
    weakest_components: list[str] = Field(default_factory=list)
    manifest_sha256: str = ""
    scope_disclaimer: str = (
        "Engineering analysis aid only (V1-V4 weakest-link graded). Not validated as live weapon-control software "
        "and does not replace official Range Safety Officers or AR 385-63 / UK JSP 403 / RCC 321-23."
    )


def build_cuas_scenario_manifest(
    *,
    scenario_name: str = "C-UAS Multi-Shotgun Engineering Scenario",
    seed: int = 42,
    site_altitude_m: float = 0.0,
    weather: dict[str, Any] | None = None,
    array_config: dict[str, Any] | None = None,
    target_config: dict[str, Any] | None = None,
    fcs_config: dict[str, Any] | None = None,
    safety_config: dict[str, Any] | None = None,
    calibrated: bool = False,
) -> dict[str, Any]:
    """Build deterministic Pydantic-validated Scenario & Provenance Manifest (`M01` / `M16`)."""
    arr = dict(array_config or {})
    tgt = dict(target_config or {})
    fcs = dict(fcs_config or {})
    saf = dict(safety_config or {})
    wx = dict(weather or {"temp_c": 15.0, "rh_pct": 0.0})

    param_items: list[ParameterProvenanceItem] = [
        ParameterProvenanceItem(
            name="site_altitude_m",
            value=round(float(site_altitude_m), 1),
            unit="m ASL",
            grade="V1",
            source_id="S04",
            bounds="0..3500 m",
            editable=True,
            note="Tropospheric barometric pressure (`AT-F01`) feeding CIPM-2007 air density.",
        ),
        ParameterProvenanceItem(
            name="drag_model",
            value=str(arr.get("drag_model", "walsh_1976")),
            unit="-",
            grade="V1" if str(arr.get("drag_model", "")).lower() in ("walsh_1976", "nol_1972", "loth_2008") else "V2",
            source_id="S71",
            bounds="walsh_1976 | nol_1972 | loth_2008 | compton_1996",
            editable=True,
            note="Sphere drag correlation (`EB-T01`, `EB-F12`).",
        ),
        ParameterProvenanceItem(
            name="drag_multiplier_kd",
            value=round(float(arr.get("drag_multiplier_kd", 1.0)), 3),
            unit="-",
            grade="V2",
            source_id="S71",
            bounds="0.90..1.35",
            editable=True,
            note="Pellet deformation multiplier (+10-30% for soft lead).",
        ),
        ParameterProvenanceItem(
            name="aim_error_mrad",
            value=round(float(arr.get("aim_error_mrad", fcs.get("aim_error_mrad", 2.0))), 2),
            unit="mrad",
            grade="V4" if not calibrated else "V2",
            source_id="S85",
            bounds="0.5..15.0 mrad",
            editable=True,
            note="Shared tracker + prediction + mount boresight aim error.",
        ),
        ParameterProvenanceItem(
            name="target_vulnerability",
            value=str(tgt.get("drone_model", tgt.get("drone_class", "fpv_cfrp"))),
            unit="-",
            grade="V3",
            source_id="S82",
            bounds="Wen (2000) / V50 ogive",
            editable=True,
            note="Composite FRP / rotor critical-component vulnerability threshold.",
        ),
        ParameterProvenanceItem(
            name="dead_time_s",
            value=round(float(fcs.get("dead_time_s", 0.141)), 4),
            unit="s",
            grade="V3",
            source_id="S85",
            bounds="0.03..0.30 s",
            editable=True,
            note="End-to-end sensor + ATR + servo + lock serial latency (`FC-T04`).",
        ),
    ]

    comp_grades = {it.name: it.grade for it in param_items}
    grade_res = resolve_verification_grade(comp_grades, calibrated=calibrated)

    core_payload = {
        "schema_version": "2.1.0",
        "app_version": __version__,
        "scenario_name": str(scenario_name),
        "seed": int(seed),
        "site_altitude_m": float(site_altitude_m),
        "weather": wx,
        "array_config": arr,
        "target_config": tgt,
        "fcs_config": fcs,
        "safety_config": saf,
        "active_formula_ids": ["AT-F01", "AT-F04", "AT-F05", "AT-F07", "EB-F01", "EB-F12", "WB-F01", "WB-F02", "FC-F12", "HP-F35", "HP-F37", "HP-F39", "MC-F18", "MC-F19", "MC-F21"],
        "active_source_ids": sorted({it.source_id for it in param_items} | {"S04", "S66", "S70", "S71", "S82", "S85", "S93"}),
        "parameter_grades": [it.model_dump() for it in param_items],
        "worst_verification_grade": str(grade_res["grade"]),
        "weakest_components": list(grade_res["weakest_components"]),
    }
    canon_bytes = json.dumps(core_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canon_bytes).hexdigest()

    manifest = CuasScenarioManifest(
        **core_payload,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        manifest_sha256=digest,
    )
    return manifest.model_dump()


def serialize_manifest_json(manifest: dict[str, Any]) -> str:
    """Serialize Scenario Manifest dictionary as formatted JSON."""
    return json.dumps(manifest, indent=2, sort_keys=False)


def serialize_manifest_yaml(manifest: dict[str, Any]) -> str:
    """Serialize Scenario Manifest dictionary as YAML."""
    return yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)


def load_cuas_scenario_manifest(raw_text: str) -> tuple[bool, dict[str, Any] | None, str]:
    """Parse JSON or YAML Scenario Manifest, validate Pydantic schema & verify SHA-256."""
    text = str(raw_text or "").strip()
    if not text:
        return False, None, "Manifest content is empty."
    try:
        parsed = json.loads(text) if text.startswith("{") else yaml.safe_load(text)
    except Exception as exc:
        return False, None, f"Parse error (JSON/YAML): {exc}"
    if not isinstance(parsed, dict):
        return False, None, "Manifest root must be a mapping/object."
    try:
        obj = CuasScenarioManifest(**parsed)
    except Exception as exc:
        return False, None, f"Schema validation error: {exc}"

    check_dict = obj.model_dump()
    supplied_hash = str(check_dict.pop("manifest_sha256", "") or "")
    check_dict.pop("created_utc", None)
    check_dict.pop("scope_disclaimer", None)
    recomputed = hashlib.sha256(
        json.dumps(check_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if supplied_hash and supplied_hash != recomputed:
        return False, obj.model_dump(), f"SHA-256 mismatch: file has {supplied_hash[:12]}, recomputed {recomputed[:12]}."
    return True, obj.model_dump(), f"Manifest v{obj.schema_version} validated (SHA-256 {recomputed[:12]}...)."


# -------------------------------------------------------------------------
# 4. PURE-SCIPY MATERN-5/2 GP SURROGATE & PARETO EXPLORER (M17)
# -------------------------------------------------------------------------

class CuasGaussianProcessSurrogate:
    """Pure NumPy/SciPy Matern-5/2 Gaussian Process emulator with closed-form LOO-CV."""

    def __init__(self, jitter: float = 1e-6) -> None:
        self.jitter = float(jitter)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.lengthscales: np.ndarray | None = None
        self.sigma_f: float = 1.0
        self.noise: float = 1e-4
        self.chol_l: Any = None
        self.alpha: np.ndarray | None = None
        self.loo_r2: float = 0.0
        self.loo_rmse: float = 0.0

    @staticmethod
    def _matern52(x1: np.ndarray, x2: np.ndarray, ls: np.ndarray, sig_f: float) -> np.ndarray:
        scaled_1 = x1 / ls[None, :]
        scaled_2 = x2 / ls[None, :]
        d2 = np.sum((scaled_1[:, None, :] - scaled_2[None, :, :]) ** 2, axis=2)
        r = np.sqrt(np.maximum(d2, 0.0))
        s5r = math.sqrt(5.0) * r
        return (sig_f ** 2) * (1.0 + s5r + (5.0 / 3.0) * (r * r)) * np.exp(-s5r)

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> CuasGaussianProcessSurrogate:
        x = np.asarray(x_raw, dtype=np.float64)
        y = np.asarray(y_raw, dtype=np.float64)
        n, d = x.shape

        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.maximum(np.std(x, axis=0), 1e-6)
        self.y_mean = float(np.mean(y))
        self.y_std = max(float(np.std(y)), 1e-6)

        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std
        self.x_train = xs
        self.y_train = ys

        def _neg_lml(theta: np.ndarray) -> float:
            ls = np.exp(theta[:d])
            sf = math.exp(float(theta[d]))
            ns = math.exp(float(theta[d + 1])) + self.jitter
            k_mat = self._matern52(xs, xs, ls, sf) + ns * np.eye(n)
            try:
                c_fac = linalg.cho_factor(k_mat, lower=True)
                alp = linalg.cho_solve(c_fac, ys)
                l_diag = np.diag(c_fac[0])
                logdet = 2.0 * float(np.sum(np.log(np.maximum(l_diag, 1e-12))))
                return 0.5 * float(ys @ alp) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)
            except np.linalg.LinAlgError:
                return 1e12

        theta0 = np.concatenate([np.zeros(d), [0.0, math.log(1e-3)]])
        bounds = [(-2.0, 2.5)] * d + [(-2.0, 2.0), (-11.0, -1.0)]
        res = optimize.minimize(_neg_lml, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
        th_opt = res.x if res.success or res.fun < _neg_lml(theta0) else theta0

        self.lengthscales = np.exp(th_opt[:d])
        self.sigma_f = math.exp(float(th_opt[d]))
        self.noise = math.exp(float(th_opt[d + 1])) + self.jitter

        k_opt = self._matern52(xs, xs, self.lengthscales, self.sigma_f) + self.noise * np.eye(n)
        self.chol_l = linalg.cho_factor(k_opt, lower=True)
        self.alpha = linalg.cho_solve(self.chol_l, ys)

        # Exact closed-form LOO-CV (Rasmussen & Williams Eq. 5.12)
        k_inv = linalg.cho_solve(self.chol_l, np.eye(n))
        diag_kinv = np.maximum(np.diag(k_inv), 1e-9)
        y_loo_s = ys - self.alpha / diag_kinv
        y_loo = y_loo_s * self.y_std + self.y_mean
        err_loo = y - y_loo
        ss_res = float(np.sum(err_loo ** 2))
        ss_tot = max(float(np.sum((y - self.y_mean) ** 2)), 1e-12)
        self.loo_r2 = float(1.0 - ss_res / ss_tot)
        self.loo_rmse = float(math.sqrt(ss_res / max(1, n)))
        return self

    def predict(self, x_query_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.x_train is not None and self.alpha is not None and self.lengthscales is not None
        xq = (np.asarray(x_query_raw, dtype=np.float64) - self.x_mean) / self.x_std
        k_star = self._matern52(self.x_train, xq, self.lengthscales, self.sigma_f)
        mu_s = k_star.T @ self.alpha
        v_mat = linalg.solve_triangular(self.chol_l[0], k_star, lower=True)
        var_s = np.maximum((self.sigma_f ** 2) - np.sum(v_mat * v_mat, axis=0), 0.0)
        return mu_s * self.y_std + self.y_mean, np.sqrt(var_s) * self.y_std


def _eval_design_vector_physics(
    x_vec: np.ndarray,
    *,
    drone_class: str,
    drone_speed_mps: float,
    drone_altitude_m: float,
    aim_error_mrad: float,
    site_altitude_m: float,
    drag_law: str,
) -> dict[str, float]:
    """Evaluate exact closed-form physics for a continuous 5D C-UAS turret design vector."""
    n_bbl = int(round(float(x_vec[0])))
    p40 = float(np.clip(x_vec[1], 0.40, 0.80))
    d_mm = float(np.clip(x_vec[2], 2.4, 6.1))
    rho_g_cm3 = float(np.clip(x_vec[3], 7.85, 18.0))
    ripple_ms = float(np.clip(x_vec[4], 0.0, 25.0))

    rho_p = rho_g_cm3 * 1000.0
    payload_g = 35.4
    v0 = 405.0
    m_p_kg = rho_p * math.pi * ((d_mm / 1000.0) ** 3) / 6.0
    m_p_g = m_p_kg * 1000.0
    n_pellets = max(4.0, payload_g / max(m_p_g, 1e-4))
    sigma_ang_rad = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(1.0 - p40))

    if "plastic" in str(drone_class).lower():
        a_v_m2 = 0.0080
        v50_mps = math.sqrt(2.0 * 5.0 / max(m_p_kg, 1e-6))
    else:
        a_v_m2 = 0.0055
        wen = wen_2000_frp_ballistic_limit(m_p_kg, d_mm / 1000.0, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
        v50_mps = float(wen["v_bl_mps"])

    station_p = tropospheric_pressure_pa(site_altitude_m)
    stations = (15.0, 25.0, 35.0, 40.0, 50.0, 60.0, 75.0, 85.0)
    traj_pts = _single_pass_rk4_stations(v0, d_mm, rho_p, m_p_g, stations, drag_law=drag_law, k_d=1.0, temp_c=15.0, pressure_pa=station_p)
    x_wreck = max(0.5, float(drone_speed_mps)) * math.sqrt(2.0 * max(1.0, float(drone_altitude_m)) / 9.80665)
    r_min_safe = x_wreck + 8.0

    qualifying: list[float] = []
    pk_40 = 0.0
    for r_m, v_ret, tof_s in traj_pts:
        pe = velocity_ogive_peff(v_ret, v50_mps, spread_s_v=0.10)
        sig_p = max(1e-3, sigma_ang_rad * r_m)
        sig_a = math.hypot((aim_error_mrad / 1000.0) * r_m, 0.5 * 5.0 * ((0.141 + tof_s) ** 2))
        lam0 = (n_pellets / (2.0 * math.pi * (sig_p ** 2))) * a_v_m2 * pe
        br = pk_salvo_brackets(lam0, sig_p, sig_a, n_shots=n_bbl)
        if abs(r_m - 40.0) < 0.5:
            pk_40 = br["pk_shared_lower"] * 100.0
        if br["pk_shared_lower"] >= 0.50:
            qualifying.append(r_m)

    r_max = max(qualifying) if qualifying else 0.0
    usable_w = max(0.0, r_max - r_min_safe)

    saami = compute_saami_recoil_impulse(shot_mass_g=payload_g, wad_mass_g=2.79, propellant_mass_g=2.164, v0_mps=v0, gun_mass_kg=3.18)
    imp_1 = float(saami["impulse_ns"])
    # Peak mount force scales smoothly with simultaneous impulse overlapping within a 12 ms buffer stroke time-constant
    overlap = sum(math.exp(-k * ripple_ms / 12.0) for k in range(n_bbl))
    peak_recoil_kn = max(2.0, (imp_1 / 0.012) * overlap / 1000.0)

    # Smooth continuous per-shell cost ($1.5 lead/steel up to ~$7.2 TSS-18) / salvo Pk
    unit_cost = 1.50 + 0.52 * max(0.0, rho_g_cm3 - 7.85)
    salvo_cost = unit_cost * n_bbl
    cost_per_kill = salvo_cost / max(pk_40 / 100.0, 0.05)

    # First-order high-elevation SDZ range indicator (Journee scaled by density ratio sqrt, SE-F02 / M14)
    sdz_est_m = 100.0 * d_mm * math.sqrt(rho_g_cm3 / 11.2)

    return {
        "n_barrels": float(n_bbl),
        "choke_p40": round(p40, 3),
        "pellet_diam_mm": round(d_mm, 2),
        "pellet_density_g_cm3": round(rho_g_cm3, 2),
        "ripple_delay_ms": round(ripple_ms, 1),
        "usable_window_m": round(usable_w, 2),
        "pk_shared_40m_pct": round(pk_40, 2),
        "peak_recoil_kn": round(peak_recoil_kn, 2),
        "cost_per_kill_usd": round(cost_per_kill, 2),
        "sdz_max_range_m": round(sdz_est_m, 1),
    }


def run_cuas_surrogate_pareto_study(
    *,
    drone_class: str = "fpv_cfrp",
    drone_speed_mps: float = 25.0,
    drone_altitude_m: float = 15.0,
    aim_error_mrad: float = 2.0,
    site_altitude_m: float = 0.0,
    drag_law: str = "walsh_1976",
    n_doe: int = 48,
    n_candidates: int = 360,
    recoil_limit_kn: float = 22.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit Matern-5/2 GP surrogates on Sobol DoE runs and extract Verified Multi-Objective Pareto Front (`M17`)."""
    bounds = np.array(
        [
            [2.0, 12.0],    # n_barrels
            [0.40, 0.80],   # choke p40
            [2.40, 6.10],   # pellet diam mm
            [7.85, 18.0],   # density g/cm3
            [0.0, 25.0],    # ripple delay ms
        ],
        dtype=np.float64,
    )
    n_d = max(24, int(n_doe))
    sampler = sp_stats.qmc.LatinHypercube(d=5, seed=int(seed))
    u_doe = sampler.random(n=n_d)
    x_doe = bounds[:, 0] + u_doe * (bounds[:, 1] - bounds[:, 0])
    x_doe[:, 0] = np.round(x_doe[:, 0])

    doe_rows = [
        _eval_design_vector_physics(
            x_doe[i],
            drone_class=drone_class,
            drone_speed_mps=drone_speed_mps,
            drone_altitude_m=drone_altitude_m,
            aim_error_mrad=aim_error_mrad,
            site_altitude_m=site_altitude_m,
            drag_law=drag_law,
        )
        for i in range(n_d)
    ]

    metrics = ("usable_window_m", "pk_shared_40m_pct", "peak_recoil_kn", "cost_per_kill_usd")
    gps: dict[str, CuasGaussianProcessSurrogate] = {}
    for m_key in metrics:
        y_vec = np.array([r[m_key] for r in doe_rows], dtype=np.float64)
        gps[m_key] = CuasGaussianProcessSurrogate().fit(x_doe, y_vec)

    # Dense Sobol candidate sweep over GP posterior
    n_c = max(128, int(n_candidates))
    cand_qmc = sp_stats.qmc.LatinHypercube(d=5, seed=int(seed) + 101)
    u_cand = cand_qmc.random(n=n_c)
    x_cand = bounds[:, 0] + u_cand * (bounds[:, 1] - bounds[:, 0])
    x_cand[:, 0] = np.round(x_cand[:, 0])

    pred_mu: dict[str, np.ndarray] = {}
    pred_sd: dict[str, np.ndarray] = {}
    for m_key, gp in gps.items():
        mu_a, sd_a = gp.predict(x_cand)
        pred_mu[m_key] = mu_a
        pred_sd[m_key] = sd_a

    candidates: list[dict[str, Any]] = []
    for idx in range(n_c):
        pk_p = float(np.clip(pred_mu["pk_shared_40m_pct"][idx], 0.0, 100.0))
        win_p = max(0.0, float(pred_mu["usable_window_m"][idx]))
        rec_p = max(1.0, float(pred_mu["peak_recoil_kn"][idx]))
        cpk_p = max(1.0, float(pred_mu["cost_per_kill_usd"][idx]))
        candidates.append({
            "cand_id": idx,
            "n_barrels": int(x_cand[idx, 0]),
            "choke_p40": round(float(x_cand[idx, 1]), 3),
            "pellet_diam_mm": round(float(x_cand[idx, 2]), 2),
            "pellet_density_g_cm3": round(float(x_cand[idx, 3]), 2),
            "ripple_delay_ms": round(float(x_cand[idx, 4]), 1),
            "pred_usable_window_m": round(win_p, 2),
            "pred_usable_window_std_m": round(float(pred_sd["usable_window_m"][idx]), 2),
            "pred_pk_40m_pct": round(pk_p, 1),
            "pred_pk_40m_std_pct": round(float(pred_sd["pk_shared_40m_pct"][idx]), 2),
            "pred_peak_recoil_kn": round(rec_p, 2),
            "pred_cost_per_kill_usd": round(cpk_p, 2),
            "feasible_recoil": bool(rec_p <= float(recoil_limit_kn)),
            "is_pareto": False,
            "vec": [float(v) for v in x_cand[idx]],
        })

    # Extract non-dominated Pareto front: maximize (win_p, pk_p), minimize (rec_p, cpk_p)
    feasible_idxs = [i for i, c in enumerate(candidates) if c["feasible_recoil"]] or list(range(len(candidates)))
    pareto_front: list[dict[str, Any]] = []
    for i in feasible_idxs:
        ci = candidates[i]
        dominated = False
        for j in feasible_idxs:
            if i == j:
                continue
            cj = candidates[j]
            ge_all = (
                cj["pred_usable_window_m"] >= ci["pred_usable_window_m"]
                and cj["pred_pk_40m_pct"] >= ci["pred_pk_40m_pct"]
                and cj["pred_peak_recoil_kn"] <= ci["pred_peak_recoil_kn"]
                and cj["pred_cost_per_kill_usd"] <= ci["pred_cost_per_kill_usd"]
            )
            gt_one = (
                cj["pred_usable_window_m"] > ci["pred_usable_window_m"]
                or cj["pred_pk_40m_pct"] > ci["pred_pk_40m_pct"]
                or cj["pred_peak_recoil_kn"] < ci["pred_peak_recoil_kn"]
                or cj["pred_cost_per_kill_usd"] < ci["pred_cost_per_kill_usd"]
            )
            if ge_all and gt_one:
                dominated = True
                break
        if not dominated:
            ci["is_pareto"] = True
            pareto_front.append(ci)

    pareto_front.sort(key=lambda c: c["pred_usable_window_m"], reverse=True)

    # Verify top 3 Pareto highlights directly with exact physics
    picks: list[tuple[str, dict[str, Any]]] = []
    if pareto_front:
        picks.append(("Max Window & Reach", pareto_front[0]))
        best_pk_c = max(pareto_front, key=lambda c: c["pred_pk_40m_pct"] - 0.3 * c["pred_peak_recoil_kn"])
        picks.append(("Balanced Pk / Recoil", best_pk_c))
        best_eco_c = min(pareto_front, key=lambda c: c["pred_cost_per_kill_usd"])
        picks.append(("Minimum $/Kill", best_eco_c))

    verified_top: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for role_lbl, cand in picks:
        if cand["cand_id"] in seen_ids:
            continue
        seen_ids.add(cand["cand_id"])
        ex = _eval_design_vector_physics(
            np.asarray(cand["vec"], dtype=np.float64),
            drone_class=drone_class,
            drone_speed_mps=drone_speed_mps,
            drone_altitude_m=drone_altitude_m,
            aim_error_mrad=aim_error_mrad,
            site_altitude_m=site_altitude_m,
            drag_law=drag_law,
        )
        verified_top.append({
            "role": role_lbl,
            **{k: v for k, v in cand.items() if k != "vec"},
            "exact_usable_window_m": ex["usable_window_m"],
            "exact_pk_40m_pct": ex["pk_shared_40m_pct"],
            "exact_peak_recoil_kn": ex["peak_recoil_kn"],
            "exact_cost_per_kill_usd": ex["cost_per_kill_usd"],
            "exact_sdz_max_range_m": ex["sdz_max_range_m"],
        })

    for c in candidates:
        c.pop("vec", None)

    loo_metrics = {k: round(gps[k].loo_r2, 4) for k in metrics}
    return {
        "summary": {
            "n_doe": n_d,
            "n_candidates": n_c,
            "n_pareto_optimal": len(pareto_front),
            "recoil_limit_kn": float(recoil_limit_kn),
            "loo_r2": loo_metrics,
            "min_loo_r2": min(loo_metrics.values()),
            "best_verified_window_m": max((r["exact_usable_window_m"] for r in verified_top), default=0.0),
            "best_verified_pk_40m_pct": max((r["exact_pk_40m_pct"] for r in verified_top), default=0.0),
        },
        "loo_validation": [
            {"output_metric": k, "loo_r2": round(gps[k].loo_r2, 4), "loo_rmse": round(gps[k].loo_rmse, 3)}
            for k in metrics
        ],
        "candidates": candidates,
        "pareto_front": [{k: v for k, v in c.items() if k != "vec"} for c in pareto_front],
        "verified_top_designs": verified_top,
    }
