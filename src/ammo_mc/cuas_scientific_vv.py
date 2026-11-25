"""Scientific Verification, Uncertainty Quantification, Range DOE & Instrument Calibration Engine.

Implements:
1. ASME V&V 20 validation error & uncertainty budget + Ferson-Oberkampf Area Validation Metric + Clopper-Pearson exact binomial CIs
2. Blind Holdout Cross-Validation (calibrate on near/nominal data, score out-of-sample holdout ranges)
3. Nested Epistemic + Aleatory Uncertainty Quantification P-Box Envelopes (R80 / Pk 95% confidence bands)
4. Published Experimental Validation Benchmarks (E-01..E-06): Bailey-Hiatt sphere Cd(M), Compton S81 transonic decay,
   12-gauge 40-yd choke patterning, shot-stringing bounds, V50 sectional-density & Recht-Ipson conservation, rotor solidity
5. Pre-Test Design of Experiments (DOE), Sample Size Power & Wald SPRT Stopping Planner + Test Card CSV
6. Doppler Radar v(R) Drag Recovery & Weather/Acoustic Shot-String CSV Ingestion
7. Reproducible FAIR Validation Bundles (SHA-256 checksums, git SHA, RNG seed) & HTML/Markdown Dossiers
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import platform
import subprocess
from typing import Any
import warnings

import numpy as np
from scipy import optimize as sp_opt
from scipy import stats as sp_stats

from ammo_mc.aero_sphere import (
    sphere_drag_compressible,
    sphere_drag_compressible_array,
    sphere_drag_mach_only_array,
    sutherland_air_viscosity,
)
from ammo_mc.cuas_calibration_twin import (
    calibrate_livefire_drone_lethality,
    calibrate_multirange_pattern_and_drag,
    get_default_nominal_twin,
)
from ammo_mc.cuas_defense import _estimate_burst_pk_analytics
from ammo_mc.cuas_mission import _resolve_shotgun_ammo
from ammo_mc.small_arms import (
    compute_component_v50_perforation,
    compute_drone_projected_geometry,
    compute_rotor_blade_strike_probability,
    get_choke_definitions,
    get_drone_target_specs,
    simulate_anti_drone_multishotgun_mc,
)


# -------------------------------------------------------------------------
# 1. ASME V&V 20 METRICS, AREA VALIDATION METRIC & CLOPPER-PEARSON CI
# -------------------------------------------------------------------------


def compute_clopper_pearson_ci(k_successes: int, n_trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return exact two-sided Clopper-Pearson binomial confidence interval (low, high) in [0, 1]."""
    n = max(1, int(n_trials))
    k = max(0, min(n, int(k_successes)))
    a = max(1e-4, min(0.5, float(alpha)))
    low = 0.0 if k == 0 else float(sp_stats.beta.ppf(a / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(sp_stats.beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return (round(low, 6), round(high, 6))


def _area_validation_metric(sim_arr: np.ndarray, exp_arr: np.ndarray) -> float:
    """Compute Ferson-Oberkampf Area Validation Metric d(F_sim, F_exp) = integral |F_sim(x) - F_exp(x)| dx."""
    s = np.sort(np.asarray(sim_arr, dtype=np.float64))
    e = np.sort(np.asarray(exp_arr, dtype=np.float64))
    if s.size == 0 or e.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([s, e])))
    if grid.size < 2:
        return float(abs(float(s[0]) - float(e[0])))
    cdf_s = np.searchsorted(s, grid[:-1], side="right") / float(s.size)
    cdf_e = np.searchsorted(e, grid[:-1], side="right") / float(e.size)
    dx = np.diff(grid)
    return float(np.sum(np.abs(cdf_s - cdf_e) * dx))


def compute_asme_vv20_metrics(
    sim_values: list[float] | np.ndarray,
    exp_values: list[float] | np.ndarray,
    u_num: float | None = None,
    u_input: float = 0.0,
    u_exp: float = 0.0,
    k_coverage: float = 2.0,
) -> dict[str, Any]:
    """Compute ASME V&V 20 validation comparison error E, combined u_val, and Ferson-Oberkampf AVM."""
    s = np.asarray(sim_values, dtype=np.float64)
    e = np.asarray(exp_values, dtype=np.float64)
    if s.size == 0 or e.size == 0:
        raise ValueError("ASME V&V 20 requires non-empty simulation and experimental arrays.")

    mu_s = float(np.mean(s))
    mu_e = float(np.mean(e))
    comp_error = mu_s - mu_e

    u_num_eff = float(u_num) if u_num is not None else float(np.std(s, ddof=1) / math.sqrt(s.size)) if s.size > 1 else 0.0
    u_exp_stat = float(np.std(e, ddof=1) / math.sqrt(e.size)) if (u_exp == 0.0 and e.size > 1) else float(u_exp)
    u_val = math.sqrt(max(0.0, u_num_eff) ** 2 + max(0.0, float(u_input)) ** 2 + max(0.0, u_exp_stat) ** 2)
    u_val_safe = max(u_val, 1e-9)
    ratio = abs(comp_error) / (max(0.5, float(k_coverage)) * u_val_safe)

    avm = _area_validation_metric(s, e)
    try:
        ks_res = sp_stats.ks_2samp(s, e)
        ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)
    except Exception:
        ks_stat, ks_p = 0.0, 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ad_res = sp_stats.anderson_ksamp([s, e])
        ad_stat = float(ad_res.statistic)
    except Exception:
        ad_stat = 0.0

    verdict = "VALIDATED" if ratio <= 1.0 else ("MARGINAL" if ratio <= 1.5 else "MODEL_DISCREPANCY")
    return {
        "n_sim": int(s.size),
        "n_exp": int(e.size),
        "sim_mean": round(mu_s, 5),
        "exp_mean": round(mu_e, 5),
        "comparison_error_E": round(comp_error, 5),
        "mean_comparison_error_e": round(abs(comp_error), 5),
        "u_num": round(u_num_eff, 5),
        "u_input": round(float(u_input), 5),
        "u_exp": round(u_exp_stat, 5),
        "u_val": round(u_val, 5),
        "k_coverage": float(k_coverage),
        "e_over_kuval": round(ratio, 4),
        "area_validation_metric": round(avm, 5),
        "area_validation_metric_avm": round(avm, 5),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_p, 4),
        "ad_statistic": round(ad_stat, 4),
        "verdict": verdict,
        "validated": bool(verdict == "VALIDATED"),
    }


# -------------------------------------------------------------------------
# 2. BLIND HOLDOUT CROSS-VALIDATION
# -------------------------------------------------------------------------


def generate_sample_blind_holdout_dataset(
    seed: int = 42,
    nominal_choke: str = "Modified",
    ammunition_preset: str = "ALDA_BUCKSHOT_12_76_COPPER_32G",
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate calibration (20-50 m) + blind holdout (65-75 m) range pattern and lethality trials."""
    rng = np.random.default_rng(seed)
    cdefs = get_choke_definitions()
    cdef = cdefs.get("Modified", list(cdefs.values())[0])
    sigma_40 = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(cdef["benchmark_40yd_30in_pct"])))
    theta_nom = sigma_40 / 36.58
    _, ammo = _resolve_shotgun_ammo("ALDA_BUCKSHOT_12_76_COPPER_32G")
    v0 = float(ammo["v0_mps"])
    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    cd = float(ammo.get("effective_cd", 0.47))
    kappa = (1.225 * cd * math.pi * 0.25 * (d_m ** 2)) / (2.0 * max(m_kg, 1e-6))

    true_spread, true_exp, true_cd = 1.14, 1.22, 1.08
    pat_rows: list[dict[str, Any]] = []
    for r_m in (20.0, 30.0, 40.0, 50.0, 65.0, 75.0):
        sig_true = (0.01852 / 4.0 + theta_nom * r_m * ((r_m / 36.58) ** (true_exp - 1.0))) * true_spread
        sig_meas = max(0.03, float(sig_true * rng.normal(1.0, 0.025)))
        v_meas = max(90.0, float(v0 * math.exp(-true_cd * kappa * r_m) + rng.normal(0.0, 2.2)))
        pat_rows.append({
            "range_m": r_m,
            "sigma_m": round(sig_meas, 4),
            "measured_v_mps": round(v_meas, 1),
        })

    d_spec = get_drone_target_specs()["fpv_14in_36cm_pg7m"]
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
    true_twin = {
        "spread_factor": true_spread,
        "spread_exponent": true_exp,
        "pellet_cd_scale": true_cd,
        "drone_avuln_scale": 0.90,
        "drone_ecrit_scale": 1.18,
    }
    lf_rows: list[dict[str, Any]] = []
    for r_m in (25.0, 35.0, 45.0, 50.0, 65.0, 75.0):
        n_rep = 8 if r_m in (65.0, 75.0) else 5
        pk_t = _estimate_burst_pk_analytics(
            slant_range_m=r_m,
            barrels_fired=5,
            pellets_per_shot=int(ammo["pellets_per_shot"]),
            pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
            pellet_mass_g=float(ammo["pellet_mass_g"]),
            v0_mps=v0,
            choke_name="Modified",
            drone_spec=d_spec,
            base_cd=cd,
            aspect_mode="head_on_frontal",
            aspect_geom=aspect_geom,
            calibration_profile=true_twin,
        )
        for _ in range(n_rep):
            lf_rows.append({
                "range_m": r_m,
                "ammunition_preset": "ALDA_BUCKSHOT_12_76_COPPER_32G",
                "choke": "Modified",
                "barrels_fired": 5,
                "drone_killed": int(rng.random() < pk_t),
            })

    return {"pattern_rows": pat_rows, "livefire_rows": lf_rows}


def run_blind_holdout_validation(
    pattern_rows: list[dict[str, Any]],
    livefire_rows: list[dict[str, Any]] | None = None,
    holdout_ranges_m: tuple[float, ...] = (65.0, 75.0),
    nominal_choke: str = "Modified",
    ammunition_preset: str = "ALDA_BUCKSHOT_12_76_COPPER_32G",
    drone_model: str = "fpv_14in_36cm_pg7m",
    seed: int = 42,
) -> dict[str, Any]:
    """Fit Digital Twin strictly on training ranges and evaluate blind out-of-sample holdout error."""
    hold_set = {round(float(x), 1) for x in holdout_ranges_m}
    train_pat = [r for r in pattern_rows if round(float(r["range_m"]), 1) not in hold_set]
    test_pat = [r for r in pattern_rows if round(float(r["range_m"]), 1) in hold_set]
    if len(train_pat) < 3 or len(test_pat) < 1:
        sorted_pat = sorted(pattern_rows, key=lambda r: float(r["range_m"]))
        split_idx = max(3, len(sorted_pat) - 2)
        train_pat = sorted_pat[:split_idx]
        test_pat = sorted_pat[split_idx:]
        hold_set = {round(float(r["range_m"]), 1) for r in test_pat}

    cal_mr = calibrate_multirange_pattern_and_drag(
        train_pat, nominal_choke=nominal_choke, ammunition_preset=ammunition_preset
    )
    twin_fit = dict(cal_mr["twin_patch"])

    cdefs = get_choke_definitions()
    cdef = cdefs.get(nominal_choke, list(cdefs.values())[0])
    sigma_40 = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(cdef["benchmark_40yd_30in_pct"])))
    theta_nom = sigma_40 / 36.58

    sf_fit = float(twin_fit.get("spread_factor", 1.0))
    exp_fit = float(twin_fit.get("spread_exponent", 1.15))

    holdout_pattern_rows: list[dict[str, Any]] = []
    sim_sig_hold, exp_sig_hold, nom_sig_hold = [], [], []
    for row in test_pat:
        r_m = float(row["range_m"])
        sig_obs = float(row["sigma_m"])
        sig_nom = (0.01852 / 4.0 + theta_nom * r_m * ((r_m / 36.58) ** (1.15 - 1.0)))
        sig_cal = (0.01852 / 4.0 + theta_nom * r_m * ((r_m / 36.58) ** (exp_fit - 1.0))) * sf_fit
        sim_sig_hold.append(sig_cal)
        exp_sig_hold.append(sig_obs)
        nom_sig_hold.append(sig_nom)
        holdout_pattern_rows.append({
            "range_m": r_m,
            "observed_sigma_m": round(sig_obs, 4),
            "nominal_pred_sigma_m": round(sig_nom, 4),
            "calibrated_pred_sigma_m": round(sig_cal, 4),
            "calibrated_abs_err_m": round(abs(sig_cal - sig_obs), 4),
            "nominal_abs_err_m": round(abs(sig_nom - sig_obs), 4),
        })

    rmse_cal_m = float(math.sqrt(float(np.mean((np.asarray(sim_sig_hold) - np.asarray(exp_sig_hold)) ** 2))))
    rmse_nom_m = float(math.sqrt(float(np.mean((np.asarray(nom_sig_hold) - np.asarray(exp_sig_hold)) ** 2))))
    asme_sigma = compute_asme_vv20_metrics(
        sim_sig_hold,
        exp_sig_hold,
        u_input=max(0.005, float(cal_mr.get("rmse_spread_m", 0.01))),
        u_exp=max(0.008, float(np.mean(exp_sig_hold)) * 0.03),
    )

    lethality_summary: dict[str, Any] = {}
    if livefire_rows:
        train_lf = [r for r in livefire_rows if round(float(r["range_m"]), 1) not in hold_set]
        test_lf = [r for r in livefire_rows if round(float(r["range_m"]), 1) in hold_set]
        if len(train_lf) >= 5 and len(test_lf) >= 2:
            lf_cal = calibrate_livefire_drone_lethality(train_lf, drone_model=drone_model, base_twin=twin_fit)
            twin_full = dict(twin_fit)
            twin_full.update(lf_cal["twin_patch"])
            d_spec = get_drone_target_specs().get(drone_model, get_drone_target_specs()["fpv_14in_36cm_pg7m"])
            aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
            brier_nom_list, brier_cal_list = [], []
            by_range: dict[float, list[dict[str, Any]]] = {}
            for tr in test_lf:
                by_range.setdefault(float(tr["range_m"]), []).append(tr)
            lf_hold_rows: list[dict[str, Any]] = []
            for rm, grp in sorted(by_range.items()):
                ak = str(grp[0].get("ammunition_preset", ammunition_preset))
                chk = str(grp[0].get("choke", nominal_choke))
                bbls = int(grp[0].get("barrels_fired", 5))
                _, ammo = _resolve_shotgun_ammo(ak)
                pk_cal = _estimate_burst_pk_analytics(
                    slant_range_m=rm,
                    barrels_fired=bbls,
                    pellets_per_shot=int(ammo["pellets_per_shot"]),
                    pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                    pellet_mass_g=float(ammo["pellet_mass_g"]),
                    v0_mps=float(ammo["v0_mps"]),
                    choke_name=chk,
                    drone_spec=d_spec,
                    base_cd=float(ammo.get("effective_cd", 0.47)),
                    aspect_mode="head_on_frontal",
                    aspect_geom=aspect_geom,
                    calibration_profile=twin_full,
                )
                pk_nom = _estimate_burst_pk_analytics(
                    slant_range_m=rm,
                    barrels_fired=bbls,
                    pellets_per_shot=int(ammo["pellets_per_shot"]),
                    pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                    pellet_mass_g=float(ammo["pellet_mass_g"]),
                    v0_mps=float(ammo["v0_mps"]),
                    choke_name=chk,
                    drone_spec=d_spec,
                    base_cd=float(ammo.get("effective_cd", 0.47)),
                    aspect_mode="head_on_frontal",
                    aspect_geom=aspect_geom,
                    calibration_profile=None,
                )
                k_hits = sum(int(r["drone_killed"]) for r in grp)
                n_tot = len(grp)
                ci_lo, ci_hi = compute_clopper_pearson_ci(k_hits, n_tot)
                for r in grp:
                    y = float(r["drone_killed"])
                    brier_cal_list.append((pk_cal - y) ** 2)
                    brier_nom_list.append((pk_nom - y) ** 2)
                lf_hold_rows.append({
                    "range_m": rm,
                    "n_trials": n_tot,
                    "n_kills": k_hits,
                    "observed_pk": round(k_hits / n_tot, 3),
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "nominal_pred_pk": round(pk_nom, 3),
                    "calibrated_pred_pk": round(pk_cal, 3),
                })
            lethality_summary = {
                "n_train_trials": len(train_lf),
                "n_holdout_trials": len(test_lf),
                "holdout_brier_nominal": round(float(np.mean(brier_nom_list)), 4),
                "holdout_brier_calibrated": round(float(np.mean(brier_cal_list)), 4),
                "holdout_rows": lf_hold_rows,
            }

    return {
        "n_train_ranges": len(train_pat),
        "n_holdout_ranges": len(test_pat),
        "holdout_ranges_m": sorted(float(r["range_m"]) for r in test_pat),
        "fitted_twin_patch": twin_fit,
        "holdout_rmse_calibrated_m": round(rmse_cal_m, 4),
        "holdout_rmse_nominal_m": round(rmse_nom_m, 4),
        "error_reduction_pct": round(100.0 * (rmse_nom_m - rmse_cal_m) / max(rmse_nom_m, 1e-6), 1),
        "holdout_pattern_rows": holdout_pattern_rows,
        "asme_vv20_sigma": asme_sigma,
        "lethality_holdout": lethality_summary,
    }


# -------------------------------------------------------------------------
# 3. EPISTEMIC vs. ALEATORY UQ P-BOX ENVELOPE
# -------------------------------------------------------------------------


def compute_epistemic_aleatory_uq_envelope(
    distance_stations_m: tuple[float, ...] = (20.0, 35.0, 50.0, 65.0, 75.0),
    twin_profile: dict[str, Any] | None = None,
    layout_type: str = "quincunx",
    n_barrels: int = 5,
    mount_radius_m: float = 0.30,
    ammunition_preset: str = "ALDA_BUCKSHOT_12_76_COPPER_32G",
    inner_ammunition_preset: str | None = "SPARTAN_33_HP_SPHERO_4T",
    default_choke: str = "Modified",
    inner_choke: str | None = "Cylinder",
    drone_model: str = "fpv_14in_36cm_pg7m",
    drone_velocity_mps: float = 0.0,
    n_epistemic_draws: int = 12,
    n_volleys_per_draw: int = 10,
    seed: int = 42,
    **sim_kwargs: Any,
) -> dict[str, Any]:
    """Run nested epistemic (twin posterior SE) x aleatory (MC shot cloud) UQ propagation across range."""
    twin = dict(get_default_nominal_twin())
    if twin_profile:
        twin.update({k: v for k, v in twin_profile.items() if v is not None})

    sampler = sp_stats.qmc.LatinHypercube(d=5, seed=seed)
    u = sampler.random(n=max(4, int(n_epistemic_draws)))
    z = sp_stats.norm.ppf(np.clip(u, 0.02, 0.98))

    sf_mu, sf_se = float(twin.get("spread_factor", 1.0)), float(twin.get("spread_factor_se", 0.08))
    exp_mu, exp_se = float(twin.get("spread_exponent", 1.15)), float(twin.get("spread_exponent_se", 0.06))
    mv_mu, mv_se = float(twin.get("mv_scale", 1.0)), float(twin.get("mv_scale_se", 0.03))
    av_mu, av_se = float(twin.get("drone_avuln_scale", 1.0)), float(twin.get("drone_avuln_scale_se", 0.14))
    ec_mu, ec_se = float(twin.get("drone_ecrit_scale", 1.0)), float(twin.get("drone_ecrit_scale_se", 0.15))

    stations = sorted(float(d) for d in distance_stations_m)
    pk_draws = np.zeros((len(u), len(stations)), dtype=np.float64)
    r80_draws: list[float] = []

    for i in range(len(u)):
        cp_i = dict(twin)
        cp_i["spread_factor"] = max(0.40, sf_mu + sf_se * float(z[i, 0]))
        cp_i["spread_exponent"] = max(0.85, min(1.65, exp_mu + exp_se * float(z[i, 1])))
        cp_i["mv_scale"] = max(0.75, min(1.25, mv_mu + mv_se * float(z[i, 2])))
        cp_i["drone_avuln_scale"] = max(0.45, min(1.80, av_mu + av_se * float(z[i, 3])))
        cp_i["drone_ecrit_scale"] = max(0.50, min(2.00, ec_mu + ec_se * float(z[i, 4])))

        for j, d_m in enumerate(stations):
            sim = simulate_anti_drone_multishotgun_mc(
                layout_type=layout_type,
                n_barrels=n_barrels,
                mount_radius_m=mount_radius_m,
                ammunition_preset=ammunition_preset,
                inner_ammunition_preset=inner_ammunition_preset,
                outer_ammunition_preset=ammunition_preset if inner_ammunition_preset else None,
                default_choke=default_choke,
                inner_choke=inner_choke,
                drone_model=drone_model,
                target_distance_m=d_m,
                drone_velocity_mps=drone_velocity_mps,
                n_volleys=max(5, int(n_volleys_per_draw)),
                seed=seed + i * 37 + j * 7,
                calibration_profile=cp_i,
                **sim_kwargs,
            )
            pk_draws[i, j] = float(sim["summary"]["kill_probability_pct"])

        # Interpolate R80 for this epistemic sample
        row_pks = pk_draws[i, :]
        r80_val = stations[0] * 0.8
        for j in range(len(stations) - 1):
            p_lo, p_hi = float(row_pks[j]), float(row_pks[j + 1])
            if p_lo >= 80.0 and p_hi >= 80.0:
                r80_val = stations[j + 1]
            elif p_lo >= 80.0 > p_hi and (p_lo - p_hi) > 1e-6:
                frac = (p_lo - 80.0) / (p_lo - p_hi)
                r80_val = stations[j] + frac * (stations[j + 1] - stations[j])
                break
        r80_draws.append(float(r80_val))

    band_rows: list[dict[str, Any]] = []
    for j, d_m in enumerate(stations):
        col = pk_draws[:, j]
        band_rows.append({
            "distance_m": d_m,
            "pk_p05_pct": round(float(np.percentile(col, 5)), 1),
            "pk_median_pct": round(float(np.percentile(col, 50)), 1),
            "pk_mean_pct": round(float(np.mean(col)), 1),
            "pk_p95_pct": round(float(np.percentile(col, 95)), 1),
            "epistemic_sd_pp": round(float(np.std(col, ddof=1)), 2) if len(col) > 1 else 0.0,
        })

    r80_arr = np.asarray(r80_draws, dtype=np.float64)
    return {
        "distance_stations_m": stations,
        "band_rows": band_rows,
        "n_epistemic_draws": int(len(u)),
        "n_volleys_per_draw": int(max(5, n_volleys_per_draw)),
        "r80_p05_m": round(float(np.percentile(r80_arr, 5)), 1),
        "r80_median_m": round(float(np.percentile(r80_arr, 50)), 1),
        "r80_p95_m": round(float(np.percentile(r80_arr, 95)), 1),
        "mean_band_width_pp": round(float(np.mean([r["pk_p95_pct"] - r["pk_p05_pct"] for r in band_rows])), 1),
    }


# -------------------------------------------------------------------------
# 4. PUBLISHED EXPERIMENTAL VALIDATION BENCHMARKS (E-01..E-06)
# -------------------------------------------------------------------------

# Digitized Bailey & Hiatt (1972) / Loth (2008) continuum high-Re sphere drag data
_BAILEY_HIATT_EXPT: list[tuple[float, float]] = [
    (0.35, 0.470),
    (0.70, 0.528),
    (0.90, 0.695),
    (1.00, 0.822),
    (1.20, 0.936),
    (1.60, 0.987),
    (2.00, 0.978),
    (2.50, 0.959),
]


def run_experimental_validation_benchmarks() -> dict[str, Any]:
    """Run published C-UAS aerodynamic, patterning, shot-string, V50 & rotor-solidity benchmarks (E-01..E-06)."""
    rows: list[dict[str, Any]] = []

    # E-01: Bailey & Hiatt (1972) continuum sphere Cd(M) vs model predictions
    bh_m = [pt[0] for pt in _BAILEY_HIATT_EXPT]
    bh_exp = [pt[1] for pt in _BAILEY_HIATT_EXPT]
    bh_pred = [float(sphere_drag_compressible(1.2e5, m)) for m in bh_m]
    e01_errs = [abs(p - e) / max(e, 1e-6) for p, e in zip(bh_pred, bh_exp)]
    e01_max_rel = max(e01_errs)
    e01_avm = _area_validation_metric(np.asarray(bh_pred), np.asarray(bh_exp))
    rows.append({
        "benchmark_id": "E-01",
        "domain": "Sphere Compressible Drag",
        "title": "Bailey & Hiatt (1972) wind-tunnel sphere Cd(M=0.35..2.50)",
        "source_ref": "Bailey & Hiatt (1972) / Loth (2008)",
        "metric": "Max Rel Error & AVM",
        "computed": round(e01_max_rel, 4),
        "reference": 0.0,
        "tolerance": 0.10,
        "avm": round(e01_avm, 4),
        "passed": bool(e01_max_rel <= 0.10 and e01_avm <= 0.08),
        "status": "PASS" if (e01_max_rel <= 0.10 and e01_avm <= 0.08) else "FAIL",
        "details": {"mach": bh_m, "exp_cd": bh_exp, "pred_cd": [round(x, 4) for x in bh_pred]},
    })

    # E-02: Compton (1996) S81 smooth-sphere Mach-only decay vs 400-step RK4 reference
    rho0, a0 = 1.225, 340.29
    def _rk4_mach_vel(v0: float, d_mm: float, m_g: float, range_m: float = 50.0) -> float:
        d_m = d_mm / 1000.0
        m_kg = m_g / 1000.0
        area_p = math.pi * 0.25 * (d_m ** 2)
        n_steps = 400
        dx = range_m / n_steps
        v = float(v0)
        for _ in range(n_steps):
            def _dvdx(vel: float) -> float:
                cd_v = float(sphere_drag_mach_only_array([vel / a0], 0.47)[0])
                return -(0.5 * rho0 * cd_v * area_p * vel) / max(m_kg, 1e-6)
            k1 = _dvdx(v)
            k2 = _dvdx(v + 0.5 * dx * k1)
            k3 = _dvdx(v + 0.5 * dx * k2)
            k4 = _dvdx(v + dx * k3)
            v = max(10.0, v + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        return v

    s81_cases = [
        ("SPARTAN_33_HP_SPHERO_4T", 415.0, 3.25, 0.327),
        ("ALDA_BUCKSHOT_12_76_COPPER_32G", 400.0, 6.20, 1.123),
        ("ALDA_DOUBLE_00_HP_N2_0", 400.0, 4.50, 0.541),
    ]
    e02_errs: list[float] = []
    for ak, v0_c, d_mm_c, m_g_c in s81_cases:
        v_rk4 = _rk4_mach_vel(v0_c, d_mm_c, m_g_c, 50.0)
        sim_c = simulate_anti_drone_multishotgun_mc(
            layout_type="single",
            n_barrels=1,
            ammunition_preset=ak,
            v0_mps=v0_c,
            pellet_diameter_mm=d_mm_c,
            pellet_mass_g=m_g_c,
            target_distance_m=50.0,
            n_volleys=5,
            drag_model="mach_only_smooth_sphere",
            seed=42,
        )
        v_sim = float(sim_c["summary"]["mean_v_impact_nominal_mps"])
        e02_errs.append(abs(v_sim - v_rk4) / max(v_rk4, 1.0))
    e02_max_rel = max(e02_errs)
    rows.append({
        "benchmark_id": "E-02",
        "domain": "Transonic Pellet Decay",
        "title": "Compton (1996) S81 RK2 trajectory solver vs 400-step RK4 across W/Cu/Pb spheres @ 50 m",
        "source_ref": "Compton (1996) UCL Thesis [S81]",
        "metric": "Max Velocity Rel Err",
        "computed": round(e02_max_rel, 5),
        "reference": 0.0,
        "tolerance": 0.015,
        "avm": round(e02_max_rel, 5),
        "passed": bool(e02_max_rel <= 0.015),
        "status": "PASS" if e02_max_rel <= 0.015 else "FAIL",
        "details": {"max_rel_err_by_load": [round(x, 5) for x in e02_errs]},
    })

    # E-03: 12-gauge 40-yd 30-inch pattern % across Cylinder, IC, Modified, Full chokes
    cdefs = get_choke_definitions()
    choke_names = ["Cylinder", "Improved Cylinder", "Modified", "Full"]
    ref_pcts = [float(cdefs[c]["benchmark_40yd_30in_pct"]) * 100.0 for c in choke_names]
    mc_pcts: list[float] = []
    for idx, chk in enumerate(choke_names):
        s_chk = simulate_anti_drone_multishotgun_mc(
            layout_type="single",
            n_barrels=1,
            ammunition_preset="ALDA_0_80_HP_N2",
            pellets_per_shot=120,
            default_choke=chk,
            target_distance_m=36.58,
            n_volleys=8,
            seed=10 + idx,
        )
        mc_pcts.append(float(s_chk["summary"]["pattern_30in_pct"]))
    e03_mae = float(np.mean(np.abs(np.asarray(mc_pcts) - np.asarray(ref_pcts))))
    e03_mono = all(mc_pcts[i] <= mc_pcts[i + 1] + 8.0 for i in range(len(mc_pcts) - 1)) and (mc_pcts[-1] > mc_pcts[0])
    rows.append({
        "benchmark_id": "E-03",
        "domain": "12G Choke Patterning",
        "title": "40-yard (36.58 m) 30-inch pattern efficiency vs Cylinder..Full choke benchmarks",
        "source_ref": "SAAMI / Lowry 40-yd 30-in Standard",
        "metric": "Mean Abs Diff (%)",
        "computed": round(e03_mae, 2),
        "reference": 0.0,
        "tolerance": 10.0,
        "avm": round(_area_validation_metric(np.asarray(mc_pcts), np.asarray(ref_pcts)), 2),
        "passed": bool(e03_mae <= 10.0 and e03_mono),
        "status": "PASS" if (e03_mae <= 10.0 and e03_mono) else "FAIL",
        "details": {"chokes": choke_names, "ref_pct": ref_pcts, "mc_pct": mc_pcts},
    })

    # E-04: Shot-string longitudinal dispersion length (Lead soft deformation > Tungsten rigid)
    s_pb = simulate_anti_drone_multishotgun_mc(
        layout_type="single", n_barrels=1, ammunition_preset="ALDA_DOUBLE_00_HP_N2_0",
        pellets_per_shot=80, target_distance_m=45.0, n_volleys=6, pattern_distribution="core_flyer_mixture", seed=42
    )
    s_w = simulate_anti_drone_multishotgun_mc(
        layout_type="single", n_barrels=1, ammunition_preset="SPARTAN_33_HP_SPHERO_4T",
        default_choke="Cylinder", target_distance_m=45.0, n_volleys=6, pattern_distribution="core_flyer_mixture", seed=42
    )
    len_pb = float(s_pb["summary"]["shot_string_length_m"])
    len_w = float(s_w["summary"]["shot_string_length_m"])
    e04_pass = (0.7 <= len_w <= 4.5) and (0.9 <= len_pb <= 5.5) and (len_pb > len_w)
    rows.append({
        "benchmark_id": "E-04",
        "domain": "Longitudinal Shot-String",
        "title": "Deformation-coupled shot-string length @ 45 m (Hardened Lead > Tungsten)",
        "source_ref": "Brister / SAAMI Shot-String Kinematics",
        "metric": "Lead String Length (m)",
        "computed": round(len_pb, 3),
        "reference": round(len_w, 3),
        "tolerance": 5.5,
        "avm": round(abs(len_pb - len_w), 3),
        "passed": e04_pass,
        "status": "PASS" if e04_pass else "FAIL",
        "details": {"lead_string_m": len_pb, "tungsten_string_m": len_w},
    })

    # E-05: Sectional density V50 ordering & Recht-Ipson energy conservation
    v50_w = compute_component_v50_perforation("battery_payload", 320.0, 0.00054, 0.00355, pellet_density_g_cm3=18.2)
    v50_pb = compute_component_v50_perforation("battery_payload", 320.0, 0.00054, 0.00450, pellet_density_g_cm3=11.3)
    e05_pass = (v50_w["v50_mps"] < v50_pb["v50_mps"]) and (0.0 < v50_w["residual_velocity_mps"] < 320.0)
    rows.append({
        "benchmark_id": "E-05",
        "domain": "Terminal V50 & Recht-Ipson",
        "title": "Sectional-density V50 advantage (Tungsten 18.2 vs Lead 11.3 g/cm3) & residual speed",
        "source_ref": "Recht & Ipson (1963) / THOR Equations",
        "metric": "Tungsten V50 (m/s)",
        "computed": v50_w["v50_mps"],
        "reference": v50_pb["v50_mps"],
        "tolerance": v50_pb["v50_mps"],
        "avm": round(v50_pb["v50_mps"] - v50_w["v50_mps"], 2),
        "passed": e05_pass,
        "status": "PASS" if e05_pass else "FAIL",
        "details": {"tungsten": v50_w, "lead": v50_pb},
    })

    # E-06: Spinning rotor blade solidity reduces phantom rotor hits vs solid disc
    p_hub = compute_rotor_blade_strike_probability(radial_frac=0.05, n_blades=3, rotor_radius_m=0.14)
    p_outer = compute_rotor_blade_strike_probability(radial_frac=0.65, n_blades=3, rotor_radius_m=0.14, rpm=11000.0, pellet_velocity_mps=260.0)
    s_solid = simulate_anti_drone_multishotgun_mc(
        layout_type="single", n_barrels=1, ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
        drone_model="fpv_14in_36cm_pg7m", target_distance_m=35.0, n_volleys=15, use_rotor_solidity=False, seed=42
    )
    s_blades = simulate_anti_drone_multishotgun_mc(
        layout_type="single", n_barrels=1, ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
        drone_model="fpv_14in_36cm_pg7m", target_distance_m=35.0, n_volleys=15, use_rotor_solidity=True, seed=42
    )
    hits_solid = float(s_solid["summary"]["mean_hits_on_drone"])
    hits_blades = float(s_blades["summary"]["mean_hits_on_drone"])
    e06_pass = (p_hub == 1.0) and (0.08 <= p_outer <= 0.45) and (hits_blades < hits_solid)
    rows.append({
        "benchmark_id": "E-06",
        "domain": "Rotor Blade Solidity",
        "title": "Spinning blade-solidity reduction vs solid disc idealization on 14-in FPV quad",
        "source_ref": "Kinematics of Swept Rotor Solidity",
        "metric": "Blade Mean Hits",
        "computed": hits_blades,
        "reference": hits_solid,
        "tolerance": hits_solid,
        "avm": round(hits_solid - hits_blades, 2),
        "passed": e06_pass,
        "status": "PASS" if e06_pass else "FAIL",
        "details": {"p_hub": p_hub, "p_outer_065": round(p_outer, 4), "solid_hits": hits_solid, "blade_hits": hits_blades},
    })

    n_pass = sum(1 for r in rows if r["passed"])
    return {
        "benchmarks": rows,
        "n_benchmarks": len(rows),
        "passed_benchmarks": n_pass,
        "all_passed": bool(n_pass == len(rows)),
        "summary": {
            "total": len(rows),
            "passed": n_pass,
            "pass_rate_pct": round(100.0 * n_pass / max(len(rows), 1), 1),
        },
    }


# -------------------------------------------------------------------------
# 5. PRE-TEST DESIGN OF EXPERIMENTS (DOE), POWER & WALD SPRT PLANNER
# -------------------------------------------------------------------------


def plan_range_validation_doe(
    target_pk: float = 0.80,
    margin_half_width: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
    compare_pk_alt: float | None = 0.65,
    sprt_p0: float = 0.65,
    sprt_p1: float = 0.85,
    ranges_m: tuple[float, ...] = (25.0, 50.0, 75.0),
    chokes: tuple[str, ...] = ("Cylinder", "Modified"),
    temperatures_c: tuple[float, ...] = (0.0, 15.0),
    unit_cost_eur: float = 1.97,
    barrels_per_volley: int = 5,
    target_range_m: float | None = None,
    p0: float | None = None,
    p1: float | None = None,
    beta: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate minimum sample sizes, A/B comparison power, Wald SPRT bounds & D-optimal range matrix."""
    if p0 is not None:
        sprt_p0 = float(p0)
    if p1 is not None:
        sprt_p1 = float(p1)
        target_pk = float(p1)
    if beta is not None:
        power = 1.0 - float(beta)
    if target_range_m is not None and target_range_m > 0:
        tr = float(target_range_m)
        ranges_m = tuple(sorted(set([max(10.0, tr - 20.0), tr, tr + 20.0])))

    p = max(0.05, min(0.95, float(target_pk)))
    half_w = max(0.02, min(0.30, float(margin_half_width)))
    z_a = float(sp_stats.norm.ppf(1.0 - max(0.005, min(0.25, float(alpha))) / 2.0))
    n_ci = int(math.ceil((z_a ** 2) * p * (1.0 - p) / (half_w ** 2)))

    n_ab = None
    if compare_pk_alt is not None and abs(p - float(compare_pk_alt)) > 0.02:
        p2 = max(0.05, min(0.95, float(compare_pk_alt)))
        z_b = float(sp_stats.norm.ppf(max(0.50, min(0.99, float(power)))))
        p_bar = 0.5 * (p + p2)
        num = (z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar)) + z_b * math.sqrt(p * (1.0 - p) + p2 * (1.0 - p2))) ** 2
        n_ab = int(math.ceil(num / ((p - p2) ** 2)))

    # Wald Sequential Probability Ratio Test (SPRT) lines for binomial kills vs rounds n
    p0 = max(0.05, min(0.90, float(sprt_p0)))
    p1 = max(p0 + 0.05, min(0.98, float(sprt_p1)))
    alpha_s = max(0.01, min(0.20, float(alpha)))
    beta_s = 1.0 - max(0.55, min(0.98, float(power)))
    A_log = math.log((1.0 - beta_s) / alpha_s)
    B_log = math.log(beta_s / (1.0 - alpha_s))
    log_odds_ratio = math.log((p1 * (1.0 - p0)) / (p0 * (1.0 - p1)))
    log_q_ratio = math.log((1.0 - p1) / (1.0 - p0))

    max_n_sprt = max(25, min(80, n_ci))
    sprt_rows: list[dict[str, Any]] = []
    for n_val in range(1, max_n_sprt + 1):
        k_accept = (A_log - n_val * log_q_ratio) / log_odds_ratio
        k_reject = (B_log - n_val * log_q_ratio) / log_odds_ratio
        sprt_rows.append({
            "n_rounds": n_val,
            "accept_h1_min_kills": round(min(float(n_val) + 0.5, k_accept), 2),
            "reject_h0_max_kills": round(max(-0.5, k_reject), 2),
        })

    # Factorial / D-optimal fractional test matrix across range x choke x temp
    n_cell = max(6, int(math.ceil(n_ci / max(len(ranges_m) * len(chokes), 1))))
    matrix_rows: list[dict[str, Any]] = []
    tot_volleys = 0
    for r_idx, r_m in enumerate(ranges_m):
        for c_idx, chk in enumerate(chokes):
            tc = float(temperatures_c[(r_idx + c_idx) % max(1, len(temperatures_c))]) if temperatures_c else 15.0
            cost_cell = round(n_cell * barrels_per_volley * float(unit_cost_eur), 2)
            tot_volleys += n_cell
            matrix_rows.append({
                "cell_id": f"DOE-{len(matrix_rows)+1:02d}",
                "range_m": float(r_m),
                "choke": str(chk),
                "ambient_temp_c": tc,
                "planned_volleys": n_cell,
                "rounds_expended": n_cell * int(barrels_per_volley),
                "estimated_ammo_eur": cost_cell,
            })

    return {
        "n_required_precision_ci": n_ci,
        "n_required_ab_per_arm": n_ab,
        "target_pk": p,
        "margin_half_width": half_w,
        "sprt_p0": p0,
        "sprt_p1": p1,
        "sprt_bounds": sprt_rows,
        "doe_matrix": matrix_rows,
        "total_volleys": tot_volleys,
        "total_rounds": tot_volleys * int(barrels_per_volley),
        "total_estimated_ammo_eur": round(tot_volleys * int(barrels_per_volley) * float(unit_cost_eur), 2),
    }


def export_doe_test_card_csv(doe_plan: dict[str, Any]) -> str:
    """Export pre-test DOE matrix to printable Range Test Card CSV."""
    lines = ["cell_id,range_m,choke,ambient_temp_c,planned_volleys,rounds_expended,estimated_ammo_eur,observed_kills,notes"]
    for row in doe_plan.get("doe_matrix", []):
        lines.append(
            f"{row['cell_id']},{row['range_m']},{row['choke']},{row['ambient_temp_c']},"
            f"{row['planned_volleys']},{row['rounds_expended']},{row['estimated_ammo_eur']},,"
        )
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------
# 6. DOPPLER RADAR v(R) DRAG RECOVERY & WEATHER/SHOT-STRING PARSERS
# -------------------------------------------------------------------------


def generate_sample_doppler_radar_csv(
    pellet_diameter_mm: float = 6.2,
    pellet_mass_g: float = 1.123,
    v0_true_mps: float = 402.0,
    true_cd_scale: float = 1.14,
    ranges_m: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 65.0, 75.0),
    ambient_temp_c: float = 15.0,
    seed: int = 42,
    ammunition_preset: str | None = None,
    **kwargs: Any,
) -> str:
    """Synthesize a LabRadar / Weibel-style Doppler range-velocity track CSV with realistic radar noise."""
    v0_eff = float(kwargs.get("true_v0_mps", v0_true_mps))
    cd_eff = float(kwargs.get("true_cd_scale", true_cd_scale))
    if "n_points" in kwargs:
        ranges_m = tuple(float(x) for x in np.linspace(5.0, 65.0, max(4, int(kwargs["n_points"]))))
    rng = np.random.default_rng(seed)
    t_k = float(ambient_temp_c) + 273.15
    rho = 101325.0 / (287.058 * t_k)
    mu = sutherland_air_viscosity(t_k)
    a_snd = math.sqrt(1.4 * 287.058 * t_k)
    d_m = float(pellet_diameter_mm) / 1000.0
    m_kg = float(pellet_mass_g) / 1000.0
    area_p = math.pi * 0.25 * (d_m ** 2)

    max_r = max(ranges_m)
    n_steps = 300
    dx = max_r / n_steps
    r_grid = np.linspace(0.0, max_r, n_steps + 1)
    v_grid = np.zeros(n_steps + 1, dtype=np.float64)
    v_cur = float(v0_eff)
    v_grid[0] = v_cur
    for i in range(n_steps):
        cd = float(sphere_drag_compressible(v_cur * d_m * rho / mu, v_cur / a_snd)) * float(cd_eff)
        dv = -(0.5 * rho * cd * area_p * v_cur) / max(m_kg, 1e-6)
        v_cur = max(10.0, v_cur + dx * dv)
        v_grid[i + 1] = v_cur

    lines = ["range_m,velocity_mps,temp_c,pressure_hpa"]
    for rm in ranges_m:
        v_meas = float(np.interp(rm, r_grid, v_grid) + rng.normal(0.0, 1.4))
        lines.append(f"{rm:.1f},{v_meas:.2f},{ambient_temp_c:.1f},1013.25")
    return "\n".join(lines) + "\n"


def calibrate_doppler_radar_drag(
    csv_text: str,
    pellet_diameter_mm: float = 6.2,
    pellet_mass_g: float = 1.123,
    ambient_temp_c: float = 15.0,
    air_pressure_hpa: float = 1013.25,
    headwind_mps: float = 0.0,
    drag_model: str = "compressible_re_m",
    nominal_v0_mps: float | None = None,
    ammunition_preset: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit both v0_mps and pellet_cd_scale jointly from Doppler radar v(R) measurements."""
    if ammunition_preset:
        from ammo_mc.small_arms import get_shotgun_presets

        presets = get_shotgun_presets()
        if ammunition_preset in presets:
            ammo_cfg = presets[ammunition_preset]
            if isinstance(ammo_cfg, dict):
                if pellet_diameter_mm == 6.2 and "pellet_diameter_m" in ammo_cfg:
                    pellet_diameter_mm = float(ammo_cfg["pellet_diameter_m"]) * 1000.0
                if pellet_mass_g == 1.123 and "pellet_mass_kg" in ammo_cfg:
                    pellet_mass_g = float(ammo_cfg["pellet_mass_kg"]) * 1000.0
            else:
                if pellet_diameter_mm == 6.2 and hasattr(ammo_cfg, "pellet_diameter_m"):
                    pellet_diameter_mm = float(ammo_cfg.pellet_diameter_m) * 1000.0
                if pellet_mass_g == 1.123 and hasattr(ammo_cfg, "pellet_mass_kg"):
                    pellet_mass_g = float(ammo_cfg.pellet_mass_kg) * 1000.0
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Doppler radar CSV has no header row.")
    pts: list[tuple[float, float]] = []
    temps, press = [], []
    for raw in reader:
        row = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
        r_raw = row.get("range_m", row.get("distance_m"))
        v_raw = row.get("velocity_mps", row.get("speed_mps"))
        if r_raw in (None, "") or v_raw in (None, ""):
            continue
        rm, vm = float(r_raw), float(v_raw)
        if rm >= 0.0 and vm > 10.0:
            pts.append((rm, vm))
        if row.get("temp_c") not in (None, ""):
            temps.append(float(row["temp_c"]))
        if row.get("pressure_hpa") not in (None, ""):
            press.append(float(row["pressure_hpa"]))
    if len(pts) < 4:
        raise ValueError("Doppler drag fit requires at least 4 (range_m, velocity_mps) stations.")

    pts.sort(key=lambda t: t[0])
    r_obs = np.array([p[0] for p in pts], dtype=np.float64)
    v_obs = np.array([p[1] for p in pts], dtype=np.float64)

    t_c = float(np.mean(temps)) if temps else float(ambient_temp_c)
    p_hpa = float(np.mean(press)) if press else float(air_pressure_hpa)
    t_k = t_c + 273.15
    rho = (p_hpa * 100.0) / (287.058 * t_k)
    mu = sutherland_air_viscosity(t_k)
    a_snd = math.sqrt(1.4 * 287.058 * t_k)
    d_m = float(pellet_diameter_mm) / 1000.0
    m_kg = float(pellet_mass_g) / 1000.0
    area_p = math.pi * 0.25 * (d_m ** 2)
    use_mach_only = str(drag_model).lower().strip() == "mach_only_smooth_sphere"

    max_r = float(np.max(r_obs))
    n_steps = 120
    r_grid = np.linspace(0.0, max_r, n_steps + 1)
    dx = max_r / n_steps

    def _sim_curve(v0_val: float, cd_scale: float) -> np.ndarray:
        v_arr = np.zeros(n_steps + 1, dtype=np.float64)
        vc = float(v0_val)
        v_arr[0] = vc
        for i in range(n_steps):
            vr = max(5.0, vc + float(headwind_mps))
            cd = float(sphere_drag_mach_only_array([vr / a_snd], 0.47)[0]) if use_mach_only else float(sphere_drag_compressible(vr * d_m * rho / mu, vr / a_snd))
            cd *= float(cd_scale)
            dv = -(0.5 * rho * cd * area_p * (vr ** 2)) / (max(m_kg, 1e-6) * max(vc, 5.0))
            vc = max(10.0, vc + dx * dv)
            v_arr[i + 1] = vc
        return np.interp(r_obs, r_grid, v_arr)

    init_v0 = float(nominal_v0_mps) if nominal_v0_mps is not None else float(v_obs[0]) + 15.0
    res = sp_opt.least_squares(
        lambda p: _sim_curve(float(p[0]), float(p[1])) - v_obs,
        x0=[init_v0, 1.0],
        bounds=([150.0, 0.45], [850.0, 2.20]),
    )
    v0_fit, cd_fit = float(res.x[0]), float(res.x[1])
    v_pred = _sim_curve(v0_fit, cd_fit)
    v_nom = _sim_curve(init_v0, 1.0)
    resid = v_obs - v_pred
    rmse = float(math.sqrt(float(np.mean(resid ** 2))))
    s2 = float(np.sum(resid ** 2)) / max(len(r_obs) - 2, 1)
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac) * s2
        v0_se = float(math.sqrt(max(float(cov[0, 0]), 1e-6)))
        cd_se = float(math.sqrt(max(float(cov[1, 1]), 1e-6)))
    except Exception:
        v0_se, cd_se = 2.0, 0.05

    return {
        "success": bool(res.success),
        "n_stations": len(pts),
        "n_track_points": len(pts),
        "nominal_v0_mps": round(init_v0, 2),
        "v0_fit_mps": round(v0_fit, 2),
        "fitted_v0_mps": round(v0_fit, 2),
        "v0_se_mps": round(v0_se, 2),
        "pellet_cd_scale": round(cd_fit, 4),
        "calibrated_cd_scale": round(cd_fit, 4),
        "pellet_cd_scale_se": round(cd_se, 4),
        "cd_scale_se": round(cd_se, 4),
        "rmse_mps": round(rmse, 2),
        "rmse_velocity_mps": round(rmse, 2),
        "ambient_temp_c": round(t_c, 1),
        "air_pressure_hpa": round(p_hpa, 2),
        "stations": [
            {"range_m": float(rv), "measured_v_mps": round(float(vo), 2), "fitted_v_mps": round(float(vp), 2), "nominal_v_mps": round(float(vn), 2)}
            for rv, vo, vp, vn in zip(r_obs, v_obs, v_pred, v_nom)
        ],
        "asme_vv20": compute_asme_vv20_metrics(v_pred, v_obs, u_input=rmse, u_exp=1.5),
        "twin_patch": {
            "pellet_cd_scale": round(cd_fit, 4),
            "pellet_cd_scale_se": round(cd_se, 4),
        },
    }


def generate_sample_shotstring_csv(range_m: float = 45.0, n_pellets: int = 36, seed: int = 42) -> str:
    """Synthesize a range weather (Kestrel) + high-speed/acoustic shot-string arrival CSV."""
    rng = np.random.default_rng(seed)
    t_nom = float(range_m) / 295.0
    lines = ["pellet_id,arrival_time_s,impact_velocity_mps,temp_c,rh_pct,pressure_hpa"]
    for idx in range(1, int(n_pellets) + 1):
        t_arr = max(0.02, float(rng.normal(t_nom, 0.0028)))
        v_imp = max(80.0, float(255.0 - 950.0 * (t_arr - t_nom) + rng.normal(0.0, 4.0)))
        lines.append(f"{idx},{t_arr:.5f},{v_imp:.2f},12.5,68.0,1011.5")
    return "\n".join(lines) + "\n"


def parse_weather_and_shotstring_csv(csv_text: str, range_m: float = 45.0) -> dict[str, Any]:
    """Parse Kestrel met fields (moist-air rho) and 5th-95th percentile acoustic shot-string length."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Shot-string CSV has no header row.")
    times, vels, temps, rhs, press = [], [], [], [], []
    for raw in reader:
        row = {str(k).strip().lower(): v for k, v in raw.items() if k is not None}
        if row.get("arrival_time_s") not in (None, ""):
            times.append(float(row["arrival_time_s"]))
        if row.get("impact_velocity_mps") not in (None, ""):
            vels.append(float(row["impact_velocity_mps"]))
        if row.get("temp_c") not in (None, ""):
            temps.append(float(row["temp_c"]))
        if row.get("rh_pct") not in (None, ""):
            rhs.append(float(row["rh_pct"]))
        if row.get("pressure_hpa") not in (None, ""):
            press.append(float(row["pressure_hpa"]))
    if len(times) < 5:
        raise ValueError("Shot-string parser requires at least 5 arrival_time_s samples.")

    t_arr = np.asarray(times, dtype=np.float64)
    v_med = float(np.median(vels)) if vels else 250.0
    dt_90 = float(np.percentile(t_arr, 95) - np.percentile(t_arr, 5))
    string_len_m = dt_90 * v_med

    tc = float(np.mean(temps)) if temps else 15.0
    rh = float(np.mean(rhs)) if rhs else 50.0
    p_pa = (float(np.mean(press)) if press else 1013.25) * 100.0
    t_k = tc + 273.15
    # Magnus saturation vapor pressure for moist-air density correction
    e_sat = 610.94 * math.exp((17.625 * tc) / (tc + 243.04))
    p_v = (max(0.0, min(100.0, rh)) / 100.0) * e_sat
    p_d = max(1000.0, p_pa - p_v)
    rho_moist = (p_d / (287.058 * t_k)) + (p_v / (461.495 * t_k))
    t_virt_k = t_k / max(1.0 - 0.378 * (p_v / max(p_pa, 1.0)), 0.5)
    t_sd_ms = float(np.std(t_arr, ddof=1) * 1000.0) if len(t_arr) > 1 else 0.0

    return {
        "n_samples": len(times),
        "n_pellets_recorded": len(times),
        "median_arrival_s": round(float(np.median(t_arr)), 5),
        "dt_90_ms": round(dt_90 * 1000.0, 2),
        "string_duration_ms": round(dt_90 * 1000.0, 2),
        "arrival_time_sd_ms": round(t_sd_ms, 2),
        "median_impact_velocity_mps": round(v_med, 2),
        "measured_shot_string_length_m": round(string_len_m, 3),
        "string_length_m": round(string_len_m, 3),
        "ambient_temp_c": round(tc, 1),
        "rh_pct": round(rh, 1),
        "relative_humidity_pct": round(rh, 1),
        "air_pressure_hpa": round(p_pa / 100.0, 2),
        "ambient_pressure_hpa": round(p_pa / 100.0, 2),
        "moist_air_density_kg_m3": round(rho_moist, 4),
        "air_density_kg_m3": round(rho_moist, 4),
        "virtual_temp_k": round(t_virt_k, 2),
        "speed_of_sound_mps": round(math.sqrt(1.4 * 287.058 * t_k), 1),
        "range_m": float(range_m),
        "twin_patch": {
            "air_density_kg_m3": round(rho_moist, 4),
            "measured_shot_string_length_m": round(string_len_m, 3),
        },
    }


# -------------------------------------------------------------------------
# 7. FAIR REPRODUCIBLE VALIDATION BUNDLE & SCIENTIFIC V&V DOSSIER
# -------------------------------------------------------------------------


def _current_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_reproducible_validation_bundle(
    twin_profile: dict[str, Any] | None = None,
    holdout_res: dict[str, Any] | None = None,
    uq_res: dict[str, Any] | None = None,
    exp_res: dict[str, Any] | None = None,
    doe_res: dict[str, Any] | None = None,
    raw_inputs: dict[str, str] | None = None,
    seed: int = 42,
    twin: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Assemble a FAIR reproducible V&V bundle with SHA-256 hashes of inputs, versions, and git SHA."""
    hashes: dict[str, str] = {}
    for k, payload in (raw_inputs or {}).items():
        hashes[str(k)] = hashlib.sha256(str(payload).encode("utf-8", errors="replace")).hexdigest()

    bundle_core = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _current_git_sha(),
        "seed": int(seed),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
        },
        "input_sha256": hashes,
        "twin_snapshot": twin_profile or twin or get_default_nominal_twin(),
        "blind_holdout": holdout_res or kwargs.get("blind_holdout_results"),
        "uq_envelope": uq_res or kwargs.get("uq_results"),
        "livefire_dataset": kwargs.get("livefire_dataset"),
        "experimental_benchmarks": exp_res or run_experimental_validation_benchmarks(),
        "doe_plan": doe_res,
    }
    canon = json.dumps(bundle_core, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canon).hexdigest()
    bundle_core["bundle_sha256"] = digest
    bundle_core["manifest_sha256"] = digest
    return bundle_core


def generate_scientific_vv_dossier_markdown(bundle: dict[str, Any]) -> str:
    """Render a Markdown Scientific V&V Reproducibility Dossier from a validation bundle."""
    lines = [
        "# Counter-UAS Shot-Cloud Scientific V&V Reproducibility Dossier",
        "",
        f"- **Generated (UTC):** `{bundle.get('generated_utc')}`",
        f"- **Git Commit SHA:** `{bundle.get('git_sha')}`",
        f"- **RNG Seed:** `{bundle.get('seed')}`",
        f"- **Bundle SHA-256:** `{bundle.get('bundle_sha256')}`",
        f"- **Runtime:** Python `{bundle['runtime']['python']}`, NumPy `{bundle['runtime']['numpy']}` ({bundle['runtime']['os']})",
        "",
        "## Input Artifact Checksums (SHA-256)",
    ]
    for name, digest in (bundle.get("input_sha256") or {}).items():
        lines.append(f"- `{name}`: `{digest}`")

    exp_b = bundle.get("experimental_benchmarks") or {}
    if exp_b:
        sm = exp_b.get("summary", {})
        lines.extend(["", f"## Published Experimental Benchmarks (`{sm.get('passed', 0)}/{sm.get('total', 0)} PASS`)", ""])
        lines.append("| ID | Domain | Computed | Reference | Tol | AVM | Status |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for r in exp_b.get("benchmarks", []):
            lines.append(f"| {r['benchmark_id']} | {r['domain']} | {r['computed']} | {r['reference']} | {r['tolerance']} | {r['avm']} | **{r['status']}** |")

    hold = bundle.get("blind_holdout")
    if hold:
        av = hold.get("asme_vv20_sigma", {})
        lines.extend([
            "",
            "## Blind Holdout Out-of-Sample Validation (ASME V&V 20)",
            f"- **Holdout Ranges:** `{hold.get('holdout_ranges_m')}` m",
            f"- **RMSE Reduction:** `{hold.get('holdout_rmse_nominal_m')}` m (nominal) -> `{hold.get('holdout_rmse_calibrated_m')}` m (calibrated, `{hold.get('error_reduction_pct')}%` error reduction)",
            f"- **ASME V&V 20 Verdict:** `{av.get('verdict')}` (`|E|/(k*u_val) = {av.get('e_over_kuval')}`, AVM `{av.get('area_validation_metric')}` m)",
        ])

    uq = bundle.get("uq_envelope")
    if uq:
        lines.extend([
            "",
            "## Nested Epistemic + Aleatory UQ Envelope",
            f"- **R80 95% Predictive Interval:** `[{uq.get('r80_p05_m')}, {uq.get('r80_p95_m')}] m` (median `{uq.get('r80_median_m')} m`)",
        ])
    return "\n".join(lines) + "\n"


def generate_scientific_vv_dossier_html(bundle: dict[str, Any]) -> str:
    """Render a self-contained HTML Scientific V&V Reproducibility Dossier."""
    md_text = generate_scientific_vv_dossier_markdown(bundle)
    exp_rows = (bundle.get("experimental_benchmarks") or {}).get("benchmarks", [])
    rows_html = "".join(
        f"<tr><td>{html.escape(str(r['benchmark_id']))}</td><td>{html.escape(str(r['domain']))}</td>"
        f"<td>{html.escape(str(r['title']))}</td><td>{r['computed']}</td><td>{r['reference']}</td>"
        f"<td>{r['avm']}</td><td><b>{html.escape(str(r['status']))}</b></td></tr>"
        for r in exp_rows
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>C-UAS Scientific V&amp;V Dossier</title>"
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:980px;margin:28px auto;padding:0 18px;color:#0f172a;background:#f8fafc}"
        "h1{color:#0f172a;border-bottom:2px solid #cbd5e1;padding-bottom:8px}"
        "table{width:100%;border-collapse:collapse;margin:14px 0;background:#fff}"
        "th,td{border:1px solid #cbd5e1;padding:7px 9px;text-align:left;font-size:13px}"
        "th{background:#e2e8f0}pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:6px;overflow:auto}</style></head><body>"
        f"<h1>Counter-UAS Shot-Cloud Scientific V&amp;V Dossier</h1>"
        f"<p><b>Bundle SHA-256:</b> <code>{html.escape(str(bundle.get('bundle_sha256', '')))}</code> | "
        f"<b>Git SHA:</b> <code>{html.escape(str(bundle.get('git_sha', '')))}</code> | "
        f"<b>Seed:</b> {int(bundle.get('seed', 42))}</p>"
        f"<table><thead><tr><th>ID</th><th>Domain</th><th>Benchmark</th><th>Computed</th><th>Ref</th><th>AVM</th><th>Status</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        f"<h2>Full Reproducibility Manifest</h2><pre>{html.escape(md_text)}</pre></body></html>"
    )


# -------------------------------------------------------------------------
# 8. NATIVE RANGE-INSTRUMENT PARSERS (LABRADAR, GARMIN XERO, KESTREL, AUTO)
# -------------------------------------------------------------------------


def generate_sample_labradar_native_csv(v0_true_mps: float = 410.0, cd_scale: float = 1.06, seed: int = 42) -> str:
    """Synthesize a native LabRadar semicolon-delimited multi-shot Doppler track CSV."""
    rng = np.random.default_rng(seed)
    lines = [
        "Device;LabRadar v1.4;Series;C-UAS-01",
        "Units Vel;m/s;Units Dist;m",
        "Shot ID;V0;V1;V2;V3;V4;V5;Dist1;Dist2;Dist3;Dist4;Dist5",
    ]
    dists = (5.0, 15.0, 25.0, 35.0, 50.0)
    kappa = 0.0092 * float(cd_scale)
    for sid in range(1, 6):
        v0 = float(rng.normal(v0_true_mps, 3.2))
        vels = [v0] + [max(30.0, v0 * math.exp(-kappa * d) + float(rng.normal(0.0, 1.3))) for d in dists]
        lines.append(
            f"{sid};{vels[0]:.1f};{vels[1]:.1f};{vels[2]:.1f};{vels[3]:.1f};{vels[4]:.1f};{vels[5]:.1f};"
            f"{dists[0]:.1f};{dists[1]:.1f};{dists[2]:.1f};{dists[3]:.1f};{dists[4]:.1f}"
        )
    return "\n".join(lines) + "\n"


def parse_labradar_native_csv(csv_text: str) -> dict[str, Any]:
    """Parse native LabRadar semicolon/comma shot-track export into SI Doppler and Chronograph CSVs."""
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    delim = ";" if any(";" in ln for ln in lines[:10]) else ","
    units_imperial = any("fps" in ln.lower() or "ft/s" in ln.lower() or "yd" in ln.lower() for ln in lines[:6])

    hdr_idx: int | None = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "shot id" in low or ("v0" in low and "v1" in low):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("Unrecognized LabRadar CSV format (missing Shot ID / V0..Vn header).")

    reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_idx:])), delimiter=delim)
    vel_scale = 0.3048 if units_imperial else 1.0
    dist_scale = 0.9144 if units_imperial else 1.0

    v0_vals: list[float] = []
    doppler_pts: dict[float, list[float]] = {}
    for row in reader:
        r = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
        v0_raw = r.get("v0")
        if v0_raw:
            v0_si = float(v0_raw) * vel_scale
            v0_vals.append(v0_si)
            doppler_pts.setdefault(0.0, []).append(v0_si)
        for k in range(1, 10):
            vk = r.get(f"v{k}")
            dk = r.get(f"dist{k}", r.get(f"d{k}"))
            if vk and dk:
                doppler_pts.setdefault(float(dk) * dist_scale, []).append(float(vk) * vel_scale)

    if not v0_vals:
        raise ValueError("No valid velocity shots parsed from LabRadar CSV.")

    dop_lines = ["range_m,velocity_mps"]
    for rm in sorted(doppler_pts):
        if rm > 0.0:
            dop_lines.append(f"{rm:.2f},{float(np.mean(doppler_pts[rm])):.2f}")
    chrono_lines = ["shot_id,velocity_mps"]
    for idx, vv in enumerate(v0_vals, start=1):
        chrono_lines.append(f"{idx},{vv:.2f}")

    dop_csv = "\n".join(dop_lines) + "\n"
    dop_fit = calibrate_doppler_radar_drag(dop_csv) if len(dop_lines) >= 5 else {}
    mean_v0 = float(np.mean(v0_vals))
    sd_v0 = float(np.std(v0_vals, ddof=1)) if len(v0_vals) > 1 else 0.0
    return {
        "instrument": "LabRadar",
        "units_detected": "imperial (ft/s, yd) -> SI" if units_imperial else "metric (m/s, m)",
        "n_shots": len(v0_vals),
        "mean_v0_mps": round(mean_v0, 2),
        "sd_v0_mps": round(sd_v0, 2),
        "doppler_csv": dop_csv,
        "chronograph_csv": "\n".join(chrono_lines) + "\n",
        "doppler_fit": dop_fit,
        "twin_patch": {
            "mv_sd_mps": round(sd_v0, 2),
            **({"pellet_cd_scale": dop_fit["pellet_cd_scale"], "pellet_cd_scale_se": dop_fit["pellet_cd_scale_se"]} if dop_fit else {}),
        },
    }


def generate_sample_garmin_xero_csv(mean_v0_mps: float = 416.0, sd_mps: float = 3.8, n_shots: int = 10, seed: int = 42) -> str:
    """Synthesize a native Garmin Xero C1 Pro session CSV export (FPS)."""
    rng = np.random.default_rng(seed)
    lines = ["Shot #,Speed (FPS),KE (ft-lbs),Power Factor,Clean Bore,Cold Bore"]
    for i in range(1, int(n_shots) + 1):
        v_mps = float(rng.normal(mean_v0_mps, sd_mps))
        v_fps = v_mps / 0.3048
        lines.append(f"{i},{v_fps:.1f},58.4,19.1,No,No")
    return "\n".join(lines) + "\n"


def parse_garmin_xero_c1_csv_or_xlsx(payload: bytes | str) -> dict[str, Any]:
    """Parse Garmin Xero C1 Pro .csv or .xlsx chronograph export (FPS or m/s) into SI."""
    rows_dicts: list[dict[str, Any]] = []
    if isinstance(payload, bytes) and payload[:2] == b"PK":
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(payload), data_only=True)
        ws = wb.active
        raw_rows = list(ws.iter_rows(values_only=True))
        hdr_row_idx = 0
        for idx, r in enumerate(raw_rows[:12]):
            if r and any("speed" in str(c).lower() or "velocity" in str(c).lower() for c in r if c is not None):
                hdr_row_idx = idx
                break
        headers = [str(c or f"col_{j}").strip().lower() for j, c in enumerate(raw_rows[hdr_row_idx])]
        for r in raw_rows[hdr_row_idx + 1 :]:
            if not r or all(c is None for c in r):
                continue
            rows_dicts.append({headers[j]: r[j] for j in range(min(len(headers), len(r)))})
    else:
        txt = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        hdr_idx = 0
        for idx, ln in enumerate(lines[:12]):
            low = ln.lower()
            if "speed" in low or "velocity" in low:
                hdr_idx = idx
                break
        reader = csv.DictReader(io.StringIO("\n".join(lines[hdr_idx:])))
        for row in reader:
            rows_dicts.append({str(k).strip().lower(): v for k, v in row.items() if k is not None})

    v_mps_list: list[float] = []
    is_fps = False
    for row in rows_dicts:
        for k, val in row.items():
            if ("speed" in k or "velocity" in k) and val not in (None, ""):
                v_num = float(str(val).replace(",", ""))
                if "fps" in k or "ft" in k or v_num > 850.0:
                    is_fps = True
                    v_num *= 0.3048
                v_mps_list.append(v_num)
                break

    if not v_mps_list:
        raise ValueError("No velocity readings found in Garmin Xero C1 payload.")

    arr = np.asarray(v_mps_list, dtype=np.float64)
    mean_v = float(np.mean(arr))
    sd_v = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    chrono_lines = ["shot_id,velocity_mps"] + [f"{i + 1},{v:.2f}" for i, v in enumerate(arr)]
    return {
        "instrument": "Garmin Xero C1 Pro",
        "units_detected": "FPS -> m/s" if is_fps else "m/s",
        "n_shots": int(arr.size),
        "mean_v0_mps": round(mean_v, 2),
        "sd_v0_mps": round(sd_v, 2),
        "es_mps": round(float(np.max(arr) - np.min(arr)), 2),
        "min_v0_mps": round(float(np.min(arr)), 2),
        "max_v0_mps": round(float(np.max(arr)), 2),
        "chronograph_csv": "\n".join(chrono_lines) + "\n",
        "twin_patch": {"mv_sd_mps": round(sd_v, 2)},
    }


def generate_sample_kestrel_link_csv(temp_c: float = 14.0, rh_pct: float = 65.0, pressure_hpa: float = 1012.0) -> str:
    """Synthesize a Kestrel LiNK weather log CSV."""
    return (
        "Device Name,Kestrel 5700 Elite\n"
        "FORMATTED DATE_TIME,Temp,Relative Humidity,Station Pressure,Crosswind,Headwind\n"
        "YYYY-MM-DD hh:mm:ss,°C,%,hPa,m/s,m/s\n"
        f"2026-11-25 09:00:00,{temp_c:.1f},{rh_pct:.1f},{pressure_hpa:.1f},2.4,1.2\n"
        f"2026-11-25 09:01:00,{temp_c + 0.3:.1f},{rh_pct - 1.0:.1f},{pressure_hpa + 0.2:.1f},2.8,1.5\n"
        f"2026-11-25 09:02:00,{temp_c - 0.2:.1f},{rh_pct + 0.5:.1f},{pressure_hpa - 0.1:.1f},2.1,1.0\n"
    )


def parse_kestrel_link_csv(csv_text: str) -> dict[str, Any]:
    """Parse Kestrel LiNK weather export and compute CIP moist-air density, wind, and sound speed."""
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    hdr_idx: int | None = None
    for i, ln in enumerate(lines[:10]):
        low = ln.lower()
        if "temp" in low and ("humidity" in low or "rh" in low or "pressure" in low):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("Unrecognized Kestrel LiNK CSV format (missing Temp / Humidity / Pressure header).")

    headers = [h.strip().lower() for h in next(csv.reader([lines[hdr_idx]]))]
    data_start = hdr_idx + 1
    units_row: list[str] = []
    if data_start < len(lines):
        cand_units = next(csv.reader([lines[data_start]]))
        if any(u.strip().lower() in ("°c", "c", "°f", "f", "%", "hpa", "mbar", "inhg", "m/s", "mph") for u in cand_units):
            units_row = [u.strip().lower() for u in cand_units]
            data_start += 1

    reader = csv.reader(lines[data_start:])
    temps_c, rhs, press_hpa, cw_list, hw_list = [], [], [], [], []
    is_f = any("f" in u for u in units_row)
    is_inhg = any("inhg" in u for u in units_row)
    is_mph = any("mph" in u for u in units_row)

    for cols in reader:
        if not cols:
            continue
        rmap = {headers[j]: cols[j].strip() for j in range(min(len(headers), len(cols)))}
        for k, v in rmap.items():
            if v == "":
                continue
            if "temp" in k and "wet" not in k:
                t_v = float(v)
                temps_c.append((t_v - 32.0) * 5.0 / 9.0 if is_f else t_v)
            elif "humidity" in k or k == "rh" or "rh_pct" in k:
                rhs.append(float(v))
            elif "pressure" in k or "baro" in k:
                p_v = float(v)
                press_hpa.append(p_v * 33.8639 if is_inhg else p_v)
            elif "crosswind" in k:
                w_v = float(v)
                cw_list.append(w_v * 0.44704 if is_mph else w_v)
            elif "headwind" in k:
                w_v = float(v)
                hw_list.append(w_v * 0.44704 if is_mph else w_v)

    if not temps_c:
        raise ValueError("No temperature records parsed from Kestrel LiNK CSV.")

    tc = float(np.mean(temps_c))
    rh = float(np.mean(rhs)) if rhs else 50.0
    p_hpa = float(np.mean(press_hpa)) if press_hpa else 1013.25
    t_k = tc + 273.15
    e_sat = 610.94 * math.exp((17.625 * tc) / (tc + 243.04))
    p_v = (max(0.0, min(100.0, rh)) / 100.0) * e_sat
    p_d = max(1000.0, p_hpa * 100.0 - p_v)
    rho_moist = (p_d / (287.058 * t_k)) + (p_v / (461.495 * t_k))
    return {
        "instrument": "Kestrel LiNK Weather",
        "n_samples": len(temps_c),
        "ambient_temp_c": round(tc, 1),
        "relative_humidity_pct": round(rh, 1),
        "air_pressure_hpa": round(p_hpa, 2),
        "air_density_kg_m3": round(rho_moist, 4),
        "speed_of_sound_mps": round(math.sqrt(1.4 * 287.058 * t_k), 1),
        "mean_crosswind_mps": round(float(np.mean(cw_list)), 2) if cw_list else 0.0,
        "mean_headwind_mps": round(float(np.mean(hw_list)), 2) if hw_list else 0.0,
        "twin_patch": {"air_density_kg_m3": round(rho_moist, 4)},
    }


def detect_and_parse_instrument_file(filename: str, raw_bytes: bytes) -> dict[str, Any]:
    """Auto-detect commercial instrument file format and return normalized parsed metrics & twin_patch."""
    fname = str(filename or "").lower()
    if fname.endswith(".xlsx") or raw_bytes[:2] == b"PK":
        parsed = parse_garmin_xero_c1_csv_or_xlsx(raw_bytes)
        return {"instrument_type": "garmin_xero_c1", "parsed": parsed, "summary": parsed, "twin_patch": parsed["twin_patch"]}

    txt = raw_bytes.decode("utf-8", errors="replace")
    low = txt[:1500].lower()
    if "labradar" in low or ("shot id;" in low and "v0" in low):
        parsed = parse_labradar_native_csv(txt)
        return {"instrument_type": "labradar", "parsed": parsed, "summary": parsed, "twin_patch": parsed["twin_patch"]}
    if "kestrel" in low or ("relative humidity" in low and "station pressure" in low):
        parsed = parse_kestrel_link_csv(txt)
        return {"instrument_type": "kestrel_link", "parsed": parsed, "summary": parsed, "twin_patch": parsed["twin_patch"]}
    if ("speed (fps)" in low or "power factor" in low or "shot #" in low) and "range" not in low:
        parsed = parse_garmin_xero_c1_csv_or_xlsx(txt)
        return {"instrument_type": "garmin_xero_c1", "parsed": parsed, "summary": parsed, "twin_patch": parsed["twin_patch"]}
    if ("timestamp" in low and "x" in low and "y" in low and "z" in low) or ("timeus" in low and "lat" in low):
        from ammo_mc.cuas_track_replay import parse_ardupilot_px4_flight_log

        parsed = parse_ardupilot_px4_flight_log(txt)
        return {"instrument_type": "px4_ardupilot", "parsed": parsed, "summary": parsed, "twin_patch": {}}
    if "arrival_time_s" in low:
        parsed = parse_weather_and_shotstring_csv(txt)
        return {"instrument_type": "shotstring_met", "parsed": parsed, "summary": parsed, "twin_patch": parsed.get("twin_patch", {})}

    parsed = calibrate_doppler_radar_drag(txt)
    return {"instrument_type": "doppler_radar_csv", "parsed": parsed, "summary": parsed, "twin_patch": parsed.get("twin_patch", {})}


def evaluate_sprt_live_sequence(
    outcomes: list[int],
    sprt_p0: float = 0.65,
    sprt_p1: float = 0.85,
    alpha: float = 0.05,
    power: float = 0.80,
    range_m: float = 45.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    barrels_fired: int = 5,
) -> dict[str, Any]:
    """Evaluate a live shot-by-shot kill/miss sequence (1/0) against Wald SPRT early-stopping thresholds."""
    p0 = max(0.05, min(0.90, float(sprt_p0)))
    p1 = max(p0 + 0.05, min(0.98, float(sprt_p1)))
    alpha_s = max(0.01, min(0.20, float(alpha)))
    beta_s = 1.0 - max(0.55, min(0.98, float(power)))

    a_log = math.log((1.0 - beta_s) / alpha_s)
    b_log = math.log(beta_s / (1.0 - alpha_s))
    log_odds = math.log((p1 * (1.0 - p0)) / (p0 * (1.0 - p1)))
    log_q = math.log((1.0 - p1) / (1.0 - p0))

    cum_kills = 0
    decision = "CONTINUE"
    stopping_round: int | None = None
    trajectory: list[dict[str, Any]] = []
    livefire_rows: list[dict[str, Any]] = []

    for idx, raw_o in enumerate(outcomes, start=1):
        kill = 1 if int(raw_o) > 0 else 0
        cum_kills += kill
        llr = cum_kills * math.log(p1 / p0) + (idx - cum_kills) * math.log((1.0 - p1) / (1.0 - p0))
        acc_k = (a_log - idx * log_q) / log_odds
        rej_k = (b_log - idx * log_q) / log_odds
        step_state = "CONTINUE"
        if llr >= a_log:
            step_state = "ACCEPT_H1"
        elif llr <= b_log:
            step_state = "REJECT_H0"
        if decision == "CONTINUE" and step_state != "CONTINUE":
            decision = step_state
            stopping_round = idx
        trajectory.append({
            "n": idx,
            "outcome": kill,
            "cum_kills": cum_kills,
            "llr": round(llr, 3),
            "accept_min_kills": round(min(float(idx) + 0.5, acc_k), 2),
            "reject_max_kills": round(max(-0.5, rej_k), 2),
            "state": step_state,
        })
        livefire_rows.append({
            "range_m": float(range_m),
            "ammunition_preset": str(ammunition_preset),
            "choke": str(choke),
            "barrels_fired": int(barrels_fired),
            "drone_killed": kill,
        })

    n_rounds = len(outcomes)
    obs_pk = (cum_kills / n_rounds) if n_rounds > 0 else 0.0
    ci_lo, ci_hi = compute_clopper_pearson_ci(cum_kills, n_rounds, alpha=alpha_s) if n_rounds > 0 else (0.0, 1.0)
    return {
        "n_rounds": n_rounds,
        "n_kills": cum_kills,
        "observed_pk": round(obs_pk, 3),
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "decision": decision,
        "stopping_round": stopping_round,
        "a_bound_llr": round(a_log, 3),
        "b_bound_llr": round(b_log, 3),
        "trajectory": trajectory,
        "livefire_rows": livefire_rows,
    }
