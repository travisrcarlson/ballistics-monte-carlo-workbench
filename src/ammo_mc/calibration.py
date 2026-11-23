"""Empirical Field-Test Calibration & Residual Diagnostics Engine.

Fits ballistics model parameters against measured range CSV data using SciPy nonlinear
least squares (`scipy.optimize.least_squares` / Levenberg-Marquardt & Trust-Region):
1. Shotgun Patterning Board Calibration:
   - Point-of-Impact (POI) windage & elevation bias (m & mrad)
   - Rayleigh radial dispersion scale fitting (`k_choke_cal = sigma_fit / sigma_nominal`)
   - Kolmogorov-Smirnov goodness-of-fit test & radial CDF residuals
2. Chronograph Muzzle Velocity & Propellant Temperature Lot Calibration:
   - Mean MV, lot SD, Extreme Spread (ES), bias vs nominal
   - Optional FM 6-40 / Table E linear temperature coefficient V(T) = V_std + k_T*(T - T_std)
3. Exterior Ballistics Drag Form Factor (`i`) Fitting:
   - Calls vectorized trajectory integrator vs measured range drop / Doppler velocity stations
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np
from scipy import optimize as sp_opt
from scipy import stats as sp_stats

from ammo_mc.environment import AtmosphereEnvironment
from ammo_mc.exterior import BatchTrajectoryConfig, integrate_trajectories_batch
from ammo_mc.presets import get_preset
from ammo_mc.small_arms import get_choke_definitions, get_shotgun_presets


def generate_sample_pattern_csv(
    range_m: float = 36.58,
    nominal_choke: str = "Modified",
    true_spread_multiplier: float = 1.22,
    bias_y_m: float = 0.06,
    bias_z_m: float = -0.04,
    n_pellets: int = 108,
    seed: int = 7,
) -> str:
    """Generate a realistic synthetic 2-volley patterning-board pellet CSV for calibration testing."""
    rng = np.random.default_rng(seed)
    cdef = get_choke_definitions().get(nominal_choke, get_choke_definitions()["Modified"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(cdef["benchmark_40yd_30in_pct"])))
    sigma_true = sigma_ref * (float(range_m) / 36.58) ** 1.15 * float(true_spread_multiplier)
    ys = rng.normal(bias_y_m, sigma_true, size=n_pellets)
    zs = rng.normal(bias_z_m, sigma_true, size=n_pellets)
    lines = ["pellet_id,y_m,z_m"]
    for idx, (yv, zv) in enumerate(zip(ys, zs), start=1):
        lines.append(f"{idx},{yv:.4f},{zv:.4f}")
    return "\n".join(lines) + "\n"


def generate_sample_chronograph_csv(
    v_std_mps: float = 412.0,
    dv_dt_mps_per_c: float = 0.72,
    sd_mps: float = 4.2,
    seed: int = 42,
) -> str:
    """Generate a realistic field chronograph log CSV across three propellant temperatures."""
    rng = np.random.default_rng(seed)
    temps = [5.0] * 8 + [21.1] * 8 + [38.0] * 8
    lines = ["shot_id,temp_c,velocity_mps"]
    for idx, tc in enumerate(temps, start=1):
        v_mean = v_std_mps + dv_dt_mps_per_c * (tc - 21.1)
        v_meas = float(rng.normal(v_mean, sd_mps))
        lines.append(f"{idx},{tc:.1f},{v_meas:.2f}")
    return "\n".join(lines) + "\n"


def generate_sample_range_drop_csv(
    preset_id: str = "NOTIONAL-SA-01",
    true_form_factor: float = 1.18,
    ranges_m: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0),
    seed: int = 42,
) -> str:
    """Generate synthetic field range drop & Doppler velocity measurements for a preset."""
    ammo, scn = get_preset(preset_id)
    rng = np.random.default_rng(seed)
    env = AtmosphereEnvironment()
    mv = float(ammo.launch.mv.mu)
    max_r = max(ranges_m)
    max_t = max(2.5, 2.2 * max_r / max(mv * 0.45, 80.0))
    cfg = BatchTrajectoryConfig(
        calibre_m=ammo.calibre_m,
        drag_ref=ammo.aero.drag_ref,
        custom_cd_table=ammo.aero.cd_table,
        fidelity="L1",
        include_coriolis=False,
        dt=0.004,
        max_time_s=max_t,
    )
    st0 = np.array([[0.0, 0.0, 0.0, mv, 0.0, 0.0]], dtype=np.float64)
    res = integrate_trajectories_batch(
        st0,
        np.array([ammo.mass_kg], dtype=np.float64),
        np.array([float(ammo.aero.form_factor.mu) * true_form_factor], dtype=np.float64),
        env,
        cfg,
        terrain_fn=lambda x, _y: np.full_like(x, -2000.0),
        record_single_path=True,
    )
    path = res["sample_path"]
    assert path is not None
    r_arr = np.asarray(ranges_m, dtype=np.float64)
    drops = np.interp(r_arr, path["range_m"], path["alt_m"]) + rng.normal(0.0, 0.006, size=r_arr.size)
    vels = np.interp(r_arr, path["range_m"], path["speed_mps"]) + rng.normal(0.0, 0.8, size=r_arr.size)

    lines = ["range_m,drop_m,velocity_mps"]
    for rv, dv, vv in zip(r_arr, drops, vels):
        lines.append(f"{rv:.1f},{dv:.4f},{vv:.2f}")
    return "\n".join(lines) + "\n"


def calibrate_shotgun_pattern_from_csv(
    csv_text: str,
    range_m: float = 36.58,
    nominal_choke: str = "Modified",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
) -> dict[str, Any]:
    """Calibrate Rayleigh pattern sigma, choke spread multiplier, and POI bias from CSV hits."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Pattern CSV has no header row.")
    fmap = {f.strip().lower(): f for f in reader.fieldnames}

    scale = 1.0
    if "y_m" in fmap and "z_m" in fmap:
        cy, cz, scale = fmap["y_m"], fmap["z_m"], 1.0
    elif "x_m" in fmap and "y_m" in fmap:
        cy, cz, scale = fmap["x_m"], fmap["y_m"], 1.0
    elif "y_cm" in fmap and "z_cm" in fmap:
        cy, cz, scale = fmap["y_cm"], fmap["z_cm"], 0.01
    elif "y_in" in fmap and "z_in" in fmap:
        cy, cz, scale = fmap["y_in"], fmap["z_in"], 0.0254
    elif "x_in" in fmap and "y_in" in fmap:
        cy, cz, scale = fmap["x_in"], fmap["y_in"], 0.0254
    else:
        raise ValueError("Pattern CSV requires `(y_m, z_m)`, `(y_cm, z_cm)`, or `(y_in, z_in)` columns.")

    ys: list[float] = []
    zs: list[float] = []
    for row in reader:
        try:
            yv = float(row[cy]) * scale
            zv = float(row[cz]) * scale
            if math.isfinite(yv) and math.isfinite(zv):
                ys.append(yv)
                zs.append(zv)
        except (TypeError, ValueError):
            continue

    if len(ys) < 5:
        raise ValueError("Need at least 5 pellet impact coordinates to fit Rayleigh dispersion.")

    y_arr = np.asarray(ys, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)
    n_pts = len(y_arr)
    bias_y = float(np.mean(y_arr))
    bias_z = float(np.mean(z_arr))
    radii = np.hypot(y_arr - bias_y, z_arr - bias_z)
    r_sorted = np.sort(radii)

    sigma_mle = float(math.sqrt(float(np.mean(r_sorted * r_sorted)) / 2.0))
    ecdf = (np.arange(1, n_pts + 1, dtype=np.float64) - 0.5) / float(n_pts)

    def _res_rayleigh(params: np.ndarray) -> np.ndarray:
        s = max(float(params[0]), 1e-4)
        pred_cdf = 1.0 - np.exp(-(r_sorted ** 2) / (2.0 * s * s))
        return pred_cdf - ecdf

    ls_res = sp_opt.least_squares(_res_rayleigh, x0=[max(sigma_mle, 0.02)], bounds=([1e-3], [10.0]))
    sigma_fit = float(ls_res.x[0])

    choke_defs = get_choke_definitions()
    c_nom = choke_defs.get(nominal_choke, choke_defs["Modified"])
    sigma_40yd_nom = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_nom["benchmark_40yd_30in_pct"])))
    sigma_nom_at_range = sigma_40yd_nom * (float(range_m) / 36.58) ** 1.15
    spread_factor = sigma_fit / max(sigma_nom_at_range, 1e-6)

    # Back-project fitted dispersion to 40-yard 30-inch (0.381 m radius) benchmark
    sigma_fit_40yd = sigma_fit / max((float(range_m) / 36.58) ** 1.15, 1e-6)
    p30_fit = 1.0 - math.exp(-(0.381 ** 2) / (2.0 * max(sigma_fit_40yd, 1e-6) ** 2))
    closest_choke = min(
        choke_defs.keys(),
        key=lambda ck: abs(float(choke_defs[ck]["benchmark_40yd_30in_pct"]) - p30_fit),
    )

    pred_fit_cdf = 1.0 - np.exp(-(r_sorted ** 2) / (2.0 * sigma_fit * sigma_fit))
    pred_nom_cdf = 1.0 - np.exp(-(r_sorted ** 2) / (2.0 * sigma_nom_at_range * sigma_nom_at_range))
    cdf_resid = ecdf - pred_fit_cdf
    rmse_cdf = float(math.sqrt(float(np.mean(cdf_resid ** 2))))
    ss_tot = float(np.sum((ecdf - np.mean(ecdf)) ** 2))
    r2_cdf = float(1.0 - np.sum(cdf_resid ** 2) / max(ss_tot, 1e-12))

    ks_res = sp_stats.kstest(r_sorted, "rayleigh", args=(0.0, sigma_fit))
    bias_y_mrad = math.degrees(math.atan2(bias_y, max(range_m, 1.0))) * 17.4532925199
    bias_z_mrad = math.degrees(math.atan2(bias_z, max(range_m, 1.0))) * 17.4532925199

    return {
        "range_m": float(range_m),
        "nominal_choke": nominal_choke,
        "ammunition_preset": ammunition_preset,
        "n_pellets": n_pts,
        "poi_bias_y_m": round(bias_y, 4),
        "poi_bias_z_m": round(bias_z, 4),
        "poi_bias_y_mrad": round(bias_y_mrad, 2),
        "poi_bias_z_mrad": round(bias_z_mrad, 2),
        "sigma_mle_m": round(sigma_mle, 4),
        "sigma_fit_m": round(sigma_fit, 4),
        "sigma_nominal_m": round(sigma_nom_at_range, 4),
        "calibrated_spread_factor": round(spread_factor, 3),
        "fitted_40yd_30in_pct": round(p30_fit * 100.0, 1),
        "closest_standard_choke": closest_choke,
        "r50_fit_m": round(sigma_fit * math.sqrt(2.0 * math.log(2.0)), 3),
        "r90_fit_m": round(sigma_fit * math.sqrt(2.0 * math.log(10.0)), 3),
        "r2_cdf": round(r2_cdf, 4),
        "rmse_cdf": round(rmse_cdf, 4),
        "ks_stat": round(float(ks_res.statistic), 4),
        "ks_pvalue": round(float(ks_res.pvalue), 4),
        "y_coords_m": [round(float(v), 4) for v in y_arr],
        "z_coords_m": [round(float(v), 4) for v in z_arr],
        "r_sorted_m": [round(float(v), 4) for v in r_sorted],
        "ecdf": [round(float(v), 4) for v in ecdf],
        "pred_fit_cdf": [round(float(v), 4) for v in pred_fit_cdf],
        "pred_nom_cdf": [round(float(v), 4) for v in pred_nom_cdf],
        "cdf_residuals": [round(float(v), 4) for v in cdf_resid],
    }


def calibrate_chronograph_from_csv(
    csv_text: str,
    nominal_mv_mps: float | None = None,
    std_temp_c: float = 21.1,
) -> dict[str, Any]:
    """Fit mean muzzle velocity, lot SD, and optional FM 6-40 dMV/dT slope from chronograph CSV."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Chronograph CSV has no header row.")
    fmap = {f.strip().lower(): f for f in reader.fieldnames}

    v_col = fmap.get("velocity_mps") or fmap.get("mv_mps")
    v_scale = 1.0
    if v_col is None:
        v_col = fmap.get("velocity_fps") or fmap.get("mv_fps")
        v_scale = 0.3048
    if v_col is None:
        raise ValueError("Chronograph CSV requires `velocity_mps` (or `velocity_fps`).")

    t_col = fmap.get("temp_c") or fmap.get("temperature_c")
    t_is_f = False
    if t_col is None:
        t_col = fmap.get("temp_f") or fmap.get("temperature_f")
        t_is_f = t_col is not None

    vels: list[float] = []
    temps: list[float] = []
    for row in reader:
        try:
            vv = float(row[v_col]) * v_scale
            if not math.isfinite(vv):
                continue
            vels.append(vv)
            if t_col is not None and row.get(t_col, "") != "":
                tv = float(row[t_col])
                temps.append((tv - 32.0) * (5.0 / 9.0) if t_is_f else tv)
        except (TypeError, ValueError):
            continue

    if len(vels) < 3:
        raise ValueError("Chronograph CSV needs at least 3 valid velocity measurements.")

    v_arr = np.asarray(vels, dtype=np.float64)
    mean_mv = float(np.mean(v_arr))
    sd_mv = float(np.std(v_arr, ddof=1))
    es_mv = float(np.ptp(v_arr))
    mv_bias = (mean_mv - float(nominal_mv_mps)) if nominal_mv_mps is not None else 0.0

    has_temp_fit = len(temps) == len(vels) and float(np.ptp(temps)) > 2.0
    dv_dt = 0.0
    v_std_fit = mean_mv
    residuals = v_arr - mean_mv
    r2_temp = 0.0
    if has_temp_fit:
        t_arr = np.asarray(temps, dtype=np.float64)
        dt_arr = t_arr - float(std_temp_c)

        def _res_temp(p: np.ndarray) -> np.ndarray:
            return (p[0] + p[1] * dt_arr) - v_arr

        fit = sp_opt.least_squares(_res_temp, x0=[mean_mv, 0.5])
        v_std_fit = float(fit.x[0])
        dv_dt = float(fit.x[1])
        v_pred = v_std_fit + dv_dt * dt_arr
        residuals = v_arr - v_pred
        ss_tot = float(np.sum((v_arr - mean_mv) ** 2))
        r2_temp = float(1.0 - np.sum(residuals ** 2) / max(ss_tot, 1e-12))

    resid_sd = float(np.std(residuals, ddof=2 if has_temp_fit else 1))
    return {
        "n_shots": len(vels),
        "mean_mv_mps": round(mean_mv, 2),
        "v_std_fit_mps": round(v_std_fit, 2),
        "sd_mps": round(sd_mv, 2),
        "residual_sd_mps": round(resid_sd, 2),
        "extreme_spread_mps": round(es_mv, 2),
        "min_mv_mps": round(float(np.min(v_arr)), 2),
        "max_mv_mps": round(float(np.max(v_arr)), 2),
        "mv_bias_mps": round(mv_bias, 2),
        "has_temp_fit": bool(has_temp_fit),
        "std_temp_c": float(std_temp_c),
        "dv_dt_mps_per_c": round(dv_dt, 4),
        "r2_temp_fit": round(r2_temp, 4),
        "velocities_mps": [round(float(v), 2) for v in v_arr],
        "temps_c": [round(float(t), 2) for t in temps] if has_temp_fit else [],
        "residuals_mps": [round(float(r), 2) for r in residuals],
    }


def calibrate_drag_form_factor_from_csv(
    csv_text: str,
    preset_id: str = "NOTIONAL-SA-01",
    fit_mv: bool = False,
) -> dict[str, Any]:
    """Fit optimal drag form factor multiplier (and optional MV scale) against field drop/velocity CSV."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Range calibration CSV has no header row.")
    fmap = {f.strip().lower(): f for f in reader.fieldnames}

    r_col = fmap.get("range_m")
    if r_col is None:
        raise ValueError("Range calibration CSV requires a `range_m` column.")

    d_col = fmap.get("drop_m") or fmap.get("drop_cm") or fmap.get("drop_in")
    d_scale = 0.01 if fmap.get("drop_cm") else (0.0254 if fmap.get("drop_in") else 1.0)
    v_col = fmap.get("velocity_mps") or fmap.get("velocity_fps")
    v_scale = 0.3048 if fmap.get("velocity_fps") and not fmap.get("velocity_mps") else 1.0

    if d_col is None and v_col is None:
        raise ValueError("Range calibration CSV needs `drop_m` (or cm/in) and/or `velocity_mps` (or fps).")

    rs: list[float] = []
    drops_obs: list[float] = []
    vels_obs: list[float] = []
    for row in reader:
        try:
            rv = float(row[r_col])
            if not math.isfinite(rv) or rv <= 0.0:
                continue
            dv = float(row[d_col]) * d_scale if (d_col and row.get(d_col, "") != "") else float("nan")
            vv = float(row[v_col]) * v_scale if (v_col and row.get(v_col, "") != "") else float("nan")
            rs.append(rv)
            drops_obs.append(dv)
            vels_obs.append(vv)
        except (TypeError, ValueError):
            continue

    if len(rs) < 3:
        raise ValueError("Need at least 3 valid downrange observation stations to fit form factor.")

    order = np.argsort(rs)
    r_arr = np.asarray(rs, dtype=np.float64)[order]
    d_arr = np.asarray(drops_obs, dtype=np.float64)[order]
    v_arr = np.asarray(vels_obs, dtype=np.float64)[order]
    has_drop = bool(np.all(np.isfinite(d_arr)))
    has_vel = bool(np.all(np.isfinite(v_arr)))

    ammo, _scn = get_preset(preset_id)
    env = AtmosphereEnvironment()
    mv_nom = float(ammo.launch.mv.mu)
    ff_nom = float(ammo.aero.form_factor.mu)
    max_t = max(2.5, 2.5 * float(np.max(r_arr)) / max(mv_nom * 0.4, 80.0))
    cfg = BatchTrajectoryConfig(
        calibre_m=ammo.calibre_m,
        drag_ref=ammo.aero.drag_ref,
        custom_cd_table=ammo.aero.cd_table,
        fidelity="L1",
        include_coriolis=False,
        dt=0.004,
        max_time_s=max_t,
    )
    ground_sink = lambda x, _y: np.full_like(x, -2000.0)
    m_arr = np.array([ammo.mass_kg], dtype=np.float64)

    def _eval_model(ff_mult: float, mv_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        mv_eff = mv_nom * mv_mult
        st0 = np.array([[0.0, 0.0, 0.0, mv_eff, 0.0, 0.0]], dtype=np.float64)
        sim = integrate_trajectories_batch(
            st0,
            m_arr,
            np.array([ff_nom * ff_mult], dtype=np.float64),
            env,
            cfg,
            terrain_fn=ground_sink,
            record_single_path=True,
        )
        p = sim["sample_path"]
        assert p is not None
        d_pred = np.interp(r_arr, p["range_m"], p["alt_m"])
        v_pred = np.interp(r_arr, p["range_m"], p["speed_mps"])
        return d_pred, v_pred

    d_nom_pred, v_nom_pred = _eval_model(1.0, 1.0)

    def _residual_vec(x: np.ndarray) -> np.ndarray:
        ff_m = float(x[0])
        mv_m = float(x[1]) if fit_mv else 1.0
        dp, vp = _eval_model(ff_m, mv_m)
        parts: list[np.ndarray] = []
        if has_drop:
            scale_d = max(float(np.ptp(d_arr)), 0.1)
            parts.append((dp - d_arr) / scale_d)
        if has_vel:
            scale_v = max(float(np.ptp(v_arr)), 10.0)
            parts.append((vp - v_arr) / scale_v)
        return np.concatenate(parts)

    x0 = [1.0, 1.0] if fit_mv else [1.0]
    bounds = ([0.4, 0.8], [2.5, 1.2]) if fit_mv else ([0.4], [2.5])
    opt = sp_opt.least_squares(_residual_vec, x0=x0, bounds=bounds, diff_step=1e-3, max_nfev=40)

    ff_mult_fit = float(opt.x[0])
    mv_mult_fit = float(opt.x[1]) if fit_mv else 1.0
    ff_calibrated = ff_nom * ff_mult_fit
    mv_calibrated = mv_nom * mv_mult_fit
    d_cal_pred, v_cal_pred = _eval_model(ff_mult_fit, mv_mult_fit)

    rmse_drop_nom = float(math.sqrt(float(np.mean((d_nom_pred - d_arr) ** 2)))) if has_drop else None
    rmse_drop_cal = float(math.sqrt(float(np.mean((d_cal_pred - d_arr) ** 2)))) if has_drop else None
    rmse_vel_nom = float(math.sqrt(float(np.mean((v_nom_pred - v_arr) ** 2)))) if has_vel else None
    rmse_vel_cal = float(math.sqrt(float(np.mean((v_cal_pred - v_arr) ** 2)))) if has_vel else None

    if has_drop:
        ss_tot = float(np.sum((d_arr - np.mean(d_arr)) ** 2))
        r2_fit = float(1.0 - np.sum((d_cal_pred - d_arr) ** 2) / max(ss_tot, 1e-12))
    else:
        ss_tot = float(np.sum((v_arr - np.mean(v_arr)) ** 2))
        r2_fit = float(1.0 - np.sum((v_cal_pred - v_arr) ** 2) / max(ss_tot, 1e-12))

    stations: list[dict[str, Any]] = []
    for i in range(len(r_arr)):
        stations.append(
            {
                "range_m": round(float(r_arr[i]), 1),
                "drop_obs_m": round(float(d_arr[i]), 4) if has_drop else None,
                "drop_nom_m": round(float(d_nom_pred[i]), 4),
                "drop_cal_m": round(float(d_cal_pred[i]), 4),
                "drop_resid_cal_m": round(float(d_arr[i] - d_cal_pred[i]), 4) if has_drop else None,
                "vel_obs_mps": round(float(v_arr[i]), 2) if has_vel else None,
                "vel_nom_mps": round(float(v_nom_pred[i]), 2),
                "vel_cal_mps": round(float(v_cal_pred[i]), 2),
                "vel_resid_cal_mps": round(float(v_arr[i] - v_cal_pred[i]), 2) if has_vel else None,
            }
        )

    return {
        "preset_id": preset_id,
        "ammo_name": ammo.name,
        "drag_ref": ammo.aero.drag_ref,
        "n_stations": len(r_arr),
        "has_drop": has_drop,
        "has_vel": has_vel,
        "nominal_form_factor": round(ff_nom, 4),
        "calibrated_form_factor_multiplier": round(ff_mult_fit, 4),
        "calibrated_form_factor": round(ff_calibrated, 4),
        "nominal_mv_mps": round(mv_nom, 2),
        "calibrated_mv_mps": round(mv_calibrated, 2),
        "rmse_drop_before_m": round(rmse_drop_nom, 4) if rmse_drop_nom is not None else None,
        "rmse_drop_after_m": round(rmse_drop_cal, 4) if rmse_drop_cal is not None else None,
        "rmse_vel_before_mps": round(rmse_vel_nom, 2) if rmse_vel_nom is not None else None,
        "rmse_vel_after_mps": round(rmse_vel_cal, 2) if rmse_vel_cal is not None else None,
        "r2_fit": round(r2_fit, 5),
        "stations": stations,
    }
