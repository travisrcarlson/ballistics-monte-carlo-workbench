"""Closed-Loop Empirical Test-to-Design Recalibration & Persistent Digital Twin Engine.

Turns lab, bench, and live-fire range measurements into a persistent calibrated Digital
Twin persisted in SQLite, diagnoses model drift vs nominal physics, prescribes targeted
design fixes, and plans Bayesian active field tests to shrink uncertainty fastest:
  1. Multi-Range Patterning & Pellet Drag Fit (sigma(R) exponent + Cd scale)
  2. Target-Board Photo-to-Pattern Digitizer (pure NumPy/SciPy illumination + blob fit)
  3. Live-Fire Drone Shoot-Down Bayesian Lethality Fit (A_vuln & E_crit MAP + reliability)
  4. Hardware Bench-Test Calibration (load-cell recoil, servo slew/settle, barrel zeroing, SWaP)
  5. Sensor & Tracker Telemetry Calibration (angular RMS error, latency, OTM stabilization jitter)
  6. Model Drift -> Top-3 Design Fix Prescription Engine + Bayesian Active Test Planner
"""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
import uuid

import numpy as np
from scipy import ndimage as sp_ndimage
from scipy import optimize as sp_opt

from ammo_mc.calibration import calibrate_shotgun_pattern_from_csv
from ammo_mc.cuas_defense import CUAS_SENSOR_PRESETS, _estimate_burst_pk_analytics
from ammo_mc.cuas_hardware import MOUNT_CLASS_SPECS, estimate_turret_swap_c
from ammo_mc.cuas_mission import CUAS_AMMO_ECONOMICS_CATALOG, _resolve_shotgun_ammo
from ammo_mc.scenario_library import resolve_scenario_db_path
from ammo_mc.small_arms import (
    compute_drone_projected_geometry,
    get_choke_definitions,
    get_drone_target_specs,
)


def get_default_nominal_twin() -> dict[str, Any]:
    """Return baseline uncalibrated Digital Twin dictionary with prior uncertainties."""
    return {
        "spread_factor": 1.0,
        "spread_factor_se": 0.08,
        "spread_exponent": 1.15,
        "spread_exponent_se": 0.06,
        "aim_bias_mrad": 0.0,
        "poi_bias_y_mrad": 0.0,
        "poi_bias_z_mrad": 0.0,
        "mv_scale": 1.0,
        "mv_scale_se": 0.03,
        "mv_sd_mps": 4.0,
        "dv_dt_mps_per_c": 0.72,
        "pellet_cd_scale": 1.0,
        "pellet_cd_scale_se": 0.06,
        "drone_avuln_scale": 1.0,
        "drone_avuln_scale_se": 0.14,
        "drone_ecrit_scale": 1.0,
        "drone_ecrit_scale_se": 0.15,
        "lethality_brier_score": None,
        "lethality_n_trials": 0,
        "recoil_force_scale": 1.0,
        "recoil_force_scale_se": 0.08,
        "measured_settle_time_s": None,
        "measured_slew_rate_deg_s": None,
        "measured_slew_accel_deg_s2": None,
        "barrel_zero_scatter_mrad": 0.0,
        "measured_mass_kg": None,
        "measured_peak_power_w": None,
        "tracker_jitter_mrad": None,
        "tracker_jitter_mrad_se": 0.5,
        "tracker_latency_ms": None,
        "otm_jitter_mrad": 0.0,
        "lot_number": "",
        "barrel_serial": "",
        "history": [],
    }


def merge_twin_updates(
    base_twin: dict[str, Any] | None,
    updates: dict[str, Any],
    domain: str = "",
    summary: str = "",
) -> dict[str, Any]:
    """Merge partial calibration fit outputs into a complete Digital Twin with audit history."""
    merged = get_default_nominal_twin()
    if base_twin:
        for k, v in base_twin.items():
            if k == "history" and isinstance(v, list):
                merged["history"] = [dict(item) for item in v if isinstance(item, dict)]
            elif v is not None:
                merged[k] = v

    for k, v in (updates or {}).items():
        if k == "history":
            continue
        if v is not None:
            merged[k] = v

    if domain or summary:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist = list(merged.get("history", []))
        hist.append(
            {
                "timestamp": stamp,
                "domain": domain or "calibration",
                "summary": summary or "Updated twin parameters from field measurements.",
                "updates": {k: v for k, v in (updates or {}).items() if k != "history"},
            }
        )
        merged["history"] = hist[-25:]
    return merged


def _coerce_rows_table(rows_or_csv: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    """Normalize CSV text or tabular list-of-dicts into stripped lowercased-key records."""
    if isinstance(rows_or_csv, str):
        reader = csv.DictReader(io.StringIO(rows_or_csv.strip()))
        if not reader.fieldnames:
            raise ValueError("CSV input has no header row.")
        out: list[dict[str, Any]] = []
        for row in reader:
            cleaned = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
            if any(str(val).strip() != "" for val in cleaned.values()):
                out.append(cleaned)
        return out
    return [
        {str(k).strip().lower(): v for k, v in row.items()}
        for row in (rows_or_csv or [])
        if isinstance(row, dict)
    ]


def _interp_range_threshold(ranges: list[float], pks: list[float], threshold_pct: float) -> float:
    """Linearly interpolate maximum slant range where P_kill >= threshold_pct."""
    r_best = 0.0
    for i, pk_v in enumerate(pks):
        if pk_v >= threshold_pct:
            r_best = float(ranges[i])
        elif i > 0 and pks[i - 1] >= threshold_pct and pk_v < threshold_pct:
            frac = (pks[i - 1] - threshold_pct) / max(pks[i - 1] - pk_v, 1e-6)
            return round(float(ranges[i - 1]) + frac * (float(ranges[i]) - float(ranges[i - 1])), 1)
    return round(r_best, 1)


def _choke_sigma_at_40yd(nominal_choke: str) -> float:
    c_defs = get_choke_definitions()
    c_nom = c_defs.get(nominal_choke, c_defs["Modified"])
    p_40 = max(0.15, min(0.99, float(c_nom["benchmark_40yd_30in_pct"])))
    return 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_40))


# -------------------------------------------------------------------------
# PERSISTENT SQLITE CALIBRATION PROFILE STORE
# -------------------------------------------------------------------------


class CalibrationProfileStore:
    """Thread-safe SQLite store for versioned empirical C-UAS Digital Twin profiles."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = resolve_scenario_db_path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    lot_number TEXT NOT NULL,
                    barrel_serial TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    twin_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "name": str(row["name"]),
            "lot_number": str(row["lot_number"]),
            "barrel_serial": str(row["barrel_serial"]),
            "tags": json.loads(row["tags_json"]),
            "notes": str(row["notes"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "twin": merge_twin_updates(None, json.loads(row["twin_json"])),
        }

    def save_profile(
        self,
        name: str,
        twin: dict[str, Any],
        notes: str = "",
        tags: list[str] | None = None,
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update a named Digital Twin calibration profile."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pid = str(profile_id).strip() if profile_id else uuid.uuid4().hex[:10]
        norm_twin = merge_twin_updates(None, twin)
        clean_tags = sorted({str(t).strip() for t in (tags or []) if str(t).strip()})
        tags_str = json.dumps(clean_tags)
        twin_str = json.dumps(norm_twin, sort_keys=True, default=str)
        lot = str(norm_twin.get("lot_number", ""))
        serial = str(norm_twin.get("barrel_serial", ""))

        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM calibration_profiles WHERE id = ?", (pid,)).fetchone()
            created = str(existing["created_at"]) if existing else now_iso
            conn.execute(
                """
                INSERT INTO calibration_profiles
                    (id, name, lot_number, barrel_serial, tags_json, notes, created_at, updated_at, twin_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    lot_number = excluded.lot_number,
                    barrel_serial = excluded.barrel_serial,
                    tags_json = excluded.tags_json,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at,
                    twin_json = excluded.twin_json
                """,
                (pid, str(name).strip() or f"CalTwin-{pid}", lot, serial, tags_str, str(notes), created, now_iso, twin_str),
            )
            conn.commit()
            saved = conn.execute("SELECT * FROM calibration_profiles WHERE id = ?", (pid,)).fetchone()
        assert saved is not None
        return self._row_to_dict(saved)

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM calibration_profiles WHERE id = ?", (profile_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_profiles(self, search: str = "") -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM calibration_profiles ORDER BY updated_at DESC").fetchall()
        q = search.strip().lower()
        items = [self._row_to_dict(r) for r in rows]
        if not q:
            return items
        return [
            it
            for it in items
            if q in f"{it['id']} {it['name']} {it['lot_number']} {it['barrel_serial']} {it['notes']} {' '.join(it['tags'])}".lower()
        ]

    def delete_profile(self, profile_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM calibration_profiles WHERE id = ?", (profile_id,))
            conn.commit()
            return cur.rowcount > 0

    def export_profiles_json(self) -> str:
        return json.dumps({"schema": "ammo_mc_calibration_profiles_v1", "profiles": self.list_profiles()}, indent=2)

    def import_profiles_json(self, raw_json: str) -> int:
        payload = json.loads(raw_json)
        items = payload.get("profiles", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("Calibration profile JSON must contain a list of profiles.")
        count = 0
        for row in items:
            if not isinstance(row, dict) or "name" not in row or "twin" not in row:
                continue
            self.save_profile(
                name=str(row["name"]),
                twin=dict(row["twin"]),
                notes=str(row.get("notes", "")),
                tags=list(row.get("tags", [])),
                profile_id=row.get("id"),
            )
            count += 1
        return count


# -------------------------------------------------------------------------
# 1. MULTI-RANGE PATTERNING & PELLET DRAG CALIBRATOR
# -------------------------------------------------------------------------


def generate_sample_multirange_pattern_rows(
    nominal_choke: str = "Modified",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    true_spread_factor: float = 1.18,
    true_exponent: float = 1.24,
    true_cd_scale: float = 1.10,
    ranges_m: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate realistic multi-distance patterning & downrange chronograph test measurements."""
    rng = np.random.default_rng(seed)
    sigma_40 = _choke_sigma_at_40yd(nominal_choke)
    _, ammo = _resolve_shotgun_ammo(ammunition_preset)
    v0 = float(ammo["v0_mps"])
    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    cd = float(ammo.get("effective_cd", 0.47))
    kappa = (1.225 * cd * math.pi * 0.25 * (d_m ** 2)) / (2.0 * max(m_kg, 1e-6))

    rows: list[dict[str, Any]] = []
    for r_m in ranges_m:
        sig_true = (0.01852 / 4.0 + (sigma_40 / 36.58) * r_m * ((r_m / 36.58) ** (true_exponent - 1.0))) * true_spread_factor
        sig_meas = max(0.02, float(sig_true * rng.normal(1.0, 0.025)))
        v_meas = max(80.0, float(v0 * math.exp(-true_cd_scale * kappa * r_m) + rng.normal(0.0, 2.5)))
        rows.append(
            {
                "range_m": round(float(r_m), 1),
                "sigma_m": round(sig_meas, 4),
                "p30in_pct": round(100.0 * (1.0 - math.exp(-(0.381 ** 2) / (2.0 * sig_meas * sig_meas))), 1),
                "measured_v_mps": round(v_meas, 1),
            }
        )
    return rows


def calibrate_multirange_pattern_and_drag(
    rows_or_csv: list[dict[str, Any]] | str,
    nominal_choke: str = "Modified",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
) -> dict[str, Any]:
    """Fit spread_factor, range-growth exponent p, and optional pellet Cd scale from multi-range data."""
    raw_rows = _coerce_rows_table(rows_or_csv)
    if not raw_rows:
        raise ValueError("Multi-range patterning table is empty.")

    sigma_40 = _choke_sigma_at_40yd(nominal_choke)
    theta_nom = sigma_40 / 36.58

    # Support either pre-summarized range stations OR raw pellet (y_m, z_m) coordinates grouped by range_m
    stations_map: dict[float, dict[str, Any]] = {}
    if "y_m" in raw_rows[0] and "z_m" in raw_rows[0] and "sigma_m" not in raw_rows[0]:
        grouped: dict[float, list[tuple[float, float, float | None]]] = {}
        for row in raw_rows:
            try:
                rv = float(row["range_m"])
                yv = float(row["y_m"])
                zv = float(row["z_m"])
                vv = float(row["measured_v_mps"]) if row.get("measured_v_mps") not in (None, "") else None
                if rv > 1.0 and math.isfinite(yv) and math.isfinite(zv):
                    grouped.setdefault(round(rv, 2), []).append((yv, zv, vv))
            except (TypeError, ValueError, KeyError):
                continue
        for rv, pts in sorted(grouped.items()):
            if len(pts) < 4:
                continue
            ys = np.array([p[0] for p in pts], dtype=np.float64)
            zs = np.array([p[1] for p in pts], dtype=np.float64)
            rads = np.hypot(ys - float(np.mean(ys)), zs - float(np.mean(zs)))
            sig_mle = float(math.sqrt(float(np.sum(rads * rads)) / max(2.0 * (len(pts) - 1), 1.0)))
            vs = [p[2] for p in pts if p[2] is not None and math.isfinite(p[2])]
            stations_map[rv] = {"range_m": rv, "sigma_m": sig_mle, "measured_v_mps": float(np.mean(vs)) if vs else float("nan")}
    else:
        for row in raw_rows:
            try:
                rv = float(row.get("range_m", 0.0))
                if rv <= 1.0 or not math.isfinite(rv):
                    continue
                sig_v = float("nan")
                if row.get("sigma_m") not in (None, "") and math.isfinite(float(row["sigma_m"])):
                    sig_v = float(row["sigma_m"])
                elif row.get("r50_m") not in (None, "") and math.isfinite(float(row["r50_m"])):
                    sig_v = float(row["r50_m"]) / math.sqrt(2.0 * math.log(2.0))
                elif row.get("p30in_pct") not in (None, "") and math.isfinite(float(row["p30in_pct"])):
                    p30 = max(0.05, min(0.99, float(row["p30in_pct"]) / 100.0))
                    sig_v = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p30))
                if not math.isfinite(sig_v) or sig_v <= 0.005:
                    continue
                vv = float("nan")
                v_raw = row.get("measured_v_mps", row.get("velocity_mps"))
                if v_raw not in (None, "") and math.isfinite(float(v_raw)):
                    vv = float(v_raw)
                stations_map[round(rv, 2)] = {"range_m": rv, "sigma_m": sig_v, "measured_v_mps": vv}
            except (TypeError, ValueError):
                continue

    stations = sorted(stations_map.values(), key=lambda d: d["range_m"])
    if len(stations) < 3:
        raise ValueError("Need at least 3 range stations to jointly calibrate spread multiplier and range exponent.")

    r_arr = np.array([s["range_m"] for s in stations], dtype=np.float64)
    sig_arr = np.array([s["sigma_m"] for s in stations], dtype=np.float64)

    def _pred_sigma(params: np.ndarray) -> np.ndarray:
        kf, exp_p = float(params[0]), float(params[1])
        return (0.01852 / 4.0 + theta_nom * r_arr * np.power(np.maximum(r_arr / 36.58, 0.05), exp_p - 1.0)) * kf

    ls_sig = sp_opt.least_squares(
        lambda p: _pred_sigma(p) - sig_arr,
        x0=[1.0, 1.15],
        bounds=([0.35, 0.85], [3.0, 1.75]),
    )
    kf_fit, exp_fit = float(ls_sig.x[0]), float(ls_sig.x[1])
    sig_pred = _pred_sigma(ls_sig.x)
    resid = sig_arr - sig_pred
    dof = max(len(r_arr) - 2, 1)
    s2 = float(np.sum(resid ** 2)) / dof
    try:
        cov = np.linalg.pinv(ls_sig.jac.T @ ls_sig.jac) * s2
        kf_se = float(math.sqrt(max(float(cov[0, 0]), 1e-6)))
        exp_se = float(math.sqrt(max(float(cov[1, 1]), 1e-6)))
    except Exception:
        kf_se, exp_se = 0.06, 0.05

    ss_tot = float(np.sum((sig_arr - float(np.mean(sig_arr))) ** 2))
    r2_sig = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-12)
    rmse_sig = math.sqrt(float(np.mean(resid ** 2)))

    # Optional downrange velocity fit -> pellet_cd_scale
    v_obs = np.array([s["measured_v_mps"] for s in stations], dtype=np.float64)
    v_mask = np.isfinite(v_obs)
    cd_scale_fit = 1.0
    cd_scale_se = 0.06
    rmse_v = None
    if int(np.sum(v_mask)) >= 2:
        _, ammo = _resolve_shotgun_ammo(ammunition_preset)
        v0 = float(ammo["v0_mps"])
        d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
        m_kg = float(ammo["pellet_mass_g"]) / 1000.0
        cd_base = float(ammo.get("effective_cd", 0.47))
        kappa_base = (1.225 * cd_base * math.pi * 0.25 * (d_m ** 2)) / (2.0 * max(m_kg, 1e-6))
        r_v = r_arr[v_mask]
        v_v = v_obs[v_mask]
        ls_v = sp_opt.least_squares(
            lambda p: (v0 * np.exp(-float(p[0]) * kappa_base * r_v)) - v_v,
            x0=[1.0],
            bounds=([0.45], [2.20]),
        )
        cd_scale_fit = float(ls_v.x[0])
        v_pred = v0 * np.exp(-cd_scale_fit * kappa_base * r_v)
        rmse_v = round(float(math.sqrt(float(np.mean((v_v - v_pred) ** 2)))), 2)
        dof_v = max(len(r_v) - 1, 1)
        s2_v = float(np.sum(ls_v.fun ** 2)) / dof_v
        try:
            cov_v = np.linalg.pinv(ls_v.jac.T @ ls_v.jac) * s2_v
            cd_scale_se = float(math.sqrt(max(float(cov_v[0, 0]), 1e-5)))
        except Exception:
            cd_scale_se = 0.05

    station_rows = []
    for idx, st_row in enumerate(stations):
        sig_nom = float(0.01852 / 4.0 + theta_nom * st_row["range_m"] * ((st_row["range_m"] / 36.58) ** 0.15))
        station_rows.append(
            {
                "range_m": st_row["range_m"],
                "sigma_measured_m": round(float(sig_arr[idx]), 4),
                "sigma_fitted_m": round(float(sig_pred[idx]), 4),
                "sigma_nominal_m": round(sig_nom, 4),
                "measured_v_mps": round(float(v_obs[idx]), 1) if math.isfinite(v_obs[idx]) else None,
            }
        )

    return {
        "nominal_choke": nominal_choke,
        "ammunition_preset": ammunition_preset,
        "n_stations": len(stations),
        "spread_factor": round(kf_fit, 3),
        "spread_factor_se": round(kf_se, 3),
        "spread_exponent": round(exp_fit, 3),
        "spread_exponent_se": round(exp_se, 3),
        "pellet_cd_scale": round(cd_scale_fit, 3),
        "pellet_cd_scale_se": round(cd_scale_se, 3),
        "r2_spread": round(r2_sig, 4),
        "rmse_spread_m": round(rmse_sig, 4),
        "rmse_velocity_mps": rmse_v,
        "stations": station_rows,
        "twin_patch": {
            "spread_factor": round(kf_fit, 3),
            "spread_factor_se": round(kf_se, 3),
            "spread_exponent": round(exp_fit, 3),
            "spread_exponent_se": round(exp_se, 3),
            "pellet_cd_scale": round(cd_scale_fit, 3) if rmse_v is not None else None,
            "pellet_cd_scale_se": round(cd_scale_se, 3) if rmse_v is not None else None,
        },
    }


# -------------------------------------------------------------------------
# 2. TARGET-BOARD PHOTO-TO-PATTERN DIGITIZER
# -------------------------------------------------------------------------


def generate_synthetic_pattern_board_png(
    range_m: float = 36.58,
    nominal_choke: str = "Modified",
    true_spread_factor: float = 1.18,
    n_pellets: int = 44,
    board_width_m: float = 1.0,
    board_height_m: float = 1.0,
    seed: int = 42,
) -> bytes:
    """Synthesize a realistic paper patterning-board target image with distinct pellet holes."""
    rng = np.random.default_rng(seed)
    w_px, h_px = 640, 640
    img = rng.normal(0.92, 0.014, size=(h_px, w_px)).astype(np.float64)

    # Subtle alignment ticks at crosshair (avoid solid continuous lines that fuse holes)
    cy, cx = h_px // 2, w_px // 2
    img[cy - 18 : cy + 19 : 3, cx] = 0.72
    img[cy, cx - 18 : cx + 19 : 3] = 0.72

    sigma_nom = _choke_sigma_at_40yd(nominal_choke) * ((float(range_m) / 36.58) ** 1.15)
    sigma_true = sigma_nom * float(true_spread_factor)

    yy, xx = np.ogrid[:h_px, :w_px]
    placed: list[tuple[float, float]] = []
    attempts = 0
    while len(placed) < int(n_pellets) and attempts < int(n_pellets) * 30:
        attempts += 1
        y_m = float(rng.normal(0.02, sigma_true))
        z_m = float(rng.normal(-0.015, sigma_true))
        px_x = cx + (y_m / max(board_width_m, 0.1)) * w_px
        px_y = cy - (z_m / max(board_height_m, 0.1)) * h_px
        if not (38.0 <= px_x <= w_px - 38.0 and 38.0 <= px_y <= h_px - 38.0):
            continue
        if any(math.hypot(px_x - ox, px_y - oy) < 15.0 for ox, oy in placed):
            continue
        placed.append((px_x, px_y))
        rad = float(rng.uniform(4.0, 5.4))
        dist2 = (xx - px_x) ** 2 + (yy - px_y) ** 2
        mask = dist2 <= (rad ** 2)
        img[mask] = np.minimum(img[mask], 0.14 + 0.12 * (dist2[mask] / max(rad ** 2, 1.0)))

    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(img_u8, mode="L").convert("RGB").save(buf, format="PNG")
    except Exception:
        from matplotlib import image as mpimg

        mpimg.imsave(buf, img_u8, cmap="gray", format="png")
    return buf.getvalue()


def digitize_pattern_board_image(
    image_bytes: bytes,
    board_width_m: float = 1.0,
    board_height_m: float = 1.0,
    range_m: float = 36.58,
    nominal_choke: str = "Modified",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    threshold_sigma: float = 2.6,
    min_blob_px: int = 4,
    max_blob_px: int = 550,
    roi_margin_frac: float = 0.04,
) -> dict[str, Any]:
    """Detect pellet puncture holes in a target-board image and fit calibrated Rayleigh dispersion."""
    if not image_bytes:
        raise ValueError("Uploaded image bytes are empty.")

    buf = io.BytesIO(image_bytes)
    try:
        from PIL import Image

        pil_img = Image.open(buf).convert("L")
        arr = np.asarray(pil_img, dtype=np.float64) / 255.0
    except Exception:
        from matplotlib import image as mpimg

        buf.seek(0)
        raw = np.asarray(mpimg.imread(buf), dtype=np.float64)
        if raw.max() > 1.5:
            raw = raw / 255.0
        arr = raw[..., :3].mean(axis=2) if raw.ndim == 3 else raw

    if arr.ndim != 2 or min(arr.shape) < 32:
        raise ValueError("Target-board image is too small or invalid grayscale dimensions.")

    h_px, w_px = arr.shape
    bg = sp_ndimage.gaussian_filter(arr, sigma=max(10.0, min(h_px, w_px) / 30.0))
    diff = np.maximum(bg - arr, 0.0)

    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    robust_sd = max(1.4826 * mad, 0.025)
    thresh = max(0.14, med + float(threshold_sigma) * robust_sd)

    binary = diff >= thresh
    mx = max(1, int(round(roi_margin_frac * w_px)))
    my = max(1, int(round(roi_margin_frac * h_px)))
    binary[:my, :] = False
    binary[-my:, :] = False
    binary[:, :mx] = False
    binary[:, -mx:] = False

    labeled, n_raw = sp_ndimage.label(binary)
    centroids: list[tuple[float, float]] = []
    if n_raw > 0:
        slices = sp_ndimage.find_objects(labeled)
        for lbl_idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys_s, xs_s = slc
            sub = labeled[slc] == lbl_idx
            area = int(np.sum(sub))
            if area < int(min_blob_px) or area > int(max_blob_px):
                continue
            bh = max(1, ys_s.stop - ys_s.start)
            bw = max(1, xs_s.stop - xs_s.start)
            aspect = max(bh / bw, bw / bh)
            if aspect > 2.8:
                continue  # Reject lines / creases
            ry_loc, rx_loc = np.nonzero(sub)
            cy_px = float(ys_s.start) + float(np.mean(ry_loc))
            cx_px = float(xs_s.start) + float(np.mean(rx_loc))
            centroids.append((cx_px, cy_px))

    if len(centroids) < 5:
        raise ValueError(
            f"Only {len(centroids)} circular pellet holes detected (need >= 5). Lower threshold or check image contrast."
        )

    csv_lines = ["pellet_id,y_m,z_m"]
    for idx, (cx_px, cy_px) in enumerate(centroids, start=1):
        y_m = ((cx_px / w_px) - 0.5) * float(board_width_m)
        z_m = (0.5 - (cy_px / h_px)) * float(board_height_m)
        csv_lines.append(f"{idx},{y_m:.4f},{z_m:.4f}")
    extracted_csv = "\n".join(csv_lines) + "\n"

    pat_res = calibrate_shotgun_pattern_from_csv(
        extracted_csv,
        range_m=float(range_m),
        nominal_choke=nominal_choke,
        ammunition_preset=ammunition_preset,
    )
    return {
        "n_detected_holes": len(centroids),
        "image_shape": [int(h_px), int(w_px)],
        "pixel_centroids": [[round(cx_v, 1), round(cy_v, 1)] for cx_v, cy_v in centroids],
        "extracted_csv": extracted_csv,
        "pattern_calibration": pat_res,
        "twin_patch": {
            "spread_factor": float(pat_res["calibrated_spread_factor"]),
            "spread_factor_se": round(float(pat_res["sigma_fit_se_m"] / max(pat_res["sigma_nominal_m"], 1e-4)), 3),
            "poi_bias_y_mrad": float(pat_res["poi_bias_y_mrad"]),
            "poi_bias_z_mrad": float(pat_res["poi_bias_z_mrad"]),
            "aim_bias_mrad": round(math.hypot(pat_res["poi_bias_y_mrad"], pat_res["poi_bias_z_mrad"]), 2),
        },
    }


# -------------------------------------------------------------------------
# 3. LIVE-FIRE DRONE SHOOT-DOWN BAYESIAN LETHALITY CALIBRATOR
# -------------------------------------------------------------------------


def generate_sample_livefire_drone_trials(
    drone_model: str = "fpv_quad_small",
    true_avuln_scale: float = 0.85,
    true_ecrit_scale: float = 1.24,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate realistic live-fire test range engagement log across ranges and loads."""
    rng = np.random.default_rng(seed)
    d_spec = get_drone_target_specs().get(drone_model, get_drone_target_specs()["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
    true_twin = {"drone_avuln_scale": true_avuln_scale, "drone_ecrit_scale": true_ecrit_scale}

    configs = [
        (25.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
        (32.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
        (38.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
        (42.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
        (46.0, "12G_NO4_BUCK_C_UAS", "Full", 2),
        (52.0, "12G_NO4_BUCK_C_UAS", "Full", 2),
        (45.0, "12G_TSS_NO2_C_UAS", "Full", 2),
        (55.0, "12G_TSS_NO2_C_UAS", "Full", 2),
        (62.0, "12G_TSS_NO2_C_UAS", "Extra Full", 2),
        (35.0, "12G_NO4_BUCK_C_UAS", "Modified", 1),
        (40.0, "12G_NO4_BUCK_C_UAS", "Full", 3),
        (58.0, "12G_TSS_NO2_C_UAS", "Full", 1),
        (48.0, "12G_NO4_BUCK_C_UAS", "Modified", 3),
        (28.0, "12G_NO4_BUCK_C_UAS", "Cylinder", 2),
        (50.0, "12G_TSS_NO2_C_UAS", "Modified", 2),
        (66.0, "12G_TSS_NO2_C_UAS", "Extra Full", 2),
        (44.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
        (36.0, "12G_NO4_BUCK_C_UAS", "Modified", 2),
    ]
    rows: list[dict[str, Any]] = []
    for rng_m, ammo_key, chk, bbls in configs:
        _, ammo = _resolve_shotgun_ammo(ammo_key)
        pk_true = _estimate_burst_pk_analytics(
            slant_range_m=rng_m,
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
            calibration_profile=true_twin,
        )
        killed = int(rng.random() < pk_true)
        rows.append(
            {
                "range_m": rng_m,
                "ammunition_preset": ammo_key,
                "choke": chk,
                "barrels_fired": bbls,
                "drone_killed": killed,
            }
        )
    return rows


def calibrate_livefire_drone_lethality(
    trials_or_csv: list[dict[str, Any]] | str,
    drone_model: str = "fpv_quad_small",
    base_twin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bayesian MAP fit of drone A_vuln & E_crit multipliers against binary shoot-down trials."""
    raw_rows = _coerce_rows_table(trials_or_csv)
    parsed: list[dict[str, Any]] = []
    for row in raw_rows:
        try:
            rm = float(row.get("range_m", 0.0))
            if rm <= 1.0 or not math.isfinite(rm):
                continue
            ammo_key = str(row.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")).strip() or "12G_NO4_BUCK_C_UAS"
            chk = str(row.get("choke", "Modified")).strip() or "Modified"
            bbls = max(1, int(round(float(row.get("barrels_fired", 2)))))
            k_raw = row.get("drone_killed", row.get("killed", 0))
            if isinstance(k_raw, bool):
                killed = 1 if k_raw else 0
            elif isinstance(k_raw, str):
                killed = 1 if k_raw.strip().lower() in ("1", "true", "yes", "y", "kill", "killed") else 0
            else:
                killed = 1 if float(k_raw) >= 0.5 else 0
            parsed.append({"range_m": rm, "ammunition_preset": ammo_key, "choke": chk, "barrels_fired": bbls, "drone_killed": killed})
        except (TypeError, ValueError):
            continue

    if len(parsed) < 5:
        raise ValueError("Live-fire lethality calibration requires at least 5 engagement trials.")

    d_spec = get_drone_target_specs().get(drone_model, get_drone_target_specs()["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
    twin_base = dict(base_twin or {})
    ammo_cache: dict[str, dict[str, Any]] = {}

    def _trial_pk(tr: dict[str, Any], av_s: float, ec_s: float) -> float:
        ak = tr["ammunition_preset"]
        if ak not in ammo_cache:
            _, ammo_cache[ak] = _resolve_shotgun_ammo(ak)
        ammo = ammo_cache[ak]
        cp = dict(twin_base)
        cp["drone_avuln_scale"] = av_s
        cp["drone_ecrit_scale"] = ec_s
        return _estimate_burst_pk_analytics(
            slant_range_m=tr["range_m"],
            barrels_fired=tr["barrels_fired"],
            pellets_per_shot=int(ammo["pellets_per_shot"]),
            pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
            pellet_mass_g=float(ammo["pellet_mass_g"]),
            v0_mps=float(ammo["v0_mps"]),
            choke_name=tr["choke"],
            drone_spec=d_spec,
            base_cd=float(ammo.get("effective_cd", 0.47)),
            aspect_mode="head_on_frontal",
            aspect_geom=aspect_geom,
            calibration_profile=cp,
        )

    ys = np.array([float(tr["drone_killed"]) for tr in parsed], dtype=np.float64)
    nom_pks = np.array([_trial_pk(tr, 1.0, 1.0) for tr in parsed], dtype=np.float64)

    prior_sd = 0.45

    def _neg_log_post(theta: np.ndarray) -> float:
        av_s = math.exp(float(theta[0]))
        ec_s = math.exp(float(theta[1]))
        nll = 0.0
        for tr, y_i in zip(parsed, ys):
            p = max(1e-4, min(1.0 - 1e-4, _trial_pk(tr, av_s, ec_s)))
            nll -= y_i * math.log(p) + (1.0 - y_i) * math.log(1.0 - p)
        prior_pen = 0.5 * ((float(theta[0]) / prior_sd) ** 2 + (float(theta[1]) / prior_sd) ** 2)
        return nll + prior_pen

    opt = sp_opt.minimize(
        _neg_log_post,
        x0=np.array([0.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(math.log(0.35), math.log(2.2)), (math.log(0.40), math.log(2.4))],
    )
    th_hat = np.asarray(opt.x, dtype=np.float64)
    av_fit = float(math.exp(float(th_hat[0])))
    ec_fit = float(math.exp(float(th_hat[1])))

    # 2x2 central-difference Hessian for Bayesian Laplace posterior SEs
    h = 0.03
    f0 = _neg_log_post(th_hat)
    hess = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        ei = np.zeros(2, dtype=np.float64)
        ei[i] = h
        hess[i, i] = (_neg_log_post(th_hat + ei) - 2.0 * f0 + _neg_log_post(th_hat - ei)) / (h * h)
        for j in range(i + 1, 2):
            ej = np.zeros(2, dtype=np.float64)
            ej[j] = h
            hij = (
                _neg_log_post(th_hat + ei + ej)
                - _neg_log_post(th_hat + ei - ej)
                - _neg_log_post(th_hat - ei + ej)
                + _neg_log_post(th_hat - ei - ej)
            ) / (4.0 * h * h)
            hess[i, j] = hij
            hess[j, i] = hij
    try:
        eig_vals = np.linalg.eigvalsh(hess)
        if np.min(eig_vals) <= 1e-4:
            hess += np.eye(2) * (1.0 / (prior_sd ** 2))
        cov_log = np.linalg.pinv(hess)
        av_log_sd = float(math.sqrt(max(float(cov_log[0, 0]), 1e-4)))
        ec_log_sd = float(math.sqrt(max(float(cov_log[1, 1]), 1e-4)))
    except Exception:
        av_log_sd, ec_log_sd = 0.14, 0.15

    cal_pks = np.array([_trial_pk(tr, av_fit, ec_fit) for tr in parsed], dtype=np.float64)
    brier_before = float(np.mean((nom_pks - ys) ** 2))
    brier_after = float(np.mean((cal_pks - ys) ** 2))
    eps = 1e-4
    ll_before = -float(np.mean(ys * np.log(np.clip(nom_pks, eps, 1 - eps)) + (1 - ys) * np.log(np.clip(1 - nom_pks, eps, 1 - eps))))
    ll_after = -float(np.mean(ys * np.log(np.clip(cal_pks, eps, 1 - eps)) + (1 - ys) * np.log(np.clip(1 - cal_pks, eps, 1 - eps))))

    # 4-bin Predicted vs Observed Reliability Curve
    edges = [0.0, 0.25, 0.50, 0.75, 1.01]
    reliability_bins: list[dict[str, Any]] = []
    for b_idx in range(4):
        lo, hi = edges[b_idx], edges[b_idx + 1]
        mask = (cal_pks >= lo) & (cal_pks < hi)
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        obs_p = float(np.mean(ys[mask]))
        pred_p = float(np.mean(cal_pks[mask]))
        # Wilson 95% score interval half-width approximation
        z = 1.96
        denom_w = 1.0 + (z * z) / n_b
        hw = (z / denom_w) * math.sqrt((obs_p * (1.0 - obs_p)) / n_b + (z * z) / (4.0 * n_b * n_b))
        reliability_bins.append(
            {
                "bin": f"{int(lo * 100)}–{int(min(100, hi * 100))}%",
                "n_trials": n_b,
                "pred_pk_pct": round(100.0 * pred_p, 1),
                "obs_pk_pct": round(100.0 * obs_p, 1),
                "ci95_half_pct": round(100.0 * hw, 1),
            }
        )

    trial_rows = []
    for tr, p_nom, p_cal in zip(parsed, nom_pks, cal_pks):
        trial_rows.append(
            {
                **tr,
                "nom_pk_pct": round(100.0 * float(p_nom), 1),
                "cal_pk_pct": round(100.0 * float(p_cal), 1),
            }
        )

    return {
        "drone_model": drone_model,
        "n_trials": len(parsed),
        "observed_kill_rate_pct": round(100.0 * float(np.mean(ys)), 1),
        "drone_avuln_scale": round(av_fit, 3),
        "drone_avuln_scale_se": round(av_fit * av_log_sd, 3),
        "drone_avuln_ci95": [round(math.exp(float(th_hat[0]) - 1.96 * av_log_sd), 3), round(math.exp(float(th_hat[0]) + 1.96 * av_log_sd), 3)],
        "drone_ecrit_scale": round(ec_fit, 3),
        "drone_ecrit_scale_se": round(ec_fit * ec_log_sd, 3),
        "drone_ecrit_ci95": [round(math.exp(float(th_hat[1]) - 1.96 * ec_log_sd), 3), round(math.exp(float(th_hat[1]) + 1.96 * ec_log_sd), 3)],
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
        "logloss_before": round(ll_before, 4),
        "logloss_after": round(ll_after, 4),
        "reliability_bins": reliability_bins,
        "trials": trial_rows,
        "twin_patch": {
            "drone_avuln_scale": round(av_fit, 3),
            "drone_avuln_scale_se": round(av_fit * av_log_sd, 3),
            "drone_ecrit_scale": round(ec_fit, 3),
            "drone_ecrit_scale_se": round(ec_fit * ec_log_sd, 3),
            "lethality_brier_score": round(brier_after, 4),
            "lethality_n_trials": len(parsed),
        },
    }


# -------------------------------------------------------------------------
# 4 & 5. HARDWARE BENCH & TRACKER TELEMETRY CALIBRATORS
# -------------------------------------------------------------------------


def generate_sample_hardware_bench_tests(n_barrels: int = 6, seed: int = 42) -> dict[str, Any]:
    """Generate sample prototype load-cell, IMU, servo step-response & barrel zeroing bench measurements."""
    rng = np.random.default_rng(seed)
    zeros = []
    for i in range(max(2, int(n_barrels))):
        zeros.append(
            {
                "barrel_id": f"B{i+1}",
                "yaw_err_mrad": round(float(rng.normal(0.15, 0.85)), 2),
                "pitch_err_mrad": round(float(rng.normal(-0.10, 0.85)), 2),
            }
        )
    return {
        "peak_recoil_kn_samples": [5.12, 5.28, 4.95, 5.19, 5.04],
        "settle_time_ms_samples": [205.0, 218.0, 212.0, 224.0],
        "slew_rate_deg_s_samples": [152.0, 149.0, 154.0, 150.5],
        "slew_accel_deg_s2_samples": [585.0, 570.0, 595.0],
        "barrel_zero_offsets_mrad": zeros,
        "weighed_mass_kg": 63.4,
        "measured_peak_power_w": 468.0,
    }


def calibrate_hardware_bench_tests(
    nominal_turret_cfg: dict[str, Any] | None,
    bench_measurements: dict[str, Any],
) -> dict[str, Any]:
    """Compare bench test data vs nominal SWaP-C/dynamics model and compute calibrated multipliers."""
    cfg = dict(nominal_turret_cfg or {})
    nom_swap = estimate_turret_swap_c(
        n_barrels=int(cfg.get("n_barrels", 6)),
        barrel_length_in=float(cfg.get("barrel_length_in", 24.0)),
        layout_type=str(cfg.get("layout_type", "circular")),
        mount_radius_m=float(cfg.get("mount_radius_m", 0.20)),
        ammunition_preset=str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS")),
        magazine_type=str(cfg.get("magazine_type", "box_10rd")),
        slew_rate_deg_s=float(cfg.get("slew_rate_deg_s", 160.0)),
        slew_accel_deg_s2=float(cfg.get("slew_accel_deg_s2", 640.0)),
        sensor_preset=str(cfg.get("sensor_preset", "Sensor_Fused_Radar_EOIR")),
        firing_mode=str(cfg.get("firing_mode", "balanced_pair_ripple")),
        ripple_delay_ms=float(cfg.get("ripple_delay_ms", 10.0)),
        hybrid_mode=bool(cfg.get("hybrid_mode", False)),
    )["summary"]

    bm = dict(bench_measurements or {})
    comparisons: list[dict[str, Any]] = []
    flags: list[str] = []

    def _add_cmp(metric: str, unit: str, nom_val: float, meas_vals: list[float] | float | None, warn_pct: float = 10.0, oos_pct: float = 18.0) -> float | None:
        if meas_vals is None:
            return None
        if isinstance(meas_vals, (list, tuple)):
            vals = [float(v) for v in meas_vals if v is not None and math.isfinite(float(v))]
            if not vals:
                return None
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            mean_v = float(meas_vals)
            sd_v = 0.0
        drift_pct = 100.0 * (mean_v - nom_val) / max(abs(nom_val), 1e-6)
        ad = abs(drift_pct)
        status = "OUT_OF_SPEC" if ad >= oos_pct else ("DRIFT_WARN" if ad >= warn_pct else "IN_SPEC")
        if status != "IN_SPEC":
            flags.append(f"{status}: {metric} measured {mean_v:.2f} {unit} vs nominal {nom_val:.2f} {unit} ({drift_pct:+.1f}%)")
        comparisons.append(
            {
                "metric": metric,
                "unit": unit,
                "nominal": round(nom_val, 2),
                "measured_mean": round(mean_v, 2),
                "measured_sd": round(sd_v, 2),
                "drift_pct": round(drift_pct, 1),
                "status": status,
            }
        )
        return mean_v

    rec_mean = _add_cmp("Peak Recoil Mount Force", "kN", float(nom_swap["peak_recoil_force_kn"]), bm.get("peak_recoil_kn_samples"), warn_pct=8.0, oos_pct=15.0)
    settle_ms_mean = _add_cmp("Recoil Settle Time", "ms", float(cfg.get("settle_time_s", 0.16)) * 1000.0, bm.get("settle_time_ms_samples"), warn_pct=12.0, oos_pct=25.0)
    slew_mean = _add_cmp("Slew Velocity", "deg/s", float(cfg.get("slew_rate_deg_s", 160.0)), bm.get("slew_rate_deg_s_samples"), warn_pct=8.0, oos_pct=15.0)
    accel_mean = _add_cmp("Slew Acceleration", "deg/s²", float(cfg.get("slew_accel_deg_s2", 640.0)), bm.get("slew_accel_deg_s2_samples"), warn_pct=10.0, oos_pct=20.0)
    mass_meas = _add_cmp("Weighed Turret Mass", "kg", float(nom_swap["total_mass_kg"]), bm.get("weighed_mass_kg"), warn_pct=6.0, oos_pct=12.0)
    pwr_meas = _add_cmp("Measured Peak Bus Power", "W", float(nom_swap["peak_power_w"]), bm.get("measured_peak_power_w"), warn_pct=10.0, oos_pct=20.0)

    zeros = bm.get("barrel_zero_offsets_mrad") or []
    rad_errs: list[float] = []
    for zrow in zeros:
        if isinstance(zrow, dict):
            ye = float(zrow.get("yaw_err_mrad", 0.0))
            pe = float(zrow.get("pitch_err_mrad", 0.0))
        else:
            ye, pe = float(zrow[0]), float(zrow[1])
        rad_errs.append(math.hypot(ye, pe))
    b_scatter = float(math.sqrt(float(np.mean(np.square(rad_errs))))) if rad_errs else 0.0
    if rad_errs:
        z_status = "OUT_OF_SPEC" if b_scatter >= 2.0 else ("DRIFT_WARN" if b_scatter >= 1.2 else "IN_SPEC")
        if z_status != "IN_SPEC":
            flags.append(f"{z_status}: Barrel zero RMS scatter {b_scatter:.2f} mrad (>= 1.2 mrad threshold)")
        comparisons.append(
            {
                "metric": "Barrel-to-Barrel Zero RMS",
                "unit": "mrad",
                "nominal": 0.50,
                "measured_mean": round(b_scatter, 2),
                "measured_sd": round(float(np.std(rad_errs)), 2),
                "drift_pct": round(100.0 * (b_scatter - 0.50) / 0.50, 1),
                "status": z_status,
            }
        )

    rec_scale = (rec_mean / max(float(nom_swap["peak_recoil_force_kn"]), 0.1)) if rec_mean is not None else 1.0
    twin_patch: dict[str, Any] = {
        "recoil_force_scale": round(rec_scale, 3),
        "recoil_force_scale_se": 0.04 if rec_mean is not None else 0.08,
        "barrel_zero_scatter_mrad": round(b_scatter, 2),
    }
    if settle_ms_mean is not None:
        twin_patch["measured_settle_time_s"] = round(settle_ms_mean / 1000.0, 3)
    if slew_mean is not None:
        twin_patch["measured_slew_rate_deg_s"] = round(slew_mean, 1)
    if accel_mean is not None:
        twin_patch["measured_slew_accel_deg_s2"] = round(accel_mean, 1)
    if mass_meas is not None:
        twin_patch["measured_mass_kg"] = round(mass_meas, 2)
    if pwr_meas is not None:
        twin_patch["measured_peak_power_w"] = round(pwr_meas, 1)

    return {
        "comparisons": comparisons,
        "out_of_spec_flags": flags,
        "n_out_of_spec": sum(1 for c in comparisons if c["status"] == "OUT_OF_SPEC"),
        "twin_patch": twin_patch,
    }


def generate_sample_tracker_test_rows(
    sensor_preset: str = "Sensor_Fused_Radar_EOIR",
    true_rms_mrad: float = 1.15,
    true_latency_ms: float = 42.0,
    true_otm_jitter_mrad: float = 0.85,
    n_samples: int = 16,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate realistic tracker error + latency test telemetry rows."""
    _ = sensor_preset
    rng = np.random.default_rng(seed)
    axis_sd = true_rms_mrad / math.sqrt(2.0)
    rows: list[dict[str, Any]] = []
    for i in range(n_samples):
        rows.append(
            {
                "sample_id": i + 1,
                "az_err_mrad": round(float(rng.normal(0.10, axis_sd)), 3),
                "el_err_mrad": round(float(rng.normal(-0.08, axis_sd)), 3),
                "latency_ms": round(max(5.0, float(rng.normal(true_latency_ms, 5.5))), 1),
                "otm_jitter_mrad": round(max(0.0, float(rng.normal(true_otm_jitter_mrad, 0.18))), 2),
            }
        )
    return rows


def calibrate_tracker_telemetry(
    rows_or_csv: list[dict[str, Any]] | str,
    sensor_preset: str = "Sensor_Fused_Radar_EOIR",
) -> dict[str, Any]:
    """Calibrate sensor RMS angular tracking jitter, acquisition latency, and OTM LOS jitter."""
    raw_rows = _coerce_rows_table(rows_or_csv)
    az_errs: list[float] = []
    el_errs: list[float] = []
    lats: list[float] = []
    otms: list[float] = []

    for row in raw_rows:
        try:
            if "az_err_mrad" in row and "el_err_mrad" in row:
                ae = float(row["az_err_mrad"])
                ee = float(row["el_err_mrad"])
            elif "meas_az_mrad" in row and "true_az_mrad" in row:
                ae = float(row["meas_az_mrad"]) - float(row["true_az_mrad"])
                ee = float(row.get("meas_el_mrad", 0.0)) - float(row.get("true_el_mrad", 0.0))
            else:
                continue
            if not (math.isfinite(ae) and math.isfinite(ee)):
                continue
            az_errs.append(ae)
            el_errs.append(ee)
            if row.get("latency_ms") not in (None, "") and math.isfinite(float(row["latency_ms"])):
                lats.append(float(row["latency_ms"]))
            if row.get("otm_jitter_mrad") not in (None, "") and math.isfinite(float(row["otm_jitter_mrad"])):
                otms.append(float(row["otm_jitter_mrad"]))
        except (TypeError, ValueError):
            continue

    if len(az_errs) < 4:
        raise ValueError("Tracker calibration requires at least 4 valid azimuth/elevation error rows.")

    az_arr = np.asarray(az_errs, dtype=np.float64)
    el_arr = np.asarray(el_errs, dtype=np.float64)
    rad_arr = np.hypot(az_arr, el_arr)
    rms_mrad = float(math.sqrt(float(np.mean(rad_arr ** 2))))
    rms_se = rms_mrad / math.sqrt(max(2.0 * len(rad_arr), 1.0))

    sp_spec = CUAS_SENSOR_PRESETS.get(sensor_preset, CUAS_SENSOR_PRESETS["Sensor_Fused_Radar_EOIR"])
    nom_rms = math.hypot(float(sp_spec["sigma_az_mrad"]), float(sp_spec["sigma_el_mrad"]))
    nom_lat = float(sp_spec["fcs_latency_ms"])
    mean_lat = float(np.mean(lats)) if lats else nom_lat
    p95_lat = float(np.percentile(lats, 95)) if lats else nom_lat
    otm_mean = float(np.mean(otms)) if otms else 0.0

    return {
        "sensor_preset": sensor_preset,
        "n_samples": len(az_errs),
        "nominal_rms_mrad": round(nom_rms, 3),
        "measured_rms_mrad": round(rms_mrad, 3),
        "measured_rms_se_mrad": round(rms_se, 3),
        "az_bias_mrad": round(float(np.mean(az_arr)), 3),
        "el_bias_mrad": round(float(np.mean(el_arr)), 3),
        "nominal_latency_ms": round(nom_lat, 1),
        "measured_latency_ms": round(mean_lat, 1),
        "latency_p95_ms": round(p95_lat, 1),
        "otm_jitter_mrad": round(otm_mean, 2),
        "twin_patch": {
            "tracker_jitter_mrad": round(rms_mrad, 3),
            "tracker_jitter_mrad_se": round(rms_se, 3),
            "tracker_latency_ms": round(mean_lat, 1),
            "otm_jitter_mrad": round(otm_mean, 2),
        },
    }


# -------------------------------------------------------------------------
# 6. MODEL DRIFT -> DESIGN FIX PRESCRIPTION & ACTIVE TEST PLANNER
# -------------------------------------------------------------------------


def _evaluate_range_sweep_for_cfg(
    cfg: dict[str, Any],
    twin: dict[str, Any] | None,
    d_spec: dict[str, Any],
    aspect_geom: dict[str, Any],
    ranges_m: list[float],
    req_range_m: float,
) -> dict[str, Any]:
    """Evaluate analytical P_kill(R), R80, R50 & P_kill@R_req cleanly without code duplication."""
    outer_key = str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))
    _, a_out = _resolve_shotgun_ammo(outer_key)
    is_hyb = bool(cfg.get("hybrid_mode", False)) and int(cfg.get("n_barrels", 6)) >= 4
    inner_key = str(cfg.get("inner_ammunition_preset") or "12G_TSS_NO2_C_UAS") if is_hyb else outer_key
    _, a_in = _resolve_shotgun_ammo(inner_key)
    sw_rng = float(cfg.get("hybrid_switch_range_m", 42.0))
    bpb = int(cfg.get("barrels_per_burst", 2))
    outer_chk = str(cfg.get("choke", cfg.get("default_choke", "Modified")))
    inner_chk = str(cfg.get("inner_choke", "Extra Full"))
    curt_fac = 0.96 if str(cfg.get("pointing_mode", "convergent")) == "divergent_curtain" else 1.0

    def _pk(r_m: float) -> float:
        use_in = is_hyb and r_m > sw_rng
        a_sel = a_in if use_in else a_out
        chk_sel = inner_chk if use_in else outer_chk
        p = _estimate_burst_pk_analytics(
            slant_range_m=r_m,
            barrels_fired=bpb,
            pellets_per_shot=int(a_sel["pellets_per_shot"]),
            pellet_diameter_mm=float(a_sel["pellet_diameter_mm"]),
            pellet_mass_g=float(a_sel["pellet_mass_g"]),
            v0_mps=float(a_sel["v0_mps"]),
            choke_name=chk_sel,
            drone_spec=d_spec,
            base_cd=float(a_sel.get("effective_cd", 0.47)),
            aspect_mode="head_on_frontal",
            aspect_geom=aspect_geom,
            calibration_profile=twin,
        )
        return round(min(99.8, 100.0 * p * curt_fac), 1)

    pks = [_pk(rv) for rv in ranges_m]
    return {
        "ranges_m": list(ranges_m),
        "pk_pct": pks,
        "r80_m": _interp_range_threshold(ranges_m, pks, 80.0),
        "r50_m": _interp_range_threshold(ranges_m, pks, 50.0),
        "pk_at_req_pct": _pk(req_range_m),
    }


def analyze_calibration_design_delta(
    turret_cfg: dict[str, Any] | None,
    calibration_twin: dict[str, Any] | None,
    drone_model: str = "fpv_quad_small",
    required_range_m: float = 45.0,
    mount_class: str = "light_ugv",
    ranges_m: tuple[float, ...] | list[float] = (15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0),
) -> dict[str, Any]:
    """Compare Nominal vs Recalibrated Digital Twin performance and rank Top-3 design fixes."""
    base_cfg = dict(turret_cfg or {})
    base_cfg.setdefault("n_barrels", 6)
    base_cfg.setdefault("barrels_per_burst", 2)
    base_cfg.setdefault("ammunition_preset", "12G_NO4_BUCK_C_UAS")
    base_cfg.setdefault("choke", str(base_cfg.get("default_choke", "Modified")))
    base_cfg.setdefault("layout_type", "circular")
    base_cfg.setdefault("pointing_mode", "convergent")

    twin = merge_twin_updates(None, calibration_twin or {})
    d_spec = get_drone_target_specs().get(drone_model, get_drone_target_specs()["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
    r_list = sorted(float(r) for r in ranges_m)
    r_req = float(required_range_m)

    nom_curve = _evaluate_range_sweep_for_cfg(base_cfg, None, d_spec, aspect_geom, r_list, r_req)
    cal_curve = _evaluate_range_sweep_for_cfg(base_cfg, twin, d_spec, aspect_geom, r_list, r_req)

    nom_swap = estimate_turret_swap_c(
        n_barrels=int(base_cfg["n_barrels"]),
        layout_type=str(base_cfg["layout_type"]),
        ammunition_preset=str(base_cfg["ammunition_preset"]),
        hybrid_mode=bool(base_cfg.get("hybrid_mode", False)),
    )["summary"]
    cal_swap = estimate_turret_swap_c(
        n_barrels=int(base_cfg["n_barrels"]),
        layout_type=str(base_cfg["layout_type"]),
        ammunition_preset=str(base_cfg["ammunition_preset"]),
        hybrid_mode=bool(base_cfg.get("hybrid_mode", False)),
        calibration_profile=twin,
    )["summary"]

    mc_spec = MOUNT_CLASS_SPECS.get(mount_class, MOUNT_CLASS_SPECS["light_ugv"])
    n_b = int(base_cfg["n_barrels"])
    bpb = int(base_cfg["barrels_per_burst"])

    # Candidate surgical design modifications evaluated under the recalibrated twin
    candidates: list[tuple[str, str, dict[str, Any]]] = [
        ("Tighten Choke -> Full", "Compensates wider measured pattern dispersion with tighter choke constriction.", {"choke": "Full", "default_choke": "Full"}),
        ("Upgrade Payload -> TSS #2", "Replaces lead #4 Buck with dense tungsten TSS #2 for retained terminal KE.", {"ammunition_preset": "12G_TSS_NO2_C_UAS"}),
        (
            "Enable Hybrid Dual-Ring (#4 Buck + TSS #2)",
            "Mixes close-in pattern density on outer barrels with long-range TSS #2 core.",
            {
                "n_barrels": max(4, n_b),
                "layout_type": "concentric",
                "hybrid_mode": True,
                "ammunition_preset": "12G_NO4_BUCK_C_UAS",
                "inner_ammunition_preset": "12G_TSS_NO2_C_UAS",
                "inner_choke": "Extra Full",
                "hybrid_switch_range_m": 38.0,
            },
        ),
        (
            f"Increase Burst -> {min(max(n_b, bpb + 1), bpb + 1)} Barrels",
            "Raises volley pellet density per trigger pull to restore P_kill against tougher target.",
            {"barrels_per_burst": min(max(4, n_b), bpb + 1), "n_barrels": max(n_b, bpb + 1)},
        ),
        (
            "TSS #2 + Full Choke Precision",
            "Combines long-range tungsten retained kinetic energy with Full choke concentration.",
            {"ammunition_preset": "12G_TSS_NO2_C_UAS", "choke": "Full", "default_choke": "Full"},
        ),
        (
            f"Scale Array -> {min(12, n_b + 2)} Barrels + Pair Ripple",
            "Adds 2 barrels with balanced pair ripple to grow salvo depth without excess shock.",
            {"n_barrels": min(12, n_b + 2), "firing_mode": "balanced_pair_ripple"},
        ),
        (
            "Divergent Curtain Splay (4.5 mrad)",
            "Absorbs measured POI aim bias, barrel zero scatter, or tracker jitter.",
            {"pointing_mode": "divergent_curtain", "splay_angle_mrad": 4.5},
        ),
    ]

    fixes: list[dict[str, Any]] = []
    for fix_name, rationale, patch in candidates:
        c_cfg = copy.deepcopy(base_cfg)
        c_cfg.update(patch)
        c_curve = _evaluate_range_sweep_for_cfg(c_cfg, twin, d_spec, aspect_geom, r_list, r_req)
        c_swap = estimate_turret_swap_c(
            n_barrels=int(c_cfg["n_barrels"]),
            layout_type=str(c_cfg["layout_type"]),
            ammunition_preset=str(c_cfg["ammunition_preset"]),
            hybrid_mode=bool(c_cfg.get("hybrid_mode", False)),
            firing_mode=str(c_cfg.get("firing_mode", "balanced_pair_ripple")),
            calibration_profile=twin,
        )["summary"]
        ok_mount = bool(c_swap["total_mass_kg"] <= mc_spec["max_mass_kg"] and c_swap["peak_recoil_force_kn"] <= mc_spec["max_recoil_kn"])
        delta_r80 = round(c_curve["r80_m"] - cal_curve["r80_m"], 1)
        fixes.append(
            {
                "fix_name": fix_name,
                "rationale": rationale,
                "r80_m": c_curve["r80_m"],
                "r80_gain_m": delta_r80,
                "pk_at_req_pct": c_curve["pk_at_req_pct"],
                "mass_kg": c_swap["total_mass_kg"],
                "peak_recoil_kn": c_swap["peak_recoil_force_kn"],
                "mount_feasible": ok_mount,
                "stage1_patch": patch,
                "_rank": (1 if ok_mount else 0) * 200.0 + c_curve["r80_m"] + 0.5 * c_curve["pk_at_req_pct"],
            }
        )

    fixes.sort(key=lambda r: r["_rank"], reverse=True)
    top3 = [{k: v for k, v in row.items() if k != "_rank"} for row in fixes[:3]]
    r80_drift = round(cal_curve["r80_m"] - nom_curve["r80_m"], 1)
    pk_drift = round(cal_curve["pk_at_req_pct"] - nom_curve["pk_at_req_pct"], 1)

    diag_msgs: list[str] = []
    if twin["spread_factor"] > 1.05 or twin["spread_exponent"] > 1.18:
        diag_msgs.append(f"Pattern spread wider than nominal ({twin['spread_factor']:.2f}x @ exp {twin['spread_exponent']:.2f}).")
    if twin["drone_ecrit_scale"] > 1.08 or twin["drone_avuln_scale"] < 0.92:
        diag_msgs.append(f"Target drone tougher than nominal (A_vuln {twin['drone_avuln_scale']:.2f}x, E_crit {twin['drone_ecrit_scale']:.2f}x).")
    if twin["recoil_force_scale"] > 1.08:
        diag_msgs.append(f"Measured recoil load {twin['recoil_force_scale']:.2f}x higher than nominal ({cal_swap['peak_recoil_force_kn']:.2f} kN).")
    if not diag_msgs:
        diag_msgs.append("Calibrated Digital Twin closely matches nominal physics baselines.")

    return {
        "nominal": {**nom_curve, "mass_kg": nom_swap["total_mass_kg"], "peak_recoil_kn": nom_swap["peak_recoil_force_kn"]},
        "calibrated": {**cal_curve, "mass_kg": cal_swap["total_mass_kg"], "peak_recoil_kn": cal_swap["peak_recoil_force_kn"]},
        "r80_drift_m": r80_drift,
        "pk_at_req_drift_pp": pk_drift,
        "required_range_m": r_req,
        "mount_class": mount_class,
        "diagnosis": " ".join(diag_msgs),
        "recommended_fixes": top3,
    }


def recommend_next_range_tests(
    turret_cfg: dict[str, Any] | None,
    calibration_twin: dict[str, Any] | None,
    drone_model: str = "fpv_quad_small",
    target_range_m: float = 45.0,
) -> dict[str, Any]:
    """Bayesian information-gain planner ranking highest-value next range/bench tests."""
    cfg = dict(turret_cfg or {})
    cfg.setdefault("n_barrels", 6)
    cfg.setdefault("barrels_per_burst", 2)
    cfg.setdefault("ammunition_preset", "12G_NO4_BUCK_C_UAS")
    cfg.setdefault("choke", str(cfg.get("default_choke", "Modified")))

    twin = merge_twin_updates(None, calibration_twin or {})
    d_spec = get_drone_target_specs().get(drone_model, get_drone_target_specs()["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")
    r_grid = [15.0, 22.0, 30.0, 38.0, 45.0, 52.0, 60.0, 68.0, 76.0]

    base_eval = _evaluate_range_sweep_for_cfg(cfg, twin, d_spec, aspect_geom, r_grid, float(target_range_m))
    r80_0 = float(base_eval["r80_m"])

    # Compute local derivative of R80 w.r.t. each uncertain twin parameter
    params_meta = [
        {
            "key": "spread_factor",
            "se_key": "spread_factor_se",
            "default_se": 0.08,
            "step": 0.05,
            "domain": "Patterning Board",
            "action": f"Shoot 2 patterning boards (54+ pellets) at {max(25.0, target_range_m - 5.0):.0f} m & {min(65.0, target_range_m + 10.0):.0f} m",
            "samples": "2 volleys (2 shells)",
            "est_cost_usd": 120.0,
        },
        {
            "key": "spread_exponent",
            "se_key": "spread_exponent_se",
            "default_se": 0.06,
            "step": 0.04,
            "domain": "Multi-Range Pattern",
            "action": "Patterning boards at 25 m and 60 m long baseline to pin dispersion exponent",
            "samples": "4 boards (4 shells)",
            "est_cost_usd": 180.0,
        },
        {
            "key": "drone_ecrit_scale",
            "se_key": "drone_ecrit_scale_se",
            "default_se": 0.15,
            "step": 0.08,
            "domain": "Live-Fire Drone",
            "action": f"Live-fire burst trials near transition range {max(35.0, r80_0):.0f} m–{min(70.0, r80_0 + 12.0):.0f} m",
            "samples": "6 drone sorties",
            "est_cost_usd": 2400.0,
        },
        {
            "key": "drone_avuln_scale",
            "se_key": "drone_avuln_scale_se",
            "default_se": 0.14,
            "step": 0.08,
            "domain": "Live-Fire Drone",
            "action": f"Head-on vs oblique FPV presentation shoots at {target_range_m:.0f} m",
            "samples": "6 drone sorties",
            "est_cost_usd": 2400.0,
        },
        {
            "key": "pellet_cd_scale",
            "se_key": "pellet_cd_scale_se",
            "default_se": 0.06,
            "step": 0.05,
            "domain": "Doppler / Downrange Chrono",
            "action": f"Measure downrange striking velocity at 35 m & {max(55.0, target_range_m + 15.0):.0f} m",
            "samples": "10 chronograph shots",
            "est_cost_usd": 260.0,
        },
        {
            "key": "mv_scale",
            "se_key": "mv_scale_se",
            "default_se": 0.03,
            "step": 0.03,
            "domain": "Chronograph Lot",
            "action": "Cold (5 °C) vs warm (35 °C) muzzle velocity lot sweep",
            "samples": "12 chronograph shots",
            "est_cost_usd": 140.0,
        },
    ]

    ranked: list[dict[str, Any]] = []
    tot_var = 0.0
    for pm in params_meta:
        val = float(twin.get(pm["key"], 1.0) or 1.0)
        se = float(twin.get(pm["se_key"], pm["default_se"]) or pm["default_se"])
        h = pm["step"]
        t_plus = dict(twin)
        t_minus = dict(twin)
        t_plus[pm["key"]] = val + h
        t_minus[pm["key"]] = max(0.25, val - h)
        r_plus = float(_evaluate_range_sweep_for_cfg(cfg, t_plus, d_spec, aspect_geom, r_grid, float(target_range_m))["r80_m"])
        r_minus = float(_evaluate_range_sweep_for_cfg(cfg, t_minus, d_spec, aspect_geom, r_grid, float(target_range_m))["r80_m"])
        d_r80_dp = (r_plus - r_minus) / (2.0 * h)
        sigma_r80_contrib = abs(d_r80_dp) * se
        var_c = sigma_r80_contrib ** 2
        tot_var += var_c
        # Assume new batch cuts parameter SE by ~50% (4x information gain)
        ci_shrink_m = round(1.96 * sigma_r80_contrib * 0.50, 2)
        ranked.append(
            {
                "parameter": pm["key"],
                "domain": pm["domain"],
                "prior_se": round(se, 3),
                "sensitivity_m_per_unit": round(d_r80_dp, 1),
                "r80_ci95_contrib_m": round(1.96 * sigma_r80_contrib, 2),
                "expected_ci_shrink_m": ci_shrink_m,
                "recommended_test": pm["action"],
                "sample_size": pm["samples"],
                "est_cost_usd": pm["est_cost_usd"],
                "value_per_kusd": round(ci_shrink_m / max(pm["est_cost_usd"] / 1000.0, 0.05), 2),
                "_var": var_c,
            }
        )

    ranked.sort(key=lambda r: (r["expected_ci_shrink_m"], r["value_per_kusd"]), reverse=True)
    r80_sd = math.sqrt(max(tot_var, 0.0))
    for i, r in enumerate(ranked, start=1):
        r["priority"] = i
        r["variance_share_pct"] = round(100.0 * r["_var"] / max(tot_var, 1e-6), 1)
    clean_rows = [{k: v for k, v in r.items() if k != "_var"} for r in ranked]

    return {
        "baseline_r80_m": round(r80_0, 1),
        "r80_ci95_halfwidth_m": round(1.96 * r80_sd, 2),
        "top_recommendations": clean_rows[:5],
        "all_parameters": clean_rows,
    }


def export_range_test_card_csv(test_plan: dict[str, Any]) -> str:
    """Export printable field test card CSV ready for range logging and re-upload."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["priority", "domain", "target_parameter", "recommended_test", "sample_size", "measured_value", "notes"])
    for row in test_plan.get("top_recommendations", []):
        writer.writerow(
            [
                row.get("priority", ""),
                row.get("domain", ""),
                row.get("parameter", ""),
                row.get("recommended_test", ""),
                row.get("sample_size", ""),
                "",
                "",
            ]
        )
    return buf.getvalue()
