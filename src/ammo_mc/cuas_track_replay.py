"""Recorded Drone-Track CSV / MAVLink Replay & Closed-Loop C-UAS Engagement Engine.

Parses real radar, EO/IR, or GPS/MAVLink telemetry logs into East-North-Up (ENU) kinematics
(instantaneous speed, lateral G-load, crossing angular rate, and 3D presentation aspect),
then replays candidate calibrated multi-barrel shotgun turrets against the measured paths.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np

from ammo_mc.aero_sphere import exact_quadratic_drag_tof_and_velocity
from ammo_mc.cuas_adversarial import normalize_blue_config
from ammo_mc.cuas_defense import CUAS_SENSOR_PRESETS, _trapezoidal_slew_time
from ammo_mc.cuas_mission import CUAS_AMMO_ECONOMICS_CATALOG, _resolve_shotgun_ammo
from ammo_mc.gis import EARTH_RADIUS_M
from ammo_mc.small_arms import (
    compute_drone_projected_geometry,
    get_choke_definitions,
    get_drone_target_specs,
)


def generate_sample_drone_track_csv(scenario: str = "evasive_fpv_dive", seed: int = 42) -> str:
    """Generate realistic synthetic recorded telemetry logs (`track_id,t_s,x_m,y_m,z_m,drone_model`)."""
    rng = np.random.default_rng(seed)
    rows: list[str] = ["track_id,t_s,x_m,y_m,z_m,drone_model"]

    def _emit_track(
        tid: str,
        model: str,
        az_deg: float,
        r_start: float,
        r_end: float,
        spd_mps: float,
        z_start: float,
        z_end: float,
        amp_lat: float,
        amp_vert: float,
        freq_hz: float,
        t_offset: float = 0.0,
        dt: float = 0.20,
        popup_t: float | None = None,
    ) -> None:
        dur = max(1.0, (r_start - r_end) / max(spd_mps, 2.0))
        n_pts = int(round(dur / dt)) + 1
        az_r = math.radians(az_deg)
        ux, uy = math.cos(az_r), math.sin(az_r)
        lx, ly = -uy, ux
        phase = float(rng.uniform(0.0, math.pi))
        for k in range(n_pts):
            t_loc = k * dt
            frac = min(1.0, t_loc / dur)
            r_nom = r_start - (r_start - r_end) * frac
            w_lat = amp_lat * math.sin(2.0 * math.pi * freq_hz * t_loc + phase)
            w_vert = amp_vert * math.cos(2.0 * math.pi * freq_hz * 1.2 * t_loc)
            if popup_t is not None and t_loc > popup_t:
                w_vert += 8.0 * math.sin(min(1.0, (t_loc - popup_t) / 1.1) * math.pi * 0.5)
            x = r_nom * ux + w_lat * lx + float(rng.normal(0.0, 0.12))
            y = r_nom * uy + w_lat * ly + float(rng.normal(0.0, 0.12))
            z = max(2.5, z_start + (z_end - z_start) * frac + w_vert + float(rng.normal(0.0, 0.08)))
            rows.append(f"{tid},{t_offset + t_loc:.2f},{x:.3f},{y:.3f},{z:.3f},{model}")

    sc = str(scenario).strip().lower()
    if sc == "multi_drone_pincer":
        _emit_track("FPV-North", "fpv_quad_small", 42.0, 115.0, 9.0, 29.0, 28.0, 12.0, 4.5, 2.2, 0.55, t_offset=0.0)
        _emit_track("Quad-East", "commercial_quad_medium", 118.0, 125.0, 10.0, 23.0, 32.0, 14.0, 3.2, 1.5, 0.42, t_offset=0.6)
        _emit_track("Wing-West", "fixed_wing_loitering", 235.0, 140.0, 11.0, 34.0, 42.0, 16.0, 2.8, 3.0, 0.35, t_offset=1.0)
    elif sc == "low_altitude_terrain_pop":
        _emit_track("Ridge-FPV-1", "fpv_quad_small", 55.0, 110.0, 10.0, 27.0, 5.5, 22.0, 3.0, 1.0, 0.45, popup_t=2.0)
        _emit_track("Ridge-FPV-2", "fpv_quad_small", 68.0, 122.0, 10.0, 28.0, 6.0, 24.0, 3.8, 1.4, 0.50, t_offset=1.1, popup_t=2.1)
    else:
        _emit_track("Evasive-FPV-01", "fpv_quad_small", 35.0, 118.0, 9.0, 28.0, 34.0, 11.0, 5.8, 3.2, 0.62, popup_t=2.4)

    return "\n".join(rows) + "\n"


def _classify_aspect_from_vel(pos: np.ndarray, vel: np.ndarray) -> str:
    spd = float(np.linalg.norm(vel))
    r_mag = float(np.linalg.norm(pos))
    if spd < 1e-3 or r_mag < 1e-3:
        return "head_on_frontal"
    # Vector from drone to turret vs drone velocity vector
    u_to_turret = -pos / r_mag
    u_vel = vel / spd
    cos_head = float(np.clip(np.dot(u_vel, u_to_turret), -1.0, 1.0))
    elev_deg = math.degrees(math.asin(float(np.clip(pos[2] / r_mag, -1.0, 1.0))))
    if abs(elev_deg) > 62.0:
        return "planform_top"
    if cos_head > 0.78:
        return "head_on_frontal"
    if cos_head > 0.35:
        return "oblique_quartering"
    return "flank_side"


def parse_drone_track_csv(
    csv_text: str,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    turret_alt_m: float = 0.0,
) -> dict[str, Any]:
    """Parse local ENU or GPS/MAVLink drone telemetry CSV into sorted multi-track kinematics."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("Track CSV has no header row.")
    fmap = {f.strip().lower(): f for f in reader.fieldnames}

    t_col = next((fmap[c] for c in ("t_s", "time_s", "timestamp", "time") if c in fmap), None)
    if t_col is None:
        raise ValueError("Track CSV requires a time column (`t_s`, `time_s`, or `timestamp`).")

    id_col = next((fmap[c] for c in ("track_id", "drone_id", "id", "target_id") if c in fmap), None)
    mod_col = next((fmap[c] for c in ("drone_model", "model", "airframe") if c in fmap), None)

    x_col = next((fmap[c] for c in ("x_m", "east_m", "x") if c in fmap), None)
    y_col = next((fmap[c] for c in ("y_m", "north_m", "y") if c in fmap), None)
    z_col = next((fmap[c] for c in ("z_m", "alt_m", "altitude_m", "z") if c in fmap), None)

    lat_col = next((fmap[c] for c in ("lat_deg", "lat", "latitude") if c in fmap), None)
    lon_col = next((fmap[c] for c in ("lon_deg", "lon", "longitude") if c in fmap), None)

    is_gps = (x_col is None or y_col is None) and (lat_col is not None and lon_col is not None)
    if (x_col is None or y_col is None) and not is_gps:
        raise ValueError(
            "Track CSV requires local (`x_m`, `y_m`, `z_m`) or GPS (`lat`/`lat_deg`, `lon`/`lon_deg`, `alt_m`) columns."
        )

    raw_rows = list(reader)
    if len(raw_rows) < 2:
        raise ValueError("Track CSV requires at least 2 telemetry samples.")

    grouped_raw: dict[str, list[tuple[float, float, float, float, str]]] = {}
    if is_gps:
        lats: list[float] = []
        lons: list[float] = []
        for row in raw_rows:
            try:
                lats.append(float(row[lat_col]))  # type: ignore[index]
                lons.append(float(row[lon_col]))  # type: ignore[index]
            except (TypeError, ValueError):
                continue
        if not lats:
            raise ValueError("No valid GPS latitude/longitude rows parsed.")
        lat0 = float(origin_lat) if origin_lat is not None else float(lats[-1])
        lon0 = float(origin_lon) if origin_lon is not None else float(lons[-1])
        cos_lat = max(0.05, math.cos(math.radians(lat0)))
        for row in raw_rows:
            try:
                tv = float(row[t_col])
                lat_v = float(row[lat_col])  # type: ignore[index]
                lon_v = float(row[lon_col])  # type: ignore[index]
                alt_v = float(row[z_col]) - float(turret_alt_m) if z_col else 20.0
                xm = math.radians(lon_v - lon0) * EARTH_RADIUS_M * cos_lat
                ym = math.radians(lat_v - lat0) * EARTH_RADIUS_M
                tid = str(row[id_col]).strip() if (id_col and str(row[id_col]).strip()) else "Track-1"
                dmod = str(row[mod_col]).strip() if (mod_col and str(row[mod_col]).strip()) else "fpv_quad_small"
                if all(math.isfinite(v) for v in (tv, xm, ym, alt_v)):
                    grouped_raw.setdefault(tid, []).append((tv, xm, ym, alt_v, dmod))
            except (TypeError, ValueError):
                continue
    else:
        for row in raw_rows:
            try:
                tv = float(row[t_col])
                xm = float(row[x_col])  # type: ignore[index]
                ym = float(row[y_col])  # type: ignore[index]
                zm = float(row[z_col]) - float(turret_alt_m) if z_col else 20.0
                tid = str(row[id_col]).strip() if (id_col and str(row[id_col]).strip()) else "Track-1"
                dmod = str(row[mod_col]).strip() if (mod_col and str(row[mod_col]).strip()) else "fpv_quad_small"
                if all(math.isfinite(v) for v in (tv, xm, ym, zm)):
                    grouped_raw.setdefault(tid, []).append((tv, xm, ym, zm, dmod))
            except (TypeError, ValueError):
                continue

    if not grouped_raw:
        raise ValueError("No valid finite trajectory points found in CSV.")

    tracks: dict[str, dict[str, Any]] = {}
    min_t_global = float("inf")
    max_t_global = float("-inf")
    for tid, pts in grouped_raw.items():
        pts.sort(key=lambda item: item[0])
        if len(pts) < 2:
            continue
        ts = np.array([p[0] for p in pts], dtype=np.float64)
        pos = np.array([[p[1], p[2], p[3]] for p in pts], dtype=np.float64)
        min_t_global = min(min_t_global, float(ts[0]))
        max_t_global = max(max_t_global, float(ts[-1]))

        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        for j in range(3):
            vel[:, j] = np.gradient(pos[:, j], ts, edge_order=1)
            if len(ts) >= 3:
                acc[:, j] = np.gradient(vel[:, j], ts, edge_order=1)

        rngs = np.linalg.norm(pos, axis=1)
        spds = np.linalg.norm(vel, axis=1)
        azs = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
        horiz = np.hypot(pos[:, 0], pos[:, 1])
        els = np.degrees(np.arctan2(pos[:, 2], np.maximum(horiz, 1e-4)))

        lat_g = np.zeros(len(ts), dtype=np.float64)
        omega_deg_s = np.zeros(len(ts), dtype=np.float64)
        aspects: list[str] = []
        for k in range(len(ts)):
            aspects.append(_classify_aspect_from_vel(pos[k], vel[k]))
            v_k = vel[k]
            s_k = max(float(spds[k]), 1e-3)
            u_v = v_k / s_k
            a_k = acc[k]
            a_lat_vec = a_k - float(np.dot(a_k, u_v)) * u_v
            lat_g[k] = float(np.linalg.norm(a_lat_vec)) / 9.80665
            r_k = max(float(rngs[k]), 1.0)
            v_tang = float(np.linalg.norm(v_k - float(np.dot(v_k, pos[k] / r_k)) * (pos[k] / r_k)))
            omega_deg_s[k] = math.degrees(v_tang / r_k)

        tracks[tid] = {
            "track_id": tid,
            "drone_model": pts[0][4],
            "t_s": ts,
            "pos_m": pos,
            "vel_mps": vel,
            "acc_mps2": acc,
            "range_m": rngs,
            "speed_mps": spds,
            "az_deg": azs,
            "el_deg": els,
            "lat_g": lat_g,
            "omega_deg_s": omega_deg_s,
            "aspects": aspects,
        }

    if not tracks:
        raise ValueError("All parsed tracks contained fewer than 2 distinct samples.")

    return {
        "coord_mode": "gps_enu" if is_gps else "local_enu",
        "n_tracks": len(tracks),
        "t_min_s": round(min_t_global, 3),
        "t_max_s": round(max_t_global, 3),
        "tracks": tracks,
    }


def simulate_recorded_track_engagement_mc(
    csv_text_or_parsed: str | dict[str, Any],
    turret_config: dict[str, Any] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    breach_radius_m: float = 12.0,
    n_mc_trials: int = 40,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate a configured/calibrated C-UAS shotgun turret engaging recorded drone flight tracks."""
    parsed = (
        parse_drone_track_csv(csv_text_or_parsed)
        if isinstance(csv_text_or_parsed, str)
        else csv_text_or_parsed
    )
    blue_n = normalize_blue_config(turret_config or {})
    cal = dict(calibration_profile or {})

    spread_factor = float(cal.get("spread_factor", 1.0))
    aim_bias_mrad = float(cal.get("aim_bias_mrad", 0.0))
    mv_scale = float(cal.get("mv_scale", 1.0))
    dv_dt = float(cal.get("dv_dt_mps_per_c", 0.72))
    ambient_temp_c = float(cal.get("ambient_temp_c", 21.1))
    mv_sd_mps = float(cal.get("mv_sd_mps", 0.0))

    choke_defs = get_choke_definitions()
    drone_specs = get_drone_target_specs()
    sensor = CUAS_SENSOR_PRESETS.get(blue_n["sensor_preset"], CUAS_SENSOR_PRESETS["Sensor_Fused_Radar_EOIR"])
    sigma_az_rad = float(sensor["sigma_az_mrad"]) / 1000.0
    t_lat = float(sensor["fcs_latency_ms"]) / 1000.0
    slew_rad_s = math.radians(blue_n["slew_rate_deg_s"])
    rho_air = 1.225

    def _ring_const(ammo_k: str, choke_k: str) -> dict[str, float]:
        _, a_spec = _resolve_shotgun_ammo(ammo_k)
        ec_spec = CUAS_AMMO_ECONOMICS_CATALOG.get(ammo_k, CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])
        d_m = float(a_spec["pellet_diameter_mm"]) / 1000.0
        m_kg = float(a_spec["pellet_mass_g"]) / 1000.0
        cd_v = float(a_spec.get("effective_cd", 0.47))
        area_p = math.pi * 0.25 * (d_m ** 2)
        c_inf = choke_defs.get(choke_k, choke_defs["Modified"])
        sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_inf["benchmark_40yd_30in_pct"])))
        return {
            "unit_cost_usd": float(ec_spec["unit_cost_usd"]),
            "pellets": float(a_spec["pellets_per_shot"]),
            "m_kg": m_kg,
            "v0": float(a_spec["v0_mps"]),
            "kappa": (rho_air * cd_v * area_p) / (2.0 * max(m_kg, 1e-6)),
            "theta_choke": sigma_ref / 36.58,
        }

    outer_r = _ring_const(blue_n["ammunition_preset"], blue_n["choke"])
    is_hybrid = bool(blue_n.get("hybrid_mode", False))
    inner_r = _ring_const(blue_n["inner_ammunition_preset"], blue_n["inner_choke"]) if is_hybrid else outer_r
    n_barrels = int(blue_n["n_barrels"])
    if is_hybrid:
        n_inner = max(1, min(n_barrels - 1, round(n_barrels * float(blue_n["inner_barrel_fraction"]))))
        n_outer = n_barrels - n_inner
    else:
        n_inner = 0
        n_outer = n_barrels
    bpb = int(blue_n["barrels_per_burst"])
    rel_t = float(blue_n["reload_time_s"])
    slew_rate = float(blue_n["slew_rate_deg_s"])
    slew_accel = max(420.0, slew_rate * 4.0)
    engage_max = float(blue_n["max_engagement_range_m"])

    # Precompute track geometry lookups & time-indexed kinematic interpolations
    tracks_raw = parsed["tracks"]
    t0_glob = float(parsed["t_min_s"])
    t1_glob = float(parsed["t_max_s"])
    track_meta: dict[str, dict[str, Any]] = {}
    for tid, tr in tracks_raw.items():
        d_mod = tr["drone_model"]
        d_sp = drone_specs.get(d_mod, drone_specs["fpv_quad_small"])
        asp_areas: dict[str, float] = {}
        for asp in ("head_on_frontal", "oblique_quartering", "flank_side", "planform_top"):
            g_res = compute_drone_projected_geometry(d_sp, aspect_mode=asp)
            asp_areas[asp] = float(g_res.get("presented_area_m2", 0.02))
        track_meta[tid] = {
            "spec": d_sp,
            "asp_areas": asp_areas,
            "e_rotor": float(d_sp.get("lethal_energy_rotor_j", 8.0)),
            "e_core": float(d_sp.get("lethal_energy_core_j", 24.0)),
            "n_crit": int(d_sp.get("critical_hits_for_kill", 3)),
        }

    def _state_at(tr: dict[str, Any], t_q: float) -> dict[str, Any] | None:
        ts = tr["t_s"]
        if t_q < float(ts[0]) - 1e-6 or t_q > float(ts[-1]) + 1e-6:
            return None
        idx = int(np.searchsorted(ts, t_q, side="left"))
        idx = max(0, min(len(ts) - 1, idx))
        if idx > 0 and idx < len(ts):
            w = float((t_q - ts[idx - 1]) / max(ts[idx] - ts[idx - 1], 1e-6))
            w = max(0.0, min(1.0, w))
            pos = (1.0 - w) * tr["pos_m"][idx - 1] + w * tr["pos_m"][idx]
            spd = float((1.0 - w) * tr["speed_mps"][idx - 1] + w * tr["speed_mps"][idx])
            lat_g = float((1.0 - w) * tr["lat_g"][idx - 1] + w * tr["lat_g"][idx])
            omega_ds = float((1.0 - w) * tr["omega_deg_s"][idx - 1] + w * tr["omega_deg_s"][idx])
        else:
            pos = tr["pos_m"][idx]
            spd = float(tr["speed_mps"][idx])
            lat_g = float(tr["lat_g"][idx])
            omega_ds = float(tr["omega_deg_s"][idx])
        rng_m = float(np.linalg.norm(pos))
        az_d = math.degrees(math.atan2(float(pos[1]), float(pos[0]))) % 360.0
        el_d = math.degrees(math.atan2(float(pos[2]), max(math.hypot(float(pos[0]), float(pos[1])), 1e-4)))
        return {
            "x_m": float(pos[0]),
            "y_m": float(pos[1]),
            "z_m": float(pos[2]),
            "range_m": rng_m,
            "speed_mps": spd,
            "lat_g": lat_g,
            "omega_deg_s": omega_ds,
            "az_deg": az_d,
            "el_deg": el_d,
            "aspect": tr["aspects"][idx],
        }

    def _eval_shot_pk(
        st: dict[str, Any],
        meta: dict[str, Any],
        ring_p: dict[str, float],
        n_bbls: int,
        ray_draw: float,
        mv_delta: float,
    ) -> float:
        R = max(float(st["range_m"]), float(breach_radius_m))
        v0_eff = max(180.0, ring_p["v0"] * mv_scale + dv_dt * (ambient_temp_c - 21.1) + mv_delta)
        t_tof, v_imp = exact_quadratic_drag_tof_and_velocity(R, v0_eff, ring_p["kappa"])
        ke_j = 0.5 * ring_p["m_kg"] * (v_imp ** 2)
        if ke_j < meta["e_rotor"]:
            return 0.0
        t_eff = t_tof + t_lat
        a_eff = min(6.0, float(st["lat_g"])) * 9.80665
        e_maneuver = (0.24 if blue_n["predictor_mode"] == "kalman_alpha_beta_gamma" else 0.52) * a_eff * (t_eff ** 2)
        omega_req = math.radians(float(st["omega_deg_s"]))
        servo_lag_m = max(0.0, omega_req - slew_rad_s) * t_eff * R
        sigma_meas = max(0.02, sigma_az_rad * R * 1.15)
        bias_m = (aim_bias_mrad / 1000.0) * R
        e_lead_tot = e_maneuver + servo_lag_m + ray_draw * sigma_meas + 0.35 * bias_m

        sigma_r = (0.00463 + ring_p["theta_choke"] * R * ((R / 36.58) ** 0.15)) * spread_factor
        if blue_n["pointing_mode"] == "divergent_curtain":
            sigma_eff = math.hypot(sigma_r, 0.58 * (float(blue_n["splay_angle_mrad"]) / 1000.0) * R)
        elif blue_n["pointing_mode"] == "staggered_tiling":
            sigma_eff = 1.24 * sigma_r
        else:
            sigma_eff = sigma_r

        sigma_eff_sq = max(sigma_eff ** 2, 1e-6)
        r_eq_sq = meta["asp_areas"].get(st["aspect"], 0.02) / math.pi
        p_pel_c = 1.0 - math.exp(-r_eq_sq / (2.0 * sigma_eff_sq))
        off_ax = math.exp(-(e_lead_tot ** 2) / (2.0 * sigma_eff_sq))
        p_pel = p_pel_c * off_ax * (0.45 + 0.55 * (sigma_r / max(sigma_eff, sigma_r)))
        energy_fac = 0.55 + 0.45 * min(1.0, max(0.0, ke_j - meta["e_rotor"]) / max(meta["e_core"] - meta["e_rotor"], 1.0))
        lam = min(60.0, n_bbls * ring_p["pellets"] * p_pel * 0.80 * energy_fac)
        p_surv = 0.0
        term = math.exp(-lam)
        for k in range(meta["n_crit"]):
            p_surv += term
            term *= lam / float(k + 1)
        return float(max(0.0, min(1.0, 1.0 - p_surv)))

    rng = np.random.default_rng(seed)
    n_trials = max(1, int(n_mc_trials))
    track_ids = list(tracks_raw.keys())
    track_kill_acc = {tid: 0 for tid in track_ids}
    track_int_ranges: dict[str, list[float]] = {tid: [] for tid in track_ids}
    zero_breach_trials = 0
    rounds_list: list[int] = []
    costs_list: list[float] = []
    sample_events: list[dict[str, Any]] = []
    dt = 0.10

    for ep in range(n_trials):
        alive = {tid: True for tid in track_ids}
        breached = {tid: False for tid in track_ids}
        t_now = t0_glob
        tur_az = float(tracks_raw[track_ids[0]]["az_deg"][0])
        tur_el = float(tracks_raw[track_ids[0]]["el_deg"][0])
        inner_ready = n_inner
        outer_ready = n_outer
        b_ready = n_barrels
        busy_until = t0_glob
        ep_rounds = 0
        ep_cost = 0.0
        mv_delta = float(rng.normal(0.0, mv_sd_mps)) if mv_sd_mps > 0.0 else 0.0

        while t_now <= t1_glob + 1e-6:
            cands: list[tuple[float, str, dict[str, Any]]] = []
            for tid in track_ids:
                if not alive[tid]:
                    continue
                st = _state_at(tracks_raw[tid], t_now)
                if st is None:
                    continue
                if st["range_m"] <= breach_radius_m:
                    alive[tid] = False
                    breached[tid] = True
                    continue
                if st["range_m"] <= engage_max:
                    ttb = max(0.0, st["range_m"] - breach_radius_m) / max(st["speed_mps"], 1.0)
                    cands.append((ttb, tid, st))

            if not any(alive.values()):
                break

            if t_now >= busy_until:
                if b_ready <= 0:
                    busy_until = t_now + rel_t
                    inner_ready = n_inner
                    outer_ready = n_outer
                    b_ready = n_barrels
                elif cands:
                    cands.sort(key=lambda item: item[0])
                    _, tgt_id, tgt_st = cands[0]
                    d_az = abs(((tgt_st["az_deg"] - tur_az + 180.0) % 360.0) - 180.0)
                    d_el = abs(tgt_st["el_deg"] - tur_el)
                    t_slew = _trapezoidal_slew_time(math.hypot(d_az, d_el), slew_rate, slew_accel)
                    if t_slew > 0.12:
                        tur_az, tur_el = tgt_st["az_deg"], tgt_st["el_deg"]
                        busy_until = t_now + t_slew
                    else:
                        if is_hybrid:
                            use_in = (tgt_st["range_m"] > float(blue_n["hybrid_switch_range_m"]) and inner_ready > 0) or (outer_ready <= 0)
                            if use_in:
                                r_shot = min(inner_ready, bpb)
                                inner_ready -= r_shot
                                ring_sel = inner_r
                            else:
                                r_shot = min(outer_ready, bpb)
                                outer_ready -= r_shot
                                ring_sel = outer_r
                            b_ready = inner_ready + outer_ready
                            n_bbls_eff = r_shot
                        else:
                            r_shot = min(b_ready, bpb)
                            b_ready -= r_shot
                            ring_sel = outer_r
                            n_bbls_eff = bpb

                        ep_rounds += r_shot
                        ep_cost += r_shot * ring_sel["unit_cost_usd"]
                        pk_shot = _eval_shot_pk(tgt_st, track_meta[tgt_id], ring_sel, n_bbls_eff, float(rng.rayleigh(1.0)), mv_delta)
                        killed = float(rng.random()) < pk_shot
                        settle = 0.25 + float(blue_n["ripple_delay_ms"]) * max(0, r_shot - 1) / 1000.0
                        busy_until = t_now + settle
                        tur_az, tur_el = tgt_st["az_deg"], tgt_st["el_deg"]
                        if ep == 0:
                            sample_events.append(
                                {
                                    "t_s": round(t_now, 2),
                                    "track_id": tgt_id,
                                    "x_m": round(tgt_st["x_m"], 2),
                                    "y_m": round(tgt_st["y_m"], 2),
                                    "z_m": round(tgt_st["z_m"], 2),
                                    "range_m": round(tgt_st["range_m"], 1),
                                    "burst_pk_pct": round(100.0 * pk_shot, 1),
                                    "killed": killed,
                                }
                            )
                        if killed:
                            alive[tgt_id] = False
                            track_kill_acc[tgt_id] += 1
                            track_int_ranges[tgt_id].append(tgt_st["range_m"])

            t_now += dt

        for tid in track_ids:
            if alive[tid]:
                # Check if trajectory ended inside or clipped keepout radius
                min_r = float(np.min(tracks_raw[tid]["range_m"]))
                if min_r <= breach_radius_m + 0.5:
                    breached[tid] = True

        if not any(breached.values()):
            zero_breach_trials += 1
        rounds_list.append(ep_rounds)
        costs_list.append(ep_cost)

    # Analytical P_kill time series over primary track for clean inspection plots
    prim_tid = track_ids[0]
    prim_tr = tracks_raw[prim_tid]
    prim_meta = track_meta[prim_tid]
    time_series: list[dict[str, Any]] = []
    for k in range(len(prim_tr["t_s"])):
        r_k = float(prim_tr["range_m"][k])
        st_k = {
            "range_m": r_k,
            "speed_mps": float(prim_tr["speed_mps"][k]),
            "lat_g": float(prim_tr["lat_g"][k]),
            "omega_deg_s": float(prim_tr["omega_deg_s"][k]),
            "aspect": prim_tr["aspects"][k],
        }
        ring_k = inner_r if (is_hybrid and r_k > float(blue_n["hybrid_switch_range_m"])) else outer_r
        pk_k = _eval_shot_pk(st_k, prim_meta, ring_k, bpb, 0.85, 0.0) if r_k <= engage_max else 0.0
        time_series.append(
            {
                "t_s": round(float(prim_tr["t_s"][k]), 2),
                "range_m": round(r_k, 1),
                "speed_mps": round(float(prim_tr["speed_mps"][k]), 1),
                "lat_g": round(float(prim_tr["lat_g"][k]), 2),
                "omega_deg_s": round(float(prim_tr["omega_deg_s"][k]), 1),
                "aspect": prim_tr["aspects"][k],
                "inst_burst_pk_pct": round(100.0 * pk_k, 1),
                "x_m": round(float(prim_tr["pos_m"][k, 0]), 2),
                "y_m": round(float(prim_tr["pos_m"][k, 1]), 2),
                "z_m": round(float(prim_tr["pos_m"][k, 2]), 2),
            }
        )

    track_rows = [
        {
            "track_id": tid,
            "drone_model": tracks_raw[tid]["drone_model"],
            "n_samples": int(len(tracks_raw[tid]["t_s"])),
            "peak_speed_mps": round(float(np.max(tracks_raw[tid]["speed_mps"])), 1),
            "peak_lat_g": round(float(np.max(tracks_raw[tid]["lat_g"])), 2),
            "min_range_m": round(float(np.min(tracks_raw[tid]["range_m"])), 1),
            "kill_rate_pct": round(100.0 * track_kill_acc[tid] / n_trials, 1),
            "mean_intercept_range_m": round(float(np.mean(track_int_ranges[tid])), 1) if track_int_ranges[tid] else 0.0,
        }
        for tid in track_ids
    ]

    mean_kr = float(np.mean([r["kill_rate_pct"] for r in track_rows])) if track_rows else 0.0
    all_int_r = [r for sub in track_int_ranges.values() for r in sub]
    return {
        "summary": {
            "coord_mode": parsed["coord_mode"],
            "n_tracks": parsed["n_tracks"],
            "turret_name": blue_n["name"],
            "zero_breach_pct": round(100.0 * zero_breach_trials / n_trials, 1),
            "mean_kill_rate_pct": round(mean_kr, 1),
            "mean_intercept_range_m": round(float(np.mean(all_int_r)), 1) if all_int_r else 0.0,
            "mean_rounds_expended": round(float(np.mean(rounds_list)), 1),
            "mean_ammo_cost_usd": round(float(np.mean(costs_list)), 2),
            "peak_lat_g": round(max(r["peak_lat_g"] for r in track_rows), 2),
            "peak_speed_mps": round(max(r["peak_speed_mps"] for r in track_rows), 1),
            "calibrated_twin_active": bool(calibration_profile),
        },
        "tracks": track_rows,
        "track_paths": {
            tid: {
                "x_m": [round(float(v), 2) for v in tr["pos_m"][:, 0]],
                "y_m": [round(float(v), 2) for v in tr["pos_m"][:, 1]],
                "z_m": [round(float(v), 2) for v in tr["pos_m"][:, 2]],
                "t_s": [round(float(v), 2) for v in tr["t_s"]],
            }
            for tid, tr in tracks_raw.items()
        },
        "time_series": time_series,
        "sample_events": sample_events,
    }
