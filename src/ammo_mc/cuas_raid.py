"""Unified Dynamic 3D Cooperative Multi-Turret C-UAS Tactical Raid Engine.

Unifies:
- Multi-turret site defense geometry, 3D building obstacles, friendly No-Fire Zones (NFZs)
  and optional Digital Elevation Model (DEM) terrain line-of-sight masking
- Multi-sensor Kill Chain (Ku-band micro-Doppler radar, EO/IR thermal, acoustic, RF/C2 DF)
  with weather attenuation, fiber-optic RF-immunity, and C2 handoff latency
- Dynamic multi-drone swarm waves (concentrated, pincer, or 360-deg multi-vector; evasive weaves)
- Cooperative Weapon-Target Assignment (WTA), trapezoidal turret slewing, shoot-look-shoot
  cycle times, magazine depletion & reloads, Newton barrel thermal accumulation, and 3D replay
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.cuas_defense import (
    _estimate_burst_pk_analytics,
    _trapezoidal_slew_time,
    evaluate_evasive_drone_state,
)
from ammo_mc.cuas_mission import (
    CUAS_AMMO_ECONOMICS_CATALOG,
    KILL_CHAIN_SENSOR_CATALOG,
    MAGAZINE_SYSTEM_SPECS,
    SITE_DEFENSE_PRESETS,
    _AMMO_PRESET_ALIASES,
    _angle_diff_deg,
    _compute_sensor_nominal_range,
    _ray_hits_circle_2d,
    _resolve_shotgun_ammo,
    _segment_crosses_obstacle_3d,
)
from ammo_mc.small_arms import compute_drone_projected_geometry, get_drone_target_specs


def simulate_unified_tactical_raid_mc(
    site_preset: str = "fob_compound_4_turret",
    custom_turrets: list[dict[str, Any]] | None = None,
    custom_obstacles: list[dict[str, Any]] | None = None,
    custom_nfz: list[dict[str, Any]] | None = None,
    terrain_dem: Any | None = None,
    sensors: list[str] | tuple[str, ...] = ("micro_doppler_radar", "eoir_thermal_optical", "rf_c2_df"),
    weather_condition: str = "clear_day",
    ambient_wind_mps: float = 4.0,
    c2_handoff_s: float = 0.25,
    n_drones: int = 10,
    fiber_optic_fraction: float = 0.30,
    drone_model: str = "fpv_quad_small",
    approach_pattern: str = "multi_vector_360",
    evasion_profile: str = "jinking_weave",
    maneuver_g: float = 1.6,
    threat_speed_mps: float = 22.0,
    threat_altitude_m: float = 22.0,
    stagger_window_s: float = 5.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    magazine_type: str = "box_10rd",
    weapon_max_range_m: float = 65.0,
    elevation_sdz_range_m: float = 280.0,
    salvo_barrels: int = 2,
    salvo_cycle_time_s: float = 0.45,
    turret_slew_rate_deg_s: float = 180.0,
    turret_slew_accel_deg_s2: float = 720.0,
    ambient_temp_c: float = 22.0,
    barrel_mass_kg: float = 1.25,
    n_mc_trials: int = 30,
    seed: int = 42,
    swarm_package: list[dict[str, Any]] | None = None,
    outer_layer_type: str = "none",
) -> dict[str, Any]:
    """Run Monte Carlo time-stepped cooperative multi-turret 3D C-UAS raid defense."""
    rng = np.random.default_rng(seed)
    site = SITE_DEFENSE_PRESETS.get(site_preset, SITE_DEFENSE_PRESETS["fob_compound_4_turret"])
    turrets_in = [dict(t) for t in (custom_turrets if custom_turrets is not None else site["turrets"])]
    obstacles = [dict(o) for o in (custom_obstacles if custom_obstacles is not None else site["obstacles"])]
    nfz_zones = [dict(n) for n in (custom_nfz if custom_nfz is not None else site["nfz_sectors"])]
    asset_r = float(site.get("asset_radius_m", 40.0))

    if terrain_dem is not None:
        from ammo_mc.terrain_dem import segment_crosses_terrain_dem

        obs_z_bases = [float(terrain_dem.elevation_at(float(o["x_m"]), float(o["y_m"]))) for o in obstacles]
        turret_ground_z = [float(terrain_dem.elevation_at(float(t["x_m"]), float(t["y_m"]))) for t in turrets_in]
        center_ground_z = float(terrain_dem.elevation_at(0.0, 0.0))
        terrain_source = terrain_dem.name
    else:
        obs_z_bases = [0.0 for _ in obstacles]
        turret_ground_z = [0.0 for _ in turrets_in]
        center_ground_z = 0.0
        terrain_source = "Flat 0 m Datum"

    canon_ammo, ammo = _resolve_shotgun_ammo(ammunition_preset)
    econ_key = _AMMO_PRESET_ALIASES.get(str(ammunition_preset).lower(), canon_ammo)
    econ_spec = CUAS_AMMO_ECONOMICS_CATALOG.get(econ_key, CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])
    mag_spec = MAGAZINE_SYSTEM_SPECS.get(magazine_type, MAGAZINE_SYSTEM_SPECS["box_10rd"])

    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode="head_on_frontal")
    pres_area_cm2 = float(aspect_geom.get("presented_area_cm2", 250.0))

    active_sensors = [s for s in sensors if s in KILL_CHAIN_SENSOR_CATALOG]
    if not active_sensors:
        active_sensors = ["micro_doppler_radar"]

    nom_det_rf = {
        s: _compute_sensor_nominal_range(s, KILL_CHAIN_SENSOR_CATALOG[s], pres_area_cm2, weather_condition, ambient_wind_mps, False)
        for s in active_sensors
    }
    nom_det_fo = {
        s: _compute_sensor_nominal_range(s, KILL_CHAIN_SENSOR_CATALOG[s], pres_area_cm2, weather_condition, ambient_wind_mps, True)
        for s in active_sensors
    }

    dt = 0.20
    max_t = 35.0
    breach_r = max(10.0, asset_r * 0.48)
    start_r = asset_r + max(float(weapon_max_range_m) + 55.0, 120.0)
    c_steel = 0.49  # kJ / (kg * K)
    tau_cool_s = 180.0
    cookoff_c = 220.0
    heat_per_shell_kj = float(econ_spec["heat_per_shot_kj"])
    unit_cost_usd = float(econ_spec["unit_cost_usd"])
    cap_per_bbl = int(mag_spec["capacity_per_barrel"])
    reload_dur_s = float(mag_spec["reload_time_s"])
    barrels_per_salvo = max(1, int(salvo_barrels))
    if swarm_package:
        n_dr = max(1, sum(max(1, int(pkg.get("count", 1))) for pkg in swarm_package))
    else:
        n_dr = max(1, int(n_drones))
    n_trials = max(1, int(n_mc_trials))

    zero_breach_trials = 0
    kill_rates: list[float] = []
    leakers_list: list[int] = []
    intercept_ranges: list[float] = []
    rounds_totals: list[int] = []
    costs_totals: list[float] = []
    peak_temps: list[float] = []
    reload_fracs: list[float] = []
    handoffs_list: list[int] = []
    overheat_list: list[int] = []
    outer_kills_list: list[int] = []
    pkg_geom_cache: dict[str, tuple[dict[str, Any], dict[str, Any], float]] = {}

    turret_acc: dict[str, dict[str, float]] = {
        t["id"]: {"kills": 0.0, "rounds": 0.0, "reloads": 0.0, "peak_temp_c": 0.0, "handoffs": 0.0}
        for t in turrets_in
    }

    rep_frames: list[dict[str, Any]] = []
    rep_timeline: list[dict[str, Any]] = []
    rep_event_log: list[dict[str, Any]] = []
    rep_idx = min(n_trials // 2, n_trials - 1)
    norm_outer_layer = str(outer_layer_type or "none").strip().lower()
    outer_engage_r = max(float(weapon_max_range_m) + 25.0, asset_r + 45.0)

    for tr_idx in range(n_trials):
        drones: list[dict[str, Any]] = []
        if swarm_package:
            d_counter = 0
            for pkg in swarm_package:
                cnt = max(1, int(pkg.get("count", 1)))
                d_mod = str(pkg.get("drone_model", drone_model))
                if d_mod not in pkg_geom_cache:
                    d_sp = drone_specs.get(d_mod, drone_specs["fpv_quad_small"])
                    d_ag = compute_drone_projected_geometry(d_sp, aspect_mode="head_on_frontal")
                    pkg_geom_cache[d_mod] = (d_sp, d_ag, float(d_ag.get("presented_area_cm2", 250.0)))
                d_sp, d_ag, d_pa = pkg_geom_cache[d_mod]
                pkg_spd = float(pkg.get("speed_mps", threat_speed_mps))
                pkg_alt = float(pkg.get("altitude_m", threat_altitude_m))
                pkg_mg = float(pkg.get("maneuver_g", maneuver_g))
                pkg_fo_frac = float(np.clip(pkg.get("fiber_optic_fraction", fiber_optic_fraction), 0.0, 1.0))
                pkg_delay = float(pkg.get("spawn_delay_s", 0.0))
                az_ctr = pkg.get("az_center_deg")
                az_spr = float(pkg.get("az_spread_deg", 25.0))
                role = str(pkg.get("role", "striker"))
                n_fo_p = int(round(cnt * pkg_fo_frac))
                fo_p = np.array([j < n_fo_p for j in range(cnt)], dtype=bool)
                rng.shuffle(fo_p)
                for j in range(cnt):
                    d_counter += 1
                    if az_ctr is not None:
                        az = (float(az_ctr) + float(rng.uniform(-az_spr, az_spr))) % 360.0
                    else:
                        az = ((d_counter + float(rng.uniform(0.15, 0.85))) * (360.0 / n_dr)) % 360.0
                    az_rad = math.radians(az)
                    ux_in, uy_in = -math.cos(az_rad), -math.sin(az_rad)
                    px0, py0 = start_r * math.cos(az_rad), start_r * math.sin(az_rad)
                    is_fo = bool(fo_p[j])
                    det_map = {
                        s: _compute_sensor_nominal_range(s, KILL_CHAIN_SENSOR_CATALOG[s], d_pa, weather_condition, ambient_wind_mps, is_fo)
                        for s in active_sensors
                    }
                    best_det_r = 0.0
                    det_conf_s = 0.45
                    for s in active_sensors:
                        nr = det_map.get(s, 0.0)
                        if nr <= 0.0:
                            continue
                        sig_f = float(KILL_CHAIN_SENSOR_CATALOG[s]["range_sigma_frac"])
                        r_draw = max(0.0, float(rng.normal(nr, nr * sig_f)))
                        if r_draw > best_det_r:
                            best_det_r = r_draw
                            det_conf_s = max(0.15, float(rng.normal(float(KILL_CHAIN_SENSOR_CATALOG[s]["confirm_latency_s"]), 0.07)))
                    spawn_t = pkg_delay + (0.0 if stagger_window_s <= 0.0 else float(rng.uniform(0.0, max(1.0, stagger_window_s * 0.45))))
                    g0 = center_ground_z if terrain_dem is None else float(terrain_dem.elevation_at(px0, py0))
                    drones.append(
                        {
                            "id": f"D-{d_counter}",
                            "role": role,
                            "az_deg": az,
                            "ux_in": ux_in,
                            "uy_in": uy_in,
                            "lat_x": -uy_in,
                            "lat_y": ux_in,
                            "x_m": px0,
                            "y_m": py0,
                            "z_m": max(g0 + 3.0, center_ground_z + pkg_alt),
                            "spawn_t_s": spawn_t,
                            "speed_mps": pkg_spd,
                            "altitude_m": pkg_alt,
                            "maneuver_g": pkg_mg,
                            "drone_spec": d_sp,
                            "aspect_geom": d_ag,
                            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
                            "is_fiber_optic": is_fo,
                            "det_range_m": best_det_r,
                            "c2_delay_s": det_conf_s + float(c2_handoff_s),
                            "detected_at_s": None,
                            "track_confirmed": False,
                            "outer_checked": False,
                            "status": "inbound",
                            "engaged_by": None,
                            "ever_assigned_by": set(),
                        }
                    )
        else:
            n_fo = int(round(n_dr * float(np.clip(fiber_optic_fraction, 0.0, 1.0))))
            fo_flags = np.array([i < n_fo for i in range(n_dr)], dtype=bool)
            rng.shuffle(fo_flags)

            for i in range(n_dr):
                if approach_pattern == "concentrated_axis":
                    az = (45.0 + float(rng.normal(0.0, 14.0))) % 360.0
                elif approach_pattern == "pincer_dual_axis":
                    base_az = 45.0 if (i % 2 == 0) else 225.0
                    az = (base_az + float(rng.normal(0.0, 12.0))) % 360.0
                else:
                    az = ((i + float(rng.uniform(0.15, 0.85))) * (360.0 / n_dr)) % 360.0

                az_rad = math.radians(az)
                ux_in, uy_in = -math.cos(az_rad), -math.sin(az_rad)
                px0, py0 = start_r * math.cos(az_rad), start_r * math.sin(az_rad)
                is_fo = bool(fo_flags[i])
                det_map = nom_det_fo if is_fo else nom_det_rf
                best_det_r = 0.0
                det_conf_s = 0.45
                for s in active_sensors:
                    nr = det_map.get(s, 0.0)
                    if nr <= 0.0:
                        continue
                    sig_f = float(KILL_CHAIN_SENSOR_CATALOG[s]["range_sigma_frac"])
                    r_draw = max(0.0, float(rng.normal(nr, nr * sig_f)))
                    if r_draw > best_det_r:
                        best_det_r = r_draw
                        det_conf_s = max(0.15, float(rng.normal(float(KILL_CHAIN_SENSOR_CATALOG[s]["confirm_latency_s"]), 0.07)))

                spawn_t = 0.0 if stagger_window_s <= 0.0 else float(rng.uniform(0.0, stagger_window_s))
                g0 = center_ground_z if terrain_dem is None else float(terrain_dem.elevation_at(px0, py0))
                drones.append(
                    {
                        "id": f"D-{i+1}",
                        "role": "striker",
                        "az_deg": az,
                        "ux_in": ux_in,
                        "uy_in": uy_in,
                        "lat_x": -uy_in,
                        "lat_y": ux_in,
                        "x_m": px0,
                        "y_m": py0,
                        "z_m": max(g0 + 3.0, center_ground_z + float(threat_altitude_m)),
                        "spawn_t_s": spawn_t,
                        "speed_mps": float(threat_speed_mps),
                        "altitude_m": float(threat_altitude_m),
                        "maneuver_g": float(maneuver_g),
                        "drone_spec": drone_spec,
                        "aspect_geom": aspect_geom,
                        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
                        "is_fiber_optic": is_fo,
                        "det_range_m": best_det_r,
                        "c2_delay_s": det_conf_s + float(c2_handoff_s),
                        "detected_at_s": None,
                        "track_confirmed": False,
                        "outer_checked": False,
                        "status": "inbound",
                        "engaged_by": None,
                        "ever_assigned_by": set(),
                    }
                )

        t_states: list[dict[str, Any]] = []
        for ti, t in enumerate(turrets_in):
            n_bbl = max(1, int(t.get("n_barrels", 6)))
            mag_cap = n_bbl * cap_per_bbl
            tz = turret_ground_z[ti] + float(t.get("z_m", 3.5))
            t_states.append(
                {
                    "id": t["id"],
                    "x_m": float(t["x_m"]),
                    "y_m": float(t["y_m"]),
                    "z_m": tz,
                    "boresight_az_deg": float(t.get("boresight_az_deg", 0.0)),
                    "sector_half_deg": float(t.get("sector_half_deg", 180.0)),
                    "max_range_m": float(t.get("max_range_m", weapon_max_range_m)),
                    "n_barrels": n_bbl,
                    "az_deg": float(t.get("boresight_az_deg", 0.0)),
                    "elev_deg": 18.0,
                    "busy_until_s": 0.0,
                    "reload_until_s": 0.0,
                    "mag_cap": mag_cap,
                    "mag_left": mag_cap,
                    "temp_c": float(ambient_temp_c),
                    "peak_temp_c": float(ambient_temp_c),
                    "overheated": False,
                    "rounds_fired": 0,
                    "kills": 0,
                    "reloads": 0,
                    "reload_time_s": 0.0,
                    "handoffs": 0,
                }
            )

        trial_kills = 0
        trial_outer_kills = 0
        outer_ab_rounds = 8
        trial_leakers = 0
        trial_rounds = 0
        trial_handoffs = 0
        trial_overheats = 0
        trial_int_ranges: list[float] = []
        is_rep = tr_idx == rep_idx

        n_steps = int(round(max_t / dt)) + 1
        for step_idx in range(n_steps):
            t_now = round(step_idx * dt, 2)

            # 1. Update drone positions & detection confirmation
            active_alive = 0
            tracked_now = 0
            for d in drones:
                if d["status"] in ("killed", "breach"):
                    continue
                active_alive += 1
                d_spd = float(d["speed_mps"])
                d_alt = float(d["altitude_m"])
                d_mg = float(d["maneuver_g"])
                t_fly = max(0.0, t_now - float(d["spawn_t_s"]))
                r_nom = start_r - d_spd * t_fly
                if r_nom <= breach_r:
                    d["status"] = "breach"
                    trial_leakers += 1
                    bx = breach_r * math.cos(math.radians(d["az_deg"]))
                    by = breach_r * math.sin(math.radians(d["az_deg"]))
                    d["x_m"], d["y_m"] = bx, by
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "BREACH",
                                "actor": d["id"],
                                "detail": f"{d['id']} ({'Fiber-Optic' if d['is_fiber_optic'] else 'RF'}) breached inner perimeter",
                            }
                        )
                    continue

                ev_map = {
                    "jinking_weave": "sinusoidal_weave",
                    "sinusoidal_weave": "sinusoidal_weave",
                    "pop_up_terminal": "terminal_popup_jink",
                    "terminal_popup_jink": "terminal_popup_jink",
                    "corkscrew_spiral": "corkscrew_spiral",
                    "straight_dive": "linear_cruise",
                    "linear_cruise": "linear_cruise",
                }.get(str(evasion_profile), "sinusoidal_weave")
                t_ev = (
                    (float(weapon_max_range_m) - r_nom) / max(d_spd, 1.0)
                    if ev_map == "terminal_popup_jink"
                    else t_fly
                )
                ev_off = evaluate_evasive_drone_state(
                    t_s=t_ev,
                    target_distance_m=start_r,
                    velocity_mps=d_spd,
                    heading_deg=0.0,
                    maneuver_profile=ev_map,
                    max_maneuver_g=d_mg,
                    phase_rad=float(d["phase"]),
                )
                lat_disp = 7.5 * math.tanh(float(ev_off["y"]) / 7.5)
                vert_disp = 6.0 * math.tanh(float(ev_off["z"]) / 6.0)
                d_cx = start_r * math.cos(math.radians(d["az_deg"])) + d_spd * t_fly * d["ux_in"]
                d_cy = start_r * math.sin(math.radians(d["az_deg"])) + d_spd * t_fly * d["uy_in"]
                px = d_cx + lat_disp * d["lat_x"]
                py = d_cy + lat_disp * d["lat_y"]
                g_pos = center_ground_z if terrain_dem is None else float(terrain_dem.elevation_at(px, py))
                pz = max(g_pos + 3.0, center_ground_z + d_alt + vert_disp)
                d["x_m"], d["y_m"], d["z_m"] = px, py, pz

                r_from_center = math.hypot(px, py)
                if d["detected_at_s"] is None and r_from_center <= float(d["det_range_m"]):
                    d["detected_at_s"] = t_now
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "DETECT",
                                "actor": d["id"],
                                "detail": f"{d['id']} acquired at r={r_from_center:.1f} m",
                            }
                        )
                if d["detected_at_s"] is not None and (t_now - float(d["detected_at_s"])) >= float(d["c2_delay_s"]):
                    d["track_confirmed"] = True
                    tracked_now += 1

                if norm_outer_layer != "none" and not d["outer_checked"] and r_from_center <= outer_engage_r:
                    d["outer_checked"] = True
                    o_hit = False
                    if norm_outer_layer == "rf_gnss_jammer" and not d["is_fiber_optic"]:
                        o_hit = float(rng.random()) < 0.68
                    elif norm_outer_layer == "hpm_screen":
                        o_hit = float(rng.random()) < 0.45
                    elif norm_outer_layer == "airburst_35mm" and outer_ab_rounds > 0:
                        outer_ab_rounds -= 1
                        p_ab = 0.52 if d_alt >= 16.0 else 0.35
                        o_hit = float(rng.random()) < p_ab
                    if o_hit:
                        d["status"] = "killed"
                        trial_kills += 1
                        trial_outer_kills += 1
                        if is_rep:
                            rep_event_log.append(
                                {
                                    "t_s": t_now,
                                    "event": f"OUTER_{norm_outer_layer.upper()}",
                                    "actor": norm_outer_layer,
                                    "detail": f"Outer layer intercepted {d['id']} ({d['role']}) at r={r_from_center:.1f} m",
                                }
                            )
                        continue

            # 2. Newton cooling & reload progression on all turrets
            frame_beams: list[dict[str, Any]] = []
            for ts in t_states:
                ts["temp_c"] = float(ambient_temp_c) + (ts["temp_c"] - float(ambient_temp_c)) * math.exp(-dt / tau_cool_s)
                if t_now < ts["reload_until_s"]:
                    ts["reload_time_s"] += dt

            # 3. Cooperative Weapon-Target Assignment (shortest TTI unobstructed threats)
            # Clear stale engagement flags if previous shooter killed or is reloading
            for d in drones:
                if d["engaged_by"] is not None:
                    sh = next((t for t in t_states if t["id"] == d["engaged_by"]), None)
                    if sh is None or t_now < sh["reload_until_s"] or t_now >= sh["busy_until_s"] + 1.2:
                        d["engaged_by"] = None

            idle_turrets = [
                ts
                for ts in t_states
                if t_now >= ts["busy_until_s"] and t_now >= ts["reload_until_s"]
            ]

            for ts in idle_turrets:
                if ts["mag_left"] <= 0:
                    ts["reloads"] += 1
                    ts["reload_until_s"] = t_now + reload_dur_s
                    ts["mag_left"] = ts["mag_cap"]
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "RELOAD",
                                "actor": ts["id"],
                                "detail": f"{ts['id']} started magazine reload ({reload_dur_s:.1f} s)",
                            }
                        )
                    continue

                candidates: list[tuple[float, float, float, float, dict[str, Any]]] = []
                for d in drones:
                    if d["status"] != "inbound" or not d["track_confirmed"]:
                        continue
                    dx = d["x_m"] - ts["x_m"]
                    dy = d["y_m"] - ts["y_m"]
                    dz = d["z_m"] - ts["z_m"]
                    hdist = math.hypot(dx, dy)
                    sdist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if sdist > ts["max_range_m"] or hdist < 1e-6:
                        continue
                    ray_az = math.degrees(math.atan2(dy, dx)) % 360.0
                    if abs(_angle_diff_deg(ray_az, ts["boresight_az_deg"])) > ts["sector_half_deg"]:
                        continue
                    if any(
                        _segment_crosses_obstacle_3d((ts["x_m"], ts["y_m"], ts["z_m"]), (d["x_m"], d["y_m"], d["z_m"]), obs, z_base_m=obs_z_bases[oi])
                        for oi, obs in enumerate(obstacles)
                    ):
                        continue
                    if terrain_dem is not None and segment_crosses_terrain_dem((ts["x_m"], ts["y_m"], ts["z_m"]), (d["x_m"], d["y_m"], d["z_m"]), terrain_dem):
                        continue
                    if any(_ray_hits_circle_2d(ts["x_m"], ts["y_m"], dx / hdist, dy / hdist, elevation_sdz_range_m, float(n["x_m"]), float(n["y_m"]), float(n["radius_m"])) for n in nfz_zones):
                        continue

                    r_center = math.hypot(d["x_m"], d["y_m"])
                    dup_penalty = 250.0 if (d["engaged_by"] is not None and d["engaged_by"] != ts["id"]) else 0.0
                    score = r_center + dup_penalty
                    ray_el = math.degrees(math.atan2(dz, hdist))
                    candidates.append((score, sdist, ray_az, ray_el, d))

                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0])
                _, sdist, ray_az, ray_el, chosen_d = candidates[0]

                slew_az_delta = abs(_angle_diff_deg(ray_az, ts["az_deg"]))
                slew_el_delta = abs(max(0.0, ray_el) - ts["elev_deg"])
                slew_delta = math.hypot(slew_az_delta, slew_el_delta)
                if slew_delta > 15.0:
                    t_slew = _trapezoidal_slew_time(
                        slew_delta, float(turret_slew_rate_deg_s), float(turret_slew_accel_deg_s2)
                    )
                    # Coarse slewing toward new target azimuth/elevation before firing
                    ts["az_deg"] = ray_az
                    ts["elev_deg"] = max(0.0, ray_el)
                    ts["busy_until_s"] = t_now + max(dt, t_slew)
                    chosen_d["engaged_by"] = ts["id"]
                    continue

                # Turret aligned -> check handoff & fire salvo!
                if chosen_d["ever_assigned_by"] and ts["id"] not in chosen_d["ever_assigned_by"]:
                    trial_handoffs += 1
                    ts["handoffs"] += 1
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "HANDOFF",
                                "actor": ts["id"],
                                "detail": f"{ts['id']} took over tracking of {chosen_d['id']}",
                            }
                        )
                chosen_d["ever_assigned_by"].add(ts["id"])
                chosen_d["engaged_by"] = ts["id"]

                shells_shot = min(barrels_per_salvo, ts["mag_left"])
                ts["mag_left"] -= shells_shot
                ts["rounds_fired"] += shells_shot
                trial_rounds += shells_shot
                ts["az_deg"] = ray_az
                ts["elev_deg"] = max(0.0, ray_el)
                ts["busy_until_s"] = t_now + max(dt, float(salvo_cycle_time_s))

                # Newton barrel thermal rise shared across active turret barrels
                d_temp = (shells_shot * heat_per_shell_kj) / max(ts["n_barrels"] * barrel_mass_kg * c_steel, 0.2)
                ts["temp_c"] += d_temp
                if ts["temp_c"] > ts["peak_temp_c"]:
                    ts["peak_temp_c"] = ts["temp_c"]
                if ts["temp_c"] >= cookoff_c and not ts["overheated"]:
                    ts["overheated"] = True
                    trial_overheats += 1
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "OVERHEAT",
                                "actor": ts["id"],
                                "detail": f"{ts['id']} crossed {cookoff_c:.0f} C cook-off limit ({ts['temp_c']:.1f} C)",
                            }
                        )

                # Barrel thermal jitter penalty when hot + evasive residual
                thermal_pen = max(0.65, 1.0 - max(0.0, ts["temp_c"] - 140.0) * 0.0025)
                ev_pen = max(0.75, 1.0 - 0.06 * float(chosen_d["maneuver_g"])) if evasion_profile != "straight_dive" else 1.0
                pk_salvo = float(
                    np.clip(
                        _estimate_burst_pk_analytics(
                            slant_range_m=sdist,
                            barrels_fired=shells_shot,
                            pellets_per_shot=int(ammo["pellets_per_shot"]),
                            pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                            pellet_mass_g=float(ammo["pellet_mass_g"]),
                            v0_mps=float(ammo["v0_mps"]),
                            choke_name=choke,
                            drone_spec=chosen_d["drone_spec"],
                            base_cd=float(ammo.get("effective_cd", 0.47)),
                            aspect_mode="head_on_frontal",
                            aspect_geom=chosen_d["aspect_geom"],
                        )
                        * thermal_pen
                        * ev_pen,
                        0.0,
                        0.995,
                    )
                )
                hit = float(rng.random()) < pk_salvo
                if is_rep:
                    frame_beams.append(
                        {
                            "turret_id": ts["id"],
                            "x0": round(ts["x_m"], 2),
                            "y0": round(ts["y_m"], 2),
                            "z0": round(ts["z_m"], 2),
                            "x1": round(chosen_d["x_m"], 2),
                            "y1": round(chosen_d["y_m"], 2),
                            "z1": round(chosen_d["z_m"], 2),
                            "hit": hit,
                        }
                    )
                if hit:
                    chosen_d["status"] = "killed"
                    trial_kills += 1
                    ts["kills"] += 1
                    trial_int_ranges.append(sdist)
                    if is_rep:
                        rep_event_log.append(
                            {
                                "t_s": t_now,
                                "event": "KILL",
                                "actor": ts["id"],
                                "detail": f"{ts['id']} neutralized {chosen_d['id']} at {sdist:.1f} m",
                            }
                        )

            if is_rep and (step_idx % 2 == 0 or frame_beams):
                rep_frames.append(
                    {
                        "t_s": t_now,
                        "drones": [
                            {
                                "id": d["id"],
                                "x_m": round(float(d["x_m"]), 2),
                                "y_m": round(float(d["y_m"]), 2),
                                "z_m": round(float(d["z_m"]), 2),
                                "status": (
                                    d["status"]
                                    if d["status"] in ("killed", "breach")
                                    else ("tracked" if d["track_confirmed"] else "undetected")
                                ),
                                "is_fiber_optic": d["is_fiber_optic"],
                            }
                            for d in drones
                        ],
                        "turrets": [
                            {
                                "id": ts["id"],
                                "x_m": round(ts["x_m"], 2),
                                "y_m": round(ts["y_m"], 2),
                                "z_m": round(ts["z_m"], 2),
                                "az_deg": round(ts["az_deg"], 1),
                                "elev_deg": round(ts["elev_deg"], 1),
                                "temp_c": round(ts["temp_c"], 1),
                                "mag_left": ts["mag_left"],
                                "reloading": t_now < ts["reload_until_s"],
                                "aim_x": round(ts["x_m"] + 24.0 * math.cos(math.radians(ts["az_deg"])), 2),
                                "aim_y": round(ts["y_m"] + 24.0 * math.sin(math.radians(ts["az_deg"])), 2),
                                "aim_z": round(ts["z_m"] + 24.0 * math.sin(math.radians(ts["elev_deg"])), 2),
                            }
                            for ts in t_states
                        ],
                        "fire_beams": frame_beams,
                    }
                )
                rep_timeline.append(
                    {
                        "t_s": t_now,
                        "active_threats": active_alive,
                        "tracked_count": tracked_now,
                        "kills_cum": trial_kills,
                        "breaches_cum": trial_leakers,
                        "rounds_cum": trial_rounds,
                        "mean_temp_c": round(float(np.mean([ts["temp_c"] for ts in t_states])), 2),
                        "max_temp_c": round(float(np.max([ts["temp_c"] for ts in t_states])), 2),
                    }
                )

            if all(d["status"] in ("killed", "breach") for d in drones):
                break

        if trial_leakers == 0:
            zero_breach_trials += 1
        kill_rates.append(100.0 * trial_kills / n_dr)
        leakers_list.append(trial_leakers)
        rounds_totals.append(trial_rounds)
        costs_totals.append(trial_rounds * unit_cost_usd)
        handoffs_list.append(trial_handoffs)
        overheat_list.append(trial_overheats)
        outer_kills_list.append(trial_outer_kills)
        if trial_int_ranges:
            intercept_ranges.extend(trial_int_ranges)

        trial_dur_s = max(t_now, dt)
        max_tr_temp = max((ts["peak_temp_c"] for ts in t_states), default=float(ambient_temp_c))
        peak_temps.append(max_tr_temp)
        rel_pct = float(np.mean([100.0 * ts["reload_time_s"] / trial_dur_s for ts in t_states]))
        reload_fracs.append(rel_pct)

        for ts in t_states:
            acc = turret_acc[ts["id"]]
            acc["kills"] += ts["kills"]
            acc["rounds"] += ts["rounds_fired"]
            acc["reloads"] += ts["reloads"]
            acc["peak_temp_c"] += ts["peak_temp_c"]
            acc["handoffs"] += ts["handoffs"]

    mean_cost = float(np.mean(costs_totals))
    mean_kills = max(float(np.mean([kr * n_dr / 100.0 for kr in kill_rates])), 1e-6)
    turret_stats = [
        {
            "turret_id": t["id"],
            "x_m": t["x_m"],
            "y_m": t["y_m"],
            "kills_mean": round(turret_acc[t["id"]]["kills"] / n_trials, 2),
            "rounds_fired_mean": round(turret_acc[t["id"]]["rounds"] / n_trials, 1),
            "reloads_mean": round(turret_acc[t["id"]]["reloads"] / n_trials, 2),
            "peak_temp_c_mean": round(turret_acc[t["id"]]["peak_temp_c"] / n_trials, 1),
            "handoffs_received_mean": round(turret_acc[t["id"]]["handoffs"] / n_trials, 2),
        }
        for t in turrets_in
    ]

    return {
        "summary": {
            "site_preset": site_preset,
            "site_name": site["name"],
            "asset_radius_m": asset_r,
            "n_turrets": len(turrets_in),
            "n_drones": n_dr,
            "fiber_optic_fraction": float(fiber_optic_fraction),
            "ammo_preset": canon_ammo,
            "ammo_name": ammo["name"],
            "weather_condition": weather_condition,
            "terrain_source": terrain_source,
            "outer_layer_type": norm_outer_layer,
            "outer_layer_kills": round(float(np.mean(outer_kills_list)), 2),
            "zero_breach_pct": round(100.0 * zero_breach_trials / n_trials, 1),
            "mean_kill_rate_pct": round(float(np.mean(kill_rates)), 1),
            "mean_leakers": round(float(np.mean(leakers_list)), 2),
            "mean_intercept_range_m": round(float(np.mean(intercept_ranges)), 1) if intercept_ranges else 0.0,
            "mean_rounds_expended": round(float(np.mean(rounds_totals)), 1),
            "mean_ammo_cost_usd": round(mean_cost, 2),
            "cost_per_kill_usd": round(mean_cost / mean_kills, 2),
            "peak_barrel_temp_c": round(float(np.max(peak_temps)), 1),
            "reload_downtime_pct": round(float(np.mean(reload_fracs)), 1),
            "handoff_count_mean": round(float(np.mean(handoffs_list)), 2),
            "overheat_events_mean": round(float(np.mean(overheat_list)), 2),
            "n_mc_trials": n_trials,
        },
        "turrets": turrets_in,
        "obstacles": obstacles,
        "nfz_zones": nfz_zones,
        "turret_stats": turret_stats,
        "frames": rep_frames,
        "timeline_series": rep_timeline,
        "event_log": rep_event_log[:60],
    }
