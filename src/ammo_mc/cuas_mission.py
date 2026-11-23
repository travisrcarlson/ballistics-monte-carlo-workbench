"""Counter-UAS Mission Engineering Modules: Kill Chain, Multi-Turret Site Defense,
3D Animated Intercept Replay, Sustained Logistics/Economics & Trade-Study Reporting.

All functions in this module are pure simulation/engineering calculations and contain zero UI imports.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

import numpy as np

from ammo_mc.cuas_defense import (
    _estimate_burst_pk_analytics,
    _trapezoidal_slew_time,
    evaluate_evasive_drone_state,
)
from ammo_mc.small_arms import (
    compute_balanced_pair_fire_times,
    compute_drone_projected_geometry,
    configure_array_pointing_vectors,
    evaluate_drone_impact,
    generate_multishotgun_array_layout,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
    get_shotgun_presets,
)

# -------------------------------------------------------------------------
# SHARED AMMO / PRESET ALIAS HELPER
# -------------------------------------------------------------------------

_AMMO_PRESET_ALIASES: dict[str, str] = {
    "12ga_04_buckshot": "12G_NO4_BUCK_C_UAS",
    "12g_no4_buck": "12G_NO4_BUCK_C_UAS",
    "12g_tss_no2": "12G_TSS_NO2_C_UAS",
    "12g_t_shot_steel": "12G_T_SHOT_STEEL_C_UAS",
    "12g_duplex_tss_buck": "12G_DUPLEX_TSS_BUCK_C_UAS",
    "12g_flechette_sabot": "12G_FLECHETTE_SABOT_C_UAS",
}


def _resolve_shotgun_ammo(ammo_key: str) -> tuple[str, dict[str, Any]]:
    """Resolve canonical or aliased shotgun preset key to its ballistic dictionary."""
    presets = get_shotgun_presets()
    canon = _AMMO_PRESET_ALIASES.get(str(ammo_key).lower(), str(ammo_key))
    if canon not in presets:
        canon = "12G_NO4_BUCK_C_UAS" if "12G_NO4_BUCK_C_UAS" in presets else next(iter(presets))
    return canon, presets[canon]


# -------------------------------------------------------------------------
# UPGRADE 1: MULTI-SENSOR ACQUISITION & KILL-CHAIN TIMELINE BUDGET
# -------------------------------------------------------------------------

KILL_CHAIN_SENSOR_CATALOG: dict[str, dict[str, Any]] = {
    "micro_doppler_radar": {
        "name": "Ku-Band Micro-Doppler Radar",
        "short_label": "Micro-Doppler Radar",
        "ref_range_m": 420.0,
        "ref_area_cm2": 250.0,
        "confirm_latency_s": 0.45,
        "range_sigma_frac": 0.08,
        "weather_factors": {
            "clear_day": 1.00,
            "overcast_dusk": 0.98,
            "night_clear": 1.00,
            "heavy_fog_rain": 0.78,
        },
        "fiber_optic_immune": False,
        "description": "All-weather 3D Ku-band radar scaling with rotor micro-Doppler & aspect RCS.",
    },
    "eoir_thermal_optical": {
        "name": "EO/IR Thermal + Optical Gimbal",
        "short_label": "EO/IR Thermal/Optical",
        "ref_range_m": 310.0,
        "ref_area_cm2": 250.0,
        "confirm_latency_s": 0.35,
        "range_sigma_frac": 0.10,
        "weather_factors": {
            "clear_day": 1.00,
            "overcast_dusk": 0.82,
            "night_clear": 0.76,
            "heavy_fog_rain": 0.28,
        },
        "fiber_optic_immune": False,
        "description": "High angular resolution day/LWIR video tracker severely attenuated by heavy fog/rain.",
    },
    "acoustic_array": {
        "name": "3D Passive Acoustic Array",
        "short_label": "Acoustic Rotor Array",
        "ref_range_m": 165.0,
        "ref_area_cm2": 250.0,
        "confirm_latency_s": 0.75,
        "range_sigma_frac": 0.14,
        "weather_factors": {
            "clear_day": 1.00,
            "overcast_dusk": 1.00,
            "night_clear": 1.04,
            "heavy_fog_rain": 0.72,
        },
        "fiber_optic_immune": False,
        "description": "Passive rotor-harmonic bearing cueing degraded by ambient wind noise.",
    },
    "rf_c2_df": {
        "name": "Passive RF / C2 Direction-Finder",
        "short_label": "RF / C2 Link DF",
        "ref_range_m": 650.0,
        "ref_area_cm2": 250.0,
        "confirm_latency_s": 0.25,
        "range_sigma_frac": 0.12,
        "weather_factors": {
            "clear_day": 1.00,
            "overcast_dusk": 1.00,
            "night_clear": 1.00,
            "heavy_fog_rain": 0.94,
        },
        "fiber_optic_immune": True,
        "description": "Long-range RF telemetry ESM; drops to 0 m against fiber-optic or RF-silent autonomous drones.",
    },
}


def get_kill_chain_sensor_catalog() -> dict[str, dict[str, Any]]:
    """Return sensor catalog specifications for the C-UAS detection-to-intercept kill chain."""
    return KILL_CHAIN_SENSOR_CATALOG


def _compute_sensor_nominal_range(
    sensor_key: str,
    spec: dict[str, Any],
    presented_area_cm2: float,
    weather: str,
    ambient_wind_mps: float,
    fiber_optic_autonomous: bool,
) -> float:
    """Compute effective mean detection range for one sensor modality."""
    if fiber_optic_autonomous and bool(spec.get("fiber_optic_immune", False)):
        return 0.0
    ref_r = float(spec["ref_range_m"])
    ref_a = max(float(spec["ref_area_cm2"]), 1.0)
    w_fac = float(spec["weather_factors"].get(weather, 1.0))

    if sensor_key == "micro_doppler_radar":
        # Radar range scales roughly with sigma^(1/4); small floor prevents collapse on nose-on micro quads
        area_scale = (max(presented_area_cm2, 30.0) / ref_a) ** 0.25
        return max(15.0, ref_r * area_scale * w_fac)
    if sensor_key == "eoir_thermal_optical":
        area_scale = (max(presented_area_cm2, 30.0) / ref_a) ** 0.28
        return max(10.0, ref_r * area_scale * w_fac)
    if sensor_key == "acoustic_array":
        wind_atten = 1.0 / (1.0 + 0.045 * max(ambient_wind_mps, 0.0) ** 1.35)
        return max(12.0, ref_r * w_fac * wind_atten)
    # RF C2 direction finding
    return max(20.0, ref_r * w_fac)


def simulate_cuas_kill_chain_mc(
    drone_model: str = "fpv_quad_small",
    aspect_mode: str = "head_on_frontal",
    drone_speed_mps: float = 28.0,
    approach_altitude_m: float = 25.0,
    weather: str = "clear_day",
    ambient_wind_mps: float = 5.0,
    fiber_optic_autonomous: bool = False,
    enabled_sensors: tuple[str, ...] | list[str] = (
        "micro_doppler_radar",
        "eoir_thermal_optical",
        "acoustic_array",
        "rf_c2_df",
    ),
    ai_classify_s: float = 0.6,
    human_in_the_loop: bool = True,
    human_auth_s: float = 1.2,
    slew_angle_deg: float = 60.0,
    turret_slew_rate_deg_s: float = 120.0,
    max_weapon_range_m: float = 55.0,
    keepout_breach_radius_m: float = 15.0,
    n_barrels: int = 6,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    salvo_cycle_time_s: float = 0.85,
    n_trials: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate full C-UAS kill chain from multi-sensor detection through classification, slew, and terminal salvos."""
    rng = np.random.default_rng(seed)
    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode=aspect_mode)
    pres_area_cm2 = float(aspect_geom.get("presented_area_cm2", 250.0))

    _, ammo = _resolve_shotgun_ammo(ammunition_preset)
    v0_mps = float(ammo["v0_mps"])
    d_mm = float(ammo["pellet_diameter_mm"])
    m_g = float(ammo["pellet_mass_g"])
    pellets = int(ammo["pellets_per_shot"])
    base_cd = float(ammo.get("effective_cd", 0.47))

    active_keys = [k for k in enabled_sensors if k in KILL_CHAIN_SENSOR_CATALOG]
    if not active_keys:
        active_keys = ["micro_doppler_radar"]

    nom_ranges: dict[str, float] = {}
    for k, spec in KILL_CHAIN_SENSOR_CATALOG.items():
        if k in active_keys:
            nom_ranges[k] = _compute_sensor_nominal_range(
                k, spec, pres_area_cm2, weather, ambient_wind_mps, fiber_optic_autonomous
            )
        else:
            nom_ranges[k] = 0.0

    # Pre-compute slew time (trapezoidal with 250 deg/s^2 accel)
    t_slew_nom = _trapezoidal_slew_time(slew_angle_deg, turret_slew_rate_deg_s, 250.0)
    barrels_per_salvo = min(2 if n_barrels >= 4 else 1, max(1, n_barrels))
    max_salvos_available = max(1, int(math.ceil(n_barrels / barrels_per_salvo)))

    det_ranges: list[float] = []
    first_shot_ranges: list[float] = []
    t_overhead_list: list[float] = []
    t_window_list: list[float] = []
    salvos_fired_list: list[int] = []
    cum_pks: list[float] = []
    breach_count = 0
    zero_detect_count = 0
    winning_sensors: dict[str, int] = {k: 0 for k in KILL_CHAIN_SENSOR_CATALOG}

    det_confirm_samples: list[float] = []
    classify_samples: list[float] = []
    auth_samples: list[float] = []
    slew_samples: list[float] = []

    for _ in range(n_trials):
        best_r = 0.0
        best_key = ""
        best_conf = 0.5
        for k in active_keys:
            r_nom = nom_ranges[k]
            if r_nom <= 0.0:
                continue
            spec = KILL_CHAIN_SENSOR_CATALOG[k]
            r_draw = max(0.0, float(rng.normal(r_nom, r_nom * float(spec["range_sigma_frac"]))))
            if r_draw > best_r:
                best_r = r_draw
                best_key = k
                best_conf = max(0.10, float(rng.normal(float(spec["confirm_latency_s"]), 0.08)))

        if best_key:
            winning_sensors[best_key] += 1
        if best_r <= keepout_breach_radius_m:
            zero_detect_count += 1
            det_ranges.append(best_r)
            first_shot_ranges.append(0.0)
            t_overhead_list.append(0.0)
            t_window_list.append(0.0)
            salvos_fired_list.append(0)
            cum_pks.append(0.0)
            breach_count += 1
            continue

        t_cls = max(0.15, float(rng.normal(ai_classify_s, 0.10)))
        t_auth = max(0.20, float(rng.normal(human_auth_s, 0.22))) if human_in_the_loop else 0.0
        t_slew = max(0.05, float(rng.normal(t_slew_nom, 0.06)))
        # Slew happens concurrently during late auth if HITL is active, adding residual latency
        t_command = best_conf + t_cls + max(t_auth, t_slew) + 0.12
        t_overhead_list.append(t_command)
        det_confirm_samples.append(best_conf)
        classify_samples.append(t_cls)
        auth_samples.append(t_auth)
        slew_samples.append(t_slew)

        det_ranges.append(best_r)
        r_after_overhead = best_r - drone_speed_mps * t_command
        # Doctrine: wait up to optimal 36m if sufficient window remains to fire full salvo sequence before breach
        r_smart_open = max(36.0, keepout_breach_radius_m + 2.0 + drone_speed_mps * max(max_salvos_available - 1, 0) * salvo_cycle_time_s)
        r_start = min(float(max_weapon_range_m), r_after_overhead, r_smart_open)
        first_shot_ranges.append(max(0.0, r_start))

        if r_start <= keepout_breach_radius_m:
            t_window_list.append(0.0)
            salvos_fired_list.append(0)
            cum_pks.append(0.0)
            breach_count += 1
            continue

        t_avail = (r_start - keepout_breach_radius_m) / max(drone_speed_mps, 1.0)
        t_window_list.append(t_avail)

        # Fire sequential salvos until breach or magazine exhausted
        p_survive = 1.0
        n_fired = 0
        t_elapsed = 0.0
        killed_in_draw = False
        while n_fired < max_salvos_available and (r_start - drone_speed_mps * t_elapsed) > keepout_breach_radius_m:
            r_horiz = r_start - drone_speed_mps * t_elapsed
            slant_r = math.sqrt(r_horiz ** 2 + approach_altitude_m ** 2)
            pk_burst = _estimate_burst_pk_analytics(
                slant_range_m=slant_r,
                barrels_fired=barrels_per_salvo,
                pellets_per_shot=pellets,
                pellet_diameter_mm=d_mm,
                pellet_mass_g=m_g,
                v0_mps=v0_mps,
                choke_name=choke,
                drone_spec=drone_spec,
                base_cd=base_cd,
                aspect_mode=aspect_mode,
                aspect_geom=aspect_geom,
            )
            p_survive *= (1.0 - pk_burst)
            n_fired += 1
            if not killed_in_draw and float(rng.random()) < pk_burst:
                killed_in_draw = True
                break
            t_elapsed += salvo_cycle_time_s

        salvos_fired_list.append(n_fired)
        cum_pks.append((1.0 - p_survive) * 100.0)
        if not killed_in_draw:
            breach_count += 1

    mean_det = float(np.mean(det_ranges)) if det_ranges else 0.0
    mean_first = float(np.mean(first_shot_ranges)) if first_shot_ranges else 0.0
    mean_over = float(np.mean(t_overhead_list)) if t_overhead_list else 0.0
    mean_win = float(np.mean(t_window_list)) if t_window_list else 0.0
    mean_salvos = float(np.mean(salvos_fired_list)) if salvos_fired_list else 0.0
    mean_pk = float(np.mean(cum_pks)) if cum_pks else 0.0
    breach_pct = (breach_count / max(n_trials, 1)) * 100.0

    # Identify limiting bottleneck
    if mean_first <= keepout_breach_radius_m or mean_win < salvo_cycle_time_s * 0.5:
        if fiber_optic_autonomous and nom_ranges.get("micro_doppler_radar", 0.0) == 0.0 and nom_ranges.get("eoir_thermal_optical", 0.0) < 120.0:
            bottleneck = "RF Silence + Optical Attenuation (Short Detection Range)"
        elif human_in_the_loop and human_auth_s >= 1.0:
            bottleneck = "Human-in-the-Loop Authorization Latency"
        elif drone_speed_mps * mean_over > mean_det - max_weapon_range_m:
            bottleneck = "C2 + Turret Slew Overhead Consuming Approach Corridor"
        else:
            bottleneck = "Short Weapon Engagement Envelope vs Drone Speed"
    elif mean_pk < 75.0:
        bottleneck = "Single-Burst Lethality vs Aspect Cross-Section"
    else:
        bottleneck = "Well-Balanced Kill Chain (No Critical Bottleneck)"

    modality_breakdown: list[dict[str, Any]] = []
    for k, spec in KILL_CHAIN_SENSOR_CATALOG.items():
        modality_breakdown.append(
            {
                "sensor_key": k,
                "sensor": spec["short_label"],
                "enabled": bool(k in active_keys),
                "nominal_range_m": round(nom_ranges[k], 1),
                "first_detect_wins": int(winning_sensors[k]),
                "first_detect_share_pct": round((winning_sensors[k] / max(n_trials, 1)) * 100.0, 1),
                "status": (
                    "RF-Dark Denied"
                    if (fiber_optic_autonomous and spec["fiber_optic_immune"] and k in active_keys)
                    else ("Active" if k in active_keys else "Disabled")
                ),
            }
        )

    c_conf = float(np.mean(det_confirm_samples)) if det_confirm_samples else 0.45
    c_cls = float(np.mean(classify_samples)) if classify_samples else ai_classify_s
    c_auth = float(np.mean(auth_samples)) if auth_samples else (human_auth_s if human_in_the_loop else 0.0)
    c_slew = float(np.mean(slew_samples)) if slew_samples else t_slew_nom
    t_cursor = 0.0
    timeline_stages: list[dict[str, Any]] = []
    for name, dur, clr in [
        ("1. Sensor Track Lock", c_conf, "#38BDF8"),
        ("2. AI Classification", c_cls, "#A855F7"),
        ("3. Operator Release" if human_in_the_loop else "3. Auto-Engage Gate", max(c_auth, 0.05), "#F59E0B"),
        ("4. Turret Slew & Settle", c_slew, "#F97316"),
        ("5. Engagement Fire Window", max(mean_win, 0.05), "#10B981"),
    ]:
        timeline_stages.append(
            {
                "stage": name,
                "start_s": round(t_cursor, 3),
                "duration_s": round(dur, 3),
                "color": clr,
            }
        )
        t_cursor += dur

    # Analytic sweep over drone speed (10 to 55 m/s) using optimal window scheduling
    speed_sweep: list[dict[str, Any]] = []
    for spd in [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0, 55.0]:
        r_smart = max(36.0, keepout_breach_radius_m + 2.0 + spd * max(max_salvos_available - 1, 0) * salvo_cycle_time_s)
        r_eng = min(float(max_weapon_range_m), max(0.0, mean_det - spd * mean_over), r_smart)
        win_s = max(0.0, (r_eng - keepout_breach_radius_m) / spd)
        n_sal = min(max_salvos_available, 1 + int(win_s // max(salvo_cycle_time_s, 0.1))) if r_eng > keepout_breach_radius_m else 0
        ps = 1.0
        for s_i in range(n_sal):
            rh = r_eng - spd * s_i * salvo_cycle_time_s
            sr = math.sqrt(max(rh, 1.0) ** 2 + approach_altitude_m ** 2)
            ps *= 1.0 - _estimate_burst_pk_analytics(
                sr, barrels_per_salvo, pellets, d_mm, m_g, v0_mps, choke, drone_spec, base_cd, aspect_mode, aspect_geom
            )
        speed_sweep.append(
            {
                "speed_mps": spd,
                "first_shot_range_m": round(r_eng, 1),
                "salvos_fired": n_sal,
                "cumulative_pk_pct": round((1.0 - ps) * 100.0, 1),
            }
        )

    return {
        "summary": {
            "drone_name": drone_spec["name"],
            "aspect_mode": aspect_mode,
            "presented_area_cm2": round(pres_area_cm2, 1),
            "weather": weather,
            "fiber_optic_autonomous": bool(fiber_optic_autonomous),
            "fused_detection_range_m": round(mean_det, 1),
            "first_shot_range_m": round(mean_first, 1),
            "t_overhead_s": round(mean_over, 2),
            "t_engagement_window_s": round(mean_win, 2),
            "mean_salvos_fired": round(mean_salvos, 2),
            "cumulative_pk_pct": round(mean_pk, 1),
            "breach_rate_pct": round(breach_pct, 1),
            "zero_detect_rate_pct": round((zero_detect_count / max(n_trials, 1)) * 100.0, 1),
            "weapon_max_range_m": float(max_weapon_range_m),
            "keepout_breach_radius_m": float(keepout_breach_radius_m),
            "limiting_bottleneck": bottleneck,
        },
        "modality_breakdown": modality_breakdown,
        "timeline_stages": timeline_stages,
        "speed_sweep": speed_sweep,
    }


# -------------------------------------------------------------------------
# UPGRADE 2: MULTI-TURRET SITE DEFENSE & NO-FIRE ZONE (NFZ) PLANNER
# -------------------------------------------------------------------------

SITE_DEFENSE_PRESETS: dict[str, dict[str, Any]] = {
    "fob_compound_4_turret": {
        "name": "Forward Operating Base (4 Corner Turrets + 2 Hangars + Convoy Gate NFZ)",
        "asset_radius_m": 45.0,
        "turrets": [
            {"id": "T1-NE", "x_m": 28.0, "y_m": 28.0, "z_m": 4.0, "boresight_az_deg": 45.0, "sector_half_deg": 135.0, "n_barrels": 6},
            {"id": "T2-NW", "x_m": -28.0, "y_m": 28.0, "z_m": 4.0, "boresight_az_deg": 135.0, "sector_half_deg": 135.0, "n_barrels": 6},
            {"id": "T3-SW", "x_m": -28.0, "y_m": -28.0, "z_m": 4.0, "boresight_az_deg": 225.0, "sector_half_deg": 135.0, "n_barrels": 6},
            {"id": "T4-SE", "x_m": 28.0, "y_m": -28.0, "z_m": 4.0, "boresight_az_deg": 315.0, "sector_half_deg": 135.0, "n_barrels": 6},
        ],
        "obstacles": [
            {"name": "Command Bunker", "x_m": 0.0, "y_m": 8.0, "width_m": 16.0, "length_m": 14.0, "height_m": 18.0},
            {"name": "UAV Hangar West", "x_m": -12.0, "y_m": -14.0, "width_m": 14.0, "length_m": 18.0, "height_m": 15.0},
        ],
        "nfz_sectors": [
            {"name": "Medevac Pad / Fuel Farm", "x_m": 75.0, "y_m": -15.0, "radius_m": 24.0},
            {"name": "Friendly Checkpoint East", "x_m": 15.0, "y_m": 115.0, "radius_m": 26.0},
        ],
    },
    "airfield_corridor_3_turret": {
        "name": "Runway Approach Shield (3 Staggered Turrets + Control Tower + Terminal NFZ)",
        "asset_radius_m": 55.0,
        "turrets": [
            {"id": "T1-North", "x_m": 0.0, "y_m": 42.0, "z_m": 5.0, "boresight_az_deg": 90.0, "sector_half_deg": 150.0, "n_barrels": 8},
            {"id": "T2-West", "x_m": -45.0, "y_m": -20.0, "z_m": 3.5, "boresight_az_deg": 200.0, "sector_half_deg": 140.0, "n_barrels": 6},
            {"id": "T3-East", "x_m": 45.0, "y_m": -20.0, "z_m": 3.5, "boresight_az_deg": 340.0, "sector_half_deg": 140.0, "n_barrels": 6},
        ],
        "obstacles": [
            {"name": "ATC Control Tower", "x_m": 18.0, "y_m": 10.0, "width_m": 12.0, "length_m": 12.0, "height_m": 28.0},
        ],
        "nfz_sectors": [
            {"name": "Civilian Apron", "x_m": -105.0, "y_m": 60.0, "radius_m": 30.0},
        ],
    },
    "vip_building_2_turret": {
        "name": "Rooftop Pair Defense (2 Opposing Rooftop Turrets + Rooftop Penthouse + Crowd NFZ)",
        "asset_radius_m": 35.0,
        "turrets": [
            {"id": "T1-Rooftop-E", "x_m": 22.0, "y_m": 0.0, "z_m": 12.0, "boresight_az_deg": 0.0, "sector_half_deg": 145.0, "n_barrels": 6},
            {"id": "T2-Rooftop-W", "x_m": -22.0, "y_m": 0.0, "z_m": 12.0, "boresight_az_deg": 180.0, "sector_half_deg": 145.0, "n_barrels": 6},
        ],
        "obstacles": [
            {"name": "Central Elevator Penthouse", "x_m": 0.0, "y_m": 0.0, "width_m": 14.0, "length_m": 14.0, "height_m": 24.0},
        ],
        "nfz_sectors": [
            {"name": "Public Plaza South", "x_m": 0.0, "y_m": -95.0, "radius_m": 28.0},
        ],
    },
    "convoy_escort_2_turret": {
        "name": "Lead/Rear Escort Pair (2 Mobile Vehicles + Armoured Truck Mask + Friendly Infantry)",
        "asset_radius_m": 32.0,
        "turrets": [
            {"id": "T1-Lead", "x_m": 24.0, "y_m": 0.0, "z_m": 3.0, "boresight_az_deg": 0.0, "sector_half_deg": 150.0, "n_barrels": 6},
            {"id": "T2-Trail", "x_m": -24.0, "y_m": 0.0, "z_m": 3.0, "boresight_az_deg": 180.0, "sector_half_deg": 150.0, "n_barrels": 6},
        ],
        "obstacles": [
            {"name": "Center Supply Semi", "x_m": 0.0, "y_m": 0.0, "width_m": 16.0, "length_m": 4.5, "height_m": 4.8},
        ],
        "nfz_sectors": [
            {"name": "Dismount Team South", "x_m": 5.0, "y_m": -70.0, "radius_m": 22.0},
        ],
    },
}


def get_site_defense_presets() -> dict[str, dict[str, Any]]:
    """Return preset multi-turret perimeter layouts with obstacles and No-Fire Zones."""
    return SITE_DEFENSE_PRESETS


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Shortest signed angular distance in degrees in [-180, 180]."""
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def _segment_crosses_obstacle_3d(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    obs: dict[str, Any],
    z_base_m: float = 0.0,
) -> bool:
    """Return True if 3D line segment p0->p1 intersects an axis-aligned obstacle prism."""
    xmin = float(obs["x_m"]) - 0.5 * float(obs["width_m"])
    xmax = float(obs["x_m"]) + 0.5 * float(obs["width_m"])
    ymin = float(obs["y_m"]) - 0.5 * float(obs["length_m"])
    ymax = float(obs["y_m"]) + 0.5 * float(obs["length_m"])
    z_min = float(z_base_m)
    z_top = z_min + float(obs["height_m"])

    # Liang-Barsky slab intersection in 3D [0, 1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    t0, t1 = 0.0, 1.0
    for p, q in [
        (-dx, p0[0] - xmin),
        (dx, xmax - p0[0]),
        (-dy, p0[1] - ymin),
        (dy, ymax - p0[1]),
        (-dz, p0[2] - z_min),
        (dz, z_top - p0[2]),
    ]:
        if abs(p) < 1e-12:
            if q < 0.0:
                return False
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
    return t0 <= t1 and 0.02 < t1 < 0.99


def _ray_hits_circle_2d(
    ox: float,
    oy: float,
    ux: float,
    uy: float,
    max_len: float,
    cx: float,
    cy: float,
    rad: float,
) -> bool:
    """Return True if 2D ray (ox, oy) + s*(ux, uy), s in [0, max_len], intersects disk."""
    fx, fy = cx - ox, cy - oy
    proj = fx * ux + fy * uy
    s_closest = max(0.0, min(max_len, proj))
    dx = ox + s_closest * ux - cx
    dy = oy + s_closest * uy - cy
    return (dx * dx + dy * dy) <= (rad * rad)


def simulate_multiturret_site_defense_mc(
    site_preset: str = "fob_compound_4_turret",
    custom_turrets: list[dict[str, Any]] | None = None,
    custom_obstacles: list[dict[str, Any]] | None = None,
    custom_nfz: list[dict[str, Any]] | None = None,
    terrain_dem: Any | None = None,
    threat_altitude_m: float = 22.0,
    threat_speed_mps: float = 22.0,
    drone_model: str = "fpv_quad_small",
    weapon_max_range_m: float = 60.0,
    elevation_sdz_range_m: float = 280.0,
    n_raid_trials: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate 360-deg coverage, obstacle & DEM terrain LOS blind spots, NFZ fallout conflicts, and raids."""
    rng = np.random.default_rng(seed)
    site = SITE_DEFENSE_PRESETS.get(site_preset, SITE_DEFENSE_PRESETS["fob_compound_4_turret"])
    turrets = [dict(t) for t in (custom_turrets if custom_turrets is not None else site["turrets"])]
    obstacles = [dict(o) for o in (custom_obstacles if custom_obstacles is not None else site["obstacles"])]
    nfz_zones = [dict(n) for n in (custom_nfz if custom_nfz is not None else site["nfz_sectors"])]
    asset_r = float(site.get("asset_radius_m", 40.0))

    if terrain_dem is not None:
        from ammo_mc.terrain_dem import segment_crosses_terrain_dem, summarize_site_terrain_fallout

        obs_z_bases = [float(terrain_dem.elevation_at(float(o["x_m"]), float(o["y_m"]))) for o in obstacles]
        turret_ground_z = [float(terrain_dem.elevation_at(float(t["x_m"]), float(t["y_m"]))) for t in turrets]
        center_ground_z = float(terrain_dem.elevation_at(0.0, 0.0))
    else:
        obs_z_bases = [0.0 for _ in obstacles]
        turret_ground_z = [0.0 for _ in turrets]
        center_ground_z = 0.0

    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode="head_on_frontal")
    _, ammo = _resolve_shotgun_ammo("12G_NO4_BUCK_C_UAS")

    # 72 azimuth sectors (every 5 degrees) along active outer intercept ring
    eval_r = min(asset_r + 0.30 * weapon_max_range_m, 0.85 * (asset_r + weapon_max_range_m))
    azimuth_profile: list[dict[str, Any]] = []
    blind_azimuths: list[float] = []
    nfz_conflict_bins = 0
    terrain_masked_bins = 0

    for idx in range(72):
        az = idx * 5.0
        az_rad = math.radians(az)
        tx_tgt = eval_r * math.cos(az_rad)
        ty_tgt = eval_r * math.sin(az_rad)
        g_tgt = float(terrain_dem.elevation_at(tx_tgt, ty_tgt)) if terrain_dem is not None else 0.0
        tz_tgt = max(g_tgt + 3.0, center_ground_z + float(threat_altitude_m))

        active_count = 0
        los_blocked_any = False
        terr_blocked_any = False
        nfz_blocked_any = False

        for t_idx, t in enumerate(turrets):
            tx, ty = float(t["x_m"]), float(t["y_m"])
            tz = turret_ground_z[t_idx] + float(t.get("z_m", 3.5))
            dx, dy, dz = tx_tgt - tx, ty_tgt - ty, tz_tgt - tz
            horiz_dist = math.hypot(dx, dy)
            slant_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            t_range = float(t.get("max_range_m", weapon_max_range_m))
            if slant_dist > t_range + 1e-6 or horiz_dist < 1e-6:
                continue

            ray_az = math.degrees(math.atan2(dy, dx)) % 360.0
            if abs(_angle_diff_deg(ray_az, float(t.get("boresight_az_deg", ray_az)))) > float(t.get("sector_half_deg", 180.0)):
                continue

            if any(
                _segment_crosses_obstacle_3d((tx, ty, tz), (tx_tgt, ty_tgt, tz_tgt), obs, z_base_m=obs_z_bases[oi])
                for oi, obs in enumerate(obstacles)
            ):
                los_blocked_any = True
                continue

            if terrain_dem is not None and segment_crosses_terrain_dem((tx, ty, tz), (tx_tgt, ty_tgt, tz_tgt), terrain_dem):
                terr_blocked_any = True
                los_blocked_any = True
                continue

            ux, uy = dx / horiz_dist, dy / horiz_dist
            if any(_ray_hits_circle_2d(tx, ty, ux, uy, elevation_sdz_range_m, float(n["x_m"]), float(n["y_m"]), float(n["radius_m"])) for n in nfz_zones):
                nfz_blocked_any = True
                continue

            active_count += 1

        if active_count == 0:
            blind_azimuths.append(az)
        if terr_blocked_any:
            terrain_masked_bins += 1
        if nfz_blocked_any:
            nfz_conflict_bins += 1

        azimuth_profile.append(
            {
                "az_deg": az,
                "active_turrets": active_count,
                "los_blocked": bool(los_blocked_any),
                "terrain_masked": bool(terr_blocked_any),
                "nfz_restricted": bool(nfz_blocked_any),
            }
        )

    cov_1plus = sum(1 for row in azimuth_profile if row["active_turrets"] >= 1) / 72.0 * 100.0
    cov_2plus = sum(1 for row in azimuth_profile if row["active_turrets"] >= 2) / 72.0 * 100.0
    blind_pct = len(blind_azimuths) / 72.0 * 100.0
    nfz_conflict_pct = nfz_conflict_bins / 72.0 * 100.0
    terr_masked_pct = terrain_masked_bins / 72.0 * 100.0

    # Monte Carlo incoming raid checks across random azimuths
    turret_kills = {t["id"]: 0 for t in turrets}
    turret_shots = {t["id"]: 0 for t in turrets}
    stopped_raids = 0
    sample_tracks: list[dict[str, Any]] = []
    start_r = asset_r + weapon_max_range_m + 35.0

    for r_i in range(n_raid_trials):
        az = float(rng.uniform(0.0, 360.0))
        az_rad = math.radians(az)
        sx, sy = start_r * math.cos(az_rad), start_r * math.sin(az_rad)
        sz_g = float(terrain_dem.elevation_at(sx, sy)) if terrain_dem is not None else 0.0
        sz = sz_g + float(threat_altitude_m)
        ux_in, uy_in = -math.cos(az_rad), -math.sin(az_rad)

        killed = False
        end_x, end_y = asset_r * 0.6 * math.cos(az_rad), asset_r * 0.6 * math.sin(az_rad)
        end_z = (float(terrain_dem.elevation_at(end_x, end_y)) if terrain_dem is not None else 0.0) + float(threat_altitude_m)
        killer_id = "NONE"

        # Step drone inward along approach path every 0.4s
        steps = max(1, int((start_r - asset_r * 0.7) / max(threat_speed_mps * 0.4, 2.0)))
        for st_idx in range(steps):
            r_now = start_r - st_idx * threat_speed_mps * 0.4
            if r_now <= asset_r * 0.7:
                break
            dx_pos = sx + st_idx * threat_speed_mps * 0.4 * ux_in
            dy_pos = sy + st_idx * threat_speed_mps * 0.4 * uy_in
            g_pos = float(terrain_dem.elevation_at(dx_pos, dy_pos)) if terrain_dem is not None else 0.0
            dz_pos = g_pos + float(threat_altitude_m)

            for t_idx, t in enumerate(turrets):
                tx, ty = float(t["x_m"]), float(t["y_m"])
                tz = turret_ground_z[t_idx] + float(t.get("z_m", 3.5))
                vx, vy, vz = dx_pos - tx, dy_pos - ty, dz_pos - tz
                hdist = math.hypot(vx, vy)
                sdist = math.sqrt(vx * vx + vy * vy + vz * vz)
                t_range = float(t.get("max_range_m", weapon_max_range_m))
                if sdist > t_range or hdist < 1e-6:
                    continue
                ray_az = math.degrees(math.atan2(vy, vx)) % 360.0
                if abs(_angle_diff_deg(ray_az, float(t.get("boresight_az_deg", ray_az)))) > float(t.get("sector_half_deg", 180.0)):
                    continue
                if any(
                    _segment_crosses_obstacle_3d((tx, ty, tz), (dx_pos, dy_pos, dz_pos), obs, z_base_m=obs_z_bases[oi])
                    for oi, obs in enumerate(obstacles)
                ):
                    continue
                if terrain_dem is not None and segment_crosses_terrain_dem((tx, ty, tz), (dx_pos, dy_pos, dz_pos), terrain_dem):
                    continue
                if any(_ray_hits_circle_2d(tx, ty, vx / hdist, vy / hdist, elevation_sdz_range_m, float(n["x_m"]), float(n["y_m"]), float(n["radius_m"])) for n in nfz_zones):
                    continue

                turret_shots[t["id"]] += 1
                pk_b = _estimate_burst_pk_analytics(
                    slant_range_m=sdist,
                    barrels_fired=2,
                    pellets_per_shot=int(ammo["pellets_per_shot"]),
                    pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                    pellet_mass_g=float(ammo["pellet_mass_g"]),
                    v0_mps=float(ammo["v0_mps"]),
                    choke_name="Modified",
                    drone_spec=drone_spec,
                    base_cd=float(ammo.get("effective_cd", 0.47)),
                    aspect_mode="head_on_frontal",
                    aspect_geom=aspect_geom,
                )
                if float(rng.random()) < pk_b:
                    killed = True
                    end_x, end_y, end_z = dx_pos, dy_pos, dz_pos
                    killer_id = t["id"]
                    turret_kills[t["id"]] += 1
                    break
            if killed:
                break

        if killed:
            stopped_raids += 1

        if r_i < 24:
            sample_tracks.append(
                {
                    "id": f"Raid-{r_i+1}",
                    "sx": round(sx, 2),
                    "sy": round(sy, 2),
                    "sz": round(sz, 2),
                    "ex": round(end_x, 2),
                    "ey": round(end_y, 2),
                    "ez": round(end_z, 2),
                    "status": "killed" if killed else "leaker",
                    "killer": killer_id,
                }
            )

    terrain_fallout: dict[str, Any] = {}
    terrain_relief_m = 0.0
    terrain_source = "Flat 0 m Datum"
    if terrain_dem is not None:
        terrain_fallout = summarize_site_terrain_fallout(turrets=turrets, dem=terrain_dem)
        terrain_relief_m = float(terrain_dem.stats()["relief_m"])
        terrain_source = terrain_dem.name

    turret_workload = [
        {
            "turret_id": t["id"],
            "x_m": t["x_m"],
            "y_m": t["y_m"],
            "ground_elev_m": round(turret_ground_z[ti], 2),
            "boresight_az_deg": t.get("boresight_az_deg", 0.0),
            "sector_half_deg": t.get("sector_half_deg", 180.0),
            "bursts_fired": turret_shots[t["id"]],
            "kills": turret_kills[t["id"]],
        }
        for ti, t in enumerate(turrets)
    ]

    return {
        "summary": {
            "site_preset": site_preset,
            "site_name": site["name"],
            "asset_radius_m": asset_r,
            "n_turrets": len(turrets),
            "weapon_max_range_m": float(weapon_max_range_m),
            "elevation_sdz_range_m": float(elevation_sdz_range_m),
            "perimeter_coverage_pct": round(cov_1plus, 1),
            "overlapping_coverage_pct": round(cov_2plus, 1),
            "blind_spot_pct": round(blind_pct, 1),
            "blind_spot_count": len(blind_azimuths),
            "blind_spot_azimuths_deg": blind_azimuths[:12],
            "terrain_masked_pct": round(terr_masked_pct, 1),
            "terrain_relief_m": round(terrain_relief_m, 2),
            "terrain_source": terrain_source,
            "flat_fallout_m": terrain_fallout.get("flat_fallout_m"),
            "terrain_fallout_min_m": terrain_fallout.get("terrain_fallout_min_m"),
            "terrain_fallout_max_m": terrain_fallout.get("terrain_fallout_max_m"),
            "nfz_fallout_conflict_pct": round(nfz_conflict_pct, 1),
            "site_zero_breach_pct": round((stopped_raids / max(n_raid_trials, 1)) * 100.0, 1),
        },
        "turrets": turrets,
        "obstacles": obstacles,
        "nfz_zones": nfz_zones,
        "azimuth_profile": azimuth_profile,
        "sample_tracks": sample_tracks,
        "turret_workload": turret_workload,
        "terrain_fallout_profile": terrain_fallout.get("azimuth_samples", []),
    }


# -------------------------------------------------------------------------
# UPGRADE 3: 3D ANIMATED INTERCEPT REPLAY DATA GENERATOR
# -------------------------------------------------------------------------

def generate_3d_intercept_replay_data(
    mode: str = "shotgun_array",
    layout_type: str = "circular",
    n_barrels: int = 6,
    target_distance_m: float = 35.0,
    drone_model: str = "commercial_quad_medium",
    aspect_mode: str = "head_on_frontal",
    drone_speed_mps: float = 18.0,
    maneuver_profile: str = "sinusoidal_weave",
    maneuver_g: float = 2.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    firing_mode: str = "balanced_pair_ripple",
    ripple_delay_ms: float = 15.0,
    n_frames: int = 32,
    max_visual_pellets: int = 90,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate frame-by-frame 3D trajectories of moving drone + shot-stringed pellets for Plotly animation."""
    rng = np.random.default_rng(seed)
    canon_ammo, ammo = _resolve_shotgun_ammo(ammunition_preset)
    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["commercial_quad_medium"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode=aspect_mode)
    M_rot = np.array(aspect_geom.get("rotation_matrix", np.eye(3)), dtype=np.float64)

    choke_defs = get_choke_definitions()
    c_info = choke_defs.get(choke, choke_defs["Modified"])
    p_bench = float(c_info["benchmark_40yd_30in_pct"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_bench))
    theta_choke = sigma_ref / 36.58

    raw_barrels = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=0.22,
        default_choke=choke,
        default_ammunition_preset=canon_ammo,
    )
    barrels = configure_array_pointing_vectors(
        raw_barrels,
        pointing_mode="divergent_curtain" if len(raw_barrels) > 2 else "convergent",
        intercept_range_m=target_distance_m,
        splay_angle_mrad=3.0 if len(raw_barrels) > 2 else 0.0,
    )
    fire_times_s = compute_balanced_pair_fire_times(barrels, firing_mode=firing_mode, ripple_delay_ms=ripple_delay_ms)

    v0_nom = float(ammo["v0_mps"])
    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    base_cd = float(ammo.get("effective_cd", 0.47))
    rho_air = 1.225
    area_m2 = math.pi * 0.25 * (d_m ** 2)
    kappa_nom = (rho_air * base_cd * area_m2) / (2.0 * max(m_kg, 1e-6))

    # Nominal flight time & FCS lead point (evaluated at nominal impact instant)
    t_flight_nom = (math.exp(kappa_nom * target_distance_m) - 1.0) / max(kappa_nom * v0_nom, 1.0)
    lead_state = evaluate_evasive_drone_state(
        t_flight_nom,
        target_distance_m=target_distance_m,
        velocity_mps=drone_speed_mps,
        heading_deg=75.0,
        maneuver_profile=maneuver_profile,
        max_maneuver_g=maneuver_g,
    )
    lead_x = float(lead_state["x"])
    lead_y = float(lead_state["y"])
    lead_z = float(lead_state["z"])

    # Allocate tracked pellets uniformly across barrels
    n_b = len(barrels)
    per_b = max(3, int(math.ceil(max_visual_pellets / max(n_b, 1))))
    pellet_records: list[dict[str, Any]] = []
    for b_idx, b in enumerate(barrels):
        t_fire = float(fire_times_s[b_idx])
        y0, z0 = float(b["y_m"]), float(b["z_m"])
        aim_dy = lead_y + target_distance_m * (float(b["aim_yaw_mrad"]) / 1000.0)
        aim_dz = lead_z + target_distance_m * (float(b["aim_pitch_mrad"]) / 1000.0)
        sigma_r = 0.01852 / 4.0 + theta_choke * target_distance_m * (target_distance_m / 36.58) ** 0.15

        for _ in range(per_b):
            cd_p = float(rng.uniform(0.88 * base_cd, 1.28 * base_cd))
            kappa_p = (rho_air * cd_p * area_m2) / (2.0 * max(m_kg, 1e-6))
            v0_p = float(rng.normal(v0_nom, 6.0))
            t_arr = t_fire + (math.exp(kappa_p * max(lead_x, 5.0)) - 1.0) / max(kappa_p * v0_p, 1.0)

            d_at_arr = evaluate_evasive_drone_state(
                t_arr, target_distance_m, drone_speed_mps, 75.0, maneuver_profile, maneuver_g
            )
            dy_spread = float(rng.normal(0.0, sigma_r))
            dz_spread = float(rng.normal(0.0, sigma_r))
            y_arr = aim_dy + dy_spread
            z_arr = aim_dz + dz_spread
            v_arr = v0_p / (1.0 + kappa_p * v0_p * max(t_arr - t_fire, 0.0))
            ke_arr = 0.5 * m_kg * (v_arr ** 2)

            hit_info = evaluate_drone_impact(
                y_arr - float(d_at_arr["y"]),
                z_arr - float(d_at_arr["z"]),
                drone_spec,
                striking_energy_j=ke_arr,
                aspect_geometry=aspect_geom,
            )
            pellet_records.append(
                {
                    "t_fire": t_fire,
                    "t_arr": t_arr,
                    "y0": y0,
                    "z0": z0,
                    "v0": v0_p,
                    "kappa": kappa_p,
                    "slope_y": (y_arr - y0) / max(float(d_at_arr["x"]), 1.0),
                    "slope_z": (z_arr - z0) / max(float(d_at_arr["x"]), 1.0),
                    "x_arr": float(d_at_arr["x"]),
                    "y_arr": y_arr,
                    "z_arr": z_arr,
                    "is_hit": bool(hit_info["is_hit"]),
                    "is_lethal": bool(hit_info["is_lethal"]),
                }
            )

    t_end = max((p["t_arr"] for p in pellet_records), default=0.10) * 1.12
    t_grid = np.linspace(0.0, t_end, max(8, int(n_frames)))

    # Pre-build drone body geometry template
    lf = float(drone_spec.get("fuselage_l_m", 0.22))
    wf = float(drone_spec.get("fuselage_w_m", 0.14))
    hf = float(drone_spec.get("fuselage_h_m", 0.08))
    box_body = np.array([
        [-lf / 2, -wf / 2, -hf / 2], [lf / 2, -wf / 2, -hf / 2], [lf / 2, wf / 2, -hf / 2], [-lf / 2, wf / 2, -hf / 2],
        [-lf / 2, -wf / 2, hf / 2], [lf / 2, -wf / 2, hf / 2], [lf / 2, wf / 2, hf / 2], [-lf / 2, wf / 2, hf / 2],
    ])
    box_rot = box_body @ M_rot.T
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    th_rot = np.linspace(0.0, 2.0 * math.pi, 16)
    rotors_rot: list[np.ndarray] = []
    for r in drone_spec.get("rotors", []):
        rx_b = float(r.get("x_m", r.get("z_m", 0.0)))
        ry_b = float(r.get("y_m", 0.0))
        rz_b = float(r.get("z_m_vert", 0.015))
        rr = float(r.get("radius_m", 0.10))
        pts_b = np.column_stack([rx_b + rr * np.cos(th_rot), ry_b + rr * np.sin(th_rot), np.full_like(th_rot, rz_b)])
        rotors_rot.append(pts_b @ M_rot.T)

    crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 3))
    frames: list[dict[str, Any]] = []
    shot_strings: list[float] = []

    for t_val in t_grid:
        t_f = float(t_val)
        d_now = evaluate_evasive_drone_state(
            t_f, target_distance_m, drone_speed_mps, 75.0, maneuver_profile, maneuver_g
        )
        dx_c, dy_c, dz_c = float(d_now["x"]), float(d_now["y"]), float(d_now["z"])

        fx: list[float | None] = []
        fy: list[float | None] = []
        fz: list[float | None] = []
        for e0, e1 in edges:
            fx.extend([round(dx_c + float(box_rot[e0, 0]), 3), round(dx_c + float(box_rot[e1, 0]), 3), None])
            fy.extend([round(dy_c + float(box_rot[e0, 1]), 3), round(dy_c + float(box_rot[e1, 1]), 3), None])
            fz.extend([round(dz_c + float(box_rot[e0, 2]), 3), round(dz_c + float(box_rot[e1, 2]), 3), None])

        rx: list[float | None] = []
        ry: list[float | None] = []
        rz: list[float | None] = []
        for rr_pts in rotors_rot:
            rx.extend([round(dx_c + float(v), 3) for v in rr_pts[:, 0]] + [None])
            ry.extend([round(dy_c + float(v), 3) for v in rr_pts[:, 1]] + [None])
            rz.extend([round(dz_c + float(v), 3) for v in rr_pts[:, 2]] + [None])

        px_fly: list[float] = []
        py_fly: list[float] = []
        pz_fly: list[float] = []
        px_hit: list[float] = []
        py_hit: list[float] = []
        pz_hit: list[float] = []
        cum_hits = 0
        cum_lethal = 0

        for p in pellet_records:
            if t_f < p["t_fire"]:
                px_fly.append(0.0)
                py_fly.append(round(p["y0"], 3))
                pz_fly.append(round(p["z0"], 3))
            elif t_f < p["t_arr"]:
                dt_p = t_f - p["t_fire"]
                x_now = math.log(1.0 + p["kappa"] * p["v0"] * dt_p) / max(p["kappa"], 1e-6)
                y_now = p["y0"] + p["slope_y"] * x_now
                z_now = p["z0"] + p["slope_z"] * x_now
                px_fly.append(round(x_now, 3))
                py_fly.append(round(y_now, 3))
                pz_fly.append(round(z_now, 3))
            else:
                if p["is_hit"]:
                    cum_hits += 1
                    if p["is_lethal"]:
                        cum_lethal += 1
                    px_hit.append(round(p["x_arr"], 3))
                    py_hit.append(round(p["y_arr"], 3))
                    pz_hit.append(round(p["z_arr"], 3))
                else:
                    # Miss continues past target plane
                    dt_p = t_f - p["t_fire"]
                    x_now = min(p["x_arr"] + 12.0, math.log(1.0 + p["kappa"] * p["v0"] * dt_p) / max(p["kappa"], 1e-6))
                    px_fly.append(round(x_now, 3))
                    py_fly.append(round(p["y0"] + p["slope_y"] * x_now, 3))
                    pz_fly.append(round(p["z0"] + p["slope_z"] * x_now, 3))

        if len(px_fly) >= 2:
            shot_strings.append(max(px_fly) - min(px_fly))

        status = "NEUTRALIZED" if cum_lethal >= crit_hits_req else ("DAMAGED" if cum_hits >= 1 else "INGRESSING")
        frames.append(
            {
                "t_ms": round(t_f * 1000.0, 1),
                "drone_pos": [round(dx_c, 3), round(dy_c, 3), round(dz_c, 3)],
                "drone_fuselage_lines": {"x": fx, "y": fy, "z": fz},
                "drone_rotor_lines": {"x": rx, "y": ry, "z": rz},
                "aim_end": [round(lead_x, 3), round(lead_y, 3), round(lead_z, 3)],
                "fly_x": px_fly,
                "fly_y": py_fly,
                "fly_z": pz_fly,
                "hit_x": px_hit,
                "hit_y": py_hit,
                "hit_z": pz_hit,
                "cumulative_hits": cum_hits,
                "cumulative_lethal": cum_lethal,
                "drone_status": status,
            }
        )

    last_f = frames[-1]
    return {
        "summary": {
            "mode": mode,
            "drone_name": drone_spec["name"],
            "ammunition_preset": canon_ammo,
            "firing_mode": firing_mode,
            "ripple_delay_ms": float(ripple_delay_ms),
            "target_distance_m": float(target_distance_m),
            "t_end_ms": round(t_end * 1000.0, 1),
            "shot_string_length_m": round(max(shot_strings, default=0.0), 2),
            "total_pellets_tracked": len(pellet_records),
            "final_hits": int(last_f["cumulative_hits"]),
            "final_lethal_hits": int(last_f["cumulative_lethal"]),
            "drone_neutralized": bool(last_f["drone_status"] == "NEUTRALIZED"),
        },
        "barrels": [{"barrel_id": b["barrel_id"], "y_m": b["y_m"], "z_m": b["z_m"]} for b in barrels],
        "frames": frames,
    }


# -------------------------------------------------------------------------
# UPGRADE 4: SUSTAINED COMBAT LOGISTICS, BARREL HEAT & $/KILL ECONOMICS
# -------------------------------------------------------------------------

CUAS_AMMO_ECONOMICS_CATALOG: dict[str, dict[str, Any]] = {
    "12G_NO4_BUCK_C_UAS": {
        "name": "12G #4 Buckshot (Lead, 27 Pellets)",
        "unit_cost_usd": 1.35,
        "cartridge_mass_g": 48.0,
        "heat_per_shot_kj": 1.85,
        "p_jam_per_shot": 0.0025,
        "pellets_per_shot": 27,
        "pellet_diameter_mm": 6.10,
        "pellet_mass_g": 1.34,
        "v0_mps": 415.0,
        "base_cd": 0.47,
        "shots_per_trigger": 1,
    },
    "12G_NO75_BIRD_TRAP": {
        "name": "12G #7.5 Birdshot (Close-In Swarm)",
        "unit_cost_usd": 0.45,
        "cartridge_mass_g": 42.0,
        "heat_per_shot_kj": 1.45,
        "p_jam_per_shot": 0.0020,
        "pellets_per_shot": 395,
        "pellet_diameter_mm": 2.41,
        "pellet_mass_g": 0.081,
        "v0_mps": 395.0,
        "base_cd": 0.47,
        "shots_per_trigger": 1,
    },
    "12G_T_SHOT_STEEL_C_UAS": {
        "name": "12G Steel T-Shot / BB (Lead-Free)",
        "unit_cost_usd": 1.65,
        "cartridge_mass_g": 46.0,
        "heat_per_shot_kj": 1.90,
        "p_jam_per_shot": 0.0028,
        "pellets_per_shot": 42,
        "pellet_diameter_mm": 5.08,
        "pellet_mass_g": 0.54,
        "v0_mps": 435.0,
        "base_cd": 0.47,
        "shots_per_trigger": 1,
    },
    "12G_TSS_NO2_C_UAS": {
        "name": "12G Tungsten Super-Shot (TSS #2)",
        "unit_cost_usd": 6.80,
        "cartridge_mass_g": 54.0,
        "heat_per_shot_kj": 2.05,
        "p_jam_per_shot": 0.0022,
        "pellets_per_shot": 52,
        "pellet_diameter_mm": 3.81,
        "pellet_mass_g": 0.52,
        "v0_mps": 410.0,
        "base_cd": 0.46,
        "shots_per_trigger": 1,
    },
    "12G_FLECHETTE_SABOT_C_UAS": {
        "name": "12G Saboted Steel Flechette Pack",
        "unit_cost_usd": 8.50,
        "cartridge_mass_g": 45.0,
        "heat_per_shot_kj": 1.95,
        "p_jam_per_shot": 0.0035,
        "pellets_per_shot": 20,
        "pellet_diameter_mm": 2.60,
        "pellet_mass_g": 1.15,
        "v0_mps": 445.0,
        "base_cd": 0.24,
        "shots_per_trigger": 1,
    },
    "CUAS_AIRBURST_CANISTER": {
        "name": "12G / 40mm Programmable Airburst",
        "unit_cost_usd": 42.00,
        "cartridge_mass_g": 92.0,
        "heat_per_shot_kj": 3.40,
        "p_jam_per_shot": 0.0018,
        "pellets_per_shot": 85,
        "pellet_diameter_mm": 3.00,
        "pellet_mass_g": 0.25,
        "v0_mps": 425.0,
        "base_cd": 0.34,
        "shots_per_trigger": 1,
    },
    "762_MINIGUN_BURST": {
        "name": "7.62×51mm Rotary Minigun (20-rd Burst)",
        "unit_cost_usd": 14.00,  # 20 rounds @ $0.70/rd per burst equivalent
        "cartridge_mass_g": 500.0,  # 20 × 25g linked belt
        "heat_per_shot_kj": 9.80,  # rapid burst thermal load across rotary cluster
        "p_jam_per_shot": 0.0012,
        "pellets_per_shot": 20,
        "pellet_diameter_mm": 7.82,
        "pellet_mass_g": 9.50,
        "v0_mps": 840.0,
        "base_cd": 0.28,
        "shots_per_trigger": 20,
    },
}

MAGAZINE_SYSTEM_SPECS: dict[str, dict[str, Any]] = {
    "tube_7rd": {"name": "7-Round Fixed Tubular Magazine", "capacity_per_barrel": 7, "reload_time_s": 9.5, "jam_clear_s": 4.0},
    "box_10rd": {"name": "10-Round Detachable Box Magazine", "capacity_per_barrel": 10, "reload_time_s": 4.2, "jam_clear_s": 3.2},
    "drum_20rd": {"name": "20-Round High-Capacity Drum", "capacity_per_barrel": 20, "reload_time_s": 6.5, "jam_clear_s": 5.5},
    "belt_500rd": {"name": "500-Round Linked Belt Hopper", "capacity_per_barrel": 500, "reload_time_s": 14.0, "jam_clear_s": 6.0},
}


def get_cuas_ammo_economics_catalog() -> dict[str, dict[str, Any]]:
    """Return catalog of C-UAS ammunition cost, mass, heat, and ballistic specs."""
    return CUAS_AMMO_ECONOMICS_CATALOG


def get_magazine_system_specs() -> dict[str, dict[str, Any]]:
    """Return magazine / feed system capacity and reload timing catalog."""
    return MAGAZINE_SYSTEM_SPECS


def simulate_cuas_sustained_logistics_mc(
    n_waves: int = 6,
    drones_per_wave: int = 4,
    wave_interval_s: float = 45.0,
    n_barrels: int = 6,
    ammo_family: str = "12G_NO4_BUCK_C_UAS",
    magazine_type: str = "box_10rd",
    barrel_mass_kg: float = 1.15,
    ambient_temp_c: float = 22.0,
    target_range_m: float = 40.0,
    drone_model: str = "fpv_quad_small",
    aspect_mode: str = "head_on_frontal",
    n_trials: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate multi-wave barrel heating, Newton cooling, reloads, stoppages, and $/kill economics."""
    rng = np.random.default_rng(seed)
    canon_ammo = _AMMO_PRESET_ALIASES.get(str(ammo_family).lower(), str(ammo_family))
    if canon_ammo not in CUAS_AMMO_ECONOMICS_CATALOG:
        canon_ammo = "12G_NO4_BUCK_C_UAS"
    ammo_spec = CUAS_AMMO_ECONOMICS_CATALOG[canon_ammo]
    mag_spec = MAGAZINE_SYSTEM_SPECS.get(magazine_type, MAGAZINE_SYSTEM_SPECS["box_10rd"])

    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode=aspect_mode)

    c_steel_kj_per_kg_k = 0.49
    tau_cool_s = 180.0
    cookoff_limit_c = 220.0
    barrels_per_burst = min(2 if n_barrels >= 2 else 1, max(1, n_barrels))
    cap_per_bbl = int(mag_spec["capacity_per_barrel"])

    base_pk = _estimate_burst_pk_analytics(
        slant_range_m=target_range_m,
        barrels_fired=barrels_per_burst,
        pellets_per_shot=int(ammo_spec["pellets_per_shot"]),
        pellet_diameter_mm=float(ammo_spec["pellet_diameter_mm"]),
        pellet_mass_g=float(ammo_spec["pellet_mass_g"]),
        v0_mps=float(ammo_spec["v0_mps"]),
        choke_name="Modified",
        drone_spec=drone_spec,
        base_cd=float(ammo_spec["base_cd"]),
        aspect_mode=aspect_mode,
        aspect_geom=aspect_geom,
    )
    # Rotary / high-density bursts enjoy an effective cluster density credit
    if canon_ammo == "CUAS_AIRBURST_CANISTER":
        base_pk = max(base_pk, min(0.96, 0.72 * math.exp(-0.008 * max(target_range_m - 45.0, 0.0))))
    elif canon_ammo == "762_MINIGUN_BURST":
        base_pk = max(base_pk, min(0.92, 0.78 * math.exp(-0.003 * max(target_range_m - 60.0, 0.0))))

    trial_peaks_c: list[float] = []
    trial_rounds: list[int] = []
    trial_kills: list[int] = []
    trial_stoppages: list[int] = []
    trial_reload_s: list[float] = []
    trial_wave_clear: list[float] = []

    sample_timeline: list[dict[str, Any]] = []
    wave_accum = [
        {"kills": 0, "rounds": 0, "temp_c_sum": 0.0, "reloads": 0, "jams": 0}
        for _ in range(n_waves)
    ]

    for t_idx in range(n_trials):
        temp_c = float(ambient_temp_c)
        mag_rem = [cap_per_bbl for _ in range(n_barrels)]
        b_cursor = 0
        tot_rds = 0
        tot_kills = 0
        tot_jams = 0
        tot_rel_s = 0.0
        peak_c = temp_c
        waves_cleared = 0
        t_clock_s = 0.0

        if t_idx == 0:
            sample_timeline.append(
                {"t_s": 0.0, "barrel_temp_c": round(temp_c, 1), "thermal_walk_mrad": 0.0, "cumulative_rounds": 0}
            )

        for w_idx in range(n_waves):
            w_rds = 0
            w_kills = 0
            w_reloads = 0
            w_jams = 0

            for _ in range(drones_per_wave):
                # Allow up to 3 bursts per drone before breach
                for _attempt in range(3):
                    need_reload = False
                    for k_b in range(barrels_per_burst):
                        bi = (b_cursor + k_b) % n_barrels
                        if mag_rem[bi] <= 0:
                            need_reload = True
                    if need_reload:
                        tot_rel_s += float(mag_spec["reload_time_s"])
                        w_reloads += 1
                        mag_rem = [cap_per_bbl for _ in range(n_barrels)]

                    for k_b in range(barrels_per_burst):
                        bi = (b_cursor + k_b) % n_barrels
                        mag_rem[bi] -= 1
                        w_rds += 1
                        tot_rds += 1
                        # Thermal rise distributed per fired barrel
                        temp_c += float(ammo_spec["heat_per_shot_kj"]) / max(barrel_mass_kg * c_steel_kj_per_kg_k * n_barrels, 0.25)
                        peak_c = max(peak_c, temp_c)

                        if float(rng.random()) < float(ammo_spec["p_jam_per_shot"]):
                            tot_jams += 1
                            w_jams += 1
                            tot_rel_s += float(mag_spec["jam_clear_s"])

                    b_cursor = (b_cursor + barrels_per_burst) % n_barrels

                    # Thermal dispersion walk penalty degrades hit probability above 120 degC
                    excess_c = max(0.0, temp_c - 120.0)
                    walk_penalty = math.exp(-0.0028 * excess_c)
                    eff_pk = max(0.02, min(0.99, base_pk * walk_penalty))
                    if float(rng.random()) < eff_pk:
                        w_kills += 1
                        tot_kills += 1
                        break

            if w_kills == drones_per_wave:
                waves_cleared += 1

            t_clock_s += 8.0  # active wave fire duration
            if t_idx == 0:
                walk_mrad = 0.045 * max(0.0, temp_c - 100.0) ** 1.15
                sample_timeline.append(
                    {
                        "t_s": round(t_clock_s, 1),
                        "barrel_temp_c": round(temp_c, 1),
                        "thermal_walk_mrad": round(walk_mrad, 2),
                        "cumulative_rounds": tot_rds,
                    }
                )

            wave_accum[w_idx]["kills"] += w_kills
            wave_accum[w_idx]["rounds"] += w_rds
            wave_accum[w_idx]["temp_c_sum"] += temp_c
            wave_accum[w_idx]["reloads"] += w_reloads
            wave_accum[w_idx]["jams"] += w_jams

            # Cool down between waves (skip after final wave)
            if w_idx < n_waves - 1:
                cool_dt = max(wave_interval_s - 8.0, 1.0)
                t_clock_s += cool_dt
                temp_c = float(ambient_temp_c) + (temp_c - float(ambient_temp_c)) * math.exp(-cool_dt / tau_cool_s)
                if t_idx == 0:
                    walk_mrad = 0.045 * max(0.0, temp_c - 100.0) ** 1.15
                    sample_timeline.append(
                        {
                            "t_s": round(t_clock_s, 1),
                            "barrel_temp_c": round(temp_c, 1),
                            "thermal_walk_mrad": round(walk_mrad, 2),
                            "cumulative_rounds": tot_rds,
                        }
                    )

        trial_peaks_c.append(peak_c)
        trial_rounds.append(tot_rds)
        trial_kills.append(tot_kills)
        trial_stoppages.append(tot_jams)
        trial_reload_s.append(tot_rel_s)
        trial_wave_clear.append((waves_cleared / max(n_waves, 1)) * 100.0)

    mean_rds = float(np.mean(trial_rounds))
    mean_kills = max(float(np.mean(trial_kills)), 0.1)
    mean_cost_total = mean_rds * float(ammo_spec["unit_cost_usd"])
    cost_per_kill = mean_cost_total / mean_kills
    mean_peak_c = float(np.mean(trial_peaks_c))

    wave_records = [
        {
            "wave": i + 1,
            "mean_drones_killed": round(wave_accum[i]["kills"] / max(n_trials, 1), 2),
            "mean_rounds_fired": round(wave_accum[i]["rounds"] / max(n_trials, 1), 1),
            "end_barrel_temp_c": round(wave_accum[i]["temp_c_sum"] / max(n_trials, 1), 1),
            "mean_reloads": round(wave_accum[i]["reloads"] / max(n_trials, 1), 2),
            "mean_stoppages": round(wave_accum[i]["jams"] / max(n_trials, 1), 2),
        }
        for i in range(n_waves)
    ]

    # Cross-Payload 7-Family Economics Table at target_range_m
    economics_comparison: list[dict[str, Any]] = []
    for k_am, sp_am in CUAS_AMMO_ECONOMICS_CATALOG.items():
        pk_est = _estimate_burst_pk_analytics(
            slant_range_m=target_range_m,
            barrels_fired=barrels_per_burst,
            pellets_per_shot=int(sp_am["pellets_per_shot"]),
            pellet_diameter_mm=float(sp_am["pellet_diameter_mm"]),
            pellet_mass_g=float(sp_am["pellet_mass_g"]),
            v0_mps=float(sp_am["v0_mps"]),
            choke_name="Modified",
            drone_spec=drone_spec,
            base_cd=float(sp_am["base_cd"]),
            aspect_mode=aspect_mode,
            aspect_geom=aspect_geom,
        )
        if k_am == "CUAS_AIRBURST_CANISTER":
            pk_est = max(pk_est, min(0.96, 0.72 * math.exp(-0.008 * max(target_range_m - 45.0, 0.0))))
        elif k_am == "762_MINIGUN_BURST":
            pk_est = max(pk_est, min(0.92, 0.78 * math.exp(-0.003 * max(target_range_m - 60.0, 0.0))))
        pk_est = max(pk_est, 0.03)
        bursts_per_kill = 1.0 / pk_est
        rds_per_kill = bursts_per_kill * barrels_per_burst
        cpk = rds_per_kill * float(sp_am["unit_cost_usd"])
        mass_10_kg = (rds_per_kill * 10.0 * float(sp_am["cartridge_mass_g"])) / 1000.0
        economics_comparison.append(
            {
                "ammo_key": k_am,
                "ammo_family": sp_am["name"],
                "unit_cost_usd": round(float(sp_am["unit_cost_usd"]), 2),
                "single_salvo_pk_pct": round(pk_est * 100.0, 1),
                "rounds_per_kill": round(rds_per_kill, 1),
                "cost_per_kill_usd": round(cpk, 2),
                "ammo_mass_per_10_kills_kg": round(mass_10_kg, 2),
                "heat_per_kill_kj": round(rds_per_kill * float(sp_am["heat_per_shot_kj"]), 1),
            }
        )

    cookoff_status = (
        "CRITICAL COOK-OFF RISK"
        if mean_peak_c >= cookoff_limit_c
        else ("HOT - THERMAL WALK ACTIVE" if mean_peak_c >= 120.0 else "NOMINAL OPERATING TEMP")
    )

    return {
        "summary": {
            "ammo_name": ammo_spec["name"],
            "magazine_name": mag_spec["name"],
            "n_waves": n_waves,
            "drones_per_wave": drones_per_wave,
            "target_range_m": float(target_range_m),
            "peak_barrel_temp_c": round(mean_peak_c, 1),
            "cookoff_limit_c": cookoff_limit_c,
            "cookoff_risk": cookoff_status,
            "total_rounds_fired": round(mean_rds, 1),
            "total_drones_killed": round(float(np.mean(trial_kills)), 1),
            "mean_cost_per_kill_usd": round(cost_per_kill, 2),
            "total_reload_time_s": round(float(np.mean(trial_reload_s)), 1),
            "mean_stoppages": round(float(np.mean(trial_stoppages)), 2),
            "wave_survival_pct": round(float(np.mean(trial_wave_clear)), 1),
        },
        "thermal_timeline": {
            "t_s": [r["t_s"] for r in sample_timeline],
            "barrel_temp_c": [r["barrel_temp_c"] for r in sample_timeline],
            "thermal_walk_mrad": [r["thermal_walk_mrad"] for r in sample_timeline],
            "cumulative_rounds": [r["cumulative_rounds"] for r in sample_timeline],
        },
        "wave_records": wave_records,
        "economics_comparison": economics_comparison,
    }


# -------------------------------------------------------------------------
# UPGRADE 5: SCENARIO CONFIG SAVE / LOAD & STANDALONE HTML REPORT EXPORT
# -------------------------------------------------------------------------

_VALID_LAYOUTS = {"circular", "grid", "concentric", "honeycomb", "fan_horizontal", "fan_vertical", "v_shape", "custom"}
_VALID_POINTING = {"convergent", "parallel", "divergent_curtain", "staggered_tiling"}
_VALID_FIRING = {"simultaneous_volley", "sequential_ripple", "balanced_pair_ripple"}


def serialize_cuas_scenario_config(config: dict[str, Any]) -> str:
    """Serialize a C-UAS turret scenario configuration into versioned JSON."""
    payload = {
        "schema_version": "cuas_scenario_v1",
        "config": config,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def validate_cuas_scenario_config(raw_json: str) -> tuple[bool, dict[str, Any], str]:
    """Validate an uploaded C-UAS JSON scenario configuration and sanitize bounds."""
    if not isinstance(raw_json, str) or not raw_json.strip():
        return False, {}, "Empty JSON payload."
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return False, {}, f"Invalid JSON syntax: {exc}"

    if not isinstance(parsed, dict):
        return False, {}, "Root JSON object must be a dictionary."

    version = parsed.get("schema_version")
    if version != "cuas_scenario_v1":
        return False, {}, f"Unsupported or missing schema_version: {version!r}."

    cfg = parsed.get("config")
    if not isinstance(cfg, dict):
        return False, {}, "Missing or invalid 'config' dictionary."

    cleaned: dict[str, Any] = {}
    if "layout_type" in cfg:
        lt = str(cfg["layout_type"])
        if lt not in _VALID_LAYOUTS:
            return False, {}, f"Unknown layout_type: {lt!r}."
        cleaned["layout_type"] = lt
    if "pointing_mode" in cfg:
        pm = str(cfg["pointing_mode"])
        if pm not in _VALID_POINTING:
            return False, {}, f"Unknown pointing_mode: {pm!r}."
        cleaned["pointing_mode"] = pm
    if "firing_mode" in cfg:
        fm = str(cfg["firing_mode"])
        if fm not in _VALID_FIRING:
            return False, {}, f"Unknown firing_mode: {fm!r}."
        cleaned["firing_mode"] = fm
    if "n_barrels" in cfg:
        try:
            nb = int(cfg["n_barrels"])
        except (TypeError, ValueError):
            return False, {}, "n_barrels must be an integer."
        if not 1 <= nb <= 16:
            return False, {}, "n_barrels must be between 1 and 16."
        cleaned["n_barrels"] = nb
    if "target_distance_m" in cfg:
        try:
            td = float(cfg["target_distance_m"])
        except (TypeError, ValueError):
            return False, {}, "target_distance_m must be numeric."
        if not 5.0 <= td <= 500.0:
            return False, {}, "target_distance_m must be between 5 and 500 m."
        cleaned["target_distance_m"] = td

    for num_key, lo, hi in [
        ("mount_radius_m", 0.05, 1.0),
        ("barrel_length_in", 10.0, 36.0),
        ("splay_angle_mrad", 0.0, 30.0),
        ("ripple_delay_ms", 0.0, 100.0),
        ("drone_speed_mps", 0.0, 80.0),
        ("drone_heading_deg", 0.0, 180.0),
        ("drone_pitch_deg", -90.0, 90.0),
        ("drone_yaw_deg", -180.0, 180.0),
        ("drone_roll_deg", -90.0, 90.0),
    ]:
        if num_key in cfg:
            try:
                val = float(cfg[num_key])
            except (TypeError, ValueError):
                return False, {}, f"{num_key} must be numeric."
            cleaned[num_key] = max(lo, min(hi, val))

    drone_specs = get_drone_target_specs()
    aspect_presets = get_drone_aspect_presets()
    choke_defs = get_choke_definitions()
    sg_presets = get_shotgun_presets()

    if "drone_model" in cfg and str(cfg["drone_model"]) in drone_specs:
        cleaned["drone_model"] = str(cfg["drone_model"])
    if "aspect_mode" in cfg and str(cfg["aspect_mode"]) in aspect_presets:
        cleaned["aspect_mode"] = str(cfg["aspect_mode"])
    if "choke" in cfg and str(cfg["choke"]) in choke_defs:
        cleaned["choke"] = str(cfg["choke"])
    if "ammunition_preset" in cfg and str(cfg["ammunition_preset"]) in sg_presets:
        cleaned["ammunition_preset"] = str(cfg["ammunition_preset"])
    if "custom_coords_text" in cfg and isinstance(cfg["custom_coords_text"], str):
        cleaned["custom_coords_text"] = cfg["custom_coords_text"]

    return True, cleaned, f"Validated {len(cleaned)} C-UAS configuration fields."


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a sanitized HTML table from a list of dicts and (key, label) pairs."""
    if not rows:
        return "<p><em>No rows generated.</em></p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def generate_cuas_trade_study_html(
    scenario_name: str,
    config: dict[str, Any],
    array_summary: dict[str, Any] | None = None,
    aspect_table: list[dict[str, Any]] | None = None,
    kill_chain_summary: dict[str, Any] | None = None,
    logistics_summary: dict[str, Any] | None = None,
    economics_table: list[dict[str, Any]] | None = None,
    sdz_summary: dict[str, Any] | None = None,
) -> str:
    """Generate a self-contained HTML engineering trade-study report ready for browser print-to-PDF."""
    safe_name = html.escape(scenario_name or "C-UAS Multi-Barrel Trade Study")
    cfg_rows = [{"param": k, "value": v} for k, v in sorted(config.items())]

    kpis: list[tuple[str, str]] = []
    if array_summary:
        kpis.append(("Intercept P_kill", f"{array_summary.get('kill_probability_pct', 0.0)}%"))
        kpis.append(("Presented Cross-Section", f"{array_summary.get('presented_area_cm2', 0.0)} cm²"))
    if kill_chain_summary:
        kpis.append(("Kill-Chain Cum. P_kill", f"{kill_chain_summary.get('cumulative_pk_pct', 0.0)}%"))
        kpis.append(("First-Shot Range", f"{kill_chain_summary.get('first_shot_range_m', 0.0)} m"))
    if logistics_summary:
        kpis.append(("Peak Barrel Temp", f"{logistics_summary.get('peak_barrel_temp_c', 0.0)} °C"))
        kpis.append(("Cost per Kill", f"${logistics_summary.get('mean_cost_per_kill_usd', 0.0)}"))
    if sdz_summary:
        kpis.append(("99% Fallout SDZ", f"{sdz_summary.get('max_fallout_range_99_m', 0.0)} m"))

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-lbl">{html.escape(lbl)}</div><div class="kpi-val">{html.escape(val)}</div></div>'
        for lbl, val in kpis
    )

    sections = [
        f"<h2>1. System Configuration Parameters</h2>{_html_table(cfg_rows, [('param', 'Parameter'), ('value', 'Value')])}"
    ]
    if aspect_table:
        sections.append(
            "<h2>2. 3D Target Drone Cross-Section Vulnerability Breakdown</h2>"
            + _html_table(
                aspect_table,
                [
                    ("name", "Presentation Aspect"),
                    ("presented_area_cm2", "Presented Area (cm²)"),
                    ("cross_section_ratio_pct", "Planform %"),
                    ("kill_probability_pct", "P_kill (%)"),
                    ("mean_hits", "Mean Hits"),
                ],
            )
        )
    if economics_table:
        sections.append(
            "<h2>3. Ammunition Family Cost-per-Kill & Mass Economics</h2>"
            + _html_table(
                economics_table,
                [
                    ("ammo_family", "Ammunition Family"),
                    ("unit_cost_usd", "Unit Cost ($)"),
                    ("single_salvo_pk_pct", "Single-Salvo P_kill (%)"),
                    ("rounds_per_kill", "Rounds / Kill"),
                    ("cost_per_kill_usd", "Cost / Kill ($)"),
                    ("ammo_mass_per_10_kills_kg", "Mass / 10 Kills (kg)"),
                ],
            )
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{safe_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 32px auto; max-width: 980px; color: #0f172a; line-height: 1.5; }}
  h1 {{ color: #1e3a8a; border-bottom: 2px solid #cbd5e1; padding-bottom: 8px; }}
  h2 {{ color: #1e293b; margin-top: 28px; }}
  .notice {{ background: #fef3c7; border-left: 5px solid #f59e0b; padding: 10px 14px; font-size: 0.9rem; margin: 16px 0; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin: 18px 0; }}
  .kpi {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px 12px; background: #f8fafc; }}
  .kpi-lbl {{ font-size: 0.8rem; color: #475569; }}
  .kpi-val {{ font-size: 1.35rem; font-weight: 700; color: #0f172a; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 7px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  @media print {{ body {{ margin: 12mm; }} }}
</style>
</head>
<body>
  <h1>🛸 C-UAS Engineering Trade-Study Report: {safe_name}</h1>
  <div class="notice">
    <strong>GOVERNANCE &amp; USE LIMITATION NOTICE:</strong> Engineering estimate based strictly on unclassified literature models (IATG, STANAG, Lowry-Journée, McCoy); not a substitute for range-certified danger-area templates.
  </div>
  <div class="kpi-grid">{kpi_html}</div>
  {''.join(sections)}
</body>
</html>
"""
