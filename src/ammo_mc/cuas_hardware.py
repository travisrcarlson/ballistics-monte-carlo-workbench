"""Hardware Prototyping Suite: SWaP-C Estimator, On-the-Move Stabilization & Muzzle-Plate CAD.

Converts logical multi-barrel C-UAS turret designs into:
  1. Sub-assembly Size, Weight, Power & Cost (SWaP-C) budgets + rotational inertia & mount compatibility,
  2. On-the-move vehicle/maritime stabilization & adverse-weather effective-range degradation curves,
  3. True-scale 2D parametric muzzle-clamp engineering drawings in AutoCAD ASCII `.dxf` and SVG.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.cuas_defense import (
    CUAS_SENSOR_PRESETS,
    _estimate_burst_pk_analytics,
    simulate_turret_recoil_and_gimbal_loads,
)
from ammo_mc.cuas_mission import MAGAZINE_SYSTEM_SPECS, _resolve_shotgun_ammo
from ammo_mc.small_arms import (
    compute_drone_projected_geometry,
    generate_multishotgun_array_layout,
    get_drone_target_specs,
)

MOUNT_CLASS_SPECS: dict[str, dict[str, Any]] = {
    "portable_tripod": {
        "name": "Heavy Tripod / Dismounted Pedestal",
        "max_mass_kg": 35.0,
        "max_recoil_kn": 2.2,
        "max_power_w": 250.0,
        "base_cost_usd": 4500.0,
    },
    "light_ugv": {
        "name": "Light UGV / Robotic Mule (e.g. 400 kg Class)",
        "max_mass_kg": 85.0,
        "max_recoil_kn": 4.0,
        "max_power_w": 750.0,
        "base_cost_usd": 12000.0,
    },
    "tactical_4x4": {
        "name": "Tactical 4x4 Roof Ring Mount (RWS Class)",
        "max_mass_kg": 220.0,
        "max_recoil_kn": 8.0,
        "max_power_w": 1800.0,
        "base_cost_usd": 24000.0,
    },
    "armored_fixed": {
        "name": "Armored APC / Fixed Shelter Pedestal",
        "max_mass_kg": 650.0,
        "max_recoil_kn": 16.0,
        "max_power_w": 4500.0,
        "base_cost_usd": 38000.0,
    },
}

CARRIER_OTM_PROFILES: dict[str, dict[str, Any]] = {
    "static_fixed_mount": {
        "name": "Static Fixed Pedestal / Tripod",
        "base_disturbance_mrad": 0.25,
    },
    "ugv_gravel_15kph": {
        "name": "Light UGV on Gravel Track (15 km/h)",
        "base_disturbance_mrad": 11.5,
    },
    "tactical_4x4_offroad_40kph": {
        "name": "Tactical 4x4 Off-Road (40 km/h)",
        "base_disturbance_mrad": 24.0,
    },
    "maritime_sea_state_3": {
        "name": "Patrol Craft Pitch/Roll (Sea State 3)",
        "base_disturbance_mrad": 32.0,
    },
}

GYRO_STABILIZATION_GRADES: dict[str, dict[str, Any]] = {
    "unstabilized": {
        "name": "Unstabilized Hard-Mount (0 dB Attenuation)",
        "attenuation_db": 0.0,
    },
    "standard_mems_gyro": {
        "name": "2-Axis Commercial MEMS Gyro (18 dB Attenuation)",
        "attenuation_db": 18.0,
    },
    "tactical_fog_gyro": {
        "name": "Tactical Closed-Loop FOG Gyro (32 dB Attenuation)",
        "attenuation_db": 32.0,
    },
}


def estimate_turret_swap_c(
    n_barrels: int = 6,
    barrel_length_in: float = 24.0,
    layout_type: str = "circular",
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    magazine_type: str = "box_10rd",
    slew_rate_deg_s: float = 160.0,
    slew_accel_deg_s2: float = 640.0,
    sensor_preset: str = "Sensor_Fused_Radar_EOIR",
    firing_mode: str = "balanced_pair_ripple",
    ripple_delay_ms: float = 10.0,
    hybrid_mode: bool = False,
) -> dict[str, Any]:
    """Estimate multi-barrel C-UAS turret mass, inertia, electrical power, recoil & BOM cost."""
    n_b = max(1, int(n_barrels))
    b_len = max(14.0, min(32.0, float(barrel_length_in)))
    b_len_m = b_len * 0.0254
    mag_spec = MAGAZINE_SYSTEM_SPECS.get(magazine_type, MAGAZINE_SYSTEM_SPECS["box_10rd"])
    cap_per_bbl = int(mag_spec["capacity_per_barrel"])

    barrels = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_b,
        mount_radius_m=float(mount_radius_m),
        spacing_y_m=float(spacing_y_m),
        spacing_z_m=float(spacing_z_m),
        default_barrel_length_in=b_len,
        default_ammunition_preset=ammunition_preset,
    )

    # Mass breakdown (kg) calibrated to 12-ga heavy-wall ordnance barrels + anodized 7075-T6 clamp + servo yoke
    per_barrel_kg = 1.05 + 0.042 * (b_len - 18.0)
    barrels_kg = n_b * per_barrel_kg
    receivers_breech_kg = n_b * 1.85
    clamp_cradle_kg = 4.2 + 0.95 * n_b + (4.5 * float(mount_radius_m))
    loaded_ammo_kg = n_b * cap_per_bbl * 0.052 + n_b * 0.42
    sensor_mass_map = {
        "EO_IR_Dual_Thermal_Tracker": 3.8,
        "Ku_Band_Micro_Doppler_Radar": 6.2,
        "Sensor_Fused_Radar_EOIR": 8.4,
    }
    sensors_fcs_kg = float(sensor_mass_map.get(sensor_preset, 7.5))

    # Yoke & servo mass scales with weapon assembly + recoil buffering
    payload_elev_kg = barrels_kg + receivers_breech_kg + clamp_cradle_kg + loaded_ammo_kg
    gimbal_drives_kg = 8.5 + 0.24 * payload_elev_kg
    total_mass_kg = payload_elev_kg + sensors_fcs_kg + gimbal_drives_kg

    # Moments of inertia around elevation trunnion and azimuth pedestal axis (kg*m^2)
    r_spread_sq = float(np.mean([(float(b["y_m"]) ** 2 + float(b["z_m"]) ** 2) for b in barrels])) if barrels else 0.04
    inertia_el = payload_elev_kg * ((b_len_m ** 2) / 10.5 + 0.015) + barrels_kg * r_spread_sq
    inertia_az = inertia_el + gimbal_drives_kg * 0.045 + sensors_fcs_kg * 0.035

    alpha_rad_s2 = math.radians(max(60.0, float(slew_accel_deg_s2)))
    omega_rad_s = math.radians(max(30.0, float(slew_rate_deg_s)))
    # Friction/unbalance + angular acceleration + drive margin
    peak_torque_nm = max(inertia_az, inertia_el) * alpha_rad_s2 * 1.45 + 6.5
    mech_power_w = peak_torque_nm * omega_rad_s
    sensor_power_w = {"EO_IR_Dual_Thermal_Tracker": 65.0, "Ku_Band_Micro_Doppler_Radar": 145.0, "Sensor_Fused_Radar_EOIR": 195.0}.get(sensor_preset, 150.0)
    peak_power_w = (mech_power_w / 0.78) + sensor_power_w + 35.0
    standby_power_w = sensor_power_w + 35.0
    bus_current_24v = peak_power_w / 24.0

    rec_res = simulate_turret_recoil_and_gimbal_loads(
        barrels=barrels,
        firing_mode=firing_mode,
        ripple_delay_ms=float(ripple_delay_ms),
        default_ammunition_preset=ammunition_preset,
        mount_mass_kg=max(20.0, total_mass_kg * 0.75),
        mount_stiffness_n_per_m=120000.0,
        mount_load_limit_kn=18.0,
        gimbal_trunnion_offset_z_m=0.18,
    )
    peak_recoil_kn = float(rec_res["summary"]["peak_transmitted_mount_force_kn"])

    # Approximate hardware recurring BOM cost breakdown (USD, notional engineering class estimate)
    sensor_cost_usd = {"EO_IR_Dual_Thermal_Tracker": 14500.0, "Ku_Band_Micro_Doppler_Radar": 22000.0, "Sensor_Fused_Radar_EOIR": 34000.0}.get(sensor_preset, 26000.0)
    ordnance_cost_usd = n_b * 1850.0 + (900.0 if hybrid_mode else 0.0) + n_b * cap_per_bbl * 12.0
    gimbal_servo_cost_usd = 7500.0 + 115.0 * peak_torque_nm + (2800.0 if slew_rate_deg_s > 170.0 else 0.0)
    compute_fcs_cost_usd = 4800.0
    unit_bom_usd = sensor_cost_usd + ordnance_cost_usd + gimbal_servo_cost_usd + compute_fcs_cost_usd

    compat_rows: list[dict[str, Any]] = []
    rec_class = "Unclassified / Exceeds Armored Pedestal"
    found_first = False
    for mc_key, mc_spec in MOUNT_CLASS_SPECS.items():
        ok_mass = total_mass_kg <= mc_spec["max_mass_kg"]
        ok_rec = peak_recoil_kn <= mc_spec["max_recoil_kn"]
        ok_pwr = peak_power_w <= mc_spec["max_power_w"]
        feasible = bool(ok_mass and ok_rec and ok_pwr)
        if feasible and not found_first:
            rec_class = mc_spec["name"]
            found_first = True
        compat_rows.append(
            {
                "mount_class_id": mc_key,
                "mount_class": mc_spec["name"],
                "feasible": feasible,
                "mass_ok": ok_mass,
                "recoil_ok": ok_rec,
                "power_ok": ok_pwr,
                "mass_utilization_pct": round(100.0 * total_mass_kg / mc_spec["max_mass_kg"], 1),
                "recoil_utilization_pct": round(100.0 * peak_recoil_kn / mc_spec["max_recoil_kn"], 1),
                "power_utilization_pct": round(100.0 * peak_power_w / mc_spec["max_power_w"], 1),
            }
        )

    return {
        "summary": {
            "n_barrels": n_b,
            "barrel_length_in": round(b_len, 1),
            "layout_type": layout_type,
            "total_mass_kg": round(total_mass_kg, 2),
            "barrels_kg": round(barrels_kg, 2),
            "receivers_breech_kg": round(receivers_breech_kg, 2),
            "clamp_cradle_kg": round(clamp_cradle_kg, 2),
            "loaded_ammo_kg": round(loaded_ammo_kg, 2),
            "gimbal_drives_kg": round(gimbal_drives_kg, 2),
            "sensors_fcs_kg": round(sensors_fcs_kg, 2),
            "inertia_az_kg_m2": round(inertia_az, 3),
            "inertia_el_kg_m2": round(inertia_el, 3),
            "peak_torque_nm": round(peak_torque_nm, 1),
            "peak_power_w": round(peak_power_w, 1),
            "standby_power_w": round(standby_power_w, 1),
            "bus_current_24v_a": round(bus_current_24v, 1),
            "peak_recoil_force_kn": round(peak_recoil_kn, 2),
            "unit_bom_usd": round(unit_bom_usd, 0),
            "recommended_mount_class": rec_class,
        },
        "mass_breakdown": {
            "Barrels": round(barrels_kg, 2),
            "Receivers & Breech": round(receivers_breech_kg, 2),
            "Cradle & Clamp Plate": round(clamp_cradle_kg, 2),
            "Loaded Ammunition": round(loaded_ammo_kg, 2),
            "Gimbal & Servo Drives": round(gimbal_drives_kg, 2),
            "Radar/EO-IR Sensors": round(sensors_fcs_kg, 2),
        },
        "cost_breakdown_usd": {
            "Ordnance & Receivers": round(ordnance_cost_usd, 0),
            "Sensors & Trackers": round(sensor_cost_usd, 0),
            "Gimbal Drives": round(gimbal_servo_cost_usd, 0),
            "FCS Compute & Power": round(compute_fcs_cost_usd, 0),
        },
        "compatibility_table": compat_rows,
    }


def evaluate_otm_stabilization_degradation(
    turret_config: dict[str, Any] | None = None,
    carrier_profile: str = "tactical_4x4_offroad_40kph",
    gyro_grade: str = "standard_mems_gyro",
    weather: str = "clear_day",
    ambient_temp_c: float = 21.1,
    crosswind_mps: float = 5.0,
    drone_model: str = "fpv_quad_small",
    ranges_m: tuple[float, ...] | list[float] = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0),
) -> dict[str, Any]:
    """Quantify chassis disturbance, gyro stabilization attenuation, and weather impact on P_kill(R)."""
    cfg = dict(turret_config or {})
    n_bbl = int(cfg.get("n_barrels", 6))
    bpb = int(cfg.get("barrels_per_burst", min(3, n_bbl)))
    ammo_key = str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))
    choke_name = str(cfg.get("choke", "Modified"))
    _, ammo = _resolve_shotgun_ammo(ammo_key)
    v0_temp = max(200.0, float(ammo["v0_mps"]) + 0.72 * (float(ambient_temp_c) - 21.1))

    drone_specs = get_drone_target_specs()
    d_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")

    carrier = CARRIER_OTM_PROFILES.get(carrier_profile, CARRIER_OTM_PROFILES["tactical_4x4_offroad_40kph"])
    base_dist = float(carrier["base_disturbance_mrad"])

    wx_jitter = {
        "clear_day": 0.0,
        "overcast_dusk": 0.4,
        "night_clear": 0.8,
        "heavy_fog_rain": 2.1,
    }.get(str(weather), 0.3)

    rng_list = [float(r) for r in ranges_m]
    curves: dict[str, list[dict[str, float]]] = {}
    r80_by_mode: dict[str, float] = {}

    mode_defs = [
        ("static_ideal", "Static Baseline", 0.25, 0.0),
        ("tactical_fog_gyro", "Tactical FOG Gyro (32 dB)", base_dist, 32.0),
        ("standard_mems_gyro", "Standard MEMS Gyro (18 dB)", base_dist, 18.0),
        ("unstabilized", "Unstabilized Chassis", base_dist, 0.0),
    ]

    for m_key, m_lbl, dist_v, atten_db in mode_defs:
        atten_fac = 10.0 ** (-atten_db / 20.0)
        res_los_mrad = math.hypot(dist_v * atten_fac, wx_jitter)
        rows_m: list[dict[str, float]] = []
        for r_m in rng_list:
            pk_nom = _estimate_burst_pk_analytics(
                slant_range_m=r_m,
                barrels_fired=bpb,
                pellets_per_shot=int(ammo["pellets_per_shot"]),
                pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                pellet_mass_g=float(ammo["pellet_mass_g"]),
                v0_mps=v0_temp,
                choke_name=choke_name,
                drone_spec=d_spec,
                base_cd=float(ammo.get("effective_cd", 0.47)),
                aspect_mode="head_on_frontal",
                aspect_geom=aspect_geom,
            )
            # RMS angular aim jitter + crosswind dispersion dilution
            wind_err_m = float(crosswind_mps) * 0.011 * r_m * (r_m / 40.0)
            los_err_m = (res_los_mrad / 1000.0) * r_m
            sig_est = max(0.18, 0.014 * r_m * ((r_m / 36.58) ** 0.15))
            # Expected Rayleigh attenuation of E[exp(-e^2 / (2 sigma^2))] ~ 1 / (1 + sigma_los^2 / sigma^2)
            jitter_var = los_err_m ** 2 + 0.35 * (wind_err_m ** 2)
            atten_pk = 1.0 / (1.0 + jitter_var / (2.0 * sig_est * sig_est))
            pk_eff = float(np.clip(pk_nom * atten_pk * 100.0, 0.0, 99.8))
            rows_m.append(
                {
                    "range_m": r_m,
                    "pk_pct": round(pk_eff, 1),
                    "residual_los_mrad": round(res_los_mrad, 2),
                }
            )
        curves[m_key] = rows_m
        # Interpolate R80
        pks = [r["pk_pct"] for r in rows_m]
        r80_val = 0.0
        for i, p_v in enumerate(pks):
            if p_v >= 80.0:
                r80_val = rng_list[i]
            elif i > 0 and pks[i - 1] >= 80.0 and p_v < 80.0:
                f = (pks[i - 1] - 80.0) / max(pks[i - 1] - p_v, 1e-6)
                r80_val = rng_list[i - 1] + f * (rng_list[i] - rng_list[i - 1])
                break
        r80_by_mode[m_key] = round(r80_val, 1)

    sel_gyro = gyro_grade if gyro_grade in r80_by_mode else "standard_mems_gyro"
    sel_atten_db = GYRO_STABILIZATION_GRADES.get(sel_gyro, GYRO_STABILIZATION_GRADES["standard_mems_gyro"])["attenuation_db"]
    res_sel_mrad = round(math.hypot(base_dist * (10.0 ** (-sel_atten_db / 20.0)), wx_jitter), 2)
    r80_stat = r80_by_mode["static_ideal"]
    r80_act = r80_by_mode[sel_gyro]
    loss_pct = round(100.0 * max(0.0, r80_stat - r80_act) / max(r80_stat, 1e-6), 1) if r80_stat > 0.0 else 0.0

    return {
        "carrier_profile": carrier["name"],
        "gyro_grade": GYRO_STABILIZATION_GRADES.get(sel_gyro, GYRO_STABILIZATION_GRADES["standard_mems_gyro"])["name"],
        "selected_gyro_key": sel_gyro,
        "weather": weather,
        "residual_los_jitter_mrad": res_sel_mrad,
        "r80_static_m": r80_stat,
        "r80_active_m": r80_act,
        "range_loss_pct": loss_pct,
        "r80_by_mode": r80_by_mode,
        "curves": curves,
    }


def _compute_plate_geometry(barrels: list[dict[str, Any]], layout_type: str) -> dict[str, Any]:
    """Compute flange dimensions (mm), barrel centers (mm), and M8 mounting bolt pattern."""
    centers_mm = [
        {
            "barrel_id": str(b.get("barrel_id", f"B{i+1}")),
            "x_mm": float(b.get("y_m", 0.0)) * 1000.0,
            "y_mm": float(b.get("z_m", 0.0)) * 1000.0,
            "choke": str(b.get("choke", "Modified")),
            "od_mm": 21.6,
            "bore_id_mm": 18.5,
        }
        for i, b in enumerate(barrels)
    ]
    if not centers_mm:
        centers_mm = [{"barrel_id": "B1", "x_mm": 0.0, "y_mm": 0.0, "choke": "Modified", "od_mm": 21.6, "bore_id_mm": 18.5}]

    xs = [c["x_mm"] for c in centers_mm]
    ys = [c["y_mm"] for c in centers_mm]
    r_max_mm = max(math.hypot(c["x_mm"], c["y_mm"]) for c in centers_mm)
    is_circular_flange = str(layout_type) in ("circular", "concentric", "honeycomb")
    flange_r_mm = max(55.0, r_max_mm + 34.0)
    rect_half_w_mm = max(55.0, max(abs(x) for x in xs) + 34.0)
    rect_half_h_mm = max(55.0, max(abs(y) for y in ys) + 34.0)

    # 8x M8 (8.4 mm thru) mounting bolt holes around outer flange perimeter or rectangle corners
    bolt_holes: list[tuple[float, float]] = []
    if is_circular_flange:
        bolt_r = flange_r_mm - 12.0
        for k in range(8):
            ang = math.radians(k * 45.0 + 22.5)
            bolt_holes.append((round(bolt_r * math.cos(ang), 2), round(bolt_r * math.sin(ang), 2)))
    else:
        bx = rect_half_w_mm - 12.0
        by = rect_half_h_mm - 12.0
        bolt_holes = [(-bx, -by), (0.0, -by), (bx, -by), (bx, 0.0), (bx, by), (0.0, by), (-bx, by), (-bx, 0.0)]

    return {
        "centers_mm": centers_mm,
        "is_circular": is_circular_flange,
        "flange_r_mm": round(flange_r_mm, 1),
        "rect_half_w_mm": round(rect_half_w_mm, 1),
        "rect_half_h_mm": round(rect_half_h_mm, 1),
        "bolt_holes_mm": bolt_holes,
        "bolt_diam_mm": 8.4,
    }


def export_muzzle_plate_svg(
    barrels: list[dict[str, Any]],
    layout_type: str = "circular",
    title: str = "C-UAS Muzzle Clamp Plate",
    splay_angle_mrad: float = 0.0,
    pointing_mode: str = "convergent",
) -> str:
    """Render true-scale parametric 2D blueprint SVG of the multi-barrel muzzle clamp plate."""
    geom = _compute_plate_geometry(barrels, layout_type)
    half_ext = max(geom["flange_r_mm"], geom["rect_half_w_mm"], geom["rect_half_h_mm"]) + 28.0
    vb_min = -half_ext
    vb_span = 2.0 * half_ext

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_min:.1f} {vb_min:.1f} {vb_span:.1f} {vb_span:.1f}" '
        f'style="width:100%;max-height:420px;background:#0F172A;border:1px solid #334155;border-radius:8px;">',
        f'<text x="{vb_min + 10:.1f}" y="{vb_min + 16:.1f}" fill="#38BDF8" font-family="monospace" font-size="11">'
        f'{title} | {layout_type} | {len(geom["centers_mm"])}x 12ga (21.6 mm OD) | {pointing_mode} ({splay_angle_mrad:.1f} mrad)</text>',
        f'<line x1="{-half_ext + 10:.1f}" y1="0" x2="{half_ext - 10:.1f}" y2="0" stroke="#334155" stroke-width="0.9" stroke-dasharray="6,4" />',
        f'<line x1="0" y1="{-half_ext + 22:.1f}" x2="0" y2="{half_ext - 10:.1f}" stroke="#334155" stroke-width="0.9" stroke-dasharray="6,4" />',
    ]

    if geom["is_circular"]:
        parts.append(
            f'<circle cx="0" cy="0" r="{geom["flange_r_mm"]:.2f}" fill="#1E293B" stroke="#94A3B8" stroke-width="2.0" />'
        )
    else:
        rw = 2.0 * geom["rect_half_w_mm"]
        rh = 2.0 * geom["rect_half_h_mm"]
        parts.append(
            f'<rect x="{-geom["rect_half_w_mm"]:.2f}" y="{-geom["rect_half_h_mm"]:.2f}" width="{rw:.2f}" height="{rh:.2f}" '
            f'rx="10" fill="#1E293B" stroke="#94A3B8" stroke-width="2.0" />'
        )

    for bx, by in geom["bolt_holes_mm"]:
        parts.append(
            f'<circle cx="{bx:.2f}" cy="{-by:.2f}" r="{geom["bolt_diam_mm"] * 0.5:.2f}" '
            f'fill="#0F172A" stroke="#F59E0B" stroke-width="1.2" />'
        )

    for c in geom["centers_mm"]:
        cx = c["x_mm"]
        cy_svg = -c["y_mm"]
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy_svg:.2f}" r="{c["od_mm"] * 0.5:.2f}" '
            f'fill="none" stroke="#38BDF8" stroke-width="1.5" />'
        )
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy_svg:.2f}" r="{c["bore_id_mm"] * 0.5:.2f}" '
            f'fill="#0B1120" stroke="#10B981" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{cx:.2f}" y="{cy_svg + 2.5:.2f}" text-anchor="middle" fill="#F8FAFC" '
            f'font-family="monospace" font-size="7.5">{c["barrel_id"]}</text>'
        )

    flange_lbl = (
        f"OD {round(2 * geom['flange_r_mm'], 1)} mm"
        if geom["is_circular"]
        else f"{round(2 * geom['rect_half_w_mm'], 1)}x{round(2 * geom['rect_half_h_mm'], 1)} mm"
    )
    parts.append(
        f'<text x="{vb_min + 10:.1f}" y="{half_ext - 10:.1f}" fill="#94A3B8" font-family="monospace" font-size="9">'
        f'Scale: 1:1 mm | Flange: {flange_lbl} | 8x M8 (8.4 mm)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def export_muzzle_plate_dxf(
    barrels: list[dict[str, Any]],
    layout_type: str = "circular",
    title: str = "C-UAS Muzzle Clamp Plate",
    splay_angle_mrad: float = 0.0,
    pointing_mode: str = "convergent",
) -> str:
    """Emit standards-compliant AutoCAD ASCII DXF (mm units) for the parametric muzzle clamp plate."""
    geom = _compute_plate_geometry(barrels, layout_type)
    lines: list[str] = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "9", "$INSUNITS", "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LAYER",
        "70", "5",
    ]
    for lname, col in (("OUTLINE", 7), ("BARREL_BORES", 3), ("BOLT_HOLES", 2), ("CENTERLINES", 8), ("ANNOTATIONS", 4)):
        lines.extend(["0", "LAYER", "2", lname, "70", "0", "62", str(col), "6", "CONTINUOUS"])
    lines.extend(["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"])

    def _add_circle(layer: str, cx: float, cy: float, r: float, color: int) -> None:
        lines.extend(["0", "CIRCLE", "8", layer, "62", str(color), "10", f"{cx:.4f}", "20", f"{cy:.4f}", "30", "0.0", "40", f"{r:.4f}"])

    def _add_line(layer: str, x1: float, y1: float, x2: float, y2: float, color: int) -> None:
        lines.extend([
            "0", "LINE", "8", layer, "62", str(color),
            "10", f"{x1:.4f}", "20", f"{y1:.4f}", "30", "0.0",
            "11", f"{x2:.4f}", "21", f"{y2:.4f}", "31", "0.0",
        ])

    def _add_text(layer: str, x: float, y: float, height: float, text: str, color: int) -> None:
        lines.extend(["0", "TEXT", "8", layer, "62", str(color), "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0.0", "40", f"{height:.2f}", "1", text])

    if geom["is_circular"]:
        _add_circle("OUTLINE", 0.0, 0.0, geom["flange_r_mm"], 7)
        ext = geom["flange_r_mm"] + 15.0
    else:
        hw, hh = geom["rect_half_w_mm"], geom["rect_half_h_mm"]
        _add_line("OUTLINE", -hw, -hh, hw, -hh, 7)
        _add_line("OUTLINE", hw, -hh, hw, hh, 7)
        _add_line("OUTLINE", hw, hh, -hw, hh, 7)
        _add_line("OUTLINE", -hw, hh, -hw, -hh, 7)
        ext = max(hw, hh) + 15.0

    _add_line("CENTERLINES", -ext, 0.0, ext, 0.0, 8)
    _add_line("CENTERLINES", 0.0, -ext, 0.0, ext, 8)

    for bx, by in geom["bolt_holes_mm"]:
        _add_circle("BOLT_HOLES", bx, by, geom["bolt_diam_mm"] * 0.5, 2)

    for c in geom["centers_mm"]:
        _add_circle("BARREL_BORES", c["x_mm"], c["y_mm"], c["od_mm"] * 0.5, 5)
        _add_circle("BARREL_BORES", c["x_mm"], c["y_mm"], c["bore_id_mm"] * 0.5, 3)
        _add_text("ANNOTATIONS", c["x_mm"] + 12.5, c["y_mm"] - 3.0, 5.0, f"{c['barrel_id']} {c['choke'][:3]}", 4)

    _add_text(
        "ANNOTATIONS",
        -ext + 5.0,
        -ext - 12.0,
        6.0,
        f"{title} ({layout_type}, {pointing_mode}, splay {splay_angle_mrad:.1f} mrad)",
        7,
    )
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"
