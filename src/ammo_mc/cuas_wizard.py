"""3-Click Turret Sizing Wizard & Interactive Red-vs-Blue Duel Diagnostic Engine."""

from __future__ import annotations

import math
from typing import Any

from ammo_mc.cuas_adversarial import (
    evaluate_red_vs_blue_matchup_mc,
    normalize_blue_config,
    normalize_red_config,
)
from ammo_mc.cuas_defense import _estimate_burst_pk_analytics
from ammo_mc.cuas_hardware import MOUNT_CLASS_SPECS, estimate_turret_swap_c
from ammo_mc.cuas_mission import CUAS_AMMO_ECONOMICS_CATALOG, _resolve_shotgun_ammo
from ammo_mc.small_arms import compute_drone_projected_geometry, get_drone_target_specs

WIZARD_ARCHETYPES: list[dict[str, Any]] = [
    {
        "name": "2-Bbl Ultralight #4 Buck Tripod Rapid",
        "layout_type": "grid",
        "n_barrels": 2,
        "barrels_per_burst": 1,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "convergent",
        "firing_mode": "sequential_ripple",
        "ripple_delay_ms": 8.0,
        "sensor_preset": "EO_IR_Dual_Thermal_Tracker",
        "max_engagement_range_m": 45.0,
        "slew_rate_deg_s": 160.0,
        "reload_time_s": 2.4,
    },
    {
        "name": "3-Bbl Light TSS #2 Tripod Precision",
        "layout_type": "circular",
        "n_barrels": 3,
        "barrels_per_burst": 1,
        "ammunition_preset": "12G_TSS_NO2_C_UAS",
        "choke": "Full",
        "pointing_mode": "convergent",
        "firing_mode": "sequential_ripple",
        "ripple_delay_ms": 10.0,
        "sensor_preset": "EO_IR_Dual_Thermal_Tracker",
        "max_engagement_range_m": 62.0,
        "slew_rate_deg_s": 165.0,
        "reload_time_s": 2.6,
    },
    {
        "name": "4-Bbl Agile #4 Buck Curtain (UGV)",
        "layout_type": "circular",
        "n_barrels": 4,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "divergent_curtain",
        "splay_angle_mrad": 5.0,
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 8.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 50.0,
        "slew_rate_deg_s": 180.0,
        "reload_time_s": 2.6,
    },
    {
        "name": "4-Bbl Hybrid Dual-Ring (#4 Buck + TSS #2)",
        "layout_type": "concentric",
        "n_barrels": 4,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "convergent",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 10.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 64.0,
        "slew_rate_deg_s": 175.0,
        "reload_time_s": 2.8,
        "hybrid_mode": True,
        "inner_ammunition_preset": "12G_TSS_NO2_C_UAS",
        "inner_choke": "Extra Full",
        "hybrid_switch_range_m": 38.0,
    },
    {
        "name": "6-Bbl Balanced #4 Buck Economical",
        "layout_type": "circular",
        "n_barrels": 6,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "convergent",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 8.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 55.0,
        "slew_rate_deg_s": 170.0,
        "reload_time_s": 3.0,
    },
    {
        "name": "6-Bbl Long-Range TSS #2 Full Choke",
        "layout_type": "concentric",
        "n_barrels": 6,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_TSS_NO2_C_UAS",
        "choke": "Full",
        "pointing_mode": "convergent",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 10.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 72.0,
        "slew_rate_deg_s": 165.0,
        "reload_time_s": 3.0,
    },
    {
        "name": "6-Bbl Deep-Reach Flechette Sabot",
        "layout_type": "grid",
        "n_barrels": 6,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_FLECHETTE_SABOT_C_UAS",
        "choke": "Extra Full",
        "pointing_mode": "convergent",
        "firing_mode": "sequential_ripple",
        "ripple_delay_ms": 10.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 78.0,
        "slew_rate_deg_s": 155.0,
        "reload_time_s": 3.4,
    },
    {
        "name": "8-Bbl Concentric Hybrid (#4 Buck + TSS #2)",
        "layout_type": "concentric",
        "n_barrels": 8,
        "barrels_per_burst": 2,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "divergent_curtain",
        "splay_angle_mrad": 4.2,
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 9.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 70.0,
        "slew_rate_deg_s": 175.0,
        "reload_time_s": 3.0,
        "hybrid_mode": True,
        "inner_ammunition_preset": "12G_TSS_NO2_C_UAS",
        "inner_choke": "Extra Full",
        "hybrid_switch_range_m": 42.0,
    },
    {
        "name": "8-Bbl Duplex High-Density Curtain",
        "layout_type": "honeycomb",
        "n_barrels": 8,
        "barrels_per_burst": 3,
        "ammunition_preset": "12G_DUPLEX_TSS_BUCK_C_UAS",
        "choke": "Full",
        "pointing_mode": "divergent_curtain",
        "splay_angle_mrad": 4.8,
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 8.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 66.0,
        "slew_rate_deg_s": 160.0,
        "reload_time_s": 3.2,
    },
    {
        "name": "10-Bbl Honeycomb Wide-Net Area Denial",
        "layout_type": "honeycomb",
        "n_barrels": 10,
        "barrels_per_burst": 3,
        "ammunition_preset": "12G_DUPLEX_TSS_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "staggered_tiling",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 8.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 64.0,
        "slew_rate_deg_s": 150.0,
        "reload_time_s": 3.4,
    },
    {
        "name": "10-Bbl Heavy Hybrid Shield (#4 Buck + Flechette)",
        "layout_type": "concentric",
        "n_barrels": 10,
        "barrels_per_burst": 3,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "convergent",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 10.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 76.0,
        "slew_rate_deg_s": 155.0,
        "reload_time_s": 3.2,
        "hybrid_mode": True,
        "inner_ammunition_preset": "12G_FLECHETTE_SABOT_C_UAS",
        "inner_choke": "Extra Full",
        "hybrid_switch_range_m": 46.0,
    },
    {
        "name": "12-Bbl Armored Citadel Saturation Turret",
        "layout_type": "concentric",
        "n_barrels": 12,
        "barrels_per_burst": 4,
        "ammunition_preset": "12G_TSS_NO2_C_UAS",
        "choke": "Full",
        "pointing_mode": "divergent_curtain",
        "splay_angle_mrad": 4.5,
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 7.0,
        "sensor_preset": "Sensor_Fused_Radar_EOIR",
        "max_engagement_range_m": 78.0,
        "slew_rate_deg_s": 150.0,
        "reload_time_s": 3.5,
    },
]


def _interp_threshold_range(ranges: list[float], pks: list[float], target_pk: float) -> float:
    r_out = 0.0
    for i, pk_v in enumerate(pks):
        if pk_v >= target_pk:
            r_out = ranges[i]
        elif i > 0 and pks[i - 1] >= target_pk and pk_v < target_pk:
            frac = (pks[i - 1] - target_pk) / max(pks[i - 1] - pk_v, 1e-6)
            return round(ranges[i - 1] + frac * (ranges[i] - ranges[i - 1]), 1)
    return round(r_out, 1)


def run_turret_sizing_wizard(
    drone_model: str = "fpv_quad_small",
    threat_speed_mps: float = 25.0,
    required_range_m: float = 45.0,
    mount_class: str = "light_ugv",
    budget_priority: str = "balanced",
    seed: int = 42,
    calibration_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Size and rank Top 3 feasible C-UAS multi-barrel shotgun turret configurations from 3 inputs."""
    _ = seed  # Deterministic analytical envelope & SWaP-C sizing
    mc_spec = MOUNT_CLASS_SPECS.get(mount_class, MOUNT_CLASS_SPECS["light_ugv"])
    drone_specs = get_drone_target_specs()
    d_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(d_spec, aspect_mode="head_on_frontal")

    r_req = float(max(15.0, min(85.0, required_range_m)))
    spd = float(max(10.0, min(60.0, threat_speed_mps)))
    spd_pen = max(0.72, 1.0 - max(0.0, spd - 15.0) * 0.0055)
    sweep_ranges = [15.0, 22.0, 30.0, 38.0, 45.0, 52.0, 60.0, 68.0, 76.0]

    evaluated: list[dict[str, Any]] = []
    for raw_arch in WIZARD_ARCHETYPES:
        cfg = normalize_blue_config(raw_arch, max_barrels=12)
        swap = estimate_turret_swap_c(
            n_barrels=cfg["n_barrels"],
            barrel_length_in=24.0,
            layout_type=cfg["layout_type"],
            ammunition_preset=cfg["ammunition_preset"],
            slew_rate_deg_s=cfg["slew_rate_deg_s"],
            sensor_preset=cfg["sensor_preset"],
            firing_mode=cfg["firing_mode"],
            ripple_delay_ms=cfg["ripple_delay_ms"],
            hybrid_mode=bool(cfg.get("hybrid_mode", False)),
            calibration_profile=calibration_profile,
        )
        sw_sm = swap["summary"]
        mass_ok = sw_sm["total_mass_kg"] <= mc_spec["max_mass_kg"]
        rec_ok = sw_sm["peak_recoil_force_kn"] <= mc_spec["max_recoil_kn"]
        feasible = bool(mass_ok and rec_ok)

        outer_key = cfg["ammunition_preset"]
        _, a_out = _resolve_shotgun_ammo(outer_key)
        c_out = float(CUAS_AMMO_ECONOMICS_CATALOG.get(outer_key, CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])["unit_cost_usd"])
        is_hyb = bool(cfg.get("hybrid_mode", False))
        inner_key = cfg["inner_ammunition_preset"] if is_hyb else outer_key
        _, a_in = _resolve_shotgun_ammo(inner_key)
        c_in = float(CUAS_AMMO_ECONOMICS_CATALOG.get(inner_key, CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])["unit_cost_usd"])

        def _pk_at(r_m: float) -> float:
            use_in = is_hyb and r_m > float(cfg["hybrid_switch_range_m"])
            a_sel = a_in if use_in else a_out
            chk_sel = cfg["inner_choke"] if use_in else cfg["choke"]
            pk = _estimate_burst_pk_analytics(
                slant_range_m=r_m,
                barrels_fired=cfg["barrels_per_burst"],
                pellets_per_shot=int(a_sel["pellets_per_shot"]),
                pellet_diameter_mm=float(a_sel["pellet_diameter_mm"]),
                pellet_mass_g=float(a_sel["pellet_mass_g"]),
                v0_mps=float(a_sel["v0_mps"]),
                choke_name=chk_sel,
                drone_spec=d_spec,
                base_cd=float(a_sel.get("effective_cd", 0.47)),
                aspect_mode="head_on_frontal",
                aspect_geom=aspect_geom,
                calibration_profile=calibration_profile,
            )
            curt_fac = 0.96 if cfg["pointing_mode"] == "divergent_curtain" else 1.0
            return round(min(99.8, 100.0 * pk * spd_pen * curt_fac), 1)

        pk_req = _pk_at(r_req)
        sweep_pks = [_pk_at(rv) for rv in sweep_ranges]
        r80 = _interp_threshold_range(sweep_ranges, sweep_pks, 80.0)
        r50 = _interp_threshold_range(sweep_ranges, sweep_pks, 50.0)
        avg_shell_cost = 0.5 * (c_out + c_in) if is_hyb else c_out
        cpk = round((cfg["barrels_per_burst"] * avg_shell_cost) / max(pk_req / 100.0, 0.15), 2)

        if budget_priority == "low_cost":
            score = pk_req - 0.8 * cpk - 1.5 * cfg["n_barrels"]
        elif budget_priority == "max_lethality":
            score = 1.2 * pk_req + 0.6 * r80
        else:
            score = pk_req + 0.35 * r80 - 0.35 * cpk

        if not feasible:
            score -= 80.0 + max(0.0, sw_sm["total_mass_kg"] - mc_spec["max_mass_kg"]) * 0.5 + max(0.0, sw_sm["peak_recoil_force_kn"] - mc_spec["max_recoil_kn"]) * 12.0

        reasons: list[str] = []
        if is_hyb:
            reasons.append(f"dual-ring {cfg['ammunition_preset']} + {cfg['inner_ammunition_preset']}")
        else:
            reasons.append(f"{cfg['n_barrels']}B {cfg['layout_type']} with {cfg['ammunition_preset']}")
        reasons.append(f"R80={r80:.0f} m (vs {r_req:.0f} m req)")
        reasons.append(f"{sw_sm['total_mass_kg']:.1f} kg & {sw_sm['peak_recoil_force_kn']:.2f} kN recoil")
        if not feasible:
            gap_parts = []
            if not mass_ok:
                gap_parts.append(f"mass {sw_sm['total_mass_kg']:.1f}>{mc_spec['max_mass_kg']:.0f} kg")
            if not rec_ok:
                gap_parts.append(f"recoil {sw_sm['peak_recoil_force_kn']:.2f}>{mc_spec['max_recoil_kn']:.1f} kN")
            reasons.append("EXCEEDS MOUNT (" + ", ".join(gap_parts) + ")")

        evaluated.append(
            {
                "name": cfg["name"],
                "config": cfg,
                "pk_at_range_pct": pk_req,
                "r80_m": r80,
                "r50_m": r50,
                "mass_kg": sw_sm["total_mass_kg"],
                "peak_recoil_kn": sw_sm["peak_recoil_force_kn"],
                "peak_power_w": sw_sm["peak_power_w"],
                "cost_per_kill_usd": cpk,
                "unit_bom_usd": sw_sm["unit_bom_usd"],
                "mount_feasible": feasible,
                "rationale": " • ".join(reasons),
                "_score": round(score, 2),
            }
        )

    evaluated.sort(key=lambda row: row["_score"], reverse=True)
    feasible_rows = [r for r in evaluated if r["mount_feasible"]]
    top_pool = feasible_rows[:3] if len(feasible_rows) >= 3 else (feasible_rows + [r for r in evaluated if not r["mount_feasible"]])[:3]
    clean_eval = [{k: v for k, v in r.items() if k != "_score"} for r in evaluated]
    clean_top = [{k: v for k, v in r.items() if k != "_score"} for r in top_pool]

    return {
        "top_3": clean_top,
        "all_evaluated": clean_eval,
        "has_feasible": len(feasible_rows) > 0,
        "n_feasible": len(feasible_rows),
        "constraints": {
            "mount_class": mount_class,
            "mount_name": mc_spec["name"],
            "max_mass_kg": mc_spec["max_mass_kg"],
            "max_recoil_kn": mc_spec["max_recoil_kn"],
            "max_power_w": mc_spec["max_power_w"],
            "required_range_m": r_req,
            "drone_model": drone_model,
            "threat_speed_mps": spd,
        },
        "summary": {
            "top1_name": clean_top[0]["name"] if clean_top else "N/A",
            "top1_pk_at_range_pct": clean_top[0]["pk_at_range_pct"] if clean_top else 0.0,
            "top1_r80_m": clean_top[0]["r80_m"] if clean_top else 0.0,
            "top1_mass_kg": clean_top[0]["mass_kg"] if clean_top else 0.0,
            "top1_recoil_kn": clean_top[0]["peak_recoil_kn"] if clean_top else 0.0,
            "n_feasible": len(feasible_rows),
        },
    }


def run_red_vs_blue_duel_diagnostic(
    blue_config: dict[str, Any],
    red_config: dict[str, Any],
    mount_load_limit_kn: float = 4.0,
    n_episodes: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Pit a Blue turret against a Red threat and diagnose exact breach root causes & prescriptions."""
    blue_n = normalize_blue_config(blue_config, max_barrels=12)
    red_n = normalize_red_config(red_config, max_swarm_size=12)

    mc_res = evaluate_red_vs_blue_matchup_mc(
        blue=blue_n,
        red=red_n,
        n_episodes=max(8, int(n_episodes)),
        mount_load_limit_kn=float(mount_load_limit_kn),
        seed=seed,
    )
    swap = estimate_turret_swap_c(
        n_barrels=blue_n["n_barrels"],
        layout_type=blue_n["layout_type"],
        ammunition_preset=blue_n["ammunition_preset"],
        slew_rate_deg_s=blue_n["slew_rate_deg_s"],
        sensor_preset=blue_n["sensor_preset"],
        firing_mode=blue_n["firing_mode"],
        ripple_delay_ms=blue_n["ripple_delay_ms"],
        hybrid_mode=bool(blue_n.get("hybrid_mode", False)),
    )

    zb = float(mc_res["zero_breach_pct"])
    if zb >= 80.0:
        verdict = "BLUE DOMINANT"
    elif zb >= 45.0:
        verdict = "CONTESTED EDGE"
    else:
        verdict = "RED BREACH ADVANTAGE"

    fm = mc_res["failure_modes"]
    prescriptions: list[str] = []
    if fm["reload_gap_pct"] >= 25.0 or (fm["total_leakers"] > 0 and blue_n["n_barrels"] < red_n["swarm_size"]):
        prescriptions.append(
            f"Reload-Window Exhaustion ({fm['reload_gap_pct']:.0f}% of leakers): increase barrel count above "
            f"{blue_n['n_barrels']}, reduce burst size below {blue_n['barrels_per_burst']}, or activate Dynamic RL Fire Control to stop over-shooting."
        )
    if fm["tracking_evasion_miss_pct"] >= 25.0:
        prescriptions.append(
            f"Evasive Tracking Lag ({fm['tracking_evasion_miss_pct']:.0f}% of leakers): switch to 3-state Kalman "
            f"predictor (`kalman_alpha_beta_gamma`), widen pointing to `divergent_curtain` (~4-5 mrad splay), or raise slew rate >=180 deg/s."
        )
    if fm["slew_bottleneck_pct"] >= 25.0:
        prescriptions.append(
            f"Slew-Rate Timeline Bottleneck ({fm['slew_bottleneck_pct']:.0f}% of leakers): extend engagement envelope "
            f"(`max_engagement_range_m` >= 60 m with TSS #2), switch priority to `minimum_time_to_breach`, or add cooperative 2nd turret."
        )
    if fm["insufficient_lethality_spread_pct"] >= 25.0:
        prescriptions.append(
            f"Insufficient Pattern Density / Pellet Energy ({fm['insufficient_lethality_spread_pct']:.0f}% of leakers): "
            f"upgrade to tungsten TSS #2 / Full choke, or enable Hybrid Concentric Mix for inner high-energy reach."
        )
    if not mc_res["mount_feasible"]:
        prescriptions.append(
            f"Structural Recoil Overload ({mc_res['peak_mount_force_kn']:.2f} kN > {mount_load_limit_kn:.2f} kN limit): "
            f"enable `balanced_pair_ripple` (>=8 ms delay) or lower barrels per burst."
        )
    if not prescriptions:
        prescriptions.append(
            f"Balanced Defense Verified ({zb:.1f}% zero-breach): array handles the configured threat package cleanly within recoil margins."
        )

    return {
        "verdict": verdict,
        "blue_config": blue_n,
        "red_config": red_n,
        "matchup": mc_res,
        "failure_modes": fm,
        "prescriptions": prescriptions[:4],
        "swap_c_summary": swap["summary"],
    }
