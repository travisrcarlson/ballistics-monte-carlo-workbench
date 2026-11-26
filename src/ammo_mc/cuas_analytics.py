"""Counter-UAS Trade-Study Comparator, Layered Defense, GIS Exporters & Tornado Sensitivity Engine.

Provides strictly unclassified, deterministic-seeded engineering analytics built on top of
`cuas_defense`, `cuas_mission`, and the core `ammo_mc.pipeline` ballistics engine.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ammo_mc.cuas_defense import (
    _estimate_burst_pk_analytics,
    generate_multishotgun_array_layout,
    get_shotgun_presets,
    simulate_cuas_pellet_fallout_hazard,
    simulate_drone_swarm_saturation_mc,
    simulate_turret_recoil_and_gimbal_loads,
)
from ammo_mc.cuas_mission import (
    _AMMO_PRESET_ALIASES,
    get_cuas_ammo_economics_catalog,
    get_magazine_system_specs,
    simulate_cuas_kill_chain_mc,
    simulate_cuas_sustained_logistics_mc,
)
from ammo_mc.environment import AtmosphereEnvironment
from ammo_mc.exterior import BatchTrajectoryConfig, integrate_trajectories_batch
from ammo_mc.gis import EARTH_RADIUS_M
from ammo_mc.small_arms import compute_drone_projected_geometry, get_drone_target_specs
from ammo_mc.schemas import AmmoDefinition, Scenario

# -------------------------------------------------------------------------
# 1. A/B/C TURRET ARCHITECTURE COMPARATOR
# -------------------------------------------------------------------------

CUAS_COMPARATOR_PRESETS: dict[str, dict[str, Any]] = {
    "concentric_tss_buck": {
        "preset_key": "concentric_tss_buck",
        "name": "6-Bbl Concentric (TSS #2 + #4 Buck, Balanced Ripple)",
        "layout_type": "concentric",
        "n_barrels": 6,
        "ammunition_preset": "12G_TSS_NO2_C_UAS",
        "choke": "Full",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 10.0,
        "magazine_type": "box_10rd",
        "max_weapon_range_m": 62.0,
        "salvo_cycle_time_s": 0.75,
    },
    "airburst_4bbl_ripple": {
        "preset_key": "airburst_4bbl_ripple",
        "name": "4-Bbl Programmable Airburst Canister (Ripple)",
        "layout_type": "circular",
        "n_barrels": 4,
        "ammunition_preset": "CUAS_AIRBURST_CANISTER",
        "choke": "Modified",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 12.0,
        "magazine_type": "box_10rd",
        "max_weapon_range_m": 68.0,
        "salvo_cycle_time_s": 0.90,
    },
    "workhorse_6bbl_no4buck": {
        "preset_key": "workhorse_6bbl_no4buck",
        "name": "6-Bbl Circular #4 Buckshot (Simultaneous Volley)",
        "layout_type": "circular",
        "n_barrels": 6,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "firing_mode": "simultaneous_volley",
        "ripple_delay_ms": 0.0,
        "magazine_type": "tube_7rd",
        "max_weapon_range_m": 55.0,
        "salvo_cycle_time_s": 0.85,
    },
    "highvol_8bbl_tshot": {
        "preset_key": "highvol_8bbl_tshot",
        "name": "8-Bbl Steel T-Shot Drum (Sustained Swarm)",
        "layout_type": "circular",
        "n_barrels": 8,
        "ammunition_preset": "12G_T_SHOT_STEEL_C_UAS",
        "choke": "Modified",
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 9.0,
        "magazine_type": "drum_20rd",
        "max_weapon_range_m": 58.0,
        "salvo_cycle_time_s": 0.70,
    },
}

_ECON_TO_SHOTGUN_PRESET: dict[str, str] = {
    "CUAS_AIRBURST_CANISTER": "12G_TSS_NO2_C_UAS",
    "762_MINIGUN_BURST": "12G_FLECHETTE_SABOT_C_UAS",
}


def get_cuas_comparator_presets() -> dict[str, dict[str, Any]]:
    """Return curated turret architecture archetypes for A/B/C comparisons."""
    return CUAS_COMPARATOR_PRESETS


def _resolve_shotgun_for_config(ammo_key: str) -> str:
    sg_presets = get_shotgun_presets()
    canon = _AMMO_PRESET_ALIASES.get(str(ammo_key).lower(), str(ammo_key))
    if canon in sg_presets:
        return canon
    mapped = _ECON_TO_SHOTGUN_PRESET.get(canon, "12G_NO4_BUCK_C_UAS")
    return mapped if mapped in sg_presets else next(iter(sg_presets))


def _resolve_econ_for_config(ammo_key: str) -> str:
    econ_cat = get_cuas_ammo_economics_catalog()
    canon = _AMMO_PRESET_ALIASES.get(str(ammo_key).lower(), str(ammo_key))
    if canon in econ_cat:
        return canon
    return "12G_NO4_BUCK_C_UAS"


def _minmax_score(values: list[float], higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span < 1e-9:
        return [75.0 for _ in values]
    scores: list[float] = []
    for v in values:
        norm = (v - vmin) / span
        if not higher_is_better:
            norm = 1.0 - norm
        scores.append(round(20.0 + 80.0 * norm, 1))
    return scores


def compare_cuas_architectures_ab(
    configs: list[dict[str, Any]] | None = None,
    drone_model: str = "fpv_quad_small",
    aspect_mode: str = "head_on_frontal",
    target_distance_m: float = 40.0,
    drone_speed_mps: float = 25.0,
    weather: str = "clear_day",
    fiber_optic_autonomous: bool = False,
    n_swarm_drones: int = 6,
    n_waves: int = 4,
    drones_per_wave: int = 3,
    seed: int = 42,
    use_common_random_numbers: bool = True,
) -> dict[str, Any]:
    """Compare 2 to 4 turret architectures side by side across 6 operational metrics."""
    if not configs:
        selected = [
            CUAS_COMPARATOR_PRESETS["concentric_tss_buck"],
            CUAS_COMPARATOR_PRESETS["workhorse_6bbl_no4buck"],
            CUAS_COMPARATOR_PRESETS["airburst_4bbl_ripple"],
        ]
    else:
        selected = list(configs)[:4]

    raw_rows: list[dict[str, Any]] = []
    trial_pk_arrays: list[np.ndarray] = []
    for idx, cfg in enumerate(selected):
        name = str(cfg.get("name", f"Config {chr(65 + idx)}"))
        layout_type = str(cfg.get("layout_type", "circular"))
        n_barrels = max(2, int(cfg.get("n_barrels", 6)))
        ammo_raw = str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))
        sg_ammo = _resolve_shotgun_for_config(ammo_raw)
        econ_ammo = _resolve_econ_for_config(ammo_raw)
        choke = str(cfg.get("choke", "Modified"))
        firing_mode = str(cfg.get("firing_mode", "balanced_pair_ripple"))
        ripple_ms = float(cfg.get("ripple_delay_ms", 10.0))
        mag_type = str(cfg.get("magazine_type", "box_10rd"))
        max_range = float(cfg.get("max_weapon_range_m", 58.0))
        cycle_s = float(cfg.get("salvo_cycle_time_s", 0.85))

        n_kc_trials = 45
        cfg_seed_kc = seed if use_common_random_numbers else (seed + idx * 97)
        kc_res = simulate_cuas_kill_chain_mc(
            drone_model=drone_model,
            aspect_mode=aspect_mode,
            drone_speed_mps=drone_speed_mps,
            weather=weather,
            fiber_optic_autonomous=fiber_optic_autonomous,
            n_barrels=n_barrels,
            ammunition_preset=sg_ammo,
            choke=choke,
            max_weapon_range_m=max_range,
            salvo_cycle_time_s=cycle_s,
            n_trials=n_kc_trials,
            seed=cfg_seed_kc,
        )
        kc_sm = kc_res["summary"]
        trial_pk_arrays.append(np.asarray(kc_res.get("trial_cum_pks", []), dtype=np.float64))

        b_per_burst = 2 if firing_mode != "simultaneous_volley" else min(n_barrels, 4)
        sw_res = simulate_drone_swarm_saturation_mc(
            swarm_size=n_swarm_drones,
            n_barrels=n_barrels,
            barrels_per_burst=b_per_burst,
            ammunition_preset=sg_ammo,
            default_choke=choke,
            drone_model=drone_model,
            mean_drone_speed_mps=drone_speed_mps,
            aspect_mode=aspect_mode,
            max_engagement_range_m=max_range,
            n_waves=12,
            seed=seed + 100 if use_common_random_numbers else (seed + 100 + idx),
            include_doctrine_sweep=False,
        )
        sw_sm = sw_res["summary"]
        swarm_survival_pct = round(max(0.0, 100.0 - float(sw_sm.get("leaker_rate_pct", 0.0))), 1)

        barrels = generate_multishotgun_array_layout(
            layout_type=layout_type if layout_type != "custom" else "circular",
            n_barrels=n_barrels,
            default_choke=choke,
            default_ammunition_preset=sg_ammo,
        )
        rc_res = simulate_turret_recoil_and_gimbal_loads(
            barrels=barrels,
            firing_mode=firing_mode,
            ripple_delay_ms=ripple_ms,
            default_ammunition_preset=sg_ammo,
        )
        rc_sm = rc_res["summary"]

        fl_res = simulate_cuas_pellet_fallout_hazard(
            layout_type=layout_type if layout_type != "custom" else "circular",
            n_barrels=n_barrels,
            ammunition_preset=sg_ammo,
            default_choke=choke,
            intercept_slant_range_m=target_distance_m,
            n_sample_pellets=75,
            seed=seed + 200 if use_common_random_numbers else (seed + 200 + idx),
        )
        fl_sm = fl_res["summary"]

        lg_res = simulate_cuas_sustained_logistics_mc(
            n_waves=n_waves,
            drones_per_wave=drones_per_wave,
            n_barrels=n_barrels,
            ammo_family=econ_ammo,
            magazine_type=mag_type,
            target_range_m=target_distance_m,
            drone_model=drone_model,
            aspect_mode=aspect_mode,
            n_trials=20,
            seed=seed + 300 if use_common_random_numbers else (seed + 300 + idx),
        )
        lg_sm = lg_res["summary"]
        peak_temp_c = float(lg_sm["peak_barrel_temp_c"])
        cookoff_margin_c = round(float(lg_sm.get("cookoff_limit_c", 220.0)) - peak_temp_c, 1)

        # Pure physics cumulative kill-chain Pk without artificial preset floor clamping
        pk_kc = float(kc_sm["cumulative_pk_pct"])

        raw_rows.append(
            {
                "label": f"{chr(65 + idx)}: {name}",
                "short_name": name,
                "config_id": str(cfg.get("preset_key", f"cfg_{idx}")),
                "layout_type": layout_type,
                "n_barrels": n_barrels,
                "ammunition": econ_ammo,
                "firing_mode": firing_mode,
                "kill_chain_pk_pct": round(pk_kc, 1),
                "swarm_survival_pct": swarm_survival_pct,
                "peak_recoil_kn": round(float(rc_sm["peak_transmitted_mount_force_kn"]), 2),
                "gimbal_walk_mrad": round(float(rc_sm["max_muzzle_climb_mrad"]), 2),
                "sdz_max_range_m": round(float(fl_sm["max_fallout_range_99_m"]), 1),
                "peak_barrel_temp_c": round(peak_temp_c, 1),
                "cookoff_margin_c": cookoff_margin_c,
                "cost_per_kill_usd": round(float(lg_sm["mean_cost_per_kill_usd"]), 2),
                "wave_survival_pct": round(float(lg_sm["wave_survival_pct"]), 1),
            }
        )

    s_kc = _minmax_score([r["kill_chain_pk_pct"] for r in raw_rows], True)
    s_sw = _minmax_score([r["swarm_survival_pct"] for r in raw_rows], True)
    s_rc = _minmax_score([r["peak_recoil_kn"] for r in raw_rows], False)
    s_sd = _minmax_score([r["sdz_max_range_m"] for r in raw_rows], False)
    s_th = _minmax_score([r["cookoff_margin_c"] for r in raw_rows], True)
    s_cs = _minmax_score([r["cost_per_kill_usd"] for r in raw_rows], False)

    base = raw_rows[0]
    arr_base = trial_pk_arrays[0] if trial_pk_arrays else np.array([], dtype=np.float64)
    t_crit = float(sp_stats.t.ppf(0.975, df=max(1, len(arr_base) - 1)))
    for i, r in enumerate(raw_rows):
        mei = round(
            0.30 * s_kc[i]
            + 0.22 * s_sw[i]
            + 0.14 * s_rc[i]
            + 0.10 * s_sd[i]
            + 0.12 * s_th[i]
            + 0.12 * s_cs[i],
            1,
        )
        r["mei_score"] = mei
        r["radar_scores"] = {
            "Kill-Chain Pk": s_kc[i],
            "Swarm Endurance": s_sw[i],
            "Recoil Stability": s_rc[i],
            "SDZ Compactness": s_sd[i],
            "Thermal Margin": s_th[i],
            "Cost Efficiency": s_cs[i],
        }
        r["delta_pk_vs_a"] = round(r["kill_chain_pk_pct"] - base["kill_chain_pk_pct"], 1)
        r["delta_cost_vs_a_usd"] = round(r["cost_per_kill_usd"] - base["cost_per_kill_usd"], 2)

        arr_i = trial_pk_arrays[i] if i < len(trial_pk_arrays) else np.array([], dtype=np.float64)
        n_i = len(arr_i)
        if n_i > 1:
            mean_i = float(np.mean(arr_i))
            se_i = float(np.std(arr_i, ddof=1) / math.sqrt(n_i))
            r["kill_chain_pk_ci95_pct"] = [
                round(max(0.0, mean_i - t_crit * se_i), 1),
                round(min(100.0, mean_i + t_crit * se_i), 1),
            ]
        else:
            r["kill_chain_pk_ci95_pct"] = [r["kill_chain_pk_pct"], r["kill_chain_pk_pct"]]

        if i == 0:
            r["delta_pk_vs_a_ci95_pct"] = [0.0, 0.0]
            r["crn_corr_vs_a"] = 1.0
        elif use_common_random_numbers and n_i == len(arr_base) and n_i > 1:
            diffs = arr_i - arr_base
            d_mean = float(np.mean(diffs))
            se_diff = float(np.std(diffs, ddof=1) / math.sqrt(n_i))
            r["delta_pk_vs_a_ci95_pct"] = [
                round(d_mean - t_crit * se_diff, 1),
                round(d_mean + t_crit * se_diff, 1),
            ]
            sd_i = float(np.std(arr_i, ddof=1))
            sd_b = float(np.std(arr_base, ddof=1))
            if sd_i > 1e-9 and sd_b > 1e-9:
                corr = float(np.corrcoef(arr_i, arr_base)[0, 1])
                r["crn_corr_vs_a"] = round(0.0 if math.isnan(corr) else corr, 3)
            else:
                r["crn_corr_vs_a"] = 0.0
        elif n_i > 1 and len(arr_base) > 1:
            d_mean = float(np.mean(arr_i) - np.mean(arr_base))
            se_diff = math.sqrt((float(np.var(arr_i, ddof=1)) + float(np.var(arr_base, ddof=1))) / min(n_i, len(arr_base)))
            r["delta_pk_vs_a_ci95_pct"] = [
                round(d_mean - t_crit * se_diff, 1),
                round(d_mean + t_crit * se_diff, 1),
            ]
            r["crn_corr_vs_a"] = 0.0
        else:
            r["delta_pk_vs_a_ci95_pct"] = [r["delta_pk_vs_a"], r["delta_pk_vs_a"]]
            r["crn_corr_vs_a"] = 0.0

    winner = max(raw_rows, key=lambda x: x["mei_score"])
    return {
        "summary": {
            "n_configs": len(raw_rows),
            "winner_label": winner["label"],
            "winner_mei": winner["mei_score"],
            "baseline_label": base["label"],
            "drone_model": drone_model,
            "weather": weather,
            "fiber_optic_autonomous": fiber_optic_autonomous,
            "crn_active": bool(use_common_random_numbers),
        },
        "scorecard": raw_rows,
    }


# -------------------------------------------------------------------------
# 2. LAYERED C-UAS DEFENSE SIMULATOR (OUTER LAYER + INNER MULTI-SHOTGUN)
# -------------------------------------------------------------------------

OUTER_LAYER_CATALOG: dict[str, dict[str, Any]] = {
    "rf_gnss_jammer": {
        "name": "Sectoral RF / C2 / GNSS Soft-Kill Jammer",
        "effective_range_m": 450.0,
        "base_pk_rf": 0.72,
        "base_pk_fiber_optic": 0.0,
        "cost_per_engagement_usd": 0.05,
        "magazine_capacity": 9999,
        "weather_penalty_fog": 0.96,
        "description": "Low-cost electronic attack vs RF-linked UAS; zero effect against fiber-optic or INS/visual autonomous drones.",
    },
    "interceptor_drone_fpv": {
        "name": "Kinetic Hit-to-Kill Interceptor Drone",
        "effective_range_m": 800.0,
        "base_pk_rf": 0.78,
        "base_pk_fiber_optic": 0.76,
        "cost_per_engagement_usd": 650.0,
        "magazine_capacity": 8,
        "weather_penalty_fog": 0.68,
        "description": "Reusable/expendable agile quad interceptor with magazine limit; defeats both RF and fiber-optic threats.",
    },
    "autocannon_30mm_airburst": {
        "name": "30mm AHEAD Programmable Airburst Turret",
        "effective_range_m": 1500.0,
        "base_pk_rf": 0.85,
        "base_pk_fiber_optic": 0.85,
        "cost_per_engagement_usd": 420.0,
        "magazine_capacity": 40,
        "weather_penalty_fog": 0.90,
        "description": "Medium-range all-threat airburst ring; high unit round cost motivates screening leakers to shotgun last-ditch tier.",
    },
    "directed_energy_hel": {
        "name": "10 kW Compact Tactical Laser (HEL)",
        "effective_range_m": 1100.0,
        "base_pk_rf": 0.80,
        "base_pk_fiber_optic": 0.78,
        "cost_per_engagement_usd": 2.50,
        "magazine_capacity": 60,
        "weather_penalty_fog": 0.32,
        "description": "Deep magazine & ultra-low $/shot in clear air, sharply degraded by heavy fog, rain, and dust backscatter.",
    },
}


def get_outer_layer_catalog() -> dict[str, dict[str, Any]]:
    """Return outer defense tier catalog for layered C-UAS simulations."""
    return OUTER_LAYER_CATALOG


def simulate_layered_cuas_defense_mc(
    outer_layer_id: str = "rf_gnss_jammer",
    outer_magazine: int | None = None,
    n_waves: int = 5,
    drones_per_wave: int = 5,
    wave_interval_s: float = 40.0,
    fiber_optic_fraction: float = 0.25,
    weather: str = "clear_day",
    inner_n_barrels: int = 6,
    inner_ammo_family: str = "12G_NO4_BUCK_C_UAS",
    inner_magazine_type: str = "box_10rd",
    target_range_m: float = 40.0,
    drone_model: str = "fpv_quad_small",
    n_trials: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired Monte Carlo comparison of Layered (Outer + Inner Turret) vs Inner Turret Only."""
    rng = np.random.default_rng(seed)
    outer_spec = OUTER_LAYER_CATALOG.get(outer_layer_id, OUTER_LAYER_CATALOG["rf_gnss_jammer"])
    outer_cap = int(outer_magazine if outer_magazine is not None else outer_spec["magazine_capacity"])

    econ_cat = get_cuas_ammo_economics_catalog()
    mag_cat = get_magazine_system_specs()
    canon_ammo = _resolve_econ_for_config(inner_ammo_family)
    ammo_spec = econ_cat[canon_ammo]
    mag_spec = mag_cat.get(inner_magazine_type, mag_cat["box_10rd"])

    drone_specs = get_drone_target_specs()
    drone_spec = drone_specs.get(drone_model, drone_specs["fpv_quad_small"])
    aspect_geom = compute_drone_projected_geometry(drone_spec, aspect_mode="head_on_frontal")

    barrels_per_burst = min(2 if inner_n_barrels >= 2 else 1, max(1, inner_n_barrels))
    base_inner_pk = _estimate_burst_pk_analytics(
        slant_range_m=target_range_m,
        barrels_fired=barrels_per_burst,
        pellets_per_shot=int(ammo_spec["pellets_per_shot"]),
        pellet_diameter_mm=float(ammo_spec["pellet_diameter_mm"]),
        pellet_mass_g=float(ammo_spec["pellet_mass_g"]),
        v0_mps=float(ammo_spec["v0_mps"]),
        choke_name="Modified",
        drone_spec=drone_spec,
        base_cd=float(ammo_spec["base_cd"]),
        aspect_mode="head_on_frontal",
        aspect_geom=aspect_geom,
    )

    weather_mult = float(outer_spec["weather_penalty_fog"]) if weather == "heavy_fog_rain" else 1.0
    c_steel = 0.49
    barrel_mass_kg = 1.15
    tau_cool_s = 180.0
    ambient_c = 22.0
    cap_per_bbl = int(mag_spec["capacity_per_barrel"])

    def _run_inner_eng(temp_c: float, mag_rem: list[int], b_cursor: int) -> tuple[bool, float, int, int]:
        rds = 0
        for _ in range(3):
            if any(mag_rem[(b_cursor + kb) % inner_n_barrels] <= 0 for kb in range(barrels_per_burst)):
                mag_rem[:] = [cap_per_bbl for _ in range(inner_n_barrels)]
            for kb in range(barrels_per_burst):
                bi = (b_cursor + kb) % inner_n_barrels
                mag_rem[bi] -= 1
                rds += 1
                temp_c += float(ammo_spec["heat_per_shot_kj"]) / max(barrel_mass_kg * c_steel * inner_n_barrels, 0.25)
            b_cursor = (b_cursor + barrels_per_burst) % inner_n_barrels
            walk_pen = math.exp(-0.0028 * max(0.0, temp_c - 120.0))
            eff_pk = max(0.02, min(0.99, base_inner_pk * walk_pen))
            if float(rng.random()) < eff_pk:
                return True, temp_c, b_cursor, rds
        return False, temp_c, b_cursor, rds

    wave_rows = [
        {"wave": w + 1, "outer_kills": 0.0, "inner_kills": 0.0, "leakers": 0.0, "baseline_leakers": 0.0}
        for w in range(n_waves)
    ]

    lay_kills_tot, lay_outer_kills_tot, lay_inner_kills_tot = 0, 0, 0
    lay_inner_rds_tot, lay_outer_shots_tot, lay_peak_temps = 0, 0, []
    base_kills_tot, base_inner_rds_tot, base_peak_temps = 0, 0, []

    total_threats = n_trials * n_waves * drones_per_wave

    for _ in range(n_trials):
        l_temp, b_temp = ambient_c, ambient_c
        l_peak, b_peak = ambient_c, ambient_c
        l_mag = [cap_per_bbl for _ in range(inner_n_barrels)]
        b_mag = [cap_per_bbl for _ in range(inner_n_barrels)]
        l_cur, b_cur = 0, 0
        outer_rem = outer_cap

        for w_idx in range(n_waves):
            for _d in range(drones_per_wave):
                is_fo = float(rng.random()) < float(fiber_optic_fraction)

                # Inner-only baseline always engages every drone
                b_killed, b_temp, b_cur, b_rds = _run_inner_eng(b_temp, b_mag, b_cur)
                b_peak = max(b_peak, b_temp)
                base_inner_rds_tot += b_rds
                if b_killed:
                    base_kills_tot += 1
                else:
                    wave_rows[w_idx]["baseline_leakers"] += 1.0

                # Layered engages outer first if magazine remains
                killed_outer = False
                if outer_rem > 0:
                    outer_rem -= 1
                    lay_outer_shots_tot += 1
                    p_outer = (
                        float(outer_spec["base_pk_fiber_optic"]) if is_fo else float(outer_spec["base_pk_rf"])
                    ) * weather_mult
                    if float(rng.random()) < p_outer:
                        killed_outer = True
                        lay_kills_tot += 1
                        lay_outer_kills_tot += 1
                        wave_rows[w_idx]["outer_kills"] += 1.0

                if not killed_outer:
                    l_killed, l_temp, l_cur, l_rds = _run_inner_eng(l_temp, l_mag, l_cur)
                    l_peak = max(l_peak, l_temp)
                    lay_inner_rds_tot += l_rds
                    if l_killed:
                        lay_kills_tot += 1
                        lay_inner_kills_tot += 1
                        wave_rows[w_idx]["inner_kills"] += 1.0
                    else:
                        wave_rows[w_idx]["leakers"] += 1.0

            if w_idx < n_waves - 1:
                cool_dt = max(wave_interval_s - 8.0, 1.0)
                decay = math.exp(-cool_dt / tau_cool_s)
                l_temp = ambient_c + (l_temp - ambient_c) * decay
                b_temp = ambient_c + (b_temp - ambient_c) * decay

        lay_peak_temps.append(l_peak)
        base_peak_temps.append(b_peak)

    for row in wave_rows:
        row["outer_kills"] = round(row["outer_kills"] / max(n_trials, 1), 2)
        row["inner_kills"] = round(row["inner_kills"] / max(n_trials, 1), 2)
        row["leakers"] = round(row["leakers"] / max(n_trials, 1), 2)
        row["baseline_leakers"] = round(row["baseline_leakers"] / max(n_trials, 1), 2)

    lay_survival = (lay_kills_tot / max(total_threats, 1)) * 100.0
    base_survival = (base_kills_tot / max(total_threats, 1)) * 100.0
    mean_lay_rds = lay_inner_rds_tot / max(n_trials, 1)
    mean_base_rds = base_inner_rds_tot / max(n_trials, 1)
    mean_lay_peak = float(np.mean(lay_peak_temps))
    mean_base_peak = float(np.mean(base_peak_temps))

    total_lay_cost = (lay_outer_shots_tot * float(outer_spec["cost_per_engagement_usd"])) + (
        lay_inner_rds_tot * float(ammo_spec["unit_cost_usd"])
    )
    blended_cpk = total_lay_cost / max(lay_kills_tot, 1)
    inner_only_cpk = (base_inner_rds_tot * float(ammo_spec["unit_cost_usd"])) / max(base_kills_tot, 1)

    return {
        "summary": {
            "outer_layer_id": outer_layer_id,
            "outer_layer_name": outer_spec["name"],
            "weather": weather,
            "fiber_optic_fraction_pct": round(fiber_optic_fraction * 100.0, 1),
            "layered_survival_pct": round(lay_survival, 1),
            "inner_only_survival_pct": round(base_survival, 1),
            "survival_gain_pct": round(lay_survival - base_survival, 1),
            "outer_kill_share_pct": round((lay_outer_kills_tot / max(lay_kills_tot, 1)) * 100.0, 1),
            "inner_kill_share_pct": round((lay_inner_kills_tot / max(lay_kills_tot, 1)) * 100.0, 1),
            "layered_inner_rounds": round(mean_lay_rds, 1),
            "inner_only_rounds": round(mean_base_rds, 1),
            "inner_rounds_saved": round(mean_base_rds - mean_lay_rds, 1),
            "layered_peak_temp_c": round(mean_lay_peak, 1),
            "inner_only_peak_temp_c": round(mean_base_peak, 1),
            "peak_temp_reduction_c": round(mean_base_peak - mean_lay_peak, 1),
            "blended_cost_per_kill_usd": round(blended_cpk, 2),
            "inner_only_cost_per_kill_usd": round(inner_only_cpk, 2),
        },
        "wave_timeline": wave_rows,
    }


# -------------------------------------------------------------------------
# 3. REAL-WORLD GIS SITE DEFENSE EXPORTERS (GEOJSON / KML / JSON)
# -------------------------------------------------------------------------

def local_xy_to_lonlat(
    x_m: float,
    y_m: float,
    origin_lat: float,
    origin_lon: float,
    heading_deg: float = 0.0,
) -> tuple[float, float]:
    """Convert local ENU-aligned (x=East, y=North) coordinates in meters to WGS84 (lon, lat)."""
    rad = math.radians(heading_deg)
    east_m = x_m * math.cos(rad) + y_m * math.sin(rad)
    north_m = -x_m * math.sin(rad) + y_m * math.cos(rad)
    lat = origin_lat + math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = max(abs(math.cos(math.radians(origin_lat))), 1e-4)
    lon = origin_lon + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return round(float(lon), 7), round(float(lat), 7)


def _circle_ring_lonlat(
    cx_m: float,
    cy_m: float,
    radius_m: float,
    origin_lat: float,
    origin_lon: float,
    heading_deg: float = 0.0,
    n_pts: int = 48,
) -> list[list[float]]:
    ring: list[list[float]] = []
    for k in range(n_pts):
        ang = 2.0 * math.pi * k / n_pts
        lo, la = local_xy_to_lonlat(cx_m + radius_m * math.cos(ang), cy_m + radius_m * math.sin(ang), origin_lat, origin_lon, heading_deg)
        ring.append([lo, la])
    ring.append(ring[0])
    return ring


def _sector_ring_lonlat(
    tx_m: float,
    ty_m: float,
    boresight_az_deg: float,
    sector_half_deg: float,
    max_range_m: float,
    origin_lat: float,
    origin_lon: float,
    heading_deg: float = 0.0,
    n_arc: int = 24,
) -> list[list[float]]:
    c_lo, c_la = local_xy_to_lonlat(tx_m, ty_m, origin_lat, origin_lon, heading_deg)
    ring: list[list[float]] = [[c_lo, c_la]]
    az0 = boresight_az_deg - sector_half_deg
    az1 = boresight_az_deg + sector_half_deg
    for az in np.linspace(az0, az1, n_arc):
        rad = math.radians(float(az))
        lo, la = local_xy_to_lonlat(tx_m + max_range_m * math.cos(rad), ty_m + max_range_m * math.sin(rad), origin_lat, origin_lon, heading_deg)
        ring.append([lo, la])
    ring.append([c_lo, c_la])
    return ring


def build_cuas_site_geojson(
    site_res: dict[str, Any],
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
    sdz_radius_m: float = 280.0,
    heading_deg: float = 0.0,
    shotfall_contour_xy_m: list[list[float]] | list[tuple[float, float]] | None = None,
    uxo_ellipse_xy_m: list[list[float]] | list[tuple[float, float]] | None = None,
    shotfall_label: str = "High-Elevation Shotfall 95% Dispersion",
) -> dict[str, Any]:
    """Build RFC 7946 GeoJSON FeatureCollection for multi-turret site defense, obstacles, NFZs, and SDZ."""
    sm = site_res["summary"]
    asset_r = float(sm.get("asset_radius_m", 40.0))
    w_range = float(sm.get("weapon_max_range_m", 60.0))
    features: list[dict[str, Any]] = []

    # 1. Extended Fallout SDZ
    features.append(
        {
            "type": "Feature",
            "properties": {
                "layer": "fallout_sdz",
                "name": f"Extended Pellet Fallout SDZ ({sdz_radius_m:.0f} m)",
                "radius_m": sdz_radius_m,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_circle_ring_lonlat(0.0, 0.0, asset_r + sdz_radius_m, origin_lat, origin_lon, heading_deg)],
            },
        }
    )

    # 2. Defended Asset Perimeter
    features.append(
        {
            "type": "Feature",
            "properties": {
                "layer": "defended_asset",
                "name": str(sm.get("site_name", "Defended Asset")),
                "asset_radius_m": asset_r,
                "perimeter_coverage_pct": sm.get("perimeter_coverage_pct", 0.0),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_circle_ring_lonlat(0.0, 0.0, asset_r, origin_lat, origin_lon, heading_deg)],
            },
        }
    )

    # 3. Turrets & Coverage Sectors
    for t in site_res.get("turrets", []):
        tx, ty = float(t["x_m"]), float(t["y_m"])
        lo, la = local_xy_to_lonlat(tx, ty, origin_lat, origin_lon, heading_deg)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "turret",
                    "id": str(t["id"]),
                    "height_m": float(t.get("z_m", 3.5)),
                    "boresight_az_deg": float(t.get("boresight_az_deg", 0.0)),
                    "n_barrels": int(t.get("n_barrels", 6)),
                },
                "geometry": {"type": "Point", "coordinates": [lo, la]},
            }
        )
        t_rng = float(t.get("max_range_m", w_range))
        features.append(
            {
                "type": "Feature",
                "properties": {"layer": "turret_sector", "turret_id": str(t["id"]), "range_m": t_rng},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _sector_ring_lonlat(
                            tx,
                            ty,
                            float(t.get("boresight_az_deg", 0.0)),
                            float(t.get("sector_half_deg", 145.0)),
                            t_rng,
                            origin_lat,
                            origin_lon,
                            heading_deg,
                        )
                    ],
                },
            }
        )

    # 4. 3D Building / Terrain Obstacles
    for obs in site_res.get("obstacles", []):
        ox, oy = float(obs["x_m"]), float(obs["y_m"])
        hw, hl = 0.5 * float(obs["width_m"]), 0.5 * float(obs["length_m"])
        corners_local = [(ox - hw, oy - hl), (ox + hw, oy - hl), (ox + hw, oy + hl), (ox - hw, oy + hl), (ox - hw, oy - hl)]
        ring = [list(local_xy_to_lonlat(cx, cy, origin_lat, origin_lon, heading_deg)) for cx, cy in corners_local]
        features.append(
            {
                "type": "Feature",
                "properties": {"layer": "obstacle", "name": str(obs["name"]), "height_m": float(obs["height_m"])},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    # 5. Friendly No-Fire Zones
    for nfz in site_res.get("nfz_zones", []):
        features.append(
            {
                "type": "Feature",
                "properties": {"layer": "nfz", "name": str(nfz["name"]), "radius_m": float(nfz["radius_m"])},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _circle_ring_lonlat(
                            float(nfz["x_m"]),
                            float(nfz["y_m"]),
                            float(nfz["radius_m"]),
                            origin_lat,
                            origin_lon,
                            heading_deg,
                        )
                    ],
                },
            }
        )

    # 6. Blind-Spot Wedges (active_turrets == 0)
    eval_r = asset_r + w_range
    for row in site_res.get("azimuth_profile", []):
        if int(row.get("active_turrets", 0)) == 0:
            az = float(row["az_deg"])
            features.append(
                {
                    "type": "Feature",
                    "properties": {"layer": "blind_spot", "az_deg": az},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            _sector_ring_lonlat(0.0, 0.0, az, 2.5, eval_r, origin_lat, origin_lon, heading_deg, n_arc=5)
                        ],
                    },
                }
            )

    # 7. Optional High-Elevation Wind-Sheared Shotfall 95% Dispersion & Post-Kill UXO Carry-On Ellipse
    if shotfall_contour_xy_m and len(shotfall_contour_xy_m) >= 3:
        sf_ring = [
            list(local_xy_to_lonlat(float(pt[0]), float(pt[1]), origin_lat, origin_lon, heading_deg))
            for pt in shotfall_contour_xy_m
        ]
        if sf_ring[0] != sf_ring[-1]:
            sf_ring.append(sf_ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "shotfall_95_dispersion",
                    "name": shotfall_label,
                    "scope_note": "Engineering scoping aid only; not official Range Safety certification.",
                },
                "geometry": {"type": "Polygon", "coordinates": [sf_ring]},
            }
        )

    if uxo_ellipse_xy_m and len(uxo_ellipse_xy_m) >= 3:
        uxo_ring = [
            list(local_xy_to_lonlat(float(pt[0]), float(pt[1]), origin_lat, origin_lon, heading_deg))
            for pt in uxo_ellipse_xy_m
        ]
        if uxo_ring[0] != uxo_ring[-1]:
            uxo_ring.append(uxo_ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "uxo_carryon_ellipse",
                    "name": "Post-Kill UXO & Ballistic Wreck Carry-On Ellipse (SE-C20)",
                    "scope_note": "Engineering scoping aid only; EOD standoff subject to Range Safety approval.",
                },
                "geometry": {"type": "Polygon", "coordinates": [uxo_ring]},
            }
        )

    return {
        "type": "FeatureCollection",
        "properties": {"origin_lat": origin_lat, "origin_lon": origin_lon, "heading_deg": heading_deg},
        "features": features,
    }


def export_cuas_site_kml(
    site_res: dict[str, Any],
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
    sdz_radius_m: float = 280.0,
    heading_deg: float = 0.0,
    name: str = "CUAS_Site_Defense",
) -> str:
    """Export multi-turret C-UAS layout, coverage sectors, 3D obstacles, NFZs & SDZ as KML 2.2."""
    geo = build_cuas_site_geojson(
        site_res=site_res,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        sdz_radius_m=sdz_radius_m,
        heading_deg=heading_deg,
    )
    esc_name = html.escape(str(name))
    placemarks: list[str] = []

    style_map = {
        "fallout_sdz": ("sdzStyle", "7f00a5ff", "4000a5ff", "0"),
        "defended_asset": ("assetStyle", "ccff8238", "35ff8238", "0"),
        "turret_sector": ("sectorStyle", "cc34a853", "2a34a853", "0"),
        "obstacle": ("obsStyle", "ff666666", "b0666666", "1"),
        "nfz": ("nfzStyle", "ff0000dc", "450000dc", "0"),
        "blind_spot": ("blindStyle", "ff0000ef", "550000ef", "0"),
    }

    for feat in geo["features"]:
        props = feat["properties"]
        layer = props.get("layer", "other")
        geom = feat["geometry"]
        if geom["type"] == "Point":
            lo, la = geom["coordinates"]
            z_alt = float(props.get("height_m", 3.5))
            t_id = html.escape(str(props.get("id", "Turret")))
            desc = html.escape(f"Barrels: {props.get('n_barrels', 6)}, Boresight: {props.get('boresight_az_deg', 0.0)}°")
            placemarks.append(
                f"""    <Placemark>
      <name>{t_id}</name>
      <description>{desc}</description>
      <Point>
        <altitudeMode>relativeToGround</altitudeMode>
        <coordinates>{lo},{la},{z_alt}</coordinates>
      </Point>
    </Placemark>"""
            )
        elif geom["type"] == "Polygon":
            style_id, _, _, extrude = style_map.get(layer, ("assetStyle", "ffffffff", "33ffffff", "0"))
            alt_z = float(props.get("height_m", 0.0)) if extrude == "1" else 0.0
            ring = geom["coordinates"][0]
            coord_str = " ".join(f"{pt[0]},{pt[1]},{alt_z}" for pt in ring)
            pm_name = html.escape(str(props.get("name", props.get("turret_id", layer))))
            alt_mode = "relativeToGround" if extrude == "1" else "clampToGround"
            placemarks.append(
                f"""    <Placemark>
      <name>{pm_name}</name>
      <styleUrl>#{style_id}</styleUrl>
      <Polygon>
        <extrude>{extrude}</extrude>
        <altitudeMode>{alt_mode}</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coord_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""
            )

    styles_xml = "\n".join(
        f"""    <Style id="{sid}">
      <LineStyle><color>{lcol}</color><width>2.2</width></LineStyle>
      <PolyStyle><color>{pcol}</color></PolyStyle>
    </Style>"""
        for sid, lcol, pcol, _ in style_map.values()
    )
    body_xml = "\n".join(placemarks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{esc_name}</name>
{styles_xml}
{body_xml}
  </Document>
</kml>
"""


def serialize_site_layout_json(layout_dict: dict[str, Any]) -> str:
    """Serialize custom multi-turret site defense layout to versioned JSON."""
    payload = {
        "schema": "cuas_site_v1",
        "origin_lat": float(layout_dict.get("origin_lat", 32.9)),
        "origin_lon": float(layout_dict.get("origin_lon", -106.4)),
        "heading_deg": float(layout_dict.get("heading_deg", 0.0)),
        "turrets": list(layout_dict.get("turrets", [])),
        "obstacles": list(layout_dict.get("obstacles", [])),
        "nfz_zones": list(layout_dict.get("nfz_zones", [])),
    }
    return json.dumps(payload, indent=2)


def validate_site_layout_json(raw_json: str) -> tuple[bool, dict[str, Any], str]:
    """Validate and sanitize uploaded custom C-UAS site layout JSON."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return False, {}, f"Invalid JSON: {exc}"
    if not isinstance(data, dict):
        return False, {}, "Root JSON element must be an object."
    if data.get("schema") != "cuas_site_v1":
        return False, {}, "Unsupported schema (expected 'cuas_site_v1')."
    turrets = data.get("turrets")
    if not isinstance(turrets, list) or len(turrets) == 0:
        return False, {}, "Site JSON must define at least one turret."

    clean_turrets = [
        {
            "id": str(t.get("id", f"T{i + 1}"))[:24],
            "x_m": float(np.clip(float(t.get("x_m", 0.0)), -500.0, 500.0)),
            "y_m": float(np.clip(float(t.get("y_m", 0.0)), -500.0, 500.0)),
            "z_m": float(np.clip(float(t.get("z_m", 3.5)), 0.5, 100.0)),
            "boresight_az_deg": float(float(t.get("boresight_az_deg", 0.0)) % 360.0),
            "sector_half_deg": float(np.clip(float(t.get("sector_half_deg", 140.0)), 15.0, 180.0)),
            "n_barrels": int(np.clip(int(t.get("n_barrels", 6)), 1, 16)),
        }
        for i, t in enumerate(turrets[:12])
        if isinstance(t, dict)
    ]
    if not clean_turrets:
        return False, {}, "No valid turret entries found."

    clean_obstacles = [
        {
            "name": str(o.get("name", f"Obstacle {i + 1}"))[:40],
            "x_m": float(np.clip(float(o.get("x_m", 0.0)), -500.0, 500.0)),
            "y_m": float(np.clip(float(o.get("y_m", 0.0)), -500.0, 500.0)),
            "width_m": float(np.clip(float(o.get("width_m", 10.0)), 1.0, 200.0)),
            "length_m": float(np.clip(float(o.get("length_m", 10.0)), 1.0, 200.0)),
            "height_m": float(np.clip(float(o.get("height_m", 12.0)), 1.0, 120.0)),
        }
        for i, o in enumerate(data.get("obstacles", [])[:20])
        if isinstance(o, dict)
    ]
    clean_nfz = [
        {
            "name": str(n.get("name", f"NFZ {i + 1}"))[:40],
            "x_m": float(np.clip(float(n.get("x_m", 60.0)), -1000.0, 1000.0)),
            "y_m": float(np.clip(float(n.get("y_m", 0.0)), -1000.0, 1000.0)),
            "radius_m": float(np.clip(float(n.get("radius_m", 25.0)), 5.0, 300.0)),
        }
        for i, n in enumerate(data.get("nfz_zones", [])[:20])
        if isinstance(n, dict)
    ]
    cleaned = {
        "origin_lat": float(np.clip(float(data.get("origin_lat", 32.9)), -85.0, 85.0)),
        "origin_lon": float(np.clip(float(data.get("origin_lon", -106.4)), -180.0, 180.0)),
        "heading_deg": float(float(data.get("heading_deg", 0.0)) % 360.0),
        "turrets": clean_turrets,
        "obstacles": clean_obstacles,
        "nfz_zones": clean_nfz,
    }
    return True, cleaned, f"Loaded {len(clean_turrets)} turrets, {len(clean_obstacles)} obstacles, {len(clean_nfz)} NFZs."


# -------------------------------------------------------------------------
# 4. AUTOMATED SENSITIVITY & TORNADO BOTTLENECK ANALYZERS
# -------------------------------------------------------------------------

def analyze_cuas_sensitivity_tornado(
    baseline_cfg: dict[str, Any] | None = None,
    perturbation_pct: float = 20.0,
    n_trials: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    """One-at-a-time ±perturbation Tornado sensitivity sweep over C-UAS Kill Chain."""
    cfg = dict(baseline_cfg or {})
    base_kwargs: dict[str, Any] = {
        "drone_model": str(cfg.get("drone_model", "fpv_quad_small")),
        "aspect_mode": str(cfg.get("aspect_mode", "head_on_frontal")),
        "drone_speed_mps": float(cfg.get("drone_speed_mps", 28.0)),
        "approach_altitude_m": float(cfg.get("approach_altitude_m", 25.0)),
        "weather": str(cfg.get("weather", "clear_day")),
        "ambient_wind_mps": float(cfg.get("ambient_wind_mps", 5.0)),
        "fiber_optic_autonomous": bool(cfg.get("fiber_optic_autonomous", False)),
        "ai_classify_s": float(cfg.get("ai_classify_s", 0.6)),
        "human_in_the_loop": bool(cfg.get("human_in_the_loop", True)),
        "human_auth_s": float(cfg.get("human_auth_s", 1.2)),
        "slew_angle_deg": float(cfg.get("slew_angle_deg", 60.0)),
        "turret_slew_rate_deg_s": float(cfg.get("turret_slew_rate_deg_s", 120.0)),
        "max_weapon_range_m": float(cfg.get("max_weapon_range_m", 55.0)),
        "keepout_breach_radius_m": float(cfg.get("keepout_breach_radius_m", 15.0)),
        "n_barrels": int(cfg.get("n_barrels", 6)),
        "ammunition_preset": _resolve_shotgun_for_config(str(cfg.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))),
        "choke": str(cfg.get("choke", "Modified")),
        "salvo_cycle_time_s": float(cfg.get("salvo_cycle_time_s", 0.85)),
        "n_trials": int(n_trials),
        "seed": int(seed),
    }
    base_res = simulate_cuas_kill_chain_mc(**base_kwargs)
    pk_base = float(base_res["summary"]["cumulative_pk_pct"])

    frac = max(0.05, min(0.50, float(perturbation_pct) / 100.0))

    specs: list[tuple[str, str, float, float, float, bool, str]] = [
        ("max_weapon_range_m", "Max Weapon Engagement Range (m)", 25.0, 95.0, False, "Extend ballistic pellet energy / tungsten range to open the firing window earlier."),
        ("drone_speed_mps", "Threat Approach Speed (m/s)", 8.0, 55.0, False, "Higher threat speed compresses engagement salvos; add outer-layer screening or faster ripple cadence."),
        ("salvo_cycle_time_s", "Salvo Cycle Cadence Δt (s)", 0.25, 2.0, False, "Switch from single-shot reload cadence to balanced-pair ripple barrels to pack more bursts before breach."),
        ("n_barrels", "Active Turret Barrels (N)", 2.0, 16.0, True, "Add barrel count or concentrated core loadouts to boost single-burst P_kill against frontal drones."),
        ("ai_classify_s", "AI Classify + C2 Latency (s)", 0.1, 3.0, False, "Automate track fusion & pre-authorize C2 fire control to reduce sensor-to-trigger dead time."),
        ("turret_slew_rate_deg_s", "Turret Gimbal Slew Rate (°/s)", 30.0, 240.0, True, "Upgrade servo slew torque or pre-cue gimbal from passive RF/radar sectors."),
        ("keepout_breach_radius_m", "Keep-Out Breach Radius (m)", 5.0, 35.0, False, "Tighten perimeter exclusion or reposition turrets outward to recover close-in engagement depth."),
        ("ambient_wind_mps", "Ambient Crosswind (m/s)", 0.5, 22.0, False, "Higher-BC Tungsten TSS / flechette payloads cut wind drift and acoustic sensor SNR loss."),
    ]

    rows: list[dict[str, Any]] = []
    for key, label, lo_bound, hi_bound, is_int, rec in specs:
        v_base = float(base_kwargs[key])
        v_low = max(lo_bound, min(hi_bound, v_base * (1.0 - frac)))
        v_high = max(lo_bound, min(hi_bound, v_base * (1.0 + frac)))
        if is_int:
            v_low_c: Any = max(int(lo_bound), int(round(v_low)))
            v_high_c: Any = min(int(hi_bound), int(round(v_high)))
            if v_low_c == int(v_base) and v_low_c > int(lo_bound):
                v_low_c -= 1
            if v_high_c == int(v_base) and v_high_c < int(hi_bound):
                v_high_c += 1
        else:
            v_low_c, v_high_c = round(v_low, 3), round(v_high, 3)

        kw_low = dict(base_kwargs)
        kw_low[key] = v_low_c
        pk_low = float(simulate_cuas_kill_chain_mc(**kw_low)["summary"]["cumulative_pk_pct"])

        kw_high = dict(base_kwargs)
        kw_high[key] = v_high_c
        pk_high = float(simulate_cuas_kill_chain_mc(**kw_high)["summary"]["cumulative_pk_pct"])

        d_low = round(pk_low - pk_base, 1)
        d_high = round(pk_high - pk_base, 1)
        swing = round(max(abs(d_low), abs(d_high), abs(pk_high - pk_low)), 1)

        rows.append(
            {
                "parameter": key,
                "label": label,
                "low_val": v_low_c,
                "base_val": round(v_base, 3) if not is_int else int(v_base),
                "high_val": v_high_c,
                "val_at_low": round(pk_low, 1),
                "val_at_base": round(pk_base, 1),
                "val_at_high": round(pk_high, 1),
                "delta_low": d_low,
                "delta_high": d_high,
                "swing": swing,
                "recommendation": rec,
            }
        )

    rows.sort(key=lambda r: r["swing"], reverse=True)
    return {
        "summary": {
            "baseline_value": round(pk_base, 1),
            "metric_name": "Kill-Chain P_kill (%)",
            "perturbation_pct": float(perturbation_pct),
            "top_driver": rows[0]["label"],
            "top_swing": rows[0]["swing"],
            "second_driver": rows[1]["label"],
        },
        "top_bottlenecks": [rows[0], rows[1]],
        "rows": rows,
    }


def analyze_artillery_sensitivity_tornado(
    ammo: AmmoDefinition,
    scenario: Scenario,
    perturbation_pct: float = 20.0,
    solved_qe_rad: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fast vectorized finite-difference ±% Tornado sweep over artillery/munition CEP50."""
    del seed  # deterministic analytical-batch unit-effect evaluation
    v0 = max(float(ammo.launch.mv.mu), 30.0)
    ff_mu = max(float(ammo.aero.form_factor.mu), 0.2)
    aim_r = max(float(scenario.aim.range_m or 0.0), 100.0)

    if ammo.launch.mode in ("static", "bomb", "guided_miss") or scenario.fidelity == "L0":
        # Analytic linearized sensitivity proxy for static / L0 modes
        r0 = aim_r
        dR_dV0 = 0.02 * r0 / max(v0, 30.0)
        dR_dQE_mil = 0.002 * r0
        dR_dWx = 0.008 * r0
        dD_dWz = 0.005 * r0
        dR_dFF_frac = 0.35 * r0
    else:
        env = AtmosphereEnvironment(
            dT_K=float(scenario.met.deviations.dT_K),
            constant_wind_mps=(
                float(scenario.met.wind_range_mps),
                float(scenario.met.wind_cross_right_mps),
                0.0,
            ),
        )
        drag_ref = str(ammo.aero.drag_ref or "G7")
        traj_cfg = BatchTrajectoryConfig(
            calibre_m=max(float(ammo.calibre_m), 0.004),
            drag_ref=drag_ref,
            custom_cd_table=ammo.aero.cd_table,
            fidelity="L1",
            rocket_burn_time_s=float(ammo.launch.rocket_burn_s),
            rocket_mdot_kg_s=float(ammo.launch.rocket_mdot_kg_s),
            rocket_ve_mps=float(ammo.launch.rocket_ve_mps),
            base_bleed_burn_time_s=float(ammo.launch.base_bleed_burn_s),
            base_bleed_dcd=float(ammo.launch.base_bleed_dcd),
            dt=0.06,
            include_coriolis=bool(scenario.include_coriolis),
            vacuum=bool(scenario.vacuum),
        )
        z0 = float(scenario.weapon_location.alt)

        if solved_qe_rad is None:
            qe_grid = (
                np.linspace(math.radians(47.0), math.radians(78.0), 12)
                if scenario.aim.high_angle
                else np.linspace(math.radians(5.0), math.radians(45.0), 12)
            )
            states_sw = np.column_stack(
                [
                    np.zeros(12),
                    np.zeros(12),
                    np.full(12, z0),
                    v0 * np.cos(qe_grid),
                    np.zeros(12),
                    v0 * np.sin(qe_grid),
                ]
            )
            sw_res = integrate_trajectories_batch(
                initial_states=states_sw,
                masses_kg=np.full(12, float(ammo.mass_kg)),
                form_factors=np.full(12, ff_mu),
                env=env,
                cfg=traj_cfg,
            )
            r_sw = np.asarray(sw_res["range_m"], dtype=np.float64)
            order = np.argsort(r_sw)
            r_sorted, q_sorted = r_sw[order], qe_grid[order]
            r_tgt = float(np.clip(aim_r, float(r_sorted[0]), float(r_sorted[-1])))
            qe_nom = float(np.interp(r_tgt, r_sorted, q_sorted))
        else:
            qe_nom = float(solved_qe_rad)

        d_mil = 2.0 * math.pi / 6400.0
        q_arr = np.array([qe_nom, qe_nom, qe_nom + d_mil, qe_nom, qe_nom, qe_nom], dtype=np.float64)
        v_arr = np.array([v0, v0 + 1.0, v0, v0, v0, v0], dtype=np.float64)
        ff_arr = np.array([ff_mu, ff_mu, ff_mu, ff_mu, ff_mu, ff_mu * 1.05], dtype=np.float64)
        w_off = np.zeros((6, 3), dtype=np.float64)
        w_off[3, 0] = 1.0
        w_off[4, 1] = -1.0  # +1 m/s crosswind right (y_left = -y_right)

        states_fd = np.column_stack(
            [
                np.zeros(6),
                np.zeros(6),
                np.full(6, z0),
                v_arr * np.cos(q_arr),
                np.zeros(6),
                v_arr * np.sin(q_arr),
            ]
        )
        fd_res = integrate_trajectories_batch(
            initial_states=states_fd,
            masses_kg=np.full(6, float(ammo.mass_kg)),
            form_factors=ff_arr,
            env=env,
            cfg=traj_cfg,
            trial_wind_offset=w_off,
        )
        r_vec = np.asarray(fd_res["range_m"], dtype=np.float64)
        d_vec = np.asarray(fd_res["deflection_m"], dtype=np.float64)

        r0 = max(float(r_vec[0]), 50.0)
        dR_dV0 = abs(float(r_vec[1] - r_vec[0]))
        dR_dQE_mil = abs(float(r_vec[2] - r_vec[0]))
        dR_dWx = abs(float(r_vec[3] - r_vec[0]))
        dD_dWz = abs(float(d_vec[4] - d_vec[0]))
        dR_dFF_frac = abs(float(r_vec[5] - r_vec[0])) / 0.05

    base_state: dict[str, float] = {
        "tle_sigma_m": float(scenario.errors.bias.tle_range_sigma_m),
        "mv_sigma_lot": float(ammo.launch.mv.sigma_lot),
        "mv_sigma_rr": float(ammo.launch.mv.sigma_rr),
        "aim_qe_bias_mils": float(scenario.errors.bias.aim_qe_bias_sigma_mils),
        "aim_qe_rand_mils": float(scenario.errors.random.aim_qe_rand_sigma_mils),
        "wind_bias_mps": float(scenario.errors.bias.wind_range_bias_sigma_mps),
        "form_factor_mu": ff_mu,
    }
    az_sigma_rad = math.hypot(
        float(scenario.errors.bias.aim_az_bias_sigma_mils),
        float(scenario.errors.random.aim_az_rand_sigma_mils),
    ) * (2.0 * math.pi / 6400.0)
    ff_sigma_base = max(float(getattr(ammo.aero.form_factor, "sigma_rr", 0.02) or 0.02), 0.015)

    def _compute_cep50(st_map: dict[str, float]) -> float:
        ff_scale = max(st_map["form_factor_mu"] / ff_mu, 0.25)
        sig_r = math.sqrt(
            st_map["tle_sigma_m"] ** 2
            + (dR_dV0 * math.hypot(st_map["mv_sigma_rr"], st_map["mv_sigma_lot"])) ** 2
            + (dR_dQE_mil * math.hypot(st_map["aim_qe_bias_mils"], st_map["aim_qe_rand_mils"])) ** 2
            + (dR_dWx * st_map["wind_bias_mps"] * ff_scale) ** 2
            + (dR_dFF_frac * ff_sigma_base * ff_scale) ** 2
        )
        sig_d = math.sqrt(
            st_map["tle_sigma_m"] ** 2
            + (r0 * az_sigma_rad) ** 2
            + (dD_dWz * st_map["wind_bias_mps"] * ff_scale) ** 2
        )
        return 0.5887 * (sig_r + sig_d)

    base_cep = _compute_cep50(base_state)
    frac = max(0.05, min(0.50, float(perturbation_pct) / 100.0))

    param_specs = [
        ("tle_sigma_m", "Target Location Error TLE σ (m)", 5.0, "Improve FO/UAS target mensuration & coordinate accuracy."),
        ("mv_sigma_lot", "Occasion/Lot MV Sigma (m/s)", 1.0, "Measure propellant temperature conditioning & chronograph MV calibration."),
        ("mv_sigma_rr", "Round-to-Round MV Sigma (m/s)", 1.0, "Tighten projectile weight-zone sorting & primer/charge lot uniformity."),
        ("aim_qe_bias_mils", "QE Lay Bias Sigma (mils)", 0.5, "Calibrate gun trunnion level & digital inertial laying unit."),
        ("aim_qe_rand_mils", "QE Round-to-Round Jitter (mils)", 0.5, "Inspect cradle/trunnion play & recoil repeatability."),
        ("wind_bias_mps", "Met Wind Error Sigma (m/s)", 1.5, "Update STANAG 4082 METCM upper-air sounding frequency."),
        ("form_factor_mu", "Ballistic Form Factor (i)", 0.05, "Low-drag base-bleed / precision-machined ogive lowers downrange dispersion."),
    ]

    rows: list[dict[str, Any]] = []
    for key, label, floor_step, rec in param_specs:
        v_base = base_state[key]
        if abs(v_base) < 1e-6:
            v_low, v_high = 0.0, floor_step
        else:
            v_low = max(0.0 if key != "form_factor_mu" else 0.4, v_base * (1.0 - frac))
            v_high = v_base * (1.0 + frac)

        s_lo = dict(base_state)
        s_lo[key] = v_low
        cep_low = _compute_cep50(s_lo)

        s_hi = dict(base_state)
        s_hi[key] = v_high
        cep_high = _compute_cep50(s_hi)

        d_low = round(cep_low - base_cep, 2)
        d_high = round(cep_high - base_cep, 2)
        swing = round(max(abs(d_low), abs(d_high), abs(cep_high - cep_low)), 2)

        rows.append(
            {
                "parameter": key,
                "label": label,
                "low_val": round(v_low, 3),
                "base_val": round(v_base, 3),
                "high_val": round(v_high, 3),
                "val_at_low": round(cep_low, 2),
                "val_at_base": round(base_cep, 2),
                "val_at_high": round(cep_high, 2),
                "delta_low": d_low,
                "delta_high": d_high,
                "swing": swing,
                "recommendation": rec,
            }
        )

    rows.sort(key=lambda r: r["swing"], reverse=True)
    return {
        "summary": {
            "ammo_id": ammo.id,
            "baseline_value": round(base_cep, 2),
            "metric_name": "CEP 50% (m)",
            "perturbation_pct": float(perturbation_pct),
            "top_driver": rows[0]["label"],
            "top_swing": rows[0]["swing"],
            "second_driver": rows[1]["label"],
        },
        "top_bottlenecks": [rows[0], rows[1]],
        "rows": rows,
    }
