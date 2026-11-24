"""Multi-Turret & Heterogeneous Swarm 3D Battlespace Co-Evolution Engine.

Co-evolves Blue cooperative multi-turret placements & outer layered defense pairings
against heterogeneous Red attack packages (decoy sponges, trailing heavy bombers,
fiber-optic pincers, ridge-masked corridors) over 3D DEM terrain, solving exact
zero-sum Nash equilibria and emitting a 1-click handoff payload for 3D Tactical Raid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.cuas_adversarial import solve_zero_sum_nash_equilibrium
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.terrain_dem import build_preset_terrain_dem

BLUE_BATTLESPACE_GEOMETRIES: tuple[str, ...] = (
    "perimeter_symmetric",
    "ridge_offset",
    "asymmetric_corridor_guard",
    "clustered_inner_citadel",
)

OUTER_LAYER_OPTIONS: tuple[str, ...] = (
    "none",
    "rf_gnss_jammer",
    "hpm_screen",
    "airburst_35mm",
)

RED_MIXED_PACKAGES: dict[str, dict[str, Any]] = {
    "decoy_sponge_bomber": {
        "name": "Decoy FPV Sponge + Trailing Heavy Bombers",
        "evasion_profile": "jinking_weave",
        "packages": [
            {
                "role": "decoy_sponge",
                "drone_model": "fpv_quad_small",
                "count": 5,
                "speed_mps": 32.0,
                "altitude_m": 20.0,
                "maneuver_g": 2.6,
                "fiber_optic_fraction": 0.2,
                "spawn_delay_s": 0.0,
                "az_center_deg": 45.0,
                "az_spread_deg": 40.0,
            },
            {
                "role": "heavy_bomber",
                "drone_model": "heavy_octocopter_large",
                "count": 3,
                "speed_mps": 17.0,
                "altitude_m": 28.0,
                "maneuver_g": 1.1,
                "fiber_optic_fraction": 0.67,
                "spawn_delay_s": 3.5,
                "az_center_deg": 45.0,
                "az_spread_deg": 20.0,
            },
        ],
    },
    "fo_pincer_saturation": {
        "name": "Dual-Axis Fiber-Optic Pincer Saturation",
        "evasion_profile": "pop_up_terminal",
        "packages": [
            {
                "role": "north_pincer_fo",
                "drone_model": "fpv_quad_small",
                "count": 4,
                "speed_mps": 28.0,
                "altitude_m": 22.0,
                "maneuver_g": 3.0,
                "fiber_optic_fraction": 1.0,
                "spawn_delay_s": 0.0,
                "az_center_deg": 45.0,
                "az_spread_deg": 18.0,
            },
            {
                "role": "south_pincer_fo",
                "drone_model": "commercial_quad_medium",
                "count": 4,
                "speed_mps": 24.0,
                "altitude_m": 24.0,
                "maneuver_g": 2.2,
                "fiber_optic_fraction": 1.0,
                "spawn_delay_s": 1.2,
                "az_center_deg": 225.0,
                "az_spread_deg": 18.0,
            },
        ],
    },
    "ridge_masked_corridor": {
        "name": "NE Ridge-Masked Corridor Dive + Fixed-Wing",
        "evasion_profile": "pop_up_terminal",
        "packages": [
            {
                "role": "ridge_skimmer",
                "drone_model": "fpv_quad_small",
                "count": 5,
                "speed_mps": 30.0,
                "altitude_m": 16.0,
                "maneuver_g": 2.8,
                "fiber_optic_fraction": 0.6,
                "spawn_delay_s": 0.0,
                "az_center_deg": 42.0,
                "az_spread_deg": 14.0,
            },
            {
                "role": "diver_wing",
                "drone_model": "fixed_wing_loitering",
                "count": 3,
                "speed_mps": 38.0,
                "altitude_m": 32.0,
                "maneuver_g": 2.4,
                "fiber_optic_fraction": 0.33,
                "spawn_delay_s": 2.5,
                "az_center_deg": 48.0,
                "az_spread_deg": 16.0,
            },
        ],
    },
    "omni_high_speed_mixed": {
        "name": "360-Deg Omni Mixed Wave (RF + Fiber-Optic)",
        "evasion_profile": "jinking_weave",
        "packages": [
            {
                "role": "omni_rf_vanguard",
                "drone_model": "fpv_quad_small",
                "count": 6,
                "speed_mps": 27.0,
                "altitude_m": 22.0,
                "maneuver_g": 2.4,
                "fiber_optic_fraction": 0.0,
                "spawn_delay_s": 0.0,
                "az_center_deg": None,
                "az_spread_deg": 180.0,
            },
            {
                "role": "omni_fo_strikers",
                "drone_model": "commercial_quad_medium",
                "count": 4,
                "speed_mps": 22.0,
                "altitude_m": 26.0,
                "maneuver_g": 2.0,
                "fiber_optic_fraction": 1.0,
                "spawn_delay_s": 2.0,
                "az_center_deg": None,
                "az_spread_deg": 180.0,
            },
        ],
    },
}


def build_turret_ring_layout(
    geometry_type: str = "perimeter_symmetric",
    n_turrets: int = 4,
    asset_radius_m: float = 38.0,
    n_barrels: int = 6,
    weapon_max_range_m: float = 65.0,
    focus_az_deg: float = 45.0,
) -> list[dict[str, Any]]:
    """Generate parametric multi-turret 3D site coordinates and sector assignments."""
    nt = max(2, min(8, int(n_turrets)))
    r_base = float(asset_radius_m)
    turrets: list[dict[str, Any]] = []

    if geometry_type == "clustered_inner_citadel":
        rad = max(14.0, r_base * 0.58)
        for i in range(nt):
            az = (focus_az_deg + i * (360.0 / nt)) % 360.0
            a_rad = math.radians(az)
            turrets.append(
                {
                    "id": f"T-{i+1}",
                    "x_m": round(rad * math.cos(a_rad), 1),
                    "y_m": round(rad * math.sin(a_rad), 1),
                    "z_m": 4.2,
                    "boresight_az_deg": round(az, 1),
                    "sector_half_deg": 145.0,
                    "n_barrels": int(n_barrels),
                    "max_range_m": float(weapon_max_range_m),
                }
            )
    elif geometry_type == "asymmetric_corridor_guard":
        # Heavy forward pair oriented toward focus_az_deg, rear pair covering flanks
        for i in range(nt):
            if i < max(2, nt // 2):
                offset = -32.0 + i * (64.0 / max(1, max(2, nt // 2) - 1))
                az = (focus_az_deg + offset) % 360.0
                rad = r_base * 0.92
                sec = 110.0
            else:
                az = (focus_az_deg + 140.0 + (i - nt // 2) * 80.0) % 360.0
                rad = r_base * 0.72
                sec = 135.0
            a_rad = math.radians(az)
            turrets.append(
                {
                    "id": f"T-{i+1}",
                    "x_m": round(rad * math.cos(a_rad), 1),
                    "y_m": round(rad * math.sin(a_rad), 1),
                    "z_m": 4.0,
                    "boresight_az_deg": round(az, 1),
                    "sector_half_deg": sec,
                    "n_barrels": int(n_barrels),
                    "max_range_m": float(weapon_max_range_m),
                }
            )
    elif geometry_type == "ridge_offset":
        # Shift turrets closer to perimeter on high ground overlooking masked draw
        for i in range(nt):
            az = (focus_az_deg + 15.0 + i * (360.0 / nt)) % 360.0
            a_rad = math.radians(az)
            rad = r_base * (1.05 if (i % 2 == 0) else 0.76)
            turrets.append(
                {
                    "id": f"T-{i+1}",
                    "x_m": round(rad * math.cos(a_rad), 1),
                    "y_m": round(rad * math.sin(a_rad), 1),
                    "z_m": 5.5,
                    "boresight_az_deg": round(az, 1),
                    "sector_half_deg": 135.0,
                    "n_barrels": int(n_barrels),
                    "max_range_m": float(weapon_max_range_m),
                }
            )
    else:
        for i in range(nt):
            az = (i * (360.0 / nt) + 22.5) % 360.0
            a_rad = math.radians(az)
            turrets.append(
                {
                    "id": f"T-{i+1}",
                    "x_m": round(r_base * 0.90 * math.cos(a_rad), 1),
                    "y_m": round(r_base * 0.90 * math.sin(a_rad), 1),
                    "z_m": 3.8,
                    "boresight_az_deg": round(az, 1),
                    "sector_half_deg": 130.0,
                    "n_barrels": int(n_barrels),
                    "max_range_m": float(weapon_max_range_m),
                }
            )
    return turrets


def _format_bs_blue_name(cfg: dict[str, Any]) -> str:
    geom_s = cfg["geometry_type"][:9]
    ol_s = cfg["outer_layer_type"].replace("_", "")[:8]
    am_s = {"12G_NO4_BUCK_C_UAS": "#4B", "12G_TSS_NO2_C_UAS": "TSS", "12G_DUPLEX_TSS_BUCK_C_UAS": "Dup"}.get(
        cfg["ammunition_preset"], "Ammo"
    )
    return f"{cfg['n_turrets']}T {geom_s} {am_s}+{ol_s}"


def run_battlespace_coevolution_mc(
    terrain_preset: str = "rolling_hills_fob",
    n_generations: int = 3,
    n_mc_trials: int = 6,
    max_turrets: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Co-evolve multi-turret 3D battlespace architectures against heterogeneous Red swarm packages."""
    rng = np.random.default_rng(seed)
    terrain_dem = None if terrain_preset == "flat_airfield" else build_preset_terrain_dem(terrain_preset)
    max_t = max(2, min(6, int(max_turrets)))

    def _normalize_blue_bs(raw: dict[str, Any]) -> dict[str, Any]:
        geom_t = str(raw.get("geometry_type", "perimeter_symmetric"))
        if geom_t not in BLUE_BATTLESPACE_GEOMETRIES:
            geom_t = "perimeter_symmetric"
        nt = max(2, min(max_t, int(raw.get("n_turrets", min(4, max_t)))))
        ol = str(raw.get("outer_layer_type", "none"))
        if ol not in OUTER_LAYER_OPTIONS:
            ol = "none"
        ammo_p = str(raw.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"))
        rmax = float(max(45.0, min(85.0, float(raw.get("weapon_max_range_m", 65.0)))))
        focus_az = float(raw.get("focus_az_deg", 45.0)) % 360.0
        n_bbl = int(raw.get("n_barrels", 6))
        turrets = build_turret_ring_layout(
            geometry_type=geom_t,
            n_turrets=nt,
            asset_radius_m=36.0,
            n_barrels=n_bbl,
            weapon_max_range_m=rmax,
            focus_az_deg=focus_az,
        )
        cfg = {
            "geometry_type": geom_t,
            "n_turrets": nt,
            "n_barrels": n_bbl,
            "ammunition_preset": ammo_p,
            "outer_layer_type": ol,
            "weapon_max_range_m": round(rmax, 1),
            "focus_az_deg": round(focus_az, 1),
            "turrets": turrets,
        }
        cfg["name"] = str(raw.get("name") or _format_bs_blue_name(cfg))
        return cfg

    blue_pool: list[dict[str, Any]] = [
        _normalize_blue_bs(
            {
                "name": "Symmetric Perimeter #4 Buck",
                "geometry_type": "perimeter_symmetric",
                "n_turrets": min(max_t, 4),
                "ammunition_preset": "12G_NO4_BUCK_C_UAS",
                "outer_layer_type": "none",
                "weapon_max_range_m": 60.0,
            }
        ),
        _normalize_blue_bs(
            {
                "name": "Ridge Offset + RF Jammer",
                "geometry_type": "ridge_offset",
                "n_turrets": min(max_t, 4),
                "ammunition_preset": "12G_TSS_NO2_C_UAS",
                "outer_layer_type": "rf_gnss_jammer",
                "weapon_max_range_m": 68.0,
                "focus_az_deg": 45.0,
            }
        ),
        _normalize_blue_bs(
            {
                "name": "Corridor Guard + 35mm Airburst",
                "geometry_type": "asymmetric_corridor_guard",
                "n_turrets": min(max_t, 4),
                "ammunition_preset": "12G_DUPLEX_TSS_BUCK_C_UAS",
                "outer_layer_type": "airburst_35mm",
                "weapon_max_range_m": 66.0,
                "focus_az_deg": 45.0,
            }
        ),
        _normalize_blue_bs(
            {
                "name": "Inner Citadel + HPM Screen",
                "geometry_type": "clustered_inner_citadel",
                "n_turrets": min(max_t, 4),
                "ammunition_preset": "12G_TSS_NO2_C_UAS",
                "outer_layer_type": "hpm_screen",
                "weapon_max_range_m": 72.0,
                "focus_az_deg": 45.0,
            }
        ),
    ]

    red_pool: list[dict[str, Any]] = []
    for key, spec in RED_MIXED_PACKAGES.items():
        pkgs = [dict(p) for p in spec["packages"]]
        tot_n = sum(int(p["count"]) for p in pkgs)
        red_pool.append(
            {
                "package_id": key,
                "name": spec["name"],
                "evasion_profile": spec["evasion_profile"],
                "swarm_package": pkgs,
                "total_drones": tot_n,
                "mean_speed_mps": round(sum(p["speed_mps"] * p["count"] for p in pkgs) / max(1, tot_n), 1),
            }
        )

    eval_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def _eval_match(bi: int, rj: int) -> dict[str, Any]:
        if (bi, rj) in eval_cache:
            return eval_cache[(bi, rj)]
        b = blue_pool[bi]
        r = red_pool[rj]
        raid = simulate_unified_tactical_raid_mc(
            site_preset="fob_compound_4_turret",
            custom_turrets=b["turrets"],
            terrain_dem=terrain_dem,
            weather_condition="clear_day",
            evasion_profile=r["evasion_profile"],
            ammunition_preset=b["ammunition_preset"],
            weapon_max_range_m=float(b["weapon_max_range_m"]),
            outer_layer_type=b["outer_layer_type"],
            swarm_package=r["swarm_package"],
            n_mc_trials=max(3, int(n_mc_trials)),
            seed=seed + bi * 53 + rj * 19,
        )
        sm = raid["summary"]
        outer_cost_pen = {
            "none": 0.0,
            "rf_gnss_jammer": 1.8,
            "hpm_screen": 3.0,
            "airburst_35mm": 4.2,
        }.get(b["outer_layer_type"], 0.0)
        util = float(
            max(
                0.0,
                min(
                    100.0,
                    0.55 * float(sm["zero_breach_pct"])
                    + 0.45 * float(sm["mean_kill_rate_pct"])
                    - 0.25 * float(sm["cost_per_kill_usd"])
                    - outer_cost_pen
                    - 0.8 * max(0, int(b["n_turrets"]) - 3),
                ),
            )
        )
        out = {
            "blue_utility": round(util, 2),
            "zero_breach_pct": sm["zero_breach_pct"],
            "mean_kill_rate_pct": sm["mean_kill_rate_pct"],
            "outer_layer_kills": sm.get("outer_layer_kills", 0.0),
            "cost_per_kill_usd": sm["cost_per_kill_usd"],
        }
        eval_cache[(bi, rj)] = out
        return out

    # Gen 0 matrix
    for bi in range(len(blue_pool)):
        for rj in range(len(red_pool)):
            _eval_match(bi, rj)

    def _matrix() -> np.ndarray:
        M = np.zeros((len(blue_pool), len(red_pool)), dtype=np.float64)
        for bi in range(len(blue_pool)):
            for rj in range(len(red_pool)):
                M[bi, rj] = eval_cache[(bi, rj)]["blue_utility"]
        return M

    M0 = _matrix()
    nash0 = solve_zero_sum_nash_equilibrium(M0)
    base_worst = float(np.max(np.min(M0, axis=1)))
    gen_history: list[dict[str, Any]] = [
        {
            "generation": 0,
            "nash_value": nash0["nash_value"],
            "best_worst_case": round(base_worst, 2),
            "n_blue": len(blue_pool),
            "n_red": len(red_pool),
            "new_blue": "Seed Multi-Turret Sites",
            "new_red": "Seed Mixed Packages",
        }
    ]

    for gen in range(1, max(1, int(n_generations)) + 1):
        M_cur = _matrix()
        nash_cur = solve_zero_sum_nash_equilibrium(M_cur)
        best_bi = int(nash_cur["pure_minimax_blue_idx"])
        parent_b = blue_pool[best_bi]

        # Generate 3 mutated Blue battlespace variations & pick best worst-case
        cand_blues = []
        for _ in range(3):
            raw_b = {
                "geometry_type": str(rng.choice(BLUE_BATTLESPACE_GEOMETRIES)),
                "n_turrets": max(2, min(max_t, int(parent_b["n_turrets"]) + int(rng.choice([-1, 0, 1])))),
                "ammunition_preset": str(rng.choice(["12G_NO4_BUCK_C_UAS", "12G_TSS_NO2_C_UAS", "12G_DUPLEX_TSS_BUCK_C_UAS"])),
                "outer_layer_type": str(rng.choice(OUTER_LAYER_OPTIONS)),
                "weapon_max_range_m": float(np.clip(parent_b["weapon_max_range_m"] + rng.uniform(-6.0, 8.0), 50.0, 82.0)),
                "focus_az_deg": float(rng.choice([30.0, 45.0, 60.0, 135.0, 225.0])),
            }
            c_b = _normalize_blue_bs(raw_b)
            c_b["name"] = f"G{gen} {c_b['name']}"
            cand_blues.append(c_b)

        # Evaluate candidates against current Red support quickly (2 most weighted threats)
        q_red = np.asarray(nash_cur["red_strategy"], dtype=np.float64)
        top_rj = np.argsort(-q_red)[:2]
        best_c_score = -1e9
        best_c_obj = cand_blues[0]
        for cb in cand_blues:
            blue_pool.append(cb)
            bi_tmp = len(blue_pool) - 1
            sc = float(np.mean([_eval_match(bi_tmp, int(rj))["blue_utility"] for rj in top_rj]))
            blue_pool.pop()
            eval_cache = {k: v for k, v in eval_cache.items() if k[0] != bi_tmp}
            if sc > best_c_score:
                best_c_score = sc
                best_c_obj = cb

        blue_pool.append(best_c_obj)
        new_bi = len(blue_pool) - 1
        for rj in range(len(red_pool)):
            _eval_match(new_bi, rj)

        # Evolve a Red adaptive counter-package shifting decoy ratio, FO share, and corridor angle
        worst_rj = int(np.argmin(M_cur[best_bi, :]))
        base_r = red_pool[worst_rj]
        new_pkgs = []
        fo_shift = float(rng.uniform(0.15, 0.35))
        az_shift = float(rng.choice([35.0, 48.0, 135.0, 225.0]))
        for idx_p, pkg in enumerate(base_r["swarm_package"]):
            npkg = dict(pkg)
            if idx_p == 0:
                npkg["count"] = max(2, min(8, int(pkg["count"]) + int(rng.choice([0, 1, 2]))))
                npkg["fiber_optic_fraction"] = round(min(1.0, float(pkg["fiber_optic_fraction"]) + fo_shift), 2)
            else:
                npkg["fiber_optic_fraction"] = round(min(1.0, float(pkg["fiber_optic_fraction"]) + 0.25), 2)
            if npkg.get("az_center_deg") is not None:
                npkg["az_center_deg"] = round((az_shift + (idx_p * 180.0 if "pincer" in base_r["package_id"] else 0.0)) % 360.0, 1)
            new_pkgs.append(npkg)
        tot_n_new = sum(int(p["count"]) for p in new_pkgs)
        new_red = {
            "package_id": f"{base_r['package_id']}_g{gen}",
            "name": f"G{gen} Adapted {base_r['name'][:26]}",
            "evasion_profile": str(rng.choice(["jinking_weave", "pop_up_terminal", "corkscrew_spiral"])),
            "swarm_package": new_pkgs,
            "total_drones": tot_n_new,
            "mean_speed_mps": round(sum(p["speed_mps"] * p["count"] for p in new_pkgs) / max(1, tot_n_new), 1),
        }
        red_pool.append(new_red)
        new_rj = len(red_pool) - 1
        for bi in range(len(blue_pool)):
            _eval_match(bi, new_rj)

        M_end = _matrix()
        nash_end = solve_zero_sum_nash_equilibrium(M_end)
        gen_history.append(
            {
                "generation": gen,
                "nash_value": nash_end["nash_value"],
                "best_worst_case": round(float(np.max(np.min(M_end, axis=1))), 2),
                "n_blue": len(blue_pool),
                "n_red": len(red_pool),
                "new_blue": best_c_obj["name"],
                "new_red": new_red["name"],
            }
        )

    final_M = _matrix()
    final_nash = solve_zero_sum_nash_equilibrium(final_M)
    best_bi_fin = int(final_nash["pure_minimax_blue_idx"])
    best_blue_bs = blue_pool[best_bi_fin]
    deadliest_rj = int(np.argmin(np.mean(final_M, axis=0)))
    deadliest_red_pkg = red_pool[deadliest_rj]

    raid_handoff_payload = {
        "label": f"{best_blue_bs['name']} vs {deadliest_red_pkg['name']}",
        "site_preset": "fob_compound_4_turret",
        "custom_turrets": best_blue_bs["turrets"],
        "terrain_preset": terrain_preset,
        "swarm_package": deadliest_red_pkg["swarm_package"],
        "outer_layer_type": best_blue_bs["outer_layer_type"],
        "ammunition_preset": best_blue_bs["ammunition_preset"],
        "evasion_profile": deadliest_red_pkg["evasion_profile"],
        "n_drones": deadliest_red_pkg["total_drones"],
        "threat_speed_mps": deadliest_red_pkg["mean_speed_mps"],
        "weapon_max_range_m": best_blue_bs["weapon_max_range_m"],
    }

    best_eval = eval_cache[(best_bi_fin, deadliest_rj)]
    return {
        "blue_pool": blue_pool,
        "red_pool": red_pool,
        "payoff_matrix": [[round(float(v), 2) for v in row] for row in final_M],
        "nash": final_nash,
        "nash_value": final_nash["nash_value"],
        "best_blue_battlespace": best_blue_bs,
        "deadliest_red_package": deadliest_red_pkg,
        "generation_history": gen_history,
        "raid_handoff_payload": raid_handoff_payload,
        "summary": {
            "terrain_preset": terrain_preset,
            "n_generations": n_generations,
            "n_blue_architectures": len(blue_pool),
            "n_red_packages": len(red_pool),
            "nash_value": final_nash["nash_value"],
            "best_blue_name": best_blue_bs["name"],
            "best_outer_layer": best_blue_bs["outer_layer_type"],
            "deadliest_red_name": deadliest_red_pkg["name"],
            "duel_zero_breach_pct": best_eval["zero_breach_pct"],
            "duel_kill_rate_pct": best_eval["mean_kill_rate_pct"],
            "duel_outer_kills": best_eval["outer_layer_kills"],
        },
    }
