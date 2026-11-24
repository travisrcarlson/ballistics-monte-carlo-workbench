"""Adversarial Red-vs-Blue Competing Monte Carlo & Gaussian Process ML Protection Optimizer.

Couples high-fidelity Counter-UAS shotgun dispersion, exact quadratic-drag pellet TOF,
Kalman evasive tracking residuals, 2-axis turret recoil structural limits, and event-driven
swarm timelines into a two-player minimax co-evolution loop driven by a pure-NumPy/SciPy
Gaussian Process (Matern-5/2) Bayesian surrogate and exact linear-programming Nash solver.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linprog, minimize

from ammo_mc.aero_sphere import exact_quadratic_drag_tof_and_velocity
from ammo_mc.cuas_defense import (
    CUAS_SENSOR_PRESETS,
    _trapezoidal_slew_time,
    simulate_turret_recoil_and_gimbal_loads,
)
from ammo_mc.cuas_mission import CUAS_AMMO_ECONOMICS_CATALOG, _resolve_shotgun_ammo
from ammo_mc.small_arms import (
    compute_drone_projected_geometry,
    generate_multishotgun_array_layout,
    get_choke_definitions,
    get_drone_target_specs,
)

# -------------------------------------------------------------------------
# SEARCH SPACES & BASELINE ARCHETYPES
# -------------------------------------------------------------------------

BLUE_SEARCH_SPACE: dict[str, Any] = {
    "layout_type": ["circular", "concentric", "honeycomb", "grid", "fan_horizontal"],
    "n_barrels": (2, 12),
    "barrels_per_burst": (1, 6),
    "ammunition_preset": [
        "12G_NO4_BUCK_C_UAS",
        "12G_TSS_NO2_C_UAS",
        "12G_DUPLEX_TSS_BUCK_C_UAS",
        "12G_FLECHETTE_SABOT_C_UAS",
    ],
    "choke": ["Cylinder", "Modified", "Full", "Extra Full"],
    "pointing_mode": ["convergent", "divergent_curtain", "staggered_tiling"],
    "splay_angle_mrad": (0.0, 10.0),
    "firing_mode": ["simultaneous_volley", "balanced_pair_ripple", "sequential_ripple"],
    "ripple_delay_ms": (0.0, 20.0),
    "sensor_preset": list(CUAS_SENSOR_PRESETS.keys())[:3],
    "predictor_mode": ["kalman_alpha_beta_gamma", "linear_alpha_beta"],
    "max_engagement_range_m": (40.0, 80.0),
    "slew_rate_deg_s": (100.0, 260.0),
    "reload_time_s": (1.8, 5.2),
    "priority_policy": ["minimum_time_to_breach", "nearest_slant_range", "minimum_slew_angle"],
}

RED_SEARCH_SPACE: dict[str, Any] = {
    "drone_model": [
        "fpv_quad_small",
        "commercial_quad_medium",
        "heavy_octocopter_large",
        "fixed_wing_loitering",
    ],
    "swarm_size": (1, 12),
    "attack_tactic": [
        "synchronized_omni_wave",
        "narrow_corridor_saturation",
        "staggered_multi_altitude",
    ],
    "maneuver_profile": [
        "linear_cruise",
        "sinusoidal_weave",
        "corkscrew_spiral",
        "terminal_popup_jink",
    ],
    "maneuver_g": (0.0, 4.5),
    "drone_speed_mps": (12.0, 42.0),
    "aspect_mode": ["head_on_frontal", "oblique_quartering", "flank_side", "planform_top"],
    "fiber_optic_fraction": (0.0, 1.0),
}

FEATURE_NAMES: list[str] = [
    "blue_n_barrels",
    "blue_barrels_per_burst",
    "blue_pellet_mass_g",
    "blue_pellets_per_shot",
    "blue_pellet_cd",
    "blue_choke_40yd_pct",
    "blue_splay_angle_mrad",
    "blue_ripple_delay_ms",
    "blue_max_range_m",
    "blue_slew_rate_deg_s",
    "blue_reload_time_s",
    "blue_sensor_sigma_mrad",
    "blue_sensor_latency_ms",
    "blue_kalman_3state",
    "blue_ammo_unit_cost_usd",
    "red_swarm_size",
    "red_drone_speed_mps",
    "red_maneuver_g",
    "red_presented_area_cm2",
    "red_drone_core_energy_j",
    "red_drone_crit_hits",
    "red_fiber_optic_fraction",
    "red_omni_wave",
    "red_corridor_rush",
    "red_corkscrew_evasion",
    "red_popup_jink",
]

FEATURE_SIDES: list[str] = (["Blue"] * 15) + (["Red"] * 11)


def _format_blue_label(cfg: dict[str, Any]) -> str:
    ammo_short = {
        "12G_NO4_BUCK_C_UAS": "#4Buck",
        "12G_TSS_NO2_C_UAS": "TSS#2",
        "12G_DUPLEX_TSS_BUCK_C_UAS": "Duplex",
        "12G_FLECHETTE_SABOT_C_UAS": "Flech",
    }.get(str(cfg.get("ammunition_preset", "")), str(cfg.get("ammunition_preset", ""))[:6])
    pt = str(cfg.get("pointing_mode", "convergent"))[:4]
    return f"{int(cfg['n_barrels'])}b/{int(cfg['barrels_per_burst'])}s {ammo_short} {cfg['choke'][:3]} {pt}"


def _format_red_label(cfg: dict[str, Any]) -> str:
    dm_short = {
        "fpv_quad_small": "FPV",
        "commercial_quad_medium": "Quad",
        "heavy_octocopter_large": "Octo",
        "fixed_wing_loitering": "Wing",
    }.get(str(cfg.get("drone_model", "")), "UAS")
    man_short = {
        "linear_cruise": "Lin",
        "sinusoidal_weave": "Weave",
        "corkscrew_spiral": "Cork",
        "terminal_popup_jink": "Jink",
    }.get(str(cfg.get("maneuver_profile", "")), "Evd")
    return f"{int(cfg['swarm_size'])}x {dm_short} {man_short}@{float(cfg['maneuver_g']):.1f}g {float(cfg['drone_speed_mps']):.0f}m/s"


def normalize_blue_config(cfg: dict[str, Any], max_barrels: int = 12) -> dict[str, Any]:
    """Sanitize and normalize a Blue turret platform specification dict."""
    n_b = int(max(2, min(max_barrels, round(float(cfg.get("n_barrels", 6))))))
    bpb = int(max(1, min(min(n_b, 6), round(float(cfg.get("barrels_per_burst", 2))))))
    lt = str(cfg.get("layout_type") or cfg.get("layout") or "circular")
    if lt not in BLUE_SEARCH_SPACE["layout_type"]:
        lt = "circular"
    ammo = str(cfg.get("ammunition_preset") or cfg.get("ammo") or "12G_NO4_BUCK_C_UAS")
    if ammo not in BLUE_SEARCH_SPACE["ammunition_preset"]:
        ammo = "12G_NO4_BUCK_C_UAS"
    chk = str(cfg.get("choke", "Modified"))
    if chk not in BLUE_SEARCH_SPACE["choke"]:
        chk = "Modified"
    pm = str(cfg.get("pointing_mode") or cfg.get("pointing") or "convergent")
    if pm not in BLUE_SEARCH_SPACE["pointing_mode"]:
        pm = "convergent"
    if "splay_angle_deg" in cfg and "splay_angle_mrad" not in cfg:
        splay = float(math.radians(float(cfg["splay_angle_deg"])) * 1000.0)
    else:
        splay = float(max(0.0, min(10.0, float(cfg.get("splay_angle_mrad", 4.0 if pm == "divergent_curtain" else 0.0)))))
    if pm != "divergent_curtain":
        splay = 0.0
    fm = str(cfg.get("firing_mode", "balanced_pair_ripple"))
    if fm not in BLUE_SEARCH_SPACE["firing_mode"]:
        fm = "balanced_pair_ripple"
    rdt = float(max(0.0, min(20.0, float(cfg.get("ripple_delay_ms", 8.0 if fm != "simultaneous_volley" else 0.0)))))
    if fm == "simultaneous_volley":
        rdt = 0.0
    sp = str(cfg.get("sensor_preset", "Sensor_Fused_Radar_EOIR"))
    if sp not in CUAS_SENSOR_PRESETS:
        sp = "Sensor_Fused_Radar_EOIR"
    pr = str(cfg.get("predictor_mode", "kalman_alpha_beta_gamma"))
    if pr not in BLUE_SEARCH_SPACE["predictor_mode"]:
        pr = "kalman_alpha_beta_gamma"
    rmax = float(max(40.0, min(80.0, float(cfg.get("max_engagement_range_m", 60.0)))))
    slew = float(max(100.0, min(260.0, float(cfg.get("slew_rate_deg_s", 175.0)))))
    rel = float(max(1.8, min(5.2, float(cfg.get("reload_time_s", 3.0)))))
    pol = str(cfg.get("priority_policy", "minimum_time_to_breach"))
    if pol not in BLUE_SEARCH_SPACE["priority_policy"]:
        pol = "minimum_time_to_breach"

    splay_deg = round(math.degrees(splay / 1000.0), 2)
    out = {
        "name": str(cfg.get("name", "")).strip(),
        "layout_type": lt,
        "layout": lt,
        "n_barrels": n_b,
        "barrels_per_burst": bpb,
        "ammunition_preset": ammo,
        "ammo": ammo,
        "choke": chk,
        "pointing_mode": pm,
        "pointing": pm,
        "splay_angle_mrad": round(splay, 2),
        "splay_angle_deg": splay_deg,
        "firing_mode": fm,
        "ripple_delay_ms": round(rdt, 1),
        "sensor_preset": sp,
        "predictor_mode": pr,
        "max_engagement_range_m": round(rmax, 1),
        "slew_rate_deg_s": round(slew, 1),
        "reload_time_s": round(rel, 2),
        "priority_policy": pol,
        "burst_count": int(cfg.get("burst_count", 1)),
    }
    if not out["name"]:
        out["name"] = _format_blue_label(out)
    return out


def normalize_red_config(cfg: dict[str, Any], max_swarm_size: int = 12) -> dict[str, Any]:
    """Sanitize and normalize a Red adversarial drone threat configuration dict."""
    dm = str(cfg.get("drone_model", "fpv_quad_small"))
    if dm not in RED_SEARCH_SPACE["drone_model"]:
        dm = "fpv_quad_small"
    sw = int(max(1, min(max_swarm_size, round(float(cfg.get("swarm_size", 6))))))
    raw_tac = str(cfg.get("attack_tactic") or cfg.get("tactic") or "synchronized_omni_wave")
    tactic_aliases = {
        "simultaneous_divergent": "synchronized_omni_wave",
        "convergent_pincer": "narrow_corridor_saturation",
        "staggered_trail": "staggered_multi_altitude",
        "low_altitude_terrain_mask": "narrow_corridor_saturation",
        "linear_trail": "staggered_multi_altitude",
    }
    tac = tactic_aliases.get(raw_tac, raw_tac)
    if tac not in RED_SEARCH_SPACE["attack_tactic"]:
        tac = "synchronized_omni_wave"

    raw_man = str(cfg.get("maneuver_profile") or cfg.get("evasion") or "sinusoidal_weave")
    evasion_aliases = {
        "weaving_jink": "sinusoidal_weave",
        "barrel_roll": "corkscrew_spiral",
        "corkscrew": "corkscrew_spiral",
        "random_walk": "terminal_popup_jink",
        "none": "linear_cruise",
    }
    man = evasion_aliases.get(raw_man, raw_man)
    if man not in RED_SEARCH_SPACE["maneuver_profile"]:
        man = "sinusoidal_weave"
    mg = float(max(0.0, min(4.5, float(cfg.get("maneuver_g", 2.0)))))
    if man == "linear_cruise":
        mg = 0.0
    spd = float(max(12.0, min(42.0, float(cfg.get("drone_speed_mps") or cfg.get("speed_mps") or 24.0))))
    asp = str(cfg.get("aspect_mode", "head_on_frontal"))
    if asp not in RED_SEARCH_SPACE["aspect_mode"]:
        asp = "head_on_frontal"
    fo = float(max(0.0, min(1.0, float(cfg.get("fiber_optic_fraction") or cfg.get("stealth_level") or 0.3))))

    out = {
        "name": str(cfg.get("name", "")).strip(),
        "drone_model": dm,
        "swarm_size": sw,
        "attack_tactic": tac,
        "tactic": tac,
        "maneuver_profile": man,
        "evasion": man,
        "maneuver_g": round(mg, 2),
        "drone_speed_mps": round(spd, 1),
        "speed_mps": round(spd, 1),
        "altitude_m": float(cfg.get("altitude_m", 25.0)),
        "aspect_mode": asp,
        "fiber_optic_fraction": round(fo, 2),
    }
    if not out["name"]:
        out["name"] = _format_red_label(out)
    return out


def get_default_blue_seed_pool(
    max_barrels: int = 10,
    stage1_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return initial diverse Blue protection architectures, optionally seeded by Stage 1."""
    pool: list[dict[str, Any]] = []
    if isinstance(stage1_profile, dict) and stage1_profile:
        nb_s1 = int(stage1_profile.get("n_barrels", 6))
        s1_cfg = normalize_blue_config(
            {
                "name": "Stage 1 Active Turret",
                "layout_type": stage1_profile.get("layout_type", "circular"),
                "n_barrels": min(max_barrels, nb_s1),
                "barrels_per_burst": min(max_barrels, max(1, min(4, nb_s1 // 2))),
                "ammunition_preset": stage1_profile.get("ammunition_preset", "12G_NO4_BUCK_C_UAS"),
                "choke": stage1_profile.get("choke", "Modified"),
                "pointing_mode": stage1_profile.get("pointing_mode", "convergent"),
                "splay_angle_mrad": stage1_profile.get("splay_angle_mrad", 0.0),
                "firing_mode": stage1_profile.get("firing_mode", "simultaneous_volley"),
                "ripple_delay_ms": stage1_profile.get("ripple_delay_ms", 0.0),
                "sensor_preset": "Sensor_Fused_Radar_EOIR",
                "predictor_mode": "kalman_alpha_beta_gamma",
                "max_engagement_range_m": min(75.0, max(38.0, float(stage1_profile.get("target_distance_m", 35.0)) + 20.0)),
            },
            max_barrels=max_barrels,
        )
        pool.append(s1_cfg)

    defaults = [
        {
            "name": "6-Bbl #4 Buck Mod Convergent",
            "layout_type": "circular",
            "n_barrels": min(max_barrels, 6),
            "barrels_per_burst": 2,
            "ammunition_preset": "12G_NO4_BUCK_C_UAS",
            "choke": "Modified",
            "pointing_mode": "convergent",
            "firing_mode": "simultaneous_volley",
            "sensor_preset": "EO_IR_Dual_Thermal_Tracker",
            "predictor_mode": "linear_alpha_beta",
            "max_engagement_range_m": 50.0,
            "slew_rate_deg_s": 110.0,
            "reload_time_s": 4.2,
            "priority_policy": "nearest_slant_range",
        },
        {
            "name": "8-Bbl TSS#2 Curtain Kalman",
            "layout_type": "concentric",
            "n_barrels": min(max_barrels, 8),
            "barrels_per_burst": 2,
            "ammunition_preset": "12G_TSS_NO2_C_UAS",
            "choke": "Full",
            "pointing_mode": "divergent_curtain",
            "splay_angle_mrad": 4.5,
            "firing_mode": "balanced_pair_ripple",
            "ripple_delay_ms": 8.0,
            "sensor_preset": "Sensor_Fused_Radar_EOIR",
            "predictor_mode": "kalman_alpha_beta_gamma",
            "max_engagement_range_m": 65.0,
            "slew_rate_deg_s": 145.0,
            "reload_time_s": 3.4,
            "priority_policy": "minimum_time_to_breach",
        },
        {
            "name": "10-Bbl Duplex Wide-Net Staggered",
            "layout_type": "honeycomb",
            "n_barrels": min(max_barrels, 10),
            "barrels_per_burst": 3,
            "ammunition_preset": "12G_DUPLEX_TSS_BUCK_C_UAS",
            "choke": "Modified",
            "pointing_mode": "staggered_tiling",
            "firing_mode": "balanced_pair_ripple",
            "ripple_delay_ms": 10.0,
            "sensor_preset": "Ku_Band_Micro_Doppler_Radar",
            "predictor_mode": "kalman_alpha_beta_gamma",
            "max_engagement_range_m": 60.0,
            "slew_rate_deg_s": 135.0,
            "reload_time_s": 4.0,
            "priority_policy": "minimum_time_to_breach",
        },
        {
            "name": "6-Bbl Long-Range Flechette Sniper",
            "layout_type": "grid",
            "n_barrels": min(max_barrels, 6),
            "barrels_per_burst": 2,
            "ammunition_preset": "12G_FLECHETTE_SABOT_C_UAS",
            "choke": "Full",
            "pointing_mode": "convergent",
            "firing_mode": "sequential_ripple",
            "ripple_delay_ms": 8.0,
            "sensor_preset": "Sensor_Fused_Radar_EOIR",
            "predictor_mode": "kalman_alpha_beta_gamma",
            "max_engagement_range_m": 72.0,
            "slew_rate_deg_s": 125.0,
            "reload_time_s": 3.6,
            "priority_policy": "minimum_time_to_breach",
        },
        {
            "name": "4-Bbl Agile Rapid Buck Turret",
            "layout_type": "circular",
            "n_barrels": min(max_barrels, 4),
            "barrels_per_burst": 2,
            "ammunition_preset": "12G_NO4_BUCK_C_UAS",
            "choke": "Cylinder",
            "pointing_mode": "divergent_curtain",
            "splay_angle_mrad": 6.0,
            "firing_mode": "balanced_pair_ripple",
            "ripple_delay_ms": 6.0,
            "sensor_preset": "EO_IR_Dual_Thermal_Tracker",
            "predictor_mode": "kalman_alpha_beta_gamma",
            "max_engagement_range_m": 44.0,
            "slew_rate_deg_s": 170.0,
            "reload_time_s": 2.4,
            "priority_policy": "minimum_time_to_breach",
        },
    ]
    for item in defaults:
        if len(pool) >= 5:
            break
        norm = normalize_blue_config(item, max_barrels=max_barrels)
        if not any(existing["name"] == norm["name"] for existing in pool):
            pool.append(norm)
    return pool


def get_default_red_seed_pool(max_swarm_size: int = 10) -> list[dict[str, Any]]:
    """Return initial diverse Red drone threat doctrines for co-evolution."""
    seeds = [
        {
            "name": "Agile FPV Jinking Swarm",
            "drone_model": "fpv_quad_small",
            "swarm_size": min(max_swarm_size, 7),
            "attack_tactic": "synchronized_omni_wave",
            "maneuver_profile": "corkscrew_spiral",
            "maneuver_g": 3.2,
            "drone_speed_mps": 29.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.4,
        },
        {
            "name": "Armored Heavy Octo Flight",
            "drone_model": "heavy_octocopter_large",
            "swarm_size": min(max_swarm_size, 4),
            "attack_tactic": "staggered_multi_altitude",
            "maneuver_profile": "sinusoidal_weave",
            "maneuver_g": 1.2,
            "drone_speed_mps": 18.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.2,
        },
        {
            "name": "High-Speed Wing Dive",
            "drone_model": "fixed_wing_loitering",
            "swarm_size": min(max_swarm_size, 4),
            "attack_tactic": "narrow_corridor_saturation",
            "maneuver_profile": "terminal_popup_jink",
            "maneuver_g": 2.8,
            "drone_speed_mps": 38.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.6,
        },
        {
            "name": "Narrow-Corridor Saturation Rush",
            "drone_model": "commercial_quad_medium",
            "swarm_size": min(max_swarm_size, 10),
            "attack_tactic": "narrow_corridor_saturation",
            "maneuver_profile": "sinusoidal_weave",
            "maneuver_g": 1.6,
            "drone_speed_mps": 23.0,
            "aspect_mode": "oblique_quartering",
            "fiber_optic_fraction": 0.3,
        },
        {
            "name": "Fiber-Optic Pop-Up Stealth Wave",
            "drone_model": "fpv_quad_small",
            "swarm_size": min(max_swarm_size, 6),
            "attack_tactic": "staggered_multi_altitude",
            "maneuver_profile": "terminal_popup_jink",
            "maneuver_g": 3.8,
            "drone_speed_mps": 32.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 1.0,
        },
    ]
    return [normalize_red_config(s, max_swarm_size=max_swarm_size) for s in seeds]


# -------------------------------------------------------------------------
# PHYSICS FEATURE ENCODER (BLUE + RED -> R^26)
# -------------------------------------------------------------------------


def encode_matchup_features(blue: dict[str, Any], red: dict[str, Any]) -> np.ndarray:
    """Encode a (Blue turret, Red threat) matchup into a physical feature vector."""
    _, ammo = _resolve_shotgun_ammo(str(blue["ammunition_preset"]))
    choke_defs = get_choke_definitions()
    c_info = choke_defs.get(str(blue["choke"]), choke_defs["Modified"])
    sensor = CUAS_SENSOR_PRESETS.get(str(blue["sensor_preset"]), CUAS_SENSOR_PRESETS["Sensor_Fused_Radar_EOIR"])
    econ = CUAS_AMMO_ECONOMICS_CATALOG.get(str(blue["ammunition_preset"]), CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])

    drone_specs = get_drone_target_specs()
    d_spec = drone_specs.get(str(red["drone_model"]), drone_specs["fpv_quad_small"])
    geom = compute_drone_projected_geometry(d_spec, aspect_mode=str(red["aspect_mode"]))

    splay_eff = float(blue["splay_angle_mrad"]) if blue["pointing_mode"] == "divergent_curtain" else (
        2.2 if blue["pointing_mode"] == "staggered_tiling" else 0.0
    )

    return np.array(
        [
            float(blue["n_barrels"]),
            float(blue["barrels_per_burst"]),
            float(ammo["pellet_mass_g"]),
            float(ammo["pellets_per_shot"]),
            float(ammo.get("effective_cd", 0.47)),
            float(c_info["benchmark_40yd_30in_pct"]),
            splay_eff,
            float(blue["ripple_delay_ms"]),
            float(blue["max_engagement_range_m"]),
            float(blue["slew_rate_deg_s"]),
            float(blue["reload_time_s"]),
            float(sensor["sigma_az_mrad"]),
            float(sensor["fcs_latency_ms"]),
            1.0 if blue["predictor_mode"] == "kalman_alpha_beta_gamma" else 0.0,
            float(econ["unit_cost_usd"]),
            float(red["swarm_size"]),
            float(red["drone_speed_mps"]),
            float(red["maneuver_g"]),
            float(geom["presented_area_cm2"]),
            float(d_spec.get("lethal_energy_core_j", 24.0)),
            float(d_spec.get("critical_hits_for_kill", 3)),
            float(red["fiber_optic_fraction"]),
            1.0 if red["attack_tactic"] == "synchronized_omni_wave" else 0.0,
            1.0 if red["attack_tactic"] == "narrow_corridor_saturation" else 0.0,
            1.0 if red["maneuver_profile"] == "corkscrew_spiral" else 0.0,
            1.0 if red["maneuver_profile"] == "terminal_popup_jink" else 0.0,
        ],
        dtype=np.float64,
    )


# -------------------------------------------------------------------------
# PURE-NUMPY / SCIPY GAUSSIAN PROCESS SURROGATE (MATERN-5/2 + MLL TUNING)
# -------------------------------------------------------------------------


class GaussianProcessSurrogate:
    """Matern-5/2 Gaussian Process regressor trained via Cholesky & marginal likelihood."""

    def __init__(
        self,
        lengthscale: float = 1.6,
        signal_std: float = 1.0,
        noise_std: float = 0.22,
    ) -> None:
        self.lengthscale = float(lengthscale)
        self.signal_std = float(signal_std)
        self.noise_std = float(noise_std)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_scale: float = 1.0
        self.X_norm: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.L_cho: np.ndarray | None = None

    @staticmethod
    def _matern52_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float, signal_var: float) -> np.ndarray:
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1, keepdims=True).T
            - 2.0 * (X1 @ X2.T)
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        # Scale Euclidean distance by sqrt(D) so isotropic lengthscale stays invariant to feature count
        d_scale = math.sqrt(max(1, X1.shape[1]))
        r = math.sqrt(5.0) * np.sqrt(sq_dist) / max(lengthscale * d_scale, 1e-4)
        return signal_var * (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)

    def _fit_cholesky_with_params(
        self, Xn: np.ndarray, yn: np.ndarray, l_val: float, sf_val: float, sn_val: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        n = Xn.shape[0]
        K_base = self._matern52_kernel(Xn, Xn, l_val, sf_val ** 2)
        diag_noise = max(sn_val ** 2, 1e-4)
        for jitter in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                K = K_base + (diag_noise + jitter) * np.eye(n)
                c_fac, lower = cho_factor(K, lower=True, check_finite=False)
                alpha = cho_solve((c_fac, lower), yn, check_finite=False)
                log_det = 2.0 * float(np.sum(np.log(np.maximum(np.diag(c_fac), 1e-12))))
                nll = 0.5 * float(yn @ alpha) + 0.5 * log_det + 0.5 * n * math.log(2.0 * math.pi)
                return nll, alpha, c_fac
            except np.linalg.LinAlgError:
                continue
        K_fb = K_base + max(diag_noise, 0.05) * np.eye(n)
        c_fac, lower = cho_factor(K_fb, lower=True, check_finite=False)
        alpha = cho_solve((c_fac, lower), yn, check_finite=False)
        return 1e9, alpha, c_fac

    def fit(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> "GaussianProcessSurrogate":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        self.x_mean = np.mean(X_arr, axis=0)
        x_sd = np.std(X_arr, axis=0)
        self.x_std = np.where(x_sd < 1e-8, 1.0, x_sd)
        self.X_norm = (X_arr - self.x_mean) / self.x_std

        self.y_mean = float(np.mean(y_arr))
        y_sd = float(np.std(y_arr))
        self.y_scale = y_sd if y_sd > 1e-6 else 1.0
        yn = (y_arr - self.y_mean) / self.y_scale

        best_params = (self.lengthscale, self.signal_std, self.noise_std)
        if optimize_hyperparams and len(y_arr) >= 12:
            candidates = [
                (0.8, 1.0, 0.18),
                (1.4, 1.0, 0.22),
                (2.2, 1.0, 0.25),
            ]
            best_nll = float("inf")
            for c_l, c_sf, c_sn in candidates:
                nll_c, _, _ = self._fit_cholesky_with_params(self.X_norm, yn, c_l, c_sf, c_sn)
                if nll_c < best_nll:
                    best_nll = nll_c
                    best_params = (c_l, c_sf, c_sn)

            def _obj(theta: np.ndarray) -> float:
                l_v, sf_v, sn_v = math.exp(theta[0]), math.exp(theta[1]), math.exp(theta[2])
                nll_v, _, _ = self._fit_cholesky_with_params(self.X_norm, yn, l_v, sf_v, sn_v)
                return float(nll_v)

            x0 = np.log(np.array(best_params, dtype=np.float64))
            bounds = [(math.log(0.35), math.log(6.0)), (math.log(0.25), math.log(3.0)), (math.log(0.05), math.log(0.9))]
            opt = minimize(_obj, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 18})
            if opt.success or math.isfinite(opt.fun):
                l_opt, sf_opt, sn_opt = np.exp(opt.x)
                best_params = (float(l_opt), float(sf_opt), float(sn_opt))

        self.lengthscale, self.signal_std, self.noise_std = best_params
        _, self.alpha, self.L_cho = self._fit_cholesky_with_params(
            self.X_norm, yn, self.lengthscale, self.signal_std, self.noise_std
        )
        return self

    def predict(self, X_new: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        assert self.X_norm is not None and self.x_mean is not None and self.x_std is not None
        assert self.alpha is not None and self.L_cho is not None
        Xn_new = (np.asarray(X_new, dtype=np.float64) - self.x_mean) / self.x_std
        K_star = self._matern52_kernel(Xn_new, self.X_norm, self.lengthscale, self.signal_std ** 2)
        mu_n = K_star @ self.alpha
        mu = np.clip(self.y_mean + self.y_scale * mu_n, 0.0, 100.0)
        if not return_std:
            return mu, None
        V = cho_solve((self.L_cho, True), K_star.T, check_finite=False)
        diag_kss = self.signal_std ** 2
        var_n = np.maximum(diag_kss - np.sum(K_star.T * V, axis=0), 1e-6)
        std = self.y_scale * np.sqrt(var_n)
        return mu, std

    def to_bundle(self, X_raw: np.ndarray, y_raw: np.ndarray) -> dict[str, Any]:
        """Serialize GP parameters and training data into JSON-friendly bundle."""
        return {
            "lengthscale": round(self.lengthscale, 5),
            "signal_std": round(self.signal_std, 5),
            "noise_std": round(self.noise_std, 5),
            "X_train": np.asarray(X_raw, dtype=np.float64).tolist(),
            "y_train": np.asarray(y_raw, dtype=np.float64).tolist(),
        }

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> "GaussianProcessSurrogate":
        gp = cls(
            lengthscale=float(bundle.get("lengthscale", 1.6)),
            signal_std=float(bundle.get("signal_std", 1.0)),
            noise_std=float(bundle.get("noise_std", 0.22)),
        )
        X_tr = np.asarray(bundle["X_train"], dtype=np.float64)
        y_tr = np.asarray(bundle["y_train"], dtype=np.float64)
        gp.fit(X_tr, y_tr, optimize_hyperparams=False)
        return gp

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k_folds: int = 5, seed: int = 42) -> dict[str, Any]:
        """Run K-fold out-of-sample cross-validation and return R^2, RMSE, MAE & predictions."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = len(y_arr)
        folds = max(2, min(k_folds, n))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        oof = np.zeros(n, dtype=np.float64)
        for f_idx in range(folds):
            val_idx = perm[f_idx::folds]
            tr_mask = np.ones(n, dtype=bool)
            tr_mask[val_idx] = False
            sub_gp = GaussianProcessSurrogate(self.lengthscale, self.signal_std, self.noise_std)
            sub_gp.fit(X_arr[tr_mask], y_arr[tr_mask], optimize_hyperparams=False)
            pred_v, _ = sub_gp.predict(X_arr[val_idx], return_std=False)
            oof[val_idx] = pred_v

        err = y_arr - oof
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        return {
            "cv_r2": round(r2, 4),
            "cv_rmse": round(rmse, 2),
            "cv_mae": round(mae, 2),
            "oof_predictions": [round(float(v), 2) for v in oof],
        }

    def compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        feature_sides: list[str] | None = None,
        n_repeats: int = 4,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Rank Blue and Red physical levers via permutation RMSE sensitivity."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        f_names = feature_names or FEATURE_NAMES
        f_sides = feature_sides or FEATURE_SIDES
        rng = np.random.default_rng(seed)

        base_pred, _ = self.predict(X_arr, return_std=False)
        base_rmse = float(np.sqrt(np.mean((y_arr - base_pred) ** 2)))

        raw_imps: list[float] = []
        rows: list[dict[str, Any]] = []
        n_feats = min(X_arr.shape[1], len(f_names))
        for j in range(n_feats):
            if float(np.std(X_arr[:, j])) < 1e-9:
                delta_rmse = 0.0
            else:
                deltas = []
                X_perm = X_arr.copy()
                col_orig = X_arr[:, j].copy()
                for _ in range(n_repeats):
                    X_perm[:, j] = rng.permutation(col_orig)
                    p_perm, _ = self.predict(X_perm, return_std=False)
                    rmse_p = float(np.sqrt(np.mean((y_arr - p_perm) ** 2)))
                    deltas.append(max(0.0, rmse_p - base_rmse))
                delta_rmse = float(np.mean(deltas))
            raw_imps.append(delta_rmse)

        tot = sum(raw_imps)
        for j in range(n_feats):
            pct = (100.0 * raw_imps[j] / tot) if tot > 1e-9 else 0.0
            rows.append(
                {
                    "feature": f_names[j],
                    "side": f_sides[j] if j < len(f_sides) else "Shared",
                    "importance_rmse": round(raw_imps[j], 3),
                    "importance_pct": round(pct, 1),
                }
            )
        rows.sort(key=lambda r: r["importance_rmse"], reverse=True)
        return rows


# -------------------------------------------------------------------------
# EXACT ZERO-SUM LINEAR PROGRAMMING NASH EQUILIBRIUM SOLVER
# -------------------------------------------------------------------------


def solve_zero_sum_nash_equilibrium(payoff_matrix: np.ndarray | list[list[float]]) -> dict[str, Any]:
    """Solve exact two-player zero-sum mixed-strategy Nash equilibrium via HiGHS LP."""
    M = np.asarray(payoff_matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
        raise ValueError("payoff_matrix must be a non-empty 2D array.")
    m, n = M.shape

    row_mins = np.min(M, axis=1)
    pure_blue_idx = int(np.argmax(row_mins))
    pure_blue_val = float(row_mins[pure_blue_idx])
    red_br_idx = int(np.argmin(M[pure_blue_idx, :]))

    if m == 1 and n == 1:
        return {
            "blue_strategy": [1.0],
            "red_strategy": [1.0],
            "nash_value": round(float(M[0, 0]), 3),
            "duality_gap": 0.0,
            "pure_minimax_blue_idx": 0,
            "pure_minimax_blue_value": round(float(M[0, 0]), 3),
            "red_best_response_to_best_blue_idx": 0,
        }

    # Charnes-Cooper positive shift ensures strictly positive value v_plus > 0
    shift = max(1.0, -float(np.min(M)) + 1.0)
    Mp = M + shift

    # Blue (row maximin): min sum(x_i) s.t. -(Mp^T) x <= -1, x >= 0
    res_b = linprog(
        c=np.ones(m),
        A_ub=-Mp.T,
        b_ub=-np.ones(n),
        bounds=(0.0, None),
        method="highs",
    )
    if res_b.success and float(np.sum(res_b.x)) > 1e-12:
        x_b = np.maximum(res_b.x, 0.0)
        v_blue = (1.0 / float(np.sum(x_b))) - shift
        p_blue = x_b / float(np.sum(x_b))
    else:
        p_blue = np.zeros(m)
        p_blue[pure_blue_idx] = 1.0
        v_blue = pure_blue_val

    # Red (col minimax): max sum(y_j) -> min -sum(y_j) s.t. Mp y <= 1, y >= 0
    res_r = linprog(
        c=-np.ones(n),
        A_ub=Mp,
        b_ub=np.ones(m),
        bounds=(0.0, None),
        method="highs",
    )
    if res_r.success and float(np.sum(res_r.x)) > 1e-12:
        y_r = np.maximum(res_r.x, 0.0)
        v_red = (1.0 / float(np.sum(y_r))) - shift
        q_red = y_r / float(np.sum(y_r))
    else:
        q_red = np.zeros(n)
        q_red[red_br_idx] = 1.0
        v_red = pure_blue_val

    duality_gap = abs(float(v_blue) - float(v_red))
    nash_v = 0.5 * (float(v_blue) + float(v_red))
    return {
        "blue_strategy": [round(float(v), 4) for v in p_blue],
        "red_strategy": [round(float(v), 4) for v in q_red],
        "nash_value": round(nash_v, 3),
        "duality_gap": round(duality_gap, 8),
        "pure_minimax_blue_idx": pure_blue_idx,
        "pure_minimax_blue_value": round(pure_blue_val, 3),
        "red_best_response_to_best_blue_idx": red_br_idx,
    }


# -------------------------------------------------------------------------
# COUPLED PHYSICS RED-VS-BLUE MONTE CARLO EVALUATOR
# -------------------------------------------------------------------------


def evaluate_red_vs_blue_matchup_mc(
    blue: dict[str, Any],
    red: dict[str, Any],
    n_episodes: int = 20,
    mount_load_limit_kn: float = 3.5,
    breach_keepout_radius_m: float = 12.0,
    w_zero_breach: float = 0.55,
    w_kill_rate: float = 0.45,
    cost_penalty_weight: float = 0.08,
    recoil_penalty_per_kn: float = 8.0,
    seed: int = 42,
    _recoil_cache: dict[tuple[Any, ...], float] | None = None,
    _geom_cache: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Evaluate one (Blue turret, Red swarm) pairing with coupled recoil, FCS & swarm MC."""
    rng = np.random.default_rng(seed)
    blue_n = normalize_blue_config(blue)
    red_n = normalize_red_config(red)

    # 1. Recoil & Structural Mount Force (for barrels_per_burst active salvo)
    rdt_bucket = round(float(blue_n["ripple_delay_ms"]), 0)
    rec_key = (
        int(blue_n["barrels_per_burst"]),
        blue_n["ammunition_preset"],
        blue_n["firing_mode"],
        rdt_bucket,
    )
    if _recoil_cache is not None and rec_key in _recoil_cache:
        peak_mount_kn = _recoil_cache[rec_key]
    else:
        b_salvo = generate_multishotgun_array_layout(
            layout_type=blue_n["layout_type"],
            n_barrels=int(blue_n["barrels_per_burst"]),
            mount_radius_m=0.20,
            default_choke=blue_n["choke"],
            default_barrel_length_in=24.0,
            default_ammunition_preset=blue_n["ammunition_preset"],
        )
        rec_res = simulate_turret_recoil_and_gimbal_loads(
            barrels=b_salvo,
            firing_mode=blue_n["firing_mode"],
            ripple_delay_ms=float(rdt_bucket),
            default_ammunition_preset=blue_n["ammunition_preset"],
            mount_mass_kg=45.0,
            mount_stiffness_n_per_m=120000.0,
            mount_load_limit_kn=mount_load_limit_kn,
            gimbal_trunnion_offset_z_m=0.18,
        )
        peak_mount_kn = float(rec_res["summary"]["peak_transmitted_mount_force_kn"])
        if _recoil_cache is not None:
            _recoil_cache[rec_key] = peak_mount_kn

    mount_feasible = bool(peak_mount_kn <= mount_load_limit_kn)

    # 2. Target Projected Area & Vulnerability
    drone_specs = get_drone_target_specs()
    d_spec = drone_specs.get(red_n["drone_model"], drone_specs["fpv_quad_small"])
    g_key = (str(red_n["drone_model"]), str(red_n["aspect_mode"]))
    if _geom_cache is not None and g_key in _geom_cache:
        area_vuln = _geom_cache[g_key]
    else:
        geom = compute_drone_projected_geometry(d_spec, aspect_mode=red_n["aspect_mode"])
        area_vuln = float(geom.get("presented_area_m2", 0.02))
        if _geom_cache is not None:
            _geom_cache[g_key] = area_vuln

    e_rotor = float(d_spec.get("lethal_energy_rotor_j", 8.0))
    e_core = float(d_spec.get("lethal_energy_core_j", 24.0))
    n_crit = int(d_spec.get("critical_hits_for_kill", 3))

    # 3. Ammo & Dispersion Constants
    _, ammo = _resolve_shotgun_ammo(blue_n["ammunition_preset"])
    econ = CUAS_AMMO_ECONOMICS_CATALOG.get(blue_n["ammunition_preset"], CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])
    unit_cost_usd = float(econ["unit_cost_usd"])
    pellets_per_shot = int(ammo["pellets_per_shot"])
    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    v0_nom = float(ammo["v0_mps"])
    base_cd = float(ammo.get("effective_cd", 0.47))
    area_p = math.pi * 0.25 * (d_m ** 2)
    rho_air = 1.225
    kappa = (rho_air * base_cd * area_p) / (2.0 * max(m_kg, 1e-6))

    choke_defs = get_choke_definitions()
    c_info = choke_defs.get(blue_n["choke"], choke_defs["Modified"])
    p_bench = float(c_info["benchmark_40yd_30in_pct"])
    sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_bench))
    theta_choke = sigma_ref / 36.58

    sensor = CUAS_SENSOR_PRESETS.get(blue_n["sensor_preset"], CUAS_SENSOR_PRESETS["Sensor_Fused_Radar_EOIR"])
    sigma_az_rad = float(sensor["sigma_az_mrad"]) / 1000.0
    t_lat = float(sensor["fcs_latency_ms"]) / 1000.0
    slew_rad_s = math.radians(blue_n["slew_rate_deg_s"])

    prof_mult = {
        "linear_cruise": 0.0,
        "sinusoidal_weave": 1.0,
        "corkscrew_spiral": 1.28,
        "terminal_popup_jink": 1.50,
    }.get(red_n["maneuver_profile"], 1.0)
    a_eff = float(red_n["maneuver_g"]) * 9.80665 * prof_mult
    r_eq_sq = area_vuln / math.pi

    def _sample_burst_pk(slant_range_m: float, d_speed_mps: float, e_lead_draw: float) -> float:
        R = max(slant_range_m, breach_keepout_radius_m)
        t_tof, v_imp = exact_quadratic_drag_tof_and_velocity(R, v0_nom, kappa)
        ke_j = 0.5 * m_kg * (v_imp ** 2)
        if ke_j < e_rotor:
            return 0.0

        t_eff = t_tof + t_lat
        # Unmodeled maneuvering acceleration residual during latency + flight time
        if blue_n["predictor_mode"] == "kalman_alpha_beta_gamma":
            e_maneuver = 0.24 * a_eff * (t_eff ** 2)
        else:
            e_maneuver = 0.52 * a_eff * (t_eff ** 2)

        # Angular servo tracking lag when crossing speed approaches/exceeds gimbal rate
        omega_req = 0.55 * d_speed_mps / R
        servo_lag_m = max(0.0, omega_req - slew_rad_s) * t_eff * R
        sigma_meas = max(0.02, sigma_az_rad * R * 1.15)
        e_lead_tot = e_maneuver + servo_lag_m + e_lead_draw * sigma_meas

        sigma_r = 0.00463 + theta_choke * R * ((R / 36.58) ** 0.15)
        if blue_n["pointing_mode"] == "divergent_curtain":
            sigma_curt = 0.58 * (float(blue_n["splay_angle_mrad"]) / 1000.0) * R
            sigma_eff = math.hypot(sigma_r, sigma_curt)
        elif blue_n["pointing_mode"] == "staggered_tiling":
            sigma_eff = 1.24 * sigma_r
        else:
            sigma_eff = sigma_r

        sigma_eff_sq = max(sigma_eff ** 2, 1e-6)
        p_pel_center = 1.0 - math.exp(-r_eq_sq / (2.0 * sigma_eff_sq))
        off_axis_atten = math.exp(-(e_lead_tot ** 2) / (2.0 * sigma_eff_sq))
        # Curtains dilute peak center density slightly unless lead offset is caught by spread
        density_dilution = sigma_r / max(sigma_eff, sigma_r)
        p_pel = p_pel_center * off_axis_atten * (0.45 + 0.55 * density_dilution)

        energy_fac = 0.55 + 0.45 * min(1.0, max(0.0, ke_j - e_rotor) / max(e_core - e_rotor, 1.0))
        lam = blue_n["barrels_per_burst"] * pellets_per_shot * p_pel * 0.80 * energy_fac
        lam = min(lam, 60.0)
        p_surv = 0.0
        term = math.exp(-lam)
        for k in range(n_crit):
            p_surv += term
            term *= lam / float(k + 1)
        return float(max(0.0, min(1.0, 1.0 - p_surv)))

    # 4. Event-Driven Swarm Timeline Across N Episodes
    zero_leaker_eps = 0
    killed_counts: list[int] = []
    leaker_counts: list[int] = []
    rounds_counts: list[int] = []
    reloads_counts: list[int] = []
    burst_pks: list[float] = []
    intercept_ranges: list[float] = []

    eff_engage_range = min(
        float(blue_n["max_engagement_range_m"]),
        float(blue_n["max_engagement_range_m"]) * (1.0 - 0.18 * float(red_n["fiber_optic_fraction"])) + 1.5,
    )
    eff_engage_range = max(breach_keepout_radius_m + 6.0, eff_engage_range)

    n_barrels = int(blue_n["n_barrels"])
    bpb = int(blue_n["barrels_per_burst"])
    rel_t = float(blue_n["reload_time_s"])
    settle_t = 0.08 + float(blue_n["ripple_delay_ms"]) * max(0, bpb - 1) / 2500.0
    slew_rate = float(blue_n["slew_rate_deg_s"])
    slew_accel = max(420.0, slew_rate * 4.0)
    pol = blue_n["priority_policy"]
    nom_spd = float(red_n["drone_speed_mps"])

    for _ in range(n_episodes):
        drones: list[dict[str, Any]] = []
        for _di in range(int(red_n["swarm_size"])):
            if red_n["attack_tactic"] == "narrow_corridor_saturation":
                az_d = float(rng.uniform(-24.0, 24.0))
                el_d = float(rng.uniform(10.0, 30.0))
                r0 = float(eff_engage_range + 4.0 + _di * 0.45 * nom_spd + rng.uniform(0.0, 12.0))
            elif red_n["attack_tactic"] == "staggered_multi_altitude":
                az_d = float(rng.uniform(-90.0, 90.0))
                el_d = float(rng.choice([12.0, 34.0, 52.0]) + rng.uniform(-4.0, 4.0))
                r0 = float(eff_engage_range + 8.0 + _di * 0.72 * nom_spd + rng.uniform(0.0, 16.0))
            else:
                az_d = float(rng.uniform(-125.0, 125.0))
                el_d = float(rng.uniform(10.0, 48.0))
                r0 = float(eff_engage_range + 6.0 + _di * 0.62 * nom_spd + rng.uniform(0.0, 14.0))
            spd_d = float(max(8.0, rng.normal(nom_spd, 0.10 * nom_spd)))
            drones.append({"az": az_d, "el": el_d, "r0": r0, "spd": spd_d, "alive": True, "leaked": False})

        t_now = 0.0
        b_ready = n_barrels
        tur_az, tur_el = 0.0, 24.0
        ep_reloads = 0
        ep_rounds = 0

        while True:
            for dr in drones:
                if dr["alive"] and (dr["r0"] - dr["spd"] * t_now) <= breach_keepout_radius_m:
                    dr["alive"] = False
                    dr["leaked"] = True

            alive = [dr for dr in drones if dr["alive"]]
            if not alive:
                break

            if b_ready <= 0:
                t_now += rel_t
                b_ready = n_barrels
                ep_reloads += 1
                continue

            in_range = [dr for dr in alive if (dr["r0"] - dr["spd"] * t_now) <= eff_engage_range]
            if not in_range:
                t_ent = min((dr["r0"] - eff_engage_range) / dr["spd"] for dr in alive)
                t_now = max(t_now, t_ent + 1e-3)
                in_range = [dr for dr in alive if (dr["r0"] - dr["spd"] * t_now) <= eff_engage_range]
                if not in_range:
                    continue

            if pol == "nearest_slant_range":
                tgt = min(in_range, key=lambda d: d["r0"] - d["spd"] * t_now)
            elif pol == "minimum_slew_angle":
                tgt = min(
                    in_range,
                    key=lambda d: abs(((d["az"] - tur_az + 180.0) % 360.0) - 180.0) + abs(d["el"] - tur_el),
                )
            else:
                tgt = min(
                    in_range,
                    key=lambda d: max(0.0, (d["r0"] - d["spd"] * t_now) - breach_keepout_radius_m) / max(d["spd"], 1.0),
                )

            d_az = ((tgt["az"] - tur_az + 180.0) % 360.0) - 180.0
            d_el = tgt["el"] - tur_el
            t_slew = _trapezoidal_slew_time(math.hypot(d_az, d_el), slew_rate, slew_accel)
            t_now += t_slew + settle_t
            tur_az, tur_el = tgt["az"], tgt["el"]

            r_fire = tgt["r0"] - tgt["spd"] * t_now
            if r_fire <= breach_keepout_radius_m:
                tgt["alive"] = False
                tgt["leaked"] = True
                continue

            rounds_fired = min(b_ready, bpb)
            b_ready -= rounds_fired
            ep_rounds += rounds_fired

            ray_draw = float(rng.rayleigh(1.0))
            pk_burst = _sample_burst_pk(r_fire, tgt["spd"], ray_draw)
            burst_pks.append(pk_burst)
            if rng.random() < pk_burst:
                tgt["alive"] = False
                intercept_ranges.append(r_fire)

        n_leaked = sum(1 for d in drones if d["leaked"])
        n_killed = int(red_n["swarm_size"]) - n_leaked
        if n_leaked == 0:
            zero_leaker_eps += 1
        killed_counts.append(n_killed)
        leaker_counts.append(n_leaked)
        rounds_counts.append(ep_rounds)
        reloads_counts.append(ep_reloads)

    zero_breach_pct = 100.0 * zero_leaker_eps / max(1, n_episodes)
    kill_rate_pct = 100.0 * float(np.mean(killed_counts)) / max(1, int(red_n["swarm_size"]))
    leaker_rate_pct = 100.0 - kill_rate_pct
    mean_cost_wave = float(np.mean(rounds_counts)) * unit_cost_usd
    mean_killed = float(np.mean(killed_counts))
    cost_per_kill = mean_cost_wave / max(mean_killed, 0.25)

    recoil_pen = recoil_penalty_per_kn * max(0.0, peak_mount_kn - mount_load_limit_kn)
    raw_utility = (
        w_zero_breach * zero_breach_pct
        + w_kill_rate * kill_rate_pct
        - cost_penalty_weight * mean_cost_wave
        - recoil_pen
    )
    blue_utility = float(max(0.0, min(100.0, raw_utility)))

    return {
        "blue_utility": round(blue_utility, 2),
        "utility_score": round(blue_utility, 2),
        "red_utility": round(100.0 - blue_utility, 2),
        "zero_breach_pct": round(zero_breach_pct, 1),
        "breach_rate_pct": round(100.0 - zero_breach_pct, 1),
        "kill_rate_pct": round(kill_rate_pct, 1),
        "leaker_rate_pct": round(leaker_rate_pct, 1),
        "mean_burst_pk_pct": round(100.0 * float(np.mean(burst_pks)), 1) if burst_pks else 0.0,
        "peak_mount_force_kn": round(peak_mount_kn, 2),
        "mount_feasible": mount_feasible,
        "mean_cost_per_wave_usd": round(mean_cost_wave, 2),
        "cost_per_kill_usd": round(cost_per_kill, 2),
        "mean_reloads_per_wave": round(float(np.mean(reloads_counts)), 2),
        "mean_intercept_range_m": round(float(np.mean(intercept_ranges)), 1) if intercept_ranges else 0.0,
        "presented_area_cm2": round(area_vuln * 10000.0, 1),
    }


# -------------------------------------------------------------------------
# CANDIDATE SAMPLERS & CO-EVOLUTION BAYESIAN ML LOOP
# -------------------------------------------------------------------------


def _sample_blue_candidates(
    n_candidates: int,
    max_barrels: int,
    elite_pool: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    n_mutate = n_candidates // 2
    for idx in range(n_candidates):
        if idx < n_mutate and elite_pool:
            base = dict(elite_pool[int(rng.integers(0, len(elite_pool)))])
            base["name"] = ""
            n_b = int(base["n_barrels"]) + int(rng.choice([-2, 0, 2]))
            bpb = int(base["barrels_per_burst"]) + int(rng.choice([-1, 0, 1]))
            base["n_barrels"] = max(2, min(max_barrels, n_b))
            base["barrels_per_burst"] = max(1, min(base["n_barrels"], min(6, bpb)))
            if rng.random() < 0.35:
                base["ammunition_preset"] = str(rng.choice(BLUE_SEARCH_SPACE["ammunition_preset"]))
            if rng.random() < 0.35:
                base["choke"] = str(rng.choice(BLUE_SEARCH_SPACE["choke"]))
            if rng.random() < 0.40:
                base["pointing_mode"] = str(rng.choice(BLUE_SEARCH_SPACE["pointing_mode"]))
            if base["pointing_mode"] == "divergent_curtain":
                base["splay_angle_mrad"] = float(np.clip(float(base["splay_angle_mrad"]) + rng.normal(0.0, 2.0), 1.5, 9.5))
            if rng.random() < 0.35:
                base["firing_mode"] = str(rng.choice(BLUE_SEARCH_SPACE["firing_mode"]))
            if base["firing_mode"] != "simultaneous_volley":
                base["ripple_delay_ms"] = float(np.clip(float(base.get("ripple_delay_ms", 8.0)) + rng.normal(0.0, 4.0), 3.0, 18.0))
            base["max_engagement_range_m"] = float(np.clip(float(base["max_engagement_range_m"]) + rng.normal(0.0, 8.0), 40.0, 80.0))
            base["slew_rate_deg_s"] = float(np.clip(float(base["slew_rate_deg_s"]) + rng.normal(0.0, 25.0), 100.0, 260.0))
            base["reload_time_s"] = float(np.clip(float(base["reload_time_s"]) + rng.normal(0.0, 0.7), 1.8, 5.2))
            cands.append(normalize_blue_config(base, max_barrels=max_barrels))
        else:
            n_b = int(rng.integers(2, max_barrels + 1))
            bpb = int(rng.integers(1, min(n_b, 6) + 1))
            pm = str(rng.choice(BLUE_SEARCH_SPACE["pointing_mode"]))
            fm = str(rng.choice(BLUE_SEARCH_SPACE["firing_mode"]))
            raw = {
                "layout_type": str(rng.choice(BLUE_SEARCH_SPACE["layout_type"])),
                "n_barrels": n_b,
                "barrels_per_burst": bpb,
                "ammunition_preset": str(rng.choice(BLUE_SEARCH_SPACE["ammunition_preset"])),
                "choke": str(rng.choice(BLUE_SEARCH_SPACE["choke"])),
                "pointing_mode": pm,
                "splay_angle_mrad": float(rng.uniform(2.0, 9.0)) if pm == "divergent_curtain" else 0.0,
                "firing_mode": fm,
                "ripple_delay_ms": float(rng.uniform(4.0, 16.0)) if fm != "simultaneous_volley" else 0.0,
                "sensor_preset": str(rng.choice(BLUE_SEARCH_SPACE["sensor_preset"])),
                "predictor_mode": str(rng.choice(BLUE_SEARCH_SPACE["predictor_mode"])),
                "max_engagement_range_m": float(rng.uniform(42.0, 79.0)),
                "slew_rate_deg_s": float(rng.uniform(110.0, 255.0)),
                "reload_time_s": float(rng.uniform(1.8, 5.0)),
                "priority_policy": str(rng.choice(BLUE_SEARCH_SPACE["priority_policy"])),
            }
            cands.append(normalize_blue_config(raw, max_barrels=max_barrels))
    return cands


def _sample_red_candidates(
    n_candidates: int,
    max_swarm_size: int,
    elite_pool: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    n_mutate = n_candidates // 2
    for idx in range(n_candidates):
        if idx < n_mutate and elite_pool:
            base = dict(elite_pool[int(rng.integers(0, len(elite_pool)))])
            base["name"] = ""
            sw = int(base["swarm_size"]) + int(rng.choice([-2, 0, 1, 2]))
            base["swarm_size"] = max(1, min(max_swarm_size, sw))
            if rng.random() < 0.35:
                base["drone_model"] = str(rng.choice(RED_SEARCH_SPACE["drone_model"]))
            if rng.random() < 0.35:
                base["maneuver_profile"] = str(rng.choice(RED_SEARCH_SPACE["maneuver_profile"]))
            if rng.random() < 0.35:
                base["attack_tactic"] = str(rng.choice(RED_SEARCH_SPACE["attack_tactic"]))
            base["maneuver_g"] = float(np.clip(float(base["maneuver_g"]) + rng.normal(0.0, 0.8), 0.2, 4.5))
            base["drone_speed_mps"] = float(np.clip(float(base["drone_speed_mps"]) + rng.normal(0.0, 5.0), 12.0, 42.0))
            base["fiber_optic_fraction"] = float(np.clip(float(base["fiber_optic_fraction"]) + rng.normal(0.0, 0.25), 0.0, 1.0))
            if rng.random() < 0.30:
                base["aspect_mode"] = str(rng.choice(RED_SEARCH_SPACE["aspect_mode"]))
            cands.append(normalize_red_config(base, max_swarm_size=max_swarm_size))
        else:
            man = str(rng.choice(RED_SEARCH_SPACE["maneuver_profile"]))
            raw = {
                "drone_model": str(rng.choice(RED_SEARCH_SPACE["drone_model"])),
                "swarm_size": int(rng.integers(2, max_swarm_size + 1)),
                "attack_tactic": str(rng.choice(RED_SEARCH_SPACE["attack_tactic"])),
                "maneuver_profile": man,
                "maneuver_g": 0.0 if man == "linear_cruise" else float(rng.uniform(0.8, 4.5)),
                "drone_speed_mps": float(rng.uniform(14.0, 41.0)),
                "aspect_mode": str(rng.choice(RED_SEARCH_SPACE["aspect_mode"])),
                "fiber_optic_fraction": float(rng.uniform(0.0, 1.0)),
            }
            cands.append(normalize_red_config(raw, max_swarm_size=max_swarm_size))
    return cands


def run_adversarial_coevolution_ml(
    n_generations: int = 5,
    n_mc_episodes: int = 18,
    mount_load_limit_kn: float = 3.5,
    max_barrels: int = 10,
    max_swarm_size: int = 8,
    exploration_beta: float = 1.25,
    w_zero_breach: float = 0.55,
    w_kill_rate: float = 0.45,
    cost_penalty_weight: float = 0.08,
    stage1_profile: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run adversarial Red-vs-Blue co-evolution backed by GP Bayesian optimization & LP Nash solver."""
    rng = np.random.default_rng(seed)
    recoil_cache: dict[tuple[Any, ...], float] = {}
    geom_cache: dict[tuple[str, str], float] = {}

    blue_pool = get_default_blue_seed_pool(max_barrels=max_barrels, stage1_profile=stage1_profile)
    red_pool = get_default_red_seed_pool(max_swarm_size=max_swarm_size)
    n_base_blue = len(blue_pool)

    eval_cache: dict[tuple[int, int], dict[str, Any]] = {}
    dataset_records: list[dict[str, Any]] = []
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    def _eval_pair(bi: int, rj: int, gen_idx: int) -> dict[str, Any]:
        if (bi, rj) in eval_cache:
            return eval_cache[(bi, rj)]
        b_cfg = blue_pool[bi]
        r_cfg = red_pool[rj]
        m_res = evaluate_red_vs_blue_matchup_mc(
            blue=b_cfg,
            red=r_cfg,
            n_episodes=n_mc_episodes,
            mount_load_limit_kn=mount_load_limit_kn,
            w_zero_breach=w_zero_breach,
            w_kill_rate=w_kill_rate,
            cost_penalty_weight=cost_penalty_weight,
            seed=seed + bi * 97 + rj * 31,
            _recoil_cache=recoil_cache,
            _geom_cache=geom_cache,
        )
        eval_cache[(bi, rj)] = m_res
        feat_vec = encode_matchup_features(b_cfg, r_cfg)
        X_list.append(feat_vec)
        y_list.append(float(m_res["blue_utility"]))
        dataset_records.append(
            {
                "generation": gen_idx,
                "blue_idx": bi,
                "blue_name": b_cfg["name"],
                "blue_layout": b_cfg["layout_type"],
                "blue_barrels": b_cfg["n_barrels"],
                "blue_burst": b_cfg["barrels_per_burst"],
                "blue_ammo": b_cfg["ammunition_preset"],
                "blue_choke": b_cfg["choke"],
                "blue_pointing": b_cfg["pointing_mode"],
                "blue_splay_mrad": b_cfg["splay_angle_mrad"],
                "blue_firing": b_cfg["firing_mode"],
                "blue_sensor": b_cfg["sensor_preset"],
                "red_idx": rj,
                "red_name": r_cfg["name"],
                "red_drone": r_cfg["drone_model"],
                "red_swarm_size": r_cfg["swarm_size"],
                "red_tactic": r_cfg["attack_tactic"],
                "red_maneuver": r_cfg["maneuver_profile"],
                "red_maneuver_g": r_cfg["maneuver_g"],
                "red_speed_mps": r_cfg["drone_speed_mps"],
                "red_fiber_optic_frac": r_cfg["fiber_optic_fraction"],
                "blue_utility": m_res["blue_utility"],
                "utility_score": m_res["blue_utility"],
                "zero_breach_pct": m_res["zero_breach_pct"],
                "kill_rate_pct": m_res["kill_rate_pct"],
                "leaker_rate_pct": m_res["leaker_rate_pct"],
                "peak_mount_force_kn": m_res["peak_mount_force_kn"],
                "mount_feasible": m_res["mount_feasible"],
                "cost_per_wave_usd": m_res["mean_cost_per_wave_usd"],
            }
        )
        return m_res

    def _build_payoff_matrix() -> np.ndarray:
        mat = np.zeros((len(blue_pool), len(red_pool)), dtype=np.float64)
        for bi in range(len(blue_pool)):
            for rj in range(len(red_pool)):
                mat[bi, rj] = eval_cache[(bi, rj)]["blue_utility"]
        return mat

    # Gen 0: Evaluate initial Blue x Red matrix
    for bi in range(len(blue_pool)):
        for rj in range(len(red_pool)):
            _eval_pair(bi, rj, gen_idx=0)

    gp = GaussianProcessSurrogate()
    gen_history: list[dict[str, Any]] = []

    mat0 = _build_payoff_matrix()
    nash0 = solve_zero_sum_nash_equilibrium(mat0)
    base_worst_idx = int(np.argmax(np.min(mat0, axis=1)))
    base_worst_val = float(np.min(mat0[base_worst_idx, :]))
    gen_history.append(
        {
            "generation": 0,
            "n_blue_pool": len(blue_pool),
            "n_red_pool": len(red_pool),
            "total_matchups_evaluated": len(X_list),
            "nash_value": nash0["nash_value"],
            "best_blue_worst_case": round(base_worst_val, 2),
            "baseline_blue_worst_case": round(base_worst_val, 2),
            "deadliest_red_min_score": round(float(np.min(mat0[:, int(np.argmin(mat0[base_worst_idx, :]))])), 2),
            "new_blue": "Initial Seed Pool",
            "new_red": "Initial Threat Pool",
        }
    )

    for gen in range(1, max(1, int(n_generations)) + 1):
        X_arr = np.vstack(X_list)
        y_arr = np.asarray(y_list, dtype=np.float64)
        gp.fit(X_arr, y_arr, optimize_hyperparams=True)

        mat_cur = _build_payoff_matrix()
        nash_cur = solve_zero_sum_nash_equilibrium(mat_cur)
        q_red_weights = np.asarray(nash_cur["red_strategy"], dtype=np.float64)

        # Rank current elite Blue designs to seed mutations
        b_min_scores = np.min(mat_cur, axis=1)
        top_b_indices = np.argsort(-b_min_scores)[:3]
        elite_blue = [blue_pool[int(i)] for i in top_b_indices]

        # Score 120 Blue candidates against all current Red threats using GP UCB minimax objective
        blue_cands = _sample_blue_candidates(120, max_barrels=max_barrels, elite_pool=elite_blue, rng=rng)
        seen_blue_names = {b["name"] for b in blue_pool}
        novel_blue_cands = [b for b in blue_cands if b["name"] not in seen_blue_names] or blue_cands

        X_b_batch = np.vstack([encode_matchup_features(bc, rc) for bc in novel_blue_cands for rc in red_pool])
        mu_b, std_b = gp.predict(X_b_batch, return_std=True)
        assert std_b is not None
        mu_b_mat = mu_b.reshape(len(novel_blue_cands), len(red_pool))
        std_b_mat = std_b.reshape(len(novel_blue_cands), len(red_pool))
        ucb_b_mat = mu_b_mat + float(exploration_beta) * std_b_mat
        # Blend minimax worst-case UCB with Nash-expected UCB for robust optimization
        b_acq_scores = 0.65 * np.min(ucb_b_mat, axis=1) + 0.35 * (ucb_b_mat @ q_red_weights)
        best_bc = novel_blue_cands[int(np.argmax(b_acq_scores))]
        best_bc["name"] = f"G{gen} {best_bc['name']}"
        blue_pool.append(best_bc)
        new_bi = len(blue_pool) - 1
        for rj in range(len(red_pool)):
            _eval_pair(new_bi, rj, gen_idx=gen)

        # Refit GP quickly with new Blue observations, then pick deadliest Red counter-strategy
        X_arr = np.vstack(X_list)
        y_arr = np.asarray(y_list, dtype=np.float64)
        gp.fit(X_arr, y_arr, optimize_hyperparams=False)

        mat_mid = _build_payoff_matrix()
        nash_mid = solve_zero_sum_nash_equilibrium(mat_mid)
        p_blue_weights = np.asarray(nash_mid["blue_strategy"], dtype=np.float64)

        r_max_scores = np.min(mat_mid, axis=0)
        top_r_indices = np.argsort(r_max_scores)[:3]
        elite_red = [red_pool[int(j)] for j in top_r_indices]

        red_cands = _sample_red_candidates(120, max_swarm_size=max_swarm_size, elite_pool=elite_red, rng=rng)
        seen_red_names = {r["name"] for r in red_pool}
        novel_red_cands = [r for r in red_cands if r["name"] not in seen_red_names] or red_cands

        X_r_batch = np.vstack([encode_matchup_features(bc, rc) for rc in novel_red_cands for bc in blue_pool])
        mu_r, std_r = gp.predict(X_r_batch, return_std=True)
        assert std_r is not None
        mu_r_mat = mu_r.reshape(len(novel_red_cands), len(blue_pool))
        std_r_mat = std_r.reshape(len(novel_red_cands), len(blue_pool))
        # Red seeks LOWEST Blue utility (with UCB bonus for uncertain exploits)
        red_lcb_score = -(mu_r_mat @ p_blue_weights) + float(exploration_beta) * (std_r_mat @ p_blue_weights)
        best_rc = novel_red_cands[int(np.argmax(red_lcb_score))]
        best_rc["name"] = f"G{gen} {best_rc['name']}"
        red_pool.append(best_rc)
        new_rj = len(red_pool) - 1
        for bi in range(len(blue_pool)):
            _eval_pair(bi, new_rj, gen_idx=gen)

        mat_end = _build_payoff_matrix()
        nash_end = solve_zero_sum_nash_equilibrium(mat_end)
        best_evolved_worst = float(np.max(np.min(mat_end, axis=1)))
        baseline_worst_now = float(np.max(np.min(mat_end[:n_base_blue, :], axis=1)))
        gen_history.append(
            {
                "generation": gen,
                "n_blue_pool": len(blue_pool),
                "n_red_pool": len(red_pool),
                "total_matchups_evaluated": len(X_list),
                "nash_value": nash_end["nash_value"],
                "best_blue_worst_case": round(best_evolved_worst, 2),
                "baseline_blue_worst_case": round(baseline_worst_now, 2),
                "deadliest_red_min_score": round(float(np.min(mat_end[:n_base_blue, new_rj])), 2),
                "new_blue": best_bc["name"],
                "new_red": best_rc["name"],
            }
        )

    # Final GP fit, 5-fold cross-validation & feature importance audit
    X_final = np.vstack(X_list)
    y_final = np.asarray(y_list, dtype=np.float64)
    gp.fit(X_final, y_final, optimize_hyperparams=True)
    cv_metrics = gp.cross_validate(X_final, y_final, k_folds=5, seed=seed)
    feat_importance = gp.compute_permutation_importance(
        X_final, y_final, FEATURE_NAMES, FEATURE_SIDES, n_repeats=4, seed=seed
    )

    final_matrix = _build_payoff_matrix()
    final_nash = solve_zero_sum_nash_equilibrium(final_matrix)
    best_blue_idx = int(final_nash["pure_minimax_blue_idx"])
    best_blue = blue_pool[best_blue_idx]

    # Identify deadliest Red threat (lowest average score against baseline pool)
    red_vs_base_mean = np.mean(final_matrix[:n_base_blue, :], axis=0)
    deadliest_red_idx = int(np.argmin(red_vs_base_mean))
    deadliest_red = red_pool[deadliest_red_idx]

    baseline_worst_final = float(np.max(np.min(final_matrix[:n_base_blue, :], axis=1)))
    evolved_worst_final = float(final_nash["pure_minimax_blue_value"])

    return {
        "blue_pool": blue_pool,
        "red_pool": red_pool,
        "n_blue_pool": len(blue_pool),
        "n_red_pool": len(red_pool),
        "matchups_evaluated": len(y_final),
        "payoff_matrix": [[round(float(v), 2) for v in row] for row in final_matrix],
        "nash": final_nash,
        "nash_value": final_nash["nash_value"],
        "nash_blue_strategy": final_nash["blue_strategy"],
        "nash_red_strategy": final_nash["red_strategy"],
        "best_blue_config": best_blue,
        "deadliest_red_threat": deadliest_red,
        "worst_case_defense_gain": {
            "baseline_worst_case": round(baseline_worst_final, 2),
            "evolved_worst_case": round(evolved_worst_final, 2),
            "worst_case_gain_pct": round(evolved_worst_final - baseline_worst_final, 2),
        },
        "gp_surrogate_diagnostics": cv_metrics,
        "generation_history": gen_history,
        "cv_metrics": cv_metrics,
        "actual_utilities": [round(float(v), 2) for v in y_final],
        "feature_importance": feat_importance,
        "surrogate_bundle": gp.to_bundle(X_final, y_final),
        "dataset_records": dataset_records,
        "summary": {
            "n_generations": n_generations,
            "n_mc_episodes": n_mc_episodes,
            "total_matchups": len(y_final),
            "n_blue_platforms": len(blue_pool),
            "n_red_threats": len(red_pool),
            "nash_value": final_nash["nash_value"],
            "duality_gap": final_nash["duality_gap"],
            "baseline_worst_case": round(baseline_worst_final, 2),
            "evolved_worst_case": round(evolved_worst_final, 2),
            "worst_case_gain": round(evolved_worst_final - baseline_worst_final, 2),
            "best_blue_idx": best_blue_idx,
            "best_blue_name": best_blue["name"],
            "best_blue_config": best_blue,
            "deadliest_red_idx": deadliest_red_idx,
            "deadliest_red_name": deadliest_red["name"],
            "deadliest_red_config": deadliest_red,
            "deadliest_red_mean_score": round(float(red_vs_base_mean[deadliest_red_idx]), 2),
            "cv_r2": cv_metrics["cv_r2"],
            "cv_rmse": cv_metrics["cv_rmse"],
            "cv_mae": cv_metrics["cv_mae"],
            "mount_load_limit_kn": mount_load_limit_kn,
            "max_barrels": max_barrels,
            "max_swarm_size": max_swarm_size,
        },
    }


def recommend_ml_defense_policy(
    adv_result: dict[str, Any] | None = None,
    target_red_threat: dict[str, Any] | None = None,
    mount_load_limit_kn: float = 3.5,
    max_barrels: int = 10,
    n_candidates: int = 160,
    verify_top_mc: bool = True,
    n_mc_episodes: int = 20,
    seed: int = 99,
    coevolution_result: dict[str, Any] | None = None,
    target_threat: dict[str, Any] | None = None,
    top_k: int = 5,
    verify_with_mc: bool | None = None,
) -> dict[str, Any]:
    """Use the trained GP surrogate to instantly recommend & optionally verify the best Blue turret."""
    actual_res = coevolution_result if coevolution_result is not None else adv_result
    if actual_res is None:
        raise ValueError("Either adv_result or coevolution_result must be provided.")
    actual_threat = target_threat if target_threat is not None else target_red_threat
    if actual_threat is None:
        raise ValueError("Either target_red_threat or target_threat must be provided.")
    actual_verify = verify_with_mc if verify_with_mc is not None else verify_top_mc

    gp = GaussianProcessSurrogate.from_bundle(actual_res["surrogate_bundle"])
    red_norm = normalize_red_config(actual_threat, max_swarm_size=12)
    rng = np.random.default_rng(seed)

    existing_pool = [normalize_blue_config(b, max_barrels=max_barrels) for b in actual_res.get("blue_pool", [])]
    sampled = _sample_blue_candidates(max(20, int(n_candidates)), max_barrels=max_barrels, elite_pool=existing_pool, rng=rng)
    all_cands: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in existing_pool + sampled:
        key = (
            f"{c['layout_type']}|{c['n_barrels']}|{c['barrels_per_burst']}|{c['ammunition_preset']}|"
            f"{c['choke']}|{c['pointing_mode']}|{c['splay_angle_mrad']}|{c['firing_mode']}|"
            f"{c['ripple_delay_ms']}|{c['sensor_preset']}|{c['predictor_mode']}|{c['max_engagement_range_m']}|"
            f"{c['slew_rate_deg_s']}|{c['reload_time_s']}|{c['priority_policy']}"
        )
        if key not in seen:
            seen.add(key)
            all_cands.append(c)

    X_q = np.vstack([encode_matchup_features(bc, red_norm) for bc in all_cands])
    mu_q, std_q = gp.predict(X_q, return_std=True)
    assert std_q is not None

    recoil_cache: dict[tuple[Any, ...], float] = {}
    ranked_rows: list[dict[str, Any]] = []
    for idx, bc in enumerate(all_cands):
        rdt_bucket = round(float(bc["ripple_delay_ms"]), 0)
        rec_key = (
            int(bc["barrels_per_burst"]),
            bc["ammunition_preset"],
            bc["firing_mode"],
            rdt_bucket,
        )
        if rec_key in recoil_cache:
            peak_kn = recoil_cache[rec_key]
        else:
            b_salvo = generate_multishotgun_array_layout(
                layout_type=bc["layout_type"],
                n_barrels=int(bc["barrels_per_burst"]),
                mount_radius_m=0.20,
                default_choke=bc["choke"],
                default_barrel_length_in=24.0,
                default_ammunition_preset=bc["ammunition_preset"],
            )
            rec_res = simulate_turret_recoil_and_gimbal_loads(
                barrels=b_salvo,
                firing_mode=bc["firing_mode"],
                ripple_delay_ms=float(rdt_bucket),
                default_ammunition_preset=bc["ammunition_preset"],
                mount_mass_kg=45.0,
                mount_stiffness_n_per_m=120000.0,
                mount_load_limit_kn=mount_load_limit_kn,
                gimbal_trunnion_offset_z_m=0.18,
            )
            peak_kn = float(rec_res["summary"]["peak_transmitted_mount_force_kn"])
            recoil_cache[rec_key] = peak_kn

        feasible = bool(peak_kn <= mount_load_limit_kn)
        pred_u = float(mu_q[idx])
        pred_sd = float(std_q[idx])
        econ = CUAS_AMMO_ECONOMICS_CATALOG.get(bc["ammunition_preset"], CUAS_AMMO_ECONOMICS_CATALOG["12G_NO4_BUCK_C_UAS"])
        est_wave_cost = float(econ["unit_cost_usd"]) * bc["n_barrels"]
        penalized_score = pred_u - (12.0 + 3.0 * max(0.0, peak_kn - mount_load_limit_kn) if not feasible else 0.0)

        ranked_rows.append(
            {
                "name": bc["name"],
                "ammunition_preset": bc["ammunition_preset"],
                "choke": bc["choke"],
                "layout_type": bc["layout_type"],
                "n_barrels": bc["n_barrels"],
                "barrels_per_burst": bc["barrels_per_burst"],
                "pointing_mode": bc["pointing_mode"],
                "splay_angle_mrad": bc["splay_angle_mrad"],
                "firing_mode": bc["firing_mode"],
                "ripple_delay_ms": bc["ripple_delay_ms"],
                "sensor_preset": bc["sensor_preset"],
                "predictor_mode": bc["predictor_mode"],
                "pred_utility": round(pred_u, 2),
                "pred_std": round(pred_sd, 2),
                "ci_95_low": round(max(0.0, pred_u - 1.96 * pred_sd), 2),
                "ci_95_high": round(min(100.0, pred_u + 1.96 * pred_sd), 2),
                "peak_mount_force_kn": round(peak_kn, 2),
                "feasible": feasible,
                "est_salvo_cost_usd": round(est_wave_cost, 2),
                "_rank_score": penalized_score,
                "_config": bc,
            }
        )

    ranked_rows.sort(key=lambda r: r["_rank_score"], reverse=True)
    top_row = ranked_rows[0]
    top_cfg = dict(top_row["_config"])

    verified_mc = None
    if actual_verify:
        verified_mc = evaluate_red_vs_blue_matchup_mc(
            blue=top_cfg,
            red=red_norm,
            n_episodes=n_mc_episodes,
            mount_load_limit_kn=mount_load_limit_kn,
            seed=seed + 5,
        )

    clean_candidates = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ranked_rows]

    recommended_policies = []
    for idx, r in enumerate(ranked_rows[:max(1, top_k)]):
        cfg = r["_config"]
        policy_dict = {
            "blue_config": {
                "name": cfg["name"],
                "layout": cfg["layout_type"],
                "n_barrels": cfg["n_barrels"],
                "ammo": cfg["ammunition_preset"],
                "choke": cfg["choke"],
                "pointing": cfg["pointing_mode"],
                "splay_angle_deg": round(math.degrees(cfg["splay_angle_mrad"] / 1000.0), 1),
                "firing_mode": cfg["firing_mode"],
                "ripple_delay_ms": cfg["ripple_delay_ms"],
                "burst_count": cfg.get("burst_count", 1),
            },
            "predicted_defense_score": r["pred_utility"],
            "predicted_std": r["pred_std"],
            "predicted_mount_load_kn": r["peak_mount_force_kn"],
            "feasible": r["feasible"],
            "mc_verified": {
                "kill_rate": verified_mc["kill_rate_pct"] / 100.0,
                "zero_breach_rate": verified_mc["zero_breach_pct"] / 100.0,
                "peak_mount_load_kn": verified_mc["peak_mount_force_kn"],
            }
            if (idx == 0 and verified_mc is not None)
            else {},
        }
        recommended_policies.append(policy_dict)

    return {
        "target_red_threat": red_norm,
        "recommended_config": top_cfg,
        "recommended_summary": {k: v for k, v in top_row.items() if not k.startswith("_")},
        "recommended_policies": recommended_policies,
        "top_5_policies": clean_candidates[:5],
        "all_candidates": clean_candidates,
        "verified_mc": verified_mc,
        "mount_load_limit_kn": mount_load_limit_kn,
    }


def export_adversarial_dataset_csv(dataset_records: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Serialize simulated Red-vs-Blue Monte Carlo episodes into CSV format."""
    if isinstance(dataset_records, dict):
        dataset_records = dataset_records.get("dataset_records", [])
    if not dataset_records:
        return ""
    buf = io.StringIO()
    fieldnames = list(dataset_records[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in dataset_records:
        writer.writerow(row)
    return buf.getvalue()
