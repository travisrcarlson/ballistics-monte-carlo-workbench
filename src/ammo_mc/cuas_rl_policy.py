"""Pure-NumPy Dynamic Fire-Control ML Policy (Natural Evolution Strategies) for C-UAS Turrets.

Trains a lightweight neural controller that decides shot-by-shot during an active drone swarm:
  1. Target prioritization score across inbound threats,
  2. Adaptive burst size (1 to max barrels/burst) balancing P_kill against magazine conservation,
  3. Hold-fire gating when a long-range maneuvering drone can be waited into a tighter shot envelope.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ammo_mc.aero_sphere import exact_quadratic_drag_tof_and_velocity
from ammo_mc.cuas_adversarial import normalize_blue_config, normalize_red_config
from ammo_mc.cuas_defense import (
    CUAS_SENSOR_PRESETS,
    _estimate_burst_pk_analytics,
    _trapezoidal_slew_time,
)
from ammo_mc.cuas_mission import _resolve_shotgun_ammo
from ammo_mc.small_arms import (
    compute_drone_projected_geometry,
    get_choke_definitions,
    get_drone_target_specs,
)

RL_FEATURE_NAMES: list[str] = [
    "norm_slant_range",
    "norm_time_to_breach",
    "norm_slew_time",
    "norm_target_speed",
    "norm_maneuver_g",
    "ammo_ready_fraction",
    "alive_swarm_fraction",
    "est_single_barrel_pk",
]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class NeuralFireControlPolicy:
    """Two-layer tanh MLP (8 -> 12 -> 3, 147 weights) for real-time C-UAS fire control."""

    INPUT_DIM = 8
    HIDDEN_DIM = 12
    OUTPUT_DIM = 3

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.25, size=(self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float64)
        self.b1 = np.zeros(self.HIDDEN_DIM, dtype=np.float64)
        self.W2 = np.zeros((self.HIDDEN_DIM, self.OUTPUT_DIM), dtype=np.float64)
        self.b2 = np.array([0.0, -0.25, -0.85], dtype=np.float64)

    @property
    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_flat_params(self, theta: np.ndarray) -> None:
        vec = np.asarray(theta, dtype=np.float64).ravel()
        if vec.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {vec.size}.")
        idx = 0
        s1 = self.INPUT_DIM * self.HIDDEN_DIM
        self.W1 = vec[idx : idx + s1].reshape((self.INPUT_DIM, self.HIDDEN_DIM)).copy()
        idx += s1
        self.b1 = vec[idx : idx + self.HIDDEN_DIM].copy()
        idx += self.HIDDEN_DIM
        s2 = self.HIDDEN_DIM * self.OUTPUT_DIM
        self.W2 = vec[idx : idx + s2].reshape((self.HIDDEN_DIM, self.OUTPUT_DIM)).copy()
        idx += s2
        self.b2 = vec[idx : idx + self.OUTPUT_DIM].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cuas_neural_fc_policy_v1",
            "input_dim": self.INPUT_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "output_dim": self.OUTPUT_DIM,
            "feature_names": list(RL_FEATURE_NAMES),
            "params": [round(float(v), 6) for v in self.get_flat_params()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuralFireControlPolicy":
        pol = cls()
        pol.set_flat_params(np.asarray(payload["params"], dtype=np.float64))
        return pol

    @staticmethod
    def build_target_features(
        slant_range_m: float,
        max_engage_range_m: float,
        ttb_s: float,
        slew_time_s: float,
        speed_mps: float,
        maneuver_g: float,
        b_ready: int,
        n_barrels: int,
        n_alive: int,
        swarm_size: int,
        est_pk_1bbl: float,
    ) -> np.ndarray:
        """Normalize operational target & turret states onto bounded ML inputs."""
        return np.array(
            [
                min(1.5, max(0.0, float(slant_range_m) / max(float(max_engage_range_m), 1.0))),
                min(1.0, max(0.0, float(ttb_s) / 5.0)),
                min(1.0, max(0.0, float(slew_time_s) / 1.5)),
                min(1.0, max(0.0, float(speed_mps) / 45.0)),
                min(1.0, max(0.0, float(maneuver_g) / 4.5)),
                min(1.0, max(0.0, float(b_ready) / max(1, int(n_barrels)))),
                min(1.0, max(0.0, float(n_alive) / max(1, int(swarm_size)))),
                min(1.0, max(0.0, float(est_pk_1bbl))),
            ],
            dtype=np.float64,
        )

    def forward(self, candidates_features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(candidates_features, dtype=np.float64))
        H = np.tanh(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2

    def select_action(
        self,
        candidates_features: np.ndarray,
        b_ready: int,
        max_burst: int,
    ) -> dict[str, Any]:
        """Choose target index, discrete burst size (1..min(b_ready, max_burst)), and hold-fire gate."""
        outs = self.forward(candidates_features)
        idx = int(np.argmax(outs[:, 0]))
        top = outs[idx]
        cap = max(1, min(int(b_ready), int(max_burst)))
        burst_frac = float(_sigmoid(float(top[1])))
        burst_size = max(1, min(cap, int(round(1.0 + (cap - 1) * burst_frac))))
        hold_prob = float(_sigmoid(float(top[2])))
        return {
            "target_idx": idx,
            "priority_score": float(top[0]),
            "burst_size": burst_size,
            "hold_fire_prob": hold_prob,
        }


def get_default_rl_training_curriculum() -> list[dict[str, Any]]:
    """Curated multi-threat benchmark suite spanning fast FPVs, pincers, heavy octos & high-G weaves."""
    raw = [
        {
            "name": "Fast FPV Omni Saturation",
            "drone_model": "fpv_quad_small",
            "swarm_size": 7,
            "attack_tactic": "synchronized_omni_wave",
            "maneuver_profile": "corkscrew_spiral",
            "maneuver_g": 3.0,
            "drone_speed_mps": 30.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.4,
        },
        {
            "name": "High-Speed Wing Corridor Rush",
            "drone_model": "fixed_wing_loitering",
            "swarm_size": 5,
            "attack_tactic": "narrow_corridor_saturation",
            "maneuver_profile": "terminal_popup_jink",
            "maneuver_g": 2.8,
            "drone_speed_mps": 36.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.5,
        },
        {
            "name": "Heavy Octo Multi-Altitude",
            "drone_model": "heavy_octocopter_large",
            "swarm_size": 4,
            "attack_tactic": "staggered_multi_altitude",
            "maneuver_profile": "sinusoidal_weave",
            "maneuver_g": 1.3,
            "drone_speed_mps": 18.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.2,
        },
        {
            "name": "Commercial Quad 8-Swarm",
            "drone_model": "commercial_quad_medium",
            "swarm_size": 8,
            "attack_tactic": "narrow_corridor_saturation",
            "maneuver_profile": "sinusoidal_weave",
            "maneuver_g": 1.8,
            "drone_speed_mps": 24.0,
            "aspect_mode": "oblique_quartering",
            "fiber_optic_fraction": 0.3,
        },
        {
            "name": "Stealth Fiber-Optic Pop-Up",
            "drone_model": "fpv_quad_small",
            "swarm_size": 6,
            "attack_tactic": "staggered_multi_altitude",
            "maneuver_profile": "terminal_popup_jink",
            "maneuver_g": 3.8,
            "drone_speed_mps": 28.0,
            "aspect_mode": "head_on_frontal",
            "fiber_optic_fraction": 0.9,
        },
    ]
    return [normalize_red_config(s, max_swarm_size=12) for s in raw]


def evaluate_fire_control_policy_mc(
    blue_config: dict[str, Any],
    policy_or_heuristic: NeuralFireControlPolicy | str,
    red_scenarios: list[dict[str, Any]] | None = None,
    n_episodes_per_scenario: int = 12,
    breach_keepout_radius_m: float = 12.0,
    seed: int = 42,
    physics_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a NeuralFireControlPolicy or named heuristic across a multi-threat swarm curriculum."""
    pf = dict(physics_fidelity) if isinstance(physics_fidelity, dict) else None
    blue_n = normalize_blue_config(blue_config)
    scenarios = [normalize_red_config(s, max_swarm_size=12) for s in (red_scenarios or get_default_rl_training_curriculum())]
    is_neural = isinstance(policy_or_heuristic, NeuralFireControlPolicy)
    pol_name = "RL Neural Policy" if is_neural else str(policy_or_heuristic)

    choke_defs = get_choke_definitions()
    drone_specs = get_drone_target_specs()
    sensor = CUAS_SENSOR_PRESETS.get(blue_n["sensor_preset"], CUAS_SENSOR_PRESETS["Sensor_Fused_Radar_EOIR"])
    sigma_az_rad = float(sensor["sigma_az_mrad"]) / 1000.0
    t_lat = float(sensor["fcs_latency_ms"]) / 1000.0
    slew_rad_s = math.radians(blue_n["slew_rate_deg_s"])
    rho_air = 1.225

    def _ring_const(ammo_k: str, choke_k: str) -> dict[str, float]:
        _, a_spec = _resolve_shotgun_ammo(ammo_k)
        d_m = float(a_spec["pellet_diameter_mm"]) / 1000.0
        m_kg = float(a_spec["pellet_mass_g"]) / 1000.0
        cd_v = float(a_spec.get("effective_cd", 0.47))
        area_p = math.pi * 0.25 * (d_m ** 2)
        c_inf = choke_defs.get(choke_k, choke_defs["Modified"])
        sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - float(c_inf["benchmark_40yd_30in_pct"])))
        return {
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

    max_burst = max(2, int(blue_n["barrels_per_burst"]))
    rel_t = float(blue_n["reload_time_s"])
    slew_rate = float(blue_n["slew_rate_deg_s"])
    slew_accel = max(420.0, slew_rate * 4.0)

    def _burst_pk(
        R_in: float,
        spd_mps: float,
        a_eff_v: float,
        e_rotor_v: float,
        e_core_v: float,
        n_crit_v: int,
        r_eq_sq_v: float,
        ring_p: dict[str, float],
        n_bbls_shot: int,
        e_lead_draw: float,
    ) -> tuple[float, str]:
        R = max(R_in, breach_keepout_radius_m)
        t_tof, v_imp = exact_quadratic_drag_tof_and_velocity(R, ring_p["v0"], ring_p["kappa"])
        ke_j = 0.5 * ring_p["m_kg"] * (v_imp ** 2)
        if ke_j < e_rotor_v:
            return 0.0, "insufficient_lethality_spread"
        t_eff = t_tof + t_lat
        e_maneuver = (0.24 if blue_n["predictor_mode"] == "kalman_alpha_beta_gamma" else 0.52) * a_eff_v * (t_eff ** 2)
        omega_req = 0.55 * spd_mps / R
        servo_lag_m = max(0.0, omega_req - slew_rad_s) * t_eff * R
        sigma_meas = max(0.02, sigma_az_rad * R * 1.15)
        e_lead_tot = e_maneuver + servo_lag_m + e_lead_draw * sigma_meas

        cup_sep_m = float((pf or {}).get("shot_cup_separation_m", 0.0) or 0.0)
        R_spr = max(0.5, R - min(cup_sep_m, 0.65 * R)) if cup_sep_m > 0.0 else R
        sigma_r = 0.00463 + ring_p["theta_choke"] * R_spr * ((R_spr / 36.58) ** 0.15)
        if blue_n["pointing_mode"] == "divergent_curtain":
            sigma_curt = 0.58 * (float(blue_n["splay_angle_mrad"]) / 1000.0) * R
            sigma_eff = math.hypot(sigma_r, sigma_curt)
        elif blue_n["pointing_mode"] == "staggered_tiling":
            sigma_eff = 1.24 * sigma_r
        else:
            sigma_eff = sigma_r

        sigma_eff_sq = max(sigma_eff ** 2, 1e-6)
        r_eq_eff = r_eq_sq_v * (0.68 if (pf and bool(pf.get("use_rotor_solidity", False))) else 1.0)
        p_pel_center = 1.0 - math.exp(-r_eq_eff / (2.0 * sigma_eff_sq))
        off_axis_atten = math.exp(-(e_lead_tot ** 2) / (2.0 * sigma_eff_sq))
        density_dilution = sigma_r / max(sigma_eff, sigma_r)
        p_pel = p_pel_center * off_axis_atten * (0.45 + 0.55 * density_dilution)
        if pf and bool(pf.get("use_v50_ballistic_limit", False)):
            from ammo_mc.small_arms import compute_component_v50_perforation

            v50_res = compute_component_v50_perforation(
                "fuselage_core",
                pellet_velocity_mps=v_imp,
                pellet_mass_kg=ring_p["m_kg"],
                pellet_diameter_m=math.sqrt((ring_p["m_kg"] / 11300.0) * 6.0 / math.pi),
                enable_ricochet=bool(pf.get("use_oblique_ricochet", False)),
            )
            energy_fac = 0.30 + 0.70 * float(v50_res["p_perforation"])
        else:
            energy_fac = 0.55 + 0.45 * min(1.0, max(0.0, ke_j - e_rotor_v) / max(e_core_v - e_rotor_v, 1.0))
        fly_m = (1.0 - 0.55 * float(pf.get("flyer_fraction", 0.12))) if (pf and str(pf.get("pattern_distribution", "gaussian")) != "gaussian") else 1.0
        shld_m = 0.90 if (pf and bool(pf.get("enable_ray_shielding", False))) else 1.0
        lam = min(60.0, n_bbls_shot * ring_p["pellets"] * p_pel * 0.80 * energy_fac * fly_m * shld_m)
        p_surv = 0.0
        term = math.exp(-lam)
        for k in range(n_crit_v):
            p_surv += term
            term *= lam / float(k + 1)
        pk_val = float(max(0.0, min(1.0, 1.0 - p_surv)))
        fail_tag = "tracking_evasion_miss" if e_lead_tot > sigma_eff else "insufficient_lethality_spread"
        return pk_val, fail_tag

    zero_breach_total = 0
    total_eps = 0
    kill_rates: list[float] = []
    rounds_per_kill_list: list[float] = []
    burst_sizes: list[int] = []
    ep_rewards: list[float] = []
    hold_decisions = 0
    total_decisions = 0
    fail_counts = {
        "reload_gap": 0,
        "slew_bottleneck": 0,
        "tracking_evasion_miss": 0,
        "insufficient_lethality_spread": 0,
    }

    for sc_idx, red_n in enumerate(scenarios):
        rng = np.random.default_rng(seed + sc_idx * 1009)
        d_spec = drone_specs.get(red_n["drone_model"], drone_specs["fpv_quad_small"])
        geom = compute_drone_projected_geometry(d_spec, aspect_mode=red_n["aspect_mode"])
        area_vuln = float(geom.get("presented_area_m2", 0.02))
        r_eq_sq = area_vuln / math.pi
        e_rotor = float(d_spec.get("lethal_energy_rotor_j", 8.0))
        e_core = float(d_spec.get("lethal_energy_core_j", 24.0))
        n_crit = int(d_spec.get("critical_hits_for_kill", 3))
        prof_mult = {
            "linear_cruise": 0.0,
            "sinusoidal_weave": 1.0,
            "corkscrew_spiral": 1.28,
            "terminal_popup_jink": 1.50,
        }.get(red_n["maneuver_profile"], 1.0)
        a_eff = float(red_n["maneuver_g"]) * 9.80665 * prof_mult
        nom_spd = float(red_n["drone_speed_mps"])

        eff_engage_range = max(
            breach_keepout_radius_m + 6.0,
            min(
                float(blue_n["max_engagement_range_m"]),
                float(blue_n["max_engagement_range_m"]) * (1.0 - 0.18 * float(red_n["fiber_optic_fraction"])) + 1.5,
            ),
        )

        for _ep in range(max(1, int(n_episodes_per_scenario))):
            total_eps += 1
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
                drones.append(
                    {
                        "az": az_d,
                        "el": el_d,
                        "r0": r0,
                        "spd": spd_d,
                        "alive": True,
                        "leaked": False,
                        "shots_taken": 0,
                        "last_fail_reason": "slew_bottleneck",
                    }
                )

            t_now = 0.0
            inner_ready = n_inner
            outer_ready = n_outer
            b_ready = n_barrels
            tur_az, tur_el = 0.0, 24.0
            ep_rounds = 0
            consecutive_holds = 0
            just_reloaded = False

            while True:
                for dr in drones:
                    if dr["alive"] and (dr["r0"] - dr["spd"] * t_now) <= breach_keepout_radius_m:
                        dr["alive"] = False
                        dr["leaked"] = True
                        reason = "reload_gap" if just_reloaded else (
                            "slew_bottleneck" if dr["shots_taken"] == 0 else dr["last_fail_reason"]
                        )
                        fail_counts[reason] += 1
                just_reloaded = False

                alive = [dr for dr in drones if dr["alive"]]
                if not alive:
                    break

                if b_ready <= 0:
                    t_now += rel_t
                    inner_ready = n_inner
                    outer_ready = n_outer
                    b_ready = n_barrels
                    just_reloaded = True
                    continue

                in_range = [dr for dr in alive if (dr["r0"] - dr["spd"] * t_now) <= eff_engage_range]
                if not in_range:
                    t_ent = min((dr["r0"] - eff_engage_range) / dr["spd"] for dr in alive)
                    t_now = max(t_now, t_ent + 1e-3)
                    in_range = [dr for dr in alive if (dr["r0"] - dr["spd"] * t_now) <= eff_engage_range]
                    if not in_range:
                        continue

                if is_neural:
                    feat_rows = []
                    ring_choices: list[bool] = []
                    for dr in in_range:
                        r_cur = dr["r0"] - dr["spd"] * t_now
                        ttb = max(0.0, r_cur - breach_keepout_radius_m) / max(dr["spd"], 1.0)
                        d_az_c = ((dr["az"] - tur_az + 180.0) % 360.0) - 180.0
                        t_sl = _trapezoidal_slew_time(math.hypot(d_az_c, dr["el"] - tur_el), slew_rate, slew_accel)
                        pk_out_1, _ = _burst_pk(r_cur, dr["spd"], a_eff, e_rotor, e_core, n_crit, r_eq_sq, outer_r, 1, 0.7)
                        use_in = False
                        pk_est = pk_out_1
                        if is_hybrid and inner_ready > 0:
                            pk_in_1, _ = _burst_pk(r_cur, dr["spd"], a_eff, e_rotor, e_core, n_crit, r_eq_sq, inner_r, 1, 0.7)
                            if outer_ready <= 0 or pk_in_1 >= pk_out_1:
                                use_in = True
                                pk_est = pk_in_1
                        ring_choices.append(use_in)
                        feat_rows.append(
                            NeuralFireControlPolicy.build_target_features(
                                slant_range_m=r_cur,
                                max_engage_range_m=eff_engage_range,
                                ttb_s=ttb,
                                slew_time_s=t_sl,
                                speed_mps=dr["spd"],
                                maneuver_g=red_n["maneuver_g"],
                                b_ready=b_ready,
                                n_barrels=n_barrels,
                                n_alive=len(alive),
                                swarm_size=int(red_n["swarm_size"]),
                                est_pk_1bbl=pk_est,
                            )
                        )
                    act = policy_or_heuristic.select_action(np.vstack(feat_rows), b_ready=b_ready, max_burst=max_burst)
                    tgt_idx = act["target_idx"]
                    tgt = in_range[tgt_idx]
                    burst_req = act["burst_size"]
                    r_cur_sel = tgt["r0"] - tgt["spd"] * t_now
                    ttb_sel = max(0.0, r_cur_sel - breach_keepout_radius_m) / max(tgt["spd"], 1.0)
                    total_decisions += 1
                    if (
                        act["hold_fire_prob"] > 0.65
                        and ttb_sel > 1.35
                        and r_cur_sel > 0.72 * eff_engage_range
                        and consecutive_holds < 5
                    ):
                        t_now += 0.22
                        consecutive_holds += 1
                        hold_decisions += 1
                        continue
                    consecutive_holds = 0
                    use_inner_shot = ring_choices[tgt_idx]
                else:
                    if policy_or_heuristic == "nearest_slant_range":
                        tgt = min(in_range, key=lambda d: d["r0"] - d["spd"] * t_now)
                    elif policy_or_heuristic == "minimum_slew_angle":
                        tgt = min(
                            in_range,
                            key=lambda d: abs(((d["az"] - tur_az + 180.0) % 360.0) - 180.0) + abs(d["el"] - tur_el),
                        )
                    else:
                        tgt = min(
                            in_range,
                            key=lambda d: max(0.0, (d["r0"] - d["spd"] * t_now) - breach_keepout_radius_m) / max(d["spd"], 1.0),
                        )
                    burst_req = int(blue_n["barrels_per_burst"])
                    r_cur_sel = tgt["r0"] - tgt["spd"] * t_now
                    use_inner_shot = bool(is_hybrid and ((r_cur_sel > float(blue_n["hybrid_switch_range_m"]) and inner_ready > 0) or outer_ready <= 0))
                    total_decisions += 1

                d_az = ((tgt["az"] - tur_az + 180.0) % 360.0) - 180.0
                d_el = tgt["el"] - tur_el
                t_slew = _trapezoidal_slew_time(math.hypot(d_az, d_el), slew_rate, slew_accel)
                settle_t = 0.08 + float(blue_n["ripple_delay_ms"]) * max(0, burst_req - 1) / 2500.0
                t_now += t_slew + settle_t
                tur_az, tur_el = tgt["az"], tgt["el"]

                r_fire = tgt["r0"] - tgt["spd"] * t_now
                if r_fire <= breach_keepout_radius_m:
                    tgt["alive"] = False
                    tgt["leaked"] = True
                    fail_counts["slew_bottleneck"] += 1
                    continue

                if is_hybrid:
                    if use_inner_shot and inner_ready > 0:
                        rounds_fired = min(inner_ready, burst_req)
                        inner_ready -= rounds_fired
                        ring_sel = inner_r
                    else:
                        rounds_fired = min(outer_ready, burst_req)
                        outer_ready -= rounds_fired
                        ring_sel = outer_r
                    b_ready = inner_ready + outer_ready
                else:
                    rounds_fired = min(b_ready, burst_req)
                    b_ready -= rounds_fired
                    ring_sel = outer_r

                ep_rounds += rounds_fired
                burst_sizes.append(rounds_fired)
                tgt["shots_taken"] += 1

                ray_draw = float(rng.rayleigh(1.0))
                pk_b, fail_tag = _burst_pk(
                    r_fire,
                    tgt["spd"],
                    a_eff,
                    e_rotor,
                    e_core,
                    n_crit,
                    r_eq_sq,
                    ring_sel,
                    rounds_fired,
                    ray_draw,
                )
                tgt["last_fail_reason"] = fail_tag
                if rng.random() < pk_b:
                    tgt["alive"] = False

            n_leaked = sum(1 for d in drones if d["leaked"])
            n_killed = int(red_n["swarm_size"]) - n_leaked
            if n_leaked == 0:
                zero_breach_total += 1
            kr_ep = 100.0 * n_killed / max(1, int(red_n["swarm_size"]))
            kill_rates.append(kr_ep)
            rpk = ep_rounds / max(n_killed, 0.5)
            rounds_per_kill_list.append(rpk)
            rew = (60.0 if n_leaked == 0 else 0.0) + 0.40 * kr_ep - 1.40 * rpk
            ep_rewards.append(rew)

    total_leakers = sum(fail_counts.values())
    denom = max(total_leakers, 1)
    return {
        "policy": pol_name,
        "zero_breach_pct": round(100.0 * zero_breach_total / max(1, total_eps), 1),
        "kill_rate_pct": round(float(np.mean(kill_rates)), 1) if kill_rates else 0.0,
        "mean_rounds_per_kill": round(float(np.mean(rounds_per_kill_list)), 2) if rounds_per_kill_list else 0.0,
        "mean_burst_size": round(float(np.mean(burst_sizes)), 2) if burst_sizes else 0.0,
        "hold_fire_rate_pct": round(100.0 * hold_decisions / max(1, total_decisions), 1),
        "mean_reward": round(float(np.mean(ep_rewards)), 2) if ep_rewards else 0.0,
        "failure_modes": {
            "reload_gap_pct": round(100.0 * fail_counts["reload_gap"] / denom, 1) if total_leakers > 0 else 0.0,
            "slew_bottleneck_pct": round(100.0 * fail_counts["slew_bottleneck"] / denom, 1) if total_leakers > 0 else 0.0,
            "tracking_evasion_miss_pct": round(100.0 * fail_counts["tracking_evasion_miss"] / denom, 1) if total_leakers > 0 else 0.0,
            "insufficient_lethality_spread_pct": round(100.0 * fail_counts["insufficient_lethality_spread"] / denom, 1) if total_leakers > 0 else 0.0,
            "total_leakers": int(total_leakers),
        },
    }


def train_fire_control_policy_es(
    blue_config: dict[str, Any] | None = None,
    n_generations: int = 14,
    pop_size: int = 14,
    sigma_es: float = 0.22,
    lr: float = 0.12,
    n_eval_episodes: int = 16,
    seed: int = 42,
    physics_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a NeuralFireControlPolicy with Antithetic Natural Evolution Strategies & benchmark vs heuristics."""
    blue_n = normalize_blue_config(
        blue_config
        or {
            "name": "8-Bbl Concentric Hybrid Turret",
            "layout_type": "concentric",
            "n_barrels": 8,
            "barrels_per_burst": 3,
            "ammunition_preset": "12G_NO4_BUCK_C_UAS",
            "choke": "Modified",
            "pointing_mode": "divergent_curtain",
            "splay_angle_mrad": 4.0,
            "firing_mode": "balanced_pair_ripple",
            "ripple_delay_ms": 8.0,
            "max_engagement_range_m": 66.0,
            "slew_rate_deg_s": 165.0,
            "reload_time_s": 3.0,
            "hybrid_mode": True,
            "inner_ammunition_preset": "12G_TSS_NO2_C_UAS",
            "inner_choke": "Extra Full",
        }
    )
    curriculum = get_default_rl_training_curriculum()
    rng = np.random.default_rng(seed)

    pol = NeuralFireControlPolicy(seed=seed)
    theta = pol.get_flat_params()
    momentum = np.zeros_like(theta)
    n_pairs = max(2, int(pop_size) // 2)
    ep_train = max(2, int(n_eval_episodes) // 5)

    # Initial Gen-0 baseline check
    pol.set_flat_params(theta)
    gen0_res = evaluate_fire_control_policy_mc(blue_n, pol, curriculum, n_episodes_per_scenario=ep_train, seed=seed, physics_fidelity=physics_fidelity)
    best_theta = theta.copy()
    best_score = float(gen0_res["mean_reward"])
    training_history: list[dict[str, Any]] = [
        {
            "generation": 0,
            "mean_pop_reward": gen0_res["mean_reward"],
            "best_reward": best_score,
            "zero_breach_pct": gen0_res["zero_breach_pct"],
            "mean_rounds_per_kill": gen0_res["mean_rounds_per_kill"],
            "mean_burst_size": gen0_res["mean_burst_size"],
        }
    ]

    for gen in range(1, max(1, int(n_generations)) + 1):
        eps = rng.normal(0.0, 1.0, size=(n_pairs, theta.size))
        f_pos = np.zeros(n_pairs, dtype=np.float64)
        f_neg = np.zeros(n_pairs, dtype=np.float64)
        gen_seed = seed + gen * 37
        for i in range(n_pairs):
            pol.set_flat_params(theta + float(sigma_es) * eps[i])
            f_pos[i] = float(
                evaluate_fire_control_policy_mc(blue_n, pol, curriculum, n_episodes_per_scenario=ep_train, seed=gen_seed, physics_fidelity=physics_fidelity)["mean_reward"]
            )
            pol.set_flat_params(theta - float(sigma_es) * eps[i])
            f_neg[i] = float(
                evaluate_fire_control_policy_mc(blue_n, pol, curriculum, n_episodes_per_scenario=ep_train, seed=gen_seed, physics_fidelity=physics_fidelity)["mean_reward"]
            )

        all_f = np.concatenate([f_pos, f_neg])
        sd_f = float(np.std(all_f))
        if sd_f > 1e-6:
            diff_norm = (f_pos - f_neg) / (2.0 * sd_f)
        else:
            diff_norm = np.zeros_like(f_pos)
        grad = (diff_norm @ eps) / (max(1, n_pairs) * max(float(sigma_es), 1e-4))
        momentum = 0.65 * momentum + (float(lr) * (0.97 ** gen)) * grad
        theta = theta + momentum

        pol.set_flat_params(theta)
        cur_eval = evaluate_fire_control_policy_mc(blue_n, pol, curriculum, n_episodes_per_scenario=ep_train, seed=gen_seed, physics_fidelity=physics_fidelity)
        if float(cur_eval["mean_reward"]) >= best_score:
            best_score = float(cur_eval["mean_reward"])
            best_theta = theta.copy()

        training_history.append(
            {
                "generation": gen,
                "mean_pop_reward": round(float(np.mean(all_f)), 2),
                "best_reward": round(best_score, 2),
                "zero_breach_pct": cur_eval["zero_breach_pct"],
                "mean_rounds_per_kill": cur_eval["mean_rounds_per_kill"],
                "mean_burst_size": cur_eval["mean_burst_size"],
            }
        )

    pol.set_flat_params(best_theta)

    # Full paired benchmark against all 3 heuristic baselines
    bench_seed = seed + 555
    eval_eps = max(8, int(n_eval_episodes))
    rl_bench = evaluate_fire_control_policy_mc(blue_n, pol, curriculum, n_episodes_per_scenario=eval_eps, seed=bench_seed, physics_fidelity=physics_fidelity)
    heur_names = ["minimum_time_to_breach", "nearest_slant_range", "minimum_slew_angle"]
    heur_results = [
        evaluate_fire_control_policy_mc(blue_n, h_name, curriculum, n_episodes_per_scenario=eval_eps, seed=bench_seed, physics_fidelity=physics_fidelity)
        for h_name in heur_names
    ]
    benchmark_rows = [rl_bench] + heur_results
    best_heur = max(heur_results, key=lambda r: (r["zero_breach_pct"], r["kill_rate_pct"], -r["mean_rounds_per_kill"]))

    uplift_zb = round(rl_bench["zero_breach_pct"] - best_heur["zero_breach_pct"], 1)
    rounds_saved_pct = round(
        100.0 * (best_heur["mean_rounds_per_kill"] - rl_bench["mean_rounds_per_kill"]) / max(best_heur["mean_rounds_per_kill"], 1e-4),
        1,
    )

    # Explainability Action Surface: learned burst size & hold-fire gate over Range x Ammo Remaining
    n_barrels = max(1, int(blue_n["n_barrels"]))
    ranges_grid = [18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0]
    ammo_frac_grid = [0.15, 0.30, 0.50, 0.70, 0.85, 1.00]
    max_b = max(2, int(blue_n["barrels_per_burst"]))
    burst_surface: list[list[float]] = []
    hold_surface: list[list[float]] = []
    for af in ammo_frac_grid:
        b_row: list[float] = []
        h_row: list[float] = []
        b_cnt = max(1, int(round(af * n_barrels)))
        for rng_m in ranges_grid:
            feat = NeuralFireControlPolicy.build_target_features(
                slant_range_m=rng_m,
                max_engage_range_m=float(blue_n["max_engagement_range_m"]),
                ttb_s=(rng_m - 12.0) / 26.0,
                slew_time_s=0.18,
                speed_mps=26.0,
                maneuver_g=2.5,
                b_ready=b_cnt,
                n_barrels=n_barrels,
                n_alive=max(1, int(round(af * 6))),
                swarm_size=6,
                est_pk_1bbl=max(0.05, min(0.95, 1.1 - rng_m / 72.0)),
            )
            act = pol.select_action(feat, b_ready=b_cnt, max_burst=max_b)
            b_row.append(float(act["burst_size"]))
            h_row.append(round(100.0 * float(act["hold_fire_prob"]), 1))
        burst_surface.append(b_row)
        hold_surface.append(h_row)

    return {
        "policy_dict": pol.to_dict(),
        "training_history": training_history,
        "benchmark_rows": benchmark_rows,
        "best_heuristic_name": best_heur["policy"],
        "uplift_zero_breach_pp": uplift_zb,
        "rounds_saved_pct": rounds_saved_pct,
        "action_surface": {
            "ranges_m": ranges_grid,
            "ammo_fractions": ammo_frac_grid,
            "burst_grid": burst_surface,
            "hold_fire_prob_grid": hold_surface,
        },
        "blue_config": blue_n,
        "summary": {
            "rl_zero_breach_pct": rl_bench["zero_breach_pct"],
            "rl_kill_rate_pct": rl_bench["kill_rate_pct"],
            "rl_mean_rounds_per_kill": rl_bench["mean_rounds_per_kill"],
            "rl_mean_burst_size": rl_bench["mean_burst_size"],
            "rl_hold_fire_rate_pct": rl_bench["hold_fire_rate_pct"],
            "best_heuristic_name": best_heur["policy"],
            "best_heuristic_zero_breach_pct": best_heur["zero_breach_pct"],
            "uplift_zero_breach_pp": uplift_zb,
            "rounds_saved_pct": rounds_saved_pct,
            "n_generations": n_generations,
            "pop_size": pop_size,
            "n_eval_episodes": eval_eps,
        },
    }


def run_hil_fire_control_trace(
    policy_or_dict: NeuralFireControlPolicy | dict[str, Any] | str | None = None,
    track_csv_or_scenario: str = "evasive_fpv_dive",
    blue_config: dict[str, Any] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    dt_ms: float = 100.0,
    breach_radius_m: float = 12.0,
    seed: int = 42,
    physics_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step a trained RL or heuristic fire-control policy through recorded drone telemetry millisecond-by-millisecond."""
    import csv as _csv
    import io as _io
    import json as _json
    from ammo_mc.cuas_track_replay import generate_sample_drone_track_csv, parse_drone_track_csv

    _ = seed
    blue_n = normalize_blue_config(blue_config or {})
    cp = dict(calibration_profile or {})

    if policy_or_dict is None:
        pol: NeuralFireControlPolicy | str = NeuralFireControlPolicy(seed=42)
        pol_name = "RL Neural Policy (Default)"
    elif isinstance(policy_or_dict, NeuralFireControlPolicy):
        pol = policy_or_dict
        pol_name = "RL Neural Policy"
    elif isinstance(policy_or_dict, dict):
        p_dict = policy_or_dict.get("policy_dict", policy_or_dict)
        if isinstance(p_dict, dict) and "params" in p_dict:
            pol = NeuralFireControlPolicy.from_dict(p_dict)
            pol_name = "RL Neural Policy (Trained)"
        else:
            pol = NeuralFireControlPolicy(seed=42)
            pol_name = "RL Neural Policy (Default)"
    else:
        pol = str(policy_or_dict)
        pol_name = f"Heuristic: {pol}"

    raw_str = str(track_csv_or_scenario or "evasive_fpv_dive").strip()
    if "," in raw_str and "\n" in raw_str:
        csv_txt = raw_str
    else:
        csv_txt = generate_sample_drone_track_csv(scenario=raw_str or "evasive_fpv_dive", seed=42)
    parsed = parse_drone_track_csv(csv_txt)
    tracks = parsed["tracks"]
    t0 = float(parsed["t_min_s"])
    t1 = float(parsed["t_max_s"])

    dt_s = max(0.02, min(0.50, float(dt_ms) / 1000.0))
    n_steps = max(2, int(math.floor((t1 - t0) / dt_s)) + 1)

    drone_specs = get_drone_target_specs()
    ammo_key = blue_n["ammunition_preset"]
    _, ammo = _resolve_shotgun_ammo(ammo_key)
    slew_rate = float(cp.get("measured_slew_rate_deg_s") or blue_n["slew_rate_deg_s"])
    slew_accel = float(cp.get("measured_slew_accel_deg_s2") or 680.0)
    settle_t = float(cp.get("measured_settle_time_s") or 0.10)
    n_bbl = int(blue_n["n_barrels"])
    max_burst = int(blue_n["barrels_per_burst"])
    max_rng = float(blue_n["max_engagement_range_m"])

    b_ready = n_bbl
    reload_until_s = -1.0
    tur_az = 0.0
    tur_el = 15.0
    engaged_until_s = -1.0
    killed_targets: set[str] = set()
    breached_targets: set[str] = set()
    total_shots = 0
    hold_frames = 0
    fire_events = 0
    frames: list[dict[str, Any]] = []

    for idx in range(n_steps):
        t_cur = t0 + idx * dt_s
        if b_ready <= 0 and reload_until_s < 0.0:
            reload_until_s = t_cur + float(blue_n["reload_time_s"])
        if reload_until_s >= 0.0 and t_cur >= reload_until_s:
            b_ready = n_bbl
            reload_until_s = -1.0

        candidates: list[dict[str, Any]] = []
        for tid, tr in tracks.items():
            if tid in killed_targets or tid in breached_targets:
                continue
            ts_arr = tr["t_s"]
            if t_cur < float(ts_arr[0]) or t_cur > float(ts_arr[-1]):
                continue
            rng_m = float(np.interp(t_cur, ts_arr, tr["range_m"]))
            spd_mps = float(np.interp(t_cur, ts_arr, tr["speed_mps"]))
            az_d = float(np.interp(t_cur, ts_arr, np.unwrap(np.radians(tr["az_deg"]))))
            az_deg = math.degrees(az_d) % 360.0
            el_deg = float(np.interp(t_cur, ts_arr, tr["el_deg"]))
            lat_g = float(np.interp(t_cur, ts_arr, tr["lat_g"]))
            if rng_m <= float(breach_radius_m):
                breached_targets.add(tid)
                continue
            if rng_m > max_rng:
                continue
            d_az = ((az_deg - tur_az + 180.0) % 360.0) - 180.0
            d_el = el_deg - tur_el
            t_sl = _trapezoidal_slew_time(math.hypot(d_az, d_el), slew_rate, slew_accel)
            ttb_s = max(0.0, rng_m - float(breach_radius_m)) / max(spd_mps, 1.0)
            d_spec = drone_specs.get(tr["drone_model"], drone_specs["fpv_quad_small"])
            pk_1 = _estimate_burst_pk_analytics(
                slant_range_m=rng_m,
                barrels_fired=1,
                pellets_per_shot=int(ammo["pellets_per_shot"]),
                pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                pellet_mass_g=float(ammo["pellet_mass_g"]),
                v0_mps=float(ammo["v0_mps"]),
                choke_name=blue_n["choke"],
                drone_spec=d_spec,
                base_cd=float(ammo.get("effective_cd", 0.47)),
                calibration_profile=calibration_profile,
                physics_fidelity=physics_fidelity,
            )
            candidates.append(
                {
                    "tid": tid,
                    "range_m": rng_m,
                    "spd_mps": spd_mps,
                    "az_deg": az_deg,
                    "el_deg": el_deg,
                    "lat_g": lat_g,
                    "slew_time_s": t_sl,
                    "ttb_s": ttb_s,
                    "pk_1": pk_1,
                    "d_spec": d_spec,
                }
            )

        if not candidates or b_ready <= 0 or t_cur < engaged_until_s:
            reason = "Reloading" if b_ready <= 0 else ("Slewing / Settling" if t_cur < engaged_until_s else "No Target in Envelope")
            frames.append(
                {
                    "t_ms": round(t_cur * 1000.0, 0),
                    "target_id": candidates[0]["tid"] if candidates else "—",
                    "slant_range_m": round(candidates[0]["range_m"], 2) if candidates else None,
                    "ttb_s": round(candidates[0]["ttb_s"], 2) if candidates else None,
                    "cmd_az_deg": round(tur_az, 1),
                    "cmd_el_deg": round(tur_el, 1),
                    "cmd_burst_size": 0,
                    "hold_fire_prob": 0.0,
                    "fire_commanded": False,
                    "barrels_ready": b_ready,
                    "pk_est_pct": round(100.0 * candidates[0]["pk_1"], 1) if candidates else 0.0,
                    "decision_reason": reason,
                }
            )
            continue

        n_alive = len(candidates)
        if isinstance(pol, NeuralFireControlPolicy):
            X_cand = np.vstack(
                [
                    NeuralFireControlPolicy.build_target_features(
                        slant_range_m=c["range_m"],
                        max_engage_range_m=max_rng,
                        ttb_s=c["ttb_s"],
                        slew_time_s=c["slew_time_s"],
                        speed_mps=c["spd_mps"],
                        maneuver_g=c["lat_g"],
                        b_ready=b_ready,
                        n_barrels=n_bbl,
                        n_alive=n_alive,
                        swarm_size=max(len(tracks), n_alive),
                        est_pk_1bbl=c["pk_1"],
                    )
                    for c in candidates
                ]
            )
            act = pol.select_action(X_cand, b_ready=b_ready, max_burst=max_burst)
            sel = candidates[act["target_idx"]]
            burst_req = int(act["burst_size"])
            hold_prob = float(act["hold_fire_prob"])
            should_hold = bool(hold_prob > 0.55 and sel["ttb_s"] > 0.85 and sel["range_m"] > 0.48 * max_rng)
        else:
            if pol == "nearest_slant_range":
                sel = min(candidates, key=lambda c: c["range_m"])
            elif pol == "minimum_slew_angle":
                sel = min(candidates, key=lambda c: c["slew_time_s"])
            else:
                sel = min(candidates, key=lambda c: c["ttb_s"])
            burst_req = min(b_ready, max_burst)
            hold_prob = 0.0
            should_hold = False

        tur_az = sel["az_deg"]
        tur_el = sel["el_deg"]
        if should_hold:
            hold_frames += 1
            reason = f"HOLD FIRE (p={hold_prob:.2f}, range={sel['range_m']:.1f} m > sweet spot)"
            cmd_fire = False
            burst_cmd = 0
            pk_est = sel["pk_1"] * 100.0
        else:
            cmd_fire = True
            fire_events += 1
            burst_cmd = min(b_ready, burst_req)
            b_ready -= burst_cmd
            total_shots += burst_cmd
            engaged_until_s = t_cur + sel["slew_time_s"] + settle_t
            pk_est = 100.0 * _estimate_burst_pk_analytics(
                slant_range_m=sel["range_m"],
                barrels_fired=burst_cmd,
                pellets_per_shot=int(ammo["pellets_per_shot"]),
                pellet_diameter_mm=float(ammo["pellet_diameter_mm"]),
                pellet_mass_g=float(ammo["pellet_mass_g"]),
                v0_mps=float(ammo["v0_mps"]),
                choke_name=blue_n["choke"],
                drone_spec=sel["d_spec"],
                base_cd=float(ammo.get("effective_cd", 0.47)),
                calibration_profile=calibration_profile,
                physics_fidelity=physics_fidelity,
            )
            reason = f"FIRE {burst_cmd}B @ {sel['tid']} ({sel['range_m']:.1f} m, est Pk {pk_est:.0f}%)"
            if pk_est >= 82.0:
                killed_targets.add(sel["tid"])

        frames.append(
            {
                "t_ms": round(t_cur * 1000.0, 0),
                "target_id": sel["tid"],
                "slant_range_m": round(sel["range_m"], 2),
                "ttb_s": round(sel["ttb_s"], 2),
                "cmd_az_deg": round(tur_az, 1),
                "cmd_el_deg": round(tur_el, 1),
                "cmd_burst_size": burst_cmd,
                "hold_fire_prob": round(hold_prob, 3),
                "fire_commanded": cmd_fire,
                "barrels_ready": b_ready,
                "pk_est_pct": round(pk_est, 1),
                "decision_reason": reason,
            }
        )

    csv_buf = _io.StringIO()
    if frames:
        writer = _csv.DictWriter(csv_buf, fieldnames=list(frames[0].keys()))
        writer.writeheader()
        writer.writerows(frames)

    summary = {
        "policy_name": pol_name,
        "dt_ms": round(dt_s * 1000.0, 1),
        "n_frames": len(frames),
        "fire_commands": fire_events,
        "hold_fire_frames": hold_frames,
        "total_rounds_commanded": total_shots,
        "targets_neutralized": len(killed_targets),
        "targets_breached": len(breached_targets),
        "total_tracks": len(tracks),
    }
    return {
        "summary": summary,
        "frames": frames,
        "json_stream": _json.dumps({"summary": summary, "frames": frames}, indent=2),
        "csv_stream": csv_buf.getvalue(),
    }

