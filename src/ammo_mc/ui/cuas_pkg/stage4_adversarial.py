"""Stage 4C: Adversarial Red-vs-Blue Co-Evolution, RL Policy, HIL & Beat-the-AI Duel UI."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from ammo_mc.cuas_adversarial import (
    export_adversarial_dataset_csv,
    recommend_ml_defense_policy,
)
from ammo_mc.cuas_hardware import export_muzzle_plate_dxf
from ammo_mc.cuas_wizard import run_red_vs_blue_duel_diagnostic
from ammo_mc.scenario_library import ScenarioLibrary
from ammo_mc.small_arms import generate_multishotgun_array_layout
from ammo_mc.terrain_dem import get_terrain_dem_presets
from ammo_mc.ui.common import (
    _cached_adversarial_coevolution,
    _cached_battlespace_coevolution,
    _cached_hil_policy_trace,
    _cached_train_rl_policy,
    get_active_physics_fidelity,
    is_expert_mode,
)
from ammo_mc.vis_cuas_advanced import (
    create_battlespace_coevolution_figure,
    create_duel_root_cause_figure,
    create_hil_policy_timeline_figure,
    create_rl_action_heatmap_figure,
    create_rl_policy_benchmark_figure,
)
from ammo_mc.vis_cuas_adversarial import (
    create_coevolution_trajectory_figure,
    create_ml_feature_importance_figure,
    create_ml_policy_pareto_figure,
    create_nash_payoff_heatmap_figure,
)
from ammo_mc.ui.cuas_pkg.helpers import (
    _get_active_physics_fidelity_json,
    _get_cal_twin_json,
    _get_effective_cal_twin_json,
    _launch_battlespace_to_raid,
    render_high_fidelity_active_banner,
)


def render_stage4_adversarial(expert: bool, sg_presets_18: dict[str, Any], drone_specs_18: dict[str, Any]) -> None:
    st.markdown("#### 🤖 Adversarial Red-vs-Blue Competing Monte Carlo & GP ML Protection Optimizer")
    render_high_fidelity_active_banner()
    st.caption(
        "Co-evolve adversarial drone threats (Red: kinematics, maneuver-g, stealth, tactics) against multi-barrel shotgun platforms "
        "(Blue: barrel count, choke, layout, ripple timing, pointing mode, ammunition) using Gaussian Process Bayesian surrogate modeling and exact Nash equilibrium game theory."
    )

    # Top controls row
    c_gen, c_mc, c_mt, c_bar, c_sw = st.columns(5)
    with c_gen:
        n_generations = st.slider("Co-Evolution Generations", min_value=3, max_value=8, value=5, step=1, key="cuas_adv_ngen")
    with c_mc:
        n_mc_episodes = st.slider("MC Episodes / Matchup", min_value=10, max_value=30, value=18, step=2, key="cuas_adv_nmc")
    with c_mt:
        mount_load_limit_kn = st.slider("Mount Limit (kN)", min_value=1.5, max_value=8.0, value=3.5, step=0.5, key="cuas_adv_mount_lim")
    with c_bar:
        max_barrels = st.slider("Max Barrels Allowed", min_value=4, max_value=12, value=10, step=1, key="cuas_adv_max_barrels")
    with c_sw:
        max_swarm_size = st.slider("Max Swarm Size (Red)", min_value=2, max_value=10, value=8, step=1, key="cuas_adv_max_swarm")

    t_opt1, t_opt2, t_opt3 = st.columns(3)
    with t_opt1:
        seed_s1 = st.checkbox("Seed Gen 0 with Stage 1 Active Profile", value=True, key="cuas_adv_seed_s1")
    with t_opt2:
        allow_hybrid = st.checkbox(
            "Enable Hybrid Mixed-Load Barrels",
            value=False,
            key="cuas_adv_hybrid",
            help="Allows dual-ring turrets mixing wide #4 Buckshot perimeter + tight TSS #2 / Flechette core barrels.",
        )
    with t_opt3:
        robust_mode = st.checkbox(
            "Uncertainty-Aware Robust Mode (Calibrated + OTM)",
            value=False,
            key="cuas_adv_robust",
            help="Optimizes CVaR-20% tail survivability across propellant temp swings, OTM gimbal jitter, gusts, and calibrated pattern spread.",
        )

    cal_twin = st.session_state.get("_calibrated_digital_twin")
    if cal_twin:
        st.caption(
            f"🔗 Linked Calibration Digital Twin: spread={cal_twin.get('spread_factor', 1.0):.2f}x, "
            f"MV_scale={cal_twin.get('mv_scale', 1.0):.3f}, aim_bias={cal_twin.get('aim_bias_mrad', 0.0):.1f} mrad"
        )

    # Expert Mode Expander
    exp_beta = 1.5
    w_zb = 0.40
    w_kr = 0.40
    w_cost = 0.20
    seed_val = 42
    if is_expert_mode():
        with st.expander("🛠️ Advanced Co-Evolution & GP Surrogate Hyperparameters", expanded=False):
            xc1, xc2, xc3, xc4, xc5 = st.columns(5)
            with xc1:
                exp_beta = st.slider("Exploration Bonus (β)", 0.5, 3.0, 1.5, 0.1, key="cuas_adv_beta")
            with xc2:
                w_zb = st.slider("Zero-Breach Weight", 0.1, 0.8, 0.40, 0.05, key="cuas_adv_w_zb")
            with xc3:
                w_kr = st.slider("Kill-Rate Weight", 0.1, 0.8, 0.40, 0.05, key="cuas_adv_w_kr")
            with xc4:
                w_cost = st.slider("Cost/Barrel Penalty Weight", 0.0, 0.5, 0.20, 0.05, key="cuas_adv_w_cost")
            with xc5:
                seed_val = st.number_input("RNG Seed", 0, 999999, 42, 1, key="cuas_adv_seed")

    prof_json = ""
    if seed_s1 and st.session_state.get("_active_cuas_profile"):
        prof_json = json.dumps(st.session_state["_active_cuas_profile"], sort_keys=True)
    cal_json = json.dumps(cal_twin, sort_keys=True) if cal_twin else ""

    with st.spinner("Executing Adversarial Competing Monte Carlo Co-Evolution & Fitting GP Surrogate..."):
        try:
            adv_res = _cached_adversarial_coevolution(
                n_generations=int(n_generations),
                n_mc_episodes=int(n_mc_episodes),
                mount_load_limit_kn=float(mount_load_limit_kn),
                max_barrels=int(max_barrels),
                max_swarm_size=int(max_swarm_size),
                exploration_beta=float(exp_beta),
                w_zero_breach=float(w_zb),
                w_kill_rate=float(w_kr),
                cost_penalty_weight=float(w_cost),
                stage1_profile_json=prof_json,
                seed=int(seed_val),
                allow_hybrid=bool(allow_hybrid),
                robust_uncertainty_mode=bool(robust_mode),
                calibration_profile_json=cal_json,
                physics_fidelity_json=_get_active_physics_fidelity_json(),
            )
        except Exception as exc:
            st.error(f"Adversarial co-evolution simulation error: {exc}")
            adv_res = None

    if adv_res is not None:
        st.session_state["_report_adversarial_result"] = adv_res
        # 4-metric scorecard
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            sm1.metric(
                "Nash Defense Floor (V*)",
                f"{adv_res['nash_value']:.2f}",
                delta="Minimax Guaranteed Score",
            )
        with sm2:
            wc_gain = adv_res["worst_case_defense_gain"]
            sm2.metric(
                "Evolved Worst-Case Defense",
                f"{wc_gain['evolved_worst_case']:.1f}%",
                delta=f"+{wc_gain['worst_case_gain_pct']:.1f}% vs Gen 0",
            )
        with sm3:
            diag = adv_res["gp_surrogate_diagnostics"]
            sm3.metric(
                "GP Surrogate CV R²",
                f"{diag['cv_r2']:.3f}",
                delta=f"RMSE={diag['cv_rmse']:.2f} (5-Fold OOF)",
            )
        with sm4:
            sm4.metric(
                "Evaluated Matchups",
                f"{adv_res['matchups_evaluated']}",
                delta=f"{adv_res['n_blue_pool']} Blue × {adv_res['n_red_pool']} Red",
            )

        # 1-Click Actions: Apply to Stage 1 + Save to Library + Download Muzzle CAD
        best_blue = adv_res["best_blue_config"]
        hyb_note = (
            f" + Inner `{best_blue.get('inner_ammunition_preset', '')}` ({best_blue.get('inner_choke', '')})"
            if best_blue.get("hybrid_mode")
            else ""
        )
        act_col1, act_col2, act_col3, act_col4 = st.columns([2.6, 1.0, 0.9, 0.9])
        with act_col1:
            act_col1.info(
                f"🏆 **ML-Evolved Policy**: {best_blue['n_barrels']}× ({best_blue['layout']}), "
                f"`{best_blue['ammo']}` ({best_blue['choke']}){hyb_note}, "
                f"Splay {best_blue['splay_angle_deg']}° ({best_blue['pointing']}), "
                f"Ripple {best_blue['ripple_delay_ms']} ms"
            )
        with act_col2:
            if act_col2.button(
                "⚡ Apply ML Turret to Stage 1",
                type="primary",
                key="cuas_adv_apply_to_s1_btn",
                help="Copy the evolved robust Blue configuration to Stage 1 active array setup",
                width="stretch",
            ):
                st.session_state["cuas_layout"] = str(best_blue["layout"])
                st.session_state["cuas_n_barrels"] = int(best_blue["n_barrels"])
                st.session_state["cuas_ammo"] = str(best_blue["ammo"])
                st.session_state["cuas_choke"] = str(best_blue["choke"])
                st.session_state["cuas_pointing"] = str(best_blue["pointing"])
                st.session_state["cuas_splay"] = float(best_blue["splay_angle_deg"])
                st.session_state["cuas_fire_mode"] = str(best_blue["firing_mode"])
                st.session_state["cuas_ripple_dt"] = float(best_blue["ripple_delay_ms"])
                st.session_state["cuas_mixed_loadout"] = bool(best_blue.get("hybrid_mode", False))
                if best_blue.get("hybrid_mode"):
                    st.session_state["cuas_inner_ammo"] = str(best_blue.get("inner_ammunition_preset", "12G_TSS_NO2_C_UAS"))
                    st.session_state["cuas_inner_choke"] = str(best_blue.get("inner_choke", "Extra Full"))
                st.session_state["_active_cuas_profile"] = {
                    "label": f"ML-Optimized {best_blue['n_barrels']}B {best_blue['layout'].title()}",
                    "layout_type": best_blue["layout"],
                    "n_barrels": int(best_blue["n_barrels"]),
                    "ammunition_preset": best_blue["ammo"],
                    "choke": best_blue["choke"],
                    "pointing_mode": best_blue["pointing"],
                    "splay_angle_deg": float(best_blue["splay_angle_deg"]),
                    "splay_angle_mrad": float(best_blue.get("splay_angle_mrad", 0.0)),
                    "firing_mode": best_blue["firing_mode"],
                    "ripple_delay_ms": float(best_blue["ripple_delay_ms"]),
                    "burst_count": int(best_blue.get("burst_count", 1)),
                    "hybrid_mode": bool(best_blue.get("hybrid_mode", False)),
                    "inner_ammunition_preset": str(best_blue.get("inner_ammunition_preset", "")),
                    "inner_choke": str(best_blue.get("inner_choke", "")),
                }
                st.success("Applied evolved ML turret (including hybrid ring mix if enabled) to Stage 1!")
        with act_col3:
            if act_col3.button("💾 Save Policy", key="cuas_adv_save_lib_btn", width="stretch"):
                ScenarioLibrary().save_scenario(
                    name=f"ML Nash {best_blue['name']}",
                    domain="cuas_ml",
                    config=best_blue,
                    metrics={
                        "nash_value": adv_res["nash_value"],
                        "cv_r2": adv_res["gp_surrogate_diagnostics"]["cv_r2"],
                        "evolved_worst_case": wc_gain["evolved_worst_case"],
                    },
                    tags=["cuas", "ml", "nash"],
                )
                st.success("Saved policy to Scenario Library!")
        with act_col4:
            adv_bbls = generate_multishotgun_array_layout(
                layout_type=str(best_blue["layout"]),
                n_barrels=int(best_blue["n_barrels"]),
                default_choke=str(best_blue["choke"]),
                default_ammunition_preset=str(best_blue["ammo"]),
            )
            adv_dxf = export_muzzle_plate_dxf(
                barrels=adv_bbls,
                layout_type=str(best_blue["layout"]),
                title=str(best_blue["name"]),
                splay_angle_mrad=float(best_blue.get("splay_angle_mrad", 0.0)),
                pointing_mode=str(best_blue["pointing"]),
            )
            act_col4.download_button(
                "📐 CAD (.dxf)",
                data=adv_dxf.encode("utf-8"),
                file_name="ml_evolved_muzzle_plate.dxf",
                mime="application/dxf",
                key="cuas_adv_dl_dxf_btn",
                width="stretch",
            )

        # 7 Visualizer & Autonomy Sub-Tabs
        adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5, adv_tab6, adv_tab7 = st.tabs([
            "📈 Arms-Race & GP Convergence",
            "🎯 Red-vs-Blue Payoff Matrix (Nash)",
            "🔬 ML Feature Importance & Dataset",
            "🛡️ Live Threat Defense Recommender",
            "🧠 Dynamic Fire-Control ML Policy (RL)",
            "🗺️ Multi-Turret 3D Battlespace Co-Evolution",
            "⚔️ Beat the AI — Interactive Duel & Root-Cause",
        ])

        with adv_tab1:
            st.plotly_chart(create_coevolution_trajectory_figure(adv_res), width="stretch")
            gh_df = pd.DataFrame(adv_res["generation_history"])
            st.markdown("##### Generational Convergence History")
            st.dataframe(gh_df, width="stretch")
            deadly_red = adv_res["deadliest_red_threat"]
            st.warning(
                f"🚨 **Deadliest Red Drone Threat Discovered**: {deadly_red['drone_model']} "
                f"(Swarm Size: {deadly_red['swarm_size']}, Tactic: `{deadly_red['tactic']}`, "
                f"Evasion: `{deadly_red['evasion']}` @ {deadly_red['maneuver_g']}G, "
                f"Speed: {deadly_red['speed_mps']} m/s, Altitude: {deadly_red['altitude_m']} m)"
            )

        with adv_tab2:
            st.plotly_chart(create_nash_payoff_heatmap_figure(adv_res), width="stretch")
            np1, np2 = st.columns(2)
            with np1:
                st.markdown("##### Blue Nash Support Strategies (p > 0)")
                blue_nash_rows = []
                for i, p in enumerate(adv_res["nash_blue_strategy"]):
                    if p > 0.005:
                        b_cfg = adv_res["blue_pool"][i]
                        blue_nash_rows.append({
                            "Strategy ID": f"Blue-{i}",
                            "Nash Weight p": round(float(p), 3),
                            "Barrels": b_cfg["n_barrels"],
                            "Layout": b_cfg["layout"],
                            "Ammo": b_cfg["ammo"],
                            "Choke": b_cfg["choke"],
                            "Pointing": b_cfg["pointing"],
                            "Ripple (ms)": b_cfg["ripple_delay_ms"],
                        })
                st.dataframe(pd.DataFrame(blue_nash_rows), width="stretch")
            with np2:
                st.markdown("##### Red Nash Support Threats (q > 0)")
                red_nash_rows = []
                for j, q in enumerate(adv_res["nash_red_strategy"]):
                    if q > 0.005:
                        r_cfg = adv_res["red_pool"][j]
                        red_nash_rows.append({
                            "Threat ID": f"Red-{j}",
                            "Nash Weight q": round(float(q), 3),
                            "Drone Model": r_cfg["drone_model"],
                            "Swarm Size": r_cfg["swarm_size"],
                            "Tactic": r_cfg["tactic"],
                            "Evasion": r_cfg["evasion"],
                            "Maneuver G": r_cfg["maneuver_g"],
                            "Speed (m/s)": r_cfg["speed_mps"],
                        })
                st.dataframe(pd.DataFrame(red_nash_rows), width="stretch")

        with adv_tab3:
            st.plotly_chart(create_ml_feature_importance_figure(adv_res), width="stretch")
            st.caption(
                "Permutation feature importance calculated on the Gaussian Process surrogate: "
                "drop in 5-fold cross-validation R² when shuffling each feature across the evaluated design space."
            )
            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                csv_data = export_adversarial_dataset_csv(adv_res)
                st.download_button(
                    "📥 Download Adversarial Dataset (.csv)",
                    data=csv_data,
                    file_name="cuas_adversarial_dataset.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with exp_c2:
                st.download_button(
                    "📥 Download Full Co-Evolution Report (.json)",
                    data=json.dumps(adv_res, indent=2),
                    file_name="cuas_adversarial_report.json",
                    mime="application/json",
                    width="stretch",
                )

        with adv_tab4:
            st.markdown("##### 🎯 Targeted Threat Policy Optimization")
            st.caption(
                "Specify an adversarial drone raid scenario. The Gaussian Process surrogate searches candidate "
                "shotgun protection configurations and predicts the optimal Pareto defense policies."
            )
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                tr_model = st.selectbox("Threat Drone", list(drone_specs_18.keys()), index=1, key="cuas_adv_tr_model")
                tr_swarm = st.slider("Target Swarm Size", 1, 12, 5, 1, key="cuas_adv_tr_swarm")
            with tc2:
                tr_tactic = st.selectbox("Attack Tactic", ["simultaneous_divergent", "convergent_pincer", "staggered_trail", "low_altitude_terrain_mask"], index=0, key="cuas_adv_tr_tactic")
                tr_evasion = st.selectbox("Evasive Maneuver", ["weaving_jink", "barrel_roll", "random_walk", "corkscrew", "none"], index=0, key="cuas_adv_tr_evasion")
            with tc3:
                tr_g = st.slider("Maneuver Load (G)", 0.5, 5.0, 3.0, 0.5, key="cuas_adv_tr_g")
                tr_speed = st.slider("Threat Speed (m/s)", 15.0, 75.0, 35.0, 5.0, key="cuas_adv_tr_spd")
            with tc4:
                tr_range = st.slider("Engagement Range (m)", 15.0, 70.0, 35.0, 5.0, key="cuas_adv_tr_rng")
                tr_stealth = st.slider("Fiber-Optic Stealth", 0.0, 1.0, 0.2, 0.1, key="cuas_adv_tr_stealth")

            target_threat_dict = {
                "drone_model": tr_model,
                "swarm_size": int(tr_swarm),
                "tactic": tr_tactic,
                "evasion": tr_evasion,
                "maneuver_g": float(tr_g),
                "speed_mps": float(tr_speed),
                "altitude_m": 25.0,
                "range_m": float(tr_range),
                "stealth_level": float(tr_stealth),
            }

            rec_out = recommend_ml_defense_policy(
                coevolution_result=adv_res,
                target_threat=target_threat_dict,
                top_k=5,
                verify_with_mc=True,
            )

            st.plotly_chart(create_ml_policy_pareto_figure(rec_out), width="stretch")

            top_policies = rec_out["recommended_policies"]
            rec_rows = []
            for rank, pol in enumerate(top_policies, start=1):
                b_cfg = pol["blue_config"]
                mc_v = pol.get("mc_verified", {})
                rec_rows.append({
                    "Rank": f"#{rank}",
                    "Predicted Defense Score": f"{pol['predicted_defense_score']:.1f}",
                    "Verified Kill Rate": f"{mc_v.get('kill_rate', 0.0) * 100:.1f}%",
                    "Verified Zero Breach": f"{mc_v.get('zero_breach_rate', 0.0) * 100:.1f}%",
                    "Peak Mount Load (kN)": f"{mc_v.get('peak_mount_load_kn', pol['predicted_mount_load_kn']):.2f}",
                    "Barrels": b_cfg["n_barrels"],
                    "Layout": b_cfg["layout"],
                    "Ammunition": b_cfg["ammo"],
                    "Choke": b_cfg["choke"],
                    "Pointing": b_cfg["pointing"],
                    "Splay (°)": b_cfg["splay_angle_deg"],
                    "Ripple (ms)": b_cfg["ripple_delay_ms"],
                })
            st.dataframe(pd.DataFrame(rec_rows), width="stretch")

            if top_policies:
                p1 = top_policies[0]["blue_config"]
                if st.button(
                    f"⚡ Apply Policy #1 ({p1['n_barrels']}B {p1['layout'].title()}) to Stage 1 Array",
                    type="secondary",
                    key="cuas_adv_apply_p1_btn",
                ):
                    st.session_state["cuas_layout"] = str(p1["layout"])
                    st.session_state["cuas_n_barrels"] = int(p1["n_barrels"])
                    st.session_state["cuas_ammo"] = str(p1["ammo"])
                    st.session_state["cuas_choke"] = str(p1["choke"])
                    st.session_state["cuas_pointing"] = str(p1["pointing"])
                    st.session_state["cuas_splay"] = float(p1["splay_angle_deg"])
                    st.session_state["cuas_fire_mode"] = str(p1["firing_mode"])
                    st.session_state["cuas_ripple_dt"] = float(p1["ripple_delay_ms"])
                    st.session_state["_active_cuas_profile"] = {
                        "label": f"ML Policy #1: {p1['n_barrels']}B {p1['layout'].title()}",
                        "layout_type": p1["layout"],
                        "n_barrels": int(p1["n_barrels"]),
                        "ammunition_preset": p1["ammo"],
                        "choke": p1["choke"],
                        "pointing_mode": p1["pointing"],
                        "splay_angle_deg": float(p1["splay_angle_deg"]),
                        "firing_mode": p1["firing_mode"],
                        "ripple_delay_ms": float(p1["ripple_delay_ms"]),
                        "burst_count": int(p1.get("burst_count", 1)),
                        "hybrid_mode": bool(p1.get("hybrid_mode", False)),
                        "inner_ammunition_preset": str(p1.get("inner_ammunition_preset", "")),
                        "inner_choke": str(p1.get("inner_choke", "")),
                    }
                    st.success(f"Applied Policy #1 ({p1['n_barrels']}B {p1['layout']}) to Stage 1 Array Setup!")

        with adv_tab5:
            st.markdown("##### 🧠 Dynamic Shot-by-Shot Fire-Control ML Policy (Evolution Strategies)")
            st.caption(
                "Trains a 147-weight neural controller deciding target priority, adaptive burst size (1..max), "
                "and hold-fire gating in real time during active swarm engagements, benchmarked directly against heuristic rules."
            )
            rc1, rc2, rc3 = st.columns([1.0, 1.0, 1.4])
            with rc1:
                rl_gens = st.slider("Training Generations", 6, 20, 12, 2, key="cuas_rl_gens")
            with rc2:
                rl_eps = st.slider("Benchmark Episodes / Threat", 8, 24, 12, 4, key="cuas_rl_eps")
            with rc3:
                st.write("")
                if st.button("🚀 Train & Benchmark ML Fire-Control Policy", type="primary", key="cuas_rl_train_btn", width="stretch"):
                    with st.spinner("Training Neural Fire-Control Policy via Antithetic ES..."):
                        rl_res = _cached_train_rl_policy(
                            blue_config_json=json.dumps(best_blue, sort_keys=True),
                            n_generations=int(rl_gens),
                            pop_size=12,
                            n_eval_episodes=int(rl_eps),
                            seed=int(seed_val),
                            physics_fidelity_json=_get_active_physics_fidelity_json(),
                        )
                    st.session_state["_cuas_rl_policy_res"] = rl_res

            rl_cur = st.session_state.get("_cuas_rl_policy_res")
            if rl_cur is not None:
                rsm = rl_cur["summary"]
                rk1, rk2, rk3, rk4 = st.columns(4)
                rk1.metric("RL Zero-Breach Rate", f"{rsm['rl_zero_breach_pct']:.1f}%", delta=f"{rsm['uplift_zero_breach_pp']:+.1f} pp vs {rsm['best_heuristic_name']}")
                rk2.metric("RL Swarm Kill Rate", f"{rsm['rl_kill_rate_pct']:.1f}%", delta=f"Avg Burst: {rsm['rl_mean_burst_size']:.2f} bbls")
                rk3.metric("Rounds / Kill", f"{rsm['rl_mean_rounds_per_kill']:.2f}", delta=f"{rsm['rounds_saved_pct']:+.1f}% rounds saved")
                rk4.metric("Hold-Fire Wait Gate", f"{rsm['rl_hold_fire_rate_pct']:.1f}%", delta="Long-range patience rate")
                st.plotly_chart(create_rl_policy_benchmark_figure(rl_cur), width="stretch")
                st.plotly_chart(create_rl_action_heatmap_figure(rl_cur), width="stretch")
                st.dataframe(pd.DataFrame(rl_cur["benchmark_rows"]), width="stretch", hide_index=True)
                st.download_button(
                    "📥 Export Neural Fire-Control Weights (.json)",
                    data=json.dumps(rl_cur["policy_dict"], indent=2).encode("utf-8"),
                    file_name="cuas_neural_fire_control_policy.json",
                    mime="application/json",
                    key="cuas_rl_dl_btn",
                    width="stretch",
                )

            st.divider()
            st.markdown("##### ⚡ Live Hardware-in-the-Loop (HIL) Telemetry Replay & Command Stream")
            st.caption(
                "Step the active Neural Fire-Control Policy (or default controller) millisecond-by-millisecond over recorded ENU/GPS drone telemetry to inspect aim commands, burst sizing, and hold-fire gates."
            )
            hl1, hl2, hl3 = st.columns(3)
            with hl1:
                hil_scn = st.selectbox(
                    "Telemetry Track Scenario",
                    ["evasive_fpv_dive", "multi_drone_pincer", "low_altitude_terrain_pop"],
                    key="cuas_hil_scn",
                )
            with hl2:
                hil_dt = st.slider("Control Loop Step dt (ms)", 50.0, 200.0, 100.0, 25.0, key="cuas_hil_dt")
            with hl3:
                hil_breach = st.slider("Perimeter Keepout (m)", 8.0, 20.0, 12.0, 1.0, key="cuas_hil_br")

            _hil_pol_json = json.dumps(rl_cur["policy_dict"], sort_keys=True) if (rl_cur and rl_cur.get("policy_dict")) else ""
            hil_res = _cached_hil_policy_trace(
                policy_json=_hil_pol_json,
                track_scenario_or_csv=str(hil_scn),
                blue_config_json=json.dumps(best_blue, sort_keys=True),
                calibration_profile_json=_get_effective_cal_twin_json(),
                dt_ms=float(hil_dt),
                breach_radius_m=float(hil_breach),
                physics_fidelity_json=_get_active_physics_fidelity_json(),
            )
            hsm = hil_res["summary"]
            hm_a, hm_b, hm_c, hm_d = st.columns(4)
            hm_a.metric("Control Frames Stepped", f"{hsm['n_frames']}", delta=f"dt={hsm['dt_ms']:.0f} ms")
            hm_b.metric("Fire Commands Issued", f"{hsm['fire_commands']}", delta=f"{hsm['total_rounds_commanded']} rounds")
            hm_c.metric("Hold-Fire Gate Frames", f"{hsm['hold_fire_frames']}", delta=hsm["policy_name"][:24])
            hm_d.metric("Neutralized / Breached", f"{hsm['targets_neutralized']} / {hsm['targets_breached']}", delta=f"of {hsm['total_tracks']} tracks")
            st.plotly_chart(create_hil_policy_timeline_figure(hil_res), width="stretch", key="cuas_hil_fig")
            st.dataframe(pd.DataFrame(hil_res["frames"][:25]), width="stretch", hide_index=True)
            hdl1, hdl2 = st.columns(2)
            hdl1.download_button(
                "📥 Download HIL Command Stream (.json)",
                data=hil_res["json_stream"].encode("utf-8"),
                file_name="cuas_hil_command_stream.json",
                mime="application/json",
                key="cuas_hil_dl_json",
                width="stretch",
            )
            hdl2.download_button(
                "📥 Download HIL Command Trace (.csv)",
                data=hil_res["csv_stream"].encode("utf-8"),
                file_name="cuas_hil_command_stream.csv",
                mime="text/csv",
                key="cuas_hil_dl_csv",
                width="stretch",
            )

        with adv_tab6:
            st.markdown("##### 🗺️ Multi-Turret 3D Battlespace & Heterogeneous Swarm Co-Evolution")
            st.caption(
                "Co-evolves multi-turret site geometry & layered defense pairings against decoy-sponge + fiber-optic + heavy-bomber packages over DEM terrain."
            )
            dem_opts = get_terrain_dem_presets()
            bc1, bc2, bc3 = st.columns([1.4, 0.9, 1.3])
            with bc1:
                bs_dem = st.selectbox(
                    "Battlespace DEM Terrain",
                    list(dem_opts.keys()),
                    format_func=lambda k: dem_opts[k]["name"],
                    index=1,
                    key="cuas_bs_dem",
                )
            with bc2:
                bs_gens = st.slider("Battlespace Gens", 1, 5, 2, 1, key="cuas_bs_gens")
            with bc3:
                st.write("")
                if st.button("⚔️ Run Multi-Turret 3D Battlespace Co-Evolution", type="primary", key="cuas_bs_run_btn", width="stretch"):
                    with st.spinner("Co-evolving Cooperative Multi-Turret Site & Mixed Red Swarms..."):
                        bs_res = _cached_battlespace_coevolution(
                            terrain_preset=str(bs_dem),
                            n_generations=int(bs_gens),
                            n_mc_trials=5,
                            max_turrets=4,
                            seed=int(seed_val),
                        )
                    st.session_state["_cuas_bs_res"] = bs_res

            bs_cur = st.session_state.get("_cuas_bs_res")
            if bs_cur is not None:
                bsm = bs_cur["summary"]
                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Battlespace Nash V*", f"{bsm['nash_value']:.2f}", delta=bsm["terrain_preset"])
                bm2.metric("Best Site + Layer", bsm["best_outer_layer"], delta=bsm["best_blue_name"][:28])
                bm3.metric("Duel Zero-Breach", f"{bsm['duel_zero_breach_pct']:.1f}%", delta=f"Kill Rate: {bsm['duel_kill_rate_pct']:.1f}%")
                bm4.metric("Deadliest Red Package", bsm["deadliest_red_name"][:24], delta=f"Outer Kills: {bsm['duel_outer_kills']:.1f}")
                st.plotly_chart(create_battlespace_coevolution_figure(bs_cur), width="stretch")
                st.button(
                    "🚀 Launch Nash Duel in 3D Tactical Raid Simulator",
                    type="primary",
                    on_click=_launch_battlespace_to_raid,
                    args=(bs_cur["raid_handoff_payload"],),
                    key="cuas_bs_launch_raid_btn",
                    width="stretch",
                )

        with adv_tab7:
            st.markdown("##### ⚔️ Beat the AI — Interactive Duel & Failure-Mode Root-Cause Diagnostic")
            st.caption(
                "Test your Stage 1 turret against the AI's deadliest evolved swarm, or design a custom Red attack to challenge the AI's Nash-optimal turret."
            )
            duel_mode = st.radio(
                "Duel Perspective",
                [
                    "Test My Stage 1 Turret vs AI's Deadliest Red Threat",
                    "Challenge AI's Nash-Optimal Turret with Custom Red Attack",
                ],
                horizontal=True,
                key="cuas_duel_mode",
            )
            s1_prof = st.session_state.get("_active_cuas_profile") or best_blue
            if duel_mode.startswith("Test My"):
                duel_blue = s1_prof
                duel_red = adv_res["deadliest_red_threat"]
            else:
                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1:
                    d_mod = st.selectbox("Red Airframe", list(drone_specs_18.keys()), index=0, key="cuas_duel_mod")
                with dc2:
                    d_sw = st.slider("Red Swarm Size", 1, 12, 7, 1, key="cuas_duel_sw")
                with dc3:
                    d_mg = st.slider("Evasive Load (G)", 0.0, 4.5, 3.5, 0.5, key="cuas_duel_mg")
                with dc4:
                    d_spd = st.slider("Attack Speed (m/s)", 14.0, 42.0, 32.0, 2.0, key="cuas_duel_spd")
                duel_blue = best_blue
                duel_red = {
                    "drone_model": str(d_mod),
                    "swarm_size": int(d_sw),
                    "maneuver_profile": "terminal_popup_jink" if d_mg >= 2.5 else "sinusoidal_weave",
                    "maneuver_g": float(d_mg),
                    "drone_speed_mps": float(d_spd),
                    "attack_tactic": "narrow_corridor_saturation",
                    "fiber_optic_fraction": 0.6,
                }

            duel_out = run_red_vs_blue_duel_diagnostic(
                blue_config=duel_blue,
                red_config=duel_red,
                mount_load_limit_kn=float(mount_load_limit_kn),
                n_episodes=24,
                seed=int(seed_val),
                physics_fidelity=get_active_physics_fidelity(),
            )
            dm = duel_out["matchup"]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Duel Verdict", duel_out["verdict"], delta=f"Utility: {dm['blue_utility']:.1f}/100")
            d2.metric("Zero-Breach Stop Rate", f"{dm['zero_breach_pct']:.1f}%", delta=f"Swarm Kill Rate: {dm['kill_rate_pct']:.1f}%")
            d3.metric("Dominant Root Cause", max(
                [
                    ("Reload Gap", duel_out["failure_modes"]["reload_gap_pct"]),
                    ("Slew Bottleneck", duel_out["failure_modes"]["slew_bottleneck_pct"]),
                    ("Tracking Lag", duel_out["failure_modes"]["tracking_evasion_miss_pct"]),
                    ("Lethality/Spread", duel_out["failure_modes"]["insufficient_lethality_spread_pct"]),
                ],
                key=lambda x: x[1],
            )[0], delta=f"{duel_out['failure_modes']['total_leakers']} total leakers")
            d4.metric("Turret Mass / Recoil", f"{duel_out['swap_c_summary']['total_mass_kg']:.1f} kg", delta=f"{dm['peak_mount_force_kn']:.2f} kN peak")
            st.plotly_chart(create_duel_root_cause_figure(duel_out), width="stretch")
            st.markdown("**Actionable Engineering Prescriptions**")
            for pr in duel_out["prescriptions"]:
                st.info(pr)

