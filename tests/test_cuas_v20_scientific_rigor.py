"""Comprehensive verification suite for v2.0.0 Scientific Rigor & Deep Physics upgrades."""

from __future__ import annotations

import json
import math

from fastapi.testclient import TestClient
import numpy as np
from typer.testing import CliRunner

import ammo_mc
from ammo_mc.api import app as fastapi_app
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_analytics import compare_cuas_architectures_ab
from ammo_mc.cuas_ballistics_v2 import (
    cipm_2007_air_density,
    compute_aspect_dependent_rcs_dbsm,
    compute_saami_recoil_impulse,
    compute_serial_dead_time_budget,
    correct_saami_instrumental_velocity,
    evaluate_markov_redundant_kill_tree,
    faa_shear_plugging_v50,
    generate_singer_evasive_trajectory,
    integrate_pellet_trajectory_rk4,
    sample_heavy_tailed_atr_error_mrad,
    sphere_drag_compton_1996,
    sphere_drag_nol_1972,
    sphere_drag_walsh_1976,
    thor_tr47_metallic_penetration,
    velocity_ogive_peff,
    wen_2000_frp_ballistic_limit,
)
from ammo_mc.cuas_defense import (
    evaluate_evasive_drone_state,
    simulate_closed_loop_fcs_engagement_mc,
    simulate_turret_recoil_and_gimbal_loads,
)
from ammo_mc.cuas_rigor import (
    check_pellet_count_consistency,
    compute_agresti_coull_ci,
    compute_clopper_pearson_ci,
    compute_wilson_ci,
    evaluate_alpha_spending_checkpoint,
    evaluate_closing_engagement_window,
    get_contextual_counter_evidence,
    optimal_pattern_radius_closed_form,
    pk_salvo_brackets,
    pk_salvo_hierarchical_mc,
    pk_single_shot_closed_form,
    resolve_verification_grade,
    run_workbook_validation_cases,
    wilson_worst_case_sample_size,
)
from ammo_mc.cuas_safety_envelope import evaluate_cuas_collateral_safety_envelope
from ammo_mc.literature import build_literature_audit_summary
from ammo_mc.small_arms_pkg.arrays import (
    generate_multishotgun_array_layout,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.vis_cuas_vv import (
    create_dead_time_waterfall_figure,
    create_salvo_brackets_and_window_figure,
    create_workbook_vv_matrix_figure,
)


def test_walsh_nol_compton_cipm_and_saami_velocity() -> None:
    assert math.isclose(sphere_drag_walsh_1976(1.0), 0.71, rel_tol=1e-9)
    assert math.isclose(sphere_drag_nol_1972(1.0), 0.71, rel_tol=1e-9)
    assert math.isclose(sphere_drag_compton_1996(0.5, False), 0.52, rel_tol=1e-9)
    assert sphere_drag_compton_1996(1.2, True) > sphere_drag_compton_1996(1.2, False) * 1.2

    dry = cipm_2007_air_density(15.0, 101325.0, 0.0)
    humid = cipm_2007_air_density(30.0, 101325.0, 0.85)
    assert math.isclose(dry["rho_kg_m3"], 1.2250, rel_tol=5e-4)
    assert humid["rho_kg_m3"] < dry["rho_kg_m3"]

    saami_v = correct_saami_instrumental_velocity(405.0, 3.30, 11200.0, gauge_dist_ft=3.0)
    assert saami_v["v0_muzzle_mps"] > 405.0
    assert math.isclose(saami_v["gauge_dist_m"], 0.9144, rel_tol=1e-6)


def test_wen_faa_thor_markov_and_saami_recoil() -> None:
    wen = wen_2000_frp_ballistic_limit(0.003, 0.005, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
    assert 100.0 <= wen["v_bl_mps"] <= 122.0
    assert wen["verification_grade"] == "V1"

    faa = faa_shear_plugging_v50(0.00134, 0.0061, 0.0015, material="2024-T351 Al")
    assert faa["caveat_id"] == "TE-C09"
    assert faa["verification_grade"] == "V3"

    thor = thor_tr47_metallic_penetration(0.21, 3.30, 230.0, 0.8, material="2024T-3 Al", pellet_material="Lead")
    assert len(thor["warnings"]) >= 2
    assert thor["verification_grade"] == "V3"

    mk = evaluate_markov_redundant_kill_tree(n_rotors=6, max_lost_rotors_survivable=1, expected_hits_lambda=3.5)
    assert 0.0 < mk["p_kill"] < 1.0

    saami = compute_saami_recoil_impulse(
        shot_mass_g=589.9 * 0.06479891,
        wad_mass_g=0.0,
        propellant_mass_g=33.4 * 0.06479891,
        v0_mps=1275.0 * 0.3048,
        gun_mass_kg=7.0 * 0.45359237,
        gas_factor_fg=1.50,
    )
    assert math.isclose(saami["impulse_ns"], 16.12, rel_tol=5e-3)
    assert math.isclose(saami["free_recoil_energy_j"], 40.97, rel_tol=5e-3)


def test_rigor_grades_consistency_salvo_brackets_and_hierarchical_mc() -> None:
    vg = resolve_verification_grade({"drag": "V1", "dispersion": "V3", "vulnerability": "V4"}, calibrated=False)
    assert vg["grade"] == "V4"
    vg_cal = resolve_verification_grade({"drag": "V1", "dispersion": "V4", "vulnerability": "V3"}, calibrated=True)
    assert vg_cal["grade"] == "V3"

    chk_ok = check_pellet_count_consistency(35.4, 3.30, 11200.0, 168)
    assert chk_ok["consistent"] is True
    chk_bad = check_pellet_count_consistency(32.0, 3.30, 11200.0, 400)
    assert chk_bad["consistent"] is False and chk_bad["warning"] is not None

    br = pk_salvo_brackets(1.529181273, 0.307791848, 0.208673604, 4)
    assert br["pk_shared_lower"] < br["pk_independent_upper"]
    assert math.isclose(br["pk_shared_lower"], 0.954931, rel_tol=1e-4)
    assert math.isclose(br["pk_independent_upper"], 0.980413, rel_tol=1e-4)

    hmc = pk_salvo_hierarchical_mc(41, 0.02, 0.4, 0.4, 0.1, 0.1, n_guns=4, nb_kappa=6.0, shot_sigma_cv=0.12, n_trials=4000)
    assert 0.4 <= hmc["pk_hierarchical"] <= 0.85

    opt = optimal_pattern_radius_closed_form(41.0, 0.02, 0.30, current_sigma_p_m=0.80, splay_angle_mrad=6.0)
    assert math.isclose(opt["r_opt_disc_m"], 0.4656, abs_tol=5e-4)
    assert opt["splay_warning"] is not None

    assert len(get_contextual_counter_evidence("array", n_barrels=6, splay_mrad=5.0)) >= 3


def test_usable_window_dead_time_singer_safety_crn_and_array() -> None:
    w_slow = evaluate_closing_engagement_window(drone_speed_mps=15.0, drone_altitude_m=10.0, warhead_blast_radius_m=6.0)
    w_fast = evaluate_closing_engagement_window(drone_speed_mps=45.0, drone_altitude_m=30.0, warhead_blast_radius_m=15.0)
    assert w_fast["summary"]["r_min_safe_m"] > w_slow["summary"]["r_min_safe_m"]

    dt_bud = compute_serial_dead_time_budget("eo_60hz_1stream", tof_s=0.14, a_unmod_mps2=5.0)
    assert math.isclose(dt_bud["t_dead_ms"], 140.9, abs_tol=0.2)

    rng = np.random.default_rng(7)
    atr = sample_heavy_tailed_atr_error_mrad(rng, 2000)
    assert float(np.max(np.abs(atr))) > 3.0
    assert compute_aspect_dependent_rcs_dbsm(-20.0, aspect_deg=35.0, null_depth_db=30.0) < -22.0

    sg = generate_singer_evasive_trajectory(np.linspace(-0.5, 0.2, 15), 40.0, 22.0, 90.0, rng=rng)
    assert len(sg["y"]) == 15
    ev = evaluate_evasive_drone_state(0.1, 40.0, 22.0, 90.0, maneuver_profile="singer_correlated")
    assert "y" in ev

    sim_v2 = simulate_anti_drone_multishotgun_mc(
        n_barrels=4,
        drag_model="walsh_1976",
        drag_multiplier_kd=1.15,
        relative_humidity_pct=65.0,
        shot_sigma_cv=0.12,
        penetration_model="wen_cfrp",
        drone_model="fpv_quad_cfrp_hardened",
        n_volleys=8,
    )
    sm = sim_v2["summary"]
    assert "salvo_brackets" in sm and "verification_grade" in sm and "pellet_count_check" in sm

    fcs_v2 = simulate_closed_loop_fcs_engagement_mc(
        maneuver_profile="singer_correlated",
        use_serial_dead_time_chain=True,
        use_heavy_tailed_atr=True,
        enable_aspect_rcs_nulls=True,
        n_volleys=8,
    )
    assert fcs_v2["summary"]["use_serial_dead_time_chain"] is True

    bbls = generate_multishotgun_array_layout("circular", 4)
    rec = simulate_turret_recoil_and_gimbal_loads(bbls, recoil_gas_model="saami_proportional", saami_gas_factor_fg=1.50)
    assert rec["summary"]["recoil_gas_model"] == "saami_proportional"

    sdz = evaluate_cuas_collateral_safety_envelope(n_barrels=4, n_sample_pellets=120)
    assert sdz["doctrine_reference_overlays"]["uk_jsp_403_vol2_ch2_table3"]["lateral_cone_mils"] == 80.0
    assert sdz["doctrine_reference_overlays"]["uk_jsp_403_vol2_ch2_table3"]["vertical_cone_mils"] == 200.0
    assert sdz["falling_drone_debris"]["impact_energy_j"] > 50.0
    assert "dual_criterion_hazards" in sdz

    ab = compare_cuas_architectures_ab(use_common_random_numbers=True)
    assert ab["summary"]["crn_active"] is True
    assert "kill_chain_pk_ci95_pct" in ab["scorecard"][0]

    # Plots render cleanly without error
    assert len(create_salvo_brackets_and_window_figure(w_slow).data) >= 4
    assert len(create_dead_time_waterfall_figure(dt_bud).data) >= 1


def test_workbook_vv_matrix_cli_api_and_100_sources() -> None:
    assert ammo_mc.__version__ == "2.0.0"
    wb_vv = run_workbook_validation_cases()
    assert wb_vv["summary"]["failed"] == 0
    assert wb_vv["summary"]["pass_rate_pct"] == 100.0
    assert len(create_workbook_vv_matrix_figure(wb_vv).data) == 2

    lit = build_literature_audit_summary()
    assert lit["total_sources"] == 100
    assert lit["linked_source_coverage_pct"] == 100.0

    runner = CliRunner()
    for args in (
        ["cuas", "salvo-brackets", "--json"],
        ["cuas", "engagement-window", "--json"],
        ["cuas", "rigor-check", "--json"],
        ["cuas", "workbook-vv", "--json"],
    ):
        res = runner.invoke(cli_app, args)
        assert res.exit_code == 0, res.output
        json.loads(res.stdout)

    client = TestClient(fastapi_app)
    assert client.get("/health").json()["version"] == "2.0.0"
    r1 = client.post("/cuas/rigor/salvo-brackets", json={"lambda0": 1.529, "sigma_m": 0.308, "sigma_a_m": 0.209, "n_shots": 4})
    assert r1.status_code == 200 and r1.json()["pk_shared_lower"] < r1.json()["pk_independent_upper"]
    r2 = client.post("/cuas/rigor/engagement-window", json={"drone_speed_mps": 25.0, "drone_altitude_m": 15.0})
    assert r2.status_code == 200 and "r_min_safe_m" in r2.json()["summary"]
    r3 = client.post("/cuas/rigor/pellet-consistency", json={"payload_mass_g": 35.4, "diam_mm": 3.30, "density_kg_m3": 11200.0, "claimed_pellets": 168})
    assert r3.status_code == 200 and r3.json()["consistent"] is True
    r4 = client.get("/cuas/vv/workbook-cases")
    assert r4.status_code == 200 and r4.json()["summary"]["failed"] == 0
