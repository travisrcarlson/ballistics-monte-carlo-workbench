"""Tests for Adversarial Red-vs-Blue Competing Monte Carlo Co-Evolution & GP ML Optimizer."""

from __future__ import annotations

import csv
import io
import math
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest
from typer.testing import CliRunner

from ammo_mc.api import app
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_adversarial import (
    FEATURE_NAMES,
    GaussianProcessSurrogate,
    encode_matchup_features,
    evaluate_red_vs_blue_matchup_mc,
    export_adversarial_dataset_csv,
    normalize_blue_config,
    normalize_red_config,
    recommend_ml_defense_policy,
    run_adversarial_coevolution_ml,
    solve_zero_sum_nash_equilibrium,
)


# -------------------------------------------------------------------------
# 1. EXACT ZERO-SUM NASH EQUILIBRIUM SOLVER
# -------------------------------------------------------------------------

def test_nash_solver_matching_pennies() -> None:
    """Verify exact LP solution on classical 2x2 Matching Pennies."""
    # Blue wins if matching (+1), Red wins if mismatching (-1 for Blue)
    M = np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])
    nash = solve_zero_sum_nash_equilibrium(M)

    assert len(nash["blue_strategy"]) == 2
    assert len(nash["red_strategy"]) == 2
    np.testing.assert_allclose(nash["blue_strategy"], [0.5, 0.5], atol=1e-3)
    np.testing.assert_allclose(nash["red_strategy"], [0.5, 0.5], atol=1e-3)
    assert abs(nash["nash_value"]) < 1e-3
    assert nash["duality_gap"] < 1e-4


def test_nash_solver_rock_paper_scissors() -> None:
    """Verify exact LP solution on symmetric 3x3 Rock-Paper-Scissors."""
    M = np.array([
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ])
    nash = solve_zero_sum_nash_equilibrium(M)

    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(nash["blue_strategy"], expected, atol=1e-3)
    np.testing.assert_allclose(nash["red_strategy"], expected, atol=1e-3)
    assert abs(nash["nash_value"]) < 1e-3
    assert nash["duality_gap"] < 1e-4


def test_nash_solver_strictly_positive_asymmetric() -> None:
    """Verify LP stability and Charnes-Cooper shift on arbitrary non-zero matrix."""
    M = np.array([
        [15.0, 45.0, 30.0],
        [40.0, 20.0, 35.0],
        [25.0, 30.0, 50.0],
    ])
    nash = solve_zero_sum_nash_equilibrium(M)

    assert nash["duality_gap"] < 1e-4
    assert sum(nash["blue_strategy"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(nash["red_strategy"]) == pytest.approx(1.0, abs=1e-3)
    assert all(p >= -1e-5 for p in nash["blue_strategy"])
    assert all(q >= -1e-5 for q in nash["red_strategy"])
    assert 20.0 <= nash["nash_value"] <= 45.0


# -------------------------------------------------------------------------
# 2. PURE NUMPY/SCIPY GAUSSIAN PROCESS SURROGATE
# -------------------------------------------------------------------------

def test_gaussian_process_surrogate_synthetic_fit() -> None:
    """Test GP fitting, out-of-sample prediction variance, and CV metrics."""
    rng = np.random.default_rng(123)
    n_samples = 40
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 3))
    # Target function with slight noise
    y = np.sin(X[:, 0]) + 0.6 * np.cos(X[:, 1]) + rng.normal(0.0, 0.05, size=n_samples)

    gp = GaussianProcessSurrogate(lengthscale=1.5, signal_std=1.0, noise_std=0.1)
    gp.fit(X, y, optimize_hyperparams=False)

    # Prediction on training points should be close to ground truth
    mu, std = gp.predict(X, return_std=True)
    assert std is not None
    assert np.mean(np.abs(mu - y)) < 0.25
    assert np.all(std > 0.0)

    # Uncertainty should increase away from the training domain
    X_far = np.array([[10.0, 10.0, 10.0]])
    _, std_far = gp.predict(X_far, return_std=True)
    assert std_far is not None
    assert std_far[0] > np.mean(std)

    # Cross-validation
    cv = gp.cross_validate(X, y, k_folds=4, seed=42)
    assert "cv_r2" in cv
    assert "cv_rmse" in cv
    assert len(cv["oof_predictions"]) == n_samples
    assert cv["cv_r2"] > 0.50

    # Serialization roundtrip
    bundle = gp.to_bundle(X, y)
    gp_restored = GaussianProcessSurrogate.from_bundle(bundle)
    mu_restored, std_restored = gp_restored.predict(X[:5], return_std=True)
    np.testing.assert_allclose(mu[:5], mu_restored, atol=1e-5)
    np.testing.assert_allclose(std[:5], std_restored, atol=1e-5)


def test_gp_permutation_feature_importance() -> None:
    """Verify permutation feature sensitivity ranking correctly attributes Blue vs Red features."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.normal(0.0, 1.0, size=(n, 4))
    # Feature 0 dominates, feature 1 moderate, feature 2 & 3 noise
    y = 5.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0.0, 0.1, size=n)

    gp = GaussianProcessSurrogate(lengthscale=1.5, signal_std=1.0, noise_std=0.1)
    gp.fit(X, y, optimize_hyperparams=False)

    f_names = ["blue_barrels", "red_swarm_size", "blue_splay", "red_speed"]
    f_sides = ["Blue", "Red", "Blue", "Red"]

    imp = gp.compute_permutation_importance(X, y, feature_names=f_names, feature_sides=f_sides, n_repeats=3, seed=42)
    assert len(imp) == 4
    # Most important should be blue_barrels
    assert imp[0]["feature"] == "blue_barrels"
    assert imp[0]["side"] == "Blue"
    assert imp[0]["importance_pct"] > imp[1]["importance_pct"]


# -------------------------------------------------------------------------
# 3. FEATURE ENCODING & CONFIG NORMALIZATION
# -------------------------------------------------------------------------

def test_config_normalization_and_aliases() -> None:
    """Test normalization handles both canonical keys and ergonomic aliases."""
    blue_raw = {
        "layout": "honeycomb",
        "n_barrels": 8,
        "ammo": "12G_TSS_NO2_C_UAS",
        "choke": "Full",
        "pointing": "divergent_curtain",
        "splay_angle_deg": 3.5,
        "firing_mode": "balanced_pair_ripple",
        "ripple_delay_ms": 10.0,
    }
    b_norm = normalize_blue_config(blue_raw, max_barrels=12)
    assert b_norm["layout_type"] == "honeycomb"
    assert b_norm["layout"] == "honeycomb"
    assert b_norm["ammunition_preset"] == "12G_TSS_NO2_C_UAS"
    assert b_norm["ammo"] == "12G_TSS_NO2_C_UAS"
    assert b_norm["pointing_mode"] == "divergent_curtain"
    assert b_norm["pointing"] == "divergent_curtain"
    assert b_norm["splay_angle_deg"] == pytest.approx(3.5, abs=0.1)
    assert b_norm["splay_angle_mrad"] > 0.0

    red_raw = {
        "drone_model": "fpv_quad_small",
        "swarm_size": 5,
        "tactic": "synchronized_omni_wave",
        "evasion": "weaving_jink",
        "maneuver_g": 3.2,
        "speed_mps": 30.0,
        "stealth_level": 0.4,
    }
    r_norm = normalize_red_config(red_raw, max_swarm_size=10)
    assert r_norm["attack_tactic"] == "synchronized_omni_wave"
    assert r_norm["tactic"] == "synchronized_omni_wave"
    assert r_norm["maneuver_profile"] == "sinusoidal_weave"
    assert r_norm["evasion"] == "sinusoidal_weave"
    assert r_norm["maneuver_g"] == 3.2
    assert r_norm["drone_speed_mps"] == 30.0
    assert r_norm["fiber_optic_fraction"] == 0.4


def test_encode_matchup_features_dimensions() -> None:
    """Verify feature encoder produces exact expected dimensionality with finite values."""
    b = normalize_blue_config({})
    r = normalize_red_config({})
    feats = encode_matchup_features(b, r)

    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert len(feats) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feats))


# -------------------------------------------------------------------------
# 4. PHYSICS-BASED MATCHUP EVALUATION MONOTONICITY
# -------------------------------------------------------------------------

def test_matchup_evaluator_evasion_degrades_pk() -> None:
    """Verify high-G evasive drone maneuver increases tracking residuals and reduces kill rate."""
    blue_cfg = normalize_blue_config({
        "layout_type": "circular",
        "n_barrels": 6,
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
        "pointing_mode": "convergent",
        "firing_mode": "simultaneous_volley",
    })

    # Benign target: slow, straight line
    red_benign = normalize_red_config({
        "drone_model": "fpv_quad_small",
        "swarm_size": 2,
        "attack_tactic": "synchronized_omni_wave",
        "maneuver_profile": "linear_cruise",
        "maneuver_g": 0.0,
        "drone_speed_mps": 25.0,
        "fiber_optic_fraction": 0.0,
    })

    # Agile target: fast, high-G weaving jink
    red_agile = normalize_red_config({
        "drone_model": "fpv_quad_small",
        "swarm_size": 2,
        "attack_tactic": "synchronized_omni_wave",
        "maneuver_profile": "terminal_popup_jink",
        "maneuver_g": 4.5,
        "drone_speed_mps": 25.0,
        "fiber_optic_fraction": 0.5,
    })

    res_benign = evaluate_red_vs_blue_matchup_mc(blue_cfg, red_benign, n_episodes=30, mount_load_limit_kn=4.0, seed=42)
    res_agile = evaluate_red_vs_blue_matchup_mc(blue_cfg, red_agile, n_episodes=30, mount_load_limit_kn=4.0, seed=42)

    # Benign should yield higher kill rate and lower breach rate
    assert res_benign["kill_rate_pct"] > res_agile["kill_rate_pct"]
    assert res_benign["utility_score"] > res_agile["utility_score"]


def test_matchup_evaluator_recoil_limit_flagging() -> None:
    """Verify structural mount force limit flags feasibility correctly."""
    blue_heavy = normalize_blue_config({
        "layout_type": "circular",
        "n_barrels": 12,
        "barrels_per_burst": 6,
        "ammunition_preset": "12G_TSS_NO2_C_UAS",
        "firing_mode": "simultaneous_volley",
    })
    red_cfg = normalize_red_config({"swarm_size": 4})

    # Strict mount limit (1.0 kN) should flag infeasible
    res_strict = evaluate_red_vs_blue_matchup_mc(blue_heavy, red_cfg, n_episodes=5, mount_load_limit_kn=1.0, seed=42)
    assert not res_strict["mount_feasible"]
    assert res_strict["peak_mount_force_kn"] > 1.0

    # Generous mount limit (15.0 kN) should flag feasible
    res_generous = evaluate_red_vs_blue_matchup_mc(blue_heavy, red_cfg, n_episodes=5, mount_load_limit_kn=15.0, seed=42)
    assert res_generous["mount_feasible"]


# -------------------------------------------------------------------------
# 5. END-TO-END ADVERSARIAL CO-EVOLUTION & RECOMMENDATION
# -------------------------------------------------------------------------

def test_run_adversarial_coevolution_ml_end_to_end() -> None:
    """Verify full Bayesian co-evolution loop returns expected schema, history, and Nash solution."""
    res = run_adversarial_coevolution_ml(
        n_generations=2,
        n_mc_episodes=5,
        mount_load_limit_kn=3.5,
        max_barrels=6,
        max_swarm_size=4,
        seed=42,
    )

    # Top-level keys check
    expected_keys = [
        "blue_pool", "red_pool", "n_blue_pool", "n_red_pool", "matchups_evaluated",
        "payoff_matrix", "nash", "nash_value", "nash_blue_strategy", "nash_red_strategy",
        "best_blue_config", "deadliest_red_threat", "worst_case_defense_gain",
        "gp_surrogate_diagnostics", "generation_history", "cv_metrics", "dataset_records", "summary"
    ]
    for k in expected_keys:
        assert k in res, f"Missing key '{k}' in coevolution result"

    assert res["matchups_evaluated"] >= 25
    assert len(res["generation_history"]) == 3  # Gen 0 + Gen 1 + Gen 2
    assert len(res["dataset_records"]) == res["matchups_evaluated"]
    assert res["nash"]["duality_gap"] < 1e-4

    # Test CSV export
    csv_str = export_adversarial_dataset_csv(res)
    assert len(csv_str) > 100
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == res["matchups_evaluated"]
    assert "utility_score" in rows[0]
    assert "blue_barrels" in rows[0]

    # Test policy recommendation
    target_threat = {
        "drone_model": "fpv_quad_small",
        "swarm_size": 4,
        "tactic": "synchronized_omni_wave",
        "evasion": "weaving_jink",
        "maneuver_g": 3.0,
        "speed_mps": 28.0,
    }
    rec = recommend_ml_defense_policy(
        adv_result=res,
        target_red_threat=target_threat,
        mount_load_limit_kn=3.5,
        top_k=5,
        verify_top_mc=True,
    )

    assert "recommended_config" in rec
    assert "recommended_policies" in rec
    assert len(rec["recommended_policies"]) == 5
    assert rec["verified_mc"] is not None
    assert "kill_rate_pct" in rec["verified_mc"]


# -------------------------------------------------------------------------
# 6. CLI AND FASTAPI ENDPOINT INTEGRATION
# -------------------------------------------------------------------------

def test_cli_cuas_adversarial_command(tmp_path: Any) -> None:
    """Verify CLI `ammo-mc cuas adversarial` runs and exports CSV."""
    runner = CliRunner()
    csv_file = tmp_path / "adv_export.csv"

    result = runner.invoke(
        cli_app,
        [
            "cuas",
            "adversarial",
            "--generations", "1",
            "--episodes", "2",
            "--max-barrels", "6",
            "--max-swarm", "4",
            "--mount-limit-kn", "3.5",
            "--out-csv", str(csv_file),
            "--seed", "42",
        ],
    )

    assert result.exit_code == 0
    assert "Adversarial Co-Evolution Complete" in result.output
    assert "Nash Game Value V*" in result.output
    assert csv_file.exists()
    assert csv_file.stat().st_size > 50


def test_api_cuas_adversarial_endpoint() -> None:
    """Verify FastAPI `POST /cuas/adversarial` returns 200 with complete co-evolution payload."""
    client = TestClient(app)
    payload = {
        "n_generations": 1,
        "n_mc_episodes": 2,
        "mount_load_limit_kn": 3.5,
        "max_barrels": 6,
        "max_swarm_size": 4,
        "seed": 42,
    }
    response = client.post("/cuas/adversarial", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "nash_value" in data
    assert "best_blue_config" in data
    assert "gp_surrogate_diagnostics" in data
    assert data["matchups_evaluated"] > 0
