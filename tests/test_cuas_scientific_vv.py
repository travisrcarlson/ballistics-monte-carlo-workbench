"""Comprehensive test suite for C-UAS Scientific Grounding, ASME V&V 20, and Range DOE."""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from ammo_mc.api import app
from ammo_mc.cuas_calibration_twin import (
    _apply_homography,
    _compute_homography_matrix,
    digitize_pattern_board_image,
    generate_skewed_pattern_board_png_with_corners,
    generate_synthetic_pattern_board_png,
    get_default_nominal_twin,
)
from ammo_mc.cuas_scientific_vv import (
    build_reproducible_validation_bundle,
    calibrate_doppler_radar_drag,
    compute_asme_vv20_metrics,
    compute_clopper_pearson_ci,
    compute_epistemic_aleatory_uq_envelope,
    export_doe_test_card_csv,
    generate_sample_blind_holdout_dataset,
    generate_sample_doppler_radar_csv,
    generate_sample_shotstring_csv,
    generate_scientific_vv_dossier_html,
    generate_scientific_vv_dossier_markdown,
    parse_weather_and_shotstring_csv,
    plan_range_validation_doe,
    run_blind_holdout_validation,
    run_experimental_validation_benchmarks,
)
from ammo_mc.small_arms import (
    compute_component_v50_perforation,
    compute_drone_projected_geometry,
    compute_rotor_blade_strike_probability,
    evaluate_drone_impact,
    get_drone_target_specs,
    sample_shotcloud_dispersion,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.vis_cuas_vv import (
    create_blind_holdout_validation_figure,
    create_doppler_drag_fit_figure,
    create_epistemic_uq_envelope_figure,
    create_experimental_benchmarks_figure,
    create_sprt_stopping_boundaries_figure,
)


# -------------------------------------------------------------------------
# 1. PHYSICS GROUNDING: ROTOR SOLIDITY & V50 RECHT-IPSON & RAY SHIELDING
# -------------------------------------------------------------------------


def test_rotor_blade_strike_probability_kinematics() -> None:
    # Hub hit (small radial fraction <= 0.14) -> 1.0
    p_hub = compute_rotor_blade_strike_probability(
        radial_frac=0.05,
        rotor_radius_m=0.18,
        n_blades=3,
        mean_chord_m=0.024,
        rpm=6500.0,
        pellet_velocity_mps=250.0,
    )
    assert p_hub == 1.0

    # Inside blade span (radial_frac = 0.55), probability bounded in (0.02, 1.0]
    p_mid = compute_rotor_blade_strike_probability(
        radial_frac=0.55,
        rotor_radius_m=0.18,
        n_blades=3,
        mean_chord_m=0.024,
        rpm=6500.0,
        pellet_velocity_mps=250.0,
    )
    assert 0.05 < p_mid <= 1.0

    # Higher RPM increases swept transit probability
    p_fast = compute_rotor_blade_strike_probability(
        radial_frac=0.55,
        rotor_radius_m=0.18,
        n_blades=3,
        mean_chord_m=0.024,
        rpm=14000.0,
        pellet_velocity_mps=250.0,
    )
    assert p_fast > p_mid

    # Slower pellet gives blades more transit window to strike
    p_slow = compute_rotor_blade_strike_probability(
        radial_frac=0.55,
        rotor_radius_m=0.18,
        n_blades=3,
        mean_chord_m=0.024,
        rpm=6500.0,
        pellet_velocity_mps=120.0,
    )
    assert p_slow > p_mid


def test_component_v50_ballistic_limit_and_recht_ipson() -> None:
    # Thin plastic arm vs high velocity buckshot -> perforation guaranteed
    res_arm = compute_component_v50_perforation(
        component="arms",
        pellet_velocity_mps=350.0,
        pellet_mass_kg=0.001123,
        pellet_diameter_m=0.0062,
        obliquity_deg=0.0,
    )
    assert res_arm["p_perforation"] > 0.90
    assert 0.0 < res_arm["residual_velocity_mps"] < 350.0
    assert res_arm["v50_mps"] < 250.0

    # Armored motor vs low velocity small lead pellet -> defeated
    res_mot = compute_component_v50_perforation(
        component="motors",
        pellet_velocity_mps=60.0,
        pellet_mass_kg=0.000327,
        pellet_diameter_m=0.00325,
        obliquity_deg=45.0,
    )
    assert res_mot["residual_velocity_mps"] == 0.0
    assert res_mot["p_perforation"] < 0.20


def test_ray_shielding_front_to_back_in_evaluate_drone_impact() -> None:
    drone_spec = get_drone_target_specs()["fpv_14in_36cm_pg7m"]
    aspect_geo = compute_drone_projected_geometry(drone_spec, aspect_mode="head_on_frontal")

    hit_shielded = evaluate_drone_impact(
        y_rel_m=0.0,
        z_rel_m=0.0,
        drone_spec=drone_spec,
        striking_energy_j=45.0,
        aspect_geometry=aspect_geo,
        enable_ray_shielding=True,
        pellet_velocity_mps=250.0,
        pellet_mass_kg=0.001123,
        pellet_diameter_m=0.0062,
        use_v50_ballistic_limit=True,
    )
    assert hit_shielded["is_hit"] is True
    assert "component" in hit_shielded


def test_shotcloud_dispersion_mixtures() -> None:
    rng = np.random.default_rng(42)

    # Gaussian
    dy_g, dz_g, dm_g = sample_shotcloud_dispersion(
        rng=rng, n_pellets=200, sigma_radial=0.25, pattern_distribution="gaussian"
    )
    assert len(dy_g) == 200
    assert len(dz_g) == 200
    assert np.all(dm_g == 1.0)

    # Core-flyer mixture
    dy_cf, dz_cf, dm_cf = sample_shotcloud_dispersion(
        rng=rng, n_pellets=200, sigma_radial=0.25, pattern_distribution="core_flyer_mixture", flyer_fraction=0.15
    )
    assert len(dy_cf) == 200
    assert np.any(dm_cf > 1.0)

    # Choke-annular mixture
    dy_an, dz_an, dm_an = sample_shotcloud_dispersion(
        rng=rng, n_pellets=200, sigma_radial=0.25, pattern_distribution="choke_annular_mixture", flyer_fraction=0.20
    )
    assert len(dy_an) == 200
    assert np.any(dm_an > 1.0)


# -------------------------------------------------------------------------
# 2. HOMOGRAPHY DLT PERSPECTIVE RECTIFICATION
# -------------------------------------------------------------------------


def test_homography_dlt_mapping() -> None:
    src_corners = [(100.0, 100.0), (700.0, 120.0), (740.0, 720.0), (80.0, 700.0)]
    dst_corners = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]

    H = _compute_homography_matrix(src_corners, dst_corners)
    assert H.shape == (3, 3)

    mapped = _apply_homography(H, src_corners)
    for i in range(4):
        assert abs(mapped[i, 0] - dst_corners[i][0]) < 1e-4
        assert abs(mapped[i, 1] - dst_corners[i][1]) < 1e-4


def test_skewed_pattern_board_digitizer() -> None:
    skew_png, corners, _ = generate_skewed_pattern_board_png_with_corners(
        range_m=36.58, nominal_choke="Modified", seed=42
    )
    assert len(skew_png) > 100
    assert len(corners) == 4

    dig = digitize_pattern_board_image(
        skew_png,
        board_width_m=1.0,
        board_height_m=1.0,
        range_m=36.58,
        nominal_choke="Modified",
        ammunition_preset="ALDA_0_80_HP_N2",
        corner_points_px=corners,
    )
    assert dig["n_detected_holes"] > 10
    assert dig["homography_applied"] is True
    assert "calibrated_spread_factor" in dig["pattern_calibration"]


# -------------------------------------------------------------------------
# 3. ASME V&V 20 METRICS, CLOPPER-PEARSON & BLIND HOLDOUT
# -------------------------------------------------------------------------


def test_clopper_pearson_ci() -> None:
    low, high = compute_clopper_pearson_ci(k_successes=8, n_trials=10, alpha=0.05)
    assert 0.40 < low < 0.85
    assert 0.75 < high <= 1.0

    low0, high0 = compute_clopper_pearson_ci(k_successes=0, n_trials=20)
    assert low0 == 0.0
    assert high0 > 0.0

    low1, high1 = compute_clopper_pearson_ci(k_successes=20, n_trials=20)
    assert low1 < 1.0
    assert high1 == 1.0


def test_asme_vv20_metrics_computation() -> None:
    s_vals = [0.25, 0.45, 0.70, 0.95]
    d_vals = [0.26, 0.43, 0.72, 0.93]
    res = compute_asme_vv20_metrics(s_vals, d_vals, u_num=0.015, u_input=0.02)
    assert res["n_sim"] == 4
    assert res["comparison_error_E"] < 0.03
    assert res["u_val"] > 0.0
    assert res["verdict"] == "VALIDATED"
    assert res["area_validation_metric"] >= 0.0


def test_blind_holdout_cross_validation_workflow() -> None:
    ds = generate_sample_blind_holdout_dataset(nominal_choke="Modified")
    assert len(ds["pattern_rows"]) >= 5
    assert len(ds["livefire_rows"]) >= 5

    res = run_blind_holdout_validation(
        ds["pattern_rows"],
        livefire_rows=ds["livefire_rows"],
        holdout_ranges_m=(65.0, 75.0),
        nominal_choke="Modified",
        ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
    )
    assert res["n_train_ranges"] >= 3
    assert res["n_holdout_ranges"] >= 1
    assert res["holdout_rmse_calibrated_m"] >= 0.0
    assert "asme_vv20_sigma" in res
    assert "lethality_holdout" in res


# -------------------------------------------------------------------------
# 4. EPISTEMIC + ALEATORY UQ ENVELOPE
# -------------------------------------------------------------------------


def test_epistemic_aleatory_uq_envelope() -> None:
    twin = get_default_nominal_twin()
    uq = compute_epistemic_aleatory_uq_envelope(
        twin_profile=twin,
        distance_stations_m=(25.0, 50.0, 75.0),
        n_epistemic_draws=4,
        n_volleys_per_draw=5,
        seed=42,
    )
    assert len(uq["distance_stations_m"]) == 3
    assert len(uq["band_rows"]) == 3
    assert uq["r80_median_m"] > 0.0
    assert uq["r80_p05_m"] <= uq["r80_median_m"] <= uq["r80_p95_m"]


# -------------------------------------------------------------------------
# 5. EXPERIMENTAL VALIDATION BENCHMARKS E-01..E-06
# -------------------------------------------------------------------------


def test_run_experimental_validation_benchmarks_all_pass() -> None:
    res = run_experimental_validation_benchmarks()
    assert res["n_benchmarks"] == 6
    assert res["passed_benchmarks"] == 6
    assert res["all_passed"] is True
    for bm in res["benchmarks"]:
        assert bm["passed"] is True
        assert bm["status"] == "PASS"


# -------------------------------------------------------------------------
# 6. RANGE DOE & WALD SPRT EARLY STOPPING
# -------------------------------------------------------------------------


def test_plan_range_validation_doe_and_csv() -> None:
    doe = plan_range_validation_doe(target_pk=0.80, alpha=0.05, power=0.80)
    assert len(doe["doe_matrix"]) >= 4
    assert len(doe["sprt_bounds"]) > 10

    csv_out = export_doe_test_card_csv(doe)
    assert "Target Range (m)" in csv_out or "DOE-" in csv_out


# -------------------------------------------------------------------------
# 7. DOPPLER RADAR DRAG REGRESSION & SHOT STRING
# -------------------------------------------------------------------------


def test_calibrate_doppler_radar_drag() -> None:
    csv_text = generate_sample_doppler_radar_csv(
        true_v0_mps=412.0, true_cd_scale=1.08, n_points=35
    )
    res = calibrate_doppler_radar_drag(
        csv_text, nominal_v0_mps=415.0, ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G"
    )
    assert res["success"] is True
    assert abs(res["fitted_v0_mps"] - 412.0) < 6.0
    assert abs(res["calibrated_cd_scale"] - 1.08) < 0.15
    assert res["rmse_velocity_mps"] < 4.0
    assert "twin_patch" in res


def test_parse_weather_and_shotstring_csv() -> None:
    csv_text = generate_sample_shotstring_csv(range_m=50.0)
    res = parse_weather_and_shotstring_csv(csv_text, range_m=50.0)
    assert 1.15 < res["air_density_kg_m3"] < 1.30
    assert res["speed_of_sound_mps"] > 330.0
    assert res["string_length_m"] > 0.5
    assert res["string_duration_ms"] > 1.0
    assert res["n_pellets_recorded"] >= 10


# -------------------------------------------------------------------------
# 8. FAIR REPRODUCIBLE VALIDATION BUNDLE & DOSSIERS
# -------------------------------------------------------------------------


def test_build_reproducible_validation_bundle_and_dossiers() -> None:
    twin = get_default_nominal_twin()
    bundle = build_reproducible_validation_bundle(
        twin=twin,
        seed=42,
    )
    assert len(bundle["bundle_sha256"]) == 64
    assert "experimental_benchmarks" in bundle

    html_doc = generate_scientific_vv_dossier_html(bundle)
    assert "<html" in html_doc
    assert bundle["bundle_sha256"] in html_doc

    md_doc = generate_scientific_vv_dossier_markdown(bundle)
    assert "# Counter-UAS" in md_doc
    assert bundle["bundle_sha256"] in md_doc


# -------------------------------------------------------------------------
# 9. PLOTLY VISUALIZATION RENDERING
# -------------------------------------------------------------------------


def test_vis_cuas_vv_figures() -> None:
    # Doppler figure
    dop_csv = generate_sample_doppler_radar_csv()
    dop_res = calibrate_doppler_radar_drag(dop_csv)
    fig_dop = create_doppler_drag_fit_figure(dop_res)
    assert fig_dop is not None

    # Holdout figure
    ds = generate_sample_blind_holdout_dataset()
    hold_res = run_blind_holdout_validation(ds["pattern_rows"], ds["livefire_rows"])
    fig_hold = create_blind_holdout_validation_figure(hold_res)
    assert fig_hold is not None

    # UQ figure
    uq_res = compute_epistemic_aleatory_uq_envelope(n_epistemic_draws=4, n_volleys_per_draw=4)
    fig_uq = create_epistemic_uq_envelope_figure(uq_res)
    assert fig_uq is not None

    # Benchmarks figure
    exp_res = run_experimental_validation_benchmarks()
    fig_exp = create_experimental_benchmarks_figure(exp_res)
    assert fig_exp is not None

    # SPRT figure
    doe = plan_range_validation_doe()
    fig_sprt = create_sprt_stopping_boundaries_figure(doe)
    assert fig_sprt is not None


# -------------------------------------------------------------------------
# 10. FASTAPI REST API ENDPOINTS
# -------------------------------------------------------------------------


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_api_cuas_array_with_high_fidelity_flags(api_client: TestClient) -> None:
    resp = api_client.post(
        "/cuas/array",
        headers={"X-API-Key": "Freya2026"},
        json={
            "layout_type": "quincunx",
            "n_barrels": 5,
            "target_distance_m": 35.0,
            "drone_model": "fpv_14in_36cm_pg7m",
            "use_rotor_solidity": True,
            "use_v50_ballistic_limit": True,
            "enable_ray_shielding": True,
            "pattern_distribution": "core_flyer_mixture",
            "flyer_fraction": 0.15,
            "n_volleys": 5,
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["kill_probability_pct"] >= 0.0


def test_api_scientific_vv_endpoints(api_client: TestClient) -> None:
    # 1. Experimental benchmarks
    r_bm = api_client.get("/cuas/vv/experimental-benchmarks", headers={"X-API-Key": "Freya2026"})
    assert r_bm.status_code == 200
    d_bm = r_bm.json()
    assert d_bm["all_passed"] is True
    assert d_bm["n_benchmarks"] == 6

    # 2. Blind holdout
    r_hold = api_client.post("/cuas/vv/blind-holdout", headers={"X-API-Key": "Freya2026"}, json={})
    assert r_hold.status_code == 200
    d_hold = r_hold.json()
    assert "holdout_rmse_calibrated_m" in d_hold

    # 3. UQ envelope
    r_uq = api_client.post(
        "/cuas/vv/uq-envelope",
        headers={"X-API-Key": "Freya2026"},
        json={"distance_stations_m": [25.0, 50.0], "n_epistemic_draws": 4, "n_volleys_per_draw": 4},
    )
    assert r_uq.status_code == 200
    d_uq = r_uq.json()
    assert "r80_median_m" in d_uq

    # 4. DOE plan
    r_doe = api_client.post(
        "/cuas/vv/doe-plan", headers={"X-API-Key": "Freya2026"}, json={"target_range_m": 45.0}
    )
    assert r_doe.status_code == 200
    d_doe = r_doe.json()
    assert "doe_matrix" in d_doe

    # 5. Doppler calibration
    r_dop = api_client.post(
        "/cuas/calibrate/doppler-drag", headers={"X-API-Key": "Freya2026"}, json={}
    )
    assert r_dop.status_code == 200
    d_dop = r_dop.json()
    assert d_dop["success"] is True

    # 6. Shot-string parsing
    r_ss = api_client.post(
        "/cuas/calibrate/shotstring", headers={"X-API-Key": "Freya2026"}, json={"range_m": 50.0}
    )
    assert r_ss.status_code == 200
    d_ss = r_ss.json()
    assert d_ss["air_density_kg_m3"] > 1.0
