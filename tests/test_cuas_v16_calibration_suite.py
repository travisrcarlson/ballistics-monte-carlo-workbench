"""Verification tests for C-UAS v1.6 Closed-Loop Calibration, Digital Twin, FEA/STL CAD, Safety Envelope & HIL Trace."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient
import plotly.graph_objects as go
import pytest
from typer.testing import CliRunner

from ammo_mc.api import app
from ammo_mc.briefing_export import generate_calibration_certificate_pdf, generate_executive_briefing_pdf
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_calibration_twin import (
    CalibrationProfileStore,
    analyze_calibration_design_delta,
    calibrate_hardware_bench_tests,
    calibrate_livefire_drone_lethality,
    calibrate_multirange_pattern_and_drag,
    calibrate_tracker_telemetry,
    digitize_pattern_board_image,
    export_range_test_card_csv,
    generate_sample_hardware_bench_tests,
    generate_sample_livefire_drone_trials,
    generate_sample_multirange_pattern_rows,
    generate_sample_tracker_test_rows,
    generate_synthetic_pattern_board_png,
    merge_twin_updates,
    recommend_next_range_tests,
)
from ammo_mc.cuas_defense import _estimate_burst_pk_analytics
from ammo_mc.cuas_hardware import evaluate_muzzle_plate_fea, export_muzzle_plate_stl
from ammo_mc.cuas_rl_policy import run_hil_fire_control_trace
from ammo_mc.cuas_safety_envelope import evaluate_cuas_collateral_safety_envelope
from ammo_mc.cuas_wizard import run_turret_sizing_wizard
from ammo_mc.small_arms import (
    generate_multishotgun_array_layout,
    get_drone_target_specs,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.vis_cuas_advanced import (
    create_active_test_planner_figure,
    create_calibration_reliability_and_drift_figure,
    create_collateral_safety_envelope_figure,
    create_hil_policy_timeline_figure,
    create_swap_c_and_otm_figure,
)


def test_multirange_pattern_and_photo_digitizer() -> None:
    rows = generate_sample_multirange_pattern_rows(
        nominal_choke="Modified",
        true_spread_factor=1.24,
        true_exponent=1.28,
        true_cd_scale=1.12,
        seed=17,
    )
    mr = calibrate_multirange_pattern_and_drag(rows, nominal_choke="Modified")
    assert mr["n_stations"] == 5
    assert 1.05 <= mr["spread_factor"] <= 1.45
    assert 1.15 <= mr["spread_exponent"] <= 1.45
    assert 0.95 <= mr["pellet_cd_scale"] <= 1.30
    assert mr["r2_spread"] > 0.90

    png_bytes = generate_synthetic_pattern_board_png(
        range_m=36.58,
        nominal_choke="Modified",
        true_spread_factor=1.15,
        n_pellets=42,
        seed=42,
    )
    assert png_bytes[:4] == b"\x89PNG"
    dig = digitize_pattern_board_image(png_bytes, board_width_m=1.0, board_height_m=1.0, range_m=36.58)
    assert dig["n_detected_holes"] >= 25
    assert 0.55 <= dig["pattern_calibration"]["calibrated_spread_factor"] <= 1.85


def test_livefire_drone_bayesian_lethality_and_twin_propagation() -> None:
    trials = generate_sample_livefire_drone_trials(
        drone_model="fpv_quad_small",
        true_avuln_scale=0.80,
        true_ecrit_scale=1.35,
        seed=7,
    )
    leth = calibrate_livefire_drone_lethality(trials, drone_model="fpv_quad_small")
    assert leth["n_trials"] == len(trials)
    assert leth["brier_after"] <= leth["brier_before"] + 0.01
    assert len(leth["reliability_bins"]) >= 2

    twin = merge_twin_updates(None, {**leth["twin_patch"], "spread_factor": 1.35, "drone_ecrit_scale": 1.50}, domain="test")
    d_spec = get_drone_target_specs()["fpv_quad_small"]
    pk_nom = _estimate_burst_pk_analytics(45.0, 2, 27, 6.1, 1.34, 415.0, "Modified", d_spec)
    pk_cal = _estimate_burst_pk_analytics(45.0, 2, 27, 6.1, 1.34, 415.0, "Modified", d_spec, calibration_profile=twin)
    assert pk_cal < pk_nom

    sim_nom = simulate_anti_drone_multishotgun_mc(n_barrels=6, target_distance_m=45.0, n_volleys=15, seed=11)
    sim_cal = simulate_anti_drone_multishotgun_mc(n_barrels=6, target_distance_m=45.0, n_volleys=15, seed=11, calibration_profile=twin)
    assert sim_cal["summary"]["kill_probability_pct"] <= sim_nom["summary"]["kill_probability_pct"]

    wiz_cal = run_turret_sizing_wizard(drone_model="fpv_quad_small", required_range_m=45.0, calibration_profile=twin)
    assert len(wiz_cal["top_3"]) == 3


def test_hardware_bench_tracker_sqlite_store_drift_and_planner(tmp_path: Path) -> None:
    hb = calibrate_hardware_bench_tests({"n_barrels": 6}, generate_sample_hardware_bench_tests(6, seed=42))
    assert hb["twin_patch"]["recoil_force_scale"] > 0.5
    assert len(hb["comparisons"]) >= 5

    trk = calibrate_tracker_telemetry(generate_sample_tracker_test_rows(true_rms_mrad=1.4, true_latency_ms=48.0, seed=42))
    assert trk["measured_rms_mrad"] > 0.7
    assert trk["twin_patch"]["tracker_latency_ms"] > 30.0

    twin = merge_twin_updates(
        None,
        {
            **hb["twin_patch"],
            **trk["twin_patch"],
            "spread_factor": 1.25,
            "drone_ecrit_scale": 1.30,
            "lot_number": "LOT-V16",
            "barrel_serial": "SN-V16",
        },
        domain="suite",
    )

    store = CalibrationProfileStore(db_path=tmp_path / "cal_profiles.db")
    saved = store.save_profile("Campaign-V16", twin, notes="Range baseline", tags=["v16"])
    listed = store.list_profiles(search="Campaign")
    assert len(listed) == 1 and listed[0]["id"] == saved["id"]
    raw_export = store.export_profiles_json()
    store2 = CalibrationProfileStore(db_path=tmp_path / "cal_profiles_copy.db")
    assert store2.import_library_json(raw_export) if hasattr(store2, "import_library_json") else store2.import_profiles_json(raw_export) == 1

    drift = analyze_calibration_design_delta({"n_barrels": 6, "barrels_per_burst": 2}, twin, required_range_m=45.0)
    assert drift["r80_drift_m"] <= 0.0
    assert len(drift["recommended_fixes"]) == 3

    plan = recommend_next_range_tests({"n_barrels": 6, "barrels_per_burst": 2}, twin, target_range_m=45.0)
    assert len(plan["top_recommendations"]) >= 4
    card_csv = export_range_test_card_csv(plan)
    assert "recommended_test" in card_csv

    fig_rel = create_calibration_reliability_and_drift_figure(None, drift)
    fig_pl = create_active_test_planner_figure(plan)
    assert isinstance(fig_rel, go.Figure) and isinstance(fig_pl, go.Figure)

    cert_pdf = generate_calibration_certificate_pdf(twin, drift, plan)
    brief_pdf = generate_executive_briefing_pdf(calibration_result={"twin": twin, "drift": drift})
    assert cert_pdf.startswith(b"%PDF") and brief_pdf.startswith(b"%PDF")


def test_muzzle_plate_fea_and_stl_extrusion() -> None:
    bbls_c = generate_multishotgun_array_layout("circular", n_barrels=6, mount_radius_m=0.20)
    bbls_g = generate_multishotgun_array_layout("grid", n_barrels=6, n_cols=3, n_rows=2, spacing_y_m=0.12, spacing_z_m=0.12)

    fea_al = evaluate_muzzle_plate_fea(bbls_c, layout_type="circular", peak_recoil_force_kn=4.5, plate_thickness_mm=24.0, material="7075-T6", bolt_grade="10.9")
    fea_weak = evaluate_muzzle_plate_fea(bbls_c, layout_type="circular", peak_recoil_force_kn=22.0, plate_thickness_mm=8.0, material="6061-T6", bolt_grade="8.8")
    assert fea_al["overall_safety_factor"] > fea_weak["overall_safety_factor"]
    assert fea_al["verdict"] in ("PASS", "MARGINAL")
    assert fea_weak["verdict"] == "FAIL"

    stl_c = export_muzzle_plate_stl(bbls_c, layout_type="circular", plate_thickness_mm=22.0)
    stl_g = export_muzzle_plate_stl(bbls_g, layout_type="grid", plate_thickness_mm=22.0)
    assert stl_c.startswith("solid ") and "endsolid" in stl_c
    assert stl_c.count("facet normal") >= 500
    assert stl_g.count("facet normal") >= 500
    assert "nan" not in stl_c.lower() and "nan" not in stl_g.lower()


def test_collateral_safety_envelope_and_hil_trace() -> None:
    saf = evaluate_cuas_collateral_safety_envelope(
        n_barrels=6,
        barrels_per_burst=2,
        ammunition_preset="12G_NO4_BUCK_C_UAS",
        firing_elevation_deg=35.0,
        environment_type="urban_permissive",
    )
    sm = saf["summary"]
    assert sm["acoustic_140db_downrange_m"] > sm["acoustic_165db_downrange_m"] > sm["acoustic_175db_downrange_m"]
    assert sm["verdict"] in ("CLEARED_WITH_STANDOFF", "CLEARED_PERIMETER_CONTROLLED", "RESTRICTED_ELEVATION_OR_AMMO")
    assert isinstance(create_collateral_safety_envelope_figure(saf), go.Figure)

    hil = run_hil_fire_control_trace(None, track_csv_or_scenario="evasive_fpv_dive", dt_ms=100.0, breach_radius_m=12.0)
    assert hil["summary"]["n_frames"] >= 10
    assert hil["summary"]["fire_commands"] >= 1
    assert "t_ms" in hil["csv_stream"]
    assert isinstance(create_hil_policy_timeline_figure(hil), go.Figure)

    swap_fig = create_swap_c_and_otm_figure(
        {"summary": {"total_mass_kg": 61.0}, "mass_breakdown": {"Barrels": 10.0, "Gimbal": 20.0}},
        {"carrier_profile": "4x4", "curves": {"static_ideal": [{"range_m": 20.0, "pk_pct": 95.0}, {"range_m": 60.0, "pk_pct": 65.0}]}},
    )
    assert isinstance(swap_fig, go.Figure)


def test_cli_and_api_cuas_v16_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "v16_api.db"
    monkeypatch.setenv("AMMO_MC_SCENARIO_DB", str(db_file))
    runner = CliRunner()

    stl_out = tmp_path / "clamp.stl"
    res_sc = runner.invoke(cli_app, ["cuas", "swap-c", "--barrels", "6", "--out-stl", str(stl_out)])
    assert res_sc.exit_code == 0 and stl_out.exists() and "FEA=" in res_sc.output

    res_cal = runner.invoke(cli_app, ["cuas", "calibrate-twin", "--drone", "fpv_quad_small", "--save-name", "CLI-V16"])
    assert res_cal.exit_code == 0 and "Calibrated Digital Twin" in res_cal.output

    res_saf = runner.invoke(cli_app, ["cuas", "safety-envelope", "--barrels", "6", "--elevation", "35"])
    assert res_saf.exit_code == 0 and "Collateral Safety Envelope" in res_saf.output

    res_hil = runner.invoke(cli_app, ["cuas", "hil-trace", "--scenario", "evasive_fpv_dive"])
    assert res_hil.exit_code == 0 and "HIL Policy Trace" in res_hil.output

    client = TestClient(app)
    assert client.post("/cuas/cad/stl", json={"n_barrels": 6, "plate_thickness_mm": 22.0}).status_code == 200
    assert client.post("/cuas/hardware/fea", json={"n_barrels": 6, "material": "7075-T6"}).json()["overall_safety_factor"] > 0
    assert client.post("/cuas/safety-envelope", json={"n_barrels": 6}).status_code == 200
    assert client.post("/cuas/hil-trace", json={"dt_ms": 100.0}).status_code == 200
    assert client.post("/cuas/calibrate/multirange", json={}).status_code == 200

    png_b64 = base64.b64encode(generate_synthetic_pattern_board_png(seed=42)).decode("ascii")
    ph_resp = client.post("/cuas/calibrate/pattern-photo", json={"image_base64": png_b64})
    assert ph_resp.status_code == 200 and ph_resp.json()["n_detected_holes"] >= 20

    assert client.post("/cuas/calibrate/lethality", json={}).status_code == 200
    assert client.post("/cuas/calibrate/hardware", json={}).status_code == 200
    assert client.post("/cuas/calibrate/tracker", json={}).status_code == 200
    assert client.post("/cuas/calibrate/design-delta", json={}).status_code == 200
    assert client.post("/cuas/calibration-profiles", json={"name": "API-V16", "twin": {"spread_factor": 1.15}}).status_code == 200
    assert len(client.get("/cuas/calibration-profiles").json()["profiles"]) >= 1
    cert_resp = client.post("/export/calibration-certificate", json={})
    assert cert_resp.status_code == 200 and cert_resp.content.startswith(b"%PDF")
