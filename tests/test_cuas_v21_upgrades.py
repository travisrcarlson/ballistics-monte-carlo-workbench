"""Comprehensive test suite for C-UAS v2.1.0 scientific upgrades across Tracks 1-6."""

from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import ammo_mc
from ammo_mc.api import app as fastapi_app
from ammo_mc.briefing_export import generate_executive_briefing_pdf, generate_executive_briefing_pptx
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_analytics import build_cuas_site_geojson
from ammo_mc.cuas_ballistics_v2 import (
    integrate_pellet_trajectory_3dof_rk4,
    solve_iterative_3d_lead_intercept,
    sphere_drag_loth_2008,
    surface_layer_wind_log_law,
    sutherland_viscosity_pa_s,
    tropospheric_pressure_pa,
)
from ammo_mc.cuas_defense import (
    simulate_closed_loop_fcs_engagement_mc,
    simulate_cuas_pellet_fallout_hazard,
)
from ammo_mc.cuas_mission import simulate_multiturret_site_defense_mc
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.cuas_rigor import (
    get_empirical_calibration_anchors,
    gittins_sls_index,
    run_workbook_validation_cases,
)
from ammo_mc.cuas_safety_envelope import evaluate_cuas_collateral_safety_envelope
from ammo_mc.cuas_v21_science import (
    build_cuas_scenario_manifest,
    compare_cuas_engagement_doctrines,
    load_cuas_scenario_manifest,
    run_cuas_l0_epistemic_envelope,
    run_cuas_morris_screening,
    run_cuas_sobol_sensitivity,
    run_cuas_surrogate_pareto_study,
    serialize_manifest_json,
    serialize_manifest_yaml,
)
from ammo_mc.small_arms_pkg.arrays import evaluate_antidrone_effective_range_envelope
from ammo_mc.vis_cuas import create_antidrone_effective_range_figure
from ammo_mc.vis_cuas_analytics import create_real_map_site_defense_figure
from ammo_mc.vis_cuas_vv import (
    create_doctrine_comparator_figure,
    create_epistemic_window_and_pk_figure,
    create_morris_sobol_sensitivity_figure,
    create_surrogate_pareto_figure,
)


def test_track2_110_workbook_oracles_and_12_empirical_anchors() -> None:
    """Verify all 110 canonical VC-001..110 workbook checks + 6 extended checks + 12 empirical anchors."""
    assert ammo_mc.__version__ == "2.1.0"
    res = run_workbook_validation_cases(tier="fast")
    sm = res["summary"]
    assert sm["canonical_vc_total"] == 110
    assert sm["canonical_vc_passed"] == 110
    assert sm["total"] == 116
    assert sm["passed"] == 116
    assert sm["failed"] == 0
    assert sm["pass_rate_pct"] == 100.0

    anchors = get_empirical_calibration_anchors()
    assert len(anchors) == 12
    ids = {a["case_id"] for a in anchors}
    assert "VC-111" in ids and "VC-122" in ids
    assert all(a["evidence_grade"] in {"B", "C", "D"} for a in anchors)


def test_track1_epistemic_l0_morris_and_sobol_sensitivity() -> None:
    """Verify nested L0 LHS epistemic bounds, Morris elementary effects, and Saltelli/Jansen Sobol."""
    ep = run_cuas_l0_epistemic_envelope(
        ammo_preset="12G_TSS_NO2_C_UAS",
        drone_class="fpv_cfrp",
        n_barrels=6,
        drone_speed_mps=15.0,
        n_outer=16,
        seed=2100,
    )
    sm = ep["summary"]
    assert 0.0 <= sm["usable_window_p5_m"] <= sm["usable_window_p50_m"] <= sm["usable_window_p95_m"]
    assert sm["usable_window_p95_m"] > 0.0
    assert len(ep["curves"]) >= 5 and "range_m" in ep["curves"][0]
    fig_ep = create_epistemic_window_and_pk_figure(ep)
    assert len(fig_ep.data) >= 2

    env = evaluate_antidrone_effective_range_envelope(
        distance_stations_m=[20.0, 35.0, 50.0, 65.0],
        ammunition_preset="12G_TSS_NO2_C_UAS",
        drone_model="fpv_quad_small",
        n_barrels=6,
        n_volleys=40,
        seed=2100,
    )
    fig_s1 = create_antidrone_effective_range_figure(
        env,
        epistemic_env=ep,
        empirical_anchors=get_empirical_calibration_anchors(),
    )
    assert len(fig_s1.data) >= 4

    mor = run_cuas_morris_screening(
        ammo_preset="12G_TSS_NO2_C_UAS",
        n_barrels=6,
        drone_speed_mps=15.0,
        n_trajectories=8,
        seed=2100,
    )
    sob = run_cuas_sobol_sensitivity(
        ammo_preset="12G_TSS_NO2_C_UAS",
        n_barrels=6,
        drone_speed_mps=15.0,
        n_base=32,
        seed=2100,
    )
    assert len(mor["factors"]) >= 4
    assert all(f["mu_star_pk_pct"] >= 0.0 for f in mor["factors"])
    assert len(sob["indices"]) >= 4
    assert any(idx["ST"] > 0.05 for idx in sob["indices"])
    fig_ms = create_morris_sobol_sensitivity_figure(mor, sob)
    assert len(fig_ms.data) >= 2


def test_track3_doctrine_lab_and_gittins_wta_in_raid() -> None:
    """Verify exact VC-089 Gittins index (6.25), 4-doctrine comparison, and raid Gittins WTA."""
    # Glazebrook & Washburn (2004) Eq. 14 exact check (VC-089)
    idx_val = gittins_sls_index(delta=0.9, q=0.5, v=4.0, p=0.2, c=0.0)
    assert math.isclose(idx_val, 6.25, rel_tol=1e-9, abs_tol=1e-9)

    doc = compare_cuas_engagement_doctrines(
        ammo_preset="12G_TSS_NO2_C_UAS",
        n_barrels=4,
        magazine_shells=12,
        r_start_m=55.0,
        drone_speed_mps=20.0,
        n_trials=200,
        seed=2101,
    )
    rows = doc["doctrines"]
    assert len(rows) == 4
    ids = {r["doctrine_id"] for r in rows}
    assert {"simultaneous_salvo", "ripple_salvo", "shoot_look_shoot_wilkening", "shoot_adjust_shoot_whistler"}.issubset(ids)
    assert all(0.0 <= r["cum_pk_pct"] <= 100.0 for r in rows)
    fig_doc = create_doctrine_comparator_figure(doc)
    assert len(fig_doc.data) >= 2

    raid_gittins = simulate_unified_tactical_raid_mc(
        n_drones=6,
        n_mc_trials=6,
        seed=2101,
        wta_policy="gittins_sls_index",
    )
    assert raid_gittins["summary"]["wta_policy"] == "gittins_sls_index"
    assert 0.0 <= raid_gittins["summary"]["mean_kill_rate_pct"] <= 100.0


def test_track4_loth_altitude_loglaw_3dof_lead_and_shotfall_uxo_gis() -> None:
    """Verify Loth (2008) drag, AT-F01 pressure, AT-F07 log-law wind, 3-DoF RK4, iterative 3D lead, & GIS."""
    cd_loth = sphere_drag_loth_2008(mach=0.9, re=1e4)
    assert 0.65 < cd_loth < 0.90
    assert math.isclose(tropospheric_pressure_pa(0.0), 101325.0, abs_tol=1e-6)
    p_2k = tropospheric_pressure_pa(2000.0)
    assert 78000.0 < p_2k < 81000.0
    assert sutherland_viscosity_pa_s(15.0) > 1.7e-5

    u_low = surface_layer_wind_log_law(u_ref_mps=6.0, z_m=2.0, z_ref_m=10.0, z0_m=0.03)
    u_high = surface_layer_wind_log_law(u_ref_mps=6.0, z_m=20.0, z_ref_m=10.0, z0_m=0.03)
    assert 0.0 < u_low < 6.0 < u_high

    traj = integrate_pellet_trajectory_3dof_rk4(
        v0_mps=405.0,
        diam_mm=3.3,
        density_kg_m3=11200.0,
        aim_elevation_deg=30.0,
        drag_law="loth_2008",
        site_altitude_m=1500.0,
        wind_xyz_10m_mps=(4.0, 2.0, 0.0),
        wind_profile_mode="log_law",
    )
    assert traj["ground_range_m"] > 100.0 and traj["tof_s"] > 1.0

    lead = solve_iterative_3d_lead_intercept(
        r0_m=(60.0, 0.0, 15.0),
        vt_mps=(-22.0, 4.0, -1.5),
        t_dead_s=0.141,
        v0_mps=405.0,
        diam_mm=3.3,
        density_kg_m3=11200.0,
        site_altitude_m=500.0,
    )
    assert lead["converged"] is True
    assert lead["tof_s"] > 0.1 and lead["residual_s"] < 1e-3

    fcs = simulate_closed_loop_fcs_engagement_mc(
        n_volleys=10,
        seed=2102,
        use_iterative_3d_lead_solver=True,
        site_altitude_m=1200.0,
        target_elevation_deg=20.0,
    )
    assert fcs["summary"]["iterative_lead_solution"]["converged"] is True

    fallout = simulate_cuas_pellet_fallout_hazard(
        n_sample_pellets=140,
        seed=2102,
        site_altitude_m=1500.0,
        wind_profile_mode="log_law",
        surface_roughness_z0_m=0.03,
    )
    assert len(fallout["shotfall_contour_95_xy_m"]) >= 12
    assert fallout["summary"]["site_altitude_m"] == 1500.0

    env_saf = evaluate_cuas_collateral_safety_envelope(
        n_sample_pellets=120,
        seed=2102,
        site_altitude_m=1000.0,
        wind_profile_mode="log_law",
        intercept_range_m=45.0,
        drone_speed_mps=22.0,
        warhead_blast_radius_m=8.0,
    )
    assert "shotfall_contour_95_xy_m" in env_saf
    assert "uxo_carryon_ellipse" in env_saf

    uxo = env_saf["uxo_carryon_ellipse"]
    site_res = simulate_multiturret_site_defense_mc(site_preset="fob_compound_4_turret", n_raid_trials=6, seed=2102)
    geojson = build_cuas_site_geojson(
        site_res=site_res,
        shotfall_contour_xy_m=env_saf["shotfall_contour_95_xy_m"],
        uxo_ellipse_xy_m=uxo.get("polygon_xy_m"),
    )
    assert len(geojson["features"]) >= 4
    fig_map = create_real_map_site_defense_figure(
        site_res=site_res,
        shotfall_contour_xy_m=env_saf["shotfall_contour_95_xy_m"],
        uxo_ellipse_xy_m=uxo.get("polygon_xy_m"),
    )
    assert len(fig_map.data) >= 3


def test_track5_scenario_manifest_sha256_yaml_json_and_briefing_stamp() -> None:
    """Verify JSON/YAML Scenario Manifest SHA-256 verification & PDF/PPTX cover stamps."""
    man = build_cuas_scenario_manifest(
        scenario_name="V2.1 Verification Scenario",
        seed=2105,
        site_altitude_m=800.0,
        array_config={"ammo_preset": "12G_TSS_NO2_C_UAS", "n_barrels": 6},
    )
    assert len(man["manifest_sha256"]) == 64

    yaml_text = serialize_manifest_yaml(man)
    json_text = serialize_manifest_json(man)
    ok_y, parsed_y, msg_y = load_cuas_scenario_manifest(yaml_text)
    ok_j, parsed_j, _ = load_cuas_scenario_manifest(json_text)
    assert ok_y and parsed_y is not None and parsed_y["manifest_sha256"] == man["manifest_sha256"], msg_y
    assert ok_j and parsed_j is not None and parsed_j["manifest_sha256"] == man["manifest_sha256"]

    # Tamper detection on SHA-256
    tampered = json_text.replace("V2.1 Verification Scenario", "Tampered Scenario")
    ok_bad, _, _ = load_cuas_scenario_manifest(tampered)
    assert ok_bad is False

    pdf_bytes = generate_executive_briefing_pdf(
        title="v2.1 Manifest Brief",
        scenario_manifest=man,
    )
    pptx_bytes = generate_executive_briefing_pptx(
        title="v2.1 Manifest Brief",
        scenario_manifest=man,
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pptx_bytes) > 10_000


def test_track6_matern52_gp_surrogate_pareto_cli_and_api(tmp_path: Path) -> None:
    """Verify Matern-5/2 GP surrogate LOO R2, Pareto front, v2.1 CLI commands & FastAPI endpoints."""
    par = run_cuas_surrogate_pareto_study(
        drone_class="fpv_cfrp",
        drone_speed_mps=20.0,
        n_doe=28,
        n_candidates=160,
        recoil_limit_kn=22.0,
        seed=2103,
    )
    sm = par["summary"]
    assert sm["loo_r2"]["pk_shared_40m_pct"] > 0.80
    assert sm["n_pareto_optimal"] >= 2
    assert len(par["verified_top_designs"]) >= 1
    assert len(create_surrogate_pareto_figure(par).data) >= 2

    # CLI checks
    runner = CliRunner()
    for args in (
        ["cuas", "workbook-vv", "--tier", "fast", "--json"],
        ["cuas", "epistemic", "--outer", "6", "--no-sensitivity", "--json"],
        ["cuas", "doctrine", "--trials", "40", "--json"],
        ["cuas", "pareto-surrogate", "--doe", "24", "--candidates", "60", "--json"],
        ["cuas", "manifest", "--format", "yaml"],
    ):
        res = runner.invoke(cli_app, args)
        assert res.exit_code == 0, res.output

    # FastAPI checks
    client = TestClient(fastapi_app)
    r_wb = client.get("/cuas/vv/workbook-cases?tier=fast")
    assert r_wb.status_code == 200 and r_wb.json()["summary"]["canonical_vc_passed"] == 110

    r_anc = client.get("/cuas/empirical-anchors")
    assert r_anc.status_code == 200 and r_anc.json()["total"] == 12

    r_ep = client.post("/cuas/epistemic-sensitivity", json={"n_outer": 6, "include_sensitivity": False})
    assert r_ep.status_code == 200 and "usable_window_p50_m" in r_ep.json()["epistemic"]["summary"]

    r_doc = client.post("/cuas/doctrine-compare", json={"n_trials": 40})
    assert r_doc.status_code == 200 and len(r_doc.json()["doctrines"]) == 4

    r_par = client.post("/cuas/surrogate-pareto", json={"n_doe": 24, "n_candidates": 60})
    assert r_par.status_code == 200 and r_par.json()["summary"]["n_pareto_optimal"] >= 1

    r_man = client.post("/cuas/manifest/build", json={"scenario_name": "API Test", "fmt": "yaml"})
    assert r_man.status_code == 200
    yaml_payload = r_man.json()["yaml"]
    r_val = client.post("/cuas/manifest/validate", json={"payload": yaml_payload})
    assert r_val.status_code == 200 and r_val.json()["valid"] is True
