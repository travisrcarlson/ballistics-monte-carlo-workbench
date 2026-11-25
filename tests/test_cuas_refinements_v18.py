"""Comprehensive unit & integration tests for v1.8 C-UAS refinements."""

from __future__ import annotations

import base64
import io
import json
import struct
import warnings

from fastapi.testclient import TestClient
import numpy as np
import openpyxl
from typer.testing import CliRunner

import ammo_mc
from ammo_mc.api import app as fastapi_app
from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_adversarial import run_adversarial_coevolution_ml
from ammo_mc.cuas_calibration_twin import (
    detect_fiducial_board_corners,
    digitize_pattern_board_image,
    generate_printable_fiducial_board_pdf,
    generate_skewed_fiducial_board_png,
    get_default_nominal_twin,
)
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.cuas_rl_policy import (
    run_hil_fire_control_trace,
    train_fire_control_policy_es,
)
from ammo_mc.cuas_scientific_vv import (
    build_reproducible_validation_bundle,
    calibrate_doppler_radar_drag,
    compute_asme_vv20_metrics,
    detect_and_parse_instrument_file,
    evaluate_sprt_live_sequence,
    generate_sample_doppler_radar_csv,
    generate_sample_garmin_xero_csv,
    generate_sample_kestrel_link_csv,
    generate_sample_labradar_native_csv,
    generate_sample_shotstring_csv,
    parse_garmin_xero_c1_csv_or_xlsx,
    parse_kestrel_link_csv,
    parse_labradar_native_csv,
    parse_weather_and_shotstring_csv,
    run_experimental_validation_benchmarks,
)
from ammo_mc.cuas_track_replay import (
    generate_sample_drone_track_csv,
    generate_sample_px4_flight_csv,
    parse_ardupilot_px4_flight_log,
    simulate_recorded_track_engagement_mc,
)
from ammo_mc.cuas_wizard import (
    run_red_vs_blue_duel_diagnostic,
    run_turret_sizing_wizard,
)
from ammo_mc.literature import build_literature_audit_summary
from ammo_mc.small_arms import (
    compute_component_v50_perforation,
    compute_drone_projected_geometry,
    evaluate_drone_impact,
    evaluate_drone_impact_batch,
    evaluate_stl_pellet_impacts_batch,
    generate_sample_drone_stl_bytes,
    get_default_physics_fidelity_profile,
    get_drone_target_specs,
    parse_stl_mesh,
    project_stl_mesh_vulnerability,
    simulate_anti_drone_multishotgun_mc,
)


def test_vectorized_drone_impact_equivalence_and_oblique_ricochet() -> None:
    specs = get_drone_target_specs()
    drone = specs["fpv_quad_small"]
    geom = compute_drone_projected_geometry(drone, aspect_mode="head_on_frontal")

    rng_pts = np.random.default_rng(123)
    y_arr = rng_pts.uniform(-0.30, 0.30, size=64)
    z_arr = rng_pts.uniform(-0.25, 0.25, size=64)
    v_arr = rng_pts.uniform(180.0, 360.0, size=64)
    ke_arr = 0.5 * 0.0014 * (v_arr ** 2)

    batch_std = evaluate_drone_impact_batch(
        y_arr, z_arr, drone, ke_arr, aspect_geometry=geom
    )
    for i in range(len(y_arr)):
        scalar = evaluate_drone_impact(
            float(y_arr[i]), float(z_arr[i]), drone, float(ke_arr[i]), aspect_geometry=geom
        )
        assert bool(batch_std["is_hit"][i]) == bool(scalar["is_hit"])
        assert bool(batch_std["is_lethal"][i]) == bool(scalar["is_lethal"])
        assert batch_std["component"][i] == scalar["component"]

    # High-fidelity equivalence with seeded RNG
    batch_hf = evaluate_drone_impact_batch(
        y_arr,
        z_arr,
        drone,
        ke_arr,
        aspect_geometry=geom,
        pellet_velocity_arr=v_arr,
        pellet_mass_kg=0.0014,
        pellet_diameter_m=0.0035,
        pellet_density_g_cm3=11.3,
        use_rotor_solidity=True,
        use_v50_ballistic_limit=True,
        enable_ray_shielding=True,
        use_oblique_ricochet=True,
        rng=np.random.default_rng(77),
    )
    rng_s = np.random.default_rng(77)
    for i in range(len(y_arr)):
        scalar_hf = evaluate_drone_impact(
            float(y_arr[i]),
            float(z_arr[i]),
            drone,
            float(ke_arr[i]),
            aspect_geometry=geom,
            pellet_velocity_mps=float(v_arr[i]),
            pellet_mass_kg=0.0014,
            pellet_diameter_m=0.0035,
            pellet_density_g_cm3=11.3,
            use_rotor_solidity=True,
            use_v50_ballistic_limit=True,
            enable_ray_shielding=True,
            use_oblique_ricochet=True,
            rng=rng_s,
        )
        assert bool(batch_hf["is_hit"][i]) == bool(scalar_hf["is_hit"])
        assert bool(batch_hf["is_lethal"][i]) == bool(scalar_hf["is_lethal"])

    # Oblique ricochet monotonicity: V50 increases with obliquity and steep graze ricochets
    n0 = compute_component_v50_perforation("fuselage_core", 240.0, 0.0014, 0.0035, obliquity_deg=0.0)
    n45 = compute_component_v50_perforation("fuselage_core", 240.0, 0.0014, 0.0035, obliquity_deg=45.0)
    n75 = compute_component_v50_perforation("fuselage_core", 240.0, 0.0014, 0.0035, obliquity_deg=75.0, enable_ricochet=True)
    assert n45["v50_mps"] > n0["v50_mps"]
    assert n75["ricochet"] is True
    assert n75["p_perforation"] == 0.0


def test_shot_cup_separation_and_stl_cad_raycast() -> None:
    res_nom = simulate_anti_drone_multishotgun_mc(
        layout_type="single",
        n_barrels=1,
        target_distance_m=15.0,
        shot_cup_separation_m=0.0,
        n_volleys=12,
        seed=42,
    )
    res_cup = simulate_anti_drone_multishotgun_mc(
        layout_type="single",
        n_barrels=1,
        target_distance_m=15.0,
        shot_cup_separation_m=6.0,
        use_oblique_ricochet=True,
        use_v50_ballistic_limit=True,
        n_volleys=12,
        seed=42,
    )
    assert res_cup["summary"]["cloud_cep_50_m"] < res_nom["summary"]["cloud_cep_50_m"]
    assert res_cup["summary"]["shot_cup_separation_m"] == 6.0
    assert res_cup["summary"]["use_oblique_ricochet"] is True

    # Built-in quadcopter sample STL + synthetic binary STL
    stl_sample = generate_sample_drone_stl_bytes("quadcopter")
    mesh_sample = parse_stl_mesh(stl_sample)
    assert mesh_sample["n_triangles"] >= 12
    proj_bin = project_stl_mesh_vulnerability(mesh_sample, aspect_mode="head_on_frontal", grid_res=28)
    assert proj_bin["presented_area_cm2"] > 20.0

    # Binary STL (1 triangle)
    bin_header = b"binary_stl_test" + b"\x00" * (80 - 15)
    bin_tri = struct.pack(
        "<12fH",
        0.0, 0.0, 1.0,
        -0.1, -0.1, 0.0,
        0.1, -0.1, 0.0,
        0.0, 0.1, 0.0,
        0,
    )
    mesh_bin = parse_stl_mesh(bin_header + struct.pack("<I", 1) + bin_tri)
    assert mesh_bin["format"] == "binary"
    assert mesh_bin["n_triangles"] == 1

    # ASCII STL (minimal tetrahedron)
    stl_ascii = (
        "solid tet\n"
        "  facet normal 1 0 0\n    outer loop\n      vertex 0.1 -0.1 -0.1\n      vertex 0.1 0.1 -0.1\n      vertex 0.1 0.0 0.1\n    endloop\n  endfacet\n"
        "  facet normal -1 0 0\n    outer loop\n      vertex -0.1 -0.1 -0.1\n      vertex -0.1 0.0 0.1\n      vertex -0.1 0.1 -0.1\n    endloop\n  endfacet\n"
        "  facet normal 0 -1 0\n    outer loop\n      vertex 0.1 -0.1 -0.1\n      vertex -0.1 -0.1 -0.1\n      vertex 0.1 0.1 -0.1\n    endloop\n  endfacet\n"
        "endsolid tet\n"
    )
    mesh_asc = parse_stl_mesh(stl_ascii)
    assert mesh_asc["format"] == "ascii"
    assert mesh_asc["n_triangles"] == 3

    y_p = np.array([0.0, 0.02, -0.02, 1.5])
    z_p = np.array([0.0, 0.01, 0.0, 1.5])
    v_p = np.array([340.0, 320.0, 310.0, 300.0])
    stl_eval = evaluate_stl_pellet_impacts_batch(
        y_p, z_p, proj_bin, pellet_velocity_arr=v_p, pellet_mass_kg=0.0014, pellet_diameter_m=0.0035, rng=np.random.default_rng(5)
    )
    assert stl_eval["n_hits"] >= 1
    assert bool(stl_eval["is_hit"][-1]) is False


def test_global_high_fidelity_propagation_and_twin_defaults() -> None:
    twin = get_default_nominal_twin()
    for key in (
        "pattern_distribution",
        "flyer_fraction",
        "shot_cup_separation_m",
        "air_density_kg_m3",
        "measured_shot_string_length_m",
    ):
        assert key in twin

    hf = get_default_physics_fidelity_profile(high_fidelity=True)
    assert hf["use_rotor_solidity"] is True
    assert hf["use_oblique_ricochet"] is True
    assert hf["shot_cup_separation_m"] > 0.0

    adv = run_adversarial_coevolution_ml(n_generations=2, n_mc_episodes=4, max_barrels=6, max_swarm_size=4, physics_fidelity=hf, seed=42)
    assert adv["summary"]["physics_fidelity_mode"] == "high_fidelity"

    rl = train_fire_control_policy_es(n_generations=2, pop_size=4, n_eval_episodes=3, physics_fidelity=hf, seed=42)
    assert rl["summary"]["rl_zero_breach_pct"] >= 0.0
    hil = run_hil_fire_control_trace(rl["policy_dict"], "evasive_fpv_dive", physics_fidelity=hf)
    assert hil["summary"]["n_frames"] > 0

    raid = simulate_unified_tactical_raid_mc(n_drones=4, n_mc_trials=4, physics_fidelity=hf, seed=42)
    assert raid["summary"]["n_drones"] == 4

    trk = simulate_recorded_track_engagement_mc(generate_sample_drone_track_csv(), n_mc_trials=4, physics_fidelity=hf, seed=42)
    assert trk["summary"]["physics_fidelity_mode"] == "high_fidelity"

    wiz = run_turret_sizing_wizard(physics_fidelity=hf, seed=42)
    assert len(wiz["top_3"]) == 3
    duel = run_red_vs_blue_duel_diagnostic(wiz["top_3"][0], {"drone_model": "fpv_quad_small", "swarm_size": 3}, n_episodes=3, physics_fidelity=hf, seed=42)
    assert "verdict" in duel


def test_native_range_instruments_fiducials_live_sprt_and_vv_keys() -> None:
    # 1. LabRadar
    lab = detect_and_parse_instrument_file("labradar_shot1.csv", generate_sample_labradar_native_csv().encode("utf-8"))
    assert lab["instrument_type"] == "labradar"
    assert lab["parsed"]["mean_v0_mps"] > 350.0
    assert "pellet_cd_scale" in lab["twin_patch"] or "mv_scale" in lab["twin_patch"]

    # 2. Garmin Xero CSV & XLSX (in-memory)
    gar_csv = generate_sample_garmin_xero_csv()
    gar_parsed = detect_and_parse_instrument_file("garmin_xero_shot.csv", gar_csv.encode("utf-8"))
    assert gar_parsed["instrument_type"] == "garmin_xero_c1"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Shot #", "Speed (fps)", "Power Factor"])
    for idx, fps in enumerate((1365.0, 1372.0, 1358.0, 1370.0), start=1):
        ws.append([idx, fps, 1.0])
    buf = io.BytesIO()
    wb.save(buf)
    gar_xlsx = detect_and_parse_instrument_file("garmin_xero_c1.xlsx", buf.getvalue())
    assert gar_xlsx["instrument_type"] == "garmin_xero_c1"
    assert gar_xlsx["parsed"]["n_shots"] == 4

    # 3. Kestrel LiNK
    kes = detect_and_parse_instrument_file("kestrel_link.csv", generate_sample_kestrel_link_csv().encode("utf-8"))
    assert kes["instrument_type"] == "kestrel_link"
    assert 0.8 <= kes["parsed"]["air_density_kg_m3"] <= 1.5
    assert "air_density_kg_m3" in kes["twin_patch"]

    # 4. PX4 flight log
    px4_csv = generate_sample_px4_flight_csv()
    px4_auto = detect_and_parse_instrument_file("px4_ulog.csv", px4_csv.encode("utf-8"))
    assert px4_auto["instrument_type"] == "px4_ardupilot"
    px4_parsed = parse_ardupilot_px4_flight_log(px4_csv)
    assert px4_parsed["n_points"] > 10
    assert "t_s,x_m,y_m,z_m" in px4_parsed["track_csv"]

    # 5. Printable Fiducial PDF + Auto Corner Detection
    pdf_b = generate_printable_fiducial_board_pdf()
    assert pdf_b.startswith(b"%PDF")
    fk_img, true_c, _ = generate_skewed_fiducial_board_png(seed=42)
    dig = digitize_pattern_board_image(fk_img, auto_detect_fiducials=True)
    assert dig["n_detected_holes"] >= 15
    assert dig["auto_detected_corners_px"] is not None
    assert len(dig["detected_holes_px"]) == dig["n_detected_holes"]

    # 6. Live SPRT decisions & Clopper-Pearson
    s_acc = evaluate_sprt_live_sequence([1] * 14)
    assert s_acc["decision"] == "ACCEPT_H1"
    assert s_acc["stopping_round"] is not None
    assert 0.0 < s_acc["ci95_low"] < s_acc["ci95_high"] <= 1.0
    s_rej = evaluate_sprt_live_sequence([0] * 12)
    assert s_rej["decision"] == "REJECT_H0"
    assert len(s_rej["livefire_rows"]) == 12

    # 7. Zero UserWarning & UI-expected keys across cuas_scientific_vv
    with warnings.catch_warnings(record=True) as rec_w:
        warnings.simplefilter("always", UserWarning)
        asme = compute_asme_vv20_metrics([1.0, 2.0, 3.0], [1.02, 1.99, 3.01], u_exp=0.05)
        user_w = [w for w in rec_w if issubclass(w.category, UserWarning) and "p-value capped" in str(w.message)]
        assert not user_w
    for k in ("mean_comparison_error_e", "area_validation_metric_avm", "validated"):
        assert k in asme
    dop = calibrate_doppler_radar_drag(generate_sample_doppler_radar_csv())
    for k in ("nominal_v0_mps", "cd_scale_se", "n_track_points"):
        assert k in dop
    ss = parse_weather_and_shotstring_csv(generate_sample_shotstring_csv())
    for k in ("arrival_time_sd_ms", "virtual_temp_k", "twin_patch"):
        assert k in ss
    bun = build_reproducible_validation_bundle(get_default_nominal_twin(), target_range_m=45.0)
    assert len(bun["manifest_sha256"]) == 64

    # 8. Briefing V&V slides (PDF & PPTX)
    vv_payload = {
        "experimental_benchmarks": run_experimental_validation_benchmarks(),
        "live_sprt": s_acc,
    }
    bp_pdf = generate_executive_briefing_pdf(vv_result=vv_payload)
    bp_pptx = generate_executive_briefing_pptx(vv_result=vv_payload)
    assert bp_pdf.startswith(b"%PDF")
    assert len(bp_pptx) > 10_000


def test_v18_cli_api_version_and_literature_100pct() -> None:
    assert ammo_mc.__version__ == "1.8.0"
    lit = build_literature_audit_summary()
    assert lit["total_sources"] == 87
    assert lit["linked_source_coverage_pct"] == 100.0

    runner = CliRunner()
    for args in (
        ["cuas", "vv-benchmarks", "--json"],
        ["cuas", "doe-plan", "--json"],
        ["cuas", "parse-instrument", "--json"],
        ["cuas", "stl-vuln", "--json"],
    ):
        res = runner.invoke(cli_app, args)
        assert res.exit_code == 0, res.output
        json.loads(res.stdout)

    client = TestClient(fastapi_app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["version"] == "1.8.0"

    arr_r = client.post("/cuas/array", json={"n_barrels": 4, "use_oblique_ricochet": True, "shot_cup_separation_m": 4.0, "n_volleys": 6})
    assert arr_r.status_code == 200
    assert arr_r.json()["summary"]["use_oblique_ricochet"] is True

    sprt_r = client.post("/cuas/vv/sprt-live", json={"outcomes": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    assert sprt_r.status_code == 200
    assert "decision" in sprt_r.json()

    inst_r = client.post("/cuas/calibrate/parse-instrument", json={"filename": "kestrel.csv", "csv_text": generate_sample_kestrel_link_csv()})
    assert inst_r.status_code == 200
    assert inst_r.json()["instrument_type"] == "kestrel_link"

    pdf_r = client.get("/cuas/calibrate/fiducial-board-pdf")
    assert pdf_r.status_code == 200
    assert pdf_r.content.startswith(b"%PDF")

    stl_r = client.post("/cuas/geometry/stl-vulnerability", json={"drone_type": "quadcopter"})
    assert stl_r.status_code == 200
    assert stl_r.json()["presented_area_cm2"] > 0.0
