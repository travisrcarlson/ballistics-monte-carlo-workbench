"""Comprehensive unit & integration tests for the 5-Pillar C-UAS Autonomy, Digital Twin, Hardware CAD & UX Upgrades."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import plotly.graph_objects as go
from typer.testing import CliRunner

from ammo_mc.api import app
from ammo_mc.briefing_export import generate_executive_briefing_pdf, generate_executive_briefing_pptx
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_adversarial import (
    FEATURE_NAMES,
    FEATURE_SIDES,
    encode_matchup_features,
    evaluate_red_vs_blue_matchup_mc,
    normalize_blue_config,
    normalize_red_config,
    run_adversarial_coevolution_ml,
)
from ammo_mc.cuas_battlespace_ml import run_battlespace_coevolution_mc
from ammo_mc.cuas_hardware import (
    estimate_turret_swap_c,
    evaluate_otm_stabilization_degradation,
    export_muzzle_plate_dxf,
    export_muzzle_plate_svg,
)
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.cuas_rl_policy import (
    NeuralFireControlPolicy,
    evaluate_fire_control_policy_mc,
    get_default_rl_training_curriculum,
    train_fire_control_policy_es,
)
from ammo_mc.cuas_track_replay import (
    generate_sample_drone_track_csv,
    parse_drone_track_csv,
    simulate_recorded_track_engagement_mc,
)
from ammo_mc.cuas_wizard import run_red_vs_blue_duel_diagnostic, run_turret_sizing_wizard
from ammo_mc.small_arms import generate_multishotgun_array_layout
from ammo_mc.vis_cuas_advanced import (
    create_battlespace_coevolution_figure,
    create_duel_root_cause_figure,
    create_recorded_track_replay_figure,
    create_rl_action_heatmap_figure,
    create_rl_policy_benchmark_figure,
    create_swap_c_and_otm_figure,
)


def test_hybrid_robust_calibrated_coevolution_and_failure_modes() -> None:
    b_hyb = normalize_blue_config(
        {
            "layout_type": "concentric",
            "n_barrels": 8,
            "barrels_per_burst": 2,
            "ammunition_preset": "12G_NO4_BUCK_C_UAS",
            "choke": "Modified",
            "hybrid_mode": True,
            "inner_ammunition_preset": "12G_TSS_NO2_C_UAS",
            "inner_choke": "Extra Full",
            "hybrid_switch_range_m": 42.0,
        }
    )
    r_cfg = normalize_red_config({"swarm_size": 7, "maneuver_g": 3.5, "drone_speed_mps": 32.0})
    assert len(encode_matchup_features(b_hyb, r_cfg)) == len(FEATURE_NAMES) == len(FEATURE_SIDES)

    cal_twin = {
        "spread_factor": 1.12,
        "spread_factor_se": 0.05,
        "aim_bias_mrad": 1.2,
        "mv_scale": 0.98,
        "mv_sd_mps": 4.5,
        "platform_jitter_mrad": 1.8,
    }
    res = evaluate_red_vs_blue_matchup_mc(
        b_hyb,
        r_cfg,
        n_episodes=15,
        calibration_profile=cal_twin,
        robust_uncertainty_mode=True,
        seed=42,
    )
    assert res["hybrid_mode"] is True
    assert res["robust_uncertainty_mode"] is True
    assert 0.0 <= res["cvar20_utility"] <= 100.0
    fm = res["failure_modes"]
    assert set(fm["counts"].keys()) == {
        "reload_gap",
        "slew_bottleneck",
        "tracking_evasion_miss",
        "insufficient_lethality_spread",
    }
    if fm["total_leakers"] > 0:
        pct_sum = (
            fm["reload_gap_pct"]
            + fm["slew_bottleneck_pct"]
            + fm["tracking_evasion_miss_pct"]
            + fm["insufficient_lethality_spread_pct"]
        )
        assert 99.0 <= pct_sum <= 101.0

    coevo = run_adversarial_coevolution_ml(
        n_generations=1,
        n_mc_episodes=4,
        max_barrels=8,
        max_swarm_size=5,
        allow_hybrid=True,
        robust_uncertainty_mode=True,
        calibration_profile=cal_twin,
        seed=42,
    )
    assert coevo["summary"]["allow_hybrid"] is True
    assert coevo["summary"]["robust_uncertainty_mode"] is True
    assert coevo["summary"]["calibrated_digital_twin"] is True


def test_rl_fire_control_policy_training_and_roundtrip() -> None:
    pol = NeuralFireControlPolicy(seed=7)
    f0 = NeuralFireControlPolicy.build_target_features(35.0, 60.0, 1.0, 0.15, 25.0, 2.0, 4, 8, 3, 6, 0.72)
    f1 = NeuralFireControlPolicy.build_target_features(52.0, 60.0, 2.1, 0.30, 22.0, 1.5, 4, 8, 3, 6, 0.35)
    act = pol.select_action(np.vstack([f0, f1]), b_ready=4, max_burst=3)
    assert act["target_idx"] in (0, 1)
    assert 1 <= act["burst_size"] <= 3
    assert 0.0 <= act["hold_fire_prob"] <= 1.0

    restored = NeuralFireControlPolicy.from_dict(pol.to_dict())
    np.testing.assert_allclose(pol.get_flat_params(), restored.get_flat_params(), atol=1e-5)

    ev = evaluate_fire_control_policy_mc(
        {"n_barrels": 6, "barrels_per_burst": 2},
        "minimum_time_to_breach",
        get_default_rl_training_curriculum()[:2],
        n_episodes_per_scenario=3,
    )
    assert 0.0 <= ev["zero_breach_pct"] <= 100.0

    train_res = train_fire_control_policy_es(
        n_generations=3,
        pop_size=6,
        n_eval_episodes=8,
        seed=42,
    )
    assert len(train_res["training_history"]) == 4
    assert len(train_res["benchmark_rows"]) == 4
    assert isinstance(create_rl_policy_benchmark_figure(train_res), go.Figure)
    assert isinstance(create_rl_action_heatmap_figure(train_res), go.Figure)


def test_battlespace_coevolution_and_heterogeneous_raid() -> None:
    pkg = [
        {"role": "decoy", "drone_model": "fpv_quad_small", "count": 3, "speed_mps": 30.0, "altitude_m": 20.0, "maneuver_g": 2.0, "fiber_optic_fraction": 0.0},
        {"role": "bomber", "drone_model": "heavy_octocopter_large", "count": 2, "speed_mps": 16.0, "altitude_m": 26.0, "maneuver_g": 1.0, "fiber_optic_fraction": 1.0},
    ]
    raid = simulate_unified_tactical_raid_mc(
        swarm_package=pkg,
        outer_layer_type="rf_gnss_jammer",
        n_mc_trials=6,
        seed=42,
    )
    assert raid["summary"]["n_drones"] == 5
    assert raid["summary"]["outer_layer_type"] == "rf_gnss_jammer"
    assert raid["summary"]["outer_layer_kills"] >= 0.0

    bs = run_battlespace_coevolution_mc(
        terrain_preset="rolling_hills_fob",
        n_generations=1,
        n_mc_trials=3,
        max_turrets=4,
        seed=42,
    )
    assert "raid_handoff_payload" in bs
    assert bs["raid_handoff_payload"]["custom_turrets"]
    assert bs["nash"]["duality_gap"] < 1e-4
    assert isinstance(create_battlespace_coevolution_figure(bs), go.Figure)


def test_recorded_track_replay_local_and_gps() -> None:
    csv_local = generate_sample_drone_track_csv("multi_drone_pincer", seed=42)
    parsed = parse_drone_track_csv(csv_local)
    assert parsed["coord_mode"] == "local_enu"
    assert parsed["n_tracks"] == 3
    res_loc = simulate_recorded_track_engagement_mc(parsed, n_mc_trials=12, seed=42)
    assert res_loc["summary"]["n_tracks"] == 3
    assert isinstance(create_recorded_track_replay_figure(res_loc), go.Figure)

    gps_csv = "\n".join([
        "track_id,t_s,lat_deg,lon_deg,alt_m",
        "GPS-1,0.0,32.90090, -106.40000, 30.0",
        "GPS-1,1.0,32.90065, -106.40000, 26.0",
        "GPS-1,2.0,32.90040, -106.40000, 22.0",
        "GPS-1,3.0,32.90018, -106.40000, 18.0",
    ])
    parsed_gps = parse_drone_track_csv(gps_csv, origin_lat=32.9000, origin_lon=-106.4000, turret_alt_m=0.0)
    assert parsed_gps["coord_mode"] == "gps_enu"
    assert parsed_gps["tracks"]["GPS-1"]["speed_mps"][0] > 15.0


def test_hardware_swap_c_otm_and_dxf_svg_cad() -> None:
    sw4 = estimate_turret_swap_c(n_barrels=4, layout_type="circular")
    sw10 = estimate_turret_swap_c(n_barrels=10, layout_type="concentric", hybrid_mode=True)
    assert sw10["summary"]["total_mass_kg"] > sw4["summary"]["total_mass_kg"]
    assert sw10["summary"]["inertia_az_kg_m2"] > sw4["summary"]["inertia_az_kg_m2"]
    assert sw10["summary"]["unit_bom_usd"] > sw4["summary"]["unit_bom_usd"]

    otm = evaluate_otm_stabilization_degradation(
        {"n_barrels": 6, "barrels_per_burst": 2, "ammunition_preset": "12G_NO4_BUCK_C_UAS"},
        carrier_profile="tactical_4x4_offroad_40kph",
        gyro_grade="standard_mems_gyro",
    )
    assert otm["r80_by_mode"]["static_ideal"] >= otm["r80_by_mode"]["tactical_fog_gyro"]
    assert otm["r80_by_mode"]["tactical_fog_gyro"] >= otm["r80_by_mode"]["standard_mems_gyro"]
    assert otm["r80_by_mode"]["standard_mems_gyro"] >= otm["r80_by_mode"]["unstabilized"]
    assert isinstance(create_swap_c_and_otm_figure(sw10, otm), go.Figure)

    bbls = generate_multishotgun_array_layout(layout_type="concentric", n_barrels=8)
    svg = export_muzzle_plate_svg(bbls, layout_type="concentric")
    dxf = export_muzzle_plate_dxf(bbls, layout_type="concentric")
    assert "<svg" in svg and svg.count("<circle") >= 16
    assert "SECTION" in dxf and "ENTITIES" in dxf and "CIRCLE" in dxf and dxf.strip().endswith("EOF")


def test_turret_sizing_wizard_duel_and_briefings() -> None:
    wiz = run_turret_sizing_wizard(
        drone_model="fpv_quad_small",
        threat_speed_mps=25.0,
        required_range_m=45.0,
        mount_class="light_ugv",
    )
    assert len(wiz["top_3"]) == 3
    assert wiz["has_feasible"] is True
    assert all(r["mount_feasible"] for r in wiz["top_3"])

    duel = run_red_vs_blue_duel_diagnostic(
        blue_config={"n_barrels": 4, "barrels_per_burst": 2, "ammunition_preset": "12G_NO4_BUCK_C_UAS"},
        red_config={"drone_model": "fpv_quad_small", "swarm_size": 8, "maneuver_g": 3.5, "drone_speed_mps": 32.0},
        n_episodes=15,
    )
    assert duel["verdict"] in ("BLUE DOMINANT", "CONTESTED EDGE", "RED BREACH ADVANTAGE")
    assert len(duel["prescriptions"]) >= 1
    assert isinstance(create_duel_root_cause_figure(duel), go.Figure)

    adv_res = run_adversarial_coevolution_ml(n_generations=1, n_mc_episodes=3, max_barrels=6, max_swarm_size=4)
    pdf_bytes = generate_executive_briefing_pdf(adversarial_result=adv_res)
    pptx_bytes = generate_executive_briefing_pptx(adversarial_result=adv_res)
    assert pdf_bytes.startswith(b"%PDF")
    assert pptx_bytes[:2] == b"PK"


def test_cli_and_api_cuas_v15_endpoints(tmp_path: Path) -> None:
    runner = CliRunner()
    dxf_out = tmp_path / "muzzle.dxf"
    svg_out = tmp_path / "muzzle.svg"
    res_sc = runner.invoke(
        cli_app,
        ["cuas", "swap-c", "--barrels", "6", "--out-dxf", str(dxf_out), "--out-svg", str(svg_out)],
    )
    assert res_sc.exit_code == 0
    assert dxf_out.exists() and svg_out.exists()

    res_wiz = runner.invoke(cli_app, ["cuas", "wizard", "--range-m", "45", "--mount", "light_ugv"])
    assert res_wiz.exit_code == 0 and "Turret Sizing Wizard" in res_wiz.output

    res_trk = runner.invoke(cli_app, ["cuas", "track-replay", "--scenario", "evasive_fpv_dive", "--trials", "8"])
    assert res_trk.exit_code == 0 and "Recorded Track Replay" in res_trk.output

    res_rl = runner.invoke(cli_app, ["cuas", "rl-policy", "--generations", "2", "--episodes", "4"])
    assert res_rl.exit_code == 0 and "RL Fire-Control Training Complete" in res_rl.output

    res_bs = runner.invoke(cli_app, ["cuas", "battlespace-ml", "--generations", "1", "--trials", "3"])
    assert res_bs.exit_code == 0 and "3D Battlespace Co-Evolution Complete" in res_bs.output

    client = TestClient(app)
    assert client.post("/cuas/wizard", json={"required_range_m": 45.0}).status_code == 200
    assert client.post("/cuas/duel", json={"blue_config": {"n_barrels": 6}, "red_config": {"swarm_size": 5}, "n_episodes": 8}).status_code == 200
    assert client.post("/cuas/swap-c", json={"n_barrels": 8}).status_code == 200
    cad_resp = client.post("/cuas/cad/dxf", json={"n_barrels": 8, "layout_type": "concentric"})
    assert cad_resp.status_code == 200 and b"ENTITIES" in cad_resp.content
    svg_resp = client.post("/cuas/cad/svg", json={"n_barrels": 8, "layout_type": "concentric"})
    assert svg_resp.status_code == 200 and b"<svg" in svg_resp.content
    assert client.post("/cuas/track-replay", json={"n_mc_trials": 8}).status_code == 200
    assert client.post("/cuas/rl-policy", json={"n_generations": 2, "pop_size": 6, "n_eval_episodes": 6}).status_code == 200
    assert client.post("/cuas/battlespace-ml", json={"n_generations": 1, "n_mc_trials": 3}).status_code == 200
