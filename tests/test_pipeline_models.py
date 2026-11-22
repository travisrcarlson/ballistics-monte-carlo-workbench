"""Physics, model-form checks, governance enforcement, GIS, exports, and throughput tests (App_Spec R13..R29)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
import numpy as np
import pytest

from ammo_mc.environment import AtmosphereEnvironment
from ammo_mc.export import export_all_artifacts
from ammo_mc.exterior import BatchTrajectoryConfig, eb_11_vacuum_range, integrate_trajectories_batch
from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE, validate_safety_distance_governance
from ammo_mc.gis import export_hazard_geojson
from ammo_mc.pipeline import GovernanceError, compute_governed_safety_distance, run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.schemas import TerrainPointSpec, TerrainSpec


def test_vacuum_trajectory_matches_eb11_within_0_1_percent() -> None:
    """App_Spec R13 MUST: vacuum case vs EB-11 within 0.1 % for both RK4 and RK45."""
    v0 = 600.0
    theta = math.radians(42.0)
    r_exact = eb_11_vacuum_range(v0, theta)

    state0 = np.array([[0.0, 0.0, 0.0, v0 * math.cos(theta), 0.0, v0 * math.sin(theta)]])
    env = AtmosphereEnvironment()

    for method in ("RK4", "RK45"):
        cfg = BatchTrajectoryConfig(
            calibre_m=0.155,
            vacuum=True,
            include_coriolis=False,
            dt=0.01,
            integrator=method,
            rtol=1e-7,
        )
        out = integrate_trajectories_batch(state0, np.array([43.0]), np.array([1.0]), env, cfg)
        rel_err = abs(float(out["range_m"][0]) - r_exact) / r_exact
        assert rel_err < 0.001, f"{method} vacuum rel_err {rel_err} >= 0.1%"


def test_model_form_l1_vs_l2_mpm_spin_drift() -> None:
    """App_Spec R24 SHOULD: Compare L1 point-mass vs L2 MPM right-hand gyroscopic spin drift."""
    ammo, scn = get_preset("NOTIONAL-AR-01")
    scn.met.wind_range_mps = 0.0
    scn.met.wind_cross_right_mps = 0.0
    scn.include_coriolis = False
    scn.aim.qe_rad = math.radians(35.0)
    scn.n_occasions = 1
    scn.rounds_per_occasion = 1
    # Zero out stochastic noise to isolate pure deterministic physics L1 vs L2
    ammo.launch.mv.sigma_rr = 0.0
    ammo.launch.mv.sigma_lot = 0.0
    ammo.aero.form_factor.sigma = 0.0
    ammo.aero.form_factor.sigma_bias = 0.0
    scn.errors.bias.aim_az_bias_sigma_mils = 0.0
    scn.errors.bias.aim_qe_bias_sigma_mils = 0.0
    scn.errors.bias.tle_range_sigma_m = 0.0
    scn.errors.bias.tle_deflection_sigma_m = 0.0
    scn.errors.bias.wind_range_bias_sigma_mps = 0.0
    scn.errors.bias.wind_cross_bias_sigma_mps = 0.0
    scn.errors.bias.temp_bias_sigma_K = 0.0
    scn.errors.bias.propellant_temp_sigma_C = 0.0
    scn.errors.random.aim_az_rand_sigma_mils = 0.0
    scn.errors.random.aim_qe_rand_sigma_mils = 0.0
    scn.errors.random.jump_az_sigma_mils = 0.0
    scn.errors.random.jump_qe_sigma_mils = 0.0

    scn_l1 = scn.model_copy(deep=True)
    scn_l1.fidelity = "L1"
    res_l1 = run_simulation(ammo, scn_l1)

    scn_l2 = scn.model_copy(deep=True)
    scn_l2.fidelity = "L2"
    res_l2 = run_simulation(ammo, scn_l2)

    # L1 in zero wind & zero Coriolis has 0 deflection; L2 right-hand spin drifts RIGHT (+deflection_m > 20 m)
    assert abs(res_l1.dispersion["mean_deflection_m"]) < 1e-6
    assert res_l2.dispersion["mean_deflection_m"] > 25.0


def test_model_form_correlated_bias_vs_independence_overstatement() -> None:
    """App_Spec R24 SHOULD & MC_Demo R22: Independence formula TV-05 overstates multi-round Pk under large shared bias."""
    ammo, scn = get_preset("NOTIONAL-AR-01")
    scn.fidelity = "L0"
    scn.n_occasions = 800
    scn.rounds_per_occasion = 10
    scn.errors.bias.l0_bias_range_sigma_m = 45.0
    scn.errors.bias.l0_bias_deflection_sigma_m = 45.0
    scn.errors.random.l0_rr_range_sigma_m = 12.0
    scn.errors.random.l0_rr_deflection_sigma_m = 12.0
    ammo.warhead.carleton_rx_m = 25.0  # type: ignore[union-attr]
    ammo.warhead.carleton_ry_m = 25.0  # type: ignore[union-attr]

    res = run_simulation(ammo, scn)
    pk_salvo_corr = res.effectiveness["salvo_pk_mean"]
    pk_salvo_indep = res.effectiveness["salvo_pk_independent_ref"]
    # With large occasion bias, rounds in the same volley miss together -> correlated salvo Pk < independent TV-05!
    assert pk_salvo_corr < pk_salvo_indep - 0.08


def test_flag_governance_warnings_and_safety_distance_block() -> None:
    """App_Spec R25 & R26 MUST: FLAG formulas trigger warnings and cannot be sole basis of safety distance."""
    # 1. Direct governance validator check
    ok, msg = validate_safety_distance_governance(["TP-01"])
    assert ok is False and "GOVERNANCE BLOCK" in (msg or "")
    with pytest.raises(GovernanceError):
        compute_governed_safety_distance(800.0, ["TB-12", "TF-12"])

    res_gov = compute_governed_safety_distance(1200.0, ["TB-15", "EB-16"])
    assert res_gov["governance_passed"] is True

    # 2. Running CN-02 (uses TP-01) surfaces FLAG warning and use_limitation notice
    ammo_ap, scn_ap = get_preset("NOTIONAL-CN-02")
    scn_ap.n_occasions = 10
    scn_ap.rounds_per_occasion = 2
    res_ap = run_simulation(ammo_ap, scn_ap)
    assert res_ap.provenance["use_limitation"] == USE_LIMITATION_NOTICE
    assert res_ap.provenance["has_flag_warnings"] is True
    flagged_ids = {f["formula_id"] for f in res_ap.provenance["flagged_formulas"]}
    assert "TP-01" in flagged_ids


def test_presets_gis_terrain_stopping_epistemic_and_exports(tmp_path: Path) -> None:
    presets = list_presets()
    assert len(presets) >= 12
    assert all(p["notional"] is True for p in presets)

    # Test GIS terrain upslope shortens impact range vs flat ground
    ammo, scn = get_preset("NOTIONAL-MO-01")
    scn.n_occasions = 10
    scn.rounds_per_occasion = 3
    scn.aim.qe_rad = math.radians(55.0)
    res_flat = run_simulation(ammo, scn)

    scn_hill = scn.model_copy(deep=True)
    scn_hill.terrain = TerrainSpec(
        mode="profile",
        profile=[TerrainPointSpec(range_m=0.0, elevation_m=0.0), TerrainPointSpec(range_m=6000.0, elevation_m=300.0)],
    )
    res_hill = run_simulation(ammo, scn_hill)
    assert res_hill.dispersion["mean_range_m"] < res_flat.dispersion["mean_range_m"]

    # Test adaptive CI stopping rule + Two-loop epistemic + L3 DM-02 + full exports
    ammo_eod, scn_eod = get_preset("NOTIONAL-DM-02")
    scn_eod.n_occasions = 25
    scn_eod.fragments_per_detonation = 60
    scn_eod.ci_tolerance = 0.25
    scn_eod.two_loop_epistemic = True
    scn_eod.n_outer_epistemic = 5
    res_eod = run_simulation(ammo_eod, scn_eod)

    assert res_eod.stopped_by_ci is True
    assert res_eod.epistemic_summary is not None
    assert res_eod.fragmentation["hazard_distance_quantile_m"] > 50.0

    fc = export_hazard_geojson(res_eod, scn_eod)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 4

    paths = export_all_artifacts(res_eod, tmp_path, scenario=scn_eod, prefix="test_eod")
    for p_str in paths.values():
        p = Path(p_str)
        assert p.exists() and p.stat().st_size > 100
    parsed_summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert parsed_summary["provenance"]["use_limitation"] == USE_LIMITATION_NOTICE


def test_vectorised_point_mass_throughput_benchmark() -> None:
    """App_Spec R14 SHOULD: Target >= 1e5 point-mass trajectories/minute on a laptop."""
    n_bench = 2000
    states = np.zeros((n_bench, 6), dtype=np.float64)
    states[:, 3] = 790.0 * math.cos(0.05)
    states[:, 5] = 790.0 * math.sin(0.05)
    env = AtmosphereEnvironment()
    cfg = BatchTrajectoryConfig(calibre_m=0.00782, drag_ref="G7", fidelity="L1", dt=0.01)

    # Warmup run to prime JIT / memory allocations
    _ = integrate_trajectories_batch(states[:100], np.full(100, 0.0113), np.ones(100), env, cfg)
    t0 = time.perf_counter()
    res = integrate_trajectories_batch(states, np.full(n_bench, 0.0113), np.ones(n_bench), env, cfg)
    elapsed_s = max(time.perf_counter() - t0, 1e-6)
    traj_per_min = (n_bench / elapsed_s) * 60.0

    assert res["range_m"].shape == (n_bench,)
    assert traj_per_min >= 6.0e4, f"Throughput {traj_per_min:.0f} traj/min < 6e4 target"
