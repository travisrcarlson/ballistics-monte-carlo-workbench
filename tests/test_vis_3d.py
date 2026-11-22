"""Tests for 3D visualization, flight plume bundles, and detonation modeling (ammo_mc.vis_3d)."""

from __future__ import annotations

from pathlib import Path
import plotly.graph_objects as go
import pytest
from typer.testing import CliRunner

from ammo_mc.cli import app
from ammo_mc.environment import aerodynamic_reynolds_number, sutherland_viscosity_Pa_s
from ammo_mc.exterior import generate_ballistic_firing_table, simulate_mpm_vs_3dof_trajectory
from ammo_mc.interior import LumpedIBConfig, solve_lumped_interior_ballistics
from ammo_mc.mcdrag import McDragInputs
from ammo_mc.pipeline import run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.terminal_blast import tb_13_axelsson_chest_wall, tb_15_simple_safety_distances
from ammo_mc.vis_3d import (
    create_3d_animated_flight_figure,
    create_3d_blast_and_frag_figure,
    create_3d_iatg_safety_zones_figure,
    create_3d_multi_facility_site_planner_figure,
    create_3d_multi_preset_comparison_figure,
    create_3d_phase_space_figure,
    create_3d_projectile_geometry_figure,
    create_3d_trajectory_bundle_figure,
    export_3d_html,
)

runner = CliRunner()


@pytest.fixture(scope="module")
def sim_result():
    ammo, scn = get_preset("NOTIONAL-SA-01")
    scn.fidelity = "L1"
    scn.n_occasions = 2
    scn.rounds_per_occasion = 5
    scn.record_trajectories_count = 4
    res = run_simulation(ammo, scn)
    return res, ammo, scn


def test_create_3d_trajectory_bundle_figure(sim_result) -> None:
    res, ammo, scn = sim_result
    fig = create_3d_trajectory_bundle_figure(
        result=res,
        scenario=scn,
        color_by="speed",
        show_terrain=True,
        show_target=True,
        show_poi=True,
        max_trajectories=4,
    )
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if t.name is not None]
    assert any("Terrain" in n for n in trace_names)
    assert any("Weapon" in n for n in trace_names)
    assert any("Target" in n for n in trace_names)
    assert any("Trajectory" in n for n in trace_names)

    # Test altitude coloring
    fig_alt = create_3d_trajectory_bundle_figure(result=res, scenario=scn, color_by="altitude")
    assert isinstance(fig_alt, go.Figure)


def test_create_3d_blast_and_frag_figure(sim_result) -> None:
    res, ammo, scn = sim_result
    fig = create_3d_blast_and_frag_figure(
        result=res,
        scenario=scn,
        burst_alt_m=10.0,
        n_frag_traces=40,
        charge_mass_kg=5.0,
        explosive_name="Composition B",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4  # 3 overpressure surfaces + burst point + fragment rays


def test_export_3d_html(sim_result, tmp_path: Path) -> None:
    res, ammo, scn = sim_result
    fig = create_3d_trajectory_bundle_figure(res, scenario=scn, max_trajectories=2)
    out_file = tmp_path / "trajectory_3d.html"
    exported_path = export_3d_html(fig, out_file)
    assert Path(exported_path).exists()
    assert Path(exported_path).stat().st_size > 50_000


def test_cli_sim_3d_command(tmp_path: Path) -> None:
    out_html = tmp_path / "cli_3d.html"
    cmd_args = [
        "sim-3d",
        "--preset", "NOTIONAL-SA-01",
        "--fidelity", "L1",
        "--trials", "10",
        "--trajectories", "3",
        "--out", str(out_html),
    ]
    res = runner.invoke(app, cmd_args)
    assert res.exit_code == 0
    assert "Interactive 3D Simulation saved to" in res.output
    assert out_html.exists()
    assert out_html.stat().st_size > 50_000


def test_create_3d_animated_and_cad_and_comparison(sim_result) -> None:
    res, ammo, scn = sim_result
    fig_anim = create_3d_animated_flight_figure(res, scenario=scn, max_trajectories=3, n_frames=10)
    assert isinstance(fig_anim, go.Figure)
    assert len(fig_anim.frames) == 10

    fig_cad = create_3d_projectile_geometry_figure(
        McDragInputs(d_ref_mm=155.0, total_length_cal=5.2, nose_length_cal=2.8, boattail_length_cal=0.6)
    )
    assert isinstance(fig_cad, go.Figure)
    assert len(fig_cad.data) >= 3

    fig_comp = create_3d_multi_preset_comparison_figure([(res, scn)], show_terrain=True)
    assert isinstance(fig_comp, go.Figure)

    ib_cfg = LumpedIBConfig(
        calibre_m=0.155,
        travel_m=6.0,
        chamber_volume_m3=0.023,
        proj_mass_kg=43.2,
        charge_mass_kg=11.5,
    )
    ib_res = solve_lumped_interior_ballistics(ib_cfg, dt=2e-5, record_history=True)
    assert "history" in ib_res
    assert len(ib_res["history"]["t_ms"]) > 10
    assert "v_gas_L" in ib_res["history"]
    assert ib_res["muzzle_velocity_mps"] > 400.0

    fig_ps = create_3d_phase_space_figure(res, scenario=scn, color_metric="dynamic_pressure_kPa", max_trajectories=3)
    assert isinstance(fig_ps, go.Figure)
    assert any("Sonic Barrier" in (t.name or "") for t in fig_ps.data)

    mu_288 = float(sutherland_viscosity_Pa_s(288.15))
    assert 1.75e-5 < mu_288 < 1.82e-5
    re_val = float(aerodynamic_reynolds_number(1.225, 800.0, 0.155, 288.15))
    assert re_val > 5.0e6

    # UNODA IATG 01.80 / 02.20 QD & 3D PES Safety Domes + Axelsson History
    qd_1000 = tb_15_simple_safety_distances(auw_kg=2500.0, neq_kg=1000.0)
    assert abs(float(qd_1000["process_building_distance_m"]) - 80.0) < 1e-6
    assert abs(float(qd_1000["public_traffic_route_distance_m"]) - 148.0) < 1e-6
    assert abs(float(qd_1000["inhabited_building_distance_m"]) - 222.0) < 1e-6
    assert abs(float(qd_1000["vulnerable_building_distance_m"]) - 444.0) < 1e-6

    fig_iatg = create_3d_iatg_safety_zones_figure(neq_kg=1000.0, auw_kg=2500.0, explosive_name="Composition B")
    assert isinstance(fig_iatg, go.Figure)
    assert any("IBD" in (t.name or "") for t in fig_iatg.data)

    ax_res = tb_13_axelsson_chest_wall(Ps_kPa=150.0, td_ms=5.0, record_history=True)
    assert "history" in ax_res
    assert len(ax_res["history"]["t_ms"]) > 10
    assert len(ax_res["history"]["v_mps"]) == len(ax_res["history"]["t_ms"])


def test_expanded_presets_and_failure_modes_and_scientific_upgrades() -> None:
    presets = list_presets()
    assert len(presets) >= 26
    assert all(p["notional"] is True for p in presets)
    rk_presets = [p for p in presets if p["class_id"].startswith("RK-")]
    assert len(rk_presets) == 4

    # Test new system presets & multi-stage Failure Modes (FMEA) including Rocket presets
    ammo_rk, scn_rk = get_preset("NOTIONAL-RK-03")
    scn_rk.n_occasions = 8
    scn_rk.rounds_per_occasion = 10
    ammo_rk.failure_modes.p_propulsion_anomaly = 0.12
    ammo_rk.failure_modes.p_fin_or_sabot_anomaly = 0.10
    res_rk = run_simulation(ammo_rk, scn_rk)
    assert res_rk.failure_mode_summary["total_rounds"] == 80
    assert res_rk.failure_mode_summary["any_anomaly_or_failure_rate"] > 0.05

    ammo, scn = get_preset("NOTIONAL-NV-01")
    scn.n_occasions = 10
    scn.rounds_per_occasion = 10
    ammo.failure_modes.p_propulsion_anomaly = 0.08
    ammo.failure_modes.p_fin_or_sabot_anomaly = 0.06
    ammo.failure_modes.p_fuze_arming_fail = 0.05
    ammo.failure_modes.p_fuze_early_burst = 0.05
    ammo.failure_modes.p_low_order_detonation = 0.08
    res_nv = run_simulation(ammo, scn)

    assert "failure_mode" in res_nv.trials
    assert len(res_nv.trials["failure_mode"]) == 100
    assert res_nv.failure_mode_summary["total_rounds"] == 100
    assert res_nv.failure_mode_summary["any_anomaly_or_failure_rate"] > 0.05
    assert "Nominal Function" in res_nv.failure_mode_summary["counts"]

    # Test STANAG 4355 MPM vs 3-DoF trajectory comparison
    mpm_cmp = simulate_mpm_vs_3dof_trajectory(v0_mps=685.0, qe_deg=30.0)
    assert mpm_cmp["terminal_spin_drift_m"] > 10.0
    assert len(mpm_cmp["alpha_e_mrad"]) > 20

    # Test Automated Ballistic Firing Table & Met Unit-Effects (including Rocket boost)
    ft_rows = generate_ballistic_firing_table(
        v0_mps=float(ammo_rk.launch.mv.mu),
        calibre_m=float(ammo_rk.calibre_m),
        mass_kg=float(ammo_rk.mass_kg),
        qe_min_deg=15.0,
        qe_max_deg=35.0,
        n_elevations=3,
        rocket_burn_time_s=float(ammo_rk.launch.rocket_burn_s),
        rocket_mdot_kg_s=float(ammo_rk.launch.rocket_mdot_kg_s),
        rocket_ve_mps=float(ammo_rk.launch.rocket_ve_mps),
    )
    assert len(ft_rows) == 3
    assert ft_rows[1]["Range (m)"] > ft_rows[0]["Range (m)"]
    assert ft_rows[1]["Range (m)"] > 10000.0

    # Test Multi-Facility 3D Site Planner & Sympathetic Detonation Matrix
    fig_site, comp_rows = create_3d_multi_facility_site_planner_figure()
    assert isinstance(fig_site, go.Figure)
    assert len(comp_rows) >= 6
    assert any("Inter-Magazine IMD" in str(r.get("Relation Type", "")) for r in comp_rows)

    # Test Multi-Barrel 2D & 3D Drone Cross-Section Visualizations
    from ammo_mc.small_arms import simulate_multibarrel_burst_mc, benchmark_multibarrel_aspect_angles_mc
    from ammo_mc.vis_3d import create_2d_multibarrel_target_figure, create_multibarrel_aspect_radar_figure, create_3d_multibarrel_dispersion_figure

    mb_sim = simulate_multibarrel_burst_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        target_type="drone",
        drone_model="commercial_quad_medium",
        aspect_mode="head_on_frontal",
        burst_rounds=30,
        target_range_m=100.0,
        seed=42,
    )
    fig_2d = create_2d_multibarrel_target_figure(mb_sim)
    assert isinstance(fig_2d, go.Figure)
    assert len(fig_2d.data) >= 3

    fig_3d = create_3d_multibarrel_dispersion_figure(mb_sim)
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_3d.data) >= 4

    mb_bench = benchmark_multibarrel_aspect_angles_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        burst_rounds=24,
        target_range_m=100.0,
        seed=42,
    )
    fig_radar = create_multibarrel_aspect_radar_figure(mb_bench)
    assert isinstance(fig_radar, go.Figure)
    assert len(fig_radar.data) == 2

