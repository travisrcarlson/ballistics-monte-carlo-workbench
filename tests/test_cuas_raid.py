"""Verification tests for the Unified Dynamic 3D Cooperative Multi-Turret Raid Simulator."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ammo_mc.api import app
from ammo_mc.cli import app as cli_app
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.terrain_dem import build_preset_terrain_dem
from ammo_mc.vis_cuas_raid import (
    create_raid_timeline_and_heat_figure,
    create_unified_tactical_raid_3d_figure,
)


def test_unified_raid_deterministic_and_physics_scaling() -> None:
    r1 = simulate_unified_tactical_raid_mc(n_drones=8, fiber_optic_fraction=0.25, n_mc_trials=10, seed=77)
    r2 = simulate_unified_tactical_raid_mc(n_drones=8, fiber_optic_fraction=0.25, n_mc_trials=10, seed=77)
    assert r1["summary"] == r2["summary"]
    assert len(r1["frames"]) >= 10
    assert len(r1["timeline_series"]) >= 10
    assert len(r1["turret_stats"]) == 4
    assert r1["summary"]["peak_barrel_temp_c"] >= 22.0

    # Fiber-optic threats blind passive RF-only sensors (0 detections when sensor is ONLY rf_c2_df)
    rf_only_fo = simulate_unified_tactical_raid_mc(
        sensors=["rf_c2_df"],
        n_drones=6,
        fiber_optic_fraction=1.0,
        n_mc_trials=4,
        seed=42,
    )
    assert rf_only_fo["summary"]["mean_kill_rate_pct"] == 0.0
    assert rf_only_fo["summary"]["mean_leakers"] == 6.0


def test_unified_raid_with_dem_figures_api_and_cli() -> None:
    dem = build_preset_terrain_dem("rolling_hills_fob")
    res = simulate_unified_tactical_raid_mc(terrain_dem=dem, n_drones=8, n_mc_trials=8, seed=42)
    f3 = create_unified_tactical_raid_3d_figure(res, terrain_dem=dem)
    f2 = create_raid_timeline_and_heat_figure(res)
    assert len(f3.frames) == len(res["frames"])
    assert len(f2.data) >= 4

    client = TestClient(app)
    api_r = client.post(
        "/cuas/unified-raid",
        json={"n_drones": 6, "n_mc_trials": 5, "terrain_preset": "rolling_hills_fob"},
    )
    assert api_r.status_code == 200
    assert "mean_kill_rate_pct" in api_r.json()["summary"]

    runner = CliRunner()
    cli_r = runner.invoke(cli_app, ["cuas", "unified-raid", "--drones", "6", "--trials", "4", "--json"])
    assert cli_r.exit_code == 0
    assert json.loads(cli_r.output)["summary"]["n_drones"] == 6
