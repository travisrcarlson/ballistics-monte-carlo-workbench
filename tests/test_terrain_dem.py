"""Verification tests for Digital Elevation Model (DEM) Terrain LOS Masking & Slope Fallout."""

from __future__ import annotations

import numpy as np

from ammo_mc.cuas_mission import simulate_multiturret_site_defense_mc
from ammo_mc.terrain_dem import (
    TerrainDEMGrid,
    build_preset_terrain_dem,
    compute_terrain_adjusted_fallout_sdz,
    compute_terrain_mask_horizon,
    get_terrain_dem_presets,
    parse_dem_csv,
    parse_esri_ascii_grid,
    segment_crosses_terrain_dem,
)
from ammo_mc.vis_cuas_mission import create_terrain_dem_site_3d_figure


def test_terrain_dem_presets_and_los_masking() -> None:
    presets = get_terrain_dem_presets()
    assert {"flat_airfield", "rolling_hills_fob", "ridge_valley_corridor", "coastal_bluff"}.issubset(presets.keys())

    ridge_dem = build_preset_terrain_dem("ridge_valley_corridor")
    st = ridge_dem.stats()
    assert st["relief_m"] > 60.0

    # Ridge sits between compound (~28,28, alt~6m) and low target (~55,55, alt~15m)
    assert segment_crosses_terrain_dem((20.0, 20.0, 6.0), (65.0, 65.0, 15.0), ridge_dem) is True
    # High-altitude target well above crest clears LOS
    assert segment_crosses_terrain_dem((20.0, 20.0, 6.0), (65.0, 65.0, 90.0), ridge_dem) is False

    horizon = compute_terrain_mask_horizon(0.0, 0.0, 4.0, ridge_dem)
    assert len(horizon) == 72
    assert max(r["mask_elev_deg"] for r in horizon) > 10.0


def test_terrain_dem_csv_ascii_parsers_and_slope_fallout() -> None:
    # CSV parser (slope ramp rising +z along +x)
    csv_rows = ["x_m,y_m,z_m"]
    for xv in np.linspace(-500.0, 500.0, 11):
        for yv in np.linspace(-500.0, 500.0, 11):
            zv = 0.08 * xv  # upslope in +x (az=0), downslope in -x (az=180)
            csv_rows.append(f"{xv:.1f},{yv:.1f},{zv:.2f}")
    slope_dem = parse_dem_csv("\n".join(csv_rows))

    upslope = compute_terrain_adjusted_fallout_sdz(0.0, 0.0, 3.5, firing_az_deg=0.0, firing_elev_deg=30.0, dem=slope_dem)
    downslope = compute_terrain_adjusted_fallout_sdz(0.0, 0.0, 3.5, firing_az_deg=180.0, firing_elev_deg=30.0, dem=slope_dem)
    assert upslope["terrain_fallout_range_m"] < upslope["flat_fallout_range_m"]
    assert downslope["terrain_fallout_range_m"] > downslope["flat_fallout_range_m"]

    # ESRI ASCII grid parser
    asc_txt = (
        "ncols 4\nnrows 4\nxllcorner -150\nyllcorner -150\ncellsize 100\nNODATA_value -9999\n"
        "10 10 20 20\n"
        "5 15 -9999 25\n"
        "0 5 10 15\n"
        "0 0 5 5\n"
    )
    asc_dem = parse_esri_ascii_grid(asc_txt)
    assert asc_dem.z_grid_m.shape == (4, 4)
    assert float(asc_dem.elevation_at(-100.0, 100.0)) >= 5.0

    # Serialization roundtrip
    rt = TerrainDEMGrid.from_dict(asc_dem.to_dict())
    assert rt.stats()["relief_m"] == asc_dem.stats()["relief_m"]


def test_site_defense_with_dem_and_3d_figure() -> None:
    ridge_dem = build_preset_terrain_dem("ridge_valley_corridor")
    res = simulate_multiturret_site_defense_mc(
        site_preset="fob_compound_4_turret",
        terrain_dem=ridge_dem,
        threat_altitude_m=15.0,
        n_raid_trials=20,
        seed=42,
    )
    sm = res["summary"]
    assert sm["terrain_masked_pct"] > 0.0
    assert sm["terrain_fallout_min_m"] < sm["flat_fallout_m"] < sm["terrain_fallout_max_m"]

    fig = create_terrain_dem_site_3d_figure(res, ridge_dem)
    assert len(fig.data) >= 4
