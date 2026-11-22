"""Test suite for upgrades:
1. Artillery Battery Sheafs, MRSI / TOT & Salvo Optimization (battery.py)
2. Modern Multi-Layer Armor Defeat, Tandem Jets & ERA Mechanics (advanced_armor.py)
3. Geospatial Terrain, Crest Clearance Masking & KML/CZML Exporters (geospatial.py)
"""

import math
import numpy as np
import plotly.graph_objects as go
import pytest

from ammo_mc.battery import (
    generate_battery_layout,
    calculate_sheaf_aim_points,
    find_elevation_for_range,
    solve_mrsi_trajectory_plan,
    simulate_battery_salvo,
)
from ammo_mc.advanced_armor import (
    calculate_era_flyer_velocity,
    simulate_tandem_shaped_charge_vs_era,
)
from ammo_mc.geospatial import (
    evaluate_terrain_profile,
    audit_crest_clearance_and_dead_space,
    export_trajectory_to_kml,
    export_trajectory_to_czml,
)
from ammo_mc.vis_3d import (
    create_3d_battery_salvo_figure,
    create_3d_armor_defeat_figure,
    create_3d_terrain_clearance_figure,
)


def test_battery_sheafs_and_mrsi_solver() -> None:
    # 1. Test battery layout generator
    guns = generate_battery_layout(n_guns=6, formation="staggered", gun_spacing_m=75.0)
    assert len(guns) == 6
    assert guns[0]["gun_id"] == 1
    assert any(g["x_m"] > 0 for g in guns) and any(g["x_m"] < 0 for g in guns)

    # 2. Test sheaf aim points for different tactical distributions
    aim_conv = calculate_sheaf_aim_points(guns, target_pos_m=(14000.0, 0.0), sheaf_type="converged")
    assert all(p == (14000.0, 0.0) for p in aim_conv)

    aim_lin = calculate_sheaf_aim_points(guns, target_pos_m=(14000.0, 0.0), sheaf_type="linear", linear_length_m=300.0)
    assert len(aim_lin) == 6
    assert aim_lin[0] != aim_lin[-1]

    # 3. Test elevation solver
    qe_low = find_elevation_for_range(target_range_m=12000.0, high_angle=False)
    assert 5.0 < qe_low < 35.0

    qe_high = find_elevation_for_range(target_range_m=12000.0, high_angle=True)
    assert 50.0 < qe_high < 80.0

    # 4. Test MRSI trajectory plan
    mrsi = solve_mrsi_trajectory_plan(target_range_m=12000.0, max_rounds=3)
    assert mrsi["mrsi_feasible"] is True
    assert mrsi["rounds_planned"] >= 2
    assert mrsi["firing_window_s"] > 10.0
    assert mrsi["plan"][0]["tof_s"] > mrsi["plan"][-1]["tof_s"]

    # 5. Test complete battery salvo simulation
    salvo = simulate_battery_salvo(n_guns=4, rounds_per_gun=2, target_pos_m=(12000.0, 0.0), seed=42)
    assert salvo["total_rounds"] == 8
    assert salvo["cep50_m"] < 100.0
    assert len(salvo["rounds"]) == 8
    assert len(salvo["trajectories"]) == 8

    # 6. Test 3D Battery Salvo Figure
    fig = create_3d_battery_salvo_figure(salvo)
    assert isinstance(fig, go.Figure)
    assert any("Battery Gun" in (t.name or "") for t in fig.data)


def test_advanced_armor_and_tandem_era_mechanics() -> None:
    # 1. Flyer plate Gurney velocity
    v_flyer = calculate_era_flyer_velocity(c_exp_kg_m2=4.5, m_plate_kg_m2=16.0, gurney_vel_mps=2400.0)
    assert 800.0 < v_flyer < 1800.0

    # 2. Monolithic vs Tandem against ERA
    res_mono = simulate_tandem_shaped_charge_vs_era(
        warhead_type="monolithic",
        has_era=True,
        era_type="heavy_era",
        obliquity_deg=60.0,
        rha_base_thickness_mm=380.0,
    )
    assert res_mono["era_activated"] is True
    assert res_mono["jet_disruption_pct"] > 50.0
    assert res_mono["perforated"] is False

    res_tandem = simulate_tandem_shaped_charge_vs_era(
        warhead_type="tandem",
        has_era=True,
        era_type="heavy_era",
        obliquity_deg=30.0,
        rha_base_thickness_mm=380.0,
    )
    assert res_tandem["precursor_success"] is True
    assert res_tandem["jet_disruption_pct"] < 15.0
    assert res_tandem["perforated"] is True
    assert res_tandem["behind_armor_spall"]["spall_fragment_count"] > 30
    assert res_tandem["behind_armor_spall"]["crew_incapacitation_prob"] > 0.80

    # 3. Test 3D Armor Defeat Figure
    fig_armor = create_3d_armor_defeat_figure(res_tandem)
    assert isinstance(fig_armor, go.Figure)
    assert any("RHA Backing" in (t.name or "") for t in fig_armor.data)


def test_geospatial_crest_clearance_and_kml_czml() -> None:
    # 1. Terrain profile evaluation
    z_crest = evaluate_terrain_profile(6500.0, ridge_pos_m=6500.0, ridge_height_m=620.0, base_elev_m=150.0)
    assert z_crest > 700.0

    # 2. Crest clearance audit
    audit = audit_crest_clearance_and_dead_space(
        gun_elev_m=150.0,
        target_dist_m=14000.0,
        target_elev_m=90.0,
        ridge_pos_m=6500.0,
        ridge_height_m=620.0,
        required_clearance_m=50.0,
    )
    assert audit["crest_x_m"] == 6500.0
    assert audit["dead_space_extent_m"] > 500.0
    assert audit["clearance_status"].startswith("✅")

    # 3. OGC KML Generation
    kml = export_trajectory_to_kml(
        trajectory_x_m=[0.0, 5000.0, 12000.0],
        trajectory_y_m=[0.0, 30.0, 85.0],
        trajectory_z_m=[0.0, 2400.0, 0.0],
        origin_lat=34.0522,
        origin_lon=-118.2437,
        name="155mm Howitzer Trajectory",
        cep50_m=35.0,
    )
    assert "<kml xmlns=" in kml
    assert "<LineString>" in kml
    assert "<Polygon>" in kml
    assert "-118.2437" in kml

    # 4. Cesium CZML Generation
    czml = export_trajectory_to_czml(
        trajectory_x_m=[0.0, 5000.0, 12000.0],
        trajectory_y_m=[0.0, 30.0, 85.0],
        trajectory_z_m=[0.0, 2400.0, 0.0],
        time_s=[0.0, 22.0, 48.0],
        origin_lat=34.0522,
        origin_lon=-118.2437,
    )
    assert "cartographicDegrees" in czml
    assert "projectile_entity" in czml

    # 5. Test 3D Terrain Clearance Figure
    fig_terr = create_3d_terrain_clearance_figure(audit, [0.0, 6500.0, 14000.0], [0.0, 0.0, 0.0], [150.0, 1100.0, 90.0])
    assert isinstance(fig_terr, go.Figure)
    assert any("Mountain Ridge" in (t.name or "") for t in fig_terr.data)
