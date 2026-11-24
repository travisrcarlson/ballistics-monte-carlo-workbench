"""Verification tests for 12-gauge FPV Counter-Drone Shot-Cloud Modeling Specification."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient
import plotly.graph_objects as go

from ammo_mc.aero_sphere import sphere_drag_mach_only
from ammo_mc.api import app
from ammo_mc.cuas_hardware import evaluate_muzzle_plate_fea, export_muzzle_plate_dxf, export_muzzle_plate_stl
from ammo_mc.literature import search_literature
from ammo_mc.small_arms import (
    PELLET_MATERIAL_DENSITIES_G_CM3,
    build_custom_drone_spec,
    build_custom_shell_spec,
    compute_annex_c_analytical_pk,
    compute_drone_projected_geometry,
    compute_shotgun_temperature_effects,
    evaluate_drone_impact,
    generate_multishotgun_array_layout,
    get_drone_target_specs,
    get_shotgun_presets,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.vis_3d import (
    create_2d_antidrone_intercept_figure,
    create_3d_antidrone_defense_figure,
    create_3d_pattern_vs_distance_figure,
)


def test_single_square_and_quincunx_layouts_and_cad() -> None:
    b_single = generate_multishotgun_array_layout("single")
    assert len(b_single) == 1
    assert b_single[0]["y_m"] == 0.0 and b_single[0]["z_m"] == 0.0

    b_sq = generate_multishotgun_array_layout("square", spacing_y_m=0.30, spacing_z_m=0.30)
    assert len(b_sq) == 4
    ys_sq = sorted(set(round(b["y_m"], 4) for b in b_sq))
    zs_sq = sorted(set(round(b["z_m"], 4) for b in b_sq))
    assert ys_sq == [-0.15, 0.15] and zs_sq == [-0.15, 0.15]

    b_qn = generate_multishotgun_array_layout(
        "quincunx",
        mount_radius_m=0.30,
        default_ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
        inner_ammunition_preset="SPARTAN_33_HP_SPHERO_4T",
        default_choke="Modified",
        inner_choke="Full",
    )
    assert len(b_qn) == 5
    assert b_qn[0]["y_m"] == 0.0 and b_qn[0]["z_m"] == 0.0
    assert b_qn[0]["ammunition_preset"] == "SPARTAN_33_HP_SPHERO_4T"
    # Tungsten load must enforce Cylinder choke even if Full was passed
    assert b_qn[0]["choke"] == "Cylinder" and b_qn[0]["choke_enforced_cylinder"] is True
    for corner in b_qn[1:]:
        assert abs(math.hypot(corner["y_m"], corner["z_m"]) - 0.30) < 1e-3
        assert corner["ammunition_preset"] == "ALDA_BUCKSHOT_12_76_COPPER_32G"
        assert corner["choke"] == "Modified"

    # CAD / FEA compatibility
    assert evaluate_muzzle_plate_fea(b_qn, layout_type="quincunx")["overall_safety_factor"] > 0.0
    assert "CIRCLE" in export_muzzle_plate_dxf(b_qn, layout_type="quincunx")
    stl = export_muzzle_plate_stl(b_sq, layout_type="square")
    assert stl.startswith("solid ") and "endsolid" in stl


def test_standard_shotshells_densities_mixed_cost_and_tungsten_choke() -> None:
    presets = get_shotgun_presets()
    expected_keys = [
        "ALDA_BUCKSHOT_12_76_COPPER_32G",
        "ALDA_DOUBLE_00_HP_N2_0",
        "ALDA_0_80_HP_N2",
        "SPARTAN_43_HP_1276_SPHERO_3T",
        "SPARTAN_33_HP_SPHERO_4T",
    ]
    for k in expected_keys:
        assert k in presets

    assert presets["ALDA_BUCKSHOT_12_76_COPPER_32G"]["pellets_per_shot"] == 32
    assert presets["ALDA_BUCKSHOT_12_76_COPPER_32G"]["pellet_diameter_mm"] == 6.20
    assert presets["ALDA_BUCKSHOT_12_76_COPPER_32G"]["unit_price_eur"] == 1.97
    assert presets["SPARTAN_33_HP_SPHERO_4T"]["pellets_per_shot"] == 210
    assert presets["SPARTAN_33_HP_SPHERO_4T"]["pellet_diameter_mm"] == 3.25
    assert presets["SPARTAN_33_HP_SPHERO_4T"]["unit_price_eur"] == 7.50
    assert presets["SPARTAN_33_HP_SPHERO_4T"]["choke_restriction"] == "Cylinder"

    assert PELLET_MATERIAL_DENSITIES_G_CM3["Sweet Copper (9.0 g/cm³)"] == 9.00
    assert PELLET_MATERIAL_DENSITIES_G_CM3["Hardened Lead (11.3 g/cm³)"] == 11.30
    assert PELLET_MATERIAL_DENSITIES_G_CM3["Steel (7.85 g/cm³)"] == 7.85
    assert PELLET_MATERIAL_DENSITIES_G_CM3["Tungsten (18.2 g/cm³)"] == 18.20

    # Custom tungsten enforces Cylinder choke
    cust_w = build_custom_shell_spec("Custom Tungsten 3T", "Tungsten (18.2 g/cm³)", 3.55, 100, default_choke="Full", unit_price_eur=9.90)
    assert cust_w["default_choke"] == "Cylinder"
    assert cust_w["choke_restriction"] == "Cylinder"
    assert cust_w["unit_price_eur"] == 9.90

    # Mixed 5-barrel Quincunx: 4x ALDA 6.2mm Copper (€1.97) + 1x SPARTAN 4T Tungsten (€7.50) = €15.38
    sim = simulate_anti_drone_multishotgun_mc(
        layout_type="quincunx",
        mount_radius_m=0.30,
        ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
        inner_ammunition_preset="SPARTAN_33_HP_SPHERO_4T",
        default_choke="Full",
        inner_choke="Full",
        drone_model="fpv_14in_36cm_pg7m",
        drone_velocity_mps=0.0,
        target_distance_m=50.0,
        n_volleys=12,
        seed=42,
    )
    sm = sim["summary"]
    assert sm["total_pellets_per_volley"] == 4 * 32 + 210
    assert abs(sm["cost_per_volley_eur"] - (4 * 1.97 + 7.50)) < 1e-2
    assert sim["barrels"][0]["choke"] == "Cylinder"
    assert 0.0 <= sm["kill_probability_pct"] <= sm["hit_probability_pct"] <= 100.0
    assert sm["nominal_pellet_impulse_ns"] > 0.0
    assert sm["nominal_pellet_energy_j"] > 0.0


def test_fpv_drones_top_battery_and_underslung_pg7m_warhead() -> None:
    specs = get_drone_target_specs()
    d7 = specs["fpv_7in_18cm_pg7m"]
    d14 = specs["fpv_14in_36cm_pg7m"]
    assert d7["span_m"] == 0.18 and d7["battery_offset_z_m"] > 0 and d7["warhead_offset_z_m"] < 0
    assert d14["span_m"] == 0.36 and d14["battery_offset_z_m"] > 0 and d14["warhead_offset_z_m"] < 0
    assert d14["has_warhead"] is True and d14["warhead_type"] == "PG-7M"

    geom_front = compute_drone_projected_geometry(d14, aspect_mode="head_on_frontal")
    geom_side = compute_drone_projected_geometry(d14, aspect_mode="flank_side")
    geom_top = compute_drone_projected_geometry(d14, aspect_mode="planform_top")
    assert len(geom_front["warhead_polygon"]) >= 3
    assert geom_side["warhead_area_cm2"] > geom_front["warhead_area_cm2"]
    assert (geom_side["fuselage_area_cm2"] + geom_side["battery_area_cm2"] + geom_side["warhead_area_cm2"]) > (
        geom_front["fuselage_area_cm2"] + geom_front["battery_area_cm2"] + geom_front["warhead_area_cm2"]
    )
    assert geom_top["presented_area_cm2"] > geom_front["presented_area_cm2"]

    # Pellet impact directly on underslung PG-7M warhead triggers warhead_pg7m lethal check
    wh_y = float(sum(p[0] for p in geom_side["warhead_polygon"]) / len(geom_side["warhead_polygon"]))
    wh_z = float(sum(p[1] for p in geom_side["warhead_polygon"]) / len(geom_side["warhead_polygon"]))
    hit = evaluate_drone_impact(wh_y, wh_z, d14, striking_energy_j=30.0, aspect_geometry=geom_side)
    assert hit["is_hit"] is True and hit["component"] == "warhead_pg7m" and hit["is_lethal"] is True

    cust = build_custom_drone_spec(span_m=0.22, battery_on_top=True, has_warhead=True)
    assert cust["battery_offset_z_m"] > 0 and cust["warhead_offset_z_m"] < 0 and cust["has_warhead"] is True


def test_mach_drag_compton_headwind_and_annex_a_temperature() -> None:
    cd_sub = sphere_drag_mach_only(0.25)
    cd_trans = sphere_drag_mach_only(1.20)
    assert abs(cd_sub - 0.47) < 1e-6
    assert cd_trans > 0.85

    hits = search_literature(query="Compton")
    assert any(r["source_id"] == "S81" for r in hits)

    cold = compute_shotgun_temperature_effects(-10.0, "Hardened Lead", "single_base")
    std = compute_shotgun_temperature_effects(15.0, "Hardened Lead", "double_base")
    hot = compute_shotgun_temperature_effects(38.0, "Hardened Lead", "single_base")
    assert cold["mv_delta_mps"] < 0.0 and cold["air_density_kg_m3"] > std["air_density_kg_m3"]
    assert cold["temp_spread_mult"] < 1.0
    assert hot["mv_delta_mps"] > 0.0 and hot["air_density_kg_m3"] < std["air_density_kg_m3"]
    assert hot["temp_spread_mult"] > 1.0

    # Headwind slows pellets -> lower striking energy & impulse vs tailwind
    sim_hw = simulate_anti_drone_multishotgun_mc(
        layout_type="single",
        ammunition_preset="ALDA_0_80_HP_N2",
        pellets_per_shot=115,
        target_distance_m=50.0,
        drone_model="fpv_14in_36cm_pg7m",
        drone_velocity_mps=0.0,
        headwind_mps=15.0,
        drag_model="mach_only_smooth_sphere",
        n_volleys=10,
        seed=9,
    )
    sim_tw = simulate_anti_drone_multishotgun_mc(
        layout_type="single",
        ammunition_preset="ALDA_0_80_HP_N2",
        pellets_per_shot=115,
        target_distance_m=50.0,
        drone_model="fpv_14in_36cm_pg7m",
        drone_velocity_mps=0.0,
        headwind_mps=-15.0,
        drag_model="mach_only_smooth_sphere",
        n_volleys=10,
        seed=9,
    )
    assert sim_hw["summary"]["mean_v_impact_nominal_mps"] < sim_tw["summary"]["mean_v_impact_nominal_mps"]
    assert sim_hw["summary"]["nominal_pellet_impulse_ns"] < sim_tw["summary"]["nominal_pellet_impulse_ns"]
    assert sim_hw["summary"]["nominal_gravity_drop_m"] > 0.0


def test_annex_c_numerical_table_and_2d_3d_visualizations() -> None:
    # Worked Numerical Example from Annex C: N=170, Av=0.025, theta_p=0.0067, theta_aim=0.015
    cases = [
        (20.0, 0.89, 99.6, 100.0),
        (30.0, 0.79, 89.0, 99.9),
        (40.0, 0.70, 66.8, 96.3),
        (50.0, 0.62, 46.5, 84.7),
        (75.0, 0.46, 18.6, 46.1),
        (100.0, 0.34, 8.2, 22.7),
    ]
    for d_m, pk_h, exp_pk1, exp_pk3 in cases:
        res = compute_annex_c_analytical_pk(d_m, 170, 0.025, 0.0067, 0.015, pk_h, n_shots=3)
        assert abs(res["pk_1_pct"] - exp_pk1) < 1.2
        assert abs(res["pk_n_pct"] - exp_pk3) < 1.2

    slices = [
        simulate_anti_drone_multishotgun_mc(
            layout_type="square",
            spacing_y_m=0.30,
            spacing_z_m=0.30,
            ammunition_preset="ALDA_BUCKSHOT_12_76_COPPER_32G",
            target_distance_m=r_m,
            drone_model="fpv_14in_36cm_pg7m",
            drone_velocity_mps=0.0,
            n_volleys=8,
            seed=42,
        )
        for r_m in (25.0, 50.0, 75.0)
    ]
    assert isinstance(create_2d_antidrone_intercept_figure(slices[1]), go.Figure)
    assert isinstance(create_3d_antidrone_defense_figure(slices[1]), go.Figure)
    assert isinstance(create_3d_pattern_vs_distance_figure(slices), go.Figure)

    client = TestClient(app)
    r_arr = client.post(
        "/cuas/array",
        json={
            "layout_type": "quincunx",
            "mount_radius_m": 0.30,
            "ammunition_preset": "ALDA_BUCKSHOT_12_76_COPPER_32G",
            "inner_ammunition_preset": "SPARTAN_33_HP_SPHERO_4T",
            "drone_model": "fpv_14in_36cm_pg7m",
            "target_distance_m": 50.0,
            "drone_velocity_mps": 0.0,
            "ambient_temp_c": 5.0,
            "headwind_mps": 5.0,
            "drag_model": "mach_only_smooth_sphere",
            "n_volleys": 8,
        },
    )
    assert r_arr.status_code == 200
    assert r_arr.json()["summary"]["cost_per_volley_eur"] > 15.0

    r_ac = client.post("/cuas/annex-c-pk", json={"distance_m": 50.0, "n_pellets": 170, "pk_given_hit": 0.62, "n_shots": 3})
    assert r_ac.status_code == 200 and 44.0 <= r_ac.json()["pk_1_pct"] <= 48.0
