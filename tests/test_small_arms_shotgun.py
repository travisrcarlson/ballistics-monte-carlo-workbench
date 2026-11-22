"""Unit and integration test suite for Small Arms, Shotgun Patterning, and Multi-Barrel Dispersion."""

from __future__ import annotations

import math
import pytest

from ammo_mc.small_arms import (
    CHOKE_DEFINITIONS,
    DRONE_TARGET_SPECS,
    MULTIBARREL_PRESETS,
    PELLET_SPECIFICATIONS,
    SHOTGUN_PRESETS,
    STANDARD_BARREL_LENGTHS,
    benchmark_drone_aspect_angles_mc,
    benchmark_multibarrel_aspect_angles_mc,
    check_target_hit,
    compare_anti_drone_layouts,
    compare_barrel_lengths_dispersion,
    compute_drone_projected_geometry,
    configure_array_pointing_vectors,
    evaluate_drone_impact,
    evaluate_multibarrel_dispersion_vs_distance,
    evaluate_shotgun_dispersion_vs_distance,
    generate_multishotgun_array_layout,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
    get_multibarrel_presets,
    get_pellet_specifications,
    get_shotgun_presets,
    get_standard_barrel_lengths,
    simulate_anti_drone_multishotgun_mc,
    simulate_multibarrel_burst_mc,
    simulate_shotgun_pattern_mc,
)
from ammo_mc.vis_3d import (
    create_2d_antidrone_intercept_figure,
    create_2d_barrel_length_comparison_figure,
    create_2d_multi_distance_shotgun_comparison_figure,
    create_2d_multibarrel_target_figure,
    create_2d_target_pattern_figure,
    create_3d_antidrone_defense_figure,
    create_3d_multi_distance_multibarrel_figure,
    create_3d_multi_distance_shotgun_figure,
    create_3d_multibarrel_dispersion_figure,
    create_3d_shotgun_plume_figure,
    create_antidrone_layout_comparison_figure,
    create_barrel_length_diagnostic_curves,
    create_drone_aspect_radar_figure,
    create_multibarrel_aspect_radar_figure,
    create_range_curves_multibarrel_figure,
    create_range_curves_shotgun_figure,
)


def test_shotgun_presets_and_catalogs():
    """Verify standard shotgun presets, choke constrictions, and pellet specifications."""
    sg_presets = get_shotgun_presets()
    assert "12G_00_BUCK_TACTICAL" in sg_presets
    assert "12G_NO75_BIRD_TRAP" in sg_presets
    assert len(sg_presets) >= 5

    chokes = get_choke_definitions()
    assert "Cylinder" in chokes
    assert "Modified" in chokes
    assert "Full" in chokes
    assert chokes["Cylinder"]["constriction_mm"] == 0.0
    assert chokes["Full"]["constriction_mm"] > chokes["Modified"]["constriction_mm"]

    pellets = get_pellet_specifications()
    assert "00 Buckshot" in pellets
    assert pellets["00 Buckshot"]["diameter_mm"] == pytest.approx(8.38, abs=0.01)


def test_target_silhouette_detection():
    """Verify target silhouette hit detection logic across humanoid, clay, and drone shapes."""
    # 30-inch circle (radius 0.381m)
    assert check_target_hit(0.0, 0.0, target_type="30_inch_circle") is True
    assert check_target_hit(0.20, 0.20, target_type="30_inch_circle") is True
    assert check_target_hit(0.50, 0.0, target_type="30_inch_circle") is False

    # 10-inch circle (radius 0.127m)
    assert check_target_hit(0.0, 0.0, target_type="10_inch_circle") is True
    assert check_target_hit(0.10, 0.0, target_type="10_inch_circle") is True
    assert check_target_hit(0.20, 0.0, target_type="10_inch_circle") is False

    # E-type humanoid silhouette
    assert check_target_hit(0.0, 0.0, target_type="e_type_silhouette") is True  # torso center
    assert check_target_hit(0.0, 0.35, target_type="e_type_silhouette") is True  # head
    assert check_target_hit(0.40, 0.0, target_type="e_type_silhouette") is False  # outside shoulder

    # Clay pigeon
    assert check_target_hit(0.0, 0.0, target_type="clay_pigeon") is True
    assert check_target_hit(0.10, 0.10, target_type="clay_pigeon") is False


def test_shotgun_pattern_choke_comparison():
    """Verify that tighter chokes (Full) produce tighter patterns and higher 30-in hit percentages than Cylinder."""
    # Run Cylinder choke simulation
    res_cyl = simulate_shotgun_pattern_mc(
        n_shots=15,
        pellets_per_shot=9,
        pellet_diameter_mm=8.38,
        pellet_mass_g=3.49,
        v0_mps=405.0,
        choke_name="Cylinder",
        target_distance_m=36.58,  # 40 yards
        seed=100,
    )

    # Run Full choke simulation
    res_full = simulate_shotgun_pattern_mc(
        n_shots=15,
        pellets_per_shot=9,
        pellet_diameter_mm=8.38,
        pellet_mass_g=3.49,
        v0_mps=405.0,
        choke_name="Full",
        target_distance_m=36.58,  # 40 yards
        seed=100,
    )

    assert res_cyl["total_pellets_fired"] == 135
    assert res_full["total_pellets_fired"] == 135

    # Full choke should have significantly higher 30-inch circle hit % than Cylinder
    assert res_full["hits_30in_pct"] > res_cyl["hits_30in_pct"]

    # Full choke should have smaller CEP50 than Cylinder
    assert res_full["cep_50_m"] < res_cyl["cep_50_m"]

    # Kinetic energy and velocity should be positive
    assert res_cyl["mean_v_impact_mps"] > 300.0
    assert res_cyl["mean_pellet_energy_j"] > 150.0

    # Shot string length should be physically realistic (0.5 to 4.0m)
    assert 0.4 <= res_cyl["mean_shot_string_length_m"] <= 4.0


def test_shot_stringing_and_moving_target():
    """Verify moving target interaction with longitudinal shot stringing and lead compensation."""
    res_static = simulate_shotgun_pattern_mc(
        n_shots=10,
        pellets_per_shot=350,  # #7.5 birdshot
        pellet_diameter_mm=2.41,
        pellet_mass_g=0.081,
        v0_mps=395.0,
        choke_name="Modified",
        target_distance_m=25.0,
        target_type="clay_pigeon",
        moving_target_velocity_mps=0.0,
        seed=42,
    )

    # Without lead compensation (shooting directly at target), target escapes pattern
    res_no_lead = simulate_shotgun_pattern_mc(
        n_shots=10,
        pellets_per_shot=350,
        pellet_diameter_mm=2.41,
        pellet_mass_g=0.081,
        v0_mps=395.0,
        choke_name="Modified",
        target_distance_m=25.0,
        target_type="clay_pigeon",
        moving_target_velocity_mps=20.0,  # 20 m/s crossing speed
        moving_target_crossing_angle_deg=90.0,
        lead_compensation_pct=0.0,
        seed=42,
    )

    # Static clay target should receive more hits than fast crossing clay target without aim lead
    assert res_static["hits_target_count"] > res_no_lead["hits_target_count"]
    assert res_no_lead["hits_target_count"] == 0
    assert len(res_static["sample_pellet_records"]) > 0


def test_multibarrel_crossover_geometry():
    """Verify multi-barrel Side-by-Side (SxS) convergence and crossover behavior."""
    crossover_m = 35.0

    # Case 1: Pre-crossover (R = 15m) -> Converging
    res_pre = simulate_multibarrel_burst_mc(
        system_type="sxs_double",
        n_barrels=2,
        bore_spacing_y_mm=28.0,
        crossover_range_m=crossover_m,
        burst_rounds=20,
        target_range_m=15.0,
        seed=42,
    )
    assert res_pre["crossover_status"] == "Converging (Pre-Crossover)"
    assert res_pre["crossover_ratio"] < 1.0

    # Case 2: Exact Crossover (R = 35m) -> Focus
    res_cross = simulate_multibarrel_burst_mc(
        system_type="sxs_double",
        n_barrels=2,
        bore_spacing_y_mm=28.0,
        crossover_range_m=crossover_m,
        burst_rounds=20,
        target_range_m=35.0,
        seed=42,
    )
    assert res_cross["crossover_status"] == "Exact Crossover Focus"

    # Case 3: Post-crossover (R = 70m) -> Diverging
    res_post = simulate_multibarrel_burst_mc(
        system_type="sxs_double",
        n_barrels=2,
        bore_spacing_y_mm=28.0,
        crossover_range_m=crossover_m,
        burst_rounds=20,
        target_range_m=70.0,
        seed=42,
    )
    assert res_post["crossover_status"] == "Diverging (Post-Crossover)"
    assert res_post["crossover_ratio"] > 1.0

    # Cluster centroids for barrel 1 and barrel 2 should flip signs between pre and post crossover
    b1_pre_y = res_pre["barrel_stats"][0]["centroid_y_m"]
    b1_post_y = res_post["barrel_stats"][0]["centroid_y_m"]
    # Barrel 1 starts at negative Y (-0.014m), at pre-crossover it is still negative, at post-crossover it has crossed to positive Y
    assert b1_pre_y < 0.0
    assert b1_post_y > 0.0


def test_rotary_gatling_minigun_burst():
    """Verify 6-barrel rotary Gatling Minigun burst dispersion and per-barrel diagnostics."""
    res_gat = simulate_multibarrel_burst_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        rotor_radius_mm=45.0,
        crossover_range_m=400.0,
        burst_rounds=60,
        target_range_m=400.0,
        cyclic_rate_rpm=3000.0,
        inherent_dispersion_mrad=2.0,
        barrel_pointing_bias_mrad=1.0,
        thermal_drift_mrad_per_100rd=0.5,
        target_type="e_type_silhouette",
        seed=42,
    )

    assert res_gat["n_barrels"] == 6
    assert len(res_gat["barrel_stats"]) == 6
    assert res_gat["burst_rounds"] == 60
    assert res_gat["salvo_cep_m"] > 0.0
    assert res_gat["salvo_hit_pct"] > 0.0

    for b in res_gat["barrel_stats"]:
        assert b["rounds_fired"] == 10
        assert b["cep_m"] > 0.0


def test_small_arms_figures_generation():
    """Verify that 3D shotgun plume, 2D target board, and 3D multi-barrel figures generate cleanly."""
    sg = simulate_shotgun_pattern_mc(n_shots=3, pellets_per_shot=9, target_distance_m=36.58)
    fig_plume = create_3d_shotgun_plume_figure(sg)
    assert fig_plume is not None
    assert len(fig_plume.data) > 0

    fig_2d = create_2d_target_pattern_figure(sg)
    assert fig_2d is not None
    assert len(fig_2d.data) >= 4

    mb = simulate_multibarrel_burst_mc(system_type="rotary_gatling", n_barrels=6, burst_rounds=30, target_range_m=300.0)
    fig_mb = create_3d_multibarrel_dispersion_figure(mb)
    assert fig_mb is not None
    assert len(fig_mb.data) > 0


def test_multi_distance_shotgun_sweep():
    """Verify shotgun dispersion sweep across multiple distances (10m to 55m)."""
    sweep = evaluate_shotgun_dispersion_vs_distance(distances_m=[10.0, 25.0, 40.0, 55.0])
    assert len(sweep["range_records"]) == 4
    assert len(sweep["station_snapshots"]) == 4

    # 30-inch containment should decay with distance
    p_close = sweep["range_records"][0]["hits_30in_pct"]
    p_far = sweep["range_records"][-1]["hits_30in_pct"]
    assert p_close > p_far

    # Velocity and energy should decay with distance
    v_close = sweep["range_records"][0]["mean_v_impact_mps"]
    v_far = sweep["range_records"][-1]["mean_v_impact_mps"]
    assert v_close > v_far

    # Pattern diameter should grow with distance
    d_close = sweep["range_records"][0]["pattern_diameter_90_m"]
    d_far = sweep["range_records"][-1]["pattern_diameter_90_m"]
    assert d_far > d_close


def test_multi_distance_multibarrel_sweep():
    """Verify multi-barrel convergence and divergence progression across distance stations."""
    crossover_m = 40.0
    mb_sweep = evaluate_multibarrel_dispersion_vs_distance(
        system_type="sxs_double",
        crossover_range_m=crossover_m,
        distances_m=[20.0, 40.0, 80.0],
    )
    records = mb_sweep["range_records"]
    assert len(records) == 3

    # Status at 20m should be Converging
    assert records[0]["crossover_status"] == "Converging (Pre-Crossover)"
    # Status at 40m should be Exact Crossover Focus
    assert records[1]["crossover_status"] == "Exact Crossover Focus"
    # Status at 80m should be Diverging
    assert records[2]["crossover_status"] == "Diverging (Post-Crossover)"

    # Theoretical spacing should be minimum at crossover
    assert records[1]["inter_barrel_spacing_m"] == pytest.approx(0.0, abs=0.001)
    assert records[0]["inter_barrel_spacing_m"] > 0.0
    assert records[2]["inter_barrel_spacing_m"] > 0.0


def test_barrel_length_dispersion_comparison():
    """Verify barrel length scaling: velocity gain and gas wash penalty for short barrels."""
    b_comp = compare_barrel_lengths_dispersion(
        barrel_lengths_in=[14.0, 18.5, 28.0, 32.0],
        base_v0_mps=405.0,
        reference_length_in=28.0,
    )
    b_recs = b_comp["barrel_records"]
    assert len(b_recs) == 4

    # Muzzle velocity must increase with barrel length
    v0_14 = b_recs[0]["v0_mps"]
    v0_28 = b_recs[2]["v0_mps"]
    v0_32 = b_recs[3]["v0_mps"]
    assert v0_14 < v0_28 < v0_32

    # Gas wash factor must be highest for 14-inch barrel
    assert b_recs[0]["gas_wash_factor"] > b_recs[2]["gas_wash_factor"]
    assert b_recs[2]["gas_wash_factor"] == 1.0


def test_multi_distance_and_barrel_length_figures():
    """Verify that multi-distance 3D/2D and barrel-length comparison figures generate without error."""
    sg_sweep = evaluate_shotgun_dispersion_vs_distance(distances_m=[15.0, 30.0, 45.0])
    fig_sg_3d = create_3d_multi_distance_shotgun_figure(sg_sweep)
    assert len(fig_sg_3d.data) > 0

    fig_sg_2d = create_2d_multi_distance_shotgun_comparison_figure(sg_sweep)
    assert len(fig_sg_2d.data) > 0

    fig_sg_rc = create_range_curves_shotgun_figure(sg_sweep)
    assert len(fig_sg_rc.data) > 0

    b_comp = compare_barrel_lengths_dispersion(barrel_lengths_in=[14.0, 18.5, 28.0])
    fig_b_2d = create_2d_barrel_length_comparison_figure(b_comp)
    assert len(fig_b_2d.data) > 0

    fig_b_diag = create_barrel_length_diagnostic_curves(b_comp)
    assert len(fig_b_diag.data) > 0

    mb_sweep = evaluate_multibarrel_dispersion_vs_distance(system_type="rotary_gatling", crossover_range_m=400.0, distances_m=[200.0, 400.0, 600.0])
    fig_mb_3d = create_3d_multi_distance_multibarrel_figure(mb_sweep)
    assert len(fig_mb_3d.data) > 0

    fig_mb_rc = create_range_curves_multibarrel_figure(mb_sweep)
    assert len(fig_mb_rc.data) > 0


def test_antidrone_specs_and_layouts():
    """Verify anti-drone target vulnerability catalogs and multi-shotgun array layout generation."""
    specs = get_drone_target_specs()
    assert "fpv_quad_small" in specs
    assert "commercial_quad_medium" in specs
    assert "heavy_octocopter_large" in specs
    assert "fixed_wing_loitering" in specs

    assert specs["fpv_quad_small"]["critical_hits_for_kill"] == 2
    assert specs["commercial_quad_medium"]["critical_hits_for_kill"] == 3

    # Test layout generation across multiple architectures
    ring_6 = generate_multishotgun_array_layout("circular", n_barrels=6, mount_radius_m=0.20)
    assert len(ring_6) == 6
    # All barrels should be approximately at radius 0.20
    for b in ring_6:
        r = math.hypot(b["y_m"], b["z_m"])
        assert r == pytest.approx(0.20, abs=0.01)

    grid_3x2 = generate_multishotgun_array_layout("grid", n_rows=2, n_cols=3, spacing_y_m=0.10, spacing_z_m=0.10)
    assert len(grid_3x2) == 6

    concentric_8 = generate_multishotgun_array_layout("concentric", n_barrels=8, mount_radius_m=0.25)
    assert len(concentric_8) == 8

    honeycomb_7 = generate_multishotgun_array_layout("honeycomb", n_barrels=7, mount_radius_m=0.20)
    assert len(honeycomb_7) >= 7
    # Center barrel should be at (0, 0)
    assert honeycomb_7[0]["y_m"] == 0.0 and honeycomb_7[0]["z_m"] == 0.0

    fan_h = generate_multishotgun_array_layout("fan_horizontal", n_barrels=5, spacing_y_m=0.10)
    assert len(fan_h) == 5
    for b in fan_h:
        assert b["z_m"] == 0.0

    custom_b = generate_multishotgun_array_layout(
        "custom",
        custom_specs=[
            {"barrel_id": "C1", "y_m": -0.15, "z_m": 0.15, "choke": "Full"},
            {"barrel_id": "C2", "y_m": 0.15, "z_m": 0.15, "choke": "Modified"},
        ],
    )
    assert len(custom_b) == 2
    assert custom_b[0]["choke"] == "Full"


def test_antidrone_pointing_and_impact():
    """Verify array aiming convergence, parallel bores, and drone component impact detection."""
    barrels = generate_multishotgun_array_layout("circular", n_barrels=4, mount_radius_m=0.20)

    # Convergent mode at 40m
    conv_b = configure_array_pointing_vectors(barrels, pointing_mode="convergent", intercept_range_m=40.0)
    for b in conv_b:
        # At 40m, aim_yaw = -y0 / 40 * 1000 = -y0 * 25
        assert b["aim_yaw_mrad"] == pytest.approx(-b["y_m"] * 25.0, abs=0.1)
        assert b["aim_pitch_mrad"] == pytest.approx(-b["z_m"] * 25.0, abs=0.1)

    # Parallel mode
    par_b = configure_array_pointing_vectors(barrels, pointing_mode="parallel", intercept_range_m=40.0)
    for b in par_b:
        assert b["aim_yaw_mrad"] == 0.0 and b["aim_pitch_mrad"] == 0.0

    # Impact checks on commercial quad
    spec = get_drone_target_specs()["commercial_quad_medium"]
    # Direct center hit (fuselage core)
    hit_core = evaluate_drone_impact(0.0, 0.0, spec, striking_energy_j=35.0)
    assert hit_core["is_hit"] is True
    assert hit_core["component"] == "fuselage_core"
    assert hit_core["is_lethal"] is True

    # Rotor hit at (0.15, 0.15) with high energy
    hit_rotor = evaluate_drone_impact(0.15, 0.15, spec, striking_energy_j=15.0)
    assert hit_rotor["is_hit"] is True
    assert "rotor" in hit_rotor["component"]
    assert hit_rotor["is_lethal"] is True

    # Rotor hit with insufficient energy (below 10 J)
    hit_weak = evaluate_drone_impact(0.15, 0.15, spec, striking_energy_j=5.0)
    assert hit_weak["is_hit"] is True
    assert hit_weak["is_lethal"] is False

    # Miss outside drone boundary
    miss = evaluate_drone_impact(1.0, 1.0, spec, striking_energy_j=40.0)
    assert miss["is_hit"] is False
    assert miss["component"] == "miss"


def test_antidrone_multishotgun_mc_simulation():
    """Verify Monte Carlo simulation of multi-shotgun anti-drone defense volleys."""
    sim = simulate_anti_drone_multishotgun_mc(
        layout_type="circular",
        n_barrels=6,
        mount_radius_m=0.20,
        pointing_mode="convergent",
        firing_mode="simultaneous_volley",
        ammunition_preset="12G_NO4_BUCK_C_UAS",
        target_distance_m=30.0,
        drone_model="commercial_quad_medium",
        drone_velocity_mps=15.0,
        n_volleys=15,
        seed=42,
    )

    summary = sim["summary"]
    assert summary["n_barrels"] == 6
    assert summary["total_pellets_per_volley"] == 6 * 27
    assert 0.0 <= summary["kill_probability_pct"] <= 100.0
    assert summary["mean_hits_on_drone"] > 0
    assert summary["mean_lethal_hits"] >= 0
    assert len(sim["barrel_stats"]) == 6
    assert len(sim["volley_records"]) == 15
    assert len(sim["sample_scatter"]["y"]) > 0

    # Sequential ripple firing mode
    sim_ripple = simulate_anti_drone_multishotgun_mc(
        layout_type="grid",
        n_rows=2,
        n_cols=3,
        firing_mode="sequential_ripple",
        ripple_delay_ms=10.0,
        target_distance_m=25.0,
        n_volleys=10,
        seed=42,
    )
    assert sim_ripple["summary"]["firing_mode"] == "sequential_ripple"
    assert len(sim_ripple["volley_records"]) == 10


def test_compare_anti_drone_layouts():
    """Verify layout comparison sweeps across ranges."""
    comp = compare_anti_drone_layouts(
        layout_configs=[
            {"name": "4-Barrel Ring", "layout_type": "circular", "n_barrels": 4, "mount_radius_m": 0.18},
            {"name": "2x2 Grid", "layout_type": "grid", "n_rows": 2, "n_cols": 2, "spacing_y_m": 0.12, "spacing_z_m": 0.12},
        ],
        distance_stations_m=[20.0, 35.0, 50.0],
        drone_model="commercial_quad_medium",
        n_volleys=10,
        seed=42,
    )
    assert len(comp["comparison_table"]) == 6  # 2 layouts * 3 distances
    assert "4-Barrel Ring" in comp["layout_curves"]
    assert len(comp["layout_curves"]["4-Barrel Ring"]["pk_pct"]) == 3


def test_antidrone_figures_generation():
    """Verify that all 3D and 2D anti-drone figures generate without error."""
    sim = simulate_anti_drone_multishotgun_mc(
        layout_type="circular",
        n_barrels=6,
        target_distance_m=35.0,
        n_volleys=10,
        seed=42,
    )
    fig_3d = create_3d_antidrone_defense_figure(sim)
    assert len(fig_3d.data) > 0

    fig_2d = create_2d_antidrone_intercept_figure(sim)
    assert len(fig_2d.data) > 0

    comp = compare_anti_drone_layouts(
        distance_stations_m=[20.0, 40.0],
        n_volleys=5,
        seed=42,
    )
    fig_comp = create_antidrone_layout_comparison_figure(comp)
    assert len(fig_comp.data) > 0


def test_drone_aspect_projected_geometry_and_cross_section():
    """Verify 3D drone target cross-section projections and aspect-dependent hit detection."""
    presets = get_drone_aspect_presets()
    assert "head_on_frontal" in presets
    assert "flank_side" in presets
    assert "oblique_quartering" in presets
    assert "tail_rear" in presets
    assert "planform_overhead" in presets

    specs = get_drone_target_specs()
    for drone_key, spec in specs.items():
        geom_head_on = compute_drone_projected_geometry(spec, aspect_mode="head_on_frontal", drone_pitch_deg=15.0)
        geom_planform = compute_drone_projected_geometry(spec, aspect_mode="planform_overhead")

        # Head-on frontal cross-section area must be significantly smaller than top-down planform
        assert geom_head_on["presented_area_m2"] < geom_planform["presented_area_m2"]
        assert geom_head_on["cross_section_ratio_pct"] < 90.0

        # Rotor discs projected edge-on must have finite thickness (not collapse to zero)
        for r_info in geom_head_on["rotors"]:
            assert r_info["minor_axis_m"] >= spec.get("rotor_thickness_m", 0.02) * 0.45

    # Target hit evaluation using 3D projected aspect geometry
    spec_fpv = specs["fpv_quad_small"]
    geom_front = compute_drone_projected_geometry(spec_fpv, aspect_mode="head_on_frontal", drone_pitch_deg=15.0)

    # 1. Fuselage hit at center (0, 0)
    hit_fuse = evaluate_drone_impact(0.0, 0.0, spec_fpv, striking_energy_j=35.0, aspect_geometry=geom_front)
    assert hit_fuse["is_hit"] is True
    assert hit_fuse["component"] in ("fuselage_core", "battery_payload")
    assert hit_fuse["is_lethal"] is True

    # 2. Large miss offset
    hit_miss = evaluate_drone_impact(1.5, 1.5, spec_fpv, striking_energy_j=50.0, aspect_geometry=geom_front)
    assert hit_miss["is_hit"] is False
    assert hit_miss["component"] == "miss"


def test_benchmark_drone_aspect_angles_mc():
    """Verify stochastic Monte Carlo benchmarking across all 6 aspect angle presets."""
    bench = benchmark_drone_aspect_angles_mc(
        layout_type="circular",
        n_barrels=6,
        mount_radius_m=0.20,
        target_distance_m=35.0,
        drone_model="fpv_quad_small",
        n_volleys=12,
        seed=42,
    )
    assert "aspect_results" in bench
    assert "comparison_table" in bench
    assert len(bench["comparison_table"]) == 6

    # Verify all presets are included
    bench_modes = [row["aspect_mode"] for row in bench["comparison_table"]]
    assert "head_on_frontal" in bench_modes
    assert "flank_side" in bench_modes
    assert "planform_top" in bench_modes

    # Frontal cross section must present less area than planform top
    frontal_row = next(r for r in bench["comparison_table"] if r["aspect_mode"] == "head_on_frontal")
    planform_row = next(r for r in bench["comparison_table"] if r["aspect_mode"] == "planform_top")
    assert frontal_row["presented_area_cm2"] < planform_row["presented_area_cm2"]

    # Verify Plotly aspect comparative radar / bar figure
    fig_radar = create_drone_aspect_radar_figure(bench)
    assert len(fig_radar.data) >= 2


def test_multibarrel_drone_aspect_cross_section_burst():
    """Verify multi-barrel Gatling burst against 3D projected drone target in head-on vs planform aspects."""
    # 1. Head-on frontal engagement (narrow cross section)
    res_front = simulate_multibarrel_burst_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        rotor_radius_mm=45.0,
        crossover_range_m=100.0,
        burst_rounds=40,
        target_range_m=100.0,
        cyclic_rate_rpm=3000.0,
        target_type="drone",
        drone_model="commercial_quad_medium",
        aspect_mode="head_on_frontal",
        drone_pitch_deg=15.0,
        drone_yaw_deg=0.0,
        drone_roll_deg=0.0,
        bullet_energy_j=1800.0,
        seed=42,
    )

    assert res_front["target_type"] == "drone"
    assert res_front["aspect_geometry"] is not None
    assert res_front["presented_area_cm2"] is not None
    assert res_front["presented_area_cm2"] > 0
    assert "fuselage_hits" in res_front
    assert "battery_hits" in res_front
    assert "rotor_hits" in res_front
    assert "is_drone_killed" in res_front

    # 2. Planform overhead engagement (broad cross section)
    res_top = simulate_multibarrel_burst_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        rotor_radius_mm=45.0,
        crossover_range_m=100.0,
        burst_rounds=40,
        target_range_m=100.0,
        cyclic_rate_rpm=3000.0,
        target_type="drone",
        drone_model="commercial_quad_medium",
        aspect_mode="planform_overhead",
        drone_pitch_deg=0.0,
        drone_yaw_deg=0.0,
        drone_roll_deg=0.0,
        bullet_energy_j=1800.0,
        seed=42,
    )

    # Planform presented area must be significantly greater than head-on
    assert res_top["presented_area_cm2"] > res_front["presented_area_cm2"]
    assert res_top["cross_section_ratio_pct"] > res_front["cross_section_ratio_pct"]

    # Visualizations
    fig_2d = create_2d_multibarrel_target_figure(res_front)
    assert len(fig_2d.data) >= 3
    fig_3d = create_3d_multibarrel_dispersion_figure(res_front)
    assert len(fig_3d.data) >= 3


def test_benchmark_multibarrel_aspect_angles_mc():
    """Verify multi-barrel aspect angle benchmarking across all 6 standard orientations."""
    bench = benchmark_multibarrel_aspect_angles_mc(
        system_type="rotary_gatling",
        n_barrels=6,
        rotor_radius_mm=45.0,
        crossover_range_m=100.0,
        burst_rounds=30,
        target_range_m=100.0,
        cyclic_rate_rpm=3000.0,
        drone_model="commercial_quad_medium",
        bullet_energy_j=1800.0,
        seed=42,
    )

    assert "comparison_table" in bench
    assert len(bench["comparison_table"]) == 6
    assert bench["max_presented_area_cm2"] > bench["min_presented_area_cm2"]
    assert bench["area_reduction_factor"] > 1.0

    fig_bench = create_multibarrel_aspect_radar_figure(bench)
    assert len(fig_bench.data) == 2


