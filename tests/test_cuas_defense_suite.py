"""Comprehensive unit, physics, and visualization tests for the 5-Module C-UAS Turret Engineering Suite."""

from __future__ import annotations

import math
import pytest

from ammo_mc.small_arms import (
    compute_balanced_pair_fire_times,
    generate_multishotgun_array_layout,
    get_shotgun_presets,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.cuas_defense import (
    compare_cuas_fallout_across_loads,
    compare_cuas_payload_families,
    get_cuas_payload_catalog,
    get_cuas_sensor_presets,
    optimize_antidrone_array_configuration,
    simulate_closed_loop_fcs_engagement_mc,
    simulate_cuas_advanced_payload_mc,
    simulate_cuas_pellet_fallout_hazard,
    simulate_drone_swarm_saturation_mc,
    simulate_turret_recoil_and_gimbal_loads,
    sweep_recoil_vs_ripple_delay,
)
from ammo_mc.vis_cuas import (
    create_2d_cuas_fallout_safety_map_figure,
    create_3d_cuas_fallout_footprint_figure,
    create_3d_swarm_hemisphere_figure,
    create_airburst_standoff_cone_figure,
    create_antidrone_optimizer_pareto_figure,
    create_cuas_payload_range_comparison_figure,
    create_fcs_tracking_and_evasion_figure,
    create_swarm_doctrine_tradeoff_figure,
    create_turret_recoil_dynamics_figure,
)


def test_small_arms_heterogeneous_and_balanced_pair():
    """Verify TSS presets, heterogeneous core+perimeter array mix, and symmetric pair ripple timings."""
    presets = get_shotgun_presets()
    assert "12G_TSS_NO2_C_UAS" in presets
    assert "12G_DUPLEX_TSS_BUCK_C_UAS" in presets
    assert "12G_FLECHETTE_SABOT_C_UAS" in presets

    honey = generate_multishotgun_array_layout(
        layout_type="honeycomb",
        n_barrels=7,
        default_ammunition_preset="12G_NO4_BUCK_C_UAS",
        inner_ammunition_preset="12G_TSS_NO2_C_UAS",
        outer_ammunition_preset="12G_NO4_BUCK_C_UAS",
        inner_choke="Full",
    )
    assert len(honey) == 7
    assert honey[0]["ammunition_preset"] == "12G_TSS_NO2_C_UAS"
    assert honey[0]["choke"] == "Full"
    assert honey[1]["ammunition_preset"] == "12G_NO4_BUCK_C_UAS"

    # Symmetric 6-barrel circular pair firing must produce 3 pairs of equal timestamps
    ring6 = generate_multishotgun_array_layout("circular", n_barrels=6, mount_radius_m=0.20)
    t_pairs = compute_balanced_pair_fire_times(ring6, firing_mode="balanced_pair_ripple", ripple_delay_ms=10.0)
    assert len(set(t_pairs)) == 3
    assert max(t_pairs) == pytest.approx(0.020, abs=1e-6)

    sim_mixed = simulate_anti_drone_multishotgun_mc(
        layout_type="honeycomb",
        n_barrels=7,
        ammunition_preset="12G_NO4_BUCK_C_UAS",
        inner_ammunition_preset="12G_TSS_NO2_C_UAS",
        inner_choke="Full",
        firing_mode="balanced_pair_ripple",
        ripple_delay_ms=10.0,
        n_volleys=10,
        seed=42,
    )
    # 1 inner TSS (70) + 6 outer #4 Buck (6*27 = 162) = 232 pellets/volley
    assert sim_mixed["summary"]["total_pellets_per_volley"] == 232
    assert sim_mixed["barrel_stats"][0]["ammunition_preset"] == "12G_TSS_NO2_C_UAS"


def test_upgrade1_closed_loop_fcs_and_evasive_maneuvers():
    """Verify Kalman (alpha-beta-gamma) lead reduces lead error on jinking drones vs unled aim."""
    sensors = get_cuas_sensor_presets()
    assert "EO_IR_Dual_Thermal_Tracker" in sensors
    assert "Ku_Band_Micro_Doppler_Radar" in sensors

    res_kalman = simulate_closed_loop_fcs_engagement_mc(
        sensor_preset="EO_IR_Dual_Thermal_Tracker",
        predictor_mode="kalman_alpha_beta_gamma",
        maneuver_profile="terminal_popup_jink",
        max_maneuver_g=3.0,
        drone_velocity_mps=25.0,
        target_distance_m=35.0,
        n_volleys=25,
        seed=42,
        include_comparison=True,
    )
    res_unled = simulate_closed_loop_fcs_engagement_mc(
        sensor_preset="EO_IR_Dual_Thermal_Tracker",
        predictor_mode="unled_boresight",
        maneuver_profile="terminal_popup_jink",
        max_maneuver_g=3.0,
        drone_velocity_mps=25.0,
        target_distance_m=35.0,
        n_volleys=25,
        seed=42,
        include_comparison=False,
    )
    assert res_kalman["summary"]["mean_lead_error_m"] < res_unled["summary"]["mean_lead_error_m"]
    assert res_kalman["summary"]["kill_probability_pct"] > res_unled["summary"]["kill_probability_pct"]
    assert len(res_kalman["architecture_comparison"]) == 3


def test_upgrade2_drone_swarm_saturation_and_doctrine():
    """Verify coordinated swarm timeline conservation (killed + leakers == swarm_size) & doctrine sweep."""
    sw = simulate_drone_swarm_saturation_mc(
        swarm_size=8,
        n_barrels=8,
        barrels_per_burst=2,
        reload_time_s=4.0,
        n_waves=25,
        seed=42,
    )
    sm = sw["summary"]
    assert sm["mean_drones_killed"] + sm["mean_leakers"] == pytest.approx(8.0, abs=0.05)
    assert 0.0 <= sm["zero_leaker_wave_pct"] <= 100.0
    assert len(sw["sample_wave_tracks"]) == 8
    assert len(sw["doctrine_sweep"]) >= 3


def test_upgrade3_advanced_payloads_and_airburst():
    """Verify TSS & Airburst superiority at 75m and sensitivity of airburst to fuze jitter."""
    cat = get_cuas_payload_catalog()
    assert "12G_TSS_NO2" in cat
    assert "CUAS_AIRBURST_CANISTER" in cat
    assert "12G_FLECHETTE_PACK" in cat

    # At 75 m, Flechette retains much more sub-projectile velocity than steel T-shot
    res_flech = simulate_cuas_advanced_payload_mc("12G_FLECHETTE_PACK", target_distance_m=75.0, n_volleys=15, seed=42)
    res_steel = simulate_cuas_advanced_payload_mc("12G_STEEL_T_SHOT", target_distance_m=75.0, n_volleys=15, seed=42)
    assert res_flech["summary"]["mean_subprojectile_velocity_mps"] > res_steel["summary"]["mean_subprojectile_velocity_mps"] + 80.0

    # Airburst: crisp 0.5 ms jitter beats noisy 10.0 ms jitter on single-shot airburst (fewer late bursts past target)
    res_ab_good = simulate_cuas_advanced_payload_mc(
        "CUAS_AIRBURST_CANISTER", n_barrels=1, target_distance_m=75.0, programmed_standoff_m=3.0, fuze_time_jitter_ms=0.5, n_volleys=25, seed=42
    )
    res_ab_bad = simulate_cuas_advanced_payload_mc(
        "CUAS_AIRBURST_CANISTER", n_barrels=1, target_distance_m=75.0, programmed_standoff_m=3.0, fuze_time_jitter_ms=10.0, n_volleys=25, seed=42
    )
    assert res_ab_good["summary"]["kill_probability_pct"] > res_ab_bad["summary"]["kill_probability_pct"]
    assert res_ab_bad["summary"]["late_burst_behind_target_pct"] > res_ab_good["summary"]["late_burst_behind_target_pct"]

    comp = compare_cuas_payload_families(distances_m=[25.0, 75.0], n_volleys=8, seed=42)
    assert len(comp["curves"]) == 7


def test_upgrade4_recoil_gimbal_and_pareto_optimizer():
    """Verify recoil impulse conservation, ripple force relief, balanced-pair yaw cancellation & optimizer."""
    ring6 = generate_multishotgun_array_layout("circular", n_barrels=6, mount_radius_m=0.22)
    rec_sim = simulate_turret_recoil_and_gimbal_loads(ring6, firing_mode="simultaneous_volley", ripple_delay_ms=0.0, mount_load_limit_kn=18.0)
    rec_seq = simulate_turret_recoil_and_gimbal_loads(ring6, firing_mode="sequential_ripple", ripple_delay_ms=10.0, mount_load_limit_kn=18.0)
    rec_pair = simulate_turret_recoil_and_gimbal_loads(ring6, firing_mode="balanced_pair_ripple", ripple_delay_ms=10.0, mount_load_limit_kn=18.0)

    # Total impulse must be identical across firing modes
    assert rec_sim["summary"]["total_impulse_ns"] == pytest.approx(rec_seq["summary"]["total_impulse_ns"], rel=1e-4)
    assert rec_sim["summary"]["total_impulse_ns"] == pytest.approx(rec_pair["summary"]["total_impulse_ns"], rel=1e-4)

    # Ripple fire strictly lowers peak transmitted mount shock force
    assert rec_seq["summary"]["peak_transmitted_mount_force_kn"] < rec_sim["summary"]["peak_transmitted_mount_force_kn"]
    assert rec_pair["summary"]["peak_transmitted_mount_force_kn"] < rec_sim["summary"]["peak_transmitted_mount_force_kn"]

    # Balanced symmetric pair ripple eliminates yaw imbalance walk on symmetric ring vs sequential ripple
    assert rec_pair["summary"]["max_yaw_deflection_mrad"] < 0.05
    assert rec_pair["summary"]["max_yaw_deflection_mrad"] < rec_seq["summary"]["max_yaw_deflection_mrad"]

    opt = optimize_antidrone_array_configuration(
        n_barrels=6,
        target_distance_m=35.0,
        mount_load_limit_kn=18.0,
        candidate_layouts=["circular", "honeycomb"],
        n_volleys=8,
        seed=42,
    )
    assert len(opt["candidates"]) > 0
    assert len(opt["pareto_frontier"]) >= 1
    assert opt["best_feasible"] is not None
    assert opt["best_feasible"]["feasible"] is True


def test_upgrade5_elevated_fallout_and_sdz():
    """Verify 3-DoF elevated fallout range, material ballistic retention & tailwind shift."""
    res_00 = simulate_cuas_pellet_fallout_hazard(ammunition_preset="12G_00_BUCK_TACTICAL", firing_elevation_deg=35.0, n_sample_pellets=120, seed=42)
    res_75 = simulate_cuas_pellet_fallout_hazard(ammunition_preset="12G_NO75_BIRD_TRAP", firing_elevation_deg=35.0, n_sample_pellets=120, seed=42)
    res_00_wind = simulate_cuas_pellet_fallout_hazard(ammunition_preset="12G_00_BUCK_TACTICAL", firing_elevation_deg=35.0, wind_downrange_mps=12.0, n_sample_pellets=120, seed=42)

    assert res_00["summary"]["max_apogee_m"] > 30.0
    assert res_00["summary"]["mean_fallout_range_m"] > res_75["summary"]["mean_fallout_range_m"]
    assert res_00["summary"]["mean_impact_energy_j"] > res_75["summary"]["mean_impact_energy_j"] * 5.0
    assert res_00_wind["summary"]["mean_fallout_range_m"] > res_00["summary"]["mean_fallout_range_m"]

    comp = compare_cuas_fallout_across_loads(firing_elevation_deg=35.0, n_sample_pellets=80, seed=42)
    assert len(comp["records"]) == 5


def test_all_cuas_visualizations_render():
    """Verify all 9 interactive Plotly visualizations for Upgrades 1-5 build without errors."""
    fcs_res = simulate_closed_loop_fcs_engagement_mc(n_volleys=10, seed=42, include_comparison=True)
    assert len(create_fcs_tracking_and_evasion_figure(fcs_res).data) > 0

    sw_res = simulate_drone_swarm_saturation_mc(n_waves=12, seed=42)
    assert len(create_3d_swarm_hemisphere_figure(sw_res).data) > 0
    assert len(create_swarm_doctrine_tradeoff_figure(sw_res).data) > 0

    ab_res = simulate_cuas_advanced_payload_mc("CUAS_AIRBURST_CANISTER", n_volleys=10, seed=42)
    assert len(create_airburst_standoff_cone_figure(ab_res).data) > 0

    pay_comp = compare_cuas_payload_families(distances_m=[25.0, 65.0], n_volleys=6, seed=42)
    assert len(create_cuas_payload_range_comparison_figure(pay_comp).data) > 0

    ring6 = generate_multishotgun_array_layout("circular", n_barrels=6)
    rec = simulate_turret_recoil_and_gimbal_loads(ring6, firing_mode="balanced_pair_ripple", ripple_delay_ms=10.0)
    swp = sweep_recoil_vs_ripple_delay(ring6, ripple_delays_ms=[0.0, 10.0, 20.0])
    assert len(create_turret_recoil_dynamics_figure(rec, swp).data) > 0

    opt = optimize_antidrone_array_configuration(candidate_layouts=["circular"], n_volleys=6, seed=42)
    assert len(create_antidrone_optimizer_pareto_figure(opt).data) > 0

    sd_res = simulate_cuas_pellet_fallout_hazard(n_sample_pellets=80, seed=42)
    sd_comp = compare_cuas_fallout_across_loads(n_sample_pellets=60, seed=42)
    assert len(create_3d_cuas_fallout_footprint_figure(sd_res).data) > 0
    assert len(create_2d_cuas_fallout_safety_map_figure(sd_res, sd_comp).data) > 0
