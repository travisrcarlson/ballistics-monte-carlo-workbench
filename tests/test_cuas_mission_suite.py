"""Unit tests for C-UAS mission engineering engines and visualizations."""

from __future__ import annotations

import plotly.graph_objects as go

from ammo_mc.cuas_mission import (
    generate_3d_intercept_replay_data,
    generate_cuas_trade_study_html,
    get_cuas_ammo_economics_catalog,
    get_kill_chain_sensor_catalog,
    get_magazine_system_specs,
    get_site_defense_presets,
    serialize_cuas_scenario_config,
    simulate_cuas_kill_chain_mc,
    simulate_cuas_sustained_logistics_mc,
    simulate_multiturret_site_defense_mc,
    validate_cuas_scenario_config,
)
from ammo_mc.vis_3d import (
    create_3d_animated_intercept_figure,
    create_kill_chain_timeline_figure,
    create_logistics_thermal_and_cost_figure,
    create_multiturret_site_map_figure,
)


def test_kill_chain_fiber_optic_dark_drone() -> None:
    cat = get_kill_chain_sensor_catalog()
    assert {"micro_doppler_radar", "eoir_thermal_optical", "acoustic_array", "rf_c2_df"} <= set(cat.keys())

    res_rf = simulate_cuas_kill_chain_mc(weather="clear_day", fiber_optic_autonomous=False, n_trials=25, seed=42)
    res_dark = simulate_cuas_kill_chain_mc(weather="clear_day", fiber_optic_autonomous=True, n_trials=25, seed=42)
    rf_on = next(m for m in res_rf["modality_breakdown"] if m["sensor_key"] == "rf_c2_df")
    rf_off = next(m for m in res_dark["modality_breakdown"] if m["sensor_key"] == "rf_c2_df")
    assert rf_on["nominal_range_m"] > 100.0
    assert rf_off["nominal_range_m"] == 0.0
    assert res_dark["summary"]["fused_detection_range_m"] < res_rf["summary"]["fused_detection_range_m"]


def test_kill_chain_weather_and_hitl_latency() -> None:
    res_auto = simulate_cuas_kill_chain_mc(
        weather="clear_day", human_in_the_loop=False, fiber_optic_autonomous=True, n_trials=25, seed=7
    )
    res_fog_hitl = simulate_cuas_kill_chain_mc(
        weather="heavy_fog_rain", human_in_the_loop=True, human_auth_s=2.0, fiber_optic_autonomous=True, n_trials=25, seed=7
    )
    eoir_clear = next(m for m in res_auto["modality_breakdown"] if m["sensor_key"] == "eoir_thermal_optical")
    eoir_fog = next(m for m in res_fog_hitl["modality_breakdown"] if m["sensor_key"] == "eoir_thermal_optical")
    assert eoir_fog["nominal_range_m"] < eoir_clear["nominal_range_m"]
    assert res_fog_hitl["summary"]["t_overhead_s"] > res_auto["summary"]["t_overhead_s"]

    fig_kc = create_kill_chain_timeline_figure(res_fog_hitl)
    assert isinstance(fig_kc, go.Figure)
    assert len(fig_kc.data) >= 2


def test_kill_chain_speed_sweep_window_drop() -> None:
    res = simulate_cuas_kill_chain_mc(n_trials=25, seed=7)
    sweep = res["speed_sweep"]
    assert len(sweep) >= 6
    assert sweep[0]["cumulative_pk_pct"] >= sweep[-1]["cumulative_pk_pct"]


def test_multiturret_site_defense_los_masking() -> None:
    presets = get_site_defense_presets()
    assert "fob_compound_4_turret" in presets

    res_clear = simulate_multiturret_site_defense_mc(
        site_preset="vip_building_2_turret", custom_obstacles=[], custom_nfz=[], n_raid_trials=20, seed=42
    )
    res_obs = simulate_multiturret_site_defense_mc(
        site_preset="vip_building_2_turret", custom_nfz=[], n_raid_trials=20, seed=42
    )
    assert res_clear["summary"]["perimeter_coverage_pct"] >= res_obs["summary"]["perimeter_coverage_pct"]


def test_multiturret_nfz_fallout_restriction_and_map() -> None:
    res_nfz = simulate_multiturret_site_defense_mc(site_preset="vip_building_2_turret", n_raid_trials=20, seed=42)
    assert res_nfz["summary"]["nfz_fallout_conflict_pct"] > 0.0

    fig_site = create_multiturret_site_map_figure(res_nfz)
    assert isinstance(fig_site, go.Figure)
    assert len(fig_site.data) >= 4


def test_3d_animated_intercept_replay_frames_and_shot_string() -> None:
    replay = generate_3d_intercept_replay_data(
        layout_type="circular",
        n_barrels=6,
        firing_mode="balanced_pair_ripple",
        ripple_delay_ms=20.0,
        n_frames=20,
        max_visual_pellets=60,
        seed=42,
    )
    assert len(replay["frames"]) == 20
    assert replay["summary"]["shot_string_length_m"] > 0.2
    assert replay["summary"]["total_pellets_tracked"] >= 18

    fig_anim = create_3d_animated_intercept_figure(replay)
    assert isinstance(fig_anim, go.Figure)
    assert len(fig_anim.frames) == 20
    assert len(fig_anim.data) >= 5
    assert list(fig_anim.frames[0].traces) == [0, 1, 2, 3, 4]


def test_sustained_logistics_thermal_rise_and_cooling() -> None:
    assert "12G_NO4_BUCK_C_UAS" in get_cuas_ammo_economics_catalog()
    assert "box_10rd" in get_magazine_system_specs()

    log_fast = simulate_cuas_sustained_logistics_mc(
        n_waves=8, drones_per_wave=6, wave_interval_s=15.0, n_barrels=4, barrel_mass_kg=0.75, n_trials=15, seed=42
    )
    log_cool = simulate_cuas_sustained_logistics_mc(
        n_waves=8, drones_per_wave=6, wave_interval_s=110.0, n_barrels=4, barrel_mass_kg=0.75, n_trials=15, seed=42
    )
    assert log_fast["summary"]["peak_barrel_temp_c"] > log_cool["summary"]["peak_barrel_temp_c"]
    assert log_fast["summary"]["peak_barrel_temp_c"] > 22.0


def test_sustained_logistics_economics_comparison_and_figure() -> None:
    log_res = simulate_cuas_sustained_logistics_mc(n_waves=6, drones_per_wave=4, n_trials=15, seed=42)
    assert len(log_res["wave_records"]) == 6
    assert len(log_res["economics_comparison"]) == 7
    assert all(row["cost_per_kill_usd"] > 0.0 for row in log_res["economics_comparison"])

    fig_log = create_logistics_thermal_and_cost_figure(log_res)
    assert isinstance(fig_log, go.Figure)
    assert len(fig_log.data) >= 4


def test_scenario_config_serialization_roundtrip() -> None:
    cfg = {
        "layout_type": "grid",
        "pointing_mode": "divergent_curtain",
        "firing_mode": "balanced_pair_ripple",
        "n_barrels": 6,
        "target_distance_m": 35.0,
        "drone_model": "fpv_quad_small",
        "aspect_mode": "head_on_frontal",
        "ammunition_preset": "12G_NO4_BUCK_C_UAS",
        "choke": "Modified",
    }
    raw = serialize_cuas_scenario_config(cfg)
    ok, cleaned, msg = validate_cuas_scenario_config(raw)
    assert ok is True, msg
    assert cleaned["layout_type"] == "grid"
    assert cleaned["n_barrels"] == 6
    assert cleaned["target_distance_m"] == 35.0


def test_scenario_config_rejects_invalid_inputs() -> None:
    for bad in [
        "",
        "{not json",
        '["list"]',
        '{"schema_version": "unknown_v0", "config": {}}',
        '{"schema_version": "cuas_scenario_v1", "config": {"n_barrels": 99}}',
        '{"schema_version": "cuas_scenario_v1", "config": {"layout_type": "invalid_layout"}}',
    ]:
        ok, _, _ = validate_cuas_scenario_config(bad)
        assert ok is False


def test_trade_study_html_report_escaping() -> None:
    report_html = generate_cuas_trade_study_html(
        scenario_name="C-UAS <Script>Test</Script>",
        config={"n_barrels": 6, "target_distance_m": 35.0},
        array_summary={"kill_probability_pct": 94.0, "presented_area_cm2": 230.0},
        aspect_table=[
            {
                "name": "Head-On",
                "presented_area_cm2": 230.0,
                "cross_section_ratio_pct": 40.2,
                "kill_probability_pct": 94.0,
                "mean_hits": 7.8,
            }
        ],
        kill_chain_summary={"cumulative_pk_pct": 88.5, "first_shot_range_m": 52.0},
        logistics_summary={"peak_barrel_temp_c": 68.4, "mean_cost_per_kill_usd": 4.25},
        sdz_summary={"max_fallout_range_99_m": 310.0},
    )
    assert "<!DOCTYPE html>" in report_html
    assert "<Script>" not in report_html
    assert "&lt;Script&gt;" in report_html
    assert "GOVERNANCE" in report_html
