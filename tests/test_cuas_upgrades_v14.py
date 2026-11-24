"""Tests for Custom Drone & Shell Builders, Effective Range/Heatmap Sweeps & Spec Sheets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_SRC = str(Path(__file__).resolve().parents[1] / "src")
if ROOT_SRC not in sys.path:
    sys.path.insert(0, ROOT_SRC)

from ammo_mc.briefing_export import generate_cuas_array_spec_sheet_pdf
from ammo_mc.cuas_mission import (
    generate_cuas_array_spec_sheet_html,
    serialize_cuas_scenario_config,
    validate_cuas_scenario_config,
)
from ammo_mc.small_arms import (
    PELLET_MATERIAL_DENSITIES_G_CM3,
    benchmark_drone_aspect_angles_mc,
    build_custom_drone_spec,
    build_custom_shell_spec,
    compare_anti_drone_layouts,
    compute_drone_projected_geometry,
    evaluate_antidrone_effective_range_envelope,
    evaluate_antidrone_range_speed_heatmap,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.vis_cuas import (
    create_antidrone_effective_range_figure,
    create_antidrone_range_speed_heatmap_figure,
)


def test_build_custom_shell_spec_density_and_pattern_clamping() -> None:
    assert "Tungsten Super-Shot (TSS 18 g/cm³)" in PELLET_MATERIAL_DENSITIES_G_CM3
    sh_tss = build_custom_shell_spec(
        name="Custom #2 TSS",
        pellet_material="Tungsten Super-Shot (TSS 18 g/cm³)",
        pellet_diameter_mm=3.81,
        pellets_per_shot=70,
        v0_mps=415.0,
        custom_pattern_40yd_30in_pct=82.0,  # percentage form normalized to 0.82
    )
    sh_steel = build_custom_shell_spec(
        name="Custom Steel",
        pellet_material="Steel (Soft Iron)",
        pellet_diameter_mm=3.81,
        pellets_per_shot=70,
        v0_mps=415.0,
    )
    assert sh_tss["is_custom"] is True
    assert sh_tss["pellet_mass_g"] > sh_steel["pellet_mass_g"] * 2.0
    assert abs(sh_tss["payload_mass_g"] - sh_tss["pellet_mass_g"] * 70) < 0.5
    assert sh_tss["custom_pattern_40yd_30in_pct"] == 0.82

    sh_clamped = build_custom_shell_spec(pellet_mass_g=1.25, custom_pattern_40yd_30in_pct=1.05)
    assert sh_clamped["pellet_mass_g"] == 1.25
    assert sh_clamped["custom_pattern_40yd_30in_pct"] == 0.99


def test_build_custom_drone_spec_all_architectures() -> None:
    for d_type, exp_rotors in [("quadcopter", 4), ("hexacopter", 6), ("octocopter", 8), ("fixed_wing", 1)]:
        spec = build_custom_drone_spec(
            name=f"Threat-{d_type}",
            drone_type=d_type,
            span_m=0.80,
            fuselage_l_m=0.35,
            fuselage_w_m=0.14,
            fuselage_h_m=0.12,
            rotor_radius_m=0.13,
        )
        assert spec["is_custom"] is True
        assert spec["drone_type"] == d_type
        assert len(spec["rotors"]) == exp_rotors
        proj = compute_drone_projected_geometry(spec, aspect_mode="head_on_frontal")
        assert proj["presented_area_cm2"] > 0.0
        if d_type == "fixed_wing":
            assert "wingspan_m" in spec and len(proj["wings"]) == 2


def test_custom_specs_in_mc_benchmarks_and_pattern_tightening() -> None:
    dr = build_custom_drone_spec("Custom Hex", "hexacopter", 0.58, 0.26, 0.14, 0.11, 0.12)
    sh_tight = build_custom_shell_spec("Tight TSS", "Tungsten Super-Shot (TSS 18 g/cm³)", 3.81, 60, 420.0, custom_pattern_40yd_30in_pct=0.90)
    sh_wide = build_custom_shell_spec("Wide TSS", "Tungsten Super-Shot (TSS 18 g/cm³)", 3.81, 60, 420.0, custom_pattern_40yd_30in_pct=0.35)

    sim_tight = simulate_anti_drone_multishotgun_mc(
        n_barrels=4,
        target_distance_m=35.0,
        n_volleys=10,
        seed=7,
        custom_drone_spec=dr,
        custom_shell_spec=sh_tight,
    )
    sim_wide = simulate_anti_drone_multishotgun_mc(
        n_barrels=4,
        target_distance_m=35.0,
        n_volleys=10,
        seed=7,
        custom_drone_spec=dr,
        custom_shell_spec=sh_wide,
    )
    assert sim_tight["summary"]["drone_name"] == "Custom Hex"
    assert sim_tight["summary"]["ammunition_name"] == "Tight TSS"
    assert sim_tight["summary"]["cloud_diameter_90_m"] < sim_wide["summary"]["cloud_diameter_90_m"]

    asp = benchmark_drone_aspect_angles_mc(
        n_barrels=4,
        n_volleys=6,
        custom_drone_spec=dr,
        custom_shell_spec=sh_tight,
    )
    assert len(asp["aspect_results"]) == 6

    cmp = compare_anti_drone_layouts(
        layout_configs=[{"name": "4-Bbl", "layout_type": "circular", "n_barrels": 4, "mount_radius_m": 0.18}],
        distance_stations_m=[20.0, 40.0],
        n_volleys=6,
        custom_drone_spec=dr,
        custom_shell_spec=sh_tight,
    )
    assert len(cmp["comparison_table"]) == 2


def test_effective_range_envelope_and_heatmap_figures() -> None:
    dr = build_custom_drone_spec("Quad", "quadcopter", 0.45, 0.22, 0.12, 0.10, 0.11)
    sh = build_custom_shell_spec("TSS", "Tungsten Super-Shot (TSS 18 g/cm³)", 3.81, 70, 415.0)
    env = evaluate_antidrone_effective_range_envelope(
        distance_stations_m=[15.0, 30.0, 45.0, 65.0, 85.0],
        n_barrels=6,
        n_volleys=8,
        custom_drone_spec=dr,
        custom_shell_spec=sh,
        compare_ammo_presets=["12G_NO4_BUCK_C_UAS"],
    )
    recs = env["records"]
    assert len(recs) == 5
    assert recs[0]["kill_probability_pct"] >= recs[-1]["kill_probability_pct"]
    assert len(env["comparison_curves"]) == 1
    assert "r80_label" in env["summary"] and "r50_label" in env["summary"]

    fig_env = create_antidrone_effective_range_figure(env)
    assert len(fig_env.data) >= 5

    hm = evaluate_antidrone_range_speed_heatmap(
        distance_stations_m=[15.0, 35.0, 60.0],
        speed_stations_mps=[10.0, 25.0],
        n_barrels=6,
        n_volleys=6,
        custom_drone_spec=dr,
        custom_shell_spec=sh,
    )
    assert len(hm["pk_matrix"]) == 2
    assert len(hm["pk_matrix"][0]) == 3
    assert len(hm["r80_by_speed_m"]) == 2
    assert 0.0 <= hm["summary"]["viable_80pct_share"] <= 100.0
    fig_hm = create_antidrone_range_speed_heatmap_figure(hm)
    assert len(fig_hm.data) == 1


def test_scenario_roundtrip_custom_specs_and_html_pdf_exports() -> None:
    dr = build_custom_drone_spec("SpecQuad", "quadcopter", 0.46, 0.22, 0.12, 0.10, 0.11)
    sh = build_custom_shell_spec("SpecTSS", "Tungsten Super-Shot (TSS 18 g/cm³)", 3.81, 70, 415.0, custom_pattern_40yd_30in_pct=0.78)
    cfg = {
        "layout_type": "circular",
        "n_barrels": 6,
        "target_distance_m": 35.0,
        "drone_model": "CUSTOM_DRONE",
        "ammunition_preset": "CUSTOM_SHELL",
        "custom_drone_enabled": True,
        "custom_shell_enabled": True,
        "custom_drone_spec": dr,
        "custom_shell_spec": sh,
    }
    raw = serialize_cuas_scenario_config(cfg)
    ok, cleaned, msg = validate_cuas_scenario_config(raw)
    assert ok, msg
    assert cleaned["custom_drone_spec"]["name"] == "SpecQuad"
    assert cleaned["custom_shell_spec"]["custom_pattern_40yd_30in_pct"] == 0.78

    bad_raw = json.dumps({"schema_version": "cuas_scenario_v1", "config": {"custom_drone_spec": "bad"}})
    ok_bad, _, _ = validate_cuas_scenario_config(bad_raw)
    assert not ok_bad

    sim = simulate_anti_drone_multishotgun_mc(n_barrels=6, target_distance_m=35.0, n_volleys=8, custom_drone_spec=dr, custom_shell_spec=sh)
    env = evaluate_antidrone_effective_range_envelope(distance_stations_m=[20.0, 40.0, 60.0], n_barrels=6, n_volleys=6, custom_drone_spec=dr, custom_shell_spec=sh)
    asp = benchmark_drone_aspect_angles_mc(n_barrels=6, n_volleys=5, custom_drone_spec=dr, custom_shell_spec=sh)

    html_doc = generate_cuas_array_spec_sheet_html(cfg, sim, env, asp)
    assert "<!DOCTYPE html>" in html_doc
    assert "http://" not in html_doc and "https://" not in html_doc
    assert "TNO" not in html_doc.upper()

    pdf_bytes = generate_cuas_array_spec_sheet_pdf(cfg, sim, env, asp)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000
