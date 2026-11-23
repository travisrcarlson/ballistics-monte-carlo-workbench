"""Unit tests for A/B Turret Comparator, Layered C-UAS Defense, GIS Exporters & Tornado Sensitivity."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import plotly.graph_objects as go

from ammo_mc.cuas_analytics import (
    analyze_artillery_sensitivity_tornado,
    analyze_cuas_sensitivity_tornado,
    build_cuas_site_geojson,
    compare_cuas_architectures_ab,
    export_cuas_site_kml,
    get_cuas_comparator_presets,
    get_outer_layer_catalog,
    local_xy_to_lonlat,
    serialize_site_layout_json,
    simulate_layered_cuas_defense_mc,
    validate_site_layout_json,
)
from ammo_mc.cuas_mission import simulate_multiturret_site_defense_mc
from ammo_mc.presets import get_preset
from ammo_mc.vis_3d import (
    create_ab_comparator_radar_and_bars_figure,
    create_layered_defense_figure,
    create_real_map_site_defense_figure,
    create_sensitivity_tornado_figure,
)


def test_ab_comparator_and_figure() -> None:
    presets = get_cuas_comparator_presets()
    assert len(presets) >= 3
    cfgs = list(presets.values())[:3]
    res = compare_cuas_architectures_ab(configs=cfgs, drone_model="fpv_quad_small", n_waves=3, drones_per_wave=3, seed=42)
    sc = res["scorecard"]
    assert len(sc) == 3
    for row in sc:
        assert 0.0 <= row["mei_score"] <= 100.0
        assert 0.0 <= row["kill_chain_pk_pct"] <= 100.0
        assert len(row["radar_scores"]) == 6
    assert res["summary"]["winner_label"] in {r["label"] for r in sc}

    fig = create_ab_comparator_radar_and_bars_figure(res)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 6


def test_layered_defense_fiber_optic_vs_kinetic_outer_tier() -> None:
    cat = get_outer_layer_catalog()
    assert "rf_gnss_jammer" in cat
    assert "interceptor_drone_fpv" in cat

    # Pure fiber-optic swarm (100%): RF jammer should score 0 outer kills while interceptor engages all drones
    rf_fo = simulate_layered_cuas_defense_mc(
        outer_layer_id="rf_gnss_jammer",
        fiber_optic_fraction=1.0,
        n_waves=4,
        drones_per_wave=4,
        n_trials=30,
        seed=42,
    )
    assert rf_fo["summary"]["outer_kill_share_pct"] == 0.0

    kin_mix = simulate_layered_cuas_defense_mc(
        outer_layer_id="interceptor_drone_fpv",
        outer_magazine=20,
        fiber_optic_fraction=0.5,
        n_waves=5,
        drones_per_wave=4,
        n_trials=40,
        seed=42,
    )
    ksm = kin_mix["summary"]
    assert ksm["outer_kill_share_pct"] > 20.0
    assert ksm["layered_survival_pct"] >= ksm["inner_only_survival_pct"]
    assert ksm["inner_rounds_saved"] > 0.0
    assert ksm["peak_temp_reduction_c"] >= 0.0

    fig = create_layered_defense_figure(kin_mix)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5


def test_gis_geojson_kml_site_json_and_scattermap() -> None:
    lo, la = local_xy_to_lonlat(0.0, 0.0, 32.9, -106.4, heading_deg=0.0)
    assert abs(lo - (-106.4)) < 1e-6 and abs(la - 32.9) < 1e-6

    site_res = simulate_multiturret_site_defense_mc(n_raid_trials=15, seed=42)
    geo = build_cuas_site_geojson(site_res, origin_lat=32.9, origin_lon=-106.4, sdz_radius_m=280.0)
    assert geo["type"] == "FeatureCollection"
    layers = {f["properties"]["layer"] for f in geo["features"]}
    assert {"fallout_sdz", "defended_asset", "turret", "turret_sector", "obstacle", "nfz"}.issubset(layers)

    kml = export_cuas_site_kml(site_res, origin_lat=32.9, origin_lon=-106.4, name="Test & <Site>")
    root = ET.fromstring(kml)
    assert root.tag.endswith("kml")

    layout_json = serialize_site_layout_json(
        {
            "origin_lat": 36.896,
            "origin_lon": -76.299,
            "heading_deg": 45.0,
            "turrets": site_res["turrets"],
            "obstacles": site_res["obstacles"],
            "nfz_zones": site_res["nfz_zones"],
        }
    )
    ok, cleaned, _ = validate_site_layout_json(layout_json)
    assert ok is True
    assert cleaned["origin_lat"] == 36.896
    assert len(cleaned["turrets"]) == len(site_res["turrets"])

    bad_ok, _, _ = validate_site_layout_json(json.dumps({"schema": "wrong"}))
    assert bad_ok is False

    fig_map = create_real_map_site_defense_figure(site_res, origin_lat=32.9, origin_lon=-106.4)
    assert isinstance(fig_map, go.Figure)
    assert any(tr.type == "scattermap" for tr in fig_map.data)
    assert fig_map.layout.map.style == "open-street-map"


def test_sensitivity_tornado_cuas_and_artillery() -> None:
    cuas_tor = analyze_cuas_sensitivity_tornado(perturbation_pct=20.0, n_trials=40, seed=42)
    rows_c = cuas_tor["rows"]
    assert len(rows_c) >= 6
    swings_c = [r["swing"] for r in rows_c]
    assert swings_c == sorted(swings_c, reverse=True)
    assert swings_c[0] > 0.0
    assert len(cuas_tor["top_bottlenecks"]) == 2
    fig_c = create_sensitivity_tornado_figure(cuas_tor, metric_label="Kill-Chain P_kill (%)", higher_is_better=True)
    assert isinstance(fig_c, go.Figure) and len(fig_c.data) >= 1

    ammo, scn = get_preset("NOTIONAL-AR-01")
    art_tor = analyze_artillery_sensitivity_tornado(ammo, scn, perturbation_pct=20.0, seed=42)
    rows_a = art_tor["rows"]
    assert len(rows_a) == 7
    swings_a = [r["swing"] for r in rows_a]
    assert swings_a == sorted(swings_a, reverse=True)
    assert swings_a[0] > 0.0
    fig_a = create_sensitivity_tornado_figure(art_tor, metric_label="CEP 50% (m)", higher_is_better=False)
    assert isinstance(fig_a, go.Figure) and len(fig_a.data) >= 1
