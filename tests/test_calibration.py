"""Verification tests for Field-Test Calibration & Empirical Residual Diagnostics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ammo_mc.api import app
from ammo_mc.calibration import (
    calibrate_chronograph_from_csv,
    calibrate_drag_form_factor_from_csv,
    calibrate_shotgun_pattern_from_csv,
    generate_sample_chronograph_csv,
    generate_sample_pattern_csv,
    generate_sample_range_drop_csv,
)
from ammo_mc.vis_calibration import (
    create_chronograph_calibration_figure,
    create_drag_calibration_figure,
    create_pattern_calibration_figure,
)


def test_pattern_chronograph_and_drag_parameter_recovery() -> None:
    pat_csv = generate_sample_pattern_csv(
        range_m=36.58,
        nominal_choke="Modified",
        true_spread_multiplier=1.22,
        bias_y_m=0.06,
        bias_z_m=-0.04,
        n_pellets=400,
        seed=42,
    )
    pat_res = calibrate_shotgun_pattern_from_csv(pat_csv, range_m=36.58, nominal_choke="Modified")
    assert 1.10 < pat_res["calibrated_spread_factor"] < 1.35
    assert abs(pat_res["poi_bias_y_m"] - 0.06) < 0.05
    assert pat_res["r2_cdf"] > 0.95
    assert len(create_pattern_calibration_figure(pat_res).data) >= 6

    ch_csv = generate_sample_chronograph_csv(v_std_mps=412.0, dv_dt_mps_per_c=0.72, sd_mps=3.0, seed=42)
    ch_res = calibrate_chronograph_from_csv(ch_csv, nominal_mv_mps=405.0, std_temp_c=21.1)
    assert ch_res["has_temp_fit"] is True
    assert abs(ch_res["v_std_fit_mps"] - 412.0) < 4.0
    assert abs(ch_res["dv_dt_mps_per_c"] - 0.72) < 0.25
    assert ch_res["r2_temp_fit"] > 0.85
    assert len(create_chronograph_calibration_figure(ch_res).data) >= 2

    dg_csv = generate_sample_range_drop_csv(preset_id="NOTIONAL-SA-01", true_form_factor=1.18, seed=42)
    dg_res = calibrate_drag_form_factor_from_csv(dg_csv, preset_id="NOTIONAL-SA-01")
    assert abs(dg_res["calibrated_form_factor_multiplier"] - 1.18) < 0.04
    assert dg_res["rmse_drop_after_m"] < dg_res["rmse_drop_before_m"]
    assert dg_res["r2_fit"] > 0.999
    assert len(create_drag_calibration_figure(dg_res).data) >= 4


def test_calibration_rest_endpoints() -> None:
    client = TestClient(app)

    r1 = client.post(
        "/calibrate/pattern",
        json={"csv_text": generate_sample_pattern_csv(n_pellets=60)},
    )
    assert r1.status_code == 200
    assert "calibrated_spread_factor" in r1.json()

    r2 = client.post(
        "/calibrate/chronograph",
        json={"csv_text": generate_sample_chronograph_csv()},
    )
    assert r2.status_code == 200
    assert r2.json()["has_temp_fit"] is True

    r3 = client.post(
        "/calibrate/drag",
        json={"csv_text": generate_sample_range_drop_csv()},
    )
    assert r3.status_code == 200
    assert r3.json()["r2_fit"] > 0.99
