"""Comprehensive verification suite for v1.3.0 scientific audit upgrades & V&V certificates."""

from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
from scipy.stats import qmc
from typer.testing import CliRunner

import ammo_mc
from ammo_mc.advanced_armor import tate_alekseevskii_penetration
from ammo_mc.aero_sphere import (
    didion_crosswind_drift_m,
    exact_quadratic_drag_tof_and_velocity,
    sphere_drag_compressible,
    sphere_drag_incompressible_morrison,
    sutherland_air_viscosity,
)
from ammo_mc.api import app
from ammo_mc.calibration import (
    calibrate_chronograph_from_csv,
    calibrate_drag_form_factor_from_csv,
    calibrate_shotgun_pattern_from_csv,
    generate_sample_pattern_csv,
)
from ammo_mc.cli import app as cli_app
from ammo_mc.data import get_kb
from ammo_mc.environment import eb_04_isa_atmosphere
from ammo_mc.launch import compute_exact_elliptical_cep
from ammo_mc.literature import build_literature_audit_summary
from ammo_mc.small_arms import get_drone_target_specs
from ammo_mc.small_arms_pkg.arrays import generate_multishotgun_array_layout
from ammo_mc.small_arms_pkg.catalogs import PELLET_SPECIFICATIONS, get_shotgun_presets
from ammo_mc.small_arms_pkg.geometry import compute_drone_projected_geometry
from ammo_mc.stats import compute_sobol_saltelli_indices
from ammo_mc.surveillance import generate_surveillance_sampling_plan
from ammo_mc.terminal_blast import (
    evaluate_airburst_mach_stem_blast,
    tno_eardrum_rupture_probit,
    tno_lung_lethality_probit,
    tno_thermal_lethality_probit,
)
from ammo_mc.terminal_concrete import (
    forrestal_frew_concrete_penetration,
    ndrc_concrete_perforation_and_scabbing,
)
from ammo_mc.terrain_dem import _wgs84_radii_m
from ammo_mc.validation import (
    generate_vv_certificate_html,
    generate_vv_certificate_markdown,
    run_extended_vv_benchmarks,
)
from ammo_mc.vulnerability import tv_04_sspk_carleton_elliptical


def test_aero_sphere_physics_and_didion() -> None:
    mu_sl = sutherland_air_viscosity(288.15)
    assert math.isclose(mu_sl, 1.789e-5, rel_tol=0.01)

    cd_inc = sphere_drag_incompressible_morrison(1e5)
    assert 0.38 <= cd_inc <= 0.48

    cd_sub = sphere_drag_compressible(2e5, 0.2)
    cd_sup = sphere_drag_compressible(2e5, 1.2)
    assert cd_sup - cd_sub > 0.40

    tof, v_end = exact_quadratic_drag_tof_and_velocity(50.0, 405.0, 0.01)
    assert math.isclose(v_end, 405.0 * math.exp(-0.5), rel_tol=1e-10)
    assert math.isclose(tof, (math.exp(0.5) - 1.0) / (0.01 * 405.0), rel_tol=1e-10)

    drift = didion_crosswind_drift_m(5.0, tof, 50.0, 405.0)
    assert 0.0 < drift < 5.0 * tof


def test_concrete_penetration_and_tate_alekseevskii() -> None:
    ff = forrestal_frew_concrete_penetration(10.0, 0.105, 800.0, crh_psi=3.0, fc_mpa=35.0)
    assert ff["regime"] == "deep_tunnel"
    assert math.isclose(ff["s_resistance_factor"], 82.6 * (35.0 ** -0.544), rel_tol=1e-9)
    assert ff["penetration_m"] > ff["crater_threshold_m"]

    ff_shallow = forrestal_frew_concrete_penetration(0.5, 0.105, 120.0, crh_psi=3.0, fc_mpa=45.0)
    assert ff_shallow["regime"] == "shallow_crater"

    ndrc = ndrc_concrete_perforation_and_scabbing(10.0, 0.105, 800.0, fc_mpa=35.0, wall_thickness_m=0.6)
    assert ndrc["scabbing_limit_m"] > ndrc["perforation_limit_m"] > ndrc["penetration_m"]
    assert ndrc["perforates_wall"] is True
    assert ndrc["spalls_back_face"] is True

    tate = tate_alekseevskii_penetration(1650.0, 0.50)
    vcrit_expected = math.sqrt(2.0 * (5.5e9 - 1.5e9) / 17600.0)
    assert math.isclose(tate["critical_velocity_mps"], vcrit_expected, rel_tol=1e-9)
    u0 = float(tate["history"]["interface_velocity_mps"][0])
    bernoulli_res = abs((0.5 * 17600.0 * ((1650.0 - u0) ** 2) + 1.5e9) - (0.5 * 7850.0 * (u0 ** 2) + 5.5e9))
    assert bernoulli_res < 1.0
    assert 0.50 * tate["hydrodynamic_limit_p_over_l"] < tate["penetration_over_length"] < 0.75 * tate["hydrodynamic_limit_p_over_l"]


def test_mach_stem_tno_probits_and_isa_50km() -> None:
    ms = evaluate_airburst_mach_stem_blast(50.0, 6.0, 100.0, target_height_m=1.5)
    assert ms["in_mach_stem"] is True
    assert ms["triple_point_height_m"] > 1.5
    assert ms["ps_mach_stem_kPa"] > ms["ps_free_air_kPa"]

    ear_50 = tno_eardrum_rupture_probit(math.exp((5.0 + 12.6) / 1.524))
    assert math.isclose(float(ear_50["probability"]), 0.5, rel_tol=1e-9)

    therm_50 = tno_thermal_lethality_probit(math.exp((5.0 + 14.9) / 2.56))
    assert math.isclose(float(therm_50["probability"]), 0.5, rel_tol=1e-9)

    lung_hi = tno_lung_lethality_probit(800_000.0, 4000.0)
    assert float(lung_hi["probability"]) > 0.5

    isa50 = eb_04_isa_atmosphere(50_000.0)
    assert math.isclose(float(isa50["T_K"]), 270.65, abs_tol=1e-6)
    assert math.isclose(float(isa50["p_Pa"]), 75.945, rel_tol=0.01)


def test_stats_cep_carleton_and_surveillance() -> None:
    eng = qmc.Sobol(d=6, scramble=True, seed=42)
    u = eng.random_base2(11)
    a = -math.pi + 2.0 * math.pi * u[:, :3]
    b = -math.pi + 2.0 * math.pi * u[:, 3:]

    def f_ish(m: np.ndarray) -> np.ndarray:
        return np.sin(m[:, 0]) + 7.0 * (np.sin(m[:, 1]) ** 2) + 0.1 * (m[:, 2] ** 4) * np.sin(m[:, 0])

    ya, yb = f_ish(a), f_ish(b)
    y_ab = []
    for j in range(3):
        ab = a.copy()
        ab[:, j] = b[:, j]
        y_ab.append(f_ish(ab))
    sob = compute_sobol_saltelli_indices(ya, yb, y_ab, names=["x1", "x2", "x3"])
    assert math.isclose(sob[0]["sobol_s1"], 0.3139, abs_tol=0.04)
    assert math.isclose(sob[1]["sobol_s1"], 0.4424, abs_tol=0.04)
    assert abs(sob[2]["sobol_s1"]) < 0.04
    assert math.isclose(sob[2]["sobol_st"], 0.24368, abs_tol=0.03)

    cep_c = compute_exact_elliptical_cep(20.0, 20.0)
    assert math.isclose(cep_c["cep50_exact_m"], 20.0 * math.sqrt(2.0 * math.log(2.0)), rel_tol=1e-6)

    carl = tv_04_sspk_carleton_elliptical(20.0, 10.0, 15.0, 8.0, bias_x_m=5.0, bias_y_m=3.0)
    dx, dy = 20.0 ** 2 + 2.0 * (15.0 ** 2), 10.0 ** 2 + 2.0 * (8.0 ** 2)
    expected_carl = (20.0 * 10.0 / math.sqrt(dx * dy)) * math.exp(-(25.0 / dx + 9.0 / dy))
    assert math.isclose(carl, expected_carl, rel_tol=1e-10)

    plan = generate_surveillance_sampling_plan(lot_size=5000, aql_percent=0.65)
    assert plan["sample_size_n"] == 200
    assert plan["acceptance_number_ac"] == 3
    assert plan["rejection_number_re"] == 4


def test_small_arms_terrain_and_calibration_fixes() -> None:
    assert "#2 Buckshot" in PELLET_SPECIFICATIONS
    presets = get_shotgun_presets()
    assert presets["20G_NO2_BUCK_LIGHT"]["pellet_type"] == "#2 Buckshot"

    honeycomb = generate_multishotgun_array_layout(layout_type="honeycomb", n_barrels=7)
    assert len(honeycomb) == 7
    concentric1 = generate_multishotgun_array_layout(layout_type="concentric", n_barrels=1)
    assert len(concentric1) == 1

    drone_spec = get_drone_target_specs()["commercial_quad_medium"]
    proj = compute_drone_projected_geometry(
        drone_spec,
        aspect_mode="custom",
        drone_pitch_deg=-25.0,
        drone_yaw_deg=35.0,
        drone_roll_deg=70.0,
    )
    rot_m = np.array(proj["rotation_matrix"], dtype=float)
    assert math.isclose(float(np.linalg.det(rot_m)), 1.0, abs_tol=1e-9)

    r_m, r_n = _wgs84_radii_m(32.9)
    assert 6_350_000.0 < r_m < r_n < 6_400_000.0

    chrono_csv = "velocity_mps,temp_c\n410,20\n412,NaN\n405,10\n418,30\n411,22\n"
    c_fit = calibrate_chronograph_from_csv(chrono_csv)
    assert c_fit["n_shots"] == 4
    assert np.isfinite(c_fit["v_std_fit_mps"]) and np.isfinite(c_fit["dv_dt_mps_per_c"])
    assert c_fit["dv_dt_ci95_mps_per_c"][0] < c_fit["dv_dt_mps_per_c"] < c_fit["dv_dt_ci95_mps_per_c"][1]

    drag_csv = "range_m,velocity_mps,drop_m\n0,850,0.0\n100,775,0.068\n200,705,0.295\n300,640,0.715\n"
    d_fit = calibrate_drag_form_factor_from_csv(drag_csv, preset_id="NOTIONAL-SA-01")
    assert d_fit["auto_negated_drop"] is True
    assert 0.5 < d_fit["calibrated_form_factor"] < 2.0

    pat_csv = generate_sample_pattern_csv(seed=42)
    p_fit = calibrate_shotgun_pattern_from_csv(pat_csv)
    assert p_fit["n_pellets"] >= 10
    assert p_fit["sigma_fit_ci95_m"][0] < p_fit["sigma_fit_m"] < p_fit["sigma_fit_ci95_m"][1]


def test_extended_vv_benchmarks_certificates_api_and_cli(tmp_path: Path) -> None:
    assert ammo_mc.__version__ == "1.8.0"

    kb = get_kb()
    assert len(kb.formulas_by_id) == 96
    assert len(kb.sources) == 87
    assert {f["formula_id"] for f in kb.formulas if f.get("is_flagged")} == {"TB-12", "TF-12", "TP-01"}

    lit = build_literature_audit_summary()
    assert lit["total_sources"] == 87
    assert lit["linked_source_coverage_pct"] == 100.0
    assert sorted(lit["flagged_formula_ids"]) == ["TB-12", "TF-12", "TP-01"]

    report = run_extended_vv_benchmarks(seed=42, n_mc_v08=25_000)
    summary = report["summary"]
    assert summary["total_checks"] == 40
    assert summary["physics_total"] == 18
    assert summary["physics_passed"] == 18
    assert summary["workbook_excel_passed"] == 22

    md = generate_vv_certificate_markdown(report)
    htm = generate_vv_certificate_html(report)
    assert "B-01" in md and "B-18" in md and "V-12" in md
    assert "V&amp;V Audit Certificate" in htm and "B-18" in htm and "https://cdn" not in htm.lower()

    client = TestClient(app)
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["version"] == "1.8.0"

    res_b = client.get("/validate/benchmarks?seed=42&n_mc=20000")
    assert res_b.status_code == 200
    b_data = res_b.json()
    assert b_data["summary"]["physics_passed"] == 18
    assert len(b_data["workbook_cases"]) == 22

    res_html = client.get("/validate/certificate?format=html&n_mc=20000")
    assert res_html.status_code == 200
    assert "<!DOCTYPE html>" in res_html.text

    res_md = client.get("/validate/certificate?format=markdown&n_mc=20000")
    assert res_md.status_code == 200
    assert "Executive Verification Scorecard" in res_md.text

    res_json = client.get("/validate/certificate?format=json&n_mc=20000")
    assert res_json.status_code == 200
    assert res_json.json()["summary"]["total_checks"] == 40

    assert client.get("/validate/certificate?format=xml").status_code == 400

    runner = CliRunner()
    html_out = tmp_path / "cert.html"
    md_out = tmp_path / "cert.md"
    cli_res = runner.invoke(
        cli_app,
        [
            "validate",
            "--extended",
            "--n-mc",
            "20000",
            "--out-html",
            str(html_out),
            "--out-md",
            str(md_out),
        ],
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert "ALL CHECKS PASSED" in cli_res.output
    assert html_out.exists() and "V&amp;V Audit Certificate" in html_out.read_text(encoding="utf-8")
    assert md_out.exists() and "B-18" in md_out.read_text(encoding="utf-8")
