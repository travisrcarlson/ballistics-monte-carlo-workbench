"""Complete coverage tests for all 96 registered Formula IDs, 16 distributions, IB ODE, and MCDRAG."""

from __future__ import annotations

import numpy as np
import pytest

import ammo_mc
from ammo_mc.data import get_kb
from ammo_mc.formulas_registry import FORMULA_META, FORMULA_REGISTRY, ProvenanceTracker
from ammo_mc.interior import LumpedIBConfig, solve_lumped_interior_ballistics
from ammo_mc.mcdrag import McDragInputs, estimate_mcdrag
from ammo_mc.stats import sample_distribution, sample_unit_hypercube


def test_all_96_formulas_registered_and_metadata_present() -> None:
    kb = get_kb()
    expected_ids = set(kb.formulas_by_id.keys())
    registered_ids = set(FORMULA_REGISTRY.keys())
    assert len(expected_ids) == 96
    assert expected_ids == registered_ids, f"Missing: {expected_ids - registered_ids}"

    flagged_ids = {fid for fid, m in FORMULA_META.items() if m.is_flagged}
    assert flagged_ids == {"TB-12", "TF-12", "TP-01"}


@pytest.mark.parametrize(
    "dist_type,params",
    [
        ("normal", {"mu": 10.0, "sigma": 2.0}),
        ("truncated_normal", {"mu": 0.0, "sigma": 1.0, "a": -2.0, "b": 2.0}),
        ("lognormal", {"mu_ln": 0.0, "sigma_ln": 0.4}),
        ("uniform", {"a": 1.0, "b": 5.0}),
        ("triangular", {"a": 1.0, "c": 2.0, "b": 5.0}),
        ("beta", {"alpha": 2.0, "beta": 8.0}),
        ("bernoulli", {"p": 0.3}),
        ("binomial", {"N": 20, "p": 0.2}),
        ("poisson", {"lambda": 4.5}),
        ("rayleigh", {"sigma": 10.0}),
        ("bivariate_normal", {"mu_x": 0.0, "mu_y": 0.0, "sx": 5.0, "sy": 3.0, "rho": 0.5}),
        ("mott", {"MK": 0.12}),
        ("weibull", {"k": 2.0, "lambda": 10.0}),
        ("exponential", {"lambda": 0.5}),
        ("von_mises", {"mu": 0.0, "kappa": 3.0}),
        ("empirical", {"data": [1.0, 2.5, 4.0, 8.0], "bandwidth": 0.05}),
    ],
)
def test_all_16_distributions_direct_and_lhs(dist_type: str, params: dict) -> None:
    rng = np.random.default_rng(101)
    s_direct = sample_distribution(dist_type, params, size=200, rng=rng)
    assert s_direct.shape[0] == 200
    assert np.all(np.isfinite(s_direct))

    if dist_type != "bivariate_normal":
        u_lhs = sample_unit_hypercube(128, 1, method="LHS", rng=rng)[:, 0]
        s_lhs = sample_distribution(dist_type, params, u=u_lhs, rng=rng)
        assert s_lhs.shape == (128,)
        assert np.all(np.isfinite(s_lhs))


def test_sobol_and_mc_hypercube_samplers() -> None:
    u_sobol = sample_unit_hypercube(64, 5, method="Sobol", rng=42)
    assert u_sobol.shape == (64, 5)
    assert np.all((u_sobol > 0.0) & (u_sobol < 1.0))

    u_mc = sample_unit_hypercube(64, 5, method="MC", rng=42)
    assert u_mc.shape == (64, 5)


def test_lumped_interior_ballistics_ode() -> None:
    with ProvenanceTracker() as tracker:
        cfg = LumpedIBConfig(
            calibre_m=0.155,
            travel_m=5.2,
            chamber_volume_m3=0.018,
            proj_mass_kg=43.2,
            charge_mass_kg=11.5,
        )
        res = solve_lumped_interior_ballistics(cfg)
    assert 500.0 < res["muzzle_velocity_mps"] < 1050.0
    assert res["peak_breech_pressure_Pa"] > 100.0e6
    assert 0.15 < res["ballistic_efficiency"] < 0.55
    assert {"IB-01", "IB-02", "IB-04", "IB-05", "IB-06"}.issubset(tracker.used_formula_ids)


def test_mcdrag_port_and_diagnostics() -> None:
    inp = McDragInputs(
        d_ref_mm=155.0,
        total_length_cal=5.2,
        nose_length_cal=2.6,
        rt_over_r=0.6,
        boattail_length_cal=0.6,
        base_dia_cal=0.84,
        meplat_dia_cal=0.12,
        band_dia_cal=1.02,
        boundary_layer_code="L/T",
        identification="155mm MPM Shell",
    )
    out = estimate_mcdrag(inp)
    assert len(out["table"]) == 26
    assert out["diagnostics"] == []
    # Transonic/low-supersonic drag peak must exceed subsonic drag
    cd_sub = out["table"][0]["cd0"]  # M=0.5
    cd_sup = max(r["cd0"] for r in out["table"] if 1.0 <= r["mach"] <= 1.5)
    assert 0.08 < cd_sub < 0.25
    assert cd_sup > 1.5 * cd_sub

    # Trigger short-nose & blunt-nose diagnostics
    bad = estimate_mcdrag(McDragInputs(d_ref_mm=30.0, total_length_cal=2.0, nose_length_cal=0.7, meplat_dia_cal=0.6))
    assert len(bad["diagnostics"]) >= 2


def test_remaining_specialized_formulas() -> None:
    f = ammo_mc.formulas
    #IB-03 Noble-Abel EOS
    p_na = f.ib_03_noble_abel_eos(C_kg=5.0, z=0.5, F_J_kg=1.0e6, V_g_m3=0.015, eta_m3_kg=0.001)
    assert p_na > 1.0e8

    # DA-08 RSS error budget
    assert f.da_08_error_budget_rss([2.0, 3.0], [4.0, 2.0]) == pytest.approx(10.0)

    # MC-03 & MC-04
    z1, z2 = f.mc_04_box_muller(0.5, 0.25)
    assert abs(z1) < 1e-12 and z2 > 0.0
    assert f.mc_03_inverse_transform(0.5, lambda u: 10.0 * u) == pytest.approx(5.0)

    # TF-02 Gurney sandwiches (all 4 configs)
    for cfg_s in ("symmetric", "asymmetric", "open", "tamped"):
        v_s = f.tf_02_gurney_sandwiches(2400.0, 2.0, 1.0, cfg_s, N_tamper_kg=3.0)
        assert 500.0 < v_s < 2500.0

    # TF-13 De Marre
    t_in = f.tf_13_de_marre_penetration(D_in=6.0, W_lb=100.0, V_fps=2200.0)
    assert 3.0 < t_in < 15.0

    # TB-08 Friedlander waveform
    p0 = f.tb_08_friedlander_waveform(0.0, Ps_kPa=50.0, td_ms=20.0, Is_kPa_ms=350.0)
    pend = f.tb_08_friedlander_waveform(20.0, Ps_kPa=50.0, td_ms=20.0, Is_kPa_ms=350.0)
    assert p0 == pytest.approx(50.0)
    assert pend == pytest.approx(0.0, abs=1e-9)
