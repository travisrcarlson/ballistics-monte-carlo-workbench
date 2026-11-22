"""Regression unit tests reproducing every live worked calculator block on the `Calculators` sheet."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ammo_mc.formulas import (
    da_02_pe_to_sigma,
    da_03_cep_circular,
    da_04_cep_elliptical,
    eb_08_miller_twist_rule,
    eb_17_fragment_velocity_decay,
    fz_02_uxo_count,
    fz_04_zero_failure_ucb,
    tb_01_scaled_distance,
    tb_04_kinney_graham,
    tb_05_dynamic_pressure,
    tb_07_normal_reflected_pressure,
    tb_10_brick_house_damage_range,
    tb_12_lung_lethality_probit,
    tb_15_simple_safety_distances,
    tf_01_gurney_cyl_sphere,
    tf_03_gurney_estimate_axial,
    tf_04_mott_distribution,
    tf_05_mott_sampler,
    tf_06_fragment_presented_area,
    tf_07_fragment_spray_density,
    tf_09_poisson_hit_kill,
    tf_11_hazardous_fragment_ke,
    tf_12_thor_penetration,
    tp_01_lanz_odermatt,
    tp_02_shaped_charge_hydrodynamic,
    tv_03_sspk_cookie_cutter,
    tv_04_sspk_carleton,
)


def test_calc_gurney_tf01_tf03() -> None:
    assert tf_01_gurney_cyl_sphere(2438.0, 2.0, 1.0, "cylinder") == pytest.approx(1541.9265870981, rel=1e-10)
    assert tf_01_gurney_cyl_sphere(2438.0, 2.0, 1.0, "sphere") == pytest.approx(1511.98341464245, rel=1e-10)
    assert tf_03_gurney_estimate_axial(6900.0)["gurney_est_mps"] == pytest.approx(2332.2, rel=1e-10)


def test_calc_mott_tf04_tf05() -> None:
    res = tf_04_mott_distribution(M0_lb=20.0, t_in=0.5, d_in=5.0, B=0.0779, m_thresh_lb=0.01)
    assert res["MK_lb_sqrt"] == pytest.approx(0.117990833931361, rel=1e-10)
    assert res["total_fragments"] == pytest.approx(718.296017810483, rel=1e-10)
    assert res["m_avg_g"] == pytest.approx(12.6296779810264, rel=1e-10)
    assert res["n_heavier"] == pytest.approx(307.771709014238, rel=1e-10)

    # Large sample mean of TF-05 must converge to 2 * MK^2 * 453.59237 g
    rng = np.random.default_rng(7)
    samples_g = tf_05_mott_sampler(res["MK_lb_sqrt"], n_fragments=150_000, rng=rng, unit="g")
    assert float(np.mean(samples_g)) == pytest.approx(12.6296779810264, rel=0.015)


def test_calc_fragment_decay_hazard_tf06_eb17_tf11() -> None:
    area = tf_06_fragment_presented_area(m_g=10.0, shape="random")
    assert area["A_m2"] == pytest.approx(0.000241316203459528, rel=1e-10)
    v_rem = eb_17_fragment_velocity_decay(1500.0, 200.0, 0.010, area["A_m2"], cd=1.29, rho_air_kg_m3=1.225)
    assert v_rem == pytest.approx(33.1095275356551, rel=1e-10)
    haz, ke = tf_11_hazardous_fragment_ke(0.010, v_rem)
    assert ke == pytest.approx(5.48120406817151, rel=1e-10)
    assert haz is False


def test_calc_spray_poisson_tf07_tf09() -> None:
    rho_f = tf_07_fragment_spray_density(400.0, 50.0, math.radians(80.0), math.radians(100.0))
    assert rho_f == pytest.approx(0.0733229430819209, rel=1e-10)
    hit = tf_09_poisson_hit_kill(rho_f, A_target_m2=0.56)
    assert hit["p_hit"] == pytest.approx(0.0402292720790743, rel=1e-10)


def test_calc_kinney_graham_tb01_04_05_07() -> None:
    z = tb_01_scaled_distance(30.0, 100.0)
    assert z == pytest.approx(6.46330407009565, rel=1e-10)
    ps = tb_04_kinney_graham(z, Pa_kPa=101.325)
    assert ps == pytest.approx(18.8282391876212, rel=1e-10)
    q = tb_05_dynamic_pressure(ps, 101.325)
    assert q == pytest.approx(1.21721265552193, rel=1e-10)
    pr = tb_07_normal_reflected_pressure(ps, 101.325)
    assert pr == pytest.approx(40.5777887484949, rel=1e-10)


def test_calc_iatg_safety_distances_tb15() -> None:
    d = tb_15_simple_safety_distances(50.0)
    assert d["frag_public_m"] == pytest.approx(1216.88888772455, rel=1e-10)
    assert d["bare_explosive_m"] == pytest.approx(478.92409482325, rel=1e-10)
    assert d["multi_item_linear_m"] == pytest.approx(809.087934718026, rel=1e-10)
    assert d["vertical_single_m"] == pytest.approx(1156.78589057308, rel=1e-10)
    assert d["vertical_multi_m"] == pytest.approx(1027.76034950668, rel=1e-10)
    assert d["noise_140dB_m"] == pytest.approx(792.066772207683, rel=1e-10)


def test_calc_scilly_damage_ranges_tb10() -> None:
    assert tb_10_brick_house_damage_range(1000.0, "A") == pytest.approx(32.1474992220417, rel=1e-10)
    assert tb_10_brick_house_damage_range(1000.0, "B") == pytest.approx(47.5515092659367, rel=1e-10)
    assert tb_10_brick_house_damage_range(1000.0, "CB") == pytest.approx(83.0477063236078, rel=1e-10)
    assert tb_10_brick_house_damage_range(1000.0, "CA") == pytest.approx(142.65452779781, rel=1e-10)
    assert tb_10_brick_house_damage_range(1000.0, "D") == pytest.approx(285.30905559562, rel=1e-10)


def test_calc_dispersion_sspk_da02_04_tv03_04() -> None:
    sigma = da_02_pe_to_sigma(30.0)
    assert sigma == pytest.approx(44.477390659748, rel=1e-10)
    cep = da_03_cep_circular(sigma)
    assert cep == pytest.approx(52.3681255381234, rel=1e-10)
    assert tv_03_sspk_cookie_cutter(40.0, cep_m=cep) == pytest.approx(0.332623031281179, rel=1e-10)
    assert tv_04_sspk_carleton(40.0, sigma) == pytest.approx(0.287952263053853, rel=1e-10)
    assert da_04_cep_elliptical(40.0, 20.0) == pytest.approx(35.322, rel=1e-10)


def test_calc_miller_eb08_and_thor_tf12() -> None:
    assert eb_08_miller_twist_rule(175.0, 0.308, 1.24, 10.0, 2600.0) == pytest.approx(2.40030931601195, rel=1e-10)

    thor = tf_12_thor_penetration(16.0, 0.38, 2000.0, 0.0, "RHA armor", "random")
    assert thor["A_in2"] == pytest.approx(0.511679680289326, rel=1e-10)
    assert thor["v_drop_fps"] == pytest.approx(1653.44707161833, rel=1e-10)
    assert thor["v_residual_mps"] == pytest.approx(1496.02934869779, rel=1e-10)
    assert thor["v50_mps"] == pytest.approx(490.694383204968, rel=1e-10)


def test_calc_lanz_odermatt_tp01_shaped_charge_tp02() -> None:
    # Note TP-01 is scale-invariant in Lw and D (passing mm returns mm)
    lo = tp_01_lanz_odermatt(500.0, 25.0, 1600.0, 17600.0, 7850.0, 300.0, 0.0, "Tungsten alloy")
    assert lo["s2_m2_s2"] == pytest.approx(1840909.09090909, rel=1e-10)
    assert lo["penetration"] == pytest.approx(364.778654655395, rel=1e-10)
    assert tp_02_shaped_charge_hydrodynamic(300.0, 8960.0, 7850.0) == pytest.approx(320.509149088886, rel=1e-10)


def test_calc_uxo_fz02_fz04_and_lung_tb12() -> None:
    uxo = fz_02_uxo_count(600, 0.02)
    assert uxo["expected_uxo"] == pytest.approx(12.0, rel=1e-12)
    assert uxo["p_at_least_one_uxo"] == pytest.approx(0.999994559417309, rel=1e-10)
    assert fz_04_zero_failure_ucb(300, 0.95) == pytest.approx(0.00993608194445772, rel=1e-10)

    lung = tb_12_lung_lethality_probit(200000.0, 0.005, p0_Pa=101325.0, m_body_kg=70.0)
    assert lung["t_scaled_s"] == pytest.approx(0.00500061694120785, rel=1e-10)
    assert lung["V"] == pytest.approx(0.21625990505955, rel=1e-10)
    assert lung["probit"] == pytest.approx(-3.72826368371049, rel=1e-10)
    assert lung["probability"] == pytest.approx(1.2930607173383e-18, rel=1e-6)
