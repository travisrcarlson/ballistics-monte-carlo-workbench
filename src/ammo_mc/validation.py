"""Built-in validation runner reproducing all 22 rows (`V-01a`..`V-12`) of `Validation_Cases`."""

from __future__ import annotations

from typing import Any
import numpy as np

from ammo_mc.data import get_kb
from ammo_mc.environment import eb_04_isa_atmosphere
from ammo_mc.exterior import eb_08_miller_twist_rule
from ammo_mc.launch import da_01_impact_error_model, da_03_cep_circular
from ammo_mc.terminal_blast import (
    tb_01_scaled_distance,
    tb_03_kingery_bulmash,
    tb_05_dynamic_pressure,
    tb_14_probit_to_probability,
)
from ammo_mc.terminal_frag import tf_01_gurney_cyl_sphere, tf_12_thor_penetration
from ammo_mc.vulnerability import tv_01_carleton_damage, tv_04_sspk_carleton


def compute_validation_case_value(test_id: str, seed: int = 42, n_mc_v08: int = 100_000) -> float:
    """Compute the live `ammo_mc` output for a given `Validation_Cases` test ID."""
    tid = test_id.strip()

    if tid == "V-01a":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "Ps"))
    if tid == "V-01b":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "Is"))
    if tid == "V-01c":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "Pr"))
    if tid == "V-01d":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "td"))
    if tid == "V-01e":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "ta"))
    if tid == "V-01f":
        z = tb_01_scaled_distance(50.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "U"))
    if tid == "V-01g":
        z = tb_01_scaled_distance(50.0, 1000.0)
        ps = float(tb_03_kingery_bulmash(z, 1000.0, "Ps"))
        return float(tb_05_dynamic_pressure(ps, 101.325))

    if tid == "V-02a":
        z = tb_01_scaled_distance(50.0, 10000.0)
        return float(tb_03_kingery_bulmash(z, 10000.0, "Ps"))
    if tid == "V-02b":
        z = tb_01_scaled_distance(50.0, 10000.0)
        return float(tb_03_kingery_bulmash(z, 10000.0, "Ir"))
    if tid == "V-02c":
        z = tb_01_scaled_distance(50.0, 10000.0)
        return float(tb_03_kingery_bulmash(z, 10000.0, "td"))

    if tid == "V-03a":
        z = tb_01_scaled_distance(50.0, 100000.0)
        return float(tb_03_kingery_bulmash(z, 100000.0, "Ps"))
    if tid == "V-03b":
        z = tb_01_scaled_distance(50.0, 100000.0)
        return float(tb_03_kingery_bulmash(z, 100000.0, "td"))
    if tid == "V-03c":
        z = tb_01_scaled_distance(50.0, 100000.0)
        return float(tb_03_kingery_bulmash(z, 100000.0, "Pr"))

    if tid == "V-04":
        z = tb_01_scaled_distance(222.0, 1000.0)
        return float(tb_03_kingery_bulmash(z, 1000.0, "Ps"))

    if tid == "V-05":
        return float(da_03_cep_circular(1.0))

    if tid == "V-06":
        return float(tb_14_probit_to_probability(6.28))

    if tid == "V-07":
        return float(tf_01_gurney_cyl_sphere(2438.0, 2.0, 1.0, "cylinder"))

    if tid == "V-08":
        # MC_Demo pattern: N >= 1e5 rounds in occasions of 10, sigma_rr=25, sigma_b=15, R=30, p_dud=0.03
        rng = np.random.default_rng(seed)
        r_per_occ = 10
        n_occ = max(1, n_mc_v08 // r_per_occ)
        imp_x = np.asarray(da_01_impact_error_model(0.0, 15.0, 25.0, n_occ, r_per_occ, rng=rng)["impacts"])
        imp_y = np.asarray(da_01_impact_error_model(0.0, 15.0, 25.0, n_occ, r_per_occ, rng=rng)["impacts"])
        functions = (rng.random(n_occ * r_per_occ) >= 0.03).astype(np.float64)
        pk = functions * np.asarray(tv_01_carleton_damage(imp_x, imp_y, 30.0))
        # Also exercise TV-04 analytic benchmark
        _ = tv_04_sspk_carleton(30.0, float(np.hypot(25.0, 15.0)), p_dud=0.03)
        return float(np.mean(pk))

    if tid == "V-09":
        return float(eb_04_isa_atmosphere(0.0)["rho_kg_m3"])

    if tid == "V-10":
        return float(eb_04_isa_atmosphere(11000.0)["a_mps"])

    if tid == "V-11":
        return float(eb_08_miller_twist_rule(175.0, 0.308, 1.24, 10.0, 2600.0))

    if tid == "V-12":
        res = tf_12_thor_penetration(16.0, 0.38, 2000.0, 0.0, "RHA armor", "random")
        return float(res["v_residual_mps"])

    raise KeyError(f"Unknown validation test_id '{test_id}'.")


def run_validation_suite(seed: int = 42, n_mc_v08: int = 100_000) -> list[dict[str, Any]]:
    """Run all workbook `Validation_Cases` and compare against both Excel and Expected source values."""
    kb = get_kb()
    results: list[dict[str, Any]] = []

    for row in kb.validation_cases:
        tid = row["test_id"]
        val = compute_validation_case_value(tid, seed=seed, n_mc_v08=n_mc_v08)
        excel_val = row["excel_value"]
        exp_val = row["expected"]
        tol = row["rel_tolerance"]

        rel_err_excel = (
            abs(val - excel_val) / max(abs(excel_val), 1e-15) if excel_val is not None else 0.0
        )
        rel_err_expected = (
            abs(val - exp_val) / max(abs(exp_val), 1e-15) if exp_val is not None else None
        )

        # Deterministic rows must match Excel value to ~1e-9; V-08 is stochastic so compare vs analytic expected
        if tid == "V-08":
            passed_excel = (rel_err_expected is not None) and (rel_err_expected <= 0.02)
            passed_source = (rel_err_expected is not None) and (rel_err_expected <= (tol or 0.1))
            status = "PASS" if (passed_excel and passed_source) else "FAIL"
        elif tid == "V-12":
            # V-12 is documented as an OPEN ISSUE [FLAG] in workbook row 29: Excel=1496.0293 vs S07=1052
            passed_excel = rel_err_excel < 1e-9
            passed_source = False
            status = "PASS (EXCEL MATCH / FLAG OPEN ISSUE)" if passed_excel else "FAIL"
        elif exp_val is None or tol is None:
            passed_excel = rel_err_excel < 1e-9
            passed_source = True
            status = "PASS (INFO)" if passed_excel else "FAIL"
        else:
            passed_excel = rel_err_excel < 1e-9
            passed_source = (rel_err_expected is not None) and (rel_err_expected <= tol)
            status = "PASS" if (passed_excel and passed_source) else "FAIL"

        results.append({
            "test_id": tid,
            "formula": row.get("formula", row.get("formula_id", "")),
            "formula_id": row.get("formula_id", row.get("formula", "")),
            "case": row["case"],
            "unit": row["unit"],
            "computed": val,
            "excel_value": excel_val,
            "expected": exp_val,
            "rel_tolerance": tol,
            "rel_err_vs_excel": rel_err_excel,
            "rel_err_vs_expected": rel_err_expected,
            "passed_excel": passed_excel,
            "passed_source": passed_source,
            "status": status,
        })

    return results


def _make_benchmark_row(
    *,
    benchmark_id: str,
    domain: str,
    title: str,
    module: str,
    source_ids: list[str],
    unit: str,
    computed: float,
    reference: float,
    tolerance: float,
    extra_passed: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    """Build a standardized benchmark record with normalized tolerance utilization."""
    abs_err = abs(float(computed) - float(reference))
    rel_err = abs_err / max(abs(float(reference)), 1e-12)
    util_pct = (rel_err / max(float(tolerance), 1e-15)) * 100.0
    passed = bool(np.isfinite(computed) and (rel_err <= tolerance) and extra_passed)
    return {
        "benchmark_id": benchmark_id,
        "domain": domain,
        "title": title,
        "module": module,
        "source_ids": source_ids,
        "unit": unit,
        "computed": float(computed),
        "reference": float(reference),
        "abs_error": float(abs_err),
        "rel_error": float(rel_err),
        "tolerance": float(tolerance),
        "utilization_pct": float(util_pct),
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "notes": notes,
    }


def _run_physics_benchmarks_b01_b18(seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute 18 independent physical, analytical, and standards V&V benchmarks (B-01..B-18)."""
    import math
    from scipy import stats as sp_stats
    from scipy.stats import qmc

    from ammo_mc.advanced_armor import tate_alekseevskii_penetration
    from ammo_mc.aero_sphere import (
        exact_quadratic_drag_tof_and_velocity,
        sphere_drag_compressible,
    )
    from ammo_mc.exterior import eb_07_gyroscopic_stability, eb_11_vacuum_range
    from ammo_mc.launch import compute_exact_elliptical_cep
    from ammo_mc.literature import build_literature_audit_summary
    from ammo_mc.metcm import generate_stanag_4082_profile
    from ammo_mc.six_dof import simulate_6dof_trajectory
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
    from ammo_mc.terminal_frag import tf_10_aastp1_stack_density

    rows: list[dict[str, Any]] = []

    # B-01: US Standard Atmosphere 1976 at 50 km stratopause
    isa50 = eb_04_isa_atmosphere(50_000.0)
    rows.append(_make_benchmark_row(
        benchmark_id="B-01",
        domain="Atmosphere",
        title="ISA-1976 50 km stratopause hydrostatic pressure & temperature",
        module="ammo_mc.environment",
        source_ids=["S16"],
        unit="Pa",
        computed=float(isa50["p_Pa"]),
        reference=75.945,
        tolerance=0.01,
        extra_passed=(
            abs(float(isa50["T_K"]) - 270.65) < 1e-6
            and abs(float(isa50["rho_kg_m3"]) - 0.0009775) / 0.0009775 < 0.01
        ),
        notes=f"T={float(isa50['T_K']):.2f} K (ref 270.65 K), rho={float(isa50['rho_kg_m3']):.6f} kg/m^3",
    ))

    # B-02: Subcritical smooth-sphere continuum drag plateau (Clift-Grace-Weber / Morrison)
    cd_sub = float(sphere_drag_compressible(1e5, 0.1))
    rows.append(_make_benchmark_row(
        benchmark_id="B-02",
        domain="Aerodynamics",
        title="Subcritical smooth-sphere continuum Cd (Re=1e5, M=0.1)",
        module="ammo_mc.aero_sphere",
        source_ids=["S70", "S72"],
        unit="dimensionless",
        computed=cd_sub,
        reference=0.44,
        tolerance=0.08,
        extra_passed=0.38 <= cd_sub <= 0.48,
        notes="Clift-Grace-Weber / Morrison subcritical plateau (~0.42-0.44)",
    ))

    # B-03: Compressible transonic/low-supersonic sphere drag rise (Miller & Bailey / Bailey-Hiatt)
    cd_m02 = float(sphere_drag_compressible(2e5, 0.2))
    cd_m12 = float(sphere_drag_compressible(2e5, 1.2))
    rows.append(_make_benchmark_row(
        benchmark_id="B-03",
        domain="Aerodynamics",
        title="Low-supersonic compressible sphere wave-drag Cd (Re=2e5, M=1.2)",
        module="ammo_mc.aero_sphere",
        source_ids=["S69", "S71"],
        unit="dimensionless",
        computed=cd_m12,
        reference=0.94,
        tolerance=0.12,
        extra_passed=(cd_m12 - cd_m02) > 0.40,
        notes=f"Delta Cd (M=1.2 vs M=0.2) = {cd_m12 - cd_m02:.3f} (> 0.40)",
    ))

    # B-04: Exact analytical quadratic-drag TOF and residual velocity
    tof_q, v_q = exact_quadratic_drag_tof_and_velocity(50.0, 405.0, 0.01)
    v_q_ref = 405.0 * math.exp(-0.5)
    t_q_ref = (math.exp(0.5) - 1.0) / (0.01 * 405.0)
    rows.append(_make_benchmark_row(
        benchmark_id="B-04",
        domain="Exterior Ballistics",
        title="Analytical quadratic-drag residual velocity v0*exp(-kappa*R) & TOF",
        module="ammo_mc.aero_sphere",
        source_ids=["S14", "S42"],
        unit="m/s",
        computed=float(v_q),
        reference=v_q_ref,
        tolerance=1e-10,
        extra_passed=abs(float(tof_q) - t_q_ref) / t_q_ref < 1e-10,
        notes=f"TOF={float(tof_q):.6f} s vs exact {t_q_ref:.6f} s",
    ))

    # B-05: Galilean vacuum trajectory maximum range R = v0^2 / g0
    r_vac = float(eb_11_vacuum_range(827.0, math.pi / 4.0))
    r_vac_ref = (827.0 ** 2) / 9.80665
    rows.append(_make_benchmark_row(
        benchmark_id="B-05",
        domain="Exterior Ballistics",
        title="Galilean flat-Earth vacuum range maximum at 45 deg",
        module="ammo_mc.exterior",
        source_ids=["S15", "S27"],
        unit="m",
        computed=r_vac,
        reference=r_vac_ref,
        tolerance=1e-10,
        notes="Exact v0^2 / g0 closed-form verification",
    ))

    # B-06: 6-DoF rigid-body muzzle stability vs closed-form EB-07 + physical spin decay
    sim6 = simulate_6dof_trajectory(v0_mps=827.0, elev_deg=30.0, dt_s=0.004, max_t_s=2.0)
    p_spin0 = 2.0 * math.pi * 827.0 / (20.0 * 0.155)
    sg_ref = float(eb_07_gyroscopic_stability(0.145, 1.35, p_spin0, 1.225, 0.155, 827.0, c_ma=4.2))
    rows.append(_make_benchmark_row(
        benchmark_id="B-06",
        domain="6-DoF Rigid Body",
        title="6-DoF muzzle gyroscopic stability factor Sg vs EB-07 closed form",
        module="ammo_mc.six_dof",
        source_ids=["S15", "S27"],
        unit="Sg",
        computed=float(sim6["muzzle_sg"]),
        reference=sg_ref,
        tolerance=0.03,
        extra_passed=(
            float(sim6["min_sg"]) >= 1.3
            and float(sim6["terminal_spin_rpm"]) < float(sim6["initial_spin_rpm"])
            and float(sim6["max_yaw_of_repose_deg"]) < 10.0
        ),
        notes=f"min_Sg={float(sim6['min_sg']):.3f}, spin {float(sim6['initial_spin_rpm']):.0f}->{float(sim6['terminal_spin_rpm']):.0f} rpm",
    ))

    # B-07: STANAG 4082 MetCM vertical sounding structure & monotonicity
    met = generate_stanag_4082_profile()
    zones = met["zones"]
    pressures = [float(z["pressure_hpa"]) for z in zones]
    monotone_p = all(pressures[i] > pressures[i + 1] for i in range(len(pressures) - 1))
    rows.append(_make_benchmark_row(
        benchmark_id="B-07",
        domain="Meteorology",
        title="STANAG 4082 MetCM 20-zone sounding surface pressure & lapse profile",
        module="ammo_mc.metcm",
        source_ids=["S32", "S16"],
        unit="hPa",
        computed=pressures[0],
        reference=1013.25,
        tolerance=0.002,
        extra_passed=(len(zones) == 20 and monotone_p and float(zones[-1]["altitude_m"]) == 20000.0),
        notes=f"zones={len(zones)}, top={zones[-1]['altitude_m']} m, P_min={pressures[-1]:.1f} hPa",
    ))

    # B-08: Forrestal-Frew ogive concrete penetration (exact two-stage analytic check)
    ff = forrestal_frew_concrete_penetration(10.0, 0.105, 800.0, crh_psi=3.0, fc_mpa=35.0)
    s_ff = 82.6 * (35.0 ** -0.544)
    n_ff = (8.0 * 3.0 - 1.0) / (24.0 * (3.0 ** 2))
    vol_cr = math.pi * ((0.105 / 2.0) ** 2) * (2.0 * 0.105)
    v1_sq = (10.0 * (800.0 ** 2) - vol_cr * s_ff * 35e6) / (10.0 + vol_cr * n_ff * 2300.0)
    p_ff_ref = 2.0 * 0.105 + (10.0 / (2.0 * math.pi * ((0.105 / 2.0) ** 2) * n_ff * 2300.0)) * math.log(
        1.0 + (n_ff * 2300.0 * v1_sq) / (s_ff * 35e6)
    )
    rows.append(_make_benchmark_row(
        benchmark_id="B-08",
        domain="Concrete Terminal",
        title="Forrestal-Frew ogive-nose concrete depth of penetration (10 kg @ 800 m/s)",
        module="ammo_mc.terminal_concrete",
        source_ids=["S73"],
        unit="m",
        computed=float(ff["penetration_m"]),
        reference=p_ff_ref,
        tolerance=1e-9,
        notes=f"S={float(ff['s_resistance_factor']):.3f}, regime={ff['regime']}",
    ))

    # B-09: Modified NDRC concrete perforation & scabbing slab hierarchy
    nd = ndrc_concrete_perforation_and_scabbing(10.0, 0.105, 800.0, fc_mpa=35.0, wall_thickness_m=0.6)
    w_lb = 10.0 * 2.20462262185
    d_in = 0.105 / 0.0254
    v_fps = 800.0 / 0.3048
    fc_psi = 35.0 * 145.0377377
    g_ndrc = (180.0 * 1.0 * w_lb) / (math.sqrt(fc_psi) * (d_in ** 2.8)) * ((v_fps / 1000.0) ** 1.8)
    x_over_d = g_ndrc + 1.0
    perf_ref_m = (1.32 + 1.24 * x_over_d) * 0.105
    rows.append(_make_benchmark_row(
        benchmark_id="B-09",
        domain="Concrete Terminal",
        title="Modified NDRC slab perforation limit & scabbing threshold hierarchy",
        module="ammo_mc.terminal_concrete",
        source_ids=["S73", "S01"],
        unit="m",
        computed=float(nd["perforation_limit_m"]),
        reference=perf_ref_m,
        tolerance=1e-8,
        extra_passed=(
            float(nd["scabbing_limit_m"]) > float(nd["perforation_limit_m"]) > float(nd["penetration_m"])
            and bool(nd["perforates_wall"])
            and bool(nd["spalls_back_face"])
        ),
        notes=f"scab={float(nd['scabbing_limit_m']):.3f} > perf={float(nd['perforation_limit_m']):.3f} > dop={float(nd['penetration_m']):.3f} m",
    ))

    # B-10: Tate-Alekseevskii long-rod ODE vs hydrodynamic limit, analytical Vcrit & Bernoulli balance
    tate = tate_alekseevskii_penetration(1650.0, 0.50)
    vcrit_ref = math.sqrt(2.0 * (5.5e9 - 1.5e9) / 17600.0)
    hydro_lim = math.sqrt(17600.0 / 7850.0)
    p_over_l = float(tate["penetration_over_length"])
    u0 = float(tate["history"]["interface_velocity_mps"][0])
    bernoulli_res = abs((0.5 * 17600.0 * ((1650.0 - u0) ** 2) + 1.5e9) - (0.5 * 7850.0 * (u0 ** 2) + 5.5e9))
    rows.append(_make_benchmark_row(
        benchmark_id="B-10",
        domain="Armor / Long-Rod",
        title="Tate-Alekseevskii WHA long-rod critical velocity, Bernoulli balance & P/L",
        module="ammo_mc.advanced_armor",
        source_ids=["S74", "S18"],
        unit="m/s",
        computed=float(tate["critical_velocity_mps"]),
        reference=vcrit_ref,
        tolerance=1e-6,
        extra_passed=(0.50 * hydro_lim < p_over_l < 0.75 * hydro_lim and bernoulli_res < 10.0),
        notes=f"P/L={p_over_l:.3f} vs hydro={hydro_lim:.3f}, Bernoulli err={bernoulli_res:.2e} Pa",
    ))

    # B-11: UFC 3-340-02 airburst Mach-stem coalescence enhancement
    ms = evaluate_airburst_mach_stem_blast(50.0, 6.0, 100.0, target_height_m=1.5)
    enh_ratio = float(ms["ps_mach_stem_kPa"]) / max(float(ms["ps_free_air_kPa"]), 1e-9)
    rows.append(_make_benchmark_row(
        benchmark_id="B-11",
        domain="Blast / Airburst",
        title="UFC 3-340-02 airburst Mach-stem overpressure enhancement vs free air",
        module="ammo_mc.terminal_blast",
        source_ids=["S01", "S02"],
        unit="ratio",
        computed=enh_ratio,
        reference=1.31,
        tolerance=0.10,
        extra_passed=(
            bool(ms["in_mach_stem"])
            and float(ms["triple_point_height_m"]) > 1.5
            and float(ms["ps_mach_stem_kPa"]) > float(ms["ps_free_air_kPa"])
        ),
        notes=f"H_tp={float(ms['triple_point_height_m']):.2f} m, Ps_ms={float(ms['ps_mach_stem_kPa']):.2f} > Ps_free={float(ms['ps_free_air_kPa']):.2f} kPa",
    ))

    # B-12: TNO Green Book (CPR 16E) exact LD50 / RD50 probit=5.0 inversions
    tdu_50 = math.exp((5.0 + 14.9) / 2.56)
    ps_ear_50 = math.exp((5.0 + 12.6) / 1.524)
    p_th_50 = float(tno_thermal_lethality_probit(tdu_50)["probability"])
    p_ear_50 = float(tno_eardrum_rupture_probit(ps_ear_50)["probability"])
    p_lung_high = float(tno_lung_lethality_probit(800_000.0, 4000.0)["probability"])
    rows.append(_make_benchmark_row(
        benchmark_id="B-12",
        domain="Injury / Toxicology",
        title="TNO Green Book (CPR 16E) thermal/eardrum/lung probit LD50=50% inversion",
        module="ammo_mc.terminal_blast",
        source_ids=["S78"],
        unit="probability",
        computed=p_th_50,
        reference=0.5,
        tolerance=1e-9,
        extra_passed=(abs(p_ear_50 - 0.5) < 1e-9 and p_lung_high > 0.5),
        notes=f"P_eardrum(RD50)={p_ear_50:.6f}, P_lung(800 kPa, 4000 Pa*s)={p_lung_high:.3f}",
    ))

    # B-13: AASTP-1 multi-stack fragment density vs exact IATG Table 28 formula & ECM reduction
    aa_open = tf_10_aastp1_stack_density(100.0, 500, 15.0, 1500.0, L_decay_m=150.0, N_surface=10, N_total=100, storage_mode="open")
    aa_ecm = tf_10_aastp1_stack_density(100.0, 500, 15.0, 1500.0, L_decay_m=150.0, N_surface=10, N_total=100, storage_mode="ecm")
    aa_far = tf_10_aastp1_stack_density(250.0, 500, 15.0, 1500.0, L_decay_m=150.0, N_surface=10, N_total=100, storage_mode="open")
    n_eff_ref = 0.9 * 10 + 0.1 * 100
    v_i_ref = 1500.0 * math.exp(-100.0 / 150.0)
    m_f_ref = min(2.0 * 79.0 / (v_i_ref ** 2), (2.0 * 79.0 / (9.81 * 150.0)) ** 0.75)
    m_avg_ref = 15.0 / 500.0
    qf_ref = ((500.0 * n_eff_ref / (4.0 * math.pi)) / (100.0 ** 2)) * math.exp(-math.sqrt(2.0 * m_f_ref / m_avg_ref))
    rows.append(_make_benchmark_row(
        benchmark_id="B-13",
        domain="Fragmentation Safety",
        title="AASTP-1 / IATG Table 28 multi-stack hazardous fragment density & ECM attenuation",
        module="ammo_mc.terminal_frag",
        source_ids=["S01", "S08"],
        unit="frags/m^2",
        computed=float(aa_open["q_f_m2"]),
        reference=qf_ref,
        tolerance=1e-9,
        extra_passed=(float(aa_ecm["N_effective"]) == 17.0 < float(aa_open["N_effective"]) and float(aa_far["q_f_m2"]) < float(aa_open["q_f_m2"])),
        notes=f"N_eff(open)={float(aa_open['N_effective']):.1f} > N_eff(ecm)={float(aa_ecm['N_effective']):.1f}",
    ))

    # B-14: Craig exact elliptical CEP50 circular Rayleigh limit & Grubbs-Patnaik check
    cep_circ = compute_exact_elliptical_cep(20.0, 20.0)
    cep_ell = compute_exact_elliptical_cep(20.0, 10.0)
    rayleigh_ref = 20.0 * math.sqrt(2.0 * math.log(2.0))
    gp_gap = abs(float(cep_ell["cep50_exact_m"]) - float(cep_ell["cep50_grubbs_patnaik_m"])) / float(cep_ell["cep50_exact_m"])
    rows.append(_make_benchmark_row(
        benchmark_id="B-14",
        domain="Delivery Accuracy",
        title="Craig exact elliptical CEP50 vs Rayleigh circular limit & Grubbs-Patnaik",
        module="ammo_mc.launch",
        source_ids=["S76"],
        unit="m",
        computed=float(cep_circ["cep50_exact_m"]),
        reference=rayleigh_ref,
        tolerance=1e-6,
        extra_passed=(gp_gap < 0.01),
        notes=f"Elliptical (20,10) exact={float(cep_ell['cep50_exact_m']):.3f} m (G-P diff {gp_gap*100:.2f}%)",
    ))

    # B-15: Carleton diffuse damage function with non-zero elliptical dispersion & MPI bias
    from ammo_mc.vulnerability import tv_04_sspk_carleton_elliptical
    carl = float(tv_04_sspk_carleton_elliptical(20.0, 10.0, 15.0, 8.0, bias_x_m=5.0, bias_y_m=3.0))
    dx = 20.0 ** 2 + 2.0 * (15.0 ** 2)
    dy = 10.0 ** 2 + 2.0 * (8.0 ** 2)
    carl_ref = (20.0 * 10.0 / math.sqrt(dx * dy)) * math.exp(-(25.0 / dx + 9.0 / dy))
    rows.append(_make_benchmark_row(
        benchmark_id="B-15",
        domain="Vulnerability",
        title="Elliptical Carleton SSPK closed-form integral with MPI bias",
        module="ammo_mc.vulnerability",
        source_ids=["S22", "S01"],
        unit="SSPK",
        computed=carl,
        reference=carl_ref,
        tolerance=1e-10,
        notes="Rx=20 m, Ry=10 m, sx=15 m, sy=8 m, bias=(5,3) m",
    ))

    # B-16: Saltelli Sobol indices on canonical Ishigami analytical test function
    eng = qmc.Sobol(d=6, scramble=True, seed=seed)
    u = eng.random_base2(11)
    mat_a = -math.pi + 2.0 * math.pi * u[:, :3]
    mat_b = -math.pi + 2.0 * math.pi * u[:, 3:]

    def _ishigami(mat: np.ndarray) -> np.ndarray:
        return np.sin(mat[:, 0]) + 7.0 * (np.sin(mat[:, 1]) ** 2) + 0.1 * (mat[:, 2] ** 4) * np.sin(mat[:, 0])

    ya = _ishigami(mat_a)
    yb = _ishigami(mat_b)
    y_ab = []
    for j in range(3):
        mat_ab = mat_a.copy()
        mat_ab[:, j] = mat_b[:, j]
        y_ab.append(_ishigami(mat_ab))
    sob = compute_sobol_saltelli_indices(ya, yb, y_ab, names=["x1", "x2", "x3"])
    rows.append(_make_benchmark_row(
        benchmark_id="B-16",
        domain="Global Sensitivity",
        title="Saltelli Sobol S1 & ST recovery on canonical Ishigami function (N=2048)",
        module="ammo_mc.stats",
        source_ids=["S77", "S33"],
        unit="ST(x3)",
        computed=float(sob[2]["sobol_st"]),
        reference=0.24368,
        tolerance=0.08,
        extra_passed=(
            abs(float(sob[0]["sobol_s1"]) - 0.3139) < 0.04
            and abs(float(sob[1]["sobol_s1"]) - 0.4424) < 0.04
            and abs(float(sob[2]["sobol_s1"])) < 0.04
        ),
        notes=f"S1=({sob[0]['sobol_s1']:.3f}, {sob[1]['sobol_s1']:.3f}, {sob[2]['sobol_s1']:.3f}), ST3={sob[2]['sobol_st']:.3f}",
    ))

    # B-17: ISO 2859-1 lot acceptance sampling & Binomial OC producer risk
    sv = generate_surveillance_sampling_plan(lot_size=5000, aql_percent=0.65)
    alpha_ref_pct = 100.0 * (1.0 - float(sp_stats.binom.cdf(3, 200, 0.0065)))
    rows.append(_make_benchmark_row(
        benchmark_id="B-17",
        domain="Quality Surveillance",
        title="ISO 2859-1 lot surveillance OC producer risk (Lot=5000, AQL=0.65%)",
        module="ammo_mc.surveillance",
        source_ids=["S80"],
        unit="%",
        computed=float(sv["producer_risk_alpha_pct"]),
        reference=alpha_ref_pct,
        tolerance=0.02,
        extra_passed=(
            int(sv["sample_size_n"]) == 200
            and int(sv["acceptance_number_ac"]) == 3
            and int(sv["rejection_number_re"]) == 4
        ),
        notes=f"n={sv['sample_size_n']}, Ac={sv['acceptance_number_ac']}, Re={sv['rejection_number_re']}",
    ))

    # B-18: Literature provenance completeness & flagged formula governance
    lit_summary = build_literature_audit_summary()
    rows.append(_make_benchmark_row(
        benchmark_id="B-18",
        domain="Literature Governance",
        title="101-source provenance linkage & flagged formula registry integrity",
        module="ammo_mc.literature",
        source_ids=["S01", "S101"],
        unit="%",
        computed=float(lit_summary["linked_source_coverage_pct"]),
        reference=100.0,
        tolerance=1e-9,
        extra_passed=(
            int(lit_summary["total_sources"]) == 101
            and int(lit_summary["total_formulas"]) == 96
            and sorted(lit_summary["flagged_formula_ids"]) == ["TB-12", "TF-12", "TP-01"]
        ),
        notes=f"{lit_summary['total_sources']} sources, {lit_summary['mapped_modules']} modules, {lit_summary['total_formulas']} formulas",
    ))

    # B-19..B-22: v2.0 C-UAS Deep Physics & Statistical Oracles
    from ammo_mc.cuas_ballistics_v2 import (
        cipm_2007_air_density,
        compute_saami_recoil_impulse,
        integrate_pellet_trajectory_wb_f05,
    )
    from ammo_mc.cuas_rigor import (
        compute_wilson_ci,
        pk_salvo_brackets,
        pk_single_shot_closed_form,
        run_workbook_validation_cases,
    )

    wb_40 = integrate_pellet_trajectory_wb_f05(
        v0_mps=405.0, diam_mm=3.30, density_kg_m3=11200.0, range_m=40.0, drag_law="walsh_1976", k_d=1.0, temp_c=15.0, pressure_pa=101325.0, dx_m=1.0
    )
    rows.append(_make_benchmark_row(
        benchmark_id="B-19",
        domain="C-UAS Exterior Ballistics",
        title="Walsh (1976) NASA TN D-8120 continuum Cd & #4 lead 40 m retained velocity",
        module="ammo_mc.cuas_ballistics_v2",
        source_ids=["S88"],
        unit="m/s",
        computed=wb_40["v_retained_mps"],
        reference=227.858844,
        tolerance=1e-4,
        notes="WB-F05 1 m exponential integrator on Walsh table EB-T01",
    ))

    pk_a1 = pk_single_shot_closed_form(0.5, 0.4, 0.3)
    br_4 = pk_salvo_brackets(1.529181273, 0.307791848, 0.208673604, 4)
    rows.append(_make_benchmark_row(
        benchmark_id="B-20",
        domain="C-UAS Salvo Theory",
        title="WB-F01/WB-F02 exact closed-form single-shot & shared-error salvo brackets",
        module="ammo_mc.cuas_rigor",
        source_ids=["S94", "S95"],
        unit="Pk",
        computed=pk_a1,
        reference=0.268190769393115,
        tolerance=1e-9,
        extra_passed=abs(br_4["pk_shared_lower"] - 0.954930612) < 1e-4 and abs(br_4["pk_independent_upper"] - 0.980413136) < 1e-4,
        notes=f"4-gun shared={br_4['pk_shared_lower']:.6f} vs indep={br_4['pk_independent_upper']:.6f}",
    ))

    cipm_dry = cipm_2007_air_density(15.0, 101325.0, 0.0)
    saami_7 = compute_saami_recoil_impulse(589.9 * 0.06479891, 0.0, 33.4 * 0.06479891, 1275.0 * 0.3048, 7.0 * 0.45359237, 1.50)
    rows.append(_make_benchmark_row(
        benchmark_id="B-21",
        domain="Atmosphere & Recoil",
        title="CIPM-2007 air density & SAAMI proportional propellant-gas recoil impulse",
        module="ammo_mc.cuas_ballistics_v2",
        source_ids=["S90", "S91"],
        unit="N*s",
        computed=saami_7["impulse_ns"],
        reference=16.12,
        tolerance=5e-3,
        extra_passed=abs(cipm_dry["rho_kg_m3"] - 1.22552) < 1e-3,
        notes=f"CIPM rho={cipm_dry['rho_kg_m3']:.5f} kg/m^3, FRE={saami_7['free_recoil_energy_j']:.2f} J",
    ))

    wb_vv = run_workbook_validation_cases()
    w_lo, w_hi = compute_wilson_ci(4, 20, 0.10)
    rows.append(_make_benchmark_row(
        benchmark_id="B-22",
        domain="C-UAS Workbook Oracles",
        title=f"Deterministic C-UAS workbook matrix ({wb_vv['summary']['total']} oracles & binomial CIs)",
        module="ammo_mc.cuas_rigor",
        source_ids=["S94", "S101"],
        unit="%",
        computed=float(wb_vv["summary"]["pass_rate_pct"]),
        reference=100.0,
        tolerance=1e-9,
        extra_passed=int(wb_vv["summary"]["failed"]) == 0 and abs(w_lo - 0.093118017) < 1e-7 and abs(w_hi - 0.378376686) < 1e-7,
        notes=f"{wb_vv['summary']['passed']}/{wb_vv['summary']['total']} oracles pass",
    ))

    return rows, lit_summary


def _build_domain_breakdown(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate benchmark pass counts and max tolerance utilization by physics domain."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in benchmarks:
        grouped.setdefault(str(row["domain"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for domain, items in grouped.items():
        n_total = len(items)
        n_pass = sum(1 for it in items if it["passed"])
        max_util = max(float(it["utilization_pct"]) for it in items)
        summary_rows.append({
            "domain": domain,
            "total": n_total,
            "passed": n_pass,
            "pass_rate_pct": (100.0 * n_pass / n_total) if n_total else 0.0,
            "max_utilization_pct": max_util,
        })
    return summary_rows


def run_extended_vv_benchmarks(seed: int = 42, n_mc_v08: int = 50_000) -> dict[str, Any]:
    """Run both the 22-case Knowledge Base suite and 18 extended physics V&V benchmarks (40 total)."""
    from datetime import datetime, timezone
    import time

    t0 = time.perf_counter()
    workbook_cases = run_validation_suite(seed=seed, n_mc_v08=n_mc_v08)
    physics_benchmarks, lit_summary = _run_physics_benchmarks_b01_b18(seed=seed)
    elapsed_s = round(time.perf_counter() - t0, 3)

    wb_excel_passed = sum(1 for r in workbook_cases if r["passed_excel"])
    wb_source_passed = sum(1 for r in workbook_cases if r["passed_source"])
    wb_flagged_open = [str(r["test_id"]) for r in workbook_cases if not r["passed_source"]]
    phys_passed = sum(1 for r in physics_benchmarks if r["passed"])
    phys_total = len(physics_benchmarks)

    return {
        "version": "2.0.1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "n_mc_v08": int(n_mc_v08),
        "elapsed_s": elapsed_s,
        "workbook_cases": workbook_cases,
        "physics_benchmarks": physics_benchmarks,
        "summary": {
            "total_checks": len(workbook_cases) + phys_total,
            "workbook_total": len(workbook_cases),
            "workbook_excel_passed": wb_excel_passed,
            "workbook_source_passed": wb_source_passed,
            "workbook_flagged_open_ids": wb_flagged_open,
            "physics_total": phys_total,
            "physics_passed": phys_passed,
            "overall_physics_pass_rate_pct": round(100.0 * phys_passed / max(phys_total, 1), 2),
            "domain_breakdown": _build_domain_breakdown(physics_benchmarks),
            "literature": {
                "total_sources": int(lit_summary["total_sources"]),
                "total_formulas": int(lit_summary["total_formulas"]),
                "mapped_modules": int(lit_summary["mapped_modules"]),
                "flagged_formula_ids": list(lit_summary["flagged_formula_ids"]),
                "linked_source_coverage_pct": float(lit_summary["linked_source_coverage_pct"]),
            },
        },
    }


def generate_vv_certificate_markdown(report: dict[str, Any]) -> str:
    """Render a standalone GitHub-Flavored Markdown V&V Audit Certificate."""
    from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE

    summary = report["summary"]
    lit = summary["literature"]
    lines = [
        f"# Ammunition Monte Carlo (`ammo-mc`) v{report.get('version', '2.0.1')} — Verification & Validation Audit Certificate",
        "",
        f"- **Generated (UTC):** `{report.get('generated_utc', '')}`",
        f"- **RNG Seed:** `{report.get('seed', 42)}` | **V-08 MC Samples:** `{report.get('n_mc_v08', 50_000):,}` | **Runtime:** `{report.get('elapsed_s', 0.0):.3f} s`",
        f"- **Use Limitation:** {USE_LIMITATION_NOTICE}",
        "",
        "## Executive Verification Scorecard",
        "",
        "| Metric | Passed / Observed | Total / Target | Status |",
        "| --- | ---: | ---: | --- |",
        f"| Extended Physics V&V Benchmarks (`B-01..B-18`) | {summary['physics_passed']} | {summary['physics_total']} | {'PASS' if summary['physics_passed'] == summary['physics_total'] else 'FAIL'} ({summary['overall_physics_pass_rate_pct']:.1f}%) |",
        f"| Workbook Regression Cases (`V-01a..V-12` Excel Match) | {summary['workbook_excel_passed']} | {summary['workbook_total']} | {'PASS' if summary['workbook_excel_passed'] == summary['workbook_total'] else 'FAIL'} |",
        f"| Workbook Source Tolerances | {summary['workbook_source_passed']} | {summary['workbook_total']} | Open Governed Issue: {', '.join(summary['workbook_flagged_open_ids']) or 'None'} |",
        f"| Authoritative Literature Linkage | {lit['total_sources']} sources ({lit['linked_source_coverage_pct']:.1f}%) | {lit['mapped_modules']} modules / {lit['total_formulas']} formulas | GOVERNED ({', '.join(lit['flagged_formula_ids'])}) |",
        "",
        "## Domain Breakdown (`B-01..B-18`)",
        "",
        "| Domain | Passed | Total | Pass Rate | Max Tolerance Utilization |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dom in summary["domain_breakdown"]:
        lines.append(
            f"| {dom['domain']} | {dom['passed']} | {dom['total']} | {dom['pass_rate_pct']:.1f}% | {dom['max_utilization_pct']:.2f}% |"
        )

    lines.extend([
        "",
        "## Extended Physics & Standards Benchmarks (`B-01..B-18`)",
        "",
        "| ID | Domain | Benchmark | Computed | Reference | Rel Error | Tol | Util % | Status | Sources |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for b in report["physics_benchmarks"]:
        srcs = ", ".join(b.get("source_ids", []))
        lines.append(
            f"| `{b['benchmark_id']}` | {b['domain']} | {b['title']} | {b['computed']:.6g} {b['unit']} | "
            f"{b['reference']:.6g} | {b['rel_error']:.2e} | {b['tolerance']:.1e} | {b['utilization_pct']:.2f}% | **{b['status']}** | {srcs} |"
        )

    lines.extend([
        "",
        "## Knowledge Base Workbook Vectors (`V-01a..V-12`)",
        "",
        "| ID | Formula | Computed | Excel Value | Expected | Rel Err (Excel) | Status | Case |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for w in report["workbook_cases"]:
        exc = f"{w['excel_value']:.6g}" if w["excel_value"] is not None else "--"
        exp = f"{w['expected']:.6g}" if w["expected"] is not None else "--"
        lines.append(
            f"| `{w['test_id']}` | `{w['formula']}` | {w['computed']:.6g} {w['unit']} | {exc} | {exp} | "
            f"{w['rel_err_vs_excel']:.2e} | {w['status']} | {w['case']} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_vv_certificate_html(report: dict[str, Any]) -> str:
    """Render a self-contained offline HTML5 V&V Audit Certificate."""
    import html
    from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE

    summary = report["summary"]
    lit = summary["literature"]
    all_phys_ok = summary["physics_passed"] == summary["physics_total"]
    badge_bg = "#0f766e" if all_phys_ok else "#b91c1c"
    badge_txt = "VERIFIED &amp; AUDIT-READY" if all_phys_ok else "VERIFICATION ATTENTION REQUIRED"

    dom_rows_html = []
    for dom in summary["domain_breakdown"]:
        dom_rows_html.append(
            f"<tr><td>{html.escape(str(dom['domain']))}</td>"
            f"<td>{dom['passed']} / {dom['total']}</td>"
            f"<td>{dom['pass_rate_pct']:.1f}%</td>"
            f"<td>{dom['max_utilization_pct']:.2f}%</td></tr>"
        )

    bench_rows_html = []
    for b in report["physics_benchmarks"]:
        st_cls = "pill-pass" if b["status"] == "PASS" else "pill-fail"
        bench_rows_html.append(
            f"<tr><td><code>{html.escape(b['benchmark_id'])}</code></td>"
            f"<td>{html.escape(b['domain'])}</td>"
            f"<td>{html.escape(b['title'])}<div class='subnote'>{html.escape(b['notes'])}</div></td>"
            f"<td class='num'>{b['computed']:.6g} {html.escape(b['unit'])}</td>"
            f"<td class='num'>{b['reference']:.6g}</td>"
            f"<td class='num'>{b['rel_error']:.2e}</td>"
            f"<td class='num'>{b['tolerance']:.1e}</td>"
            f"<td class='num'>{b['utilization_pct']:.2f}%</td>"
            f"<td><span class='pill {st_cls}'>{html.escape(b['status'])}</span></td>"
            f"<td>{html.escape(', '.join(b.get('source_ids', [])))}</td></tr>"
        )

    wb_rows_html = []
    for w in report["workbook_cases"]:
        st_str = str(w["status"])
        st_cls = "pill-flag" if "FLAG" in st_str else ("pill-pass" if "PASS" in st_str else "pill-fail")
        exc_s = f"{w['excel_value']:.6g}" if w["excel_value"] is not None else "--"
        exp_s = f"{w['expected']:.6g}" if w["expected"] is not None else "--"
        wb_rows_html.append(
            f"<tr><td><code>{html.escape(str(w['test_id']))}</code></td>"
            f"<td><code>{html.escape(str(w['formula']))}</code></td>"
            f"<td>{html.escape(str(w['case']))}</td>"
            f"<td class='num'>{w['computed']:.6g} {html.escape(str(w['unit']))}</td>"
            f"<td class='num'>{exc_s}</td>"
            f"<td class='num'>{exp_s}</td>"
            f"<td class='num'>{w['rel_err_vs_excel']:.2e}</td>"
            f"<td><span class='pill {st_cls}'>{html.escape(st_str)}</span></td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ammo-mc v{html.escape(str(report.get('version', '2.0.1')))} V&amp;V Audit Certificate</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --accent:#0f4c81; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:28px; line-height:1.45; }}
  .wrap {{ max-width:1240px; margin:0 auto; }}
  .hero {{ background:linear-gradient(135deg,#0f172a,#1e3a5f); color:#fff; padding:24px 28px; border-radius:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; }}
  .hero h1 {{ margin:0 0 6px 0; font-size:1.45rem; }}
  .hero p {{ margin:0; color:#cbd5e1; font-size:0.9rem; }}
  .status-badge {{ background:{badge_bg}; color:#fff; padding:8px 14px; border-radius:999px; font-weight:700; font-size:0.85rem; letter-spacing:0.04em; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin:20px 0; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
  .kpi .lbl {{ font-size:0.78rem; text-transform:uppercase; color:var(--muted); letter-spacing:0.04em; }}
  .kpi .val {{ font-size:1.55rem; font-weight:700; margin-top:4px; color:var(--accent); }}
  .kpi .sub {{ font-size:0.82rem; color:var(--muted); margin-top:4px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:20px; overflow-x:auto; }}
  .card h2 {{ margin:0 0 12px 0; font-size:1.1rem; color:var(--accent); }}
  table {{ width:100%; border-collapse:collapse; font-size:0.84rem; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#f1f5f9; font-weight:600; color:#1e293b; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .subnote {{ font-size:0.76rem; color:var(--muted); margin-top:2px; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:600; font-size:0.74rem; white-space:nowrap; }}
  .pill-pass {{ background:#dcfce7; color:#166534; }}
  .pill-flag {{ background:#fef3c7; color:#92400e; }}
  .pill-fail {{ background:#fee2e2; color:#991b1b; }}
  .gov-box {{ background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:12px 16px; font-size:0.85rem; color:#78350f; }}
  @media print {{ body {{ background:#fff; padding:0; }} .card, .kpi {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div>
      <h1>Ammunition Monte Carlo (ammo-mc) v{html.escape(str(report.get('version', '2.0.1')))} — V&amp;V Audit Certificate</h1>
      <p>Generated UTC: {html.escape(str(report.get('generated_utc', '')))} &middot; Seed: {report.get('seed', 42)} &middot; V-08 MC: {report.get('n_mc_v08', 50000):,} &middot; Runtime: {report.get('elapsed_s', 0.0):.3f} s</p>
    </div>
    <div class="status-badge">{badge_txt}</div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="lbl">Extended Physics Benchmarks</div><div class="val">{summary['physics_passed']} / {summary['physics_total']}</div><div class="sub">{summary['overall_physics_pass_rate_pct']:.1f}% Pass Rate (B-01..B-18)</div></div>
    <div class="kpi"><div class="lbl">Workbook Excel Match</div><div class="val">{summary['workbook_excel_passed']} / {summary['workbook_total']}</div><div class="sub">Source Tolerance Passed: {summary['workbook_source_passed']} / {summary['workbook_total']}</div></div>
    <div class="kpi"><div class="lbl">Literature Provenance</div><div class="val">{lit['linked_source_coverage_pct']:.1f}%</div><div class="sub">{lit['total_sources']} Primary Sources &middot; {lit['mapped_modules']} Mapped Modules</div></div>
    <div class="kpi"><div class="lbl">Formula Registry Governance</div><div class="val">{lit['total_formulas']} Formulas</div><div class="sub">Flagged Open Issues: {html.escape(', '.join(lit['flagged_formula_ids']))}</div></div>
  </div>

  <div class="gov-box">
    <strong>Governance &amp; Use Limitation Notice:</strong> {html.escape(USE_LIMITATION_NOTICE)}
    Flagged formulas requiring secondary verification: <code>{html.escape(', '.join(lit['flagged_formula_ids']))}</code>
    (Workbook open issue: <code>{html.escape(', '.join(summary['workbook_flagged_open_ids']) or 'None')}</code>).
  </div>
  <br/>

  <div class="card">
    <h2>Domain Summary Scorecard (B-01..B-18)</h2>
    <table>
      <thead><tr><th>Physics / Engineering Domain</th><th>Passed / Total</th><th>Pass Rate</th><th>Max Tolerance Utilization</th></tr></thead>
      <tbody>{"".join(dom_rows_html)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Extended Physics &amp; Standards Benchmarks (B-01..B-18)</h2>
    <table>
      <thead><tr><th>ID</th><th>Domain</th><th>Benchmark &amp; Notes</th><th class="num">Computed</th><th class="num">Reference</th><th class="num">Rel Error</th><th class="num">Tolerance</th><th class="num">Util %</th><th>Status</th><th>Sources</th></tr></thead>
      <tbody>{"".join(bench_rows_html)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Knowledge Base Workbook Suite (V-01a..V-12)</h2>
    <table>
      <thead><tr><th>Test ID</th><th>Formula</th><th>Case Description</th><th class="num">Computed</th><th class="num">Excel Val</th><th class="num">Expected</th><th class="num">Rel Err (Excel)</th><th>Status</th></tr></thead>
      <tbody>{"".join(wb_rows_html)}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
