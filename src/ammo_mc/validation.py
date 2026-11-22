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
