"""Unit tests for all 22 rows (`V-01a`..`V-12`) of the `Validation_Cases` sheet (App_Spec R23 MUST)."""

from __future__ import annotations

import pytest

import ammo_mc  # ensures all 96 formulas are registered
from ammo_mc.data import get_kb
from ammo_mc.formulas_registry import FORMULA_META
from ammo_mc.validation import compute_validation_case_value, run_validation_suite
from ammo_mc.vulnerability import tv_04_sspk_carleton


@pytest.mark.parametrize(
    "case_row",
    get_kb().validation_cases,
    ids=[r["test_id"] for r in get_kb().validation_cases],
)
def test_workbook_validation_case(case_row: dict) -> None:
    tid = case_row["test_id"]
    excel_val = case_row["excel_value"]
    expected = case_row["expected"]
    rel_tol = case_row["rel_tolerance"]

    if tid == "V-08":
        # Stochastic MC_Demo benchmark: test with fixed seed and N = 200,000 per workbook row 30 note
        mc_val = compute_validation_case_value("V-08", seed=12345, n_mc_v08=200_000)
        analytic_val = tv_04_sspk_carleton(30.0, (25.0**2 + 15.0**2) ** 0.5, p_dud=0.03)
        assert abs(analytic_val - expected) < 1e-12
        rel_diff = abs(mc_val - expected) / expected
        assert rel_diff < 0.015  # well inside workbook's 0.10 tolerance
        return

    val = compute_validation_case_value(tid)

    # 1. Must reproduce live Excel formula value to floating-point precision
    assert val == pytest.approx(excel_val, rel=1e-9), (
        f"{tid}: computed {val} != Excel {excel_val}"
    )

    # 2. Must agree with published source Expected value within stated rel_tolerance
    #    (except V-11 which is INFO with blank Expected, and V-12 which is documented OPEN ISSUE [FLAG])
    if tid == "V-12":
        assert FORMULA_META["TF-12"].is_flagged is True
    elif expected is not None and rel_tol is not None:
        rel_err = abs(val - expected) / abs(expected)
        assert rel_err <= rel_tol, f"{tid}: rel_err {rel_err} > {rel_tol}"


def test_run_validation_suite_summary() -> None:
    summary = run_validation_suite(seed=42, n_mc_v08=100_000)
    assert len(summary) == 22
    for row in summary:
        assert row["passed_excel"] is True, f"Failed Excel match on {row['test_id']}: {row}"
        if row["test_id"] != "V-12":
            assert row["passed_source"] is True, f"Failed source match on {row['test_id']}: {row}"
