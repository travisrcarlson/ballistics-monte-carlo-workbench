"""Fuze function, system reliability, graze/obliquity effects, and UXO models (FZ-01..FZ-07)."""

from __future__ import annotations

from typing import Any
import numpy as np

from ammo_mc.formulas_registry import register_formula


@register_formula("FZ-01")
def fz_01_system_reliability(
    reliabilities: list[float] | np.ndarray,
) -> tuple[float, float]:
    """FZ-01: Series system reliability R_sys = prod(R_i) and dud probability P_dud = 1 - R_sys."""
    r_arr = np.clip(np.asarray(reliabilities, dtype=np.float64), 0.0, 1.0)
    r_sys = float(np.prod(r_arr))
    return r_sys, float(1.0 - r_sys)


@register_formula("FZ-02")
def fz_02_uxo_count(
    n_rounds: int,
    p_dud: float | np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """FZ-02: Expected UXO count, P(>= 1 UXO), and Binomial(N, p_dud) sample."""
    p_arr = np.clip(np.asarray(p_dud, dtype=np.float64), 0.0, 1.0)
    expected = n_rounds * p_arr
    p_any = 1.0 - np.power(1.0 - p_arr, n_rounds)
    gen = rng if rng is not None else np.random.default_rng()
    sampled = gen.binomial(n_rounds, p_arr)
    if np.ndim(p_dud) == 0:
        return {
            "expected_uxo": float(expected),
            "p_at_least_one_uxo": float(p_any),
            "sampled_uxo": int(sampled),
        }
    return {
        "expected_uxo": expected,
        "p_at_least_one_uxo": p_any,
        "sampled_uxo": sampled,
    }


@register_formula("FZ-03")
def fz_03_bayesian_dud_rate(
    failures: float,
    n_tests: float,
    a0: float = 1.0,
    b0: float = 1.0,
    size: int | None = None,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """FZ-03: Bayesian epistemic dud-rate posterior sample p ~ Beta(a0 + f, b0 + n - f)."""
    alpha = max(1e-6, a0 + failures)
    beta = max(1e-6, b0 + max(0.0, n_tests - failures))
    gen = rng if rng is not None else np.random.default_rng()
    draws = gen.beta(alpha, beta, size=size)
    return float(draws) if size is None else draws


@register_formula("FZ-04")
def fz_04_zero_failure_ucb(
    n_tests: int | float,
    confidence: float = 0.95,
) -> float:
    """FZ-04: Clopper-Pearson upper confidence bound for 0 failures: p_U = 1 - alpha^(1/n)."""
    alpha = max(1e-15, 1.0 - confidence)
    return float(1.0 - (alpha ** (1.0 / max(float(n_tests), 1.0))))


@register_formula("FZ-05")
def fz_05_time_proximity_function(
    fuze_mode: str,
    t_set_s: float = 0.0,
    sigma_t_s: float = 0.0,
    hob_nom_m: float = 0.0,
    sigma_hob_m: float = 0.0,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """FZ-05: Sample time-fuze function times t_f ~ N(t_set, sigma_t) or proximity HOB ~ N(HOB, sigma_HOB)."""
    gen = rng if rng is not None else np.random.default_rng()
    mode = fuze_mode.strip().lower()
    if mode in ("time", "mt", "et", "delay_air"):
        tf = gen.normal(t_set_s, max(sigma_t_s, 0.0), size=n_samples)
        hob = np.zeros(n_samples, dtype=np.float64)
    elif mode in ("proximity", "vt", "airburst", "hob"):
        tf = np.full(n_samples, np.nan, dtype=np.float64)
        hob = np.maximum(0.0, gen.normal(hob_nom_m, max(sigma_hob_m, 0.0), size=n_samples))
    else:
        tf = np.full(n_samples, np.nan, dtype=np.float64)
        hob = np.zeros(n_samples, dtype=np.float64)
    return {"t_function_s": tf, "hob_m": hob}


@register_formula("FZ-06")
def fz_06_impact_graze_function(
    angle_of_fall_rad: float | np.ndarray,
    v_impact_mps: float | np.ndarray,
    graze_threshold_deg: float = 8.0,
    min_arm_vel_mps: float = 60.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """FZ-06: Impact fuze function probability conditioned on graze angle and impact speed."""
    ang_deg = np.degrees(np.abs(np.asarray(angle_of_fall_rad, dtype=np.float64)))
    v_arr = np.asarray(v_impact_mps, dtype=np.float64)

    # Smooth S-curve transition below graze angle threshold & minimum impact speed
    p_ang = np.clip(ang_deg / max(graze_threshold_deg, 0.1), 0.15, 1.0)
    p_vel = np.where(v_arr >= min_arm_vel_mps, 1.0, np.clip(v_arr / max(min_arm_vel_mps, 1.0), 0.0, 1.0))
    p_fn = p_ang * p_vel

    gen = rng if rng is not None else np.random.default_rng()
    fn = gen.random(size=p_fn.shape) < p_fn
    if np.ndim(angle_of_fall_rad) == 0 and np.ndim(v_impact_mps) == 0:
        return {"p_function": float(p_fn), "functions": bool(fn)}
    return {"p_function": p_fn, "functions": fn}


@register_formula("FZ-07")
def fz_07_self_destruct_redundancy(
    p_dud_primary: float | np.ndarray,
    p_fail_sd: float | np.ndarray,
    common_cause_beta: float = 0.0,
) -> float | np.ndarray:
    """FZ-07: Submunition UXO probability with self-destruct redundancy (+ optional common-cause beta)."""
    pd = np.clip(np.asarray(p_dud_primary, dtype=np.float64), 0.0, 1.0)
    psd = np.clip(np.asarray(p_fail_sd, dtype=np.float64), 0.0, 1.0)
    beta = float(np.clip(common_cause_beta, 0.0, 1.0))
    # Independent term (1 - beta)*pd*psd + common-cause failure beta*min(pd, psd)
    p_uxo = (1.0 - beta) * pd * psd + beta * np.minimum(pd, psd)
    return float(p_uxo) if np.ndim(p_uxo) == 0 else p_uxo
