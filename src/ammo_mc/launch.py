"""Delivery accuracy, dispersion conversions, aiming errors, and correlated sampling (DA-01..DA-09)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ammo_mc.formulas_registry import register_formula

NATO_MIL_PER_CIRCLE = 6400.0
RAD_PER_NATO_MIL = 2.0 * np.pi / NATO_MIL_PER_CIRCLE
PE_OVER_SIGMA = 0.6745  # DA-02 exact workbook factor


@register_formula("DA-01")
def da_01_impact_error_model(
    x_aim: float | np.ndarray,
    sigma_bias: float | np.ndarray,
    sigma_rr: float | np.ndarray,
    n_occasions: int = 1,
    rounds_per_occasion: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float]:
    """DA-01: Impact error model x_{k,i} = x_aim + b_k + e_{k,i} with shared occasion bias."""
    gen = rng if rng is not None else np.random.default_rng()
    aim = np.asarray(x_aim, dtype=np.float64)
    sb = np.asarray(sigma_bias, dtype=np.float64)
    sr = np.asarray(sigma_rr, dtype=np.float64)
    shape_extra = np.broadcast(aim, sb, sr).shape

    bias = gen.normal(0.0, 1.0, size=(n_occasions,) + shape_extra) * sb
    disp = gen.normal(0.0, 1.0, size=(n_occasions, rounds_per_occasion) + shape_extra) * sr
    impacts = aim + bias[:, None, ...] + disp
    sigma_tot = np.sqrt(sb**2 + sr**2)
    return {
        "impacts": impacts.reshape((n_occasions * rounds_per_occasion,) + shape_extra),
        "impacts_by_occasion": impacts,
        "occasion_bias": bias,
        "sigma_total": float(sigma_tot) if sigma_tot.ndim == 0 else sigma_tot,
    }


@register_formula("DA-02")
def da_02_pe_to_sigma(
    pe: float | np.ndarray | None = None,
    *,
    sigma: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """DA-02: Convert probable error PE <-> standard deviation sigma (PE = 0.6745 * sigma)."""
    if pe is not None:
        res = np.asarray(pe, dtype=np.float64) / PE_OVER_SIGMA
        return float(res) if np.ndim(pe) == 0 else res
    if sigma is not None:
        res = np.asarray(sigma, dtype=np.float64) * PE_OVER_SIGMA
        return float(res) if np.ndim(sigma) == 0 else res
    raise ValueError("Provide either `pe` (returns sigma) or `sigma=` (returns PE).")


@register_formula("DA-03")
def da_03_cep_circular(sigma: float | np.ndarray) -> float | np.ndarray:
    """DA-03: Circular Error Probable of circular normal CEP = sigma * sqrt(2 * ln(2))."""
    res = np.asarray(sigma, dtype=np.float64) * np.sqrt(2.0 * np.log(2.0))
    return float(res) if np.ndim(sigma) == 0 else res


@register_formula("DA-04")
def da_04_cep_elliptical(
    sigma_x: float | np.ndarray,
    sigma_y: float | np.ndarray,
) -> float | np.ndarray:
    """DA-04: Elliptical normal CEP approximation CEP ~ 0.5887 * (sigma_x + sigma_y)."""
    res = 0.5887 * (np.asarray(sigma_x, dtype=np.float64) + np.asarray(sigma_y, dtype=np.float64))
    return float(res) if np.ndim(sigma_x) == 0 and np.ndim(sigma_y) == 0 else res


@register_formula("DA-05")
def da_05_radial_miss_cdf(
    R_m: float | np.ndarray,
    *,
    sigma: float | np.ndarray | None = None,
    cep: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """DA-05: Rayleigh radial miss CDF P(r <= R) = 1 - exp(-R^2/(2*sigma^2)) = 1 - 0.5^((R/CEP)^2)."""
    r_arr = np.maximum(np.asarray(R_m, dtype=np.float64), 0.0)
    if cep is not None:
        res = 1.0 - np.power(0.5, (r_arr / np.asarray(cep, dtype=np.float64)) ** 2)
    elif sigma is not None:
        res = 1.0 - np.exp(-(r_arr**2) / (2.0 * (np.asarray(sigma, dtype=np.float64) ** 2)))
    else:
        raise ValueError("Provide either `sigma` or `cep`.")
    return float(res) if np.ndim(res) == 0 else res


@register_formula("DA-06")
def da_06_rect_hit_probability(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    sx: float,
    sy: float,
    mx: float = 0.0,
    my: float = 0.0,
) -> float:
    """DA-06: Single-shot hit probability on rectangular target [x1, x2] x [y1, y2] with MPI (mx, my)."""
    px = float(sp_stats.norm.cdf((x2 - mx) / sx) - sp_stats.norm.cdf((x1 - mx) / sx))
    py = float(sp_stats.norm.cdf((y2 - my) / sy) - sp_stats.norm.cdf((y1 - my) / sy))
    return max(0.0, min(1.0, px * py))


@register_formula("DA-07")
def da_07_angular_to_linear_error(
    range_m: float | np.ndarray,
    sigma_mils: float | np.ndarray | None = None,
    *,
    sigma_rad: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """DA-07: Angular error (NATO mils or rad) to linear dispersion sigma_lin = R * sigma_ang."""
    if sigma_rad is not None:
        ang = np.asarray(sigma_rad, dtype=np.float64)
    elif sigma_mils is not None:
        ang = np.asarray(sigma_mils, dtype=np.float64) * RAD_PER_NATO_MIL
    else:
        raise ValueError("Provide either `sigma_mils` or `sigma_rad`.")
    res = np.asarray(range_m, dtype=np.float64) * ang
    return float(res) if np.ndim(res) == 0 else res


@register_formula("DA-08")
def da_08_error_budget_rss(
    sensitivities: list[float] | np.ndarray,
    sigmas: list[float] | np.ndarray,
) -> float:
    """DA-08: Linearised root-sum-square error budget sigma_tot = sqrt(sum_j (J_j * s_j)^2)."""
    j_arr = np.asarray(sensitivities, dtype=np.float64)
    s_arr = np.asarray(sigmas, dtype=np.float64)
    return float(np.sqrt(np.sum((j_arr * s_arr) ** 2)))


@register_formula("DA-09")
def da_09_correlated_bivariate_normal(
    sx: float,
    sy: float,
    rho: float = 0.0,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
    z_std: np.ndarray | None = None,
) -> np.ndarray:
    """DA-09: Correlated bivariate normal sampling via Cholesky decomposition.

    Also generalises to k-dimensional covariance matrices via `sample_correlated_mvn`.
    """
    rho_c = float(np.clip(rho, -0.999999, 0.999999))
    if z_std is not None:
        z = np.asarray(z_std, dtype=np.float64)
        z1, z2 = z[..., 0], z[..., 1]
    else:
        gen = rng if rng is not None else np.random.default_rng()
        z1 = gen.normal(size=n_samples)
        z2 = gen.normal(size=n_samples)
    x = sx * z1
    y = sy * (rho_c * z1 + np.sqrt(1.0 - rho_c**2) * z2)
    return np.column_stack([x, y])


def sample_correlated_mvn(
    mean: np.ndarray,
    sigmas: np.ndarray,
    corr_matrix: np.ndarray | None = None,
    n_samples: int = 1,
    u_unit: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """General k-dimensional Cholesky sampler compatible with MC, LHS, and Sobol unit draws."""
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sigmas, dtype=np.float64)
    k = len(mu)
    if u_unit is not None:
        z = sp_stats.norm.ppf(np.clip(u_unit, 1e-12, 1.0 - 1e-12))
    else:
        gen = rng if rng is not None else np.random.default_rng()
        z = gen.normal(size=(n_samples, k))

    if corr_matrix is None:
        return mu + z * sd

    corr = np.asarray(corr_matrix, dtype=np.float64)
    cov = np.outer(sd, sd) * corr
    # Small jitter on diagonal if positive-semidefinite
    chol = np.linalg.cholesky(cov + np.eye(k) * 1e-15)
    return mu + z @ chol.T
