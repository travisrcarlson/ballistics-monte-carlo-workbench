"""Monte Carlo statistics, numerics, RNG stream management, and distribution samplers.

Implements Formula IDs MC-01 through MC-10 plus the full Distributions sheet library.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import qmc

from ammo_mc.formulas_registry import register_formula


@register_formula("MC-01")
def mc_01_estimator_mean_se(
    x: np.ndarray | list[float],
    target_half_width: float | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """MC-01: Sample mean, standard error, CI half-width, and required N for target tolerance."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci_half_width": float("nan"), "n_req": float("nan")}
    mean = float(np.mean(arr))
    if n < 2:
        return {
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "ci_half_width": 0.0,
            "n_req": 1.0,
        }
    s = float(np.std(arr, ddof=1))
    se = s / np.sqrt(n)
    z = float(sp_stats.norm.ppf(0.5 + 0.5 * confidence))
    half_width = z * se
    n_req = float((z * s / target_half_width) ** 2) if (target_half_width and target_half_width > 0) else float(n)
    return {
        "mean": mean,
        "std": s,
        "se": float(se),
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_half_width": float(half_width),
        "n_req": n_req,
    }


@register_formula("MC-02")
def mc_02_wilson_interval(
    k: int | float,
    n: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """MC-02: Proportion estimate p_hat = k/N and Wilson score confidence interval."""
    if n <= 0:
        return {"p_hat": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "half_width": float("nan")}
    p = float(np.clip(k / n, 0.0, 1.0))
    z = float(sp_stats.norm.ppf(0.5 + 0.5 * confidence))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    low = float(max(0.0, center - margin))
    high = float(min(1.0, center + margin))
    return {
        "p_hat": p,
        "ci_low": low,
        "ci_high": high,
        "half_width": 0.5 * (high - low),
    }


@register_formula("MC-03")
def mc_03_inverse_transform(
    u: float | np.ndarray,
    ppf_fn: Callable[[np.ndarray], np.ndarray],
) -> float | np.ndarray:
    """MC-03: Inverse-transform sampling x = F^{-1}(U), U in (0, 1)."""
    u_clipped = np.clip(np.asarray(u, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    res = ppf_fn(u_clipped)
    if np.ndim(u) == 0:
        return float(res)
    return res


@register_formula("MC-04")
def mc_04_box_muller(
    u1: float | np.ndarray,
    u2: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """MC-04: Box-Muller standard normal deviate generator from Uniform(0,1) pairs."""
    u1_c = np.clip(np.asarray(u1, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    u2_c = np.asarray(u2, dtype=np.float64)
    r = np.sqrt(-2.0 * np.log(u1_c))
    theta = 2.0 * np.pi * u2_c
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    if np.ndim(u1) == 0 and np.ndim(u2) == 0:
        return float(z1), float(z2)
    return z1, z2


@register_formula("MC-05")
def mc_05_latin_hypercube(
    n_samples: int,
    n_dims: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """MC-05: Latin Hypercube Sampling in [0, 1]^d per McKay, Beckman & Conover (1979)."""
    gen = np.random.default_rng(rng)
    u = np.empty((n_samples, n_dims), dtype=np.float64)
    for j in range(n_dims):
        perm = gen.permutation(n_samples)
        u[:, j] = (perm + gen.random(n_samples)) / n_samples
    return u


def sample_unit_hypercube(
    n_samples: int,
    n_dims: int,
    method: str = "LHS",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate unit hypercube samples [N x d] via MC, LHS, or Sobol quasi-MC."""
    gen = np.random.default_rng(rng)
    m_up = method.strip().upper()
    if m_up == "LHS":
        return mc_05_latin_hypercube(n_samples, n_dims, gen)
    if m_up == "SOBOL":
        seed_int = int(gen.integers(0, 2**31 - 1))
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed_int)
        u = sampler.random(n_samples)
        return np.clip(u, 1e-12, 1.0 - 1e-12)
    if m_up in ("MC", "SRS", "RANDOM"):
        return np.clip(gen.random((n_samples, n_dims)), 1e-12, 1.0 - 1e-12)
    raise ValueError(f"Unknown sampling method '{method}'. Expected 'MC', 'LHS', or 'Sobol'.")


@register_formula("MC-06")
def mc_06_two_loop_mc(
    n_outer: int,
    n_inner: int,
    epistemic_sampler: Callable[[np.random.Generator], dict[str, Any]],
    aleatory_runner: Callable[[dict[str, Any], int, np.random.Generator], dict[str, float]],
    seed: int | None = 42,
) -> dict[str, Any]:
    """MC-06: Nested two-loop Monte Carlo separating epistemic (outer) and aleatory (inner) uncertainty."""
    streams = mc_08_reproducible_rng_streams(seed, n_outer + 1)
    epistemic_rng = streams[0]
    outer_records: list[dict[str, float]] = []
    epistemic_draws: list[dict[str, Any]] = []

    for k in range(n_outer):
        theta = epistemic_sampler(epistemic_rng)
        epistemic_draws.append(theta)
        stats_k = aleatory_runner(theta, n_inner, streams[k + 1])
        outer_records.append(stats_k)

    summary: dict[str, dict[str, float]] = {}
    if outer_records:
        for key in outer_records[0]:
            vals = np.array([r[key] for r in outer_records], dtype=np.float64)
            summary[key] = {
                "mean": float(np.mean(vals)),
                "p05": float(np.quantile(vals, 0.05)),
                "p50": float(np.quantile(vals, 0.50)),
                "p95": float(np.quantile(vals, 0.95)),
                "std_epistemic": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
    return {
        "n_outer": n_outer,
        "n_inner": n_inner,
        "outer_records": outer_records,
        "epistemic_draws": epistemic_draws,
        "summary": summary,
    }


@register_formula("MC-07")
def mc_07_rk4_integrator(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    dt: float,
) -> np.ndarray:
    """MC-07: Classical 4th-order Runge-Kutta step (supports vectorised [N x state] arrays)."""
    k1 = f(t0, y0)
    k2 = f(t0 + 0.5 * dt, y0 + 0.5 * dt * k1)
    k3 = f(t0 + 0.5 * dt, y0 + 0.5 * dt * k2)
    k4 = f(t0 + dt, y0 + dt * k3)
    return y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@register_formula("MC-08")
def mc_08_reproducible_rng_streams(
    seed: int | None,
    n_streams: int = 8,
) -> list[np.random.Generator]:
    """MC-08: Independent parallel-safe RNG streams via SeedSequence.spawn."""
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in ss.spawn(n_streams)]


@register_formula("MC-09")
def mc_09_quantile_hazard_distance(
    max_ranges: np.ndarray | list[float],
    q: float | list[float] = 0.999,
) -> float | dict[str, float]:
    """MC-09: Quantile hazard distance D_q = quantile(d_max, q)."""
    arr = np.asarray(max_ranges, dtype=np.float64)
    if isinstance(q, (int, float)):
        return float(np.quantile(arr, float(q)))
    return {f"q_{val:g}": float(np.quantile(arr, float(val))) for val in q}


@register_formula("MC-10")
def mc_10_sensitivity_indices(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """MC-10: Global sensitivity ranking via Spearman rank correlation & variance contribution."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    n_samples, n_features = X_arr.shape
    names = feature_names or [f"x_{j}" for j in range(n_features)]

    results: list[dict[str, Any]] = []
    var_y = float(np.var(y_arr))
    for j in range(n_features):
        col = X_arr[:, j]
        if np.std(col) < 1e-14 or var_y < 1e-14 or n_samples < 4:
            rho_s, p_val, s1_approx = 0.0, 1.0, 0.0
        else:
            sp_res = sp_stats.spearmanr(col, y_arr)
            rho_s = float(0.0 if np.isnan(sp_res.statistic) else sp_res.statistic)
            p_val = float(1.0 if np.isnan(sp_res.pvalue) else sp_res.pvalue)
            # Binned conditional-expectation first-order Sobol estimator Var(E[Y|X_j])/Var(Y)
            n_bins = max(4, min(20, int(np.sqrt(n_samples))))
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(col, quantiles))
            if len(edges) > 2:
                bin_idx = np.digitize(col, edges[1:-1])
                cond_means = [
                    np.mean(y_arr[bin_idx == b])
                    for b in range(len(edges) - 1)
                    if np.any(bin_idx == b)
                ]
                s1_approx = float(np.clip(np.var(cond_means) / var_y, 0.0, 1.0))
            else:
                s1_approx = float(rho_s**2)
        results.append({
            "parameter": names[j],
            "spearman_rho": rho_s,
            "abs_rho": abs(rho_s),
            "importance_score": abs(rho_s),
            "p_value": p_val,
            "sobol_s1_est": s1_approx,
        })

    results.sort(key=lambda r: r["abs_rho"], reverse=True)
    return results


def compute_sobol_saltelli_indices(
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_ab_list: list[np.ndarray] | np.ndarray,
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Saltelli (2010) variance-based first-order S_i and total-order S_Ti Sobol sensitivity indices.

    Given model outputs from base matrices A, B and Saltelli hybrid matrices AB^(i)
    (where column i comes from B and remaining columns from A):
      S_i  = mean(Y_B * (Y_AB^(i) - Y_A)) / Var(Y)
      S_Ti = 0.5 * mean((Y_A - Y_AB^(i))^2) / Var(Y)     [Jansen 1999 / Saltelli 2010 Eq. b & f]
    """
    ya = np.asarray(y_a, dtype=np.float64).ravel()
    yb = np.asarray(y_b, dtype=np.float64).ravel()
    y_ab = [np.asarray(arr, dtype=np.float64).ravel() for arr in y_ab_list]
    k = len(y_ab)
    lbls = names if (names is not None and len(names) == k) else [f"x_{i}" for i in range(k)]
    var_y = float(np.var(np.concatenate([ya, yb]), ddof=1)) if (ya.size + yb.size) > 2 else 0.0
    out: list[dict[str, Any]] = []
    for i in range(k):
        yabi = y_ab[i]
        if var_y <= 1e-18:
            s1, st = 0.0, 0.0
        else:
            s1 = float(np.mean(yb * (yabi - ya)) / var_y)
            st = float(0.5 * np.mean((ya - yabi) ** 2) / var_y)
        out.append({
            "parameter": lbls[i],
            "sobol_s1": float(np.clip(s1, -0.05, 1.05)),
            "sobol_st": float(np.clip(st, 0.0, 1.05)),
        })
    out.sort(key=lambda r: r["sobol_st"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Full Distributions Sheet Sampler Library (16 distributions)
# ---------------------------------------------------------------------------

def sample_distribution(
    dist_type: str,
    params: dict[str, Any],
    size: int | tuple[int, ...] = 1,
    rng: np.random.Generator | None = None,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """Sample from any of the 16 distributions listed on the Distributions sheet.

    Supports both direct RNG draws and inverse-CDF transformation of LHS/Sobol `u` in (0,1).
    """
    gen = rng if rng is not None else np.random.default_rng()
    dt = dist_type.strip().lower().replace("-", "_").replace(" ", "_")

    if u is not None:
        u_c = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    else:
        u_c = None

    if dt == "normal":
        mu, sigma = float(params.get("mu", 0.0)), float(params.get("sigma", 1.0))
        if sigma <= 0.0:
            return np.full(u_c.shape if u_c is not None else size, mu, dtype=np.float64)
        return sp_stats.norm.ppf(u_c, loc=mu, scale=sigma) if u_c is not None else gen.normal(mu, sigma, size=size)

    if dt in ("truncated_normal", "truncnorm"):
        mu, sigma = float(params["mu"]), float(params["sigma"])
        a, b = float(params["a"]), float(params["b"])
        if sigma <= 0.0:
            return np.full(u_c.shape if u_c is not None else size, np.clip(mu, a, b), dtype=np.float64)
        alpha, beta = (a - mu) / sigma, (b - mu) / sigma
        dist = sp_stats.truncnorm(alpha, beta, loc=mu, scale=sigma)
        return dist.ppf(u_c) if u_c is not None else dist.rvs(size=size, random_state=gen)

    if dt == "lognormal":
        mu_ln, sigma_ln = float(params["mu_ln"]), float(params["sigma_ln"])
        if u_c is not None:
            return np.exp(sp_stats.norm.ppf(u_c, loc=mu_ln, scale=sigma_ln))
        return gen.lognormal(mu_ln, sigma_ln, size=size)

    if dt == "uniform":
        a, b = float(params.get("a", 0.0)), float(params.get("b", 1.0))
        return a + (b - a) * u_c if u_c is not None else gen.uniform(a, b, size=size)

    if dt == "triangular":
        a, c, b = float(params["a"]), float(params["c"]), float(params["b"])
        if b <= a:
            return np.full(u_c.shape if u_c is not None else size, a, dtype=np.float64)
        if u_c is not None:
            fc = (c - a) / (b - a)
            return np.where(
                u_c < fc,
                a + np.sqrt(u_c * (b - a) * (c - a)),
                b - np.sqrt((1.0 - u_c) * (b - a) * (b - c)),
            )
        return gen.triangular(a, c, b, size=size)

    if dt == "beta":
        alpha, beta = float(params["alpha"]), float(params["beta"])
        return sp_stats.beta.ppf(u_c, alpha, beta) if u_c is not None else gen.beta(alpha, beta, size=size)

    if dt == "bernoulli":
        p = float(params["p"])
        draws = u_c if u_c is not None else gen.random(size=size)
        return (draws < p).astype(np.float64)

    if dt == "binomial":
        n_b, p = int(params["N"]), float(params["p"])
        if u_c is not None:
            return sp_stats.binom.ppf(u_c, n_b, p).astype(np.float64)
        return gen.binomial(n_b, p, size=size).astype(np.float64)

    if dt == "poisson":
        lam = float(params["lambda"])
        if u_c is not None:
            return sp_stats.poisson.ppf(u_c, lam).astype(np.float64)
        return gen.poisson(lam, size=size).astype(np.float64)

    if dt == "rayleigh":
        sigma = float(params["sigma"])
        draws = u_c if u_c is not None else np.clip(gen.random(size=size), 1e-15, 1.0)
        return sigma * np.sqrt(-2.0 * np.log(1.0 - draws))

    if dt == "bivariate_normal":
        mu_x, mu_y = float(params.get("mu_x", 0.0)), float(params.get("mu_y", 0.0))
        sx, sy = float(params["sx"]), float(params["sy"])
        rho = float(params.get("rho", 0.0))
        if u_c is not None and np.asarray(u_c).ndim >= 2 and np.asarray(u_c).shape[-1] >= 2:
            z1 = sp_stats.norm.ppf(u_c[..., 0])
            z2 = sp_stats.norm.ppf(u_c[..., 1])
        else:
            n_pts = size if isinstance(size, int) else int(np.prod(size))
            z1 = gen.normal(size=n_pts)
            z2 = gen.normal(size=n_pts)
        x = mu_x + sx * z1
        y = mu_y + sy * (rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
        return np.column_stack([x, y])

    if dt == "mott":
        mk = float(params["MK"])
        draws = u_c if u_c is not None else np.clip(gen.random(size=size), 1e-15, 1.0)
        return (mk * np.log(1.0 / draws)) ** 2

    if dt == "weibull":
        k_shape, lam = float(params["k"]), float(params["lambda"])
        if u_c is not None:
            return lam * (-np.log(1.0 - u_c)) ** (1.0 / k_shape)
        return lam * gen.weibull(k_shape, size=size)

    if dt == "exponential":
        lam = float(params["lambda"])
        if u_c is not None:
            return -np.log(1.0 - u_c) / lam
        return gen.exponential(1.0 / lam, size=size)

    if dt == "von_mises":
        mu, kappa = float(params.get("mu", 0.0)), float(params["kappa"])
        if u_c is not None:
            return sp_stats.vonmises.ppf(u_c, kappa, loc=mu)
        return gen.vonmises(mu, kappa, size=size)

    if dt in ("empirical", "kde"):
        data = np.asarray(params["data"], dtype=np.float64)
        bw = float(params.get("bandwidth", 0.0))
        if u_c is not None:
            data_src = np.sort(data)
            idx = np.floor(u_c * len(data_src)).astype(int).clip(0, len(data_src) - 1)
        else:
            data_src = data
            idx = gen.integers(0, len(data_src), size=size)
        res = data_src[idx]
        if bw > 0.0:
            res = res + gen.normal(0.0, bw, size=res.shape)
        return res

    raise ValueError(f"Unsupported distribution type '{dist_type}'.")
