"""Epistemic Rigor (`V1`-`V4`), Closed-Form Salvo Oracles & Workbook V&V Matrix (`v2.0.0`).

Provides:
  - V1-V4 weakest-link verification depth propagation (never falsely labels outputs validated)
  - Contextual Counter-Evidence & Open Gaps registry for inline UI disclosures
  - Pellet-count physical mass consistency audit (SE-F14 / SE-C17)
  - Exact closed-form single-shot & shared/independent salvo brackets (WB-F01 / WB-F02 / HP-F25)
  - 3-level hierarchical engagement/gun/shot correlation & Negative-Binomial clumping (HP-F35 / HP-F36 / PD-F11)
  - Optimal uniform-disc & Gaussian pattern radius guardrails (HP-F33 / HP-F34 / HP-T07)
  - Usable closing-target engagement window [R_min_safe, R_max] with wreck carry-on inertia (DT-F09 / SE-F06)
  - Binomial statistical intervals (Wilson, Clopper-Pearson, Agresti-Coull, alpha-spending)
  - Deterministic Workbook Validation_Cases & Calc_Checks regression suite (38+ oracles)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import integrate, optimize, stats as sp_stats
from scipy.special import gammainc, gammaln, i0

from ammo_mc.cuas_ballistics_v2 import (
    cipm_2007_air_density,
    compute_saami_recoil_impulse,
    integrate_pellet_trajectory_rk4,
    integrate_pellet_trajectory_wb_f05,
    tropospheric_pressure_pa,
    velocity_ogive_peff,
    wen_2000_frp_ballistic_limit,
)

# -------------------------------------------------------------------------
# 1. VERIFICATION DEPTH RUBRIC (V1-V4) & WEAKEST-LINK PROPAGATION
# -------------------------------------------------------------------------

VERIFICATION_RUBRIC: dict[str, dict[str, Any]] = {
    "V1": {
        "grade": "V1",
        "rank": 1,
        "label": "V1 — Primary Source Verified",
        "color": "#16A34A",
        "summary": "Value or equation verified directly in primary peer-reviewed or standards text.",
    },
    "V2": {
        "grade": "V2",
        "rank": 2,
        "label": "V2 — Secondary Literature",
        "color": "#2563EB",
        "summary": "Obtained from reputable secondary citation, abstract, or calibrated fit.",
    },
    "V3": {
        "grade": "V3",
        "rank": 3,
        "label": "V3 — Handbook / Extrapolated",
        "color": "#D97706",
        "summary": "Standard textbook correlation or extrapolated outside original fragment/caliber regime.",
    },
    "V4": {
        "grade": "V4",
        "rank": 4,
        "label": "V4 — Analyst Estimate (Uncalibrated)",
        "color": "#DC2626",
        "summary": "Analyst engineering placeholder; uncalibrated against live-fire range trials.",
    },
}


def resolve_verification_grade(
    component_grades: dict[str, str],
    calibrated: bool = False,
    calibrated_keys: tuple[str, ...] = ("dispersion", "muzzle_velocity", "drag"),
) -> dict[str, Any]:
    """Propagate weakest verification grade across active pipeline components (`V1` best -> `V4` weakest)."""
    resolved: dict[str, str] = {}
    worst_rank = 1
    worst_grade = "V1"
    bottlenecks: list[str] = []

    for comp, raw_g in component_grades.items():
        g = str(raw_g).upper().strip()
        if g not in VERIFICATION_RUBRIC:
            g = "V4"
        if calibrated and comp in calibrated_keys and VERIFICATION_RUBRIC[g]["rank"] > 2:
            g = "V2"
        resolved[comp] = g
        rk = int(VERIFICATION_RUBRIC[g]["rank"])
        if rk > worst_rank:
            worst_rank = rk
            worst_grade = g
            bottlenecks = [comp]
        elif rk == worst_rank:
            bottlenecks.append(comp)

    meta = dict(VERIFICATION_RUBRIC[worst_grade])
    meta["component_grades"] = resolved
    meta["weakest_components"] = bottlenecks
    meta["calibrated_twin_applied"] = bool(calibrated)
    meta["disclaimer"] = (
        "Analytical/Monte Carlo engineering estimate graded by weakest link. "
        "Closed-form verification does not establish live-fire empirical validity."
    )
    return meta


def check_pellet_count_consistency(
    payload_mass_g: float,
    diam_mm: float,
    density_kg_m3: float,
    claimed_pellets: int,
    tol_pct: float = 12.0,
) -> dict[str, Any]:
    """Check physical mass conservation between payload mass, sphere geometry, and pellet count (`SE-F14`)."""
    d_m = max(1e-4, float(diam_mm) / 1000.0)
    m_p_g = float(density_kg_m3) * math.pi * (d_m ** 3) / 6.0 * 1000.0
    n_phys = max(1e-6, float(payload_mass_g)) / max(m_p_g, 1e-9)
    n_claimed = max(1, int(claimed_pellets))
    discrepancy_pct = abs(n_claimed - n_phys) / max(n_phys, 1e-6) * 100.0
    consistent = discrepancy_pct <= float(tol_pct)
    return {
        "pellet_mass_g": round(m_p_g, 4),
        "physical_pellet_count": round(n_phys, 2),
        "claimed_pellet_count": n_claimed,
        "discrepancy_pct": round(discrepancy_pct, 2),
        "consistent": bool(consistent),
        "warning": (
            None
            if consistent
            else (
                f"SE-C17 physical consistency flag: claimed {n_claimed} pellets vs. "
                f"{n_phys:.1f} expected from {payload_mass_g:.1f} g at {diam_mm:.2f} mm ({discrepancy_pct:.1f}% gap)."
            )
        ),
    }


# -------------------------------------------------------------------------
# 2. COUNTER-EVIDENCE & OPEN LITERATURE GAPS REGISTRY
# -------------------------------------------------------------------------

COUNTER_EVIDENCE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "WB-C01",
        "domain": "Engagement / Defaults",
        "grade": "V4",
        "title": "Plastic-shell quadcopter defaults are more favorable than hardened CFRP FPV evidence",
        "detail": "5 J plastic-shell benchmarks overstate kill probability against woven 3 mm CFRP FPV frames (~16-21 J steel V_bl).",
        "mitigation": "Switch target preset to CFRP-Frame FPV and enable Wen (2000) / V50 perforation gating.",
        "stages": ["array", "effective_range", "comparator"],
    },
    {
        "id": "HP-C01",
        "domain": "Salvo Correlation",
        "grade": "V1",
        "title": "Independent-shot binomial 1-(1-Pk)^n overstates multi-barrel salvo Pk under shared tracker bias",
        "detail": "When barrel aimpoints share tracker, servo, or unmodeled maneuver errors, all rounds miss together toward the shared-error lower bound.",
        "mitigation": "Evaluate both shared-error lower and independent-error upper brackets (`WB-F02`).",
        "stages": ["array", "effective_range", "comparator"],
    },
    {
        "id": "HP-C05",
        "domain": "Patterning & Splay",
        "grade": "V1",
        "title": "Excessive barrel splay lowers core Pk whenever aim error sigma_a is small",
        "detail": "Deliberately diverging barrels past the HP-F33 optimal pattern radius dilutes central pellet density and reduces Peak Pk.",
        "mitigation": "Compare current pattern spread against the closed-form optimal radius R_p* = sigma_a * sqrt(2*sqrt(c)).",
        "stages": ["array", "fcs", "effective_range"],
    },
    {
        "id": "EB-C01",
        "domain": "Exterior Ballistics",
        "grade": "V1",
        "title": "Smooth-sphere drag underestimates real deformed lead pellet slowdown past 25 m",
        "detail": "In-bore setback & choke scrub deform soft lead spheres (+10-30% Cd; Compton trailing lead +23.4%).",
        "mitigation": "Inspect sensitivity with Compton trailing drag or k_d = 1.15-1.30 on soft lead.",
        "stages": ["array", "effective_range", "sdz"],
    },
    {
        "id": "EB-C08",
        "domain": "Exterior Ballistics",
        "grade": "V1",
        "title": "Transonic sphere drag correlations diverge around M=0.8-1.1 without Walsh/NOL lookup",
        "detail": "Analytical approximations with misprinted logarithms overpredict transonic drag; Walsh (1976) / NOLTR 72-34 tables track measured wind-tunnel curves.",
        "mitigation": "Select Walsh (1976) or NOL (1972) continuum tables for shotshell M=0.5-1.3.",
        "stages": ["array", "effective_range"],
    },
    {
        "id": "PD-C01",
        "domain": "Patterning",
        "grade": "V2",
        "title": "Choked shotgun patterns show spatial clumping & shot-to-shot sigma CV (10-20%)",
        "detail": "Single-shot Poisson assumptions neglect negative-binomial pellet clustering (kappa < inf) and round-to-round spread scatter.",
        "mitigation": "Enable Shot-to-Shot Pattern CV and overdispersed clumping diagnostics.",
        "stages": ["array", "effective_range"],
    },
    {
        "id": "TE-C01",
        "domain": "Terminal Effects",
        "grade": "V1",
        "title": "Project THOR TR-47 equations assume rigid steel fragments above 366 m/s on Aluminum",
        "detail": "Applying rigid-fragment equations to soft lead pellets at shotshell impact speeds (~220 m/s) overpredicts metallic perforation.",
        "mitigation": "Restrict THOR metallic checks to hard tungsten/steel pellets and observe TE-C01 velocity floor flags.",
        "stages": ["array", "effective_range"],
    },
    {
        "id": "TE-C09",
        "domain": "Terminal Effects",
        "grade": "V3",
        "title": "FAA shear-plugging Cs constants were fitted to 244-800 g rotor blade fragments",
        "detail": "Extrapolating macro-fragment shear moduli to 0.2-1.5 g shotgun pellets carries uncalibrated thickness-scaling uncertainty.",
        "mitigation": "Grade FAA shear-plugging outputs V3 unless calibrated by witness-plate test firings.",
        "stages": ["array"],
    },
    {
        "id": "FC-C01",
        "domain": "Fire Control",
        "grade": "V1",
        "title": "End-to-end serial vision+actuator dead time (~141 ms) drives quadratic maneuver miss",
        "detail": "Edge-GPU vision pipelines (84-195 ms p99) plus solenoid/lock/ignition delays add ~141 ms before shot exit; miss grows as 0.5*a*(t_dead+TOF)^2.",
        "mitigation": "Inspect the full serial dead-time chain rather than optimistic frame-rate-only latency.",
        "stages": ["fcs", "effective_range", "comparator"],
    },
    {
        "id": "FC-C04",
        "domain": "Tracking & Radar",
        "grade": "V2",
        "title": "AI optical ATR centroids exhibit heavy outlier tails (up to 14 mrad) & radar suffers 25-35 dB aspect nulls",
        "detail": "Gaussian tracker jitter models miss intermittent ATR false-centroid jumps and oblique RCS drop-outs.",
        "mitigation": "Enable Heavy-Tailed ATR Noise and Aspect RCS Nulls in Stage 2 FCS stress tests.",
        "stages": ["fcs"],
    },
    {
        "id": "DT-C06",
        "domain": "Engagement Window",
        "grade": "V1",
        "title": "Ballistic wreck carry-on inertia limits safe close-in intercept of fast FPVs",
        "detail": "A killed 25-35 m/s kamikaze drone continues along a ballistic parabola x = v*sqrt(2h/g) plus warhead blast radius, collapsing low-range windows.",
        "mitigation": "Compute R_min_safe = x_wreck + R_blast and never count sub-floor kills toward safe defense.",
        "stages": ["effective_range", "fcs"],
    },
    {
        "id": "SE-C15",
        "domain": "Range Safety",
        "grade": "V1",
        "title": "Falling shot remains an ocular hazard well past lethal kinetic energy ranges",
        "detail": "Kneubuehl (~3.0 J/cm^2) ocular and (~10.0 J/cm^2) skin criteria trigger well below RCC 14.91 J blunt casualty thresholds.",
        "mitigation": "Enforce Dual RCC + Kneubuehl criteria plus AR 385-63 / JSP 403 fan boundaries in Stage 3 SDZ.",
        "stages": ["sdz"],
    },
    {
        "id": "SE-C20",
        "domain": "Range Safety",
        "grade": "V1",
        "title": "Downed kamikaze UAS carry post-kill UXO detonation & battery thermal-runaway risk",
        "detail": "Even after rotor/link kill, intact warheads remain armed downrange until EOD clearance.",
        "mitigation": "Maintain post-kill debris Exclusion radius around impact ellipse (`SE-F04`).",
        "stages": ["sdz", "effective_range"],
    },
    {
        "id": "MC-C01",
        "domain": "Monte Carlo Statistics",
        "grade": "V1",
        "title": "Low trial counts (< 100 volleys) induce wide binomial uncertainty (+/- 10-15%)",
        "detail": "Achieving +/- 1% half-width at 95% confidence worst-case p=0.5 requires n = 9,600 trials (MC-F01).",
        "mitigation": "Pair small-sample MC runs with Wilson/Clopper-Pearson CIs and closed-form WB-F01 cross-checks.",
        "stages": ["array", "fcs", "comparator"],
    },
]

LITERATURE_GAPS_CATALOG: list[dict[str, str]] = [
    {"gap_id": "EB-G09", "domain": "Exterior", "gap": "Lack of measured buckshot Doppler radar velocity beyond 12 m.", "action": "Conduct multi-station Doppler chronograph trials across 15-60 m."},
    {"gap_id": "EB-G10", "domain": "Exterior", "gap": "Limited high-speed video measurements of plastic shot-cup separation distance.", "action": "Capture muzzle-to-15 m spark-shadowgraph sequence by choke."},
    {"gap_id": "AM-G01", "domain": "Ammunition", "gap": "Scarce lot-to-lot MV SD measurements on specialized tungsten C-UAS shells.", "action": "Sample >=20 rounds per lot using radar chronograph (SAAMI 3 ft / 8 ft)."},
    {"gap_id": "TE-G02", "domain": "Terminal", "gap": "Few open-literature V50 curves comparing 3 mm CFRP vs. consumer ABS shells for 3-6 mm shot.", "action": "Execute witness-plate ballistic limit testing per NIJ/STANAG."},
    {"gap_id": "FC-G03", "domain": "Fire Control", "gap": "Insufficient empirical multi-barrel gimbal jitter measurements under rapid ripple fire.", "action": "Record IMU pitch/yaw walk at 2 kHz across ripple intervals."},
    {"gap_id": "SE-G01", "domain": "Safety", "gap": "Limited experimental validation of post-kill UXO impact state vs. standoff burst height.", "action": "Track intact payload survival probability after rotor/ESC kill."},
]


def get_contextual_counter_evidence(
    stage: str = "array",
    *,
    range_m: float = 40.0,
    drone_class: str = "fpv_cfrp",
    n_barrels: int = 1,
    splay_mrad: float = 0.0,
    drag_law: str = "compressible_re_m",
    calibrated: bool = False,
    n_mc_trials: int = 25,
) -> list[dict[str, Any]]:
    """Return prioritized counter-evidence items relevant to the user's current context."""
    items = [dict(it) for it in COUNTER_EVIDENCE_CATALOG if stage in it.get("stages", [])]
    if not items:
        items = [dict(it) for it in COUNTER_EVIDENCE_CATALOG[:4]]

    def _priority(it: dict[str, Any]) -> int:
        score = 0
        cid = it["id"]
        if cid == "HP-C01" and int(n_barrels) > 1:
            score += 4
        if cid == "HP-C05" and float(splay_mrad) > 0.5:
            score += 4
        if cid == "EB-C01" and float(range_m) >= 25.0:
            score += 3
        if cid == "WB-C01" and "plastic" in str(drone_class).lower():
            score += 5
        if cid == "MC-C01" and int(n_mc_trials) < 500:
            score += 4
        if cid == "EB-C08" and drag_law not in ("walsh_1976", "nol_1972"):
            score += 2
        if not calibrated and it["grade"] in ("V3", "V4"):
            score += 2
        return -score

    items.sort(key=_priority)
    return items[:6]


# -------------------------------------------------------------------------
# 3. EXACT CLOSED-FORM SALVO BRACKETS & HIERARCHICAL / PATTERN ORACLES
# -------------------------------------------------------------------------

def pk_single_shot_closed_form(lambda0: float, sigma_m: float, sigma_a_m: float) -> float:
    """Exact closed-form single-shot kill probability with Gaussian pattern & aim error (`WB-F01`).

    Pk = 1 - c * lambda0^(-c) * gamma(c, lambda0), where c = (sigma / sigma_a)^2
    and gamma(c, lambda0) = Gamma(c) * gammainc(c, lambda0).
    For c > 150 uses exact asymptotic limit 1 - exp(-lambda0 * c / (c + 1)).
    """
    lam0 = max(0.0, float(lambda0))
    if lam0 <= 0.0:
        return 0.0
    sig = max(1e-9, float(sigma_m))
    sig_a = max(0.0, float(sigma_a_m))
    if sig_a <= 1e-12:
        return float(1.0 - math.exp(-lam0))
    c = (sig / sig_a) ** 2
    if c > 150.0:
        return float(1.0 - math.exp(-lam0 * c / (c + 1.0)))
    if c < 1e-12:
        return 0.0
    log_pref = math.log(c) - c * math.log(lam0) + float(gammaln(c))
    # gammainc(c, x) in scipy is normalized lower incomplete gamma P(c, x) = gamma(c, x)/Gamma(c)
    p_low = float(gammainc(c, lam0))
    val = 1.0 - math.exp(log_pref) * p_low
    return float(max(0.0, min(1.0, val)))


def pk_salvo_brackets(lambda0: float, sigma_m: float, sigma_a_m: float, n_shots: int) -> dict[str, float]:
    """Compute exact salvo brackets: Shared-Error Lower (`WB-F02`) vs Independent-Error Upper (`HP-F25`)."""
    n = max(1, int(n_shots))
    pk0 = float(1.0 - math.exp(-max(0.0, float(lambda0))))
    pk1 = pk_single_shot_closed_form(lambda0, sigma_m, sigma_a_m)
    pk_shared = pk_single_shot_closed_form(n * max(0.0, float(lambda0)), sigma_m, sigma_a_m)
    pk_indep = float(1.0 - (1.0 - pk1) ** n)
    return {
        "n_shots": n,
        "pk_perfect_aim": round(pk0, 6),
        "pk_single": round(pk1, 6),
        "pk_shared_lower": round(pk_shared, 6),
        "pk_independent_upper": round(pk_indep, 6),
        "optimism_gap_pct": round((pk_indep - pk_shared) * 100.0, 2),
        "verification_grade": "V1",
    }


def optimal_pattern_radius_closed_form(
    n_pellets: float,
    vulnerable_area_m2: float,
    sigma_a_m: float,
    *,
    current_sigma_p_m: float | None = None,
    splay_angle_mrad: float = 0.0,
) -> dict[str, Any]:
    """Closed-form optimal uniform-disc pattern (`HP-F33`) & optimal Gaussian pattern (`HP-F34`)."""
    n_p = max(1.0, float(n_pellets))
    a_v = max(1e-6, float(vulnerable_area_m2))
    sa = max(1e-4, float(sigma_a_m))

    c_par = (n_p * a_v) / (2.0 * math.pi * (sa ** 2))
    r_opt_disc = sa * math.sqrt(2.0 * math.sqrt(c_par))
    pk_opt_disc = (1.0 - math.exp(-math.sqrt(c_par))) ** 2

    def _neg_gauss_pk(s: float) -> float:
        lam = (n_p * a_v) / (2.0 * math.pi * (s ** 2))
        return -pk_single_shot_closed_form(lam, s, sa)

    res = optimize.minimize_scalar(_neg_gauss_pk, bounds=(0.02 * sa, 12.0 * sa), method="bounded")
    sigma_opt_gauss = float(res.x)
    pk_opt_gauss = float(-res.fun)

    splay_warning: str | None = None
    if current_sigma_p_m is not None and current_sigma_p_m > 1.35 * sigma_opt_gauss and float(splay_angle_mrad) > 0.5:
        splay_warning = (
            f"HP-C05 pattern dilution guardrail: active sigma_p ({current_sigma_p_m:.2f} m) with "
            f"{splay_angle_mrad:.1f} mrad splay exceeds HP-F34 optimal Gaussian spread ({sigma_opt_gauss:.2f} m) "
            f"at aim sigma_a={sa:.2f} m."
        )

    return {
        "c_param": float(c_par),
        "r_opt_disc_m": round(r_opt_disc, 4),
        "pk_opt_disc": round(pk_opt_disc, 6),
        "sigma_opt_gauss_m": round(sigma_opt_gauss, 4),
        "pk_opt_gauss": round(pk_opt_gauss, 6),
        "splay_warning": splay_warning,
        "verification_grade": "V1",
    }


def pk_salvo_hierarchical_mc(
    n_pellets: int,
    vulnerable_area_m2: float,
    sigma_p_m: float,
    sigma_common_m: float,
    sigma_gun_m: float,
    sigma_shot_m: float,
    *,
    n_guns: int = 4,
    shots_per_gun: int = 1,
    ar1_rho: float = 0.0,
    nb_kappa: float | None = None,
    shot_sigma_cv: float = 0.0,
    n_trials: int = 20_000,
    seed: int = 42,
) -> dict[str, float]:
    """3-level hierarchical engagement/gun/shot error model (`HP-F35`, `HP-F36`) + NB clumping (`PD-F11`)."""
    rng = np.random.default_rng(seed)
    n_tr = max(200, int(n_trials))
    g_cnt = max(1, int(n_guns))
    k_cnt = max(1, int(shots_per_gun))
    sp0 = max(1e-3, float(sigma_p_m))
    a_v = max(1e-6, float(vulnerable_area_m2))
    rho = max(-0.95, min(0.95, float(ar1_rho)))

    # Level 1: Engagement-common track/wind/bias error (n_tr, 1, 1, 2)
    e_c = rng.normal(0.0, max(0.0, float(sigma_common_m)), size=(n_tr, 1, 1, 2))
    # Level 2: Per-gun boresight error (n_tr, g_cnt, 1, 2)
    e_g = rng.normal(0.0, max(0.0, float(sigma_gun_m)), size=(n_tr, g_cnt, 1, 2))
    # Level 3: Per-shot AR(1) residual error (n_tr, g_cnt, k_cnt, 2)
    e_s = np.zeros((n_tr, g_cnt, k_cnt, 2), dtype=np.float64)
    s_s = max(0.0, float(sigma_shot_m))
    if s_s > 0.0:
        e_s[:, :, 0, :] = rng.normal(0.0, s_s, size=(n_tr, g_cnt, 2))
        innov_std = s_s * math.sqrt(max(0.0, 1.0 - rho * rho))
        for k in range(1, k_cnt):
            e_s[:, :, k, :] = rho * e_s[:, :, k - 1, :] + rng.normal(0.0, innov_std, size=(n_tr, g_cnt, 2))

    miss = e_c + e_g + e_s
    r2 = np.sum(miss * miss, axis=-1)  # (n_tr, g_cnt, k_cnt)

    if shot_sigma_cv > 0.0:
        sp = sp0 * np.clip(rng.normal(1.0, float(shot_sigma_cv), size=r2.shape), 0.35, 2.5)
    else:
        sp = sp0

    lam_shots = (float(n_pellets) * a_v / (2.0 * math.pi * (sp ** 2))) * np.exp(-r2 / (2.0 * (sp ** 2)))
    if nb_kappa is not None and float(nb_kappa) > 0.0:
        kap = float(nb_kappa)
        lam_shots = lam_shots * rng.gamma(shape=kap, scale=1.0 / kap, size=lam_shots.shape)

    lam_tot = np.sum(lam_shots, axis=(1, 2))
    p_kill_arr = 1.0 - np.exp(-lam_tot)
    pk_mean = float(np.mean(p_kill_arr))
    se = float(np.std(p_kill_arr, ddof=1) / math.sqrt(n_tr))
    return {
        "pk_hierarchical": round(pk_mean, 6),
        "mc_se": round(se, 6),
        "n_guns": g_cnt,
        "shots_per_gun": k_cnt,
    }


# -------------------------------------------------------------------------
# 4. USABLE ENGAGEMENT WINDOW [R_min_safe, R_max] WITH WRECK INERTIA
# -------------------------------------------------------------------------

def evaluate_closing_engagement_window(
    drone_speed_mps: float = 25.0,
    drone_altitude_m: float = 15.0,
    warhead_blast_radius_m: float = 8.0,
    range_min_m: float = 5.0,
    range_max_m: float = 90.0,
    step_m: float = 5.0,
    n_barrels: int = 4,
    salvo_interval_s: float = 0.65,
    ammo_preset: str = "12G_NO4_BUCK_C_UAS",
    choke: str = "Modified",
    drone_class: str = "fpv_cfrp",
    aim_error_mrad: float = 2.0,
    dead_time_s: float = 0.141,
    a_unmod_mps2: float = 5.0,
    pk_threshold: float = 0.50,
    drag_law: str = "walsh_1976",
    k_d: float = 1.0,
    t_react_s: float = 0.0,
    site_altitude_m: float = 0.0,
) -> dict[str, Any]:
    """Compute closing-target usable window [R_min_safe, R_max] with wreck carry-on inertia (`DT-F09`/`SE-F06`)."""
    from ammo_mc.small_arms_pkg.catalogs import CHOKE_DEFINITIONS, SHOTGUN_PRESETS, resolve_pellet_density_g_cm3

    preset = SHOTGUN_PRESETS.get(ammo_preset, SHOTGUN_PRESETS["12G_NO4_BUCK_C_UAS"])
    choke_info = CHOKE_DEFINITIONS.get(choke, CHOKE_DEFINITIONS["Modified"])
    p40 = float(choke_info["benchmark_40yd_30in_pct"])
    sigma_ang_rad = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(max(1e-6, 1.0 - p40)))

    d_mm = float(preset["pellet_diameter_mm"])
    m_g = float(preset["pellet_mass_g"])
    n_pellets = int(preset["pellets_per_shot"])
    v0 = float(preset["v0_mps"])
    rho_p = resolve_pellet_density_g_cm3(str(preset.get("material", "lead")), preset.get("material_density_g_cm3")) * 1000.0
    station_p_pa = tropospheric_pressure_pa(site_altitude_m)

    # Target vulnerable area & perforation threshold: explicit distinction between plastic consumer vs CFRP FPV
    if "plastic" in str(drone_class).lower():
        a_v_m2 = 0.0080
        e50_j = 5.0
        target_grade = "V4 (Plastic-Shell Anchor)"
        v50_mps = math.sqrt(2.0 * e50_j / max(m_g / 1000.0, 1e-6))
    else:
        # 3 mm CFRP frame benchmark (Wen 2000)
        a_v_m2 = 0.0055
        wen = wen_2000_frp_ballistic_limit(m_g / 1000.0, d_mm / 1000.0, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
        v50_mps = float(wen["v_bl_mps"])
        e50_j = float(wen["e_bl_j"])
        target_grade = "V3 (Wen 2000 CFRP FPV)"

    # Wreck carry-on inertia floor (DT-F09 / SE-F06): x_wreck = v * sqrt(2*h/g)
    v_dr = max(0.5, float(drone_speed_mps))
    h_dr = max(1.0, float(drone_altitude_m))
    x_wreck_m = v_dr * math.sqrt(2.0 * h_dr / 9.80665)
    r_min_safe_m = x_wreck_m + max(0.0, float(warhead_blast_radius_m))

    ranges = np.arange(float(range_min_m), float(range_max_m) + 1e-9, max(1.0, float(step_m)))
    stations: list[dict[str, Any]] = []
    usable_shared_pks: list[float] = []
    usable_indep_pks: list[float] = []

    for r_m in ranges:
        traj = integrate_pellet_trajectory_rk4(
            v0_mps=v0,
            diam_mm=d_mm,
            density_kg_m3=rho_p,
            range_m=float(r_m),
            drag_law=drag_law,
            k_d=k_d,
            pressure_pa=station_p_pa,
            pellet_mass_g=m_g,
            n_steps=max(20, int(r_m)),
        )
        v_ret = traj["v_retained_mps"]
        tof_s = traj["tof_s"]
        p_eff = velocity_ogive_peff(v_ret, v50_mps, spread_s_v=0.10)
        sigma_p = max(1e-3, sigma_ang_rad * float(r_m))
        aim_ang_m = (float(aim_error_mrad) / 1000.0) * float(r_m)
        aim_man_m = 0.5 * float(a_unmod_mps2) * ((float(dead_time_s) + tof_s) ** 2)
        sigma_a = math.hypot(aim_ang_m, aim_man_m)

        lam0 = (n_pellets / (2.0 * math.pi * (sigma_p ** 2))) * a_v_m2 * p_eff
        br = pk_salvo_brackets(lam0, sigma_p, sigma_a, n_shots=n_barrels)
        in_safe_window = float(r_m) >= r_min_safe_m and br["pk_shared_lower"] >= float(pk_threshold)
        if in_safe_window:
            usable_shared_pks.append(br["pk_shared_lower"])
            usable_indep_pks.append(br["pk_independent_upper"])

        stations.append({
            "range_m": round(float(r_m), 1),
            "v_retained_mps": round(v_ret, 1),
            "tof_s": round(tof_s, 4),
            "energy_j": round(traj["energy_j"], 2),
            "p_eff": round(p_eff, 4),
            "lam0": float(lam0),
            "sigma_p_m": round(sigma_p, 3),
            "sigma_a_m": round(sigma_a, 3),
            "pk_single_pct": round(br["pk_single"] * 100.0, 1),
            "pk_shared_lower_pct": round(br["pk_shared_lower"] * 100.0, 1),
            "pk_indep_upper_pct": round(br["pk_independent_upper"] * 100.0, 1),
            "in_safe_usable_window": in_safe_window,
            "wreck_exclusion": float(r_m) < r_min_safe_m,
        })

    qualifying = [s["range_m"] for s in stations if s["pk_shared_lower_pct"] >= pk_threshold * 100.0]
    r_max_shared = max(qualifying) if qualifying else 0.0
    qualifying_indep = [s["range_m"] for s in stations if s["pk_indep_upper_pct"] >= pk_threshold * 100.0]
    r_max_indep = max(qualifying_indep) if qualifying_indep else 0.0

    usable_width_m = max(0.0, r_max_shared - r_min_safe_m)
    t_react = max(0.0, float(t_react_s))
    t_corridor_s = usable_width_m / v_dr
    usable_time_s = max(0.0, t_corridor_s - t_react)
    dt_salvo = max(0.15, float(salvo_interval_s))
    n_salvos_avail = (
        int(math.floor(usable_time_s / dt_salvo)) + 1
        if (usable_width_m > 0.0 and t_corridor_s >= t_react)
        else 0
    )

    # Cumulative Pk over usable closing trajectory:
    # - Shared-error lower bound (`WB-F02`) integrates a single shared radial aim-error draw U ~ Rayleigh(1)
    #   across all salvo ranges so persistent bias does NOT falsely multiply independent survival probabilities.
    # - Independent upper bound (`HP-F25`) multiplies per-salvo survival probabilities.
    cum_shared = 0.0
    cum_indep = 0.0
    if usable_width_m > 0.0 and n_salvos_avail >= 1:
        r_start_eff = r_max_shared - v_dr * t_react
        r_shots = [
            r_start_eff - k * v_dr * dt_salvo
            for k in range(min(n_salvos_avail, 12))
        ]
        r_shots = [r for r in r_shots if r >= r_min_safe_m - 1e-6]
        if not r_shots:
            r_shots = [max(r_min_safe_m, r_max_shared)]

        salvo_params: list[tuple[float, float, float]] = []
        surv_i = 1.0
        for rs in r_shots:
            nearest = min(stations, key=lambda st: abs(st["range_m"] - float(rs)))
            surv_i *= 1.0 - nearest["pk_indep_upper_pct"] / 100.0
            salvo_params.append(
                (float(nearest["lam0"]), max(1e-3, float(nearest["sigma_p_m"])), max(0.0, float(nearest["sigma_a_m"])))
            )

        def _shared_surv_integrand(u: float) -> float:
            lam_sum = 0.0
            for lam_k, sp_k, sa_k in salvo_params:
                lam_sum += n_barrels * lam_k * math.exp(-0.5 * ((u * sa_k / sp_k) ** 2))
            return u * math.exp(-0.5 * u * u - lam_sum)

        surv_s, _ = integrate.quad(_shared_surv_integrand, 0.0, 12.0, epsabs=1e-9, epsrel=1e-8)
        cum_shared = max(0.0, min(100.0, (1.0 - surv_s) * 100.0))
        cum_indep = max(cum_shared, min(100.0, (1.0 - surv_i) * 100.0))

    window_collapsed = usable_width_m <= 0.0
    return {
        "summary": {
            "x_wreck_m": round(x_wreck_m, 2),
            "warhead_blast_radius_m": round(float(warhead_blast_radius_m), 1),
            "r_min_safe_m": round(r_min_safe_m, 1),
            "r_max_shared_m": round(r_max_shared, 1),
            "r_max_indep_m": round(r_max_indep, 1),
            "usable_window_m": round(usable_width_m, 1),
            "usable_window_time_s": round(usable_time_s, 2),
            "n_salvos_available": int(n_salvos_avail),
            "cum_pk_shared_pct": round(cum_shared, 1),
            "cum_pk_indep_pct": round(cum_indep, 1),
            "window_collapsed": bool(window_collapsed),
            "v50_mps": round(v50_mps, 1),
            "e50_j": round(e50_j, 2),
            "target_class_grade": target_grade,
            "drone_speed_mps": v_dr,
            "drone_altitude_m": h_dr,
        },
        "stations": stations,
    }


# -------------------------------------------------------------------------
# 5. STATISTICAL PRIMITIVES & WORKBOOK HIT-PROBABILITY ORACLES
# -------------------------------------------------------------------------

def compute_wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Wilson score binomial confidence interval (`MC-F01`)."""
    n_i = max(1, int(n))
    k_i = max(0, min(n_i, int(k)))
    p_hat = k_i / n_i
    z = float(sp_stats.norm.ppf(1.0 - float(alpha) / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n_i
    center = (p_hat + z2 / (2.0 * n_i)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n_i) + (z2 / (4.0 * n_i * n_i)))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def wilson_worst_case_sample_size(alpha: float = 0.05, half_width: float = 0.01) -> int:
    """Worst-case p=0.5 Wilson sample size for half-width <= eps (`MC-F01` / `VC-095`)."""
    z = float(sp_stats.norm.ppf(1.0 - float(alpha) / 2.0))
    eps = max(1e-5, float(half_width))
    # Workbook exact criterion solves n^2/(n+z^2) >= z^2/(4*eps^2) -> for alpha=0.05, eps=0.01 yields 9600
    n_exact = (z * z) * (1.0 / (4.0 * eps * eps) - 1.0)
    return int(math.ceil(n_exact))


def compute_clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson two-sided binomial interval (`MC-F02`)."""
    n_i = max(1, int(n))
    k_i = max(0, min(n_i, int(k)))
    a2 = float(alpha) / 2.0
    lo = 0.0 if k_i == 0 else float(sp_stats.beta.ppf(a2, k_i, n_i - k_i + 1))
    hi = 1.0 if k_i == n_i else float(sp_stats.beta.ppf(1.0 - a2, k_i + 1, n_i - k_i))
    return lo, hi


def compute_agresti_coull_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Generalized Agresti-Coull binomial interval (`MC-F03`)."""
    n_i = max(1, int(n))
    k_i = max(0, min(n_i, int(k)))
    z = float(sp_stats.norm.ppf(1.0 - float(alpha) / 2.0))
    z2 = z * z
    n_t = n_i + z2
    p_t = (k_i + 0.5 * z2) / n_t
    half = z * math.sqrt(p_t * (1.0 - p_t) / n_t)
    return float(max(0.0, p_t - half)), float(min(1.0, p_t + half))


def evaluate_alpha_spending_checkpoint(
    k_look: int,
    total_looks: int | None = None,
    alpha: float = 0.05,
    spending: str = "bonferroni",
) -> dict[str, float]:
    """Sequential fixed-checkpoint (`alpha/K`) or open-ended (`alpha/(k*(k+1))`) alpha spending (`MC-F05`)."""
    look = max(1, int(k_look))
    mode = str(spending).lower().strip()
    if mode == "infinite" or total_looks is None:
        per_look = float(alpha) / float(look * (look + 1))
        cum_alpha = float(alpha) * (float(look) / float(look + 1))
        return {"look": look, "total_looks": 0, "per_look_alpha": float(per_look), "cum_alpha": float(cum_alpha)}

    k_tot = max(1, int(total_looks))
    look = min(k_tot, look)
    t_frac = look / k_tot
    if mode == "pocock":
        cum_alpha = float(alpha) * math.log(1.0 + (math.e - 1.0) * t_frac)
        prev_alpha = float(alpha) * math.log(1.0 + (math.e - 1.0) * ((look - 1) / k_tot))
        per_look = cum_alpha - prev_alpha
    else:
        per_look = float(alpha) / k_tot
        cum_alpha = per_look * look
    return {"look": look, "total_looks": k_tot, "per_look_alpha": float(per_look), "cum_alpha": float(cum_alpha)}


def rice_offset_circle_prob(radius: float, sigma: float, offset_d: float = 0.0) -> float:
    """Exact hit probability on an offset circle via circular normal integration."""
    r = max(0.0, float(radius))
    s = max(1e-9, float(sigma))
    d = abs(float(offset_d))
    if d == 0.0:
        return float(1.0 - math.exp(-(r * r) / (2.0 * s * s)))
    val, _ = integrate.quad(
        lambda u: (u / (s * s)) * math.exp(-(u * u + d * d) / (2.0 * s * s)) * float(i0(u * d / (s * s))),
        0.0,
        r,
        epsabs=1e-11,
        epsrel=1e-10,
    )
    return float(val)


def ellipse_exact_prob(radius: float, sx: float, sy: float, x0: float = 0.0, y0: float = 0.0) -> float:
    """Exact probability that an offset elliptical normal falls inside a circle of radius R."""
    val, _ = integrate.quad(
        lambda y: (
            (1.0 / (math.sqrt(2.0 * math.pi) * sy))
            * math.exp(-((y - y0) ** 2) / (2.0 * sy * sy))
            * (
                float(sp_stats.norm.cdf((math.sqrt(max(0.0, radius * radius - y * y)) - x0) / sx))
                - float(sp_stats.norm.cdf((-math.sqrt(max(0.0, radius * radius - y * y)) - x0) / sx))
            )
        ),
        -radius,
        radius,
        epsabs=1e-10,
        epsrel=1e-9,
    )
    return float(val)


def ellipse_grubbs_wh_prob(radius: float, sx: float, sy: float, x0: float = 0.0, y0: float = 0.0) -> float:
    """Grubbs (1964) Patnaik Wilson-Hilferty chi-square approximation (`VC-057`, `VC-059`)."""
    mx = x0 / sx
    my = y0 / sy
    v_x = sx * sx
    v_y = sy * sy
    k1 = v_x * (1.0 + mx * mx) + v_y * (1.0 + my * my)
    k2 = 2.0 * (v_x * v_x * (1.0 + 2.0 * mx * mx) + v_y * v_y * (1.0 + 2.0 * my * my))
    m_g = k1
    v_g = k2
    c_ratio = v_g / (2.0 * m_g)
    nu = (2.0 * m_g * m_g) / v_g
    t_val = (radius * radius) / (c_ratio * nu)
    z_wh = (t_val ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * nu))) / math.sqrt(2.0 / (9.0 * nu))
    return float(sp_stats.norm.cdf(z_wh))


def washburn_sulr_solr_bounds(z: float = 2.0) -> dict[str, float]:
    """Washburn Upper Bound, SULR, and SOLR salvo approximations (`HP-F27..F29`, `VC-065`..`VC-067`)."""
    z_val = max(0.0, float(z))
    p_upper = 1.0 - math.exp(-z_val)
    p_sulr = (1.0 - math.exp(-math.sqrt(z_val))) ** 2
    s2z = math.sqrt(2.0 * z_val)
    p_solr = 1.0 - (1.0 + s2z) * math.exp(-s2z)
    return {
        "upper": float(p_upper),
        "sulr": float(p_sulr),
        "solr": float(p_solr),
    }


def thomas_gemmill_salvo_prob(s1: float, s2: float, a: float, pr: float, n_shots: int) -> float:
    """Thomas & Gemmill (1972) 2D circular-normal ring patterned salvo Pk (`HP-F31`, `VC-072`..`077`)."""
    # P = E_{U ~ N(0, s1^2 I2)} [ 1 - prod_i (1 - F_ncx2((a/s2)^2; df=2, nc=|U - a_i|^2/s2^2)) ]
    r_grid = np.linspace(0.0, 6.0 * s1, 45)
    dr = float(r_grid[1] - r_grid[0])
    th_grid = np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False)
    dth = 2.0 * math.pi / len(th_grid)

    rr, tt = np.meshgrid(r_grid, th_grid, indexing="ij")
    ux = rr * np.cos(tt)
    uy = rr * np.sin(tt)
    w_ray = (rr / (s1 * s1)) * np.exp(-(rr * rr) / (2.0 * s1 * s1)) * (dr * dth / (2.0 * math.pi))

    angles_i = np.linspace(0.0, 2.0 * math.pi, n_shots, endpoint=False)
    ax = pr * np.cos(angles_i)
    ay = pr * np.sin(angles_i)

    dx = ux[:, :, None] - ax[None, None, :]
    dy = uy[:, :, None] - ay[None, None, :]
    nc = (dx * dx + dy * dy) / (s2 * s2)
    df_x = (a / s2) ** 2
    p_hit = sp_stats.ncx2.cdf(df_x, 2, nc)
    p_kill_cond = 1.0 - np.prod(1.0 - p_hit, axis=2)
    return float(np.sum(w_ray * p_kill_cond))


def whistler_sas_prob(n_shots: int = 5, radius: float = 1.0, sigma: float = 1.0) -> float:
    """Whistler shoot-adjust-shoot rule exact closed-form equation (`HP-F37`, `VC-079`)."""
    s_sum = sum((i - 1) / i for i in range(2, max(2, int(n_shots) + 1)))
    return float(1.0 - math.exp(-((radius * radius) / (2.0 * sigma * sigma)) * s_sum))


def wilkening_ns(s: int = 2, k: float = 0.5) -> float:
    """Wilkening (1999) SLS transition threshold `n_s(k) = s + sum_{j=1..s-1} (1-k)^(-j)` (`HP-F39`)."""
    s_i = max(1, int(s))
    k_c = max(1e-6, min(1.0 - 1e-6, float(k)))
    return float(s_i + sum((1.0 - k_c) ** (-j) for j in range(1, s_i)))


def wilkening_n2(k: float = 0.5) -> float:
    """Wilkening (1999) SLS transition threshold `n_2(k) = 2 + 1/(1-k)` (`HP-F39`, `VC-088`)."""
    return wilkening_ns(2, k)


def halton_point(index_zero_based: int, bases: tuple[int, ...] = (2, 3)) -> list[float]:
    """Zero-based radical-inverse Halton point matching workbook `VC-103` (e.g., i=5 -> [5/8, 7/9])."""
    out: list[float] = []
    for b in bases:
        n = int(index_zero_based)
        f = 1.0
        r = 0.0
        while n > 0:
            f /= b
            r += f * (n % b)
            n //= b
        out.append(float(r))
    return out


def ellipse_exact_cep(sx: float, sy: float) -> float:
    """Exact elliptical 50%-CEP radius via Brent root of `ellipse_exact_prob` (`HP-F09`, `VC-060`)."""
    s_hi = max(float(sx), float(sy), 1e-6)
    return float(optimize.brentq(lambda r: ellipse_exact_prob(r, sx, sy) - 0.5, 1e-6, 4.0 * s_hi, xtol=1e-10))


def rectangular_hit_prob(
    width_a: float,
    height_b: float,
    sx: float,
    sy: float,
    mx: float = 0.0,
    my: float = 0.0,
) -> float:
    """Exact independent bivariate normal hit probability over centered rectangle (`HP-F11`, `VC-061`)."""
    hx = 0.5 * float(width_a)
    hy = 0.5 * float(height_b)
    px = float(sp_stats.norm.cdf((hx - mx) / sx) - sp_stats.norm.cdf((-hx - mx) / sx))
    py = float(sp_stats.norm.cdf((hy - my) / sy) - sp_stats.norm.cdf((-hy - my) / sy))
    return float(max(0.0, min(1.0, px * py)))


def dg_single_shot_prob(
    bx: float,
    by: float,
    sx: float,
    sy: float,
    mx: float = 0.0,
    my: float = 0.0,
) -> float:
    """Diffuse Gaussian (DG) single-shot kill probability with offset (`HP-F16`, `VC-062`..`063`)."""
    px = (bx / math.sqrt(bx * bx + sx * sx)) * math.exp(-(mx * mx) / (2.0 * (bx * bx + sx * sx)))
    py = (by / math.sqrt(by * by + sy * sy)) * math.exp(-(my * my) / (2.0 * (by * by + sy * sy)))
    return float(px * py)


def carleton_expected_damage(
    rx: float,
    ry: float,
    d0: float,
    sx: float,
    sy: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    sx_shared: float = 0.0,
    sy_shared: float = 0.0,
) -> float:
    """Carleton elliptical damage function expectation with dispersion + offset (`HP-F17`, `VC-064`)."""
    gx2 = (rx ** 2) / (2.0 * d0) + (sx ** 2) + (sx_shared ** 2)
    gy2 = (ry ** 2) / (2.0 * d0) + (sy ** 2) + (sy_shared ** 2)
    return float(
        (rx * ry)
        / (2.0 * math.sqrt(gx2 * gy2))
        * math.exp(-(offset_x ** 2) / (2.0 * gx2) - (offset_y ** 2) / (2.0 * gy2))
    )


def washburn_square_pattern_prob(
    mode: str = "cookie",
    side: float = 11.7,
    sigma_u: float = 7.5,
    sigma_x: float = 1.0,
    radius: float = 7.5,
) -> float:
    """Washburn 4-shot square salvo Pk with shared U + independent X (`HP-F26`, `VC-068`..`VC-070`)."""
    half = 0.5 * float(side)
    corners = np.array([[-half, -half], [-half, half], [half, -half], [half, half]], dtype=np.float64)
    if str(mode).lower().strip() == "cookie":
        # Uniform 2D integration avoids Hermite aliasing on hard cookie-cutter boundaries.
        bound = 4.8 * float(sigma_u)
        g = np.linspace(-bound, bound, 161)
        dg = float(g[1] - g[0])
        ux, uy = np.meshgrid(g, g, indexing="ij")
        ww = (1.0 / (2.0 * math.pi * (sigma_u ** 2))) * np.exp(-(ux * ux + uy * uy) / (2.0 * (sigma_u ** 2))) * dg * dg
        dx = ux[:, :, None] - corners[None, None, :, 0]
        dy = uy[:, :, None] - corners[None, None, :, 1]
        nc = (dx * dx + dy * dy) / (sigma_x ** 2)
        p_hit = sp_stats.ncx2.cdf((radius / sigma_x) ** 2, 2, nc)
        return float(np.sum(ww * (1.0 - np.prod(1.0 - p_hit, axis=2))))

    b_dg = float(radius) / math.sqrt(2.0)
    x_gh, w_gh = np.polynomial.hermite.hermgauss(25)
    u_1d = math.sqrt(2.0) * float(sigma_u) * x_gh
    w_1d = w_gh / math.sqrt(math.pi)
    ux, uy = np.meshgrid(u_1d, u_1d, indexing="ij")
    ww = np.outer(w_1d, w_1d)
    dx = ux[:, :, None] - corners[None, None, :, 0]
    dy = uy[:, :, None] - corners[None, None, :, 1]
    denom = b_dg * b_dg + sigma_x * sigma_x
    p_dg = (b_dg * b_dg / denom) * np.exp(-(dx * dx + dy * dy) / (2.0 * denom))
    return float(np.sum(ww * (1.0 - np.prod(1.0 - p_dg, axis=2))))


def washburn_2003_dg_inclusion_exclusion(
    rho_list: list[float] | tuple[float, ...],
    s_list: list[np.ndarray],
    sigma_list: list[np.ndarray],
    t_mat: np.ndarray,
    mu_list: list[np.ndarray] | None = None,
) -> float:
    """Exact Washburn (2003) 2^n-1 subset inclusion-exclusion for correlated DG weapons (`HP-F30`, `VC-071`)."""
    n_w = len(rho_list)
    dim = t_mat.shape[0]
    eye = np.eye(dim, dtype=np.float64)
    c_i: list[float] = []
    r_i: list[np.ndarray] = []
    m_vec = [np.zeros(dim, dtype=np.float64) if mu_list is None else np.asarray(mu_list[i], dtype=np.float64) for i in range(n_w)]

    for i in range(n_w):
        s_mat = np.asarray(s_list[i], dtype=np.float64)
        sig_mat = np.asarray(sigma_list[i], dtype=np.float64)
        a_mat = eye + sig_mat @ s_mat
        det_a = float(np.linalg.det(a_mat))
        r_mat = s_mat @ np.linalg.inv(a_mat)
        mu = m_vec[i]
        p_fac = float(rho_list[i]) * (det_a ** -0.5) * math.exp(-0.5 * float(mu.T @ r_mat @ mu))
        c_i.append(p_fac)
        r_i.append(r_mat)

    total_q = 0.0
    for mask in range(1, 1 << n_w):
        bits = [i for i in range(n_w) if (mask >> i) & 1]
        sign = 1.0 if (len(bits) % 2 == 1) else -1.0
        c_u = 1.0
        m0_u = np.zeros((dim, dim), dtype=np.float64)
        m_lin = np.zeros(dim, dtype=np.float64)
        m_quad = 0.0
        for i in bits:
            c_u *= c_i[i]
            m0_u = m0_u + r_i[i]
            # Inclusion-exclusion for zero-mean aimpoint or shifted aimpoint
            m_lin = m_lin + r_i[i] @ m_vec[i]
            m_quad += 0.0
        b_mat = eye + m0_u @ t_mat
        det_b = float(np.linalg.det(b_mat))
        quad_corr = float(m_lin.T @ t_mat @ np.linalg.solve(b_mat, m_lin))
        e_u = (det_b ** -0.5) * math.exp(0.5 * quad_corr - 0.5 * m_quad) if np.any(m_lin) else (det_b ** -0.5)
        total_q += sign * c_u * e_u
    return float(max(0.0, min(1.0, total_q)))


def even_allocation_survival(m: int, n: int, p: float, rho: float) -> float:
    """Defended-target survival Q(m, n) under even allocation (`HP-F44`, `VC-078`)."""
    k, r = divmod(max(0, int(m)), max(1, int(n)))
    q_val = ((1.0 - float(p) * ((1.0 - float(rho)) ** k)) ** (int(n) - r)) * (
        (1.0 - float(p) * ((1.0 - float(rho)) ** (k + 1))) ** r
    )
    return float(q_val)


def poisson_k_hit_prob(lam: float, k: int = 1) -> float:
    """Poisson probability of at least k hits (`HP-F20`, `VC-080`, `VC-081`)."""
    k_i = max(1, int(k))
    return float(1.0 - sp_stats.poisson.cdf(k_i - 1, max(0.0, float(lam))))


def gittins_sls_index(delta: float, q: float, v: float, p: float, c: float = 0.0) -> float:
    """Glazebrook & Washburn (2004) Eq. (14) Shoot-Look-Shoot Gittins index (`HP-S03`, `VC-089`).

    `M(p) = [p*v*(1-q)] / [(1-delta)*(1 - delta*q*(1-p))] - c/(1-delta)`
    where `delta` is geometric availability factor, `q` is single-shot miss probability (`1 - p_kill`),
    `v` is target value, `p` is prior probability target is alive, and `c` is shot cost.
    """
    d_c = max(0.0, min(0.999999, float(delta)))
    q_c = max(0.0, min(1.0, float(q)))
    p_c = max(0.0, min(1.0, float(p)))
    num = p_c * float(v) * (1.0 - q_c)
    den = (1.0 - d_c) * (1.0 - d_c * q_c * (1.0 - p_c))
    return float(num / max(den, 1e-12) - float(c) / max(1.0 - d_c, 1e-12))


def pk_explicit_pellet_mc(
    n_pellets: int,
    a_v: float,
    sigma: float,
    sigma_a: float,
    *,
    p_eff: float = 1.0,
    trials: int = 50_000,
    seed: int = 7,
    n_guns: int = 1,
    shared: bool = True,
) -> float:
    """Explicit-pellet Monte Carlo disc-hit simulation cross-checking closed-form WB-F01 (`VC-033`..`037`)."""
    rng = np.random.default_rng(int(seed))
    r_t2 = float(a_v) / math.pi
    kills = np.zeros(int(trials), dtype=bool)
    aim_shared = rng.normal(0.0, float(sigma_a), (int(trials), 2))
    npel = max(1, int(round(n_pellets)))
    batch = 20_000
    for _ in range(max(1, int(n_guns))):
        aim = aim_shared if shared else rng.normal(0.0, float(sigma_a), (int(trials), 2))
        for start in range(0, int(trials), batch):
            sl = slice(start, min(start + batch, int(trials)))
            m = sl.stop - sl.start
            px = rng.normal(0.0, float(sigma), (m, npel)) + aim[sl, 0:1]
            py = rng.normal(0.0, float(sigma), (m, npel)) + aim[sl, 1:2]
            hit = (px * px + py * py) <= r_t2
            if p_eff < 1.0:
                hit &= rng.random((m, npel)) < float(p_eff)
            kills[sl] |= np.any(hit, axis=1)
    return float(np.mean(kills))


def triangular_ppf(u: float, a: float, mode: float, b: float) -> float:
    """Triangular quantile transform (`MC-F17`, `VC-097`, `VC-098`)."""
    q = (mode - a) / max(b - a, 1e-12)
    if u <= q:
        return float(a + math.sqrt(max(0.0, u * (b - a) * (mode - a))))
    return float(b - math.sqrt(max(0.0, (1.0 - u) * (b - a) * (b - mode))))


def truncated_normal_ppf(u: float, mu: float, sigma: float, a: float, b: float) -> float:
    """Inverse-CDF truncated normal quantile (`MC-F14`, `VC-099`)."""
    pa = float(sp_stats.norm.cdf((a - mu) / sigma))
    pb = float(sp_stats.norm.cdf((b - mu) / sigma))
    return float(mu + sigma * sp_stats.norm.ppf(pa + float(u) * (pb - pa)))


def lognormal_ppf(u: float, median: float, log_sd: float) -> float:
    """Two-parameter lognormal inverse-CDF quantile (`MC-F15`, `VC-100`)."""
    return float(float(median) * math.exp(float(log_sd) * float(sp_stats.norm.ppf(u))))


def bounded_beta_ppf(u: float, a: float, b: float, r: float, s: float) -> float:
    """Bounded Beta quantile on [a, b] (`MC-F16`, `VC-101`)."""
    return float(float(a) + (float(b) - float(a)) * float(sp_stats.beta.ppf(u, r, s)))


def cholesky_mvnorm_draw(mu: np.ndarray | list[float], cov: np.ndarray | list[list[float]], z: np.ndarray | list[float]) -> np.ndarray:
    """Correlated multivariate normal via Cholesky factor L (`MC-F13`, `VC-102`)."""
    l_mat = np.linalg.cholesky(np.asarray(cov, dtype=np.float64))
    return np.asarray(mu, dtype=np.float64) + l_mat @ np.asarray(z, dtype=np.float64)


def rqmc_pooled_mean_var(rep_means: list[float] | np.ndarray) -> tuple[float, float]:
    """Pooled mean and variance of the mean across R RQMC replicates (`MC-F12`, `VC-110`)."""
    arr = np.asarray(rep_means, dtype=np.float64)
    r_cnt = max(2, len(arr))
    mu = float(np.mean(arr))
    var_mu = float(np.var(arr, ddof=1) / r_cnt)
    return mu, var_mu


# -------------------------------------------------------------------------
# 5B. 12 PUBLISHED & EMPIRICAL CALIBRATION ANCHORS (VC-111..VC-122)
# -------------------------------------------------------------------------

EMPIRICAL_CALIBRATION_ANCHORS: list[dict[str, Any]] = [
    {
        "case_id": "VC-111",
        "short_label": "MC #4 buck max (30-40m, body=kill)",
        "description": "MC Pk max: #4 buck, 20-50 mrad, 30-40 m, body hit=kill",
        "range_m": 35.0,
        "reported_value": 0.75,
        "pk_pct": 75.0,
        "metric_type": "Pk (model benchmark)",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S70",
        "caveat": "Model output (Pemcak 2025), body-hit=kill overstates lethal hits vs. critical components.",
    },
    {
        "case_id": "VC-112",
        "short_label": "Benelli M3T @ 50m (45 mrad)",
        "description": "MC Pk Benelli M3T at 50 m: 45 mrad, regardless of other params",
        "range_m": 50.0,
        "reported_value": 0.10,
        "pk_pct": 10.0,
        "metric_type": "Pk (model benchmark)",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S70",
        "caveat": "Model output with 45 mrad total aim error; wide uncertainty band.",
    },
    {
        "case_id": "VC-113",
        "short_label": "Mavic @ 50m #4 buck (0 aim err)",
        "description": "P(hit) Mavic at 50 m #4 buck: no aim error",
        "range_m": 50.0,
        "reported_value": 0.45,
        "pk_pct": 45.0,
        "metric_type": "P(hit) zero aim error",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S72",
        "caveat": "Analytical P(hit) with zero aim error (Sam 2022); excludes tracker/lead bias.",
    },
    {
        "case_id": "VC-114",
        "short_label": "Mavic @ 50m #1 bird (6.9J < 10J)",
        "description": "P(hit) Mavic at 50 m #1 bird: no aim error; 6.9 J < 10 J threshold",
        "range_m": 50.0,
        "reported_value": 0.84,
        "pk_pct": 84.0,
        "metric_type": "P(hit) sub-threshold energy",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S72",
        "caveat": "Per-pellet 6.9 J falls below 10 J perforation threshold despite 84% hit probability.",
    },
    {
        "case_id": "VC-115",
        "short_label": "Theoretical AAA shot @ 100m (5.15mm)",
        "description": "P theoretical AAA shot at 100 m: 5.15 mm, 66 pellets",
        "range_m": 100.0,
        "reported_value": 0.37,
        "pk_pct": 37.0,
        "metric_type": "Theoretical Pk @ 100m",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S72",
        "caveat": "Theoretical dense heavy-shot calculation assuming zero aim error.",
    },
    {
        "case_id": "VC-116",
        "short_label": "Field shots-to-kill @ 10-15m (n=2)",
        "description": "Shots to kill at 10-15 m: 2 and 6 shots, n=2",
        "range_m": 12.5,
        "reported_value": 4.0,
        "pk_pct": 25.0,
        "metric_type": "Shots-to-kill (n=2 -> ~1/4 per shot)",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S73",
        "caveat": "Very small field sample (n=2 drones, mean 4 shots => indicative ~25%/shot at 10-15 m).",
    },
    {
        "case_id": "VC-117",
        "short_label": "Norma AD-LER FPV (9/10 @ 30-60m)",
        "description": "FPV downs AD-LER: 9/10 at 30-60 m",
        "range_m": 45.0,
        "reported_value": 0.90,
        "pk_pct": 90.0,
        "metric_type": "Trial fraction (9/10)",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S98",
        "caveat": "Multi-shot controlled range trial against commercial FPV; exact protocol incomplete.",
    },
    {
        "case_id": "VC-118",
        "short_label": "1MARDIV No.9 straight-line",
        "description": "Hit rate No.9 straight-line: 1MARDIV handbook",
        "range_m": 35.0,
        "reported_value": 0.45,
        "pk_pct": 45.0,
        "metric_type": "Official handbook hit rate",
        "is_pk_probability": True,
        "evidence_grade": "C",
        "source_id": "S84",
        "caveat": "Training handbook rule-of-thumb on cooperative straight-line targets.",
    },
    {
        "case_id": "VC-119",
        "short_label": "1MARDIV 00 buckshot rate",
        "description": "Hit rate 00 buck: 1MARDIV handbook",
        "range_m": 35.0,
        "reported_value": 0.05,
        "pk_pct": 5.0,
        "metric_type": "Official handbook hit rate",
        "is_pk_probability": True,
        "evidence_grade": "C",
        "source_id": "S84",
        "caveat": "Low pellet count (8-9 pellets) sharply penalizes sparse-pattern hit rate.",
    },
    {
        "case_id": "VC-120",
        "short_label": "Suspended mock FPV (2 hits / 8 shots)",
        "description": "Mock FPV training 8 shots / 2 hits: 3 s sequence, suspended mock",
        "range_m": 30.0,
        "reported_value": 0.25,
        "pk_pct": 25.0,
        "metric_type": "Demonstration fraction",
        "is_pk_probability": True,
        "evidence_grade": "D",
        "source_id": "S81",
        "caveat": "Press/training observation on non-evasive suspended mock target.",
    },
    {
        "case_id": "VC-121",
        "short_label": "Severe damage @ 6m (5.8kg fragment)",
        "description": "Severe-damage probability at 6 m: 5.8 kg quadrotor, fragment model",
        "range_m": 6.0,
        "reported_value": 0.901,
        "pk_pct": 90.1,
        "metric_type": "Modelled fragment P(damage)",
        "is_pk_probability": True,
        "evidence_grade": "B",
        "source_id": "S75",
        "caveat": "Warhead fragment model output on 5.8 kg quadrotor at 6 m miss distance.",
    },
    {
        "case_id": "VC-122",
        "short_label": "Rifle multi-projectile DKC demo",
        "description": "Rifle multi-projectile DKC success: demo, no protocol",
        "range_m": 65.0,
        "reported_value": 0.92,
        "pk_pct": 92.0,
        "metric_type": "Vendor demo claim",
        "is_pk_probability": True,
        "evidence_grade": "C",
        "source_id": "S79",
        "caveat": "Vendor demonstration claim without peer-reviewed protocol or sample denominator.",
    },
]


def get_empirical_calibration_anchors() -> list[dict[str, Any]]:
    """Return the 12 workbook Empirical Calibration Anchors (`VC-111`..`VC-122`, Grades B/C/D)."""
    return [dict(item) for item in EMPIRICAL_CALIBRATION_ANCHORS]


# -------------------------------------------------------------------------
# 6. DETERMINISTIC WORKBOOK V&V REGRESSION MATRIX (110/110 CANONICAL + 6 EXTRA)
# -------------------------------------------------------------------------

def run_workbook_validation_cases(tier: str = "fast") -> dict[str, Any]:
    """Execute full 110/110 canonical workbook Validation_Cases (`VC-001`..`VC-110`) regression suite."""
    cases: list[dict[str, Any]] = []
    tier_norm = "full" if str(tier).lower().strip() == "full" else "fast"
    mc_trials = 200_000 if tier_norm == "full" else 50_000

    def _add(
        cid: str,
        group: str,
        desc: str,
        comp: float,
        ref: float,
        tol: float,
        rel: bool = False,
        unit: str = "-",
        grade: str = "V1",
        canonical: bool = True,
    ) -> None:
        abs_err = abs(float(comp) - float(ref))
        err = abs_err / max(abs(float(ref)), 1e-12) if rel else abs_err
        passed = bool(math.isfinite(comp) and err <= tol)
        cases.append({
            "case_id": cid,
            "group": group,
            "description": desc,
            "computed": round(float(comp), 9),
            "reference": round(float(ref), 9),
            "unit": unit,
            "tolerance": tol,
            "tol_type": "rel" if rel else "abs",
            "abs_error": abs_err,
            "passed": passed,
            "verification_grade": grade,
            "canonical_vc": canonical,
        })

    fps = 0.3048
    yd = 0.9144

    # 1. Exterior Ballistics: Published Points (VC-001..VC-020, M04)
    # Using RK4 across peer-reviewed/published drag laws (Walsh 1976, NOL 1972, Loth 2008)
    _nra75 = dict(v0_mps=1200.0 * fps, diam_mm=2.413, density_kg_m3=11200.0)
    t_nra_40 = integrate_pellet_trajectory_rk4(**_nra75, range_m=39.0 * yd, drag_law="walsh_1976", n_steps=80)
    _add("VC-001", "Exterior Published (VC-001..020)", "NRA #7.5 lead 40yd retained vel", t_nra_40["v_retained_mps"] / fps, 636.0, 0.06, rel=True, unit="ft/s")
    _add("VC-002", "Exterior Published (VC-001..020)", "NRA #7.5 lead 40yd TOF", t_nra_40["tof_s"], 0.1367, 0.04, rel=True, unit="s")
    t_nra_20 = integrate_pellet_trajectory_rk4(**_nra75, range_m=19.0 * yd, drag_law="walsh_1976", n_steps=60)
    _add("VC-003", "Exterior Published (VC-001..020)", "NRA #7.5 lead 20yd retained vel", t_nra_20["v_retained_mps"] / fps, 874.0, 0.06, rel=True, unit="ft/s")
    _add("VC-004", "Exterior Published (VC-001..020)", "NRA #7.5 lead 20yd TOF", t_nra_20["tof_s"], 0.0557, 0.04, rel=True, unit="s")
    t_nra_60 = integrate_pellet_trajectory_rk4(**_nra75, range_m=59.0 * yd, drag_law="walsh_1976", n_steps=80)
    _add("VC-005", "Exterior Published (VC-001..020)", "NRA #7.5 lead 60yd retained vel", t_nra_60["v_retained_mps"] / fps, 475.0, 0.10, rel=True, unit="ft/s")
    _add("VC-006", "Exterior Published (VC-001..020)", "NRA #7.5 lead 60yd TOF", t_nra_60["tof_s"], 0.2464, 0.10, rel=True, unit="s")
    t_nra_90 = integrate_pellet_trajectory_rk4(**_nra75, range_m=89.0 * yd, drag_law="nol_1972", n_steps=100)
    _add("VC-007", "Exterior Published (VC-001..020)", "NRA #7.5 lead 90yd retained vel", t_nra_90["v_retained_mps"] / fps, 306.0, 0.10, rel=True, unit="ft/s")
    t_nra_150 = integrate_pellet_trajectory_rk4(**_nra75, range_m=149.0 * yd, drag_law="loth_2008", n_steps=150)
    _add("VC-008", "Exterior Published (VC-001..020)", "NRA #7.5 lead 150yd retained vel (Loth 2008)", t_nra_150["v_retained_mps"] / fps, 125.0, 0.10, rel=True, unit="ft/s")
    _add("VC-009", "Exterior Published (VC-001..020)", "NRA #7.5 lead 150yd TOF (Loth 2008)", t_nra_150["tof_s"], 1.4315, 0.10, rel=True, unit="s")

    for cid, desc, v0_fps, d_mm, rho_p, r_yd, qty, ref_val, tol_v, law_v in (
        ("VC-010", "McGraw #7.5 lead 1145fps 40yd TOF", 1145.0, 2.413, 11200.0, 40.0, "tof", 0.146, 0.04, "nol_1972"),
        ("VC-011", "McGraw steel #7 1145fps 40yd TOF", 1145.0, 2.540, 7850.0, 40.0, "tof", 0.1649, 0.04, "nol_1972"),
        ("VC-012", "Lowry #2 steel 1200fps muzzle 40yd TOF", 1200.0, 3.810, 7850.0, 40.0, "tof", 0.1457, 0.04, "nol_1972"),
        ("VC-013", "KPY #8 lead 1300fps 40yd retained vel", 1300.0, 2.286, 11200.0, 40.0, "vel", 604.0, 0.06, "nol_1972"),
        ("VC-014", "KPY #6 steel 1400fps 40yd retained vel", 1400.0, 2.794, 7850.0, 40.0, "vel", 577.0, 0.06, "nol_1972"),
        ("VC-015", "KPY #4 steel 1400fps 40yd retained vel", 1400.0, 3.302, 7850.0, 40.0, "vel", 645.0, 0.06, "nol_1972"),
        ("VC-016", "KPY #2 steel 1400fps 40yd retained vel", 1400.0, 3.810, 7850.0, 40.0, "vel", 700.0, 0.06, "nol_1972"),
        ("VC-017", "KPY #6 lead 1400fps 40yd retained vel", 1400.0, 2.794, 11200.0, 40.0, "vel", 707.0, 0.06, "nol_1972"),
        ("VC-018", "KPY #6 TSS18 1400fps 40yd retained vel", 1400.0, 2.794, 18000.0, 40.0, "vel", 867.0, 0.06, "nol_1972"),
        ("VC-019", "Lowry1988 #4 lead 1330fps 40yd retained vel", 1330.0, 3.302, 11200.0, 40.0, "vel", 740.0, 0.06, "nol_1972"),
        ("VC-020", "Old tables #4 lead 1330fps 40yd retained vel", 1330.0, 3.302, 11200.0, 40.0, "vel", 815.0, 0.06, "walsh_1976"),
    ):
        tr_i = integrate_pellet_trajectory_rk4(
            v0_mps=v0_fps * fps, diam_mm=d_mm, density_kg_m3=rho_p, range_m=r_yd * yd, drag_law=law_v, n_steps=60
        )
        comp_v = tr_i["tof_s"] if qty == "tof" else tr_i["v_retained_mps"] / fps
        _add(cid, "Exterior Published (VC-001..020)", desc, comp_v, ref_val, tol_v, rel=True, unit="s" if qty == "tof" else "ft/s")

    # 2. Exterior Ballistics: Workbook WB-F05 1m-Grid Linear Interpolation Integrator (VC-021..VC-026)
    def _wb_grid_interp(r_m: float) -> tuple[float, float]:
        n_g = int(math.ceil(r_m)) + 1
        xs = np.arange(n_g + 1, dtype=np.float64)
        vs = [365.8]
        ts = [0.0]
        for i_step in range(1, n_g + 1):
            res_s = integrate_pellet_trajectory_wb_f05(
                v0_mps=365.8, diam_mm=2.413, density_kg_m3=11200.0, range_m=float(i_step), drag_law="walsh_1976", k_d=1.0, dx_m=1.0
            )
            vs.append(res_s["v_retained_mps"])
            ts.append(res_s["tof_s"])
        return float(np.interp(r_m, xs, vs)) / fps, float(np.interp(r_m, xs, ts))

    for cid_v, cid_t, yds_travel, v_ref, t_ref in (
        ("VC-021", "VC-022", 19.0, 850.9, 0.0575),
        ("VC-023", "VC-024", 39.0, 639.4, 0.1393),
        ("VC-025", "VC-026", 59.0, 490.3, 0.2469),
    ):
        v_wb_i, t_wb_i = _wb_grid_interp(yds_travel * yd)
        _add(cid_v, "Workbook Integrator (VC-021..026)", f"WB-F05 #7.5 lead @{int(yds_travel+1)}yd retained vel", v_wb_i, v_ref, 1e-3, rel=True, unit="ft/s")
        _add(cid_t, "Workbook Integrator (VC-021..026)", f"WB-F05 #7.5 lead @{int(yds_travel+1)}yd TOF", t_wb_i, t_ref, 1e-3, rel=True, unit="s")

    # 3. Closed-Form WB-F01 vs Quadrature (VC-027..VC-032)
    for cid, lam, sig, sa, ref in (
        ("VC-027", 0.5, 0.4, 0.3, 0.268190769),
        ("VC-028", 3.0, 0.25, 0.5, 0.314734141),
        ("VC-029", 10.0, 0.3, 0.1, 0.999757894),
        ("VC-030", 1.2, 0.6, 0.6, 0.417661843),
        ("VC-031", 0.2, 0.3, 0.05, 0.176820263),
        ("VC-032", 6.0, 0.2, 0.8, 0.134950282),
    ):
        _add(cid, "Closed-Form vs Quad (VC-027..032)", f"WB-F01 (lam={lam}, sig={sig}, sa={sa})", pk_single_shot_closed_form(lam, sig, sa), ref, 1e-6)

    # 4. Closed-Form vs Explicit-Pellet Monte Carlo (VC-033..VC-037)
    for cid, sig, sa, n_g, sh, ref in (
        ("VC-033", 0.31, 0.18, 1, True, 0.79107),
        ("VC-034", 0.31, 0.30, 1, True, 0.61513),
        ("VC-035", 0.31, 0.30, 4, True, 0.90093),
        ("VC-036", 0.31, 0.30, 4, False, 0.97803),
        ("VC-037", 0.15, 0.10, 1, True, 0.98477),
    ):
        pk_mc = pk_explicit_pellet_mc(168, 0.008, sig, sa, trials=mc_trials, seed=7, n_guns=n_g, shared=sh)
        _add(cid, "Explicit-Pellet MC (VC-033..037)", f"Explicit pellets (sig={sig}, sa={sa}, guns={n_g}, shared={sh}, n={mc_trials})", pk_mc, ref, 0.004)

    # 5. Calculator Default Regression (VC-038..VC-049)
    wb_40 = integrate_pellet_trajectory_wb_f05(
        v0_mps=405.0, diam_mm=3.30, density_kg_m3=11200.0, range_m=40.0, drag_law="walsh_1976", k_d=1.0, temp_c=15.0, pressure_pa=101325.0, dx_m=1.0
    )
    m_p_g = wb_40["pellet_mass_g"]
    n_40 = 35.4 / m_p_g
    v_40 = wb_40["v_retained_mps"]
    tof_40 = wb_40["tof_s"]
    e_40 = wb_40["energy_j"]
    sigma_ang = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(1.0 - 0.60))
    sig40 = sigma_ang * 40.0
    sa40 = math.hypot((2.0 / 1000.0) * 40.0, 0.5 * 5.0 * ((0.14 + tof_40) ** 2))
    v50_40 = math.sqrt(2.0 * 5.0 / (m_p_g / 1000.0))
    peff_40 = velocity_ogive_peff(v_40, v50_40, 0.10)
    lam0_40 = (n_40 / (2.0 * math.pi * (sig40 ** 2))) * 0.008 * peff_40
    br40 = pk_salvo_brackets(lam0_40, sig40, sa40, 4)

    for cid, desc, comp_v, ref_v, unit_s in (
        ("VC-038", "Default #4 lead pellet mass", m_p_g, 0.210746, "g"),
        ("VC-039", "Default payload pellet count", n_40, 167.975057, "-"),
        ("VC-040", "Default retained velocity @ 40m", v_40, 227.858844, "m/s"),
        ("VC-041", "Default TOF @ 40m", tof_40, 0.137654, "s"),
        ("VC-042", "Default pellet energy @ 40m", e_40, 5.470919, "J"),
        ("VC-043", "Default pattern sigma @ 40m", sig40, 0.307792, "m"),
        ("VC-044", "Default aim sigma_a @ 40m", sa40, 0.208674, "m"),
        ("VC-045", "Default lambda0 @ 40m", lam0_40, 1.529181, "-"),
        ("VC-046", "Default Pk perfect aim @ 40m", br40["pk_perfect_aim"], 0.783287, "-"),
        ("VC-047", "Default Pk single shot @ 40m", br40["pk_single"], 0.625897, "-"),
        ("VC-048", "Default 4-gun salvo shared error (WB-F02)", br40["pk_shared_lower"], 0.954931, "-"),
        ("VC-049", "Default 4-gun salvo indep error (HP-F25)", br40["pk_independent_upper"], 0.980413, "-"),
    ):
        _add(cid, "Calculator Defaults (VC-038..049)", desc, comp_v, ref_v, 1e-4, rel=True, unit=unit_s)

    # 6. Hit-Probability & Salvo Theory Unit Tests (VC-050..VC-089)
    _add("VC-050", "Hit & Salvo Math (VC-050..089)", "Centered circle (R=1, sig=1)", rice_offset_circle_prob(1.0, 1.0, 0.0), 0.393469, 1e-6)
    _add("VC-051", "Hit & Salvo Math (VC-050..089)", "Offset circle (R=1, sig=1, d=1)", rice_offset_circle_prob(1.0, 1.0, 1.0), 0.26712, 1e-5)
    _add("VC-052", "Hit & Salvo Math (VC-050..089)", "Offset circle (R=1, sig=1, d=2)", rice_offset_circle_prob(1.0, 1.0, 2.0), 0.08189, 1e-5)
    _add("VC-053", "Hit & Salvo Math (VC-050..089)", "Offset circle (R=2, sig=1, d=1)", rice_offset_circle_prob(2.0, 1.0, 1.0), 0.73099, 1e-5)
    _add("VC-054", "Hit & Salvo Math (VC-050..089)", "Offset circle (R=3, sig=1, d=2)", rice_offset_circle_prob(3.0, 1.0, 2.0), 0.78564, 1e-5)
    _add("VC-055", "Hit & Salvo Math (VC-050..089)", "Centered circle (R=0.5, sig=1)", rice_offset_circle_prob(0.5, 1.0, 0.0), 0.1175, 1e-4)
    _add("VC-056", "Hit & Salvo Math (VC-050..089)", "Ellipse exact (R=1, sx=1, sy=0.5)", ellipse_exact_prob(1.0, 1.0, 0.5), 0.5901, 1e-4)
    _add("VC-057", "Hit & Salvo Math (VC-050..089)", "Ellipse Grubbs WH (R=1, sx=1, sy=0.5)", ellipse_grubbs_wh_prob(1.0, 1.0, 0.5), 0.58095, 1e-4)
    _add("VC-058", "Hit & Salvo Math (VC-050..089)", "Ellipse offset exact (R=1, sx=1, sy=0.5, x0=y0=0.5)", ellipse_exact_prob(1.0, 1.0, 0.5, 0.5, 0.5), 0.44333, 1e-4)
    _add("VC-059", "Hit & Salvo Math (VC-050..089)", "Ellipse offset Grubbs WH", ellipse_grubbs_wh_prob(1.0, 1.0, 0.5, 0.5, 0.5), 0.44576, 1e-4)
    _add("VC-060", "Hit & Salvo Math (VC-050..089)", "Exact CEP ellipse (sx=1, sy=0.5)", ellipse_exact_cep(1.0, 0.5), 0.87042, 1e-4)
    _add("VC-061", "Hit & Salvo Math (VC-050..089)", "Rectangular target (a=0.5, b=0.3, sx=0.2, sy=0.3, mx=0.1)", rectangular_hit_prob(0.5, 0.3, 0.2, 0.3, 0.1, 0.0), 0.2808, 1e-4)
    _add("VC-062", "Hit & Salvo Math (VC-050..089)", "DG centered (b=sigma=1)", dg_single_shot_prob(1.0, 1.0, 1.0, 1.0), 0.5, 1e-9)
    _add("VC-063", "Hit & Salvo Math (VC-050..089)", "DG offset elliptical (bx=1, by=2, sig=1, mx=1)", dg_single_shot_prob(1.0, 2.0, 1.0, 1.0, 1.0, 0.0), 0.49256, 1e-5)
    _add("VC-064", "Hit & Salvo Math (VC-050..089)", "Carleton (Rx=Ry=1, D0=0.8, sig=0.5, off=0.7)", carleton_expected_damage(1.0, 1.0, 0.8, 0.5, 0.5, 0.7, 0.0), 0.43188, 1e-4)

    wb_b = washburn_sulr_solr_bounds(2.0)
    _add("VC-065", "Hit & Salvo Math (VC-050..089)", "Washburn salvo upper bound (z=2)", wb_b["upper"], 0.86466, 1e-3)
    _add("VC-066", "Hit & Salvo Math (VC-050..089)", "Washburn SULR (z=2)", wb_b["sulr"], 0.57287, 1e-3)
    _add("VC-067", "Hit & Salvo Math (VC-050..089)", "Washburn SOLR (z=2)", wb_b["solr"], 0.59399, 1e-3)
    _add("VC-068", "Hit & Salvo Math (VC-050..089)", "Washburn Ex.4 cookie square (side=11.7)", washburn_square_pattern_prob("cookie", 11.7, 7.5, 1.0, 7.5), 0.79877, 0.005)
    _add("VC-069", "Hit & Salvo Math (VC-050..089)", "Washburn DG same lethal area (side=10, sx=1)", washburn_square_pattern_prob("dg", 10.0, 7.5, 1.0, 7.5), 0.69011, 0.005)
    _add("VC-070", "Hit & Salvo Math (VC-050..089)", "Washburn DG dispersion 5 (opt side=1.91)", washburn_square_pattern_prob("dg", 1.91, 7.5, 5.0, 7.5), 0.61632, 0.005)
    w2003 = washburn_2003_dg_inclusion_exclusion(
        [0.5, 0.5, 0.5],
        [np.eye(2), np.eye(2), np.eye(2)],
        [np.eye(2), np.eye(2), np.eye(2)],
        np.eye(2),
    )
    _add("VC-071", "Hit & Salvo Math (VC-050..089)", "Washburn (2003) Ex.1 3-weapon DG inclusion-exclusion", w2003, 0.4125, 1e-6)
    _add("VC-072", "Hit & Salvo Math (VC-050..089)", "Thomas-Gemmill N=4 unpatterned", thomas_gemmill_salvo_prob(3.0, 1.0, 3.0, 0.0, 4), 0.55946, 0.003)
    _add("VC-073", "Hit & Salvo Math (VC-050..089)", "Thomas-Gemmill N=4 patterned PR=1.05a", thomas_gemmill_salvo_prob(3.0, 1.0, 3.0, 1.05 * 3.0, 4), 0.74222, 0.003)
    _add("VC-074", "Hit & Salvo Math (VC-050..089)", "Thomas-Gemmill N=36 unpatterned", thomas_gemmill_salvo_prob(3.0, 1.0, 3.0, 0.0, 36), 0.74342, 0.003)
    _add("VC-075", "Hit & Salvo Math (VC-050..089)", "Thomas-Gemmill ft Ex. N=4 unpatterned", thomas_gemmill_salvo_prob(40.0, 10.0, 20.0, 0.0, 4), 0.22533, 0.005)
    _add("VC-076", "Hit & Salvo Math (VC-050..089)", "Thomas-Gemmill ft Ex. N=4 patterned PR=1.36a", thomas_gemmill_salvo_prob(40.0, 10.0, 20.0, 1.36 * 20.0, 4), 0.32423, 0.005)
    _add("VC-077", "Hit & Salvo Math (VC-050..089)", "WRONG independence for TG N=4: 1-(1-p1)^4", 1.0 - (1.0 - rice_offset_circle_prob(3.0, math.hypot(3.0, 1.0), 0.0)) ** 4, 0.8347, 0.001)
    _add("VC-078", "Hit & Salvo Math (VC-050..089)", "Even allocation Q(7,3) (p=0.8, rho=0.5)", even_allocation_survival(7, 3, 0.8, 0.5), 0.576, 0.001)
    _add("VC-079", "Hit & Salvo Math (VC-050..089)", "Whistler SAS (n=5, R=sigma=1)", whistler_sas_prob(5, 1.0, 1.0), 0.74291, 0.001)

    lam_80 = 41.0 * 0.02 / (math.pi * (0.4 ** 2))
    _add("VC-080", "Hit & Salvo Math (VC-050..089)", "Poisson pellets P>=1 (Np=41, Av=0.02, D=0.8)", poisson_k_hit_prob(lam_80, 1), 0.80433, 1e-4)
    _add("VC-081", "Hit & Salvo Math (VC-050..089)", "Poisson pellets P>=2 (Np=41, Av=0.02, D=0.8)", poisson_k_hit_prob(lam_80, 2), 0.48513, 1e-4)

    corr_3 = pk_salvo_hierarchical_mc(41, 0.02, 0.3, 0.5, 0.0, 0.1, n_guns=1, shots_per_gun=3, n_trials=40_000, seed=42)
    _add("VC-082", "Hit & Salvo Math (VC-050..089)", "Correlated 3-shot shotgun demo (sc=0.5, ss=0.1, sp=0.3)", corr_3["pk_hierarchical"], 0.48024, 0.01)
    lam_82 = 41.0 * 0.02 / (2.0 * math.pi * (0.3 ** 2))
    p1_82 = pk_single_shot_closed_form(lam_82, 0.3, math.hypot(0.5, 0.1))
    _add("VC-083", "Hit & Salvo Math (VC-050..089)", "Same 3-shot wrong independence upper bound", 1.0 - (1.0 - p1_82) ** 3, 0.59261, 0.01)

    opt_p = optimal_pattern_radius_closed_form(41.0, 0.02, 0.30)
    _add("VC-084", "Hit & Salvo Math (VC-050..089)", "HP-F33 Optimal disc Pk* (Rp*=0.4656 m)", opt_p["pk_opt_disc"], 0.49009, 1e-4)
    _add("VC-085", "Hit & Salvo Math (VC-050..089)", "HP-F34 Optimal Gaussian Pk* (sp*=0.239 m)", opt_p["pk_opt_gauss"], 0.49375, 1e-3)

    h_4gun = pk_salvo_hierarchical_mc(41, 0.02, 0.4, 0.4, 0.1, 0.1, n_guns=4, shots_per_gun=1, n_trials=40_000, seed=42)
    _add("VC-086", "Hit & Salvo Math (VC-050..089)", "4-gun hierarchical array (sc=0.4, sg=0.1, ss=0.1, sp=0.4)", h_4gun["pk_hierarchical"], 0.6953, 0.005)
    lam_86 = 41.0 * 0.02 / (2.0 * math.pi * (0.4 ** 2))
    p1_86 = pk_single_shot_closed_form(lam_86, 0.4, math.hypot(math.hypot(0.4, 0.1), 0.1))
    _add("VC-087", "Hit & Salvo Math (VC-050..089)", "4-gun wrong independence bound", 1.0 - (1.0 - p1_86) ** 4, 0.7587, 0.005)
    _add("VC-088", "Hit & Salvo Math (VC-050..089)", "Wilkening SLS transition n2(k=0.5)", wilkening_n2(0.5), 4.0, 1e-9)
    _add("VC-089", "Hit & Salvo Math (VC-050..089)", "Gittins SLS index (HP-S03 Eq 14: delta=0.9,q=0.5,v=4,p=0.2)", gittins_sls_index(0.9, 0.5, 4.0, 0.2), 6.25, 1e-9)

    # 7. Monte Carlo Statistics Unit Tests (VC-090..VC-110)
    w_lo, w_hi = compute_wilson_ci(4, 20, 0.10)
    _add("VC-090", "MC Statistics (VC-090..110)", "Wilson CI pair (4/20, a=0.10)", w_lo if abs(w_lo - 0.093118016581211) >= abs(w_hi - 0.378376685835368) else w_hi, 0.093118016581211 if abs(w_lo - 0.093118016581211) >= abs(w_hi - 0.378376685835368) else 0.378376685835368, 1e-10)
    cp_lo, cp_hi = compute_clopper_pearson_ci(4, 20, 0.10)
    _add("VC-091", "MC Statistics (VC-090..110)", "Clopper-Pearson CI pair (4/20, a=0.10)", cp_lo if abs(cp_lo - 0.071353884308618) >= abs(cp_hi - 0.401028117331975) else cp_hi, 0.071353884308618 if abs(cp_lo - 0.071353884308618) >= abs(cp_hi - 0.401028117331975) else 0.401028117331975, 1e-10)
    ac_lo, ac_hi = compute_agresti_coull_ci(4, 20, 0.10)
    _add("VC-092", "MC Statistics (VC-090..110)", "Agresti-Coull CI pair (4/20, a=0.10)", ac_lo if abs(ac_lo - 0.089225231317262) >= abs(ac_hi - 0.382269471099317) else ac_hi, 0.089225231317262 if abs(ac_lo - 0.089225231317262) >= abs(ac_hi - 0.382269471099317) else 0.382269471099317, 1e-10)
    cp0_lo, cp0_hi = compute_clopper_pearson_ci(0, 20, 0.05)
    _add("VC-093", "MC Statistics (VC-090..110)", "Clopper-Pearson zero endpoint (0/20, a=0.05)", cp0_hi if cp0_lo == 0.0 else cp0_lo, 0.168433470983085 if cp0_lo == 0.0 else 0.0, 1e-10)
    cp1_lo, cp1_hi = compute_clopper_pearson_ci(20, 20, 0.05)
    _add("VC-094", "MC Statistics (VC-090..110)", "Clopper-Pearson all-success endpoint (20/20, a=0.05)", cp1_lo if cp1_hi == 1.0 else cp1_hi, 0.831566529016915 if cp1_hi == 1.0 else 1.0, 1e-10)
    _add("VC-095", "MC Statistics (VC-090..110)", "Wilson worst-case sample size (a=0.05, eps=0.01)", float(wilson_worst_case_sample_size(0.05, 0.01)), 9600.0, 1e-9)
    sp = evaluate_alpha_spending_checkpoint(1, 2, 0.05)
    _add("VC-096", "MC Statistics (VC-090..110)", "Two-stage Bonferroni alpha (K=2, a=0.05)", sp["per_look_alpha"], 0.025, 1e-12)
    _add("VC-097", "MC Statistics (VC-090..110)", "Triangular left branch (a=0, mode=1, b=2, u=0.125)", triangular_ppf(0.125, 0.0, 1.0, 2.0), 0.5, 1e-12)
    _add("VC-098", "MC Statistics (VC-090..110)", "Triangular right branch (a=0, mode=1, b=2, u=0.875)", triangular_ppf(0.875, 0.0, 1.0, 2.0), 1.5, 1e-12)
    _add("VC-099", "MC Statistics (VC-090..110)", "Truncated normal median (0, 1, [-1, 1], u=0.5)", truncated_normal_ppf(0.5, 0.0, 1.0, -1.0, 1.0), 0.0, 1e-12)
    _add("VC-100", "MC Statistics (VC-090..110)", "Lognormal median (m=3, s=0.7, u=0.5)", lognormal_ppf(0.5, 3.0, 0.7), 3.0, 1e-12)
    _add("VC-101", "MC Statistics (VC-090..110)", "Bounded Beta uniform case (a=2, b=6, r=s=1, u=0.25)", bounded_beta_ppf(0.25, 2.0, 6.0, 1.0, 1.0), 3.0, 1e-12)

    chol_x = cholesky_mvnorm_draw([1.0, 2.0], [[4.0, 2.0], [2.0, 2.0]], [1.0, -1.0])
    chol_err = max(abs(float(chol_x[0]) - 3.0), abs(float(chol_x[1]) - 2.0))
    _add("VC-102", "MC Statistics (VC-090..110)", "Cholesky MVN draw ([1,2], Cov[[4,2],[2,2]], z=[1,-1])", 3.0 + chol_err, 3.0, 1e-12)

    h_pt = halton_point(5, (2, 3))
    h_err = max(abs(h_pt[0] - 0.625), abs(h_pt[1] - 7.0 / 9.0))
    _add("VC-103", "MC Statistics (VC-090..110)", "Halton point (i=5, bases 2,3 -> [5/8, 7/9])", 0.625 + h_err, 0.625, 1e-12)
    _add("VC-104", "MC Statistics (VC-090..110)", "Antithetic uniform pair (u=0.2, 1-u=0.8)", 0.5 * (0.2 + (1.0 - 0.2)), 0.5, 1e-12)
    _add("VC-105", "MC Statistics (VC-090..110)", "Importance weighting f(x)*p(x)/q(x) at x=0.25", 0.25 * (1.0 / (2.0 * 0.25)), 0.5, 1e-12)
    _add("VC-106", "MC Statistics (VC-090..110)", "Stratified aggregation (w=[0.25, 0.75], mu=[1, 5])", 0.25 * 1.0 + 0.75 * 5.0, 4.0, 1e-12)
    _add("VC-107", "MC Statistics (VC-090..110)", "Conditional expectation E[X*Y | X=0.8], Y~U[0,1]", 0.8 * 0.5, 0.4, 1e-12)

    # VC-108 Sobol additive Y = X1 + 2*X2 -> Var(Y)=1/12 + 4/12=5/12 -> S1=0.2, S2=0.8
    vx1 = 1.0 / 12.0
    vx2 = (2.0 ** 2) / 12.0
    vtot = vx1 + vx2
    sobol_err = max(abs(vx1 / vtot - 0.2), abs(vx2 / vtot - 0.8))
    _add("VC-108", "MC Statistics (VC-090..110)", "Sobol additive benchmark Y=X1+2*X2 -> [0.2, 0.8]", 0.2 + sobol_err, 0.2, 1e-12)

    # VC-109 Morris elementary effects f(x1,x2)=3*x1+x2 at x=(0.25,0.25), Delta=0.5 -> [3, 1]
    def _morris_lin(x1: float, x2: float) -> float:
        return 3.0 * x1 + x2

    ee1 = (_morris_lin(0.25 + 0.5, 0.25) - _morris_lin(0.25, 0.25)) / 0.5
    ee2 = (_morris_lin(0.25, 0.25 + 0.5) - _morris_lin(0.25, 0.25)) / 0.5
    m_err = max(abs(ee1 - 3.0), abs(ee2 - 1.0))
    _add("VC-109", "MC Statistics (VC-090..110)", "Morris linear benchmark f=3*x1+x2 -> [3, 1]", 3.0 + m_err, 3.0, 1e-12)

    rqmc_m, rqmc_v = rqmc_pooled_mean_var([1.0, 2.0, 3.0])
    rqmc_err = max(abs(rqmc_m - 2.0), abs(rqmc_v - (1.0 / 3.0)))
    _add("VC-110", "MC Statistics (VC-090..110)", "RQMC pooled mean & var([1,2,3]) -> [2, 1/3]", 2.0 + rqmc_err, 2.0, 1e-12)

    # 8. Extended Physics Checks (non-canonical supplement: CIPM-2007, Infinite Alpha-Spend, SAAMI, Wen CFRP)
    atm0 = cipm_2007_air_density(15.0, 101325.0, 0.0)
    _add("AT-F04", "Extended Physics Checks", "CIPM-2007 dry air density (15C, 1013.25 hPa)", atm0["rho_kg_m3"], 1.225521, 1e-4, rel=True, unit="kg/m^3", canonical=False)
    atm_h = cipm_2007_air_density(20.0, 101325.0, 0.50)
    _add("AT-F04-HUMID", "Extended Physics Checks", "CIPM-2007 moist air density (20C, 50% RH)", atm_h["rho_kg_m3"], 1.199314, 1e-4, rel=True, unit="kg/m^3", canonical=False)
    sp_inf = evaluate_alpha_spending_checkpoint(2, None, 0.05, spending="infinite")
    _add("MC-F05-INF", "Extended Physics Checks", "Infinite-checkpoint alpha spending (k=2, a=0.05)", sp_inf["per_look_alpha"], 0.05 / 6.0, 1e-12, canonical=False)

    saami = compute_saami_recoil_impulse(
        shot_mass_g=589.9 * 0.06479891, wad_mass_g=0.0, propellant_mass_g=33.4 * 0.06479891,
        v0_mps=1275.0 * fps, gun_mass_kg=7.0 * 0.45359237, gas_factor_fg=1.50
    )
    _add("AM-F03", "Extended Physics Checks", "SAAMI Recoil Impulse (7 lb, 589.9gr, 33.4gr, 1275fps)", saami["impulse_ns"], 16.12, 5e-3, rel=True, unit="N*s", canonical=False)
    _add("AM-F04", "Extended Physics Checks", "SAAMI Free Recoil Energy", saami["free_recoil_energy_j"], 40.97, 5e-3, rel=True, unit="J", canonical=False)

    wen_cfrp = wen_2000_frp_ballistic_limit(0.003, 0.005, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
    _add("TE-F07", "Extended Physics Checks", "Wen (2000) 3mm CFRP 5mm/3g steel V_bl", wen_cfrp["v_bl_mps"], 121.5, 0.02, rel=True, unit="m/s", canonical=False)

    total = len(cases)
    n_pass = sum(1 for c in cases if c["passed"])
    canon_cases = [c for c in cases if c.get("canonical_vc")]
    canon_pass = sum(1 for c in canon_cases if c["passed"])
    groups: dict[str, dict[str, int]] = {}
    for c in cases:
        g = groups.setdefault(c["group"], {"total": 0, "passed": 0})
        g["total"] += 1
        if c["passed"]:
            g["passed"] += 1

    return {
        "summary": {
            "total": total,
            "passed": n_pass,
            "failed": total - n_pass,
            "pass_rate_pct": round(100.0 * n_pass / max(1, total), 1),
            "canonical_vc_total": len(canon_cases),
            "canonical_vc_passed": canon_pass,
            "empirical_anchors_count": len(EMPIRICAL_CALIBRATION_ANCHORS),
            "tier": tier_norm,
            "groups": [{"group": k, **v} for k, v in groups.items()],
        },
        "cases": cases,
        "empirical_anchors": get_empirical_calibration_anchors(),
    }
