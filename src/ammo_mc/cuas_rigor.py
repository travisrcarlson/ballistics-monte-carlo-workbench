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
) -> dict[str, Any]:
    """Compute closing-target usable window [R_min_safe, R_max] with wreck carry-on inertia (`DT-F09`/`SE-F06`)."""
    from ammo_mc.small_arms_pkg.catalogs import CHOKE_DEFINITIONS, PELLET_MATERIAL_DENSITIES_G_CM3, SHOTGUN_PRESETS

    preset = SHOTGUN_PRESETS.get(ammo_preset, SHOTGUN_PRESETS["12G_NO4_BUCK_C_UAS"])
    choke_info = CHOKE_DEFINITIONS.get(choke, CHOKE_DEFINITIONS["Modified"])
    p40 = float(choke_info["benchmark_40yd_30in_pct"])
    sigma_ang_rad = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(max(1e-6, 1.0 - p40)))

    d_mm = float(preset["pellet_diameter_mm"])
    m_g = float(preset["pellet_mass_g"])
    n_pellets = int(preset["pellets_per_shot"])
    v0 = float(preset["v0_mps"])
    mat_low = str(preset.get("material", "lead")).lower()
    if "tungsten" in mat_low:
        rho_p = 18000.0
    elif "steel" in mat_low:
        rho_p = 7850.0
    elif "copper" in mat_low:
        rho_p = 8960.0
    else:
        rho_p = 11200.0

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
        if float(r_m) >= r_min_safe_m and br["pk_shared_lower"] >= float(pk_threshold):
            usable_shared_pks.append(br["pk_shared_lower"])
            usable_indep_pks.append(br["pk_independent_upper"])

        stations.append({
            "range_m": round(float(r_m), 1),
            "v_retained_mps": round(v_ret, 1),
            "tof_s": round(tof_s, 4),
            "energy_j": round(traj["energy_j"], 2),
            "p_eff": round(p_eff, 4),
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
    usable_time_s = usable_width_m / v_dr
    n_salvos_avail = int(math.floor(usable_time_s / max(0.15, float(salvo_interval_s)))) + (1 if usable_width_m > 0.0 else 0)

    # Approximate cumulative Pk over usable closing trajectory at discrete salvo intervals
    cum_shared = 0.0
    cum_indep = 0.0
    if usable_width_m > 0.0 and n_salvos_avail >= 1:
        r_shots = np.linspace(r_max_shared, r_min_safe_m + 0.5, min(n_salvos_avail, 12))
        surv_s, surv_i = 1.0, 1.0
        for rs in r_shots:
            nearest = min(stations, key=lambda st: abs(st["range_m"] - float(rs)))
            surv_s *= (1.0 - nearest["pk_shared_lower_pct"] / 100.0)
            surv_i *= (1.0 - nearest["pk_indep_upper_pct"] / 100.0)
        cum_shared = (1.0 - surv_s) * 100.0
        cum_indep = (1.0 - surv_i) * 100.0

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
    total_looks: int,
    alpha: float = 0.05,
    spending: str = "bonferroni",
) -> dict[str, float]:
    """Sequential fixed-checkpoint alpha spending (`MC-F05` / `VC-096`)."""
    k_tot = max(1, int(total_looks))
    look = max(1, min(k_tot, int(k_look)))
    t_frac = look / k_tot
    if str(spending).lower() == "pocock":
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
    """Washburn SULR vs SOLR salvo bounds at single-shot Poisson parameter z (`VC-065`..`VC-067`)."""
    r_opt = math.sqrt(2.0 * math.sqrt(z))
    p_upper = 1.0 - math.exp(-z)
    p_sulr = (1.0 - math.exp(-math.sqrt(z))) ** 2

    def _solr_integrand(u: float) -> float:
        return math.exp(-u) * (1.0 - math.exp(-(z * z) / max(1e-9, 4.0 * u * u)) if False else 1.0)

    # Optimal circular uniform cookie-cutter under Gaussian aim error: maximize 1 - E[exp(-n hit)] -> Washburn Eq.
    def _eval_solr(R_u: float) -> float:
        val, _ = integrate.quad(
            lambda r: r * math.exp(-0.5 * r * r) * math.exp(-2.0 * z / (R_u * R_u) * (1.0 if r <= R_u else 0.0)),
            0.0,
            max(R_u + 8.0, 12.0),
        )
        return 1.0 - val

    # SOLR uses Gaussian pattern optimizing disc radius for target vs aim-error circular integral:
    opt = optimize.minimize_scalar(lambda r_u: -((1.0 - math.exp(-0.5 * r_u * r_u)) * (1.0 - math.exp(-2.0 * z / (r_u * r_u)))), bounds=(0.1, 6.0), method="bounded")
    p_solr = float(-opt.fun)
    # Exact Washburn SOLR analytical limit: integrate 1 - exp(-lambda) over optimal aim-disc -> 1 - min
    # Refined SOLR exact evaluation:
    def _solr_exact(r_u: float) -> float:
        term_in = (1.0 - math.exp(-0.5 * r_u * r_u)) * math.exp(-2.0 * z / (r_u * r_u))
        term_out = math.exp(-0.5 * r_u * r_u)
        return 1.0 - (term_in + term_out)

    opt2 = optimize.minimize_scalar(lambda r_u: -_solr_exact(r_u), bounds=(0.1, 8.0), method="bounded")
    return {
        "upper": float(p_upper),
        "sulr": float(p_sulr),
        "solr": float(-opt2.fun),
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


def wilkening_n2(k: float = 0.5) -> float:
    """Wilkening (1999) SLS transition threshold n2(k) (`HP-F39`, `VC-088`)."""
    return float(2.0 / max(1e-6, float(k)))


def halton_point(index_zero_based: int, bases: tuple[int, ...] = (2, 3)) -> list[float]:
    """Deterministic zero-based Halton radical-inverse point matching Scipy/Owen (`VC-103`)."""
    # Check: VC-103 i=5 -> [0.625, 0.777777777777778]. Note 5/8 = 0.625 (101_2 -> 0.101_2), 7/9 = 0.777777... (5+1=6 -> 20_3 -> .02, wait: index+1 = 6? 6 in base 2 = 110 -> .011 = 3/8=0.375; 5 in base 2 = 101 -> .101_2 = 0.625! And 5 in base 3 = 12_3 -> .21_3 = 2/3 + 1/9 = 7/9 = 0.7777778! Wait: index=5 directly (not 6)!! Look: 5 in base 2 = 101 -> reversed = 101/8 = 0.625! 5 in base 3 = 12 -> reversed = 21/9 = 7/9 = 0.777778!
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


# -------------------------------------------------------------------------
# 6. DETERMINISTIC WORKBOOK V&V REGRESSION MATRIX (38+ ORACLES)
# -------------------------------------------------------------------------

def run_workbook_validation_cases() -> dict[str, Any]:
    """Execute deterministic regression verification against workbook Validation_Cases & Calc_Checks."""
    cases: list[dict[str, Any]] = []

    def _add(cid: str, group: str, desc: str, comp: float, ref: float, tol: float, rel: bool = False, unit: str = "-", grade: str = "V1") -> None:
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
        })

    # Group A: Exterior Ballistics (WB-F05 1m exponential step + Walsh 1976 & RK4 cross-check)
    from ammo_mc.cuas_ballistics_v2 import sphere_drag_walsh_1976

    rho_isa = 101325.0 / (287.053 * 288.15)
    a_isa = math.sqrt(1.4 * 287.053 * 288.15)
    d_m_4 = 0.00330
    m_p_4_g = 11200.0 * math.pi * (d_m_4 ** 3) / 6.0 * 1000.0
    k_4 = (3.0 * rho_isa) / (4.0 * 11200.0 * d_m_4)
    v_wb, t_wb = 405.0, 0.0
    for _ in range(40):
        cd_s = sphere_drag_walsh_1976(v_wb / a_isa)
        q_s = k_4 * cd_s
        t_wb += (math.exp(q_s) - 1.0) / (q_s * v_wb)
        v_wb *= math.exp(-q_s)
    e_wb = 0.5 * (m_p_4_g / 1000.0) * (v_wb ** 2)
    traj_40 = integrate_pellet_trajectory_rk4(
        v0_mps=405.0, diam_mm=3.30, density_kg_m3=11200.0, range_m=40.0, drag_law="walsh_1976", k_d=1.0, temp_c=15.0, pressure_pa=101325.0, n_steps=80
    )
    _add("CK-C1", "Exterior & Default Calc", "#4 lead pellet mass", m_p_4_g, 0.210745575, 1e-4, rel=True, unit="g")
    _add("CK-C3", "Exterior & Default Calc", "#4 lead retained velocity @ 40m (WB-F05)", v_wb, 227.858844, 1e-4, rel=True, unit="m/s")
    _add("CK-C4", "Exterior & Default Calc", "#4 lead TOF @ 40m (WB-F05)", t_wb, 0.137654125, 1e-4, rel=True, unit="s")
    _add("CK-C5", "Exterior & Default Calc", "#4 lead energy @ 40m (WB-F05)", e_wb, 5.4709185, 1e-4, rel=True, unit="J")
    _add("CK-C3-RK4", "Exterior & Default Calc", "#4 lead RK4 continuous vs WB-F05 @ 40m", traj_40["v_retained_mps"], 227.858844, 5e-3, rel=True, unit="m/s")

    # VC-001/002 NRA #7.5 lead 1200 fps @ 1yd to 40yd (39yd=35.66m) -> 636 fps, 0.1367 s
    traj_75 = integrate_pellet_trajectory_rk4(
        v0_mps=1200.0 * 0.3048, diam_mm=2.413, density_kg_m3=11200.0, range_m=39.0 * 0.9144, drag_law="walsh_1976", k_d=1.0, n_steps=60
    )
    _add("VC-001", "Exterior Published", "NRA #7.5 lead 40yd retained vel", traj_75["v_retained_mps"] / 0.3048, 636.0, 0.06, rel=True, unit="ft/s")
    _add("VC-002", "Exterior Published", "NRA #7.5 lead 40yd TOF", traj_75["tof_s"] + (0.9144 / (1200.0 * 0.3048)), 0.1367, 0.05, rel=True, unit="s")

    # VC-018 KPY #6 TSS18 1400 fps 40 yd (36.576 m) -> 867 fps
    traj_tss = integrate_pellet_trajectory_rk4(
        v0_mps=1400.0 * 0.3048, diam_mm=2.794, density_kg_m3=18000.0, range_m=36.576, drag_law="walsh_1976", k_d=1.0, n_steps=60
    )
    _add("VC-018", "Exterior Published", "KPY #6 TSS18 40yd retained vel", traj_tss["v_retained_mps"] / 0.3048, 867.0, 0.06, rel=True, unit="ft/s")

    # CIPM-2007 dry ISA density
    atm0 = cipm_2007_air_density(15.0, 101325.0, 0.0)
    _add("AT-F04", "Atmosphere", "CIPM-2007 dry ISA density", atm0["rho_kg_m3"], 1.2250, 5e-4, rel=True, unit="kg/m^3")

    # Group B: WB-F01 Closed-Form vs Quadrature Oracles (CK-A1..CK-A6)
    for cid, lam, sig, sa, ref in (
        ("CK-A1", 0.5, 0.4, 0.3, 0.268190769393115),
        ("CK-A2", 3.0, 0.25, 0.5, 0.314734141332934),
        ("CK-A3", 10.0, 0.3, 0.1, 0.999757893605025),
        ("CK-A4", 1.2, 0.6, 0.6, 0.417661843260168),
        ("CK-A5", 0.2, 0.3, 0.05, 0.176820262752669),
        ("CK-A6", 6.0, 0.2, 0.8, 0.134950281619550),
    ):
        _add(cid, "Closed-Form WB-F01", f"WB-F01 vs quadrature (lam={lam}, sig={sig}, sa={sa})", pk_single_shot_closed_form(lam, sig, sa), ref, 1e-9)

    # Group C: Full Calculator Salvo Defaults (CK-C6..C12)
    sigma_ang = (0.381 / 36.576) / math.sqrt(-2.0 * math.log(1.0 - 0.60))
    sig40 = sigma_ang * 40.0
    sa40 = math.hypot((2.0 / 1000.0) * 40.0, 0.5 * 5.0 * ((0.14 + 0.137654125) ** 2))
    n_40 = 35.4 / 0.210745575
    v50_40 = math.sqrt(2.0 * 5.0 / (0.210745575 / 1000.0))
    peff_40 = velocity_ogive_peff(227.858844, v50_40, 0.10)
    lam0_40 = (n_40 / (2.0 * math.pi * (sig40 ** 2))) * 0.008 * peff_40
    br40 = pk_salvo_brackets(lam0_40, sig40, sa40, 4)
    _add("CK-C6", "Exterior & Default Calc", "Pattern sigma at 40m", sig40, 0.307791848, 1e-4, rel=True, unit="m")
    _add("CK-C7", "Exterior & Default Calc", "Aim sigma_a at 40m", sa40, 0.208673604, 1e-4, rel=True, unit="m")
    _add("CK-C8", "Exterior & Default Calc", "Default lambda0 @ 40m", lam0_40, 1.529181273, 1e-4, rel=True)
    _add("CK-C9", "Exterior & Default Calc", "Pk perfect aim @ 40m", br40["pk_perfect_aim"], 0.783286977, 1e-4, rel=True)
    _add("CK-C10", "Exterior & Default Calc", "Pk single shot @ 40m", br40["pk_single"], 0.625896970, 1e-4, rel=True)
    _add("CK-C11", "Exterior & Default Calc", "Pk 4-gun salvo shared error (WB-F02)", br40["pk_shared_lower"], 0.954930612, 1e-4, rel=True)
    _add("CK-C12", "Exterior & Default Calc", "Pk 4-gun salvo indep error (HP-F25)", br40["pk_independent_upper"], 0.980413136, 1e-4, rel=True)

    # Group D: Hit-Probability & Salvo Theory Unit Tests (VC-050..VC-088)
    _add("VC-050", "Hit & Salvo Theory", "Centered circle (R=1, sig=1)", rice_offset_circle_prob(1.0, 1.0, 0.0), 0.393469, 1e-5)
    _add("VC-051", "Hit & Salvo Theory", "Offset circle (R=1, sig=1, d=1)", rice_offset_circle_prob(1.0, 1.0, 1.0), 0.26712, 5e-5)
    _add("VC-052", "Hit & Salvo Theory", "Offset circle (R=1, sig=1, d=2)", rice_offset_circle_prob(1.0, 1.0, 2.0), 0.08189, 5e-5)
    _add("VC-056", "Hit & Salvo Theory", "Ellipse exact (R=1, sx=1, sy=0.5)", ellipse_exact_prob(1.0, 1.0, 0.5), 0.5901, 2e-4)
    _add("VC-057", "Hit & Salvo Theory", "Ellipse Grubbs WH (R=1, sx=1, sy=0.5)", ellipse_grubbs_wh_prob(1.0, 1.0, 0.5), 0.58095, 2e-4)
    _add("VC-072", "Hit & Salvo Theory", "Thomas-Gemmill N=4 unpatterned", thomas_gemmill_salvo_prob(3.0, 1.0, 3.0, 0.0, 4), 0.55946, 0.003)
    _add("VC-073", "Hit & Salvo Theory", "Thomas-Gemmill N=4 patterned PR=1.05a", thomas_gemmill_salvo_prob(3.0, 1.0, 3.0, 1.05 * 3.0, 4), 0.74222, 0.003)
    _add("VC-077", "Hit & Salvo Theory", "WRONG independence for TG N=4: 1-(1-p1)^4", 1.0 - (1.0 - rice_offset_circle_prob(3.0, math.hypot(3.0, 1.0), 0.0)) ** 4, 0.8347, 0.003)
    _add("VC-079", "Hit & Salvo Theory", "Whistler SAS (n=5, R=sigma=1)", whistler_sas_prob(5, 1.0, 1.0), 0.74291, 5e-4)
    opt_p = optimal_pattern_radius_closed_form(41.0, 0.02, 0.30)
    _add("VC-084R", "Hit & Salvo Theory", "HP-F33 Optimal disc radius (Np=41,Av=0.02,sa=0.3)", opt_p["r_opt_disc_m"], 0.4656, 5e-4, unit="m")
    _add("VC-084P", "Hit & Salvo Theory", "HP-F33 Optimal disc Pk*", opt_p["pk_opt_disc"], 0.49009, 5e-4)
    _add("VC-088", "Hit & Salvo Theory", "Wilkening SLS transition n2(k=0.5)", wilkening_n2(0.5), 4.0, 1e-9)

    # Group E: Monte Carlo Stats, SAAMI Recoil & Terminal Penetration Oracles
    w_lo, w_hi = compute_wilson_ci(4, 20, 0.10)
    _add("VC-090L", "MC Statistics", "Wilson CI lower (4/20, a=0.10)", w_lo, 0.093118016581211, 1e-10)
    _add("VC-090U", "MC Statistics", "Wilson CI upper (4/20, a=0.10)", w_hi, 0.378376685835368, 1e-10)
    cp_lo, cp_hi = compute_clopper_pearson_ci(4, 20, 0.10)
    _add("VC-091L", "MC Statistics", "Clopper-Pearson CI lower (4/20, a=0.10)", cp_lo, 0.071353884308618, 1e-10)
    _add("VC-091U", "MC Statistics", "Clopper-Pearson CI upper (4/20, a=0.10)", cp_hi, 0.401028117331975, 1e-10)
    ac_lo, ac_hi = compute_agresti_coull_ci(4, 20, 0.10)
    _add("VC-092L", "MC Statistics", "Agresti-Coull CI lower (4/20, a=0.10)", ac_lo, 0.089225231317262, 1e-10)
    _add("VC-095", "MC Statistics", "Wilson worst-case sample size (a=0.05, eps=0.01)", float(wilson_worst_case_sample_size(0.05, 0.01)), 9600.0, 1e-9)
    sp = evaluate_alpha_spending_checkpoint(1, 2, 0.05)
    _add("VC-096", "MC Statistics", "Two-stage Bonferroni alpha (K=2, a=0.05)", sp["per_look_alpha"], 0.025, 1e-12)
    h_pt = halton_point(5, (2, 3))
    _add("VC-103", "MC Statistics", "Halton dim-0 (i=5, base=2)", h_pt[0], 0.625, 1e-12)
    _add("VC-103B", "MC Statistics", "Halton dim-1 (i=5, base=3)", h_pt[1], 7.0 / 9.0, 1e-12)

    saami = compute_saami_recoil_impulse(
        shot_mass_g=589.9 * 0.06479891, wad_mass_g=0.0, propellant_mass_g=33.4 * 0.06479891,
        v0_mps=1275.0 * 0.3048, gun_mass_kg=7.0 * 0.45359237, gas_factor_fg=1.50
    )
    _add("AM-F03", "Recoil SAAMI", "SAAMI Recoil Impulse (7 lb, 589.9gr, 33.4gr, 1275fps)", saami["impulse_ns"], 16.12, 5e-3, rel=True, unit="N*s")
    _add("AM-F04", "Recoil SAAMI", "SAAMI Free Recoil Energy", saami["free_recoil_energy_j"], 40.97, 5e-3, rel=True, unit="J")

    wen_cfrp = wen_2000_frp_ballistic_limit(0.003, 0.005, 0.003, rho_target_kg_m3=1550.0, sigma_e_pa=260e6)
    _add("TE-F07", "Terminal Wen CFRP", "Wen (2000) 3mm CFRP 5mm/3g steel V_bl", wen_cfrp["v_bl_mps"], 110.0, 0.12, rel=True, unit="m/s")

    total = len(cases)
    n_pass = sum(1 for c in cases if c["passed"])
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
            "groups": [{"group": k, **v} for k, v in groups.items()],
        },
        "cases": cases,
    }
