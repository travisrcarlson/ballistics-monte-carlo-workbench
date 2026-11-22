"""End-to-end Stage 0–8 Monte Carlo Simulation Pipeline (`run_simulation`).

Implements `MC_Pipeline` Stages 0..8 and `App_Spec` R13..R27:
- Stage 0 (SCN): Correlated LHS/Sobol/MC sampling, SeedSequence streams, two-loop epistemic MC
- Stage 1 (ENV): US Std Atm 1976 + met deviations + altitude wind profile + shared occasion met bias
- Stage 2 (IB): Firing-table MV + propellant temperature + wear + lot bias + round-to-round scatter
- Stage 3 (LAY): Occasion aiming/TLE bias + per-round laying & jump errors + aircraft release kinematics
- Stage 4 (EB): L0 analytic, L1 3-DOF point mass, L2 STANAG 4355 MPM, 6-DOF & Guidance plugins, DEM terrain
- Stage 5 (FZ): Bayesian Beta dud rate, graze angle function, time/HOB burst, SD redundancy, submunitions
- Stage 6 (TE): KB/Kinney-Graham blast, Mott/Gurney/EB-16/EB-17 fragments, THOR [FLAG], KE/HEAT penetration
- Stage 7 (TV): Carleton, cookie-cutter, blast/frag/thermal union, salvo correlation, area fractional damage
- Stage 8 (AGG): Dispersion, CEP/REP/DEP, Wilson CI, stopping rule, sensitivity (MC-10), FLAG governance
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from ammo_mc.data import get_kb
from ammo_mc.environment import AtmosphereEnvironment, eb_04_isa_atmosphere
from ammo_mc.exterior import (
    BatchTrajectoryConfig,
    eb_08_miller_twist_rule,
    eb_11_vacuum_range,
    eb_14_free_fall_bomb,
    eb_16_fragment_trajectory,
    eb_17_fragment_velocity_decay,
    integrate_trajectories_batch,
    solve_quadrant_elevation,
)
from ammo_mc.formulas_registry import (
    ProvenanceTracker,
    validate_safety_distance_governance,
)
from ammo_mc.fuze import (
    fz_01_system_reliability,
    fz_02_uxo_count,
    fz_03_bayesian_dud_rate,
    fz_04_zero_failure_ucb,
    fz_05_time_proximity_function,
    fz_06_impact_graze_function,
    fz_07_self_destruct_redundancy,
)
from ammo_mc.gis import build_terrain_callable
from ammo_mc.interior import ib_07_propellant_temp_correction, ib_08_mv_error_budget
from ammo_mc.launch import (
    RAD_PER_NATO_MIL,
    da_01_impact_error_model,
    da_02_pe_to_sigma,
    da_03_cep_circular,
    da_04_cep_elliptical,
    da_05_radial_miss_cdf,
    da_06_rect_hit_probability,
    da_07_angular_to_linear_error,
    da_09_correlated_bivariate_normal,
    sample_correlated_mvn,
)
from ammo_mc.penetration import (
    tp_01_lanz_odermatt,
    tp_02_shaped_charge_hydrodynamic,
    tp_03_standoff_degradation,
    tp_04_lambert_jonas,
    tp_05_los_thickness,
    tp_06_perforation_probability,
)
from ammo_mc.plugins import get_6dof_plugin, get_guidance_plugin
from ammo_mc.schemas import AmmoDefinition, Scenario, SimulationResult
from ammo_mc.stats import (
    mc_01_estimator_mean_se,
    mc_02_wilson_interval,
    mc_06_two_loop_mc,
    mc_08_reproducible_rng_streams,
    mc_09_quantile_hazard_distance,
    mc_10_sensitivity_indices,
    sample_unit_hypercube,
)
from ammo_mc.terminal_blast import (
    evaluate_blast_field,
    tb_10_brick_house_damage_range,
    tb_11_glass_breakage_swiss,
    tb_12_lung_lethality_probit,
    tb_13_axelsson_chest_wall,
    tb_15_simple_safety_distances,
    tb_16_thermal_dose,
)
from ammo_mc.terminal_frag import (
    tf_01_gurney_cyl_sphere,
    tf_03_gurney_estimate_axial,
    tf_04_mott_distribution,
    tf_05_mott_sampler,
    tf_06_fragment_presented_area,
    tf_07_fragment_spray_density,
    tf_08_taylor_angle,
    tf_09_poisson_hit_kill,
    tf_10_aastp1_stack_density,
    tf_11_hazardous_fragment_ke,
    tf_12_thor_penetration,
    tf_14_rc_debris_throw,
    tf_15_tp14_consequence_union,
)
from ammo_mc.vulnerability import (
    tv_01_carleton_damage,
    tv_02_cookie_cutter_damage,
    tv_03_sspk_cookie_cutter,
    tv_04_sspk_carleton,
    tv_05_multi_round_independent,
    tv_06_area_fractional_damage,
    tv_07_lethal_area_from_field,
    tv_08_mechanism_union,
)


class GovernanceError(RuntimeError):
    """Raised when App_Spec R25 governance prohibits a safety-distance output solely from FLAG formulas."""


def compute_governed_safety_distance(
    distance_m: float,
    basis_formula_ids: list[str],
) -> dict[str, Any]:
    """Validate and return a safety distance output subject to App_Spec R25 FLAG governance."""
    allowed, reason = validate_safety_distance_governance(basis_formula_ids)
    if not allowed:
        raise GovernanceError(reason or "Safety distance blocked by FLAG governance.")
    return {
        "safety_distance_m": float(distance_m),
        "basis_formula_ids": list(basis_formula_ids),
        "governance_passed": True,
    }


def _build_environment(scenario: Scenario) -> AtmosphereEnvironment:
    bands = [
        (b.alt_m, b.wind_range_mps, -b.wind_cross_right_mps, b.wind_up_mps)
        for b in scenario.met.wind_profile
    ]
    return AtmosphereEnvironment(
        dT_K=float(scenario.met.deviations.dT_K),
        p_sl_Pa=float(scenario.met.deviations.p_sl_Pa),
        wind_bands=bands,
        constant_wind_mps=(
            float(scenario.met.wind_range_mps),
            -float(scenario.met.wind_cross_right_mps),
            0.0,
        ),
    )


def _build_traj_config(ammo: AmmoDefinition, scenario: Scenario) -> BatchTrajectoryConfig:
    default_dt = 0.005 if ammo.calibre_m < 0.04 else 0.015
    dt = float(scenario.dt_s) if scenario.dt_s is not None else default_dt
    traj_fid = "L2" if scenario.fidelity == "L2" else "L1"
    # L3 uses MPM flight if inertia & spin are configured, else point-mass
    if scenario.fidelity == "L3" and ammo.inertia.ix > 0 and ammo.launch.twist_cal != 0:
        traj_fid = "L2"

    mpm = ammo.aero.mpm_coeffs
    return BatchTrajectoryConfig(
        calibre_m=ammo.calibre_m,
        drag_ref=ammo.aero.drag_ref,
        custom_cd_table=ammo.aero.cd_table,
        fidelity=traj_fid,
        ix_kg_m2=ammo.inertia.ix,
        iy_kg_m2=ammo.inertia.iy,
        twist_cal_per_turn=ammo.launch.twist_cal,
        c_da2=mpm.c_da2,
        c_la=mpm.c_la,
        c_ma=mpm.c_ma,
        c_mag=mpm.c_mag,
        c_spin=mpm.c_spin,
        lat_rad=math.radians(scenario.weapon_location.lat),
        az_rad=scenario.aim.az_rad,
        include_coriolis=scenario.include_coriolis,
        vacuum=scenario.vacuum,
        base_bleed_burn_time_s=ammo.launch.base_bleed_burn_s,
        base_bleed_dcd=ammo.launch.base_bleed_dcd,
        rocket_burn_time_s=ammo.launch.rocket_burn_s,
        rocket_mdot_kg_s=ammo.launch.rocket_mdot_kg_s,
        rocket_ve_mps=ammo.launch.rocket_ve_mps,
        dt=dt,
        max_time_s=240.0,
        integrator=scenario.integrator,
    )


def _run_batch_occasions(
    ammo: AmmoDefinition,
    scenario: Scenario,
    n_occ: int,
    r_per_occ: int,
    qe_nom_rad: float,
    env: AtmosphereEnvironment,
    traj_cfg: BatchTrajectoryConfig,
    terrain_fn: Any,
    streams: list[np.random.Generator],
    epistemic_overrides: dict[str, float] | None = None,
    record_path: bool = False,
) -> dict[str, Any]:
    """Execute Stages 0..7 for `n_occ` occasions x `r_per_occ` rounds (`N = n_occ * r_per_occ`)."""
    rng_occ, rng_rnd, rng_fuze, rng_frag, rng_plug = streams[:5]
    n_tot = n_occ * r_per_occ
    ep = epistemic_overrides or {}

    # --- Stage 0, 2 & 3: Occasion-level shared biases vs round-level random draws ---
    b_spec = scenario.errors.bias
    r_spec = scenario.errors.random

    # 8-dim occasion hypercube: [lot_mv, t_prop, tle_r, tle_d, aim_az_b, aim_qe_b, wind_r_b, wind_c_b, temp_b, ff_b]
    u_occ = sample_unit_hypercube(n_occ, 10, method=scenario.sampler, rng=rng_occ)
    occ_means = np.zeros(10, dtype=np.float64)
    occ_sigmas = np.array([
        ammo.launch.mv.sigma_lot,
        b_spec.propellant_temp_sigma_C,
        b_spec.tle_range_sigma_m,
        b_spec.tle_deflection_sigma_m,
        b_spec.aim_az_bias_sigma_mils * RAD_PER_NATO_MIL,
        b_spec.aim_qe_bias_sigma_mils * RAD_PER_NATO_MIL,
        b_spec.wind_range_bias_sigma_mps,
        b_spec.wind_cross_bias_sigma_mps,
        b_spec.temp_bias_sigma_K,
        ammo.aero.form_factor.sigma_bias,
    ], dtype=np.float64)
    occ_draws = sample_correlated_mvn(occ_means, occ_sigmas, u_unit=u_occ, rng=rng_occ)

    # Expand occasion biases across rounds_per_occasion -> shape [n_tot, 10]
    occ_exp = np.repeat(occ_draws, r_per_occ, axis=0)
    lot_mv_off = occ_exp[:, 0]
    t_prop_off = occ_exp[:, 1]
    tle_r_off = occ_exp[:, 2]
    tle_d_off = occ_exp[:, 3]
    aim_az_b = occ_exp[:, 4]
    aim_qe_b = occ_exp[:, 5]
    wind_r_b = occ_exp[:, 6]
    wind_c_b = occ_exp[:, 7]
    temp_b_K = occ_exp[:, 8]
    ff_b = occ_exp[:, 9]

    # 6-dim round-to-round hypercube: [mv_rr, mass_rr, ff_rr, aim_az_rr, aim_qe_rr, jump_az_rr, jump_qe_rr]
    u_rnd = sample_unit_hypercube(n_tot, 7, method=scenario.sampler, rng=rng_rnd)
    rnd_sigmas = np.array([
        ammo.launch.mv.sigma_rr,
        ammo.mass_sigma_kg,
        ammo.aero.form_factor.sigma,
        r_spec.aim_az_rand_sigma_mils * RAD_PER_NATO_MIL,
        r_spec.aim_qe_rand_sigma_mils * RAD_PER_NATO_MIL,
        r_spec.jump_az_sigma_mils * RAD_PER_NATO_MIL,
        r_spec.jump_qe_sigma_mils * RAD_PER_NATO_MIL,
    ], dtype=np.float64)
    rnd_draws = sample_correlated_mvn(
        np.zeros(7), rnd_sigmas, corr_matrix=scenario.errors.correlation_matrix, u_unit=u_rnd, rng=rng_rnd
    )

    # IB-07 & IB-08 Muzzle Velocity synthesis
    _ = ib_08_mv_error_budget(
        s_lot=ammo.launch.mv.sigma_lot,
        s_wear=abs(ammo.launch.wear_efc * ammo.launch.wear_coeff_mps_per_efc),
        s_temp=abs(b_spec.propellant_temp_sigma_C * ammo.launch.temp_coeff_mps_per_C),
        s_rr=ammo.launch.mv.sigma_rr,
    )
    t_prop_C = scenario.aim.propellant_temp_C + t_prop_off
    mv_temp_corr = np.asarray(
        ib_07_propellant_temp_correction(
            ammo.launch.mv.mu,
            t_prop_C,
            ammo.launch.temp_coeff_mps_per_C,
            ammo.launch.T_std_C,
        ),
        dtype=np.float64,
    )
    wear_dv = ammo.launch.wear_efc * ammo.launch.wear_coeff_mps_per_efc
    mv_trials = np.maximum(mv_temp_corr + wear_dv + lot_mv_off + rnd_draws[:, 0], 0.0)
    mass_trials = np.maximum(ammo.mass_kg + rnd_draws[:, 1], 1e-5)
    ff_trials = np.maximum(ammo.aero.form_factor.mu + ff_b + rnd_draws[:, 2], 0.1)

    az_err_rad = aim_az_b + rnd_draws[:, 3] + rnd_draws[:, 5]
    qe_err_rad = aim_qe_b + rnd_draws[:, 4] + rnd_draws[:, 6]
    _ = da_07_angular_to_linear_error(max(scenario.aim.range_m, 1.0), sigma_rad=float(np.std(az_err_rad) + 1e-12))

    # --- Multi-Stage Failure Modes & Effects Analysis (FMEA) Pre-Flight Draws ---
    fm_spec = ammo.failure_modes
    rng_fmea = np.random.default_rng((int(scenario.seed) if scenario.seed is not None else 42) + 77777 + n_tot)
    fm_propulsion = (
        rng_fmea.random(size=n_tot) < fm_spec.p_propulsion_anomaly
        if fm_spec.p_propulsion_anomaly > 0.0
        else np.zeros(n_tot, dtype=bool)
    )
    fm_fin_sabot = (
        rng_fmea.random(size=n_tot) < fm_spec.p_fin_or_sabot_anomaly
        if fm_spec.p_fin_or_sabot_anomaly > 0.0
        else np.zeros(n_tot, dtype=bool)
    )
    fm_sa_fail = (
        rng_fmea.random(size=n_tot) < fm_spec.p_fuze_arming_fail
        if fm_spec.p_fuze_arming_fail > 0.0
        else np.zeros(n_tot, dtype=bool)
    )
    fm_early_burst = (
        rng_fmea.random(size=n_tot) < fm_spec.p_fuze_early_burst
        if fm_spec.p_fuze_early_burst > 0.0
        else np.zeros(n_tot, dtype=bool)
    )
    fm_low_order = (
        rng_fmea.random(size=n_tot) < fm_spec.p_low_order_detonation
        if fm_spec.p_low_order_detonation > 0.0
        else np.zeros(n_tot, dtype=bool)
    )

    if np.any(fm_propulsion):
        mv_trials = np.where(fm_propulsion, mv_trials * max(0.1, 1.0 - fm_spec.mv_drop_frac), mv_trials)
    if np.any(fm_fin_sabot):
        tipoff_az = rng_fmea.normal(0.0, 4.5 * RAD_PER_NATO_MIL, size=n_tot)
        tipoff_qe = rng_fmea.normal(0.0, 4.5 * RAD_PER_NATO_MIL, size=n_tot)
        az_err_rad = np.where(fm_fin_sabot, az_err_rad + tipoff_az, az_err_rad)
        qe_err_rad = np.where(fm_fin_sabot, qe_err_rad + tipoff_qe, qe_err_rad)
        ff_trials = np.where(fm_fin_sabot, ff_trials * 1.18, ff_trials)

    # --- Stage 5 pre-sampling: Time/HOB fuze settings ---
    fuze_draws = fz_05_time_proximity_function(
        fuze_mode=ammo.fuze.type,
        t_set_s=ammo.fuze.time_set_s or 0.0,
        sigma_t_s=ammo.fuze.time_sigma_s,
        hob_nom_m=ammo.fuze.hob.mu,
        sigma_hob_m=ammo.fuze.hob.sigma,
        n_samples=n_tot,
        rng=rng_fuze,
    )
    fuze_times = fuze_draws["t_function_s"] if ammo.fuze.type == "time" and ammo.fuze.time_set_s else None
    hob_trials = fuze_draws["hob_m"]

    # --- Stage 4: Exterior Ballistics ---
    sample_path = None
    sample_paths: list[dict[str, Any]] = []
    if ammo.launch.mode == "static":
        imp_range = np.full(n_tot, scenario.aim.range_m, dtype=np.float64) + tle_r_off
        imp_defl = np.zeros(n_tot, dtype=np.float64) + tle_d_off
        imp_alt = np.full(n_tot, scenario.weapon_location.alt, dtype=np.float64)
        v_impact = np.zeros(n_tot, dtype=np.float64)
        aof_rad = np.full(n_tot, 0.5 * math.pi, dtype=np.float64)
        tof_s = np.zeros(n_tot, dtype=np.float64)
        unstable = np.zeros(n_tot, dtype=bool)
    elif scenario.fidelity == "L0" or ammo.launch.mode == "guided_miss":
        # L0 Analytic Bivariate Normal Impact Error Model (DA-01 + DA-09)
        da01_r = da_01_impact_error_model(
            scenario.aim.range_m,
            b_spec.l0_bias_range_sigma_m,
            0.0,
            n_occasions=n_occ,
            rounds_per_occasion=r_per_occ,
            rng=rng_occ,
        )
        da01_d = da_01_impact_error_model(
            0.0,
            b_spec.l0_bias_deflection_sigma_m,
            0.0,
            n_occasions=n_occ,
            rounds_per_occasion=r_per_occ,
            rng=rng_occ,
        )
        rr_xy = da_09_correlated_bivariate_normal(
            sx=r_spec.l0_rr_range_sigma_m,
            sy=r_spec.l0_rr_deflection_sigma_m,
            rho=scenario.errors.rho_range_deflection,
            n_samples=n_tot,
            rng=rng_rnd,
        )
        imp_range = np.asarray(da01_r["impacts"], dtype=np.float64) + rr_xy[:, 0]
        imp_defl = np.asarray(da01_d["impacts"], dtype=np.float64) + rr_xy[:, 1]
        imp_alt = np.zeros(n_tot, dtype=np.float64)
        v_impact = mv_trials * 0.55
        aof_rad = np.full(n_tot, 0.65, dtype=np.float64)
        tof_s = np.full(n_tot, max(scenario.aim.range_m, 1.0) / np.maximum(mv_trials * 0.7, 1.0), dtype=np.float64)
        unstable = np.zeros(n_tot, dtype=bool)
    else:
        if ammo.launch.mode == "bomb":
            rel_alt = np.maximum(10.0, ammo.launch.release_alt_m + rng_rnd.normal(0.0, r_spec.release_alt_sigma_m, size=n_tot))
            rel_tas = np.maximum(20.0, ammo.launch.release_tas_mps + rng_rnd.normal(0.0, r_spec.release_tas_sigma_mps, size=n_tot))
            states0 = eb_14_free_fall_bomb(
                release_alt_m=rel_alt,
                tas_mps=rel_tas,
                dive_angle_rad=ammo.launch.release_dive_rad + qe_err_rad,
                cross_vel_left_mps=-rel_tas * np.sin(az_err_rad),
            )
            dt_rel = rng_rnd.normal(0.0, r_spec.release_time_sigma_s, size=n_tot)
            states0[:, 0] += rel_tas * np.cos(ammo.launch.release_dive_rad) * dt_rel + tle_r_off
            states0[:, 1] += -tle_d_off
        else:
            qe_trials = qe_nom_rad + qe_err_rad
            # Right-handed internal frame: x = range, y_left = -deflection_right, z = alt
            vx = mv_trials * np.cos(qe_trials) * np.cos(az_err_rad)
            vy_left = -mv_trials * np.cos(qe_trials) * np.sin(az_err_rad)
            vz = mv_trials * np.sin(qe_trials)
            states0 = np.column_stack([
                tle_r_off,
                -tle_d_off,
                np.full(n_tot, scenario.weapon_location.alt, dtype=np.float64),
                vx,
                vy_left,
                vz,
            ])

        trial_wind_off = np.column_stack([wind_r_b, -wind_c_b, np.zeros(n_tot, dtype=np.float64)])
        if ammo.six_dof_plugin:
            plugin_6dof = get_6dof_plugin(ammo.six_dof_plugin)
            eb_res = plugin_6dof.integrate_batch(
                initial_states=states0,
                masses_kg=mass_trials,
                form_factors=ff_trials,
                env=env,
                cfg=traj_cfg,
                extra_params=ammo.guidance_params,
                fuze_times_s=fuze_times,
                terrain_fn=terrain_fn,
            )
        else:
            eb_res = integrate_trajectories_batch(
                initial_states=states0,
                masses_kg=mass_trials,
                form_factors=ff_trials,
                env=env,
                cfg=traj_cfg,
                fuze_times_s=fuze_times,
                terrain_fn=terrain_fn,
                record_single_path=record_path,
                record_n_paths=(scenario.record_trajectories_count if record_path else 0),
                trial_wind_offset=trial_wind_off,
                trial_dT_offset=temp_b_K,
            )

        imp_range = eb_res["range_m"]
        imp_defl = eb_res["deflection_m"]
        imp_alt = eb_res["alt_m"]
        v_impact = eb_res["v_impact_mps"]
        aof_rad = eb_res["angle_of_fall_rad"]
        tof_s = eb_res["tof_s"]
        unstable = eb_res["unstable"]
        sample_path = eb_res.get("sample_path")
        sample_paths = eb_res.get("sample_paths") or ([sample_path] if sample_path else [])

        # For bomb release where aim.range_m == 0 or nominal, align target with nominal bomb range if needed
        if ammo.launch.mode == "bomb" and scenario.aim.range_m > 0:
            # Shift release point so nominal drop aims at scenario.aim.range_m
            st_nom = eb_14_free_fall_bomb(ammo.launch.release_alt_m, ammo.launch.release_tas_mps, ammo.launch.release_dive_rad)[None, :]
            r_nom_drop = float(integrate_trajectories_batch(st_nom, np.array([ammo.mass_kg]), np.array([ammo.aero.form_factor.mu]), env, traj_cfg)["range_m"][0])
            imp_range = imp_range + (scenario.aim.range_m - r_nom_drop)

    # Optional Guidance Plugin (AR-05 Course-Correction / MS-01 Seeker)
    if ammo.guidance_plugin:
        g_plug = get_guidance_plugin(ammo.guidance_plugin)
        g_out = g_plug.apply_guidance(
            ballistic_range_m=imp_range,
            ballistic_deflection_m=imp_defl,
            aim_range_m=scenario.aim.range_m,
            aim_deflection_m=0.0,
            params=ammo.guidance_params,
            rng=rng_plug,
        )
        imp_range = g_out["range_m"]
        imp_defl = g_out["deflection_m"]

    # --- Stage 5: Fuze Reliability, Graze Function & UXO ---
    rel_spec = ammo.fuze.reliability
    if "p_dud" in ep:
        p_dud_base = float(ep["p_dud"])
    elif rel_spec.p_dud_mean is not None:
        p_dud_base = float(rel_spec.p_dud_mean)
        _, _ = fz_01_system_reliability([1.0 - p_dud_base])
    elif ammo.fuze.type == "none":
        p_dud_base = 0.0
    else:
        # Draw epistemic lot dud rate from Beta(failures, successes) via FZ-03
        p_dud_base = float(
            fz_03_bayesian_dud_rate(
                failures=max(rel_spec.b - 1.0, 0.0),
                n_tests=max(rel_spec.a + rel_spec.b - 2.0, 1.0),
                a0=1.0,
                b0=1.0,
                rng=rng_fuze,
            )
        )

    if ammo.fuze.type in ("PD", "PIBD", "delay") and ammo.launch.mode != "static" and scenario.fidelity != "L0":
        graze_res = fz_06_impact_graze_function(
            angle_of_fall_rad=aof_rad,
            v_impact_mps=v_impact,
            graze_threshold_deg=ammo.fuze.graze_threshold_deg,
            min_arm_vel_mps=ammo.fuze.min_arm_vel_mps,
            rng=rng_fuze,
        )
        p_graze_fail = 1.0 - np.asarray(graze_res["p_function"], dtype=np.float64)
        p_dud_primary = 1.0 - (1.0 - p_dud_base) * (1.0 - p_graze_fail)
    else:
        p_graze_fail = np.zeros(n_tot, dtype=np.float64)
        p_dud_primary = np.full(n_tot, p_dud_base, dtype=np.float64)

    if ammo.fuze.sd_fail_prob < 1.0:
        p_uxo_round = np.asarray(
            fz_07_self_destruct_redundancy(p_dud_primary, ammo.fuze.sd_fail_prob, ammo.fuze.common_cause_beta),
            dtype=np.float64,
        )
    else:
        p_uxo_round = p_dud_primary

    is_dud = rng_fuze.random(size=n_tot) < p_dud_primary
    is_uxo = rng_fuze.random(size=n_tot) < p_uxo_round if ammo.fuze.sd_fail_prob < 1.0 else is_dud.copy()
    if ammo.fuze.type == "none":
        is_dud[:] = False
        is_uxo[:] = False

    # Apply FMEA fuze failure modes (Safe & Arm failure vs Premature airburst)
    if np.any(fm_sa_fail):
        is_dud = is_dud | fm_sa_fail
        is_uxo = is_uxo | fm_sa_fail
    if np.any(fm_early_burst):
        imp_range = np.where(fm_early_burst, imp_range * 0.58, imp_range)
        is_dud = np.where(fm_early_burst, False, is_dud)
        is_uxo = np.where(fm_early_burst, False, is_uxo)

    # Submunition UXO dispersion if carrier shell (AR-03 / AB-03)
    sub_uxo_points: list[dict[str, float]] = []
    sub_uxo_counts = np.zeros(n_tot, dtype=int)
    if ammo.fuze.type == "submunition" and ammo.fuze.submunition_count > 0:
        p_sub_uxo = float(
            fz_07_self_destruct_redundancy(p_dud_base, ammo.fuze.sd_fail_prob, ammo.fuze.common_cause_beta)
        )
        for idx_r in range(n_tot):
            if not is_dud[idx_r]:
                uxo_draw = fz_02_uxo_count(ammo.fuze.submunition_count, p_sub_uxo, rng=rng_fuze)
                k_u = int(uxo_draw["sampled_uxo"])
                sub_uxo_counts[idx_r] = k_u
                if k_u > 0 and len(sub_uxo_points) < 500:
                    rad_u = ammo.fuze.submunition_pattern_radius_m * np.sqrt(rng_fuze.random(k_u))
                    ang_u = 2.0 * math.pi * rng_fuze.random(k_u)
                    for ru, au in zip(rad_u, ang_u):
                        sub_uxo_points.append({
                            "range_m": float(imp_range[idx_r] + ru * math.cos(au)),
                            "deflection_m": float(imp_defl[idx_r] + ru * math.sin(au)),
                        })

    # --- Stage 6 & 7: Terminal Effects & Target Vulnerability ---
    miss_x = imp_range - scenario.aim.range_m
    miss_y = imp_defl
    miss_r = np.hypot(miss_x, miss_y)
    # For static demolitions / QD (DM-01, DM-02, ST-01), evaluate effects at Exposed Site distance
    eval_r = (
        np.full(n_tot, max(float(scenario.target.geometry.get("exposed_site_distance_m", 300.0)), 1.0))
        if ammo.launch.mode == "static"
        else np.maximum(miss_r, 0.25)
    )

    p_kill_blast = np.zeros(n_tot, dtype=np.float64)
    p_kill_frag = np.zeros(n_tot, dtype=np.float64)
    p_kill_therm = np.zeros(n_tot, dtype=np.float64)
    p_kill_round = np.zeros(n_tot, dtype=np.float64)
    ps_at_target = np.zeros(n_tot, dtype=np.float64)
    max_frag_ranges: np.ndarray | None = None
    penetration_depths: np.ndarray | None = None
    perforated_flags: np.ndarray | None = None

    f_tnt_mult = float(ep.get("f_tnt_mult", 1.0))
    mott_b_mult = float(ep.get("mott_b_mult", 1.0))

    if ammo.warhead is not None and ammo.warhead.fill.mass_kg > 0.0:
        wh = ammo.warhead
        kb_exp = get_kb().get_explosive_properties(wh.fill.explosive)
        fp_eff = (wh.fill.f_tnt_p_override or kb_exp["f_tnt_p"]) * f_tnt_mult
        fi_eff = (wh.fill.f_tnt_i_override or kb_exp["f_tnt_i"]) * f_tnt_mult

        hob_eff = hob_trials if ammo.fuze.type in ("proximity", "time") else np.zeros(n_tot)
        blast_res = evaluate_blast_field(
            R_m=eval_r,
            w_fill_kg=wh.fill.mass_kg,
            explosive=wh.fill.explosive,
            hob_m=hob_eff,
            alt_m=imp_alt,
            dT_K=scenario.met.deviations.dT_K,
            p_sl_Pa=scenario.met.deviations.p_sl_Pa,
            f_p_override=fp_eff,
            f_i_override=fi_eff,
        )
        ps_at_target = np.asarray(blast_res["Ps_kPa"], dtype=np.float64)
        td_at_target = np.asarray(blast_res["td_ms"], dtype=np.float64)

        if scenario.target.enable_lung_probit_flag or ammo.class_id == "LM-01":
            lung_res = tb_12_lung_lethality_probit(ps_at_target * 1000.0, td_at_target * 1e-3)
            p_kill_blast = np.asarray(lung_res["probability"], dtype=np.float64)
        else:
            # Smooth overpressure incapacitation curve (50% around 100 kPa for personnel, 35 kPa for structures)
            ps_50 = 35.0 if scenario.target.type == "structure" else 110.0
            p_kill_blast = 1.0 / (1.0 + np.power(ps_50 / np.maximum(ps_at_target, 1e-4), 3.5))

        if wh.thermal_flux_kw_m2_at_100m > 0.0:
            q_r = wh.thermal_flux_kw_m2_at_100m * ((100.0 / np.maximum(eval_r, 5.0)) ** 2)
            th_res = tb_16_thermal_dose(q_r, wh.thermal_duration_s)
            p_kill_therm = np.clip(np.asarray(th_res["dose"], dtype=np.float64) / 1000.0, 0.0, 1.0)

        # Fragmentation effects
        if wh.geometry != "bare" and wh.case.mass_kg > 0.0:
            gurney_c = wh.gurney_mps or kb_exp["gurney_mps"]
            v0_gurney = float(
                tf_01_gurney_cyl_sphere(gurney_c, wh.case.mass_kg, wh.fill.mass_kg, wh.geometry)
            )
            det_vel = float(kb_exp.get("det_vel_mps") or kb_exp.get("detonation_velocity_mps") or 6900.0)
            ax_info = tf_03_gurney_estimate_axial(
                D_det_mps=det_vel,
                x_m=0.5 * wh.case.length_m,
                L_m=wh.case.length_m,
                d_m=ammo.calibre_m,
                v_gurney_mps=v0_gurney,
            )
            v0_eff = float(ax_info["v_axial_mps"])
            mott_info = tf_04_mott_distribution(
                M0_kg=wh.case.mass_kg,
                t_in=wh.case.t_in,
                d_in=wh.case.d_in,
                B=wh.mott_B * mott_b_mult,
            )
            nt_frags = float(mott_info["total_fragments"])
            m_avg_kg = float(mott_info["m_avg_kg"])
            area_f = tf_06_fragment_presented_area(m_kg=m_avg_kg, shape=wh.frag_shape)
            _ = tf_08_taylor_angle(v0_eff, det_vel)

            v_rem = np.asarray(
                eb_17_fragment_velocity_decay(v0_eff, eval_r, m_avg_kg, float(area_f["A_m2"])),
                dtype=np.float64,
            )
            is_haz, ke_rem = tf_11_hazardous_fragment_ke(m_avg_kg, v_rem)
            rho_f = np.asarray(tf_07_fragment_spray_density(nt_frags, eval_r), dtype=np.float64)

            a_targ = float(scenario.target.geometry.get("presented_area_m2", 0.56))
            if scenario.target.posture == "prone":
                a_targ = min(a_targ, 0.18)
            pk_given_hit = np.where(is_haz, np.clip(np.asarray(ke_rem, dtype=np.float64) / 150.0, 0.25, 0.95), 0.0)
            pois = tf_09_poisson_hit_kill(rho_f, A_target_m2=a_targ, pk_given_hit=pk_given_hit)
            p_kill_frag = np.asarray(pois["p_kill"], dtype=np.float64)

            # L3 Fragment-Resolved Flyout (EB-16) for Quantile Hazard Distance (MC-09)
            if scenario.fidelity == "L3" or ammo.class_id in ("DM-02", "ST-01"):
                n_det_sim = min(n_tot, 120)
                k_f = max(20, int(scenario.fragments_per_detonation))
                total_f = n_det_sim * k_f
                m_samples = tf_05_mott_sampler(float(mott_info["MK_lb_sqrt"]), n_fragments=total_f, rng=rng_frag, unit="kg")
                v_samples = v0_eff * np.clip(rng_frag.normal(1.0, 0.06, size=total_f), 0.6, 1.35)
                # Elevation angles biased around optimal ballistic throw (10..55 deg) for max hazard envelope
                el_samples = rng_frag.uniform(math.radians(8.0), math.radians(52.0), size=total_f)
                fly_res = eb_16_fragment_trajectory(
                    m_frag_kg=m_samples,
                    v0_mps=v_samples,
                    elev_rad=el_samples,
                    z0_m=max(float(ammo.fuze.hob.mu), 0.5),
                    rho_m_kg_m3=wh.case.rho_kg_m3,
                    dt=0.02,
                )
                r_fly = np.asarray(fly_res["range_m"], dtype=np.float64).reshape(n_det_sim, k_f)
                ke_fly = np.asarray(fly_res["ke_impact_J"], dtype=np.float64).reshape(n_det_sim, k_f)
                r_haz_fly = np.where(ke_fly >= 79.0, r_fly, 0.0)
                max_frag_ranges = np.max(r_haz_fly, axis=1)

        # Combine damage for HE rounds:
        if scenario.fidelity == "L0":
            # Exact Carleton damage function (TV-01) so L0 MC matches analytic TV-04 (V-08 / MC_Demo)!
            p_kill_given_fn = np.asarray(
                tv_01_carleton_damage(miss_x, miss_y, wh.carleton_rx_m, wh.carleton_ry_m),
                dtype=np.float64,
            )
        elif scenario.fidelity in ("L1", "L2") and ammo.launch.mode != "static":
            # Blend Carleton lethal zone and mechanistic blast+frag union
            p_carl = np.asarray(tv_01_carleton_damage(miss_x, miss_y, wh.carleton_rx_m, wh.carleton_ry_m), dtype=np.float64)
            p_mech = np.asarray(tv_08_mechanism_union(p_kill_blast, p_kill_frag, p_kill_therm), dtype=np.float64)
            p_kill_given_fn = 0.5 * p_carl + 0.5 * p_mech
        else:
            p_kill_given_fn = np.asarray(
                tv_08_mechanism_union(p_kill_blast, p_kill_frag, p_kill_therm),
                dtype=np.float64,
            )
        p_kill_round = np.where(is_dud, 0.0, p_kill_given_fn)

    # Kinetic / Shaped-Charge Armour Penetration (SA-01, SA-02, CN-02, CN-03)
    if ammo.penetrator is not None and ammo.penetrator.type != "none":
        pen_spec = ammo.penetrator
        t_plate = float(scenario.target.geometry.get("armour_thickness_m", 0.35))
        obliq_rad = math.radians(float(scenario.target.geometry.get("obliquity_deg", 30.0)))
        t_los = float(tp_05_los_thickness(t_plate, obliq_rad))

        hx = float(scenario.target.geometry.get("half_width_x_m", 1.2))
        hy = float(scenario.target.geometry.get("half_width_y_m", 1.5))
        # For direct-fire vertical target plane at aim.range_m, convert range error via tan(aof) to vertical error
        vert_miss = miss_x * np.tan(np.maximum(aof_rad, 0.02)) if scenario.fidelity != "L0" else miss_x
        hit_box = (np.abs(vert_miss) <= hx) & (np.abs(miss_y) <= hy)

        if pen_spec.type == "long_rod":
            bhn_trials = np.maximum(150.0, rng_rnd.normal(float(scenario.target.geometry.get("armour_bhn", 300.0)), 12.0, size=n_tot))
            lo_res = tp_01_lanz_odermatt(
                Lw_m=pen_spec.L_m,
                D_m=pen_spec.D_m,
                V_mps=np.maximum(v_impact, 100.0),
                rho_p_kg_m3=pen_spec.rho,
                rho_t_kg_m3=float(scenario.target.geometry.get("armour_rho_kg_m3", 7850.0)),
                bhn_t=bhn_trials,
                obliquity_rad=obliq_rad,
                rod_material=pen_spec.rod_material,
            )
            penetration_depths = np.asarray(lo_res["penetration"], dtype=np.float64)
            # TP-01 computes oblique channel depth directly; compare against LOS thickness
            p_perf = np.asarray(tp_06_perforation_probability(penetration_depths, t_los, sigma_v50_mps=0.04 * t_los), dtype=np.float64)
            perforated_flags = penetration_depths >= t_los
        elif pen_spec.type == "shaped_charge":
            p_hydro = tp_02_shaped_charge_hydrodynamic(
                pen_spec.jet_length_m,
                pen_spec.jet_rho_kg_m3,
                float(scenario.target.geometry.get("armour_rho_kg_m3", 7850.0)),
            )
            so_trials = np.maximum(0.5, rng_rnd.normal(pen_spec.standoff_cd, 0.4, size=n_tot))
            penetration_depths = np.asarray(tp_03_standoff_degradation(p_hydro, so_trials), dtype=np.float64) * rng_rnd.normal(1.0, 0.06, size=n_tot)
            p_perf = np.where(~is_dud, np.asarray(tp_06_perforation_probability(penetration_depths, t_los, sigma_v50_mps=0.05 * t_los), dtype=np.float64), 0.0)
            perforated_flags = (penetration_depths >= t_los) & (~is_dud)
        else:  # bullet_barrier
            _ = tp_04_lambert_jonas(v_impact, pen_spec.v50_mps)
            p_perf = np.asarray(tp_06_perforation_probability(v_impact, pen_spec.v50_mps, pen_spec.sigma_v50_mps), dtype=np.float64)
            penetration_depths = v_impact  # remaining striking speed proxy
            perforated_flags = v_impact >= pen_spec.v50_mps

        p_kill_pen = np.where(hit_box, p_perf, 0.0)
        p_kill_round = np.maximum(p_kill_round, p_kill_pen)

    # Apply FMEA terminal effects (Premature Airburst, Rocket Motor Burnout Anomaly & Low-Order Partial Detonation)
    if ammo.launch.mode == "rocket" and np.any(fm_propulsion):
        imp_range = np.where(fm_propulsion, imp_range * max(0.25, 1.0 - 1.5 * fm_spec.mv_drop_frac), imp_range)
    if np.any(fm_early_burst):
        p_kill_round = np.where(fm_early_burst, 0.0, p_kill_round)
        ps_at_target = np.where(fm_early_burst, 0.0, ps_at_target)
    if np.any(fm_low_order):
        lo_mask = fm_low_order & (~is_dud) & (~fm_early_burst)
        p_kill_round = np.where(lo_mask, p_kill_round * fm_spec.low_order_yield_frac, p_kill_round)
        ps_at_target = np.where(lo_mask, ps_at_target * fm_spec.low_order_yield_frac, ps_at_target)

    # Classify each trial into a primary Failure Mode & Effects Analysis (FMEA) category
    failure_modes_arr = np.full(n_tot, "Nominal Function", dtype=object)
    for i_r in range(n_tot):
        if fm_sa_fail[i_r]:
            failure_modes_arr[i_r] = "Safe & Arm (S&A) Fail"
        elif fm_early_burst[i_r]:
            failure_modes_arr[i_r] = "Premature Airburst"
        elif is_dud[i_r]:
            if p_graze_fail[i_r] > 0.25 and p_graze_fail[i_r] > p_dud_base:
                failure_modes_arr[i_r] = "Impact Graze / Ricochet Fail (FZ-06)"
            else:
                failure_modes_arr[i_r] = "Fuze Initiation Dud (FZ-03)"
        elif fm_low_order[i_r]:
            failure_modes_arr[i_r] = "Low-Order Partial Detonation"
        elif unstable[i_r]:
            failure_modes_arr[i_r] = "Ballistic Instability (Sg < 1)"
        elif fm_propulsion[i_r]:
            if ammo.launch.mode == "rocket":
                failure_modes_arr[i_r] = "Rocket Motor Thrust / Burnout Anomaly (EB-13)"
            else:
                failure_modes_arr[i_r] = "Propulsion / Ignition Anomaly"
        elif fm_fin_sabot[i_r]:
            if ammo.launch.mode == "rocket":
                failure_modes_arr[i_r] = "Wrap-Around Fin / Nozzle Tip-Off Anomaly"
            else:
                failure_modes_arr[i_r] = "Fin / Sabot Tip-Off Anomaly"

    # Salvo kill per occasion (captures shared occasion bias correlation per MC_Demo!)
    pk_by_occ = p_kill_round.reshape(n_occ, r_per_occ)
    salvo_pk_per_occ = 1.0 - np.prod(1.0 - np.clip(pk_by_occ, 0.0, 1.0), axis=1)

    # Stochastic input matrix for MC-10 sensitivity ranking
    X_sens = np.column_stack([
        mv_trials,
        ff_trials,
        mass_trials,
        az_err_rad / RAD_PER_NATO_MIL,
        qe_err_rad / RAD_PER_NATO_MIL,
        wind_r_b,
        wind_c_b,
        temp_b_K,
    ])
    sens_names = [
        "muzzle_velocity_mps",
        "drag_form_factor",
        "projectile_mass_kg",
        "azimuth_error_mils",
        "elevation_error_mils",
        "wind_range_bias_mps",
        "wind_cross_bias_mps",
        "air_temp_bias_K",
    ]

    return {
        "n_occ": n_occ,
        "r_per_occ": r_per_occ,
        "imp_range": imp_range,
        "imp_defl": imp_defl,
        "imp_alt": imp_alt,
        "v_impact": v_impact,
        "aof_rad": aof_rad,
        "tof_s": tof_s,
        "mv_trials": mv_trials,
        "unstable": unstable,
        "is_dud": is_dud,
        "is_uxo": is_uxo,
        "failure_modes": failure_modes_arr,
        "sub_uxo_counts": sub_uxo_counts,
        "sub_uxo_points": sub_uxo_points,
        "p_dud_base": p_dud_base,
        "p_kill_round": p_kill_round,
        "salvo_pk_per_occ": salvo_pk_per_occ,
        "ps_at_target": ps_at_target,
        "max_frag_ranges": max_frag_ranges,
        "penetration_depths": penetration_depths,
        "perforated_flags": perforated_flags,
        "X_sens": X_sens,
        "sens_names": sens_names,
        "sample_path": sample_path,
        "sample_paths": sample_paths,
    }


def run_simulation(
    ammo: AmmoDefinition,
    scenario: Scenario,
) -> SimulationResult:
    """Run the complete Ammunition Effects & Dispersion Monte Carlo Simulator."""
    with ProvenanceTracker() as tracker:
        streams = mc_08_reproducible_rng_streams(scenario.seed, 16)
        env = _build_environment(scenario)
        _ = eb_04_isa_atmosphere(scenario.weapon_location.alt, dT_K=scenario.met.deviations.dT_K, p_sl_Pa=scenario.met.deviations.p_sl_Pa)
        traj_cfg = _build_traj_config(ammo, scenario)
        terrain_fn = build_terrain_callable(scenario.terrain, base_alt_m=0.0)

        # Determine occasion & round counts
        r_per_occ = max(1, int(scenario.rounds_per_occasion))
        if scenario.n_trials is not None and scenario.n_trials > 0:
            if scenario.n_occasions <= 1:
                n_occ = 1
                r_per_occ = int(scenario.n_trials)
            else:
                n_occ = max(1, int(math.ceil(scenario.n_trials / r_per_occ)))
        else:
            n_occ = max(1, int(scenario.n_occasions))

        # Solve Quadrant Elevation if needed
        if scenario.aim.qe_rad is not None:
            qe_solved = float(scenario.aim.qe_rad)
        elif ammo.launch.mode in ("gun", "rocket") and scenario.fidelity != "L0" and scenario.aim.range_m > 0:
            qe_solved = solve_quadrant_elevation(
                target_range_m=scenario.aim.range_m,
                mv_mps=max(ammo.launch.mv.mu, 10.0),
                mass_kg=ammo.mass_kg,
                env=env,
                cfg=traj_cfg,
                high_angle=scenario.aim.high_angle,
                z0_m=scenario.weapon_location.alt,
            )
            if scenario.vacuum:
                _ = eb_11_vacuum_range(ammo.launch.mv.mu, qe_solved)
        else:
            qe_solved = 0.0

        # Small-arms Miller twist stability check (EB-08) when length_m and spin exist
        miller_sg: float | None = None
        if ammo.calibre_m <= 0.020 and ammo.length_m and ammo.launch.twist_cal > 0:
            miller_sg = eb_08_miller_twist_rule(
                m_grains=ammo.mass_kg * 15432.3584,
                d_in=ammo.calibre_m / 0.0254,
                length_in=ammo.length_m / 0.0254,
                twist_in_per_turn=(ammo.launch.twist_cal * ammo.calibre_m) / 0.0254,
                v_fps=ammo.launch.mv.mu * 3.28084,
                h_ft=scenario.weapon_location.alt * 3.28084,
            )

        # Execute batch(es) with optional 95% CI half-width adaptive stopping rule (App_Spec R16)
        batches: list[dict[str, Any]] = []
        stopped_by_ci = False
        total_rounds_so_far = 0
        batch_idx = 0

        while True:
            batch_streams = mc_08_reproducible_rng_streams(
                None if scenario.seed is None else (scenario.seed + 10007 * batch_idx),
                6,
            )
            b_out = _run_batch_occasions(
                ammo=ammo,
                scenario=scenario,
                n_occ=n_occ,
                r_per_occ=r_per_occ,
                qe_nom_rad=qe_solved,
                env=env,
                traj_cfg=traj_cfg,
                terrain_fn=terrain_fn,
                streams=batch_streams,
                record_path=(batch_idx == 0),
            )
            batches.append(b_out)
            total_rounds_so_far += n_occ * r_per_occ
            batch_idx += 1

            if scenario.ci_tolerance is None or scenario.ci_tolerance <= 0.0:
                break
            all_pk = np.concatenate([b["p_kill_round"] for b in batches])
            all_rng = np.concatenate([b["imp_range"] for b in batches])
            if np.any(all_pk > 0):
                hw = mc_01_estimator_mean_se(all_pk)["ci_half_width"]
            else:
                hw = mc_01_estimator_mean_se(all_rng)["ci_half_width"]
            if hw < scenario.ci_tolerance:
                stopped_by_ci = True
                break
            if total_rounds_so_far >= scenario.n_max:
                break

        # Merge batches
        def cat(key: str) -> np.ndarray:
            return np.concatenate([b[key] for b in batches], axis=0)

        n_occ_tot = sum(b["n_occ"] for b in batches)
        imp_range = cat("imp_range")
        imp_defl = cat("imp_defl")
        v_impact = cat("v_impact")
        aof_rad = cat("aof_rad")
        tof_s = cat("tof_s")
        mv_trials = cat("mv_trials")
        unstable = cat("unstable")
        is_dud = cat("is_dud")
        is_uxo = cat("is_uxo")
        failure_modes_all = cat("failure_modes")
        sub_uxo_counts = cat("sub_uxo_counts")
        p_kill_round = cat("p_kill_round")
        salvo_pk_per_occ = cat("salvo_pk_per_occ")
        ps_at_target = cat("ps_at_target")
        X_sens = cat("X_sens")
        sens_names = batches[0]["sens_names"]
        sample_path = batches[0]["sample_path"]
        sample_paths = batches[0].get("sample_paths") or ([sample_path] if sample_path else [])

        max_frag_ranges = (
            np.concatenate([b["max_frag_ranges"] for b in batches if b["max_frag_ranges"] is not None])
            if any(b["max_frag_ranges"] is not None for b in batches)
            else None
        )
        penetration_depths = (
            np.concatenate([b["penetration_depths"] for b in batches if b["penetration_depths"] is not None])
            if any(b["penetration_depths"] is not None for b in batches)
            else None
        )
        perforated_flags = (
            np.concatenate([b["perforated_flags"] for b in batches if b["perforated_flags"] is not None])
            if any(b["perforated_flags"] is not None for b in batches)
            else None
        )

        # Optional Two-Loop Epistemic Outer Loop (MC-06)
        epistemic_summary: dict[str, Any] | None = None
        if scenario.two_loop_epistemic:
            rel_s = ammo.fuze.reliability

            def ep_sampler(rng_ep: np.random.Generator) -> dict[str, Any]:
                p_d = (
                    float(rel_s.p_dud_mean)
                    if rel_s.p_dud_mean is not None
                    else float(
                        fz_03_bayesian_dud_rate(
                            max(rel_s.b - 1.0, 0.1),
                            max(rel_s.a + rel_s.b - 2.0, 1.0),
                            rng=rng_ep,
                        )
                    )
                )
                return {
                    "p_dud": p_d,
                    "f_tnt_mult": float(rng_ep.uniform(0.92, 1.08)),
                    "mott_b_mult": float(rng_ep.normal(1.0, 0.05)),
                }

            def al_runner(theta: dict[str, Any], n_in: int, rng_in: np.random.Generator) -> dict[str, float]:
                n_occ_in = max(4, n_in // r_per_occ)
                sub_st = mc_08_reproducible_rng_streams(int(rng_in.integers(0, 2**31 - 1)), 6)
                res_in = _run_batch_occasions(
                    ammo=ammo,
                    scenario=scenario,
                    n_occ=n_occ_in,
                    r_per_occ=r_per_occ,
                    qe_nom_rad=qe_solved,
                    env=env,
                    traj_cfg=traj_cfg,
                    terrain_fn=terrain_fn,
                    streams=sub_st,
                    epistemic_overrides=theta,
                )
                sr_i = float(np.std(res_in["imp_range"], ddof=1)) if res_in["imp_range"].size > 1 else 0.0
                sd_i = float(np.std(res_in["imp_defl"], ddof=1)) if res_in["imp_defl"].size > 1 else 0.0
                return {
                    "sspk": float(np.mean(res_in["p_kill_round"])),
                    "salvo_pk": float(np.mean(res_in["salvo_pk_per_occ"])),
                    "cep_m": float(da_04_cep_elliptical(sr_i, sd_i)),
                    "uxo_fraction": float(np.mean(res_in["is_uxo"])),
                }

            epistemic_summary = mc_06_two_loop_mc(
                n_outer=scenario.n_outer_epistemic,
                n_inner=max(40, min(160, n_occ * r_per_occ)),
                epistemic_sampler=ep_sampler,
                aleatory_runner=al_runner,
                seed=None if scenario.seed is None else scenario.seed + 99991,
            )["summary"]

        # --- Stage 8: Aggregation, Standard Report, Hazard Grid & Governance ---
        n_tot = imp_range.size
        rng_ci = mc_01_estimator_mean_se(imp_range)
        defl_ci = mc_01_estimator_mean_se(imp_defl)
        std_r = float(np.std(imp_range, ddof=1)) if n_tot > 1 else 0.0
        std_d = float(np.std(imp_defl, ddof=1)) if n_tot > 1 else 0.0

        rep_m = float(da_02_pe_to_sigma(sigma=std_r))
        dep_m = float(da_02_pe_to_sigma(sigma=std_d))
        cep_ellip = float(da_04_cep_elliptical(std_r, std_d))
        cep_circ_eq = float(da_03_cep_circular(0.5 * (std_r + std_d)))
        radial_from_mpi = np.hypot(imp_range - rng_ci["mean"], imp_defl - defl_ci["mean"])
        cep_emp = float(np.median(radial_from_mpi))
        mpi_offset_m = float(math.hypot(rng_ci["mean"] - scenario.aim.range_m, defl_ci["mean"]))

        dispersion = {
            "mean_range_m": rng_ci["mean"],
            "range_ci95_half_width_m": rng_ci["ci_half_width"],
            "std_range_m": std_r,
            "p05_range_m": float(np.quantile(imp_range, 0.05)),
            "p50_range_m": float(np.quantile(imp_range, 0.50)),
            "p95_range_m": float(np.quantile(imp_range, 0.95)),
            "mean_deflection_m": defl_ci["mean"],
            "deflection_ci95_half_width_m": defl_ci["ci_half_width"],
            "std_deflection_m": std_d,
            "p05_deflection_m": float(np.quantile(imp_defl, 0.05)),
            "p50_deflection_m": float(np.quantile(imp_defl, 0.50)),
            "p95_deflection_m": float(np.quantile(imp_defl, 0.95)),
            "rep_m": rep_m,
            "dep_m": dep_m,
            "cep_m": cep_ellip,
            "cep_empirical_m": cep_emp,
            "cep_circular_ref_m": cep_circ_eq,
            "mpi_offset_m": mpi_offset_m,
            "p_within_25m": float(da_05_radial_miss_cdf(25.0, cep=max(cep_ellip, 1e-6))),
            "mean_v_impact_mps": float(np.mean(v_impact)),
            "mean_tof_s": float(np.mean(tof_s)),
            "unstable_fraction": float(np.mean(unstable)),
        }
        if miller_sg is not None:
            dispersion["miller_sg"] = miller_sg

        # Effectiveness & Salvo Statistics
        # Sample Bernoulli kill realizations from p_kill_round for exact Wilson score CI (MC-02)
        kill_rng = streams[8]
        kill_draws = kill_rng.random(size=n_tot) < p_kill_round
        wilson_single = mc_02_wilson_interval(int(np.sum(kill_draws)), n_tot)
        sspk_mean = float(np.mean(p_kill_round))
        salvo_ci = mc_01_estimator_mean_se(salvo_pk_per_occ)
        salvo_indep_ref = tv_05_multi_round_independent(sspk_mean, n_rounds=r_per_occ)

        # Analytic comparison metrics (DA-06, TV-03, TV-04)
        hx = float(scenario.target.geometry.get("half_width_x_m", 25.0))
        hy = float(scenario.target.geometry.get("half_width_y_m", 25.0))
        p_rect_hit = (
            da_06_rect_hit_probability(-hx, hx, -hy, hy, max(std_r, 0.1), max(std_d, 0.1), rng_ci["mean"] - scenario.aim.range_m, defl_ci["mean"])
            if ammo.launch.mode != "static"
            else 1.0
        )

        lr_ref = ammo.warhead.lethal_radius_m if ammo.warhead is not None else 5.0
        analytic_sspk_cc = float(tv_03_sspk_cookie_cutter(lr_ref, cep_m=max(cep_ellip, 0.1)))
        analytic_sspk_carl = float(
            tv_04_sspk_carleton(
                ammo.warhead.carleton_rx_m if ammo.warhead else lr_ref,
                0.5 * (std_r + std_d),
                bias_m=0.0,
                p_dud=float(np.mean(is_dud)),
            )
        )
        _ = tv_02_cookie_cutter_damage(0.5 * lr_ref, lr_ref)

        # Area fractional damage (TV-06) if area target & warhead
        area_fd = 0.0
        if ammo.warhead is not None and ammo.launch.mode != "static":
            n_occ_fd = min(n_occ_tot, 30)
            fds = []
            for k_occ in range(n_occ_fd):
                sl = slice(k_occ * r_per_occ, (k_occ + 1) * r_per_occ)
                imp_xy_k = np.column_stack([imp_range[sl] - scenario.aim.range_m, imp_defl[sl]])
                fn_mask_k = ~is_dud[sl]
                fds.append(
                    tv_06_area_fractional_damage(
                        imp_xy_k, hx, hy, ammo.warhead.carleton_rx_m, ammo.warhead.carleton_ry_m, fn_mask_k, grid_n=21
                    )
                )
            area_fd = float(np.mean(fds)) if fds else 0.0

        # Numerical Lethal Area & 2-D Hazard Grid around burst point (TV-07)
        hazard_grid: dict[str, Any] | None = None
        lethal_area_m2 = 0.0
        carleton_al_ref_m2 = 0.0
        if ammo.warhead is not None and ammo.warhead.fill.mass_kg > 0.0:
            wh = ammo.warhead
            carleton_al_ref_m2 = float(math.pi * wh.carleton_rx_m * wh.carleton_ry_m)
            span = max(3.0 * wh.lethal_radius_m, 60.0)
            gx = np.linspace(-span, span, 41)
            gy = np.linspace(-span, span, 41)
            dx_g = float(gx[1] - gx[0])
            xx, yy = np.meshgrid(gx, gy, indexing="ij")
            rr = np.maximum(np.hypot(xx, yy), 0.5)

            bf_grid = evaluate_blast_field(rr, wh.fill.mass_kg, wh.fill.explosive, hob_m=ammo.fuze.hob.mu)
            ps_g = np.asarray(bf_grid["Ps_kPa"], dtype=np.float64)
            pk_b_g = 1.0 / (1.0 + np.power(110.0 / np.maximum(ps_g, 1e-4), 3.5))

            if wh.geometry != "bare" and wh.case.mass_kg > 0.0:
                kb_e = get_kb().get_explosive_properties(wh.fill.explosive)
                v0_g = float(tf_01_gurney_cyl_sphere(wh.gurney_mps or kb_e["gurney_mps"], wh.case.mass_kg, wh.fill.mass_kg, wh.geometry))
                mt_g = tf_04_mott_distribution(M0_kg=wh.case.mass_kg, t_in=wh.case.t_in, d_in=wh.case.d_in, B=wh.mott_B)
                af_g = tf_06_fragment_presented_area(m_kg=float(mt_g["m_avg_kg"]), shape=wh.frag_shape)
                vr_g = np.asarray(eb_17_fragment_velocity_decay(v0_g, rr, float(mt_g["m_avg_kg"]), float(af_g["A_m2"])), dtype=np.float64)
                hz_g, ke_g = tf_11_hazardous_fragment_ke(float(mt_g["m_avg_kg"]), vr_g)
                rh_g = np.asarray(tf_07_fragment_spray_density(float(mt_g["total_fragments"]), rr), dtype=np.float64)
                pkh_g = np.where(hz_g, np.clip(np.asarray(ke_g, dtype=np.float64) / 150.0, 0.25, 0.95), 0.0)
                pk_f_g = np.asarray(tf_09_poisson_hit_kill(rh_g, 0.56, pkh_g)["p_kill"], dtype=np.float64)
            else:
                pk_f_g = np.zeros_like(pk_b_g)

            pk_tot_g = np.asarray(tv_08_mechanism_union(pk_b_g, pk_f_g, 0.0), dtype=np.float64)
            al_info = tv_07_lethal_area_from_field(pk_tot_g, dx_g, dx_g)
            lethal_area_m2 = al_info["lethal_area_m2"]
            hazard_grid = {
                "x_m": gx.tolist(),
                "y_m": gy.tolist(),
                "pk_grid": np.round(pk_tot_g, 4).tolist(),
                "ps_kpa_grid": np.round(ps_g, 2).tolist(),
            }

        effectiveness = {
            "sspk": sspk_mean,
            "sspk_wilson_ci_low": wilson_single["ci_low"],
            "sspk_wilson_ci_high": wilson_single["ci_high"],
            "sspk_wilson_half_width": wilson_single["half_width"],
            "salvo_pk_mean": salvo_ci["mean"],
            "salvo_pk_ci_low": max(0.0, salvo_ci.get("ci_low", salvo_ci["mean"])),
            "salvo_pk_ci_high": min(1.0, salvo_ci.get("ci_high", salvo_ci["mean"])),
            "salvo_pk_independent_ref": salvo_indep_ref,
            "rect_target_hit_prob": p_rect_hit,
            "area_fractional_damage": area_fd,
            "lethal_area_m2": lethal_area_m2,
            "carleton_ref_lethal_area_m2": carleton_al_ref_m2,
            "analytic_sspk_cookie_cutter": analytic_sspk_cc,
            "analytic_sspk_carleton": analytic_sspk_carl,
        }

        # Blast & Safety Summary
        blast_and_safety: dict[str, Any] = {}
        if ammo.warhead is not None and ammo.warhead.fill.mass_kg > 0.0:
            wh = ammo.warhead
            auw = wh.auw_kg if wh.auw_kg is not None else ammo.mass_kg
            tb15 = tb_15_simple_safety_distances(auw_kg=auw, neq_kg=wh.fill.mass_kg)
            es_dist = max(float(scenario.target.geometry.get("exposed_site_distance_m", 300.0)), 2.0)
            es_blast = evaluate_blast_field(es_dist, wh.fill.mass_kg, wh.fill.explosive)
            glass = tb_11_glass_breakage_swiss(float(es_blast["Ps_kPa"]), float(es_blast["Is_kPa_ms"]), scenario.target.window_size)
            axel = tb_13_axelsson_chest_wall(float(es_blast["Ps_kPa"]), float(es_blast["td_ms"]), float(es_blast["Is_kPa_ms"]))
            brick_cb = float(tb_10_brick_house_damage_range(wh.fill.mass_kg, category=scenario.target.brick_damage_category))

            blast_and_safety = {
                "mean_ps_at_target_kPa": float(np.mean(ps_at_target)),
                "exposed_site_distance_m": es_dist,
                "ps_at_exposed_site_kPa": float(es_blast["Ps_kPa"]),
                "is_at_exposed_site_kPa_ms": float(es_blast["Is_kPa_ms"]),
                "glass_breakage_at_es": glass["breaks"],
                "axelsson_asii_at_es": axel["ASII"],
                "axelsson_injury_at_es": axel["injury_level"],
                "brick_house_cb_range_m": brick_cb,
                "iatg_tb15_distances_m": tb15,
            }

        # Fragmentation Summary
        fragmentation: dict[str, Any] = {"quantile": scenario.hazard_quantile, "hazard_distance_quantile_m": None}
        if ammo.warhead is not None and ammo.warhead.geometry != "bare" and ammo.warhead.case.mass_kg > 0.0:
            wh = ammo.warhead
            kb_e = get_kb().get_explosive_properties(wh.fill.explosive)
            v0_g = float(tf_01_gurney_cyl_sphere(wh.gurney_mps or kb_e["gurney_mps"], wh.case.mass_kg, wh.fill.mass_kg, wh.geometry))
            mt_g = tf_04_mott_distribution(M0_kg=wh.case.mass_kg, t_in=wh.case.t_in, d_in=wh.case.d_in, B=wh.mott_B)
            fragmentation.update({
                "gurney_v0_mps": v0_g,
                "mott_total_fragments": mt_g["total_fragments"],
                "mott_mean_frag_mass_g": mt_g["m_avg_g"],
            })
            if max_frag_ranges is not None and max_frag_ranges.size > 0:
                q_dist = float(mc_09_quantile_hazard_distance(max_frag_ranges, scenario.hazard_quantile))
                fragmentation["hazard_distance_quantile_m"] = q_dist
                fragmentation["hazard_distance_p95_m"] = float(mc_09_quantile_hazard_distance(max_frag_ranges, 0.95))
                fragmentation["hazard_distance_max_m"] = float(np.max(max_frag_ranges))

            if scenario.target.enable_thor_penetration_flag:
                thor_res = tf_12_thor_penetration(
                    m_g=mt_g["m_avg_g"],
                    t_cm=float(scenario.target.geometry.get("thor_plate_cm", 0.635)),
                    v_s_mps=v0_g * 0.75,
                    target_material=scenario.target.thor_material,
                    shape=wh.frag_shape,
                )
                fragmentation["thor_v50_mps"] = thor_res["v50_mps"]
                fragmentation["thor_v_residual_mps"] = thor_res["v_residual_mps"]

            if wh.stack_total_items > 1 or ammo.class_id == "ST-01":
                es_d = max(float(scenario.target.geometry.get("exposed_site_distance_m", 400.0)), 5.0)
                st_res = tf_10_aastp1_stack_density(
                    R_m=es_d,
                    N_t_per_item=mt_g["total_fragments"],
                    M0_kg_per_item=wh.case.mass_kg,
                    V0_mps=v0_g,
                    N_surface=wh.stack_surface_items,
                    N_total=wh.stack_total_items,
                    storage_mode=wh.storage_mode,
                )
                fragmentation["stack_qf_at_es_m2"] = st_res["q_f_m2"]
                fragmentation["stack_exceeds_1_per_56m2"] = st_res["exceeds_1_per_56m2"]
                if wh.rc_room_vol_m3 > 0 and wh.rc_wall_t_m > 0:
                    rc_res = tf_14_rc_debris_throw(wh.fill.mass_kg, wh.rc_room_vol_m3, wh.rc_wall_t_m)
                    fragmentation["rc_debris_dlv_mps"] = rc_res["dlv_mps"]
                    fragmentation["rc_debris_avg_mass_kg"] = rc_res["m_avg_debris_kg"]
                tp14_u = tf_15_tp14_consequence_union([0.56, 0.56], [st_res["q_f_m2"] * 0.6, st_res["q_f_m2"] * 0.4], [0.8, 0.5])
                fragmentation["tp14_consequence_prob_at_es"] = tp14_u["p_total"]

        # Penetration Summary
        pen_summary: dict[str, Any] | None = None
        if penetration_depths is not None and perforated_flags is not None:
            perf_w = mc_02_wilson_interval(int(np.sum(perforated_flags)), n_tot)
            pen_summary = {
                "penetrator_type": ammo.penetrator.type if ammo.penetrator else "none",
                "mean_penetration_metric": float(np.mean(penetration_depths)),
                "p05_penetration_metric": float(np.quantile(penetration_depths, 0.05)),
                "p95_penetration_metric": float(np.quantile(penetration_depths, 0.95)),
                "p_perforation_given_impact": perf_w["p_hat"],
                "p_perforation_ci_low": perf_w["ci_low"],
                "p_perforation_ci_high": perf_w["ci_high"],
            }

        # UXO Summary & Location Map (App_Spec R20)
        p_uxo_emp = float(np.mean(is_uxo))
        if ammo.fuze.type == "submunition" and ammo.fuze.submunition_count > 0:
            tot_sub_uxo = int(np.sum(sub_uxo_counts))
            exp_uxo_per_salvo = float(np.mean(sub_uxo_counts) * r_per_occ)
            p_any_uxo_salvo = float(1.0 - np.mean(sub_uxo_counts.reshape(n_occ_tot, r_per_occ).sum(axis=1) == 0))
            uxo_locs = [pt for b in batches for pt in b["sub_uxo_points"]][:500]
            uxo_summary = {
                "mode": "submunition_carrier",
                "submunitions_per_round": ammo.fuze.submunition_count,
                "total_uxo_sampled": tot_sub_uxo,
                "expected_uxo_per_salvo": exp_uxo_per_salvo,
                "p_at_least_one_uxo_per_salvo": p_any_uxo_salvo,
                "zero_failure_100_test_ucb95": fz_04_zero_failure_ucb(100, 0.95),
            }
        else:
            fz02_salvo = fz_02_uxo_count(r_per_occ, p_uxo_emp, rng=streams[9])
            uxo_idx = np.flatnonzero(is_uxo)[:500]
            uxo_locs = [
                {"range_m": float(imp_range[i]), "deflection_m": float(imp_defl[i])}
                for i in uxo_idx
            ]
            uxo_summary = {
                "mode": "unitary",
                "empirical_uxo_rate": p_uxo_emp,
                "total_uxo_sampled": int(np.sum(is_uxo)),
                "expected_uxo_per_salvo": fz02_salvo["expected_uxo"],
                "p_at_least_one_uxo_per_salvo": fz02_salvo["p_at_least_one_uxo"],
                "expected_uxo_total_run": float(n_tot * p_uxo_emp),
                "zero_failure_100_test_ucb95": fz_04_zero_failure_ucb(100, 0.95),
            }

        # Sensitivity Ranking (MC-10)
        if ammo.launch.mode == "static":
            y_target_sens = p_kill_round
        elif penetration_depths is not None and ammo.penetrator and ammo.penetrator.type in ("long_rod", "shaped_charge"):
            y_target_sens = penetration_depths
        else:
            y_target_sens = imp_range
        sensitivity = mc_10_sensitivity_indices(X_sens, y_target_sens, feature_names=sens_names)

        # Convergence Curve (25 checkpoints)
        step_pts = np.unique(np.linspace(max(2, n_tot // 25), n_tot, 25, dtype=int))
        conv_n: list[float] = []
        conv_mean_r: list[float] = []
        conv_hw_r: list[float] = []
        conv_pk: list[float] = []
        conv_hw_pk: list[float] = []
        for k_n in step_pts:
            c_r = mc_01_estimator_mean_se(imp_range[:k_n])
            c_p = mc_01_estimator_mean_se(p_kill_round[:k_n])
            conv_n.append(float(k_n))
            conv_mean_r.append(c_r["mean"])
            conv_hw_r.append(c_r["ci_half_width"])
            conv_pk.append(c_p["mean"])
            conv_hw_pk.append(c_p["ci_half_width"])

        convergence = {
            "n_rounds": conv_n,
            "mean_range_m": conv_mean_r,
            "range_ci95_half_width_m": conv_hw_r,
            "sspk": conv_pk,
            "sspk_ci95_half_width": conv_hw_pk,
        }

        # Governance Check for Safety Distance Outputs (App_Spec R25)
        safety_basis_ids = [
            fid for fid in ("TB-15", "EB-16", "TF-11", "MC-09", "TB-03", "TB-10") if fid in tracker.used_formula_ids
        ]
        if not safety_basis_ids:
            safety_basis_ids = sorted(tracker.used_formula_ids)
        gov_ok, gov_msg = validate_safety_distance_governance(safety_basis_ids)
        safety_governance = {
            "safety_distance_mode": scenario.safety_distance_mode,
            "basis_formula_ids": safety_basis_ids,
            "allowed": gov_ok,
            "message": gov_msg or "Safety-distance outputs backed by verified non-FLAG formulas.",
        }
        if scenario.safety_distance_mode and not gov_ok:
            raise GovernanceError(gov_msg or "Safety distance output blocked by FLAG governance.")

        prov = tracker.to_provenance_dict(
            seed=scenario.seed,
            sampler=scenario.sampler,
            fidelity=scenario.fidelity,
        )

    # Build Failure Modes & Effects Analysis (FMEA) Summary
    unique_fms, counts_fms = np.unique(failure_modes_all, return_counts=True)
    fm_counts: dict[str, int] = {str(k): int(v) for k, v in zip(unique_fms, counts_fms)}
    fm_fractions: dict[str, float] = {str(k): round(float(v) / max(n_tot, 1), 5) for k, v in zip(unique_fms, counts_fms)}
    nominal_count = fm_counts.get("Nominal Function", 0)
    failure_mode_summary: dict[str, Any] = {
        "total_rounds": int(n_tot),
        "nominal_function_count": int(nominal_count),
        "overall_system_reliability": round(float(nominal_count) / max(n_tot, 1), 5),
        "any_anomaly_or_failure_rate": round(1.0 - float(nominal_count) / max(n_tot, 1), 5),
        "dud_rate": round(float(np.mean(is_dud)), 5),
        "uxo_rate": round(float(np.mean(is_uxo)), 5),
        "self_destruct_neutralized_count": int(np.sum(is_dud & (~is_uxo))),
        "counts": fm_counts,
        "fractions": fm_fractions,
    }

    # Cap inline trial arrays to 5000 rows in JSON payload if gigantic, or keep all if <= 10000
    max_rows = min(n_tot, 10000)
    trials_dict: dict[str, list[float | int | bool | str]] = {
        "trial_id": list(range(max_rows)),
        "occasion_id": (np.arange(max_rows) // r_per_occ).tolist(),
        "range_m": np.round(imp_range[:max_rows], 4).tolist(),
        "deflection_m": np.round(imp_defl[:max_rows], 4).tolist(),
        "v_impact_mps": np.round(v_impact[:max_rows], 3).tolist(),
        "angle_of_fall_deg": np.round(np.degrees(aof_rad[:max_rows]), 3).tolist(),
        "tof_s": np.round(tof_s[:max_rows], 4).tolist(),
        "mv_mps": np.round(mv_trials[:max_rows], 3).tolist(),
        "is_dud": is_dud[:max_rows].tolist(),
        "is_uxo": is_uxo[:max_rows].tolist(),
        "failure_mode": [str(x) for x in failure_modes_all[:max_rows]],
        "p_kill": np.round(p_kill_round[:max_rows], 5).tolist(),
        "ps_target_kPa": np.round(ps_at_target[:max_rows], 3).tolist(),
    }
    if penetration_depths is not None:
        trials_dict["penetration_metric"] = np.round(penetration_depths[:max_rows], 5).tolist()

    sample_traj_out = (
        {k: np.round(v, 4).tolist() for k, v in sample_path.items()}
        if sample_path is not None
        else None
    )
    sample_trajs_out = [
        {k: np.round(v, 4).tolist() for k, v in p.items()}
        for p in sample_paths
    ]

    return SimulationResult(
        ammo_id=ammo.id,
        ammo_name=ammo.name,
        class_id=ammo.class_id,
        scenario_name=scenario.name,
        fidelity=scenario.fidelity,
        sampler=scenario.sampler,
        seed=scenario.seed,
        n_occasions=n_occ_tot,
        rounds_per_occasion=r_per_occ,
        n_total_rounds=n_tot,
        stopped_by_ci=stopped_by_ci,
        solved_qe_rad=qe_solved,
        solved_qe_mils=qe_solved / RAD_PER_NATO_MIL,
        dispersion=dispersion,
        effectiveness=effectiveness,
        blast_and_safety=blast_and_safety,
        fragmentation=fragmentation,
        penetration=pen_summary,
        uxo=uxo_summary,
        failure_mode_summary=failure_mode_summary,
        epistemic_summary=epistemic_summary,
        sensitivity=sensitivity,
        convergence=convergence,
        safety_governance=safety_governance,
        provenance=prov,
        trials=trials_dict,
        uxo_locations=uxo_locs,
        sample_trajectory=sample_traj_out,
        sample_trajectories=sample_trajs_out,
        hazard_grid=hazard_grid,
    )
