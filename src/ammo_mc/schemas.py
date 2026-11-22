"""Pydantic v2 data models for AmmoDefinition, Scenario, SimulationResult, and Provenance.

Conforms to `App_Spec` R11 (`AmmoDefinition`) and R12 (`Scenario`), plus optional
GIS terrain (`App_Spec` R29) and 6-DOF / Guided plugin (`App_Spec` R28) schemas.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class InertiaSpec(BaseModel):
    ix: float = Field(default=0.0, description="Axial moment of inertia Ix (kg*m^2)")
    iy: float = Field(default=0.0, description="Transverse moment of inertia Iy (kg*m^2)")


class NormalSpec(BaseModel):
    mu: float = 1.0
    sigma: float = 0.0
    sigma_bias: float = 0.0


class MPMCoeffsSpec(BaseModel):
    c_da2: float = 4.0
    c_la: float = 2.2
    c_la3: float = 0.0
    c_ma: float = 2.8
    c_mag: float = 0.25
    c_spin: float = -0.015


class AeroSpec(BaseModel):
    drag_ref: Literal["G7", "G1", "FRAGMENT", "custom"] = "G7"
    cd_table: list[dict[str, float]] | None = None
    form_factor: NormalSpec = Field(default_factory=lambda: NormalSpec(mu=1.0, sigma=0.015))
    mpm_coeffs: MPMCoeffsSpec = Field(default_factory=MPMCoeffsSpec)


class MVSpec(BaseModel):
    mu: float = 800.0
    sigma_rr: float = 2.0
    sigma_lot: float = 1.5


class LaunchSpec(BaseModel):
    mode: Literal["gun", "rocket", "bomb", "static", "guided_miss"] = "gun"
    mv: MVSpec = Field(default_factory=MVSpec)
    twist_cal: float = Field(default=20.0, description="Rifling twist in calibres/turn (0 for fin-stabilised)")
    temp_coeff_mps_per_C: float = 0.35
    T_std_C: float = 21.1
    wear_efc: float = 0.0
    wear_coeff_mps_per_efc: float = 0.0
    # Extended-range / rocket / air-delivery parameters
    base_bleed_burn_s: float = 0.0
    base_bleed_dcd: float = 0.0
    rocket_burn_s: float = 0.0
    rocket_mdot_kg_s: float = 0.0
    rocket_ve_mps: float = 0.0
    release_alt_m: float = 0.0
    release_tas_mps: float = 0.0
    release_dive_rad: float = 0.0


class BetaReliabilitySpec(BaseModel):
    a: float = Field(default=98.0, description="Beta alpha parameter (successes + prior, or failures in FZ-03)")
    b: float = Field(default=2.0, description="Beta beta parameter (failures + prior)")
    p_dud_mean: float | None = Field(
        default=None,
        description="Optional direct deterministic/nominal dud probability override",
    )


class FuzeSpec(BaseModel):
    type: Literal["none", "PD", "delay", "time", "proximity", "PIBD", "submunition", "static"] = "PD"
    reliability: BetaReliabilitySpec = Field(default_factory=BetaReliabilitySpec)
    time_set_s: float | None = None
    time_sigma_s: float = 0.05
    hob: NormalSpec = Field(default_factory=lambda: NormalSpec(mu=0.0, sigma=0.0))
    graze_threshold_deg: float = 8.0
    min_arm_vel_mps: float = 50.0
    sd_fail_prob: float = 1.0  # 1.0 = no self-destruct backup; <1.0 = SD redundancy (FZ-07)
    common_cause_beta: float = 0.05
    submunition_count: int = 0
    submunition_pattern_radius_m: float = 50.0


class FillSpec(BaseModel):
    explosive: str = "Composition B"
    mass_kg: float = 6.8
    f_tnt_p_override: float | None = None
    f_tnt_i_override: float | None = None


class CaseSpec(BaseModel):
    mass_kg: float = 34.0
    t_in: float = 0.55
    d_in: float = 6.10
    length_m: float = 0.65
    material: str = "HF-1 / AISI 1045 Steel"
    rho_kg_m3: float = 7850.0


class WarheadSpec(BaseModel):
    fill: FillSpec = Field(default_factory=FillSpec)
    case: CaseSpec = Field(default_factory=CaseSpec)
    geometry: Literal["cylinder", "sphere", "bare"] = "cylinder"
    gurney_mps: float | None = None
    mott_B: float = 0.0779
    frag_shape: Literal["random", "cube", "sphere"] = "random"
    lethal_radius_m: float = 25.0
    carleton_rx_m: float = 25.0
    carleton_ry_m: float = 25.0
    # Open detonation / stack QD fields (DM-01, DM-02, ST-01)
    auw_kg: float | None = None
    stack_total_items: int = 1
    stack_surface_items: int = 1
    storage_mode: Literal["open", "ecm"] = "open"
    rc_room_vol_m3: float = 0.0
    rc_wall_t_m: float = 0.0
    thermal_flux_kw_m2_at_100m: float = 0.0
    thermal_duration_s: float = 2.5


class PenetratorSpec(BaseModel):
    type: Literal["none", "long_rod", "shaped_charge", "bullet_barrier"] = "none"
    L_m: float = 0.55
    D_m: float = 0.025
    rho: float = 17600.0
    rod_material: str = "Tungsten alloy"
    jet_length_m: float = 0.45
    jet_rho_kg_m3: float = 8960.0
    standoff_cd: float = 3.5
    v50_mps: float = 450.0
    sigma_v50_mps: float = 25.0


class FailureModesSpec(BaseModel):
    """Multi-stage stochastic failure mode probabilities (Launch -> Flight -> Fuze -> Warhead FMEA)."""

    p_propulsion_anomaly: float = Field(
        default=0.0,
        description="Probability of propellant hangfire, base-bleed ignition failure, or motor under-burn",
    )
    mv_drop_frac: float = Field(
        default=0.15,
        description="Fractional muzzle/burnout velocity loss when a propulsion anomaly occurs",
    )
    p_fin_or_sabot_anomaly: float = Field(
        default=0.0,
        description="Probability of fin deployment failure or asymmetric sabot discard jump inflation",
    )
    p_fuze_arming_fail: float = Field(
        default=0.0,
        description="Probability of Safe & Arm (S&A) mechanical/electronic arming failure",
    )
    p_fuze_early_burst: float = Field(
        default=0.0,
        description="Probability of premature in-flight / rain-impact airburst along trajectory",
    )
    p_low_order_detonation: float = Field(
        default=0.0,
        description="Probability of incomplete explosive train transfer -> partial low-order detonation",
    )
    low_order_yield_frac: float = Field(
        default=0.25,
        description="Effective warhead yield fraction when a low-order partial detonation occurs",
    )


class AmmoDefinition(BaseModel):
    """Complete JSON-serializable Ammunition Definition per App_Spec R11."""

    id: str
    class_id: str
    name: str
    notional: bool = True
    description: str = ""
    calibre_m: float
    mass_kg: float
    mass_sigma_kg: float = 0.0
    length_m: float | None = None
    inertia: InertiaSpec = Field(default_factory=InertiaSpec)
    aero: AeroSpec = Field(default_factory=AeroSpec)
    launch: LaunchSpec = Field(default_factory=LaunchSpec)
    fuze: FuzeSpec = Field(default_factory=FuzeSpec)
    warhead: WarheadSpec | None = None
    penetrator: PenetratorSpec | None = None
    failure_modes: FailureModesSpec = Field(default_factory=FailureModesSpec)
    guidance_plugin: str | None = None
    guidance_params: dict[str, Any] = Field(default_factory=dict)
    six_dof_plugin: str | None = None
    sources: list[str] = Field(default_factory=lambda: ["S01", "S10", "S14", "S32"])


# ---------------------------------------------------------------------------
# Scenario Models (App_Spec R12)
# ---------------------------------------------------------------------------

class GeoLocation(BaseModel):
    lat: float = 45.0
    lon: float = 0.0
    alt: float = 0.0


class TargetSpec(BaseModel):
    type: Literal["point", "area", "personnel", "structure", "armour"] = "point"
    posture: Literal["standing", "prone", "crouching", "open", "brick_house", "armoured"] = "standing"
    geometry: dict[str, float] = Field(
        default_factory=lambda: {
            "half_width_x_m": 50.0,
            "half_width_y_m": 50.0,
            "presented_area_m2": 0.56,
            "armour_thickness_m": 0.40,
            "armour_bhn": 300.0,
            "armour_rho_kg_m3": 7850.0,
            "obliquity_deg": 30.0,
            "thor_plate_cm": 0.635,
            "exposed_site_distance_m": 400.0,
        }
    )
    thor_material: str = "RHA armor"
    window_size: Literal["small", "medium", "large"] = "medium"
    brick_damage_category: Literal["A", "B", "CB", "CA", "D"] = "CB"
    enable_lung_probit_flag: bool = False
    enable_thor_penetration_flag: bool = False


class AimSpec(BaseModel):
    range_m: float = 10000.0
    az_rad: float = 0.0
    qe_rad: float | None = None
    high_angle: bool = False
    propellant_temp_C: float = 21.1


class MetDeviations(BaseModel):
    dT_K: float = 0.0
    p_sl_Pa: float = 101325.0


class WindBandSpec(BaseModel):
    alt_m: float
    wind_range_mps: float = 0.0
    wind_cross_right_mps: float = 0.0
    wind_up_mps: float = 0.0


class MetSpec(BaseModel):
    mode: Literal["ISA", "message"] = "ISA"
    deviations: MetDeviations = Field(default_factory=MetDeviations)
    wind_profile: list[WindBandSpec] = Field(default_factory=list)
    wind_range_mps: float = 0.0
    wind_cross_right_mps: float = 0.0


class BiasErrorsSpec(BaseModel):
    """Per-occasion shared biases (DA-01 / Uncertainty_Inputs)."""

    tle_range_sigma_m: float = 0.0
    tle_deflection_sigma_m: float = 0.0
    aim_az_bias_sigma_mils: float = 0.5
    aim_qe_bias_sigma_mils: float = 0.5
    wind_range_bias_sigma_mps: float = 1.0
    wind_cross_bias_sigma_mps: float = 1.0
    temp_bias_sigma_K: float = 1.0
    propellant_temp_sigma_C: float = 0.5
    # Direct L0 analytic bias sigmas (m) if running L0 miss-distance mode
    l0_bias_range_sigma_m: float = 15.0
    l0_bias_deflection_sigma_m: float = 15.0


class RandomErrorsSpec(BaseModel):
    """Per-round independent random errors."""

    aim_az_rand_sigma_mils: float = 0.8
    aim_qe_rand_sigma_mils: float = 0.8
    jump_az_sigma_mils: float = 0.3
    jump_qe_sigma_mils: float = 0.3
    # Direct L0 analytic round-to-round sigmas (m)
    l0_rr_range_sigma_m: float = 25.0
    l0_rr_deflection_sigma_m: float = 20.0
    # Air-delivery release errors
    release_alt_sigma_m: float = 10.0
    release_tas_sigma_mps: float = 1.5
    release_time_sigma_s: float = 0.05


class ErrorBudgetSpec(BaseModel):
    bias: BiasErrorsSpec = Field(default_factory=BiasErrorsSpec)
    random: RandomErrorsSpec = Field(default_factory=RandomErrorsSpec)
    rho_range_deflection: float = 0.0
    correlation_matrix: list[list[float]] | None = None


class TerrainPointSpec(BaseModel):
    range_m: float
    elevation_m: float


class TerrainSpec(BaseModel):
    """Optional DEM / 1-D or 2-D elevation profile per App_Spec R29."""

    mode: Literal["flat", "slope", "profile", "grid"] = "flat"
    slope_grade: float = 0.0
    profile: list[TerrainPointSpec] = Field(default_factory=list)
    grid_ranges_m: list[float] = Field(default_factory=list)
    grid_deflections_m: list[float] = Field(default_factory=list)
    grid_elevations_m: list[list[float]] = Field(default_factory=list)


class Scenario(BaseModel):
    """Complete JSON-serializable Scenario definition per App_Spec R12."""

    name: str = "Standard Evaluation Scenario"
    weapon_location: GeoLocation = Field(default_factory=GeoLocation)
    target: TargetSpec = Field(default_factory=TargetSpec)
    aim: AimSpec = Field(default_factory=AimSpec)
    met: MetSpec = Field(default_factory=MetSpec)
    errors: ErrorBudgetSpec = Field(default_factory=ErrorBudgetSpec)
    terrain: TerrainSpec = Field(default_factory=TerrainSpec)

    fidelity: Literal["L0", "L1", "L2", "L3"] = "L1"
    integrator: Literal["RK4", "RK45"] = "RK4"
    dt_s: float | None = None
    vacuum: bool = False
    include_coriolis: bool = True

    n_occasions: int = 50
    rounds_per_occasion: int = 4
    n_trials: int | None = None

    two_loop_epistemic: bool = False
    n_outer_epistemic: int = 15

    ci_tolerance: float | None = None
    n_max: int = 20000
    hazard_quantile: float = 0.99
    fragments_per_detonation: int = 120
    safety_distance_mode: bool = False

    sampler: Literal["MC", "LHS", "Sobol"] = "LHS"
    seed: int | None = 42
    record_trajectories_count: int = 12


class SimulationResult(BaseModel):
    """Structured simulation result & Standard Report per App_Spec R21 & R22."""

    ammo_id: str
    ammo_name: str
    class_id: str
    scenario_name: str
    fidelity: str
    sampler: str
    seed: int | None
    n_occasions: int
    rounds_per_occasion: int
    n_total_rounds: int
    stopped_by_ci: bool = False
    solved_qe_rad: float = 0.0
    solved_qe_mils: float = 0.0

    dispersion: dict[str, float]
    effectiveness: dict[str, Any]
    blast_and_safety: dict[str, Any]
    fragmentation: dict[str, Any]
    penetration: dict[str, Any] | None = None
    uxo: dict[str, Any]
    failure_mode_summary: dict[str, Any] = Field(default_factory=dict)
    epistemic_summary: dict[str, Any] | None = None
    sensitivity: list[dict[str, Any]]
    convergence: dict[str, list[float]]
    safety_governance: dict[str, Any]
    provenance: dict[str, Any]

    # Sample / trial arrays for plotting & CSV export
    trials: dict[str, list[float | int | bool | str]]
    uxo_locations: list[dict[str, float]] = Field(default_factory=list)
    sample_trajectory: dict[str, list[float]] | None = None
    sample_trajectories: list[dict[str, list[float]]] = Field(default_factory=list)
    hazard_grid: dict[str, Any] | None = None
