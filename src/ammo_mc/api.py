"""FastAPI REST API server for the Ammunition Monte Carlo Simulator (ammo-mc)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel, Field

from ammo_mc.auth import verify_app_password
from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.calibration import (
    calibrate_chronograph_from_csv,
    calibrate_drag_form_factor_from_csv,
    calibrate_shotgun_pattern_from_csv,
)
from ammo_mc.cuas_adversarial import run_adversarial_coevolution_ml
from ammo_mc.cuas_analytics import (
    analyze_cuas_sensitivity_tornado,
    build_cuas_site_geojson,
    compare_cuas_architectures_ab,
    get_cuas_comparator_presets,
    get_outer_layer_catalog,
    simulate_layered_cuas_defense_mc,
)
from ammo_mc.cuas_defense import (
    get_shotgun_presets,
    simulate_anti_drone_multishotgun_mc,
)
from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
from ammo_mc.cuas_mission import (
    generate_cuas_trade_study_html,
    get_cuas_ammo_economics_catalog,
    get_kill_chain_sensor_catalog,
    get_magazine_system_specs,
    get_site_defense_presets,
    simulate_cuas_kill_chain_mc,
    simulate_cuas_sustained_logistics_mc,
    simulate_multiturret_site_defense_mc,
)
from ammo_mc.data import get_kb
from ammo_mc.export import geojson_to_str, summary_to_json_str
from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.gis import export_hazard_geojson
from ammo_mc.literature import (
    build_literature_audit_summary,
    get_provenance_map,
    search_literature,
)
from ammo_mc.mcdrag import McDragInputs, estimate_mcdrag
from ammo_mc.pipeline import GovernanceError, run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.schemas import AmmoDefinition, Scenario, SimulationResult
from ammo_mc.small_arms import (
    generate_multishotgun_array_layout,
    get_drone_aspect_presets,
    get_drone_target_specs,
)
from ammo_mc.validation import (
    generate_vv_certificate_html,
    generate_vv_certificate_markdown,
    run_extended_vv_benchmarks,
    run_validation_suite,
)

app = FastAPI(
    title="Ammunition Monte Carlo API (ammo-mc)",
    description=(
        "Production REST API for stochastic ammunition effects, trajectory dispersion, "
        "fuze reliability, blast/fragmentation hazard modeling, and Counter-UAS trade studies."
    ),
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars into plain JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _verify_api_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Optional Bearer / X-API-Key authentication gated by AMMO_MC_API_REQUIRE_AUTH."""
    req_auth = os.environ.get("AMMO_MC_API_REQUIRE_AUTH", "").strip().lower() in ("1", "true", "yes")
    if not req_auth:
        return
    token = ""
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not verify_app_password(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.3.0"
    uptime_seconds: float
    use_limitation: str = USE_LIMITATION_NOTICE


class McDragRequest(BaseModel):
    calibre_m: float = Field(0.155, description="Calibre in metres")
    total_length_cal: float = Field(5.0, description="Total length in calibres")
    nose_length_cal: float = Field(2.5, description="Nose length in calibres")
    rt_over_r: float = Field(0.5, description="Tangent-radius / actual-ogive-radius")
    boattail_length_cal: float = Field(0.5, description="Boattail length in calibres")
    base_dia_cal: float = Field(0.85, description="Base diameter in calibres")
    meplat_dia_cal: float = Field(0.15, description="Meplat diameter in calibres")
    band_dia_cal: float = Field(1.0, description="Rotating band diameter in calibres")
    boundary_layer_code: str = Field("L/T", description="Boundary layer condition ('L/L', 'L/T', 'T/T')")
    identification: str = Field("Projectile", description="Identification tag")


class SimulateRequest(BaseModel):
    preset_id: Optional[str] = Field(None, description="Preset identifier to run (e.g. 'NOTIONAL-AR-01')")
    ammo: Optional[AmmoDefinition] = Field(None, description="Custom AmmoDefinition payload")
    scenario: Optional[Scenario] = Field(None, description="Custom Scenario payload")
    fidelity_override: Optional[str] = Field(None, description="Override fidelity ('L0', 'L1', 'L2', 'L3')")
    trials_override: Optional[int] = Field(None, description="Override trial / round count")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return API health status, version, and official use limitation notice."""
    return HealthResponse(
        status="ok",
        version="1.3.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
        use_limitation=USE_LIMITATION_NOTICE,
    )


@app.get("/taxonomy", tags=["Knowledge Base"])
def get_taxonomy() -> list[dict[str, Any]]:
    """List all 14 munition taxonomy classes with reference physical parameters."""
    kb = get_kb()
    return kb.taxonomy


@app.get("/formulas", tags=["Knowledge Base"])
def list_all_formulas(flagged_only: bool = Query(False, description="Filter for flagged formulas only")) -> list[dict[str, Any]]:
    """List all 96 governing formulas with source IDs, verification statuses, and FLAG warnings."""
    kb = get_kb()
    if flagged_only:
        return [f for f in kb.formulas if f.get("is_flagged")]
    return kb.formulas


@app.get("/literature", tags=["Knowledge Base"])
def get_literature_endpoint(
    query: str = Query("", description="Search author, title, DOI, or keyword"),
    category: Optional[str] = Query(None, description="Filter by domain category"),
    workbench_slug: Optional[str] = Query(None, description="Filter by workbench slug"),
    doi_only: bool = Query(False, description="Only entries with peer-reviewed/RFC DOIs"),
) -> dict[str, Any]:
    """Return scientific literature sources, module provenance map, and audit summary."""
    return {
        "summary": build_literature_audit_summary(),
        "sources": search_literature(query=query, category=category, workbench_slug=workbench_slug, doi_only=doi_only),
        "module_provenance": get_provenance_map(),
    }


@app.get("/presets", tags=["Presets"])
def get_all_presets() -> list[dict[str, Any]]:
    """Return all shipped notional ammunition presets across taxonomy families."""
    return list_presets()


@app.get("/presets/{preset_id}", tags=["Presets"])
def get_preset_by_id(preset_id: str) -> dict[str, Any]:
    """Retrieve full AmmoDefinition and default Scenario JSON for a specified preset."""
    try:
        ammo, scn = get_preset(preset_id)
        return {"ammo": ammo.model_dump(), "scenario": scn.model_dump()}
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.post("/mcdrag", tags=["Aerodynamics"])
def post_mcdrag(req: McDragRequest) -> dict[str, Any]:
    """Run McCoy (1974) MCDRAG aerodynamic drag coefficient estimator."""
    inp = McDragInputs(
        d_ref_mm=req.calibre_m * 1000.0,
        total_length_cal=req.total_length_cal,
        nose_length_cal=req.nose_length_cal,
        rt_over_r=req.rt_over_r,
        boattail_length_cal=req.boattail_length_cal,
        base_dia_cal=req.base_dia_cal,
        meplat_dia_cal=req.meplat_dia_cal,
        band_dia_cal=req.band_dia_cal,
        boundary_layer_code=req.boundary_layer_code,
        identification=req.identification,
    )
    try:
        return estimate_mcdrag(inp)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.get("/validate", tags=["Verification"])
def get_validation_suite(seed: int = 42, n_mc: int = 50_000) -> list[dict[str, Any]]:
    """Run the 22-case validation suite against Excel values and source tolerances."""
    return run_validation_suite(seed=seed, n_mc_v08=n_mc)


@app.get("/validate/benchmarks", tags=["Verification"])
def get_extended_vv_benchmarks(seed: int = 42, n_mc: int = 50_000) -> dict[str, Any]:
    """Run the 40-check V&V suite (22 workbook cases + 18 independent physics benchmarks)."""
    return _to_json_safe(run_extended_vv_benchmarks(seed=seed, n_mc_v08=n_mc))


@app.get("/validate/certificate", tags=["Verification"])
def get_vv_audit_certificate(
    format: str = Query("html", description="Certificate format: 'html', 'markdown', 'md', or 'json'."),
    seed: int = 42,
    n_mc: int = 50_000,
) -> Any:
    """Export an HTML, Markdown, or JSON V&V Audit Certificate."""
    fmt = format.strip().lower()
    if fmt not in ("html", "markdown", "md", "json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Choose from: 'html', 'markdown', 'md', or 'json'.",
        )
    report = run_extended_vv_benchmarks(seed=seed, n_mc_v08=n_mc)
    if fmt == "html":
        return HTMLResponse(content=generate_vv_certificate_html(report))
    if fmt in ("markdown", "md"):
        return Response(content=generate_vv_certificate_markdown(report), media_type="text/markdown; charset=utf-8")
    return _to_json_safe(report)


@app.post("/simulate", response_model=SimulationResult, tags=["Simulation"])
def post_simulate(req: SimulateRequest) -> SimulationResult:
    """Run a full Monte Carlo dispersion and terminal effects simulation."""
    if req.preset_id:
        try:
            ammo, scn = get_preset(req.preset_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    elif req.ammo and req.scenario:
        ammo = req.ammo
        scn = req.scenario
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either 'preset_id' or both 'ammo' and 'scenario'.",
        )

    if req.fidelity_override:
        scn.fidelity = req.fidelity_override.upper()  # type: ignore[assignment]
    if req.trials_override is not None:
        scn.n_trials = req.trials_override

    try:
        return run_simulation(ammo, scn)
    except GovernanceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Governance Block: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/export/geojson", tags=["Export"])
def post_export_geojson(req: SimulateRequest) -> dict[str, Any]:
    """Execute simulation and return RFC 7946 GeoJSON FeatureCollection."""
    sim_res = post_simulate(req)
    if req.preset_id:
        _, scn = get_preset(req.preset_id)
    else:
        scn = req.scenario
    return export_hazard_geojson(sim_res, scenario=scn)


# -------------------------------------------------------------------------
# COUNTER-UAS TURRET ENGINEERING & TRADE-STUDY ENDPOINTS
# -------------------------------------------------------------------------

class CuasArrayRequest(BaseModel):
    layout_type: str = "circular"
    n_barrels: int = 6
    mount_radius_m: float = 0.20
    spacing_y_m: float = 0.30
    spacing_z_m: float = 0.30
    pointing_mode: str = "convergent"
    splay_angle_mrad: float = 0.0
    firing_mode: str = "balanced_pair_ripple"
    ripple_delay_ms: float = 10.0
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    inner_ammunition_preset: Optional[str] = None
    pellets_per_shot: Optional[int] = None
    default_choke: str = "Modified"
    default_barrel_length_in: float = 26.0
    target_distance_m: float = 35.0
    drone_model: str = "commercial_quad_medium"
    aspect_mode: str = "head_on_frontal"
    drone_velocity_mps: float = 15.0
    crosswind_mps: float = 0.0
    headwind_mps: float = 0.0
    ambient_temp_c: float = 15.0
    propellant_type: str = "double_base"
    drag_model: str = "compressible_re_m"
    n_volleys: int = 20
    seed: int = 42


class CuasKillChainRequest(BaseModel):
    drone_model: str = "fpv_quad_small"
    aspect_mode: str = "head_on_frontal"
    drone_speed_mps: float = 28.0
    approach_altitude_m: float = 25.0
    weather: str = "clear_day"
    ambient_wind_mps: float = 5.0
    fiber_optic_autonomous: bool = False
    enabled_sensors: list[str] = Field(
        default_factory=lambda: ["micro_doppler_radar", "eoir_thermal_optical", "acoustic_array", "rf_c2_df"]
    )
    ai_classify_s: float = 0.6
    human_in_the_loop: bool = True
    human_auth_s: float = 1.2
    slew_angle_deg: float = 60.0
    turret_slew_rate_deg_s: float = 120.0
    max_weapon_range_m: float = 55.0
    keepout_breach_radius_m: float = 15.0
    n_barrels: int = 6
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    choke: str = "Modified"
    salvo_cycle_time_s: float = 0.85
    n_trials: int = 80
    seed: int = 42


class CuasSiteDefenseRequest(BaseModel):
    site_preset: str = "fob_compound_4_turret"
    custom_turrets: Optional[list[dict[str, Any]]] = None
    custom_obstacles: Optional[list[dict[str, Any]]] = None
    custom_nfz: Optional[list[dict[str, Any]]] = None
    terrain_preset: Optional[str] = None
    threat_altitude_m: float = 22.0
    threat_speed_mps: float = 22.0
    drone_model: str = "fpv_quad_small"
    weapon_max_range_m: float = 60.0
    elevation_sdz_range_m: float = 280.0
    n_raid_trials: int = 60
    seed: int = 42
    include_geojson: bool = False
    origin_lat: float = 32.9000
    origin_lon: float = -106.4000
    heading_deg: float = 0.0


class CuasUnifiedRaidRequest(BaseModel):
    site_preset: str = "fob_compound_4_turret"
    terrain_preset: Optional[str] = None
    sensors: list[str] = Field(
        default_factory=lambda: ["micro_doppler_radar", "eoir_thermal_optical", "rf_c2_df"]
    )
    weather_condition: str = "clear_day"
    n_drones: int = 10
    fiber_optic_fraction: float = 0.30
    drone_model: str = "fpv_quad_small"
    approach_pattern: str = "multi_vector_360"
    evasion_profile: str = "jinking_weave"
    threat_speed_mps: float = 22.0
    threat_altitude_m: float = 22.0
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    magazine_type: str = "box_10rd"
    weapon_max_range_m: float = 65.0
    n_mc_trials: int = 20
    seed: int = 42


class CuasLogisticsRequest(BaseModel):
    n_waves: int = 6
    drones_per_wave: int = 4
    wave_interval_s: float = 45.0
    n_barrels: int = 6
    ammo_family: str = "12G_NO4_BUCK_C_UAS"
    magazine_type: str = "box_10rd"
    barrel_mass_kg: float = 1.15
    ambient_temp_c: float = 22.0
    target_range_m: float = 40.0
    drone_model: str = "fpv_quad_small"
    aspect_mode: str = "head_on_frontal"
    n_trials: int = 40
    seed: int = 42


class CuasCompareABRequest(BaseModel):
    configs: Optional[list[dict[str, Any]]] = None
    drone_model: str = "fpv_quad_small"
    aspect_mode: str = "head_on_frontal"
    target_distance_m: float = 40.0
    drone_speed_mps: float = 25.0
    weather: str = "clear_day"
    fiber_optic_autonomous: bool = False
    n_swarm_drones: int = 6
    n_waves: int = 4
    drones_per_wave: int = 3
    seed: int = 42


class CuasLayeredRequest(BaseModel):
    outer_layer_id: str = "rf_gnss_jammer"
    outer_magazine: Optional[int] = None
    n_waves: int = 5
    drones_per_wave: int = 5
    wave_interval_s: float = 40.0
    fiber_optic_fraction: float = 0.25
    weather: str = "clear_day"
    inner_n_barrels: int = 6
    inner_ammo_family: str = "12G_NO4_BUCK_C_UAS"
    inner_magazine_type: str = "box_10rd"
    target_range_m: float = 40.0
    drone_model: str = "fpv_quad_small"
    n_trials: int = 50
    seed: int = 42


class CuasSensitivityRequest(BaseModel):
    baseline_cfg: Optional[dict[str, Any]] = None
    perturbation_pct: float = 20.0
    n_trials: int = 60
    seed: int = 42


class CuasAdversarialRequest(BaseModel):
    n_generations: int = 5
    n_mc_episodes: int = 18
    mount_load_limit_kn: float = 3.5
    max_barrels: int = 10
    max_swarm_size: int = 8
    exploration_beta: float = 1.5
    w_zero_breach: float = 0.40
    w_kill_rate: float = 0.40
    cost_penalty_weight: float = 0.20
    stage1_profile: Optional[dict[str, Any]] = None
    seed: int = 42
    allow_hybrid: bool = False
    robust_uncertainty_mode: bool = False
    calibration_profile: Optional[dict[str, Any]] = None


class CuasReportRequest(BaseModel):
    scenario_name: str = "Multi-Shotgun C-UAS Turret Engineering Trade Study"
    config: dict[str, Any] = Field(
        default_factory=lambda: {"n_barrels": 6, "ammo_family": "12G_NO4_BUCK_C_UAS", "target_range_m": 40.0}
    )
    array_summary: Optional[dict[str, Any]] = None
    aspect_table: Optional[list[dict[str, Any]]] = None
    kill_chain_summary: Optional[dict[str, Any]] = None
    logistics_summary: Optional[dict[str, Any]] = None
    economics_table: Optional[list[dict[str, Any]]] = None
    sdz_summary: Optional[dict[str, Any]] = None
    format: str = "html"


@app.get("/cuas/catalogs", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def get_cuas_catalogs() -> dict[str, Any]:
    """Return C-UAS shotgun, drone, aspect, sensor, site, comparator & outer-layer catalogs."""
    return _to_json_safe(
        {
            "shotgun_presets": get_shotgun_presets(),
            "drone_models": get_drone_target_specs(),
            "aspect_presets": get_drone_aspect_presets(),
            "kill_chain_sensors": get_kill_chain_sensor_catalog(),
            "site_presets": get_site_defense_presets(),
            "ammo_economics": get_cuas_ammo_economics_catalog(),
            "magazines": get_magazine_system_specs(),
            "comparator_presets": get_cuas_comparator_presets(),
            "outer_layers": get_outer_layer_catalog(),
        }
    )


@app.post("/cuas/array", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_array(req: CuasArrayRequest) -> dict[str, Any]:
    """Simulate Multi-Shotgun C-UAS Defense Array Monte Carlo volley engagement."""
    res = simulate_anti_drone_multishotgun_mc(**req.model_dump())
    return _to_json_safe(
        {
            "summary": res["summary"],
            "barrel_stats": res.get("barrel_stats", []),
            "component_kills": res.get("component_kills", {}),
        }
    )


class AnnexCPkRequest(BaseModel):
    distance_m: float = 50.0
    n_pellets: int = 170
    vulnerable_area_m2: float = 0.025
    theta_p_rad: float = 0.0067
    theta_aim_rad: float = 0.015
    pk_given_hit: float = 0.62
    n_shots: int = 3


@app.post("/cuas/annex-c-pk", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_annex_c_pk(req: AnnexCPkRequest) -> dict[str, Any]:
    """Compute Poisson-Gaussian analytical single-shot Pk,1 and multi-shot Pk,n."""
    from ammo_mc.small_arms import compute_annex_c_analytical_pk

    return _to_json_safe(compute_annex_c_analytical_pk(**req.model_dump()))


@app.post("/cuas/kill-chain", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_kill_chain(req: CuasKillChainRequest) -> dict[str, Any]:
    """Simulate Multi-Sensor C-UAS acquisition & human-in-the-loop kill chain."""
    return _to_json_safe(simulate_cuas_kill_chain_mc(**req.model_dump()))


@app.post("/cuas/site-defense", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_site_defense(req: CuasSiteDefenseRequest) -> dict[str, Any]:
    """Simulate multi-turret 360-deg coverage, obstacles, DEM terrain masking, NFZ fallout & GeoJSON."""
    dem_obj = None
    if req.terrain_preset and req.terrain_preset != "flat_airfield":
        from ammo_mc.terrain_dem import build_preset_terrain_dem

        dem_obj = build_preset_terrain_dem(req.terrain_preset, origin_lat=req.origin_lat, origin_lon=req.origin_lon)

    res = simulate_multiturret_site_defense_mc(
        site_preset=req.site_preset,
        custom_turrets=req.custom_turrets,
        custom_obstacles=req.custom_obstacles,
        custom_nfz=req.custom_nfz,
        terrain_dem=dem_obj,
        threat_altitude_m=req.threat_altitude_m,
        threat_speed_mps=req.threat_speed_mps,
        drone_model=req.drone_model,
        weapon_max_range_m=req.weapon_max_range_m,
        elevation_sdz_range_m=req.elevation_sdz_range_m,
        n_raid_trials=req.n_raid_trials,
        seed=req.seed,
    )
    if req.include_geojson:
        res["geojson"] = build_cuas_site_geojson(
            res,
            origin_lat=req.origin_lat,
            origin_lon=req.origin_lon,
            sdz_radius_m=req.elevation_sdz_range_m,
            heading_deg=req.heading_deg,
        )
    return _to_json_safe(res)


@app.post("/cuas/unified-raid", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_unified_raid(req: CuasUnifiedRaidRequest) -> dict[str, Any]:
    """Run Unified Dynamic 3D Cooperative Multi-Turret C-UAS Tactical Raid simulation."""
    dem_obj = None
    if req.terrain_preset and req.terrain_preset != "flat_airfield":
        from ammo_mc.terrain_dem import build_preset_terrain_dem

        dem_obj = build_preset_terrain_dem(req.terrain_preset)
    res = simulate_unified_tactical_raid_mc(
        site_preset=req.site_preset,
        terrain_dem=dem_obj,
        sensors=req.sensors,
        weather_condition=req.weather_condition,
        n_drones=req.n_drones,
        fiber_optic_fraction=req.fiber_optic_fraction,
        drone_model=req.drone_model,
        approach_pattern=req.approach_pattern,
        evasion_profile=req.evasion_profile,
        threat_speed_mps=req.threat_speed_mps,
        threat_altitude_m=req.threat_altitude_m,
        ammunition_preset=req.ammunition_preset,
        magazine_type=req.magazine_type,
        weapon_max_range_m=req.weapon_max_range_m,
        n_mc_trials=req.n_mc_trials,
        seed=req.seed,
    )
    return _to_json_safe(res)


@app.post("/cuas/logistics", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_logistics(req: CuasLogisticsRequest) -> dict[str, Any]:
    """Simulate sustained multi-wave barrel heating, reloads, and $/kill economics."""
    return _to_json_safe(simulate_cuas_sustained_logistics_mc(**req.model_dump()))


@app.post("/cuas/compare-ab", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_compare_ab(req: CuasCompareABRequest) -> dict[str, Any]:
    """Run normalized A/B/C turret architecture mission scorecard."""
    return _to_json_safe(compare_cuas_architectures_ab(**req.model_dump()))


@app.post("/cuas/layered-defense", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_layered(req: CuasLayeredRequest) -> dict[str, Any]:
    """Simulate Outer Soft/Hard-Kill + Inner Multi-Shotgun paired layered defense."""
    return _to_json_safe(simulate_layered_cuas_defense_mc(**req.model_dump()))


@app.post("/cuas/sensitivity", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_sensitivity(req: CuasSensitivityRequest) -> dict[str, Any]:
    """Run +/-% C-UAS Kill-Chain Tornado sensitivity & bottleneck ranking."""
    return _to_json_safe(analyze_cuas_sensitivity_tornado(**req.model_dump()))


@app.post("/cuas/adversarial", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_adversarial(req: CuasAdversarialRequest) -> dict[str, Any]:
    """Run Adversarial Red-vs-Blue Competing Monte Carlo co-evolution & GP ML protection optimizer."""
    return _to_json_safe(run_adversarial_coevolution_ml(**req.model_dump()))


class CuasRlPolicyRequest(BaseModel):
    blue_config: Optional[dict[str, Any]] = None
    n_generations: int = 10
    pop_size: int = 12
    n_eval_episodes: int = 12
    seed: int = 42


class CuasBattlespaceMlRequest(BaseModel):
    terrain_preset: str = "rolling_hills_fob"
    n_generations: int = 2
    n_mc_trials: int = 4
    max_turrets: int = 4
    seed: int = 42


class CuasTrackReplayRequest(BaseModel):
    csv_text: Optional[str] = None
    scenario: str = "evasive_fpv_dive"
    turret_config: Optional[dict[str, Any]] = None
    calibration_profile: Optional[dict[str, Any]] = None
    breach_radius_m: float = 12.0
    n_mc_trials: int = 25
    seed: int = 42


class CuasSwapCRequest(BaseModel):
    n_barrels: int = 6
    barrel_length_in: float = 24.0
    layout_type: str = "circular"
    mount_radius_m: float = 0.20
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    magazine_type: str = "box_10rd"
    slew_rate_deg_s: float = 160.0
    default_choke: str = "Modified"
    pointing_mode: str = "convergent"
    splay_angle_mrad: float = 0.0
    firing_mode: str = "balanced_pair_ripple"
    ripple_delay_ms: float = 10.0
    hybrid_mode: bool = False


class CuasWizardRequest(BaseModel):
    drone_model: str = "fpv_quad_small"
    threat_speed_mps: float = 25.0
    required_range_m: float = 45.0
    mount_class: str = "light_ugv"
    budget_priority: str = "balanced"
    seed: int = 42


class CuasDuelRequest(BaseModel):
    blue_config: dict[str, Any] = Field(default_factory=dict)
    red_config: dict[str, Any] = Field(default_factory=dict)
    mount_load_limit_kn: float = 4.0
    n_episodes: int = 24
    seed: int = 42


@app.post("/cuas/rl-policy", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_rl_policy(req: CuasRlPolicyRequest) -> dict[str, Any]:
    """Train and benchmark dynamic shot-by-shot neural fire-control policy."""
    from ammo_mc.cuas_rl_policy import train_fire_control_policy_es

    return _to_json_safe(train_fire_control_policy_es(**req.model_dump()))


@app.post("/cuas/battlespace-ml", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_battlespace_ml(req: CuasBattlespaceMlRequest) -> dict[str, Any]:
    """Co-evolve multi-turret 3D battlespace architectures against heterogeneous Red swarm packages."""
    from ammo_mc.cuas_battlespace_ml import run_battlespace_coevolution_mc

    return _to_json_safe(run_battlespace_coevolution_mc(**req.model_dump()))


@app.post("/cuas/track-replay", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_track_replay(req: CuasTrackReplayRequest) -> dict[str, Any]:
    """Replay recorded ENU/GPS drone flight tracks against calibrated multi-barrel turret."""
    from ammo_mc.cuas_track_replay import (
        generate_sample_drone_track_csv,
        simulate_recorded_track_engagement_mc,
    )

    csv_txt = req.csv_text or generate_sample_drone_track_csv(scenario=req.scenario, seed=req.seed)
    try:
        res = simulate_recorded_track_engagement_mc(
            csv_txt,
            turret_config=req.turret_config,
            calibration_profile=req.calibration_profile,
            breach_radius_m=req.breach_radius_m,
            n_mc_trials=req.n_mc_trials,
            seed=req.seed,
        )
        return _to_json_safe(res)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/swap-c", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_swap_c(req: CuasSwapCRequest) -> dict[str, Any]:
    """Compute C-UAS turret sub-assembly SWaP-C mass, power, inertia & BOM budget."""
    from ammo_mc.cuas_hardware import estimate_turret_swap_c

    return _to_json_safe(
        estimate_turret_swap_c(
            n_barrels=req.n_barrels,
            barrel_length_in=req.barrel_length_in,
            layout_type=req.layout_type,
            mount_radius_m=req.mount_radius_m,
            ammunition_preset=req.ammunition_preset,
            magazine_type=req.magazine_type,
            slew_rate_deg_s=req.slew_rate_deg_s,
            firing_mode=req.firing_mode,
            ripple_delay_ms=req.ripple_delay_ms,
            hybrid_mode=req.hybrid_mode,
        )
    )


@app.post("/cuas/cad/dxf", tags=["C-UAS", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_cad_dxf(req: CuasSwapCRequest) -> Response:
    """Export parametric 2D muzzle-clamp plate AutoCAD ASCII DXF drawing."""
    from ammo_mc.cuas_hardware import export_muzzle_plate_dxf

    bbls = generate_multishotgun_array_layout(
        layout_type=req.layout_type,
        n_barrels=req.n_barrels,
        mount_radius_m=req.mount_radius_m,
        default_choke=req.default_choke,
        default_barrel_length_in=req.barrel_length_in,
        default_ammunition_preset=req.ammunition_preset,
    )
    dxf_txt = export_muzzle_plate_dxf(
        barrels=bbls,
        layout_type=req.layout_type,
        title=f"{req.n_barrels}B {req.layout_type.title()} Muzzle Clamp",
        splay_angle_mrad=req.splay_angle_mrad,
        pointing_mode=req.pointing_mode,
    )
    return Response(
        content=dxf_txt.encode("utf-8"),
        media_type="application/dxf",
        headers={"Content-Disposition": "attachment; filename=cuas_muzzle_plate.dxf"},
    )


@app.post("/cuas/cad/svg", tags=["C-UAS", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_cad_svg(req: CuasSwapCRequest) -> Response:
    """Export parametric 2D muzzle-clamp plate scaled vector SVG blueprint."""
    from ammo_mc.cuas_hardware import export_muzzle_plate_svg

    bbls = generate_multishotgun_array_layout(
        layout_type=req.layout_type,
        n_barrels=req.n_barrels,
        mount_radius_m=req.mount_radius_m,
        default_choke=req.default_choke,
        default_barrel_length_in=req.barrel_length_in,
        default_ammunition_preset=req.ammunition_preset,
    )
    svg_txt = export_muzzle_plate_svg(
        barrels=bbls,
        layout_type=req.layout_type,
        title=f"{req.n_barrels}B {req.layout_type.title()} Muzzle Clamp",
        splay_angle_mrad=req.splay_angle_mrad,
        pointing_mode=req.pointing_mode,
    )
    return Response(
        content=svg_txt.encode("utf-8"),
        media_type="image/svg+xml",
        headers={"Content-Disposition": "attachment; filename=cuas_muzzle_plate.svg"},
    )


@app.post("/cuas/wizard", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_wizard(req: CuasWizardRequest) -> dict[str, Any]:
    """Run 3-Click C-UAS Turret Sizing Wizard across 12 turret archetypes & mount constraints."""
    from ammo_mc.cuas_wizard import run_turret_sizing_wizard

    return _to_json_safe(run_turret_sizing_wizard(**req.model_dump()))


@app.post("/cuas/duel", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_duel(req: CuasDuelRequest) -> dict[str, Any]:
    """Run Red-vs-Blue duel diagnostic with leaker failure-mode attribution & prescriptions."""
    from ammo_mc.cuas_wizard import run_red_vs_blue_duel_diagnostic

    return _to_json_safe(run_red_vs_blue_duel_diagnostic(**req.model_dump()))


class CuasStlAndFeaRequest(CuasSwapCRequest):
    plate_thickness_mm: float = 22.0
    material: str = "7075-T6"
    bolt_grade: str = "8.8"


@app.post("/cuas/cad/stl", tags=["C-UAS", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_cad_stl(req: CuasStlAndFeaRequest) -> Response:
    """Export parametric extruded 3D watertight ASCII STL solid of the muzzle clamp plate."""
    from ammo_mc.cuas_hardware import export_muzzle_plate_stl

    bbls = generate_multishotgun_array_layout(
        layout_type=req.layout_type,
        n_barrels=req.n_barrels,
        mount_radius_m=req.mount_radius_m,
        default_choke=req.default_choke,
        default_barrel_length_in=req.barrel_length_in,
        default_ammunition_preset=req.ammunition_preset,
    )
    stl_txt = export_muzzle_plate_stl(
        barrels=bbls,
        layout_type=req.layout_type,
        plate_thickness_mm=req.plate_thickness_mm,
        title=f"cuas_{req.layout_type}_{req.n_barrels}bbl_clamp",
    )
    return Response(
        content=stl_txt.encode("utf-8"),
        media_type="model/stl",
        headers={"Content-Disposition": "attachment; filename=cuas_muzzle_plate.stl"},
    )


@app.post("/cuas/hardware/fea", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_hardware_fea(req: CuasStlAndFeaRequest) -> dict[str, Any]:
    """Run first-principles bolt-shear, bearing & geometric ligament FEA sanity check."""
    from ammo_mc.cuas_hardware import estimate_turret_swap_c, evaluate_muzzle_plate_fea

    swap = estimate_turret_swap_c(
        n_barrels=req.n_barrels,
        barrel_length_in=req.barrel_length_in,
        layout_type=req.layout_type,
        mount_radius_m=req.mount_radius_m,
        ammunition_preset=req.ammunition_preset,
        firing_mode=req.firing_mode,
        ripple_delay_ms=req.ripple_delay_ms,
        hybrid_mode=req.hybrid_mode,
    )
    bbls = generate_multishotgun_array_layout(
        layout_type=req.layout_type,
        n_barrels=req.n_barrels,
        mount_radius_m=req.mount_radius_m,
        default_choke=req.default_choke,
        default_barrel_length_in=req.barrel_length_in,
        default_ammunition_preset=req.ammunition_preset,
    )
    return _to_json_safe(
        evaluate_muzzle_plate_fea(
            barrels=bbls,
            layout_type=req.layout_type,
            peak_recoil_force_kn=float(swap["summary"]["peak_recoil_force_kn"]),
            plate_thickness_mm=req.plate_thickness_mm,
            material=req.material,
            bolt_grade=req.bolt_grade,
        )
    )


class CuasSafetyEnvelopeRequest(BaseModel):
    n_barrels: int = 6
    barrels_per_burst: int = 2
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    default_choke: str = "Modified"
    default_barrel_length_in: float = 24.0
    firing_elevation_deg: float = 35.0
    firing_azimuth_deg: float = 0.0
    turret_height_m: float = 2.0
    wind_downrange_mps: float = 0.0
    wind_crossrange_mps: float = 0.0
    environment_type: str = "urban_permissive"
    calibration_profile: Optional[dict[str, Any]] = None
    n_sample_pellets: int = 200
    seed: int = 42


@app.post("/cuas/safety-envelope", tags=["C-UAS", "Safety"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_safety_envelope(req: CuasSafetyEnvelopeRequest) -> dict[str, Any]:
    """Compute acoustic muzzle-blast dB contours, danger cone & falling-pellet collateral risk tiers."""
    from ammo_mc.cuas_safety_envelope import evaluate_cuas_collateral_safety_envelope

    return _to_json_safe(evaluate_cuas_collateral_safety_envelope(**req.model_dump()))


class CuasHilTraceRequest(BaseModel):
    policy_dict: Optional[dict[str, Any]] = None
    track_csv_or_scenario: str = "evasive_fpv_dive"
    blue_config: Optional[dict[str, Any]] = None
    calibration_profile: Optional[dict[str, Any]] = None
    dt_ms: float = 100.0
    breach_radius_m: float = 12.0


@app.post("/cuas/hil-trace", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_hil_trace(req: CuasHilTraceRequest) -> dict[str, Any]:
    """Step RL fire-control policy through recorded drone telemetry millisecond-by-millisecond."""
    from ammo_mc.cuas_rl_policy import run_hil_fire_control_trace

    return _to_json_safe(
        run_hil_fire_control_trace(
            policy_or_dict=req.policy_dict,
            track_csv_or_scenario=req.track_csv_or_scenario,
            blue_config=req.blue_config,
            calibration_profile=req.calibration_profile,
            dt_ms=req.dt_ms,
            breach_radius_m=req.breach_radius_m,
        )
    )


class CalibrateMultiRangeRequest(BaseModel):
    rows_or_csv: Optional[Any] = None
    nominal_choke: str = "Modified"
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"


class CalibratePhotoPatternRequest(BaseModel):
    image_base64: Optional[str] = None
    use_synthetic_sample: bool = True
    board_width_m: float = 1.0
    board_height_m: float = 1.0
    range_m: float = 36.58
    nominal_choke: str = "Modified"
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    threshold_sigma: float = 2.6


class CalibrateLethalityRequest(BaseModel):
    trials_or_csv: Optional[Any] = None
    drone_model: str = "fpv_quad_small"
    base_twin: Optional[dict[str, Any]] = None


class CalibrateHardwareBenchRequest(BaseModel):
    nominal_turret_cfg: Optional[dict[str, Any]] = None
    bench_measurements: Optional[dict[str, Any]] = None


class CalibrateTrackerRequest(BaseModel):
    rows_or_csv: Optional[Any] = None
    sensor_preset: str = "Sensor_Fused_Radar_EOIR"


class CalibrateDesignDeltaRequest(BaseModel):
    turret_cfg: Optional[dict[str, Any]] = None
    calibration_twin: Optional[dict[str, Any]] = None
    drone_model: str = "fpv_quad_small"
    required_range_m: float = 45.0
    mount_class: str = "light_ugv"


class CalibrationProfileSaveRequest(BaseModel):
    name: str
    twin: dict[str, Any]
    notes: str = ""
    tags: Optional[list[str]] = None
    profile_id: Optional[str] = None


@app.post("/cuas/calibrate/multirange", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_multirange(req: CalibrateMultiRangeRequest) -> dict[str, Any]:
    """Fit multi-range pattern spread factor, range exponent & pellet Cd scale."""
    from ammo_mc.cuas_calibration_twin import (
        calibrate_multirange_pattern_and_drag,
        generate_sample_multirange_pattern_rows,
    )

    rows = req.rows_or_csv if req.rows_or_csv is not None else generate_sample_multirange_pattern_rows(nominal_choke=req.nominal_choke)
    try:
        return _to_json_safe(calibrate_multirange_pattern_and_drag(rows, nominal_choke=req.nominal_choke, ammunition_preset=req.ammunition_preset))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/calibrate/pattern-photo", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_pattern_photo(req: CalibratePhotoPatternRequest) -> dict[str, Any]:
    """Digitize target-board photo holes and fit calibrated Rayleigh dispersion."""
    import base64
    from ammo_mc.cuas_calibration_twin import (
        digitize_pattern_board_image,
        generate_synthetic_pattern_board_png,
    )

    if req.image_base64:
        img_bytes = base64.b64decode(req.image_base64)
    elif req.use_synthetic_sample:
        img_bytes = generate_synthetic_pattern_board_png(range_m=req.range_m, nominal_choke=req.nominal_choke)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide `image_base64` or set `use_synthetic_sample=true`.")
    try:
        return _to_json_safe(
            digitize_pattern_board_image(
                img_bytes,
                board_width_m=req.board_width_m,
                board_height_m=req.board_height_m,
                range_m=req.range_m,
                nominal_choke=req.nominal_choke,
                ammunition_preset=req.ammunition_preset,
                threshold_sigma=req.threshold_sigma,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/calibrate/lethality", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_lethality(req: CalibrateLethalityRequest) -> dict[str, Any]:
    """Bayesian MAP calibration of drone A_vuln & E_crit against live-fire shoot-down trials."""
    from ammo_mc.cuas_calibration_twin import (
        calibrate_livefire_drone_lethality,
        generate_sample_livefire_drone_trials,
    )

    trials = req.trials_or_csv if req.trials_or_csv is not None else generate_sample_livefire_drone_trials(drone_model=req.drone_model)
    try:
        return _to_json_safe(calibrate_livefire_drone_lethality(trials, drone_model=req.drone_model, base_twin=req.base_twin))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/calibrate/hardware", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_hardware(req: CalibrateHardwareBenchRequest) -> dict[str, Any]:
    """Calibrate turret recoil force, settle/slew times, mass & barrel zero scatter vs bench measurements."""
    from ammo_mc.cuas_calibration_twin import (
        calibrate_hardware_bench_tests,
        generate_sample_hardware_bench_tests,
    )

    bm = req.bench_measurements if req.bench_measurements is not None else generate_sample_hardware_bench_tests()
    return _to_json_safe(calibrate_hardware_bench_tests(req.nominal_turret_cfg, bm))


@app.post("/cuas/calibrate/tracker", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_tracker(req: CalibrateTrackerRequest) -> dict[str, Any]:
    """Calibrate sensor RMS angular tracking jitter, latency & OTM residual LOS jitter."""
    from ammo_mc.cuas_calibration_twin import (
        calibrate_tracker_telemetry,
        generate_sample_tracker_test_rows,
    )

    rows = req.rows_or_csv if req.rows_or_csv is not None else generate_sample_tracker_test_rows(sensor_preset=req.sensor_preset)
    try:
        return _to_json_safe(calibrate_tracker_telemetry(rows, sensor_preset=req.sensor_preset))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/calibrate/design-delta", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_design_delta(req: CalibrateDesignDeltaRequest) -> dict[str, Any]:
    """Compute Nominal vs Calibrated design drift, Top-3 design fixes & active test recommendations."""
    from ammo_mc.cuas_calibration_twin import (
        analyze_calibration_design_delta,
        recommend_next_range_tests,
    )

    drift = analyze_calibration_design_delta(**req.model_dump())
    plan = recommend_next_range_tests(
        req.turret_cfg,
        req.calibration_twin,
        drone_model=req.drone_model,
        target_range_m=req.required_range_m,
    )
    return _to_json_safe({"drift": drift, "test_plan": plan})


@app.get("/cuas/calibration-profiles", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def get_calibration_profiles(search: str = Query(default="")) -> dict[str, Any]:
    """List saved Digital Twin calibration profiles from persistent SQLite library."""
    from ammo_mc.cuas_calibration_twin import CalibrationProfileStore

    return _to_json_safe({"profiles": CalibrationProfileStore().list_profiles(search=search)})


@app.post("/cuas/calibration-profiles", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibration_profile(req: CalibrationProfileSaveRequest) -> dict[str, Any]:
    """Save or update a named Digital Twin calibration profile in SQLite store."""
    from ammo_mc.cuas_calibration_twin import CalibrationProfileStore

    return _to_json_safe(CalibrationProfileStore().save_profile(**req.model_dump()))


@app.post("/export/calibration-certificate", tags=["Calibration", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_export_calibration_certificate(req: CalibrateDesignDeltaRequest) -> Response:
    """Render and download a landscape PDF Empirical Calibration Certificate."""
    from ammo_mc.briefing_export import generate_calibration_certificate_pdf
    from ammo_mc.cuas_calibration_twin import (
        analyze_calibration_design_delta,
        merge_twin_updates,
        recommend_next_range_tests,
    )

    twin = merge_twin_updates(None, req.calibration_twin or {})
    drift = analyze_calibration_design_delta(
        req.turret_cfg, twin, drone_model=req.drone_model, required_range_m=req.required_range_m, mount_class=req.mount_class
    )
    plan = recommend_next_range_tests(req.turret_cfg, twin, drone_model=req.drone_model, target_range_m=req.required_range_m)
    pdf_bytes = generate_calibration_certificate_pdf(twin=twin, drift_res=drift, test_plan=plan)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cuas_calibration_certificate.pdf"},
    )


class CalibratePatternRequest(BaseModel):
    csv_text: str
    range_m: float = 36.58
    nominal_choke: str = "Modified"
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"


class CalibrateChronographRequest(BaseModel):
    csv_text: str
    nominal_mv_mps: Optional[float] = None
    std_temp_c: float = 21.1


class CalibrateDragRequest(BaseModel):
    csv_text: str
    preset_id: str = "NOTIONAL-SA-01"
    fit_mv: bool = False


@app.post("/calibrate/pattern", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_pattern(req: CalibratePatternRequest) -> dict[str, Any]:
    """Fit Rayleigh choke dispersion & POI bias from range patterning-board CSV."""
    try:
        return _to_json_safe(calibrate_shotgun_pattern_from_csv(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/calibrate/chronograph", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_chronograph(req: CalibrateChronographRequest) -> dict[str, Any]:
    """Fit chronograph MV mean, lot SD, and optional FM 6-40 dMV/dT temperature sensitivity."""
    try:
        return _to_json_safe(calibrate_chronograph_from_csv(**req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/calibrate/drag", tags=["Calibration"], dependencies=[Depends(_verify_api_auth)])
def post_calibrate_drag(req: CalibrateDragRequest) -> dict[str, Any]:
    """Fit exterior ballistic drag form factor (i) against range drop/velocity CSV."""
    try:
        return _to_json_safe(calibrate_drag_form_factor_from_csv(**req.model_dump()))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/cuas/report", tags=["C-UAS"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_report(req: CuasReportRequest) -> Any:
    """Generate standalone HTML C-UAS Trade-Study Report (or JSON envelope)."""
    html_doc = generate_cuas_trade_study_html(
        scenario_name=req.scenario_name,
        config=req.config,
        array_summary=req.array_summary,
        aspect_table=req.aspect_table,
        kill_chain_summary=req.kill_chain_summary,
        logistics_summary=req.logistics_summary,
        economics_table=req.economics_table,
        sdz_summary=req.sdz_summary,
    )
    if req.format.lower() == "json":
        return {"scenario_name": req.scenario_name, "html": html_doc}
    return HTMLResponse(content=html_doc)


class BriefingExportRequest(BaseModel):
    title: str = "C-UAS & Ballistics Executive Briefing"
    subtitle: str = "Monte Carlo Engineering Trade Study"
    ab_result: Optional[dict[str, Any]] = None
    site_result: Optional[dict[str, Any]] = None
    tornado_result: Optional[dict[str, Any]] = None
    logistics_result: Optional[dict[str, Any]] = None
    raid_result: Optional[dict[str, Any]] = None
    simulator_summary: Optional[dict[str, Any]] = None
    source_ids: Optional[list[str]] = None
    adversarial_result: Optional[dict[str, Any]] = None
    calibration_result: Optional[dict[str, Any]] = None


@app.post("/cuas/briefing/pdf", tags=["C-UAS", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_briefing_pdf(req: BriefingExportRequest) -> Response:
    """Export multi-page landscape Executive Briefing PDF."""
    pdf_bytes = generate_executive_briefing_pdf(**req.model_dump())
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cuas_executive_briefing.pdf"},
    )


@app.post("/cuas/briefing/pptx", tags=["C-UAS", "Export"], dependencies=[Depends(_verify_api_auth)])
def post_cuas_briefing_pptx(req: BriefingExportRequest) -> Response:
    """Export native 16:9 widescreen PowerPoint (.pptx) Executive Briefing deck."""
    pptx_bytes = generate_executive_briefing_pptx(**req.model_dump())
    return Response(
        content=pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": "attachment; filename=cuas_executive_briefing.pptx"},
    )
