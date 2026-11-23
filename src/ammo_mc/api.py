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
from ammo_mc.small_arms import get_drone_aspect_presets, get_drone_target_specs
from ammo_mc.validation import run_validation_suite

app = FastAPI(
    title="Ammunition Monte Carlo API (ammo-mc)",
    description=(
        "Production REST API for stochastic ammunition effects, trajectory dispersion, "
        "fuze reliability, blast/fragmentation hazard modeling, and Counter-UAS trade studies."
    ),
    version="1.2.0",
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
    version: str = "1.0.0"
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
        version="1.0.0",
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
    pointing_mode: str = "convergent"
    splay_angle_mrad: float = 0.0
    firing_mode: str = "balanced_pair_ripple"
    ripple_delay_ms: float = 10.0
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS"
    default_choke: str = "Modified"
    target_distance_m: float = 35.0
    drone_model: str = "commercial_quad_medium"
    aspect_mode: str = "head_on_frontal"
    drone_velocity_mps: float = 15.0
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
