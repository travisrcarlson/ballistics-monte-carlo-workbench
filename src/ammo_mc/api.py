"""FastAPI REST API server for the Ammunition Monte Carlo Simulator (ammo-mc)."""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ammo_mc.data import get_kb
from ammo_mc.export import geojson_to_str, summary_to_json_str
from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.gis import export_hazard_geojson
from ammo_mc.mcdrag import McDragInputs, estimate_mcdrag
from ammo_mc.pipeline import GovernanceError, run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.schemas import AmmoDefinition, Scenario, SimulationResult
from ammo_mc.validation import run_validation_suite

app = FastAPI(
    title="Ammunition Monte Carlo API (ammo-mc)",
    description=(
        "Production REST API for stochastic ammunition effects, trajectory dispersion, "
        "fuze reliability, and blast/fragmentation hazard modeling."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


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
