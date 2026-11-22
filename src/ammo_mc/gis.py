"""GIS Terrain / DEM Evaluator and RFC 7946 GeoJSON Hazard Contour Exporter (App_Spec R29)."""

from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.schemas import Scenario, SimulationResult, TerrainSpec

EARTH_RADIUS_M = 6378137.0


def build_terrain_callable(
    terrain: TerrainSpec | None,
    base_alt_m: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    """Return vectorised terrain function `z_ground(range_m, deflection_m)` or None if flat at 0."""
    if terrain is None:
        if abs(base_alt_m) < 1e-9:
            return None
        return lambda r, d: np.full_like(np.asarray(r, dtype=np.float64), base_alt_m)

    mode = terrain.mode.lower()
    if mode == "flat":
        if abs(base_alt_m) < 1e-9:
            return None
        return lambda r, d: np.full_like(np.asarray(r, dtype=np.float64), base_alt_m)

    if mode == "slope":
        grade = float(terrain.slope_grade)
        return lambda r, d: base_alt_m + grade * np.asarray(r, dtype=np.float64)

    if mode == "profile" and len(terrain.profile) >= 2:
        pts = sorted(terrain.profile, key=lambda p: p.range_m)
        rx = np.array([p.range_m for p in pts], dtype=np.float64)
        ez = np.array([p.elevation_m for p in pts], dtype=np.float64)
        return lambda r, d: np.interp(np.asarray(r, dtype=np.float64), rx, ez)

    if (
        mode == "grid"
        and len(terrain.grid_ranges_m) >= 2
        and len(terrain.grid_deflections_m) >= 2
        and len(terrain.grid_elevations_m) == len(terrain.grid_ranges_m)
    ):
        gx = np.asarray(terrain.grid_ranges_m, dtype=np.float64)
        gy = np.asarray(terrain.grid_deflections_m, dtype=np.float64)
        gz = np.asarray(terrain.grid_elevations_m, dtype=np.float64)
        interp = RegularGridInterpolator(
            (gx, gy),
            gz,
            bounds_error=False,
            fill_value=None,
        )

        def eval_grid(r: np.ndarray, d: np.ndarray) -> np.ndarray:
            r_arr = np.asarray(r, dtype=np.float64)
            d_arr = np.broadcast_to(np.asarray(d, dtype=np.float64), r_arr.shape)
            pts_q = np.column_stack([r_arr.ravel(), d_arr.ravel()])
            return interp(pts_q).reshape(r_arr.shape)

        return eval_grid

    return None


def range_deflection_to_lonlat(
    range_m: float | np.ndarray,
    deflection_right_m: float | np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    az_rad: float = 0.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert local ballistic `(range_m, deflection_right_m)` to WGS84 `(lon_deg, lat_deg)`.

    `az_rad` is clockwise from True North (0 = North, pi/2 = East).
    """
    r_arr = np.asarray(range_m, dtype=np.float64)
    d_arr = np.asarray(deflection_right_m, dtype=np.float64)

    # Local ENU offsets from weapon
    north_m = r_arr * math.cos(az_rad) - d_arr * math.sin(az_rad)
    east_m = r_arr * math.sin(az_rad) + d_arr * math.cos(az_rad)

    lat0_rad = math.radians(lat0_deg)
    dlat_deg = np.degrees(north_m / EARTH_RADIUS_M)
    dlon_deg = np.degrees(east_m / max(EARTH_RADIUS_M * math.cos(lat0_rad), 1.0))

    lat = lat0_deg + dlat_deg
    lon = lon0_deg + dlon_deg
    if np.ndim(range_m) == 0 and np.ndim(deflection_right_m) == 0:
        return float(lon), float(lat)
    return lon, lat


def _ellipse_ring_lonlat(
    center_range_m: float,
    center_defl_m: float,
    semi_range_m: float,
    semi_defl_m: float,
    lat0_deg: float,
    lon0_deg: float,
    az_rad: float,
    n_vertices: int = 64,
) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    r_pts = center_range_m + semi_range_m * np.cos(angles)
    d_pts = center_defl_m + semi_defl_m * np.sin(angles)
    lons, lats = range_deflection_to_lonlat(r_pts, d_pts, lat0_deg, lon0_deg, az_rad)
    ring = [[round(float(lo), 7), round(float(la), 7)] for lo, la in zip(lons, lats)]
    ring.append(ring[0])  # RFC 7946 closed linear ring
    return ring


def export_hazard_geojson(
    result: SimulationResult,
    scenario: Scenario | None = None,
) -> dict[str, Any]:
    """Export dispersion ellipses, hazard contours, and UXO map as RFC 7946 GeoJSON."""
    lat0 = scenario.weapon_location.lat if scenario is not None else 45.0
    lon0 = scenario.weapon_location.lon if scenario is not None else 0.0
    az_rad = scenario.aim.az_rad if scenario is not None else 0.0
    aim_range = scenario.aim.range_m if scenario is not None else result.dispersion.get("mean_range_m", 0.0)

    features: list[dict[str, Any]] = []

    # 1. Weapon Position
    w_lon, w_lat = range_deflection_to_lonlat(0.0, 0.0, lat0, lon0, az_rad)
    features.append({
        "type": "Feature",
        "properties": {"feature_type": "weapon_location", "label": "Weapon / Origin"},
        "geometry": {"type": "Point", "coordinates": [round(float(w_lon), 7), round(float(w_lat), 7)]},
    })

    # 2. Target / Aim Point
    t_lon, t_lat = range_deflection_to_lonlat(aim_range, 0.0, lat0, lon0, az_rad)
    features.append({
        "type": "Feature",
        "properties": {"feature_type": "aim_point", "label": "Aim / Target Point", "range_m": aim_range},
        "geometry": {"type": "Point", "coordinates": [round(float(t_lon), 7), round(float(t_lat), 7)]},
    })

    # 3. Mean Point of Impact (MPI) + CEP & 90% Dispersion Ellipses
    mpi_r = float(result.dispersion.get("mean_range_m", aim_range))
    mpi_d = float(result.dispersion.get("mean_deflection_m", 0.0))
    sig_r = float(result.dispersion.get("std_range_m", 0.0))
    sig_d = float(result.dispersion.get("std_deflection_m", 0.0))

    m_lon, m_lat = range_deflection_to_lonlat(mpi_r, mpi_d, lat0, lon0, az_rad)
    features.append({
        "type": "Feature",
        "properties": {
            "feature_type": "mpi",
            "label": "Mean Point of Impact",
            "mean_range_m": mpi_r,
            "mean_deflection_m": mpi_d,
            "cep_m": result.dispersion.get("cep_m", 0.0),
        },
        "geometry": {"type": "Point", "coordinates": [round(float(m_lon), 7), round(float(m_lat), 7)]},
    })

    if sig_r > 0.05 or sig_d > 0.05:
        sr = max(sig_r, 0.1)
        sd = max(sig_d, 0.1)
        # 50% bivariate normal ellipse scale: sqrt(-2 ln(0.5)) = 1.17741
        k50 = math.sqrt(2.0 * math.log(2.0))
        # 90% bivariate normal ellipse scale: sqrt(-2 ln(0.1)) = 2.14597
        k90 = math.sqrt(-2.0 * math.log(0.1))

        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "dispersion_ellipse",
                "probability": 0.50,
                "label": "50% Dispersion Contour (CEP)",
                "cep_m": result.dispersion.get("cep_m", 0.0),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_ring_lonlat(mpi_r, mpi_d, k50 * sr, k50 * sd, lat0, lon0, az_rad)],
            },
        })
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "dispersion_ellipse",
                "probability": 0.90,
                "label": "90% Dispersion Contour",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_ring_lonlat(mpi_r, mpi_d, k90 * sr, k90 * sd, lat0, lon0, az_rad)],
            },
        })

    # 4. Blast & Fragmentation Hazard Rings centered at MPI
    frag_haz_r = float(result.fragmentation.get("hazard_distance_quantile_m", 0.0) or 0.0)
    if frag_haz_r > 0.5:
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "fragment_hazard_contour",
                "quantile": result.fragmentation.get("quantile", 0.99),
                "radius_m": frag_haz_r,
                "label": f"MC Fragment 79 J Hazard Ring ({frag_haz_r:.1f} m)",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_ring_lonlat(mpi_r, mpi_d, frag_haz_r, frag_haz_r, lat0, lon0, az_rad)],
            },
        })

    iatg_lines = result.blast_and_safety.get("iatg_tb15_distances_m", {})
    if isinstance(iatg_lines, dict) and iatg_lines.get("frag_public_m", 0.0) > 0.5:
        r_pub = float(iatg_lines["frag_public_m"])
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "iatg_tb15_public_safety",
                "radius_m": r_pub,
                "formula_id": "TB-15",
                "label": f"IATG TB-15 Public Fragment Reference ({r_pub:.1f} m)",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_ring_lonlat(mpi_r, mpi_d, r_pub, r_pub, lat0, lon0, az_rad)],
            },
        })

    brick_cb = float(result.blast_and_safety.get("brick_house_cb_range_m", 0.0) or 0.0)
    if brick_cb > 0.5:
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "blast_damage_cb",
                "radius_m": brick_cb,
                "formula_id": "TB-10",
                "label": f"Brick House Serious Damage Cb ({brick_cb:.1f} m)",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_ring_lonlat(mpi_r, mpi_d, brick_cb, brick_cb, lat0, lon0, az_rad)],
            },
        })

    # 5. Sampled UXO Locations (up to 500 points to keep GeoJSON compact)
    for idx_u, uxo_pt in enumerate(result.uxo_locations[:500]):
        ur = float(uxo_pt["range_m"])
        ud = float(uxo_pt["deflection_m"])
        u_lon, u_lat = range_deflection_to_lonlat(ur, ud, lat0, lon0, az_rad)
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "uxo_location",
                "uxo_index": idx_u,
                "range_m": ur,
                "deflection_m": ud,
            },
            "geometry": {"type": "Point", "coordinates": [round(float(u_lon), 7), round(float(u_lat), 7)]},
        })

    return {
        "type": "FeatureCollection",
        "properties": {
            "ammo_id": result.ammo_id,
            "ammo_name": result.ammo_name,
            "scenario_name": result.scenario_name,
            "use_limitation": USE_LIMITATION_NOTICE,
            "has_flag_warnings": result.provenance.get("has_flag_warnings", False),
            "governance_warnings": result.provenance.get("governance_warnings", []),
        },
        "features": features,
    }
