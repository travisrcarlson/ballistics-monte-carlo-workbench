"""Digital Elevation Model (DEM) Terrain Grid, 3D Line-of-Sight Masking & Terrain-Adjusted Fallout.

Pure NumPy/SciPy domain engine (zero GDAL/rasterio dependencies) supporting:
- Deterministic synthetic tactical terrain presets (flat airfield, rolling hills, ridge/valley, coastal bluff)
- CSV DEM parsing (`x_m, y_m, z_m` or `lat, lon, elev_m`)
- ESRI ASCII Grid (.asc) DEM parsing
- Optional online Open-Meteo Elevation API batch DEM fetch for any real-world lat/lon anchor
- 3D ray-cast Line-of-Sight (LOS) terrain masking & 360-deg horizon mask profile
- Terrain-slope-adjusted elevated shotgun pellet Fallout SDZ & terminal kinetic energy
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

from ammo_mc.cuas_defense import _sphere_drag_coeff
from ammo_mc.gis import EARTH_RADIUS_M
from ammo_mc.small_arms import get_shotgun_presets

TERRAIN_DEM_PRESETS: dict[str, dict[str, str]] = {
    "flat_airfield": {
        "name": "Flat Airfield / Desert Ramp (0 m Datum)",
        "description": "Zero-relief paved apron baseline without terrain ridge masking.",
    },
    "rolling_hills_fob": {
        "name": "Rolling Hills FOB Knoll (+4 m Compound / +24 m Perimeter Hills)",
        "description": "Gently undulating foothills surrounding an elevated compound knoll.",
    },
    "ridge_valley_corridor": {
        "name": "Flanking Ridge & Valley Corridor (+36 m NE Ridge / -10 m SW Draw)",
        "description": "Steep 36 m ridgeline ~95 m NE that shadows low-altitude approaches plus downslope SW draw.",
    },
    "coastal_bluff": {
        "name": "Coastal Bluff / Escarpment (+22 m West Bluff -> 1 m East Apron)",
        "description": "Elevated coastal defense bluff overlooking low sea-level eastern approach.",
    },
}


def get_terrain_dem_presets() -> dict[str, dict[str, str]]:
    """Return catalog of built-in tactical DEM terrain presets."""
    return TERRAIN_DEM_PRESETS


@dataclass
class TerrainDEMGrid:
    """Regular 2D Easting/Northing Digital Elevation Model with fast bilinear lookup."""

    name: str
    source_label: str
    x_m: np.ndarray  # sorted 1D Easting offsets (m), length nx
    y_m: np.ndarray  # sorted 1D Northing offsets (m), length ny
    z_grid_m: np.ndarray  # 2D elevation grid (m MSL or relative datum), shape (ny, nx)
    origin_lat: float = 32.9000
    origin_lon: float = -106.4000

    def __post_init__(self) -> None:
        self.x_m = np.asarray(self.x_m, dtype=np.float64)
        self.y_m = np.asarray(self.y_m, dtype=np.float64)
        self.z_grid_m = np.asarray(self.z_grid_m, dtype=np.float64)
        if self.x_m.ndim != 1 or self.y_m.ndim != 1:
            raise ValueError("DEM x_m and y_m must be 1D coordinate vectors.")
        if self.z_grid_m.shape != (self.y_m.size, self.x_m.size):
            raise ValueError(
                f"DEM z_grid_m shape {self.z_grid_m.shape} does not match (len(y_m)={self.y_m.size}, len(x_m)={self.x_m.size})."
            )
        if self.x_m.size < 2 or self.y_m.size < 2:
            raise ValueError("DEM grid must be at least 2x2.")
        self._interp = RegularGridInterpolator(
            (self.y_m, self.x_m),
            self.z_grid_m,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def elevation_at(
        self,
        x_m: float | np.ndarray,
        y_m: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return bilinearly interpolated ground elevation (m), clamped safely to DEM extent."""
        x_arr = np.asarray(x_m, dtype=np.float64)
        y_arr = np.asarray(y_m, dtype=np.float64)
        xc = np.clip(x_arr, float(self.x_m[0]), float(self.x_m[-1]))
        yc = np.clip(y_arr, float(self.y_m[0]), float(self.y_m[-1]))
        if xc.ndim == 0 and yc.ndim == 0:
            return float(self._interp(np.array([[float(yc), float(xc)]], dtype=np.float64))[0])
        xc_b, yc_b = np.broadcast_arrays(xc, yc)
        pts = np.column_stack([yc_b.ravel(), xc_b.ravel()])
        out = self._interp(pts).reshape(xc_b.shape)
        return out

    def slope_deg_at(self, x_m: float, y_m: float, h_m: float = 4.0) -> float:
        """Compute local terrain slope magnitude in degrees via central finite differences."""
        dz_dx = (self.elevation_at(x_m + h_m, y_m) - self.elevation_at(x_m - h_m, y_m)) / (2.0 * h_m)
        dz_dy = (self.elevation_at(x_m, y_m + h_m) - self.elevation_at(x_m, y_m - h_m)) / (2.0 * h_m)
        return float(math.degrees(math.atan(math.hypot(float(dz_dx), float(dz_dy)))))

    def stats(self) -> dict[str, Any]:
        """Compute summary relief and slope statistics across the DEM."""
        dx = float(np.mean(np.diff(self.x_m)))
        dy = float(np.mean(np.diff(self.y_m)))
        gy, gx = np.gradient(self.z_grid_m, max(dy, 1e-6), max(dx, 1e-6))
        slopes = np.degrees(np.arctan(np.hypot(gx, gy)))
        z_min = float(np.min(self.z_grid_m))
        z_max = float(np.max(self.z_grid_m))
        return {
            "name": self.name,
            "source_label": self.source_label,
            "min_m": round(z_min, 2),
            "max_m": round(z_max, 2),
            "mean_m": round(float(np.mean(self.z_grid_m)), 2),
            "relief_m": round(z_max - z_min, 2),
            "mean_slope_deg": round(float(np.mean(slopes)), 2),
            "max_slope_deg": round(float(np.max(slopes)), 2),
            "grid_shape": f"{self.z_grid_m.shape[0]}x{self.z_grid_m.shape[1]}",
            "extent_m": round(max(float(self.x_m[-1] - self.x_m[0]), float(self.y_m[-1] - self.y_m[0])), 1),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize DEM into a JSON-compatible dictionary for Streamlit caching."""
        return {
            "name": self.name,
            "source_label": self.source_label,
            "x_m": [round(float(v), 3) for v in self.x_m],
            "y_m": [round(float(v), 3) for v in self.y_m],
            "z_grid_m": [[round(float(v), 3) for v in row] for row in self.z_grid_m],
            "origin_lat": float(self.origin_lat),
            "origin_lon": float(self.origin_lon),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainDEMGrid":
        return cls(
            name=str(data.get("name", "Custom DEM")),
            source_label=str(data.get("source_label", "Serialized DEM")),
            x_m=np.asarray(data["x_m"], dtype=np.float64),
            y_m=np.asarray(data["y_m"], dtype=np.float64),
            z_grid_m=np.asarray(data["z_grid_m"], dtype=np.float64),
            origin_lat=float(data.get("origin_lat", 32.9000)),
            origin_lon=float(data.get("origin_lon", -106.4000)),
        )


def build_preset_terrain_dem(
    preset_key: str = "ridge_valley_corridor",
    extent_m: float = 520.0,
    grid_n: int = 57,
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
) -> TerrainDEMGrid:
    """Build one of the built-in synthetic tactical DEM terrain grids."""
    n = max(15, int(grid_n))
    x_v = np.linspace(-extent_m, extent_m, n, dtype=np.float64)
    y_v = np.linspace(-extent_m, extent_m, n, dtype=np.float64)
    X, Y = np.meshgrid(x_v, y_v, indexing="xy")
    r2 = X * X + Y * Y

    if preset_key == "flat_airfield":
        Z = np.zeros_like(X, dtype=np.float64)
    elif preset_key == "rolling_hills_fob":
        knoll = 4.0 * np.exp(-r2 / (70.0 ** 2))
        nw_hill = 38.0 * np.exp(-(((X + 185.0) ** 2 + (Y - 170.0) ** 2) / (110.0 ** 2)))
        se_hill = 32.0 * np.exp(-(((X - 205.0) ** 2 + (Y + 150.0) ** 2) / (115.0 ** 2)))
        sw_valley = -18.0 * np.exp(-(((X + 210.0) ** 2 + (Y + 210.0) ** 2) / (130.0 ** 2)))
        undulation = 2.5 * np.sin(X / 65.0) * np.cos(Y / 70.0)
        Z = knoll + nw_hill + se_hill + sw_valley + undulation
    elif preset_key == "coastal_bluff":
        escarpment = 28.0 / (1.0 + np.exp((X - 30.0) / 34.0))
        berm = 18.0 * np.exp(-((X - 42.0) ** 2 + (Y - 10.0) ** 2) / (28.0 ** 2))
        Z = 1.0 + escarpment + berm
    else:
        preset_key = "ridge_valley_corridor"
        # Intervening 36 m rocky ridge berm around ~(38, 38) between compound turrets and outer NE ring,
        # plus outer +58 m NE hillside and -32 m SW draw inside +/-520 m so Fallout SDZ sees full slope delta.
        inner_ridge = 36.0 * np.exp(
            -(((0.707 * (X - 38.0) + 0.707 * (Y - 38.0)) / 6.5) ** 2)
            - (((-0.707 * (X - 38.0) + 0.707 * (Y - 38.0)) / 36.0) ** 2)
        )
        outer_ridge = 58.0 * np.exp(-(((X - 210.0) ** 2 + (Y - 210.0) ** 2) / (110.0 ** 2)))
        draw = -32.0 * np.exp(-(((X + 230.0) ** 2 + (Y + 230.0) ** 2) / (140.0 ** 2)))
        subtle = 1.5 * np.sin(X / 80.0) * np.sin(Y / 80.0)
        Z = np.maximum(inner_ridge + outer_ridge + draw + subtle + 1.0, -34.0)

    info = TERRAIN_DEM_PRESETS.get(preset_key, TERRAIN_DEM_PRESETS["ridge_valley_corridor"])
    return TerrainDEMGrid(
        name=info["name"],
        source_label=f"Synthetic Tactical Preset ({preset_key})",
        x_m=x_v,
        y_m=y_v,
        z_grid_m=Z,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def parse_dem_csv(
    csv_text: str,
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
    name: str = "Uploaded CSV DEM",
    grid_n: int = 41,
) -> TerrainDEMGrid:
    """Parse scattered or gridded DEM CSV (`x_m,y_m,z_m` or `lat,lon,elev_m`) onto regular grid."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("CSV DEM has no header row.")
    fields_low = {f.strip().lower(): f for f in reader.fieldnames}

    def _find_col(*cands: str) -> str | None:
        for c in cands:
            if c in fields_low:
                return fields_low[c]
        return None

    col_x = _find_col("x_m", "easting_m", "east_m", "x")
    col_y = _find_col("y_m", "northing_m", "north_m", "y")
    col_lat = _find_col("lat", "latitude", "lat_deg")
    col_lon = _find_col("lon", "lng", "longitude", "lon_deg")
    col_z = _find_col("z_m", "elev_m", "elevation_m", "elevation", "z", "alt_m")

    if col_z is None:
        raise ValueError("CSV DEM requires an elevation column (`z_m`, `elev_m`, or `elevation_m`).")

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    cos_lat = max(math.cos(math.radians(origin_lat)), 0.1)

    for row in reader:
        try:
            zv = float(row[col_z])
            if col_x and col_y:
                xv = float(row[col_x])
                yv = float(row[col_y])
            elif col_lat and col_lon:
                lat_v = float(row[col_lat])
                lon_v = float(row[col_lon])
                xv = math.radians(lon_v - origin_lon) * EARTH_RADIUS_M * cos_lat
                yv = math.radians(lat_v - origin_lat) * EARTH_RADIUS_M
            else:
                raise ValueError("CSV DEM needs either (`x_m`, `y_m`) or (`lat`, `lon`) coordinate columns.")
            if math.isfinite(xv) and math.isfinite(yv) and math.isfinite(zv):
                xs.append(xv)
                ys.append(yv)
                zs.append(zv)
        except (TypeError, ValueError):
            continue

    if len(xs) < 4:
        raise ValueError("CSV DEM requires at least 4 valid elevation sample rows.")

    pts_x = np.asarray(xs, dtype=np.float64)
    pts_y = np.asarray(ys, dtype=np.float64)
    pts_z = np.asarray(zs, dtype=np.float64)

    n = max(15, int(grid_n))
    gx = np.linspace(float(np.min(pts_x)), float(np.max(pts_x)), n)
    gy = np.linspace(float(np.min(pts_y)), float(np.max(pts_y)), n)
    if np.ptp(gx) < 1e-6 or np.ptp(gy) < 1e-6:
        raise ValueError("CSV DEM points must span a non-zero 2D area.")
    GX, GY = np.meshgrid(gx, gy, indexing="xy")

    z_lin = griddata((pts_x, pts_y), pts_z, (GX, GY), method="linear")
    z_near = griddata((pts_x, pts_y), pts_z, (GX, GY), method="nearest")
    z_grid = np.where(np.isnan(z_lin), z_near, z_lin)

    return TerrainDEMGrid(
        name=name,
        source_label=f"CSV DEM ({len(xs)} points)",
        x_m=gx,
        y_m=gy,
        z_grid_m=z_grid,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def parse_esri_ascii_grid(
    asc_text: str,
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
    name: str = "Uploaded ESRI ASCII DEM",
) -> TerrainDEMGrid:
    """Parse an ESRI ASCII Grid (.asc) elevation raster into a local TerrainDEMGrid."""
    lines = [ln.strip() for ln in asc_text.splitlines() if ln.strip()]
    header: dict[str, float] = {}
    data_start = 0
    for idx, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) == 2 and parts[0].lower() in {
            "ncols",
            "nrows",
            "xllcorner",
            "yllcorner",
            "xllcenter",
            "yllcenter",
            "cellsize",
            "nodata_value",
        }:
            header[parts[0].lower()] = float(parts[1])
            data_start = idx + 1
        else:
            break

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    cell = float(header.get("cellsize", 0.0))
    if ncols < 2 or nrows < 2 or cell <= 0.0:
        raise ValueError("Invalid ESRI ASCII DEM header: requires ncols>=2, nrows>=2, cellsize>0.")
    nodata = float(header.get("nodata_value", -9999.0))

    vals: list[float] = []
    for ln in lines[data_start:]:
        vals.extend(float(tok) for tok in ln.split())
    if len(vals) < ncols * nrows:
        raise ValueError(f"ESRI ASCII DEM expected {ncols * nrows} cells, found {len(vals)}.")

    arr = np.asarray(vals[: ncols * nrows], dtype=np.float64).reshape((nrows, ncols))
    valid_mask = np.isfinite(arr) & (np.abs(arr - nodata) > 1e-6)
    fill_v = float(np.median(arr[valid_mask])) if np.any(valid_mask) else 0.0
    arr = np.where(valid_mask, arr, fill_v)
    # Row 0 in ESRI ASCII is top (northmost), flipud so y_m increases south->north
    arr_sn = np.flipud(arr)

    width_m = (ncols - 1) * cell
    height_m = (nrows - 1) * cell
    # Center large UTM/geographic coordinates around site local (0, 0)
    xll = header.get("xllcorner", header.get("xllcenter", -0.5 * width_m))
    yll = header.get("yllcorner", header.get("yllcenter", -0.5 * height_m))
    if abs(xll) > 5000.0 or abs(yll) > 5000.0:
        x_v = np.linspace(-0.5 * width_m, 0.5 * width_m, ncols)
        y_v = np.linspace(-0.5 * height_m, 0.5 * height_m, nrows)
    else:
        offset_x = 0.5 * cell if "xllcorner" in header else 0.0
        offset_y = 0.5 * cell if "yllcorner" in header else 0.0
        x_v = xll + offset_x + np.arange(ncols, dtype=np.float64) * cell
        y_v = yll + offset_y + np.arange(nrows, dtype=np.float64) * cell

    return TerrainDEMGrid(
        name=name,
        source_label=f"ESRI ASCII Grid ({nrows}x{ncols}, {cell:.1f}m)",
        x_m=x_v,
        y_m=y_v,
        z_grid_m=arr_sn,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )


def fetch_open_meteo_elevation_dem(
    origin_lat: float,
    origin_lon: float,
    extent_m: float = 320.0,
    grid_n: int = 9,
    timeout_s: float = 5.0,
) -> tuple[bool, TerrainDEMGrid | None, str]:
    """Fetch real-world SRTM/Copernicus 90m/30m DEM patch from Open-Meteo Elevation API.

    Gracefully returns `(False, None, reason)` if offline, rate-limited, or timed out.
    """
    try:
        import httpx
    except ImportError:
        return False, None, "httpx package unavailable for online DEM lookup."

    gn = max(5, min(10, int(grid_n)))
    xv = np.linspace(-extent_m, extent_m, gn)
    yv = np.linspace(-extent_m, extent_m, gn)
    cos_lat = max(math.cos(math.radians(origin_lat)), 0.1)

    lats: list[str] = []
    lons: list[str] = []
    for y_val in yv:
        lat_p = origin_lat + math.degrees(float(y_val) / EARTH_RADIUS_M)
        for x_val in xv:
            lon_p = origin_lon + math.degrees(float(x_val) / (EARTH_RADIUS_M * cos_lat))
            lats.append(f"{lat_p:.6f}")
            lons.append(f"{lon_p:.6f}")

    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": ",".join(lats), "longitude": ",".join(lons)},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        elevs = payload.get("elevation")
        if not isinstance(elevs, list) or len(elevs) != gn * gn:
            return False, None, "Unexpected elevation payload returned by Open-Meteo API."
    except Exception as exc:
        return False, None, f"Online DEM lookup unavailable ({exc}). Using local preset/uploaded DEM."

    z_coarse = np.asarray(elevs, dtype=np.float64).reshape((gn, gn))
    # Upsample cleanly to 33x33 for smooth contour plots and LOS ray marching
    interp_c = RegularGridInterpolator((yv, xv), z_coarse, method="cubic", bounds_error=False, fill_value=None)
    xf = np.linspace(-extent_m, extent_m, 33)
    yf = np.linspace(-extent_m, extent_m, 33)
    XF, YF = np.meshgrid(xf, yf, indexing="xy")
    zf = interp_c(np.column_stack([YF.ravel(), XF.ravel()])).reshape((33, 33))

    dem = TerrainDEMGrid(
        name=f"Live Terrain ({origin_lat:.4f}°, {origin_lon:.4f}°)",
        source_label="Open-Meteo Copernicus/SRTM DEM API",
        x_m=xf,
        y_m=yf,
        z_grid_m=zf,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    return True, dem, f"Loaded real terrain patch around ({origin_lat:.4f}°, {origin_lon:.4f}°) [relief {dem.stats()['relief_m']:.1f} m]."


def segment_crosses_terrain_dem(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    dem: TerrainDEMGrid,
    n_samples: int = 48,
    clearance_m: float = 0.5,
) -> bool:
    """Return True if 3D sightline p0->p1 intersects the terrain DEM surface."""
    t_s = np.linspace(0.03, 0.98, max(8, int(n_samples)))
    px = float(p0[0]) + t_s * (float(p1[0]) - float(p0[0]))
    py = float(p0[1]) + t_s * (float(p1[1]) - float(p0[1]))
    pz = float(p0[2]) + t_s * (float(p1[2]) - float(p0[2]))
    z_ground = np.asarray(dem.elevation_at(px, py), dtype=np.float64)
    return bool(np.any(pz < (z_ground + float(clearance_m))))


def compute_terrain_mask_horizon(
    turret_x: float,
    turret_y: float,
    turret_agl_m: float,
    dem: TerrainDEMGrid,
    max_range_m: float = 280.0,
    n_az: int = 72,
    n_steps: int = 40,
) -> list[dict[str, float]]:
    """Compute minimum clear elevation angle (deg) around 360 deg from a sited turret."""
    z_turret = float(dem.elevation_at(turret_x, turret_y)) + float(turret_agl_m)
    rs = np.linspace(8.0, float(max_range_m), max(10, int(n_steps)))
    rows: list[dict[str, float]] = []
    for idx in range(int(n_az)):
        az = idx * (360.0 / float(n_az))
        az_r = math.radians(az)
        sx = turret_x + rs * math.cos(az_r)
        sy = turret_y + rs * math.sin(az_r)
        sz = np.asarray(dem.elevation_at(sx, sy), dtype=np.float64)
        angles_deg = np.degrees(np.arctan2(sz - z_turret, rs))
        max_mask_deg = float(np.max(angles_deg))
        blocking_idx = int(np.argmax(angles_deg))
        rows.append(
            {
                "az_deg": round(az, 1),
                "mask_elev_deg": round(max_mask_deg, 2),
                "crest_dist_m": round(float(rs[blocking_idx]), 1),
            }
        )
    return rows


def compute_terrain_adjusted_fallout_sdz(
    turret_x: float,
    turret_y: float,
    turret_agl_m: float,
    firing_az_deg: float,
    firing_elev_deg: float,
    dem: TerrainDEMGrid,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_barrel_length_in: float = 24.0,
) -> dict[str, Any]:
    """Integrate center pellet arc to both flat datum and real terrain DEM intersection."""
    presets = get_shotgun_presets()
    ammo = presets.get(ammunition_preset, presets["12G_NO4_BUCK_C_UAS"])

    d_m = float(ammo["pellet_diameter_mm"]) / 1000.0
    m_kg = float(ammo["pellet_mass_g"]) / 1000.0
    area_m2 = math.pi * 0.25 * (d_m ** 2)
    v0 = float(ammo["v0_mps"]) + 2.2 * (float(default_barrel_length_in) - 28.0)
    eff_cd = float(ammo.get("effective_cd", 0.47))
    is_flechette = eff_cd < 0.25

    el_rad = math.radians(firing_elev_deg)
    az_rad = math.radians(firing_az_deg)
    ux = math.cos(el_rad) * math.cos(az_rad)
    uy = math.cos(el_rad) * math.sin(az_rad)
    uz = math.sin(el_rad)

    z0_ground = float(dem.elevation_at(turret_x, turret_y))
    px, py, pz = float(turret_x), float(turret_y), z0_ground + float(turret_agl_m)
    vx, vy, vz = v0 * ux, v0 * uy, v0 * uz

    dt = 0.025
    flat_range_m: float | None = None
    terr_range_m: float | None = None
    terr_impact_v: float | None = None
    terr_impact_z: float | None = None
    t_now = 0.0

    for _ in range(1600):
        vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
        z_agl_est = max(pz - z0_ground, 0.0)
        temp_ratio = max(1.0 - 2.2558e-5 * z_agl_est, 0.1)
        rho_air = 1.225 * (temp_ratio ** 4.2559)
        a_snd = 340.29 * math.sqrt(temp_ratio)

        if is_flechette:
            cd_v = eff_cd
        else:
            re = max(vmag * d_m / 1.48e-5, 1.0)
            mach = vmag / max(a_snd, 1.0)
            cd_v = float(_sphere_drag_coeff(np.array([re]), np.array([mach]))[0])

        k_drag = -(0.5 * rho_air * cd_v * area_m2 / max(m_kg, 1e-6)) * vmag
        ax = k_drag * vx
        ay = k_drag * vy
        az_a = k_drag * vz - 9.80665

        px_n = px + vx * dt + 0.5 * ax * dt * dt
        py_n = py + vy * dt + 0.5 * ay * dt * dt
        pz_n = pz + vz * dt + 0.5 * az_a * dt * dt
        vx_n = vx + ax * dt
        vy_n = vy + ay * dt
        vz_n = vz + az_a * dt
        t_now += dt

        if t_now > 0.08:
            if flat_range_m is None and pz_n <= z0_ground:
                frac = (pz - z0_ground) / max(pz - pz_n, 1e-9)
                ix = px + frac * (px_n - px)
                iy = py + frac * (py_n - py)
                flat_range_m = math.hypot(ix - turret_x, iy - turret_y)

            g_here = float(dem.elevation_at(px_n, py_n))
            g_prev = float(dem.elevation_at(px, py))
            if terr_range_m is None and pz_n <= g_here:
                prev_diff = pz - g_prev
                next_diff = pz_n - g_here
                frac = prev_diff / max(prev_diff - next_diff, 1e-9)
                frac = min(max(frac, 0.0), 1.0)
                ix = px + frac * (px_n - px)
                iy = py + frac * (py_n - py)
                terr_range_m = math.hypot(ix - turret_x, iy - turret_y)
                terr_impact_v = math.sqrt(vx_n * vx_n + vy_n * vy_n + vz_n * vz_n)
                terr_impact_z = float(dem.elevation_at(ix, iy))

        px, py, pz = px_n, py_n, pz_n
        vx, vy, vz = vx_n, vy_n, vz_n

        if flat_range_m is not None and terr_range_m is not None:
            break

    if flat_range_m is None:
        flat_range_m = math.hypot(px - turret_x, py - turret_y)
    if terr_range_m is None:
        terr_range_m = math.hypot(px - turret_x, py - turret_y)
        terr_impact_v = math.sqrt(vx * vx + vy * vy + vz * vz)
        terr_impact_z = float(dem.elevation_at(px, py))

    impact_ke_j = 0.5 * m_kg * ((terr_impact_v or 0.0) ** 2)
    return {
        "firing_az_deg": round(float(firing_az_deg), 1),
        "firing_elev_deg": round(float(firing_elev_deg), 1),
        "flat_fallout_range_m": round(float(flat_range_m), 1),
        "terrain_fallout_range_m": round(float(terr_range_m), 1),
        "range_delta_m": round(float(terr_range_m - flat_range_m), 1),
        "impact_ground_elev_m": round(float(terr_impact_z or 0.0), 2),
        "impact_velocity_mps": round(float(terr_impact_v or 0.0), 1),
        "impact_ke_j": round(float(impact_ke_j), 2),
    }


def summarize_site_terrain_fallout(
    turrets: list[dict[str, Any]],
    dem: TerrainDEMGrid,
    firing_elev_deg: float = 30.0,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
) -> dict[str, Any]:
    """Evaluate 12-azimuth terrain-adjusted Fallout SDZ envelope from centroid turret siting."""
    if not turrets:
        cx, cy, cz = 0.0, 0.0, 3.5
    else:
        cx = float(np.mean([float(t["x_m"]) for t in turrets]))
        cy = float(np.mean([float(t["y_m"]) for t in turrets]))
        cz = float(np.mean([float(t.get("z_m", 3.5)) for t in turrets]))

    rows = [
        compute_terrain_adjusted_fallout_sdz(
            turret_x=cx,
            turret_y=cy,
            turret_agl_m=cz,
            firing_az_deg=az,
            firing_elev_deg=firing_elev_deg,
            dem=dem,
            ammunition_preset=ammunition_preset,
        )
        for az in range(0, 360, 30)
    ]
    return {
        "flat_fallout_m": rows[0]["flat_fallout_range_m"],
        "terrain_fallout_min_m": round(min(r["terrain_fallout_range_m"] for r in rows), 1),
        "terrain_fallout_max_m": round(max(r["terrain_fallout_range_m"] for r in rows), 1),
        "azimuth_samples": rows,
    }
