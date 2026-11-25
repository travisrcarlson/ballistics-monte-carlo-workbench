"""Small Arms Ballistics, Shotgun Patterning & Anti-Drone Dispersion Package — arrays submodule."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from ammo_mc.aero_sphere import (
    didion_crosswind_drift_m,
    exact_quadratic_drag_tof_and_velocity,
    sphere_drag_compressible,
    sphere_drag_compressible_array,
    sphere_drag_mach_only,
    sphere_drag_mach_only_array,
    sutherland_air_viscosity,
)
from .catalogs import (
    DRONE_ASPECT_PRESETS,
    DRONE_TARGET_SPECS,
    PELLET_MATERIAL_DENSITIES_G_CM3,
    SHOTGUN_PRESETS,
    get_choke_definitions,
    get_drone_target_specs,
    get_shotgun_presets,
)
from .geometry import compute_drone_projected_geometry, evaluate_drone_impact

def generate_multishotgun_array_layout(
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    custom_specs: list[dict[str, Any]] | None = None,
    default_ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
) -> list[dict[str, Any]]:
    """Generate the 2D mounting coordinates (y, z) and attributes for multiple shotgun barrels.

    Supports single, square (4-barrel), quincunx (5-barrel center + corners), circular, concentric,
    grid, horizontal/vertical fans, honeycomb, chevron, and custom coordinates, plus homogeneous
    or heterogeneous Core (inner) + Perimeter (outer) ammunition/choke mixes.

    Coordinates on turret face (X=0):
      y_m: horizontal crossrange position (m), positive = right
      z_m: vertical height position (m), positive = up
    """
    barrels: list[dict[str, Any]] = []

    if layout_type == "single":
        barrels.append({
            "barrel_id": "B1",
            "label": "Barrel 1 (Single Bore)",
            "y_m": 0.0,
            "z_m": 0.0,
            "choke": default_choke,
            "barrel_length_in": float(default_barrel_length_in),
            "mount_group": "inner_ring",
        })

    elif layout_type == "square":
        # 4 barrels in a square pattern with spacing_y_m (0.20 - 0.40 m)
        sy = max(0.05, float(spacing_y_m))
        sz = max(0.05, float(spacing_z_m if spacing_z_m != 0.12 else sy))
        corners = [
            ("Top-Left", -0.5 * sy, +0.5 * sz),
            ("Top-Right", +0.5 * sy, +0.5 * sz),
            ("Bottom-Left", -0.5 * sy, -0.5 * sz),
            ("Bottom-Right", +0.5 * sy, -0.5 * sz),
        ]
        for idx, (c_lbl, y_i, z_i) in enumerate(corners, start=1):
            barrels.append({
                "barrel_id": f"B{idx}",
                "label": f"Barrel {idx} (Square {c_lbl})",
                "y_m": round(y_i, 4),
                "z_m": round(z_i, 4),
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "quincunx":
        # 5 barrels: 1 center + 4 corners at exact radial distance mount_radius_m (0.20 - 0.40 m)
        r_q = max(0.05, float(mount_radius_m))
        d_q = r_q / math.sqrt(2.0)
        barrels.append({
            "barrel_id": "B1",
            "label": "Barrel 1 (Center Quincunx)",
            "y_m": 0.0,
            "z_m": 0.0,
            "choke": default_choke,
            "barrel_length_in": float(default_barrel_length_in),
            "mount_group": "inner_ring",
        })
        corners_q = [
            ("Top-Left", -d_q, +d_q),
            ("Top-Right", +d_q, +d_q),
            ("Bottom-Left", -d_q, -d_q),
            ("Bottom-Right", +d_q, -d_q),
        ]
        for idx, (c_lbl, y_i, z_i) in enumerate(corners_q, start=2):
            barrels.append({
                "barrel_id": f"B{idx}",
                "label": f"Barrel {idx} (Quincunx {c_lbl})",
                "y_m": round(y_i, 4),
                "z_m": round(z_i, 4),
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "circular":
        # N barrels evenly spaced around a circle of radius mount_radius_m
        for i in range(max(1, n_barrels)):
            angle = (2.0 * math.pi * i) / n_barrels
            # Start at top (90 deg)
            phi = angle + math.pi / 2.0
            y_i = round(mount_radius_m * math.cos(phi), 4)
            z_i = round(mount_radius_m * math.sin(phi), 4)
            deg = int(round(math.degrees(phi))) % 360
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Ring {deg}°)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "concentric":
        # Dual-ring: center core + outer ring
        if n_barrels <= 1:
            inner_count = 1
            outer_count = 0
        elif n_barrels <= 6:
            inner_count = 1
            outer_count = n_barrels - inner_count
        elif n_barrels <= 9:
            inner_count = 3
            outer_count = n_barrels - inner_count
        else:
            inner_count = max(3, n_barrels // 3)
            outer_count = max(1, n_barrels - inner_count)

        # Inner ring / center core
        r_inner = mount_radius_m * 0.45
        for i in range(inner_count):
            if inner_count == 1:
                y_i, z_i = 0.0, 0.0
                lbl = "Barrel 1 (Center Core)"
            else:
                angle = (2.0 * math.pi * i) / inner_count + math.pi / 2.0
                y_i = round(r_inner * math.cos(angle), 4)
                z_i = round(r_inner * math.sin(angle), 4)
                lbl = f"Barrel {i+1} (Inner Ring)"
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": lbl,
                "y_m": y_i,
                "z_m": z_i,
                "choke": "Full" if default_choke != "Extra Full" else default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "inner_ring",
            })

        # Outer ring
        for i in range(outer_count):
            angle = (2.0 * math.pi * i) / outer_count + math.pi / 2.0
            y_i = round(mount_radius_m * math.cos(angle), 4)
            z_i = round(mount_radius_m * math.sin(angle), 4)
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": f"Barrel {len(barrels)+1} (Outer Ring)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "grid":
        # Matrix grid: n_rows x n_cols
        b_idx = 1
        for r in range(n_rows):
            z_i = round((r - (n_rows - 1) / 2.0) * spacing_z_m, 4)
            for c in range(n_cols):
                y_i = round((c - (n_cols - 1) / 2.0) * spacing_y_m, 4)
                barrels.append({
                    "barrel_id": f"B{b_idx}",
                    "label": f"Barrel {b_idx} (Row {r+1}, Col {c+1})",
                    "y_m": y_i,
                    "z_m": z_i,
                    "choke": default_choke,
                    "barrel_length_in": float(default_barrel_length_in),
                    "mount_group": "grid_matrix",
                })
                b_idx += 1

    elif layout_type == "fan_horizontal":
        # Linear horizontal row of N barrels
        for i in range(max(1, n_barrels)):
            y_i = round((i - (n_barrels - 1) / 2.0) * spacing_y_m, 4)
            z_i = 0.0
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Pos {i+1} H-Fan)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "fan_horizontal",
            })

    elif layout_type == "fan_vertical":
        # Linear vertical stack of N barrels
        for i in range(max(1, n_barrels)):
            y_i = 0.0
            z_i = round((i - (n_barrels - 1) / 2.0) * spacing_z_m, 4)
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Pos {i+1} V-Stack)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "fan_vertical",
            })

    elif layout_type == "honeycomb":
        # Honeycomb: 1 center + perimeter hex at mount_radius_m
        barrels.append({
            "barrel_id": "B1",
            "label": "Barrel 1 (Center Core)",
            "y_m": 0.0,
            "z_m": 0.0,
            "choke": "Full",
            "barrel_length_in": float(default_barrel_length_in),
            "mount_group": "honeycomb_center",
        })
        hex_count = max(0, int(n_barrels) - 1)
        for i in range(hex_count):
            angle = (2.0 * math.pi * i) / hex_count
            y_i = round(mount_radius_m * math.cos(angle), 4)
            z_i = round(mount_radius_m * math.sin(angle), 4)
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": f"Barrel {len(barrels)+1} (Hex Petal {i+1})",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "honeycomb_outer",
            })

    elif layout_type == "v_shape":
        # V-shape Chevron: apex in center, wings angled up/back
        for i in range(max(1, n_barrels)):
            t = (i - (n_barrels - 1) / 2.0)
            y_i = round(t * spacing_y_m, 4)
            z_i = round(abs(t) * spacing_z_m, 4)
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Chevron V)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "v_chevron",
            })

    elif layout_type == "custom":
        if custom_specs:
            for idx, spec in enumerate(custom_specs):
                b_custom: dict[str, Any] = {
                    "barrel_id": spec.get("barrel_id", f"B{idx+1}"),
                    "label": spec.get("label", f"Custom Barrel {idx+1}"),
                    "y_m": float(spec.get("y_m", 0.0)),
                    "z_m": float(spec.get("z_m", 0.0)),
                    "choke": str(spec.get("choke", default_choke)),
                    "barrel_length_in": float(spec.get("barrel_length_in", default_barrel_length_in)),
                    "mount_group": "custom",
                }
                if "ammunition_preset" in spec:
                    b_custom["ammunition_preset"] = str(spec["ammunition_preset"])
                barrels.append(b_custom)
        else:
            # Default 4-quad mount
            barrels = [
                {"barrel_id": "B1", "label": "B1 (Top-Left)", "y_m": -0.12, "z_m": 0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B2", "label": "B2 (Top-Right)", "y_m": 0.12, "z_m": 0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B3", "label": "B3 (Bottom-Left)", "y_m": -0.12, "z_m": -0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B4", "label": "B4 (Bottom-Right)", "y_m": 0.12, "z_m": -0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
            ]

    # Assign homogeneous or heterogeneous Core + Perimeter ammunition & choke
    max_r = max((math.hypot(float(b["y_m"]), float(b["z_m"])) for b in barrels), default=0.0)
    eff_outer_ammo = outer_ammunition_preset or default_ammunition_preset
    eff_inner_ammo = inner_ammunition_preset or default_ammunition_preset
    any_center = any(
        b.get("mount_group") in ("inner_ring", "honeycomb_center")
        or (max_r > 0.04 and math.hypot(float(b["y_m"]), float(b["z_m"])) <= max_r * 0.55)
        for b in barrels
    )
    for idx, b in enumerate(barrels):
        r_b = math.hypot(float(b["y_m"]), float(b["z_m"]))
        if any_center:
            is_inner = b.get("mount_group") in ("inner_ring", "honeycomb_center") or (
                max_r > 0.04 and r_b <= max_r * 0.55 and b.get("mount_group") != "outer_ring"
            )
        else:
            # All barrels sit on a single perimeter (e.g. hollow circular ring): alternate symmetrically if mixed
            is_inner = (idx % 2 == 0) if inner_ammunition_preset is not None else False
        if "ammunition_preset" not in b:
            b["ammunition_preset"] = eff_inner_ammo if is_inner else eff_outer_ammo
        if is_inner and inner_choke is not None and b.get("mount_group") != "custom":
            b["choke"] = inner_choke
        p_spec = SHOTGUN_PRESETS.get(str(b.get("ammunition_preset", "")), {})
        if p_spec.get("choke_restriction") == "Cylinder" and b.get("choke") != "Cylinder":
            b["choke"] = "Cylinder"
            b["choke_enforced_cylinder"] = True
        else:
            b["choke_enforced_cylinder"] = bool(p_spec.get("choke_restriction") == "Cylinder")

    return barrels


def compute_shotgun_temperature_effects(
    ambient_temp_c: float = 15.0,
    pellet_material: str = "Hardened Lead",
    propellant_type: str = "double_base",
) -> dict[str, Any]:
    """Compute Annex A internal + external temperature corrections on 12-gauge shot cloud ballistics."""
    t_c = float(ambient_temp_c)
    t_k = max(210.0, t_c + 273.15)
    rho_air = 1.225 * (288.15 / t_k)
    a_sound = math.sqrt(1.4 * 287.05 * t_k)
    mu_air = sutherland_air_viscosity(t_k)

    k_prop = 0.55 if str(propellant_type).lower().strip() == "single_base" else 0.32
    mv_prop_delta = k_prop * (t_c - 15.0)
    # Stiff polyethylene wad blow-by below 5 C reduces muzzle velocity slightly
    wad_penalty_mps = -0.22 * max(0.0, 5.0 - t_c)
    mv_delta_mps = mv_prop_delta + wad_penalty_mps

    mat_low = str(pellet_material).lower()
    if "tungsten" in mat_low or "steel" in mat_low:
        deform_sens = 0.15
    elif "copper" in mat_low:
        deform_sens = 0.35
    else:
        deform_sens = 1.00

    if t_c < 15.0:
        temp_spread_mult = max(0.88, 1.0 - 0.0028 * (15.0 - t_c))
        cd_upper_mult = 1.12 + 0.0015 * (15.0 - t_c)
        regime = "cold" if t_c < 5.0 else "cool"
    else:
        excess_15 = t_c - 15.0
        excess_30 = max(0.0, t_c - 30.0)
        temp_spread_mult = 1.0 + 0.0020 * excess_15 + 0.0055 * excess_30 * deform_sens
        cd_upper_mult = 1.12 + 0.0040 * max(0.0, t_c - 25.0) * deform_sens
        regime = "hot" if t_c > 30.0 else "standard"

    v0_sd_mps = 4.2 + 0.08 * max(0.0, t_c - 30.0) + 0.10 * max(0.0, 5.0 - t_c)
    return {
        "ambient_temp_c": round(t_c, 1),
        "ambient_temp_k": round(t_k, 2),
        "regime": regime,
        "air_density_kg_m3": round(rho_air, 4),
        "speed_of_sound_mps": round(a_sound, 2),
        "air_viscosity_pa_s": mu_air,
        "mv_delta_mps": round(mv_delta_mps, 2),
        "wad_blowby_delta_mps": round(wad_penalty_mps, 2),
        "temp_spread_mult": round(temp_spread_mult, 4),
        "cd_upper_mult": round(cd_upper_mult, 4),
        "v0_sd_mps": round(v0_sd_mps, 2),
    }


def compute_annex_c_analytical_pk(
    distance_m: float,
    n_pellets: int,
    vulnerable_area_m2: float,
    theta_p_rad: float,
    theta_aim_rad: float,
    pk_given_hit: float,
    n_shots: int = 1,
) -> dict[str, Any]:
    """Compute Annex C Poisson-Gaussian analytical single-shot Pk,1 and multi-shot Pk,n."""
    d = max(1.0, float(distance_m))
    n_pel = max(1, int(n_pellets))
    a_v = max(1e-6, float(vulnerable_area_m2))
    sigma_p_m = max(1e-5, float(theta_p_rad) * d)
    sigma_aim_m = max(0.0, float(theta_aim_rad) * d)
    sigma_total_sq = (sigma_p_m ** 2) + (sigma_aim_m ** 2)
    lam = (n_pel * a_v) / max(2.0 * math.pi * sigma_total_sq, 1e-12)
    p_cond = max(0.0, min(1.0, float(pk_given_hit)))
    pk_1 = 1.0 - math.exp(-lam * p_cond)
    n_burst = max(1, int(n_shots))
    pk_n = 1.0 - ((1.0 - pk_1) ** n_burst)
    return {
        "distance_m": round(d, 2),
        "n_pellets": n_pel,
        "vulnerable_area_m2": round(a_v, 5),
        "sigma_p_m": round(sigma_p_m, 4),
        "sigma_aim_m": round(sigma_aim_m, 4),
        "sigma_total_m": round(math.sqrt(sigma_total_sq), 4),
        "expected_hits_lambda": round(lam, 3),
        "pk_given_hit": round(p_cond, 3),
        "n_shots": n_burst,
        "pk_1_pct": round(pk_1 * 100.0, 1),
        "pk_n_pct": round(pk_n * 100.0, 1),
    }


def compute_balanced_pair_fire_times(
    barrels: list[dict[str, Any]],
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
) -> list[float]:
    """Compute per-barrel firing timestamps (seconds) supporting simultaneous, sequential, or symmetric pair ripple."""
    n = len(barrels)
    if n == 0 or firing_mode == "simultaneous_volley":
        return [0.0] * n

    dt_s = max(float(ripple_delay_ms), 0.0) / 1000.0
    if firing_mode == "sequential_ripple":
        return [round(i * dt_s, 6) for i in range(n)]

    # "balanced_pair_ripple": pair opposite barrels so net yaw/pitch gimbal moment cancels
    fire_times = [0.0] * n
    unassigned = list(range(n))
    step_idx = 0

    # Fire central core barrels (r < 0.03 m) at step 0 since they exert near-zero moment arm
    center_indices = [i for i in unassigned if math.hypot(float(barrels[i]["y_m"]), float(barrels[i]["z_m"])) < 0.03]
    if center_indices:
        for ci in center_indices:
            fire_times[ci] = 0.0
            unassigned.remove(ci)
        step_idx = 1

    # Greedily match each remaining barrel with its most opposite unassigned partner
    while unassigned:
        i = unassigned.pop(0)
        fire_times[i] = round(step_idx * dt_s, 6)
        if unassigned:
            yi, zi = float(barrels[i]["y_m"]), float(barrels[i]["z_m"])
            best_j = min(
                unassigned,
                key=lambda j: math.hypot(yi + float(barrels[j]["y_m"]), zi + float(barrels[j]["z_m"])),
            )
            unassigned.remove(best_j)
            fire_times[best_j] = round(step_idx * dt_s, 6)
        step_idx += 1

    return fire_times


def configure_array_pointing_vectors(
    barrels: list[dict[str, Any]],
    pointing_mode: str = "convergent",
    intercept_range_m: float = 35.0,
    splay_angle_mrad: float = 0.0,
    tiling_target_radius_m: float = 0.50,
    custom_angles: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Compute the pointing yaw and pitch aiming vectors for each barrel in the multi-shotgun array.

    Angles in milliradians (mrad):
      aim_yaw_mrad: horizontal deflection (+ = right)
      aim_pitch_mrad: vertical deflection (+ = up)
    """
    R = max(intercept_range_m, 1.0)
    configured: list[dict[str, Any]] = []

    for idx, b in enumerate(barrels):
        b_copy = dict(b)
        y0 = float(b["y_m"])
        z0 = float(b["z_m"])

        if pointing_mode == "convergent":
            # All barrels angle inward so nominal centerlines intersect at (0, 0) at range R
            yaw_mrad = -(y0 / R) * 1000.0
            pitch_mrad = -(z0 / R) * 1000.0

        elif pointing_mode == "parallel":
            # Parallel bore axes: no angular convergence
            yaw_mrad = 0.0
            pitch_mrad = 0.0

        elif pointing_mode == "divergent_curtain":
            # Barrels angle outward from cluster centroid by splay_angle_mrad to maximize intercept net
            r_dist = math.hypot(y0, z0)
            if r_dist > 1e-4:
                u_y = y0 / r_dist
                u_z = z0 / r_dist
            else:
                u_y, u_z = 0.0, 0.0
            yaw_mrad = -(y0 / R) * 1000.0 + (splay_angle_mrad * u_y)
            pitch_mrad = -(z0 / R) * 1000.0 + (splay_angle_mrad * u_z)

        elif pointing_mode == "staggered_tiling":
            # Tiles pattern centers across a circle of radius tiling_target_radius_m
            n_tot = len(barrels)
            r_dist = math.hypot(y0, z0)
            if n_tot == 1 or r_dist < 0.03:
                y_tgt, z_tgt = 0.0, 0.0
            else:
                angle = math.atan2(z0, y0)
                y_tgt = tiling_target_radius_m * math.cos(angle)
                z_tgt = tiling_target_radius_m * math.sin(angle)
            yaw_mrad = ((y_tgt - y0) / R) * 1000.0
            pitch_mrad = ((z_tgt - z0) / R) * 1000.0

        elif pointing_mode == "custom":
            if custom_angles and idx < len(custom_angles):
                yaw_mrad, pitch_mrad = custom_angles[idx]
            else:
                yaw_mrad, pitch_mrad = 0.0, 0.0
        else:
            yaw_mrad, pitch_mrad = -(y0 / R) * 1000.0, -(z0 / R) * 1000.0

        b_copy["aim_yaw_mrad"] = round(yaw_mrad, 3)
        b_copy["aim_pitch_mrad"] = round(pitch_mrad, 3)
        b_copy["focal_range_m"] = R
        configured.append(b_copy)

    return configured


def sample_shotcloud_dispersion(
    rng: Any,
    n_pellets: int,
    sigma_radial: float,
    pattern_distribution: str = "gaussian",
    flyer_fraction: float = 0.12,
    flyer_scale: float = 2.15,
    choke_ring_bias: float = 0.35,
    material: str = "Hardened Lead",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample transverse (dy, dz) pellet offsets and coupled drag multipliers (core + flyers).

    Supports:
      - 'gaussian': baseline isotropic Rayleigh-Gaussian pattern (cd_extra = 1.0)
      - 'core_flyer_mixture': central core plus deformed-pellet flyer tail with elevated Cd
      - 'choke_annular_mixture': annular ring-skewed pattern plus flyer drag tail
    """
    n_p = max(1, int(n_pellets))
    mode = str(pattern_distribution or "gaussian").lower().strip()
    if mode == "gaussian":
        dy = rng.normal(0.0, sigma_radial, size=n_p)
        dz = rng.normal(0.0, sigma_radial, size=n_p)
        return dy, dz, np.ones(n_p, dtype=np.float64)

    mat_low = str(material).lower()
    mat_scale = 0.55 if "tungsten" in mat_low else (0.80 if "steel" in mat_low or "copper" in mat_low else 1.15)
    eff_fly = max(0.02, min(0.40, float(flyer_fraction) * mat_scale))
    fly_mask = rng.random(n_p) < eff_fly

    if mode == "choke_annular_mixture":
        angles = rng.uniform(0.0, 2.0 * math.pi, size=n_p)
        radii = np.abs(rng.normal(float(choke_ring_bias) * sigma_radial, sigma_radial * 0.88, size=n_p))
        dy = radii * np.cos(angles)
        dz = radii * np.sin(angles)
    else:
        dy = rng.normal(0.0, sigma_radial * 0.90, size=n_p)
        dz = rng.normal(0.0, sigma_radial * 0.90, size=n_p)

    cd_extra = np.ones(n_p, dtype=np.float64)
    n_fly = int(np.sum(fly_mask))
    if n_fly > 0:
        dy[fly_mask] = rng.normal(0.0, sigma_radial * float(flyer_scale), size=n_fly)
        dz[fly_mask] = rng.normal(0.0, sigma_radial * float(flyer_scale), size=n_fly)
        cd_extra[fly_mask] = 1.0 + rng.uniform(0.14, 0.28, size=n_fly)
    return dy, dz, cd_extra


def simulate_anti_drone_multishotgun_mc(
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    pellets_per_shot: int | None = None,
    pellet_diameter_mm: float | None = None,
    pellet_mass_g: float | None = None,
    v0_mps: float | None = None,
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    target_distance_m: float = 35.0,
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    drone_heading_deg: float = 90.0,
    aim_error_mrad: float = 1.5,
    crosswind_mps: float = 0.0,
    n_volleys: int = 25,
    seed: int = 42,
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
    barrel_pitch_offsets_mrad: list[float] | None = None,
    barrel_yaw_offsets_mrad: list[float] | None = None,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    custom_drone_spec: dict[str, Any] | None = None,
    custom_shell_spec: dict[str, Any] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    headwind_mps: float = 0.0,
    ambient_temp_c: float = 15.0,
    propellant_type: str = "double_base",
    drag_model: str = "compressible_re_m",
    default_barrel_length_cm: float | None = None,
    n_shots_per_barrel: int = 1,
    use_rotor_solidity: bool = False,
    use_v50_ballistic_limit: bool = False,
    enable_ray_shielding: bool = False,
    pattern_distribution: str = "gaussian",
    flyer_fraction: float = 0.12,
) -> dict[str, Any]:
    """Execute Monte Carlo simulation of an anti-drone defense volley using a multi-shotgun array."""
    rng = np.random.default_rng(seed)
    cp = dict(calibration_profile or {})
    cal_mv_scale = max(0.50, min(1.50, float(cp.get("mv_scale", 1.0) or 1.0)))
    cal_spread_factor = max(0.35, float(cp.get("spread_factor", 1.0) or 1.0))
    cal_spread_exp = max(0.80, min(1.80, float(cp.get("spread_exponent", 1.15) or 1.15)))
    cal_cd_scale = max(0.40, min(2.20, float(cp.get("pellet_cd_scale", 1.0) or 1.0)))
    cal_avuln_scale = max(0.25, min(2.50, float(cp.get("drone_avuln_scale", 1.0) or 1.0)))
    cal_avuln_lin = math.sqrt(cal_avuln_scale)
    cal_ecrit_scale = max(0.30, min(2.60, float(cp.get("drone_ecrit_scale", 1.0) or 1.0)))
    cal_aim_extra_mrad = math.sqrt(
        (float(cp.get("aim_bias_mrad", 0.0) or 0.0) ** 2)
        + (float(cp.get("barrel_zero_scatter_mrad", 0.0) or 0.0) ** 2)
        + (float(cp.get("otm_jitter_mrad", 0.0) or 0.0) ** 2)
        + (max(0.0, float(cp.get("tracker_jitter_mrad")) - 1.2) ** 2 if cp.get("tracker_jitter_mrad") is not None else 0.0)
    )
    eff_barrel_len_in = float(default_barrel_length_cm) / 2.54 if default_barrel_length_cm is not None else float(default_barrel_length_in)

    # 1. Resolve Baseline Nominal Ammunition Parameters
    presets = get_shotgun_presets()

    def _resolve_barrel_ammo(preset_key: str) -> dict[str, Any]:
        use_custom_shell = (
            custom_shell_spec is not None
            and (preset_key == ammunition_preset or preset_key == "CUSTOM_SHELL" or preset_key not in presets)
        )
        allow_pel_override = (
            pellets_per_shot is not None
            and int(pellets_per_shot) > 0
            and (preset_key == ammunition_preset or inner_ammunition_preset is None)
        )
        if use_custom_shell and custom_shell_spec is not None:
            return {
                "pellets": int(pellets_per_shot) if allow_pel_override else int(custom_shell_spec.get("pellets_per_shot", 27)),
                "d_mm": float(pellet_diameter_mm or custom_shell_spec.get("pellet_diameter_mm", 6.10)),
                "m_g": float(pellet_mass_g or custom_shell_spec.get("pellet_mass_g", 1.34)),
                "v0": float(v0_mps or custom_shell_spec.get("v0_mps", 415.0)),
                "effective_cd": float(custom_shell_spec.get("effective_cd", 0.47)),
                "material": str(custom_shell_spec.get("pellet_material", custom_shell_spec.get("material", "Hardened Lead"))),
                "unit_price_eur": float(custom_shell_spec.get("unit_price_eur", 2.00)),
                "choke_restriction": custom_shell_spec.get("choke_restriction"),
                "custom_pattern_40yd_30in_pct": custom_shell_spec.get("custom_pattern_40yd_30in_pct"),
            }
        if preset_key in presets:
            a = presets[preset_key]
            return {
                "pellets": int(pellets_per_shot) if allow_pel_override else int(a["pellets_per_shot"]),
                "d_mm": float(pellet_diameter_mm or a["pellet_diameter_mm"]),
                "m_g": float(pellet_mass_g or a["pellet_mass_g"]),
                "v0": float(v0_mps or a["v0_mps"]),
                "effective_cd": float(a.get("effective_cd", 0.47)),
                "material": str(a.get("material", a.get("pellet_type", "Hardened Lead"))),
                "unit_price_eur": float(a.get("unit_price_eur", 1.50)),
                "choke_restriction": a.get("choke_restriction"),
                "custom_pattern_40yd_30in_pct": None,
            }
        return {
            "pellets": int(pellets_per_shot or 27),
            "d_mm": float(pellet_diameter_mm or 6.10),
            "m_g": float(pellet_mass_g or 1.34),
            "v0": float(v0_mps or 415.0),
            "effective_cd": 0.47,
            "material": "Hardened Lead",
            "unit_price_eur": 1.50,
            "choke_restriction": None,
            "custom_pattern_40yd_30in_pct": None,
        }

    nom_ammo = _resolve_barrel_ammo(ammunition_preset)
    temp_fx = compute_shotgun_temperature_effects(
        ambient_temp_c=ambient_temp_c,
        pellet_material=nom_ammo["material"],
        propellant_type=propellant_type,
    )
    eff_pellets = nom_ammo["pellets"]
    eff_d_mm = nom_ammo["d_mm"]
    eff_m_g = nom_ammo["m_g"]
    eff_v0 = (nom_ammo["v0"] + temp_fx["mv_delta_mps"]) * cal_mv_scale

    # 2. Resolve Drone Specs & 3D Projected Cross-Section Geometry
    drone_specs_cat = get_drone_target_specs()
    if custom_drone_spec is not None:
        drone_spec = dict(custom_drone_spec)
    else:
        drone_spec = dict(drone_specs_cat.get(drone_model, drone_specs_cat["commercial_quad_medium"]))
    for ek in ("lethal_energy_rotor_j", "lethal_energy_core_j", "lethal_energy_battery_j", "lethal_energy_arm_j", "lethal_energy_warhead_j"):
        if ek in drone_spec:
            drone_spec[ek] = float(drone_spec[ek]) * cal_ecrit_scale
    crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 3))

    aspect_geom = compute_drone_projected_geometry(
        drone_spec=drone_spec,
        aspect_mode=aspect_mode,
        drone_pitch_deg=drone_pitch_deg,
        drone_yaw_deg=drone_yaw_deg,
        drone_roll_deg=drone_roll_deg,
    )

    # 3. Generate and Configure Barrel Array
    barrels_raw = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        spacing_y_m=spacing_y_m,
        spacing_z_m=spacing_z_m,
        n_cols=n_cols,
        n_rows=n_rows,
        default_choke=default_choke,
        default_barrel_length_in=eff_barrel_len_in,
        custom_specs=custom_barrel_specs,
        default_ammunition_preset=ammunition_preset,
        inner_ammunition_preset=inner_ammunition_preset,
        outer_ammunition_preset=outer_ammunition_preset,
        inner_choke=inner_choke,
    )
    barrels = configure_array_pointing_vectors(
        barrels=barrels_raw,
        pointing_mode=pointing_mode,
        intercept_range_m=target_distance_m,
        splay_angle_mrad=splay_angle_mrad,
    )
    n_act_barrels = len(barrels)
    fire_times_s = compute_balanced_pair_fire_times(barrels, firing_mode=firing_mode, ripple_delay_ms=ripple_delay_ms)

    # 4. Aerodynamic Integration Helper (Compton Mach-Only vs Compressible Re+M + Head/Tailwind)
    rho_air = float(temp_fx["air_density_kg_m3"])
    mu_air = float(temp_fx["air_viscosity_pa_s"])
    a_sound = float(temp_fx["speed_of_sound_mps"])
    use_mach_only = str(drag_model).lower().strip() == "mach_only_smooth_sphere"

    def _integrate_pellets_path(
        v0_vec: np.ndarray,
        diam_m: float,
        mass_kg: float,
        range_m: float,
        base_cd_ov: float,
        cd_mult_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        area_p = math.pi * 0.25 * (diam_m ** 2)
        n_q = 8
        dx = max(float(range_m), 0.5) / n_q
        v_cur = np.maximum(np.asarray(v0_vec, dtype=np.float64), 10.0)
        tof = np.zeros_like(v_cur)
        hw = float(headwind_mps)
        is_fixed_low = base_cd_ov < 0.25
        re_scale = diam_m * rho_air / mu_air
        for _ in range(n_q):
            v_rel = np.maximum(5.0, v_cur + hw)
            if is_fixed_low:
                cd_arr = np.full_like(v_cur, base_cd_ov)
            elif use_mach_only:
                cd_arr = sphere_drag_mach_only_array(v_rel / a_sound, base_cd_ov)
            else:
                cd_arr = sphere_drag_compressible_array(v_rel * re_scale, v_rel / a_sound)
            cd_arr = cd_arr * cd_mult_vec
            dv_dx_1 = -(0.5 * rho_air * cd_arr * area_p * (v_rel ** 2)) / (max(mass_kg, 1e-6) * np.maximum(v_cur, 5.0))
            v_mid = np.maximum(5.0, v_cur + 0.5 * dx * dv_dx_1)
            v_rel_mid = np.maximum(5.0, v_mid + hw)
            if is_fixed_low:
                cd_mid = cd_arr
            elif use_mach_only:
                cd_mid = sphere_drag_mach_only_array(v_rel_mid / a_sound, base_cd_ov) * cd_mult_vec
            else:
                cd_mid = sphere_drag_compressible_array(v_rel_mid * re_scale, v_rel_mid / a_sound) * cd_mult_vec
            dv_dx_2 = -(0.5 * rho_air * cd_mid * area_p * (v_rel_mid ** 2)) / (max(mass_kg, 1e-6) * np.maximum(v_mid, 5.0))
            v_next = np.maximum(5.0, v_cur + dx * dv_dx_2)
            tof += dx / np.maximum(0.5 * (v_cur + v_next), 5.0)
            v_cur = v_next
        return tof, v_cur

    d_m_nom = eff_d_mm / 1000.0
    m_kg_nom = eff_m_g / 1000.0
    t_nom_arr, v_nom_arr = _integrate_pellets_path(
        np.array([eff_v0], dtype=np.float64),
        d_m_nom,
        m_kg_nom,
        target_distance_m,
        nom_ammo["effective_cd"],
        np.array([cal_cd_scale], dtype=np.float64),
    )
    t_flight_nominal = float(t_nom_arr[0])
    v_impact_nominal = float(v_nom_arr[0])
    nominal_gravity_drop_m = 0.5 * 9.80665 * (t_flight_nominal ** 2)

    # Moving target vector in engagement plane: heading 0 = closing (-X), 90 = crossing (+Y)
    heading_rad = math.radians(drone_heading_deg)
    v_drone_x = -drone_velocity_mps * math.cos(heading_rad)
    v_drone_y = drone_velocity_mps * math.sin(heading_rad)
    v_drone_z = 0.0

    # 5. Volley Simulation Loop
    volley_records: list[dict[str, Any]] = []
    kills_count = 0
    hits_volley_count = 0
    all_hits_counts: list[int] = []
    all_lethal_counts: list[int] = []
    all_energy_on_drone: list[float] = []
    all_impulse_on_drone: list[float] = []
    blade_pass_through_count = 0
    flyer_count_total = 0
    v50_samples: list[float] = []
    component_kills_tally: dict[str, int] = {"fuselage_core": 0, "battery_payload": 0, "warhead_pg7m": 0, "rotors_motors": 0, "structural_arms": 0}

    barrel_ammo_specs = [_resolve_barrel_ammo(str(b.get("ammunition_preset", ammunition_preset))) for b in barrels]
    for b_idx, b in enumerate(barrels):
        b_sp = barrel_ammo_specs[b_idx]
        if b_sp.get("choke_restriction") == "Cylinder" and b.get("choke") != "Cylinder":
            b["choke"] = "Cylinder"
            b["choke_enforced_cylinder"] = True

    tot_pellets_per_volley = sum(spec["pellets"] for spec in barrel_ammo_specs)
    cost_per_volley_eur = round(sum(float(spec.get("unit_price_eur", 1.50)) for spec in barrel_ammo_specs), 2)
    cost_per_engagement_eur = round(cost_per_volley_eur * max(1, int(n_shots_per_barrel)), 2)

    # Per-barrel hit accumulator
    barrel_stats_map = {
        b["barrel_id"]: {
            "barrel_id": b["barrel_id"],
            "label": b["label"],
            "y_m": b["y_m"],
            "z_m": b["z_m"],
            "choke": b["choke"],
            "choke_enforced_cylinder": bool(b.get("choke_enforced_cylinder", False)),
            "ammunition_preset": b.get("ammunition_preset", ammunition_preset),
            "unit_price_eur": round(float(barrel_ammo_specs[idx].get("unit_price_eur", 1.50)), 2),
            "fire_time_ms": round(fire_times_s[idx] * 1000.0, 2),
            "aim_yaw_mrad": b["aim_yaw_mrad"],
            "aim_pitch_mrad": b["aim_pitch_mrad"],
            "pellets_fired": 0,
            "hits_on_drone": 0,
            "lethal_hits": 0,
            "rotor_hits": 0,
            "fuselage_hits": 0,
        }
        for idx, b in enumerate(barrels)
    }

    sample_scatter: dict[str, list[Any]] = {
        "y": [],
        "z": [],
        "barrel_id": [],
        "is_hit": [],
        "is_lethal": [],
        "component": [],
        "striking_energy_j": [],
        "striking_impulse_ns": [],
        "striking_velocity_mps": [],
        "flight_time_s": [],
    }

    eff_aim_err_mrad = math.hypot(max(0.0, float(aim_error_mrad)), cal_aim_extra_mrad)
    aim_sigma_m = target_distance_m * (eff_aim_err_mrad / 1000.0) if eff_aim_err_mrad > 0 else 0.0
    choke_defs = get_choke_definitions()
    theta_chokes_list: list[float] = []
    barrel_kin_params: list[dict[str, Any]] = []
    for b_idx, b in enumerate(barrels):
        b_choke = str(b.get("choke", default_choke))
        b_len = float(b.get("barrel_length_in", eff_barrel_len_in))
        b_spec = barrel_ammo_specs[b_idx]
        b_temp_fx = compute_shotgun_temperature_effects(ambient_temp_c, b_spec["material"], propellant_type)
        v0_barrel = (float(b_spec["v0"]) + 2.2 * (b_len - 28.0) + b_temp_fx["mv_delta_mps"]) * cal_mv_scale
        wash_turb = 1.0 + 0.35 * (max(0.0, 24.0 - b_len) / 24.0) ** 1.5
        c_info = choke_defs.get(b_choke, choke_defs["Modified"])
        custom_pat = b_spec.get("custom_pattern_40yd_30in_pct")
        raw_p_bench = float(custom_pat) if custom_pat is not None else float(c_info["benchmark_40yd_30in_pct"])
        p_bench = max(0.15, min(0.99, raw_p_bench))
        sigma_ref = 0.381 / math.sqrt(-2.0 * math.log(1.0 - p_bench))
        theta_choke = sigma_ref / 36.58
        theta_chokes_list.append(theta_choke * wash_turb * b_temp_fx["temp_spread_mult"] * cal_spread_factor)
        sigma_radial = (
            (0.01852 / 4.0 + theta_choke * target_distance_m * ((target_distance_m / 36.58) ** (cal_spread_exp - 1.0)))
            * wash_turb
            * b_temp_fx["temp_spread_mult"]
            * cal_spread_factor
        )
        dyn_yaw = float(barrel_yaw_offsets_mrad[b_idx]) if barrel_yaw_offsets_mrad and b_idx < len(barrel_yaw_offsets_mrad) else 0.0
        dyn_pitch = float(barrel_pitch_offsets_mrad[b_idx]) if barrel_pitch_offsets_mrad and b_idx < len(barrel_pitch_offsets_mrad) else 0.0
        barrel_kin_params.append({
            "b_id": b["barrel_id"],
            "y0": float(b["y_m"]),
            "z0": float(b["z_m"]),
            "b_pellets": int(b_spec["pellets"]),
            "b_d_m": float(b_spec["d_mm"]) / 1000.0,
            "b_m_kg": float(b_spec["m_g"]) / 1000.0,
            "b_cd_base": float(b_spec["effective_cd"]),
            "b_material": str(b_spec["material"]),
            "density_g_cm3": float(PELLET_MATERIAL_DENSITIES_G_CM3.get(str(b_spec["material"]), 11.3)),
            "b_temp_fx": b_temp_fx,
            "t_fire": fire_times_s[b_idx],
            "v0_barrel": v0_barrel,
            "sigma_radial": sigma_radial,
            "nom_yaw_mrad": b["aim_yaw_mrad"] + dyn_yaw,
            "nom_pitch_mrad": b["aim_pitch_mrad"] + dyn_pitch,
        })

    for v_idx in range(n_volleys):
        # Turret pointing jitter for this volley
        aim_err_y = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0
        aim_err_z = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0

        v_hits = 0
        v_lethal = 0
        v_energy = 0.0
        v_impulse = 0.0
        v_fuselage_lethal = 0
        v_battery_lethal = 0
        v_warhead_lethal = 0
        v_rotors_killed: set[str] = set()

        for b_idx, bk in enumerate(barrel_kin_params):
            b_id = bk["b_id"]
            y0 = bk["y0"]
            z0 = bk["z0"]
            b_pellets = bk["b_pellets"]
            b_d_m = bk["b_d_m"]
            b_m_kg = bk["b_m_kg"]
            b_cd_base = bk["b_cd_base"]
            b_temp_fx = bk["b_temp_fx"]
            t_fire = bk["t_fire"]
            v0_barrel = bk["v0_barrel"]
            sigma_radial = bk["sigma_radial"]

            # Nominal bore center at target plane
            y_bore = y0 + target_distance_m * (bk["nom_yaw_mrad"] / 1000.0) + aim_err_y
            z_bore = z0 + target_distance_m * (bk["nom_pitch_mrad"] / 1000.0) + aim_err_z

            barrel_stats_map[b_id]["pellets_fired"] += b_pellets

            # Sample velocity & drag variances for this barrel's pellets (Annex B) and integrate trajectories
            v0_vec = np.clip(rng.normal(v0_barrel, b_temp_fx["v0_sd_mps"], size=b_pellets), 100.0, 900.0)
            cd_mult_vec = rng.uniform(0.94, b_temp_fx["cd_upper_mult"], size=b_pellets) * cal_cd_scale
            if str(pattern_distribution or "gaussian").lower().strip() == "gaussian":
                t_flight_vec, v_impact_vec = _integrate_pellets_path(
                    v0_vec, b_d_m, b_m_kg, target_distance_m, b_cd_base, cd_mult_vec
                )
                dy_vec = rng.normal(0.0, sigma_radial, size=b_pellets)
                dz_vec = rng.normal(0.0, sigma_radial, size=b_pellets)
            else:
                dy_vec, dz_vec, cd_fly_mult = sample_shotcloud_dispersion(
                    rng,
                    b_pellets,
                    sigma_radial,
                    pattern_distribution=pattern_distribution,
                    flyer_fraction=flyer_fraction,
                    material=bk["b_material"],
                )
                flyer_count_total += int(np.sum(cd_fly_mult > 1.01))
                t_flight_vec, v_impact_vec = _integrate_pellets_path(
                    v0_vec, b_d_m, b_m_kg, target_distance_m, b_cd_base, cd_mult_vec * cd_fly_mult
                )

            use_hf_impact = bool(use_rotor_solidity or use_v50_ballistic_limit or enable_ray_shielding)

            for p_idx in range(b_pellets):
                t_flight = float(t_flight_vec[p_idx])
                v_impact = float(v_impact_vec[p_idx])
                t_arrival = t_fire + t_flight

                # Radial dispersion + Didion crosswind lag drift + differential gravity drop
                wind_lag_y = didion_crosswind_drift_m(crosswind_mps, t_flight, target_distance_m, float(v0_vec[p_idx]))
                diff_grav_z = -0.5 * 9.80665 * ((t_flight ** 2) - (t_flight_nominal ** 2))
                y_impact_raw = y_bore + float(dy_vec[p_idx]) + wind_lag_y
                z_impact_raw = z_bore + float(dz_vec[p_idx]) + diff_grav_z

                # Moving target displacement relative to nominal lead at arrival time
                dt_target = (t_arrival - t_flight_nominal) * (v_impact / max(v_impact - v_drone_x, 10.0))
                tgt_offset_y = v_drone_y * dt_target
                tgt_offset_z = v_drone_z * dt_target

                y_rel = y_impact_raw - tgt_offset_y
                z_rel = z_impact_raw - tgt_offset_z

                ke_pellet = 0.5 * b_m_kg * (v_impact ** 2)
                imp_pellet = b_m_kg * v_impact

                # Component hit check (incorporate empirical A_vuln area scaling cleanly)
                if use_hf_impact:
                    hit_info = evaluate_drone_impact(
                        y_rel / cal_avuln_lin,
                        z_rel / cal_avuln_lin,
                        drone_spec,
                        ke_pellet,
                        aspect_geometry=aspect_geom,
                        pellet_velocity_mps=v_impact,
                        pellet_mass_kg=b_m_kg,
                        pellet_diameter_m=b_d_m,
                        pellet_density_g_cm3=bk["density_g_cm3"],
                        use_rotor_solidity=use_rotor_solidity,
                        use_v50_ballistic_limit=use_v50_ballistic_limit,
                        enable_ray_shielding=enable_ray_shielding,
                        rng=rng,
                    )
                    if hit_info.get("blade_pass_through"):
                        blade_pass_through_count += 1
                    if hit_info.get("v50_mps") is not None:
                        v50_samples.append(float(hit_info["v50_mps"]))
                else:
                    hit_info = evaluate_drone_impact(
                        y_rel / cal_avuln_lin,
                        z_rel / cal_avuln_lin,
                        drone_spec,
                        ke_pellet,
                        aspect_geometry=aspect_geom,
                    )

                if hit_info["is_hit"]:
                    v_hits += 1
                    barrel_stats_map[b_id]["hits_on_drone"] += 1
                    v_energy += ke_pellet
                    v_impulse += imp_pellet

                    comp = hit_info["component"]
                    if hit_info["is_lethal"]:
                        v_lethal += 1
                        barrel_stats_map[b_id]["lethal_hits"] += 1
                        if comp == "fuselage_core":
                            v_fuselage_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                            component_kills_tally["fuselage_core"] += 1
                        elif comp == "battery_payload":
                            v_battery_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                            component_kills_tally["battery_payload"] += 1
                        elif comp == "warhead_pg7m":
                            v_warhead_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                            component_kills_tally["warhead_pg7m"] += 1
                        elif comp.startswith("rotor_"):
                            v_rotors_killed.add(comp)
                            barrel_stats_map[b_id]["rotor_hits"] += 1
                            component_kills_tally["rotors_motors"] += 1
                        elif comp.startswith("motor_"):
                            r_id = comp.replace("motor_", "rotor_")
                            v_rotors_killed.add(r_id)
                            barrel_stats_map[b_id]["rotor_hits"] += 1
                            component_kills_tally["rotors_motors"] += 1
                        elif comp == "structural_arm":
                            component_kills_tally["structural_arms"] += 1

                # Record sample volley for visualization (first volley)
                if v_idx == 0:
                    sample_scatter["y"].append(round(y_rel, 4))
                    sample_scatter["z"].append(round(z_rel, 4))
                    sample_scatter["barrel_id"].append(b_id)
                    sample_scatter["is_hit"].append(hit_info["is_hit"])
                    sample_scatter["is_lethal"].append(hit_info["is_lethal"])
                    sample_scatter["component"].append(hit_info["component"])
                    sample_scatter["striking_energy_j"].append(round(ke_pellet, 2))
                    sample_scatter["striking_impulse_ns"].append(round(imp_pellet, 4))
                    sample_scatter["striking_velocity_mps"].append(round(v_impact, 1))
                    sample_scatter["flight_time_s"].append(round(t_arrival, 4))

        if v_hits >= 1:
            hits_volley_count += 1

        # Volley Kill Condition (architecture-specific rotor kill requirement)
        d_type = str(drone_spec.get("drone_type", "quadcopter"))
        req_rotors = 1 if d_type == "fixed_wing" else (2 if d_type == "quadcopter" else 3)
        is_killed = bool(
            (v_lethal >= crit_hits_req)
            or (v_fuselage_lethal >= 1)
            or (v_battery_lethal >= 1)
            or (v_warhead_lethal >= 1)
            or (len(v_rotors_killed) >= req_rotors)
        )
        if is_killed:
            kills_count += 1

        all_hits_counts.append(v_hits)
        all_lethal_counts.append(v_lethal)
        all_energy_on_drone.append(v_energy)
        all_impulse_on_drone.append(v_impulse)

        volley_records.append({
            "volley_idx": v_idx + 1,
            "hits_on_drone": v_hits,
            "lethal_hits": v_lethal,
            "energy_on_drone_j": round(v_energy, 1),
            "impulse_on_drone_ns": round(v_impulse, 3),
            "fuselage_lethal": v_fuselage_lethal,
            "battery_lethal": v_battery_lethal,
            "warhead_lethal": v_warhead_lethal,
            "rotors_killed": len(v_rotors_killed),
            "is_killed": is_killed,
        })

    # Summary Statistics
    hit_prob_pct = round((hits_volley_count / max(1, n_volleys)) * 100.0, 1)
    pk_pct = round((kills_count / max(1, n_volleys)) * 100.0, 1)
    mean_hits = round(float(np.mean(all_hits_counts)), 1)
    mean_lethal = round(float(np.mean(all_lethal_counts)), 1)
    mean_energy = round(float(np.mean(all_energy_on_drone)), 1)
    mean_impulse = round(float(np.mean(all_impulse_on_drone)), 3)

    # Cloud geometry & Annex B pattern percentages in sample volley
    if sample_scatter["y"]:
        sy_arr = np.asarray(sample_scatter["y"], dtype=np.float64)
        sz_arr = np.asarray(sample_scatter["z"], dtype=np.float64)
        st_arr = np.asarray(sample_scatter["flight_time_s"], dtype=np.float64)
        rads_com = np.hypot(sy_arr - float(np.mean(sy_arr)), sz_arr - float(np.mean(sz_arr)))
        cep_50 = round(float(np.percentile(rads_com, 50)), 3)
        r_90 = round(float(np.percentile(rads_com, 90)), 3)
        cloud_diam_90 = round(2.0 * r_90, 3)
        pattern_30in_pct = round(float(np.mean(np.hypot(sy_arr, sz_arr) <= 0.381)) * 100.0, 1)
        pattern_1m2_pct = round(float(np.mean((np.abs(sy_arr) <= 0.50) & (np.abs(sz_arr) <= 0.50))) * 100.0, 1)
        shot_string_len_m = round(float((np.percentile(st_arr, 95) - np.percentile(st_arr, 5)) * v_impact_nominal), 3)
    else:
        cep_50, r_90, cloud_diam_90 = 0.5, 1.0, 2.0
        pattern_30in_pct, pattern_1m2_pct, shot_string_len_m = 0.0, 0.0, 0.0

    # Annex C Poisson-Gaussian analytical cross-check
    nom_pellet_energy_j = round(0.5 * m_kg_nom * (v_impact_nominal ** 2), 2)
    nom_pellet_impulse_ns = round(m_kg_nom * v_impact_nominal, 4)
    e_crit_avg = 0.5 * (float(drone_spec.get("lethal_energy_rotor_j", 8.0)) + float(drone_spec.get("lethal_energy_core_j", 20.0)))
    pk_given_hit_est = 1.0 / (1.0 + math.exp(-0.22 * (nom_pellet_energy_j - 0.65 * e_crit_avg)))
    theta_p_eff = float(np.mean(theta_chokes_list)) if theta_chokes_list else 0.0085
    annex_c_analytical = compute_annex_c_analytical_pk(
        distance_m=target_distance_m,
        n_pellets=tot_pellets_per_volley,
        vulnerable_area_m2=float(aspect_geom["presented_area_m2"]) * cal_avuln_scale,
        theta_p_rad=theta_p_eff,
        theta_aim_rad=eff_aim_err_mrad / 1000.0,
        pk_given_hit=pk_given_hit_est,
        n_shots=n_shots_per_barrel,
    )

    # Calculate per-barrel summary list
    barrel_stats_list: list[dict[str, Any]] = []
    tot_hits_all = sum(b["hits_on_drone"] for b in barrel_stats_map.values())
    for b_id, b_data in barrel_stats_map.items():
        b_fired = b_data["pellets_fired"]
        b_hits = b_data["hits_on_drone"]
        hit_pct = round((b_hits / max(1, b_fired)) * 100.0, 1)
        contrib_pct = round((b_hits / max(1, tot_hits_all)) * 100.0, 1) if tot_hits_all > 0 else 0.0
        b_data["hit_rate_pct"] = hit_pct
        b_data["contribution_pct"] = contrib_pct
        barrel_stats_list.append(b_data)

    return {
        "summary": {
            "hit_probability_pct": hit_prob_pct,
            "kill_probability_pct": pk_pct,
            "mean_hits_on_drone": mean_hits,
            "mean_lethal_hits": mean_lethal,
            "mean_energy_on_drone_j": mean_energy,
            "mean_impulse_on_drone_ns": mean_impulse,
            "nominal_pellet_energy_j": nom_pellet_energy_j,
            "nominal_pellet_impulse_ns": nom_pellet_impulse_ns,
            "total_pellets_per_volley": tot_pellets_per_volley,
            "n_barrels": n_act_barrels,
            "n_volleys": n_volleys,
            "n_shots_per_barrel": int(n_shots_per_barrel),
            "cost_per_volley_eur": cost_per_volley_eur,
            "cost_per_engagement_eur": cost_per_engagement_eur,
            "expected_cost_per_kill_eur": round(cost_per_volley_eur / (pk_pct / 100.0), 2) if pk_pct > 0.0 else None,
            "pattern_30in_pct": pattern_30in_pct,
            "pattern_1m2_pct": pattern_1m2_pct,
            "shot_string_length_m": shot_string_len_m,
            "cloud_cep_50_m": cep_50,
            "cloud_r_90_m": r_90,
            "cloud_diameter_90_m": cloud_diam_90,
            "target_distance_m": target_distance_m,
            "drone_model": drone_model,
            "drone_name": drone_spec.get("name", drone_model),
            "ammunition_preset": ammunition_preset,
            "ammunition_name": (
                str(custom_shell_spec.get("name", ammunition_preset))
                if custom_shell_spec is not None
                else str(presets.get(ammunition_preset, {}).get("name", ammunition_preset))
            ),
            "drone_velocity_mps": drone_velocity_mps,
            "v0_nominal_mps": round(eff_v0, 1),
            "mean_v_impact_nominal_mps": round(v_impact_nominal, 1),
            "firing_mode": firing_mode,
            "pointing_mode": pointing_mode,
            "layout_type": layout_type,
            "aspect_mode": aspect_mode,
            "aspect_name": aspect_geom["name"],
            "drone_pitch_deg": aspect_geom["pitch_deg"],
            "drone_yaw_deg": aspect_geom["yaw_deg"],
            "drone_roll_deg": aspect_geom["roll_deg"],
            "presented_area_m2": aspect_geom["presented_area_m2"],
            "presented_area_cm2": aspect_geom["presented_area_cm2"],
            "planform_area_m2": aspect_geom["planform_area_m2"],
            "cross_section_ratio_pct": aspect_geom["cross_section_ratio_pct"],
            "ambient_temp_c": round(float(ambient_temp_c), 1),
            "headwind_mps": round(float(headwind_mps), 1),
            "propellant_type": propellant_type,
            "drag_model": drag_model,
            "air_density_kg_m3": temp_fx["air_density_kg_m3"],
            "nominal_gravity_drop_m": round(nominal_gravity_drop_m, 3),
            "temp_mv_delta_mps": temp_fx["mv_delta_mps"],
            "temp_spread_mult": temp_fx["temp_spread_mult"],
            "annex_c_analytical": annex_c_analytical,
            "physics_fidelity_mode": (
                "high_fidelity"
                if (use_rotor_solidity or use_v50_ballistic_limit or enable_ray_shielding or str(pattern_distribution) != "gaussian")
                else "standard"
            ),
            "use_rotor_solidity": bool(use_rotor_solidity),
            "use_v50_ballistic_limit": bool(use_v50_ballistic_limit),
            "enable_ray_shielding": bool(enable_ray_shielding),
            "pattern_distribution": str(pattern_distribution),
            "blade_pass_through_count": int(blade_pass_through_count),
            "flyer_count": int(flyer_count_total),
            "mean_v50_mps": round(float(np.mean(v50_samples)), 1) if v50_samples else None,
        },
        "drone_spec": drone_spec,
        "aspect_geometry": aspect_geom,
        "barrels": barrels,
        "barrel_stats": barrel_stats_list,
        "component_kills": component_kills_tally,
        "sample_scatter": sample_scatter,
        "volley_records": volley_records,
    }


def benchmark_drone_aspect_angles_mc(
    drone_model: str = "commercial_quad_medium",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 8.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    default_barrel_length_in: float = 24.0,
    target_distance_m: float = 35.0,
    drone_velocity_mps: float = 15.0,
    aim_error_mrad: float = 1.0,
    crosswind_mps: float = 0.0,
    default_choke: str = "Full",
    n_volleys: int = 25,
    seed: int = 42,
    custom_drone_spec: dict[str, Any] | None = None,
    custom_shell_spec: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark anti-drone interception effectiveness across all 6 standard presentation aspect angles."""
    aspects = [
        "head_on_frontal",
        "flank_side",
        "oblique_quartering",
        "tail_rear",
        "planform_top",
        "planform_bottom",
    ]
    results_list: list[dict[str, Any]] = []
    for asp in aspects:
        sim = simulate_anti_drone_multishotgun_mc(
            layout_type=layout_type,
            n_barrels=n_barrels,
            mount_radius_m=mount_radius_m,
            spacing_y_m=spacing_y_m,
            spacing_z_m=spacing_z_m,
            n_cols=n_cols,
            n_rows=n_rows,
            pointing_mode=pointing_mode,
            splay_angle_mrad=splay_angle_mrad,
            firing_mode=firing_mode,
            ripple_delay_ms=ripple_delay_ms,
            custom_barrel_specs=custom_barrel_specs,
            default_barrel_length_in=default_barrel_length_in,
            target_distance_m=target_distance_m,
            ammunition_preset=ammunition_preset,
            default_choke=default_choke,
            drone_model=drone_model,
            drone_velocity_mps=drone_velocity_mps,
            aim_error_mrad=aim_error_mrad,
            crosswind_mps=crosswind_mps,
            aspect_mode=asp,
            n_volleys=n_volleys,
            seed=seed,
            custom_drone_spec=custom_drone_spec,
            custom_shell_spec=custom_shell_spec,
            **kwargs,
        )
        sm = sim["summary"]
        results_list.append({
            "aspect_mode": asp,
            "name": sm.get("aspect_name", asp),
            "pitch_deg": sm.get("drone_pitch_deg", 0.0),
            "yaw_deg": sm.get("drone_yaw_deg", 0.0),
            "presented_area_cm2": sm.get("presented_area_cm2", 0.0),
            "cross_section_ratio_pct": sm.get("cross_section_ratio_pct", 100.0),
            "kill_probability_pct": sm["kill_probability_pct"],
            "mean_hits": sm["mean_hits_on_drone"],
            "mean_lethal_hits": sm["mean_lethal_hits"],
            "mean_energy_j": sm["mean_energy_on_drone_j"],
        })
    return {
        "drone_model": drone_model,
        "target_distance_m": target_distance_m,
        "n_barrels": n_barrels,
        "ammunition_preset": ammunition_preset,
        "aspect_results": results_list,
        "comparison_table": results_list,
    }


def compare_anti_drone_layouts(
    layout_configs: list[dict[str, Any]] | None = None,
    distance_stations_m: list[float] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    pointing_mode: str = "convergent",
    firing_mode: str = "simultaneous_volley",
    n_volleys: int = 15,
    seed: int = 42,
    custom_drone_spec: dict[str, Any] | None = None,
    custom_shell_spec: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compare multiple multi-shotgun array placement configurations across engagement ranges."""
    if layout_configs is None:
        layout_configs = [
            {"name": "6-Barrel Circular Ring", "layout_type": "circular", "n_barrels": 6, "mount_radius_m": 0.20},
            {"name": "2x3 Grid Matrix", "layout_type": "grid", "n_rows": 2, "n_cols": 3, "spacing_y_m": 0.12, "spacing_z_m": 0.12},
            {"name": "7-Barrel Honeycomb Core", "layout_type": "honeycomb", "n_barrels": 7, "mount_radius_m": 0.20},
            {"name": "6-Barrel Horizontal Fan", "layout_type": "fan_horizontal", "n_barrels": 6, "spacing_y_m": 0.12},
            {"name": "8-Barrel Concentric Dual-Ring", "layout_type": "concentric", "n_barrels": 8, "mount_radius_m": 0.25},
        ]

    if distance_stations_m is None:
        distance_stations_m = [15.0, 25.0, 35.0, 50.0, 65.0]

    comparison_table: list[dict[str, Any]] = []
    layout_curves: dict[str, dict[str, list[float]]] = {}

    for l_idx, cfg in enumerate(layout_configs):
        c_name = cfg.get("name", f"Layout {l_idx+1}")
        layout_curves[c_name] = {
            "distances_m": list(distance_stations_m),
            "pk_pct": [],
            "mean_hits": [],
            "mean_lethal": [],
            "mean_energy_j": [],
            "cloud_diam_m": [],
        }

        for d_m in distance_stations_m:
            sim = simulate_anti_drone_multishotgun_mc(
                layout_type=cfg.get("layout_type", "circular"),
                n_barrels=int(cfg.get("n_barrels", 6)),
                mount_radius_m=float(cfg.get("mount_radius_m", 0.20)),
                spacing_y_m=float(cfg.get("spacing_y_m", 0.12)),
                spacing_z_m=float(cfg.get("spacing_z_m", 0.12)),
                n_cols=int(cfg.get("n_cols", 3)),
                n_rows=int(cfg.get("n_rows", 2)),
                pointing_mode=pointing_mode,
                firing_mode=firing_mode,
                ammunition_preset=ammunition_preset,
                target_distance_m=d_m,
                drone_model=drone_model,
                drone_velocity_mps=drone_velocity_mps,
                n_volleys=n_volleys,
                seed=seed + l_idx * 17,
                custom_drone_spec=custom_drone_spec,
                custom_shell_spec=custom_shell_spec,
                **kwargs,
            )
            sm = sim["summary"]
            layout_curves[c_name]["pk_pct"].append(sm["kill_probability_pct"])
            layout_curves[c_name]["mean_hits"].append(sm["mean_hits_on_drone"])
            layout_curves[c_name]["mean_lethal"].append(sm["mean_lethal_hits"])
            layout_curves[c_name]["mean_energy_j"].append(sm["mean_energy_on_drone_j"])
            layout_curves[c_name]["cloud_diam_m"].append(sm["cloud_diameter_90_m"])

            comparison_table.append({
                "layout_name": c_name,
                "layout_type": cfg.get("layout_type", "circular"),
                "n_barrels": sm["n_barrels"],
                "distance_m": d_m,
                "kill_probability_pct": sm["kill_probability_pct"],
                "mean_hits": sm["mean_hits_on_drone"],
                "mean_lethal": sm["mean_lethal_hits"],
                "mean_energy_j": sm["mean_energy_on_drone_j"],
                "cloud_diameter_m": sm["cloud_diameter_90_m"],
            })

    return {
        "layout_configs": layout_configs,
        "distance_stations_m": distance_stations_m,
        "drone_model": drone_model,
        "ammunition_preset": ammunition_preset,
        "comparison_table": comparison_table,
        "layout_curves": layout_curves,
    }


def _interpolate_pk_range_threshold(
    distances_m: list[float],
    pk_pcts: list[float],
    threshold_pct: float,
) -> tuple[float | None, str]:
    """Interpolate effective maximum range where P_kill stays >= threshold_pct."""
    if not distances_m or not pk_pcts or len(distances_m) != len(pk_pcts):
        return None, "N/A"
    pairs = sorted(zip(distances_m, pk_pcts), key=lambda p: p[0])
    d_vals = [float(p[0]) for p in pairs]
    p_vals = [float(p[1]) for p in pairs]

    if all(p >= threshold_pct for p in p_vals):
        return d_vals[-1], f">{d_vals[-1]:.0f} m"
    if all(p < threshold_pct for p in p_vals):
        return None, f"<{d_vals[0]:.0f} m"

    # Scan from the farthest station above threshold to the next falling crossing
    for idx in range(len(d_vals) - 2, -1, -1):
        p_hi, p_lo = p_vals[idx], p_vals[idx + 1]
        d_lo, d_hi = d_vals[idx], d_vals[idx + 1]
        if p_hi >= threshold_pct and p_lo < threshold_pct:
            denom = max(1e-6, p_hi - p_lo)
            frac = (p_hi - threshold_pct) / denom
            r_interp = d_lo + frac * (d_hi - d_lo)
            return round(r_interp, 1), f"{r_interp:.1f} m"

    # Fallback if it stayed >= threshold up to an interior peak then didn't drop
    above_idx = [i for i, p in enumerate(p_vals) if p >= threshold_pct]
    if above_idx:
        r_best = d_vals[above_idx[-1]]
        return round(r_best, 1), f"{r_best:.1f} m"
    return None, f"<{d_vals[0]:.0f} m"


def evaluate_antidrone_effective_range_envelope(
    distance_stations_m: list[float] | None = None,
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Full",
    default_barrel_length_in: float = 24.0,
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    drone_heading_deg: float = 90.0,
    aim_error_mrad: float = 1.5,
    crosswind_mps: float = 0.0,
    n_volleys: int = 15,
    seed: int = 42,
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    custom_drone_spec: dict[str, Any] | None = None,
    custom_shell_spec: dict[str, Any] | None = None,
    compare_ammo_presets: list[str] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    headwind_mps: float = 0.0,
    ambient_temp_c: float = 15.0,
    propellant_type: str = "double_base",
    drag_model: str = "compressible_re_m",
    pellets_per_shot: int | None = None,
    use_rotor_solidity: bool = False,
    use_v50_ballistic_limit: bool = False,
    enable_ray_shielding: bool = False,
    pattern_distribution: str = "gaussian",
    flyer_fraction: float = 0.12,
) -> dict[str, Any]:
    """Sweep slant range for an anti-drone array to derive P_kill(R), R80, and R50 thresholds."""
    stations = sorted(float(d) for d in (distance_stations_m or [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]))

    def _run_curve(ammo_key: str, custom_sh: dict[str, Any] | None, curve_seed: int) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for idx, d_m in enumerate(stations):
            sim = simulate_anti_drone_multishotgun_mc(
                layout_type=layout_type,
                n_barrels=n_barrels,
                mount_radius_m=mount_radius_m,
                spacing_y_m=spacing_y_m,
                spacing_z_m=spacing_z_m,
                n_cols=n_cols,
                n_rows=n_rows,
                pointing_mode=pointing_mode,
                splay_angle_mrad=splay_angle_mrad,
                firing_mode=firing_mode,
                ripple_delay_ms=ripple_delay_ms,
                custom_barrel_specs=custom_barrel_specs,
                ammunition_preset=ammo_key,
                pellets_per_shot=pellets_per_shot,
                default_choke=default_choke,
                default_barrel_length_in=default_barrel_length_in,
                target_distance_m=d_m,
                drone_model=drone_model,
                drone_velocity_mps=drone_velocity_mps,
                drone_heading_deg=drone_heading_deg,
                aim_error_mrad=aim_error_mrad,
                crosswind_mps=crosswind_mps,
                n_volleys=n_volleys,
                seed=curve_seed + idx * 7,
                inner_ammunition_preset=inner_ammunition_preset if custom_sh is None else None,
                outer_ammunition_preset=outer_ammunition_preset if custom_sh is None else None,
                inner_choke=inner_choke,
                aspect_mode=aspect_mode,
                drone_pitch_deg=drone_pitch_deg,
                drone_yaw_deg=drone_yaw_deg,
                drone_roll_deg=drone_roll_deg,
                custom_drone_spec=custom_drone_spec,
                custom_shell_spec=custom_sh,
                calibration_profile=calibration_profile,
                headwind_mps=headwind_mps,
                ambient_temp_c=ambient_temp_c,
                propellant_type=propellant_type,
                drag_model=drag_model,
                use_rotor_solidity=use_rotor_solidity,
                use_v50_ballistic_limit=use_v50_ballistic_limit,
                enable_ray_shielding=enable_ray_shielding,
                pattern_distribution=pattern_distribution,
                flyer_fraction=flyer_fraction,
            )
            sm = sim["summary"]
            records.append({
                "distance_m": d_m,
                "kill_probability_pct": sm["kill_probability_pct"],
                "mean_hits_on_drone": sm["mean_hits_on_drone"],
                "mean_lethal_hits": sm["mean_lethal_hits"],
                "mean_energy_on_drone_j": sm["mean_energy_on_drone_j"],
                "mean_v_impact_mps": sm["mean_v_impact_nominal_mps"],
                "cloud_diameter_90_m": sm["cloud_diameter_90_m"],
                "ammunition_name": sm.get("ammunition_name", ammo_key),
                "drone_name": sm.get("drone_name", drone_model),
            })

        pks = [r["kill_probability_pct"] for r in records]
        r80_m, r80_label = _interpolate_pk_range_threshold(stations, pks, 80.0)
        r50_m, r50_label = _interpolate_pk_range_threshold(stations, pks, 50.0)
        return {
            "ammunition_preset": ammo_key,
            "ammunition_name": records[0]["ammunition_name"] if records else ammo_key,
            "records": records,
            "r80_m": r80_m,
            "r80_label": r80_label,
            "r50_m": r50_m,
            "r50_label": r50_label,
            "max_pk_pct": max(pks) if pks else 0.0,
            "min_pk_pct": min(pks) if pks else 0.0,
        }

    primary = _run_curve(ammunition_preset, custom_shell_spec, seed)
    comp_curves: list[dict[str, Any]] = []
    if compare_ammo_presets:
        for c_idx, alt_key in enumerate(compare_ammo_presets):
            if alt_key == ammunition_preset and custom_shell_spec is None:
                continue
            comp_curves.append(_run_curve(alt_key, None, seed + 101 * (c_idx + 1)))

    return {
        "distance_stations_m": stations,
        "records": primary["records"],
        "comparison_curves": comp_curves,
        "summary": {
            "r80_m": primary["r80_m"],
            "r80_label": primary["r80_label"],
            "r50_m": primary["r50_m"],
            "r50_label": primary["r50_label"],
            "max_pk_pct": primary["max_pk_pct"],
            "min_pk_pct": primary["min_pk_pct"],
            "ammunition_preset": ammunition_preset,
            "ammunition_name": primary["ammunition_name"],
            "drone_name": primary["records"][0]["drone_name"] if primary["records"] else drone_model,
            "n_barrels": n_barrels,
            "layout_type": layout_type,
        },
    }


def evaluate_antidrone_range_speed_heatmap(
    distance_stations_m: list[float] | None = None,
    speed_stations_mps: list[float] | None = None,
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    default_choke: str = "Full",
    default_barrel_length_in: float = 24.0,
    drone_model: str = "commercial_quad_medium",
    drone_heading_deg: float = 90.0,
    aim_error_mrad: float = 1.5,
    crosswind_mps: float = 0.0,
    n_volleys: int = 12,
    seed: int = 42,
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    custom_drone_spec: dict[str, Any] | None = None,
    custom_shell_spec: dict[str, Any] | None = None,
    calibration_profile: dict[str, Any] | None = None,
    headwind_mps: float = 0.0,
    ambient_temp_c: float = 15.0,
    propellant_type: str = "double_base",
    drag_model: str = "compressible_re_m",
    pellets_per_shot: int | None = None,
    use_rotor_solidity: bool = False,
    use_v50_ballistic_limit: bool = False,
    enable_ray_shielding: bool = False,
    pattern_distribution: str = "gaussian",
    flyer_fraction: float = 0.12,
) -> dict[str, Any]:
    """Compute a 2D Speed x Slant-Range P_kill engagement envelope heatmap."""
    d_stations = sorted(float(d) for d in (distance_stations_m or [15.0, 25.0, 35.0, 45.0, 60.0, 75.0]))
    v_stations = sorted(float(v) for v in (speed_stations_mps or [5.0, 12.0, 20.0, 30.0, 40.0]))

    pk_matrix: list[list[float]] = []
    mean_hits_matrix: list[list[float]] = []
    mean_lethal_matrix: list[list[float]] = []
    grid_records: list[dict[str, Any]] = []
    r80_by_speed_m: list[dict[str, Any]] = []

    for v_idx, v_mps in enumerate(v_stations):
        pk_row: list[float] = []
        hits_row: list[float] = []
        lethal_row: list[float] = []
        for d_idx, d_m in enumerate(d_stations):
            sim = simulate_anti_drone_multishotgun_mc(
                layout_type=layout_type,
                n_barrels=n_barrels,
                mount_radius_m=mount_radius_m,
                spacing_y_m=spacing_y_m,
                spacing_z_m=spacing_z_m,
                n_cols=n_cols,
                n_rows=n_rows,
                pointing_mode=pointing_mode,
                splay_angle_mrad=splay_angle_mrad,
                firing_mode=firing_mode,
                ripple_delay_ms=ripple_delay_ms,
                custom_barrel_specs=custom_barrel_specs,
                ammunition_preset=ammunition_preset,
                pellets_per_shot=pellets_per_shot,
                default_choke=default_choke,
                default_barrel_length_in=default_barrel_length_in,
                target_distance_m=d_m,
                drone_model=drone_model,
                drone_velocity_mps=v_mps,
                drone_heading_deg=drone_heading_deg,
                aim_error_mrad=aim_error_mrad,
                crosswind_mps=crosswind_mps,
                n_volleys=n_volleys,
                seed=seed + v_idx * 31 + d_idx * 7,
                inner_ammunition_preset=inner_ammunition_preset if custom_shell_spec is None else None,
                outer_ammunition_preset=outer_ammunition_preset if custom_shell_spec is None else None,
                inner_choke=inner_choke,
                aspect_mode=aspect_mode,
                drone_pitch_deg=drone_pitch_deg,
                drone_yaw_deg=drone_yaw_deg,
                drone_roll_deg=drone_roll_deg,
                custom_drone_spec=custom_drone_spec,
                custom_shell_spec=custom_shell_spec,
                calibration_profile=calibration_profile,
                headwind_mps=headwind_mps,
                ambient_temp_c=ambient_temp_c,
                propellant_type=propellant_type,
                drag_model=drag_model,
                use_rotor_solidity=use_rotor_solidity,
                use_v50_ballistic_limit=use_v50_ballistic_limit,
                enable_ray_shielding=enable_ray_shielding,
                pattern_distribution=pattern_distribution,
                flyer_fraction=flyer_fraction,
            )
            sm = sim["summary"]
            pk = float(sm["kill_probability_pct"])
            pk_row.append(pk)
            hits_row.append(float(sm["mean_hits_on_drone"]))
            lethal_row.append(float(sm["mean_lethal_hits"]))
            grid_records.append({
                "drone_velocity_mps": v_mps,
                "distance_m": d_m,
                "kill_probability_pct": pk,
                "mean_hits_on_drone": sm["mean_hits_on_drone"],
                "mean_lethal_hits": sm["mean_lethal_hits"],
                "mean_energy_on_drone_j": sm["mean_energy_on_drone_j"],
            })

        pk_matrix.append(pk_row)
        mean_hits_matrix.append(hits_row)
        mean_lethal_matrix.append(lethal_row)
        r80_v, r80_lbl = _interpolate_pk_range_threshold(d_stations, pk_row, 80.0)
        r80_by_speed_m.append({
            "drone_velocity_mps": v_mps,
            "r80_m": r80_v,
            "r80_label": r80_lbl,
        })

    n_viable_80 = sum(1 for r in grid_records if r["kill_probability_pct"] >= 80.0)
    return {
        "distance_stations_m": d_stations,
        "speed_stations_mps": v_stations,
        "pk_matrix": pk_matrix,
        "mean_hits_matrix": mean_hits_matrix,
        "mean_lethal_matrix": mean_lethal_matrix,
        "grid_records": grid_records,
        "r80_by_speed_m": r80_by_speed_m,
        "summary": {
            "total_cells": len(grid_records),
            "viable_80pct_cells": n_viable_80,
            "viable_80pct_share": round((n_viable_80 / max(1, len(grid_records))) * 100.0, 1),
            "max_pk_pct": max((r["kill_probability_pct"] for r in grid_records), default=0.0),
            "mean_pk_pct": round(float(np.mean([r["kill_probability_pct"] for r in grid_records])), 1) if grid_records else 0.0,
        },
    }


