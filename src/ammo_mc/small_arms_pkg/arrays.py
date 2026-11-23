"""Small Arms Ballistics, Shotgun Patterning & Anti-Drone Dispersion Package — arrays submodule."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from .catalogs import DRONE_ASPECT_PRESETS, DRONE_TARGET_SPECS, SHOTGUN_PRESETS, get_choke_definitions, get_drone_target_specs, get_shotgun_presets
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

    Supports homogeneous or heterogeneous Core (inner) + Perimeter (outer) ammunition/choke mixes.

    Coordinates on turret face (X=0):
      y_m: horizontal crossrange position (m), positive = right
      z_m: vertical height position (m), positive = up
    """
    barrels: list[dict[str, Any]] = []

    if layout_type == "circular":
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
        if n_barrels <= 3:
            inner_count = 1
        elif n_barrels <= 6:
            inner_count = 1
        elif n_barrels <= 9:
            inner_count = 3
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
        hex_count = max(6, n_barrels - 1)
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

    return barrels


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
) -> dict[str, Any]:
    """Execute Monte Carlo simulation of an anti-drone defense volley using a multi-shotgun array.

    Evaluates:
      - Multi-barrel spatial array geometry (circular, grid, fan, honeycomb, concentric, custom)
      - Heterogeneous core vs perimeter ammunition presets and chokes
      - Pointing convergence, parallel bores, or splay curtain + optional dynamic recoil walk offsets
      - Simultaneous volley vs sequential ripple vs balanced symmetric pair ripple fire
      - 3D longitudinal shot stringing vs moving drone kinematics
      - 3D target drone cross-section projection (head-on frontal, flank, oblique, tail, planform)
      - Drone component-level damage (fuselage core, battery payload, motor pods, rotors, arms, wings)
      - Overall drone kill probability P_kill, lethal hit counts, and energy deposition
    """
    rng = np.random.default_rng(seed)

    # 1. Resolve Baseline Nominal Ammunition Parameters
    presets = get_shotgun_presets()

    def _resolve_barrel_ammo(preset_key: str) -> dict[str, Any]:
        if preset_key in presets:
            a = presets[preset_key]
            return {
                "pellets": int(pellets_per_shot or a["pellets_per_shot"]),
                "d_mm": float(pellet_diameter_mm or a["pellet_diameter_mm"]),
                "m_g": float(pellet_mass_g or a["pellet_mass_g"]),
                "v0": float(v0_mps or a["v0_mps"]),
                "effective_cd": float(a.get("effective_cd", 0.47)),
            }
        return {
            "pellets": int(pellets_per_shot or 27),
            "d_mm": float(pellet_diameter_mm or 6.10),
            "m_g": float(pellet_mass_g or 1.34),
            "v0": float(v0_mps or 415.0),
            "effective_cd": 0.47,
        }

    nom_ammo = _resolve_barrel_ammo(ammunition_preset)
    eff_pellets = nom_ammo["pellets"]
    eff_d_mm = nom_ammo["d_mm"]
    eff_m_g = nom_ammo["m_g"]
    eff_v0 = nom_ammo["v0"]

    # 2. Resolve Drone Specs & 3D Projected Cross-Section Geometry
    drone_specs_cat = get_drone_target_specs()
    drone_spec = drone_specs_cat.get(drone_model, drone_specs_cat["commercial_quad_medium"])
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
        default_barrel_length_in=default_barrel_length_in,
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

    # 4. Aerodynamic deceleration constants (true area-to-mass ratio across Lead/Steel/Tungsten/Flechette)
    rho_air = 1.225
    d_m_nom = eff_d_mm / 1000.0
    area_nom = math.pi * 0.25 * (d_m_nom ** 2)
    kappa_nom = (rho_air * nom_ammo["effective_cd"] * area_nom) / (2.0 * (eff_m_g / 1000.0))
    v_impact_nominal = eff_v0 * math.exp(-kappa_nom * target_distance_m)
    t_flight_nominal = target_distance_m / max((eff_v0 + v_impact_nominal) / 2.0, 10.0)

    # Moving target vector
    heading_rad = math.radians(drone_heading_deg)
    v_drone_y = drone_velocity_mps * math.sin(heading_rad)
    v_drone_z = drone_velocity_mps * math.cos(heading_rad)

    # 5. Volley Simulation Loop
    volley_records: list[dict[str, Any]] = []
    kills_count = 0
    all_hits_counts: list[int] = []
    all_lethal_counts: list[int] = []
    all_energy_on_drone: list[float] = []

    # Per-barrel hit accumulator
    barrel_stats_map = {
        b["barrel_id"]: {
            "barrel_id": b["barrel_id"],
            "label": b["label"],
            "y_m": b["y_m"],
            "z_m": b["z_m"],
            "choke": b["choke"],
            "ammunition_preset": b.get("ammunition_preset", ammunition_preset),
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
        "striking_velocity_mps": [],
        "flight_time_s": [],
    }

    aim_sigma_m = (target_distance_m * (aim_error_mrad / 1000.0)) / 2.0 if aim_error_mrad > 0 else 0.0
    choke_defs = get_choke_definitions()

    barrel_ammo_specs = [_resolve_barrel_ammo(str(b.get("ammunition_preset", ammunition_preset))) for b in barrels]
    tot_pellets_per_volley = sum(spec["pellets"] for spec in barrel_ammo_specs)

    for v_idx in range(n_volleys):
        # Turret pointing jitter for this volley
        aim_err_y = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0
        aim_err_z = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0

        v_hits = 0
        v_lethal = 0
        v_energy = 0.0
        v_fuselage_lethal = 0
        v_battery_lethal = 0
        v_rotors_killed: set[str] = set()

        for b_idx, b in enumerate(barrels):
            b_id = b["barrel_id"]
            y0 = float(b["y_m"])
            z0 = float(b["z_m"])
            b_choke = str(b.get("choke", default_choke))
            b_len = float(b.get("barrel_length_in", default_barrel_length_in))
            b_spec = barrel_ammo_specs[b_idx]
            b_pellets = int(b_spec["pellets"])
            b_d_m = float(b_spec["d_mm"]) / 1000.0
            b_m_kg = float(b_spec["m_g"]) / 1000.0
            b_area_m2 = math.pi * 0.25 * (b_d_m ** 2)
            b_cd_base = float(b_spec["effective_cd"])

            # Firing time offset (simultaneous vs sequential ripple vs balanced pair ripple)
            t_fire = fire_times_s[b_idx]

            # Barrel length velocity scaling & gas wash turbulence
            v0_barrel = float(b_spec["v0"]) + 2.2 * (b_len - 28.0)
            wash_turb = 1.0 + 0.35 * (max(0.0, 24.0 - b_len) / 24.0) ** 1.5

            # Choke radial spread parameter
            c_info = choke_defs.get(b_choke, choke_defs["Modified"])
            p_bench = float(c_info["benchmark_40yd_30in_pct"])
            r_bench = 0.381
            sigma_ref = r_bench / math.sqrt(-2.0 * math.log(1.0 - p_bench))
            theta_choke = sigma_ref / 36.58
            sigma_radial = (0.01852 / 4.0 + theta_choke * target_distance_m * (target_distance_m / 36.58) ** 0.15) * wash_turb

            # Optional dynamic recoil muzzle climb / yaw walk offsets
            dyn_yaw = float(barrel_yaw_offsets_mrad[b_idx]) if barrel_yaw_offsets_mrad and b_idx < len(barrel_yaw_offsets_mrad) else 0.0
            dyn_pitch = float(barrel_pitch_offsets_mrad[b_idx]) if barrel_pitch_offsets_mrad and b_idx < len(barrel_pitch_offsets_mrad) else 0.0

            # Nominal bore center at target plane
            y_bore = y0 + target_distance_m * ((b["aim_yaw_mrad"] + dyn_yaw) / 1000.0) + aim_err_y
            z_bore = z0 + target_distance_m * ((b["aim_pitch_mrad"] + dyn_pitch) / 1000.0) + aim_err_z

            barrel_stats_map[b_id]["pellets_fired"] += b_pellets

            # Generate pellets for this barrel
            for _ in range(b_pellets):
                # Radial dispersion
                dy = rng.normal(0.0, sigma_radial) + (crosswind_mps * 0.008 * target_distance_m)
                dz = rng.normal(0.0, sigma_radial)

                y_impact_raw = y_bore + dy
                z_impact_raw = z_bore + dz

                # Pellet drag variations & longitudinal stringing
                cd_pellet = rng.uniform(0.92 * b_cd_base, 1.25 * b_cd_base)
                kappa_p = (rho_air * cd_pellet * b_area_m2) / (2.0 * max(b_m_kg, 1e-6))
                v_impact = v0_barrel * math.exp(-kappa_p * target_distance_m)
                t_flight = target_distance_m / max((v0_barrel + v_impact) / 2.0, 10.0)
                t_arrival = t_fire + t_flight

                # Moving target position relative to nominal lead at arrival time
                dt_target = t_arrival - t_flight_nominal
                tgt_offset_y = v_drone_y * dt_target
                tgt_offset_z = v_drone_z * dt_target

                # Relative impact position on drone target frame
                y_rel = y_impact_raw - tgt_offset_y
                z_rel = z_impact_raw - tgt_offset_z

                ke_pellet = 0.5 * b_m_kg * (v_impact**2)

                # Component hit check
                hit_info = evaluate_drone_impact(y_rel, z_rel, drone_spec, ke_pellet, aspect_geometry=aspect_geom)

                if hit_info["is_hit"]:
                    v_hits += 1
                    barrel_stats_map[b_id]["hits_on_drone"] += 1
                    v_energy += ke_pellet

                    comp = hit_info["component"]
                    if hit_info["is_lethal"]:
                        v_lethal += 1
                        barrel_stats_map[b_id]["lethal_hits"] += 1
                        if comp == "fuselage_core":
                            v_fuselage_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                        elif comp == "battery_payload":
                            v_battery_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                        elif comp.startswith("rotor_"):
                            v_rotors_killed.add(comp)
                            barrel_stats_map[b_id]["rotor_hits"] += 1
                        elif comp.startswith("motor_"):
                            r_id = comp.replace("motor_", "rotor_")
                            v_rotors_killed.add(r_id)
                            barrel_stats_map[b_id]["rotor_hits"] += 1

                # Record sample volley for visualization (first volley)
                if v_idx == 0:
                    sample_scatter["y"].append(round(y_rel, 4))
                    sample_scatter["z"].append(round(z_rel, 4))
                    sample_scatter["barrel_id"].append(b_id)
                    sample_scatter["is_hit"].append(hit_info["is_hit"])
                    sample_scatter["is_lethal"].append(hit_info["is_lethal"])
                    sample_scatter["component"].append(hit_info["component"])
                    sample_scatter["striking_energy_j"].append(round(ke_pellet, 2))
                    sample_scatter["striking_velocity_mps"].append(round(v_impact, 1))
                    sample_scatter["flight_time_s"].append(round(t_arrival, 4))

        # Volley Kill Condition
        is_killed = bool(
            (v_lethal >= crit_hits_req)
            or (v_fuselage_lethal >= 1)
            or (v_battery_lethal >= 1)
            or (len(v_rotors_killed) >= 2)
        )
        if is_killed:
            kills_count += 1

        all_hits_counts.append(v_hits)
        all_lethal_counts.append(v_lethal)
        all_energy_on_drone.append(v_energy)

        volley_records.append({
            "volley_idx": v_idx + 1,
            "hits_on_drone": v_hits,
            "lethal_hits": v_lethal,
            "energy_on_drone_j": round(v_energy, 1),
            "fuselage_lethal": v_fuselage_lethal,
            "battery_lethal": v_battery_lethal,
            "rotors_killed": len(v_rotors_killed),
            "is_killed": is_killed,
        })

    # Summary Statistics
    pk_pct = round((kills_count / max(1, n_volleys)) * 100.0, 1)
    mean_hits = round(float(np.mean(all_hits_counts)), 1)
    mean_lethal = round(float(np.mean(all_lethal_counts)), 1)
    mean_energy = round(float(np.mean(all_energy_on_drone)), 1)

    # Cloud geometry in sample volley
    if sample_scatter["y"]:
        rads = np.hypot(sample_scatter["y"], sample_scatter["z"])
        cep_50 = round(float(np.percentile(rads, 50)), 3)
        r_90 = round(float(np.percentile(rads, 90)), 3)
        cloud_diam_90 = round(2.0 * r_90, 3)
    else:
        cep_50, r_90, cloud_diam_90 = 0.5, 1.0, 2.0

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
            "kill_probability_pct": pk_pct,
            "mean_hits_on_drone": mean_hits,
            "mean_lethal_hits": mean_lethal,
            "mean_energy_on_drone_j": mean_energy,
            "total_pellets_per_volley": tot_pellets_per_volley,
            "n_barrels": n_act_barrels,
            "n_volleys": n_volleys,
            "cloud_cep_50_m": cep_50,
            "cloud_r_90_m": r_90,
            "cloud_diameter_90_m": cloud_diam_90,
            "target_distance_m": target_distance_m,
            "drone_model": drone_model,
            "drone_name": drone_spec.get("name", drone_model),
            "drone_velocity_mps": drone_velocity_mps,
            "v0_nominal_mps": eff_v0,
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
        },
        "drone_spec": drone_spec,
        "aspect_geometry": aspect_geom,
        "barrels": barrels,
        "barrel_stats": barrel_stats_list,
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


