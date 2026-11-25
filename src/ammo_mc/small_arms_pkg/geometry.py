"""Small Arms Ballistics, Shotgun Patterning & Anti-Drone Dispersion Package — geometry submodule."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from .catalogs import DRONE_ASPECT_PRESETS

def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull of points (monotone chain algorithm). Returns vertices in CCW order."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) <= 3:
        return pts
    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) <= 3:
        return unique_pts
    idx = np.lexsort((unique_pts[:, 1], unique_pts[:, 0]))
    pts_sorted = unique_pts[idx]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts_sorted:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def polygon_area(pts: np.ndarray) -> float:
    """Calculate area of a 2D polygon using the Shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def point_in_convex_polygon(y: float, z: float, poly_verts: np.ndarray) -> bool:
    """Test if point (y, z) is inside a 2D convex polygon."""
    n = len(poly_verts)
    if n < 3:
        return False
    signs: list[bool] = []
    for i in range(n):
        p1 = poly_verts[i]
        p2 = poly_verts[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (z - p1[1]) - (p2[1] - p1[1]) * (y - p1[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    if not signs:
        return False
    return all(s == signs[0] for s in signs)


def compute_drone_projected_geometry(
    drone_spec: dict[str, Any],
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
) -> dict[str, Any]:
    """Compute the 2D projected cross-section silhouette and component boundaries.

    Transforms 3D drone components (fuselage, battery, rotors, motor pods, arms,
    wings) into the 2D line-of-fire target plane (Y = horizontal, Z = vertical)
    under arbitrary 3-axis Euler angles (pitch, yaw, roll).
    """
    preset = DRONE_ASPECT_PRESETS.get(aspect_mode, DRONE_ASPECT_PRESETS["head_on_frontal"])
    if aspect_mode != "custom":
        yaw_deg = float(preset["yaw_deg"])
        pitch_deg = float(preset["pitch_deg"])
        roll_deg = float(preset["roll_deg"])
    else:
        yaw_deg = float(drone_yaw_deg)
        pitch_deg = float(drone_pitch_deg)
        roll_deg = float(drone_roll_deg)

    psi_r = math.radians(yaw_deg)
    theta_r = math.radians(pitch_deg)
    phi_r = math.radians(roll_deg)

    # 1. Forward unit vector (nose)
    ux_fwd = -math.cos(psi_r) * math.cos(theta_r)
    uy_fwd = math.sin(psi_r) * math.cos(theta_r)
    uz_fwd = math.sin(theta_r)
    u_fwd = np.array([ux_fwd, uy_fwd, uz_fwd], dtype=float)
    norm_f = np.linalg.norm(u_fwd)
    u_fwd = u_fwd / norm_f if norm_f > 1e-9 else np.array([-1.0, 0.0, 0.0])

    # 2. Lateral unit vector (starboard, right-handed wrt u_fwd facing -X at yaw=0)
    if abs(abs(theta_r) - math.pi / 2) < 1e-4:
        u_r0 = np.array([0.0, -1.0, 0.0], dtype=float)
    else:
        u_r0 = np.array([-math.sin(psi_r), -math.cos(psi_r), 0.0], dtype=float)
        norm_r0 = np.linalg.norm(u_r0)
        u_r0 = u_r0 / norm_r0 if norm_r0 > 1e-9 else np.array([0.0, -1.0, 0.0])

    # Rodrigues roll around u_fwd
    u_right = (
        u_r0 * math.cos(phi_r)
        + np.cross(u_fwd, u_r0) * math.sin(phi_r)
        + u_fwd * (np.dot(u_fwd, u_r0)) * (1.0 - math.cos(phi_r))
    )
    norm_r = np.linalg.norm(u_right)
    u_right = u_right / norm_r if norm_r > 1e-9 else np.array([0.0, -1.0, 0.0])

    # 3. Upward unit vector (right-handed: u_up = u_fwd x u_right, det(M) == +1.0 everywhere)
    u_up = np.cross(u_fwd, u_right)
    norm_up = np.linalg.norm(u_up)
    u_up = u_up / norm_up if norm_up > 1e-9 else np.array([0.0, 0.0, 1.0])

    # 3x3 rotation matrix: p_los = M @ [x_b, y_b, z_b]
    M = np.column_stack([u_fwd, u_right, u_up])

    # Fuselage 3D Box
    l_f = float(drone_spec.get("fuselage_l_m", drone_spec.get("span_m", 0.4) * 0.55))
    w_f = float(drone_spec.get("fuselage_w_m", 0.12))
    h_f = float(drone_spec.get("fuselage_h_m", 0.09))

    verts_b = np.array([
        [+l_f / 2, +w_f / 2, +h_f / 2],
        [+l_f / 2, +w_f / 2, -h_f / 2],
        [+l_f / 2, -w_f / 2, +h_f / 2],
        [+l_f / 2, -w_f / 2, -h_f / 2],
        [-l_f / 2, +w_f / 2, +h_f / 2],
        [-l_f / 2, +w_f / 2, -h_f / 2],
        [-l_f / 2, -w_f / 2, +h_f / 2],
        [-l_f / 2, -w_f / 2, -h_f / 2],
    ], dtype=float)
    verts_los = verts_b @ M.T
    fuse_proj = verts_los[:, [1, 2]]
    fuse_hull = _convex_hull_2d(fuse_proj)
    area_fuse = polygon_area(fuse_hull)

    # Battery 3D Box
    l_b = float(drone_spec.get("battery_l_m", l_f * 0.6))
    w_b = float(drone_spec.get("battery_w_m", w_f * 0.6))
    h_b = float(drone_spec.get("battery_h_m", h_f * 0.55))
    z_b_off = float(drone_spec.get("battery_offset_z_m", -h_f / 2 - h_b / 2))

    b_verts_b = np.array([
        [+l_b / 2, +w_b / 2, z_b_off + h_b / 2],
        [+l_b / 2, +w_b / 2, z_b_off - h_b / 2],
        [+l_b / 2, -w_b / 2, z_b_off + h_b / 2],
        [+l_b / 2, -w_b / 2, z_b_off - h_b / 2],
        [-l_b / 2, +w_b / 2, z_b_off + h_b / 2],
        [-l_b / 2, +w_b / 2, z_b_off - h_b / 2],
        [-l_b / 2, -w_b / 2, z_b_off + h_b / 2],
        [-l_b / 2, -w_b / 2, z_b_off - h_b / 2],
    ], dtype=float)
    b_verts_los = b_verts_b @ M.T
    batt_proj = b_verts_los[:, [1, 2]]
    batt_hull = _convex_hull_2d(batt_proj)
    area_batt = polygon_area(batt_hull)

    # Underslung Warhead Payload 3D Axisymmetric Hull (e.g. PG-7M)
    warhead_hull = np.empty((0, 2), dtype=float)
    area_warhead = 0.0
    warhead_depth_x = 0.0
    if drone_spec.get("has_warhead"):
        l_wh = float(drone_spec.get("warhead_length_m", 0.290))
        d_wh = float(drone_spec.get("warhead_diam_m", 0.070))
        r_wh = 0.5 * d_wh
        z_wh_off = float(drone_spec.get("warhead_offset_z_m", -h_f / 2.0 - r_wh - 0.006))
        # Axisymmetric PG-7M warhead profile along body +X: tip, ogive/70mm cone-cylinder bulge, aft taper
        stations = [
            (+0.50 * l_wh, 0.28 * r_wh),
            (+0.22 * l_wh, 0.72 * r_wh),
            (+0.05 * l_wh, 1.00 * r_wh),
            (-0.14 * l_wh, 1.00 * r_wh),
            (-0.34 * l_wh, 0.58 * r_wh),
            (-0.50 * l_wh, 0.42 * r_wh),
        ]
        wh_pts_b = []
        for x_st, r_st in stations:
            for ang_idx in range(8):
                ang = ang_idx * (2.0 * math.pi / 8.0)
                wh_pts_b.append([x_st, r_st * math.cos(ang), z_wh_off + r_st * math.sin(ang)])
        wh_verts_los = np.array(wh_pts_b, dtype=float) @ M.T
        wh_proj = wh_verts_los[:, [1, 2]]
        warhead_hull = _convex_hull_2d(wh_proj)
        area_warhead = polygon_area(warhead_hull)
        warhead_depth_x = float(np.mean(wh_verts_los[:, 0]))

    # Rotors & Motor Pods
    rotors_spec = drone_spec.get("rotors", [])
    h_blade = float(drone_spec.get("rotor_thickness_m", 0.022))
    d_motor = float(drone_spec.get("motor_d_m", 0.038))
    h_motor = float(drone_spec.get("motor_h_m", 0.032))
    arm_d = float(drone_spec.get("arm_d_m", 0.020))

    rotors_out = []
    motors_out = []
    arms_out = []
    area_rotors_tot = 0.0
    area_motors_tot = 0.0

    n_normal_los = M @ np.array([0.0, 0.0, 1.0], dtype=float)
    nx, ny, nz = float(n_normal_los[0]), float(n_normal_los[1]), float(n_normal_los[2])
    d_yz = math.hypot(ny, nz)

    if d_yz > 1e-6:
        u_maj = np.array([-nz / d_yz, ny / d_yz], dtype=float)
        u_min = np.array([ny / d_yz, nz / d_yz], dtype=float)
    else:
        u_maj = np.array([1.0, 0.0], dtype=float)
        u_min = np.array([0.0, 1.0], dtype=float)

    c_fuse_proj = (M @ np.array([0.0, 0.0, 0.0], dtype=float))[[1, 2]]

    for r_idx, r in enumerate(rotors_spec):
        rx_b = float(r.get("x_m", r.get("z_m", 0.0)))
        ry_b = float(r.get("y_m", 0.0))
        rz_b = float(r.get("z_m_vert", 0.015))
        r_rad = float(r.get("radius_m", 0.10))

        c_hub_los = M @ np.array([rx_b, ry_b, rz_b], dtype=float)
        cy, cz = float(c_hub_los[1]), float(c_hub_los[2])

        a = r_rad
        b = max(r_rad * abs(nx), h_blade / 2.0)
        area_rot = math.pi * a * b
        area_rotors_tot += area_rot

        rotors_out.append({
            "id": r.get("id", f"R{r_idx+1}"),
            "y_m": cy,
            "z_m": cz,
            "depth_x": float(c_hub_los[0]),
            "rotor_radius_m": r_rad,
            "semi_major_m": a,
            "semi_minor_m": b,
            "major_axis_m": a,
            "minor_axis_m": b,
            "u_maj": u_maj.tolist(),
            "u_min": u_min.tolist(),
            "area_m2": area_rot,
        })

        area_mot = d_motor * h_motor
        area_motors_tot += area_mot
        motors_out.append({
            "id": r.get("id", f"M{r_idx+1}"),
            "y_m": cy,
            "z_m": cz,
            "depth_x": float(c_hub_los[0]),
            "w_m": d_motor,
            "h_m": h_motor,
            "area_m2": area_mot,
        })

        arms_out.append({
            "id": f"Arm_{r_idx+1}",
            "p1": [float(c_fuse_proj[0]), float(c_fuse_proj[1])],
            "p2": [cy, cz],
            "diameter_m": arm_d,
        })

    # Fixed-Wing components (if fixed_wing)
    wings_out = []
    area_wings = 0.0
    if drone_spec.get("drone_type") == "fixed_wing":
        b_span = float(drone_spec.get("wingspan_m", 1.60))
        c_chord = float(drone_spec.get("wing_chord_m", 0.22))
        t_thick = float(drone_spec.get("wing_thickness_m", 0.035))
        # Left Wing
        w_l_b = np.array([
            [-c_chord / 2, -b_span / 2, +t_thick / 2],
            [-c_chord / 2, -b_span / 2, -t_thick / 2],
            [+c_chord / 2, -b_span / 2, +t_thick / 2],
            [+c_chord / 2, -b_span / 2, -t_thick / 2],
            [-c_chord / 2, -w_f / 2, +t_thick / 2],
            [-c_chord / 2, -w_f / 2, -t_thick / 2],
            [+c_chord / 2, -w_f / 2, +t_thick / 2],
            [+c_chord / 2, -w_f / 2, -t_thick / 2],
        ], dtype=float)
        w_l_proj = (w_l_b @ M.T)[:, [1, 2]]
        w_l_hull = _convex_hull_2d(w_l_proj)
        wings_out.append(w_l_hull.tolist())
        area_wings += polygon_area(w_l_hull)

        # Right Wing
        w_r_b = np.array([
            [-c_chord / 2, +w_f / 2, +t_thick / 2],
            [-c_chord / 2, +w_f / 2, -t_thick / 2],
            [+c_chord / 2, +w_f / 2, +t_thick / 2],
            [+c_chord / 2, +w_f / 2, -t_thick / 2],
            [-c_chord / 2, +b_span / 2, +t_thick / 2],
            [-c_chord / 2, +b_span / 2, -t_thick / 2],
            [+c_chord / 2, +b_span / 2, +t_thick / 2],
            [+c_chord / 2, +b_span / 2, -t_thick / 2],
        ], dtype=float)
        w_r_proj = (w_r_b @ M.T)[:, [1, 2]]
        w_r_hull = _convex_hull_2d(w_r_proj)
        wings_out.append(w_r_hull.tolist())
        area_wings += polygon_area(w_r_hull)

    # Total Presented Area (accounting for fuselage-battery & rotor-motor projection overlap)
    span_m = float(drone_spec.get("span_m", 0.40))
    if drone_spec.get("drone_type") == "fixed_wing":
        b_span = float(drone_spec.get("wingspan_m", span_m))
        c_chord = float(drone_spec.get("wing_chord_m", 0.22))
        planform_area = max(b_span * c_chord + l_f * w_f * 0.75 + area_rotors_tot * 0.15, 1e-4)
    else:
        planform_rot = sum(math.pi * (float(r.get("radius_m", 0.10)) ** 2) for r in rotors_spec)
        planform_area = max(l_f * w_f + 0.75 * planform_rot, (span_m ** 2) * 0.62, 1e-4)

    batt_vis_frac = 1.0 - 0.65 * abs(nx)
    mot_vis_frac = max(0.15, 1.0 - 0.70 * abs(nx))
    wh_vis_frac = 1.0 - 0.55 * abs(nx)
    tot_presented_area = (
        area_fuse
        + area_batt * batt_vis_frac
        + area_warhead * wh_vis_frac
        + area_wings
        + area_rotors_tot * 0.75
        + area_motors_tot * mot_vis_frac
    )
    cross_section_ratio_pct = min(100.0, (tot_presented_area / planform_area) * 100.0)

    return {
        "aspect_mode": aspect_mode,
        "name": preset["name"],
        "description": preset["description"],
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "rotation_matrix": M.tolist(),
        "fuselage_polygon": fuse_hull.tolist(),
        "battery_polygon": batt_hull.tolist(),
        "warhead_polygon": warhead_hull.tolist() if len(warhead_hull) >= 3 else [],
        "rotors": rotors_out,
        "motors": motors_out,
        "arms": arms_out,
        "wings": wings_out,
        "fuselage_area_cm2": round(area_fuse * 10000.0, 1),
        "battery_area_cm2": round(area_batt * 10000.0, 1),
        "warhead_area_cm2": round(area_warhead * 10000.0, 1),
        "rotors_area_cm2": round(area_rotors_tot * 10000.0, 1),
        "presented_area_m2": round(tot_presented_area, 6),
        "presented_area_cm2": round(tot_presented_area * 10000.0, 1),
        "planform_area_m2": round(planform_area, 6),
        "cross_section_ratio_pct": round(cross_section_ratio_pct, 1),
        "planform_ratio_pct": round(cross_section_ratio_pct, 1),
        "component_depths_x": {
            "fuselage_core": float(np.mean(verts_los[:, 0])),
            "battery_payload": float(np.mean(b_verts_los[:, 0])),
            "warhead_pg7m": warhead_depth_x,
        },
        "rotor_disc_obliquity_deg": round(math.degrees(math.acos(min(1.0, abs(nx)))), 2),
    }


def check_target_hit(
    y_m: float,
    z_m: float,
    target_type: str = "30_inch_circle",
    custom_w_m: float = 0.5,
    custom_h_m: float = 1.0,
) -> bool:
    """Evaluate whether an impact point (y, z) lies within the specified target silhouette.

    Coordinates: y = horizontal (crossrange), z = vertical (height from target center).
    """
    if target_type == "30_inch_circle":
        # 30-inch diameter = 0.762 m (radius = 0.381 m)
        return float(y_m**2 + z_m**2) <= (0.381**2)

    elif target_type == "10_inch_circle":
        # 10-inch diameter = 0.254 m (radius = 0.127 m)
        return float(y_m**2 + z_m**2) <= (0.127**2)

    elif target_type == "e_type_silhouette":
        # NATO E-type humanoid silhouette:
        # Torso: width 0.45m (-0.225 to +0.225), height 0.65m (-0.45 to +0.20)
        # Head: width 0.22m (-0.11 to +0.11), height 0.25m (+0.20 to +0.45)
        in_torso = (-0.225 <= y_m <= 0.225) and (-0.45 <= z_m <= 0.20)
        in_head = (-0.11 <= y_m <= 0.11) and (0.20 < z_m <= 0.45)
        return bool(in_torso or in_head)

    elif target_type == "clay_pigeon":
        # Clay target: oval disc 11cm wide (0.11m), 3cm thick (0.03m)
        return float((y_m / 0.055) ** 2 + (z_m / 0.025) ** 2) <= 1.0

    elif target_type == "drone":
        # Micro-UAV / Drone: 0.40m x 0.40m square
        return bool(abs(y_m) <= 0.20 and abs(z_m) <= 0.20)

    elif target_type == "rectangle":
        return bool(abs(y_m) <= custom_w_m / 2.0 and abs(z_m) <= custom_h_m / 2.0)

    # Default to 30-inch circle
    return float(y_m**2 + z_m**2) <= (0.381**2)


_COMPONENT_ARMOR_PROFILES: dict[str, dict[str, float]] = {
    "fuselage_core": {"ref_v50_mps": 135.0, "plug_areal_kg_m2": 2.4, "hardness_sens": 0.25},
    "battery_payload": {"ref_v50_mps": 118.0, "plug_areal_kg_m2": 2.0, "hardness_sens": 0.28},
    "warhead_pg7m": {"ref_v50_mps": 152.0, "plug_areal_kg_m2": 2.9, "hardness_sens": 0.30},
    "motor": {"ref_v50_mps": 145.0, "plug_areal_kg_m2": 2.6, "hardness_sens": 0.32},
    "rotor": {"ref_v50_mps": 72.0, "plug_areal_kg_m2": 0.75, "hardness_sens": 0.15},
    "structural_arm": {"ref_v50_mps": 165.0, "plug_areal_kg_m2": 2.2, "hardness_sens": 0.22},
    "wing_spar": {"ref_v50_mps": 110.0, "plug_areal_kg_m2": 1.3, "hardness_sens": 0.18},
}


def compute_rotor_blade_strike_probability(
    radial_frac: float,
    n_blades: int = 3,
    mean_chord_m: float = 0.022,
    rotor_radius_m: float = 0.12,
    rpm: float = 12000.0,
    blade_thickness_m: float = 0.0025,
    pellet_diameter_m: float = 0.004,
    pellet_velocity_mps: float = 250.0,
) -> float:
    """Compute geometric + rotational sweeping probability of striking a spinning rotor blade.

    Combines instantaneous blade planform solidity (N_b * c / (2*pi*r)) with swept rotor
    volume during pellet axial transit time across blade thickness.
    """
    r_tot = max(float(rotor_radius_m), 0.02)
    r_frac = max(0.0, min(1.0, float(radial_frac)))
    r_abs = r_frac * r_tot
    if r_abs <= max(0.012, 0.14 * r_tot):
        return 1.0  # Direct rotor hub / shaft collar hit

    n_b = max(2, int(n_blades))
    chord = max(0.006, float(mean_chord_m) * (1.15 - 0.35 * r_frac))
    p_static = (n_b * chord) / (2.0 * math.pi * r_abs)
    omega = (max(0.0, float(rpm)) * 2.0 * math.pi) / 60.0
    v_eff = max(10.0, float(pellet_velocity_mps))
    p_swept = (n_b * omega * (max(0.001, float(blade_thickness_m)) + max(0.001, float(pellet_diameter_m)))) / (2.0 * math.pi * v_eff)
    return float(max(0.02, min(1.0, p_static + p_swept)))


def compute_component_v50_perforation(
    component: str,
    pellet_velocity_mps: float,
    pellet_mass_kg: float,
    pellet_diameter_m: float,
    pellet_density_g_cm3: float = 11.3,
    obliquity_deg: float = 0.0,
) -> dict[str, Any]:
    """Evaluate sectional-density & obliquity-dependent V50 perforation and Recht-Ipson residual speed."""
    ck = str(component)
    if ck.startswith("motor_"):
        prof = _COMPONENT_ARMOR_PROFILES["motor"]
    elif ck.startswith("rotor_"):
        prof = _COMPONENT_ARMOR_PROFILES["rotor"]
    else:
        prof = _COMPONENT_ARMOR_PROFILES.get(ck, _COMPONENT_ARMOR_PROFILES["fuselage_core"])

    d_m = max(0.001, float(pellet_diameter_m))
    m_kg = max(1e-5, float(pellet_mass_kg))
    rho = max(2.0, float(pellet_density_g_cm3))
    area_p = math.pi * 0.25 * (d_m ** 2)
    sd_kg_m2 = m_kg / area_p
    # Reference sphere: 6.1mm #4 Buck hardened lead (1.34g -> SD ~ 45.85 kg/m^2, rho=11.3)
    sd_ratio = max(0.25, sd_kg_m2 / 45.85)
    # Harder dense tungsten / steel resists mushrooming vs soft lead at equivalent KE
    mat_hardness_factor = (11.3 / rho) ** float(prof["hardness_sens"])
    if rho >= 16.0:
        mat_hardness_factor *= 0.86
    elif 7.4 <= rho <= 8.4:
        mat_hardness_factor *= 0.94

    theta_rad = math.radians(max(0.0, min(70.0, abs(float(obliquity_deg)))))
    obliq_factor = (1.0 / max(math.cos(theta_rad), 0.34)) ** 0.75

    v50 = float(prof["ref_v50_mps"]) * (sd_ratio ** -0.45) * mat_hardness_factor * obliq_factor
    v_cur = max(0.0, float(pellet_velocity_mps))
    slope = max(0.12 * v50, 8.0)
    z_log = max(-40.0, min(40.0, (v_cur - v50) / slope))
    p_perf = 1.0 / (1.0 + math.exp(-z_log))

    m_plug = float(prof["plug_areal_kg_m2"]) * area_p * (1.0 / max(math.cos(theta_rad), 0.34))
    a_r = m_kg / max(m_kg + m_plug, 1e-6)
    v_res = a_r * math.sqrt(max(0.0, v_cur ** 2 - v50 ** 2)) if v_cur > v50 else 0.0
    return {
        "v50_mps": round(v50, 2),
        "p_perforation": float(p_perf),
        "residual_velocity_mps": float(v_res),
        "sectional_density_kg_m2": round(sd_kg_m2, 2),
    }


def evaluate_drone_impact(
    y_rel_m: float,
    z_rel_m: float,
    drone_spec: dict[str, Any],
    striking_energy_j: float,
    aspect_geometry: dict[str, Any] | None = None,
    *,
    pellet_velocity_mps: float | None = None,
    pellet_mass_kg: float | None = None,
    pellet_diameter_m: float | None = None,
    pellet_density_g_cm3: float = 11.3,
    use_rotor_solidity: bool = False,
    use_v50_ballistic_limit: bool = False,
    enable_ray_shielding: bool = False,
    rng: Any | None = None,
) -> dict[str, Any]:
    """Evaluate whether an impact point (y, z) hits a drone component and whether it is lethal.

    Supports both:
      1. 3D Projected Cross-Section (when aspect_geometry is provided)
      2. Planform Top-Down / Overhead fallback (when aspect_geometry is None)
    Optional high-fidelity modes model spinning rotor blade solidity, sectional-density
    V50 perforation curves, and front-to-back ray shielding with Recht-Ipson residual velocity.
    """
    e_rotor_lethal = float(drone_spec.get("lethal_energy_rotor_j", 8.0))
    e_core_lethal = float(drone_spec.get("lethal_energy_core_j", 20.0))
    e_arm_lethal = float(drone_spec.get("lethal_energy_arm_j", 30.0))
    e_warhead_lethal = float(drone_spec.get("lethal_energy_warhead_j", e_core_lethal * 0.85))

    if aspect_geometry is not None and (use_rotor_solidity or use_v50_ballistic_limit or enable_ray_shielding):
        depths = aspect_geometry.get("component_depths_x", {})
        candidates: list[dict[str, Any]] = []

        fuse_poly = aspect_geometry.get("fuselage_polygon")
        if fuse_poly and len(fuse_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(fuse_poly)):
            candidates.append({
                "component": "fuselage_core",
                "depth_x": float(depths.get("fuselage_core", 0.0)),
                "e_req": e_core_lethal,
                "critical": True,
                "opaque": True,
                "obliquity_deg": min(55.0, abs(float(aspect_geometry.get("pitch_deg", 0.0)))),
            })

        batt_poly = aspect_geometry.get("battery_polygon")
        if batt_poly and len(batt_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(batt_poly)):
            candidates.append({
                "component": "battery_payload",
                "depth_x": float(depths.get("battery_payload", 0.0)),
                "e_req": e_core_lethal * 0.75,
                "critical": True,
                "opaque": True,
                "obliquity_deg": min(55.0, abs(float(aspect_geometry.get("pitch_deg", 0.0)))),
            })

        wh_poly = aspect_geometry.get("warhead_polygon")
        if wh_poly and len(wh_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(wh_poly)):
            candidates.append({
                "component": "warhead_pg7m",
                "depth_x": float(depths.get("warhead_pg7m", 0.0)),
                "e_req": e_warhead_lethal,
                "critical": True,
                "opaque": True,
                "obliquity_deg": 25.0,
            })

        for m in aspect_geometry.get("motors", []):
            my, mz = float(m["y_m"]), float(m["z_m"])
            if abs(y_rel_m - my) <= float(m["w_m"]) / 2.0 and abs(z_rel_m - mz) <= float(m["h_m"]) / 2.0:
                candidates.append({
                    "component": f"motor_{m.get('id', 'M')}",
                    "depth_x": float(m.get("depth_x", 0.0)),
                    "e_req": e_rotor_lethal,
                    "critical": True,
                    "opaque": True,
                    "obliquity_deg": 20.0,
                })

        for r in aspect_geometry.get("rotors", []):
            ry, rz = float(r["y_m"]), float(r["z_m"])
            a, b = float(r["semi_major_m"]), float(r["semi_minor_m"])
            u_maj, u_min = r.get("u_maj", [1.0, 0.0]), r.get("u_min", [0.0, 1.0])
            dy, dz = y_rel_m - ry, z_rel_m - rz
            u = dy * u_maj[0] + dz * u_maj[1]
            v = dy * u_min[0] + dz * u_min[1]
            norm_ell = (u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2
            if norm_ell <= 1.0:
                candidates.append({
                    "component": f"rotor_{r.get('id', 'R')}",
                    "depth_x": float(r.get("depth_x", 0.0)),
                    "e_req": e_rotor_lethal,
                    "critical": True,
                    "opaque": False,
                    "radial_frac": math.sqrt(max(0.0, norm_ell)),
                    "rotor_radius_m": float(r.get("rotor_radius_m", a)),
                    "obliquity_deg": float(aspect_geometry.get("rotor_disc_obliquity_deg", 15.0)),
                })

        for w_poly in aspect_geometry.get("wings", []):
            if w_poly and len(w_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(w_poly)):
                candidates.append({
                    "component": "wing_spar",
                    "depth_x": 0.05,
                    "e_req": e_arm_lethal,
                    "critical": False,
                    "opaque": True,
                    "obliquity_deg": 35.0,
                })

        for arm in aspect_geometry.get("arms", []):
            p1, p2 = arm.get("p1", [0.0, 0.0]), arm.get("p2", [0.0, 0.0])
            arm_r = float(arm.get("diameter_m", 0.020)) / 2.0
            seg_dy, seg_dz = float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])
            seg_len_sq = seg_dy ** 2 + seg_dz ** 2
            if seg_len_sq > 1e-6:
                t_p = max(0.0, min(1.0, ((y_rel_m - float(p1[0])) * seg_dy + (z_rel_m - float(p1[1])) * seg_dz) / seg_len_sq))
                if math.hypot(y_rel_m - (float(p1[0]) + t_p * seg_dy), z_rel_m - (float(p1[1]) + t_p * seg_dz)) <= arm_r:
                    candidates.append({
                        "component": "structural_arm",
                        "depth_x": 0.02,
                        "e_req": e_arm_lethal,
                        "critical": False,
                        "opaque": True,
                        "obliquity_deg": 25.0,
                    })

        if candidates:
            if enable_ray_shielding:
                candidates.sort(key=lambda c: float(c["depth_x"]))

            m_eff = float(pellet_mass_kg) if (pellet_mass_kg and pellet_mass_kg > 0) else 0.00134
            d_eff = float(pellet_diameter_m) if (pellet_diameter_m and pellet_diameter_m > 0) else 0.0061
            v_cur = float(pellet_velocity_mps) if (pellet_velocity_mps is not None and pellet_velocity_mps > 0) else math.sqrt(2.0 * max(striking_energy_j, 0.0) / m_eff)
            ke_cur = 0.5 * m_eff * (v_cur ** 2)
            blade_passed_any = False
            n_blades_spec = int(drone_spec.get("rotor_blade_count", 3))
            rpm_spec = float(drone_spec.get("rotor_rpm", 11500.0))

            first_hit: dict[str, Any] | None = None
            best_lethal: dict[str, Any] | None = None
            last_v50: float | None = None

            for cand in candidates:
                comp = str(cand["component"])
                if comp.startswith("rotor_") and use_rotor_solidity:
                    p_blade = compute_rotor_blade_strike_probability(
                        radial_frac=float(cand.get("radial_frac", 0.5)),
                        n_blades=n_blades_spec,
                        rotor_radius_m=float(cand.get("rotor_radius_m", 0.12)),
                        rpm=rpm_spec,
                        pellet_diameter_m=d_eff,
                        pellet_velocity_mps=v_cur,
                    )
                    u_draw = float(rng.random()) if rng is not None else 0.5
                    if u_draw >= p_blade:
                        blade_passed_any = True
                        continue

                v50_info: dict[str, Any] | None = None
                if use_v50_ballistic_limit:
                    v50_info = compute_component_v50_perforation(
                        comp,
                        pellet_velocity_mps=v_cur,
                        pellet_mass_kg=m_eff,
                        pellet_diameter_m=d_eff,
                        pellet_density_g_cm3=pellet_density_g_cm3,
                        obliquity_deg=float(cand.get("obliquity_deg", 0.0)),
                    )
                    last_v50 = float(v50_info["v50_mps"])
                    u_perf = float(rng.random()) if rng is not None else 0.5
                    is_perf = bool((u_perf < float(v50_info["p_perforation"])) and (ke_cur >= 0.45 * float(cand["e_req"])))
                    is_lethal = is_perf
                else:
                    is_lethal = bool(ke_cur >= float(cand["e_req"]))
                    is_perf = is_lethal

                hit_rec = {
                    "is_hit": True,
                    "component": comp,
                    "is_lethal": is_lethal,
                    "critical_system": bool(cand["critical"]),
                    "striking_energy_j": ke_cur,
                    "v50_mps": last_v50,
                    "blade_pass_through": blade_passed_any,
                }
                if first_hit is None:
                    first_hit = hit_rec
                if is_lethal and best_lethal is None:
                    best_lethal = hit_rec
                    if not enable_ray_shielding:
                        return best_lethal

                if not enable_ray_shielding:
                    return hit_rec

                # Ray shielding propagation front-to-back
                if is_perf and v50_info is not None:
                    v_cur = float(v50_info["residual_velocity_mps"])
                elif is_perf:
                    v_cur = math.sqrt(max(0.0, (2.0 * max(ke_cur - float(cand["e_req"]), 0.0)) / m_eff))
                else:
                    v_cur = 0.0
                ke_cur = 0.5 * m_eff * (v_cur ** 2)
                if cand["opaque"] and v_cur < 15.0:
                    break

            if best_lethal is not None:
                return best_lethal
            if first_hit is not None:
                return first_hit
            return {
                "is_hit": False,
                "component": "miss",
                "is_lethal": False,
                "critical_system": False,
                "striking_energy_j": striking_energy_j,
                "v50_mps": last_v50,
                "blade_pass_through": blade_passed_any,
            }

    if aspect_geometry is not None:
        # 1. Fuselage 3D Box Projected Polygon (Flight Controller, Avionics, IMU)
        fuse_poly = aspect_geometry.get("fuselage_polygon")
        if fuse_poly and len(fuse_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(fuse_poly)):
            return {
                "is_hit": True,
                "component": "fuselage_core",
                "is_lethal": bool(striking_energy_j >= e_core_lethal),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

        # 2. Battery Projected Polygon (Top or Underslung Pack)
        batt_poly = aspect_geometry.get("battery_polygon")
        if batt_poly and len(batt_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(batt_poly)):
            return {
                "is_hit": True,
                "component": "battery_payload",
                "is_lethal": bool(striking_energy_j >= e_core_lethal * 0.75),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

        # 2b. Underslung Warhead Payload (e.g. PG-7M)
        wh_poly = aspect_geometry.get("warhead_polygon")
        if wh_poly and len(wh_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(wh_poly)):
            return {
                "is_hit": True,
                "component": "warhead_pg7m",
                "is_lethal": bool(striking_energy_j >= e_warhead_lethal),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

        # 3. Motor Pods (Direct hub motor strike)
        motors = aspect_geometry.get("motors", [])
        for m in motors:
            my = float(m["y_m"])
            mz = float(m["z_m"])
            mw = float(m["w_m"])
            mh = float(m["h_m"])
            if abs(y_rel_m - my) <= mw / 2.0 and abs(z_rel_m - mz) <= mh / 2.0:
                return {
                    "is_hit": True,
                    "component": f"motor_{m.get('id', 'M')}",
                    "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                    "critical_system": True,
                    "striking_energy_j": striking_energy_j,
                }

        # 4. Spinning Rotor Discs (Projected Ellipses or Edge-on Blade Bands)
        rotors = aspect_geometry.get("rotors", [])
        for r in rotors:
            ry = float(r["y_m"])
            rz = float(r["z_m"])
            a = float(r["semi_major_m"])
            b = float(r["semi_minor_m"])
            u_maj = r.get("u_maj", [1.0, 0.0])
            u_min = r.get("u_min", [0.0, 1.0])
            dy = y_rel_m - ry
            dz = z_rel_m - rz
            u = dy * u_maj[0] + dz * u_maj[1]
            v = dy * u_min[0] + dz * u_min[1]
            if (u / a)**2 + (v / b)**2 <= 1.0:
                return {
                    "is_hit": True,
                    "component": f"rotor_{r.get('id', 'R')}",
                    "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                    "critical_system": True,
                    "striking_energy_j": striking_energy_j,
                }

        # 5. Fixed-Wing Aerodynamic Panels (if fixed_wing)
        wings = aspect_geometry.get("wings", [])
        for w_poly in wings:
            if w_poly and len(w_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(w_poly)):
                return {
                    "is_hit": True,
                    "component": "wing_spar",
                    "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                    "critical_system": False,
                    "striking_energy_j": striking_energy_j,
                }

        # 6. Structural Carbon Arms / Booms
        arms = aspect_geometry.get("arms", [])
        for arm in arms:
            p1 = arm.get("p1", [0.0, 0.0])
            p2 = arm.get("p2", [0.0, 0.0])
            arm_r = float(arm.get("diameter_m", 0.020)) / 2.0
            p1_y, p1_z = float(p1[0]), float(p1[1])
            p2_y, p2_z = float(p2[0]), float(p2[1])
            seg_dy = p2_y - p1_y
            seg_dz = p2_z - p1_z
            seg_len_sq = seg_dy**2 + seg_dz**2
            if seg_len_sq > 1e-6:
                t_p = max(0.0, min(1.0, ((y_rel_m - p1_y) * seg_dy + (z_rel_m - p1_z) * seg_dz) / seg_len_sq))
                proj_y = p1_y + t_p * seg_dy
                proj_z = p1_z + t_p * seg_dz
                dist_to_arm = math.hypot(y_rel_m - proj_y, z_rel_m - proj_z)
                if dist_to_arm <= arm_r:
                    return {
                        "is_hit": True,
                        "component": "structural_arm",
                        "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                        "critical_system": False,
                        "striking_energy_j": striking_energy_j,
                    }

        return {
            "is_hit": False,
            "component": "miss",
            "is_lethal": False,
            "critical_system": False,
            "striking_energy_j": striking_energy_j,
        }

    # Fallback to 2D Planform (Top-down)
    drone_type = drone_spec.get("drone_type", "quadcopter")
    fuse_w = float(drone_spec.get("fuselage_w_m", 0.10))
    fuse_l = float(drone_spec.get("fuselage_l_m", drone_spec.get("fuselage_h_m", 0.10)))

    # 1. Central fuselage check (core avionics, battery, IMU)
    in_fuselage = (abs(y_rel_m) <= fuse_w / 2.0) and (abs(z_rel_m) <= fuse_l / 2.0)
    if in_fuselage:
        return {
            "is_hit": True,
            "component": "fuselage_core",
            "is_lethal": bool(striking_energy_j >= e_core_lethal),
            "critical_system": True,
            "striking_energy_j": striking_energy_j,
        }

    # 2. Rotors check
    rotors = drone_spec.get("rotors", [])
    for r in rotors:
        ry = float(r["y_m"])
        rz = float(r["z_m"])
        rr = float(r["radius_m"])
        d_rotor = math.hypot(y_rel_m - ry, z_rel_m - rz)
        if d_rotor <= rr:
            return {
                "is_hit": True,
                "component": f"rotor_{r.get('id', 'R')}",
                "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

    # 3. Fixed wing check (if fixed wing)
    if drone_type == "fixed_wing":
        wingspan = float(drone_spec.get("wingspan_m", 1.60))
        chord = float(drone_spec.get("wing_chord_m", 0.22))
        if abs(y_rel_m) <= wingspan / 2.0 and abs(z_rel_m) <= chord / 2.0:
            return {
                "is_hit": True,
                "component": "wing_spar",
                "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                "critical_system": False,
                "striking_energy_j": striking_energy_j,
            }

    # 4. Structural arms check (quad/octo)
    for r in rotors:
        ry = float(r["y_m"])
        rz = float(r["z_m"])
        arm_len_sq = ry**2 + rz**2
        if arm_len_sq > 1e-4:
            t_proj = max(0.0, min(1.0, (y_rel_m * ry + z_rel_m * rz) / arm_len_sq))
            proj_y = t_proj * ry
            proj_z = t_proj * rz
            dist_to_arm = math.hypot(y_rel_m - proj_y, z_rel_m - proj_z)
            if dist_to_arm <= 0.020:  # 20mm carbon arm tube
                return {
                    "is_hit": True,
                    "component": "structural_arm",
                    "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                    "critical_system": False,
                    "striking_energy_j": striking_energy_j,
                }

    return {
        "is_hit": False,
        "component": "miss",
        "is_lethal": False,
        "critical_system": False,
        "striking_energy_j": striking_energy_j,
    }


