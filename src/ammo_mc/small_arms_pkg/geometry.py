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
        return True
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

    # 2. Lateral unit vector (starboard)
    if abs(abs(theta_r) - math.pi / 2) < 1e-4:
        u_r0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        u_r0 = np.array([math.sin(psi_r), math.cos(psi_r), 0.0], dtype=float)
        norm_r0 = np.linalg.norm(u_r0)
        u_r0 = u_r0 / norm_r0 if norm_r0 > 1e-9 else np.array([0.0, 1.0, 0.0])

    # Rodrigues roll around u_fwd
    u_right = (
        u_r0 * math.cos(phi_r)
        + np.cross(u_fwd, u_r0) * math.sin(phi_r)
        + u_fwd * (np.dot(u_fwd, u_r0)) * (1.0 - math.cos(phi_r))
    )
    norm_r = np.linalg.norm(u_right)
    u_right = u_right / norm_r if norm_r > 1e-9 else np.array([0.0, 1.0, 0.0])

    # 3. Upward unit vector (dorsal)
    u_up = np.cross(u_right, u_fwd)
    if u_up[2] < 0 and abs(theta_r) < math.pi / 2 - 0.05:
        u_up = -u_up
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

    # Total Presented Area
    span_m = float(drone_spec.get("span_m", 0.40))
    planform_area = span_m**2 * 0.62
    tot_presented_area = area_fuse + area_batt + area_wings + area_rotors_tot * 0.75 + area_motors_tot * 0.5
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
        "rotors": rotors_out,
        "motors": motors_out,
        "arms": arms_out,
        "wings": wings_out,
        "presented_area_m2": round(tot_presented_area, 6),
        "presented_area_cm2": round(tot_presented_area * 10000.0, 1),
        "planform_area_m2": round(planform_area, 6),
        "cross_section_ratio_pct": round(cross_section_ratio_pct, 1),
        "planform_ratio_pct": round(cross_section_ratio_pct, 1),
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


def evaluate_drone_impact(
    y_rel_m: float,
    z_rel_m: float,
    drone_spec: dict[str, Any],
    striking_energy_j: float,
    aspect_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether an impact point (y, z) hits a drone component and whether it is lethal.

    Supports both:
      1. 3D Projected Cross-Section (when aspect_geometry is provided)
      2. Planform Top-Down / Overhead fallback (when aspect_geometry is None)
    """
    e_rotor_lethal = float(drone_spec.get("lethal_energy_rotor_j", 8.0))
    e_core_lethal = float(drone_spec.get("lethal_energy_core_j", 20.0))
    e_arm_lethal = float(drone_spec.get("lethal_energy_arm_j", 30.0))

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

        # 2. Battery / Munition Payload Box Projected Polygon
        batt_poly = aspect_geometry.get("battery_polygon")
        if batt_poly and len(batt_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(batt_poly)):
            return {
                "is_hit": True,
                "component": "battery_payload",
                "is_lethal": bool(striking_energy_j >= e_core_lethal * 0.75),
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
    fuse_h = float(drone_spec.get("fuselage_h_m", 0.10))

    # 1. Central fuselage check (core avionics, battery, IMU)
    in_fuselage = (abs(y_rel_m) <= fuse_w / 2.0) and (abs(z_rel_m) <= fuse_h / 2.0)
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


