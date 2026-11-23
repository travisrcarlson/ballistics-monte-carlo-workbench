"""Interactive 3D & 2D Ballistics, Tactical & C-UAS Plotly Visualization Package — antidrone submodule."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ammo_mc.data import get_kb
from ammo_mc.environment import aerodynamic_reynolds_number, eb_04_isa_atmosphere
from ammo_mc.mcdrag import McDragInputs
from ammo_mc.schemas import Scenario, SimulationResult
from ammo_mc.terminal_blast import (
    evaluate_blast_field,
    tb_01_scaled_distance,
    tb_02_tnt_equivalence,
    tb_10_brick_house_damage_range,
    tb_15_simple_safety_distances,
)
from ammo_mc.terminal_frag import (
    tf_01_gurney_cyl_sphere,
    tf_04_mott_distribution,
    tf_06_fragment_presented_area,
    tf_08_taylor_angle,
    tf_11_hazardous_fragment_ke,
)
from ammo_mc.vis_cuas import (  # noqa: E402,F401
    create_2d_cuas_fallout_safety_map_figure,
    create_3d_cuas_fallout_footprint_figure,
    create_3d_swarm_hemisphere_figure,
    create_airburst_standoff_cone_figure,
    create_antidrone_optimizer_pareto_figure,
    create_cuas_payload_range_comparison_figure,
    create_fcs_tracking_and_evasion_figure,
    create_swarm_doctrine_tradeoff_figure,
    create_turret_recoil_dynamics_figure,
)
from ammo_mc.vis_cuas_mission import (  # noqa: E402,F401
    create_3d_animated_intercept_figure,
    create_kill_chain_timeline_figure,
    create_logistics_thermal_and_cost_figure,
    create_multiturret_site_map_figure,
)
from ammo_mc.vis_cuas_analytics import (  # noqa: E402,F401
    create_ab_comparator_radar_and_bars_figure,
    create_layered_defense_figure,
    create_real_map_site_defense_figure,
    create_sensitivity_tornado_figure,
)

def create_3d_antidrone_defense_figure(sim_res: dict[str, Any]) -> go.Figure:
    """Create interactive 3D WebGL figure of the anti-drone multi-shotgun array and intercept cloud."""
    summary = sim_res.get("summary", {})
    barrels = sim_res.get("barrels", [])
    drone_spec = sim_res.get("drone_spec", {})
    scatter = sim_res.get("sample_scatter", {})
    R = float(summary.get("target_distance_m", 35.0))

    fig = go.Figure()

    # Color palette for barrels
    b_colors = [
        "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4",
        "#EC4899", "#84CC16", "#F97316", "#6366F1", "#14B8A6", "#E11D48",
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
    ]

    # 1. Muzzle Mount Plate at X=0
    max_mount_r = max([math.hypot(b["y_m"], b["z_m"]) for b in barrels] + [0.20]) * 1.35
    th_circ = np.linspace(0, 2 * np.pi, 60)
    plate_y = max_mount_r * np.cos(th_circ)
    plate_z = max_mount_r * np.sin(th_circ)
    plate_x = np.zeros_like(plate_y)

    fig.add_trace(
        go.Scatter3d(
            x=plate_x,
            y=plate_y,
            z=plate_z,
            mode="lines",
            line=dict(color="#475569", width=4),
            name="Turret Mount Plate (X=0)",
            showlegend=True,
        )
    )

    # 2. Barrels at Muzzle and Bore Centerlines
    for idx, b in enumerate(barrels):
        col = b_colors[idx % len(b_colors)]
        b_id = b["barrel_id"]
        y0 = float(b["y_m"])
        z0 = float(b["z_m"])
        yaw_mrad = float(b.get("aim_yaw_mrad", 0.0))
        pitch_mrad = float(b.get("aim_pitch_mrad", 0.0))

        y_tgt = y0 + R * (yaw_mrad / 1000.0)
        z_tgt = z0 + R * (pitch_mrad / 1000.0)

        # Muzzle marker
        fig.add_trace(
            go.Scatter3d(
                x=[0.0],
                y=[y0],
                z=[z0],
                mode="markers+text",
                marker=dict(size=7, color=col, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
                text=[b_id],
                textposition="top center",
                name=f"{b_id} ({b.get('label', '')})",
                showlegend=True,
            )
        )

        # Bore trajectory centerline to target
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, R],
                y=[y0, y_tgt],
                z=[z0, z_tgt],
                mode="lines",
                line=dict(color=col, width=2.5, dash="solid"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{b_id} Bore Line: Yaw {yaw_mrad:+.1f} mrad, Pitch {pitch_mrad:+.1f} mrad",
            )
        )

    # 3. Intercept Target Plane at X=R
    plane_w = 2.5
    pw_y = np.array([-plane_w, plane_w, plane_w, -plane_w, -plane_w])
    pw_z = np.array([-plane_w, -plane_w, plane_w, plane_w, -plane_w])
    pw_x = np.full_like(pw_y, R)

    fig.add_trace(
        go.Scatter3d(
            x=pw_x,
            y=pw_y,
            z=pw_z,
            mode="lines",
            line=dict(color="rgba(100, 116, 139, 0.4)", width=2, dash="dash"),
            name=f"Target Plane (R={R:.1f}m)",
            showlegend=True,
        )
    )

    # 4. 3D Drone Representation at Interception Point (X=R, Y=0, Z=0)
    drone_type = drone_spec.get("drone_type", "quadcopter")
    aspect_geom = sim_res.get("aspect_geometry")
    M_rot = np.array(aspect_geom["rotation_matrix"], dtype=float) if aspect_geom and "rotation_matrix" in aspect_geom else np.eye(3)

    rotors = drone_spec.get("rotors", [])
    for r in rotors:
        rx_b = float(r.get("x_m", r.get("z_m", 0.0)))
        ry_b = float(r.get("y_m", 0.0))
        rz_b = float(r.get("z_m_vert", 0.015))
        r_rad = float(r.get("radius_m", 0.10))

        # 3D hub position in world space
        hub_world = np.array([R, 0.0, 0.0]) + M_rot @ np.array([rx_b, ry_b, rz_b])
        c_world = np.array([R, 0.0, 0.0]) + M_rot @ np.array([0.0, 0.0, 0.0])

        fig.add_trace(
            go.Scatter3d(
                x=[c_world[0], hub_world[0]],
                y=[c_world[1], hub_world[1]],
                z=[c_world[2], hub_world[2]],
                mode="lines",
                line=dict(color="#06B6D4", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 3D tilted rotor circle
        th_rot = np.linspace(0, 2 * np.pi, 30)
        pts_body = np.column_stack([
            rx_b + r_rad * np.cos(th_rot),
            ry_b + r_rad * np.sin(th_rot),
            np.full_like(th_rot, rz_b),
        ])
        pts_world = np.array([R, 0.0, 0.0]) + (pts_body @ M_rot.T)

        fig.add_trace(
            go.Scatter3d(
                x=pts_world[:, 0],
                y=pts_world[:, 1],
                z=pts_world[:, 2],
                mode="lines",
                line=dict(color="#38BDF8", width=3),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Rotor {r.get('id', '')} (r={r_rad*100:.0f}cm)",
            )
        )

    # Central fuselage 3D Box Wireframe
    l_f = float(drone_spec.get("fuselage_l_m", 0.20))
    w_f = float(drone_spec.get("fuselage_w_m", 0.12))
    h_f = float(drone_spec.get("fuselage_h_m", 0.09))
    box_b = np.array([
        [-l_f / 2, -w_f / 2, -h_f / 2],
        [+l_f / 2, -w_f / 2, -h_f / 2],
        [+l_f / 2, +w_f / 2, -h_f / 2],
        [-l_f / 2, +w_f / 2, -h_f / 2],
        [-l_f / 2, -w_f / 2, -h_f / 2],
        [-l_f / 2, -w_f / 2, +h_f / 2],
        [+l_f / 2, -w_f / 2, +h_f / 2],
        [+l_f / 2, +w_f / 2, +h_f / 2],
        [-l_f / 2, +w_f / 2, +h_f / 2],
        [-l_f / 2, -w_f / 2, +h_f / 2],
    ], dtype=float)
    box_w = np.array([R, 0.0, 0.0]) + (box_b @ M_rot.T)
    fig.add_trace(
        go.Scatter3d(
            x=box_w[:, 0],
            y=box_w[:, 1],
            z=box_w[:, 2],
            mode="lines",
            line=dict(color="#F59E0B", width=4),
            name="Fuselage Core (3D Oriented)",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Central fuselage marker
    fig.add_trace(
        go.Scatter3d(
            x=[R],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=10, color="#F59E0B", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            text=[f"{drone_spec.get('name', 'Drone')}"],
            textposition="bottom center",
            name="Drone Target Center",
            showlegend=True,
        )
    )

    # 5. Pellet Cloud at Target Plane
    if scatter and scatter.get("y"):
        sc_y = np.array(scatter["y"])
        sc_z = np.array(scatter["z"])
        sc_hit = np.array(scatter["is_hit"], dtype=bool)
        sc_lethal = np.array(scatter["is_lethal"], dtype=bool)
        sc_ke = np.array(scatter["striking_energy_j"])
        sc_x = np.full_like(sc_y, R)

        if np.any(sc_hit):
            fig.add_trace(
                go.Scatter3d(
                    x=sc_x[sc_hit],
                    y=sc_y[sc_hit],
                    z=sc_z[sc_hit],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=["#EF4444" if l else "#F59E0B" for l in sc_lethal[sc_hit]],
                        symbol="diamond",
                        line=dict(color="#FFFFFF", width=1),
                    ),
                    name="Pellet Hits on Drone",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[
                        f"HIT: {comp} | Energy: {e:.1f}J | Lethal: {l}"
                        for comp, e, l in zip(
                            np.array(scatter["component"])[sc_hit],
                            sc_ke[sc_hit],
                            sc_lethal[sc_hit],
                        )
                    ],
                )
            )

        miss_mask = ~sc_hit
        if np.any(miss_mask):
            miss_idx = np.where(miss_mask)[0]
            if len(miss_idx) > 250:
                miss_idx = np.random.choice(miss_idx, size=250, replace=False)
            fig.add_trace(
                go.Scatter3d(
                    x=sc_x[miss_idx],
                    y=sc_y[miss_idx],
                    z=sc_z[miss_idx],
                    mode="markers",
                    marker=dict(size=3, color="rgba(99, 102, 241, 0.45)", symbol="circle"),
                    name="Pellet Cloud (Intercept Net)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[f"Miss Pellet: ({y:.2f}m, {z:.2f}m) | Energy: {e:.1f}J" for y, z, e in zip(sc_y[miss_idx], sc_z[miss_idx], sc_ke[miss_idx])],
                )
            )

    fig.update_layout(
        title=(
            f"🛸 3D Anti-Drone Multi-Shotgun Array & Intercept Volume @ {R:.1f}m<br>"
            f"<sup>Layout: {summary.get('layout_type', '').title()} ({summary.get('n_barrels', 6)} Barrels) | "
            f"Target: {summary.get('drone_name', '')} | P_kill: {summary.get('kill_probability_pct', 0.0):.1f}%</sup>"
        ),
        scene=dict(
            xaxis=dict(title="Range X (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title="Crossrange Y (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title="Height Z (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            aspectratio=dict(x=2.4, y=1.0, z=1.0),
            camera=dict(eye=dict(x=-1.8, y=-1.5, z=0.8)),
        ),
        margin=dict(l=10, r=10, t=65, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#F8FAFC", size=10)),
        height=620,
    )
    return fig


def create_2d_antidrone_intercept_figure(
    sim_res: dict[str, Any],
    color_by: str = "barrel",
) -> go.Figure:
    """Create detailed 2D target board figure showing the drone silhouette, component impacts, and pellet cloud."""
    summary = sim_res.get("summary", {})
    barrels = sim_res.get("barrels", [])
    drone_spec = sim_res.get("drone_spec", {})
    scatter = sim_res.get("sample_scatter", {})
    R = float(summary.get("target_distance_m", 35.0))

    fig = go.Figure()

    # 1. 30-inch and 10-inch Benchmark Rings
    th = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(
        go.Scatter(
            x=0.381 * np.cos(th),
            y=0.381 * np.sin(th),
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5, dash="dash"),
            name="30-inch Circle (0.76m diam)",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=0.127 * np.cos(th),
            y=0.127 * np.sin(th),
            mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.4)", width=1.5, dash="dot"),
            name="10-inch Core (0.25m diam)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # 2. Drone Silhouette (3D Projected Cross-Section or Planform Fallback)
    aspect_geom = sim_res.get("aspect_geometry")
    if aspect_geom:
        # A. Structural Arms
        arms = aspect_geom.get("arms", [])
        for a_idx, arm in enumerate(arms):
            p1 = arm.get("p1", [0.0, 0.0])
            p2 = arm.get("p2", [0.0, 0.0])
            fig.add_trace(
                go.Scatter(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    mode="lines",
                    line=dict(color="#334155", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # B. Fuselage Core Projected Polygon
        fuse_poly = aspect_geom.get("fuselage_polygon", [])
        if fuse_poly and len(fuse_poly) >= 3:
            fp_y = [pt[0] for pt in fuse_poly] + [fuse_poly[0][0]]
            fp_z = [pt[1] for pt in fuse_poly] + [fuse_poly[0][1]]
            fig.add_trace(
                go.Scatter(
                    x=fp_y,
                    y=fp_z,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(245, 158, 11, 0.28)",
                    line=dict(color="#F59E0B", width=2.5),
                    name="Fuselage Core (Flight Controller & Avionics)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=f"Core Avionics Cross-Section (Lethal: ≥{drone_spec.get('lethal_energy_core_j', 20):.0f}J)",
                )
            )

        # C. Battery / Munition Payload Projected Polygon
        batt_poly = aspect_geom.get("battery_polygon", [])
        if batt_poly and len(batt_poly) >= 3:
            bp_y = [pt[0] for pt in batt_poly] + [batt_poly[0][0]]
            bp_z = [pt[1] for pt in batt_poly] + [batt_poly[0][1]]
            fig.add_trace(
                go.Scatter(
                    x=bp_y,
                    y=bp_z,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.32)",
                    line=dict(color="#EF4444", width=2.0),
                    name="Battery / Munition Payload",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=f"Battery Pack / Munition Core (Puncture = Instant Detonation)",
                )
            )

        # D. Motor Pods
        motors = aspect_geom.get("motors", [])
        for m_idx, m in enumerate(motors):
            my = float(m["y_m"])
            mz = float(m["z_m"])
            mw = float(m["w_m"])
            mh = float(m["h_m"])
            m_box_y = [my - mw / 2, my + mw / 2, my + mw / 2, my - mw / 2, my - mw / 2]
            m_box_z = [mz - mh / 2, mz - mh / 2, mz + mh / 2, mz + mh / 2, mz - mh / 2]
            fig.add_trace(
                go.Scatter(
                    x=m_box_y,
                    y=m_box_z,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(168, 85, 247, 0.45)",
                    line=dict(color="#A855F7", width=1.5),
                    name="Motor Pod" if m_idx == 0 else None,
                    showlegend=bool(m_idx == 0),
                    hoverinfo="text",
                    hovertext=f"Motor {m.get('id', '')} Housing (Lethal: ≥{drone_spec.get('lethal_energy_rotor_j', 8):.0f}J)",
                )
            )

        # E. Rotor Discs (Projected Ellipses or Edge-on Blade Bands)
        rotors = aspect_geom.get("rotors", [])
        for r_idx, r in enumerate(rotors):
            ry = float(r["y_m"])
            rz = float(r["z_m"])
            a = float(r["semi_major_m"])
            b = float(r["semi_minor_m"])
            u_maj = r.get("u_maj", [1.0, 0.0])
            u_min = r.get("u_min", [0.0, 1.0])

            th_rot = np.linspace(0, 2 * np.pi, 40)
            rot_y = ry + a * np.cos(th_rot) * u_maj[0] + b * np.sin(th_rot) * u_min[0]
            rot_z = rz + a * np.cos(th_rot) * u_maj[1] + b * np.sin(th_rot) * u_min[1]

            fig.add_trace(
                go.Scatter(
                    x=rot_y,
                    y=rot_z,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(6, 182, 212, 0.12)",
                    line=dict(color="#06B6D4", width=2, dash="dash"),
                    name=f"Rotor {r.get('id', r_idx+1)}" if r_idx == 0 else None,
                    showlegend=bool(r_idx == 0),
                    hoverinfo="text",
                    hovertext=f"Rotor {r.get('id', r_idx+1)} (a={a*100:.1f}cm, b={b*100:.1f}cm | Lethal: ≥{drone_spec.get('lethal_energy_rotor_j', 8):.0f}J)",
                )
            )

        # F. Fixed-Wing Aerodynamic Panels
        wings = aspect_geom.get("wings", [])
        for w_idx, w_poly in enumerate(wings):
            if w_poly and len(w_poly) >= 3:
                wp_y = [pt[0] for pt in w_poly] + [w_poly[0][0]]
                wp_z = [pt[1] for pt in w_poly] + [w_poly[0][1]]
                fig.add_trace(
                    go.Scatter(
                        x=wp_y,
                        y=wp_z,
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(14, 165, 233, 0.20)",
                        line=dict(color="#0EA5E9", width=2.0),
                        name="Wing Panel" if w_idx == 0 else None,
                        showlegend=bool(w_idx == 0),
                        hoverinfo="text",
                        hovertext="Main Wing Aerodynamic Spar",
                    )
                )
    else:
        # Fallback Planform Silhouette
        drone_type = drone_spec.get("drone_type", "quadcopter")
        rotors = drone_spec.get("rotors", [])
        fuse_w = float(drone_spec.get("fuselage_w_m", 0.10))
        fuse_h = float(drone_spec.get("fuselage_h_m", 0.10))

        for r_idx, r in enumerate(rotors):
            ry = float(r["y_m"])
            rz = float(r["z_m"])
            r_rad = float(r["radius_m"])

            fig.add_trace(
                go.Scatter(
                    x=[0.0, ry],
                    y=[0.0, rz],
                    mode="lines",
                    line=dict(color="#334155", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=ry + r_rad * np.cos(th),
                    y=rz + r_rad * np.sin(th),
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(6, 182, 212, 0.12)",
                    line=dict(color="#06B6D4", width=2, dash="dash"),
                    name=f"Rotor {r.get('id', r_idx+1)}" if r_idx == 0 else None,
                    showlegend=bool(r_idx == 0),
                    hoverinfo="text",
                    hovertext=f"Rotor {r.get('id', r_idx+1)} (Lethal: ≥{drone_spec.get('lethal_energy_rotor_j', 8):.0f}J)",
                )
            )

        # Central Fuselage (Core Avionics / Battery)
        f_box_y = [-fuse_w / 2, fuse_w / 2, fuse_w / 2, -fuse_w / 2, -fuse_w / 2]
        f_box_z = [-fuse_h / 2, -fuse_h / 2, fuse_h / 2, fuse_h / 2, -fuse_h / 2]
        fig.add_trace(
            go.Scatter(
                x=f_box_y,
                y=f_box_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.25)",
                line=dict(color="#F59E0B", width=2.5),
                name="Fuselage Core (Flight Controller & Battery)",
                showlegend=True,
                hoverinfo="text",
                hovertext=f"Core Electronics (Lethal: ≥{drone_spec.get('lethal_energy_core_j', 20):.0f}J)",
            )
        )

    # 3. Barrel Nominal Boresight Aim Points on Target Plane
    for b in barrels:
        b_id = b["barrel_id"]
        y0 = float(b["y_m"])
        z0 = float(b["z_m"])
        yaw = float(b.get("aim_yaw_mrad", 0.0))
        pitch = float(b.get("aim_pitch_mrad", 0.0))
        yb = y0 + R * (yaw / 1000.0)
        zb = z0 + R * (pitch / 1000.0)

        fig.add_trace(
            go.Scatter(
                x=[yb],
                y=[zb],
                mode="markers+text",
                marker=dict(size=9, color="#94A3B8", symbol="cross", line=dict(color="#FFFFFF", width=1.5)),
                text=[b_id],
                textposition="top center",
                name=f"Boresight {b_id}",
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{b_id} Aim Point: ({yb:+.2f}m, {zb:+.2f}m)",
            )
        )

    # 4. Pellet Impacts
    if scatter and scatter.get("y"):
        sc_y = np.array(scatter["y"])
        sc_z = np.array(scatter["z"])
        sc_b = np.array(scatter["barrel_id"])
        sc_hit = np.array(scatter["is_hit"], dtype=bool)
        sc_lethal = np.array(scatter["is_lethal"], dtype=bool)
        sc_comp = np.array(scatter["component"])
        sc_ke = np.array(scatter["striking_energy_j"])

        miss_mask = ~sc_hit
        if np.any(miss_mask):
            fig.add_trace(
                go.Scatter(
                    x=sc_y[miss_mask],
                    y=sc_z[miss_mask],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="rgba(99, 102, 241, 0.40)",
                        symbol="circle",
                    ),
                    name="Miss Pellets",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[f"Pellet from {b}: ({y:.2f}m, {z:.2f}m) | {ke:.1f}J" for b, y, z, ke in zip(sc_b[miss_mask], sc_y[miss_mask], sc_z[miss_mask], sc_ke[miss_mask])],
                )
            )

        nl_hits = sc_hit & (~sc_lethal)
        if np.any(nl_hits):
            fig.add_trace(
                go.Scatter(
                    x=sc_y[nl_hits],
                    y=sc_z[nl_hits],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color="#FACC15",
                        symbol="triangle-up",
                        line=dict(color="#000000", width=1),
                    ),
                    name="Superficial Hits (Below Lethal Energy)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[f"HIT {comp}: {ke:.1f}J (Sub-lethal)" for comp, ke in zip(sc_comp[nl_hits], sc_ke[nl_hits])],
                )
            )

        l_hits = sc_hit & sc_lethal
        if np.any(l_hits):
            fig.add_trace(
                go.Scatter(
                    x=sc_y[l_hits],
                    y=sc_z[l_hits],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color="#EF4444",
                        symbol="star",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    name="💥 Lethal Hits (Component Disabled)",
                    showlegend=True,
                    hoverinfo="text",
                    hovertext=[f"💥 LETHAL KILL on {comp}! Energy: {ke:.1f}J" for comp, ke in zip(sc_comp[l_hits], sc_ke[l_hits])],
                )
            )

    pk = summary.get("kill_probability_pct", 0.0)
    pk_color = "#10B981" if pk >= 80.0 else ("#F59E0B" if pk >= 50.0 else "#EF4444")

    aspect_info_str = ""
    if aspect_geom:
        asp_n = aspect_geom.get("name", "Cross-Section")
        a_cm2 = aspect_geom.get("presented_area_cm2", 0.0)
        a_rat = aspect_geom.get("cross_section_ratio_pct", 100.0)
        aspect_info_str = f" | Aspect: <b>{asp_n}</b> | Cross-Section: <b>{a_cm2:.1f} cm²</b> ({a_rat:.1f}% planform)"

    fig.update_layout(
        title=(
            f"🎯 2D Anti-Drone Intercept Board & Silhouette Damage @ {R:.1f}m<br>"
            f"<sup>Target: {summary.get('drone_name', '')}{aspect_info_str} | "
            f"<span style='color:{pk_color};font-weight:bold;'>P_kill: {pk:.1f}%</span> (Mean Lethal Hits: {summary.get('mean_lethal_hits', 0.0):.1f})</sup>"
        ),
        xaxis=dict(title="Horizontal Crossrange Y (m)", range=[-1.2, 1.2], zeroline=True, zerolinecolor="#475569", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Vertical Height Z (m)", range=[-1.2, 1.2], zeroline=True, zerolinecolor="#475569"),
        margin=dict(l=30, r=30, t=65, b=30),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(17, 24, 39, 0.8)", font=dict(color="#F9FAFB", size=10)),
        height=580,
    )
    return fig


def create_antidrone_layout_comparison_figure(comp_res: dict[str, Any]) -> go.Figure:
    """Create diagnostic curves comparing different multi-shotgun array geometries across ranges."""
    curves = comp_res.get("layout_curves", {})
    if not curves:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "🎯 Drone Kill Probability (P_kill %) vs Engagement Distance",
            "💥 Mean Lethal Hits on Drone vs Engagement Distance",
        ],
    )

    palette = ["#2563EB", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899"]

    for idx, (layout_name, data) in enumerate(curves.items()):
        col = palette[idx % len(palette)]
        dists = data["distances_m"]
        pks = data["pk_pct"]
        lethals = data["mean_lethal"]

        fig.add_trace(
            go.Scatter(
                x=dists,
                y=pks,
                mode="lines+markers",
                name=f"{layout_name} P_kill",
                line=dict(color=col, width=3),
                marker=dict(size=7),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dists,
                y=lethals,
                mode="lines+markers",
                name=f"{layout_name} Lethal Hits",
                line=dict(color=col, width=2.5, dash="dash"),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Engagement Distance (m)", row=2, col=1)
    fig.update_yaxes(title_text="Kill Probability (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Mean Lethal Hits", row=2, col=1)

    fig.update_layout(
        title=f"Multi-Shotgun Array Architecture Benchmark — Target: {comp_res.get('drone_model', '').replace('_', ' ').title()}",
        margin=dict(l=25, r=25, t=55, b=25),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=10)),
        height=540,
    )
    return fig


def create_drone_aspect_radar_figure(aspect_bench_res: dict[str, Any]) -> go.Figure:
    """Create dual-panel benchmark figure showing P_kill and presented cross-section area across aspect angles."""
    results = aspect_bench_res.get("aspect_results", [])
    if not results:
        return go.Figure()

    names = [r["name"].split("(")[0].strip() for r in results]
    pk_vals = [r["kill_probability_pct"] for r in results]
    areas = [r["presented_area_cm2"] for r in results]
    ratios = [r["cross_section_ratio_pct"] for r in results]
    hits = [r["mean_hits"] for r in results]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            "🎯 Kill Probability P_kill (%)<br>by Aspect Angle",
            "📐 Presented Cross-Section (cm²)<br>& Planform Ratio (%)",
        ],
    )

    colors = ["#10B981" if p >= 80 else ("#F59E0B" if p >= 50 else "#EF4444") for p in pk_vals]

    # Panel 1: Bar chart of P_kill
    fig.add_trace(
        go.Bar(
            x=names,
            y=pk_vals,
            text=[f"{p:.1f}%" for p in pk_vals],
            textposition="auto",
            marker=dict(color=colors),
            name="P_kill (%)",
            hovertext=[f"P_kill: {p:.1f}%<br>Mean Hits: {h:.1f}" for p, h in zip(pk_vals, hits)],
        ),
        row=1,
        col=1,
    )

    # Panel 2: Bar chart of presented area
    fig.add_trace(
        go.Bar(
            x=names,
            y=areas,
            text=[f"{a:.0f} cm² ({r:.0f}%)" for a, r in zip(areas, ratios)],
            textposition="auto",
            marker=dict(color="#06B6D4"),
            name="Cross-Section Area (cm²)",
            hovertext=[f"Area: {a:.1f} cm²<br>{r:.1f}% of planform" for a, r in zip(areas, ratios)],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(tickangle=-25, row=1, col=1)
    fig.update_xaxes(tickangle=-25, row=1, col=2)
    fig.update_yaxes(title_text="Kill Probability (%)", range=[0, 110], row=1, col=1)
    fig.update_yaxes(title_text="Presented Area (cm²)", row=1, col=2)

    tgt_d = aspect_bench_res.get("target_distance_m", 35.0)
    d_model = aspect_bench_res.get("drone_model", "").replace("_", " ").title()
    fig.update_layout(
        title=f"🌐 Drone Presentation Aspect Angle Benchmark — Target: {d_model} @ {tgt_d:.0f}m",
        margin=dict(l=30, r=30, t=85, b=55),
        showlegend=False,
        height=490,
    )
    fig.update_annotations(font_size=12)
    return fig


