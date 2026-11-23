"""Interactive 3D & 2D Ballistics, Tactical & C-UAS Plotly Visualization Package — small_arms submodule."""

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

def create_3d_shotgun_plume_figure(sim_res: dict[str, Any]) -> go.Figure:
    """Create interactive 3D WebGL figure of shotgun expanding pellet plume, choke cone, and target plane."""
    fig = go.Figure()
    target_dist = float(sim_res.get("target_distance_m", 36.58))
    pellet_coords = sim_res.get("pellet_coordinates", {})
    y_pellets = pellet_coords.get("y_m", [])
    z_pellets = pellet_coords.get("z_m", [])
    v_pellets = pellet_coords.get("v_impact_mps", [])

    # 1. Gun Barrel & Muzzle Representation
    r_barrel = 0.015
    for p in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        fig.add_trace(
            go.Scatter3d(
                x=[-0.7, 0.0],
                y=[r_barrel * np.cos(p), r_barrel * np.cos(p)],
                z=[r_barrel * np.sin(p), r_barrel * np.sin(p)],
                mode="lines",
                line=dict(color="#4B5563", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    phi_b = np.linspace(0, 2 * np.pi, 25)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0] * len(phi_b),
            y=r_barrel * np.cos(phi_b),
            z=r_barrel * np.sin(phi_b),
            mode="lines",
            line=dict(color="#F59E0B", width=4),
            name="Shotgun Muzzle (X=0)",
        )
    )

    # 2. Conical Plume Boundary Envelope (30-in circle cone)
    r_30in = 0.381
    for a_idx, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, target_dist],
                y=[0.0, r_30in * np.cos(angle)],
                z=[0.0, r_30in * np.sin(angle)],
                mode="lines",
                line=dict(color="rgba(59, 130, 246, 0.35)", width=2, dash="dash"),
                showlegend=bool(a_idx == 0),
                name="30-in Pattern Cone Envelope",
            )
        )

    # 3. Sampled 3D Pellet Flight Streamlines (showing stringing and expansion)
    n_sample = min(40, len(y_pellets))
    if n_sample > 0:
        step = max(1, len(y_pellets) // n_sample)
        for i in range(0, len(y_pellets), step):
            yp = y_pellets[i]
            zp = z_pellets[i]
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, target_dist],
                    y=[0.0, yp],
                    z=[0.0, zp],
                    mode="lines",
                    line=dict(color="rgba(239, 68, 68, 0.40)", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # 4. Target Plane Surface (semi-transparent)
    span = max(0.65, r_30in * 1.6)
    plane_y = np.linspace(-span, span, 5)
    plane_z = np.linspace(-span, span, 5)
    Y_p, Z_p = np.meshgrid(plane_y, plane_z)
    X_p = np.full_like(Y_p, target_dist)
    fig.add_trace(
        go.Surface(
            x=X_p,
            y=Y_p,
            z=Z_p,
            opacity=0.18,
            colorscale=[[0, "#9CA3AF"], [1, "#9CA3AF"]],
            showscale=False,
            name="Target Plane",
        )
    )

    # 5. 30-inch and 10-inch Benchmark Rings on Target Plane
    phi_ring = np.linspace(0, 2 * np.pi, 50)
    fig.add_trace(
        go.Scatter3d(
            x=[target_dist] * len(phi_ring),
            y=r_30in * np.cos(phi_ring),
            z=r_30in * np.sin(phi_ring),
            mode="lines",
            line=dict(color="#1D4ED8", width=4),
            name="30-in Benchmark Ring (0.762m)",
        )
    )
    r_10in = 0.127
    fig.add_trace(
        go.Scatter3d(
            x=[target_dist] * len(phi_ring),
            y=r_10in * np.cos(phi_ring),
            z=r_10in * np.sin(phi_ring),
            mode="lines",
            line=dict(color="#DC2626", width=3.5, dash="dot"),
            name="10-in Core Ring (0.254m)",
        )
    )

    # 6. Target Silhouette on Target Plane
    tgt_type = sim_res.get("target_type", "30_inch_circle")
    if tgt_type == "e_type_silhouette":
        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter3d(
                x=[target_dist] * len(e_y),
                y=e_y,
                z=e_z,
                mode="lines",
                line=dict(color="#10B981", width=3.5),
                name="E-Type Humanoid Target",
            )
        )
    elif tgt_type == "clay_pigeon":
        c_phi = np.linspace(0, 2 * np.pi, 30)
        fig.add_trace(
            go.Scatter3d(
                x=[target_dist] * len(c_phi),
                y=0.055 * np.cos(c_phi),
                z=0.025 * np.sin(c_phi),
                mode="lines",
                line=dict(color="#F59E0B", width=3.5),
                name="Clay Pigeon Target",
            )
        )

    # 7. Pellet Impact Scatter Points on Target Plane
    if len(y_pellets) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=[target_dist] * len(y_pellets),
                y=y_pellets,
                z=z_pellets,
                mode="markers",
                marker=dict(
                    size=4.5,
                    color=v_pellets if v_pellets else "#DC2626",
                    colorscale="Turbo",
                    colorbar=dict(title="V_impact (m/s)", len=0.6, x=1.02),
                    showscale=True,
                ),
                text=[
                    f"Pellet #{i+1}<br>V: {v:.1f} m/s<br>Y: {y:.3f}m, Z: {z:.3f}m"
                    for i, (y, z, v) in enumerate(zip(y_pellets, z_pellets, v_pellets))
                ],
                hoverinfo="text",
                name="Pellet Impacts",
            )
        )

    fig.update_layout(
        title=f"3D Shotgun Pellet Plume & Shot Stringing — Choke: {sim_res.get('choke_name', 'Modified')} | Range: {target_dist:.1f}m | 30-in Hit: {sim_res.get('hits_30in_pct', 0):.1f}%",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Vertical Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_2d_target_pattern_figure(sim_res: dict[str, Any]) -> go.Figure:
    """Create 2D target board pattern figure with concentric rings, target silhouette, and density contours."""
    fig = go.Figure()
    target_dist = float(sim_res.get("target_distance_m", 36.58))
    pellet_coords = sim_res.get("pellet_coordinates", {})
    y_pellets = np.array(pellet_coords.get("y_m", []))
    z_pellets = np.array(pellet_coords.get("z_m", []))
    v_pellets = np.array(pellet_coords.get("v_impact_mps", []))
    e_pellets = np.array(pellet_coords.get("energy_j", []))

    # 1. Benchmark 30-inch and 10-inch Rings
    r_30in = 0.381
    r_10in = 0.127
    phi_c = np.linspace(0, 2 * np.pi, 100)

    # 30-in circle
    fig.add_trace(
        go.Scatter(
            x=r_30in * np.cos(phi_c),
            y=r_30in * np.sin(phi_c),
            mode="lines",
            line=dict(color="#1D4ED8", width=3),
            name="30-inch Circle (Benchmark: 40yd)",
            hoverinfo="name",
        )
    )
    # 10-in circle
    fig.add_trace(
        go.Scatter(
            x=r_10in * np.cos(phi_c),
            y=r_10in * np.sin(phi_c),
            mode="lines",
            line=dict(color="#DC2626", width=2.5, dash="dash"),
            name="10-inch Core Ring (Vital)",
            hoverinfo="name",
        )
    )

    # 2. Target Silhouette Outline
    tgt_type = sim_res.get("target_type", "30_inch_circle")
    if tgt_type == "e_type_silhouette":
        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter(
                x=e_y,
                y=e_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.12)",
                line=dict(color="#10B981", width=2.5),
                name="E-Type Humanoid Silhouette",
            )
        )
    elif tgt_type == "clay_pigeon":
        fig.add_trace(
            go.Scatter(
                x=0.055 * np.cos(phi_c),
                y=0.025 * np.sin(phi_c),
                mode="lines",
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.20)",
                line=dict(color="#F59E0B", width=2.5),
                name="Clay Pigeon (11cm)",
            )
        )
    elif tgt_type == "drone":
        d_y = [-0.20, 0.20, 0.20, -0.20, -0.20]
        d_z = [-0.20, -0.20, 0.20, 0.20, -0.20]
        fig.add_trace(
            go.Scatter(
                x=d_y,
                y=d_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(139, 92, 246, 0.15)",
                line=dict(color="#8B5CF6", width=2.5),
                name="Micro-UAV Drone (40x40cm)",
            )
        )

    # 3. Pellet Impacts Scatter
    if len(y_pellets) > 0:
        fig.add_trace(
            go.Scatter(
                x=y_pellets,
                y=z_pellets,
                mode="markers",
                marker=dict(
                    size=7,
                    color=e_pellets if len(e_pellets) > 0 else "#DC2626",
                    colorscale="Plasma",
                    colorbar=dict(title="Energy (J)", len=0.7, x=1.02),
                    showscale=True,
                    line=dict(width=0.5, color="#FFFFFF"),
                ),
                text=[
                    f"Pellet #{i+1}<br>V: {v:.1f} m/s<br>Energy: {e:.1f} J<br>Y: {y:.3f}m, Z: {z:.3f}m"
                    for i, (y, z, v, e) in enumerate(zip(y_pellets, z_pellets, v_pellets, e_pellets))
                ],
                hoverinfo="text",
                name="Pellet Impacts",
            )
        )

    # 4. Center of Mass & CEP-50 Circle
    com_y = float(sim_res.get("center_of_mass_y_m", 0.0))
    com_z = float(sim_res.get("center_of_mass_z_m", 0.0))
    cep_50 = float(sim_res.get("cep_50_m", 0.25))

    fig.add_trace(
        go.Scatter(
            x=[com_y],
            y=[com_z],
            mode="markers+text",
            marker=dict(size=14, color="#EF4444", symbol="cross"),
            text=["Center of Mass"],
            textposition="top right",
            name=f"Pattern COM ({com_y:+.3f}m, {com_z:+.3f}m)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=com_y + cep_50 * np.cos(phi_c),
            y=com_z + cep_50 * np.sin(phi_c),
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dashdot"),
            name=f"CEP-50 ({cep_50:.3f}m)",
            hoverinfo="name",
        )
    )

    max_extent = max(0.65, r_30in * 1.4)
    fig.update_layout(
        title=f"2D Target Patterning Board — {sim_res.get('choke_name', 'Modified')} Choke @ {target_dist:.1f}m | 30-in: {sim_res.get('hits_30in_pct', 0):.1f}% | 10-in: {sim_res.get('hits_10in_pct', 0):.1f}%",
        xaxis=dict(title="Horizontal Crossrange Y (m)", range=[-max_extent, max_extent]),
        yaxis=dict(title="Vertical Height Z (m)", range=[-max_extent, max_extent], scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=10)),
        height=540,
    )
    return fig


def create_3d_multibarrel_dispersion_figure(sim_res: dict[str, Any]) -> go.Figure:
    """Create 3D WebGL figure of multi-barrel / Gatling convergence geometry and burst dispersion."""
    fig = go.Figure()
    target_dist = float(sim_res.get("target_range_m", 35.0))
    crossover_dist = float(sim_res.get("crossover_range_m", 35.0))
    n_b = int(sim_res.get("n_barrels", 2))
    muzzle_positions = sim_res.get("barrel_muzzle_positions", [])

    palette = [
        "#DC2626",
        "#2563EB",
        "#10B981",
        "#F59E0B",
        "#8B5CF6",
        "#EC4899",
        "#14B8A6",
        "#6366F1",
    ]

    # 1. Barrels Muzzles and Bore Centerlines
    for b_idx, (y0, z0) in enumerate(muzzle_positions):
        b_col = palette[b_idx % len(palette)]
        # Barrel tube back to -0.6m
        fig.add_trace(
            go.Scatter3d(
                x=[-0.6, 0.0],
                y=[y0, y0],
                z=[z0, z0],
                mode="lines",
                line=dict(color="#374151", width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Muzzle marker
        fig.add_trace(
            go.Scatter3d(
                x=[0.0],
                y=[y0],
                z=[z0],
                mode="markers+text",
                marker=dict(size=6, color=b_col, symbol="circle"),
                text=[f"B{b_idx+1}"],
                textposition="top center",
                name=f"Barrel #{b_idx+1} Muzzle",
            )
        )
        # Bore centerline extending through crossover to target
        # Bore trajectory line: y(x) = y0 * (1 - x / crossover_dist)
        x_pts = [0.0, crossover_dist, target_dist]
        y_pts = [y0, 0.0, y0 * (1.0 - target_dist / max(crossover_dist, 1.0))]
        z_pts = [z0, 0.0, z0 * (1.0 - target_dist / max(crossover_dist, 1.0))]
        fig.add_trace(
            go.Scatter3d(
                x=x_pts,
                y=y_pts,
                z=z_pts,
                mode="lines",
                line=dict(color=b_col, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 2. Regulation Crossover Point
    fig.add_trace(
        go.Scatter3d(
            x=[crossover_dist],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=8, color="#F59E0B", symbol="diamond"),
            text=[f"Regulation Crossover ({crossover_dist:.0f}m)"],
            textposition="top center",
            name="Regulation Crossover Point",
        )
    )

    # 3. Target Plane Mesh
    span_y = max(0.5, max([abs(p[0]) for p in muzzle_positions], default=0.1) * 3.0)
    span_z = max(0.5, max([abs(p[1]) for p in muzzle_positions], default=0.1) * 3.0)
    plane_y = np.linspace(-span_y, span_y, 5)
    plane_z = np.linspace(-span_z, span_z, 5)
    Y_p, Z_p = np.meshgrid(plane_y, plane_z)
    X_p = np.full_like(Y_p, target_dist)
    fig.add_trace(
        go.Surface(
            x=X_p,
            y=Y_p,
            z=Z_p,
            opacity=0.15,
            colorscale=[[0, "#6B7280"], [1, "#6B7280"]],
            showscale=False,
            name="Target Plane",
        )
    )

    # 4. Target Silhouette on Target Plane
    tgt_type = sim_res.get("target_type", "e_type_silhouette")
    if tgt_type == "e_type_silhouette":
        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter3d(
                x=[target_dist] * len(e_y),
                y=e_y,
                z=e_z,
                mode="lines",
                line=dict(color="#10B981", width=3.5),
                name="E-Type Silhouette Target",
            )
        )
    elif tgt_type == "drone":
        drone_spec = sim_res.get("drone_spec")
        if drone_spec is None:
            from ammo_mc.small_arms import get_drone_target_specs
            drone_spec = get_drone_target_specs().get("commercial_quad_medium", {})
        aspect_geom = sim_res.get("aspect_geometry")
        M_rot = (
            np.array(aspect_geom["rotation_matrix"], dtype=float)
            if aspect_geom and "rotation_matrix" in aspect_geom
            else np.eye(3)
        )
        R = target_dist

        # Central fuselage 3D Box Wireframe
        l_f = float(drone_spec.get("fuselage_l_m", 0.20))
        w_f = float(drone_spec.get("fuselage_w_m", 0.12))
        h_f = float(drone_spec.get("fuselage_h_m", 0.08))
        box_corners_body = np.array([
            [-l_f / 2, -w_f / 2, -h_f / 2],
            [l_f / 2, -w_f / 2, -h_f / 2],
            [l_f / 2, w_f / 2, -h_f / 2],
            [-l_f / 2, w_f / 2, -h_f / 2],
            [-l_f / 2, -w_f / 2, h_f / 2],
            [l_f / 2, -w_f / 2, h_f / 2],
            [l_f / 2, w_f / 2, h_f / 2],
            [-l_f / 2, w_f / 2, h_f / 2],
        ])
        box_corners_world = np.array([R, 0.0, 0.0]) + (box_corners_body @ M_rot.T)
        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        bx: list[float | None] = []
        by: list[float | None] = []
        bz: list[float | None] = []
        for e0, e1 in box_edges:
            bx.extend([float(box_corners_world[e0, 0]), float(box_corners_world[e1, 0]), None])
            by.extend([float(box_corners_world[e0, 1]), float(box_corners_world[e1, 1]), None])
            bz.extend([float(box_corners_world[e0, 2]), float(box_corners_world[e1, 2]), None])

        fig.add_trace(
            go.Scatter3d(
                x=bx,
                y=by,
                z=bz,
                mode="lines",
                line=dict(color="#F59E0B", width=4),
                name=f"Drone Fuselage ({drone_spec.get('name', 'Drone')})",
                hoverinfo="text",
                hovertext=f"Presented Cross-Section: {sim_res.get('presented_area_cm2', 0):.1f} cm² ({sim_res.get('cross_section_ratio_pct', 100):.1f}% planform)",
            )
        )

        # Rotors and structural arms
        rotors = drone_spec.get("rotors", [])
        for r in rotors:
            rx_b = float(r.get("x_m", r.get("z_m", 0.0)))
            ry_b = float(r.get("y_m", 0.0))
            rz_b = float(r.get("z_m_vert", 0.015))
            r_rad = float(r.get("radius_m", 0.10))

            hub_world = np.array([R, 0.0, 0.0]) + M_rot @ np.array([rx_b, ry_b, rz_b])
            c_world = np.array([R, 0.0, 0.0]) + M_rot @ np.array([0.0, 0.0, 0.0])

            fig.add_trace(
                go.Scatter3d(
                    x=[float(c_world[0]), float(hub_world[0])],
                    y=[float(c_world[1]), float(hub_world[1])],
                    z=[float(c_world[2]), float(hub_world[2])],
                    mode="lines",
                    line=dict(color="#06B6D4", width=5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            th_rot = np.linspace(0, 2 * np.pi, 24)
            pts_body = np.column_stack([
                rx_b + r_rad * np.cos(th_rot),
                ry_b + r_rad * np.sin(th_rot),
                np.full_like(th_rot, rz_b),
            ])
            pts_world = np.array([R, 0.0, 0.0]) + (pts_body @ M_rot.T)

            fig.add_trace(
                go.Scatter3d(
                    x=[float(v) for v in pts_world[:, 0]],
                    y=[float(v) for v in pts_world[:, 1]],
                    z=[float(v) for v in pts_world[:, 2]],
                    mode="lines",
                    line=dict(color="#38BDF8", width=3),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"Rotor {r.get('id', '')} (r={r_rad*100:.0f}cm)",
                )
            )
    elif tgt_type == "30_inch_circle":
        th_c = np.linspace(0, 2 * np.pi, 60)
        r_c = 0.381
        fig.add_trace(
            go.Scatter3d(
                x=[target_dist] * len(th_c),
                y=[float(v) for v in (r_c * np.cos(th_c))],
                z=[float(v) for v in (r_c * np.sin(th_c))],
                mode="lines",
                line=dict(color="#60A5FA", width=3, dash="dash"),
                name="30-inch Benchmark Circle",
            )
        )

    # 5. Round Impacts Grouped by Barrel
    all_imp = sim_res.get("all_impacts", {})
    y_imp = all_imp.get("y_m", [])
    z_imp = all_imp.get("z_m", [])
    b_indices = all_imp.get("barrel_indices", [])

    for b_idx in range(n_b):
        b_col = palette[b_idx % len(palette)]
        mask = [i == (b_idx + 1) for i in b_indices]
        by = [y for y, m in zip(y_imp, mask) if m]
        bz = [z for z, m in zip(z_imp, mask) if m]
        if by:
            fig.add_trace(
                go.Scatter3d(
                    x=[target_dist] * len(by),
                    y=by,
                    z=bz,
                    mode="markers",
                    marker=dict(size=5, color=b_col, symbol="circle"),
                    name=f"B#{b_idx+1} Cluster ({len(by)} rds)",
                )
            )

    fig.update_layout(
        title=f"3D Multi-Barrel Volley & Dispersion — System: {sim_res.get('system_type', '').upper()} | Status: {sim_res.get('crossover_status', '')} | Salvo CEP: {sim_res.get('salvo_cep_m', 0.0):.3f}m",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Vertical Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_multi_distance_shotgun_figure(sweep_res: dict[str, Any]) -> go.Figure:
    """Create interactive 3D WebGL figure displaying shotgun pellet dispersion across multiple distance planes."""
    fig = go.Figure()
    snapshots = sweep_res.get("station_snapshots", {})
    if not snapshots:
        return fig

    max_dist = max([float(s["target_distance_m"]) for s in snapshots.values()], default=50.0)

    # 1. Gun Barrel Muzzle Representation
    r_barrel = 0.015
    for p in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        fig.add_trace(
            go.Scatter3d(
                x=[-0.7, 0.0],
                y=[r_barrel * np.cos(p), r_barrel * np.cos(p)],
                z=[r_barrel * np.sin(p), r_barrel * np.sin(p)],
                mode="lines",
                line=dict(color="#4B5563", width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    phi_b = np.linspace(0, 2 * np.pi, 25)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0] * len(phi_b),
            y=r_barrel * np.cos(phi_b),
            z=r_barrel * np.sin(phi_b),
            mode="lines",
            line=dict(color="#F59E0B", width=4),
            name="Shotgun Muzzle (X=0)",
        )
    )

    # 2. Conical Plume Boundary Envelope extending to max_dist
    r_30in = 0.381
    for a_idx, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, max_dist],
                y=[0.0, r_30in * (max_dist / 36.58) * np.cos(angle)],
                z=[0.0, r_30in * (max_dist / 36.58) * np.sin(angle)],
                mode="lines",
                line=dict(color="rgba(59, 130, 246, 0.25)", width=2, dash="dash"),
                showlegend=bool(a_idx == 0),
                name="Dispersion Cone Envelope",
            )
        )

    # 3. Planar Slices & Pellet Scatter at each distance station
    phi_ring = np.linspace(0, 2 * np.pi, 40)
    station_colors = ["#2563EB", "#10B981", "#F59E0B", "#DC2626", "#8B5CF6"]

    for st_idx, (st_name, s_data) in enumerate(snapshots.items()):
        d_val = float(s_data.get("target_distance_m", 30.0))
        st_col = station_colors[st_idx % len(station_colors)]

        # Target plane surface (semi-transparent mesh)
        span = max(0.45, r_30in * 1.3)
        plane_y = np.linspace(-span, span, 4)
        plane_z = np.linspace(-span, span, 4)
        Y_p, Z_p = np.meshgrid(plane_y, plane_z)
        X_p = np.full_like(Y_p, d_val)
        fig.add_trace(
            go.Surface(
                x=X_p,
                y=Y_p,
                z=Z_p,
                opacity=0.12,
                colorscale=[[0, st_col], [1, st_col]],
                showscale=False,
                hoverinfo="skip",
                name=f"Plane R={d_val:.0f}m",
            )
        )

        # 30-inch circle ring at this station
        fig.add_trace(
            go.Scatter3d(
                x=[d_val] * len(phi_ring),
                y=r_30in * np.cos(phi_ring),
                z=r_30in * np.sin(phi_ring),
                mode="lines",
                line=dict(color=st_col, width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Station text billboard
        fig.add_trace(
            go.Scatter3d(
                x=[d_val],
                y=[0.0],
                z=[span + 0.08],
                mode="text",
                text=[f"R={d_val:.0f}m ({s_data.get('hits_30in_pct', 0.0):.0f}% 30-in)"],
                textposition="top center",
                showlegend=False,
            )
        )

        # Pellet scatter points at this station
        p_coords = s_data.get("pellet_coordinates", {})
        y_p = p_coords.get("y_m", [])
        z_p = p_coords.get("z_m", [])

        if y_p:
            fig.add_trace(
                go.Scatter3d(
                    x=[d_val] * len(y_p),
                    y=y_p,
                    z=z_p,
                    mode="markers",
                    marker=dict(size=4, color=st_col, symbol="circle"),
                    name=f"Station {d_val:.0f}m ({len(y_p)} pellets)",
                )
            )

    fig.update_layout(
        title=f"3D Multi-Distance Shotgun Dispersion Plume — Choke: {sweep_res.get('choke_name', 'Modified')} across {len(snapshots)} Distance Slices",
        scene=dict(
            xaxis=dict(title="Downrange Distance X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Vertical Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_2d_multi_distance_shotgun_comparison_figure(sweep_res: dict[str, Any]) -> go.Figure:
    """Create side-by-side 2D subplots comparing target pattern boards at multiple distances."""
    snapshots = sweep_res.get("station_snapshots", {})
    if not snapshots:
        return go.Figure()

    station_items = list(snapshots.items())[:4]
    n_cols = len(station_items)
    titles = [
        f"R = {s_data.get('target_distance_m', 0.0):.0f}m | 30-in: {s_data.get('hits_30in_pct', 0.0):.1f}% | E: {s_data.get('mean_pellet_energy_j', 0.0):.0f}J"
        for _, s_data in station_items
    ]

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.04)

    r_30in = 0.381
    r_10in = 0.127
    phi_c = np.linspace(0, 2 * np.pi, 80)

    for col_idx, (_, s_data) in enumerate(station_items, start=1):
        # 30-inch circle
        fig.add_trace(
            go.Scatter(
                x=r_30in * np.cos(phi_c),
                y=r_30in * np.sin(phi_c),
                mode="lines",
                line=dict(color="#1D4ED8", width=2.5),
                showlegend=bool(col_idx == 1),
                name="30-in Circle",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )
        # 10-inch circle
        fig.add_trace(
            go.Scatter(
                x=r_10in * np.cos(phi_c),
                y=r_10in * np.sin(phi_c),
                mode="lines",
                line=dict(color="#DC2626", width=2, dash="dash"),
                showlegend=bool(col_idx == 1),
                name="10-in Core",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )

        # Target silhouette (E-type)
        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter(
                x=e_y,
                y=e_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.08)",
                line=dict(color="#10B981", width=1.5),
                showlegend=bool(col_idx == 1),
                name="E-Type Silhouette",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )

        # Pellet scatter points
        p_coords = s_data.get("pellet_coordinates", {})
        y_p = p_coords.get("y_m", [])
        z_p = p_coords.get("z_m", [])
        e_p = p_coords.get("energy_j", [])

        if y_p:
            fig.add_trace(
                go.Scatter(
                    x=y_p,
                    y=z_p,
                    mode="markers",
                    marker=dict(
                        size=5.5,
                        color=e_p if e_p else "#DC2626",
                        colorscale="Plasma",
                        showscale=bool(col_idx == n_cols),
                        colorbar=dict(title="Energy (J)", len=0.7, x=1.02) if col_idx == n_cols else None,
                        line=dict(width=0.4, color="#FFFFFF"),
                    ),
                    showlegend=bool(col_idx == 1),
                    name="Pellet Impacts",
                    text=[f"Y: {y:.3f}m, Z: {z:.3f}m<br>Energy: {e:.1f}J" for y, z, e in zip(y_p, z_p, e_p)],
                    hoverinfo="text",
                ),
                row=1,
                col=col_idx,
            )

        # Center of mass crosshair
        com_y = float(s_data.get("center_of_mass_y_m", 0.0))
        com_z = float(s_data.get("center_of_mass_z_m", 0.0))
        fig.add_trace(
            go.Scatter(
                x=[com_y],
                y=[com_z],
                mode="markers",
                marker=dict(size=10, color="#EF4444", symbol="cross"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(title_text="Crossrange Y (m)", range=[-0.55, 0.55], row=1, col=col_idx)
        fig.update_yaxes(title_text="Vertical Z (m)" if col_idx == 1 else "", range=[-0.55, 0.55], scaleanchor=f"x{col_idx if col_idx > 1 else ''}", scaleratio=1, row=1, col=col_idx)

    fig.update_layout(
        title=f"2D Target Pattern Comparison Across Multiple Distances — Choke: {sweep_res.get('choke_name', 'Modified')}",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=9)),
        height=420,
    )
    return fig


def create_range_curves_shotgun_figure(sweep_res: dict[str, Any]) -> go.Figure:
    """Create 2-row diagnostic curves showing pattern spread, hit containment, and energy vs distance."""
    records = sweep_res.get("range_records", [])
    if not records:
        return go.Figure()

    dists = [r["distance_m"] for r in records]
    p_30in = [r["hits_30in_pct"] for r in records]
    p_10in = [r["hits_10in_pct"] for r in records]
    p_tgt = [r["hits_target_pct"] for r in records]
    p_diams = [r["pattern_diameter_90_m"] for r in records]
    energies = [r["mean_pellet_energy_j"] for r in records]
    string_lens = [r["shot_string_length_m"] for r in records]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "🎯 Pattern Containment & Hit Probabilities vs Range",
            "⚡ Striking Energy & Pattern Dispersion Diameter vs Range",
        ],
    )

    # Row 1: Hit % curves
    fig.add_trace(
        go.Scatter(x=dists, y=p_30in, mode="lines+markers", name="30-inch Circle Hit %", line=dict(color="#2563EB", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dists, y=p_10in, mode="lines+markers", name="10-inch Core Ring Hit %", line=dict(color="#DC2626", width=2.5, dash="dash")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dists, y=p_tgt, mode="lines+markers", name="Target Silhouette Hit %", line=dict(color="#10B981", width=2.5)),
        row=1,
        col=1,
    )

    # Row 2: Energy & Spread
    fig.add_trace(
        go.Scatter(x=dists, y=energies, mode="lines+markers", name="Striking Energy (J/pellet)", line=dict(color="#F59E0B", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dists, y=p_diams, mode="lines+markers", name="Pattern Diameter R90 (m)", line=dict(color="#8B5CF6", width=2.5, dash="dot")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dists, y=string_lens, mode="lines+markers", name="Shot String Length (m)", line=dict(color="#EC4899", width=2.0, dash="dashdot")),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Target Distance (m)", row=2, col=1)
    fig.update_yaxes(title_text="Hit Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (J) / Spread (m)", row=2, col=1)

    fig.update_layout(
        title=f"Shotgun Dispersion Diagnostic Profiles vs Range — Choke: {sweep_res.get('choke_name', 'Modified')}",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
        height=520,
    )
    return fig


def create_3d_multi_distance_multibarrel_figure(sweep_res: dict[str, Any]) -> go.Figure:
    """Create 3D WebGL figure illustrating multi-barrel convergence, crossover, and divergence across distance stations."""
    fig = go.Figure()
    snapshots = sweep_res.get("station_snapshots", {})
    if not snapshots:
        return fig

    crossover_dist = float(sweep_res.get("crossover_range_m", 400.0))
    palette = ["#DC2626", "#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#6366F1"]

    # 1. Regulation Crossover Point Marker
    fig.add_trace(
        go.Scatter3d(
            x=[crossover_dist],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=9, color="#F59E0B", symbol="diamond"),
            text=[f"Regulation Crossover ({crossover_dist:.0f}m)"],
            textposition="top center",
            name="Regulation Crossover Point",
        )
    )

    # 2. Multi-Station Slices & Barrel Clusters
    first_snap = next(iter(snapshots.values()))
    muzzle_positions = first_snap.get("barrel_muzzle_positions", [])
    n_b = len(muzzle_positions)

    # Draw bore centerlines through crossover
    max_d = max([float(s["target_range_m"]) for s in snapshots.values()], default=crossover_dist * 1.5)
    for b_idx, (y0, z0) in enumerate(muzzle_positions):
        b_col = palette[b_idx % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=[-0.6, 0.0],
                y=[y0, y0],
                z=[z0, z0],
                mode="lines",
                line=dict(color="#374151", width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        x_pts = [0.0, crossover_dist, max_d]
        y_pts = [y0, 0.0, y0 * (1.0 - max_d / max(crossover_dist, 1.0))]
        z_pts = [z0, 0.0, z0 * (1.0 - max_d / max(crossover_dist, 1.0))]
        fig.add_trace(
            go.Scatter3d(
                x=x_pts,
                y=y_pts,
                z=z_pts,
                mode="lines",
                line=dict(color=b_col, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Station planar slices
    for st_name, s_data in snapshots.items():
        dist = float(s_data.get("target_range_m", 100.0))
        st_status = s_data.get("crossover_status", "")

        span_y = max(0.6, max([abs(p[0]) for p in muzzle_positions], default=0.1) * (1.0 + dist / max(crossover_dist, 1.0)) * 2.0)
        span_z = max(0.6, max([abs(p[1]) for p in muzzle_positions], default=0.1) * (1.0 + dist / max(crossover_dist, 1.0)) * 2.0)
        plane_y = np.linspace(-span_y, span_y, 4)
        plane_z = np.linspace(-span_z, span_z, 4)
        Y_p, Z_p = np.meshgrid(plane_y, plane_z)
        X_p = np.full_like(Y_p, dist)
        fig.add_trace(
            go.Surface(
                x=X_p,
                y=Y_p,
                z=Z_p,
                opacity=0.10,
                colorscale=[[0, "#9CA3AF"], [1, "#9CA3AF"]],
                showscale=False,
                hoverinfo="skip",
                name=f"Plane {dist:.0f}m",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[dist],
                y=[0.0],
                z=[span_z + 0.1],
                mode="text",
                text=[f"R={dist:.0f}m<br>({st_status})"],
                textposition="top center",
                showlegend=False,
            )
        )

        all_imp = s_data.get("all_impacts", {})
        y_imp = all_imp.get("y_m", [])
        z_imp = all_imp.get("z_m", [])
        b_indices = all_imp.get("barrel_indices", [])

        for b_idx in range(n_b):
            b_col = palette[b_idx % len(palette)]
            mask = [i == (b_idx + 1) for i in b_indices]
            by = [y for y, m in zip(y_imp, mask) if m]
            bz = [z for z, m in zip(z_imp, mask) if m]
            if by:
                fig.add_trace(
                    go.Scatter3d(
                        x=[dist] * len(by),
                        y=by,
                        z=bz,
                        mode="markers",
                        marker=dict(size=4.5, color=b_col, symbol="circle"),
                        name=f"B#{b_idx+1} @ {dist:.0f}m",
                        showlegend=bool(dist == min([float(s["target_range_m"]) for s in snapshots.values()])),
                    )
                )

    fig.update_layout(
        title=f"3D Multi-Distance Multi-Barrel Geometry — Crossover: {crossover_dist:.0f}m across {len(snapshots)} Distance Slices",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Vertical Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_range_curves_multibarrel_figure(sweep_res: dict[str, Any]) -> go.Figure:
    """Create diagnostic range curves for multi-barrel systems showing bore separation and salvo CEP vs distance."""
    records = sweep_res.get("range_records", [])
    if not records:
        return go.Figure()

    dists = [r["distance_m"] for r in records]
    spacings_mm = [r["inter_barrel_spacing_m"] * 1000.0 for r in records]
    ceps = [r["salvo_cep_m"] for r in records]
    hits_pct = [r["salvo_hit_pct"] for r in records]
    crossover_m = float(sweep_res.get("crossover_range_m", 400.0))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "📐 Theoretical Inter-Barrel Bore Separation vs Range (Crossover Convergence)",
            "🎯 Salvo Combined CEP & Target Hit Rate vs Range",
        ],
    )

    fig.add_trace(
        go.Scatter(x=dists, y=spacings_mm, mode="lines+markers", name="Bore Center Separation (mm)", line=dict(color="#2563EB", width=3)),
        row=1,
        col=1,
    )
    fig.add_vline(x=crossover_m, line_width=2, line_dash="dash", line_color="#F59E0B", annotation_text=f"Crossover Focus ({crossover_m:.0f}m)", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=dists, y=ceps, mode="lines+markers", name="Salvo Combined CEP (m)", line=dict(color="#DC2626", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dists, y=hits_pct, mode="lines+markers", name="Target Hit Rate (%)", line=dict(color="#10B981", width=2.5, dash="dash")),
        row=2,
        col=1,
    )
    fig.add_vline(x=crossover_m, line_width=2, line_dash="dash", line_color="#F59E0B", row=2, col=1)

    fig.update_xaxes(title_text="Engagement Range (m)", row=2, col=1)
    fig.update_yaxes(title_text="Separation (mm)", row=1, col=1)
    fig.update_yaxes(title_text="CEP (m) / Hit %", row=2, col=1)

    fig.update_layout(
        title=f"Multi-Barrel Convergence & Volley Dispersion Profiles vs Range — Crossover: {crossover_m:.0f}m",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
        height=520,
    )
    return fig


def create_2d_barrel_length_comparison_figure(comp_res: dict[str, Any]) -> go.Figure:
    """Create side-by-side 2D target pattern boards comparing multiple barrel lengths at a fixed distance."""
    snapshots = comp_res.get("barrel_snapshots", {})
    if not snapshots:
        return go.Figure()

    barrel_items = list(snapshots.items())[:5]
    n_cols = len(barrel_items)
    titles = [
        f"L = {b_key} | V₀: {s_data.get('v0_mps', 0.0):.0f}m/s | 30-in: {s_data.get('hits_30in_pct', 0.0):.1f}%"
        for b_key, s_data in barrel_items
    ]

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.035)

    r_30in = 0.381
    r_10in = 0.127
    phi_c = np.linspace(0, 2 * np.pi, 80)

    for col_idx, (_, s_data) in enumerate(barrel_items, start=1):
        fig.add_trace(
            go.Scatter(
                x=r_30in * np.cos(phi_c),
                y=r_30in * np.sin(phi_c),
                mode="lines",
                line=dict(color="#1D4ED8", width=2.5),
                showlegend=bool(col_idx == 1),
                name="30-in Circle",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=r_10in * np.cos(phi_c),
                y=r_10in * np.sin(phi_c),
                mode="lines",
                line=dict(color="#DC2626", width=2, dash="dash"),
                showlegend=bool(col_idx == 1),
                name="10-in Core Ring",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )

        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter(
                x=e_y,
                y=e_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.08)",
                line=dict(color="#10B981", width=1.5),
                showlegend=bool(col_idx == 1),
                name="E-Type Silhouette",
                hoverinfo="name",
            ),
            row=1,
            col=col_idx,
        )

        p_coords = s_data.get("pellet_coordinates", {})
        y_p = p_coords.get("y_m", [])
        z_p = p_coords.get("z_m", [])
        e_p = p_coords.get("energy_j", [])

        if y_p:
            fig.add_trace(
                go.Scatter(
                    x=y_p,
                    y=z_p,
                    mode="markers",
                    marker=dict(
                        size=5.5,
                        color=e_p if e_p else "#DC2626",
                        colorscale="Viridis",
                        showscale=bool(col_idx == n_cols),
                        colorbar=dict(title="Energy (J)", len=0.7, x=1.02) if col_idx == n_cols else None,
                        line=dict(width=0.4, color="#FFFFFF"),
                    ),
                    showlegend=bool(col_idx == 1),
                    name="Pellet Impacts",
                    text=[f"Y: {y:.3f}m, Z: {z:.3f}m<br>Energy: {e:.1f}J" for y, z, e in zip(y_p, z_p, e_p)],
                    hoverinfo="text",
                ),
                row=1,
                col=col_idx,
            )

        com_y = float(s_data.get("center_of_mass_y_m", 0.0))
        com_z = float(s_data.get("center_of_mass_z_m", 0.0))
        fig.add_trace(
            go.Scatter(
                x=[com_y],
                y=[com_z],
                mode="markers",
                marker=dict(size=10, color="#EF4444", symbol="cross"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(title_text="Crossrange Y (m)", range=[-0.55, 0.55], row=1, col=col_idx)
        fig.update_yaxes(title_text="Vertical Z (m)" if col_idx == 1 else "", range=[-0.55, 0.55], scaleanchor=f"x{col_idx if col_idx > 1 else ''}", scaleratio=1, row=1, col=col_idx)

    fig.update_layout(
        title=f"2D Target Pattern Comparison Across Multiple Barrel Lengths @ {comp_res.get('target_distance_m', 36.58):.1f}m — Choke: {comp_res.get('choke_name', 'Modified')}",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=9)),
        height=420,
    )
    return fig


def create_barrel_length_diagnostic_curves(comp_res: dict[str, Any]) -> go.Figure:
    """Create diagnostic curves showing Muzzle Velocity, Pattern Containment %, and Energy vs Barrel Length."""
    records = comp_res.get("barrel_records", [])
    if not records:
        return go.Figure()

    lengths = [r["length_in"] for r in records]
    v0s = [r["v0_mps"] for r in records]
    h30s = [r["hits_30in_pct"] for r in records]
    h10s = [r["hits_10in_pct"] for r in records]
    energies = [r["mean_pellet_energy_j"] for r in records]
    washes = [r["gas_wash_factor"] for r in records]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "🚀 Muzzle Velocity V₀ & Muzzle Gas Wash Turbulence vs Barrel Length",
            "🎯 Pattern Containment & Striking Energy vs Barrel Length",
        ],
    )

    fig.add_trace(
        go.Scatter(x=lengths, y=v0s, mode="lines+markers", name="Muzzle Velocity V₀ (m/s)", line=dict(color="#2563EB", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=lengths, y=[w * 350.0 for w in washes], mode="lines+markers", name="Gas Wash Turbulence Scale", line=dict(color="#DC2626", width=2.5, dash="dash")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=lengths, y=h30s, mode="lines+markers", name="30-inch Hit %", line=dict(color="#10B981", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=lengths, y=h10s, mode="lines+markers", name="10-inch Core Hit %", line=dict(color="#F59E0B", width=2.5, dash="dash")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=lengths, y=energies, mode="lines+markers", name="Striking Energy (J/pellet)", line=dict(color="#8B5CF6", width=2.0, dash="dot")),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Barrel Length (inches)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Hit % / Energy (J)", row=2, col=1)

    fig.update_layout(
        title=f"Barrel Length Performance Scaling @ {comp_res.get('target_distance_m', 36.58):.1f}m — Choke: {comp_res.get('choke_name', 'Modified')}",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
        height=520,
    )
    return fig


def create_2d_multibarrel_target_figure(sim_res: dict[str, Any]) -> go.Figure:
    """Create 2D target board figure showing multi-barrel salvo impacts against the target silhouette."""
    fig = go.Figure()
    target_dist = float(sim_res.get("target_range_m", 35.0))
    crossover_dist = float(sim_res.get("crossover_range_m", 35.0))
    tgt_type = sim_res.get("target_type", "e_type_silhouette")
    n_b = int(sim_res.get("n_barrels", 2))
    all_imp = sim_res.get("all_impacts", {})
    y_imp = all_imp.get("y_m", [])
    z_imp = all_imp.get("z_m", [])
    b_indices = all_imp.get("barrel_indices", [])

    palette = [
        "#DC2626",
        "#2563EB",
        "#10B981",
        "#F59E0B",
        "#8B5CF6",
        "#EC4899",
        "#14B8A6",
        "#6366F1",
    ]

    # 1. 30-inch and 10-inch Benchmark Rings
    th = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(
        go.Scatter(
            x=[float(v) for v in (0.381 * np.cos(th))],
            y=[float(v) for v in (0.381 * np.sin(th))],
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5, dash="dash"),
            name="30-inch Circle (0.76m diam)",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(v) for v in (0.127 * np.cos(th))],
            y=[float(v) for v in (0.127 * np.sin(th))],
            mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.4)", width=1.5, dash="dot"),
            name="10-inch Core (0.25m diam)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # 2. Target Silhouette / Drone Geometry
    if tgt_type == "drone":
        aspect_geom = sim_res.get("aspect_geometry")
        drone_spec = sim_res.get("drone_spec")
        if aspect_geom:
            # A. Structural Arms
            arms = aspect_geom.get("arms", [])
            for arm in arms:
                p1 = arm.get("p1", [0.0, 0.0])
                p2 = arm.get("p2", [0.0, 0.0])
                fig.add_trace(
                    go.Scatter(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        mode="lines",
                        line=dict(color="#475569", width=5),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # B. Fuselage Core
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
                        hovertext=f"Presented Area: {sim_res.get('presented_area_cm2', 0):.1f} cm² ({sim_res.get('cross_section_ratio_pct', 100):.1f}% planform)",
                    )
                )

            # C. Battery / Payload
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
                        name="Battery / Payload Munition",
                        showlegend=True,
                        hoverinfo="text",
                        hovertext="Critical Battery / Detonator Core",
                    )
                )

            # D. Rotors
            rotors = aspect_geom.get("rotors", [])
            for r in rotors:
                cx = float(r.get("center_y", 0.0))
                cz = float(r.get("center_z", 0.0))
                sma = float(r.get("semi_major_m", 0.10))
                smi = float(r.get("semi_minor_m", 0.04))
                ang = float(r.get("orientation_angle_rad", 0.0))
                th_rot = np.linspace(0, 2 * np.pi, 48)
                rot_x = sma * np.cos(th_rot)
                rot_z = smi * np.sin(th_rot)
                x_el = cx + rot_x * np.cos(ang) - rot_z * np.sin(ang)
                z_el = cz + rot_x * np.sin(ang) + rot_z * np.cos(ang)
                fig.add_trace(
                    go.Scatter(
                        x=[float(v) for v in x_el],
                        y=[float(v) for v in z_el],
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(6, 182, 212, 0.15)",
                        line=dict(color="#06B6D4", width=2),
                        name="Rotor Blade Disc",
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"Rotor {r.get('id', '')}",
                    )
                )

            # E. Motors
            motors = aspect_geom.get("motors", [])
            for m in motors:
                fig.add_trace(
                    go.Scatter(
                        x=[float(m.get("center_y", 0.0))],
                        y=[float(m.get("center_z", 0.0))],
                        mode="markers",
                        marker=dict(size=8, color="#0284C7", symbol="circle"),
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"Motor {m.get('id', '')}",
                    )
                )
    elif tgt_type == "e_type_silhouette":
        e_y = [-0.225, 0.225, 0.225, 0.11, 0.11, -0.11, -0.11, -0.225, -0.225]
        e_z = [-0.45, -0.45, 0.20, 0.20, 0.45, 0.45, 0.20, 0.20, -0.45]
        fig.add_trace(
            go.Scatter(
                x=e_y,
                y=e_z,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.12)",
                line=dict(color="#10B981", width=2.5),
                name="E-Type Silhouette Target",
            )
        )

    # 3. Round Impacts Grouped by Barrel
    for b_idx in range(n_b):
        b_col = palette[b_idx % len(palette)]
        mask = [i == (b_idx + 1) for i in b_indices]
        by = [y for y, m in zip(y_imp, mask) if m]
        bz = [z for z, m in zip(z_imp, mask) if m]
        if by:
            fig.add_trace(
                go.Scatter(
                    x=by,
                    y=bz,
                    mode="markers",
                    marker=dict(size=8, color=b_col, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                    name=f"Barrel #{b_idx+1} ({len(by)} rds)",
                )
            )

    # 4. Salvo CEP Circle
    if len(y_imp) > 0:
        salvo_cep = float(sim_res.get("salvo_cep_m", 0.0))
        mean_y = float(np.mean(y_imp))
        mean_z = float(np.mean(z_imp))
        fig.add_trace(
            go.Scatter(
                x=[float(v) for v in (mean_y + salvo_cep * np.cos(th))],
                y=[float(v) for v in (mean_z + salvo_cep * np.sin(th))],
                mode="lines",
                line=dict(color="#E11D48", width=2, dash="dashdot"),
                name=f"Salvo CEP ({salvo_cep:.3f}m)",
                hoverinfo="text",
                hovertext=f"Salvo Combined CEP: {salvo_cep:.3f}m around centroid ({mean_y:.3f}, {mean_z:.3f})",
            )
        )

    # Subtitle info
    title_text = (
        f"🎯 2D Multi-Barrel Target Intercept Pattern — {sim_res.get('system_type', '').upper()} @ {target_dist:.0f}m"
        f"<br><sup>Status: {sim_res.get('crossover_status', '')} | Salvo Hit Rate: {sim_res.get('salvo_hit_pct', 0):.1f}% | CEP: {sim_res.get('salvo_cep_m', 0):.3f}m"
    )
    if tgt_type == "drone" and sim_res.get("presented_area_cm2"):
        asp_label = (sim_res.get("aspect_mode") or "head_on_frontal").replace("_", " ").title()
        title_text += f" | Aspect: {asp_label} ({sim_res.get('presented_area_cm2', 0):.0f} cm², {sim_res.get('cross_section_ratio_pct', 100):.0f}% planform)"
    title_text += "</sup>"

    fig.update_layout(
        title=title_text,
        xaxis=dict(title="Crossrange Y (m)", scaleanchor="y", scaleratio=1, zeroline=True, gridcolor="rgba(100, 116, 139, 0.25)"),
        yaxis=dict(title="Vertical Z (m)", zeroline=True, gridcolor="rgba(100, 116, 139, 0.25)"),
        margin=dict(l=40, r=20, t=75, b=40),
        legend=dict(x=1.02, y=0.98, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=10)),
        height=540,
    )
    return fig


def create_multibarrel_aspect_radar_figure(bench_res: dict[str, Any]) -> go.Figure:
    """Create dual-panel comparative visualization of multi-barrel salvo hit effectiveness across 6 aspect orientations."""
    table = bench_res.get("comparison_table", [])
    if not table:
        return go.Figure()

    labels = [r["label"] for r in table]
    hit_pcts = [float(r["salvo_hit_pct"]) for r in table]
    hits_count = [int(r["total_hits"]) for r in table]
    lethal_hits = [int(r.get("total_lethal_hits", 0)) for r in table]
    areas = [float(r["presented_area_cm2"]) for r in table]
    ratios = [float(r["cross_section_ratio_pct"]) for r in table]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            "🎯 Multi-Barrel Salvo Hit Rate (%)<br>by Presentation Aspect",
            "📐 Drone Presented Cross-Section (cm²)<br>& Planform Ratio (%)",
        ],
    )

    colors_hit = ["#10B981" if p >= 70 else ("#F59E0B" if p >= 40 else "#EF4444") for p in hit_pcts]

    # Panel 1: Salvo Hit Rate %
    fig.add_trace(
        go.Bar(
            x=labels,
            y=hit_pcts,
            text=[f"{p:.1f}%" for p in hit_pcts],
            textposition="auto",
            marker=dict(color=colors_hit),
            name="Salvo Hit Rate (%)",
            hovertext=[f"Hit Rate: {p:.1f}%<br>Total Hits: {h}<br>Lethal Hits: {l}" for p, h, l in zip(hit_pcts, hits_count, lethal_hits)],
        ),
        row=1,
        col=1,
    )

    # Panel 2: Presented Cross-Section Area (cm²)
    fig.add_trace(
        go.Bar(
            x=labels,
            y=areas,
            text=[f"{a:.0f} cm² ({r:.0f}%)" for a, r in zip(areas, ratios)],
            textposition="auto",
            marker=dict(color="#06B6D4"),
            name="Presented Area (cm²)",
            hovertext=[f"Presented Area: {a:.1f} cm²<br>{r:.1f}% of planform" for a, r in zip(areas, ratios)],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(tickangle=-25, row=1, col=1)
    fig.update_xaxes(tickangle=-25, row=1, col=2)
    fig.update_yaxes(title_text="Salvo Hit Rate (%)", range=[0, 110], row=1, col=1)
    fig.update_yaxes(title_text="Presented Area (cm²)", row=1, col=2)

    tgt_d = float(bench_res.get("target_range_m", 400.0))
    d_model = str(bench_res.get("drone_model", "")).replace("_", " ").title()
    sys_type = str(bench_res.get("system_type", "")).upper()

    fig.update_layout(
        title=f"🌐 Multi-Barrel 3D Aspect Angle Benchmark — System: {sys_type} vs {d_model} @ {tgt_d:.0f}m",
        margin=dict(l=30, r=30, t=85, b=55),
        showlegend=False,
        height=490,
    )
    fig.update_annotations(font_size=12)
    return fig


