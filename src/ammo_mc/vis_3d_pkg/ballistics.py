"""Interactive 3D & 2D Ballistics, Tactical & C-UAS Plotly Visualization Package — ballistics submodule."""

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

def _compute_terrain_z(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    scenario: Scenario | None,
    terrain_style: Literal["auto", "rolling_hills", "valley", "ridge"] = "auto",
    x_max: float = 1000.0,
    y_span: float = 300.0,
) -> np.ndarray:
    """Compute synthetic or scenario-driven 3D terrain elevation Z(X, Y)."""
    Z_grid = np.zeros_like(X_grid, dtype=np.float64)

    if scenario is not None and scenario.terrain.mode == "profile" and scenario.terrain.profile:
        p0, p1 = scenario.terrain.profile[0], scenario.terrain.profile[-1]
        slope = (p1.elevation_m - p0.elevation_m) / max(p1.range_m - p0.range_m, 1.0)
        Z_grid = np.maximum(p0.elevation_m + slope * X_grid, 0.0)

    if terrain_style == "rolling_hills":
        hill_amp = max(x_max * 0.018, 12.0)
        Z_grid = Z_grid + hill_amp * (
            0.55 * np.sin(2.0 * np.pi * X_grid / max(x_max * 0.45, 100.0))
            * np.cos(np.pi * Y_grid / max(y_span * 0.85, 50.0))
            + 0.45 * np.sin(np.pi * X_grid / max(x_max * 0.8, 100.0))
        )
        Z_grid = np.maximum(Z_grid - Z_grid[0, Z_grid.shape[1] // 2], 0.0)
    elif terrain_style == "valley":
        wall_h = max(x_max * 0.03, 25.0)
        Z_grid = Z_grid + wall_h * ((Y_grid / max(y_span, 1.0)) ** 2) * np.clip(X_grid / max(x_max * 0.3, 1.0), 0.0, 1.0)
    elif terrain_style == "ridge":
        ridge_h = max(x_max * 0.025, 20.0)
        Z_grid = Z_grid + ridge_h * np.exp(-(((X_grid - 0.65 * x_max) / max(0.18 * x_max, 20.0)) ** 2))

    return Z_grid


def create_3d_trajectory_bundle_figure(
    result: SimulationResult,
    scenario: Scenario | None = None,
    color_by: Literal["speed", "mach", "altitude", "index"] = "speed",
    show_terrain: bool = True,
    show_target: bool = True,
    show_poi: bool = True,
    show_cep_rings: bool = True,
    terrain_style: Literal["auto", "rolling_hills", "valley", "ridge"] = "auto",
    max_trajectories: int = 15,
) -> go.Figure:
    """Build an interactive 3D Plotly figure rendering the bundle of Monte Carlo trajectory paths.

    Includes 3D terrain surface (flat, sloped profile, or procedural DEM), weapon firing position,
    target 3D bounding geometry, 3D CEP 50% & 95% ellipses on the ground, wind vector indicator,
    impact scatter points on the ground, and trajectory apex markers.
    """
    fig = go.Figure()

    trajs = result.sample_trajectories or ([result.sample_trajectory] if result.sample_trajectory else [])
    trajs = [t for t in trajs if t is not None][:max_trajectories]

    # 1. 3D Terrain Surface Mesh
    target_r = float(result.dispersion.get("mean_range_m", 1000.0))
    if scenario is not None and scenario.aim.range_m > 0:
        target_r = float(scenario.aim.range_m)

    x_max = max(target_r * 1.15, 500.0)
    y_span = max(float(result.dispersion.get("rep_m", 100.0)) * 5.0, 300.0)

    gx = np.linspace(0.0, x_max, 45)
    gy = np.linspace(-y_span, y_span, 35)
    X_grid, Y_grid = np.meshgrid(gx, gy, indexing="ij")
    Z_grid = _compute_terrain_z(X_grid, Y_grid, scenario, terrain_style, x_max, y_span)

    def _interp_ground_z(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        ix = np.clip(np.searchsorted(gx, x_arr) - 1, 0, len(gx) - 1)
        iy = np.clip(np.searchsorted(gy, y_arr) - 1, 0, len(gy) - 1)
        return Z_grid[ix, iy]

    if show_terrain:
        fig.add_trace(
            go.Surface(
                x=X_grid,
                y=Y_grid,
                z=Z_grid,
                colorscale="Earth" if terrain_style != "auto" else "YlGnBu",
                cmin=0.0,
                cmax=max(float(np.max(Z_grid)), 80.0),
                opacity=0.65,
                showscale=False,
                name="Terrain Surface",
                hoverinfo="skip",
            )
        )

    # 2. Weapon Launch Marker
    z0_weapon = float(Z_grid[0, len(gy) // 2])
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[z0_weapon],
            mode="markers+text",
            marker=dict(size=8, color="#1E3A8A", symbol="diamond"),
            text=["Weapon Origin"],
            textposition="top center",
            name="Weapon Launch Location",
        )
    )

    # Optional 3D Wind Indicator Arrow near origin
    if scenario is not None:
        wx = float(scenario.met.wind_range_mps)
        wy = float(scenario.met.wind_cross_right_mps)
        w_mag = math.hypot(wx, wy)
        if w_mag > 0.1:
            scale_w = min(max(x_max * 0.08, 40.0), 250.0) / max(w_mag, 1.0)
            w_z = z0_weapon + max(x_max * 0.04, 25.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, wx * scale_w],
                    y=[0.0, wy * scale_w],
                    z=[w_z, w_z],
                    mode="lines+markers+text",
                    line=dict(color="#0284C7", width=6),
                    marker=dict(size=[2, 6], color="#0284C7", symbol="diamond"),
                    text=["", f"Wind: {w_mag:.1f} m/s"],
                    textposition="top center",
                    name=f"Wind Vector ({wx:+.1f}, {wy:+.1f} m/s)",
                )
            )

    # 3. 3D Target Volume
    if show_target and scenario is not None:
        hx = float(scenario.target.geometry.get("half_width_x_m", 25.0))
        hy = float(scenario.target.geometry.get("half_width_y_m", 25.0))
        hz = float(scenario.target.geometry.get("height_m", 4.0))

        t_z0 = float(_interp_ground_z(np.array([target_r]), np.array([0.0]))[0])

        box_x = [
            target_r - hx, target_r + hx, target_r + hx, target_r - hx, target_r - hx,
            target_r - hx, target_r + hx, target_r + hx, target_r - hx, target_r - hx,
        ]
        box_y = [-hy, -hy, hy, hy, -hy, -hy, -hy, hy, hy, -hy]
        box_z = [t_z0, t_z0, t_z0, t_z0, t_z0, t_z0 + hz, t_z0 + hz, t_z0 + hz, t_z0 + hz, t_z0 + hz]

        fig.add_trace(
            go.Scatter3d(
                x=box_x,
                y=box_y,
                z=box_z,
                mode="lines",
                line=dict(color="#10B981", width=5),
                name=f"Target Volume ({scenario.target.posture})",
            )
        )

    # 4. 3D Trajectory Traces (Bundle)
    palette = [
        "#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#059669",
        "#0284C7", "#4F46E5", "#9333EA", "#C026D3", "#D97706",
        "#16A34A", "#0D9488", "#0891B2", "#6366F1", "#84CC16",
    ]

    for idx, tr in enumerate(trajs):
        rx = np.array(tr.get("range_m", []), dtype=np.float64)
        ry = np.array(tr.get("deflection_m", []), dtype=np.float64)
        rz = np.array(tr.get("alt_m", []), dtype=np.float64)
        speed = np.array(tr.get("speed_mps", []), dtype=np.float64)
        mach = np.array(tr.get("mach", []), dtype=np.float64)
        if len(mach) != len(rx) and len(speed) == len(rx):
            mach = speed / 340.29

        if len(rx) < 2:
            continue

        c_line: Any = palette[idx % len(palette)]
        c_bar = False
        if color_by == "speed" and len(speed) == len(rx):
            c_line = speed
            c_bar = idx == 0
        elif color_by == "mach" and len(mach) == len(rx):
            c_line = mach
            c_bar = idx == 0
        elif color_by == "altitude":
            c_line = rz
            c_bar = idx == 0

        line_dict: dict[str, Any] = dict(width=4)
        if isinstance(c_line, np.ndarray):
            line_dict["color"] = c_line
            line_dict["colorscale"] = "Turbo" if color_by in ("speed", "mach") else "Viridis"
            if c_bar:
                cbar_title = {"speed": "Speed (m/s)", "mach": "Mach (-)", "altitude": "Alt (m)"}.get(color_by, "")
                line_dict["colorbar"] = dict(title=cbar_title, x=1.02, len=0.7)
        else:
            line_dict["color"] = c_line

        fig.add_trace(
            go.Scatter3d(
                x=rx,
                y=ry,
                z=rz,
                mode="lines",
                line=line_dict,
                name=f"Trajectory #{idx + 1}",
                text=[
                    f"R: {x:.0f}m, D: {y:.1f}m, Z: {z:.0f}m, M: {m:.2f}"
                    for x, y, z, m in zip(rx, ry, rz, mach if len(mach) == len(rx) else np.zeros_like(rx))
                ],
                hoverinfo="text+name",
            )
        )

        apex_idx = int(np.argmax(rz))
        if idx == 0 and rz[apex_idx] > 20.0:
            fig.add_trace(
                go.Scatter3d(
                    x=[rx[apex_idx]],
                    y=[ry[apex_idx]],
                    z=[rz[apex_idx]],
                    mode="markers+text",
                    marker=dict(size=5, color="#DC2626", symbol="circle"),
                    text=[f"Apex: {rz[apex_idx]:.0f}m"],
                    textposition="top center",
                    name="Max Ordinate (Apex)",
                )
            )

    # 5. 3D Impact POI Scatter & 3D CEP Rings
    if show_poi and result.trials:
        t_df = result.trials
        p_x = np.array(t_df.get("range_m", []), dtype=np.float64)
        p_y = np.array(t_df.get("deflection_m", []), dtype=np.float64)
        is_dud = np.array(t_df.get("is_dud", []), dtype=bool)

        p_z = _interp_ground_z(p_x, p_y)
        colors = ["#DC2626" if d else "#2563EB" for d in is_dud]
        fig.add_trace(
            go.Scatter3d(
                x=p_x,
                y=p_y,
                z=p_z + 1.2,
                mode="markers",
                marker=dict(size=4, color=colors, opacity=0.78),
                name=f"Impact POI (N={len(p_x)})",
                text=[
                    f"Impact Range: {x:.1f}m, Defl: {y:.1f}m ({'Non-func' if d else 'Nominal'})"
                    for x, y, d in zip(p_x, p_y, is_dud)
                ],
                hoverinfo="text",
            )
        )

    if show_cep_rings:
        mpi_x = float(result.dispersion.get("mean_range_m", target_r))
        mpi_y = float(result.dispersion.get("mean_deflection_m", 0.0))
        cep50 = float(result.dispersion.get("cep_m", result.dispersion.get("cep50_m", 10.0)))
        cep95 = float(result.dispersion.get("cep95_m", cep50 * 2.08))
        theta_ring = np.linspace(0.0, 2.0 * np.pi, 72)

        for r_val, r_label, r_col, r_dash in [
            (cep50, f"CEP 50% ({cep50:.1f} m)", "#F59E0B", "solid"),
            (cep95, f"CEP 95% ({cep95:.1f} m)", "#EF4444", "dash"),
        ]:
            rx_ring = mpi_x + r_val * np.cos(theta_ring)
            ry_ring = mpi_y + r_val * np.sin(theta_ring)
            rz_ring = _interp_ground_z(rx_ring, ry_ring) + 1.8
            fig.add_trace(
                go.Scatter3d(
                    x=rx_ring,
                    y=ry_ring,
                    z=rz_ring,
                    mode="lines",
                    line=dict(color=r_col, width=5, dash=r_dash),
                    name=r_label,
                )
            )

        mpi_z = float(_interp_ground_z(np.array([mpi_x]), np.array([mpi_y]))[0]) + 2.0
        fig.add_trace(
            go.Scatter3d(
                x=[mpi_x],
                y=[mpi_y],
                z=[mpi_z],
                mode="markers+text",
                marker=dict(size=6, color="#F59E0B", symbol="cross"),
                text=["MPI"],
                textposition="top center",
                name="Mean Point of Impact (MPI)",
            )
        )

    fig.update_layout(
        title=f"3D Flight Simulation: {result.ammo_name} [{result.fidelity}]",
        scene=dict(
            xaxis=dict(title="Downrange X (m)", backgroundcolor="#F9FAFB"),
            yaxis=dict(title="Crossrange Deflection Y (m)", backgroundcolor="#F9FAFB"),
            zaxis=dict(title="Altitude Z (m)", backgroundcolor="#F9FAFB"),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.2, z=0.85),
            camera=dict(
                eye=dict(x=-1.75, y=-1.75, z=0.95),
                center=dict(x=0.0, y=0.0, z=-0.08),
            ),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_animated_flight_figure(
    result: SimulationResult,
    scenario: Scenario | None = None,
    max_trajectories: int = 10,
    n_frames: int = 24,
) -> go.Figure:
    """Create a time-stepped interactive 3D Plotly animation of the ballistic salvo in flight."""
    trajs = result.sample_trajectories or ([result.sample_trajectory] if result.sample_trajectory else [])
    trajs = [t for t in trajs if t is not None and len(t.get("range_m", [])) >= 2][:max_trajectories]

    target_r = float(result.dispersion.get("mean_range_m", 1000.0))
    if scenario is not None and scenario.aim.range_m > 0:
        target_r = float(scenario.aim.range_m)

    x_max = max(target_r * 1.15, 500.0)
    y_span = max(float(result.dispersion.get("rep_m", 100.0)) * 4.5, 250.0)
    z_max = 100.0
    max_tof = 1.0

    resampled: list[dict[str, np.ndarray]] = []
    u_grid = np.linspace(0.0, 1.0, n_frames)

    for tr in trajs:
        rx = np.array(tr.get("range_m", []), dtype=np.float64)
        ry = np.array(tr.get("deflection_m", []), dtype=np.float64)
        rz = np.array(tr.get("alt_m", []), dtype=np.float64)
        t_arr = np.array(tr.get("time_s", np.linspace(0.0, 10.0, len(rx))), dtype=np.float64)
        u_orig = np.linspace(0.0, 1.0, len(rx))
        z_max = max(z_max, float(np.max(rz)) * 1.15)
        max_tof = max(max_tof, float(t_arr[-1]) if len(t_arr) > 0 else 1.0)
        resampled.append(
            {
                "x": np.interp(u_grid, u_orig, rx),
                "y": np.interp(u_grid, u_orig, ry),
                "z": np.interp(u_grid, u_orig, rz),
                "t": np.interp(u_grid, u_orig, t_arr),
            }
        )

    gx = np.linspace(0.0, x_max, 30)
    gy = np.linspace(-y_span, y_span, 24)
    X_grid, Y_grid = np.meshgrid(gx, gy, indexing="ij")
    Z_grid = _compute_terrain_z(X_grid, Y_grid, scenario, "auto", x_max, y_span)

    fig = go.Figure()

    # Trace 0: Static terrain
    fig.add_trace(
        go.Surface(
            x=X_grid,
            y=Y_grid,
            z=Z_grid,
            colorscale="YlGnBu",
            opacity=0.55,
            showscale=False,
            name="Terrain",
            hoverinfo="skip",
        )
    )

    # Trace 1: Target marker
    fig.add_trace(
        go.Scatter3d(
            x=[target_r],
            y=[0.0],
            z=[float(Z_grid[-1, len(gy) // 2])],
            mode="markers+text",
            marker=dict(size=8, color="#10B981", symbol="diamond"),
            text=["Target Center"],
            textposition="top center",
            name="Target",
        )
    )

    # Trace 2: Dynamic projectile heads
    init_x = [float(r["x"][0]) for r in resampled]
    init_y = [float(r["y"][0]) for r in resampled]
    init_z = [float(r["z"][0]) for r in resampled]
    fig.add_trace(
        go.Scatter3d(
            x=init_x,
            y=init_y,
            z=init_z,
            mode="markers",
            marker=dict(size=6, color="#EF4444", symbol="circle"),
            name="Projectiles in Flight",
        )
    )

    # Trace 3: Dynamic flight smoke trails (concatenated with None separators)
    trail_x: list[float | None] = []
    trail_y: list[float | None] = []
    trail_z: list[float | None] = []
    for r in resampled:
        trail_x.extend([float(r["x"][0]), None])
        trail_y.extend([float(r["y"][0]), None])
        trail_z.extend([float(r["z"][0]), None])

    fig.add_trace(
        go.Scatter3d(
            x=trail_x,
            y=trail_y,
            z=trail_z,
            mode="lines",
            line=dict(color="#2563EB", width=4),
            name="Flight Trails",
        )
    )

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, Any]] = []

    for k in range(n_frames):
        fk_x = [float(r["x"][k]) for r in resampled]
        fk_y = [float(r["y"][k]) for r in resampled]
        fk_z = [float(r["z"][k]) for r in resampled]

        tk_x: list[float | None] = []
        tk_y: list[float | None] = []
        tk_z: list[float | None] = []
        for r in resampled:
            tk_x.extend(r["x"][: k + 1].tolist() + [None])
            tk_y.extend(r["y"][: k + 1].tolist() + [None])
            tk_z.extend(r["z"][: k + 1].tolist() + [None])

        t_sec = (k / max(n_frames - 1, 1)) * max_tof
        frame_name = f"frame_{k}"
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(x=fk_x, y=fk_y, z=fk_z),
                    go.Scatter3d(x=tk_x, y=tk_y, z=tk_z),
                ],
                traces=[2, 3],
                name=frame_name,
            )
        )
        slider_steps.append(
            {
                "args": [
                    [frame_name],
                    {"frame": {"duration": 80, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}},
                ],
                "label": f"{t_sec:.1f}s",
                "method": "animate",
            }
        )

    fig.frames = frames
    fig.update_layout(
        title=f"Animated 3D Salvo Flight Replay: {result.ammo_name} (TOF ~ {max_tof:.1f} s)",
        scene=dict(
            xaxis=dict(title="Downrange X (m)", range=[0.0, x_max]),
            yaxis=dict(title="Deflection Y (m)", range=[-y_span, y_span]),
            zaxis=dict(title="Altitude Z (m)", range=[0.0, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.2, z=0.85),
            camera=dict(eye=dict(x=-1.7, y=-1.7, z=0.95)),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.05,
                "buttons": [
                    {
                        "label": "▶ Play 3D Salvo",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 85, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"font": {"size": 13}, "prefix": "Flight Time: ", "visible": True},
                "pad": {"b": 10, "t": 30},
                "len": 0.82,
                "x": 0.16,
                "y": 0.05,
                "steps": slider_steps,
            }
        ],
        margin=dict(l=10, r=10, t=40, b=50),
    )
    return fig


def create_3d_projectile_geometry_figure(inp: McDragInputs) -> go.Figure:
    """Build an interactive 3D axisymmetric CAD surface mesh of a projectile from McDragInputs."""
    d_mm = float(inp.d_ref_mm)
    L_tot = float(inp.total_length_cal) * d_mm
    L_nose = float(inp.nose_length_cal) * d_mm
    L_bt = float(inp.boattail_length_cal) * d_mm
    r_ref = 0.5 * d_mm
    r_meplat = 0.5 * float(inp.meplat_dia_cal) * d_mm
    r_base = 0.5 * float(inp.base_dia_cal) * d_mm
    r_band = 0.5 * float(inp.band_dia_cal) * d_mm
    rt_r = float(np.clip(inp.rt_over_r, 0.0, 1.0))

    # Axial stations x from nose (x=0) to base (x=L_tot)
    x_nose = np.linspace(0.0, L_nose, 35)
    xi = x_nose / max(L_nose, 1e-6)
    # Blend between conical (rt_r=0) and tangent ogive profile (rt_r=1)
    r_cone = r_meplat + (r_ref - r_meplat) * xi
    r_ogive = r_meplat + (r_ref - r_meplat) * np.sin(0.5 * np.pi * xi)
    r_nose_prof = (1.0 - rt_r) * r_cone + rt_r * r_ogive

    L_cyl_end = max(L_tot - L_bt, L_nose)
    band_start = max(L_cyl_end - 0.35 * d_mm, L_nose)
    band_end = max(L_cyl_end - 0.12 * d_mm, band_start + 1e-3)

    x_cyl1 = np.linspace(L_nose, band_start, 15)
    r_cyl1 = np.full_like(x_cyl1, r_ref)

    x_band = np.linspace(band_start, band_end, 8)
    r_band_prof = np.full_like(x_band, max(r_band, r_ref))

    x_cyl2 = np.linspace(band_end, L_cyl_end, 8)
    r_cyl2 = np.full_like(x_cyl2, r_ref)

    if L_bt > 1e-3:
        x_bt = np.linspace(L_cyl_end, L_tot, 15)
        r_bt = r_ref + (r_base - r_ref) * ((x_bt - L_cyl_end) / L_bt)
        x_prof = np.concatenate([x_nose, x_cyl1, x_band, x_cyl2, x_bt])
        r_prof = np.concatenate([r_nose_prof, r_cyl1, r_band_prof, r_cyl2, r_bt])
    else:
        x_prof = np.concatenate([x_nose, x_cyl1, x_band, x_cyl2])
        r_prof = np.concatenate([r_nose_prof, r_cyl1, r_band_prof, r_cyl2])

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    X_mesh, TH = np.meshgrid(x_prof, theta, indexing="ij")
    R_mesh, _ = np.meshgrid(r_prof, theta, indexing="ij")
    Y_mesh = R_mesh * np.cos(TH)
    Z_mesh = R_mesh * np.sin(TH)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X_mesh,
            y=Y_mesh,
            z=Z_mesh,
            surfacecolor=R_mesh,
            colorscale="Cividis",
            showscale=False,
            opacity=0.92,
            name=inp.identification,
            hovertemplate="Axial X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<extra></extra>",
        )
    )

    # Centerline & CG marker
    xcg_mm = float(inp.xcg_nose_cal) * d_mm if inp.xcg_nose_cal > 0 else 0.58 * L_tot
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, L_tot],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="lines",
            line=dict(color="#EF4444", width=3, dash="dash"),
            name="Spin Axis",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[xcg_mm],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=7, color="#10B981", symbol="diamond"),
            text=[f"CG ({xcg_mm / d_mm:.2f} cal)"],
            textposition="top center",
            name="Center of Gravity (CG)",
        )
    )

    fig.update_layout(
        title=f"3D Axisymmetric Projectile CAD: {inp.identification} ({d_mm:.1f} mm, {inp.total_length_cal:.2f} cal)",
        scene=dict(
            xaxis=dict(title="Axial Station from Nose X (mm)"),
            yaxis=dict(title="Transverse Y (mm)"),
            zaxis=dict(title="Vertical Z (mm)"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.4, y=-1.8, z=1.1)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def create_3d_multi_preset_comparison_figure(
    runs: list[tuple[SimulationResult, Scenario]],
    show_terrain: bool = True,
) -> go.Figure:
    """Overlay multiple presets' 3D trajectory bundles and impact footprints in a single comparative 3D scene."""
    fig = go.Figure()
    palette = ["#2563EB", "#EA580C", "#10B981", "#9333EA", "#DC2626"]

    max_x = 500.0
    max_y = 250.0
    for res, scn in runs:
        max_x = max(max_x, float(res.dispersion.get("mean_range_m", scn.aim.range_m)) * 1.15)
        max_y = max(max_y, float(res.dispersion.get("rep_m", 50.0)) * 5.0)

    if show_terrain:
        gx = np.linspace(0.0, max_x, 35)
        gy = np.linspace(-max_y, max_y, 25)
        X_grid, Y_grid = np.meshgrid(gx, gy, indexing="ij")
        Z_grid = np.zeros_like(X_grid)
        fig.add_trace(
            go.Surface(
                x=X_grid,
                y=Y_grid,
                z=Z_grid,
                colorscale="YlGnBu",
                opacity=0.45,
                showscale=False,
                name="Reference Ground Plane",
                hoverinfo="skip",
            )
        )

    for idx, (res, scn) in enumerate(runs):
        col = palette[idx % len(palette)]
        label = f"{res.ammo_id} ({res.ammo_name})"
        trajs = res.sample_trajectories or ([res.sample_trajectory] if res.sample_trajectory else [])
        trajs = [t for t in trajs if t is not None][:6]

        for t_i, tr in enumerate(trajs):
            rx = np.array(tr.get("range_m", []), dtype=np.float64)
            ry = np.array(tr.get("deflection_m", []), dtype=np.float64)
            rz = np.array(tr.get("alt_m", []), dtype=np.float64)
            if len(rx) < 2:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=rx,
                    y=ry,
                    z=rz,
                    mode="lines",
                    line=dict(color=col, width=4),
                    name=label,
                    legendgroup=res.ammo_id,
                    showlegend=(t_i == 0),
                )
            )

        if res.trials:
            px = np.array(res.trials.get("range_m", []), dtype=np.float64)
            py = np.array(res.trials.get("deflection_m", []), dtype=np.float64)
            cep_val = float(res.dispersion.get("cep_m", res.dispersion.get("cep50_m", 0.0)))
            fig.add_trace(
                go.Scatter3d(
                    x=px,
                    y=py,
                    z=np.ones_like(px) * 1.5,
                    mode="markers",
                    marker=dict(size=3.5, color=col, opacity=0.7),
                    name=f"{res.ammo_id} Impacts (CEP50={cep_val:.1f}m)",
                    legendgroup=res.ammo_id,
                    showlegend=True,
                )
            )

    fig.update_layout(
        title="Multi-Preset 3D Trajectory & Dispersion Comparison Arena",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Deflection Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="manual",
            aspectratio=dict(x=3.0, y=1.2, z=0.85),
            camera=dict(eye=dict(x=-1.75, y=-1.75, z=0.95)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_blast_and_frag_figure(
    result: SimulationResult,
    scenario: Scenario | None = None,
    burst_alt_m: float = 0.0,
    n_frag_traces: int = 120,
    charge_mass_kg: float = 10.0,
    explosive_name: str = "Composition B",
) -> go.Figure:
    """Build an interactive 3D Plotly figure rendering the detonation event:

    - 3D Overpressure Iso-Surfaces (100 kPa, 35 kPa, 14 kPa).
    - 3D Monte Carlo fragmentation spray rays projected outward to ground impact.
    """
    fig = go.Figure()
    kb = get_kb()
    kb_e = kb.get_explosive_properties(explosive_name)

    p_targets = [
        {"p_kpa": 100.0, "color": "rgba(220, 38, 38, 0.35)", "name": "100 kPa (Severe Overpressure)"},
        {"p_kpa": 35.0, "color": "rgba(245, 158, 11, 0.25)", "name": "35 kPa (Structural Threshold)"},
        {"p_kpa": 14.0, "color": "rgba(59, 130, 246, 0.18)", "name": "14 kPa (Moderate Threshold)"},
    ]

    phi = np.linspace(0, np.pi, 25)
    theta = np.linspace(0, 2 * np.pi, 35)
    PHI, THETA = np.meshgrid(phi, theta)

    for pt in p_targets:
        target_p = pt["p_kpa"]
        r_low, r_high = 0.5, 200.0
        for _ in range(25):
            r_mid = 0.5 * (r_low + r_high)
            bf = evaluate_blast_field(r_mid, charge_mass_kg, explosive_name)
            p_val = float(bf["Ps_kPa"])
            if p_val > target_p:
                r_low = r_mid
            else:
                r_high = r_mid
        r_iso = 0.5 * (r_low + r_high)

        xs = r_iso * np.sin(PHI) * np.cos(THETA)
        ys = r_iso * np.sin(PHI) * np.sin(THETA)
        zs = np.maximum(burst_alt_m + r_iso * np.cos(PHI), 0.0)

        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                colorscale=[[0, pt["color"]], [1, pt["color"]]],
                opacity=0.45,
                showscale=False,
                name=f"{pt['name']} ({r_iso:.1f} m)",
                hoverinfo="name",
            )
        )

    m_case = 32.0
    v0_g = float(tf_01_gurney_cyl_sphere(kb_e["gurney_mps"], m_case, charge_mass_kg, "cylinder"))

    rng = np.random.default_rng(42)
    u_azim = rng.uniform(0, 2 * np.pi, n_frag_traces)
    u_elev = rng.uniform(-np.pi / 6.0, np.pi / 3.0, n_frag_traces)
    v_speeds = rng.normal(v0_g, v0_g * 0.12, n_frag_traces)

    for i in range(n_frag_traces):
        azim = u_azim[i]
        elev = u_elev[i]
        v_init = v_speeds[i]

        t_steps = np.linspace(0, 1.2, 12)
        beta = 0.008
        r_dist = (1.0 / beta) * np.log(1.0 + beta * v_init * t_steps)

        fx = r_dist * np.cos(elev) * np.cos(azim)
        fy = r_dist * np.cos(elev) * np.sin(azim)
        fz = np.maximum(burst_alt_m + r_dist * np.sin(elev) - 0.5 * 9.80665 * (t_steps**2), 0.0)

        hit_idx = np.where(fz <= 0.0)[0]
        if len(hit_idx) > 0 and hit_idx[0] > 0:
            k_end = hit_idx[0] + 1
            fx, fy, fz = fx[:k_end], fy[:k_end], fz[:k_end]

        ke_val = 0.5 * 0.005 * (v_init**2)
        c_frag = "#DC2626" if ke_val >= 79.0 else "#0284C7"

        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines",
                line=dict(color=c_frag, width=2),
                showlegend=(i == 0 or i == 1),
                name="High-Energy Fragment (>=79J)" if ke_val >= 79.0 else "Low-Energy Fragment",
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[burst_alt_m],
            mode="markers+text",
            marker=dict(size=8, color="#F59E0B", symbol="cross"),
            text=["Event Center"],
            textposition="top center",
            name="Burst Point",
        )
    )

    fig.update_layout(
        title=f"3D Pressure Wave & Fragment Kinematics ({charge_mass_kg:.1f} kg {explosive_name})",
        scene=dict(
            xaxis=dict(title="X Offset (m)"),
            yaxis=dict(title="Y Offset (m)"),
            zaxis=dict(title="Z Height (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def create_3d_phase_space_figure(
    result: SimulationResult,
    scenario: Scenario | None = None,
    color_metric: Literal["dynamic_pressure_kPa", "reynolds_millions", "specific_energy_kJ_kg"] = "dynamic_pressure_kPa",
    calibre_m: float = 0.155,
    max_trajectories: int = 12,
) -> go.Figure:
    """Build a 3D Aerodynamic & Thermodynamic Phase-Space Manifold (Range X vs Altitude Z vs Mach Number M).

    Includes a semi-transparent Sonic Barrier Plane (M = 1.0) and colors trajectories by either
    dynamic pressure q = 0.5*rho*v^2 (kPa), Reynolds number Re_d (millions), or specific mechanical
    energy e = 0.5*v^2 + g*z (kJ/kg).
    """
    fig = go.Figure()
    trajs = result.sample_trajectories or ([result.sample_trajectory] if result.sample_trajectory else [])
    trajs = [t for t in trajs if t is not None and len(t.get("range_m", [])) >= 2][:max_trajectories]

    dT_K = float(scenario.met.deviations.dT_K) if scenario is not None else 0.0
    max_x = 1000.0
    max_z = 200.0

    for idx, tr in enumerate(trajs):
        rx = np.array(tr.get("range_m", []), dtype=np.float64)
        rz = np.array(tr.get("alt_m", []), dtype=np.float64)
        v_arr = np.array(tr.get("speed_mps", []), dtype=np.float64)
        max_x = max(max_x, float(np.max(rx)))
        max_z = max(max_z, float(np.max(rz)))

        atm = eb_04_isa_atmosphere(rz, dT_K=dT_K)
        rho = np.asarray(atm["rho_kg_m3"], dtype=np.float64)
        a_snd = np.asarray(atm["a_mps"], dtype=np.float64)
        t_k = np.asarray(atm["T_K"], dtype=np.float64)
        mach = v_arr / np.maximum(a_snd, 1.0)

        q_kpa = 0.5 * rho * (v_arr**2) * 1e-3
        re_mil = np.asarray(aerodynamic_reynolds_number(rho, v_arr, calibre_m, t_k), dtype=np.float64) * 1e-6
        spec_e = (0.5 * (v_arr**2) + 9.80665 * rz) * 1e-3

        if color_metric == "dynamic_pressure_kPa":
            c_arr = q_kpa
            c_title = "q (kPa)"
        elif color_metric == "reynolds_millions":
            c_arr = re_mil
            c_title = "Re_d (10^6)"
        else:
            c_arr = spec_e
            c_title = "e_mech (kJ/kg)"

        line_cfg: dict[str, Any] = dict(width=5, color=c_arr, colorscale="Plasma")
        if idx == 0:
            line_cfg["colorbar"] = dict(title=c_title, x=1.02, len=0.75)

        fig.add_trace(
            go.Scatter3d(
                x=rx,
                y=rz,
                z=mach,
                mode="lines",
                line=line_cfg,
                name=f"State Trajectory #{idx + 1}",
                text=[
                    f"X: {x:.0f}m | Alt: {z:.0f}m | M: {m:.2f} | q: {q:.1f} kPa | Re: {re:.2f}M"
                    for x, z, m, q, re in zip(rx, rz, mach, q_kpa, re_mil)
                ],
                hoverinfo="text+name",
            )
        )

    # Sonic Barrier Plane (Mach = 1.0)
    sx = np.linspace(0.0, max_x * 1.05, 10)
    sz = np.linspace(0.0, max_z * 1.05, 10)
    SX, SZ = np.meshgrid(sx, sz)
    SM = np.ones_like(SX)
    fig.add_trace(
        go.Surface(
            x=SX,
            y=SZ,
            z=SM,
            colorscale=[[0, "rgba(239, 68, 68, 0.22)"], [1, "rgba(239, 68, 68, 0.22)"]],
            opacity=0.30,
            showscale=False,
            name="Sonic Barrier (Mach = 1.0)",
            hoverinfo="name",
        )
    )

    fig.update_layout(
        title=f"3D Aerodynamic & Thermodynamic Phase-Space Manifold: {result.ammo_name}",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Altitude Z (m)"),
            zaxis=dict(title="Mach Number M (-)"),
            aspectmode="manual",
            aspectratio=dict(x=2.4, y=1.3, z=1.1),
            camera=dict(eye=dict(x=-1.65, y=-1.65, z=1.05)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def export_3d_html(fig: go.Figure, file_path: str | Path, include_plotlyjs: str = "cdn") -> str:
    """Save the interactive 3D figure to a standalone HTML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return str(path)


