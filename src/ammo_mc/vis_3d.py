"""3D Visualization and Simulation module for trajectory bundles, terrain, animated flight,
3D axisymmetric projectile geometry, multi-preset comparisons, and detonation fields (Plotly 3D).
"""

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


def create_3d_iatg_safety_zones_figure(
    neq_kg: float = 1000.0,
    auw_kg: float = 2500.0,
    explosive_name: str = "Composition B",
    show_domes: bool = True,
) -> go.Figure:
    """Build a 3D UNODA IATG 01.80 / 02.20 Potential Explosion Site (PES) & Quantity-Distance (QD) Safety Dome Map.

    Visualizes:
    - Central Potential Explosion Site (PES) storehouse / bunker at (0, 0, 0)
    - IATG 01.80 / 02.20 Quantity-Distance (QD) translucent 3D hemispherical safety domes & ground rings:
      * PBD (Z = 8.0 m/kg^(1/3)): Process Building Distance
      * PTRD (Z = 14.8 m/kg^(1/3)): Public Traffic Route Distance
      * IBD (Z = 22.2 m/kg^(1/3)): Inhabited Building Distance
      * VBD (Z = 44.4 m/kg^(1/3)): Vulnerable Building Distance (School / Hospital)
    - TB-10 UK Brick-House Structural Collapse rings (Jarrett Categories A, B, CB)
    - Representative Exposed Sites (ES) with standoff R, scaled distance Z, and peak incident overpressure Ps.
    """
    fig = go.Figure()

    wp_eq, _ = tb_02_tnt_equivalence(neq_kg, explosive=explosive_name)
    q_tnt = max(float(wp_eq), 0.1)
    qd = tb_15_simple_safety_distances(auw_kg=auw_kg, neq_kg=q_tnt)

    r_pbd = float(qd["process_building_distance_m"])
    r_ptrd = float(qd["public_traffic_route_distance_m"])
    r_ibd = float(qd["inhabited_building_distance_m"])
    r_vbd = float(qd["vulnerable_building_distance_m"])

    # Ground surface plane
    max_r = max(r_vbd * 1.12, 100.0)
    gx = np.linspace(-max_r, max_r, 30)
    gy = np.linspace(-max_r, max_r, 30)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    fig.add_trace(
        go.Surface(
            x=GX,
            y=GY,
            z=GZ,
            colorscale=[[0, "#E5E7EB"], [1, "#D1D5DB"]],
            opacity=0.35,
            showscale=False,
            name="Site Ground Plane",
            hoverinfo="skip",
        )
    )

    # Central PES Storehouse Marker
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[max(max_r * 0.015, 2.0)],
            mode="markers+text",
            marker=dict(size=10, color="#991B1B", symbol="diamond"),
            text=[f"PES ({neq_kg:.0f} kg {explosive_name} / {q_tnt:.0f} kg TNT-eq)"],
            textposition="top center",
            name="Potential Explosion Site (PES)",
        )
    )

    qd_zones = [
        ("PBD: Process Building (Z=8.0)", r_pbd, 8.0, "#DC2626", "rgba(220, 38, 38, 0.16)"),
        ("PTRD: Public Traffic Route (Z=14.8)", r_ptrd, 14.8, "#EA580C", "rgba(234, 88, 12, 0.13)"),
        ("IBD: Inhabited Building (Z=22.2)", r_ibd, 22.2, "#2563EB", "rgba(37, 99, 235, 0.11)"),
        ("VBD: Vulnerable Building (Z=44.4)", r_vbd, 44.4, "#10B981", "rgba(16, 185, 129, 0.08)"),
    ]

    theta_ring = np.linspace(0.0, 2.0 * np.pi, 100)
    u_dome = np.linspace(0.0, 2.0 * np.pi, 32)
    v_dome = np.linspace(0.0, 0.5 * np.pi, 16)

    for label, r_val, z_val, hex_col, rgba_col in qd_zones:
        bf = evaluate_blast_field(r_val, q_tnt, explosive="TNT")
        ps_val = float(bf["Ps_kPa"])

        # Ground QD Perimeter Ring
        fig.add_trace(
            go.Scatter3d(
                x=r_val * np.cos(theta_ring),
                y=r_val * np.sin(theta_ring),
                z=np.full_like(theta_ring, 0.5),
                mode="lines",
                line=dict(color=hex_col, width=5),
                name=f"{label}: {r_val:.1f} m ({ps_val:.1f} kPa)",
            )
        )

        if show_domes:
            DX = r_val * np.outer(np.cos(u_dome), np.sin(v_dome))
            DY = r_val * np.outer(np.sin(u_dome), np.sin(v_dome))
            DZ = r_val * np.outer(np.ones_like(u_dome), np.cos(v_dome))
            fig.add_trace(
                go.Surface(
                    x=DX,
                    y=DY,
                    z=DZ,
                    colorscale=[[0, rgba_col], [1, rgba_col]],
                    opacity=0.14,
                    showscale=False,
                    name=f"{label} 3D Safety Dome",
                    hoverinfo="name",
                )
            )

    # TB-10 Jarrett Brick-House Structural Collapse Rings on Ground
    for cat_code, cat_desc, col_cat in [
        ("A", "TB-10 Cat A (Complete Demolition)", "#7F1D1D"),
        ("B", "TB-10 Cat B (Uninhabitable / >50% Collapse)", "#B91C1C"),
        ("CB", "TB-10 Cat CB (Partial Collapse)", "#D97706"),
    ]:
        r_cat = float(tb_10_brick_house_damage_range(q_tnt, category=cat_code, method="jarrett"))
        fig.add_trace(
            go.Scatter3d(
                x=r_cat * np.cos(theta_ring),
                y=r_cat * np.sin(theta_ring),
                z=np.full_like(theta_ring, 1.2),
                mode="lines",
                line=dict(color=col_cat, width=3, dash="dash"),
                name=f"{cat_desc}: {r_cat:.1f} m",
            )
        )

    # Exposed Site (ES) Reference Markers along cardinal/diagonal azimuths
    es_angles = [0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]
    es_names = ["ES-1: Process Workshop", "ES-2: Public Highway", "ES-3: Inhabited House", "ES-4: Hospital / School"]
    es_radii = [r_pbd, r_ptrd, r_ibd, r_vbd]
    es_cols = ["#DC2626", "#EA580C", "#2563EB", "#10B981"]

    for es_name, r_es, ang, c_es in zip(es_names, es_radii, es_angles, es_cols):
        ex = r_es * math.cos(ang)
        ey = r_es * math.sin(ang)
        bf_es = evaluate_blast_field(r_es, q_tnt, explosive="TNT")
        ps_es = float(bf_es["Ps_kPa"])
        fig.add_trace(
            go.Scatter3d(
                x=[ex],
                y=[ey],
                z=[max(max_r * 0.012, 2.0)],
                mode="markers+text",
                marker=dict(size=7, color=c_es, symbol="square"),
                text=[f"{es_name} ({r_es:.0f}m | {ps_es:.1f} kPa)"],
                textposition="top center",
                name=es_name,
            )
        )

    fig.update_layout(
        title=(
            f"3D UNODA IATG 01.80 / 02.20 Explosive Safety & Quantity-Distance (QD) Domes "
            f"(NEQ={neq_kg:.0f} kg {explosive_name}, W_TNT={q_tnt:.0f} kg)"
        ),
        scene=dict(
            xaxis=dict(title="East-West X (m)"),
            yaxis=dict(title="North-South Y (m)"),
            zaxis=dict(title="Altitude above PES Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.45, y=-1.45, z=0.95)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_multi_facility_site_planner_figure(
    pes_list: list[dict[str, Any]] | None = None,
    es_list: list[dict[str, Any]] | None = None,
    show_ibd_domes: bool = True,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    """Build a 3D Multi-Facility Explosive Storage Area (ESA) Site Planner & Sympathetic Detonation Matrix.

    Evaluates:
    - Inter-Magazine / Process Building Separation (PES_i -> PES_j):
      * Unbarricaded Open Stack: Z_IMD = 8.0 m/kg^(1/3)
      * Barricaded Magazine: Z_IMD = 4.5 m/kg^(1/3)
      * Earth-Covered Magazine (ECM, UFC 3-340-02 / IATG 02.20): Z_IMD = 2.4 m/kg^(1/3)
    - PES -> Exposed Site (ES) Quantity-Distance compliance:
      * Process Workshop: Z = 8.0 m/kg^(1/3)
      * Public Traffic Route: Z = 14.8 m/kg^(1/3)
      * Inhabited Building: Z = 22.2 m/kg^(1/3)
      * Vulnerable Building (School/Hospital): Z = 44.4 m/kg^(1/3)
    Returns `(fig, audit_rows)`.
    """
    if pes_list is None:
        pes_list = [
            {"name": "PES-1 (ECM)", "x_m": -60.0, "y_m": 0.0, "neq_tnt_kg": 1500.0, "magazine_type": "ecm"},
            {"name": "PES-2 (Barricaded)", "x_m": 45.0, "y_m": 20.0, "neq_tnt_kg": 2000.0, "magazine_type": "barricaded"},
            {"name": "PES-3 (Open Stack)", "x_m": 0.0, "y_m": 110.0, "neq_tnt_kg": 800.0, "magazine_type": "unbarricaded"},
        ]
    if es_list is None:
        es_list = [
            {"name": "ES-1 (Process Workshop)", "x_m": -75.0, "y_m": -85.0, "es_type": "process_building"},
            {"name": "ES-2 (Public Road)", "x_m": 180.0, "y_m": -120.0, "es_type": "public_traffic_route"},
            {"name": "ES-3 (Inhabited Quarters)", "x_m": 280.0, "y_m": 60.0, "es_type": "inhabited_building"},
            {"name": "ES-4 (Base Clinic)", "x_m": 490.0, "y_m": 160.0, "es_type": "vulnerable_building"},
        ]
    fig = go.Figure()
    audit_rows: list[dict[str, Any]] = []

    all_x = [float(p["x_m"]) for p in pes_list] + [float(e["x_m"]) for e in es_list] + [0.0]
    all_y = [float(p["y_m"]) for p in pes_list] + [float(e["y_m"]) for e in es_list] + [0.0]
    span = max(max(abs(min(all_x)), abs(max(all_x)), abs(min(all_y)), abs(max(all_y))) * 1.35, 350.0)

    gx = np.linspace(-span, span, 28)
    gy = np.linspace(-span, span, 28)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    fig.add_trace(
        go.Surface(
            x=GX,
            y=GY,
            z=GZ,
            colorscale=[[0, "#E5E7EB"], [1, "#D1D5DB"]],
            opacity=0.32,
            showscale=False,
            name="ESA Ground Grid",
            hoverinfo="skip",
        )
    )

    theta_ring = np.linspace(0.0, 2.0 * np.pi, 72)
    u_dome = np.linspace(0.0, 2.0 * np.pi, 24)
    v_dome = np.linspace(0.0, 0.5 * np.pi, 12)

    imd_z_map = {
        "unbarricaded": 8.0,
        "barricaded": 4.5,
        "ecm": 2.4,
    }
    es_z_map = {
        "process_building": 8.0,
        "public_traffic_route": 14.8,
        "inhabited_building": 22.2,
        "vulnerable_building": 44.4,
    }

    # Plot each PES and its IMD & IBD rings/domes
    for idx_p, pes in enumerate(pes_list):
        pname = str(pes.get("name", f"PES-{idx_p + 1}"))
        px_m = float(pes["x_m"])
        py_m = float(pes["y_m"])
        q_tnt = max(float(pes.get("neq_tnt_kg", 1000.0)), 0.1)
        b_type = str(pes.get("magazine_type", "barricaded")).lower()
        z_imd = imd_z_map.get(b_type, 4.5)
        r_imd = z_imd * (q_tnt ** (1.0 / 3.0))
        r_ibd = 22.2 * (q_tnt ** (1.0 / 3.0))

        fig.add_trace(
            go.Scatter3d(
                x=[px_m],
                y=[py_m],
                z=[3.0],
                mode="markers+text",
                marker=dict(size=9, color="#DC2626", symbol="diamond"),
                text=[f"{pname} ({q_tnt:.0f} kg TNT | {b_type.upper()})"],
                textposition="top center",
                name=pname,
            )
        )

        # IMD Ring
        fig.add_trace(
            go.Scatter3d(
                x=px_m + r_imd * np.cos(theta_ring),
                y=py_m + r_imd * np.sin(theta_ring),
                z=np.full_like(theta_ring, 0.6),
                mode="lines",
                line=dict(color="#DC2626", width=4),
                name=f"{pname} IMD Ring (R={r_imd:.1f}m, Z={z_imd})",
            )
        )
        # IBD Ring
        fig.add_trace(
            go.Scatter3d(
                x=px_m + r_ibd * np.cos(theta_ring),
                y=py_m + r_ibd * np.sin(theta_ring),
                z=np.full_like(theta_ring, 0.8),
                mode="lines",
                line=dict(color="#2563EB", width=3, dash="dash"),
                name=f"{pname} IBD Ring (R={r_ibd:.1f}m, Z=22.2)",
            )
        )

        if show_ibd_domes:
            DX = px_m + r_ibd * np.outer(np.cos(u_dome), np.sin(v_dome))
            DY = py_m + r_ibd * np.outer(np.sin(u_dome), np.sin(v_dome))
            DZ = r_ibd * np.outer(np.ones_like(u_dome), np.cos(v_dome))
            fig.add_trace(
                go.Surface(
                    x=DX,
                    y=DY,
                    z=DZ,
                    colorscale=[[0, "rgba(37, 99, 235, 0.10)"], [1, "rgba(37, 99, 235, 0.10)"]],
                    opacity=0.12,
                    showscale=False,
                    name=f"{pname} 3D IBD Dome",
                    hoverinfo="name",
                )
            )

    # Inter-Magazine (PES_i -> PES_j) Sympathetic Detonation Audit
    for i in range(len(pes_list)):
        for j in range(i + 1, len(pes_list)):
            p1, p2 = pes_list[i], pes_list[j]
            d_ij = math.hypot(float(p1["x_m"]) - float(p2["x_m"]), float(p1["y_m"]) - float(p2["y_m"]))
            q_max = max(float(p1.get("neq_tnt_kg", 1000.0)), float(p2.get("neq_tnt_kg", 1000.0)), 0.1)
            b1 = str(p1.get("magazine_type", "barricaded")).lower()
            b2 = str(p2.get("magazine_type", "barricaded")).lower()
            z_req = max(imd_z_map.get(b1, 4.5), imd_z_map.get(b2, 4.5))
            d_req = z_req * (q_max ** (1.0 / 3.0))
            z_act = d_ij / (q_max ** (1.0 / 3.0))
            bf_ij = evaluate_blast_field(max(d_ij, 1.0), q_max, explosive="TNT")
            ok = d_ij >= d_req
            audit_rows.append(
                {
                    "Donor PES": str(p1.get("name", f"PES-{i + 1}")),
                    "Acceptor / Exposed Site": str(p2.get("name", f"PES-{j + 1}")),
                    "Relation Type": f"Inter-Magazine IMD ({b1}/{b2})",
                    "Required Z (m/kg^1/3)": round(z_req, 1),
                    "Required D_req (m)": round(d_req, 1),
                    "Actual D_act (m)": round(d_ij, 1),
                    "Actual Z_act": round(z_act, 2),
                    "Incident Ps (kPa)": round(float(bf_ij["Ps_kPa"]), 1),
                    "QD Status": "✅ COMPLIANT" if ok else "⚠️ QD VIOLATION",
                }
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[float(p1["x_m"]), float(p2["x_m"])],
                    y=[float(p1["y_m"]), float(p2["y_m"])],
                    z=[3.0, 3.0],
                    mode="lines",
                    line=dict(color="#10B981" if ok else "#EF4444", width=5),
                    name=f"IMD {p1['name']}↔{p2['name']} ({'OK' if ok else 'VIOLATION'})",
                )
            )

    # Plot each Exposed Site (ES) and audit against all PES
    for idx_e, es in enumerate(es_list):
        ename = str(es.get("name", f"ES-{idx_e + 1}"))
        ex_m = float(es["x_m"])
        ey_m = float(es["y_m"])
        etype = str(es.get("es_type", "inhabited_building")).lower()
        z_req = es_z_map.get(etype, 22.2)

        fig.add_trace(
            go.Scatter3d(
                x=[ex_m],
                y=[ey_m],
                z=[3.5],
                mode="markers+text",
                marker=dict(size=8, color="#059669", symbol="square"),
                text=[f"{ename} ({etype.replace('_', ' ').title()})"],
                textposition="top center",
                name=ename,
            )
        )

        for idx_p, pes in enumerate(pes_list):
            pname = str(pes.get("name", f"PES-{idx_p + 1}"))
            q_tnt = max(float(pes.get("neq_tnt_kg", 1000.0)), 0.1)
            d_pe = math.hypot(float(pes["x_m"]) - ex_m, float(pes["y_m"]) - ey_m)
            d_req = z_req * (q_tnt ** (1.0 / 3.0))
            z_act = d_pe / (q_tnt ** (1.0 / 3.0))
            bf_pe = evaluate_blast_field(max(d_pe, 1.0), q_tnt, explosive="TNT")
            ok = d_pe >= d_req
            audit_rows.append(
                {
                    "Donor PES": pname,
                    "Acceptor / Exposed Site": ename,
                    "Relation Type": f"PES → {etype.replace('_', ' ').title()}",
                    "Required Z (m/kg^1/3)": round(z_req, 1),
                    "Required D_req (m)": round(d_req, 1),
                    "Actual D_act (m)": round(d_pe, 1),
                    "Actual Z_act": round(z_act, 2),
                    "Incident Ps (kPa)": round(float(bf_pe["Ps_kPa"]), 1),
                    "QD Status": "✅ COMPLIANT" if ok else "⚠️ QD VIOLATION",
                }
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[float(pes["x_m"]), ex_m],
                    y=[float(pes["y_m"]), ey_m],
                    z=[3.0, 3.5],
                    mode="lines",
                    line=dict(color="#10B981" if ok else "#EF4444", width=3, dash="dot" if ok else "solid"),
                    name=f"QD {pname}→{ename} ({'OK' if ok else 'VIOLATION'})",
                )
            )

    fig.update_layout(
        title="3D Multi-Facility Explosive Storage Area (ESA) Site Planner & Sympathetic Detonation Audit",
        scene=dict(
            xaxis=dict(title="East-West X (m)"),
            yaxis=dict(title="North-South Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.4, y=-1.4, z=1.05)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig, audit_rows


def create_3d_guided_intercept_figure(traj_res: dict[str, Any]) -> go.Figure:
    """Renders a 3D visualization of a closed-loop guided missile intercept engagement."""
    hist = traj_res.get("history", {})
    m_pos = np.asarray(hist.get("missile_pos", []), dtype=np.float64)
    t_pos = np.asarray(hist.get("target_pos", []), dtype=np.float64)
    act_g = np.asarray(hist.get("act_g", []), dtype=np.float64)

    fig = go.Figure()

    if len(m_pos) > 1:
        # Missile trajectory colored by G-load
        fig.add_trace(
            go.Scatter3d(
                x=m_pos[:, 0],
                y=m_pos[:, 1],
                z=m_pos[:, 2],
                mode="lines",
                line=dict(
                    color=act_g if len(act_g) == len(m_pos) else "#2563EB",
                    colorscale="Viridis",
                    colorbar=dict(title="G-Load (g)", x=1.05),
                    width=6,
                ),
                name="Guided Missile Trajectory",
            )
        )
        # Missile start
        fig.add_trace(
            go.Scatter3d(
                x=[m_pos[0, 0]],
                y=[m_pos[0, 1]],
                z=[m_pos[0, 2]],
                mode="markers+text",
                marker=dict(size=8, color="#10B981", symbol="diamond"),
                text=["Launch Point"],
                textposition="top center",
                name="Launch Point",
            )
        )

    if len(t_pos) > 1:
        # Target trajectory
        fig.add_trace(
            go.Scatter3d(
                x=t_pos[:, 0],
                y=t_pos[:, 1],
                z=t_pos[:, 2],
                mode="lines+markers",
                line=dict(color="#DC2626", width=4, dash="dash"),
                marker=dict(size=4, color="#DC2626"),
                name="Target Track",
            )
        )
        # Final target position
        fig.add_trace(
            go.Scatter3d(
                x=[t_pos[-1, 0]],
                y=[t_pos[-1, 1]],
                z=[t_pos[-1, 2]],
                mode="markers+text",
                marker=dict(size=10, color="#DC2626", symbol="square"),
                text=[f"Target ({traj_res.get('status', 'TERMINAL')})"],
                textposition="top right",
                name="Target Position",
            )
        )

    # Sample a few Line-of-Sight (LOS) rays
    if len(m_pos) > 5 and len(t_pos) == len(m_pos):
        sample_indices = np.linspace(0, len(m_pos) - 1, min(6, len(m_pos)), dtype=int)
        for idx in sample_indices:
            fig.add_trace(
                go.Scatter3d(
                    x=[m_pos[idx, 0], t_pos[idx, 0]],
                    y=[m_pos[idx, 1], t_pos[idx, 1]],
                    z=[m_pos[idx, 2], t_pos[idx, 2]],
                    mode="lines",
                    line=dict(color="rgba(156, 163, 175, 0.45)", width=2, dash="dot"),
                    showlegend=bool(idx == sample_indices[0]),
                    name="Seeker Line-of-Sight (LOS)",
                )
            )

    # Intercept point sphere
    if len(m_pos) > 0:
        hit = traj_res.get("hit", False)
        miss_m = float(traj_res.get("miss_distance_m", 0.0))
        fig.add_trace(
            go.Scatter3d(
                x=[m_pos[-1, 0]],
                y=[m_pos[-1, 1]],
                z=[m_pos[-1, 2]],
                mode="markers+text",
                marker=dict(size=12, color="#10B981" if hit else "#F59E0B", symbol="cross"),
                text=[f"{'DIRECT HIT' if hit else 'CLOSEST APPROACH'} (Miss={miss_m:.2f}m)"],
                textposition="bottom center",
                name="Intercept Point",
            )
        )

    fig.update_layout(
        title=f"3D Closed-Loop Guided Intercept — Status: {traj_res.get('status', 'UNKNOWN')} | Miss={traj_res.get('miss_distance_m', 0.0):.2f}m",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_submunition_dispersion_figure(disp_res: dict[str, Any]) -> go.Figure:
    """Renders 3D carrier approach, submunition descent tracks, and ground UXO hazard footprint."""
    fig = go.Figure()

    bp = disp_res.get("burst_point_m", [12000.0, 0.0, 380.0])
    # Carrier approach line (arbitrary 1.5km lead-in)
    fig.add_trace(
        go.Scatter3d(
            x=[bp[0] - 1500.0, bp[0]],
            y=[bp[1], bp[1]],
            z=[bp[2] + 450.0, bp[2]],
            mode="lines+markers",
            line=dict(color="#4B5563", width=5, dash="dash"),
            marker=dict(size=6, color="#4B5563"),
            name="Carrier Approach Trajectory",
        )
    )

    # Burst point flash marker
    fig.add_trace(
        go.Scatter3d(
            x=[bp[0]],
            y=[bp[1]],
            z=[bp[2]],
            mode="markers+text",
            marker=dict(size=12, color="#F59E0B", symbol="diamond"),
            text=[f"Carrier Payload Burst (Alt={bp[2]:.0f}m, N={disp_res.get('n_submunitions', 64)})"],
            textposition="top center",
            name="Payload Expulsion Burst",
        )
    )

    # Submunition trajectories
    trajs = disp_res.get("trajectories", [])
    subs = disp_res.get("submunitions", [])
    for idx, (tr, s_info) in enumerate(zip(trajs, subs)):
        tr_arr = np.asarray(tr, dtype=np.float64)
        if len(tr_arr) < 2:
            continue
        is_uxo = s_info.get("is_uxo", False)
        status = s_info.get("status", "DETONATED")
        col = "#DC2626" if is_uxo else ("#10B981" if status == "SELF_DESTRUCT_NEUTRALIZED" else "rgba(37, 99, 235, 0.55)")
        fig.add_trace(
            go.Scatter3d(
                x=tr_arr[:, 0],
                y=tr_arr[:, 1],
                z=tr_arr[:, 2],
                mode="lines",
                line=dict(color=col, width=2),
                showlegend=bool(idx == 0),
                name="Submunition Descent Path",
            )
        )

    # Ground impact points grouped by status
    det_x = [s["impact_x_m"] for s in subs if s["status"] == "DETONATED"]
    det_y = [s["impact_y_m"] for s in subs if s["status"] == "DETONATED"]
    sd_x = [s["impact_x_m"] for s in subs if s["status"] == "SELF_DESTRUCT_NEUTRALIZED"]
    sd_y = [s["impact_y_m"] for s in subs if s["status"] == "SELF_DESTRUCT_NEUTRALIZED"]
    uxo_x = [s["impact_x_m"] for s in subs if s["status"] == "UNEXPLODED_ORDNANCE_UXO"]
    uxo_y = [s["impact_y_m"] for s in subs if s["status"] == "UNEXPLODED_ORDNANCE_UXO"]

    if det_x:
        fig.add_trace(
            go.Scatter3d(
                x=det_x,
                y=det_y,
                z=np.zeros(len(det_x)),
                mode="markers",
                marker=dict(size=6, color="#2563EB", symbol="circle"),
                name=f"Detonated Functional ({len(det_x)})",
            )
        )
    if sd_x:
        fig.add_trace(
            go.Scatter3d(
                x=sd_x,
                y=sd_y,
                z=np.zeros(len(sd_x)),
                mode="markers",
                marker=dict(size=7, color="#10B981", symbol="diamond"),
                name=f"Self-Destruct Neutralized ({len(sd_x)})",
            )
        )
    if uxo_x:
        fig.add_trace(
            go.Scatter3d(
                x=uxo_x,
                y=uxo_y,
                z=np.zeros(len(uxo_x)),
                mode="markers+text",
                marker=dict(size=11, color="#DC2626", symbol="square"),
                text=[f"UXO #{idx + 1}" for idx in range(len(uxo_x))],
                textposition="top center",
                name=f"Live UXO Dud ({len(uxo_x)}) [HAZARD]",
            )
        )

    # IATG 10.10 Clearance Cordon Ring on ground plane
    c_x = float(disp_res.get("mean_impact_x_m", bp[0]))
    c_y = float(disp_res.get("mean_impact_y_m", bp[1]))
    r_cordon = float(disp_res.get("iatg_cordon_radius_m", 250.0))
    theta = np.linspace(0, 2.0 * math.pi, 90)
    fig.add_trace(
        go.Scatter3d(
            x=c_x + r_cordon * np.cos(theta),
            y=c_y + r_cordon * np.sin(theta),
            z=np.zeros_like(theta),
            mode="lines",
            line=dict(color="#DC2626", width=4, dash="dash"),
            name=f"IATG 10.10 Clearance Cordon (R={r_cordon:.0f}m)",
        )
    )

    fig.update_layout(
        title=f"3D Cluster Munition Dispersion & UXO Footprint (N={disp_res.get('n_submunitions', 64)}, UXO={disp_res.get('n_uxo', 0)}, Area={disp_res.get('footprint_area_hectares', 0.0):.2f} ha)",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_6dof_attitude_figure(traj_res: dict[str, Any]) -> go.Figure:
    """Renders 3D trajectory with projectile orientation vectors and gyroscopic spin drift."""
    fig = go.Figure()
    hist = traj_res.get("history", {})
    x = np.asarray(hist.get("x_m", []), dtype=np.float64)
    y = np.asarray(hist.get("y_m", []), dtype=np.float64)
    z = np.asarray(hist.get("z_m", []), dtype=np.float64)
    alpha_t = np.asarray(hist.get("alpha_t_deg", []), dtype=np.float64)

    if len(x) > 1:
        # Full flight path colored by total angle of attack alpha_t
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=alpha_t if len(alpha_t) == len(x) else "#2563EB",
                    colorscale="Plasma",
                    colorbar=dict(title="Total AoA |α_t| (°)", x=1.05),
                    width=6,
                ),
                name="6-DoF Rigid-Body Trajectory",
            )
        )

        # Draw body attitude vectors at regular intervals along trajectory
        n_vectors = min(12, len(x))
        indices = np.linspace(0, len(x) - 1, n_vectors, dtype=int)
        vec_len = float(np.max(x) - np.min(x)) * 0.04
        pitch_deg = np.asarray(hist.get("pitch_deg", []), dtype=np.float64)
        yaw_deg = np.asarray(hist.get("yaw_deg", []), dtype=np.float64)

        for i in indices:
            if i < len(pitch_deg) and i < len(yaw_deg):
                prad = math.radians(pitch_deg[i])
                yrad = math.radians(yaw_deg[i])
                nx = math.cos(prad) * math.cos(yrad)
                ny = math.cos(prad) * math.sin(yrad)
                nz = math.sin(prad)
                fig.add_trace(
                    go.Scatter3d(
                        x=[x[i], x[i] + vec_len * nx],
                        y=[y[i], y[i] + vec_len * ny],
                        z=[z[i], z[i] + vec_len * nz],
                        mode="lines",
                        line=dict(color="#10B981", width=5),
                        showlegend=bool(i == indices[0]),
                        name="Projectile Body Symmetry Axis",
                    )
                )

        # Ground impact
        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[0.0],
                mode="markers+text",
                marker=dict(size=10, color="#DC2626", symbol="diamond"),
                text=[f"Impact: X={x[-1]:.0f}m, Drift={y[-1]:+.1f}m"],
                textposition="top center",
                name="6-DoF Impact Point",
            )
        )

    fig.update_layout(
        title=f"3D STANAG 4355 Full 6-DoF Trajectory & Attitude Vectors — Range: {traj_res.get('final_range_m', 0.0):.0f}m | Lateral Spin Drift: {traj_res.get('final_deflection_m', 0.0):+.1f}m (Right)",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Drift Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_battery_salvo_figure(salvo_res: dict[str, Any]) -> go.Figure:
    """Renders 3D artillery battery layout, sheaf trajectories, and ground Carleton lethality."""
    fig = go.Figure()
    guns = salvo_res.get("guns", [])
    aim_pts = salvo_res.get("aim_points", [])
    trajs = salvo_res.get("trajectories", [])
    rounds = salvo_res.get("rounds", [])

    # 1. Gun positions
    if guns:
        gx = [g["x_m"] for g in guns]
        gy = [g["y_m"] for g in guns]
        gz = [g.get("z_m", 0.0) for g in guns]
        fig.add_trace(
            go.Scatter3d(
                x=gx,
                y=gy,
                z=gz,
                mode="markers+text",
                marker=dict(size=9, color="#10B981", symbol="diamond"),
                text=[f"Gun #{g['gun_id']}" for g in guns],
                textposition="top center",
                name="Battery Gun Positions",
            )
        )

    # 2. Aim Points
    if aim_pts:
        ax = [p[0] for p in aim_pts]
        ay = [p[1] for p in aim_pts]
        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=np.zeros(len(ax)),
                mode="markers+text",
                marker=dict(size=8, color="#F59E0B", symbol="cross"),
                text=[f"Aim #{i+1}" for i in range(len(ax))],
                textposition="bottom center",
                name=f"Sheaf Aim Points ({salvo_res.get('sheaf_type', 'Converged').title()})",
            )
        )

    # 3. 3D Flight Trajectories
    palette = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DB2777", "#4B5563", "#0D9488", "#DC2626"]
    for idx, tr in enumerate(trajs):
        tr_arr = np.asarray(tr, dtype=np.float64)
        if len(tr_arr) < 2:
            continue
        col = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter3d(
                x=tr_arr[:, 0],
                y=tr_arr[:, 1],
                z=tr_arr[:, 2],
                mode="lines",
                line=dict(color=col, width=3),
                showlegend=bool(idx < len(guns)),
                name=f"Gun #{guns[idx % len(guns)]['gun_id']} Salvo Flight" if guns else "Trajectory",
            )
        )

    # 4. Actual Round Impact Points
    if rounds:
        ix = [r["impact_x_m"] for r in rounds]
        iy = [r["impact_y_m"] for r in rounds]
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=np.zeros(len(ix)),
                mode="markers",
                marker=dict(size=7, color="#DC2626", symbol="circle"),
                name=f"Ground Impacts (N={len(rounds)})",
            )
        )

    # 5. Target Center and CEP50 ring
    t_pos = salvo_res.get("target_pos_m", [15000.0, 0.0])
    cep50 = float(salvo_res.get("cep50_m", 35.0))
    theta = np.linspace(0, 2.0 * math.pi, 60)
    fig.add_trace(
        go.Scatter3d(
            x=t_pos[0] + cep50 * np.cos(theta),
            y=t_pos[1] + cep50 * np.sin(theta),
            z=np.full_like(theta, 0.5),
            mode="lines",
            line=dict(color="#DC2626", width=3, dash="dash"),
            name=f"Battery Sheaf CEP50 ({cep50:.1f}m)",
        )
    )

    fig.update_layout(
        title=f"3D Multi-Gun Battery Salvo & Sheaf Geometry — {salvo_res.get('n_guns', 6)} Guns | {salvo_res.get('total_rounds', 12)} Rounds | Sheaf: {salvo_res.get('sheaf_type', 'Converged').title()} | Neutralization: {salvo_res.get('fractional_neutralization_pct', 0.0)}%",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_armor_defeat_figure(armor_res: dict[str, Any]) -> go.Figure:
    """Renders 3D multi-layer armor sandwich, penetrating jet, and behind-armor spall cone."""
    fig = go.Figure()
    has_era = armor_res.get("has_era", True)
    perforated = armor_res.get("perforated", False)
    spall = armor_res.get("behind_armor_spall", {})

    # Plate dimensions
    w_plate = 300.0  # mm
    h_plate = 300.0  # mm
    z_base = 0.0

    # Layer 1: Front ERA Cassette (if present)
    if has_era:
        # Flyer plate 1 (Steel)
        fig.add_trace(
            go.Scatter3d(
                x=[-120.0, -120.0, -120.0, -120.0, -120.0],
                y=[-w_plate / 2, w_plate / 2, w_plate / 2, -w_plate / 2, -w_plate / 2],
                z=[-h_plate / 2, -h_plate / 2, h_plate / 2, h_plate / 2, -h_plate / 2],
                mode="lines",
                line=dict(color="#9CA3AF", width=4),
                name="ERA Front Flyer Plate (Steel, 10mm)",
            )
        )
        # ERA Explosive Interlayer
        fig.add_trace(
            go.Scatter3d(
                x=[-95.0, -95.0, -95.0, -95.0, -95.0],
                y=[-w_plate / 2, w_plate / 2, w_plate / 2, -w_plate / 2, -w_plate / 2],
                z=[-h_plate / 2, -h_plate / 2, h_plate / 2, h_plate / 2, -h_plate / 2],
                mode="lines",
                line=dict(color="#EF4444", width=5),
                name=f"ERA Explosive Core (V_flyer={armor_res.get('era_flyer_speed_mps', 1200):.0f} m/s)",
            )
        )
        # Flyer plate 2 (Backing)
        fig.add_trace(
            go.Scatter3d(
                x=[-70.0, -70.0, -70.0, -70.0, -70.0],
                y=[-w_plate / 2, w_plate / 2, w_plate / 2, -w_plate / 2, -w_plate / 2],
                z=[-h_plate / 2, -h_plate / 2, h_plate / 2, h_plate / 2, -h_plate / 2],
                mode="lines",
                line=dict(color="#9CA3AF", width=4),
                name="ERA Rear Flyer Plate",
            )
        )

    # Layer 2: Ceramic Tile (SiC / Al2O3)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, 0.0, 0.0, 0.0, 0.0],
            y=[-w_plate / 2, w_plate / 2, w_plate / 2, -w_plate / 2, -w_plate / 2],
            z=[-h_plate / 2, -h_plate / 2, h_plate / 2, h_plate / 2, -h_plate / 2],
            mode="lines",
            line=dict(color="#06B6D4", width=6),
            name="Silicon Carbide (SiC) Ceramic Dwell Tile",
        )
    )

    # Layer 3: High-Hardness RHA Steel Backing Plate
    rha_thick = float(armor_res.get("base_rha_los_thickness_mm", 450.0))
    fig.add_trace(
        go.Scatter3d(
            x=[rha_thick, rha_thick, rha_thick, rha_thick, rha_thick],
            y=[-w_plate / 2, w_plate / 2, w_plate / 2, -w_plate / 2, -w_plate / 2],
            z=[-h_plate / 2, -h_plate / 2, h_plate / 2, h_plate / 2, -h_plate / 2],
            mode="lines",
            line=dict(color="#374151", width=6),
            name=f"RHA Backing Plate ({rha_thick:.0f}mm LOS)",
        )
    )

    # Penetrating Shaped-Charge Jet Line
    max_p = float(armor_res.get("post_era_jet_penetration_mm", 650.0))
    x_jet_end = min(max_p, rha_thick + 150.0 if perforated else max_p)
    fig.add_trace(
        go.Scatter3d(
            x=[-250.0, x_jet_end],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="lines+markers",
            line=dict(color="#F59E0B", width=7),
            marker=dict(size=[6, 10], color=["#F59E0B", "#DC2626"]),
            name=f"Shaped-Charge Jet (Residual Pen: {max_p:.0f}mm)",
        )
    )

    # Behind-Armor Spall Cone
    if perforated and spall.get("spall_fragment_count", 0) > 0:
        n_sp = min(30, int(spall.get("spall_fragment_count", 50)))
        cone_ang = math.radians(float(spall.get("spall_cone_half_angle_deg", 28.0)))
        rng_sp = np.random.default_rng(42)
        for _ in range(n_sp):
            th_sp = float(rng_sp.uniform(0, 2.0 * math.pi))
            r_sp = math.tan(cone_ang * float(rng_sp.uniform(0.2, 1.0))) * 220.0
            fig.add_trace(
                go.Scatter3d(
                    x=[rha_thick, rha_thick + 220.0],
                    y=[0.0, r_sp * math.cos(th_sp)],
                    z=[0.0, r_sp * math.sin(th_sp)],
                    mode="lines",
                    line=dict(color="#DC2626", width=2),
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[rha_thick + 220.0],
                y=[0.0],
                z=[0.0],
                mode="markers+text",
                marker=dict(size=8, color="#DC2626", symbol="diamond"),
                text=[f"Spall Cone (N={spall.get('spall_fragment_count', 0)}, Pk={spall.get('crew_incapacitation_prob', 0.0):.2f})"],
                textposition="top right",
                name="Behind-Armor Spall Cone",
            )
        )

    fig.update_layout(
        title=f"3D Multi-Layer Armor Defeat & Terminal Spall Cone — Status: {armor_res.get('perforation_status', 'UNKNOWN')}",
        scene=dict(
            xaxis=dict(title="Penetration Depth X (mm)"),
            yaxis=dict(title="Lateral Plate Width Y (mm)"),
            zaxis=dict(title="Vertical Plate Height Z (mm)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_terrain_clearance_figure(
    clear_res: dict[str, Any],
    traj_x: list[float] | None = None,
    traj_y: list[float] | None = None,
    traj_z: list[float] | None = None,
) -> go.Figure:
    """Renders 3D mountain ridge terrain surface, masking crest clearance buffer, and dead space shadow."""
    fig = go.Figure()
    from ammo_mc.geospatial import evaluate_terrain_profile

    # Extract trajectory if not explicitly passed
    if traj_x is None or traj_y is None or traj_z is None:
        t_dict = clear_res.get("trajectory", {})
        target_dist = float(clear_res.get("target_dist_m", 14000.0))
        traj_x = t_dict.get("x_m", [0.0, target_dist])
        traj_y = t_dict.get("y_m", [0.0, 0.0])
        traj_z = t_dict.get("z_m", [0.0, 0.0])

    # Terrain profile along X
    x_max = max(float(traj_x[-1]) if traj_x else 14000.0, 14000.0)
    tx_grid = np.linspace(0.0, x_max, 100)
    ty_grid = np.linspace(-600.0, 600.0, 25)
    TX, TY = np.meshgrid(tx_grid, ty_grid)

    r_pos = float(clear_res.get("crest_x_m", 6500.0))
    TZ = np.zeros_like(TX)
    for i in range(TX.shape[0]):
        TZ[i, :] = evaluate_terrain_profile(TX[i, :], ridge_pos_m=r_pos)

    # 1. 3D Terrain Surface
    fig.add_trace(
        go.Surface(
            x=TX,
            y=TY,
            z=TZ,
            colorscale="Earth",
            opacity=0.65,
            showscale=False,
            name="Mountain Ridge Terrain DEM",
        )
    )

    # 2. 3D Flight Trajectory
    if len(traj_x) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode="lines",
                line=dict(color="#2563EB", width=5),
                name="Artillery Trajectory Arc",
            )
        )

    # 3. Masking Crest Peak Point & Clearance Vertical Line
    cx = float(clear_res.get("crest_x_m", 6500.0))
    cz = float(clear_res.get("crest_z_m", 620.0))
    c_traj_z = float(clear_res.get("trajectory_z_at_crest_m", cz + 100.0))
    ok = clear_res.get("clears_crest", True)
    fig.add_trace(
        go.Scatter3d(
            x=[cx, cx],
            y=[0.0, 0.0],
            z=[cz, c_traj_z],
            mode="lines+markers",
            line=dict(color="#10B981" if ok else "#EF4444", width=6),
            marker=dict(size=[8, 8], color=["#374151", "#10B981" if ok else "#EF4444"]),
            name=f"Crest Safety Buffer (ΔZ = {clear_res.get('actual_clearance_m', 0.0):.1f}m)",
        )
    )

    # 4. Dead Space Shadow Zone behind the ridge
    ds_start = float(clear_res.get("dead_space_start_m", cx + 100.0))
    ds_end = float(clear_res.get("dead_space_end_m", cx + 1200.0))
    if ds_end > ds_start:
        ds_x = np.linspace(ds_start, ds_end, 30)
        ds_z = [float(evaluate_terrain_profile(x_val, ridge_pos_m=r_pos)) + 2.0 for x_val in ds_x]
        fig.add_trace(
            go.Scatter3d(
                x=ds_x,
                y=np.zeros_like(ds_x),
                z=ds_z,
                mode="lines",
                line=dict(color="#DC2626", width=8),
                name=f"Dead Space Shadow (Extent: {clear_res.get('dead_space_extent_m', 0.0):.0f}m)",
            )
        )

    fig.update_layout(
        title=f"3D Terrain Crest Clearance & Dead Space Audit — Status: {clear_res.get('clearance_status', 'UNKNOWN')} | Clearance: {clear_res.get('actual_clearance_m', 0.0):.1f}m (Min: {clear_res.get('required_clearance_m', 50.0):.0f}m)",
        scene=dict(
            xaxis=dict(title="Downrange Distance X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Terrain Elevation Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_counter_battery_radar_figure(
    radar_res: dict[str, Any],
    reconstruction_res: dict[str, Any],
) -> go.Figure:
    """Renders 3D Weapon-Locating Radar tracking, beam scan volume, POO battery, and POI warning zone."""
    fig = go.Figure()
    rx, ry, rz = radar_res.get("radar_pos_m", [4000.0, -2500.0, 50.0])

    # 1. Radar Emplacement Emissive Marker
    fig.add_trace(
        go.Scatter3d(
            x=[rx],
            y=[ry],
            z=[rz],
            mode="markers+text",
            marker=dict(size=10, color="#10B981", symbol="diamond"),
            text=["📡 Weapon-Locating Radar (AN/TPQ-53)"],
            textposition="top center",
            name="Counter-Battery Radar",
        )
    )

    # 2. Radar Fan Search Cone / Beam Coverage Wireframe
    r_beam = 14000.0
    for az_offset in [-25.0, 0.0, 25.0]:
        rad_az = math.radians(45.0 + az_offset)
        for el in [4.0, 25.0]:
            rad_el = math.radians(el)
            bx = rx + r_beam * math.cos(rad_el) * math.sin(rad_az)
            by = ry + r_beam * math.cos(rad_el) * math.cos(rad_az)
            bz = rz + r_beam * math.sin(rad_el)
            fig.add_trace(
                go.Scatter3d(
                    x=[rx, bx],
                    y=[ry, by],
                    z=[rz, bz],
                    mode="lines",
                    line=dict(color="rgba(16, 185, 129, 0.35)", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # 3. Noisy Track Point Observations
    tracks = radar_res.get("track_points", [])
    if tracks:
        ox = [p["obs_x"] for p in tracks]
        oy = [p["obs_y"] for p in tracks]
        oz = [p["obs_z"] for p in tracks]
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="markers",
                marker=dict(size=5, color="#F59E0B", symbol="cross"),
                name=f"Radar Detections (N={len(tracks)})",
            )
        )

    # 4. Reconstructed Trajectory (Inverse Back-Propagation + Forward Extrapolation)
    re_traj = reconstruction_res.get("reconstructed_trajectory", {})
    if re_traj and "x_m" in re_traj:
        fig.add_trace(
            go.Scatter3d(
                x=re_traj["x_m"],
                y=re_traj["y_m"],
                z=re_traj["z_m"],
                mode="lines",
                line=dict(color="#2563EB", width=5),
                name="Reconstructed Ballistic Arc",
            )
        )

    # 5. Point of Origin (POO) Hostile Gun Position & Confidence Ellipse
    poo_x = reconstruction_res.get("poo_x_m", 0.0)
    poo_y = reconstruction_res.get("poo_y_m", 0.0)
    poo_z = reconstruction_res.get("poo_z_m", 0.0)
    fig.add_trace(
        go.Scatter3d(
            x=[poo_x],
            y=[poo_y],
            z=[poo_z],
            mode="markers+text",
            marker=dict(size=12, color="#DC2626", symbol="square"),
            text=["🎯 Estimated Hostile Battery (POO)"],
            textposition="top left",
            name="Point of Origin (POO)",
        )
    )
    poo_ell = reconstruction_res.get("poo_error_ellipse", {})
    a_poo = poo_ell.get("semi_major_m", 45.0)
    b_poo = poo_ell.get("semi_minor_m", 25.0)
    theta_pts = np.linspace(0, 2.0 * math.pi, 30)
    ell_x = [poo_x + a_poo * math.cos(th) for th in theta_pts]
    ell_y = [poo_y + b_poo * math.sin(th) for th in theta_pts]
    fig.add_trace(
        go.Scatter3d(
            x=ell_x,
            y=ell_y,
            z=[poo_z] * len(theta_pts),
            mode="lines",
            line=dict(color="#DC2626", width=3, dash="dot"),
            name=f"POO Error Ellipse ({a_poo:.0f}m x {b_poo:.0f}m)",
        )
    )

    # 6. Point of Impact (POI) Warning Zone
    poi_x = reconstruction_res.get("poi_x_m", 14000.0)
    poi_y = reconstruction_res.get("poi_y_m", 0.0)
    poi_z = reconstruction_res.get("poi_z_m", 0.0)
    fig.add_trace(
        go.Scatter3d(
            x=[poi_x],
            y=[poi_y],
            z=[poi_z],
            mode="markers+text",
            marker=dict(size=12, color="#7C3AED", symbol="circle"),
            text=[f"⚠️ Predicted POI (Warn: {reconstruction_res.get('early_warning_countdown_s', 0):.1f}s)"],
            textposition="top right",
            name="Predicted Impact (POI)",
        )
    )

    fig.update_layout(
        title=f"3D Counter-Battery Radar Trajectory Reconstruction (NATO STANAG 4114) — POO: ({poo_x:.0f}m, {poo_y:.0f}m) | Warning Time: {reconstruction_res.get('early_warning_countdown_s', 0):.1f}s",
        scene=dict(
            xaxis=dict(title="East X (m)"),
            yaxis=dict(title="North Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_atmospheric_shear_figure(
    met_res: dict[str, Any],
    sounding_profile: dict[str, Any],
) -> go.Figure:
    """Renders 3D comparison of trajectories fired through standard ISA atmosphere vs. sheared METCM sounding."""
    fig = go.Figure()

    # 1. Trajectory with Atmospheric Sounding & Sheared Winds
    t_met = met_res.get("trajectory_met", {})
    if t_met and "x_m" in t_met:
        fig.add_trace(
            go.Scatter3d(
                x=t_met["x_m"],
                y=t_met["y_m"],
                z=t_met["z_m"],
                mode="lines",
                line=dict(color="#2563EB", width=5),
                name=f"Sheared Sounding Trajectory ({met_res.get('range_with_met_m', 0.0):.0f}m)",
            )
        )

    # 2. Standard Uniform ISA Atmosphere Baseline Trajectory
    t_isa = met_res.get("trajectory_isa", {})
    if t_isa and "x_m" in t_isa:
        fig.add_trace(
            go.Scatter3d(
                x=t_isa["x_m"],
                y=t_isa["y_m"],
                z=t_isa["z_m"],
                mode="lines",
                line=dict(color="#9CA3AF", width=3, dash="dash"),
                name=f"Standard ISA Zero-Wind Baseline ({met_res.get('range_isa_baseline_m', 0.0):.0f}m)",
            )
        )

    # 3. 3D Wind Vector Barb Quivers at Sounding Layer Altitudes
    zones = sounding_profile.get("zones", [])
    x_center = float(met_res.get("range_with_met_m", 10000.0)) * 0.45
    y_center = 0.0
    for z_row in zones[1::2]:  # Step every 2 zones
        z_alt = z_row["altitude_m"]
        if z_alt <= met_res.get("vertex_altitude_m", 6000.0) * 1.15:
            scale = 45.0  # Scale m/s to meters for visual representation
            wx = z_row["wind_vx_mps"] * scale
            wy = z_row["wind_vy_mps"] * scale
            fig.add_trace(
                go.Scatter3d(
                    x=[x_center, x_center + wx],
                    y=[y_center, y_center + wy],
                    z=[z_alt, z_alt],
                    mode="lines+markers",
                    line=dict(color="#F59E0B", width=4),
                    marker=dict(size=[0, 4], color="#F59E0B"),
                    name=f"Wind @ {z_alt}m ({z_row['wind_speed_mps']} m/s)",
                    showlegend=(z_alt == zones[1]["altitude_m"]),
                )
            )

    fig.update_layout(
        title=f"3D Atmospheric Wind Shear & Met Sounding (STANAG 4082) — Range Shift ΔR: {met_res.get('delta_range_met_m', 0.0):+.1f}m | Crosswind Drift: {met_res.get('crossrange_drift_m', 0.0):.1f}m",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_ew_guidance_figure(
    ew_res: dict[str, Any],
    sal_res: dict[str, Any],
) -> go.Figure:
    """Renders 3D guided precision munition flight, EW GPS jamming dome, and laser designator homing."""
    fig = go.Figure()

    # 1. Guided Munition Trajectory
    traj = ew_res.get("trajectory", [])
    if traj:
        tx = [p[0] for p in traj]
        ty = [p[1] for p in traj]
        tz = [p[2] for p in traj]
        denial_idx = ew_res.get("denial_point_index", -1)

        if denial_idx > 0 and denial_idx < len(traj):
            # Pre-denial segment (GPS-aided)
            fig.add_trace(
                go.Scatter3d(
                    x=tx[: denial_idx + 1],
                    y=ty[: denial_idx + 1],
                    z=tz[: denial_idx + 1],
                    mode="lines",
                    line=dict(color="#10B981", width=5),
                    name="GPS-Aided Guidance (Precision)",
                )
            )
            # Post-denial segment (Inertial dead-reckoning drift)
            fig.add_trace(
                go.Scatter3d(
                    x=tx[denial_idx:],
                    y=ty[denial_idx:],
                    z=tz[denial_idx:],
                    mode="lines",
                    line=dict(color="#DC2626", width=5, dash="dash"),
                    name="Jammed INS Dead-Reckoning Drift",
                )
            )
            # Denial Event Marker
            fig.add_trace(
                go.Scatter3d(
                    x=[tx[denial_idx]],
                    y=[ty[denial_idx]],
                    z=[tz[denial_idx]],
                    mode="markers+text",
                    marker=dict(size=9, color="#EF4444", symbol="x"),
                    text=["⚡ GPS Lock Lost (J/S > 42 dB)"],
                    textposition="top center",
                    name="GPS Jamming Denial Point",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=tx,
                    y=ty,
                    z=tz,
                    mode="lines",
                    line=dict(color="#10B981", width=5),
                    name="Nominal Guided Trajectory",
                )
            )

    # 2. EW Standoff Noise Jammer Emplacement & Jamming Dome
    jx, jy, jz = ew_res.get("jammer_pos_m", [15000.0, 1800.0, 80.0])
    fig.add_trace(
        go.Scatter3d(
            x=[jx],
            y=[jy],
            z=[jz],
            mode="markers+text",
            marker=dict(size=12, color="#EF4444", symbol="diamond"),
            text=["📡 Standoff EW GPS Jammer"],
            textposition="top right",
            name="EW Jammer Site",
        )
    )

    # 3. Semi-Active Laser Designator & Target Spot
    laser_spot = sal_res.get("laser_spot_pos_m", [14000.0, 0.0, 0.0])
    fig.add_trace(
        go.Scatter3d(
            x=[laser_spot[0]],
            y=[laser_spot[1]],
            z=[laser_spot[2]],
            mode="markers+text",
            marker=dict(size=11, color="#F59E0B", symbol="circle"),
            text=["🎯 Target (Laser Designator Spot)"],
            textposition="top left",
            name="Target & Laser Spot",
        )
    )

    fig.update_layout(
        title=f"3D Electronic Warfare (EW) & Precision Guidance — Status: {ew_res.get('cep_category', 'Precision')} | Terminal Miss Distance: {ew_res.get('terminal_miss_distance_m', 0.0):.2f}m",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


def create_3d_direct_fire_sabot_figure(
    sabot_res: dict[str, Any],
    direct_res: dict[str, Any],
) -> go.Figure:
    """Renders 3D direct-fire tank gunnery, APFSDS sabot petal separation, flat trajectory, and moving target."""
    fig = go.Figure()

    # 1. Gun Muzzle Origin
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[1.8],
            mode="markers+text",
            marker=dict(size=10, color="#10B981", symbol="diamond"),
            text=["🛡️ Main Battle Tank Gun (Muzzle Exit)"],
            textposition="top left",
            name="120mm Gun Muzzle",
        )
    )

    # 2. Sabot Petal Radial Divergence Arcs
    petal_paths = sabot_res.get("petal_paths", [])
    colors = ["#F59E0B", "#EF4444", "#3B82F6", "#8B5CF6"]
    for idx, p_path in enumerate(petal_paths):
        px = [pt[0] for pt in p_path]
        py = [pt[1] for pt in p_path]
        pz = [pt[2] for pt in p_path]
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=4),
                name=f"Sabot Petal #{idx + 1} Discard",
            )
        )

    # 3. Supersonic Kinetic Long-Rod Penetrator Flight Path
    rod_traj = direct_res.get("trajectory", {})
    if rod_traj and "x_m" in rod_traj:
        fig.add_trace(
            go.Scatter3d(
                x=rod_traj["x_m"],
                y=rod_traj["y_m"],
                z=rod_traj["z_m"],
                mode="lines",
                line=dict(color="#2563EB", width=5),
                name=f"APFSDS Kinetic Rod (V0={direct_res.get('v0_mps', 1680):.0f} m/s)",
            )
        )

    # 4. Moving Target Tank Box
    tgt_r = float(direct_res.get("target_range_m", 2200.0))
    fcs = direct_res.get("fcs_solution", {})
    lead_m = float(fcs.get("kinematic_lead_distance_m", 15.0))
    off_x = float(direct_res.get("impact_offset_x_m", 0.0))
    off_z = float(direct_res.get("impact_offset_z_m", 0.0))

    fig.add_trace(
        go.Scatter3d(
            x=[tgt_r],
            y=[off_x],
            z=[1.8 + off_z],
            mode="markers+text",
            marker=dict(size=12, color="#DC2626" if direct_res.get("is_hit") else "#9CA3AF", symbol="square"),
            text=[f"🎯 Moving Target ({direct_res.get('engagement_status', 'STATUS')})"],
            textposition="top center",
            name="Moving Target Impact",
        )
    )

    fig.update_layout(
        title=f"3D APFSDS Sabot Separation & Direct-Fire Tank Gunnery — Result: {direct_res.get('engagement_status', 'COMPLETE')} | Lanz-Odermatt Pen: {direct_res.get('lo_penetration_normal_mm', 0.0):.0f}mm RHA",
        scene=dict(
            xaxis=dict(title="Downrange X (m)"),
            yaxis=dict(title="Crossrange Y (m)"),
            zaxis=dict(title="Altitude Z (m)"),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.72)", font=dict(color="#F9FAFB", size=10)),
    )
    return fig


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


# -------------------------------------------------------------------------
# ANTI-DRONE MULTI-SHOTGUN C-UAS VISUALIZATIONS
# -------------------------------------------------------------------------

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


def export_3d_html(fig: go.Figure, file_path: str | Path, include_plotlyjs: str = "cdn") -> str:
    """Save the interactive 3D figure to a standalone HTML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return str(path)


# Re-export complete 5-module C-UAS suite visualizations from vis_cuas
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
