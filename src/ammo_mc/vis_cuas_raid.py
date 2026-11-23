"""Plotly 3D Animated Replay & Timeline Figures for the Unified Cooperative C-UAS Tactical Raid."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_unified_tactical_raid_3d_figure(
    raid_res: dict[str, Any],
    terrain_dem: Any | None = None,
) -> go.Figure:
    """Create interactive 3D animated cooperative multi-turret C-UAS raid arena."""
    sm = raid_res.get("summary", {})
    turrets = raid_res.get("turrets", [])
    obstacles = raid_res.get("obstacles", [])
    nfzs = raid_res.get("nfz_zones", [])
    frames_data = raid_res.get("frames", [])
    asset_r = float(sm.get("asset_radius_m", 40.0))

    fig = go.Figure()

    # 1. Static Terrain Surface or Flat Ground Datum
    if terrain_dem is not None:
        fig.add_trace(
            go.Surface(
                x=terrain_dem.x_m,
                y=terrain_dem.y_m,
                z=terrain_dem.z_grid_m,
                colorscale="Earth",
                opacity=0.72,
                showscale=False,
                name="Terrain DEM",
            )
        )
    else:
        g_lim = max(asset_r + 120.0, 180.0)
        gx = np.linspace(-g_lim, g_lim, 12)
        gy = np.linspace(-g_lim, g_lim, 12)
        fig.add_trace(
            go.Surface(
                x=gx,
                y=gy,
                z=np.zeros((12, 12)),
                colorscale=[[0, "#0F172A"], [1, "#1E293B"]],
                opacity=0.65,
                showscale=False,
                name="Ground Apron",
            )
        )

    # 2. Static 3D Building Obstacles
    for idx, obs in enumerate(obstacles):
        ox, oy = float(obs["x_m"]), float(obs["y_m"])
        z0 = float(terrain_dem.elevation_at(ox, oy)) if terrain_dem is not None else 0.0
        h = float(obs["height_m"])
        hw, hl = 0.5 * float(obs["width_m"]), 0.5 * float(obs["length_m"])
        vx = [ox - hw, ox + hw, ox + hw, ox - hw, ox - hw, ox + hw, ox + hw, ox - hw]
        vy = [oy - hl, oy - hl, oy + hl, oy + hl, oy - hl, oy - hl, oy + hl, oy + hl]
        vz = [z0, z0, z0, z0, z0 + h, z0 + h, z0 + h, z0 + h]
        fig.add_trace(
            go.Mesh3d(
                x=vx,
                y=vy,
                z=vz,
                i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                color="#64748B",
                opacity=0.82,
                name="Buildings" if idx == 0 else obs["name"],
                showlegend=bool(idx == 0),
            )
        )

    # 3. Protected Perimeter Ring & NFZ Rings on Ground
    th = np.linspace(0.0, 2.0 * math.pi, 64)
    p_x = [float(asset_r * math.cos(a)) for a in th]
    p_y = [float(asset_r * math.sin(a)) for a in th]
    p_z = [
        (float(terrain_dem.elevation_at(px, py)) + 1.0) if terrain_dem is not None else 1.0
        for px, py in zip(p_x, p_y)
    ]
    fig.add_trace(
        go.Scatter3d(
            x=p_x,
            y=p_y,
            z=p_z,
            mode="lines",
            line=dict(color="#3B82F6", width=5),
            name="Protected Perimeter",
        )
    )

    for idx, nfz in enumerate(nfzs):
        nx, ny, nr = float(nfz["x_m"]), float(nfz["y_m"]), float(nfz["radius_m"])
        n_x = [float(nx + nr * math.cos(a)) for a in th]
        n_y = [float(ny + nr * math.sin(a)) for a in th]
        n_z = [
            (float(terrain_dem.elevation_at(px, py)) + 1.5) if terrain_dem is not None else 1.5
            for px, py in zip(n_x, n_y)
        ]
        fig.add_trace(
            go.Scatter3d(
                x=n_x,
                y=n_y,
                z=n_z,
                mode="lines",
                line=dict(color="#EF4444", width=4, dash="dot"),
                name="Friendly NFZ" if idx == 0 else nfz["name"],
                showlegend=bool(idx == 0),
            )
        )

    # 4. Turret Static Bases
    if turrets:
        txs = [float(t["x_m"]) for t in turrets]
        tys = [float(t["y_m"]) for t in turrets]
        tzs = [
            (float(terrain_dem.elevation_at(float(t["x_m"]), float(t["y_m"]))) if terrain_dem is not None else 0.0)
            + float(t.get("z_m", 3.5))
            for t in turrets
        ]
        fig.add_trace(
            go.Scatter3d(
                x=txs,
                y=tys,
                z=tzs,
                mode="markers+text",
                marker=dict(size=7, color="#38BDF8", symbol="diamond"),
                text=[t["id"] for t in turrets],
                textposition="top center",
                name="C-UAS Turrets",
            )
        )

    def _frame_dynamic_traces(fr: dict[str, Any]) -> tuple[go.Scatter3d, go.Scatter3d, go.Scatter3d]:
        drones = fr.get("drones", [])
        color_map = {
            "undetected": "#94A3B8",
            "tracked": "#F59E0B",
            "killed": "#10B981",
            "breach": "#EF4444",
        }
        sym_map = {
            "undetected": "circle",
            "tracked": "circle",
            "killed": "x",
            "breach": "diamond",
        }
        tr_drones = go.Scatter3d(
            x=[d["x_m"] for d in drones],
            y=[d["y_m"] for d in drones],
            z=[d["z_m"] for d in drones],
            mode="markers+text",
            marker=dict(
                size=[7 if d["status"] in ("tracked", "breach") else 5 for d in drones],
                color=[("#FB923C" if (d["status"] == "tracked" and d.get("is_fiber_optic")) else color_map.get(d["status"], "#F59E0B")) for d in drones],
                symbol=[sym_map.get(d["status"], "circle") for d in drones],
            ),
            text=[d["id"] for d in drones],
            textfont=dict(size=9),
            hovertext=[
                f"{d['id']} ({'Fiber-Optic' if d.get('is_fiber_optic') else 'RF'}): {d['status'].upper()}"
                for d in drones
            ],
            name="Swarm Drones",
        )

        ax_pts: list[float | None] = []
        ay_pts: list[float | None] = []
        az_pts: list[float | None] = []
        for ts in fr.get("turrets", []):
            ax_pts.extend([ts["x_m"], ts["aim_x"], None])
            ay_pts.extend([ts["y_m"], ts["aim_y"], None])
            az_pts.extend([ts["z_m"], ts["aim_z"], None])
        tr_aim = go.Scatter3d(
            x=ax_pts or [0.0],
            y=ay_pts or [0.0],
            z=az_pts or [0.0],
            mode="lines",
            line=dict(color="#38BDF8", width=4),
            name="Turret Aim Vectors",
        )

        bx_pts: list[float | None] = []
        by_pts: list[float | None] = []
        bz_pts: list[float | None] = []
        for bm in fr.get("fire_beams", []):
            bx_pts.extend([bm["x0"], bm["x1"], None])
            by_pts.extend([bm["y0"], bm["y1"], None])
            bz_pts.extend([bm["z0"], bm["z1"], None])
        tr_beams = go.Scatter3d(
            x=bx_pts or [None],
            y=by_pts or [None],
            z=bz_pts or [None],
            mode="lines",
            line=dict(color="#FACC15", width=6),
            name="Active Shot Salvo",
        )
        return tr_drones, tr_aim, tr_beams

    dyn_start = len(fig.data)
    if frames_data:
        t0_d, t0_a, t0_b = _frame_dynamic_traces(frames_data[0])
    else:
        t0_d = go.Scatter3d(x=[], y=[], z=[], mode="markers", name="Swarm Drones")
        t0_a = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Turret Aim Vectors")
        t0_b = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Active Shot Salvo")
    fig.add_trace(t0_d)
    fig.add_trace(t0_a)
    fig.add_trace(t0_b)

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, Any]] = []
    for fr in frames_data:
        fn = f"{fr['t_s']:.1f}s"
        td, ta, tb = _frame_dynamic_traces(fr)
        frames.append(
            go.Frame(
                name=fn,
                data=[td, ta, tb],
                traces=[dyn_start, dyn_start + 1, dyn_start + 2],
            )
        )
        slider_steps.append(
            {
                "args": [[fn], {"frame": {"duration": 180, "redraw": True}, "mode": "immediate"}],
                "label": fn,
                "method": "animate",
            }
        )
    fig.frames = frames

    fig.update_layout(
        title=(
            f"⚔️ Unified 3D Cooperative Raid — {sm.get('site_name', '')} | "
            f"Zero-Breach: {sm.get('zero_breach_pct', 0.0):.1f}% | Kill Rate: {sm.get('mean_kill_rate_pct', 0.0):.1f}%"
        ),
        scene=dict(
            xaxis_title="East X (m)",
            yaxis_title="North Y (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.96,
                "buttons": [
                    {
                        "label": "▶ Play Raid",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 220, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "x": 0.14,
                "len": 0.82,
                "y": -0.02,
                "currentvalue": {"prefix": "Raid Clock: "},
            }
        ]
        if slider_steps
        else [],
        height=640,
        margin=dict(l=10, r=10, t=72, b=45),
    )
    return fig


def create_raid_timeline_and_heat_figure(raid_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel telemetry figure: turret barrel thermal load and cumulative swarm attrition."""
    ts_list = raid_res.get("timeline_series", [])
    sm = raid_res.get("summary", {})
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=[
            "🔥 Cooperative Turret Barrel Temperature vs Cook-Off Limit (220°C)",
            "🎯 Raid Swarm Attrition, Active Threats & Cumulative Ammunition Expended",
        ],
    )
    if ts_list:
        t_v = [r["t_s"] for r in ts_list]
        fig.add_trace(
            go.Scatter(
                x=t_v,
                y=[r["max_temp_c"] for r in ts_list],
                mode="lines",
                line=dict(color="#EF4444", width=2.5),
                name="Peak Turret Barrel Temp (°C)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_v,
                y=[r["mean_temp_c"] for r in ts_list],
                mode="lines",
                line=dict(color="#F59E0B", width=2, dash="dash"),
                name="Mean Turret Barrel Temp (°C)",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=220.0, line_dash="dot", line_color="#DC2626", annotation_text="Cook-Off Limit", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=t_v,
                y=[r["kills_cum"] for r in ts_list],
                mode="lines+markers",
                line=dict(color="#10B981", width=2.5),
                name="Drones Neutralized (Cum)",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_v,
                y=[r["active_threats"] for r in ts_list],
                mode="lines",
                line=dict(color="#3B82F6", width=2),
                name="Active Inbound Threats",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_v,
                y=[r["breaches_cum"] for r in ts_list],
                mode="lines",
                line=dict(color="#EF4444", width=2),
                name="Perimeter Breaches (Cum)",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Barrel Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Drone Count", row=2, col=1)
    fig.update_xaxes(title_text="Raid Elapsed Time (s)", row=2, col=1)
    fig.update_layout(
        title=(
            f"📊 Raid Telemetry — Peak Temp: {sm.get('peak_barrel_temp_c', 0.0):.1f}°C | "
            f"Mean Rounds: {sm.get('mean_rounds_expended', 0.0):.1f} (${sm.get('cost_per_kill_usd', 0.0):.2f}/Kill)"
        ),
        height=560,
        margin=dict(l=40, r=30, t=75, b=40),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig
