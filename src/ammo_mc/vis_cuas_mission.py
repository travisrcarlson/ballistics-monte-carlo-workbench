"""Interactive Plotly Visualizations for C-UAS Mission Kill Chain, Multi-Turret Site Defense,
3D Animated Intercept Replay, and Sustained Logistics / Economics.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_kill_chain_timeline_figure(kc_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel visualization of per-sensor detection ranges and the kill-chain time budget."""
    sm = kc_res.get("summary", {})
    mods = kc_res.get("modality_breakdown", [])
    stages = kc_res.get("timeline_stages", [])

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            "🛰️ Detection Range by Sensor<br>vs Engagement & Keep-Out Envelopes",
            "⏱️ Detection-to-Intercept<br>Time Budget (seconds)",
        ],
    )

    if mods:
        names = [m["sensor"] for m in mods]
        ranges = [float(m["nominal_range_m"]) for m in mods]
        colors = [
            ("#38BDF8" if m["enabled"] and m["nominal_range_m"] > 0 else "#EF4444")
            for m in mods
        ]
        fig.add_trace(
            go.Bar(
                y=names,
                x=ranges,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{r:.0f} m ({m['status']})" for r, m in zip(ranges, mods)],
                textposition="auto",
                name="Detection Range (m)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=float(sm.get("weapon_max_range_m", 55.0)),
            line_dash="dash",
            line_color="#10B981",
            annotation_text="Weapon R_max",
            row=1,
            col=1,
        )
        fig.add_vline(
            x=float(sm.get("keepout_breach_radius_m", 15.0)),
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="Keep-Out",
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Slant Detection Range (m)", row=1, col=1)

    if stages:
        st_names = [s["stage"] for s in stages]
        durs = [float(s["duration_s"]) for s in stages]
        starts = [float(s["start_s"]) for s in stages]
        st_cols = [s["color"] for s in stages]
        fig.add_trace(
            go.Bar(
                y=st_names,
                x=durs,
                base=starts,
                orientation="h",
                marker=dict(color=st_cols),
                text=[f"{d:.2f}s" for d in durs],
                textposition="auto",
                name="Phase Duration (s)",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Elapsed Time From First Lock (s)", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        title=(
            f"🛰️ C-UAS Kill Chain — {sm.get('drone_name', '')} | "
            f"First Shot: {sm.get('first_shot_range_m', 0.0):.1f} m | Cum. P_kill: {sm.get('cumulative_pk_pct', 0.0):.1f}%"
        ),
        margin=dict(l=35, r=30, t=88, b=45),
        height=440,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_multiturret_site_map_figure(site_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel top-down multi-turret perimeter map and 360-deg azimuth coverage diagram."""
    sm = site_res.get("summary", {})
    turrets = site_res.get("turrets", [])
    obstacles = site_res.get("obstacles", [])
    nfzs = site_res.get("nfz_zones", [])
    az_prof = site_res.get("azimuth_profile", [])
    tracks = site_res.get("sample_tracks", [])
    asset_r = float(sm.get("asset_radius_m", 40.0))
    w_range = float(sm.get("weapon_max_range_m", 60.0))

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.12,
        subplot_titles=[
            "🏰 Top-Down Perimeter, Buildings,<br>No-Fire Zones & Raid Tracks",
            "🧭 360° Active Turret Coverage<br>(Mutual Support vs Blind Spots)",
        ],
    )

    th = np.linspace(0.0, 2.0 * math.pi, 100)
    # Protected asset circle
    fig.add_trace(
        go.Scatter(
            x=[float(v) for v in (asset_r * np.cos(th))],
            y=[float(v) for v in (asset_r * np.sin(th))],
            mode="lines",
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.08)",
            line=dict(color="#3B82F6", width=2, dash="dash"),
            name="Protected Perimeter",
        ),
        row=1,
        col=1,
    )

    # Obstacles
    for idx, obs in enumerate(obstacles):
        ox, oy = float(obs["x_m"]), float(obs["y_m"])
        hw, hl = 0.5 * float(obs["width_m"]), 0.5 * float(obs["length_m"])
        fig.add_trace(
            go.Scatter(
                x=[ox - hw, ox + hw, ox + hw, ox - hw, ox - hw],
                y=[oy - hl, oy - hl, oy + hl, oy + hl, oy - hl],
                mode="lines",
                fill="toself",
                fillcolor="rgba(100, 116, 139, 0.48)",
                line=dict(color="#94A3B8", width=2),
                name="LOS Obstacles" if idx == 0 else obs["name"],
                showlegend=bool(idx == 0),
                hovertext=f"{obs['name']} (H={obs['height_m']}m)",
            ),
            row=1,
            col=1,
        )

    # Friendly No-Fire Zones
    for idx, nfz in enumerate(nfzs):
        nx, ny, nr = float(nfz["x_m"]), float(nfz["y_m"]), float(nfz["radius_m"])
        fig.add_trace(
            go.Scatter(
                x=[float(nx + nr * math.cos(a)) for a in th],
                y=[float(ny + nr * math.sin(a)) for a in th],
                mode="lines",
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.16)",
                line=dict(color="#EF4444", width=2, dash="dot"),
                name="No-Fire Zones (NFZ)" if idx == 0 else nfz["name"],
                showlegend=bool(idx == 0),
                hovertext=f"NFZ: {nfz['name']} (R={nr:.0f}m)",
            ),
            row=1,
            col=1,
        )

    # Turret engagement wedges + markers
    for idx, t in enumerate(turrets):
        tx, ty = float(t["x_m"]), float(t["y_m"])
        bs = float(t.get("boresight_az_deg", 0.0))
        sh = float(t.get("sector_half_deg", 180.0))
        tr = float(t.get("max_range_m", w_range))
        angs = np.radians(np.linspace(bs - sh, bs + sh, 36))
        wx = [tx] + [float(tx + tr * math.cos(a)) for a in angs] + [tx]
        wy = [ty] + [float(ty + tr * math.sin(a)) for a in angs] + [ty]
        fig.add_trace(
            go.Scatter(
                x=wx,
                y=wy,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.09)",
                line=dict(color="rgba(16, 185, 129, 0.55)", width=1.5),
                name="Turret Engagement Envelope" if idx == 0 else t["id"],
                showlegend=bool(idx == 0),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[tx],
                y=[ty],
                mode="markers+text",
                marker=dict(size=12, color="#38BDF8", symbol="diamond"),
                text=[t["id"]],
                textposition="top center",
                name="Turret Sites" if idx == 0 else t["id"],
                showlegend=bool(idx == 0),
            ),
            row=1,
            col=1,
        )

    # Sample raid tracks
    shown_k, shown_l = False, False
    for tr in tracks:
        is_k = tr["status"] == "killed"
        show_l = bool((is_k and not shown_k) or ((not is_k) and not shown_l))
        if is_k:
            shown_k = True
        else:
            shown_l = True
        fig.add_trace(
            go.Scatter(
                x=[tr["sx"], tr["ex"]],
                y=[tr["sy"], tr["ey"]],
                mode="lines+markers",
                line=dict(color="#10B981" if is_k else "#EF4444", width=2),
                marker=dict(size=6, symbol="x" if is_k else "circle"),
                name="Intercepted Raid" if is_k else "Perimeter Breach (Leaker)",
                showlegend=show_l,
                hovertext=f"{tr['id']}: {tr['status']} ({tr['killer']})",
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="East X (m)", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="North Y (m)", row=1, col=1)

    if az_prof:
        azs = [r["az_deg"] for r in az_prof]
        cnts = [int(r["active_turrets"]) for r in az_prof]
        bar_cols = ["#EF4444" if c == 0 else ("#F59E0B" if c == 1 else "#10B981") for c in cnts]
        fig.add_trace(
            go.Bar(
                x=azs,
                y=cnts,
                marker=dict(color=bar_cols),
                name="Active Turrets",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Threat Approach Azimuth (°)", row=1, col=2)
        fig.update_yaxes(title_text="Unobstructed Firing Turrets", dtick=1, row=1, col=2)

    fig.update_layout(
        title=(
            f"🏰 Multi-Turret Site Defense — {sm.get('site_name', '')} | "
            f"Coverage: {sm.get('perimeter_coverage_pct', 0.0):.1f}% | Mutual Support (≥2): {sm.get('overlapping_coverage_pct', 0.0):.1f}%"
        ),
        margin=dict(l=30, r=30, t=88, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(15, 23, 42, 0.78)", font=dict(color="#F8FAFC", size=10)),
        height=540,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_3d_animated_intercept_figure(replay_data: dict[str, Any]) -> go.Figure:
    """Create interactive Plotly 3D animated intercept replay with Play/Pause and time scrubber."""
    sm = replay_data.get("summary", {})
    barrels = replay_data.get("barrels", [])
    frames = replay_data.get("frames", [])
    if not frames:
        return go.Figure()

    f0 = frames[0]
    d_pos0 = f0["drone_pos"]
    aim0 = f0["aim_end"]

    # Base dynamic traces (0..4) updated on every frame + static turret mounts (5+)
    data: list[Any] = [
        go.Scatter3d(
            x=f0["drone_fuselage_lines"]["x"],
            y=f0["drone_fuselage_lines"]["y"],
            z=f0["drone_fuselage_lines"]["z"],
            mode="lines",
            line=dict(color="#F59E0B", width=5),
            name="Drone Fuselage (3D Attitude)",
        ),
        go.Scatter3d(
            x=f0["drone_rotor_lines"]["x"],
            y=f0["drone_rotor_lines"]["y"],
            z=f0["drone_rotor_lines"]["z"],
            mode="lines",
            line=dict(color="#38BDF8", width=4),
            name="Drone Rotors",
        ),
        go.Scatter3d(
            x=[0.0, aim0[0]],
            y=[0.0, aim0[1]],
            z=[0.0, aim0[2]],
            mode="lines+markers",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            marker=dict(size=6, color="#EF4444", symbol="cross"),
            name="FCS Predicted Lead Aim",
        ),
        go.Scatter3d(
            x=f0["fly_x"],
            y=f0["fly_y"],
            z=f0["fly_z"],
            mode="markers",
            marker=dict(size=4, color="#38BDF8", opacity=0.85),
            name="Shot-String Pellets (In Flight)",
        ),
        go.Scatter3d(
            x=f0["hit_x"],
            y=f0["hit_y"],
            z=f0["hit_z"],
            mode="markers",
            marker=dict(size=7, color="#EF4444", symbol="diamond"),
            name="Target Drone Impacts",
        ),
    ]

    # Static barrel muzzle markers
    if barrels:
        data.append(
            go.Scatter3d(
                x=[0.0] * len(barrels),
                y=[float(b["y_m"]) for b in barrels],
                z=[float(b["z_m"]) for b in barrels],
                mode="markers",
                marker=dict(size=5, color="#10B981", symbol="circle"),
                name="Turret Muzzles (X=0)",
            )
        )

    all_x = [float(f["drone_pos"][0]) for f in frames]
    all_y = [float(f["drone_pos"][1]) for f in frames]
    all_z = [float(f["drone_pos"][2]) for f in frames]
    x_max = max(max(all_x, default=35.0), float(sm.get("target_distance_m", 35.0))) + 10.0
    y_span = max(3.5, max((abs(v) for v in all_y), default=1.0) + 2.5)
    z_span = max(2.5, max((abs(v) for v in all_z), default=1.0) + 2.0)

    go_frames = []
    slider_steps = []
    for idx, fr in enumerate(frames):
        aim = fr["aim_end"]
        go_frames.append(
            go.Frame(
                name=str(idx),
                traces=[0, 1, 2, 3, 4],
                data=[
                    go.Scatter3d(x=fr["drone_fuselage_lines"]["x"], y=fr["drone_fuselage_lines"]["y"], z=fr["drone_fuselage_lines"]["z"]),
                    go.Scatter3d(x=fr["drone_rotor_lines"]["x"], y=fr["drone_rotor_lines"]["y"], z=fr["drone_rotor_lines"]["z"]),
                    go.Scatter3d(x=[0.0, aim[0]], y=[0.0, aim[1]], z=[0.0, aim[2]]),
                    go.Scatter3d(x=fr["fly_x"], y=fr["fly_y"], z=fr["fly_z"]),
                    go.Scatter3d(x=fr["hit_x"], y=fr["hit_y"], z=fr["hit_z"]),
                ],
            )
        )
        slider_steps.append(
            {
                "args": [[str(idx)], {"frame": {"duration": 75, "redraw": True}, "mode": "immediate"}],
                "label": f"{fr['t_ms']:.0f}ms ({fr['drone_status'][:4]} · {fr['cumulative_hits']} hits)",
                "method": "animate",
            }
        )

    fig = go.Figure(data=data, frames=go_frames)
    fig.update_layout(
        title=(
            f"🎬 3D Animated Intercept Replay — {sm.get('drone_name', '')} | "
            f"Shot String: {sm.get('shot_string_length_m', 0.0):.2f} m | "
            f"Final Hits: {sm.get('final_hits', 0)} ({sm.get('final_lethal_hits', 0)} lethal)"
        ),
        scene=dict(
            xaxis=dict(title="Downrange X (m)", range=[-1.0, x_max], backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title="Crossrange Y (m)", range=[-y_span, y_span], backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title="Vertical Z (m)", range=[-z_span, z_span], backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            camera=dict(eye=dict(x=1.4, y=1.35, z=0.65)),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.02,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 75, "redraw": True}, "fromcurrent": True}],
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
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "right"},
                "pad": {"b": 10, "t": 35},
                "len": 0.78,
                "x": 0.18,
                "y": 0.04,
                "steps": slider_steps,
            }
        ],
        margin=dict(l=10, r=10, t=55, b=35),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(15, 23, 42, 0.78)", font=dict(color="#F8FAFC", size=10)),
        height=600,
    )
    return fig


def create_logistics_thermal_and_cost_figure(log_res: dict[str, Any]) -> go.Figure:
    """Create 2x2 dashboard of multi-wave barrel heating, thermal walk, $/kill economics, and carried mass."""
    sm = log_res.get("summary", {})
    tl = log_res.get("thermal_timeline", {})
    econ = log_res.get("economics_comparison", [])

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.20,
        horizontal_spacing=0.14,
        subplot_titles=[
            "🌡️ Multi-Wave Barrel Temp (°C)<br>vs 220°C Cook-Off Limit",
            "🎯 Thermal Dispersion Walk (mrad)<br>& Cumulative Fired Rounds",
            "💵 Cost per Drone Neutralized ($/Kill)<br>Across 7 Payload Families",
            "🎒 Carried Ammo Mass per 10 Kills (kg)<br>vs Single-Salvo P_kill (%)",
        ],
    )

    if tl and tl.get("t_s"):
        ts = tl["t_s"]
        fig.add_trace(
            go.Scatter(x=ts, y=tl["barrel_temp_c"], mode="lines+markers", name="Barrel Temp (°C)", line=dict(color="#F97316", width=3)),
            row=1,
            col=1,
        )
        fig.add_hline(y=float(sm.get("cookoff_limit_c", 220.0)), line_dash="dash", line_color="#EF4444", annotation_text="220°C Cook-Off", row=1, col=1)
        fig.add_hline(y=120.0, line_dash="dot", line_color="#F59E0B", annotation_text="120°C Walk Threshold", row=1, col=1)
        fig.update_xaxes(title_text="Raid Timeline (s)", row=1, col=1)
        fig.update_yaxes(title_text="Barrel Temp (°C)", row=1, col=1)

        fig.add_trace(
            go.Scatter(x=ts, y=tl["thermal_walk_mrad"], mode="lines+markers", name="Thermal Walk (mrad)", line=dict(color="#A855F7", width=3)),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Raid Timeline (s)", row=1, col=2)
        fig.update_yaxes(title_text="Thermal Dispersion Walk (mrad)", row=1, col=2)

    if econ:
        names = [e["ammo_family"].split("(")[0].strip() for e in econ]
        cpks = [float(e["cost_per_kill_usd"]) for e in econ]
        masses = [float(e["ammo_mass_per_10_kills_kg"]) for e in econ]
        pks = [float(e["single_salvo_pk_pct"]) for e in econ]

        bar_cols = ["#10B981" if c < 10.0 else ("#F59E0B" if c < 40.0 else "#EF4444") for c in cpks]
        fig.add_trace(
            go.Bar(
                x=names,
                y=cpks,
                marker=dict(color=bar_cols),
                text=[f"${c:.2f}" for c in cpks],
                textposition="auto",
                name="$/Kill",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(tickangle=-22, row=2, col=1)
        fig.update_yaxes(title_text="Cost per Neutralized Drone ($)", row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=masses,
                y=pks,
                mode="markers+text",
                marker=dict(size=12, color="#38BDF8"),
                text=names,
                textposition="top center",
                name="Mass vs P_kill",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Carried Ammo Mass per 10 Kills (kg)", row=2, col=2)
        fig.update_yaxes(title_text="Single-Salvo P_kill (%)", range=[0, 108], row=2, col=2)

    fig.update_layout(
        title=(
            f"🔥 Sustained Logistics & $/Kill — {sm.get('ammo_name', '')} | "
            f"Peak: {sm.get('peak_barrel_temp_c', 0.0):.1f}°C ({sm.get('cookoff_risk', '')}) | "
            f"Mean Cost/Kill: ${sm.get('mean_cost_per_kill_usd', 0.0):.2f}"
        ),
        margin=dict(l=35, r=30, t=88, b=45),
        showlegend=False,
        height=690,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_terrain_dem_site_3d_figure(
    site_res: dict[str, Any],
    terrain_dem: Any,
) -> go.Figure:
    """Create 3D topographic DEM site defense figure showing ridge LOS masking & raid tracks."""
    sm = site_res.get("summary", {})
    turrets = site_res.get("turrets", [])
    obstacles = site_res.get("obstacles", [])
    tracks = site_res.get("sample_tracks", [])
    az_prof = site_res.get("azimuth_profile", [])
    asset_r = float(sm.get("asset_radius_m", 40.0))

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=terrain_dem.x_m,
            y=terrain_dem.y_m,
            z=terrain_dem.z_grid_m,
            colorscale="Earth",
            opacity=0.88,
            name="Terrain DEM (m)",
            colorbar=dict(title="Elev (m)", len=0.65, x=1.02),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=False)),
        )
    )

    # 3D Building Prisms anchored to ground elevation
    for idx, obs in enumerate(obstacles):
        ox, oy = float(obs["x_m"]), float(obs["y_m"])
        z0 = float(terrain_dem.elevation_at(ox, oy))
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
                opacity=0.85,
                name="LOS Buildings" if idx == 0 else obs["name"],
                showlegend=bool(idx == 0),
                hovertext=f"{obs['name']} (AGL={h:.1f}m)",
            )
        )

    # Elevated Turrets
    if turrets:
        txs = [float(t["x_m"]) for t in turrets]
        tys = [float(t["y_m"]) for t in turrets]
        tzs = [float(terrain_dem.elevation_at(float(t["x_m"]), float(t["y_m"]))) + float(t.get("z_m", 3.5)) for t in turrets]
        fig.add_trace(
            go.Scatter3d(
                x=txs,
                y=tys,
                z=tzs,
                mode="markers+text",
                marker=dict(size=7, color="#38BDF8", symbol="diamond"),
                text=[t["id"] for t in turrets],
                textposition="top center",
                name="Turret Sites (MSL)",
            )
        )

    # Terrain-masked azimuth alert markers at outer ring
    eval_r = min(asset_r + 0.30 * float(sm.get("weapon_max_range_m", 60.0)), 140.0)
    masked_rows = [r for r in az_prof if r.get("terrain_masked")]
    if masked_rows:
        mx = [eval_r * math.cos(math.radians(r["az_deg"])) for r in masked_rows]
        my = [eval_r * math.sin(math.radians(r["az_deg"])) for r in masked_rows]
        mz = [float(terrain_dem.elevation_at(px, py)) + 16.0 for px, py in zip(mx, my)]
        fig.add_trace(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                marker=dict(size=5, color="#F97316", symbol="x"),
                name=f"Terrain-Masked Sectors ({len(masked_rows)*5}°)",
            )
        )

    # 3D Incoming Raid Tracks
    shown_k, shown_l = False, False
    for tr in tracks:
        is_k = tr["status"] == "killed"
        show_l = bool((is_k and not shown_k) or ((not is_k) and not shown_l))
        if is_k:
            shown_k = True
        else:
            shown_l = True
        sz = float(tr.get("sz", float(terrain_dem.elevation_at(tr["sx"], tr["sy"])) + 20.0))
        ez = float(tr.get("ez", float(terrain_dem.elevation_at(tr["ex"], tr["ey"])) + 20.0))
        fig.add_trace(
            go.Scatter3d(
                x=[tr["sx"], tr["ex"]],
                y=[tr["sy"], tr["ey"]],
                z=[sz, ez],
                mode="lines+markers",
                line=dict(color="#10B981" if is_k else "#EF4444", width=4),
                marker=dict(size=4),
                name="Intercepted Track" if is_k else "Breach (Leaker)",
                showlegend=show_l,
            )
        )

    fig.update_layout(
        title=(
            f"⛰️ 3D DEM Terrain LOS Masking — {sm.get('terrain_source', terrain_dem.name)} | "
            f"Terrain Masked: {sm.get('terrain_masked_pct', 0.0):.1f}% | Relief: {sm.get('terrain_relief_m', 0.0):.1f} m"
        ),
        scene=dict(
            xaxis_title="East X (m)",
            yaxis_title="North Y (m)",
            zaxis_title="Elevation MSL (m)",
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=65, b=15),
        height=600,
    )
    return fig
