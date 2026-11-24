"""Interactive Plotly Visualizations for Dynamic RL Fire Control, 3D Battlespace ML,
Recorded Drone-Track Replay, Hardware SWaP-C/OTM Stabilization & Duel Root-Cause Diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_rl_policy_benchmark_figure(rl_res: dict[str, Any]) -> go.Figure:
    """2-panel plot: ES training learning curve + paired policy vs heuristic benchmark bars."""
    hist = rl_res.get("training_history", [])
    rows = rl_res.get("benchmark_rows", [])
    sm = rl_res.get("summary", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=[
            "Evolution Strategies Training Curve (Reward vs Gen)",
            f"Head-to-Head Benchmark (Zero-Breach % | Uplift: {sm.get('uplift_zero_breach_pp', 0.0):+.1f} pp)",
        ],
    )

    if hist:
        gens = [int(h["generation"]) for h in hist]
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=[float(h["best_reward"]) for h in hist],
                mode="lines+markers",
                name="Best Evolved Policy Reward",
                line=dict(color="#10B981", width=3.2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=[float(h["mean_pop_reward"]) for h in hist],
                mode="lines+markers",
                name="Population Mean Reward",
                line=dict(color="#38BDF8", width=2.2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    if rows:
        names = [str(r["policy"]) for r in rows]
        zbs = [float(r["zero_breach_pct"]) for r in rows]
        krs = [float(r["kill_rate_pct"]) for r in rows]
        fig.add_trace(
            go.Bar(
                x=names,
                y=zbs,
                name="Zero-Breach Stop Rate (%)",
                marker_color=["#10B981" if "RL" in n else "#38BDF8" for n in names],
                text=[f"{v:.1f}%" for v in zbs],
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=krs,
                name="Mean Swarm Kill Rate (%)",
                marker_color="#F59E0B",
                opacity=0.82,
                text=[f"{v:.1f}%" for v in krs],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Training Generation", dtick=2, row=1, col=1)
    fig.update_yaxes(title_text="Curriculum Episodic Reward", row=1, col=1)
    fig.update_xaxes(title_text="Fire-Control Policy", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", range=[0, 112], row=1, col=2)
    fig.update_layout(
        title="Dynamic Real-Time Fire-Control ML Policy vs. Baseline Heuristics",
        barmode="group",
        height=470,
        margin=dict(l=40, r=30, t=80, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, x=0.0, font=dict(size=10)),
    )
    fig.update_annotations(font_size=12)
    return fig


def create_rl_action_heatmap_figure(rl_res: dict[str, Any]) -> go.Figure:
    """2-panel explainability heatmaps showing learned Burst Size & Hold-Fire Gate over Range x Ammo."""
    surf = rl_res.get("action_surface", {})
    ranges = surf.get("ranges_m", [])
    fracs = [f"{int(round(100 * f))}%" for f in surf.get("ammo_fractions", [])]
    burst_z = surf.get("burst_grid", [])
    hold_z = surf.get("hold_fire_prob_grid", [])

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            "Learned Burst Size (Barrels/Shot)",
            "Learned Hold-Fire Wait Gate Probability (%)",
        ],
    )

    if ranges and fracs and burst_z:
        fig.add_trace(
            go.Heatmap(
                z=burst_z,
                x=ranges,
                y=fracs,
                colorscale="Viridis",
                text=burst_z,
                texttemplate="%{text:.0f}b",
                colorbar=dict(x=0.44, len=0.82, title="Barrels"),
            ),
            row=1,
            col=1,
        )
    if ranges and fracs and hold_z:
        fig.add_trace(
            go.Heatmap(
                z=hold_z,
                x=ranges,
                y=fracs,
                colorscale="RdBu_r",
                zmin=0,
                zmax=100,
                text=hold_z,
                texttemplate="%{text:.0f}%",
                colorbar=dict(x=1.01, len=0.82, title="Hold %"),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Target Slant Range (m)", row=1, col=1)
    fig.update_yaxes(title_text="Magazine Ready Share", row=1, col=1)
    fig.update_xaxes(title_text="Target Slant Range (m)", row=1, col=2)
    fig.update_yaxes(title_text="Magazine Ready Share", row=1, col=2)
    fig.update_layout(
        title="Neural Fire-Control Policy State-Action Map (Range × Magazine Level)",
        height=440,
        margin=dict(l=55, r=65, t=75, b=50),
    )
    fig.update_annotations(font_size=12)
    return fig


def create_battlespace_coevolution_figure(bs_res: dict[str, Any]) -> go.Figure:
    """2-panel 3D battlespace co-evolution chart: Nash Payoff Matrix + Best Site Top-Down Turret Layout."""
    M = np.asarray(bs_res.get("payoff_matrix", [[]]), dtype=float)
    b_names = [str(b["name"])[:24] for b in bs_res.get("blue_pool", [])]
    r_names = [str(r["name"])[:24] for r in bs_res.get("red_pool", [])]
    best_b = bs_res.get("best_blue_battlespace", {})
    turrets = best_b.get("turrets", [])
    sm = bs_res.get("summary", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            f"Multi-Turret vs Mixed Swarm Payoff (Nash V*={sm.get('nash_value', 0.0):.2f})",
            f"Nash-Optimal Layout: {best_b.get('geometry_type', '')} + {best_b.get('outer_layer_type', 'none')}",
        ],
    )

    if M.size > 0 and b_names and r_names:
        fig.add_trace(
            go.Heatmap(
                z=M,
                x=r_names,
                y=b_names,
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                text=np.round(M, 1),
                texttemplate="%{text}",
                colorbar=dict(x=0.44, len=0.82, title="Utility"),
            ),
            row=1,
            col=1,
        )

    th = np.linspace(0.0, 2.0 * np.pi, 90)
    fig.add_trace(
        go.Scatter(
            x=36.0 * np.cos(th),
            y=36.0 * np.sin(th),
            mode="lines",
            line=dict(color="#3B82F6", dash="dash", width=1.8),
            name="Protected Perimeter (36 m)",
        ),
        row=1,
        col=2,
    )
    if turrets:
        tx = [float(t["x_m"]) for t in turrets]
        ty = [float(t["y_m"]) for t in turrets]
        r_eng = float(best_b.get("weapon_max_range_m", 65.0))
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="markers+text",
                text=[str(t["id"]) for t in turrets],
                textposition="top center",
                marker=dict(size=12, symbol="diamond", color="#10B981", line=dict(color="#FFFFFF", width=1)),
                name=f"Evolved Turrets (Rmax={r_eng:.0f} m)",
            ),
            row=1,
            col=2,
        )
        # Draw boresight vectors
        bx_lines: list[float | None] = []
        by_lines: list[float | None] = []
        for t in turrets:
            az_r = np.radians(float(t.get("boresight_az_deg", 0.0)))
            bx_lines.extend([float(t["x_m"]), float(t["x_m"]) + 28.0 * float(np.cos(az_r)), None])
            by_lines.extend([float(t["y_m"]), float(t["y_m"]) + 28.0 * float(np.sin(az_r)), None])
        fig.add_trace(
            go.Scatter(
                x=bx_lines,
                y=by_lines,
                mode="lines",
                line=dict(color="#38BDF8", width=1.5),
                name="Sector Boresight",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Red Heterogeneous Package", tickangle=-22, row=1, col=1)
    fig.update_xaxes(title_text="East X (m)", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="North Y (m)", row=1, col=2)
    fig.update_layout(
        title=f"3D Battlespace Co-Evolution Over `{sm.get('terrain_preset', 'terrain')}` DEM",
        height=490,
        margin=dict(l=90, r=35, t=80, b=95),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, x=0.52, font=dict(size=10)),
    )
    fig.update_annotations(font_size=12)
    return fig


def create_recorded_track_replay_figure(track_res: dict[str, Any]) -> go.Figure:
    """2-panel figure: 3D recorded drone tracks + intercept events, and kinematic/P_kill strip chart."""
    paths = track_res.get("track_paths", {})
    ts = track_res.get("time_series", [])
    events = track_res.get("sample_events", [])
    sm = track_res.get("summary", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"secondary_y": True}]],
        horizontal_spacing=0.08,
        subplot_titles=[
            f"3D Recorded Track Intercept Replay ({sm.get('coord_mode', 'local_enu')})",
            "Measured Speed, Lateral G & Instantaneous Shot P_kill(t)",
        ],
    )

    palette = ["#38BDF8", "#F59E0B", "#EC4899", "#A78BFA"]
    for idx, (tid, p) in enumerate(paths.items()):
        fig.add_trace(
            go.Scatter3d(
                x=p["x_m"],
                y=p["y_m"],
                z=p["z_m"],
                mode="lines",
                name=f"Track {tid}",
                line=dict(color=palette[idx % len(palette)], width=5),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[1.5],
            mode="markers",
            marker=dict(size=8, color="#10B981", symbol="diamond"),
            name="Turret Origin (0,0)",
        ),
        row=1,
        col=1,
    )
    if events:
        hits = [e for e in events if e.get("killed")]
        if hits:
            fig.add_trace(
                go.Scatter3d(
                    x=[e["x_m"] for e in hits],
                    y=[e["y_m"] for e in hits],
                    z=[e["z_m"] for e in hits],
                    mode="markers",
                    marker=dict(size=7, color="#EF4444", symbol="x"),
                    name="Simulated Intercept",
                ),
                row=1,
                col=1,
            )

    if ts:
        tv = [r["t_s"] for r in ts]
        fig.add_trace(
            go.Scatter(x=tv, y=[r["inst_burst_pk_pct"] for r in ts], name="Instantaneous P_kill (%)", line=dict(color="#10B981", width=2.8)),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=tv, y=[r["speed_mps"] for r in ts], name="Measured Speed (m/s)", line=dict(color="#38BDF8", width=2.0, dash="dash")),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=tv, y=[r["lat_g"] for r in ts], name="Lateral Evasion Load (G)", line=dict(color="#F59E0B", width=2.2)),
            row=1,
            col=2,
            secondary_y=True,
        )

    fig.update_layout(
        title=f"Recorded Track Closed-Loop Twin ({sm.get('zero_breach_pct', 0.0):.1f}% Zero-Breach | Kill Rate {sm.get('mean_kill_rate_pct', 0.0):.1f}%)",
        scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Alt (m)"),
        height=500,
        margin=dict(l=30, r=45, t=80, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20, x=0.0, font=dict(size=10)),
    )
    fig.update_xaxes(title_text="Recording Timestamp (s)", row=1, col=2)
    fig.update_yaxes(title_text="P_kill (%) / Speed (m/s)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Lateral Evasion (G)", row=1, col=2, secondary_y=True)
    fig.update_annotations(font_size=12)
    return fig


def create_swap_c_and_otm_figure(swap_res: dict[str, Any], otm_res: dict[str, Any]) -> go.Figure:
    """2-panel hardware chart: Sub-assembly Mass Donut + OTM Gyro Stabilization P_kill(R) curves."""
    mb = swap_res.get("mass_breakdown", {})
    curves = otm_res.get("curves", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        horizontal_spacing=0.10,
        subplot_titles=[
            f"SWaP-C Mass Budget ({swap_res.get('summary', {}).get('total_mass_kg', 0.0):.1f} kg Total)",
            f"On-the-Move P_kill(R) vs Gyro Grade ({otm_res.get('carrier_profile', '')[:28]})",
        ],
    )

    if mb:
        fig.add_trace(
            go.Pie(
                labels=list(mb.keys()),
                values=list(mb.values()),
                hole=0.45,
                textinfo="label+percent",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    style_map = {
        "static_ideal": ("Static Pedestal (Ideal)", "#10B981", "solid"),
        "tactical_fog_gyro": ("Tactical FOG Gyro (32 dB)", "#38BDF8", "solid"),
        "standard_mems_gyro": ("Standard MEMS Gyro (18 dB)", "#F59E0B", "dash"),
        "unstabilized": ("Unstabilized Hard-Mount", "#EF4444", "dot"),
    }
    all_ranges: list[float] = []
    for m_key, (lbl, col, dash_s) in style_map.items():
        pts = curves.get(m_key, [])
        if pts:
            all_ranges.extend(float(p["range_m"]) for p in pts)
            fig.add_trace(
                go.Scatter(
                    x=[p["range_m"] for p in pts],
                    y=[p["pk_pct"] for p in pts],
                    mode="lines+markers",
                    name=lbl,
                    line=dict(color=col, width=2.6, dash=dash_s),
                ),
                row=1,
                col=2,
            )

    x_lo = min(all_ranges) if all_ranges else 15.0
    x_hi = max(all_ranges) if all_ranges else 95.0
    fig.add_trace(
        go.Scatter(
            x=[x_lo, x_hi],
            y=[80.0, 80.0],
            mode="lines",
            name="R80 Target (80%)",
            line=dict(color="#94A3B8", width=1.6, dash="dash"),
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Slant Range (m)", row=1, col=2)
    fig.update_yaxes(title_text="Burst Kill Probability P_kill (%)", range=[0, 104], row=1, col=2)
    fig.update_layout(
        title="Hardware Prototyping SWaP-C Breakdown & On-the-Move Stabilization Envelope",
        height=470,
        margin=dict(l=30, r=30, t=80, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, x=0.48, font=dict(size=10)),
    )
    fig.update_annotations(font_size=12)
    return fig


def create_duel_root_cause_figure(duel_res: dict[str, Any]) -> go.Figure:
    """2-panel Beat-the-AI Duel outcome bars & Failure-Mode Root-Cause attribution breakdown."""
    mc = duel_res.get("matchup", {})
    fm = duel_res.get("failure_modes", {})
    verdict = duel_res.get("verdict", "DUEL RESULT")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            f"Duel Verdict: {verdict}",
            f"Leaker Failure-Mode Attribution ({fm.get('total_leakers', 0)} Total Breaches)",
        ],
    )

    kpis = ["Zero-Breach Stop (%)", "Swarm Kill Rate (%)", "Mean Burst Pk (%)", "Defense Utility"]
    vals = [
        float(mc.get("zero_breach_pct", 0.0)),
        float(mc.get("kill_rate_pct", 0.0)),
        float(mc.get("mean_burst_pk_pct", 0.0)),
        float(mc.get("blue_utility", 0.0)),
    ]
    fig.add_trace(
        go.Bar(
            x=kpis,
            y=vals,
            marker_color=["#10B981", "#38BDF8", "#F59E0B", "#A78BFA"],
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    reasons = ["Reload Window Gap", "Slew Bottleneck", "Evasive Tracking Lag", "Lethality / Spread"]
    rpcts = [
        float(fm.get("reload_gap_pct", 0.0)),
        float(fm.get("slew_bottleneck_pct", 0.0)),
        float(fm.get("tracking_evasion_miss_pct", 0.0)),
        float(fm.get("insufficient_lethality_spread_pct", 0.0)),
    ]
    fig.add_trace(
        go.Bar(
            y=reasons,
            x=rpcts,
            orientation="h",
            marker_color=["#EF4444", "#F59E0B", "#EC4899", "#38BDF8"],
            text=[f"{v:.1f}%" for v in rpcts],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(range=[0, 112], title_text="Percentage / Score", row=1, col=1)
    fig.update_xaxes(range=[0, 115], title_text="Share of Breaching Drones (%)", row=1, col=2)
    fig.update_layout(
        title="Red-vs-Blue Duel Performance & Automated Root-Cause Attribution",
        height=440,
        margin=dict(l=50, r=30, t=75, b=50),
    )
    fig.update_annotations(font_size=12)
    return fig
