"""Plotly 2D and 3D Interactive Visualizations for the Complete C-UAS Turret Engineering Suite."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------------------------------------------------------
# UPGRADE 1: CLOSED-LOOP FCS, KALMAN LEAD & EVASIVE MANEUVER VISUALIZATION
# -------------------------------------------------------------------------

def create_fcs_tracking_and_evasion_figure(fcs_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel visualization of evasive drone Kalman tracking/intercept and doctrine P_kill comparison."""
    sm = fcs_res.get("summary", {})
    tr = fcs_res.get("sample_track", {})
    pel = fcs_res.get("sample_pellets", {})
    comp = fcs_res.get("architecture_comparison", [])

    has_comp = len(comp) > 0
    fig = make_subplots(
        rows=1,
        cols=2 if has_comp else 1,
        horizontal_spacing=0.10 if has_comp else 0.0,
        column_widths=[0.62, 0.38] if has_comp else [1.0],
        subplot_titles=[
            "🎯 Evasive Maneuver Track, Noisy Lock<br>& Predicted Lead Intercept",
            "🛡️ Tracking & Architecture<br>Benchmark (P_kill %)",
        ][: 2 if has_comp else 1],
    )

    if tr:
        # Pellet cloud near intercept
        if pel and pel.get("y_offset"):
            py_abs = [tr["y_true_intercept"] + dy for dy in pel["y_offset"]]
            pz_abs = [tr["z_true_intercept"] + dz for dz in pel["z_offset"]]
            p_hit = pel["is_hit"]
            p_let = pel["is_lethal"]
            colors = [
                "#EF4444" if bool(l) else ("#F59E0B" if bool(h) else "rgba(99, 102, 241, 0.35)")
                for h, l in zip(p_hit, p_let)
            ]
            sizes = [9 if bool(h) else 4 for h in p_hit]
            fig.add_trace(
                go.Scatter(
                    x=py_abs,
                    y=pz_abs,
                    mode="markers",
                    marker=dict(size=sizes, color=colors),
                    name="Intercept Pellet Cloud",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=tr["y_meas"],
                y=tr["z_meas"],
                mode="markers",
                marker=dict(size=5, color="rgba(148, 163, 184, 0.55)"),
                name="Noisy Sensor Samples",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tr["y_true"],
                y=tr["z_true"],
                mode="lines",
                line=dict(color="#06B6D4", width=3),
                name="True Evasive Drone Trajectory",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[tr["y_true_intercept"]],
                y=[tr["z_true_intercept"]],
                mode="markers+text",
                marker=dict(size=13, color="#10B981", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                text=["True Drone @ TOF"],
                textposition="top right",
                name="True Drone at Impact",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[tr["y_aim_predicted"]],
                y=[tr["z_aim_predicted"]],
                mode="markers+text",
                marker=dict(size=14, color="#EF4444", symbol="cross-thin", line=dict(color="#EF4444", width=3)),
                text=["FCS Predicted Aim"],
                textposition="bottom left",
                name="FCS Lead Aimpoint",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if has_comp:
        doctrines = [c["doctrine"] for c in comp]
        pks = [c["kill_probability_pct"] for c in comp]
        bar_cols = ["#64748B", "#F59E0B", "#10B981"]
        fig.add_trace(
            go.Bar(
                x=doctrines,
                y=pks,
                marker=dict(color=[bar_cols[i % len(bar_cols)] for i in range(len(doctrines))]),
                text=[f"{v:.1f}%" for v in pks],
                textposition="outside",
                name="Doctrine P_kill (%)",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Kill Probability (%)", range=[0, 112], row=1, col=2)

    fig.update_xaxes(title_text="Crossrange Y (m)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude / Elevation Offset Z (m)", row=1, col=1)
    fig.update_layout(
        title=(
            f"📡 Closed-Loop FCS Intercept vs {sm.get('maneuver_profile', '').replace('_', ' ').title()} | "
            f"P_kill: {sm.get('kill_probability_pct', 0.0):.1f}% (Mean Lead Error: {sm.get('mean_lead_error_m', 0.0):.2f}m)"
        ),
        margin=dict(l=30, r=30, t=88, b=35),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=510,
    )
    fig.update_annotations(font_size=12)
    return fig


# -------------------------------------------------------------------------
# UPGRADE 2: SWARM SATURATION 3D HEMISPHERE & DOCTRINE CURVES
# -------------------------------------------------------------------------

def create_3d_swarm_hemisphere_figure(swarm_res: dict[str, Any]) -> go.Figure:
    """Create 3D WebGL hemispherical defense dome with incoming drone swarm tracks, kills & leaker breaches."""
    sm = swarm_res.get("summary", {})
    tracks = swarm_res.get("sample_wave_tracks", [])
    r_max = float(sm.get("max_engagement_range_m", 55.0))
    r_breach = float(sm.get("breach_keepout_radius_m", 12.0))

    fig = go.Figure()

    # Defended Keep-out ring & Max Engagement ring on ground Z=0
    th = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter3d(
            x=r_max * np.cos(th),
            y=r_max * np.sin(th),
            z=np.zeros_like(th),
            mode="lines",
            line=dict(color="rgba(59, 130, 246, 0.55)", width=3, dash="dash"),
            name=f"Max Engagement Envelope ({r_max:.0f}m)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=r_breach * np.cos(th),
            y=r_breach * np.sin(th),
            z=np.zeros_like(th),
            mode="lines",
            line=dict(color="#EF4444", width=4),
            name=f"Asset Keep-Out Breach Perimeter ({r_breach:.0f}m)",
        )
    )

    # Central C-UAS Turret
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[2.0],
            mode="markers+text",
            marker=dict(size=9, color="#38BDF8", symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
            text=["C-UAS Turret"],
            textposition="top center",
            name="C-UAS Multi-Shotgun Turret",
        )
    )

    # Drone tracks colored by outcome
    shown_killed = False
    shown_leaker = False
    for tr in tracks:
        is_killed = tr["status"] == "killed"
        col = "#10B981" if is_killed else "#EF4444"
        lbl = "Destroyed Drones" if is_killed else "Leaker (Asset Breach!)"
        show_leg = bool((is_killed and not shown_killed) or ((not is_killed) and not shown_leaker))
        if is_killed:
            shown_killed = True
        else:
            shown_leaker = True

        fig.add_trace(
            go.Scatter3d(
                x=[tr["x0"], tr["xe"]],
                y=[tr["y0"], tr["ye"]],
                z=[tr["z0"], tr["ze"]],
                mode="lines",
                line=dict(color=col, width=3.5),
                name=lbl,
                showlegend=show_leg,
                hoverinfo="text",
                hovertext=f"{tr['id']}: {tr['status']} @ R={tr['end_r_m']}m (t={tr['end_t_s']}s, {tr['bursts_absorbed']} bursts)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[tr["xe"]],
                y=[tr["ye"]],
                z=[tr["ze"]],
                mode="markers",
                marker=dict(size=6, color=col, symbol="x" if is_killed else "circle"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{tr['id']} {tr['status']}",
            )
        )

    fig.update_layout(
        title=(
            f"🐝 3D Swarm Saturation Dome ({sm.get('swarm_size', 8)} Drones) — "
            f"Zero-Leaker Rate: {sm.get('zero_leaker_wave_pct', 0.0):.1f}% | Mean Leakers: {sm.get('mean_leakers', 0.0):.2f}"
        ),
        scene=dict(
            xaxis=dict(title="East X (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title="North Y (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title="Altitude Z (m)", range=[0, r_max * 0.85], backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.85)),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#F8FAFC", size=10)),
        height=560,
    )
    return fig


def create_swarm_doctrine_tradeoff_figure(swarm_res: dict[str, Any]) -> go.Figure:
    """Plot Zero-Leaker Wave Success (%) and Mean Leakers vs Sub-Salvo Barrels Fired per Target."""
    sweep = swarm_res.get("doctrine_sweep", [])
    if not sweep:
        return go.Figure()

    bpb = [r["barrels_per_burst"] for r in sweep]
    z_pct = [r["zero_leaker_wave_pct"] for r in sweep]
    leakers = [r["mean_leakers"] for r in sweep]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=bpb,
            y=z_pct,
            mode="lines+markers",
            name="Zero-Leaker Defense Wave %",
            line=dict(color="#10B981", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bpb,
            y=leakers,
            mode="lines+markers",
            name="Mean Leakers per Wave",
            line=dict(color="#EF4444", width=2.5, dash="dash"),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Sub-Salvo Doctrine: Barrels Fired per Target Burst")
    fig.update_yaxes(title_text="Zero-Leaker Wave Success (%)", range=[0, 105], secondary_y=False)
    fig.update_yaxes(title_text="Mean Leakers per Wave", secondary_y=True)
    fig.update_layout(
        title="⚖️ Swarm Sub-Salvo Fire Allocation Trade-Off (Pattern Lethality vs Magazine Endurance & Reloads)",
        margin=dict(l=30, r=30, t=55, b=30),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=400,
    )
    return fig


# -------------------------------------------------------------------------
# UPGRADE 3: ADVANCED C-UAS PAYLOADS & AIRBURST VISUALIZATIONS
# -------------------------------------------------------------------------

def create_cuas_payload_range_comparison_figure(comp_res: dict[str, Any]) -> go.Figure:
    """Compare Kill Probability and Retained Striking Energy vs Range (20-110 m) across all 7 payload families."""
    curves = comp_res.get("curves", {})
    if not curves:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            "🎯 Standoff Drone Kill Probability P_kill (%) vs Engagement Range",
            "⚡ Retained Mean Sub-Projectile Energy (J) vs Range (Airburst, TSS & Flechette Advantage)",
        ],
    )
    palette = ["#64748B", "#06B6D4", "#8B5CF6", "#10B981", "#F59E0B", "#3B82F6", "#EF4444"]

    for idx, (p_name, data) in enumerate(curves.items()):
        col = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=data["distances_m"],
                y=data["pk_pct"],
                mode="lines+markers",
                name=p_name,
                line=dict(color=col, width=3),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data["distances_m"],
                y=data["sub_energy_j"],
                mode="lines+markers",
                name=p_name,
                line=dict(color=col, width=2.5, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Engagement Range (m)", row=2, col=1)
    fig.update_yaxes(title_text="P_kill (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Mean Sub-Projectile Energy (J)", row=2, col=1)
    fig.update_layout(
        title=f"💥 Standoff Payload Trade Study ({comp_res.get('n_barrels', 4)} Barrels vs {comp_res.get('drone_model', '').replace('_', ' ').title()})",
        margin=dict(l=30, r=30, t=55, b=25),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=560,
    )
    return fig


def create_airburst_standoff_cone_figure(ab_res: dict[str, Any]) -> go.Figure:
    """Create 3D WebGL view of programmable airburst carrier detonation points & ejected tungsten cone at target."""
    sm = ab_res.get("summary", {})
    bursts = ab_res.get("sample_burst_points", {})
    impacts = ab_res.get("sample_impacts", {})
    R_tgt = float(sm.get("target_distance_m", 75.0))

    fig = go.Figure()

    # Muzzle to Burst lines & Burst flash stars
    bx_list = bursts.get("x", [])
    by_list = bursts.get("y", [])
    bz_list = bursts.get("z", [])
    b_stats = bursts.get("status", [])

    for idx, (bx, by, bz, st_lbl) in enumerate(zip(bx_list, by_list, bz_list, b_stats)):
        col = "#F59E0B" if st_lbl == "OPTIMAL_AIRBURST" else "#EF4444"
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, float(bx)],
                y=[0.0, float(by)],
                z=[0.0, float(bz)],
                mode="lines",
                line=dict(color=col, width=3, dash="dash"),
                name="Carrier Trajectory" if idx == 0 else None,
                showlegend=bool(idx == 0),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(bx)],
                y=[float(by)],
                z=[float(bz)],
                mode="markers",
                marker=dict(size=10, color=col, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                name="Programmable Fuze Burst Point" if idx == 0 else None,
                showlegend=bool(idx == 0),
                hoverinfo="text",
                hovertext=f"Burst @ X={bx}m ({st_lbl})",
            )
        )

    # Drone target at R_tgt
    fig.add_trace(
        go.Scatter3d(
            x=[R_tgt],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=11, color="#06B6D4", symbol="square"),
            text=[f"Drone @ {R_tgt:.0f}m"],
            textposition="top center",
            name="Drone Target",
        )
    )

    # Ejected sub-projectile impact cone on target plane
    if impacts and impacts.get("y"):
        iy = np.asarray(impacts["y"])
        iz = np.asarray(impacts["z"])
        ihit = np.asarray(impacts["is_hit"], dtype=bool)
        ilet = np.asarray(impacts["is_lethal"], dtype=bool)
        ix = np.full_like(iy, R_tgt)
        cols = ["#EF4444" if bool(l) else ("#F59E0B" if bool(h) else "rgba(56, 189, 248, 0.55)") for h, l in zip(ihit, ilet)]
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="markers",
                marker=dict(size=4, color=cols),
                name="Sub-Projectile Cone @ Target",
            )
        )

    fig.update_layout(
        title=(
            f"💥 {sm.get('payload_name', '')} @ {R_tgt:.0f}m — "
            f"P_kill: {sm.get('kill_probability_pct', 0.0):.1f}% | Mean Hits: {sm.get('mean_hits_on_drone', 0.0):.1f}"
        ),
        scene=dict(
            xaxis=dict(title="Range X (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title="Crossrange Y (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title="Elevation Z (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            aspectratio=dict(x=2.6, y=1.0, z=1.0),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.6)),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#F8FAFC", size=10)),
        height=490,
    )
    return fig


# -------------------------------------------------------------------------
# UPGRADE 4: TURRET RECOIL, GIMBAL LOADS & PARETO ARRAY OPTIMIZER
# -------------------------------------------------------------------------

def create_turret_recoil_dynamics_figure(
    recoil_res: dict[str, Any],
    sweep_res: dict[str, Any] | None = None,
) -> go.Figure:
    """Create 2x2 interactive dashboard of recoil force history, gimbal walk, and ripple-delay load relief."""
    th = recoil_res.get("time_history", {})
    sm = recoil_res.get("summary", {})
    has_sweep = sweep_res is not None and len(sweep_res.get("delays_ms", [])) > 0

    fig = make_subplots(
        rows=2 if has_sweep else 1,
        cols=2,
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
        subplot_titles=[
            "🔩 Raw Breech Pulse vs<br>Transmitted Mount Reaction (kN)",
            "🎯 2-Axis Gimbal Climb (Pitch)<br>& Yaw Walk (mrad)",
            "📉 Peak Transmitted Shock vs<br>Ripple Delay Δt (kN)",
            "⚖️ Max Gimbal Walk vs Ripple Delay<br>(Sequential vs Balanced Pair)",
        ][: 4 if has_sweep else 2],
    )

    if th:
        t_ms = th["t_ms"]
        fig.add_trace(
            go.Scatter(x=t_ms, y=th["raw_breech_force_kn"], mode="lines", name="Raw Breech Force (kN)", line=dict(color="rgba(148, 163, 184, 0.55)", width=1.5, dash="dot")),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_ms, y=th["transmitted_force_kn"], mode="lines", name="Transmitted Mount Force (kN)", line=dict(color="#2563EB", width=3)),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=float(sm.get("mount_load_limit_kn", 18.0)),
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Mount Limit",
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=t_ms, y=th["pitch_climb_mrad"], mode="lines", name="Gimbal Pitch Climb (mrad)", line=dict(color="#F59E0B", width=3)),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=t_ms, y=th["yaw_walk_mrad"], mode="lines", name="Gimbal Yaw Walk (mrad)", line=dict(color="#10B981", width=2.5, dash="dash")),
            row=1,
            col=2,
        )

    if has_sweep and sweep_res is not None:
        delays = sweep_res["delays_ms"]
        fig.add_trace(
            go.Scatter(x=delays, y=sweep_res["seq_peak_force_kn"], mode="lines+markers", name="Sequential Ripple Force", line=dict(color="#EF4444", width=2.5)),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=delays, y=sweep_res["pair_peak_force_kn"], mode="lines+markers", name="Balanced Pair Force", line=dict(color="#10B981", width=3)),
            row=2,
            col=1,
        )
        fig.add_hline(y=float(sweep_res.get("mount_load_limit_kn", 18.0)), line_dash="dash", line_color="#EF4444", row=2, col=1)

        fig.add_trace(
            go.Scatter(x=delays, y=sweep_res["seq_max_walk_mrad"], mode="lines+markers", name="Sequential Max Walk", line=dict(color="#F59E0B", width=2.5)),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=delays, y=sweep_res["pair_max_walk_mrad"], mode="lines+markers", name="Balanced Pair Max Walk", line=dict(color="#06B6D4", width=3)),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Inter-Barrel Ripple Delay Δt (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Inter-Barrel Ripple Delay Δt (ms)", row=2, col=2)
        fig.update_yaxes(title_text="Peak Mount Force (kN)", row=2, col=1)
        fig.update_yaxes(title_text="Max Gimbal Walk (mrad)", row=2, col=2)

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Reaction Force (kN)", row=1, col=1)
    fig.update_yaxes(title_text="Angular Deflection (mrad)", row=1, col=2)

    fig.update_layout(
        title=(
            f"🔩 Turret Recoil & Gimbal Shock Dynamics — Peak Mount Load: {sm.get('peak_transmitted_mount_force_kn', 0.0):.1f} kN "
            f"(Limit: {sm.get('mount_load_limit_kn', 18.0):.1f} kN | Margin: {sm.get('structural_safety_margin_pct', 0.0):+.1f}%)"
        ),
        margin=dict(l=30, r=30, t=85, b=35),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(17, 24, 39, 0.75)", font=dict(color="#F9FAFB", size=9)),
        height=660 if has_sweep else 400,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_antidrone_optimizer_pareto_figure(opt_res: dict[str, Any]) -> go.Figure:
    """Create Pareto frontier scatter of Peak Mount Force (kN) vs Drone Kill Probability P_kill (%)."""
    cands = opt_res.get("candidates", [])
    pareto = opt_res.get("pareto_frontier", [])
    best_feas = opt_res.get("best_feasible")
    limit_kn = float(opt_res.get("mount_load_limit_kn", 18.0))

    fig = go.Figure()
    layout_cols = {
        "circular": "#2563EB",
        "concentric": "#10B981",
        "honeycomb": "#F59E0B",
        "grid": "#8B5CF6",
        "fan_horizontal": "#06B6D4",
    }

    by_layout: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        by_layout.setdefault(c["layout_type"], []).append(c)

    for l_type, grp in by_layout.items():
        col = layout_cols.get(l_type, "#94A3B8")
        fig.add_trace(
            go.Scatter(
                x=[c["peak_transmitted_force_kn"] for c in grp],
                y=[c["kill_probability_pct"] for c in grp],
                mode="markers",
                marker=dict(
                    size=10,
                    color=col,
                    symbol=["circle" if bool(c["feasible"]) else "x" for c in grp],
                    line=dict(color="#FFFFFF", width=1),
                ),
                name=f"Layout: {l_type.title()}",
                text=[
                    f"{c['layout_type']} | {c['pointing_mode']} | {c['firing_mode']} (Δt={c['ripple_delay_ms']}ms)"
                    f"<br>P_kill={c['kill_probability_pct']}% | Lethal={c['mean_lethal_hits']} | Force={c['peak_transmitted_force_kn']}kN"
                    for c in grp
                ],
                hoverinfo="text",
            )
        )

    if pareto:
        fig.add_trace(
            go.Scatter(
                x=[c["peak_transmitted_force_kn"] for c in pareto],
                y=[c["kill_probability_pct"] for c in pareto],
                mode="lines",
                line=dict(color="#FACC15", width=2.5, dash="dot"),
                name="Pareto Efficiency Frontier",
            )
        )

    if best_feas is not None:
        fig.add_trace(
            go.Scatter(
                x=[best_feas["peak_transmitted_force_kn"]],
                y=[best_feas["kill_probability_pct"]],
                mode="markers+text",
                marker=dict(size=16, color="#10B981", symbol="star", line=dict(color="#FFFFFF", width=2)),
                text=[f"Optimal Feasible ({best_feas['layout_type']})"],
                textposition="bottom left",
                name="Selected Optimal Configuration",
            )
        )

    fig.add_vline(x=limit_kn, line_dash="dash", line_color="#EF4444", annotation_text=f"Mount Structural Limit ({limit_kn:.1f} kN)")

    fig.update_layout(
        title="🏆 Multi-Objective Array Optimizer: Structural Reaction Load (kN) vs Drone Kill Probability (%)",
        xaxis_title="Peak Transmitted Mount Shock Force (kN)",
        yaxis_title="Drone Kill Probability P_kill (%)",
        yaxis=dict(range=[-3, 108]),
        margin=dict(l=30, r=30, t=55, b=30),
        legend=dict(x=0.01, y=0.02, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=500,
    )
    return fig


# -------------------------------------------------------------------------
# UPGRADE 5: ELEVATED-FIRE 3-DOF PELLET FALLOUT & SURFACE DANGER ZONE (SDZ)
# -------------------------------------------------------------------------

def create_3d_cuas_fallout_footprint_figure(fallout_res: dict[str, Any]) -> go.Figure:
    """Create 3D WebGL ballistic fallout trajectories climbing to apogee and descending to ground Z=0."""
    sm = fallout_res.get("summary", {})
    trajs = fallout_res.get("trajectories_3d", [])
    gi = fallout_res.get("ground_impacts", {})

    fig = go.Figure()
    color_map = {
        "safe_low_energy": "#10B981",
        "eye_hazard": "#F59E0B",
        "skin_penetration": "#EF4444",
        "incapacitating": "#DC2626",
        "iatg_lethal": "#991B1B",
    }

    shown_safe = False
    shown_haz = False
    for tr in trajs:
        hcls = tr.get("hazard_class", "safe_low_energy")
        col = color_map.get(hcls, "#3B82F6")
        is_safe = hcls == "safe_low_energy"
        show_leg = bool((is_safe and not shown_safe) or ((not is_safe) and not shown_haz))
        if is_safe:
            shown_safe = True
        else:
            shown_haz = True

        fig.add_trace(
            go.Scatter3d(
                x=tr["x"],
                y=tr["y"],
                z=tr["z"],
                mode="lines",
                line=dict(color=col, width=2.2),
                name="Safe Terminal Arcs (<0.06 J/mm²)" if is_safe else "Eye / Skin Hazard Arcs",
                showlegend=show_leg,
            )
        )

    # Ground impact footprint points on Z=0
    if gi and gi.get("x_m"):
        ix = gi["x_m"]
        iy = gi["y_m"]
        ih = gi["hazard_class"]
        ke_l = gi["energy_j"]
        se_l = gi["specific_energy_j_mm2"]
        pts_col = [color_map.get(c, "#3B82F6") for c in ih]
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=[0.0] * len(ix),
                mode="markers",
                marker=dict(size=3.5, color=pts_col, opacity=0.75),
                name="Ground Impact Footprint (Z=0)",
                hoverinfo="text",
                hovertext=[f"X={x:.0f}m, Y={y:.1f}m | E={e:.2f}J ({se:.3f} J/mm²) [{h}]" for x, y, e, se, h in zip(ix, iy, ke_l, se_l, ih)],
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[float(sm.get("turret_height_m", 2.0))],
            mode="markers+text",
            marker=dict(size=8, color="#38BDF8", symbol="diamond"),
            text=["C-UAS Turret"],
            textposition="top center",
            name="Turret Origin",
        )
    )

    fig.update_layout(
        title=(
            f"🌐 3D Elevated-Fire Pellet Fallout Trajectories (El: {sm.get('firing_elevation_deg', 35.0):.0f}°) — "
            f"99% SDZ Range: {sm.get('max_fallout_range_99_m', 0.0):.0f}m | Apogee: {sm.get('max_apogee_m', 0.0):.0f}m"
        ),
        scene=dict(
            xaxis=dict(title="Downrange X (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title="Crossrange Y (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title="Altitude Z (m)", backgroundcolor="#0F172A", gridcolor="#1E293B", showbackground=True),
            aspectratio=dict(x=2.2, y=1.0, z=0.8),
            camera=dict(eye=dict(x=-1.4, y=-1.6, z=0.7)),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#F8FAFC", size=10)),
        height=550,
    )
    return fig


def create_2d_cuas_fallout_safety_map_figure(
    fallout_res: dict[str, Any],
    load_comp_res: dict[str, Any] | None = None,
) -> go.Figure:
    """Create 2-panel top-down Surface Danger Zone (SDZ) scatter map and multi-load fallout range span comparison."""
    sm = fallout_res.get("summary", {})
    gi = fallout_res.get("ground_impacts", {})
    has_comp = load_comp_res is not None and len(load_comp_res.get("records", [])) > 0

    fig = make_subplots(
        rows=1,
        cols=2 if has_comp else 1,
        horizontal_spacing=0.12 if has_comp else 0.0,
        column_widths=[0.55, 0.45] if has_comp else [1.0],
        subplot_titles=[
            f"🗺️ Top-Down Ground SDZ Footprint<br>99% Range: {sm.get('max_fallout_range_99_m', 0.0):.0f}m",
            "📏 Downrange SDZ Span Comparison<br>Across Shotgun C-UAS Loads",
        ][: 2 if has_comp else 1],
    )

    color_map = {
        "safe_low_energy": "#10B981",
        "eye_hazard": "#F59E0B",
        "skin_penetration": "#EF4444",
        "incapacitating": "#DC2626",
        "iatg_lethal": "#991B1B",
    }

    if gi and gi.get("x_m"):
        ix = np.asarray(gi["x_m"])
        iy = np.asarray(gi["y_m"])
        ih = gi["hazard_class"]
        ke_l = gi["energy_j"]
        cols = [color_map.get(c, "#3B82F6") for c in ih]
        fig.add_trace(
            go.Scatter(
                x=ix,
                y=iy,
                mode="markers",
                marker=dict(size=6, color=cols, opacity=0.80),
                name="Ground Pellet Impacts",
                hoverinfo="text",
                hovertext=[f"X={x:.0f}m, Y={y:.1f}m | {e:.2f} J [{h}]" for x, y, e, h in zip(ix, iy, ke_l, ih)],
            ),
            row=1,
            col=1,
        )
        # Turret origin marker
        fig.add_trace(
            go.Scatter(x=[0.0], y=[0.0], mode="markers+text", marker=dict(size=10, color="#38BDF8", symbol="triangle-right"), text=["Turret (0,0)"], textposition="top right", name="Turret"),
            row=1,
            col=1,
        )
        r99 = float(sm.get("max_fallout_range_99_m", 100.0))
        th_arc = np.linspace(-math.pi / 4.0, math.pi / 4.0, 60) + math.radians(float(sm.get("firing_azimuth_deg", 0.0)))
        fig.add_trace(
            go.Scatter(
                x=r99 * np.cos(th_arc),
                y=r99 * np.sin(th_arc),
                mode="lines",
                line=dict(color="#EF4444", width=2, dash="dash"),
                name="99% SDZ Boundary",
            ),
            row=1,
            col=1,
        )

    if has_comp and load_comp_res is not None:
        recs = load_comp_res["records"]
        y_names = [r["ammunition_preset"] for r in recs]
        spans = [max(r["max_range_99_m"] - r["min_range_m"], 5.0) for r in recs]
        mins = [r["min_range_m"] for r in recs]
        bar_cols = ["#EF4444" if r["eye_hazard_pct"] > 50 else ("#F59E0B" if r["eye_hazard_pct"] > 5 else "#10B981") for r in recs]
        fig.add_trace(
            go.Bar(
                y=y_names,
                x=spans,
                base=mins,
                orientation="h",
                marker=dict(color=bar_cols, opacity=0.85),
                text=[f"{r['min_range_m']:.0f}–{r['max_range_99_m']:.0f}m (Eye Risk {r['eye_hazard_pct']:.0f}%)" for r in recs],
                textposition="inside",
                name="Fallout Range Band (m)",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Ground Fallout Range (m)", row=1, col=2)

    fig.update_xaxes(title_text="Downrange X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Crossrange Y (m)", row=1, col=1)
    fig.update_layout(
        title=(
            f"⚠️ Friendly Surface Danger Zone (SDZ) @ {sm.get('firing_elevation_deg', 35.0):.0f}° Elevation — "
            f"Ocular Risk: {sm.get('eye_hazard_pct', 0.0):.1f}% | Mean Impact Energy: {sm.get('mean_impact_energy_j', 0.0):.2f} J"
        ),
        margin=dict(l=30, r=30, t=88, b=35),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=490,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_antidrone_effective_range_figure(env_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel visualization of P_kill(R) envelope (with R80/R50 thresholds) and lethality vs range."""
    recs = env_res.get("records", [])
    comp = env_res.get("comparison_curves", [])
    sm = env_res.get("summary", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.11,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        subplot_titles=[
            "Effective Range Curve: P_kill(R) & Thresholds",
            "Terminal Lethality & Velocity Decay vs Range",
        ],
    )

    if recs:
        dists = [r["distance_m"] for r in recs]
        pks = [r["kill_probability_pct"] for r in recs]
        mean_hits = [r["mean_hits_on_drone"] for r in recs]
        mean_let = [r["mean_lethal_hits"] for r in recs]
        vels = [r["mean_v_impact_mps"] for r in recs]

        fig.add_trace(
            go.Scatter(
                x=dists,
                y=pks,
                mode="lines+markers",
                name=f"Active ({sm.get('ammunition_name', 'Active Load')})",
                line=dict(color="#10B981", width=3.5),
                marker=dict(size=8, color="#10B981"),
            ),
            row=1,
            col=1,
        )

        palette = ["#3B82F6", "#F59E0B", "#EC4899", "#8B5CF6"]
        for c_idx, c_item in enumerate(comp):
            c_recs = c_item.get("records", [])
            if c_recs:
                fig.add_trace(
                    go.Scatter(
                        x=[r["distance_m"] for r in c_recs],
                        y=[r["kill_probability_pct"] for r in c_recs],
                        mode="lines+markers",
                        name=f"{c_item.get('ammunition_name', c_item.get('ammunition_preset', 'Alt'))} (R80={c_item.get('r80_label', 'N/A')})",
                        line=dict(color=palette[c_idx % len(palette)], width=2.2, dash="dot"),
                        marker=dict(size=6),
                    ),
                    row=1,
                    col=1,
                )

        fig.add_hline(y=80.0, line_dash="dash", line_color="#EF4444", annotation_text="R80 (80%)", row=1, col=1)
        fig.add_hline(y=50.0, line_dash="dot", line_color="#F59E0B", annotation_text="R50 (50%)", row=1, col=1)

        r80_m = sm.get("r80_m")
        if r80_m is not None:
            fig.add_vline(
                x=float(r80_m),
                line_dash="dash",
                line_color="#10B981",
                annotation_text=f"R80={sm.get('r80_label', f'{r80_m:.1f} m')}",
                annotation_position="top right",
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=dists,
                y=mean_hits,
                mode="lines+markers",
                name="Mean Total Hits",
                line=dict(color="#38BDF8", width=2.2),
            ),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dists,
                y=mean_let,
                mode="lines+markers",
                name="Mean Lethal Hits",
                line=dict(color="#EF4444", width=2.8),
            ),
            row=1,
            col=2,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=dists,
                y=vels,
                mode="lines+markers",
                name="Striking Velocity (m/s)",
                line=dict(color="#F59E0B", width=2.0, dash="dash"),
            ),
            row=1,
            col=2,
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Slant Range (m)", row=1, col=1)
    fig.update_yaxes(title_text="Kill Probability P_kill (%)", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Slant Range (m)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Hits per Volley", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Striking Velocity (m/s)", row=1, col=2, secondary_y=True)
    fig.update_layout(
        title=(
            f"Effective Engagement Envelope — R80: {sm.get('r80_label', 'N/A')} | "
            f"R50: {sm.get('r50_label', 'N/A')} ({sm.get('drone_name', 'UAS')})"
        ),
        margin=dict(l=30, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, x=0.0, font=dict(size=10)),
        height=480,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_antidrone_range_speed_heatmap_figure(hm_res: dict[str, Any]) -> go.Figure:
    """Create 2D Speed x Range P_kill (%) engagement contour heatmap with R80 overlay."""
    d_vals = hm_res.get("distance_stations_m", [])
    v_vals = hm_res.get("speed_stations_mps", [])
    z_mat = hm_res.get("pk_matrix", [])
    r80_rows = hm_res.get("r80_by_speed_m", [])
    sm = hm_res.get("summary", {})

    fig = go.Figure()
    if d_vals and v_vals and z_mat:
        text_mat = [[f"{val:.0f}%" for val in row] for row in z_mat]
        fig.add_trace(
            go.Heatmap(
                x=[f"{d:.0f} m" for d in d_vals],
                y=[f"{v:.0f} m/s" for v in v_vals],
                z=z_mat,
                text=text_mat,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#111827"),
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                colorbar=dict(title="P_kill (%)"),
                hovertemplate="Range: %{x}<br>Drone Speed: %{y}<br>P_kill: %{z:.1f}%<extra></extra>",
            )
        )

    r80_note = ", ".join(f"{r['drone_velocity_mps']:.0f} m/s→{r['r80_label']}" for r in r80_rows[:4])
    fig.update_layout(
        title=(
            f"Speed x Range Engagement Envelope — {sm.get('viable_80pct_share', 0.0):.0f}% Cells >= 80% P_kill"
            + (f" ({r80_note})" if r80_note else "")
        ),
        xaxis_title="Target Slant Range (m)",
        yaxis_title="Crossing Drone Speed (m/s)",
        margin=dict(l=40, r=30, t=70, b=40),
        height=440,
    )
    return fig
