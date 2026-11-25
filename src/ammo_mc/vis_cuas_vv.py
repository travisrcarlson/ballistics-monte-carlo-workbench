"""Plotly Scientific V&V, Uncertainty Quantification, Doppler Drag & Range DOE Visualizations."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def create_epistemic_uq_envelope_figure(uq_res: dict[str, Any]) -> go.Figure:
    """Plot P_kill(R) median with shaded 5th-95th percentile epistemic/aleatory UQ P-box ribbon."""
    rows = uq_res.get("band_rows", [])
    dists = [r["distance_m"] for r in rows]
    p05 = [r["pk_p05_pct"] for r in rows]
    p50 = [r["pk_median_pct"] for r in rows]
    p95 = [r["pk_p95_pct"] for r in rows]

    fig = go.Figure()
    if dists:
        fig.add_trace(
            go.Scatter(
                x=dists + dists[::-1],
                y=p95 + p05[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.22)",
                line=dict(color="rgba(0,0,0,0)"),
                name="90% Predictive UQ Envelope (P05–P95)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dists,
                y=p50,
                mode="lines+markers",
                name="Median P_kill (%)",
                line=dict(color="#0284c7", width=3),
                marker=dict(size=8),
            )
        )
    fig.add_hline(y=80.0, line_dash="dash", line_color="#16a34a", annotation_text="R80 Doctrinal Target (80%)")
    fig.update_layout(
        title=(
            f"Epistemic + Aleatory P_kill(R) UQ P-Box Envelope "
            f"(R80 CI: [{uq_res.get('r80_p05_m', '?')}, {uq_res.get('r80_p95_m', '?')}] m)"
        ),
        xaxis_title="Intercept Slant Range (m)",
        yaxis_title="Kill Probability (%)",
        yaxis=dict(range=[0, 105]),
        height=400,
        margin=dict(l=45, r=25, t=52, b=40),
    )
    return fig


def create_blind_holdout_validation_figure(holdout_res: dict[str, Any]) -> go.Figure:
    """Plot blind holdout dispersion sigma(R): observed vs nominal vs calibrated out-of-sample predictions."""
    rows = holdout_res.get("holdout_pattern_rows", [])
    dists = [r["range_m"] for r in rows]
    obs = [r["observed_sigma_m"] for r in rows]
    nom = [r["nominal_pred_sigma_m"] for r in rows]
    cal = [r["calibrated_pred_sigma_m"] for r in rows]

    av = holdout_res.get("asme_vv20_sigma", {})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dists, y=obs, mode="markers+lines", name="Observed Holdout σ (m)", marker=dict(size=10, color="#ef4444")))
    fig.add_trace(go.Scatter(x=dists, y=nom, mode="lines+markers", name="Nominal Uncalibrated σ (m)", line=dict(dash="dash", color="#94a3b8")))
    fig.add_trace(go.Scatter(x=dists, y=cal, mode="lines+markers", name="Calibrated Out-of-Sample σ (m)", line=dict(width=3, color="#10b981")))
    fig.update_layout(
        title=f"Blind Holdout Out-of-Sample Validation — ASME V&V 20: {av.get('verdict', 'N/A')} (AVM={av.get('area_validation_metric', 0.0):.4f} m)",
        xaxis_title="Holdout Slant Range (m)",
        yaxis_title="Rayleigh Pattern Sigma σ (m)",
        height=370,
        margin=dict(l=45, r=25, t=52, b=40),
    )
    return fig


def create_doppler_drag_fit_figure(doppler_res: dict[str, Any]) -> go.Figure:
    """Plot Doppler radar measured v(R) track against nominal and jointly fitted Cd/V0 trajectories."""
    st_rows = doppler_res.get("stations", [])
    rs = [r["range_m"] for r in st_rows]
    vo = [r["measured_v_mps"] for r in st_rows]
    vf = [r["fitted_v_mps"] for r in st_rows]
    vn = [r["nominal_v_mps"] for r in st_rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs, y=vo, mode="markers", name="Doppler Measured v(R)", marker=dict(size=9, color="#f59e0b")))
    fig.add_trace(go.Scatter(x=rs, y=vn, mode="lines", name="Nominal Sphere Drag (Cd 1.0x)", line=dict(dash="dash", color="#64748b")))
    fig.add_trace(go.Scatter(x=rs, y=vf, mode="lines", name=f"Fitted Cd ({doppler_res.get('pellet_cd_scale', 1.0):.3f}x)", line=dict(width=3, color="#3b82f6")))
    fig.update_layout(
        title=f"Doppler Radar Velocity Decay Calibration (V0={doppler_res.get('v0_fit_mps', 0.0):.1f} m/s, Cd={doppler_res.get('pellet_cd_scale', 1.0):.3f}×, RMSE={doppler_res.get('rmse_mps', 0.0):.2f} m/s)",
        xaxis_title="Range (m)",
        yaxis_title="Velocity (m/s)",
        height=370,
        margin=dict(l=45, r=25, t=52, b=40),
    )
    return fig


def create_experimental_benchmarks_figure(exp_res: dict[str, Any]) -> go.Figure:
    """Plot Bailey-Hiatt experimental Cd(M) agreement and tolerance margins across E-01..E-06."""
    bms = exp_res.get("benchmarks", [])
    e01 = next((b for b in bms if b["benchmark_id"] == "E-01"), None)
    fig = go.Figure()
    if e01 and "details" in e01:
        det = e01["details"]
        fig.add_trace(go.Scatter(x=det["mach"], y=det["exp_cd"], mode="markers", name="Bailey & Hiatt (1972) Experimental Cd", marker=dict(size=10, color="#ef4444")))
        fig.add_trace(go.Scatter(x=det["mach"], y=det["pred_cd"], mode="lines+markers", name="Predicted Compressible Cd(Re, M)", line=dict(width=3, color="#2563eb")))
        fig.update_layout(
            title=f"Experimental Benchmark E-01: Sphere Cd(M) vs Bailey & Hiatt Data (AVM={e01.get('avm', 0.0):.4f})",
            xaxis_title="Mach Number M",
            yaxis_title="Drag Coefficient Cd",
        )
    else:
        fig.add_trace(go.Bar(x=[b["benchmark_id"] for b in bms], y=[1.0 if b["passed"] else 0.0 for b in bms], name="Pass"))
        fig.update_layout(title="Experimental C-UAS Benchmarks (E-01..E-06)")
    fig.update_layout(height=360, margin=dict(l=45, r=25, t=50, b=40))
    return fig


def create_sprt_stopping_boundaries_figure(
    doe_res: dict[str, Any],
    live_sprt: dict[str, Any] | None = None,
) -> go.Figure:
    """Plot Wald Sequential Probability Ratio Test (SPRT) accept/reject decision funnel vs rounds fired."""
    bounds = doe_res.get("sprt_bounds", [])
    ns = [b["n_rounds"] for b in bounds]
    hi = [b["accept_h1_min_kills"] for b in bounds]
    lo = [b["reject_h0_max_kills"] for b in bounds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=hi, mode="lines", name=f"Accept Threshold (Pk ≥ {doe_res.get('sprt_p1', 0.85):.0%})", line=dict(color="#16a34a", width=3)))
    fig.add_trace(go.Scatter(x=ns, y=lo, mode="lines", name=f"Reject Threshold (Pk ≤ {doe_res.get('sprt_p0', 0.65):.0%})", line=dict(color="#dc2626", width=3)))
    if live_sprt and live_sprt.get("trajectory"):
        traj = live_sprt["trajectory"]
        fig.add_trace(
            go.Scatter(
                x=[t["n"] for t in traj],
                y=[t["cum_kills"] for t in traj],
                mode="lines+markers",
                name=f"Live Trials ({live_sprt.get('decision', 'CONTINUE')})",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=9, color="#facc15"),
            )
        )
    fig.update_layout(
        title="Wald SPRT Sequential Live-Fire Early-Stopping Funnel",
        xaxis_title="Cumulative Rounds Fired (n)",
        yaxis_title="Cumulative Drone Kills Required",
        height=350,
        margin=dict(l=45, r=25, t=50, b=40),
    )
    return fig


def create_stl_vulnerability_figure(stl_proj: dict[str, Any]) -> go.Figure:
    """Visualize ray-cast 3D STL CAD face obliquity angles and silhouette presented area."""
    import numpy as np
    from plotly.subplots import make_subplots

    y_edges = np.asarray(stl_proj["y_edges"], dtype=np.float64)
    z_edges = np.asarray(stl_proj["z_edges"], dtype=np.float64)
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    obliq = np.asarray(stl_proj["obliquity_grid"], dtype=np.float64)
    depth = np.asarray(stl_proj["depth_x_grid"], dtype=np.float64)
    depth_plot = np.where(np.isfinite(depth), depth, np.nan)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Ray-Cast Face Obliquity Angle θ (deg)", "LOS Penetration Depth Profile x (m)"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Heatmap(
            x=yc,
            y=zc,
            z=obliq,
            colorscale="Turbo",
            zmin=0.0,
            zmax=80.0,
            colorbar=dict(title="θ°", x=0.45),
            name="Obliquity (deg)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=yc,
            y=zc,
            z=depth_plot,
            colorscale="Viridis",
            colorbar=dict(title="X (m)", x=1.0),
            name="Depth X (m)",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=f"3D STL CAD Mesh Vulnerability — Area={stl_proj.get('presented_area_cm2', 0.0):.1f} cm² | Mean θ={stl_proj.get('mean_obliquity_deg', 0.0):.1f}°",
        height=390,
        margin=dict(l=45, r=35, t=55, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    return fig


def create_pattern_board_corners_overlay_figure(
    image_bytes: bytes,
    corners_px: list[tuple[float, float]] | None = None,
    detected_holes_px: list[tuple[float, float]] | None = None,
    title: str = "Pattern-Board Rectification & Hole Overlay",
) -> go.Figure:
    """Render uploaded pattern board photo overlaid with detected corner fiducials and pellet holes."""
    import io
    import numpy as np

    buf = io.BytesIO(image_bytes)
    try:
        from PIL import Image

        arr = np.asarray(Image.open(buf).convert("RGB"))
    except Exception:
        from matplotlib import image as mpimg

        buf.seek(0)
        raw = np.asarray(mpimg.imread(buf))
        if raw.max() <= 1.5:
            raw = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
        arr = raw[..., :3] if raw.ndim == 3 else np.stack([raw, raw, raw], axis=-1)

    h_px, w_px = arr.shape[:2]
    fig = go.Figure()
    fig.add_trace(go.Image(z=arr))
    if corners_px and len(corners_px) == 4:
        xs = [float(c[0]) for c in corners_px] + [float(corners_px[0][0])]
        ys = [float(c[1]) for c in corners_px] + [float(corners_px[0][1])]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="Rectification Corners",
                line=dict(color="#10b981", width=2.5),
                marker=dict(size=10, color="#10b981", symbol="square-open"),
            )
        )
    if detected_holes_px:
        fig.add_trace(
            go.Scatter(
                x=[float(p[0]) for p in detected_holes_px],
                y=[float(p[1]) for p in detected_holes_px],
                mode="markers",
                name=f"Detected Holes ({len(detected_holes_px)})",
                marker=dict(size=7, color="#ef4444", symbol="circle-open", line=dict(width=1.8)),
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, w_px], showgrid=False),
        yaxis=dict(range=[h_px, 0], scaleanchor="x", scaleratio=1, showgrid=False),
        height=400,
        margin=dict(l=30, r=25, t=48, b=30),
    )
    return fig


def create_salvo_brackets_and_window_figure(window_res: dict[str, Any]) -> go.Figure:
    """Plot closing-target Usable Window [R_min_safe, R_max], Salvo Brackets, and Ogive P_eff(R)."""
    from plotly.subplots import make_subplots

    st_rows = window_res.get("stations", [])
    sm = window_res.get("summary", {})
    rs = [r["range_m"] for r in st_rows]
    pk_s = [r["pk_shared_lower_pct"] for r in st_rows]
    pk_i = [r["pk_indep_upper_pct"] for r in st_rows]
    pk_1 = [r["pk_single_pct"] for r in st_rows]
    vs = [r["v_retained_mps"] for r in st_rows]
    peffs = [r["p_eff"] * 100.0 for r in st_rows]

    r_safe = float(sm.get("r_min_safe_m", 0.0))
    r_max_s = float(sm.get("r_max_shared_m", 0.0))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        subplot_titles=(
            f"Salvo Brackets (WB-F01/F02) & Usable Window [{r_safe:.0f}–{r_max_s:.0f} m]",
            f"Pellet Velocity & Ogive P_eff(v) (V50={sm.get('v50_mps', 0.0):.0f} m/s)",
        ),
        horizontal_spacing=0.12,
    )
    if rs:
        fig.add_trace(
            go.Scatter(
                x=rs + rs[::-1],
                y=pk_i + pk_s[::-1],
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.20)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Correlation Bracket (Shared..Indep)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=rs, y=pk_i, mode="lines", name="Upper Bound: Independent Aim", line=dict(dash="dash", color="#60a5fa")), row=1, col=1)
        fig.add_trace(go.Scatter(x=rs, y=pk_s, mode="lines+markers", name="Lower Bound: Shared Aim (WB-F02)", line=dict(width=3, color="#2563eb")), row=1, col=1)
        fig.add_trace(go.Scatter(x=rs, y=pk_1, mode="lines", name="Single-Shot Pk (WB-F01)", line=dict(dash="dot", color="#94a3b8")), row=1, col=1)

        if r_safe > 0.0:
            fig.add_vrect(x0=min(rs), x1=min(max(rs), r_safe), fillcolor="rgba(239, 68, 68, 0.14)", line_width=1, line_color="#ef4444", annotation_text="Wreck Keep-Out", row=1, col=1)
        if not sm.get("window_collapsed") and r_max_s > r_safe:
            fig.add_vrect(x0=r_safe, x1=r_max_s, fillcolor="rgba(16, 185, 129, 0.12)", line_width=1, line_color="#10b981", annotation_text="Usable Window", row=1, col=1)

        fig.add_trace(go.Scatter(x=rs, y=peffs, mode="lines+markers", name="Ogive P_eff(v) (%)", line=dict(width=2.5, color="#f59e0b")), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=rs, y=vs, mode="lines", name="Retained Velocity (m/s)", line=dict(dash="dash", color="#14b8a6")), row=1, col=2, secondary_y=True)

    fig.update_xaxes(title_text="Slant Range R (m)", row=1, col=1)
    fig.update_yaxes(title_text="Kill Probability (%)", range=[0, 105], row=1, col=1)
    fig.update_xaxes(title_text="Slant Range R (m)", row=1, col=2)
    fig.update_yaxes(title_text="Ogive P_eff(v) (%)", range=[0, 105], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Retained Velocity (m/s)", row=1, col=2, secondary_y=True)
    fig.update_layout(height=380, margin=dict(l=45, r=45, t=55, b=40))
    return fig


def create_dead_time_waterfall_figure(budget: dict[str, Any]) -> go.Figure:
    """Plot serial fire-control dead-time chain breakdown (`FC-T04`) + ballistic TOF."""
    stages = budget.get("stages_ms", {})
    labels = [k.replace("_ms", "").replace("_", " ").title() for k in stages.keys()] + ["Pellet TOF"]
    vals = [float(v) for v in stages.values()] + [float(budget.get("tof_s", 0.0)) * 1000.0]
    colors = ["#38bdf8"] * len(stages) + ["#f59e0b"]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f} ms" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"Serial Latency Chain: Dead Time = {budget.get('t_dead_ms', 0.0):.1f} ms + TOF = {budget.get('t_total_s', 0.0)*1000.0:.1f} ms (Miss ~ {budget.get('quadratic_maneuver_miss_m', 0.0):.3f} m)",
        xaxis_title="Latency Contribution (ms)",
        height=320,
        margin=dict(l=140, r=30, t=50, b=35),
    )
    return fig


def create_workbook_vv_matrix_figure(wb_vv: dict[str, Any]) -> go.Figure:
    """Plot pass-rate breakdown across deterministic workbook V&V oracle groups."""
    groups = wb_vv.get("summary", {}).get("groups", [])
    names = [g["group"] for g in groups]
    passed = [g["passed"] for g in groups]
    failed = [max(0, g["total"] - g["passed"]) for g in groups]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Passed", x=names, y=passed, marker_color="#16a34a"))
    fig.add_trace(go.Bar(name="Failed", x=names, y=failed, marker_color="#dc2626"))
    fig.update_layout(
        barmode="stack",
        title=f"C-UAS Workbook Oracle Matrix ({wb_vv.get('summary', {}).get('passed', 0)}/{wb_vv.get('summary', {}).get('total', 0)} PASS)",
        xaxis_title="Verification Domain",
        yaxis_title="Oracle Test Count",
        height=340,
        margin=dict(l=45, r=25, t=48, b=60),
    )
    return fig
