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


def create_sprt_stopping_boundaries_figure(doe_res: dict[str, Any]) -> go.Figure:
    """Plot Wald Sequential Probability Ratio Test (SPRT) accept/reject decision funnel vs rounds fired."""
    bounds = doe_res.get("sprt_bounds", [])
    ns = [b["n_rounds"] for b in bounds]
    hi = [b["accept_h1_min_kills"] for b in bounds]
    lo = [b["reject_h0_max_kills"] for b in bounds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=hi, mode="lines", name=f"Accept Threshold (Pk ≥ {doe_res.get('sprt_p1', 0.85):.0%})", line=dict(color="#16a34a", width=3)))
    fig.add_trace(go.Scatter(x=ns, y=lo, mode="lines", name=f"Reject Threshold (Pk ≤ {doe_res.get('sprt_p0', 0.65):.0%})", line=dict(color="#dc2626", width=3)))
    fig.update_layout(
        title="Wald SPRT Sequential Live-Fire Early-Stopping Funnel",
        xaxis_title="Cumulative Rounds Fired (n)",
        yaxis_title="Cumulative Drone Kills Required",
        height=350,
        margin=dict(l=45, r=25, t=50, b=40),
    )
    return fig
