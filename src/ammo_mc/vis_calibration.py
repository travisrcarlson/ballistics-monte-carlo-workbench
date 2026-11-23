"""Plotly Residual & Goodness-of-Fit Figures for Field-Test Calibration Lab."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_pattern_calibration_figure(cal_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel pattern-board scatter + fitted Rayleigh CDF & residual comparison."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=[
            f"🎯 Empirical Pellet Hits & POI Bias ({cal_res['n_pellets']} pellets)",
            f"📈 Rayleigh Radial CDF Fit (R²={cal_res['r2_cdf']:.3f}, KS p={cal_res['ks_pvalue']:.3f})",
        ],
    )
    by = float(cal_res["poi_bias_y_m"])
    bz = float(cal_res["poi_bias_z_m"])

    fig.add_trace(
        go.Scatter(
            x=cal_res["y_coords_m"],
            y=cal_res["z_coords_m"],
            mode="markers",
            marker=dict(size=7, color="#38BDF8", opacity=0.82),
            name="Empirical Hits",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[by],
            y=[bz],
            mode="markers",
            marker=dict(size=12, color="#EF4444", symbol="cross"),
            name="Empirical POI Mean",
        ),
        row=1,
        col=1,
    )
    th = np.linspace(0.0, 2.0 * math.pi, 90)
    for mult, lbl, col in [
        (float(cal_res["r50_fit_m"]), "Fitted R50", "#10B981"),
        (float(cal_res["r90_fit_m"]), "Fitted R90", "#F59E0B"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[float(by + mult * math.cos(a)) for a in th],
                y=[float(bz + mult * math.sin(a)) for a in th],
                mode="lines",
                line=dict(color=col, width=2, dash="dash"),
                name=lbl,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="Horizontal Offset Y (m)", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Vertical Offset Z (m)", row=1, col=1)

    rs = cal_res["r_sorted_m"]
    fig.add_trace(
        go.Scatter(
            x=rs,
            y=cal_res["ecdf"],
            mode="lines+markers",
            line=dict(color="#38BDF8", width=2),
            marker=dict(size=5),
            name="Empirical CDF",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=rs,
            y=cal_res["pred_fit_cdf"],
            mode="lines",
            line=dict(color="#10B981", width=2.5),
            name=f"Fitted (k={cal_res['calibrated_spread_factor']:.2f}×)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=rs,
            y=cal_res["pred_nom_cdf"],
            mode="lines",
            line=dict(color="#94A3B8", width=1.8, dash="dot"),
            name=f"Nominal ({cal_res['nominal_choke']})",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Radial Distance from POI Center r (m)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Fraction Enclosed", range=[0, 1.05], row=1, col=2)
    fig.update_layout(height=470, margin=dict(l=35, r=30, t=70, b=40))
    return fig


def create_chronograph_calibration_figure(cal_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel chronograph histogram and MV-vs-temperature / residual diagnostic figure."""
    vels = cal_res["velocities_mps"]
    resids = cal_res["residuals_mps"]
    has_temp = bool(cal_res.get("has_temp_fit"))

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=[
            f"📏 Muzzle Velocity Spread (Mean={cal_res['mean_mv_mps']:.1f}, SD={cal_res['sd_mps']:.2f} m/s)",
            "🌡️ Propellant Temp Fit V(T) [FM 6-40]" if has_temp else "📉 Per-Shot Residual Errors (m/s)",
        ],
    )
    fig.add_trace(
        go.Histogram(x=vels, nbinsx=min(12, max(5, len(vels) // 2)), marker_color="#3B82F6", name="Measured Shots"),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Muzzle Velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Shot Count", row=1, col=1)

    if has_temp:
        temps = cal_res["temps_c"]
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=vels,
                mode="markers",
                marker=dict(size=8, color="#38BDF8"),
                name="Chrono Readings",
            ),
            row=1,
            col=2,
        )
        tx = np.linspace(min(temps), max(temps), 25)
        ty = float(cal_res["v_std_fit_mps"]) + float(cal_res["dv_dt_mps_per_c"]) * (tx - float(cal_res["std_temp_c"]))
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="lines",
                line=dict(color="#10B981", width=2.5),
                name=f"Fit dV/dT={cal_res['dv_dt_mps_per_c']:+.3f} m/s/°C",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Propellant Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Muzzle Velocity (m/s)", row=1, col=2)
    else:
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(resids) + 1)),
                y=resids,
                marker_color=["#10B981" if abs(r) <= cal_res["sd_mps"] else "#F59E0B" for r in resids],
                name="Residuals (m/s)",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Shot Index", row=1, col=2)
        fig.update_yaxes(title_text="Residual (m/s)", row=1, col=2)

    fig.update_layout(height=440, margin=dict(l=35, r=30, t=70, b=40))
    return fig


def create_drag_calibration_figure(cal_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel Nominal vs Calibrated exterior curve & station residual plot."""
    stations = cal_res["stations"]
    rs = [s["range_m"] for s in stations]
    has_drop = bool(cal_res.get("has_drop"))

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=[
            f"🚀 Range Drop Fit (i_cal={cal_res['calibrated_form_factor']:.3f}, R²={cal_res['r2_fit']:.4f})"
            if has_drop
            else f"🚀 Velocity Decay Fit (i_cal={cal_res['calibrated_form_factor']:.3f})",
            "📊 Calibrated Model Residuals",
        ],
    )
    if has_drop:
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["drop_obs_m"] for s in stations],
                mode="markers",
                marker=dict(size=9, color="#38BDF8"),
                name="Field Measured Drop",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["drop_nom_m"] for s in stations],
                mode="lines",
                line=dict(color="#94A3B8", width=2, dash="dot"),
                name=f"Nominal (i={cal_res['nominal_form_factor']:.2f})",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["drop_cal_m"] for s in stations],
                mode="lines",
                line=dict(color="#10B981", width=2.5),
                name=f"Calibrated (i={cal_res['calibrated_form_factor']:.3f})",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Trajectory Drop (m)", row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=rs,
                y=[s["drop_resid_cal_m"] for s in stations],
                marker_color="#10B981",
                name="Drop Residual (Obs - Cal)",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Drop Residual (m)", row=1, col=2)
    else:
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["vel_obs_mps"] for s in stations],
                mode="markers",
                marker=dict(size=9, color="#38BDF8"),
                name="Measured Velocity",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["vel_nom_mps"] for s in stations],
                mode="lines",
                line=dict(color="#94A3B8", width=2, dash="dot"),
                name=f"Nominal (i={cal_res['nominal_form_factor']:.2f})",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rs,
                y=[s["vel_cal_mps"] for s in stations],
                mode="lines",
                line=dict(color="#10B981", width=2.5),
                name=f"Calibrated (i={cal_res['calibrated_form_factor']:.3f})",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Remaining Velocity (m/s)", row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=rs,
                y=[s["vel_resid_cal_mps"] for s in stations],
                marker_color="#10B981",
                name="Velocity Residual (m/s)",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Velocity Residual (m/s)", row=1, col=2)

    fig.update_xaxes(title_text="Downrange Station (m)", row=1, col=1)
    fig.update_xaxes(title_text="Downrange Station (m)", row=1, col=2)
    fig.update_layout(height=450, margin=dict(l=35, r=30, t=70, b=40))
    return fig
