"""Interactive Plotly Visualizations for Adversarial Red-vs-Blue Co-Evolution & ML Optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_coevolution_trajectory_figure(adv_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel arms-race convergence plot and GP 5-fold cross-validation parity plot."""
    hist = adv_res.get("generation_history", [])
    cv = adv_res.get("cv_metrics", {})
    actual = adv_res.get("actual_utilities", [])
    preds = cv.get("oof_predictions", [])
    sm = adv_res.get("summary", {})

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.11,
        subplot_titles=[
            "Adversarial Co-Evolution: Defense Floor vs Generations",
            f"GP Surrogate Out-of-Sample Fit (5-Fold CV R²={sm.get('cv_r2', 0.0):.3f}, RMSE={sm.get('cv_rmse', 0.0):.1f})",
        ],
    )

    if hist:
        gens = [int(h["generation"]) for h in hist]
        nash_v = [float(h["nash_value"]) for h in hist]
        evolved_w = [float(h["best_blue_worst_case"]) for h in hist]
        base_w = [float(h["baseline_blue_worst_case"]) for h in hist]

        fig.add_trace(
            go.Scatter(
                x=gens,
                y=nash_v,
                mode="lines+markers",
                name="Nash Mixed Equilibrium V*",
                line=dict(color="#10B981", width=3.5),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=evolved_w,
                mode="lines+markers",
                name="Best Evolved Blue Worst-Case",
                line=dict(color="#38BDF8", width=2.8),
                marker=dict(size=7),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=base_w,
                mode="lines+markers",
                name="Initial Baseline Blue Worst-Case",
                line=dict(color="#F59E0B", width=2.2, dash="dash"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

    if actual and preds and len(actual) == len(preds):
        act_arr = np.asarray(actual, dtype=float)
        prd_arr = np.asarray(preds, dtype=float)
        err_arr = np.abs(act_arr - prd_arr)
        fig.add_trace(
            go.Scatter(
                x=act_arr,
                y=prd_arr,
                mode="markers",
                marker=dict(
                    size=8,
                    color=err_arr,
                    colorscale="Viridis_r",
                    opacity=0.82,
                    line=dict(color="#FFFFFF", width=0.6),
                ),
                name="MC Matchup (Actual vs CV Pred)",
                hovertemplate="Actual MC: %{x:.1f}<br>GP CV Pred: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        lo = min(0.0, float(np.min(act_arr)) - 3.0)
        hi = max(100.0, float(np.max(act_arr)) + 3.0)
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="#EF4444", width=2.0, dash="dash"),
                name="Ideal y = x",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Co-Evolution Generation", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="Defense Utility Score (0-100)", range=[0, 102], row=1, col=1)
    fig.update_xaxes(title_text="True Competing MC Utility", range=[0, 102], row=1, col=2)
    fig.update_yaxes(title_text="Out-of-Sample GP Predicted Utility", range=[0, 102], row=1, col=2)
    fig.update_layout(
        title=f"Adversarial Arms Race & Bayesian ML Surrogate ({sm.get('total_matchups', 0)} Matchups Evaluated)",
        margin=dict(l=35, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, x=0.0, font=dict(size=10)),
        height=480,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_nash_payoff_heatmap_figure(adv_res: dict[str, Any]) -> go.Figure:
    """Create annotated 2D Blue Platforms vs Red Threat Doctrines Nash payoff heatmap."""
    b_pool = adv_res.get("blue_pool", [])
    r_pool = adv_res.get("red_pool", [])
    mat = adv_res.get("payoff_matrix", [])
    nash = adv_res.get("nash", {})
    p_blue = nash.get("blue_strategy", [0.0] * len(b_pool))
    q_red = nash.get("red_strategy", [0.0] * len(r_pool))

    y_labels = [
        f"{'★ ' if idx < len(p_blue) and p_blue[idx] > 0.02 else ''}{b.get('name', f'B{idx+1}')}"
        for idx, b in enumerate(b_pool)
    ]
    x_labels = [
        f"{'★ ' if idx < len(q_red) and q_red[idx] > 0.02 else ''}{r.get('name', f'R{idx+1}')}"
        for idx, r in enumerate(r_pool)
    ]

    fig = go.Figure()
    if mat and y_labels and x_labels:
        text_mat = [[f"{val:.0f}" for val in row] for row in mat]
        fig.add_trace(
            go.Heatmap(
                x=x_labels,
                y=y_labels,
                z=mat,
                text=text_mat,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#111827"),
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                colorbar=dict(title="Defense Score"),
                hovertemplate="Blue: %{y}<br>Red Threat: %{x}<br>Blue Defense Score: %{z:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=(
            f"Red-vs-Blue Payoff Matrix — Nash Equilibrium V* = {nash.get('nash_value', 0.0):.1f} "
            f"(★ = Positive Nash Support Strategy)"
        ),
        xaxis_title="Adversarial Red Drone Doctrines",
        yaxis_title="Blue Protection Systems (Shotgun Turrets)",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickangle=-25),
        margin=dict(l=45, r=30, t=70, b=90),
        height=max(440, 220 + 28 * len(y_labels)),
    )
    return fig


def create_ml_feature_importance_figure(adv_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel permutation feature importance splitting Blue levers from Red threat drivers."""
    rows = adv_res.get("feature_importance", [])
    blue_rows = [r for r in rows if r.get("side") == "Blue"][:8]
    red_rows = [r for r in rows if r.get("side") == "Red"][:8]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.18,
        subplot_titles=[
            "Top Blue Protection Design Levers",
            "Top Red Adversarial Threat Drivers",
        ],
    )

    if blue_rows:
        fig.add_trace(
            go.Bar(
                y=[r["feature"].replace("blue_", "") for r in blue_rows],
                x=[r["importance_pct"] for r in blue_rows],
                orientation="h",
                marker=dict(color="#38BDF8"),
                text=[f"{r['importance_pct']:.1f}%" for r in blue_rows],
                textposition="outside",
                name="Blue Levers",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if red_rows:
        fig.add_trace(
            go.Bar(
                y=[r["feature"].replace("red_", "") for r in red_rows],
                x=[r["importance_pct"] for r in red_rows],
                orientation="h",
                marker=dict(color="#EF4444"),
                text=[f"{r['importance_pct']:.1f}%" for r in red_rows],
                textposition="outside",
                name="Red Threat Drivers",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="Permutation Importance Share (%)", row=1, col=1)
    fig.update_xaxes(title_text="Permutation Importance Share (%)", row=1, col=2)
    fig.update_layout(
        title="Learned ML Surrogate Feature Sensitivity (Physical Protection vs Threat Drivers)",
        margin=dict(l=40, r=40, t=75, b=40),
        height=440,
    )
    fig.update_annotations(font_size=12)
    return fig


def create_ml_policy_pareto_figure(policy_res: dict[str, Any]) -> go.Figure:
    """Visualize candidate Blue policies predicted by GP surrogate for a selected Red threat."""
    cands = policy_res.get("all_candidates", [])
    rec = policy_res.get("recommended_summary")
    limit_kn = float(policy_res.get("mount_load_limit_kn", 18.0))
    red_t = policy_res.get("target_red_threat", {})

    fig = go.Figure()
    palette = {
        "12G_NO4_BUCK_C_UAS": "#38BDF8",
        "12G_TSS_NO2_C_UAS": "#10B981",
        "12G_DUPLEX_TSS_BUCK_C_UAS": "#F59E0B",
        "12G_FLECHETTE_SABOT_C_UAS": "#EC4899",
    }
    by_ammo: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        by_ammo.setdefault(str(c["ammunition_preset"]), []).append(c)

    for ammo_k, grp in by_ammo.items():
        fig.add_trace(
            go.Scatter(
                x=[c["peak_mount_force_kn"] for c in grp],
                y=[c["pred_utility"] for c in grp],
                error_y=dict(type="data", array=[c["pred_std"] for c in grp], visible=False),
                mode="markers",
                marker=dict(
                    size=9,
                    color=palette.get(ammo_k, "#94A3B8"),
                    symbol=["circle" if bool(c["feasible"]) else "x" for c in grp],
                    opacity=0.82,
                    line=dict(color="#FFFFFF", width=0.8),
                ),
                name=ammo_k.replace("_C_UAS", ""),
                text=[
                    f"{c['name']}<br>Pred Score: {c['pred_utility']:.1f} ± {c['pred_std']:.1f}"
                    f"<br>Force: {c['peak_mount_force_kn']:.1f} kN | Cost/salvo: ${c['est_salvo_cost_usd']:.2f}"
                    for c in grp
                ],
                hoverinfo="text",
            )
        )

    if rec is not None:
        fig.add_trace(
            go.Scatter(
                x=[rec["peak_mount_force_kn"]],
                y=[rec["pred_utility"]],
                mode="markers+text",
                marker=dict(size=17, color="#FACC15", symbol="star", line=dict(color="#111827", width=1.5)),
                text=[f"ML #1 Pick ({rec['pred_utility']:.1f})"],
                textposition="top center",
                name="ML #1 Recommended Policy",
            )
        )

    fig.add_vline(x=limit_kn, line_dash="dash", line_color="#EF4444", annotation_text=f"Mount Limit ({limit_kn:.1f} kN)")
    fig.update_layout(
        title=f"ML Policy Space vs Threat: {red_t.get('name', 'Selected Drone Threat')}",
        xaxis_title="Peak Turret Mount Force (kN)",
        yaxis_title="GP Predicted Defense Score (0-100)",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=35, r=30, t=65, b=40),
        legend=dict(x=0.01, y=0.02, bgcolor="rgba(17, 24, 39, 0.78)", font=dict(color="#F9FAFB", size=10)),
        height=470,
    )
    return fig
