"""Plotly Visualization Suite for A/B Turret Comparison, Layered C-UAS Defense,
Real-World OpenStreetMap Tactical Layout, and Tornado Sensitivity Analysis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ammo_mc.cuas_analytics import local_xy_to_lonlat


def create_ab_comparator_radar_and_bars_figure(ab_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel Mission Effectiveness Radar + Core Metrics Bar Comparison."""
    rows = ab_res.get("scorecard", [])
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.12,
        subplot_titles=[
            "🧭 6-Axis Normalized Mission Profile<br>(100 = Best in Trade Group)",
            "⚔️ Kill-Chain Pk, Swarm Survival<br>& Composite MEI Score (%)",
        ],
    )
    palette = ["#38BDF8", "#F59E0B", "#10B981", "#A855F7"]

    if rows:
        axes = list(rows[0]["radar_scores"].keys())
        axes_closed = axes + [axes[0]]
        for idx, r in enumerate(rows):
            col = palette[idx % len(palette)]
            vals = [float(r["radar_scores"][k]) for k in axes]
            vals_closed = vals + [vals[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals_closed,
                    theta=axes_closed,
                    fill="toself",
                    name=r["label"],
                    line=dict(color=col, width=2.4),
                    opacity=0.80,
                ),
                row=1,
                col=1,
            )

        labels = [r["label"] for r in rows]
        fig.add_trace(
            go.Bar(
                name="Kill-Chain Pk (%)",
                x=labels,
                y=[r["kill_chain_pk_pct"] for r in rows],
                marker_color="#38BDF8",
                text=[f"{r['kill_chain_pk_pct']:.1f}%" for r in rows],
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                name="Swarm Survival (%)",
                x=labels,
                y=[r["swarm_survival_pct"] for r in rows],
                marker_color="#10B981",
                text=[f"{r['swarm_survival_pct']:.1f}%" for r in rows],
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                name="MEI Composite Score",
                x=labels,
                y=[r["mei_score"] for r in rows],
                marker_color="#F59E0B",
                text=[f"{r['mei_score']:.1f}" for r in rows],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=490,
        margin=dict(l=45, r=30, t=75, b=50),
        polar=dict(radialaxis=dict(visible=True, range=[0, 105])),
        yaxis=dict(title="Score / Probability (%)", range=[0, 114]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.24, x=0.0),
    )
    return fig


def create_layered_defense_figure(layer_res: dict[str, Any]) -> go.Figure:
    """Create 2-panel wave attrition & Layered vs Inner-Only gains dashboard."""
    sm = layer_res.get("summary", {})
    waves = layer_res.get("wave_timeline", [])

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.14,
        subplot_titles=[
            "🛡️ Multi-Wave Drone Disposition<br>(Mean Drones per Wave)",
            "📈 Layered vs Inner-Only Comparison<br>(Survival %, Temp °C & Rounds)",
        ],
    )
    if waves:
        x_w = [f"W{w['wave']}" for w in waves]
        fig.add_trace(
            go.Bar(
                name="Outer Tier Kills",
                x=x_w,
                y=[w["outer_kills"] for w in waves],
                marker_color="#38BDF8",
                offsetgroup="wave_stack",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Inner Turret Kills",
                x=x_w,
                y=[w["inner_kills"] for w in waves],
                marker_color="#10B981",
                offsetgroup="wave_stack",
                base=[w["outer_kills"] for w in waves],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Defense Leakers",
                x=x_w,
                y=[w["leakers"] for w in waves],
                marker_color="#EF4444",
                offsetgroup="wave_stack",
                base=[w["outer_kills"] + w["inner_kills"] for w in waves],
            ),
            row=1,
            col=1,
        )

    cats = ["Survival (%)", "Peak Temp (°C)", "Inner Rounds"]
    l_vals = [
        float(sm.get("layered_survival_pct", 0.0)),
        float(sm.get("layered_peak_temp_c", 0.0)),
        float(sm.get("layered_inner_rounds", 0.0)),
    ]
    i_vals = [
        float(sm.get("inner_only_survival_pct", 0.0)),
        float(sm.get("inner_only_peak_temp_c", 0.0)),
        float(sm.get("inner_only_rounds", 0.0)),
    ]
    fig.add_trace(
        go.Bar(
            name="Layered (Outer + Inner)",
            x=cats,
            y=l_vals,
            marker_color="#2563EB",
            offsetgroup="comp_a",
            text=[f"{v:.1f}" for v in l_vals],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Inner Turret Only",
            x=cats,
            y=i_vals,
            marker_color="#94A3B8",
            offsetgroup="comp_b",
            text=[f"{v:.1f}" for v in i_vals],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    y_top = max(max(l_vals, default=10.0), max(i_vals, default=10.0), 100.0) * 1.18
    fig.update_yaxes(title_text="Mean Drones / Wave", row=1, col=1)
    fig.update_yaxes(title_text="Metric Value", range=[0, y_top], row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=450,
        margin=dict(l=40, r=25, t=75, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, x=0.0),
    )
    return fig


def create_real_map_site_defense_figure(
    site_res: dict[str, Any],
    origin_lat: float = 32.9000,
    origin_lon: float = -106.4000,
    sdz_radius_m: float = 280.0,
    heading_deg: float = 0.0,
    map_style: str = "open-street-map",
) -> go.Figure:
    """Plot C-UAS turrets, firing sectors, obstacles, NFZs, blind spots & Fallout SDZ on a real-world map."""
    sm = site_res.get("summary", {})
    asset_r = float(sm.get("asset_radius_m", 40.0))
    w_range = float(sm.get("weapon_max_range_m", 60.0))

    fig = go.Figure()

    def _circle_coords(cx: float, cy: float, r: float, n: int = 48) -> tuple[list[float], list[float]]:
        lons, lats = [], []
        for k in range(n + 1):
            ang = 2.0 * math.pi * (k % n) / n
            lo, la = local_xy_to_lonlat(cx + r * math.cos(ang), cy + r * math.sin(ang), origin_lat, origin_lon, heading_deg)
            lons.append(lo)
            lats.append(la)
        return lons, lats

    # 1. Extended Fallout SDZ perimeter
    sdz_lons, sdz_lats = _circle_coords(0.0, 0.0, asset_r + sdz_radius_m)
    fig.add_trace(
        go.Scattermap(
            lon=sdz_lons,
            lat=sdz_lats,
            mode="lines",
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.12)",
            line=dict(color="#F59E0B", width=2),
            name=f"Fallout SDZ ({sdz_radius_m:.0f} m)",
        )
    )

    # 2. Defended Asset perimeter
    asset_lons, asset_lats = _circle_coords(0.0, 0.0, asset_r)
    fig.add_trace(
        go.Scattermap(
            lon=asset_lons,
            lat=asset_lats,
            mode="lines",
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.18)",
            line=dict(color="#38BDF8", width=2.5),
            name=f"Defended Asset (R={asset_r:.0f} m)",
        )
    )

    # 3. Turret Firing Sectors
    for idx, t in enumerate(site_res.get("turrets", [])):
        tx, ty = float(t["x_m"]), float(t["y_m"])
        c_lo, c_la = local_xy_to_lonlat(tx, ty, origin_lat, origin_lon, heading_deg)
        b_az = float(t.get("boresight_az_deg", 0.0))
        half_deg = float(t.get("sector_half_deg", 145.0))
        t_rng = float(t.get("max_range_m", w_range))
        s_lons, s_lats = [c_lo], [c_la]
        for az in np.linspace(b_az - half_deg, b_az + half_deg, 24):
            rad = math.radians(float(az))
            lo, la = local_xy_to_lonlat(tx + t_rng * math.cos(rad), ty + t_rng * math.sin(rad), origin_lat, origin_lon, heading_deg)
            s_lons.append(lo)
            s_lats.append(la)
        s_lons.append(c_lo)
        s_lats.append(c_la)
        fig.add_trace(
            go.Scattermap(
                lon=s_lons,
                lat=s_lats,
                mode="lines",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.18)",
                line=dict(color="#10B981", width=1.5),
                name="Turret Engagement Sectors" if idx == 0 else str(t["id"]),
                showlegend=(idx == 0),
                legendgroup="sectors",
            )
        )

    # 4. Blind-spot azimuth arcs (merge adjacent 5-deg bins into contiguous wedges)
    blind_bins = sorted(int(round(r["az_deg"])) for r in site_res.get("azimuth_profile", []) if int(r.get("active_turrets", 0)) == 0)
    if blind_bins:
        spans: list[tuple[int, int]] = []
        s_start = blind_bins[0]
        s_prev = blind_bins[0]
        for b in blind_bins[1:]:
            if b - s_prev <= 6:
                s_prev = b
            else:
                spans.append((s_start, s_prev + 5))
                s_start = s_prev = b
        spans.append((s_start, s_prev + 5))

        eval_r = asset_r + w_range
        c_lo, c_la = local_xy_to_lonlat(0.0, 0.0, origin_lat, origin_lon, heading_deg)
        for s_i, (a0, a1) in enumerate(spans):
            b_lons, b_lats = [c_lo], [c_la]
            for az in np.linspace(a0 - 2.5, a1 - 2.5, max(5, (a1 - a0) // 4 + 2)):
                rad = math.radians(float(az))
                lo, la = local_xy_to_lonlat(eval_r * math.cos(rad), eval_r * math.sin(rad), origin_lat, origin_lon, heading_deg)
                b_lons.append(lo)
                b_lats.append(la)
            b_lons.append(c_lo)
            b_lats.append(c_la)
            fig.add_trace(
                go.Scattermap(
                    lon=b_lons,
                    lat=b_lats,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.30)",
                    line=dict(color="#EF4444", width=1.8),
                    name="LOS / NFZ Blind Spots" if s_i == 0 else f"Blind {a0}-{a1}°",
                    showlegend=(s_i == 0),
                    legendgroup="blind",
                )
            )

    # 5. Obstacles
    for i, obs in enumerate(site_res.get("obstacles", [])):
        ox, oy = float(obs["x_m"]), float(obs["y_m"])
        hw, hl = 0.5 * float(obs["width_m"]), 0.5 * float(obs["length_m"])
        corners = [(ox - hw, oy - hl), (ox + hw, oy - hl), (ox + hw, oy + hl), (ox - hw, oy + hl), (ox - hw, oy - hl)]
        pts = [local_xy_to_lonlat(cx, cy, origin_lat, origin_lon, heading_deg) for cx, cy in corners]
        fig.add_trace(
            go.Scattermap(
                lon=[p[0] for p in pts],
                lat=[p[1] for p in pts],
                mode="lines",
                fill="toself",
                fillcolor="rgba(148, 163, 184, 0.55)",
                line=dict(color="#CBD5E1", width=2),
                name="Buildings / Obstacles" if i == 0 else str(obs["name"]),
                hovertext=f"{obs['name']} (H={obs['height_m']}m)",
                showlegend=(i == 0),
                legendgroup="obstacles",
            )
        )

    # 6. Friendly NFZs
    for i, nfz in enumerate(site_res.get("nfz_zones", [])):
        n_lons, n_lats = _circle_coords(float(nfz["x_m"]), float(nfz["y_m"]), float(nfz["radius_m"]), n=32)
        fig.add_trace(
            go.Scattermap(
                lon=n_lons,
                lat=n_lats,
                mode="lines",
                fill="toself",
                fillcolor="rgba(220, 38, 38, 0.25)",
                line=dict(color="#DC2626", width=2),
                name="Friendly No-Fire Zones" if i == 0 else str(nfz["name"]),
                hovertext=f"NFZ: {nfz['name']} (R={nfz['radius_m']}m)",
                showlegend=(i == 0),
                legendgroup="nfz",
            )
        )

    # 7. Turret Point Markers
    t_lons, t_lats, t_texts = [], [], []
    for t in site_res.get("turrets", []):
        lo, la = local_xy_to_lonlat(float(t["x_m"]), float(t["y_m"]), origin_lat, origin_lon, heading_deg)
        t_lons.append(lo)
        t_lats.append(la)
        t_texts.append(f"{t['id']} ({t.get('n_barrels', 6)} bbls)")
    if t_lons:
        fig.add_trace(
            go.Scattermap(
                lon=t_lons,
                lat=t_lats,
                mode="markers+text",
                marker=dict(size=13, color="#22C55E"),
                text=t_texts,
                textposition="top right",
                name="C-UAS Turrets",
            )
        )

    fig.update_layout(
        map=dict(
            style=map_style,
            center=dict(lat=float(origin_lat), lon=float(origin_lon)),
            zoom=15.4,
        ),
        height=560,
        margin=dict(l=10, r=10, t=42, b=10),
        title=f"🌍 Real-World Site Defense, Obstacle Masking & Fallout SDZ ({origin_lat:.4f}°, {origin_lon:.4f}°)",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01, bgcolor="rgba(15, 23, 42, 0.78)"),
    )
    return fig


def create_sensitivity_tornado_figure(
    sens_res: dict[str, Any],
    metric_label: str = "Kill-Chain P_kill (%)",
    higher_is_better: bool = True,
) -> go.Figure:
    """Create diverging Tornado sensitivity chart centered on the baseline output."""
    rows = list(reversed(sens_res.get("rows", [])))  # reverse so largest driver renders at top
    sm = sens_res.get("summary", {})
    base_val = float(sm.get("baseline_value", 0.0))

    fig = go.Figure()
    fav_x, fav_base, fav_y, fav_txt = [], [], [], []
    adv_x, adv_base, adv_y, adv_txt = [], [], [], []

    for r in rows:
        lbl = str(r["label"])
        for tag, d_key, val_key, param_val_key in [
            ("−Δ%", "delta_low", "val_at_low", "low_val"),
            ("+Δ%", "delta_high", "val_at_high", "high_val"),
        ]:
            delta = float(r[d_key])
            if abs(delta) < 1e-6:
                continue
            is_fav = (delta > 0) if higher_is_better else (delta < 0)
            txt = f"{delta:+.1f} ({tag}={r[param_val_key]})"
            if is_fav:
                fav_x.append(delta)
                fav_base.append(base_val)
                fav_y.append(lbl)
                fav_txt.append(txt)
            else:
                adv_x.append(delta)
                adv_base.append(base_val)
                adv_y.append(lbl)
                adv_txt.append(txt)

    if fav_y:
        fig.add_trace(
            go.Bar(
                name="Favorable Shift",
                y=fav_y,
                x=fav_x,
                base=fav_base,
                orientation="h",
                marker_color="#10B981",
                text=fav_txt,
                textposition="auto",
            )
        )
    if adv_y:
        fig.add_trace(
            go.Bar(
                name="Adverse Shift",
                y=adv_y,
                x=adv_x,
                base=adv_base,
                orientation="h",
                marker_color="#EF4444",
                text=adv_txt,
                textposition="auto",
            )
        )

    fig.add_vline(
        x=base_val,
        line_dash="dash",
        line_color="#F8FAFC",
        annotation_text=f"Baseline = {base_val:.1f}",
    )
    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        title=f"📊 Parameter Sensitivity Tornado (±{sm.get('perturbation_pct', 20):.0f}% Perturbation on {metric_label})",
        xaxis_title=metric_label,
        height=max(390, 62 * max(len(rows), 4)),
        margin=dict(l=220, r=30, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.0),
    )
    return fig
