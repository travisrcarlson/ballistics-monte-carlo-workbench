"""Interactive 3D & 2D Ballistics, Tactical & C-UAS Plotly Visualization Package — tactical submodule."""

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


