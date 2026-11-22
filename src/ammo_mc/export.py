"""Export utilities for CSV trials, JSON summary + provenance, GeoJSON contours, and PNG/PDF plots (App_Spec R22, R29)."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.gis import export_hazard_geojson
from ammo_mc.schemas import Scenario, SimulationResult


def trials_to_csv_bytes(result: SimulationResult) -> bytes:
    """Serialize `result.trials` to UTF-8 CSV bytes."""
    buf = io.StringIO()
    cols = list(result.trials.keys())
    if not cols:
        return b""
    n_rows = len(result.trials[cols[0]])
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for idx in range(n_rows):
        writer.writerow([result.trials[c][idx] for c in cols])
    return buf.getvalue().encode("utf-8")


def summary_to_json_str(result: SimulationResult, include_trials: bool = False) -> str:
    """Serialize Standard Report + Provenance block to formatted JSON string."""
    payload = result.model_dump()
    if not include_trials:
        payload.pop("trials", None)
        payload.pop("hazard_grid", None)
    return json.dumps(payload, indent=2)


def geojson_to_str(result: SimulationResult, scenario: Scenario | None = None) -> str:
    """Serialize RFC 7946 GeoJSON FeatureCollection to formatted JSON string."""
    fc = export_hazard_geojson(result, scenario=scenario)
    return json.dumps(fc, indent=2)


def render_report_figure_bytes(
    result: SimulationResult,
    scenario: Scenario | None = None,
    fmt: str = "png",
    dpi: int = 160,
) -> bytes:
    """Render a 2x2 engineering summary sheet (PNG or PDF) with governance notices."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.patch.set_facecolor("#FFFFFF")

    title_str = f"{result.ammo_name} — {result.scenario_name} [{result.fidelity} | {result.sampler} | N={result.n_total_rounds}]"
    fig.suptitle(title_str, fontsize=13, fontweight="bold", y=0.97)

    if result.provenance.get("has_flag_warnings"):
        warn_txt = "FLAG WARNING: " + " | ".join(result.provenance.get("governance_warnings", []))
        fig.text(
            0.5,
            0.932,
            warn_txt[:150],
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7F2704",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FCE4D6", "edgecolor": "#D95F02"},
        )

    rng_arr = np.asarray(result.trials.get("range_m", []), dtype=np.float64)
    def_arr = np.asarray(result.trials.get("deflection_m", []), dtype=np.float64)
    uxo_arr = np.asarray(result.trials.get("is_uxo", []), dtype=bool)

    # Panel 1: Impact Scatter + CEP Ellipse + UXO Locations
    ax0 = axes[0, 0]
    if rng_arr.size > 0:
        normal_mask = ~uxo_arr if uxo_arr.size == rng_arr.size else np.ones_like(rng_arr, dtype=bool)
        ax0.scatter(
            def_arr[normal_mask],
            rng_arr[normal_mask],
            s=14,
            alpha=0.55,
            color="#1f77b4",
            label="Functioned Impacts",
        )
        if np.any(uxo_arr):
            ax0.scatter(
                def_arr[uxo_arr],
                rng_arr[uxo_arr],
                s=36,
                marker="x",
                color="#d62728",
                linewidths=1.5,
                label="UXO / Dud",
            )
        mpi_r = result.dispersion.get("mean_range_m", float(np.mean(rng_arr)))
        mpi_d = result.dispersion.get("mean_deflection_m", float(np.mean(def_arr)))
        sr = max(result.dispersion.get("std_range_m", 0.0), 0.05)
        sd = max(result.dispersion.get("std_deflection_m", 0.05), 0.05)
        ang = np.linspace(0, 2.0 * math.pi, 100)
        k50 = math.sqrt(2.0 * math.log(2.0))
        ax0.plot(
            mpi_d + k50 * sd * np.sin(ang),
            mpi_r + k50 * sr * np.cos(ang),
            color="#ff7f0e",
            lw=2.0,
            label=f"CEP ({result.dispersion.get('cep_m', 0.0):.1f} m)",
        )
        if scenario is not None:
            ax0.scatter([0.0], [scenario.aim.range_m], marker="+", s=90, color="#111111", label="Aim Point")
    ax0.set_title("Impact Dispersion, CEP Ellipse & UXO Map", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("Deflection (+ Right) [m]")
    ax0.set_ylabel("Range [m]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    # Panel 2: Range Histogram or Hazard Heatmap
    ax1 = axes[0, 1]
    if result.hazard_grid is not None:
        gx = np.asarray(result.hazard_grid["x_m"])
        gy = np.asarray(result.hazard_grid["y_m"])
        pk_g = np.asarray(result.hazard_grid["pk_grid"])
        im = ax1.contourf(gy, gx, pk_g, levels=np.linspace(0.0, 1.0, 11), cmap="YlOrRd")
        fig.colorbar(im, ax=ax1, label="Single-Detonation Pk")
        ax1.set_title(
            f"Terminal Effects Hazard Field (Lethal Area = {result.effectiveness.get('lethal_area_m2', 0.0):.0f} m²)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax1.set_xlabel("Crossrange Offset [m]")
        ax1.set_ylabel("Downrange Offset [m]")
    elif rng_arr.size > 0:
        ax1.hist(rng_arr, bins=min(30, max(8, int(math.sqrt(rng_arr.size)))), color="#2ca02c", alpha=0.75, edgecolor="white")
        ax1.set_title("Impact Range Distribution", fontsize=10.5, fontweight="bold")
        ax1.set_xlabel("Range [m]")
        ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.25)

    # Panel 3: Convergence History & 95% CI Ribbon
    ax2 = axes[1, 0]
    n_c = np.asarray(result.convergence.get("n_rounds", []), dtype=np.float64)
    if n_c.size > 0:
        pk_c = np.asarray(result.convergence.get("sspk", []), dtype=np.float64)
        hw_p = np.asarray(result.convergence.get("sspk_ci95_half_width", []), dtype=np.float64)
        if np.any(pk_c > 0):
            ax2.plot(n_c, pk_c, color="#1f77b4", lw=2.0, label="Running Mean SSPK")
            ax2.fill_between(n_c, np.maximum(0.0, pk_c - hw_p), np.minimum(1.0, pk_c + hw_p), color="#1f77b4", alpha=0.2, label="95% CI")
            ax2.set_ylabel("SSPK")
        else:
            mr_c = np.asarray(result.convergence.get("mean_range_m", []), dtype=np.float64)
            hw_r = np.asarray(result.convergence.get("range_ci95_half_width_m", []), dtype=np.float64)
            ax2.plot(n_c, mr_c, color="#1f77b4", lw=2.0, label="Running Mean Range")
            ax2.fill_between(n_c, mr_c - hw_r, mr_c + hw_r, color="#1f77b4", alpha=0.2, label="95% CI")
            ax2.set_ylabel("Mean Range [m]")
        ax2.legend(loc="best", fontsize=8)
    ax2.set_title("Monte Carlo Convergence & 95% Confidence Band", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Cumulative Simulated Rounds (N)")
    ax2.grid(True, alpha=0.3)

    # Panel 4: Sensitivity Tornado (MC-10 Spearman Rank Correlation)
    ax3 = axes[1, 1]
    sens = result.sensitivity[:8]
    if sens:
        labels = [s["parameter"] for s in reversed(sens)]
        rhos = [s["spearman_rho"] for s in reversed(sens)]
        colors = ["#d62728" if r < 0 else "#1f77b4" for r in rhos]
        ax3.barh(labels, rhos, color=colors, alpha=0.85)
        ax3.axvline(0.0, color="#333333", lw=1.0)
        ax3.set_xlim(-1.05, 1.05)
    ax3.set_title("Global Sensitivity Ranking (MC-10 Spearman ρ)", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Spearman Rank Correlation ρ")
    ax3.grid(True, axis="x", alpha=0.3)

    fig.text(
        0.5,
        0.012,
        USE_LIMITATION_NOTICE,
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0.0, 0.035, 1.0, 0.91])

    out_buf = io.BytesIO()
    fig.savefig(out_buf, format=fmt.lower(), dpi=dpi)
    plt.close(fig)
    return out_buf.getvalue()


def export_all_artifacts(
    result: SimulationResult,
    output_dir: str | Path,
    scenario: Scenario | None = None,
    prefix: str = "sim_output",
) -> dict[str, str]:
    """Write CSV trials, JSON summary+provenance, GeoJSON contours, PNG, and PDF report to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{prefix}_trials.csv"
    json_path = out / f"{prefix}_summary.json"
    geojson_path = out / f"{prefix}_contours.geojson"
    png_path = out / f"{prefix}_report.png"
    pdf_path = out / f"{prefix}_report.pdf"

    csv_path.write_bytes(trials_to_csv_bytes(result))
    json_path.write_text(summary_to_json_str(result, include_trials=False), encoding="utf-8")
    geojson_path.write_text(geojson_to_str(result, scenario=scenario), encoding="utf-8")
    png_path.write_bytes(render_report_figure_bytes(result, scenario=scenario, fmt="png"))
    pdf_path.write_bytes(render_report_figure_bytes(result, scenario=scenario, fmt="pdf"))

    return {
        "trials_csv": str(csv_path),
        "summary_json": str(json_path),
        "contours_geojson": str(geojson_path),
        "report_png": str(png_path),
        "report_pdf": str(pdf_path),
    }
