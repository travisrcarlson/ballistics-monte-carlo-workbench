"""Command-line interface for the Ammunition Monte Carlo Simulator (ammo-mc)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional

import typer

from ammo_mc.export import export_all_artifacts, summary_to_json_str
from ammo_mc.mcdrag import McDragInputs, estimate_mcdrag
from ammo_mc.pipeline import run_simulation
from ammo_mc.presets import get_preset, list_presets
from ammo_mc.schemas import AmmoDefinition, Scenario
from ammo_mc.validation import run_validation_suite

app = typer.Typer(
    name="ammo-mc",
    help="Stochastic Ammunition Effects & Dispersion Monte Carlo Simulator CLI.",
    add_completion=False,
)


@app.command("presets")
def cmd_presets() -> None:
    """List all available notional ammunition presets and baseline scenarios."""
    presets = list_presets()
    typer.echo(f"\n{'Preset ID':<20} {'Class':<8} {'Fidelity':<10} {'Ammo Name'}")
    typer.echo("=" * 80)
    for p in presets:
        typer.echo(f"{p['preset_id']:<20} {p['class_id']:<8} {p['fidelity']:<10} {p['ammo_name']}")
    typer.echo(f"\nTotal: {len(presets)} presets available. All presets strictly NOTIONAL.\n")


@app.command("validate")
def cmd_validate(
    seed: int = typer.Option(42, help="RNG seed for stochastic validation cases (e.g. V-08)."),
    n_mc: int = typer.Option(100_000, help="Trial count for stochastic validation cases."),
    as_json: bool = typer.Option(False, "--json", help="Output validation results as raw JSON."),
) -> None:
    """Run the Ammunition Monte Carlo Knowledge Base validation suite (V-01a..V-12)."""
    typer.echo("Running validation suite against Excel formulas and source literature...\n")
    results = run_validation_suite(seed=seed, n_mc_v08=n_mc)

    if as_json:
        typer.echo(json.dumps(results, indent=2))
        return

    typer.echo(
        f"{'Test ID':<8} {'Formula':<10} {'Status':<16} {'Computed':<14} {'Excel Val':<14} {'Expected':<12} {'Rel Err (Exp)'}"
    )
    typer.echo("=" * 95)
    all_passed = True
    for r in results:
        status_color = typer.colors.GREEN if "PASS" in r["status"] else typer.colors.RED
        if "FAIL" in r["status"]:
            all_passed = False
        comp_s = f"{r['computed']:.5g}"
        exc_s = f"{r['excel_value']:.5g}" if r["excel_value"] is not None else "--"
        exp_s = f"{r['expected']:.5g}" if r["expected"] is not None else "--"
        err_s = f"{r['rel_err_vs_expected']*100:.2f}%" if r["rel_err_vs_expected"] is not None else "--"
        status_styled = typer.style(f"{r['status']:<16}", fg=status_color, bold=True)
        typer.echo(
            f"{r['test_id']:<8} {r['formula']:<10} {status_styled} {comp_s:<14} {exc_s:<14} {exp_s:<12} {err_s}"
        )

    typer.echo("=" * 95)
    if all_passed:
        typer.secho("\nValidation Suite: ALL CASES PASSED.", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("\nValidation Suite: FAILURES DETECTED.", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


@app.command("simulate")
def cmd_simulate(
    preset: Optional[str] = typer.Option("NOTIONAL-AR-01", "--preset", "-p", help="Preset ID to run."),
    ammo_file: Optional[Path] = typer.Option(None, "--ammo-file", help="Custom AmmoDefinition JSON path."),
    scenario_file: Optional[Path] = typer.Option(None, "--scenario-file", help="Custom Scenario JSON path."),
    fidelity: Optional[str] = typer.Option(None, "--fidelity", "-f", help="Override fidelity (L0, L1, L2, L3)."),
    trials: Optional[int] = typer.Option(None, "--trials", "-n", help="Override total trials / rounds."),
    occasions: Optional[int] = typer.Option(None, "--occasions", help="Override number of occasions."),
    rounds_per_occasion: Optional[int] = typer.Option(None, "--rounds-per-occasion", help="Override rounds per occasion."),
    sampler: Optional[str] = typer.Option(None, "--sampler", help="Sampler method (LHS, SOBOL, PSEUDO)."),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Output directory to save artifacts."),
    export_all: bool = typer.Option(False, "--export-all", help="Export CSV, JSON, GeoJSON, PNG, PDF."),
    export_3d: bool = typer.Option(False, "--export-3d", help="Export standalone interactive 3D model HTML."),
    as_json: bool = typer.Option(False, "--json", help="Output simulation summary as JSON."),
) -> None:
    """Execute an ammunition effects and dispersion Monte Carlo simulation."""
    if preset:
        ammo, scn = get_preset(preset)
    elif ammo_file and scenario_file:
        ammo = AmmoDefinition.model_validate_json(ammo_file.read_text(encoding="utf-8"))
        scn = Scenario.model_validate_json(scenario_file.read_text(encoding="utf-8"))
    else:
        typer.secho("Error: Must specify either --preset or both --ammo-file and --scenario-file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if fidelity:
        scn.fidelity = fidelity.upper()  # type: ignore[assignment]
    if trials is not None:
        scn.n_trials = trials
    if occasions is not None:
        scn.n_occasions = occasions
    if rounds_per_occasion is not None:
        scn.rounds_per_occasion = rounds_per_occasion
    if sampler:
        scn.sampler = sampler.upper()  # type: ignore[assignment]
    if seed is not None:
        scn.seed = seed

    import time
    t0 = time.perf_counter()
    typer.echo(f"Running simulation: {ammo.name} | Scenario: {scn.name} [{scn.fidelity}]...")
    res = run_simulation(ammo, scn)
    elapsed_s = time.perf_counter() - t0

    if as_json:
        typer.echo(summary_to_json_str(res))
    else:
        typer.echo("\n" + "=" * 60)
        typer.echo(f"Simulation Summary: {res.ammo_name}")
        typer.echo(f"Scenario: {res.scenario_name} (Fidelity: {res.fidelity})")
        typer.echo(f"Rounds Simulated: {res.n_total_rounds} across {res.n_occasions} occasion(s)")
        typer.echo(f"Runtime: {elapsed_s:.3f} s")
        typer.echo("-" * 60)
        disp = res.dispersion
        typer.echo(f"Mean Range:       {disp.get('mean_range_m', 0.0):.2f} m (CI95: ±{disp.get('range_ci95_half_width_m', 0.0):.2f} m)")
        typer.echo(f"Mean Deflection:  {disp.get('mean_deflection_m', 0.0):.2f} m (CI95: ±{disp.get('deflection_ci95_half_width_m', 0.0):.2f} m)")
        typer.echo(f"REP: {disp.get('rep_m', 0.0):.2f} m | DEP: {disp.get('dep_m', 0.0):.2f} m | CEP50: {disp.get('cep_m', 0.0):.2f} m")
        typer.echo(f"P(<=25m Miss):    {disp.get('p_within_25m', 0.0)*100:.1f} %")

        eff = res.effectiveness
        typer.echo("-" * 60)
        typer.echo(f"Single-Shot Pk:   {eff.get('sspk', 0.0):.4f} (Wilson 95% CI: [{eff.get('sspk_wilson_ci_low', 0.0):.4f}, {eff.get('sspk_wilson_ci_high', 0.0):.4f}])")
        typer.echo(f"Salvo Pk (Mean):  {eff.get('salvo_pk_mean', 0.0):.4f} (Independent Ref: {eff.get('salvo_pk_independent_ref', 0.0):.4f})")
        typer.echo(f"Lethal Area:      {eff.get('lethal_area_m2', 0.0):.1f} m² (Carleton Ref: {eff.get('carleton_ref_lethal_area_m2', 0.0):.1f} m²)")

        fz = res.uxo
        typer.echo("-" * 60)
        typer.echo(f"Reliability:      {fz.get('reliability_single', 1.0)*100:.2f} % | Dud Fraction: {fz.get('empirical_dud_rate', 0.0)*100:.2f} %")
        typer.echo(f"Dud Wilson 95%:   [{fz.get('dud_rate_ci_low', 0.0)*100:.2f} %, {fz.get('dud_rate_ci_high', 0.0)*100:.2f} %]")

        if res.provenance.get("has_flag_warnings"):
            typer.echo("-" * 60)
            typer.secho("GOVERNANCE NOTICES [FLAG]:", fg=typer.colors.YELLOW, bold=True)
            for w in res.provenance.get("governance_warnings", []):
                typer.secho(f"  • {w}", fg=typer.colors.YELLOW)
        typer.echo("=" * 60 + "\n")

    if export_all or export_3d or out_dir:
        target_dir = out_dir or Path("./exports")
        paths = export_all_artifacts(res, target_dir, scenario=scn, prefix=preset or "simulation")
        if export_3d or export_all:
            from ammo_mc.vis_3d import create_3d_trajectory_bundle_figure, export_3d_html
            fig_3d = create_3d_trajectory_bundle_figure(res, scenario=scn)
            p3d = target_dir / f"{preset or 'simulation'}_3d.html"
            export_3d_html(fig_3d, p3d)
            paths["3d_model_html"] = str(p3d)

        typer.secho(f"Artifacts saved to {target_dir}:", fg=typer.colors.CYAN)
        for kind, path in paths.items():
            typer.echo(f"  • {kind}: {path}")


@app.command("sim-3d")
def cmd_sim_3d(
    preset: str = typer.Option("NOTIONAL-AR-01", "--preset", "-p", help="Preset ID to simulate."),
    fidelity: str = typer.Option("L2", "--fidelity", "-f", help="Fidelity level (L0, L1, L2, L3)."),
    trials: int = typer.Option(100, "--trials", "-n", help="Total trials."),
    trajectories_3d: int = typer.Option(12, "--trajectories", "-k", help="Number of 3D trajectory lines in bundle."),
    out_html: Path = typer.Option(Path("./simulation_3d.html"), "--out", "-o", help="Output 3D HTML file path."),
    open_browser: bool = typer.Option(False, "--open", help="Open generated 3D simulation in web browser."),
) -> None:
    """Run Monte Carlo simulation and generate an interactive 3D trajectory and terrain HTML model."""
    import webbrowser
    from ammo_mc.vis_3d import create_3d_trajectory_bundle_figure, export_3d_html

    ammo, scn = get_preset(preset)
    scn.fidelity = fidelity.upper()  # type: ignore[assignment]
    scn.n_trials = trials
    scn.record_trajectories_count = trajectories_3d

    typer.echo(f"Running 3D simulation: {ammo.name} [{scn.fidelity} | N={trials}]...")
    res = run_simulation(ammo, scn)

    fig_3d = create_3d_trajectory_bundle_figure(res, scenario=scn, max_trajectories=trajectories_3d)
    out_path = export_3d_html(fig_3d, out_html)
    typer.secho(f"\nInteractive 3D Simulation saved to: {out_path}", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Mean Range: {res.dispersion.get('mean_range_m', 0.0):.1f} m | CEP50: {res.dispersion.get('cep_m', 0.0):.1f} m")

    if open_browser:
        webbrowser.open(Path(out_path).resolve().as_uri())


@app.command("mcdrag")
def cmd_mcdrag(
    calibre_m: float = typer.Option(0.155, help="Calibre diameter in metres."),
    d_meplat_cal: float = typer.Option(0.15, help="Meplat diameter in calibres."),
    l_nose_cal: float = typer.Option(2.5, help="Nose / ogive length in calibres."),
    l_cyl_cal: float = typer.Option(2.0, help="Cylindrical body length in calibres."),
    l_bt_cal: float = typer.Option(0.5, help="Boat-tail length in calibres."),
    d_base_cal: float = typer.Option(0.85, help="Base diameter in calibres."),
) -> None:
    """McCoy (1974) MCDRAG aerodynamic zero-yaw drag coefficient estimation."""
    total_len = l_nose_cal + l_cyl_cal + l_bt_cal
    inp = McDragInputs(
        d_ref_mm=calibre_m * 1000.0,
        total_length_cal=total_len,
        nose_length_cal=l_nose_cal,
        boattail_length_cal=l_bt_cal,
        base_dia_cal=d_base_cal,
        meplat_dia_cal=d_meplat_cal,
    )
    res = estimate_mcdrag(inp)

    typer.echo(f"\nMCDRAG Estimation for Calibre {calibre_m*1000:.1f} mm:")
    typer.echo(f"Total: {total_len:.2f} cal | Nose: {l_nose_cal:.2f} cal | Boat-tail: {l_bt_cal:.2f} cal | Base: {d_base_cal:.2f} cal\n")
    typer.echo(f"{'Mach':<8} {'Cd0 Total':<12} {'Cd Head':<12} {'Cd Skin':<12} {'Cd Base':<12} {'Cd Boat-tail'}")
    typer.echo("=" * 68)

    for r in res["table"]:
        typer.echo(
            f"{r['mach']:<8.2f} {r['cd0']:<12.4f} {r['cdh']:<12.4f} {r['cdsf']:<12.4f} {r['cdb']:<12.4f} {r['cdbt']:.4f}"
        )
    if res.get("diagnostics"):
        typer.echo("\nDiagnostics:")
        for d in res["diagnostics"]:
            typer.secho(f"  • {d}", fg=typer.colors.YELLOW)
    typer.echo("\n")
    typer.echo("\n")


@app.command("serve-api")
def cmd_serve_api(
    host: str = typer.Option("127.0.0.1", help="API host interface."),
    port: int = typer.Option(8000, help="API server port."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload."),
) -> None:
    """Start the FastAPI REST backend server."""
    import uvicorn
    typer.echo(f"Starting ammo-mc FastAPI server on http://{host}:{port} ...")
    uvicorn.run("ammo_mc.api:app", host=host, port=port, reload=reload)


@app.command("ui")
def cmd_ui(
    port: int = typer.Option(8501, help="Streamlit dashboard port."),
) -> None:
    """Launch the interactive Streamlit Workbench."""
    import subprocess
    app_path = Path(__file__).parent / "streamlit_app.py"
    typer.echo(f"Launching Streamlit Workbench from {app_path} on port {port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)])


if __name__ == "__main__":
    app()
