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
from ammo_mc.validation import (
    generate_vv_certificate_html,
    generate_vv_certificate_markdown,
    run_extended_vv_benchmarks,
    run_validation_suite,
)

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
    extended: bool = typer.Option(False, "--extended", help="Run extended 40-check V&V suite (22 workbook + 18 physics benchmarks)."),
    out_html: Optional[Path] = typer.Option(None, "--out-html", help="Write self-contained HTML V&V audit certificate."),
    out_md: Optional[Path] = typer.Option(None, "--out-md", help="Write Markdown V&V audit certificate."),
) -> None:
    """Run the Ammunition Monte Carlo Knowledge Base validation suite (V-01a..V-12 & B-01..B-18)."""
    run_ext = bool(extended or out_html is not None or out_md is not None)
    if run_ext:
        typer.echo("Running extended 40-check V&V suite (workbook V-01a..V-12 + physics B-01..B-18)...\n")
        report = run_extended_vv_benchmarks(seed=seed, n_mc_v08=n_mc)
        if out_html is not None:
            out_html.parent.mkdir(parents=True, exist_ok=True)
            out_html.write_text(generate_vv_certificate_html(report), encoding="utf-8")
            typer.echo(f"Wrote HTML V&V certificate -> {out_html}")
        if out_md is not None:
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(generate_vv_certificate_markdown(report), encoding="utf-8")
            typer.echo(f"Wrote Markdown V&V certificate -> {out_md}")
        if as_json:
            typer.echo(json.dumps(report, indent=2))
            return

        typer.echo(f"{'ID':<7} {'Domain':<22} {'Status':<8} {'Computed':<14} {'Reference':<14} {'Util %':<9} {'Benchmark'}")
        typer.echo("=" * 105)
        all_passed = True
        for b in report["physics_benchmarks"]:
            status_color = typer.colors.GREEN if b["passed"] else typer.colors.RED
            if not b["passed"]:
                all_passed = False
            status_styled = typer.style(f"{b['status']:<8}", fg=status_color, bold=True)
            typer.echo(
                f"{b['benchmark_id']:<7} {b['domain']:<22} {status_styled} {b['computed']:<14.5g} {b['reference']:<14.5g} "
                f"{b['utilization_pct']:<8.2f}% {b['title']}"
            )
        for w in report["workbook_cases"]:
            if "FAIL" in str(w["status"]):
                all_passed = False
        s = report["summary"]
        typer.echo("=" * 105)
        typer.echo(
            f"Summary: Physics {s['physics_passed']}/{s['physics_total']} PASS | "
            f"Workbook Excel {s['workbook_excel_passed']}/{s['workbook_total']} PASS | "
            f"Literature {s['literature']['total_sources']} sources ({s['literature']['linked_source_coverage_pct']:.0f}%)"
        )
        if all_passed:
            typer.secho("\nExtended V&V Suite: ALL CHECKS PASSED.", fg=typer.colors.GREEN, bold=True)
        else:
            typer.secho("\nExtended V&V Suite: FAILURES DETECTED.", fg=typer.colors.RED, bold=True)
            raise typer.Exit(code=1)
        return

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


# -------------------------------------------------------------------------
# COUNTER-UAS ENGINEERING SUB-CLI (ammo-mc cuas ...)
# -------------------------------------------------------------------------

cuas_app = typer.Typer(help="Counter-UAS multi-shotgun, kill-chain, site defense, layered & tornado commands.")
app.add_typer(cuas_app, name="cuas")


@cuas_app.command("kill-chain")
def cmd_cuas_kill_chain(
    drone_model: str = typer.Option("fpv_quad_small", help="Target drone preset key."),
    aspect_mode: str = typer.Option("head_on_frontal", help="Presentation aspect mode."),
    speed_mps: float = typer.Option(28.0, help="Drone approach speed (m/s)."),
    weather: str = typer.Option("clear_day", help="Weather regime (clear_day, overcast_dusk, night_clear, heavy_fog_rain)."),
    fiber_optic: bool = typer.Option(False, "--fiber-optic", help="Simulate RF-silent fiber-optic threat."),
    barrels: int = typer.Option(6, help="Number of turret barrels."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", help="Ammunition preset."),
    trials: int = typer.Option(80, help="Monte Carlo trial count."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Evaluate multi-sensor C-UAS detection & kill-chain timeline."""
    from ammo_mc.cuas_mission import simulate_cuas_kill_chain_mc

    res = simulate_cuas_kill_chain_mc(
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        drone_speed_mps=speed_mps,
        weather=weather,
        fiber_optic_autonomous=fiber_optic,
        n_barrels=barrels,
        ammunition_preset=ammo,
        n_trials=trials,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2, default=str))
        return
    sm = res["summary"]
    typer.echo(f"Drone: {sm['drone_name']} ({sm['aspect_mode']}) | Weather: {sm['weather']} | Fiber-Optic: {sm['fiber_optic_autonomous']}")
    typer.echo(f"Fused Detection: {sm['fused_detection_range_m']:.1f} m | Open Fire: {sm['first_shot_range_m']:.1f} m | Overhead: {sm['t_overhead_s']:.2f} s")
    typer.echo(f"Kill-Chain Pk:   {sm['cumulative_pk_pct']:.1f} % | Breach Rate: {sm['breach_rate_pct']:.1f} % | Bottleneck: {sm['limiting_bottleneck']}")


@cuas_app.command("site-defense")
def cmd_cuas_site_defense(
    preset: str = typer.Option("fob_compound_4_turret", help="Site defense preset key."),
    lat: float = typer.Option(32.9000, help="Origin WGS84 latitude (deg)."),
    lon: float = typer.Option(-106.4000, help="Origin WGS84 longitude (deg)."),
    heading_deg: float = typer.Option(0.0, help="Site clockwise map rotation (deg)."),
    sdz_range_m: float = typer.Option(280.0, help="Extended fallout SDZ radius (m)."),
    raids: int = typer.Option(50, help="Monte Carlo raid trials."),
    out_geojson: Optional[Path] = typer.Option(None, "--out-geojson", help="Optional GeoJSON export path."),
    out_kml: Optional[Path] = typer.Option(None, "--out-kml", help="Optional KML export path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON summary."),
) -> None:
    """Evaluate multi-turret 360-deg site coverage, NFZ conflicts, and export GIS layers."""
    from ammo_mc.cuas_analytics import build_cuas_site_geojson, export_cuas_site_kml
    from ammo_mc.cuas_mission import simulate_multiturret_site_defense_mc

    res = simulate_multiturret_site_defense_mc(
        site_preset=preset,
        elevation_sdz_range_m=sdz_range_m,
        n_raid_trials=raids,
    )
    if out_geojson:
        geo = build_cuas_site_geojson(res, origin_lat=lat, origin_lon=lon, sdz_radius_m=sdz_range_m, heading_deg=heading_deg)
        out_geojson.parent.mkdir(parents=True, exist_ok=True)
        out_geojson.write_text(json.dumps(geo, indent=2), encoding="utf-8")
    if out_kml:
        kml = export_cuas_site_kml(res, origin_lat=lat, origin_lon=lon, sdz_radius_m=sdz_range_m, heading_deg=heading_deg, name=preset)
        out_kml.parent.mkdir(parents=True, exist_ok=True)
        out_kml.write_text(kml, encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(res["summary"], indent=2))
        return
    sm = res["summary"]
    typer.echo(f"Site: {sm['site_name']} ({sm['n_turrets']} turrets)")
    typer.echo(f"Coverage (>=1): {sm['perimeter_coverage_pct']:.1f}% | Overlap (>=2): {sm['overlapping_coverage_pct']:.1f}% | Blind Spots: {sm['blind_spot_pct']:.1f}%")
    typer.echo(f"NFZ Conflict: {sm['nfz_fallout_conflict_pct']:.1f}% | Zero-Breach Raid Stop: {sm['site_zero_breach_pct']:.1f}%")


@cuas_app.command("compare-ab")
def cmd_cuas_compare_ab(
    drone_model: str = typer.Option("fpv_quad_small", help="Benchmark drone class."),
    speed_mps: float = typer.Option(25.0, help="Threat approach speed (m/s)."),
    out_csv: Optional[Path] = typer.Option(None, "--out-csv", help="Optional scorecard CSV export path."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Run side-by-side A/B/C turret architecture scorecard."""
    import pandas as pd
    from ammo_mc.cuas_analytics import compare_cuas_architectures_ab

    res = compare_cuas_architectures_ab(drone_model=drone_model, drone_speed_mps=speed_mps)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["scorecard"]).to_csv(out_csv, index=False)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(f"Winner: {res['summary']['winner_label']} (MEI={res['summary']['winner_mei']})")
    typer.echo(f"{'Architecture':<48} {'MEI':<6} {'KC Pk%':<8} {'Swarm%':<8} {'Temp(C)':<9} {'$/Kill'}")
    typer.echo("-" * 92)
    for r in res["scorecard"]:
        typer.echo(
            f"{r['label']:<48} {r['mei_score']:<6.1f} {r['kill_chain_pk_pct']:<8.1f} {r['swarm_survival_pct']:<8.1f} {r['peak_barrel_temp_c']:<9.1f} ${r['cost_per_kill_usd']:.2f}"
        )


@cuas_app.command("layered")
def cmd_cuas_layered(
    outer: str = typer.Option("rf_gnss_jammer", help="Outer layer ID (rf_gnss_jammer, interceptor_drone_fpv, autocannon_30mm_airburst, directed_energy_hel)."),
    fo_frac: float = typer.Option(0.25, help="Fraction of fiber-optic / RF-silent drones (0..1)."),
    weather: str = typer.Option("clear_day", help="Weather (clear_day, heavy_fog_rain)."),
    waves: int = typer.Option(5, help="Number of incoming attack waves."),
    drones_per_wave: int = typer.Option(5, help="Drones per wave."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Evaluate Outer Soft/Hard-Kill + Inner Multi-Shotgun layered defense."""
    from ammo_mc.cuas_analytics import simulate_layered_cuas_defense_mc

    res = simulate_layered_cuas_defense_mc(
        outer_layer_id=outer,
        fiber_optic_fraction=fo_frac,
        weather=weather,
        n_waves=waves,
        drones_per_wave=drones_per_wave,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(f"Outer Tier: {sm['outer_layer_name']} | FO Mix: {sm['fiber_optic_fraction_pct']:.0f}% | Weather: {sm['weather']}")
    typer.echo(f"Layered Survival: {sm['layered_survival_pct']:.1f}% (Inner-Only: {sm['inner_only_survival_pct']:.1f}%, Gain: {sm['survival_gain_pct']:+.1f}%)")
    typer.echo(f"Inner Rounds Saved: {sm['inner_rounds_saved']:.1f} | Temp Drop: -{sm['peak_temp_reduction_c']:.1f} C | Blended $/Kill: ${sm['blended_cost_per_kill_usd']:.2f}")


@cuas_app.command("sensitivity")
def cmd_cuas_sensitivity(
    perturb_pct: float = typer.Option(20.0, help="One-at-a-time +/- perturbation percentage."),
    trials: int = typer.Option(60, help="Trials per perturbation point."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Run C-UAS Kill-Chain parameter sensitivity Tornado analysis."""
    from ammo_mc.cuas_analytics import analyze_cuas_sensitivity_tornado

    res = analyze_cuas_sensitivity_tornado(perturbation_pct=perturb_pct, n_trials=trials)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(f"Baseline Kill-Chain Pk: {sm['baseline_value']:.1f}% | Top Driver: {sm['top_driver']} (swing {sm['top_swing']:.1f}%)")
    for r in res["rows"]:
        typer.echo(f"  {r['label']:<38} Low={r['val_at_low']:<5.1f}% Base={r['val_at_base']:<5.1f}% High={r['val_at_high']:<5.1f}% Swing={r['swing']:.1f}%")


@cuas_app.command("sweep")
def cmd_cuas_sweep(
    param: str = typer.Option("range", help="Sweep axis: 'range' or 'speed'."),
    start: float = typer.Option(25.0, help="Sweep start value."),
    stop: float = typer.Option(65.0, help="Sweep stop value."),
    steps: int = typer.Option(5, help="Number of sweep points."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", help="Ammunition preset."),
    out_csv: Optional[Path] = typer.Option(None, "--out-csv", help="Optional CSV export path."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Sweep intercept range or threat speed across C-UAS Kill Chain."""
    import numpy as np
    import pandas as pd
    from ammo_mc.cuas_mission import simulate_cuas_kill_chain_mc

    vals = np.linspace(start, stop, max(2, steps))
    rows = []
    for v in vals:
        kw: dict[str, object] = {"ammunition_preset": ammo, "n_trials": 50}
        if param.lower() == "speed":
            kw["drone_speed_mps"] = float(v)
        else:
            kw["max_weapon_range_m"] = float(v)
        r = simulate_cuas_kill_chain_mc(**kw)  # type: ignore[arg-type]
        sm = r["summary"]
        rows.append(
            {
                "param_value": round(float(v), 2),
                "kill_chain_pk_pct": sm["cumulative_pk_pct"],
                "breach_rate_pct": sm["breach_rate_pct"],
                "mean_salvos": sm["mean_salvos_fired"],
            }
        )
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    typer.echo(f"Sweep Axis: {param} ({start} -> {stop} across {len(rows)} steps)")
    for r in rows:
        typer.echo(f"  {param}={r['param_value']:<7.1f} -> Kill-Chain Pk={r['kill_chain_pk_pct']:5.1f}% | Breach={r['breach_rate_pct']:5.1f}% | Salvos={r['mean_salvos']:.2f}")


@cuas_app.command("unified-raid")
def cmd_cuas_unified_raid(
    site: str = typer.Option("fob_compound_4_turret", help="Site preset."),
    terrain: Optional[str] = typer.Option(None, "--terrain", help="Optional DEM terrain preset."),
    drones: int = typer.Option(10, help="Number of swarm drones."),
    fiber_optic: float = typer.Option(0.30, help="Fraction of RF-silent fiber-optic drones (0-1)."),
    trials: int = typer.Option(20, help="Monte Carlo trial count."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Run Unified Dynamic 3D Cooperative Multi-Turret C-UAS Raid simulation."""
    from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
    from ammo_mc.terrain_dem import build_preset_terrain_dem

    dem_obj = build_preset_terrain_dem(terrain) if terrain and terrain != "flat_airfield" else None
    res = simulate_unified_tactical_raid_mc(
        site_preset=site,
        terrain_dem=dem_obj,
        n_drones=drones,
        fiber_optic_fraction=fiber_optic,
        n_mc_trials=trials,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"Unified 3D Raid [{sm['site_name']}]: Zero-Breach={sm['zero_breach_pct']:.1f}% | "
        f"Kill-Rate={sm['mean_kill_rate_pct']:.1f}% | $/Kill=${sm['cost_per_kill_usd']:.2f} | "
        f"Peak Temp={sm['peak_barrel_temp_c']:.1f} C"
    )


@cuas_app.command("adversarial")
def cmd_cuas_adversarial(
    generations: int = typer.Option(5, "--generations", "-g", help="Number of co-evolution generations."),
    episodes: int = typer.Option(18, "--episodes", "-e", help="MC episodes per Red-vs-Blue matchup."),
    mount_limit_kn: float = typer.Option(3.5, "--mount-limit-kn", help="Structural mount force limit in kN."),
    max_barrels: int = typer.Option(10, "--max-barrels", help="Maximum allowable shotgun barrels."),
    max_swarm: int = typer.Option(8, "--max-swarm", help="Maximum Red swarm size."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid mixed-load inner/outer barrel search."),
    robust: bool = typer.Option(False, "--robust", help="Enable CVaR-20% calibrated/OTM robust optimization."),
    out_csv: Optional[Path] = typer.Option(None, "--out-csv", help="Optional CSV export path for evaluated matchup dataset."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    as_json: bool = typer.Option(False, "--json", help="Output complete co-evolution result as JSON."),
) -> None:
    """Run Adversarial Red-vs-Blue Competing Monte Carlo co-evolution & GP ML protection optimizer."""
    from ammo_mc.cuas_adversarial import (
        export_adversarial_dataset_csv,
        run_adversarial_coevolution_ml,
    )

    res = run_adversarial_coevolution_ml(
        n_generations=generations,
        n_mc_episodes=episodes,
        mount_load_limit_kn=mount_limit_kn,
        max_barrels=max_barrels,
        max_swarm_size=max_swarm,
        seed=seed,
        allow_hybrid=hybrid,
        robust_uncertainty_mode=robust,
    )
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_data = export_adversarial_dataset_csv(res)
        out_csv.write_text(csv_data, encoding="utf-8")
        typer.echo(f"Exported adversarial dataset ({res['matchups_evaluated']} matchups) to {out_csv}")
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return

    b = res["best_blue_config"]
    r = res["deadliest_red_threat"]
    wc = res["worst_case_defense_gain"]
    diag = res["gp_surrogate_diagnostics"]
    typer.echo(
        f"Adversarial Co-Evolution Complete ({res['matchups_evaluated']} matchups, {generations} gen):\n"
        f"  Nash Game Value V*         : {res['nash_value']:.2f} (Floor)\n"
        f"  Evolved Worst-Case Defense : {wc['evolved_worst_case']:.1f}% (+{wc['worst_case_gain_pct']:.1f}% vs Gen 0)\n"
        f"  GP Surrogate 5-Fold OOF    : R2={diag['cv_r2']:.3f} | RMSE={diag['cv_rmse']:.2f}\n"
        f"  Best Evolved Blue Turret   : {b['n_barrels']}B {b['layout']} ({b['ammo']}, {b['choke']}, splay={b['splay_angle_deg']} deg, ripple={b['ripple_delay_ms']}ms)\n"
        f"  Deadliest Red Threat       : {r['drone_model']} (swarm={r['swarm_size']}, tactic={r['tactic']}, {r['maneuver_g']}G {r['evasion']})"
    )


@cuas_app.command("rl-policy")
def cmd_cuas_rl_policy(
    generations: int = typer.Option(10, "--generations", "-g", help="Evolution Strategies training generations."),
    episodes: int = typer.Option(12, "--episodes", "-e", help="Benchmark episodes per threat scenario."),
    barrels: int = typer.Option(8, "--barrels", "-b", help="Turret barrel count."),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Enable dual-ring hybrid loadouts."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON training report."),
) -> None:
    """Train Dynamic Shot-by-Shot Neural Fire-Control Policy and benchmark against heuristics."""
    from ammo_mc.cuas_rl_policy import train_fire_control_policy_es

    res = train_fire_control_policy_es(
        blue_config={
            "layout_type": "concentric" if hybrid else "circular",
            "n_barrels": barrels,
            "barrels_per_burst": min(barrels, 3),
            "hybrid_mode": hybrid,
        },
        n_generations=generations,
        pop_size=12,
        n_eval_episodes=episodes,
        seed=seed,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"RL Fire-Control Training Complete ({generations} gen):\n"
        f"  RL Zero-Breach : {sm['rl_zero_breach_pct']:.1f}% ({sm['uplift_zero_breach_pp']:+.1f} pp vs {sm['best_heuristic_name']})\n"
        f"  RL Rounds/Kill : {sm['rl_mean_rounds_per_kill']:.2f} ({sm['rounds_saved_pct']:+.1f}% saved) | Avg Burst: {sm['rl_mean_burst_size']:.2f}"
    )


@cuas_app.command("battlespace-ml")
def cmd_cuas_battlespace_ml(
    terrain: str = typer.Option("rolling_hills_fob", "--terrain", help="DEM terrain preset."),
    generations: int = typer.Option(2, "--generations", "-g", help="Battlespace co-evolution generations."),
    trials: int = typer.Option(4, "--trials", "-n", help="Raid MC trials per evaluation."),
    turrets: int = typer.Option(4, "--turrets", help="Max turrets per site."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON battlespace Nash results."),
) -> None:
    """Co-evolve multi-turret 3D battlespace architectures against heterogeneous Red swarm packages."""
    from ammo_mc.cuas_battlespace_ml import run_battlespace_coevolution_mc

    res = run_battlespace_coevolution_mc(
        terrain_preset=terrain,
        n_generations=generations,
        n_mc_trials=trials,
        max_turrets=turrets,
        seed=seed,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"3D Battlespace Co-Evolution Complete [{terrain}]:\n"
        f"  Nash Game Value V*  : {sm['nash_value']:.2f}\n"
        f"  Best Site + Layer   : {sm['best_blue_name']} ({sm['best_outer_layer']})\n"
        f"  Deadliest Red Pkg   : {sm['deadliest_red_name']}"
    )


@cuas_app.command("track-replay")
def cmd_cuas_track_replay(
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="Recorded ENU/GPS drone track CSV path."),
    scenario: str = typer.Option("evasive_fpv_dive", "--scenario", help="Sample scenario if --csv omitted."),
    barrels: int = typer.Option(6, "--barrels", "-b", help="Turret barrel count."),
    breach_r: float = typer.Option(12.0, "--breach-r", help="Keepout breach radius (m)."),
    trials: int = typer.Option(25, "--trials", "-n", help="MC replay episodes."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON replay summary."),
) -> None:
    """Replay recorded drone flight tracks (CSV/MAVLink) against a calibrated shotgun turret."""
    from ammo_mc.cuas_track_replay import (
        generate_sample_drone_track_csv,
        simulate_recorded_track_engagement_mc,
    )

    csv_txt = csv_path.read_text(encoding="utf-8") if csv_path else generate_sample_drone_track_csv(scenario=scenario)
    res = simulate_recorded_track_engagement_mc(
        csv_txt,
        turret_config={"n_barrels": barrels},
        breach_radius_m=breach_r,
        n_mc_trials=trials,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"Recorded Track Replay ({sm['n_tracks']} tracks, {sm['coord_mode']}): "
        f"Zero-Breach={sm['zero_breach_pct']:.1f}% | Kill-Rate={sm['mean_kill_rate_pct']:.1f}% | Peak G={sm['peak_lat_g']:.2f}G"
    )


@cuas_app.command("swap-c")
def cmd_cuas_swap_c(
    barrels: int = typer.Option(6, "--barrels", "-b", help="Number of barrels."),
    layout: str = typer.Option("circular", "--layout", help="Muzzle layout geometry."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", "--ammo", help="Ammunition preset."),
    choke: str = typer.Option("Modified", "--choke", help="Default choke."),
    pointing: str = typer.Option("convergent", "--pointing", help="Pointing mode."),
    splay_mrad: float = typer.Option(0.0, "--splay-mrad", help="Curtain splay angle in mrad."),
    thickness_mm: float = typer.Option(22.0, "--thickness-mm", help="Clamp plate thickness in mm for FEA & STL."),
    material: str = typer.Option("7075-T6", "--material", help="Clamp plate alloy (7075-T6|6061-T6|4140-Steel)."),
    out_dxf: Optional[Path] = typer.Option(None, "--out-dxf", help="Write parametric AutoCAD DXF drawing."),
    out_svg: Optional[Path] = typer.Option(None, "--out-svg", help="Write parametric blueprint SVG."),
    out_stl: Optional[Path] = typer.Option(None, "--out-stl", help="Write extruded 3D ASCII STL CAD solid."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON SWaP-C + FEA breakdown."),
) -> None:
    """Compute turret SWaP-C mass/power/BOM, bolt-shear FEA, and export 2D/3D muzzle-plate CAD."""
    from ammo_mc.cuas_hardware import (
        estimate_turret_swap_c,
        evaluate_muzzle_plate_fea,
        export_muzzle_plate_dxf,
        export_muzzle_plate_stl,
        export_muzzle_plate_svg,
    )
    from ammo_mc.small_arms import generate_multishotgun_array_layout

    res = estimate_turret_swap_c(n_barrels=barrels, layout_type=layout, ammunition_preset=ammo)
    bbls = generate_multishotgun_array_layout(
        layout_type=layout, n_barrels=barrels, default_choke=choke, default_ammunition_preset=ammo
    )
    fea = evaluate_muzzle_plate_fea(
        bbls,
        layout_type=layout,
        peak_recoil_force_kn=float(res["summary"]["peak_recoil_force_kn"]),
        plate_thickness_mm=thickness_mm,
        material=material,
    )
    res["fea"] = fea
    if out_dxf:
        out_dxf.parent.mkdir(parents=True, exist_ok=True)
        out_dxf.write_text(
            export_muzzle_plate_dxf(bbls, layout_type=layout, splay_angle_mrad=splay_mrad, pointing_mode=pointing),
            encoding="utf-8",
        )
        typer.echo(f"Wrote AutoCAD DXF to {out_dxf}")
    if out_svg:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        out_svg.write_text(
            export_muzzle_plate_svg(bbls, layout_type=layout, splay_angle_mrad=splay_mrad, pointing_mode=pointing),
            encoding="utf-8",
        )
        typer.echo(f"Wrote SVG blueprint to {out_svg}")
    if out_stl:
        out_stl.parent.mkdir(parents=True, exist_ok=True)
        out_stl.write_text(
            export_muzzle_plate_stl(bbls, layout_type=layout, plate_thickness_mm=thickness_mm),
            encoding="utf-8",
        )
        typer.echo(f"Wrote 3D ASCII STL to {out_stl}")
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"SWaP-C [{barrels}B {layout}]: Mass={sm['total_mass_kg']:.1f} kg | Power={sm['peak_power_w']:.0f} W | "
        f"Recoil={sm['peak_recoil_force_kn']:.2f} kN | BOM=${sm['unit_bom_usd']:,.0f} | "
        f"FEA={fea['verdict']} (SF={fea['overall_safety_factor']:.2f}x) | Mount={sm['recommended_mount_class']}"
    )


@cuas_app.command("calibrate-twin")
def cmd_cuas_calibrate_twin(
    drone: str = typer.Option("fpv_quad_small", "--drone", help="Reference drone model."),
    range_m: float = typer.Option(45.0, "--range-m", help="Required intercept range (m)."),
    barrels: int = typer.Option(6, "--barrels", "-b", help="Baseline turret barrels."),
    save_name: Optional[str] = typer.Option(None, "--save-name", help="Save calibrated profile into SQLite store."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON calibration & drift summary."),
) -> None:
    """Run closed-loop empirical recalibration across range/lethality/hardware and analyze design drift."""
    from ammo_mc.cuas_calibration_twin import (
        CalibrationProfileStore,
        analyze_calibration_design_delta,
        calibrate_hardware_bench_tests,
        calibrate_livefire_drone_lethality,
        calibrate_multirange_pattern_and_drag,
        generate_sample_hardware_bench_tests,
        generate_sample_livefire_drone_trials,
        generate_sample_multirange_pattern_rows,
        merge_twin_updates,
        recommend_next_range_tests,
    )

    mr = calibrate_multirange_pattern_and_drag(generate_sample_multirange_pattern_rows())
    lf = calibrate_livefire_drone_lethality(generate_sample_livefire_drone_trials(drone_model=drone), drone_model=drone)
    hb = calibrate_hardware_bench_tests({"n_barrels": barrels}, generate_sample_hardware_bench_tests(barrels))
    twin = merge_twin_updates(None, {**mr["twin_patch"], **lf["twin_patch"], **hb["twin_patch"]}, domain="cli")
    drift = analyze_calibration_design_delta({"n_barrels": barrels, "barrels_per_burst": 2}, twin, drone_model=drone, required_range_m=range_m)
    plan = recommend_next_range_tests({"n_barrels": barrels, "barrels_per_burst": 2}, twin, drone_model=drone, target_range_m=range_m)
    saved_id = None
    if save_name:
        saved_id = CalibrationProfileStore().save_profile(save_name, twin)["id"]
    payload = {"twin": twin, "drift": drift, "test_plan": plan, "saved_profile_id": saved_id}
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(
        f"Calibrated Digital Twin [{drone}]: spread={twin['spread_factor']:.2f}x (p={twin['spread_exponent']:.2f}) | "
        f"E_crit={twin['drone_ecrit_scale']:.2f}x | R80 {drift['nominal']['r80_m']:.1f}m -> {drift['calibrated']['r80_m']:.1f}m | "
        f"Top Fix={drift['recommended_fixes'][0]['fix_name']}"
    )


@cuas_app.command("safety-envelope")
def cmd_cuas_safety_envelope(
    barrels: int = typer.Option(6, "--barrels", "-b", help="Turret barrels."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", "--ammo", help="Ammunition preset."),
    elevation: float = typer.Option(35.0, "--elevation", help="Firing elevation angle (deg)."),
    env: str = typer.Option("urban_permissive", "--env", help="Environment (urban_permissive|urban_canyon)."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON safety envelope breakdown."),
) -> None:
    """Evaluate acoustic blast dB contours, friendly danger cone & falling-pellet collateral risk."""
    from ammo_mc.cuas_safety_envelope import evaluate_cuas_collateral_safety_envelope

    res = evaluate_cuas_collateral_safety_envelope(
        n_barrels=barrels, ammunition_preset=ammo, firing_elevation_deg=elevation, environment_type=env
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"Collateral Safety Envelope [{sm['verdict']}]: 140dB={sm['acoustic_140db_downrange_m']:.0f}m | "
        f"Crew Rear={sm['crew_rear_140db_standoff_m']:.1f}m | Fallout 99%={sm['fallout_99pct_range_m']:.0f}m | "
        f"Lethal(>15J)={sm['lethal_15j_pct']:.1f}%"
    )


@cuas_app.command("hil-trace")
def cmd_cuas_hil_trace(
    scenario: str = typer.Option("evasive_fpv_dive", "--scenario", help="Track scenario or CSV path."),
    dt_ms: float = typer.Option(100.0, "--dt-ms", help="Control-loop step size (ms)."),
    breach_r: float = typer.Option(12.0, "--breach-r", help="Perimeter keepout radius (m)."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON HIL command stream."),
) -> None:
    """Step RL fire-control policy over recorded drone telemetry millisecond-by-millisecond."""
    from ammo_mc.cuas_rl_policy import run_hil_fire_control_trace

    res = run_hil_fire_control_trace(None, track_csv_or_scenario=scenario, dt_ms=dt_ms, breach_radius_m=breach_r)
    if as_json:
        typer.echo(json.dumps({"summary": res["summary"], "frames": res["frames"]}, indent=2))
        return
    sm = res["summary"]
    typer.echo(
        f"HIL Policy Trace ({sm['policy_name']}): Frames={sm['n_frames']} | Fire Cmds={sm['fire_commands']} | "
        f"Hold Gates={sm['hold_fire_frames']} | Neutralized={sm['targets_neutralized']}/{sm['total_tracks']}"
    )


@cuas_app.command("wizard")
def cmd_cuas_wizard(
    drone: str = typer.Option("fpv_quad_small", "--drone", help="Target drone model."),
    speed: float = typer.Option(25.0, "--speed", help="Threat speed (m/s)."),
    range_m: float = typer.Option(45.0, "--range-m", help="Required intercept range (m)."),
    mount: str = typer.Option("light_ugv", "--mount", help="Mount class constraint."),
    priority: str = typer.Option("balanced", "--priority", help="Ranking priority (balanced|low_cost|max_lethality)."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON wizard ranking."),
) -> None:
    """Size and rank Top 3 feasible C-UAS multi-barrel turret configs from mission inputs."""
    from ammo_mc.cuas_wizard import run_turret_sizing_wizard

    res = run_turret_sizing_wizard(
        drone_model=drone,
        threat_speed_mps=speed,
        required_range_m=range_m,
        mount_class=mount,
        budget_priority=priority,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(f"Turret Sizing Wizard ({drone} @ {range_m:.0f}m, {mount}):")
    for idx, row in enumerate(res["top_3"], start=1):
        typer.echo(
            f"  #{idx} {row['name']}: Pk={row['pk_at_range_pct']:.1f}% | R80={row['r80_m']:.0f}m | "
            f"{row['mass_kg']:.1f}kg | {row['peak_recoil_kn']:.2f}kN | $/Kill=${row['cost_per_kill_usd']:.2f}"
        )


@cuas_app.command("vv-benchmarks")
def cmd_cuas_vv_benchmarks(
    as_json: bool = typer.Option(False, "--json", help="Output JSON benchmark table."),
) -> None:
    """Run published C-UAS experimental benchmarks (E-01..E-06)."""
    from ammo_mc.cuas_scientific_vv import run_experimental_validation_benchmarks

    res = run_experimental_validation_benchmarks()
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(f"C-UAS Experimental Benchmarks: {res['passed']}/{res['total']} PASS")
    for b in res["benchmarks"]:
        typer.echo(f"  [{b['id']}] {'PASS' if b['passed'] else 'FAIL'} | {b['name']}")


@cuas_app.command("blind-holdout")
def cmd_cuas_blind_holdout(
    choke: str = typer.Option("Modified", "--choke", help="Nominal choke preset."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", "--ammo", help="Ammunition preset."),
    drone: str = typer.Option("fpv_14in_36cm_pg7m", "--drone", help="Target drone model."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON blind holdout report."),
) -> None:
    """Run blind out-of-sample range holdout validation with ASME V&V 20 metrics."""
    from ammo_mc.cuas_scientific_vv import (
        generate_sample_blind_holdout_dataset,
        run_blind_holdout_validation,
    )

    ds = generate_sample_blind_holdout_dataset(nominal_choke=choke, ammunition_preset=ammo)
    res = run_blind_holdout_validation(
        pattern_rows=ds["pattern_rows"],
        livefire_rows=ds["livefire_rows"],
        holdout_ranges_m=(65.0, 75.0),
        nominal_choke=choke,
        ammunition_preset=ammo,
        drone_model=drone,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(
        f"Blind Holdout Validation: Nominal RMSE={res['holdout_rmse_nominal_m']:.3f}m -> "
        f"Calibrated={res['holdout_rmse_calibrated_m']:.3f}m ({res['error_reduction_pct']:+.1f}%)"
    )


@cuas_app.command("uq-envelope")
def cmd_cuas_uq_envelope(
    ammo: str = typer.Option("ALDA_BUCKSHOT_12_76_COPPER_32G", "--ammo", help="Ammunition preset."),
    drone: str = typer.Option("fpv_14in_36cm_pg7m", "--drone", help="Target drone model."),
    draws: int = typer.Option(6, "--draws", help="Latin Hypercube epistemic draws."),
    volleys: int = typer.Option(6, "--volleys", help="Aleatory volleys per draw."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON UQ P-box envelope."),
) -> None:
    """Compute nested epistemic x aleatory UQ P-box R80 envelope."""
    from ammo_mc.cuas_scientific_vv import compute_epistemic_aleatory_uq_envelope

    res = compute_epistemic_aleatory_uq_envelope(
        distance_stations_m=(20.0, 35.0, 50.0, 65.0, 75.0),
        ammunition_preset=ammo,
        drone_model=drone,
        n_epistemic_draws=draws,
        n_volleys_per_draw=volleys,
    )
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(
        f"UQ P-Box R80: Median={res['r80_median_m']:.1f}m "
        f"(P05={res['r80_p05_m']:.1f}m, P95={res['r80_p95_m']:.1f}m)"
    )


@cuas_app.command("doe-plan")
def cmd_cuas_doe_plan(
    range_m: float = typer.Option(50.0, "--range-m", help="Primary validation slant range (m)."),
    p0: float = typer.Option(0.50, "--p0", help="Null-hypothesis kill probability."),
    p1: float = typer.Option(0.80, "--p1", help="Target alternative kill probability."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON DOE & SPRT plan."),
) -> None:
    """Generate pre-test range Design of Experiments and Wald SPRT boundaries."""
    from ammo_mc.cuas_scientific_vv import plan_range_validation_doe

    res = plan_range_validation_doe(target_range_m=range_m, p0=p0, p1=p1)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(
        f"DOE Plan @ {range_m:.0f}m: n_prec_ci={res['n_required_precision_ci']}, "
        f"SPRT={res['sprt_bounds']}"
    )


@cuas_app.command("doppler-drag")
def cmd_cuas_doppler_drag(
    csv_file: Optional[Path] = typer.Option(None, "--file", help="Optional Doppler velocity CSV file."),
    ammo: str = typer.Option("12G_NO4_BUCK_C_UAS", "--ammo", help="Reference ammunition preset."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON Doppler drag fit."),
) -> None:
    """Fit in-flight pellet Cd scale and V0 from Doppler radar track CSV."""
    from ammo_mc.cuas_scientific_vv import (
        calibrate_doppler_radar_drag,
        generate_sample_doppler_radar_csv,
    )

    csv_t = csv_file.read_text(encoding="utf-8", errors="replace") if csv_file else generate_sample_doppler_radar_csv(ammunition_preset=ammo)
    res = calibrate_doppler_radar_drag(csv_t, ammunition_preset=ammo)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(
        f"Doppler Drag Fit: V0={res['fitted_v0_mps']:.1f} m/s | "
        f"Cd_scale={res['calibrated_cd_scale']:.3f} | RMSE={res['rmse_velocity_mps']:.2f} m/s"
    )


@cuas_app.command("shotstring")
def cmd_cuas_shotstring(
    csv_file: Optional[Path] = typer.Option(None, "--file", help="Optional shot-string & weather CSV."),
    range_m: float = typer.Option(50.0, "--range-m", help="Target range (m)."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON shot-string summary."),
) -> None:
    """Parse acoustic/optical shot-string arrival times and moist air density."""
    from ammo_mc.cuas_scientific_vv import (
        generate_sample_shotstring_csv,
        parse_weather_and_shotstring_csv,
    )

    csv_t = csv_file.read_text(encoding="utf-8", errors="replace") if csv_file else generate_sample_shotstring_csv(range_m=range_m)
    res = parse_weather_and_shotstring_csv(csv_t, range_m=range_m)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(
        f"Shot-String @ {range_m:.0f}m: L={res['string_length_m']:.2f}m ({res['string_duration_ms']:.2f}ms) | "
        f"rho={res['air_density_kg_m3']:.3f} kg/m3"
    )


@cuas_app.command("parse-instrument")
def cmd_cuas_parse_instrument(
    file: Optional[Path] = typer.Option(None, "--file", help="Native LabRadar/Garmin/Kestrel/PX4 file path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON parsed instrument packet."),
) -> None:
    """Auto-detect and parse LabRadar, Garmin Xero C1, Kestrel LiNK, or PX4 telemetry files."""
    from ammo_mc.cuas_scientific_vv import (
        detect_and_parse_instrument_file,
        generate_sample_labradar_native_csv,
    )

    if file is not None:
        raw = file.read_bytes()
        name = file.name
    else:
        raw = generate_sample_labradar_native_csv().encode("utf-8")
        name = "labradar_sample.csv"
    res = detect_and_parse_instrument_file(name, raw)
    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return
    typer.echo(f"Instrument Auto-Parse [{res['instrument_type']}]: patch keys={list((res.get('twin_patch') or {}).keys())}")


@cuas_app.command("stl-vuln")
def cmd_cuas_stl_vuln(
    file: Optional[Path] = typer.Option(None, "--file", help="Optional .stl drone CAD mesh path."),
    aspect: str = typer.Option("head_on_frontal", "--aspect", help="Presentation aspect mode."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON STL projection summary."),
) -> None:
    """Project 3D STL drone CAD mesh triangles for presented area & surface obliquity."""
    from ammo_mc.small_arms import (
        generate_sample_drone_stl_bytes,
        parse_stl_mesh,
        project_stl_mesh_vulnerability,
    )

    stl_b = file.read_bytes() if file is not None else generate_sample_drone_stl_bytes("quadcopter")
    mesh = parse_stl_mesh(stl_b)
    proj = project_stl_mesh_vulnerability(mesh, aspect_mode=aspect)
    out = {
        "n_triangles": mesh["n_triangles"],
        "format": mesh["format"],
        "aspect_mode": aspect,
        "presented_area_cm2": proj["presented_area_cm2"],
        "span_xyz_m": mesh["span_xyz_m"],
    }
    if as_json:
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(f"STL Projection ({aspect}): {out['n_triangles']} tris, A_pres={out['presented_area_cm2']:.1f} cm2")


@app.command("literature")
def cmd_literature(
    query: str = typer.Option("", "--query", "-q", help="Filter sources by keyword, author, DOI, or formula."),
    workbench: Optional[str] = typer.Option(None, "--workbench", "-w", help="Filter by workbench slug (e.g. cuas, sim, iatg)."),
    as_json: bool = typer.Option(False, "--json", help="Output literature summary and matching entries as JSON."),
) -> None:
    """Browse the Scientific Literature Database & Model Provenance Map."""
    from ammo_mc.literature import build_literature_audit_summary, search_literature

    summary = build_literature_audit_summary()
    rows = search_literature(query=query, workbench_slug=workbench)
    if as_json:
        typer.echo(json.dumps({"summary": summary, "sources": rows}, indent=2))
        return
    typer.echo(
        f"Literature KB: {summary['total_sources']} sources ({summary['doi_sources']} DOIs) | "
        f"{summary['total_formulas']} formulas | {summary['mapped_modules']} mapped modules"
    )
    for r in rows[:20]:
        typer.echo(f"  [{r['source_id']}] ({r['year'] or 'n/a'}) {r['title'][:75]} [{r['category']}]")


if __name__ == "__main__":
    app()
