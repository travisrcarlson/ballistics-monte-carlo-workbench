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


if __name__ == "__main__":
    app()
