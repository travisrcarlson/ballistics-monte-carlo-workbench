"""Tests for ammo_mc CLI commands using Typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner
import pytest

from ammo_mc.cli import app

runner = CliRunner()


def test_cli_presets() -> None:
    res = runner.invoke(app, ["presets"])
    assert res.exit_code == 0
    assert "Preset ID" in res.output
    assert "NOTIONAL-AR-01" in res.output


def test_cli_validate() -> None:
    res = runner.invoke(app, ["validate", "--n-mc", "20000"])
    assert res.exit_code == 0
    assert "ALL CASES PASSED" in res.output
    assert "V-01a" in res.output


def test_cli_mcdrag() -> None:
    res = runner.invoke(app, ["mcdrag", "--calibre-m", "0.155"])
    assert res.exit_code == 0
    assert "Cd0 Total" in res.output
    assert "0.50" in res.output


def test_cli_simulate_preset() -> None:
    res = runner.invoke(
        app,
        ["simulate", "--preset", "NOTIONAL-SA-01", "--trials", "20", "--fidelity", "L1"],
    )
    assert res.exit_code == 0
    assert "Simulation Summary" in res.output
    assert "Mean Range:" in res.output
    assert "CEP50:" in res.output


def test_cli_cuas_subcommands(tmp_path) -> None:
    import json

    r_kc = runner.invoke(app, ["cuas", "kill-chain", "--trials", "25", "--json"])
    assert r_kc.exit_code == 0
    assert "cumulative_pk_pct" in json.loads(r_kc.output)["summary"]

    geo_p = tmp_path / "site.geojson"
    kml_p = tmp_path / "site.kml"
    r_site = runner.invoke(
        app,
        ["cuas", "site-defense", "--raids", "15", "--out-geojson", str(geo_p), "--out-kml", str(kml_p)],
    )
    assert r_site.exit_code == 0
    assert geo_p.exists() and kml_p.exists()

    csv_ab = tmp_path / "ab.csv"
    r_ab = runner.invoke(app, ["cuas", "compare-ab", "--out-csv", str(csv_ab)])
    assert r_ab.exit_code == 0
    assert "Winner:" in r_ab.output and csv_ab.exists()

    r_lay = runner.invoke(app, ["cuas", "layered", "--waves", "3", "--json"])
    assert r_lay.exit_code == 0
    assert "layered_survival_pct" in json.loads(r_lay.output)["summary"]

    r_sen = runner.invoke(app, ["cuas", "sensitivity", "--trials", "30", "--json"])
    assert r_sen.exit_code == 0
    assert len(json.loads(r_sen.output)["rows"]) >= 6

    r_swp = runner.invoke(app, ["cuas", "sweep", "--param", "speed", "--steps", "3", "--json"])
    assert r_swp.exit_code == 0
    assert len(json.loads(r_swp.output)) == 3
