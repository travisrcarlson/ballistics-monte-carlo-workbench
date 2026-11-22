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
