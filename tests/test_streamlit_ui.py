"""Headless Streamlit UI regression tests (auth gate, contextual sidebar & URL deep links)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT_APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_ui_password_gate_reject_and_unlock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMMO_MC_PASSWORD", "ci-ui-pass")
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert not at.session_state["authenticated"]

    # Wrong password keeps screen locked with error
    at.text_input(key="auth_pw_field").set_value("wrong-password")
    at.button[0].click().run()
    assert not at.session_state["authenticated"]
    assert len(at.error) >= 1

    # Valid password unlocks into default Quick-Start Home in Standard mode
    at.text_input(key="auth_pw_field").set_value("ci-ui-pass")
    at.button[0].click().run()
    assert at.session_state["authenticated"] is True
    assert not at.exception
    assert at.session_state["active_workbench"] == "Home"
    assert at.session_state["ui_expert_mode"] is False


@pytest.mark.parametrize(
    ("wb_slug", "mod_slug", "expected_wb", "expected_sub_key", "expected_sub_val"),
    [
        ("home", None, "Home", None, None),
        ("cuas", None, "C-UAS & Multi-Barrel", "cuas_stage_tool", "Anti-Drone Multi-Shotgun Array"),
        ("cuas", "array", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Anti-Drone Multi-Shotgun Array"),
        ("cuas", "comparator", "C-UAS & Multi-Barrel", "cuas_stage_tool", "A/B Turret Comparator & Layered Defense"),
        ("cuas", "site_defense", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Multi-Turret Site Defense & GIS Map"),
        ("cuas", "sensitivity", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Sensitivity & Bottleneck Tornado"),
        ("cuas", "raid", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Unified 3D Tactical Raid Simulator"),
        ("cuas", "calibration", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Field-Test Calibration & Residuals"),
        ("sim", None, "Ballistics & 3D Sim", "sim_active_subview", "Monte Carlo Pipeline"),
        ("3d", None, "Ballistics & 3D Sim", "sim_active_subview", "3D Trajectory Lab"),
        ("firing", None, "Ballistics & 3D Sim", "sim_active_subview", "Firing Tables & Met"),
        ("physics", None, "Physics & Aero Labs", "physics_active_subview", "Physics & Ballistics Labs"),
        ("mcdrag", None, "Physics & Aero Labs", "physics_active_subview", "MCDRAG & 3D CAD"),
        ("iatg", None, "Safety & IATG", None, None),
        ("validation", None, "Audit & Sources", "audit_active_subview", "V&V Suite & Certificates"),
        ("literature", None, "Audit & Sources", "audit_active_subview", "Scientific Literature & Sources"),
    ],
)
def test_ui_deeplinks_and_contextual_sidebars(
    wb_slug: str,
    mod_slug: str | None,
    expected_wb: str,
    expected_sub_key: str | None,
    expected_sub_val: str | None,
) -> None:
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.query_params["wb"] = wb_slug
    if mod_slug:
        at.query_params["mod"] = mod_slug
    at.run()

    assert not at.exception
    assert at.session_state["active_workbench"] == expected_wb
    if expected_sub_key and expected_sub_val:
        assert at.session_state[expected_sub_key] == expected_sub_val


@pytest.mark.parametrize(("wb_slug", "mod_slug"), [("cuas", "array"), ("sim", None), ("mcdrag", None), ("3d", None), ("firing", None)])
def test_ui_expert_mode_toggle_renders_cleanly(wb_slug: str, mod_slug: str | None) -> None:
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.query_params["wb"] = wb_slug
    if mod_slug:
        at.query_params["mod"] = mod_slug
    at.run()
    assert not at.exception
    assert at.session_state["ui_expert_mode"] is False

    at.toggle(key="ui_expert_mode").set_value(True).run()
    assert not at.exception
    assert at.session_state["ui_expert_mode"] is True


def test_ui_stage1_custom_builders_and_profile_sync() -> None:
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.query_params["wb"] = "cuas"
    at.query_params["mod"] = "array"
    at.run()
    assert not at.exception
    assert "_active_cuas_profile" in at.session_state
    assert "_report_range_envelope" in at.session_state

    # Enable custom drone & custom shell + update distance to 42 m
    at.checkbox(key="cuas_cust_drone_on").set_value(True)
    at.checkbox(key="cuas_cust_shell_on").set_value(True)
    at.slider(key="cuas_dist").set_value(42.0)
    at.run()
    assert not at.exception
    assert at.session_state["_active_cuas_profile"]["target_distance_m"] == 42.0

    # Switch from Stage 1 -> Stage 2 (Closed-Loop FCS, first tool in Stage 2)
    at.radio(key="cuas_stage").set_value("2. Engagement & Lethality").run()
    assert not at.exception
    assert at.session_state["cuas_stage_tool"] == "Closed-Loop FCS & Evasive Drones"
    assert at.slider(key="cuas18_fcs_dist").value == 42.0

    # Mutate FCS distance, then test 1-click re-sync back to Stage 1's 42 m
    at.slider(key="cuas18_fcs_dist").set_value(20.0).run()
    assert not at.exception
    assert at.slider(key="cuas18_fcs_dist").value == 20.0
    at.button(key="cuas_sync_btn_fcs").click().run()
    assert not at.exception
    assert at.slider(key="cuas18_fcs_dist").value == 42.0

    # Switch to Stage 3 (SDZ, first tool in Stage 3) and test sync button cleanly
    at.radio(key="cuas_stage").set_value("3. Safety, Site & Mission").run()
    assert not at.exception
    assert at.session_state["cuas_stage_tool"] == "Elevated Fallout & Safety Zone (SDZ)"
    at.button(key="cuas_sync_btn_sdz").click().run()
    assert not at.exception
