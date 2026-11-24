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
        ("cuas", "adversarial", "C-UAS & Multi-Barrel", "cuas_stage_tool", "Adversarial Red-vs-Blue ML Optimizer"),
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


def test_ui_cuas_adversarial_ml_tab_renders_and_syncs() -> None:
    """Test deep-linking into Adversarial ML Optimizer tab and 1-click sync to Stage 1 array."""
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.query_params["wb"] = "cuas"
    at.query_params["mod"] = "adversarial"
    at.run()
    assert not at.exception
    assert at.session_state["active_workbench"] == "C-UAS & Multi-Barrel"
    assert at.session_state["cuas_stage"] == "4. Compare, Optimize & Calibrate"
    assert at.session_state["cuas_stage_tool"] == "Adversarial Red-vs-Blue ML Optimizer"

    # Verify key widgets render
    assert at.slider(key="cuas_adv_ngen").value == 5
    assert at.slider(key="cuas_adv_mount_lim").value == 3.5

    # Test applying ML evolved turret to Stage 1
    at.button(key="cuas_adv_apply_to_s1_btn").click().run()
    assert not at.exception
    assert "_active_cuas_profile" in at.session_state
    prof = at.session_state["_active_cuas_profile"]
    assert "n_barrels" in prof
    assert "ammunition_preset" in prof

    # Verify new Hybrid, Robust, and Duel controls render & toggle cleanly
    assert at.checkbox(key="cuas_adv_hybrid") is not None
    assert at.checkbox(key="cuas_adv_robust") is not None
    at.checkbox(key="cuas_adv_hybrid").set_value(True).run()
    assert not at.exception
    assert at.radio(key="cuas_duel_mode") is not None


def test_ui_home_wizard_and_calibration_track_replay() -> None:
    """Verify Home 3-Click Turret Sizing Wizard & Tab 18.13 Track Replay / Twin."""
    at_home = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_home.session_state["authenticated"] = True
    at_home.query_params["wb"] = "home"
    at_home.run()
    assert not at_home.exception
    assert at_home.slider(key="home_wiz_rng").value == 45.0
    at_home.button(key="home_wiz_load_btn").click().run()
    assert not at_home.exception
    assert at_home.session_state["active_workbench"] == "C-UAS & Multi-Barrel"
    assert at_home.session_state["cuas_stage_tool"] == "Anti-Drone Multi-Shotgun Array"

    at_cal = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_cal.session_state["authenticated"] = True
    at_cal.query_params["wb"] = "cuas"
    at_cal.query_params["mod"] = "calibration"
    at_cal.run()
    assert not at_cal.exception
    at_cal.button(key="cal_pat_apply_btn").click().run()
    assert not at_cal.exception
    assert "_calibrated_digital_twin" in at_cal.session_state


def test_ui_digital_twin_sidebar_fea_sdz_and_calibration_v16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify v1.6 Digital Twin sidebar switch, Stage 1 FEA, Tab 18.5 Collateral Safety, & Tab 18.13 modes."""
    monkeypatch.setenv("AMMO_MC_SCENARIO_DB", str(tmp_path / "ui_v16.db"))

    # 1. Stage 1 Array: FEA sliders/selectors + global Digital Twin toggle
    at_arr = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_arr.session_state["authenticated"] = True
    at_arr.query_params["wb"] = "cuas"
    at_arr.query_params["mod"] = "array"
    at_arr.run()
    assert not at_arr.exception
    assert at_arr.toggle(key="ui_digital_twin_active") is not None
    assert at_arr.slider(key="cuas_fea_thk").value == 22.0
    assert at_arr.selectbox(key="cuas_fea_mat").value == "7075-T6"

    # Toggle global Digital Twin active
    at_arr.toggle(key="ui_digital_twin_active").set_value(True).run()
    assert not at_arr.exception

    # 2. Tab 18.5 SDZ Collateral Safety Envelope controls render cleanly
    at_sdz = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_sdz.session_state["authenticated"] = True
    at_sdz.query_params["wb"] = "cuas"
    at_sdz.query_params["mod"] = "sdz"
    at_sdz.run()
    assert not at_sdz.exception
    assert at_sdz.selectbox(key="cuas18_collat_env").value == "urban_permissive"

    # 3. Tab 18.13 Multi-Range, Photo Digitizer, Bayesian Lethality, Hardware Bench & Twin Save
    at_cal = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_cal.session_state["authenticated"] = True
    at_cal.query_params["wb"] = "cuas"
    at_cal.query_params["mod"] = "calibration"
    at_cal.run()
    assert not at_cal.exception

    at_cal.radio(key="cal_pat_mode").set_value("Multi-Range Spread Exponent & Pellet Drag").run()
    assert not at_cal.exception
    at_cal.button(key="cal_mr_apply_btn").click().run()
    assert not at_cal.exception

    at_cal.radio(key="cal_pat_mode").set_value("Photo-to-Pattern Hole Digitizer (PNG / JPG)").run()
    assert not at_cal.exception
    at_cal.button(key="cal_ph_apply_btn").click().run()
    assert not at_cal.exception

    at_cal.button(key="cal_lf_apply_btn").click().run()
    assert not at_cal.exception
    at_cal.button(key="cal_hb_apply_btn").click().run()
    assert not at_cal.exception
    at_cal.button(key="cal_mgr_save_btn").click().run()
    assert not at_cal.exception
    assert at_cal.session_state["ui_digital_twin_active"] is True
    at_cal.text_input(key="cal_mgr_name").set_value("Proto-V3_Diff_Check")
    at_cal.button(key="cal_mgr_save_btn").click().run()
    assert not at_cal.exception
    assert at_cal.selectbox(key="cal_diff_a") is not None
    assert at_cal.selectbox(key="cal_diff_b") is not None

    # 4. Tab 18.14 HIL telemetry widgets render cleanly
    at_adv = AppTest.from_file(ROOT_APP, default_timeout=60)
    at_adv.session_state["authenticated"] = True
    at_adv.query_params["wb"] = "cuas"
    at_adv.query_params["mod"] = "adversarial"
    at_adv.run()
    assert not at_adv.exception
    assert at_adv.selectbox(key="cuas_hil_scn").value == "evasive_fpv_dive"


