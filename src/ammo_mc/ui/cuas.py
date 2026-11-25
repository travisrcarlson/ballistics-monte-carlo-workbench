"""Counter-UAS & Multi-Barrel Small Arms UI workbench (modular facade)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ammo_mc.cuas_defense import (
    get_cuas_payload_catalog,
    get_cuas_sensor_presets,
    get_shotgun_presets,
)
from ammo_mc.cuas_mission import validate_cuas_scenario_config
from ammo_mc.small_arms import (
    PELLET_MATERIAL_DENSITIES_G_CM3,
    get_choke_definitions,
    get_drone_aspect_presets,
    get_drone_target_specs,
)
from ammo_mc.ui.common import (
    _CUAS18_MODULES,
    _CUAS_STAGE_TOOLS,
    _CUAS_STAGES,
    _CUAS_SUITES,
    _CUAS_TOOL_TO_INTERNAL,
    _SMALL_ARMS_MODES,
    _navigate_to,
    _render_sidebar_header,
    is_expert_mode,
)
from ammo_mc.ui.cuas_pkg import (
    _render_cuas_profile_sync_bar,
    _render_effective_range_tab,
    _render_stage1_hardware_cad_tab,
    render_stage1_small_arms,
    render_stage2_engagement,
    render_stage3_tactical,
    render_stage4_adversarial,
    render_stage4_calibration,
    render_stage4_trade_studies,
)


def tab_cuas_and_small_arms() -> None:
    expert = is_expert_mode()
    st.header("C-UAS & Multi-Barrel Array Workbench")
    st.caption(
        "4-stage workflow for multi-barrel shotgun arrays, closed-loop FCS engagements, elevated fallout SDZs, "
        "turret trade studies, and field calibration."
    )

    with st.sidebar:
        _render_sidebar_header("C-UAS & Multi-Barrel", "4-stage counter-drone & shotgun array workflow")
        if st.session_state.get("cuas_stage") not in _CUAS_STAGES:
            st.session_state["cuas_stage"] = _CUAS_STAGES[0]
        active_stage = st.radio("Workflow Stage", _CUAS_STAGES, key="cuas_stage")
        stage_tools = _CUAS_STAGE_TOOLS[active_stage]
        if st.session_state.get("cuas_stage_tool") not in stage_tools:
            st.session_state["cuas_stage_tool"] = stage_tools[0]
        active_tool = st.radio("Analysis Tool", stage_tools, key="cuas_stage_tool")

        cuas_suite, mapped_target, _ = _CUAS_TOOL_TO_INTERNAL[active_tool]
        st.session_state["cuas_wb_suite"] = cuas_suite
        if cuas_suite == _CUAS_SUITES[0]:
            active_cuas_mod = mapped_target
            arm_mode = _SMALL_ARMS_MODES[4]
            st.session_state["cuas18_active_module"] = active_cuas_mod
        else:
            arm_mode = mapped_target
            active_cuas_mod = _CUAS18_MODULES[0]
            st.session_state["small_arms_mode"] = arm_mode

        st.divider()
        if cuas_suite == _CUAS_SUITES[0]:
            with st.expander("📄 Trade-Study Report Readiness", expanded=False):
                _rc_items = [
                    ("Array Summary (Stage 1)", st.session_state.get("_report_array_summary") is not None),
                    ("3D Aspect Table (Stage 1)", st.session_state.get("_report_aspect_table") is not None),
                    ("Kill Chain (Stage 3)", st.session_state.get("_report_kc_summary") is not None),
                    ("Fallout SDZ (Stage 3)", st.session_state.get("_report_sdz_summary") is not None),
                ]
                for _lbl, _ready in _rc_items:
                    st.markdown(f"{'✅' if _ready else '○'} {_lbl}")
                if active_cuas_mod != _CUAS18_MODULES[8]:
                    st.button(
                        "Open Logistics & Report Export",
                        on_click=_navigate_to,
                        args=("cuas", None, "logistics"),
                        width="stretch",
                        key="cuas18_jump_report_btn",
                    )
        elif arm_mode == "🛸 Anti-Drone Multi-Shotgun Defense Array (C-UAS)":
            with st.expander("💾 Scenario Preset Save / Load (JSON)", expanded=expert):
                uploaded_cfg = st.file_uploader(
                    "Upload Saved C-UAS Scenario Preset (.json)",
                    type=["json"],
                    key="cuas_cfg_upload",
                )
                if uploaded_cfg is not None:
                    raw_bytes = uploaded_cfg.getvalue()
                    sig = f"{uploaded_cfg.name}:{len(raw_bytes)}:{hash(raw_bytes)}"
                    if st.session_state.get("_cuas_last_uploaded_sig") != sig:
                        ok_cfg, cleaned_cfg, msg_cfg = validate_cuas_scenario_config(
                            raw_bytes.decode("utf-8", errors="replace")
                        )
                        if ok_cfg:
                            _key_map = {
                                "layout_type": "cuas_layout",
                                "n_barrels": "cuas_n_barrels",
                                "mount_radius_m": "cuas_mount_r",
                                "spacing_y_m": "cuas_sp_y",
                                "spacing_z_m": "cuas_sp_z",
                                "barrel_length_in": "cuas_b_len",
                                "choke": "cuas_choke",
                                "pointing_mode": "cuas_pointing",
                                "splay_angle_mrad": "cuas_splay",
                                "firing_mode": "cuas_fire_mode",
                                "ripple_delay_ms": "cuas_ripple_dt",
                                "ammunition_preset": "cuas_ammo",
                                "drone_model": "cuas_drone_model",
                                "aspect_mode": "cuas_aspect_mode",
                                "drone_pitch_deg": "cuas_pitch_tilt",
                                "drone_yaw_deg": "cuas_yaw_angle",
                                "drone_roll_deg": "cuas_roll_angle",
                                "target_distance_m": "cuas_dist",
                                "drone_speed_mps": "cuas_speed",
                                "drone_heading_deg": "cuas_heading",
                                "custom_coords_text": "cuas_custom_text",
                            }
                            for cfg_k, ss_k in _key_map.items():
                                if cfg_k in cleaned_cfg:
                                    val_up = cleaned_cfg[cfg_k]
                                    if cfg_k == "ammunition_preset" and val_up == "CUSTOM_SHELL":
                                        st.session_state["cuas_cust_shell_on"] = True
                                        continue
                                    if cfg_k == "drone_model" and val_up == "CUSTOM_DRONE":
                                        st.session_state["cuas_cust_drone_on"] = True
                                        continue
                                    st.session_state[ss_k] = val_up
                            if "n_barrels" in cleaned_cfg:
                                st.session_state["cuas_n_barrels_fan"] = cleaned_cfg["n_barrels"]
                            if "custom_drone_enabled" in cleaned_cfg:
                                st.session_state["cuas_cust_drone_on"] = bool(cleaned_cfg["custom_drone_enabled"])
                            if "custom_shell_enabled" in cleaned_cfg:
                                st.session_state["cuas_cust_shell_on"] = bool(cleaned_cfg["custom_shell_enabled"])
                            if isinstance(cleaned_cfg.get("custom_drone_spec"), dict):
                                cds = cleaned_cfg["custom_drone_spec"]
                                st.session_state["cuas_cd_name"] = str(cds.get("name", "Custom Threat UAS"))
                                st.session_state["cuas_cd_type"] = str(cds.get("drone_type", "quadcopter"))
                                st.session_state["cuas_cd_span"] = float(cds.get("span_m", 0.45))
                                st.session_state["cuas_cd_fl"] = float(cds.get("fuselage_l_m", 0.22))
                                st.session_state["cuas_cd_fw"] = float(cds.get("fuselage_w_m", 0.12))
                                st.session_state["cuas_cd_fh"] = float(cds.get("fuselage_h_m", 0.10))
                                st.session_state["cuas_cd_rr"] = float(cds.get("rotor_radius_m", 0.11))
                                st.session_state["cuas_cd_erot"] = float(cds.get("lethal_energy_rotor_j", 10.0))
                                st.session_state["cuas_cd_ecore"] = float(cds.get("lethal_energy_core_j", 24.0))
                                st.session_state["cuas_cd_crit"] = int(cds.get("critical_hits_for_kill", 3))
                            if isinstance(cleaned_cfg.get("custom_shell_spec"), dict):
                                css = cleaned_cfg["custom_shell_spec"]
                                st.session_state["cuas_cs_name"] = str(css.get("name", "Custom 12G C-UAS Load"))
                                if css.get("pellet_material") in PELLET_MATERIAL_DENSITIES_G_CM3:
                                    st.session_state["cuas_cs_mat"] = str(css["pellet_material"])
                                st.session_state["cuas_cs_diam"] = float(css.get("pellet_diameter_mm", 3.81))
                                st.session_state["cuas_cs_n"] = int(css.get("pellets_per_shot", 70))
                                st.session_state["cuas_cs_v0"] = float(css.get("v0_mps", 415.0))
                                st.session_state["cuas_cs_cd"] = float(css.get("effective_cd", 0.44))
                                if css.get("custom_pattern_40yd_30in_pct") is not None:
                                    st.session_state["cuas_cs_pat_on"] = True
                                    st.session_state["cuas_cs_pat40"] = round(float(css["custom_pattern_40yd_30in_pct"]) * 100.0, 1)
                            st.session_state["_cuas_last_uploaded_sig"] = sig
                            st.session_state["_cuas_upload_notice"] = msg_cfg
                            st.rerun()
                        else:
                            st.error(f"❌ Invalid scenario file: {msg_cfg}")
                if st.session_state.get("_cuas_upload_notice"):
                    st.success(f"✅ {st.session_state['_cuas_upload_notice']}")

    if cuas_suite == _CUAS_SUITES[1]:
        render_stage1_small_arms(arm_mode=arm_mode, expert=expert)
        return

    st.subheader("C-UAS Turret Engineering & Mission Effectiveness")
    if expert:
        with st.expander("Governing Control, Recoil & Ballistic Equations", expanded=False):
            st.latex(
                r"""
                \hat{\mathbf{r}}(\tau) = \hat{\mathbf{p}} + \hat{\mathbf{v}}\tau + \frac{1}{2}\hat{\mathbf{a}}\tau^2, \quad
                M\ddot{x} + C\dot{x} + Kx = \sum_{i} F_i(t), \quad
                I_{\theta}\ddot{\theta} + C_{\theta}\dot{\theta} + K_{\theta}\theta = \sum_{i} F_i(t)(z_{\text{tr}} + z_i), \quad
                m_p \frac{d\mathbf{v}}{dt} = -\frac{1}{2}\rho(z) C_D(Re,M) A_p |\mathbf{v}-\mathbf{W}|\,(\mathbf{v}-\mathbf{W}) - m_p g \hat{\mathbf{k}}
                """
            )

    sg_presets_18 = get_shotgun_presets()
    choke_defs_18 = get_choke_definitions()
    drone_specs_18 = get_drone_target_specs()
    aspect_presets_18 = get_drone_aspect_presets()
    sensor_presets_18 = get_cuas_sensor_presets()
    payload_cat_18 = get_cuas_payload_catalog()

    if active_cuas_mod in _CUAS18_MODULES[0:4]:
        render_stage2_engagement(
            active_cuas_mod=active_cuas_mod,
            expert=expert,
            sg_presets_18=sg_presets_18,
            choke_defs_18=choke_defs_18,
            drone_specs_18=drone_specs_18,
            aspect_presets_18=aspect_presets_18,
            sensor_presets_18=sensor_presets_18,
            payload_cat_18=payload_cat_18,
        )
    elif active_cuas_mod in (_CUAS18_MODULES[4:8] + [_CUAS18_MODULES[11]]):
        render_stage3_tactical(
            active_cuas_mod=active_cuas_mod,
            expert=expert,
            sg_presets_18=sg_presets_18,
            choke_defs_18=choke_defs_18,
            drone_specs_18=drone_specs_18,
            aspect_presets_18=aspect_presets_18,
            sensor_presets_18=sensor_presets_18,
        )
    elif active_cuas_mod in _CUAS18_MODULES[8:11]:
        render_stage4_trade_studies(
            active_cuas_mod=active_cuas_mod,
            expert=expert,
            sg_presets_18=sg_presets_18,
            drone_specs_18=drone_specs_18,
            aspect_presets_18=aspect_presets_18,
        )
    elif active_cuas_mod == _CUAS18_MODULES[12]:
        render_stage4_calibration(
            expert=expert,
            sg_presets_18=sg_presets_18,
            drone_specs_18=drone_specs_18,
        )
    elif active_cuas_mod == _CUAS18_MODULES[13]:
        render_stage4_adversarial(
            expert=expert,
            sg_presets_18=sg_presets_18,
            drone_specs_18=drone_specs_18,
        )


__all__ = [
    "_render_cuas_profile_sync_bar",
    "_render_effective_range_tab",
    "_render_stage1_hardware_cad_tab",
    "tab_cuas_and_small_arms",
]
