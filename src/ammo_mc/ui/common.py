"""Shared constants, cached computation wrappers, auth gate, styles & URL deep-linking."""

from __future__ import annotations

import json
import math
from typing import Any

import streamlit as st

from ammo_mc.auth import LoginRateLimiter, verify_app_password
from ammo_mc.cuas_analytics import (
    analyze_artillery_sensitivity_tornado,
    analyze_cuas_sensitivity_tornado,
    compare_cuas_architectures_ab,
    simulate_layered_cuas_defense_mc,
)
from ammo_mc.cuas_defense import (
    compare_cuas_fallout_across_loads,
    compare_cuas_payload_families,
    optimize_antidrone_array_configuration,
)
from ammo_mc.cuas_mission import (
    generate_3d_intercept_replay_data,
    simulate_cuas_kill_chain_mc,
    simulate_cuas_sustained_logistics_mc,
    simulate_multiturret_site_defense_mc,
)
from ammo_mc.formulas_registry import USE_LIMITATION_NOTICE
from ammo_mc.presets import get_preset
from ammo_mc.small_arms import (
    benchmark_drone_aspect_angles_mc,
    benchmark_multibarrel_aspect_angles_mc,
    compare_anti_drone_layouts,
)

# -------------------------------------------------------------------------
# WORKBENCH & MODULE NAVIGATION CONSTANTS + URL SLUGS
# -------------------------------------------------------------------------

_WORKBENCH_OPTIONS = [
    "Home",
    "C-UAS & Multi-Barrel",
    "Ballistics & 3D Sim",
    "Physics & Aero Labs",
    "Safety & IATG",
    "Audit & Sources",
]

_SIM_SUBVIEWS = [
    "Monte Carlo Pipeline",
    "3D Trajectory Lab",
    "Firing Tables & Met",
]

_PHYSICS_SUBVIEWS = [
    "Physics & Ballistics Labs",
    "MCDRAG & 3D CAD",
]

_AUDIT_SUBVIEWS = [
    "V&V Suite & Certificates",
    "Scientific Literature & Sources",
]

_WORKBENCH_SLUGS: dict[str, str] = {
    "Home": "home",
    "C-UAS & Multi-Barrel": "cuas",
    "Ballistics & 3D Sim": "sim",
    "Physics & Aero Labs": "physics",
    "Safety & IATG": "iatg",
    "Audit & Sources": "validation",
}

_CUAS_SUITES = [
    "C-UAS Turret Suite (13 Modules)",
    "Small Arms & Multi-Barrel Array (5 Modes)",
]

_CUAS18_MODULES = [
    "📡 1. Closed-Loop FCS & Evasive Drones",
    "🐝 2. Multi-Drone Swarm Saturation",
    "💥 3. Advanced Payloads & Airburst",
    "🔩 4. Recoil, Gimbal & Pareto Optimizer",
    "⚠️ 5. Elevated Fallout & Safety Zone (SDZ)",
    "🛰️ 6. Multi-Sensor Kill Chain & Timeline",
    "🏰 7. Multi-Turret Site Defense, Custom Builder & GIS Map",
    "🎬 8. 3D Animated Intercept Replay",
    "🔥 9. Logistics, Barrel Heat, $/Kill & Report",
    "⚔️ 10. A/B Turret Comparator & Layered Defense",
    "📊 11. Sensitivity & Bottleneck Tornado",
    "⚔️ 12. Unified 3D Tactical Raid Simulator",
    "🧪 13. Field-Test Calibration & Residuals",
]

_CUAS_MOD_SLUGS: dict[str, str] = {
    _CUAS18_MODULES[0]: "fcs",
    _CUAS18_MODULES[1]: "swarm",
    _CUAS18_MODULES[2]: "payloads",
    _CUAS18_MODULES[3]: "recoil",
    _CUAS18_MODULES[4]: "sdz",
    _CUAS18_MODULES[5]: "kill_chain",
    _CUAS18_MODULES[6]: "site_defense",
    _CUAS18_MODULES[7]: "replay",
    _CUAS18_MODULES[8]: "logistics",
    _CUAS18_MODULES[9]: "comparator",
    _CUAS18_MODULES[10]: "sensitivity",
    _CUAS18_MODULES[11]: "raid",
    _CUAS18_MODULES[12]: "calibration",
}
_SLUG_TO_CUAS_MOD: dict[str, str] = {v: k for k, v in _CUAS_MOD_SLUGS.items()}

_SMALL_ARMS_MODES = [
    "🎯 Single-Range Pattern Spread & Shot Stringing",
    "📏 Multi-Distance Range Sweep & 3D Plume Slices",
    "📐 Multiple Barrel Lengths Comparison",
    "🔄 Multi-Barrel Systems, Rotary Minigun & Volley Dispersion",
    "🛸 Anti-Drone Multi-Shotgun Defense Array (C-UAS)",
]

_CUAS_STAGES = [
    "1. Array & Weapon Setup",
    "2. Engagement & Lethality",
    "3. Safety, Site & Mission",
    "4. Compare, Optimize & Calibrate",
]

_CUAS_STAGE_TOOLS: dict[str, list[str]] = {
    _CUAS_STAGES[0]: [
        "Anti-Drone Multi-Shotgun Array",
        "Single-Shotgun Pattern & Shot Stringing",
        "Range Sweep & 3D Plume Slices",
        "Barrel Length Comparison",
        "Multi-Barrel Rotary & Volley Systems",
    ],
    _CUAS_STAGES[1]: [
        "Closed-Loop FCS & Evasive Drones",
        "Advanced Payloads & Airburst",
        "Multi-Drone Swarm Saturation",
        "3D Animated Intercept Replay",
    ],
    _CUAS_STAGES[2]: [
        "Elevated Fallout & Safety Zone (SDZ)",
        "Multi-Sensor Kill Chain & Timeline",
        "Multi-Turret Site Defense & GIS Map",
        "Unified 3D Tactical Raid Simulator",
    ],
    _CUAS_STAGES[3]: [
        "A/B Turret Comparator & Layered Defense",
        "Recoil, Gimbal & Pareto Optimizer",
        "Sensitivity & Bottleneck Tornado",
        "Logistics, Heat, $/Kill & Report",
        "Field-Test Calibration & Residuals",
    ],
}

# Maps friendly UI tool label -> (suite_name, internal_mode_or_module, url_mod_slug)
_CUAS_TOOL_TO_INTERNAL: dict[str, tuple[str, str, str]] = {
    "Anti-Drone Multi-Shotgun Array": (_CUAS_SUITES[1], _SMALL_ARMS_MODES[4], "array"),
    "Single-Shotgun Pattern & Shot Stringing": (_CUAS_SUITES[1], _SMALL_ARMS_MODES[0], "small_arms"),
    "Range Sweep & 3D Plume Slices": (_CUAS_SUITES[1], _SMALL_ARMS_MODES[1], "range_sweep"),
    "Barrel Length Comparison": (_CUAS_SUITES[1], _SMALL_ARMS_MODES[2], "barrel_lengths"),
    "Multi-Barrel Rotary & Volley Systems": (_CUAS_SUITES[1], _SMALL_ARMS_MODES[3], "multibarrel"),
    "Closed-Loop FCS & Evasive Drones": (_CUAS_SUITES[0], _CUAS18_MODULES[0], "fcs"),
    "Multi-Drone Swarm Saturation": (_CUAS_SUITES[0], _CUAS18_MODULES[1], "swarm"),
    "Advanced Payloads & Airburst": (_CUAS_SUITES[0], _CUAS18_MODULES[2], "payloads"),
    "Recoil, Gimbal & Pareto Optimizer": (_CUAS_SUITES[0], _CUAS18_MODULES[3], "recoil"),
    "Elevated Fallout & Safety Zone (SDZ)": (_CUAS_SUITES[0], _CUAS18_MODULES[4], "sdz"),
    "Multi-Sensor Kill Chain & Timeline": (_CUAS_SUITES[0], _CUAS18_MODULES[5], "kill_chain"),
    "Multi-Turret Site Defense & GIS Map": (_CUAS_SUITES[0], _CUAS18_MODULES[6], "site_defense"),
    "3D Animated Intercept Replay": (_CUAS_SUITES[0], _CUAS18_MODULES[7], "replay"),
    "Logistics, Heat, $/Kill & Report": (_CUAS_SUITES[0], _CUAS18_MODULES[8], "logistics"),
    "A/B Turret Comparator & Layered Defense": (_CUAS_SUITES[0], _CUAS18_MODULES[9], "comparator"),
    "Sensitivity & Bottleneck Tornado": (_CUAS_SUITES[0], _CUAS18_MODULES[10], "sensitivity"),
    "Unified 3D Tactical Raid Simulator": (_CUAS_SUITES[0], _CUAS18_MODULES[11], "raid"),
    "Field-Test Calibration & Residuals": (_CUAS_SUITES[0], _CUAS18_MODULES[12], "calibration"),
}

_CUAS_SLUG_TO_STAGE_TOOL: dict[str, tuple[str, str]] = {}
for _st_name, _tlist in _CUAS_STAGE_TOOLS.items():
    for _tname in _tlist:
        _slug_val = _CUAS_TOOL_TO_INTERNAL[_tname][2]
        _CUAS_SLUG_TO_STAGE_TOOL[_slug_val] = (_st_name, _tname)

_CUAS_INTERNAL_TO_STAGE_TOOL: dict[str, tuple[str, str]] = {
    _info[1]: (_st_name, _tname)
    for _st_name, _tlist in _CUAS_STAGE_TOOLS.items()
    for _tname in _tlist
    for _info in (_CUAS_TOOL_TO_INTERNAL[_tname],)
}

_SLUG_TO_ROUTE: dict[str, tuple[str, str | None]] = {
    "home": ("Home", None),
    "cuas": ("C-UAS & Multi-Barrel", None),
    "sim": ("Ballistics & 3D Sim", _SIM_SUBVIEWS[0]),
    "3d": ("Ballistics & 3D Sim", _SIM_SUBVIEWS[1]),
    "firing": ("Ballistics & 3D Sim", _SIM_SUBVIEWS[2]),
    "physics": ("Physics & Aero Labs", _PHYSICS_SUBVIEWS[0]),
    "mcdrag": ("Physics & Aero Labs", _PHYSICS_SUBVIEWS[1]),
    "iatg": ("Safety & IATG", None),
    "validation": ("Audit & Sources", _AUDIT_SUBVIEWS[0]),
    "literature": ("Audit & Sources", _AUDIT_SUBVIEWS[1]),
}

_SIM_SUBVIEW_SLUGS = {
    _SIM_SUBVIEWS[0]: "sim",
    _SIM_SUBVIEWS[1]: "3d",
    _SIM_SUBVIEWS[2]: "firing",
}
_PHYSICS_SUBVIEW_SLUGS = {
    _PHYSICS_SUBVIEWS[0]: "physics",
    _PHYSICS_SUBVIEWS[1]: "mcdrag",
}
_AUDIT_SUBVIEW_SLUGS = {
    _AUDIT_SUBVIEWS[0]: "validation",
    _AUDIT_SUBVIEWS[1]: "literature",
}

_LEGACY_WORKBENCH_ALIASES: dict[str, tuple[str, str | None]] = {
    "🎯 Monte Carlo Simulator": ("sim", None),
    "🛸 C-UAS & Small Arms": ("cuas", "array"),
    "🌐 3D Simulations": ("3d", None),
    "🚀 MCDRAG & 3D CAD": ("mcdrag", None),
    "🔬 Physics & Ballistics Labs": ("physics", None),
    "🏛️ UNODA IATG Safety": ("iatg", None),
    "📋 Firing Tables & Met": ("firing", None),
    "✅ Validation Suite": ("validation", None),
    "📚 Literature & Sources": ("literature", None),
}

_PHYSICS_LABS = [
    "🔥 1. Interior Ballistics & P-V Diagram",
    "🌍 2. ISA Atmosphere, Stability & Coriolis",
    "🌊 3. Shock Physics & Rankine-Hugoniot",
    "💥 4. Gurney Energy, Taylor & Mott",
    "🛡️ 5. Continuum Penetration & Project THOR",
    "📐 6. Quasi-MC Discrepancy & KB Registry",
    "🌀 7. STANAG 4355 MPM Yaw & Spin Drift",
    "⚙️ 8. Bayesian Fuze Reliability & Graze Surface",
    "🚀 9. Precision Guidance & Proportional Nav (PN/APN)",
    "📦 10. Cluster Dispenser & UXO Hazard (CCM/IATG)",
    "🌀 11. Full 6-DoF Rigid-Body & Epicyclic Yaw",
    "🎯 12. Battery Sheafs & MRSI/TOT Salvo",
    "🛡️ 13. Multi-Layer Armor & Tandem ERA",
    "🛰️ 14. Counter-Battery Radar & POO/POI",
    "🎯 15. Direct-Fire Tank Gunnery & APFSDS Sabot",
    "📡 16. Electronic Warfare & Laser Homing",
]

_IATG_TOOLS = [
    "🏗️ 1. Quantity-Distance (QD), Open-Detonation (TB-15) & 3D PES Map",
    "🧱 2. Structural Damage (TB-10 Jarrett/GLS) & Swiss Glazing (TB-11)",
    "📦 3. NATO AASTP-1 Stack Fragmentation (TF-10) & RC Debris (TF-14)",
    "🫁 4. Axelsson Thoracic Biomechanics ODE (TB-13) & Thermal Dose (TB-16)",
    "🗺️ 5. Multi-Facility 3D Site Planner & Sympathetic Matrix",
    "🧪 6. Propellant Aging & In-Service Surveillance (IATG 07.10 / AOP-48)",
    "🗺️ 7. Terrain Crest Clearance & KML/CZML Export",
]


# -------------------------------------------------------------------------
# CACHED COMPUTATION WRAPPERS (HASHABLE ARGS FOR STREAMLIT CLOUD)
# -------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _cached_benchmark_multibarrel_aspect(
    system_type: str,
    n_barrels: int,
    bore_spacing_y_mm: float,
    bore_spacing_z_mm: float,
    rotor_radius_mm: float,
    crossover_range_m: float,
    burst_rounds: int,
    target_range_m: float,
    cyclic_rate_rpm: float,
    inherent_dispersion_mrad: float,
    barrel_pointing_bias_mrad: float,
    thermal_drift_mrad_per_100rd: float,
    drone_model: str,
    bullet_energy_j: float,
    seed: int = 42,
) -> dict[str, Any]:
    return benchmark_multibarrel_aspect_angles_mc(
        system_type=system_type,
        n_barrels=n_barrels,
        bore_spacing_y_mm=bore_spacing_y_mm,
        bore_spacing_z_mm=bore_spacing_z_mm,
        rotor_radius_mm=rotor_radius_mm,
        crossover_range_m=crossover_range_m,
        burst_rounds=burst_rounds,
        target_range_m=target_range_m,
        cyclic_rate_rpm=cyclic_rate_rpm,
        inherent_dispersion_mrad=inherent_dispersion_mrad,
        barrel_pointing_bias_mrad=barrel_pointing_bias_mrad,
        thermal_drift_mrad_per_100rd=thermal_drift_mrad_per_100rd,
        drone_model=drone_model,
        bullet_energy_j=bullet_energy_j,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_compare_anti_drone_layouts(
    distance_stations_m: tuple[float, ...],
    ammunition_preset: str,
    drone_model: str,
    drone_velocity_mps: float,
    pointing_mode: str,
    firing_mode: str,
    n_volleys: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    return compare_anti_drone_layouts(
        distance_stations_m=list(distance_stations_m),
        ammunition_preset=ammunition_preset,
        drone_model=drone_model,
        drone_velocity_mps=drone_velocity_mps,
        pointing_mode=pointing_mode,
        firing_mode=firing_mode,
        n_volleys=n_volleys,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_benchmark_drone_aspect(
    layout_type: str,
    n_barrels: int,
    mount_radius_m: float,
    spacing_y_m: float,
    spacing_z_m: float,
    n_cols: int,
    n_rows: int,
    pointing_mode: str,
    splay_angle_mrad: float,
    firing_mode: str,
    ripple_delay_ms: float,
    custom_barrel_specs_json: str,
    ammunition_preset: str,
    default_choke: str,
    default_barrel_length_in: float,
    target_distance_m: float,
    drone_model: str,
    drone_velocity_mps: float,
    aim_error_mrad: float,
    crosswind_mps: float,
    n_volleys: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    custom_specs = json.loads(custom_barrel_specs_json) if custom_barrel_specs_json else None
    return benchmark_drone_aspect_angles_mc(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        spacing_y_m=spacing_y_m,
        spacing_z_m=spacing_z_m,
        n_cols=n_cols,
        n_rows=n_rows,
        pointing_mode=pointing_mode,
        splay_angle_mrad=splay_angle_mrad,
        firing_mode=firing_mode,
        ripple_delay_ms=ripple_delay_ms,
        custom_barrel_specs=custom_specs,
        ammunition_preset=ammunition_preset,
        default_choke=default_choke,
        default_barrel_length_in=default_barrel_length_in,
        target_distance_m=target_distance_m,
        drone_model=drone_model,
        drone_velocity_mps=drone_velocity_mps,
        aim_error_mrad=aim_error_mrad,
        crosswind_mps=crosswind_mps,
        n_volleys=n_volleys,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_compare_cuas_payloads(
    distances_m: tuple[float, ...],
    n_barrels: int,
    drone_model: str,
    aspect_mode: str,
    fuze_time_jitter_ms: float,
    n_volleys: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    return compare_cuas_payload_families(
        distances_m=list(distances_m),
        n_barrels=n_barrels,
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        fuze_time_jitter_ms=fuze_time_jitter_ms,
        n_volleys=n_volleys,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_optimize_antidrone_array(
    n_barrels: int,
    mount_radius_m: float,
    target_distance_m: float,
    drone_model: str,
    drone_velocity_mps: float,
    ammunition_preset: str,
    inner_ammunition_preset: str | None,
    mount_mass_kg: float,
    mount_stiffness_n_per_m: float,
    mount_load_limit_kn: float,
    gimbal_trunnion_offset_z_m: float,
    n_volleys: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    return optimize_antidrone_array_configuration(
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        target_distance_m=target_distance_m,
        drone_model=drone_model,
        drone_velocity_mps=drone_velocity_mps,
        ammunition_preset=ammunition_preset,
        inner_ammunition_preset=inner_ammunition_preset,
        mount_mass_kg=mount_mass_kg,
        mount_stiffness_n_per_m=mount_stiffness_n_per_m,
        mount_load_limit_kn=mount_load_limit_kn,
        gimbal_trunnion_offset_z_m=gimbal_trunnion_offset_z_m,
        n_volleys=n_volleys,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_compare_cuas_fallout(
    firing_elevation_deg: float,
    wind_downrange_mps: float,
    wind_crossrange_mps: float,
    default_choke: str,
    n_sample_pellets: int = 140,
    seed: int = 42,
) -> dict[str, Any]:
    return compare_cuas_fallout_across_loads(
        firing_elevation_deg=firing_elevation_deg,
        wind_downrange_mps=wind_downrange_mps,
        wind_crossrange_mps=wind_crossrange_mps,
        default_choke=default_choke,
        n_sample_pellets=n_sample_pellets,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_kill_chain(
    drone_model: str,
    aspect_mode: str,
    drone_speed_mps: float,
    approach_altitude_m: float,
    weather: str,
    ambient_wind_mps: float,
    fiber_optic_autonomous: bool,
    enabled_sensors: tuple[str, ...],
    ai_classify_s: float,
    human_in_the_loop: bool,
    human_auth_s: float,
    slew_angle_deg: float,
    turret_slew_rate_deg_s: float,
    max_weapon_range_m: float,
    keepout_breach_radius_m: float,
    n_barrels: int,
    ammunition_preset: str,
    choke: str,
    salvo_cycle_time_s: float,
    n_trials: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    return simulate_cuas_kill_chain_mc(
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        drone_speed_mps=drone_speed_mps,
        approach_altitude_m=approach_altitude_m,
        weather=weather,
        ambient_wind_mps=ambient_wind_mps,
        fiber_optic_autonomous=fiber_optic_autonomous,
        enabled_sensors=enabled_sensors,
        ai_classify_s=ai_classify_s,
        human_in_the_loop=human_in_the_loop,
        human_auth_s=human_auth_s,
        slew_angle_deg=slew_angle_deg,
        turret_slew_rate_deg_s=turret_slew_rate_deg_s,
        max_weapon_range_m=max_weapon_range_m,
        keepout_breach_radius_m=keepout_breach_radius_m,
        n_barrels=n_barrels,
        ammunition_preset=ammunition_preset,
        choke=choke,
        salvo_cycle_time_s=salvo_cycle_time_s,
        n_trials=n_trials,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_site_defense(
    site_preset: str,
    threat_altitude_m: float,
    threat_speed_mps: float,
    drone_model: str,
    weapon_max_range_m: float,
    elevation_sdz_range_m: float,
    custom_turrets_json: str = "",
    custom_obstacles_json: str = "",
    custom_nfz_json: str = "",
    terrain_dem_json: str = "",
    n_raid_trials: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    from ammo_mc.terrain_dem import TerrainDEMGrid

    c_tur = json.loads(custom_turrets_json) if custom_turrets_json else None
    c_obs = json.loads(custom_obstacles_json) if custom_obstacles_json else None
    c_nfz = json.loads(custom_nfz_json) if custom_nfz_json else None
    dem_obj = TerrainDEMGrid.from_dict(json.loads(terrain_dem_json)) if terrain_dem_json else None
    return simulate_multiturret_site_defense_mc(
        site_preset=site_preset,
        custom_turrets=c_tur,
        custom_obstacles=c_obs,
        custom_nfz=c_nfz,
        terrain_dem=dem_obj,
        threat_altitude_m=threat_altitude_m,
        threat_speed_mps=threat_speed_mps,
        drone_model=drone_model,
        weapon_max_range_m=weapon_max_range_m,
        elevation_sdz_range_m=elevation_sdz_range_m,
        n_raid_trials=n_raid_trials,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_intercept_replay(
    mode: str,
    layout_type: str,
    n_barrels: int,
    target_distance_m: float,
    drone_model: str,
    aspect_mode: str,
    drone_speed_mps: float,
    maneuver_profile: str,
    maneuver_g: float,
    ammunition_preset: str,
    choke: str,
    firing_mode: str,
    ripple_delay_ms: float,
    n_frames: int = 30,
    max_visual_pellets: int = 84,
    seed: int = 42,
) -> dict[str, Any]:
    return generate_3d_intercept_replay_data(
        mode=mode,
        layout_type=layout_type,
        n_barrels=n_barrels,
        target_distance_m=target_distance_m,
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        drone_speed_mps=drone_speed_mps,
        maneuver_profile=maneuver_profile,
        maneuver_g=maneuver_g,
        ammunition_preset=ammunition_preset,
        choke=choke,
        firing_mode=firing_mode,
        ripple_delay_ms=ripple_delay_ms,
        n_frames=n_frames,
        max_visual_pellets=max_visual_pellets,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_sustained_logistics(
    n_waves: int,
    drones_per_wave: int,
    wave_interval_s: float,
    n_barrels: int,
    ammo_family: str,
    magazine_type: str,
    barrel_mass_kg: float,
    ambient_temp_c: float,
    target_range_m: float,
    drone_model: str,
    aspect_mode: str,
    n_trials: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    return simulate_cuas_sustained_logistics_mc(
        n_waves=n_waves,
        drones_per_wave=drones_per_wave,
        wave_interval_s=wave_interval_s,
        n_barrels=n_barrels,
        ammo_family=ammo_family,
        magazine_type=magazine_type,
        barrel_mass_kg=barrel_mass_kg,
        ambient_temp_c=ambient_temp_c,
        target_range_m=target_range_m,
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        n_trials=n_trials,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_compare_ab(
    configs_json: str,
    drone_model: str,
    aspect_mode: str,
    target_distance_m: float,
    drone_speed_mps: float,
    weather: str,
    fiber_optic_autonomous: bool,
    n_swarm_drones: int,
    n_waves: int,
    drones_per_wave: int,
    seed: int = 42,
) -> dict[str, Any]:
    return compare_cuas_architectures_ab(
        configs=json.loads(configs_json) if configs_json else None,
        drone_model=drone_model,
        aspect_mode=aspect_mode,
        target_distance_m=target_distance_m,
        drone_speed_mps=drone_speed_mps,
        weather=weather,
        fiber_optic_autonomous=fiber_optic_autonomous,
        n_swarm_drones=n_swarm_drones,
        n_waves=n_waves,
        drones_per_wave=drones_per_wave,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_layered_defense(
    outer_layer_id: str,
    outer_magazine: int,
    n_waves: int,
    drones_per_wave: int,
    wave_interval_s: float,
    fiber_optic_fraction: float,
    weather: str,
    inner_n_barrels: int,
    inner_ammo_family: str,
    inner_magazine_type: str,
    target_range_m: float,
    drone_model: str,
    n_trials: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    return simulate_layered_cuas_defense_mc(
        outer_layer_id=outer_layer_id,
        outer_magazine=outer_magazine,
        n_waves=n_waves,
        drones_per_wave=drones_per_wave,
        wave_interval_s=wave_interval_s,
        fiber_optic_fraction=fiber_optic_fraction,
        weather=weather,
        inner_n_barrels=inner_n_barrels,
        inner_ammo_family=inner_ammo_family,
        inner_magazine_type=inner_magazine_type,
        target_range_m=target_range_m,
        drone_model=drone_model,
        n_trials=n_trials,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_cuas_tornado(
    baseline_cfg_json: str,
    perturbation_pct: float,
    n_trials: int = 75,
    seed: int = 42,
) -> dict[str, Any]:
    return analyze_cuas_sensitivity_tornado(
        baseline_cfg=json.loads(baseline_cfg_json) if baseline_cfg_json else None,
        perturbation_pct=perturbation_pct,
        n_trials=n_trials,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_artillery_tornado(
    ammo_json: str,
    scn_json: str,
    perturbation_pct: float,
    solved_qe_rad: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    from ammo_mc.schemas import AmmoDefinition, Scenario

    ammo_obj = AmmoDefinition.model_validate_json(ammo_json)
    scn_obj = Scenario.model_validate_json(scn_json)
    return analyze_artillery_sensitivity_tornado(
        ammo=ammo_obj,
        scenario=scn_obj,
        perturbation_pct=perturbation_pct,
        solved_qe_rad=solved_qe_rad,
        seed=seed,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_unified_raid(
    site_preset: str,
    custom_turrets_json: str,
    custom_obstacles_json: str,
    custom_nfz_json: str,
    terrain_dem_json: str,
    sensors_tuple: tuple[str, ...],
    weather_condition: str,
    n_drones: int,
    fiber_optic_fraction: float,
    drone_model: str,
    approach_pattern: str,
    evasion_profile: str,
    maneuver_g: float,
    threat_speed_mps: float,
    threat_altitude_m: float,
    stagger_window_s: float,
    ammunition_preset: str,
    choke: str,
    magazine_type: str,
    weapon_max_range_m: float,
    n_mc_trials: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    from ammo_mc.cuas_raid import simulate_unified_tactical_raid_mc
    from ammo_mc.terrain_dem import TerrainDEMGrid

    return simulate_unified_tactical_raid_mc(
        site_preset=site_preset,
        custom_turrets=json.loads(custom_turrets_json) if custom_turrets_json else None,
        custom_obstacles=json.loads(custom_obstacles_json) if custom_obstacles_json else None,
        custom_nfz=json.loads(custom_nfz_json) if custom_nfz_json else None,
        terrain_dem=TerrainDEMGrid.from_dict(json.loads(terrain_dem_json)) if terrain_dem_json else None,
        sensors=list(sensors_tuple),
        weather_condition=weather_condition,
        n_drones=n_drones,
        fiber_optic_fraction=fiber_optic_fraction,
        drone_model=drone_model,
        approach_pattern=approach_pattern,
        evasion_profile=evasion_profile,
        maneuver_g=maneuver_g,
        threat_speed_mps=threat_speed_mps,
        threat_altitude_m=threat_altitude_m,
        stagger_window_s=stagger_window_s,
        ammunition_preset=ammunition_preset,
        choke=choke,
        magazine_type=magazine_type,
        weapon_max_range_m=weapon_max_range_m,
        n_mc_trials=n_mc_trials,
        seed=seed,
    )


# -------------------------------------------------------------------------
# FOCUS MODE & NAVIGATION CALLBACKS + URL DEEP-LINKING
# -------------------------------------------------------------------------

def is_expert_mode() -> bool:
    """Return True when the user has unlocked Expert Mode in the sidebar."""
    return bool(st.session_state.get("ui_expert_mode", False))


def render_focus_mode_toggle() -> bool:
    """Render the global Standard / Expert Mode switch in the sidebar."""
    return bool(
        st.toggle(
            "Expert Mode",
            value=bool(st.session_state.get("ui_expert_mode", False)),
            key="ui_expert_mode",
            help=(
                "Standard mode surfaces the primary inputs and results. "
                "Enable Expert Mode for FMEA, aero overrides, full telemetry, and advanced labs."
            ),
        )
    )


def _apply_cuas_slug(mod_slug: str) -> None:
    """Synchronize C-UAS stage, tool label, suite, and internal mode from a URL/tool slug."""
    m_low = str(mod_slug).strip().lower()
    stage, tool = _CUAS_SLUG_TO_STAGE_TOOL.get(m_low, _CUAS_SLUG_TO_STAGE_TOOL["array"])
    st.session_state["cuas_stage"] = stage
    st.session_state["cuas_stage_tool"] = tool
    suite_name, target_mode, _ = _CUAS_TOOL_TO_INTERNAL[tool]
    st.session_state["cuas_wb_suite"] = suite_name
    if suite_name == _CUAS_SUITES[0]:
        st.session_state["cuas18_active_module"] = target_mode
    else:
        st.session_state["small_arms_mode"] = target_mode


def _navigate_to(
    wb_slug: str,
    sub: str | None = None,
    cuas_mod_slug: str | None = None,
) -> None:
    """Navigate cleanly across consolidated workbenches, sub-views, and C-UAS tools."""
    slug = str(wb_slug).strip().lower()
    if slug not in _SLUG_TO_ROUTE:
        slug = "home"
    wb_name, default_sub = _SLUG_TO_ROUTE[slug]
    st.session_state["active_workbench"] = wb_name
    sub_val = sub or default_sub
    if wb_name == "Ballistics & 3D Sim" and sub_val in _SIM_SUBVIEWS:
        st.session_state["sim_active_subview"] = sub_val
    elif wb_name == "Physics & Aero Labs" and sub_val in _PHYSICS_SUBVIEWS:
        st.session_state["physics_active_subview"] = sub_val
    elif wb_name == "Audit & Sources" and sub_val in _AUDIT_SUBVIEWS:
        st.session_state["audit_active_subview"] = sub_val
    elif wb_name == "C-UAS & Multi-Barrel":
        _apply_cuas_slug(cuas_mod_slug or "array")


def _set_active_workbench(wb_name: str) -> None:
    """Switch active workbench cleanly, accepting both consolidated and legacy names."""
    if wb_name in _LEGACY_WORKBENCH_ALIASES:
        target_slug, mod_slug = _LEGACY_WORKBENCH_ALIASES[wb_name]
        _navigate_to(target_slug, cuas_mod_slug=mod_slug)
        return
    if wb_name in _WORKBENCH_OPTIONS:
        st.session_state["active_workbench"] = wb_name
        if wb_name == "C-UAS & Multi-Barrel" and "cuas_stage_tool" not in st.session_state:
            _apply_cuas_slug("array")


def _set_cuas_module(mod_name: str) -> None:
    """Switch C-UAS module by internal name and synchronize stage/tool selectors."""
    st.session_state["cuas_wb_suite"] = _CUAS_SUITES[0]
    st.session_state["cuas18_active_module"] = mod_name
    if mod_name in _CUAS_INTERNAL_TO_STAGE_TOOL:
        st_name, tool_name = _CUAS_INTERNAL_TO_STAGE_TOOL[mod_name]
        st.session_state["cuas_stage"] = st_name
        st.session_state["cuas_stage_tool"] = tool_name


def _active_route_slug() -> str:
    """Return the canonical URL `wb` slug corresponding to the current view state."""
    active_wb = st.session_state.get("active_workbench", _WORKBENCH_OPTIONS[0])
    if active_wb == "Ballistics & 3D Sim":
        sub = st.session_state.get("sim_active_subview", _SIM_SUBVIEWS[0])
        return _SIM_SUBVIEW_SLUGS.get(sub, "sim")
    if active_wb == "Physics & Aero Labs":
        sub = st.session_state.get("physics_active_subview", _PHYSICS_SUBVIEWS[0])
        return _PHYSICS_SUBVIEW_SLUGS.get(sub, "physics")
    if active_wb == "Audit & Sources":
        sub = st.session_state.get("audit_active_subview", _AUDIT_SUBVIEWS[0])
        return _AUDIT_SUBVIEW_SLUGS.get(sub, "validation")
    return _WORKBENCH_SLUGS.get(active_wb, "home")


def init_from_query_params() -> None:
    """Hydrate workbench & module selection from URL query params once before widgets render."""
    if st.session_state.get("_qp_initialized"):
        return
    try:
        qp = st.query_params
        wb_slug = qp.get("wb")
        if isinstance(wb_slug, list):
            wb_slug = wb_slug[0] if wb_slug else None
        mod_slug = qp.get("mod")
        if isinstance(mod_slug, list):
            mod_slug = mod_slug[0] if mod_slug else None

        if wb_slug and str(wb_slug).lower() in _SLUG_TO_ROUTE:
            _navigate_to(str(wb_slug).lower(), cuas_mod_slug=str(mod_slug).lower() if mod_slug else None)
        else:
            st.session_state.setdefault("active_workbench", _WORKBENCH_OPTIONS[0])
    except Exception:
        pass
    st.session_state["_qp_initialized"] = True


def sync_query_params() -> None:
    """Persist active workbench and C-UAS sub-module to URL query params safely."""
    try:
        wb_slug = _active_route_slug()
        st.query_params["wb"] = wb_slug
        if wb_slug == "cuas":
            tool = st.session_state.get("cuas_stage_tool", "Anti-Drone Multi-Shotgun Array")
            if tool in _CUAS_TOOL_TO_INTERNAL:
                st.query_params["mod"] = _CUAS_TOOL_TO_INTERNAL[tool][2]
            else:
                st.query_params["mod"] = "array"
        elif "mod" in st.query_params:
            del st.query_params["mod"]
    except Exception:
        pass


def build_deeplink_query() -> str:
    """Return relative shareable query string for current workbench/module."""
    wb_slug = _active_route_slug()
    if wb_slug == "cuas":
        tool = st.session_state.get("cuas_stage_tool", "Anti-Drone Multi-Shotgun Array")
        mod_slug = _CUAS_TOOL_TO_INTERNAL.get(tool, ("", "", "array"))[2]
        return f"?wb=cuas&mod={mod_slug}"
    return f"?wb={wb_slug}"


def _render_sidebar_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="sb-wb-badge">
            <div class="sb-wb-kicker">ACTIVE WORKBENCH</div>
            <div class="sb-wb-title">{title}</div>
            <div class="sb-wb-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _configure_page_and_styles() -> None:
    """Configure page metadata and inject theme-adaptive CSS styles on each run."""
    st.set_page_config(
        page_title="Ballistics & Dispersion Monte Carlo Workbench",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-title { font-size: 2.15rem; font-weight: 800; color: var(--text-color, #F8FAFC); margin-bottom: 0.15rem; }
        .sub-title { font-size: 1.02rem; color: var(--text-color, #CBD5E1); opacity: 0.80; margin-bottom: 1.0rem; }
        .gov-banner { background-color: rgba(245, 158, 11, 0.14); border: 1px solid rgba(245, 158, 11, 0.35); border-left: 5px solid #F59E0B; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.86rem; }
        .flag-banner { background-color: rgba(239, 68, 68, 0.14); border: 1px solid rgba(239, 68, 68, 0.35); border-left: 5px solid #EF4444; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.86rem; }
        .epistemic-banner { background-color: rgba(59, 130, 246, 0.14); border: 1px solid rgba(59, 130, 246, 0.35); border-left: 5px solid #3B82F6; color: var(--text-color, #F8FAFC); padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.0rem; font-size: 0.88rem; }
        .auth-card { background-color: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 10px; padding: 1.5rem 1.75rem; margin-top: 2.5rem; margin-bottom: 1rem; }
        .sb-wb-badge { background-color: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.30); border-radius: 8px; padding: 0.65rem 0.8rem; margin-bottom: 0.75rem; color: var(--text-color, #F8FAFC); }
        .sb-wb-kicker { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; opacity: 0.72; text-transform: uppercase; margin-bottom: 0.15rem; }
        .sb-wb-title { font-size: 1.02rem; font-weight: 700; line-height: 1.25; }
        .sb-wb-sub { font-size: 0.78rem; opacity: 0.80; margin-top: 0.2rem; line-height: 1.3; }
        [data-testid="stMetricValue"] { font-size: 1.55rem !important; line-height: 1.25 !important; white-space: normal !important; overflow-wrap: break-word !important; }
        [data-testid="stMetricDelta"] > div { white-space: normal !important; overflow-wrap: break-word !important; font-size: 0.80rem !important; }
        [data-testid="stMetricLabel"] p { white-space: normal !important; line-height: 1.2 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_password() -> bool:
    """Gate workbench access behind a rate-limited session password prompt on launch."""
    if st.session_state.setdefault("authenticated", False):
        return True

    limiter = st.session_state.get("_auth_rate_limiter")
    if not isinstance(limiter, LoginRateLimiter):
        limiter = LoginRateLimiter()
        st.session_state["_auth_rate_limiter"] = limiter

    st.markdown("<style>[data-testid='stSidebar'] { display: none; }</style>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1.1, 1.4, 1.1])
    with col_auth:
        st.markdown(
            """
            <div class="auth-card">
                <div style="font-size: 1.45rem; font-weight: 700; margin-bottom: 0.35rem;">
                    🔒 Ballistics & Dispersion Monte Carlo Workbench
                </div>
                <div style="font-size: 0.92rem; opacity: 0.82;">
                    Protected engineering workbench — enter the access password to continue.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("auth_login_form", clear_on_submit=False):
            entered_pw = st.text_input(
                "Access Password",
                type="password",
                placeholder="Enter password...",
                key="auth_pw_field",
            )
            submitted = st.form_submit_button("🔓 Unlock Workbench", width="stretch", type="primary")
        if submitted:
            locked, rem_s = limiter.is_locked()
            if locked:
                st.error(
                    f"⏳ Too many failed attempts. Please wait {max(1, math.ceil(rem_s))} s before retrying."
                )
            else:
                try:
                    secrets_map = st.secrets
                except Exception:
                    secrets_map = None
                if verify_app_password(entered_pw, secrets_mapping=secrets_map):
                    limiter.record_success()
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    cooldown = limiter.record_failure()
                    if cooldown > 0.0:
                        st.error(
                            f"❌ Incorrect password. Login locked for {math.ceil(cooldown)} s."
                        )
                    else:
                        st.error(
                            f"❌ Incorrect password. {limiter.remaining_attempts()} attempt(s) remaining before cooldown."
                        )
    return False


def render_header() -> None:
    st.markdown(
        '<div class="main-title" style="font-size:1.75rem;">Ballistics &amp; Counter-UAS Monte Carlo Workbench</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title" style="font-size:0.92rem; margin-bottom:0.55rem;">'
        'Stochastic flight dispersion, Counter-UAS multi-barrel engagements, terminal physics, and verified safety envelopes.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"⚖️ **Use Limitation:** {USE_LIMITATION_NOTICE}")
