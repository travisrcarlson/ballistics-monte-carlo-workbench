"""Small Arms Ballistics, Shotgun Patterning & Anti-Drone Dispersion Package — catalogs submodule."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

# Standard Shotgun Gauges Bore Diameters (meters)
GAUGE_BORE_DIAMETERS_M: dict[str, float] = {
    "12_gauge": 0.01852,  # 0.729 in
    "16_gauge": 0.01680,  # 0.662 in
    "20_gauge": 0.01563,  # 0.615 in
    "28_gauge": 0.01397,  # 0.550 in
    "410_bore": 0.01041,  # 0.410 in
}


# Standard Choke Constrictions and 40-yard (36.58m) 30-inch Circle Pattern Benchmark
CHOKE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Cylinder": {
        "constriction_mm": 0.00,
        "constriction_in": 0.000,
        "benchmark_40yd_30in_pct": 0.40,
        "description": "Zero constriction (straight bore). Maximum rapid spread for close-quarters.",
    },
    "Skeet": {
        "constriction_mm": 0.13,
        "constriction_in": 0.005,
        "benchmark_40yd_30in_pct": 0.45,
        "description": "Light constriction for skeet and close-range clays (15-25m).",
    },
    "Improved Cylinder": {
        "constriction_mm": 0.25,
        "constriction_in": 0.010,
        "benchmark_40yd_30in_pct": 0.50,
        "description": "Mild choke constriction for upland hunting and tactical defense (20-30m).",
    },
    "Modified": {
        "constriction_mm": 0.51,
        "constriction_in": 0.020,
        "benchmark_40yd_30in_pct": 0.60,
        "description": "All-around sporting and military choke (25-35m).",
    },
    "Improved Modified": {
        "constriction_mm": 0.64,
        "constriction_in": 0.025,
        "benchmark_40yd_30in_pct": 0.65,
        "description": "Tight choke for long-range trap and waterfowl.",
    },
    "Full": {
        "constriction_mm": 0.89,
        "constriction_in": 0.035,
        "benchmark_40yd_30in_pct": 0.72,
        "description": "High constriction for tight long-range patterns (35-45m).",
    },
    "Extra Full": {
        "constriction_mm": 1.02,
        "constriction_in": 0.040,
        "benchmark_40yd_30in_pct": 0.82,
        "description": "Maximum constriction for turkey hunting and long-range predator loads.",
    },
}


# Standard Pellet Specifications
PELLET_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "00 Buckshot": {
        "diameter_mm": 8.38,  # 0.330 in
        "mass_g": 3.49,        # 53.8 grains
        "standard_12g_count": 9,
        "type": "buckshot",
    },
    "000 Buckshot": {
        "diameter_mm": 9.14,  # 0.360 in
        "mass_g": 4.54,
        "standard_12g_count": 8,
        "type": "buckshot",
    },
    "#1 Buckshot": {
        "diameter_mm": 7.62,  # 0.300 in
        "mass_g": 2.62,
        "standard_12g_count": 16,
        "type": "buckshot",
    },
    "#2 Buckshot": {
        "diameter_mm": 6.86,  # 0.270 in
        "mass_g": 1.91,
        "standard_12g_count": 18,
        "type": "buckshot",
    },
    "#4 Buckshot": {
        "diameter_mm": 6.10,  # 0.240 in
        "mass_g": 1.34,
        "standard_12g_count": 27,
        "type": "buckshot",
    },
    "#2 Birdshot (Waterfowl)": {
        "diameter_mm": 3.81,  # 0.150 in
        "mass_g": 0.32,
        "standard_12g_count": 105,
        "type": "birdshot",
    },
    "#4 Birdshot (Field)": {
        "diameter_mm": 3.30,  # 0.130 in
        "mass_g": 0.21,
        "standard_12g_count": 135,
        "type": "birdshot",
    },
    "#7.5 Birdshot (Clay/Trap)": {
        "diameter_mm": 2.41,  # 0.095 in
        "mass_g": 0.081,
        "standard_12g_count": 350,
        "type": "birdshot",
    },
    "#8 Birdshot (Skeet)": {
        "diameter_mm": 2.29,  # 0.090 in
        "mass_g": 0.069,
        "standard_12g_count": 410,
        "type": "birdshot",
    },
    "1 oz Foster Rifled Slug": {
        "diameter_mm": 18.20,
        "mass_g": 28.35,
        "standard_12g_count": 1,
        "type": "slug",
    },
    "T-Shot (C-UAS)": {
        "diameter_mm": 5.08,
        "mass_g": 0.53,
        "standard_12g_count": 52,
        "type": "steel_shot",
    },
    "#2 Tungsten Super-Shot (TSS)": {
        "diameter_mm": 3.81,
        "mass_g": 0.52,
        "standard_12g_count": 70,
        "type": "tungsten_tss",
    },
    "Fin-Stabilized Flechette": {
        "diameter_mm": 1.90,
        "mass_g": 0.75,
        "standard_12g_count": 20,
        "type": "flechette",
    },
}


# Standard Shotgun Cartridge Presets
SHOTGUN_PRESETS: dict[str, dict[str, Any]] = {
    "12G_00_BUCK_TACTICAL": {
        "name": "12-Gauge 2-3/4\" 00-Buckshot Tactical (9 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "00 Buckshot",
        "pellets_per_shot": 9,
        "pellet_diameter_mm": 8.38,
        "pellet_mass_g": 3.49,
        "v0_mps": 405.0,
        "default_choke": "Improved Cylinder",
        "description": "Standard military / law enforcement combat buckshot load.",
    },
    "12G_00_BUCK_MAGNUM": {
        "name": "12-Gauge 3\" Magnum 00-Buckshot (12 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "00 Buckshot",
        "pellets_per_shot": 12,
        "pellet_diameter_mm": 8.38,
        "pellet_mass_g": 3.49,
        "v0_mps": 390.0,
        "default_choke": "Modified",
        "description": "Heavy magnum 12-pellet combat buckshot load.",
    },
    "12G_NO4_BUCK_HOME_DEFENSE": {
        "name": "12-Gauge #4 Buckshot Defense (27 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "#4 Buckshot",
        "pellets_per_shot": 27,
        "pellet_diameter_mm": 6.10,
        "pellet_mass_g": 1.34,
        "v0_mps": 385.0,
        "default_choke": "Cylinder",
        "description": "Dense pattern, reduced interior over-penetration defensive load.",
    },
    "12G_NO75_BIRD_TRAP": {
        "name": "12-Gauge #7.5 Target / Clay Birdshot (350 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "#7.5 Birdshot (Clay/Trap)",
        "pellets_per_shot": 350,
        "pellet_diameter_mm": 2.41,
        "pellet_mass_g": 0.081,
        "v0_mps": 395.0,
        "default_choke": "Modified",
        "description": "Standard 1 oz (350-pellet) sporting clay and trap competition load.",
    },
    "20G_NO2_BUCK_LIGHT": {
        "name": "20-Gauge #2 Buckshot (18 Pellets)",
        "gauge": "20_gauge",
        "pellet_type": "#2 Buckshot",
        "pellets_per_shot": 18,
        "pellet_diameter_mm": 6.86,
        "pellet_mass_g": 1.91,
        "v0_mps": 365.0,
        "default_choke": "Improved Cylinder",
        "description": "Reduced-recoil medium gauge defensive buckshot load.",
    },
    "12G_NO4_BUCK_C_UAS": {
        "name": "12-Gauge C-UAS #4 Buckshot (27 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "#4 Buckshot",
        "pellets_per_shot": 27,
        "pellet_diameter_mm": 6.10,
        "pellet_mass_g": 1.34,
        "v0_mps": 415.0,
        "default_choke": "Modified",
        "description": "High-velocity counter-drone #4 buckshot with optimal rotor-shredding density and penetration.",
    },
    "12G_T_SHOT_STEEL_C_UAS": {
        "name": "12-Gauge C-UAS T-Shot Steel (52 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "T-Shot (C-UAS)",
        "pellets_per_shot": 52,
        "pellet_diameter_mm": 5.08,
        "pellet_mass_g": 0.53,
        "v0_mps": 435.0,
        "default_choke": "Full",
        "effective_cd": 0.47,
        "description": "High-velocity heavy steel shot creating dense cloud curtains against agile FPV kamikaze drones.",
    },
    "12G_TSS_NO2_C_UAS": {
        "name": "12-Gauge C-UAS #2 Tungsten TSS (70 Pellets)",
        "gauge": "12_gauge",
        "pellet_type": "#2 Tungsten Super-Shot (TSS)",
        "pellets_per_shot": 70,
        "pellet_diameter_mm": 3.81,
        "pellet_mass_g": 0.52,
        "v0_mps": 410.0,
        "default_choke": "Full",
        "effective_cd": 0.44,
        "propellant_mass_g": 2.35,
        "wad_mass_g": 2.80,
        "description": "Ultra-dense 18.0 g/cm³ Tungsten Super-Shot for extended-range C-UAS penetration and high pellet count.",
    },
    "12G_DUPLEX_TSS_BUCK_C_UAS": {
        "name": "12-Gauge C-UAS Duplex TSS + #4 Buck (48 Projectiles)",
        "gauge": "12_gauge",
        "pellet_type": "#2 Tungsten Super-Shot (TSS)",
        "pellets_per_shot": 48,
        "pellet_diameter_mm": 4.50,
        "pellet_mass_g": 0.72,
        "v0_mps": 412.0,
        "default_choke": "Modified",
        "effective_cd": 0.46,
        "description": "Stacked duplex payload combining heavy core penetrators with dense rotor-shredding TSS sub-pellets.",
    },
    "12G_FLECHETTE_SABOT_C_UAS": {
        "name": "12-Gauge C-UAS Fin-Stabilized Flechette (20 Darts)",
        "gauge": "12_gauge",
        "pellet_type": "Fin-Stabilized Flechette",
        "pellets_per_shot": 20,
        "pellet_diameter_mm": 1.90,
        "pellet_mass_g": 0.75,
        "v0_mps": 445.0,
        "default_choke": "Cylinder",
        "effective_cd": 0.18,
        "description": "Low-drag saboted steel darts retaining lethal velocity past 80 m against armored drones.",
    },
}


# Multi-Barrel System Presets
MULTIBARREL_PRESETS: dict[str, dict[str, Any]] = {
    "SXS_DOUBLE_BARREL_12G": {
        "name": "Side-by-Side (SxS) 12G Double-Barrel",
        "system_type": "sxs_double",
        "n_barrels": 2,
        "bore_spacing_y_mm": 28.0,
        "bore_spacing_z_mm": 0.0,
        "crossover_range_m": 35.0,
        "cyclic_rate_rpm": 240.0,
        "inherent_dispersion_mrad": 3.0,
        "barrel_pointing_bias_mrad": 0.8,
        "thermal_drift_mrad_per_100rd": 0.2,
        "description": "Classic horizontal double-barrel regulated at 35m crossover.",
    },
    "OU_DOUBLE_BARREL_12G": {
        "name": "Over-Under (O/U) 12G Double-Barrel",
        "system_type": "ou_double",
        "n_barrels": 2,
        "bore_spacing_y_mm": 0.0,
        "bore_spacing_z_mm": 32.0,
        "crossover_range_m": 35.0,
        "cyclic_rate_rpm": 240.0,
        "inherent_dispersion_mrad": 2.5,
        "barrel_pointing_bias_mrad": 0.6,
        "thermal_drift_mrad_per_100rd": 0.2,
        "description": "Vertical over-under sporting shotgun regulated at 35m crossover.",
    },
    "M134_ROTARY_MINIGUN": {
        "name": "M134 7.62mm 6-Barrel Rotary Minigun",
        "system_type": "rotary_gatling",
        "n_barrels": 6,
        "rotor_radius_mm": 45.0,
        "crossover_range_m": 400.0,
        "cyclic_rate_rpm": 3000.0,
        "inherent_dispersion_mrad": 2.2,
        "barrel_pointing_bias_mrad": 1.2,
        "thermal_drift_mrad_per_100rd": 0.5,
        "description": "Six-barrel externally powered Gatling gun firing 7.62x51mm NATO.",
    },
    "GAU19_ROTARY_50BMG": {
        "name": "GAU-19/B 12.7mm (.50 BMG) 3-Barrel Rotary",
        "system_type": "rotary_gatling",
        "n_barrels": 3,
        "rotor_radius_mm": 65.0,
        "crossover_range_m": 800.0,
        "cyclic_rate_rpm": 1300.0,
        "inherent_dispersion_mrad": 1.8,
        "barrel_pointing_bias_mrad": 1.0,
        "thermal_drift_mrad_per_100rd": 0.4,
        "description": "Three-barrel heavy machine gun Gatling weapon firing 12.7x99mm NATO.",
    },
    "QUAD_AUTOCANNON_23MM": {
        "name": "Quad-Barrel 23mm Anti-Aircraft Mount",
        "system_type": "quad_mount",
        "n_barrels": 4,
        "bore_spacing_y_mm": 280.0,
        "bore_spacing_z_mm": 240.0,
        "crossover_range_m": 1200.0,
        "cyclic_rate_rpm": 3400.0,
        "inherent_dispersion_mrad": 1.5,
        "barrel_pointing_bias_mrad": 0.7,
        "thermal_drift_mrad_per_100rd": 0.3,
        "description": "Four-barrel anti-aircraft / counter-drone volley mount converging at 1200m.",
    },
}


def get_shotgun_presets() -> dict[str, dict[str, Any]]:
    """Return catalog of standard shotgun ammunition presets."""
    return SHOTGUN_PRESETS


def get_choke_definitions() -> dict[str, dict[str, Any]]:
    """Return catalog of standard shotgun choke constriction specifications."""
    return CHOKE_DEFINITIONS


def get_multibarrel_presets() -> dict[str, dict[str, Any]]:
    """Return catalog of standard multi-barrel systems and Gatling guns."""
    return MULTIBARREL_PRESETS


def get_pellet_specifications() -> dict[str, dict[str, Any]]:
    """Return standard pellet specifications."""
    return PELLET_SPECIFICATIONS


DRONE_TARGET_SPECS: dict[str, dict[str, Any]] = {
    "fpv_quad_small": {
        "name": "FPV Kamikaze / Racing Quad (250mm)",
        "drone_type": "quadcopter",
        "span_m": 0.25,
        "width_m": 0.25,
        "length_m": 0.25,
        "height_m": 0.08,
        "hub_radius_m": 0.045,
        "rotor_radius_m": 0.065,
        "rotor_thickness_m": 0.020,
        "motor_d_m": 0.032,
        "motor_h_m": 0.028,
        "arm_d_m": 0.016,
        "rotors": [
            {"id": "FL", "x_m": 0.09, "y_m": -0.09, "z_m_vert": 0.015, "z_m": 0.09, "radius_m": 0.065},
            {"id": "FR", "x_m": 0.09, "y_m": 0.09, "z_m_vert": 0.015, "z_m": 0.09, "radius_m": 0.065},
            {"id": "RL", "x_m": -0.09, "y_m": -0.09, "z_m_vert": 0.015, "z_m": -0.09, "radius_m": 0.065},
            {"id": "RR", "x_m": -0.09, "y_m": 0.09, "z_m_vert": 0.015, "z_m": -0.09, "radius_m": 0.065},
        ],
        "fuselage_l_m": 0.16,
        "fuselage_w_m": 0.08,
        "fuselage_h_m": 0.065,
        "battery_l_m": 0.090,
        "battery_w_m": 0.045,
        "battery_h_m": 0.035,
        "battery_offset_z_m": -0.045,
        "lethal_energy_rotor_j": 6.0,
        "lethal_energy_core_j": 16.0,
        "lethal_energy_arm_j": 28.0,
        "critical_hits_for_kill": 2,
        "description": "Small, high-speed agile FPV kamikaze quadcopter (15-35 m/s). Highly vulnerable to rotor loss.",
    },
    "commercial_quad_medium": {
        "name": "Commercial Recon Quad (Mavic/Phantom 400mm)",
        "drone_type": "quadcopter",
        "span_m": 0.40,
        "width_m": 0.40,
        "length_m": 0.40,
        "height_m": 0.16,
        "hub_radius_m": 0.07,
        "rotor_radius_m": 0.11,
        "rotor_thickness_m": 0.025,
        "motor_d_m": 0.042,
        "motor_h_m": 0.035,
        "arm_d_m": 0.020,
        "rotors": [
            {"id": "FL", "x_m": 0.15, "y_m": -0.15, "z_m_vert": 0.020, "z_m": 0.15, "radius_m": 0.11},
            {"id": "FR", "x_m": 0.15, "y_m": 0.15, "z_m_vert": 0.020, "z_m": 0.15, "radius_m": 0.11},
            {"id": "RL", "x_m": -0.15, "y_m": -0.15, "z_m_vert": 0.020, "z_m": -0.15, "radius_m": 0.11},
            {"id": "RR", "x_m": -0.15, "y_m": 0.15, "z_m_vert": 0.020, "z_m": -0.15, "radius_m": 0.11},
        ],
        "fuselage_l_m": 0.22,
        "fuselage_w_m": 0.12,
        "fuselage_h_m": 0.10,
        "battery_l_m": 0.130,
        "battery_w_m": 0.070,
        "battery_h_m": 0.055,
        "battery_offset_z_m": -0.065,
        "lethal_energy_rotor_j": 10.0,
        "lethal_energy_core_j": 24.0,
        "lethal_energy_arm_j": 35.0,
        "critical_hits_for_kill": 3,
        "description": "Medium commercial quadcopter used for scouting/spotting. Vulnerable to motor and battery strikes.",
    },
    "heavy_octocopter_large": {
        "name": "Heavy-Lift Bomber Octocopter (800mm)",
        "drone_type": "octocopter",
        "span_m": 0.80,
        "width_m": 0.80,
        "length_m": 0.80,
        "height_m": 0.30,
        "hub_radius_m": 0.16,
        "rotor_radius_m": 0.14,
        "rotor_thickness_m": 0.032,
        "motor_d_m": 0.058,
        "motor_h_m": 0.050,
        "arm_d_m": 0.028,
        "rotors": [
            {
                "id": f"R{i+1}",
                "x_m": round(0.32 * math.cos(i * math.pi / 4), 3),
                "y_m": round(0.32 * math.sin(i * math.pi / 4), 3),
                "z_m_vert": 0.025,
                "z_m": round(0.32 * math.cos(i * math.pi / 4), 3),
                "radius_m": 0.14,
            }
            for i in range(8)
        ],
        "fuselage_l_m": 0.35,
        "fuselage_w_m": 0.28,
        "fuselage_h_m": 0.15,
        "battery_l_m": 0.240,
        "battery_w_m": 0.160,
        "battery_h_m": 0.090,
        "battery_offset_z_m": -0.110,
        "lethal_energy_rotor_j": 18.0,
        "lethal_energy_core_j": 35.0,
        "lethal_energy_arm_j": 45.0,
        "critical_hits_for_kill": 5,
        "description": "Large multi-rotor drone carrying munition payloads. Armored electronics and redundant rotors.",
    },
    "fixed_wing_loitering": {
        "name": "Fixed-Wing Loitering Munition (1.6m Span)",
        "drone_type": "fixed_wing",
        "span_m": 1.60,
        "width_m": 1.60,
        "length_m": 1.10,
        "height_m": 0.35,
        "wingspan_m": 1.60,
        "wing_chord_m": 0.22,
        "wing_thickness_m": 0.035,
        "fuselage_l_m": 0.95,
        "fuselage_w_m": 0.20,
        "fuselage_h_m": 0.18,
        "fin_h_m": 0.24,
        "fin_chord_m": 0.18,
        "pusher_prop_radius_m": 0.15,
        "prop_thickness_m": 0.025,
        "rotors": [
            {"id": "Prop", "x_m": -0.48, "y_m": 0.0, "z_m_vert": 0.0, "z_m": -0.48, "radius_m": 0.15},
        ],
        "lethal_energy_rotor_j": 12.0,
        "lethal_energy_core_j": 30.0,
        "lethal_energy_arm_j": 20.0,
        "critical_hits_for_kill": 4,
        "description": "High-speed pusher-prop loitering drone (30-50 m/s). Wide wingspan and composite airframe.",
    },
}


def get_drone_target_specs() -> dict[str, dict[str, Any]]:
    """Return catalog of drone target vulnerability specifications."""
    return DRONE_TARGET_SPECS


DRONE_ASPECT_PRESETS: dict[str, dict[str, Any]] = {
    "head_on_frontal": {
        "name": "🎯 Head-on / Frontal Cross-Section (0° Ingress)",
        "yaw_deg": 0.0,
        "pitch_deg": -15.0,  # 15° forward cruise tilt
        "roll_deg": 0.0,
        "description": "Drone ingressing directly toward the gun. Minimal frontal silhouette with edge-on rotor discs.",
    },
    "flank_side": {
        "name": "📐 Flank / Side Profile Cross-Section (90° Crossing)",
        "yaw_deg": 90.0,
        "pitch_deg": -15.0,
        "roll_deg": 0.0,
        "description": "Drone flying perpendicular across line of fire. Presents fuselage side profile and separated rotor discs.",
    },
    "oblique_quartering": {
        "name": "↗️ Oblique / Quartering Cross-Section (45° Approach)",
        "yaw_deg": 45.0,
        "pitch_deg": -15.0,
        "roll_deg": 0.0,
        "description": "Intermediate angle approach presenting 3D compound cross-section of fuselage, arms, and rotors.",
    },
    "tail_rear": {
        "name": "🔙 Tail / Rear Cross-Section (180° Egress)",
        "yaw_deg": 180.0,
        "pitch_deg": -15.0,
        "roll_deg": 0.0,
        "description": "Drone retreating away from gun with aft-facing components, battery, and pusher prop.",
    },
    "planform_top": {
        "name": "⬇️ Planform Overhead / Top-Down (90° Elevation)",
        "yaw_deg": 0.0,
        "pitch_deg": -90.0,
        "roll_deg": 0.0,
        "description": "Full top-down planform silhouette (maximum exposed surface area).",
    },
    "planform_overhead": {
        "name": "⬇️ Planform Overhead / Top-Down (90° Elevation)",
        "yaw_deg": 0.0,
        "pitch_deg": -90.0,
        "roll_deg": 0.0,
        "description": "Full top-down planform silhouette (maximum exposed surface area).",
    },
    "planform_bottom": {
        "name": "⬆️ Planform Underbelly / Bottom-Up (-90° Elevation)",
        "yaw_deg": 0.0,
        "pitch_deg": 90.0,
        "roll_deg": 0.0,
        "description": "Full underbelly view exposing battery, underslung payload, and rotor undersides.",
    },
    "planform_underbelly": {
        "name": "⬆️ Planform Underbelly / Bottom-Up (-90° Elevation)",
        "yaw_deg": 0.0,
        "pitch_deg": 90.0,
        "roll_deg": 0.0,
        "description": "Full underbelly view exposing battery, underslung payload, and rotor undersides.",
    },
    "custom": {
        "name": "🎛️ Custom 3D Aspect & Orientation",
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "description": "Arbitrary 3-axis Euler angle orientation (pitch, roll, yaw).",
    },
}


def get_drone_aspect_presets() -> dict[str, dict[str, Any]]:
    """Return catalog of drone target aspect and presentation presets."""
    return DRONE_ASPECT_PRESETS


STANDARD_BARREL_LENGTHS: dict[str, dict[str, Any]] = {
    "14_in": {"length_in": 14.0, "length_cm": 35.6, "name": "14\" Breacher / Entry", "category": "Ultra-Short"},
    "18_5_in": {"length_in": 18.5, "length_cm": 47.0, "name": "18.5\" Tactical / Patrol", "category": "Tactical"},
    "22_in": {"length_in": 22.0, "length_cm": 55.9, "name": "22\" 3-Gun / Practical", "category": "Competition"},
    "26_in": {"length_in": 26.0, "length_cm": 66.0, "name": "26\" Upland / Field", "category": "Field"},
    "28_in": {"length_in": 28.0, "length_cm": 71.1, "name": "28\" Sporting Clays / Trap", "category": "Sporting (Reference)"},
    "30_in": {"length_in": 30.0, "length_cm": 76.2, "name": "30\" Trap / Competition", "category": "Target"},
    "32_in": {"length_in": 32.0, "length_cm": 81.3, "name": "32\" Goose / Waterfowl", "category": "Long Range"},
}


def get_standard_barrel_lengths() -> dict[str, dict[str, Any]]:
    """Return catalog of standard small-arms shotgun barrel lengths."""
    return STANDARD_BARREL_LENGTHS


