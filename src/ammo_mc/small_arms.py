"""Small Arms Ballistics, Shotgun Patterning, and Multi-Barrel Dispersion Engine.

Provides first-principles exterior ballistics and Monte Carlo dispersion modeling for:
1. Shotgun Ballistics & Choke Constriction Mechanics:
   - Standard pellet sizes (00 Buck, #1 Buck, #4 Buck, #4 Bird, #7.5 Bird, #8 Bird, Slugs, Custom)
   - Choke constrictions (Cylinder, Improved Cylinder, Modified, Improved Modified, Full, Extra Full)
   - Lowry / Journée Gaussian and Rayleigh radial dispersion models
   - 3D Longitudinal Shot Stringing dynamics and moving-target exposure
   - Sphere aerodynamic drag C_D(M), striking kinetic energy, and soft-tissue / ballistic gelatin penetration
   - Standard 30-inch and 10-inch pattern circle densities, E-type humanoid silhouette, clay pigeon, and drone target checks
2. Multi-Barrel Systems & Volley Dispersion:
   - Side-by-Side (SxS) and Over-Under (O/U) double-barrel regulation and crossover geometry
   - Rotary / Gatling multi-barrel systems (e.g., M134 Minigun 6-barrel, GAU-19 3-barrel)
   - Individual barrel pointing biases (bore run-out, crown asymmetry)
   - Rotor centrifugal / tangential dynamics and thermal walk drift
   - Combined multi-barrel salvo dispersion, individual barrel cluster CEPs, and pattern overlap
"""

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
        "description": "Standard 1-1/8 oz sporting clay and trap competition load.",
    },
    "20G_NO2_BUCK_LIGHT": {
        "name": "20-Gauge #2 Buckshot (18 Pellets)",
        "gauge": "20_gauge",
        "pellet_type": "#1 Buckshot",
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


# -------------------------------------------------------------------------
# DRONE TARGET SPECIFICATIONS (C-UAS VULNERABILITY MODELS)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# DRONE ASPECT AND 3D CROSS-SECTION PRESENTATION CATALOG
# -------------------------------------------------------------------------
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


def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull of points (monotone chain algorithm). Returns vertices in CCW order."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) <= 3:
        return pts
    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) <= 3:
        return unique_pts
    idx = np.lexsort((unique_pts[:, 1], unique_pts[:, 0]))
    pts_sorted = unique_pts[idx]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts_sorted:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


def polygon_area(pts: np.ndarray) -> float:
    """Calculate area of a 2D polygon using the Shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def point_in_convex_polygon(y: float, z: float, poly_verts: np.ndarray) -> bool:
    """Test if point (y, z) is inside a 2D convex polygon."""
    n = len(poly_verts)
    if n < 3:
        return False
    signs: list[bool] = []
    for i in range(n):
        p1 = poly_verts[i]
        p2 = poly_verts[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (z - p1[1]) - (p2[1] - p1[1]) * (y - p1[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    if not signs:
        return True
    return all(s == signs[0] for s in signs)


def compute_drone_projected_geometry(
    drone_spec: dict[str, Any],
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
) -> dict[str, Any]:
    """Compute the 2D projected cross-section silhouette and component boundaries.

    Transforms 3D drone components (fuselage, battery, rotors, motor pods, arms,
    wings) into the 2D line-of-fire target plane (Y = horizontal, Z = vertical)
    under arbitrary 3-axis Euler angles (pitch, yaw, roll).
    """
    preset = DRONE_ASPECT_PRESETS.get(aspect_mode, DRONE_ASPECT_PRESETS["head_on_frontal"])
    if aspect_mode != "custom":
        yaw_deg = float(preset["yaw_deg"])
        pitch_deg = float(preset["pitch_deg"])
        roll_deg = float(preset["roll_deg"])
    else:
        yaw_deg = float(drone_yaw_deg)
        pitch_deg = float(drone_pitch_deg)
        roll_deg = float(drone_roll_deg)

    psi_r = math.radians(yaw_deg)
    theta_r = math.radians(pitch_deg)
    phi_r = math.radians(roll_deg)

    # 1. Forward unit vector (nose)
    ux_fwd = -math.cos(psi_r) * math.cos(theta_r)
    uy_fwd = math.sin(psi_r) * math.cos(theta_r)
    uz_fwd = math.sin(theta_r)
    u_fwd = np.array([ux_fwd, uy_fwd, uz_fwd], dtype=float)
    norm_f = np.linalg.norm(u_fwd)
    u_fwd = u_fwd / norm_f if norm_f > 1e-9 else np.array([-1.0, 0.0, 0.0])

    # 2. Lateral unit vector (starboard)
    if abs(abs(theta_r) - math.pi / 2) < 1e-4:
        u_r0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        u_r0 = np.array([math.sin(psi_r), math.cos(psi_r), 0.0], dtype=float)
        norm_r0 = np.linalg.norm(u_r0)
        u_r0 = u_r0 / norm_r0 if norm_r0 > 1e-9 else np.array([0.0, 1.0, 0.0])

    # Rodrigues roll around u_fwd
    u_right = (
        u_r0 * math.cos(phi_r)
        + np.cross(u_fwd, u_r0) * math.sin(phi_r)
        + u_fwd * (np.dot(u_fwd, u_r0)) * (1.0 - math.cos(phi_r))
    )
    norm_r = np.linalg.norm(u_right)
    u_right = u_right / norm_r if norm_r > 1e-9 else np.array([0.0, 1.0, 0.0])

    # 3. Upward unit vector (dorsal)
    u_up = np.cross(u_right, u_fwd)
    if u_up[2] < 0 and abs(theta_r) < math.pi / 2 - 0.05:
        u_up = -u_up
    norm_up = np.linalg.norm(u_up)
    u_up = u_up / norm_up if norm_up > 1e-9 else np.array([0.0, 0.0, 1.0])

    # 3x3 rotation matrix: p_los = M @ [x_b, y_b, z_b]
    M = np.column_stack([u_fwd, u_right, u_up])

    # Fuselage 3D Box
    l_f = float(drone_spec.get("fuselage_l_m", drone_spec.get("span_m", 0.4) * 0.55))
    w_f = float(drone_spec.get("fuselage_w_m", 0.12))
    h_f = float(drone_spec.get("fuselage_h_m", 0.09))

    verts_b = np.array([
        [+l_f / 2, +w_f / 2, +h_f / 2],
        [+l_f / 2, +w_f / 2, -h_f / 2],
        [+l_f / 2, -w_f / 2, +h_f / 2],
        [+l_f / 2, -w_f / 2, -h_f / 2],
        [-l_f / 2, +w_f / 2, +h_f / 2],
        [-l_f / 2, +w_f / 2, -h_f / 2],
        [-l_f / 2, -w_f / 2, +h_f / 2],
        [-l_f / 2, -w_f / 2, -h_f / 2],
    ], dtype=float)
    verts_los = verts_b @ M.T
    fuse_proj = verts_los[:, [1, 2]]
    fuse_hull = _convex_hull_2d(fuse_proj)
    area_fuse = polygon_area(fuse_hull)

    # Battery 3D Box
    l_b = float(drone_spec.get("battery_l_m", l_f * 0.6))
    w_b = float(drone_spec.get("battery_w_m", w_f * 0.6))
    h_b = float(drone_spec.get("battery_h_m", h_f * 0.55))
    z_b_off = float(drone_spec.get("battery_offset_z_m", -h_f / 2 - h_b / 2))

    b_verts_b = np.array([
        [+l_b / 2, +w_b / 2, z_b_off + h_b / 2],
        [+l_b / 2, +w_b / 2, z_b_off - h_b / 2],
        [+l_b / 2, -w_b / 2, z_b_off + h_b / 2],
        [+l_b / 2, -w_b / 2, z_b_off - h_b / 2],
        [-l_b / 2, +w_b / 2, z_b_off + h_b / 2],
        [-l_b / 2, +w_b / 2, z_b_off - h_b / 2],
        [-l_b / 2, -w_b / 2, z_b_off + h_b / 2],
        [-l_b / 2, -w_b / 2, z_b_off - h_b / 2],
    ], dtype=float)
    b_verts_los = b_verts_b @ M.T
    batt_proj = b_verts_los[:, [1, 2]]
    batt_hull = _convex_hull_2d(batt_proj)
    area_batt = polygon_area(batt_hull)

    # Rotors & Motor Pods
    rotors_spec = drone_spec.get("rotors", [])
    h_blade = float(drone_spec.get("rotor_thickness_m", 0.022))
    d_motor = float(drone_spec.get("motor_d_m", 0.038))
    h_motor = float(drone_spec.get("motor_h_m", 0.032))
    arm_d = float(drone_spec.get("arm_d_m", 0.020))

    rotors_out = []
    motors_out = []
    arms_out = []
    area_rotors_tot = 0.0
    area_motors_tot = 0.0

    n_normal_los = M @ np.array([0.0, 0.0, 1.0], dtype=float)
    nx, ny, nz = float(n_normal_los[0]), float(n_normal_los[1]), float(n_normal_los[2])
    d_yz = math.hypot(ny, nz)

    if d_yz > 1e-6:
        u_maj = np.array([-nz / d_yz, ny / d_yz], dtype=float)
        u_min = np.array([ny / d_yz, nz / d_yz], dtype=float)
    else:
        u_maj = np.array([1.0, 0.0], dtype=float)
        u_min = np.array([0.0, 1.0], dtype=float)

    c_fuse_proj = (M @ np.array([0.0, 0.0, 0.0], dtype=float))[[1, 2]]

    for r_idx, r in enumerate(rotors_spec):
        rx_b = float(r.get("x_m", r.get("z_m", 0.0)))
        ry_b = float(r.get("y_m", 0.0))
        rz_b = float(r.get("z_m_vert", 0.015))
        r_rad = float(r.get("radius_m", 0.10))

        c_hub_los = M @ np.array([rx_b, ry_b, rz_b], dtype=float)
        cy, cz = float(c_hub_los[1]), float(c_hub_los[2])

        a = r_rad
        b = max(r_rad * abs(nx), h_blade / 2.0)
        area_rot = math.pi * a * b
        area_rotors_tot += area_rot

        rotors_out.append({
            "id": r.get("id", f"R{r_idx+1}"),
            "y_m": cy,
            "z_m": cz,
            "semi_major_m": a,
            "semi_minor_m": b,
            "major_axis_m": a,
            "minor_axis_m": b,
            "u_maj": u_maj.tolist(),
            "u_min": u_min.tolist(),
            "area_m2": area_rot,
        })

        area_mot = d_motor * h_motor
        area_motors_tot += area_mot
        motors_out.append({
            "id": r.get("id", f"M{r_idx+1}"),
            "y_m": cy,
            "z_m": cz,
            "w_m": d_motor,
            "h_m": h_motor,
            "area_m2": area_mot,
        })

        arms_out.append({
            "id": f"Arm_{r_idx+1}",
            "p1": [float(c_fuse_proj[0]), float(c_fuse_proj[1])],
            "p2": [cy, cz],
            "diameter_m": arm_d,
        })

    # Fixed-Wing components (if fixed_wing)
    wings_out = []
    area_wings = 0.0
    if drone_spec.get("drone_type") == "fixed_wing":
        b_span = float(drone_spec.get("wingspan_m", 1.60))
        c_chord = float(drone_spec.get("wing_chord_m", 0.22))
        t_thick = float(drone_spec.get("wing_thickness_m", 0.035))
        # Left Wing
        w_l_b = np.array([
            [-c_chord / 2, -b_span / 2, +t_thick / 2],
            [-c_chord / 2, -b_span / 2, -t_thick / 2],
            [+c_chord / 2, -b_span / 2, +t_thick / 2],
            [+c_chord / 2, -b_span / 2, -t_thick / 2],
            [-c_chord / 2, -w_f / 2, +t_thick / 2],
            [-c_chord / 2, -w_f / 2, -t_thick / 2],
            [+c_chord / 2, -w_f / 2, +t_thick / 2],
            [+c_chord / 2, -w_f / 2, -t_thick / 2],
        ], dtype=float)
        w_l_proj = (w_l_b @ M.T)[:, [1, 2]]
        w_l_hull = _convex_hull_2d(w_l_proj)
        wings_out.append(w_l_hull.tolist())
        area_wings += polygon_area(w_l_hull)

        # Right Wing
        w_r_b = np.array([
            [-c_chord / 2, +w_f / 2, +t_thick / 2],
            [-c_chord / 2, +w_f / 2, -t_thick / 2],
            [+c_chord / 2, +w_f / 2, +t_thick / 2],
            [+c_chord / 2, +w_f / 2, -t_thick / 2],
            [-c_chord / 2, +b_span / 2, +t_thick / 2],
            [-c_chord / 2, +b_span / 2, -t_thick / 2],
            [+c_chord / 2, +b_span / 2, +t_thick / 2],
            [+c_chord / 2, +b_span / 2, -t_thick / 2],
        ], dtype=float)
        w_r_proj = (w_r_b @ M.T)[:, [1, 2]]
        w_r_hull = _convex_hull_2d(w_r_proj)
        wings_out.append(w_r_hull.tolist())
        area_wings += polygon_area(w_r_hull)

    # Total Presented Area
    span_m = float(drone_spec.get("span_m", 0.40))
    planform_area = span_m**2 * 0.62
    tot_presented_area = area_fuse + area_batt + area_wings + area_rotors_tot * 0.75 + area_motors_tot * 0.5
    cross_section_ratio_pct = min(100.0, (tot_presented_area / planform_area) * 100.0)

    return {
        "aspect_mode": aspect_mode,
        "name": preset["name"],
        "description": preset["description"],
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "rotation_matrix": M.tolist(),
        "fuselage_polygon": fuse_hull.tolist(),
        "battery_polygon": batt_hull.tolist(),
        "rotors": rotors_out,
        "motors": motors_out,
        "arms": arms_out,
        "wings": wings_out,
        "presented_area_m2": round(tot_presented_area, 6),
        "presented_area_cm2": round(tot_presented_area * 10000.0, 1),
        "planform_area_m2": round(planform_area, 6),
        "cross_section_ratio_pct": round(cross_section_ratio_pct, 1),
        "planform_ratio_pct": round(cross_section_ratio_pct, 1),
    }


# -------------------------------------------------------------------------
# TARGET SILHOUETTE CHECKING
# -------------------------------------------------------------------------
def check_target_hit(
    y_m: float,
    z_m: float,
    target_type: str = "30_inch_circle",
    custom_w_m: float = 0.5,
    custom_h_m: float = 1.0,
) -> bool:
    """Evaluate whether an impact point (y, z) lies within the specified target silhouette.

    Coordinates: y = horizontal (crossrange), z = vertical (height from target center).
    """
    if target_type == "30_inch_circle":
        # 30-inch diameter = 0.762 m (radius = 0.381 m)
        return float(y_m**2 + z_m**2) <= (0.381**2)

    elif target_type == "10_inch_circle":
        # 10-inch diameter = 0.254 m (radius = 0.127 m)
        return float(y_m**2 + z_m**2) <= (0.127**2)

    elif target_type == "e_type_silhouette":
        # NATO E-type humanoid silhouette:
        # Torso: width 0.45m (-0.225 to +0.225), height 0.65m (-0.45 to +0.20)
        # Head: width 0.22m (-0.11 to +0.11), height 0.25m (+0.20 to +0.45)
        in_torso = (-0.225 <= y_m <= 0.225) and (-0.45 <= z_m <= 0.20)
        in_head = (-0.11 <= y_m <= 0.11) and (0.20 < z_m <= 0.45)
        return bool(in_torso or in_head)

    elif target_type == "clay_pigeon":
        # Clay target: oval disc 11cm wide (0.11m), 3cm thick (0.03m)
        return float((y_m / 0.055) ** 2 + (z_m / 0.025) ** 2) <= 1.0

    elif target_type == "drone":
        # Micro-UAV / Drone: 0.40m x 0.40m square
        return bool(abs(y_m) <= 0.20 and abs(z_m) <= 0.20)

    elif target_type == "rectangle":
        return bool(abs(y_m) <= custom_w_m / 2.0 and abs(z_m) <= custom_h_m / 2.0)

    # Default to 30-inch circle
    return float(y_m**2 + z_m**2) <= (0.381**2)


# -------------------------------------------------------------------------
# SHOTGUN MONTE CARLO SIMULATOR
# -------------------------------------------------------------------------
def simulate_shotgun_pattern_mc(
    n_shots: int = 10,
    pellets_per_shot: int = 9,
    pellet_diameter_mm: float = 8.38,
    pellet_mass_g: float = 3.49,
    v0_mps: float = 405.0,
    choke_name: str = "Modified",
    target_distance_m: float = 36.58,
    target_type: str = "30_inch_circle",
    moving_target_velocity_mps: float = 0.0,
    moving_target_crossing_angle_deg: float = 90.0,
    lead_compensation_pct: float = 100.0,
    crosswind_mps: float = 0.0,
    aim_error_mrad: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate shotgun pellet cloud exterior ballistics and dispersion using Monte Carlo.

    Incorporates:
    - Choke constriction geometry and benchmark containment (Lowry-Oberfell model)
    - 3D Longitudinal shot stringing due to differential pellet deformation
    - Striking velocity and kinetic energy decay
    - Ballistic gelatin penetration depth
    - Target silhouette intersection (static or moving target)
    """
    rng = np.random.default_rng(seed)

    # 1. Resolve Choke Parameters
    choke_info = CHOKE_DEFINITIONS.get(choke_name, CHOKE_DEFINITIONS["Modified"])
    benchmark_pct = float(choke_info["benchmark_40yd_30in_pct"])
    r_30in = 0.381  # 15 inches = 0.381 meters
    r_10in = 0.127  # 5 inches = 0.127 meters
    r_bench = 36.58  # 40 yards

    # Gaussian radial variance parameter sigma at benchmark range:
    # F(r) = 1 - exp(-r^2 / (2 * sigma^2)) => sigma_ref = r / sqrt(-2 * ln(1 - F))
    clamped_pct = max(0.15, min(0.95, benchmark_pct))
    sigma_ref = r_30in / math.sqrt(-2.0 * math.log(1.0 - clamped_pct))
    theta_choke_rad = sigma_ref / r_bench

    # Angular spread half-angle with turbulent expansion term:
    dist_ratio = max(0.1, target_distance_m / r_bench)
    sigma_target = 0.005 + theta_choke_rad * target_distance_m * (dist_ratio**0.15)

    # 2. Pellet Aerodynamic Deceleration Parameters
    d_m = pellet_diameter_mm / 1000.0
    m_kg = pellet_mass_g / 1000.0
    rho_air = 1.225
    cd_base = 0.47  # subsonic smooth sphere drag

    # Differential drag factor distribution due to launch setback deformation
    # Leading pellets stay rounder; trailing pellets flatten
    total_pellets = int(n_shots * pellets_per_shot)

    # Target movement vector
    rad_cross = math.radians(moving_target_crossing_angle_deg)
    v_tgt_y = moving_target_velocity_mps * math.sin(rad_cross)
    v_tgt_z = moving_target_velocity_mps * math.cos(rad_cross)

    pellet_records: list[dict[str, Any]] = []
    shot_string_lengths_m: list[float] = []

    hits_30in_total = 0
    hits_10in_total = 0
    hits_target_total = 0

    all_y_m: list[float] = []
    all_z_m: list[float] = []
    all_v_impact: list[float] = []
    all_energies: list[float] = []
    all_t_arrive: list[float] = []

    for s_idx in range(n_shots):
        # Shot-to-shot pointing error (aim jitter)
        shot_aim_y = rng.normal(0.0, aim_error_mrad * target_distance_m / 1000.0)
        shot_aim_z = rng.normal(0.0, aim_error_mrad * target_distance_m / 1000.0)

        # Crosswind drift offset: approx drift = 0.5 * rho * A * Cd * w_z * t^2 / m
        # simplified closed form
        t_est = target_distance_m / max(v0_mps * 0.8, 100.0)
        wind_drift_y = 0.5 * (crosswind_mps * 0.15) * (t_est**1.3)

        # Generate individual pellets for this shot
        # Deformation: normalized 0.0 (round front) to 1.0 (flattened rear)
        deformations = rng.uniform(0.0, 1.0, size=pellets_per_shot)
        # Radial positions follow Rayleigh distribution
        radii = rng.rayleigh(scale=sigma_target, size=pellets_per_shot)
        thetas = rng.uniform(0.0, 2.0 * math.pi, size=pellets_per_shot)

        # Longitudinal stringing: deformed pellets experience higher drag
        # Drag variation: +0% to +35%
        cd_pellets = cd_base * (1.0 + 0.35 * deformations)
        kappas = (3.0 * rho_air * cd_pellets) / (4.0 * (m_kg / (math.pi / 6.0 * d_m**3)) * d_m)

        # Arrival velocity: v(x) = v0 * exp(-kappa * x)
        v_impacts = v0_mps * np.exp(-kappas * target_distance_m)

        # Arrival time: t = integral dx / v = (exp(kappa * x) - 1) / (v0 * kappa)
        t_arrivals = (np.exp(kappas * target_distance_m) - 1.0) / (v0_mps * kappas)

        # Shot string length for this shot: distance between fastest and slowest pellet
        t_min = float(np.min(t_arrivals))
        t_max = float(np.max(t_arrivals))
        t_mean = float(np.mean(t_arrivals))
        v_mean = float(np.mean(v_impacts))
        string_len_m = (t_max - t_min) * v_mean
        shot_string_lengths_m.append(string_len_m)

        # Target lead arrival time:
        # 100% lead compensation aims at pattern arrival t_mean;
        # 0% lead compensation aims at target position at shot release (t=0)
        t_lead = (lead_compensation_pct / 100.0) * t_mean

        for p_idx in range(pellets_per_shot):
            r_p = float(radii[p_idx])
            th_p = float(thetas[p_idx])
            v_p = float(v_impacts[p_idx])
            t_p = float(t_arrivals[p_idx])

            # Position on target plane without target motion
            y_base = shot_aim_y + wind_drift_y + r_p * math.cos(th_p)
            z_base = shot_aim_z + r_p * math.sin(th_p)

            # Gravity drop: approx 0.5 * g * t^2
            z_drop = -0.5 * 9.80665 * (t_p**2)
            z_final_static = z_base + z_drop

            # Moving target displacement relative to aim point
            y_rel = y_base - v_tgt_y * (t_p - t_lead)
            z_rel = z_final_static - v_tgt_z * (t_p - t_lead)

            # Hit checks
            is_30in = float(y_rel**2 + z_rel**2) <= (r_30in**2)
            is_10in = float(y_rel**2 + z_rel**2) <= (r_10in**2)
            is_target_hit = check_target_hit(y_rel, z_rel, target_type=target_type)

            if is_30in:
                hits_30in_total += 1
            if is_10in:
                hits_10in_total += 1
            if is_target_hit:
                hits_target_total += 1

            # Kinetic energy: 0.5 * m * v^2
            ek_j = 0.5 * m_kg * (v_p**2)

            # Ballistic gelatin penetration (Fackler / MacPherson soft tissue model, cm)
            # Penetration ~ 0.00185 * (mass_g / d_mm^2)^0.7 * v^1.25
            sect_density = (pellet_mass_g) / (pellet_diameter_mm**2)
            pen_cm = 0.00185 * (sect_density**0.7) * (v_p**1.25)

            all_y_m.append(y_rel)
            all_z_m.append(z_rel)
            all_v_impact.append(v_p)
            all_energies.append(ek_j)
            all_t_arrive.append(t_p)

            # Record sampled pellets for telemetry table (limit size)
            if len(pellet_records) < 120:
                pellet_records.append(
                    {
                        "shot_index": s_idx + 1,
                        "pellet_index": p_idx + 1,
                        "y_m": round(y_rel, 4),
                        "z_m": round(z_rel, 4),
                        "radial_dist_m": round(math.sqrt(y_rel**2 + z_rel**2), 4),
                        "v_impact_mps": round(v_p, 1),
                        "energy_j": round(ek_j, 1),
                        "gel_penetration_cm": round(pen_cm, 1),
                        "arrival_time_ms": round(t_p * 1000.0, 2),
                        "hit_30in": bool(is_30in),
                        "hit_10in": bool(is_10in),
                        "hit_target": bool(is_target_hit),
                    }
                )

    # 3. Aggregate Statistical Diagnostics
    y_arr = np.array(all_y_m)
    z_arr = np.array(all_z_m)
    r_arr = np.sqrt(y_arr**2 + z_arr**2)

    pct_30in = (hits_30in_total / total_pellets) * 100.0
    pct_10in = (hits_10in_total / total_pellets) * 100.0
    pct_target = (hits_target_total / total_pellets) * 100.0

    mean_v_impact = float(np.mean(all_v_impact))
    mean_energy = float(np.mean(all_energies))
    total_energy_delivered = float(np.sum(all_energies))
    mean_string_len = float(np.mean(shot_string_lengths_m))

    # 50% and 90% radial containment (CEP50 and R90)
    sorted_r = np.sort(r_arr)
    cep_50_m = float(sorted_r[int(0.50 * (len(sorted_r) - 1))])
    r_90_m = float(sorted_r[int(0.90 * (len(sorted_r) - 1))])

    # Pattern Uniformity Index (ratio of actual 10in to 30in hits vs ideal Gaussian)
    # Ideal Gaussian ratio = (1 - exp(-r10^2/(2*sigma^2))) / (1 - exp(-r30^2/(2*sigma^2)))
    uniformity_index = (hits_10in_total / max(hits_30in_total, 1)) / (
        (r_10in / r_30in) ** 2
    )

    return {
        "n_shots": n_shots,
        "pellets_per_shot": pellets_per_shot,
        "total_pellets_fired": total_pellets,
        "pellet_diameter_mm": pellet_diameter_mm,
        "pellet_mass_g": pellet_mass_g,
        "v0_mps": v0_mps,
        "choke_name": choke_name,
        "benchmark_choke_pct": benchmark_pct * 100.0,
        "target_distance_m": target_distance_m,
        "target_type": target_type,
        "moving_target_velocity_mps": moving_target_velocity_mps,
        "hits_30in_count": hits_30in_total,
        "hits_30in_pct": round(pct_30in, 2),
        "hits_10in_count": hits_10in_total,
        "hits_10in_pct": round(pct_10in, 2),
        "hits_target_count": hits_target_total,
        "hits_target_pct": round(pct_target, 2),
        "cep_50_m": round(cep_50_m, 3),
        "r_90_m": round(r_90_m, 3),
        "mean_v_impact_mps": round(mean_v_impact, 1),
        "mean_pellet_energy_j": round(mean_energy, 1),
        "total_energy_delivered_j": round(total_energy_delivered, 1),
        "mean_shot_string_length_m": round(mean_string_len, 2),
        "pattern_uniformity_index": round(uniformity_index, 2),
        "center_of_mass_y_m": round(float(np.mean(y_arr)), 4),
        "center_of_mass_z_m": round(float(np.mean(z_arr)), 4),
        "pellet_coordinates": {
            "y_m": [round(float(v), 4) for v in y_arr],
            "z_m": [round(float(v), 4) for v in z_arr],
            "v_impact_mps": [round(float(v), 1) for v in all_v_impact],
            "energy_j": [round(float(v), 1) for v in all_energies],
            "t_arrive_ms": [round(float(v) * 1000.0, 2) for v in all_t_arrive],
        },
        "sample_pellet_records": pellet_records,
    }


# -------------------------------------------------------------------------
# MULTI-BARREL & GATLING SYSTEM SIMULATOR
# -------------------------------------------------------------------------
def simulate_multibarrel_burst_mc(
    system_type: str = "sxs_double",
    n_barrels: int = 2,
    bore_spacing_y_mm: float = 28.0,
    bore_spacing_z_mm: float = 0.0,
    rotor_radius_mm: float = 45.0,
    crossover_range_m: float = 35.0,
    burst_rounds: int = 40,
    target_range_m: float = 35.0,
    cyclic_rate_rpm: float = 3000.0,
    inherent_dispersion_mrad: float = 2.5,
    barrel_pointing_bias_mrad: float = 1.0,
    thermal_drift_mrad_per_100rd: float = 0.3,
    target_type: str = "e_type_silhouette",
    drone_model: str = "commercial_quad_medium",
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = 15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    bullet_energy_j: float = 1800.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate dispersion and point-of-impact clusters for multi-barrel and Gatling systems.

    Models:
    - Bore physical spacing and convergence angles toward regulation crossover range
    - Before crossover: converging pattern; after crossover: diverging pattern
    - Individual barrel manufacturing run-out biases (random per-barrel boresight error)
    - Rotor angular velocity impulse (for Gatling / rotary systems)
    - Continuous thermal walk across burst length
    - Target drone 3D projected cross-section geometry, component vulnerability, and kill status
    - Combined salvo CEP, per-barrel cluster centroids, and target hit probability
    """
    rng = np.random.default_rng(seed)

    # 1. Determine Muzzle Positions of Barrels (in meters)
    barrel_muzzle_pos: list[tuple[float, float]] = []

    if system_type == "sxs_double":
        n_b = 2
        half_y = (bore_spacing_y_mm / 1000.0) / 2.0
        barrel_muzzle_pos = [(-half_y, 0.0), (half_y, 0.0)]

    elif system_type == "ou_double":
        n_b = 2
        half_z = (bore_spacing_z_mm / 1000.0) / 2.0
        barrel_muzzle_pos = [(0.0, half_z), (0.0, -half_z)]

    elif system_type == "rotary_gatling":
        n_b = max(3, min(8, n_barrels))
        r_m = rotor_radius_mm / 1000.0
        for i in range(n_b):
            angle = 2.0 * math.pi * i / n_b
            barrel_muzzle_pos.append((r_m * math.cos(angle), r_m * math.sin(angle)))

    elif system_type == "quad_mount":
        n_b = 4
        half_y = (bore_spacing_y_mm / 1000.0) / 2.0
        half_z = (bore_spacing_z_mm / 1000.0) / 2.0
        barrel_muzzle_pos = [
            (-half_y, half_z),
            (half_y, half_z),
            (-half_y, -half_z),
            (half_y, -half_z),
        ]
    else:
        # Default: circular array of n_barrels
        n_b = max(2, n_barrels)
        r_m = max(0.01, rotor_radius_mm / 1000.0)
        for i in range(n_b):
            angle = 2.0 * math.pi * i / n_b
            barrel_muzzle_pos.append((r_m * math.cos(angle), r_m * math.sin(angle)))

    # 2. Individual Barrel Manufacturing Biases (fixed per barrel)
    barrel_biases: list[tuple[float, float]] = []
    for _ in range(n_b):
        bias_y = rng.normal(0.0, barrel_pointing_bias_mrad / 1000.0)
        bias_z = rng.normal(0.0, barrel_pointing_bias_mrad / 1000.0)
        barrel_biases.append((bias_y, bias_z))

    # 3. Convergence Angles toward Crossover Range
    # Barrel aims toward (crossover_range_m, 0, 0)
    barrel_aim_angles: list[tuple[float, float]] = []
    for y0, z0 in barrel_muzzle_pos:
        psi_y = -y0 / max(crossover_range_m, 1.0)  # rad
        gamma_z = -z0 / max(crossover_range_m, 1.0)  # rad
        barrel_aim_angles.append((psi_y, gamma_z))

    # 4. Target Drone Geometry & Vulnerability (when target_type == 'drone')
    aspect_geom = None
    drone_spec = None
    if target_type == "drone":
        drone_specs = get_drone_target_specs()
        drone_spec = drone_specs.get(drone_model, drone_specs.get("commercial_quad_medium", next(iter(drone_specs.values()))))
        aspect_geom = compute_drone_projected_geometry(
            drone_spec,
            aspect_mode=aspect_mode,
            drone_pitch_deg=drone_pitch_deg,
            drone_yaw_deg=drone_yaw_deg,
            drone_roll_deg=drone_roll_deg,
        )

    # 5. Fire Burst of Rounds
    sigma_shot = (inherent_dispersion_mrad / 1000.0) * target_range_m  # meters on target plane

    round_records: list[dict[str, Any]] = []
    barrel_hits: dict[int, int] = {i: 0 for i in range(n_b)}
    barrel_coords: dict[int, list[tuple[float, float]]] = {i: [] for i in range(n_b)}

    all_y: list[float] = []
    all_z: list[float] = []
    total_hits = 0
    total_lethal_hits = 0
    fuselage_hits = 0
    battery_hits = 0
    rotor_hits = 0
    motor_hits = 0
    rotors_killed: set[str] = set()

    for r_idx in range(burst_rounds):
        # Sequential barrel firing in round-robin order
        b_idx = r_idx % n_b
        y0, z0 = barrel_muzzle_pos[b_idx]
        psi_y, gamma_z = barrel_aim_angles[b_idx]
        bias_y, bias_z = barrel_biases[b_idx]

        # Nominal bore line position at target range R:
        # y_bore = y0 + R * psi_y = y0 * (1 - R / R_cross)
        y_center = y0 + target_range_m * psi_y + target_range_m * bias_y
        z_center = z0 + target_range_m * gamma_z + target_range_m * bias_z

        # Rotor spin deflection impulse (for Gatling)
        if system_type == "rotary_gatling":
            # Tangential direction: (-sin(theta), cos(theta))
            theta_b = 2.0 * math.pi * b_idx / n_b
            tangential_scale = 0.00015 * (cyclic_rate_rpm / 3000.0) * target_range_m
            y_center += -math.sin(theta_b) * tangential_scale
            z_center += math.cos(theta_b) * tangential_scale

        # Thermal walk drift across the burst (progressive heating)
        thermal_factor = (r_idx / 100.0) * (thermal_drift_mrad_per_100rd / 1000.0) * target_range_m
        y_thermal = thermal_factor * 0.4
        z_thermal = thermal_factor * 0.9  # vertical heat expansion

        # Random shot-to-shot dispersion (ammunition & crown dynamics)
        disp_y = rng.normal(0.0, sigma_shot)
        disp_z = rng.normal(0.0, sigma_shot)

        y_impact = y_center + y_thermal + disp_y
        z_impact = z_center + z_thermal + disp_z

        component_name = "miss"
        is_lethal = False

        if target_type == "drone" and drone_spec is not None and aspect_geom is not None:
            hit_info = evaluate_drone_impact(
                y_impact,
                z_impact,
                drone_spec,
                striking_energy_j=bullet_energy_j,
                aspect_geometry=aspect_geom,
            )
            is_hit = hit_info["is_hit"]
            is_lethal = hit_info["is_lethal"]
            component_name = hit_info["component"]

            if is_hit:
                total_hits += 1
                barrel_hits[b_idx] += 1
                if is_lethal:
                    total_lethal_hits += 1
                if component_name == "fuselage_core":
                    fuselage_hits += 1
                elif component_name == "battery_payload":
                    battery_hits += 1
                elif component_name.startswith("rotor_"):
                    rotor_hits += 1
                    rotors_killed.add(component_name)
                elif component_name.startswith("motor_"):
                    motor_hits += 1
                    rotors_killed.add(component_name.replace("motor_", "rotor_"))
        else:
            is_hit = check_target_hit(y_impact, z_impact, target_type=target_type)
            if is_hit:
                total_hits += 1
                barrel_hits[b_idx] += 1

        all_y.append(y_impact)
        all_z.append(z_impact)
        barrel_coords[b_idx].append((y_impact, z_impact))

        if len(round_records) < 120:
            rec: dict[str, Any] = {
                "round_index": r_idx + 1,
                "barrel_index": b_idx + 1,
                "y_impact_m": round(y_impact, 4),
                "z_impact_m": round(z_impact, 4),
                "radial_dist_m": round(math.sqrt(y_impact**2 + z_impact**2), 4),
                "hit_target": bool(is_hit),
            }
            if target_type == "drone":
                rec["component"] = component_name
                rec["is_lethal"] = bool(is_lethal)
                rec["striking_energy_j"] = round(bullet_energy_j, 1)
            round_records.append(rec)

    # 6. Statistical Diagnostics per Barrel & Combined
    y_arr = np.array(all_y)
    z_arr = np.array(all_z)
    r_arr = np.sqrt(y_arr**2 + z_arr**2)

    # Salvo Combined CEP
    sorted_r = np.sort(r_arr)
    salvo_cep_m = float(sorted_r[int(0.50 * (len(sorted_r) - 1))])
    salvo_r90_m = float(sorted_r[int(0.90 * (len(sorted_r) - 1))])

    # Per-barrel cluster centroids & individual CEPs
    barrel_stats: list[dict[str, Any]] = []
    for b_idx in range(n_b):
        pts = barrel_coords[b_idx]
        if pts:
            by = np.array([p[0] for p in pts])
            bz = np.array([p[1] for p in pts])
            b_centroid_y = float(np.mean(by))
            b_centroid_z = float(np.mean(bz))
            b_r_from_centroid = np.sqrt((by - b_centroid_y) ** 2 + (bz - b_centroid_z) ** 2)
            b_cep = float(np.median(b_r_from_centroid)) if len(b_r_from_centroid) > 0 else 0.0
            b_hits = barrel_hits[b_idx]
            b_rounds = len(pts)
            barrel_stats.append(
                {
                    "barrel_index": b_idx + 1,
                    "muzzle_y_m": round(barrel_muzzle_pos[b_idx][0], 4),
                    "muzzle_z_m": round(barrel_muzzle_pos[b_idx][1], 4),
                    "centroid_y_m": round(b_centroid_y, 4),
                    "centroid_z_m": round(b_centroid_z, 4),
                    "cep_m": round(b_cep, 4),
                    "rounds_fired": b_rounds,
                    "hits_count": b_hits,
                    "hit_pct": round((b_hits / max(1, b_rounds)) * 100.0, 1),
                }
            )

    # Geometric Crossover Ratio
    crossover_ratio = target_range_m / max(crossover_range_m, 1.0)
    crossover_status = (
        "Converging (Pre-Crossover)"
        if crossover_ratio < 0.95
        else ("Exact Crossover Focus" if crossover_ratio <= 1.05 else "Diverging (Post-Crossover)")
    )

    # Drone Kill Evaluation
    drone_killed = False
    kill_prob_pct = 0.0
    if target_type == "drone" and drone_spec is not None:
        crit_hits_req = int(drone_spec.get("critical_hits_to_kill", 2))
        drone_type_name = drone_spec.get("drone_type", "quadcopter")
        req_rotors = 1 if drone_type_name == "fixed_wing" else (2 if drone_type_name == "quadcopter" else 3)
        drone_killed = (
            (total_lethal_hits >= crit_hits_req)
            or (fuselage_hits >= 1)
            or (battery_hits >= 1)
            or (len(rotors_killed) >= req_rotors)
        )
        kill_prob_pct = 100.0 if drone_killed else min(95.0, round((total_lethal_hits / max(1, crit_hits_req)) * 100.0, 1))

    return {
        "system_type": system_type,
        "n_barrels": n_b,
        "burst_rounds": burst_rounds,
        "target_range_m": target_range_m,
        "crossover_range_m": crossover_range_m,
        "crossover_ratio": round(crossover_ratio, 3),
        "crossover_status": crossover_status,
        "cyclic_rate_rpm": cyclic_rate_rpm,
        "target_type": target_type,
        "salvo_cep_m": round(salvo_cep_m, 3),
        "salvo_r90_m": round(salvo_r90_m, 3),
        "total_hits": total_hits,
        "salvo_hit_pct": round((total_hits / max(1, burst_rounds)) * 100.0, 1),
        "barrel_muzzle_positions": barrel_muzzle_pos,
        "barrel_stats": barrel_stats,
        "round_records": round_records,
        "all_impacts": {
            "y_m": [round(float(v), 4) for v in y_arr],
            "z_m": [round(float(v), 4) for v in z_arr],
            "barrel_indices": [(i % n_b) + 1 for i in range(burst_rounds)],
        },
        # Drone aspect & cross-section attributes
        "aspect_geometry": aspect_geom,
        "drone_model": drone_model if target_type == "drone" else None,
        "drone_spec": drone_spec,
        "aspect_mode": aspect_mode if target_type == "drone" else None,
        "presented_area_cm2": aspect_geom.get("presented_area_cm2", 0.0) if aspect_geom else None,
        "cross_section_ratio_pct": aspect_geom.get("cross_section_ratio_pct", 100.0) if aspect_geom else None,
        "planform_ratio_pct": aspect_geom.get("cross_section_ratio_pct", 100.0) if aspect_geom else None,
        "total_lethal_hits": total_lethal_hits,
        "fuselage_hits": fuselage_hits,
        "battery_hits": battery_hits,
        "rotor_hits": rotor_hits,
        "motor_hits": motor_hits,
        "rotors_killed_count": len(rotors_killed),
        "is_drone_killed": bool(drone_killed),
        "drone_kill_probability_pct": kill_prob_pct,
    }


def evaluate_shotgun_dispersion_vs_distance(
    pellets_per_shot: int = 9,
    pellet_diameter_mm: float = 8.38,
    pellet_mass_g: float = 3.49,
    v0_mps: float = 405.0,
    choke_name: str = "Modified",
    distances_m: list[float] | None = None,
    target_type: str = "30_inch_circle",
    moving_target_velocity_mps: float = 0.0,
    moving_target_crossing_angle_deg: float = 90.0,
    lead_compensation_pct: float = 100.0,
    crosswind_mps: float = 0.0,
    aim_error_mrad: float = 0.0,
    n_shots_per_dist: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate shotgun pattern dispersion, containment, and energy across multiple distances."""
    if distances_m is None:
        distances_m = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    range_records: list[dict[str, Any]] = []
    station_snapshots: dict[str, dict[str, Any]] = {}

    for d_idx, dist in enumerate(distances_m):
        sim = simulate_shotgun_pattern_mc(
            n_shots=n_shots_per_dist,
            pellets_per_shot=pellets_per_shot,
            pellet_diameter_mm=pellet_diameter_mm,
            pellet_mass_g=pellet_mass_g,
            v0_mps=v0_mps,
            choke_name=choke_name,
            target_distance_m=float(dist),
            target_type=target_type,
            moving_target_velocity_mps=moving_target_velocity_mps,
            moving_target_crossing_angle_deg=moving_target_crossing_angle_deg,
            lead_compensation_pct=lead_compensation_pct,
            crosswind_mps=crosswind_mps,
            aim_error_mrad=aim_error_mrad,
            seed=seed + d_idx * 17,
        )
        range_records.append(
            {
                "distance_m": round(float(dist), 1),
                "hits_30in_pct": sim["hits_30in_pct"],
                "hits_10in_pct": sim["hits_10in_pct"],
                "hits_target_pct": sim["hits_target_pct"],
                "cep_50_m": sim["cep_50_m"],
                "r_90_m": sim["r_90_m"],
                "pattern_diameter_90_m": round(2.0 * sim["r_90_m"], 3),
                "mean_v_impact_mps": sim["mean_v_impact_mps"],
                "mean_pellet_energy_j": sim["mean_pellet_energy_j"],
                "total_energy_delivered_j": sim["total_energy_delivered_j"],
                "shot_string_length_m": sim["mean_shot_string_length_m"],
            }
        )
        station_snapshots[f"{dist:.1f}m"] = sim

    return {
        "choke_name": choke_name,
        "pellets_per_shot": pellets_per_shot,
        "pellet_diameter_mm": pellet_diameter_mm,
        "pellet_mass_g": pellet_mass_g,
        "v0_mps": v0_mps,
        "distances_m": [float(d) for d in distances_m],
        "range_records": range_records,
        "station_snapshots": station_snapshots,
    }


def evaluate_multibarrel_dispersion_vs_distance(
    system_type: str = "sxs_double",
    n_barrels: int = 2,
    bore_spacing_y_mm: float = 28.0,
    bore_spacing_z_mm: float = 0.0,
    rotor_radius_mm: float = 45.0,
    crossover_range_m: float = 35.0,
    distances_m: list[float] | None = None,
    burst_rounds: int = 36,
    cyclic_rate_rpm: float = 3000.0,
    inherent_dispersion_mrad: float = 2.5,
    barrel_pointing_bias_mrad: float = 1.0,
    thermal_drift_mrad_per_100rd: float = 0.3,
    target_type: str = "e_type_silhouette",
    drone_model: str = "commercial_quad_medium",
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = 15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    bullet_energy_j: float = 1800.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate multi-barrel / Gatling convergence and dispersion evolution across multiple distances."""
    if distances_m is None:
        rc = max(10.0, crossover_range_m)
        distances_m = [
            round(0.25 * rc, 1),
            round(0.50 * rc, 1),
            round(0.75 * rc, 1),
            round(1.00 * rc, 1),
            round(1.25 * rc, 1),
            round(1.50 * rc, 1),
            round(2.00 * rc, 1),
        ]

    range_records: list[dict[str, Any]] = []
    station_snapshots: dict[str, dict[str, Any]] = {}

    if system_type == "sxs_double":
        d0_m = bore_spacing_y_mm / 1000.0
    elif system_type == "ou_double":
        d0_m = bore_spacing_z_mm / 1000.0
    elif system_type == "rotary_gatling":
        d0_m = 2.0 * (rotor_radius_mm / 1000.0)
    elif system_type == "quad_mount":
        d0_m = math.sqrt((bore_spacing_y_mm / 1000.0) ** 2 + (bore_spacing_z_mm / 1000.0) ** 2)
    else:
        d0_m = 2.0 * (rotor_radius_mm / 1000.0)

    for d_idx, dist in enumerate(distances_m):
        sim = simulate_multibarrel_burst_mc(
            system_type=system_type,
            n_barrels=n_barrels,
            bore_spacing_y_mm=bore_spacing_y_mm,
            bore_spacing_z_mm=bore_spacing_z_mm,
            rotor_radius_mm=rotor_radius_mm,
            crossover_range_m=crossover_range_m,
            burst_rounds=burst_rounds,
            target_range_m=float(dist),
            cyclic_rate_rpm=cyclic_rate_rpm,
            inherent_dispersion_mrad=inherent_dispersion_mrad,
            barrel_pointing_bias_mrad=barrel_pointing_bias_mrad,
            thermal_drift_mrad_per_100rd=thermal_drift_mrad_per_100rd,
            target_type=target_type,
            drone_model=drone_model,
            aspect_mode=aspect_mode,
            drone_pitch_deg=drone_pitch_deg,
            drone_yaw_deg=drone_yaw_deg,
            drone_roll_deg=drone_roll_deg,
            bullet_energy_j=bullet_energy_j,
            seed=seed + d_idx * 19,
        )

        inter_spacing_m = d0_m * abs(1.0 - float(dist) / max(crossover_range_m, 1.0))

        range_records.append(
            {
                "distance_m": round(float(dist), 1),
                "crossover_ratio": sim["crossover_ratio"],
                "crossover_status": sim["crossover_status"],
                "inter_barrel_spacing_m": round(inter_spacing_m, 4),
                "salvo_cep_m": sim["salvo_cep_m"],
                "salvo_r90_m": sim["salvo_r90_m"],
                "salvo_hit_pct": sim["salvo_hit_pct"],
                "total_hits": sim["total_hits"],
                "total_lethal_hits": sim.get("total_lethal_hits", 0),
                "is_drone_killed": sim.get("is_drone_killed", False),
            }
        )
        station_snapshots[f"{dist:.1f}m"] = sim

    return {
        "system_type": system_type,
        "n_barrels": n_barrels,
        "crossover_range_m": crossover_range_m,
        "distances_m": [float(d) for d in distances_m],
        "range_records": range_records,
        "station_snapshots": station_snapshots,
    }


def benchmark_multibarrel_aspect_angles_mc(
    system_type: str = "rotary_gatling",
    n_barrels: int = 6,
    bore_spacing_y_mm: float = 28.0,
    bore_spacing_z_mm: float = 0.0,
    rotor_radius_mm: float = 45.0,
    crossover_range_m: float = 400.0,
    burst_rounds: int = 40,
    target_range_m: float = 400.0,
    cyclic_rate_rpm: float = 3000.0,
    inherent_dispersion_mrad: float = 2.0,
    barrel_pointing_bias_mrad: float = 1.0,
    thermal_drift_mrad_per_100rd: float = 0.3,
    drone_model: str = "commercial_quad_medium",
    bullet_energy_j: float = 1800.0,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark multi-barrel / Gatling salvo hit effectiveness across all 6 standard 3D target drone aspect angles.

    Evaluated aspect angles:
    - Head-on Frontal (Ingressing)
    - Flank Side (Crossing at 90°)
    - Oblique Quartering (45°)
    - Tail Rear (Egressing)
    - Planform Overhead (Top-down)
    - Planform Underbelly (Ascending)
    """
    aspect_modes = [
        "head_on_frontal",
        "flank_side",
        "oblique_quartering",
        "tail_rear",
        "planform_overhead",
        "planform_underbelly",
    ]

    aspect_results: dict[str, dict[str, Any]] = {}
    comparison_table: list[dict[str, Any]] = []

    for idx, mode in enumerate(aspect_modes):
        preset = DRONE_ASPECT_PRESETS.get(mode, DRONE_ASPECT_PRESETS["head_on_frontal"])
        sim_res = simulate_multibarrel_burst_mc(
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
            target_type="drone",
            drone_model=drone_model,
            aspect_mode=mode,
            drone_pitch_deg=preset["pitch_deg"],
            drone_yaw_deg=preset["yaw_deg"],
            drone_roll_deg=preset["roll_deg"],
            bullet_energy_j=bullet_energy_j,
            seed=seed + idx * 101,
        )

        aspect_results[mode] = sim_res

        row = {
            "aspect_mode": mode,
            "label": preset.get("name", preset.get("title", mode)),
            "presented_area_cm2": sim_res["presented_area_cm2"],
            "cross_section_ratio_pct": sim_res["cross_section_ratio_pct"],
            "salvo_hit_pct": sim_res["salvo_hit_pct"],
            "total_hits": sim_res["total_hits"],
            "total_lethal_hits": sim_res["total_lethal_hits"],
            "is_drone_killed": sim_res["is_drone_killed"],
            "salvo_cep_m": sim_res["salvo_cep_m"],
            "pitch_deg": preset["pitch_deg"],
            "yaw_deg": preset["yaw_deg"],
            "roll_deg": preset["roll_deg"],
        }
        comparison_table.append(row)

    # Sort to find best (highest presented area / hit rate) and worst
    sorted_by_area = sorted(comparison_table, key=lambda x: x["presented_area_cm2"], reverse=True)
    best_aspect = sorted_by_area[0]
    worst_aspect = sorted_by_area[-1]

    return {
        "drone_model": drone_model,
        "system_type": system_type,
        "target_range_m": target_range_m,
        "crossover_range_m": crossover_range_m,
        "burst_rounds": burst_rounds,
        "aspect_results": aspect_results,
        "comparison_table": comparison_table,
        "max_presented_area_cm2": best_aspect["presented_area_cm2"],
        "min_presented_area_cm2": worst_aspect["presented_area_cm2"],
        "area_reduction_factor": round(
            best_aspect["presented_area_cm2"] / max(worst_aspect["presented_area_cm2"], 1e-3), 2
        ),
        "best_aspect": best_aspect["label"],
        "worst_aspect": worst_aspect["label"],
    }


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


def compare_barrel_lengths_dispersion(
    barrel_lengths_in: list[float] | None = None,
    pellets_per_shot: int = 9,
    pellet_diameter_mm: float = 8.38,
    pellet_mass_g: float = 3.49,
    base_v0_mps: float = 405.0,
    reference_length_in: float = 28.0,
    choke_name: str = "Modified",
    target_distance_m: float = 36.58,
    target_type: str = "30_inch_circle",
    n_shots: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare shotgun pattern spread, muzzle velocity, and energy delivered across multiple barrel lengths simultaneously.

    Physics model:
    - Muzzle velocity scaling: delta V0 ~ 2.2 m/s per inch deviation from reference length (28 inches)
    - Muzzle gas wash / wad tipping penalty: for short barrels (< 24 inches), higher exit gas pressure
      induces aerodynamic wad disruption and increases radial spread half-angle.
    """
    if barrel_lengths_in is None:
        barrel_lengths_in = [14.0, 18.5, 24.0, 28.0, 32.0]

    barrel_records: list[dict[str, Any]] = []
    barrel_snapshots: dict[str, dict[str, Any]] = {}

    for b_idx, l_in in enumerate(barrel_lengths_in):
        # Velocity scaling: ~2.2 m/s per inch
        v0_scaled = max(280.0, base_v0_mps + (float(l_in) - reference_length_in) * 2.2)

        # Gas wash factor for short barrels (< 24 inches)
        wash_factor = 1.0 + 0.35 * (max(0.0, (24.0 - float(l_in)) / 24.0) ** 1.5)
        wash_aim_mrad = (wash_factor - 1.0) * 8.0

        sim = simulate_shotgun_pattern_mc(
            n_shots=n_shots,
            pellets_per_shot=pellets_per_shot,
            pellet_diameter_mm=pellet_diameter_mm,
            pellet_mass_g=pellet_mass_g,
            v0_mps=v0_scaled,
            choke_name=choke_name,
            target_distance_m=target_distance_m,
            target_type=target_type,
            aim_error_mrad=wash_aim_mrad,
            seed=seed + b_idx * 23,
        )

        b_key = f"{float(l_in):.1f}in"
        barrel_records.append(
            {
                "length_in": float(l_in),
                "length_cm": round(float(l_in) * 2.54, 1),
                "v0_mps": round(v0_scaled, 1),
                "gas_wash_factor": round(wash_factor, 3),
                "hits_30in_pct": sim["hits_30in_pct"],
                "hits_10in_pct": sim["hits_10in_pct"],
                "hits_target_pct": sim["hits_target_pct"],
                "cep_50_m": sim["cep_50_m"],
                "r_90_m": sim["r_90_m"],
                "pattern_diameter_90_m": round(2.0 * sim["r_90_m"], 3),
                "mean_v_impact_mps": sim["mean_v_impact_mps"],
                "mean_pellet_energy_j": sim["mean_pellet_energy_j"],
                "total_energy_delivered_j": sim["total_energy_delivered_j"],
                "shot_string_length_m": sim["mean_shot_string_length_m"],
            }
        )
        barrel_snapshots[b_key] = sim

    return {
        "choke_name": choke_name,
        "target_distance_m": target_distance_m,
        "target_type": target_type,
        "reference_length_in": reference_length_in,
        "barrel_lengths_in": [float(l) for l in barrel_lengths_in],
        "barrel_records": barrel_records,
        "barrel_snapshots": barrel_snapshots,
    }


# -------------------------------------------------------------------------
# ANTI-DRONE MULTI-SHOTGUN C-UAS SYSTEM (FLEXIBLE ARRAY GEOMETRY & MONTE CARLO)
# -------------------------------------------------------------------------

def generate_multishotgun_array_layout(
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    custom_specs: list[dict[str, Any]] | None = None,
    default_ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
) -> list[dict[str, Any]]:
    """Generate the 2D mounting coordinates (y, z) and attributes for multiple shotgun barrels.

    Supports homogeneous or heterogeneous Core (inner) + Perimeter (outer) ammunition/choke mixes.

    Coordinates on turret face (X=0):
      y_m: horizontal crossrange position (m), positive = right
      z_m: vertical height position (m), positive = up
    """
    barrels: list[dict[str, Any]] = []

    if layout_type == "circular":
        # N barrels evenly spaced around a circle of radius mount_radius_m
        for i in range(max(1, n_barrels)):
            angle = (2.0 * math.pi * i) / n_barrels
            # Start at top (90 deg)
            phi = angle + math.pi / 2.0
            y_i = round(mount_radius_m * math.cos(phi), 4)
            z_i = round(mount_radius_m * math.sin(phi), 4)
            deg = int(round(math.degrees(phi))) % 360
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Ring {deg}°)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "concentric":
        # Dual-ring: center core + outer ring
        if n_barrels <= 3:
            inner_count = 1
        elif n_barrels <= 6:
            inner_count = 1
        elif n_barrels <= 9:
            inner_count = 3
        else:
            inner_count = max(3, n_barrels // 3)
        outer_count = max(1, n_barrels - inner_count)

        # Inner ring / center core
        r_inner = mount_radius_m * 0.45
        for i in range(inner_count):
            if inner_count == 1:
                y_i, z_i = 0.0, 0.0
                lbl = "Barrel 1 (Center Core)"
            else:
                angle = (2.0 * math.pi * i) / inner_count + math.pi / 2.0
                y_i = round(r_inner * math.cos(angle), 4)
                z_i = round(r_inner * math.sin(angle), 4)
                lbl = f"Barrel {i+1} (Inner Ring)"
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": lbl,
                "y_m": y_i,
                "z_m": z_i,
                "choke": "Full" if default_choke != "Extra Full" else default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "inner_ring",
            })

        # Outer ring
        for i in range(outer_count):
            angle = (2.0 * math.pi * i) / outer_count + math.pi / 2.0
            y_i = round(mount_radius_m * math.cos(angle), 4)
            z_i = round(mount_radius_m * math.sin(angle), 4)
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": f"Barrel {len(barrels)+1} (Outer Ring)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "outer_ring",
            })

    elif layout_type == "grid":
        # Matrix grid: n_rows x n_cols
        b_idx = 1
        for r in range(n_rows):
            z_i = round((r - (n_rows - 1) / 2.0) * spacing_z_m, 4)
            for c in range(n_cols):
                y_i = round((c - (n_cols - 1) / 2.0) * spacing_y_m, 4)
                barrels.append({
                    "barrel_id": f"B{b_idx}",
                    "label": f"Barrel {b_idx} (Row {r+1}, Col {c+1})",
                    "y_m": y_i,
                    "z_m": z_i,
                    "choke": default_choke,
                    "barrel_length_in": float(default_barrel_length_in),
                    "mount_group": "grid_matrix",
                })
                b_idx += 1

    elif layout_type == "fan_horizontal":
        # Linear horizontal row of N barrels
        for i in range(max(1, n_barrels)):
            y_i = round((i - (n_barrels - 1) / 2.0) * spacing_y_m, 4)
            z_i = 0.0
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Pos {i+1} H-Fan)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "fan_horizontal",
            })

    elif layout_type == "fan_vertical":
        # Linear vertical stack of N barrels
        for i in range(max(1, n_barrels)):
            y_i = 0.0
            z_i = round((i - (n_barrels - 1) / 2.0) * spacing_z_m, 4)
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Pos {i+1} V-Stack)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "fan_vertical",
            })

    elif layout_type == "honeycomb":
        # Honeycomb: 1 center + perimeter hex at mount_radius_m
        barrels.append({
            "barrel_id": "B1",
            "label": "Barrel 1 (Center Core)",
            "y_m": 0.0,
            "z_m": 0.0,
            "choke": "Full",
            "barrel_length_in": float(default_barrel_length_in),
            "mount_group": "honeycomb_center",
        })
        hex_count = max(6, n_barrels - 1)
        for i in range(hex_count):
            angle = (2.0 * math.pi * i) / hex_count
            y_i = round(mount_radius_m * math.cos(angle), 4)
            z_i = round(mount_radius_m * math.sin(angle), 4)
            barrels.append({
                "barrel_id": f"B{len(barrels)+1}",
                "label": f"Barrel {len(barrels)+1} (Hex Petal {i+1})",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "honeycomb_outer",
            })

    elif layout_type == "v_shape":
        # V-shape Chevron: apex in center, wings angled up/back
        for i in range(max(1, n_barrels)):
            t = (i - (n_barrels - 1) / 2.0)
            y_i = round(t * spacing_y_m, 4)
            z_i = round(abs(t) * spacing_z_m, 4)
            barrels.append({
                "barrel_id": f"B{i+1}",
                "label": f"Barrel {i+1} (Chevron V)",
                "y_m": y_i,
                "z_m": z_i,
                "choke": default_choke,
                "barrel_length_in": float(default_barrel_length_in),
                "mount_group": "v_chevron",
            })

    elif layout_type == "custom":
        if custom_specs:
            for idx, spec in enumerate(custom_specs):
                b_custom: dict[str, Any] = {
                    "barrel_id": spec.get("barrel_id", f"B{idx+1}"),
                    "label": spec.get("label", f"Custom Barrel {idx+1}"),
                    "y_m": float(spec.get("y_m", 0.0)),
                    "z_m": float(spec.get("z_m", 0.0)),
                    "choke": str(spec.get("choke", default_choke)),
                    "barrel_length_in": float(spec.get("barrel_length_in", default_barrel_length_in)),
                    "mount_group": "custom",
                }
                if "ammunition_preset" in spec:
                    b_custom["ammunition_preset"] = str(spec["ammunition_preset"])
                barrels.append(b_custom)
        else:
            # Default 4-quad mount
            barrels = [
                {"barrel_id": "B1", "label": "B1 (Top-Left)", "y_m": -0.12, "z_m": 0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B2", "label": "B2 (Top-Right)", "y_m": 0.12, "z_m": 0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B3", "label": "B3 (Bottom-Left)", "y_m": -0.12, "z_m": -0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
                {"barrel_id": "B4", "label": "B4 (Bottom-Right)", "y_m": 0.12, "z_m": -0.12, "choke": default_choke, "barrel_length_in": default_barrel_length_in, "mount_group": "custom"},
            ]

    # Assign homogeneous or heterogeneous Core + Perimeter ammunition & choke
    max_r = max((math.hypot(float(b["y_m"]), float(b["z_m"])) for b in barrels), default=0.0)
    eff_outer_ammo = outer_ammunition_preset or default_ammunition_preset
    eff_inner_ammo = inner_ammunition_preset or default_ammunition_preset
    any_center = any(
        b.get("mount_group") in ("inner_ring", "honeycomb_center")
        or (max_r > 0.04 and math.hypot(float(b["y_m"]), float(b["z_m"])) <= max_r * 0.55)
        for b in barrels
    )
    for idx, b in enumerate(barrels):
        r_b = math.hypot(float(b["y_m"]), float(b["z_m"]))
        if any_center:
            is_inner = b.get("mount_group") in ("inner_ring", "honeycomb_center") or (
                max_r > 0.04 and r_b <= max_r * 0.55 and b.get("mount_group") != "outer_ring"
            )
        else:
            # All barrels sit on a single perimeter (e.g. hollow circular ring): alternate symmetrically if mixed
            is_inner = (idx % 2 == 0) if inner_ammunition_preset is not None else False
        if "ammunition_preset" not in b:
            b["ammunition_preset"] = eff_inner_ammo if is_inner else eff_outer_ammo
        if is_inner and inner_choke is not None and b.get("mount_group") != "custom":
            b["choke"] = inner_choke

    return barrels


def compute_balanced_pair_fire_times(
    barrels: list[dict[str, Any]],
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
) -> list[float]:
    """Compute per-barrel firing timestamps (seconds) supporting simultaneous, sequential, or symmetric pair ripple."""
    n = len(barrels)
    if n == 0 or firing_mode == "simultaneous_volley":
        return [0.0] * n

    dt_s = max(float(ripple_delay_ms), 0.0) / 1000.0
    if firing_mode == "sequential_ripple":
        return [round(i * dt_s, 6) for i in range(n)]

    # "balanced_pair_ripple": pair opposite barrels so net yaw/pitch gimbal moment cancels
    fire_times = [0.0] * n
    unassigned = list(range(n))
    step_idx = 0

    # Fire central core barrels (r < 0.03 m) at step 0 since they exert near-zero moment arm
    center_indices = [i for i in unassigned if math.hypot(float(barrels[i]["y_m"]), float(barrels[i]["z_m"])) < 0.03]
    if center_indices:
        for ci in center_indices:
            fire_times[ci] = 0.0
            unassigned.remove(ci)
        step_idx = 1

    # Greedily match each remaining barrel with its most opposite unassigned partner
    while unassigned:
        i = unassigned.pop(0)
        fire_times[i] = round(step_idx * dt_s, 6)
        if unassigned:
            yi, zi = float(barrels[i]["y_m"]), float(barrels[i]["z_m"])
            best_j = min(
                unassigned,
                key=lambda j: math.hypot(yi + float(barrels[j]["y_m"]), zi + float(barrels[j]["z_m"])),
            )
            unassigned.remove(best_j)
            fire_times[best_j] = round(step_idx * dt_s, 6)
        step_idx += 1

    return fire_times


def configure_array_pointing_vectors(
    barrels: list[dict[str, Any]],
    pointing_mode: str = "convergent",
    intercept_range_m: float = 35.0,
    splay_angle_mrad: float = 0.0,
    tiling_target_radius_m: float = 0.50,
    custom_angles: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Compute the pointing yaw and pitch aiming vectors for each barrel in the multi-shotgun array.

    Angles in milliradians (mrad):
      aim_yaw_mrad: horizontal deflection (+ = right)
      aim_pitch_mrad: vertical deflection (+ = up)
    """
    R = max(intercept_range_m, 1.0)
    configured: list[dict[str, Any]] = []

    for idx, b in enumerate(barrels):
        b_copy = dict(b)
        y0 = float(b["y_m"])
        z0 = float(b["z_m"])

        if pointing_mode == "convergent":
            # All barrels angle inward so nominal centerlines intersect at (0, 0) at range R
            yaw_mrad = -(y0 / R) * 1000.0
            pitch_mrad = -(z0 / R) * 1000.0

        elif pointing_mode == "parallel":
            # Parallel bore axes: no angular convergence
            yaw_mrad = 0.0
            pitch_mrad = 0.0

        elif pointing_mode == "divergent_curtain":
            # Barrels angle outward from cluster centroid by splay_angle_mrad to maximize intercept net
            r_dist = math.hypot(y0, z0)
            if r_dist > 1e-4:
                u_y = y0 / r_dist
                u_z = z0 / r_dist
            else:
                u_y, u_z = 0.0, 0.0
            yaw_mrad = -(y0 / R) * 1000.0 + (splay_angle_mrad * u_y)
            pitch_mrad = -(z0 / R) * 1000.0 + (splay_angle_mrad * u_z)

        elif pointing_mode == "staggered_tiling":
            # Tiles pattern centers across a circle of radius tiling_target_radius_m
            n_tot = len(barrels)
            r_dist = math.hypot(y0, z0)
            if n_tot == 1 or r_dist < 0.03:
                y_tgt, z_tgt = 0.0, 0.0
            else:
                angle = math.atan2(z0, y0)
                y_tgt = tiling_target_radius_m * math.cos(angle)
                z_tgt = tiling_target_radius_m * math.sin(angle)
            yaw_mrad = ((y_tgt - y0) / R) * 1000.0
            pitch_mrad = ((z_tgt - z0) / R) * 1000.0

        elif pointing_mode == "custom":
            if custom_angles and idx < len(custom_angles):
                yaw_mrad, pitch_mrad = custom_angles[idx]
            else:
                yaw_mrad, pitch_mrad = 0.0, 0.0
        else:
            yaw_mrad, pitch_mrad = -(y0 / R) * 1000.0, -(z0 / R) * 1000.0

        b_copy["aim_yaw_mrad"] = round(yaw_mrad, 3)
        b_copy["aim_pitch_mrad"] = round(pitch_mrad, 3)
        b_copy["focal_range_m"] = R
        configured.append(b_copy)

    return configured


def evaluate_drone_impact(
    y_rel_m: float,
    z_rel_m: float,
    drone_spec: dict[str, Any],
    striking_energy_j: float,
    aspect_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether an impact point (y, z) hits a drone component and whether it is lethal.

    Supports both:
      1. 3D Projected Cross-Section (when aspect_geometry is provided)
      2. Planform Top-Down / Overhead fallback (when aspect_geometry is None)
    """
    e_rotor_lethal = float(drone_spec.get("lethal_energy_rotor_j", 8.0))
    e_core_lethal = float(drone_spec.get("lethal_energy_core_j", 20.0))
    e_arm_lethal = float(drone_spec.get("lethal_energy_arm_j", 30.0))

    if aspect_geometry is not None:
        # 1. Fuselage 3D Box Projected Polygon (Flight Controller, Avionics, IMU)
        fuse_poly = aspect_geometry.get("fuselage_polygon")
        if fuse_poly and len(fuse_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(fuse_poly)):
            return {
                "is_hit": True,
                "component": "fuselage_core",
                "is_lethal": bool(striking_energy_j >= e_core_lethal),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

        # 2. Battery / Munition Payload Box Projected Polygon
        batt_poly = aspect_geometry.get("battery_polygon")
        if batt_poly and len(batt_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(batt_poly)):
            return {
                "is_hit": True,
                "component": "battery_payload",
                "is_lethal": bool(striking_energy_j >= e_core_lethal * 0.75),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

        # 3. Motor Pods (Direct hub motor strike)
        motors = aspect_geometry.get("motors", [])
        for m in motors:
            my = float(m["y_m"])
            mz = float(m["z_m"])
            mw = float(m["w_m"])
            mh = float(m["h_m"])
            if abs(y_rel_m - my) <= mw / 2.0 and abs(z_rel_m - mz) <= mh / 2.0:
                return {
                    "is_hit": True,
                    "component": f"motor_{m.get('id', 'M')}",
                    "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                    "critical_system": True,
                    "striking_energy_j": striking_energy_j,
                }

        # 4. Spinning Rotor Discs (Projected Ellipses or Edge-on Blade Bands)
        rotors = aspect_geometry.get("rotors", [])
        for r in rotors:
            ry = float(r["y_m"])
            rz = float(r["z_m"])
            a = float(r["semi_major_m"])
            b = float(r["semi_minor_m"])
            u_maj = r.get("u_maj", [1.0, 0.0])
            u_min = r.get("u_min", [0.0, 1.0])
            dy = y_rel_m - ry
            dz = z_rel_m - rz
            u = dy * u_maj[0] + dz * u_maj[1]
            v = dy * u_min[0] + dz * u_min[1]
            if (u / a)**2 + (v / b)**2 <= 1.0:
                return {
                    "is_hit": True,
                    "component": f"rotor_{r.get('id', 'R')}",
                    "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                    "critical_system": True,
                    "striking_energy_j": striking_energy_j,
                }

        # 5. Fixed-Wing Aerodynamic Panels (if fixed_wing)
        wings = aspect_geometry.get("wings", [])
        for w_poly in wings:
            if w_poly and len(w_poly) >= 3 and point_in_convex_polygon(y_rel_m, z_rel_m, np.array(w_poly)):
                return {
                    "is_hit": True,
                    "component": "wing_spar",
                    "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                    "critical_system": False,
                    "striking_energy_j": striking_energy_j,
                }

        # 6. Structural Carbon Arms / Booms
        arms = aspect_geometry.get("arms", [])
        for arm in arms:
            p1 = arm.get("p1", [0.0, 0.0])
            p2 = arm.get("p2", [0.0, 0.0])
            arm_r = float(arm.get("diameter_m", 0.020)) / 2.0
            p1_y, p1_z = float(p1[0]), float(p1[1])
            p2_y, p2_z = float(p2[0]), float(p2[1])
            seg_dy = p2_y - p1_y
            seg_dz = p2_z - p1_z
            seg_len_sq = seg_dy**2 + seg_dz**2
            if seg_len_sq > 1e-6:
                t_p = max(0.0, min(1.0, ((y_rel_m - p1_y) * seg_dy + (z_rel_m - p1_z) * seg_dz) / seg_len_sq))
                proj_y = p1_y + t_p * seg_dy
                proj_z = p1_z + t_p * seg_dz
                dist_to_arm = math.hypot(y_rel_m - proj_y, z_rel_m - proj_z)
                if dist_to_arm <= arm_r:
                    return {
                        "is_hit": True,
                        "component": "structural_arm",
                        "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                        "critical_system": False,
                        "striking_energy_j": striking_energy_j,
                    }

        return {
            "is_hit": False,
            "component": "miss",
            "is_lethal": False,
            "critical_system": False,
            "striking_energy_j": striking_energy_j,
        }

    # Fallback to 2D Planform (Top-down)
    drone_type = drone_spec.get("drone_type", "quadcopter")
    fuse_w = float(drone_spec.get("fuselage_w_m", 0.10))
    fuse_h = float(drone_spec.get("fuselage_h_m", 0.10))

    # 1. Central fuselage check (core avionics, battery, IMU)
    in_fuselage = (abs(y_rel_m) <= fuse_w / 2.0) and (abs(z_rel_m) <= fuse_h / 2.0)
    if in_fuselage:
        return {
            "is_hit": True,
            "component": "fuselage_core",
            "is_lethal": bool(striking_energy_j >= e_core_lethal),
            "critical_system": True,
            "striking_energy_j": striking_energy_j,
        }

    # 2. Rotors check
    rotors = drone_spec.get("rotors", [])
    for r in rotors:
        ry = float(r["y_m"])
        rz = float(r["z_m"])
        rr = float(r["radius_m"])
        d_rotor = math.hypot(y_rel_m - ry, z_rel_m - rz)
        if d_rotor <= rr:
            return {
                "is_hit": True,
                "component": f"rotor_{r.get('id', 'R')}",
                "is_lethal": bool(striking_energy_j >= e_rotor_lethal),
                "critical_system": True,
                "striking_energy_j": striking_energy_j,
            }

    # 3. Fixed wing check (if fixed wing)
    if drone_type == "fixed_wing":
        wingspan = float(drone_spec.get("wingspan_m", 1.60))
        chord = float(drone_spec.get("wing_chord_m", 0.22))
        if abs(y_rel_m) <= wingspan / 2.0 and abs(z_rel_m) <= chord / 2.0:
            return {
                "is_hit": True,
                "component": "wing_spar",
                "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                "critical_system": False,
                "striking_energy_j": striking_energy_j,
            }

    # 4. Structural arms check (quad/octo)
    for r in rotors:
        ry = float(r["y_m"])
        rz = float(r["z_m"])
        arm_len_sq = ry**2 + rz**2
        if arm_len_sq > 1e-4:
            t_proj = max(0.0, min(1.0, (y_rel_m * ry + z_rel_m * rz) / arm_len_sq))
            proj_y = t_proj * ry
            proj_z = t_proj * rz
            dist_to_arm = math.hypot(y_rel_m - proj_y, z_rel_m - proj_z)
            if dist_to_arm <= 0.020:  # 20mm carbon arm tube
                return {
                    "is_hit": True,
                    "component": "structural_arm",
                    "is_lethal": bool(striking_energy_j >= e_arm_lethal),
                    "critical_system": False,
                    "striking_energy_j": striking_energy_j,
                }

    return {
        "is_hit": False,
        "component": "miss",
        "is_lethal": False,
        "critical_system": False,
        "striking_energy_j": striking_energy_j,
    }


def simulate_anti_drone_multishotgun_mc(
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 10.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    pellets_per_shot: int | None = None,
    pellet_diameter_mm: float | None = None,
    pellet_mass_g: float | None = None,
    v0_mps: float | None = None,
    default_choke: str = "Modified",
    default_barrel_length_in: float = 24.0,
    target_distance_m: float = 35.0,
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    drone_heading_deg: float = 90.0,
    aim_error_mrad: float = 1.5,
    crosswind_mps: float = 0.0,
    n_volleys: int = 25,
    seed: int = 42,
    inner_ammunition_preset: str | None = None,
    outer_ammunition_preset: str | None = None,
    inner_choke: str | None = None,
    barrel_pitch_offsets_mrad: list[float] | None = None,
    barrel_yaw_offsets_mrad: list[float] | None = None,
    aspect_mode: str = "head_on_frontal",
    drone_pitch_deg: float = -15.0,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
) -> dict[str, Any]:
    """Execute Monte Carlo simulation of an anti-drone defense volley using a multi-shotgun array.

    Evaluates:
      - Multi-barrel spatial array geometry (circular, grid, fan, honeycomb, concentric, custom)
      - Heterogeneous core vs perimeter ammunition presets and chokes
      - Pointing convergence, parallel bores, or splay curtain + optional dynamic recoil walk offsets
      - Simultaneous volley vs sequential ripple vs balanced symmetric pair ripple fire
      - 3D longitudinal shot stringing vs moving drone kinematics
      - 3D target drone cross-section projection (head-on frontal, flank, oblique, tail, planform)
      - Drone component-level damage (fuselage core, battery payload, motor pods, rotors, arms, wings)
      - Overall drone kill probability P_kill, lethal hit counts, and energy deposition
    """
    rng = np.random.default_rng(seed)

    # 1. Resolve Baseline Nominal Ammunition Parameters
    presets = get_shotgun_presets()

    def _resolve_barrel_ammo(preset_key: str) -> dict[str, Any]:
        if preset_key in presets:
            a = presets[preset_key]
            return {
                "pellets": int(pellets_per_shot or a["pellets_per_shot"]),
                "d_mm": float(pellet_diameter_mm or a["pellet_diameter_mm"]),
                "m_g": float(pellet_mass_g or a["pellet_mass_g"]),
                "v0": float(v0_mps or a["v0_mps"]),
                "effective_cd": float(a.get("effective_cd", 0.47)),
            }
        return {
            "pellets": int(pellets_per_shot or 27),
            "d_mm": float(pellet_diameter_mm or 6.10),
            "m_g": float(pellet_mass_g or 1.34),
            "v0": float(v0_mps or 415.0),
            "effective_cd": 0.47,
        }

    nom_ammo = _resolve_barrel_ammo(ammunition_preset)
    eff_pellets = nom_ammo["pellets"]
    eff_d_mm = nom_ammo["d_mm"]
    eff_m_g = nom_ammo["m_g"]
    eff_v0 = nom_ammo["v0"]

    # 2. Resolve Drone Specs & 3D Projected Cross-Section Geometry
    drone_specs_cat = get_drone_target_specs()
    drone_spec = drone_specs_cat.get(drone_model, drone_specs_cat["commercial_quad_medium"])
    crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 3))

    aspect_geom = compute_drone_projected_geometry(
        drone_spec=drone_spec,
        aspect_mode=aspect_mode,
        drone_pitch_deg=drone_pitch_deg,
        drone_yaw_deg=drone_yaw_deg,
        drone_roll_deg=drone_roll_deg,
    )

    # 3. Generate and Configure Barrel Array
    barrels_raw = generate_multishotgun_array_layout(
        layout_type=layout_type,
        n_barrels=n_barrels,
        mount_radius_m=mount_radius_m,
        spacing_y_m=spacing_y_m,
        spacing_z_m=spacing_z_m,
        n_cols=n_cols,
        n_rows=n_rows,
        default_choke=default_choke,
        default_barrel_length_in=default_barrel_length_in,
        custom_specs=custom_barrel_specs,
        default_ammunition_preset=ammunition_preset,
        inner_ammunition_preset=inner_ammunition_preset,
        outer_ammunition_preset=outer_ammunition_preset,
        inner_choke=inner_choke,
    )
    barrels = configure_array_pointing_vectors(
        barrels=barrels_raw,
        pointing_mode=pointing_mode,
        intercept_range_m=target_distance_m,
        splay_angle_mrad=splay_angle_mrad,
    )
    n_act_barrels = len(barrels)
    fire_times_s = compute_balanced_pair_fire_times(barrels, firing_mode=firing_mode, ripple_delay_ms=ripple_delay_ms)

    # 4. Aerodynamic deceleration constants (true area-to-mass ratio across Lead/Steel/Tungsten/Flechette)
    rho_air = 1.225
    d_m_nom = eff_d_mm / 1000.0
    area_nom = math.pi * 0.25 * (d_m_nom ** 2)
    kappa_nom = (rho_air * nom_ammo["effective_cd"] * area_nom) / (2.0 * (eff_m_g / 1000.0))
    v_impact_nominal = eff_v0 * math.exp(-kappa_nom * target_distance_m)
    t_flight_nominal = target_distance_m / max((eff_v0 + v_impact_nominal) / 2.0, 10.0)

    # Moving target vector
    heading_rad = math.radians(drone_heading_deg)
    v_drone_y = drone_velocity_mps * math.sin(heading_rad)
    v_drone_z = drone_velocity_mps * math.cos(heading_rad)

    # 5. Volley Simulation Loop
    volley_records: list[dict[str, Any]] = []
    kills_count = 0
    all_hits_counts: list[int] = []
    all_lethal_counts: list[int] = []
    all_energy_on_drone: list[float] = []

    # Per-barrel hit accumulator
    barrel_stats_map = {
        b["barrel_id"]: {
            "barrel_id": b["barrel_id"],
            "label": b["label"],
            "y_m": b["y_m"],
            "z_m": b["z_m"],
            "choke": b["choke"],
            "ammunition_preset": b.get("ammunition_preset", ammunition_preset),
            "fire_time_ms": round(fire_times_s[idx] * 1000.0, 2),
            "aim_yaw_mrad": b["aim_yaw_mrad"],
            "aim_pitch_mrad": b["aim_pitch_mrad"],
            "pellets_fired": 0,
            "hits_on_drone": 0,
            "lethal_hits": 0,
            "rotor_hits": 0,
            "fuselage_hits": 0,
        }
        for idx, b in enumerate(barrels)
    }

    sample_scatter: dict[str, list[Any]] = {
        "y": [],
        "z": [],
        "barrel_id": [],
        "is_hit": [],
        "is_lethal": [],
        "component": [],
        "striking_energy_j": [],
        "striking_velocity_mps": [],
        "flight_time_s": [],
    }

    aim_sigma_m = (target_distance_m * (aim_error_mrad / 1000.0)) / 2.0 if aim_error_mrad > 0 else 0.0
    choke_defs = get_choke_definitions()

    barrel_ammo_specs = [_resolve_barrel_ammo(str(b.get("ammunition_preset", ammunition_preset))) for b in barrels]
    tot_pellets_per_volley = sum(spec["pellets"] for spec in barrel_ammo_specs)

    for v_idx in range(n_volleys):
        # Turret pointing jitter for this volley
        aim_err_y = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0
        aim_err_z = rng.normal(0.0, aim_sigma_m) if aim_sigma_m > 0 else 0.0

        v_hits = 0
        v_lethal = 0
        v_energy = 0.0
        v_fuselage_lethal = 0
        v_battery_lethal = 0
        v_rotors_killed: set[str] = set()

        for b_idx, b in enumerate(barrels):
            b_id = b["barrel_id"]
            y0 = float(b["y_m"])
            z0 = float(b["z_m"])
            b_choke = str(b.get("choke", default_choke))
            b_len = float(b.get("barrel_length_in", default_barrel_length_in))
            b_spec = barrel_ammo_specs[b_idx]
            b_pellets = int(b_spec["pellets"])
            b_d_m = float(b_spec["d_mm"]) / 1000.0
            b_m_kg = float(b_spec["m_g"]) / 1000.0
            b_area_m2 = math.pi * 0.25 * (b_d_m ** 2)
            b_cd_base = float(b_spec["effective_cd"])

            # Firing time offset (simultaneous vs sequential ripple vs balanced pair ripple)
            t_fire = fire_times_s[b_idx]

            # Barrel length velocity scaling & gas wash turbulence
            v0_barrel = float(b_spec["v0"]) + 2.2 * (b_len - 28.0)
            wash_turb = 1.0 + 0.35 * (max(0.0, 24.0 - b_len) / 24.0) ** 1.5

            # Choke radial spread parameter
            c_info = choke_defs.get(b_choke, choke_defs["Modified"])
            p_bench = float(c_info["benchmark_40yd_30in_pct"])
            r_bench = 0.381
            sigma_ref = r_bench / math.sqrt(-2.0 * math.log(1.0 - p_bench))
            theta_choke = sigma_ref / 36.58
            sigma_radial = (0.01852 / 4.0 + theta_choke * target_distance_m * (target_distance_m / 36.58) ** 0.15) * wash_turb

            # Optional dynamic recoil muzzle climb / yaw walk offsets
            dyn_yaw = float(barrel_yaw_offsets_mrad[b_idx]) if barrel_yaw_offsets_mrad and b_idx < len(barrel_yaw_offsets_mrad) else 0.0
            dyn_pitch = float(barrel_pitch_offsets_mrad[b_idx]) if barrel_pitch_offsets_mrad and b_idx < len(barrel_pitch_offsets_mrad) else 0.0

            # Nominal bore center at target plane
            y_bore = y0 + target_distance_m * ((b["aim_yaw_mrad"] + dyn_yaw) / 1000.0) + aim_err_y
            z_bore = z0 + target_distance_m * ((b["aim_pitch_mrad"] + dyn_pitch) / 1000.0) + aim_err_z

            barrel_stats_map[b_id]["pellets_fired"] += b_pellets

            # Generate pellets for this barrel
            for _ in range(b_pellets):
                # Radial dispersion
                dy = rng.normal(0.0, sigma_radial) + (crosswind_mps * 0.008 * target_distance_m)
                dz = rng.normal(0.0, sigma_radial)

                y_impact_raw = y_bore + dy
                z_impact_raw = z_bore + dz

                # Pellet drag variations & longitudinal stringing
                cd_pellet = rng.uniform(0.92 * b_cd_base, 1.25 * b_cd_base)
                kappa_p = (rho_air * cd_pellet * b_area_m2) / (2.0 * max(b_m_kg, 1e-6))
                v_impact = v0_barrel * math.exp(-kappa_p * target_distance_m)
                t_flight = target_distance_m / max((v0_barrel + v_impact) / 2.0, 10.0)
                t_arrival = t_fire + t_flight

                # Moving target position relative to nominal lead at arrival time
                dt_target = t_arrival - t_flight_nominal
                tgt_offset_y = v_drone_y * dt_target
                tgt_offset_z = v_drone_z * dt_target

                # Relative impact position on drone target frame
                y_rel = y_impact_raw - tgt_offset_y
                z_rel = z_impact_raw - tgt_offset_z

                ke_pellet = 0.5 * b_m_kg * (v_impact**2)

                # Component hit check
                hit_info = evaluate_drone_impact(y_rel, z_rel, drone_spec, ke_pellet, aspect_geometry=aspect_geom)

                if hit_info["is_hit"]:
                    v_hits += 1
                    barrel_stats_map[b_id]["hits_on_drone"] += 1
                    v_energy += ke_pellet

                    comp = hit_info["component"]
                    if hit_info["is_lethal"]:
                        v_lethal += 1
                        barrel_stats_map[b_id]["lethal_hits"] += 1
                        if comp == "fuselage_core":
                            v_fuselage_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                        elif comp == "battery_payload":
                            v_battery_lethal += 1
                            barrel_stats_map[b_id]["fuselage_hits"] += 1
                        elif comp.startswith("rotor_"):
                            v_rotors_killed.add(comp)
                            barrel_stats_map[b_id]["rotor_hits"] += 1
                        elif comp.startswith("motor_"):
                            r_id = comp.replace("motor_", "rotor_")
                            v_rotors_killed.add(r_id)
                            barrel_stats_map[b_id]["rotor_hits"] += 1

                # Record sample volley for visualization (first volley)
                if v_idx == 0:
                    sample_scatter["y"].append(round(y_rel, 4))
                    sample_scatter["z"].append(round(z_rel, 4))
                    sample_scatter["barrel_id"].append(b_id)
                    sample_scatter["is_hit"].append(hit_info["is_hit"])
                    sample_scatter["is_lethal"].append(hit_info["is_lethal"])
                    sample_scatter["component"].append(hit_info["component"])
                    sample_scatter["striking_energy_j"].append(round(ke_pellet, 2))
                    sample_scatter["striking_velocity_mps"].append(round(v_impact, 1))
                    sample_scatter["flight_time_s"].append(round(t_arrival, 4))

        # Volley Kill Condition
        is_killed = bool(
            (v_lethal >= crit_hits_req)
            or (v_fuselage_lethal >= 1)
            or (v_battery_lethal >= 1)
            or (len(v_rotors_killed) >= 2)
        )
        if is_killed:
            kills_count += 1

        all_hits_counts.append(v_hits)
        all_lethal_counts.append(v_lethal)
        all_energy_on_drone.append(v_energy)

        volley_records.append({
            "volley_idx": v_idx + 1,
            "hits_on_drone": v_hits,
            "lethal_hits": v_lethal,
            "energy_on_drone_j": round(v_energy, 1),
            "fuselage_lethal": v_fuselage_lethal,
            "battery_lethal": v_battery_lethal,
            "rotors_killed": len(v_rotors_killed),
            "is_killed": is_killed,
        })

    # Summary Statistics
    pk_pct = round((kills_count / max(1, n_volleys)) * 100.0, 1)
    mean_hits = round(float(np.mean(all_hits_counts)), 1)
    mean_lethal = round(float(np.mean(all_lethal_counts)), 1)
    mean_energy = round(float(np.mean(all_energy_on_drone)), 1)

    # Cloud geometry in sample volley
    if sample_scatter["y"]:
        rads = np.hypot(sample_scatter["y"], sample_scatter["z"])
        cep_50 = round(float(np.percentile(rads, 50)), 3)
        r_90 = round(float(np.percentile(rads, 90)), 3)
        cloud_diam_90 = round(2.0 * r_90, 3)
    else:
        cep_50, r_90, cloud_diam_90 = 0.5, 1.0, 2.0

    # Calculate per-barrel summary list
    barrel_stats_list: list[dict[str, Any]] = []
    tot_hits_all = sum(b["hits_on_drone"] for b in barrel_stats_map.values())
    for b_id, b_data in barrel_stats_map.items():
        b_fired = b_data["pellets_fired"]
        b_hits = b_data["hits_on_drone"]
        hit_pct = round((b_hits / max(1, b_fired)) * 100.0, 1)
        contrib_pct = round((b_hits / max(1, tot_hits_all)) * 100.0, 1) if tot_hits_all > 0 else 0.0
        b_data["hit_rate_pct"] = hit_pct
        b_data["contribution_pct"] = contrib_pct
        barrel_stats_list.append(b_data)

    return {
        "summary": {
            "kill_probability_pct": pk_pct,
            "mean_hits_on_drone": mean_hits,
            "mean_lethal_hits": mean_lethal,
            "mean_energy_on_drone_j": mean_energy,
            "total_pellets_per_volley": tot_pellets_per_volley,
            "n_barrels": n_act_barrels,
            "n_volleys": n_volleys,
            "cloud_cep_50_m": cep_50,
            "cloud_r_90_m": r_90,
            "cloud_diameter_90_m": cloud_diam_90,
            "target_distance_m": target_distance_m,
            "drone_model": drone_model,
            "drone_name": drone_spec.get("name", drone_model),
            "drone_velocity_mps": drone_velocity_mps,
            "v0_nominal_mps": eff_v0,
            "mean_v_impact_nominal_mps": round(v_impact_nominal, 1),
            "firing_mode": firing_mode,
            "pointing_mode": pointing_mode,
            "layout_type": layout_type,
            "aspect_mode": aspect_mode,
            "aspect_name": aspect_geom["name"],
            "drone_pitch_deg": aspect_geom["pitch_deg"],
            "drone_yaw_deg": aspect_geom["yaw_deg"],
            "drone_roll_deg": aspect_geom["roll_deg"],
            "presented_area_m2": aspect_geom["presented_area_m2"],
            "presented_area_cm2": aspect_geom["presented_area_cm2"],
            "planform_area_m2": aspect_geom["planform_area_m2"],
            "cross_section_ratio_pct": aspect_geom["cross_section_ratio_pct"],
        },
        "drone_spec": drone_spec,
        "aspect_geometry": aspect_geom,
        "barrels": barrels,
        "barrel_stats": barrel_stats_list,
        "sample_scatter": sample_scatter,
        "volley_records": volley_records,
    }


def benchmark_drone_aspect_angles_mc(
    drone_model: str = "commercial_quad_medium",
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    layout_type: str = "circular",
    n_barrels: int = 6,
    mount_radius_m: float = 0.20,
    spacing_y_m: float = 0.12,
    spacing_z_m: float = 0.12,
    n_cols: int = 3,
    n_rows: int = 2,
    pointing_mode: str = "convergent",
    splay_angle_mrad: float = 0.0,
    firing_mode: str = "simultaneous_volley",
    ripple_delay_ms: float = 8.0,
    custom_barrel_specs: list[dict[str, Any]] | None = None,
    default_barrel_length_in: float = 24.0,
    target_distance_m: float = 35.0,
    drone_velocity_mps: float = 15.0,
    aim_error_mrad: float = 1.0,
    crosswind_mps: float = 0.0,
    default_choke: str = "Full",
    n_volleys: int = 25,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Benchmark anti-drone interception effectiveness across all 6 standard presentation aspect angles."""
    aspects = [
        "head_on_frontal",
        "flank_side",
        "oblique_quartering",
        "tail_rear",
        "planform_top",
        "planform_bottom",
    ]
    results_list: list[dict[str, Any]] = []
    for asp in aspects:
        sim = simulate_anti_drone_multishotgun_mc(
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
            custom_barrel_specs=custom_barrel_specs,
            default_barrel_length_in=default_barrel_length_in,
            target_distance_m=target_distance_m,
            ammunition_preset=ammunition_preset,
            default_choke=default_choke,
            drone_model=drone_model,
            drone_velocity_mps=drone_velocity_mps,
            aim_error_mrad=aim_error_mrad,
            crosswind_mps=crosswind_mps,
            aspect_mode=asp,
            n_volleys=n_volleys,
            seed=seed,
            **kwargs,
        )
        sm = sim["summary"]
        results_list.append({
            "aspect_mode": asp,
            "name": sm.get("aspect_name", asp),
            "pitch_deg": sm.get("drone_pitch_deg", 0.0),
            "yaw_deg": sm.get("drone_yaw_deg", 0.0),
            "presented_area_cm2": sm.get("presented_area_cm2", 0.0),
            "cross_section_ratio_pct": sm.get("cross_section_ratio_pct", 100.0),
            "kill_probability_pct": sm["kill_probability_pct"],
            "mean_hits": sm["mean_hits_on_drone"],
            "mean_lethal_hits": sm["mean_lethal_hits"],
            "mean_energy_j": sm["mean_energy_on_drone_j"],
        })
    return {
        "drone_model": drone_model,
        "target_distance_m": target_distance_m,
        "n_barrels": n_barrels,
        "ammunition_preset": ammunition_preset,
        "aspect_results": results_list,
        "comparison_table": results_list,
    }


def compare_anti_drone_layouts(
    layout_configs: list[dict[str, Any]] | None = None,
    distance_stations_m: list[float] | None = None,
    ammunition_preset: str = "12G_NO4_BUCK_C_UAS",
    drone_model: str = "commercial_quad_medium",
    drone_velocity_mps: float = 15.0,
    pointing_mode: str = "convergent",
    firing_mode: str = "simultaneous_volley",
    n_volleys: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare multiple multi-shotgun array placement configurations across engagement ranges."""
    if layout_configs is None:
        layout_configs = [
            {"name": "6-Barrel Circular Ring", "layout_type": "circular", "n_barrels": 6, "mount_radius_m": 0.20},
            {"name": "2x3 Grid Matrix", "layout_type": "grid", "n_rows": 2, "n_cols": 3, "spacing_y_m": 0.12, "spacing_z_m": 0.12},
            {"name": "7-Barrel Honeycomb Core", "layout_type": "honeycomb", "n_barrels": 7, "mount_radius_m": 0.20},
            {"name": "6-Barrel Horizontal Fan", "layout_type": "fan_horizontal", "n_barrels": 6, "spacing_y_m": 0.12},
            {"name": "8-Barrel Concentric Dual-Ring", "layout_type": "concentric", "n_barrels": 8, "mount_radius_m": 0.25},
        ]

    if distance_stations_m is None:
        distance_stations_m = [15.0, 25.0, 35.0, 50.0, 65.0]

    comparison_table: list[dict[str, Any]] = []
    layout_curves: dict[str, dict[str, list[float]]] = {}

    for l_idx, cfg in enumerate(layout_configs):
        c_name = cfg.get("name", f"Layout {l_idx+1}")
        layout_curves[c_name] = {
            "distances_m": list(distance_stations_m),
            "pk_pct": [],
            "mean_hits": [],
            "mean_lethal": [],
            "mean_energy_j": [],
            "cloud_diam_m": [],
        }

        for d_m in distance_stations_m:
            sim = simulate_anti_drone_multishotgun_mc(
                layout_type=cfg.get("layout_type", "circular"),
                n_barrels=int(cfg.get("n_barrels", 6)),
                mount_radius_m=float(cfg.get("mount_radius_m", 0.20)),
                spacing_y_m=float(cfg.get("spacing_y_m", 0.12)),
                spacing_z_m=float(cfg.get("spacing_z_m", 0.12)),
                n_cols=int(cfg.get("n_cols", 3)),
                n_rows=int(cfg.get("n_rows", 2)),
                pointing_mode=pointing_mode,
                firing_mode=firing_mode,
                ammunition_preset=ammunition_preset,
                target_distance_m=d_m,
                drone_model=drone_model,
                drone_velocity_mps=drone_velocity_mps,
                n_volleys=n_volleys,
                seed=seed + l_idx * 17,
            )
            sm = sim["summary"]
            layout_curves[c_name]["pk_pct"].append(sm["kill_probability_pct"])
            layout_curves[c_name]["mean_hits"].append(sm["mean_hits_on_drone"])
            layout_curves[c_name]["mean_lethal"].append(sm["mean_lethal_hits"])
            layout_curves[c_name]["mean_energy_j"].append(sm["mean_energy_on_drone_j"])
            layout_curves[c_name]["cloud_diam_m"].append(sm["cloud_diameter_90_m"])

            comparison_table.append({
                "layout_name": c_name,
                "layout_type": cfg.get("layout_type", "circular"),
                "n_barrels": sm["n_barrels"],
                "distance_m": d_m,
                "kill_probability_pct": sm["kill_probability_pct"],
                "mean_hits": sm["mean_hits_on_drone"],
                "mean_lethal": sm["mean_lethal_hits"],
                "mean_energy_j": sm["mean_energy_on_drone_j"],
                "cloud_diameter_m": sm["cloud_diameter_90_m"],
            })

    return {
        "layout_configs": layout_configs,
        "distance_stations_m": distance_stations_m,
        "drone_model": drone_model,
        "ammunition_preset": ammunition_preset,
        "comparison_table": comparison_table,
        "layout_curves": layout_curves,
    }
