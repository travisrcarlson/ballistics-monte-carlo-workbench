"""Small Arms Ballistics, Shotgun Patterning & Anti-Drone Dispersion Package — ballistics submodule."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from ammo_mc.aero_sphere import (
    didion_crosswind_drift_m,
    exact_quadratic_drag_tof_and_velocity,
    sphere_drag_compressible,
    sutherland_air_viscosity,
)
from .catalogs import CHOKE_DEFINITIONS, DRONE_ASPECT_PRESETS, DRONE_TARGET_SPECS, MULTIBARREL_PRESETS, PELLET_SPECIFICATIONS, SHOTGUN_PRESETS, STANDARD_BARREL_LENGTHS, get_choke_definitions, get_drone_target_specs, get_multibarrel_presets, get_shotgun_presets
from .geometry import check_target_hit, compute_drone_projected_geometry, evaluate_drone_impact, point_in_convex_polygon

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

    # 2. Pellet Aerodynamic Deceleration Parameters (compressible Morrison/Bailey-Hiatt/Loth sphere drag)
    d_m = pellet_diameter_mm / 1000.0
    m_kg = pellet_mass_g / 1000.0
    rho_air = 1.225
    mu_air = sutherland_air_viscosity(288.15)
    a_sound = 340.29
    area_p = math.pi * 0.25 * (d_m ** 2)

    # Path-integrate baseline compressible Cd across [0, target_distance_m] so transonic deceleration is resolved
    n_quad = 8
    dx_q = max(float(target_distance_m), 0.5) / n_quad
    v_q = max(float(v0_mps), 10.0)
    t_q_nom = 0.0
    for _ in range(n_quad):
        cd1 = sphere_drag_compressible(v_q * d_m * rho_air / mu_air, v_q / a_sound)
        k1 = 0.5 * rho_air * cd1 * area_p / max(m_kg, 1e-6)
        vmid = max(5.0, v_q * math.exp(-0.5 * k1 * dx_q))
        cd2 = sphere_drag_compressible(vmid * d_m * rho_air / mu_air, vmid / a_sound)
        k2 = 0.5 * rho_air * cd2 * area_p / max(m_kg, 1e-6)
        dt_step, v_next = exact_quadratic_drag_tof_and_velocity(dx_q, v_q, k2)
        t_q_nom += dt_step
        v_q = max(5.0, v_next)
    kappa_eff_nom = max(0.0, -math.log(max(v_q, 1e-3) / max(float(v0_mps), 10.0)) / max(float(target_distance_m), 1e-3))
    cd_base = max(0.20, (2.0 * m_kg * kappa_eff_nom) / max(rho_air * area_p, 1e-12))

    # Differential drag factor distribution due to launch setback deformation
    # Round leading pellets retain near-ideal Cd; trailing flattened pellets experience +20% drag
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

        # Generate individual pellets for this shot
        # Deformation: normalized 0.0 (round front) to 1.0 (flattened rear)
        deformations = rng.uniform(0.0, 1.0, size=pellets_per_shot)
        # Radial positions follow Rayleigh distribution
        radii = rng.rayleigh(scale=sigma_target, size=pellets_per_shot)
        thetas = rng.uniform(0.0, 2.0 * math.pi, size=pellets_per_shot)

        # Longitudinal stringing: setback deformation spreads drag around baseline path Cd
        cd_pellets = cd_base * (0.94 + 0.14 * deformations)
        kappas = (rho_air * area_p * cd_pellets) / (2.0 * max(m_kg, 1e-6))

        # Arrival velocity: v(x) = v0 * exp(-kappa * x)
        v_impacts = v0_mps * np.exp(-kappas * target_distance_m)

        # Arrival time: t = integral dx / v = (exp(kappa * x) - 1) / (v0 * kappa)
        kr = np.maximum(kappas * target_distance_m, 1e-8)
        t_arrivals = np.where(
            kr > 1e-5,
            (np.exp(kr) - 1.0) / (v0_mps * kappas),
            (target_distance_m / max(v0_mps, 1.0)) * (1.0 + 0.5 * kr),
        )

        # Exact Didion crosswind lag drift: w_z * (t_flight - R / v0)
        t_mean = float(np.mean(t_arrivals))
        wind_drift_y = didion_crosswind_drift_m(crosswind_mps, t_mean, target_distance_m, v0_mps)

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

    # 50% and 90% radial containment (CEP50 and R90) centered on pattern CoM
    com_y = float(np.mean(y_arr))
    com_z = float(np.mean(z_arr))
    sorted_r = np.sort(np.hypot(y_arr - com_y, z_arr - com_z))
    cep_50_m = float(sorted_r[int(0.50 * (len(sorted_r) - 1))])
    r_90_m = float(sorted_r[int(0.90 * (len(sorted_r) - 1))])

    # Pattern Uniformity Index: actual 10in-to-30in ratio relative to ideal Rayleigh pattern density
    ideal_p10 = 1.0 - math.exp(-(r_10in ** 2) / max(2.0 * (sigma_target ** 2), 1e-9))
    ideal_p30 = 1.0 - math.exp(-(r_30in ** 2) / max(2.0 * (sigma_target ** 2), 1e-9))
    ideal_ratio = max(ideal_p10 / max(ideal_p30, 1e-6), 1e-6)
    uniformity_index = (hits_10in_total / max(hits_30in_total, 1)) / ideal_ratio

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
        crit_hits_req = int(drone_spec.get("critical_hits_for_kill", 2))
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


