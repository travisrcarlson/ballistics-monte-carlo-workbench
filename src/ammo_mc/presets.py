"""Notional example munition presets and matching baseline scenarios across Ammo_Taxonomy families.

GOVERNANCE NOTE (App_Spec R27):
All parameters in this module are strictly NOTIONAL generic class examples constructed from open
textbook/handbook orders of magnitude for verification and demonstration. Real system data must
be loaded by the user from controlled local JSON files.
"""

from __future__ import annotations

from typing import Any
from ammo_mc.schemas import (
    AeroSpec,
    AimSpec,
    AmmoDefinition,
    BetaReliabilitySpec,
    BiasErrorsSpec,
    CaseSpec,
    ErrorBudgetSpec,
    FillSpec,
    FuzeSpec,
    InertiaSpec,
    LaunchSpec,
    MetSpec,
    MPMCoeffsSpec,
    MVSpec,
    NormalSpec,
    PenetratorSpec,
    RandomErrorsSpec,
    Scenario,
    TargetSpec,
    WarheadSpec,
)


def _build_presets() -> dict[str, tuple[AmmoDefinition, Scenario]]:
    presets: dict[str, tuple[AmmoDefinition, Scenario]] = {}

    # 1. SA-01: 7.62mm Notional Ball Rifle
    presets["NOTIONAL-SA-01"] = (
        AmmoDefinition(
            id="NOTIONAL-SA-01",
            class_id="SA-01",
            name="7.62mm Notional Boat-Tail Ball (NOTIONAL)",
            description="Notional spin-stabilised small-arms FMJ rifle cartridge using G7 drag and Lambert-Jonas/V50 barrier penetration.",
            calibre_m=0.00782,
            mass_kg=0.01134,
            mass_sigma_kg=0.00005,
            length_m=0.0315,
            inertia=InertiaSpec(ix=7.2e-8, iy=8.1e-7),
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=0.96, sigma=0.015)),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=792.0, sigma_rr=3.5, sigma_lot=2.0),
                twist_cal=32.5,
                temp_coeff_mps_per_C=0.45,
            ),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="bullet_barrier",
                L_m=0.0315,
                D_m=0.00782,
                rho=10500.0,
                v50_mps=420.0,
                sigma_v50_mps=18.0,
            ),
            sources=["S13", "S14", "S30"],
        ),
        Scenario(
            name="600m Rifle Point Target & Light Barrier",
            target=TargetSpec(
                type="armour",
                posture="standing",
                geometry={"half_width_x_m": 0.25, "half_width_y_m": 0.25, "presented_area_m2": 0.45, "obliquity_deg": 15.0},
            ),
            aim=AimSpec(range_m=600.0, az_rad=0.0),
            met=MetSpec(wind_cross_right_mps=2.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.15, aim_qe_bias_sigma_mils=0.15, wind_cross_bias_sigma_mps=0.5),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.25, aim_qe_rand_sigma_mils=0.25, jump_az_sigma_mils=0.15, jump_qe_sigma_mils=0.15),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=40,
            rounds_per_occasion=5,
        ),
    )

    # 2. CN-01: 30mm Notional Programmable Airburst HE-FRAG
    presets["NOTIONAL-CN-01"] = (
        AmmoDefinition(
            id="NOTIONAL-CN-01",
            class_id="CN-01",
            name="30mm Notional Programmable Airburst HE-FRAG (NOTIONAL)",
            description="Notional medium-calibre cannon airburst round with programmable time fuze and self-destruct backup.",
            calibre_m=0.030,
            mass_kg=0.42,
            mass_sigma_kg=0.003,
            length_m=0.145,
            inertia=InertiaSpec(ix=4.8e-5, iy=5.2e-4),
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.08, sigma=0.02)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=1050.0, sigma_rr=4.0, sigma_lot=2.5), twist_cal=25.0),
            fuze=FuzeSpec(
                type="time",
                reliability=BetaReliabilitySpec(a=196.0, b=4.0),
                time_set_s=1.85,
                time_sigma_s=0.006,
                sd_fail_prob=0.01,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="RDX", mass_kg=0.055),
                case=CaseSpec(mass_kg=0.28, t_in=0.16, d_in=1.18, length_m=0.09),
                lethal_radius_m=5.5,
                carleton_rx_m=5.5,
                carleton_ry_m=5.5,
            ),
            sources=["S01", "S10", "S24"],
        ),
        Scenario(
            name="1500m Cannon Airburst Engagement",
            target=TargetSpec(type="personnel", posture="standing", geometry={"presented_area_m2": 0.56, "half_width_x_m": 8.0, "half_width_y_m": 8.0}),
            aim=AimSpec(range_m=1500.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.3, aim_qe_bias_sigma_mils=0.3),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.6, aim_qe_rand_sigma_mils=0.6),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=30,
            rounds_per_occasion=6,
        ),
    )

    # 3. CN-02: 120mm Notional Tungsten APFSDS Long Rod (triggers TP-01 FLAG warning)
    presets["NOTIONAL-CN-02"] = (
        AmmoDefinition(
            id="NOTIONAL-CN-02",
            class_id="CN-02",
            name="120mm Notional Fin-Stabilised Long-Rod APFSDS (NOTIONAL)",
            description="Notional kinetic-energy tungsten long-rod penetrator using Lanz-Odermatt (TP-01 [FLAG]).",
            calibre_m=0.026,
            mass_kg=4.8,
            mass_sigma_kg=0.02,
            length_m=0.62,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=0.82, sigma=0.02)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=1650.0, sigma_rr=5.5, sigma_lot=3.0), twist_cal=0.0),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="long_rod",
                L_m=0.56,
                D_m=0.025,
                rho=17600.0,
                rod_material="Tungsten alloy",
            ),
            sources=["S06", "S13"],
        ),
        Scenario(
            name="2000m Tank RHA Plate Engagement",
            target=TargetSpec(
                type="armour",
                posture="armoured",
                geometry={
                    "half_width_x_m": 1.15,
                    "half_width_y_m": 1.6,
                    "armour_thickness_m": 0.52,
                    "armour_bhn": 300.0,
                    "armour_rho_kg_m3": 7850.0,
                    "obliquity_deg": 30.0,
                },
            ),
            aim=AimSpec(range_m=2000.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.15, aim_qe_bias_sigma_mils=0.15),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.20, aim_qe_rand_sigma_mils=0.20, jump_az_sigma_mils=0.25, jump_qe_sigma_mils=0.25),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=50,
            rounds_per_occasion=2,
        ),
    )

    # 4. CN-03: 84mm Notional Recoilless HEAT
    presets["NOTIONAL-CN-03"] = (
        AmmoDefinition(
            id="NOTIONAL-CN-03",
            class_id="CN-03",
            name="84mm Notional Recoilless Shaped-Charge HEAT (NOTIONAL)",
            description="Notional shaped-charge anti-armour round using Birkhoff hydrodynamic penetration (TP-02) and standoff curve (TP-03).",
            calibre_m=0.084,
            mass_kg=3.1,
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.15, sigma=0.03)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=290.0, sigma_rr=3.0, sigma_lot=1.5), twist_cal=0.0),
            fuze=FuzeSpec(type="PIBD", reliability=BetaReliabilitySpec(a=97.0, b=3.0), graze_threshold_deg=15.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Octol 75/25", mass_kg=0.65),
                case=CaseSpec(mass_kg=1.6, t_in=0.18, d_in=3.3, length_m=0.22),
                lethal_radius_m=8.0,
            ),
            penetrator=PenetratorSpec(
                type="shaped_charge",
                jet_length_m=0.38,
                jet_rho_kg_m3=8960.0,
                standoff_cd=3.5,
            ),
            sources=["S01", "S10"],
        ),
        Scenario(
            name="450m Direct-Fire Shaped-Charge Engagement",
            target=TargetSpec(
                type="armour",
                posture="armoured",
                geometry={"half_width_x_m": 1.2, "half_width_y_m": 1.5, "armour_thickness_m": 0.32, "obliquity_deg": 25.0},
            ),
            aim=AimSpec(range_m=450.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.5, aim_qe_bias_sigma_mils=0.5),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.9, aim_qe_rand_sigma_mils=0.9),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=50,
            rounds_per_occasion=2,
        ),
    )

    # 5. AR-01: 155mm Notional Spin-Stabilised HE Artillery (Flagship L2 MPM preset)
    presets["NOTIONAL-AR-01"] = (
        AmmoDefinition(
            id="NOTIONAL-AR-01",
            class_id="AR-01",
            name="155mm Notional HE Spin-Stabilised Projectile (NOTIONAL)",
            description="Notional 155mm artillery shell modeled with STANAG 4355 Modified Point Mass (L2), Kingery-Bulmash blast, and Mott/Gurney fragmentation.",
            calibre_m=0.155,
            mass_kg=43.2,
            mass_sigma_kg=0.18,
            length_m=0.78,
            inertia=InertiaSpec(ix=0.145, iy=1.22),
            aero=AeroSpec(
                drag_ref="G7",
                form_factor=NormalSpec(mu=1.04, sigma=0.015, sigma_bias=0.01),
                mpm_coeffs=MPMCoeffsSpec(c_da2=4.2, c_la=2.15, c_ma=2.85, c_mag=0.24, c_spin=-0.014),
            ),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=685.0, sigma_rr=1.8, sigma_lot=1.4),
                twist_cal=20.0,
                temp_coeff_mps_per_C=0.42,
            ),
            fuze=FuzeSpec(
                type="PD",
                reliability=BetaReliabilitySpec(a=97.0, b=3.0),
                hob=NormalSpec(mu=0.0, sigma=0.0),
                graze_threshold_deg=10.0,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Composition B", mass_kg=6.9),
                case=CaseSpec(mass_kg=33.5, t_in=0.58, d_in=6.10, length_m=0.62),
                mott_B=0.0779,
                lethal_radius_m=28.0,
                carleton_rx_m=30.0,
                carleton_ry_m=26.0,
                auw_kg=43.2,
            ),
            sources=["S01", "S02", "S10", "S14", "S32"],
        ),
        Scenario(
            name="14 km Indirect-Fire Battery Volley (L2 MPM)",
            target=TargetSpec(
                type="area",
                posture="standing",
                geometry={"half_width_x_m": 60.0, "half_width_y_m": 60.0, "presented_area_m2": 0.56},
            ),
            aim=AimSpec(range_m=14000.0, az_rad=0.35),
            met=MetSpec(wind_range_mps=2.0, wind_cross_right_mps=3.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(
                    tle_range_sigma_m=12.0,
                    tle_deflection_sigma_m=12.0,
                    aim_az_bias_sigma_mils=0.6,
                    aim_qe_bias_sigma_mils=0.6,
                    wind_range_bias_sigma_mps=1.2,
                    wind_cross_bias_sigma_mps=1.2,
                ),
                random=RandomErrorsSpec(
                    aim_az_rand_sigma_mils=0.8,
                    aim_qe_rand_sigma_mils=0.8,
                ),
            ),
            fidelity="L2",
            dt_s=0.02,
            n_occasions=40,
            rounds_per_occasion=6,
        ),
    )

    # 6. AR-02: 155mm Notional Base-Bleed Extended Range
    ar02_ammo = presets["NOTIONAL-AR-01"][0].model_copy(deep=True)
    ar02_ammo.id = "NOTIONAL-AR-02"
    ar02_ammo.class_id = "AR-02"
    ar02_ammo.name = "155mm Notional Base-Bleed Extended-Range HE (NOTIONAL)"
    ar02_ammo.description = "Notional 155mm base-bleed projectile demonstrating EB-12 base-drag reduction during burn."
    ar02_ammo.launch.mv = MVSpec(mu=810.0, sigma_rr=2.2, sigma_lot=1.6)
    ar02_ammo.launch.base_bleed_burn_s = 24.0
    ar02_ammo.launch.base_bleed_dcd = 0.065
    ar02_scn = presets["NOTIONAL-AR-01"][1].model_copy(deep=True)
    ar02_scn.name = "22 km Base-Bleed Extended Range Mission"
    ar02_scn.aim.range_m = 22000.0
    presets["NOTIONAL-AR-02"] = (ar02_ammo, ar02_scn)

    # 7. AR-03: 155mm Notional Cargo Submunition Carrier
    ar03_ammo = presets["NOTIONAL-AR-01"][0].model_copy(deep=True)
    ar03_ammo.id = "NOTIONAL-AR-03"
    ar03_ammo.class_id = "AR-03"
    ar03_ammo.name = "155mm Notional Cargo Submunition Carrier (NOTIONAL)"
    ar03_ammo.description = "Notional cargo shell dispensing 48 submunitions with FZ-07 self-destruct redundancy and UXO footprint."
    ar03_ammo.fuze = FuzeSpec(
        type="submunition",
        reliability=BetaReliabilitySpec(a=96.0, b=4.0),
        sd_fail_prob=0.15,
        common_cause_beta=0.05,
        submunition_count=48,
        submunition_pattern_radius_m=55.0,
    )
    ar03_scn = presets["NOTIONAL-AR-01"][1].model_copy(deep=True)
    ar03_scn.name = "12 km Submunition Footprint & UXO Contamination Assessment"
    ar03_scn.aim.range_m = 12000.0
    ar03_scn.n_occasions = 25
    ar03_scn.rounds_per_occasion = 4
    presets["NOTIONAL-AR-03"] = (ar03_ammo, ar03_scn)

    # 8. AR-05: 155mm Notional Course-Corrected Guided Shell
    ar05_ammo = presets["NOTIONAL-AR-01"][0].model_copy(deep=True)
    ar05_ammo.id = "NOTIONAL-AR-05"
    ar05_ammo.class_id = "AR-05"
    ar05_ammo.name = "155mm Notional Course-Corrected Guided Shell (NOTIONAL)"
    ar05_ammo.description = "Notional 155mm shell fitted with GNSS course-correction fuze plugin (8m CEP with 3% unguided fallback)."
    ar05_ammo.guidance_plugin = "course_correction"
    ar05_ammo.guidance_params = {"cep_guided_m": 8.0, "max_authority_m": 280.0, "p_guidance_fail": 0.03, "gnss_bias_sigma_m": 2.0}
    ar05_scn = presets["NOTIONAL-AR-01"][1].model_copy(deep=True)
    ar05_scn.name = "15 km Precision Course-Corrected Engagement"
    ar05_scn.aim.range_m = 15000.0
    ar05_scn.rounds_per_occasion = 2
    presets["NOTIONAL-AR-05"] = (ar05_ammo, ar05_scn)

    # 9. MO-01: 81mm Notional Fin-Stabilised HE Mortar
    presets["NOTIONAL-MO-01"] = (
        AmmoDefinition(
            id="NOTIONAL-MO-01",
            class_id="MO-01",
            name="81mm Notional Fin-Stabilised HE Mortar Bomb (NOTIONAL)",
            description="Notional high-angle mortar bomb modeled with 3-DOF point-mass flight, proximity/PD fuze, and Mott fragmentation.",
            calibre_m=0.081,
            mass_kg=4.2,
            mass_sigma_kg=0.03,
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=0.95, sigma=0.025)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=255.0, sigma_rr=1.6, sigma_lot=1.1), twist_cal=0.0),
            fuze=FuzeSpec(type="proximity", reliability=BetaReliabilitySpec(a=96.0, b=4.0), hob=NormalSpec(mu=3.5, sigma=0.8)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Composition B", mass_kg=0.85),
                case=CaseSpec(mass_kg=2.9, t_in=0.26, d_in=3.19, length_m=0.24),
                lethal_radius_m=15.0,
                carleton_rx_m=15.0,
                carleton_ry_m=15.0,
                auw_kg=4.2,
            ),
            sources=["S01", "S10", "S24"],
        ),
        Scenario(
            name="3800m High-Angle Mortar Platoon Fire",
            target=TargetSpec(type="personnel", posture="standing", geometry={"presented_area_m2": 0.56, "half_width_x_m": 35.0, "half_width_y_m": 35.0}),
            aim=AimSpec(range_m=3800.0, high_angle=True),
            met=MetSpec(wind_cross_right_mps=2.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=1.2, aim_qe_bias_sigma_mils=1.2, wind_cross_bias_sigma_mps=1.0),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=1.8, aim_qe_rand_sigma_mils=1.8),
            ),
            fidelity="L1",
            dt_s=0.01,
            n_occasions=40,
            rounds_per_occasion=4,
        ),
    )

    # 10. RK-01: 122mm Notional Unguided Artillery Rocket
    presets["NOTIONAL-RK-01"] = (
        AmmoDefinition(
            id="NOTIONAL-RK-01",
            class_id="RK-01",
            name="122mm Notional Unguided Artillery Rocket (NOTIONAL)",
            description="Notional solid-propellant artillery rocket with EB-13 boost thrust + tip-off launch dispersion.",
            calibre_m=0.122,
            mass_kg=66.0,
            mass_sigma_kg=0.4,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.18, sigma=0.03)),
            launch=LaunchSpec(
                mode="rocket",
                mv=MVSpec(mu=45.0, sigma_rr=1.5, sigma_lot=1.0),
                twist_cal=0.0,
                rocket_burn_s=2.8,
                rocket_mdot_kg_s=7.2,
                rocket_ve_mps=2150.0,
            ),
            fuze=FuzeSpec(type="PD", reliability=BetaReliabilitySpec(a=95.0, b=5.0)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="RDX", mass_kg=6.4),
                case=CaseSpec(mass_kg=12.5, t_in=0.32, d_in=4.80, length_m=0.48),
                lethal_radius_m=24.0,
                carleton_rx_m=25.0,
                carleton_ry_m=25.0,
                auw_kg=66.0,
            ),
            sources=["S01", "S13", "S24"],
        ),
        Scenario(
            name="16 km Unguided Rocket Salvo",
            target=TargetSpec(type="area", posture="standing", geometry={"half_width_x_m": 120.0, "half_width_y_m": 120.0, "presented_area_m2": 0.56}),
            aim=AimSpec(range_m=16000.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=1.5, aim_qe_bias_sigma_mils=1.5),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=2.5, aim_qe_rand_sigma_mils=2.5, jump_az_sigma_mils=4.0, jump_qe_sigma_mils=3.5),
            ),
            fidelity="L1",
            dt_s=0.02,
            n_occasions=25,
            rounds_per_occasion=8,
        ),
    )

    # 11. AB-01: 250kg Notional General-Purpose Free-Fall Bomb
    presets["NOTIONAL-AB-01"] = (
        AmmoDefinition(
            id="NOTIONAL-AB-01",
            class_id="AB-01",
            name="250 kg Class Notional General-Purpose Bomb (NOTIONAL)",
            description="Notional low-drag air-delivered GP bomb initialized via EB-14 aircraft release kinematics.",
            calibre_m=0.273,
            mass_kg=227.0,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=0.92, sigma=0.02)),
            launch=LaunchSpec(
                mode="bomb",
                mv=MVSpec(mu=240.0, sigma_rr=1.5, sigma_lot=0.5),
                twist_cal=0.0,
                release_alt_m=3000.0,
                release_tas_mps=240.0,
                release_dive_rad=0.1745,  # 10 deg dive
            ),
            fuze=FuzeSpec(type="PD", reliability=BetaReliabilitySpec(a=98.0, b=2.0)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Tritonal", mass_kg=87.0),
                case=CaseSpec(mass_kg=135.0, t_in=0.55, d_in=10.75, length_m=1.5),
                lethal_radius_m=58.0,
                carleton_rx_m=60.0,
                carleton_ry_m=55.0,
                auw_kg=227.0,
            ),
            sources=["S01", "S02", "S10"],
        ),
        Scenario(
            name="3000m Release Free-Fall Bomb Delivery",
            target=TargetSpec(type="structure", posture="brick_house", geometry={"half_width_x_m": 25.0, "half_width_y_m": 25.0}),
            aim=AimSpec(range_m=4750.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(tle_range_sigma_m=8.0, tle_deflection_sigma_m=8.0),
                random=RandomErrorsSpec(release_alt_sigma_m=12.0, release_tas_sigma_mps=1.8, release_time_sigma_s=0.06),
            ),
            fidelity="L1",
            dt_s=0.02,
            n_occasions=40,
            rounds_per_occasion=2,
        ),
    )

    # 12. DM-01: 50 kg Notional Bare Explosive Open Detonation
    presets["NOTIONAL-DM-01"] = (
        AmmoDefinition(
            id="NOTIONAL-DM-01",
            class_id="DM-01",
            name="50 kg Notional Bare C-4 Open-Detonation Charge (NOTIONAL)",
            description="Notional static bare-charge demolition evaluating Kingery-Bulmash/Kinney-Graham blast contours and IATG TB-15 safety distances.",
            calibre_m=0.38,
            mass_kg=50.0,
            mass_sigma_kg=0.5,
            launch=LaunchSpec(mode="static", mv=MVSpec(mu=0.0, sigma_rr=0.0, sigma_lot=0.0), twist_cal=0.0),
            fuze=FuzeSpec(type="static", reliability=BetaReliabilitySpec(a=99.5, b=0.5)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="C-4", mass_kg=50.0),
                case=CaseSpec(mass_kg=0.0, t_in=0.01, d_in=15.0, length_m=0.35),
                geometry="bare",
                lethal_radius_m=18.0,
                auw_kg=50.0,
            ),
            sources=["S01", "S02", "S03"],
        ),
        Scenario(
            name="Static Bare-Charge Blast & Noise Range Assessment",
            target=TargetSpec(type="structure", posture="brick_house", geometry={"exposed_site_distance_m": 300.0}),
            aim=AimSpec(range_m=0.0),
            fidelity="L1",
            n_occasions=100,
            rounds_per_occasion=1,
            safety_distance_mode=True,
        ),
    )

    # 13. DM-02: Notional Open-Detonation UXO Disposal (Fragment Trajectory EB-16 vs TB-15)
    presets["NOTIONAL-DM-02"] = (
        AmmoDefinition(
            id="NOTIONAL-DM-02",
            class_id="DM-02",
            name="Notional 155mm EOD Open-Detonation Disposal (NOTIONAL)",
            description="Static EOD counter-charge disposal simulating individual Mott/Gurney natural fragments flying EB-16 trajectories vs IATG TB-15 lines.",
            calibre_m=0.155,
            mass_kg=45.0,
            launch=LaunchSpec(mode="static", mv=MVSpec(mu=0.0, sigma_rr=0.0, sigma_lot=0.0), twist_cal=0.0),
            fuze=FuzeSpec(type="static", reliability=BetaReliabilitySpec(a=99.5, b=0.5)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Composition B", mass_kg=8.5),
                case=CaseSpec(mass_kg=34.0, t_in=0.58, d_in=6.10, length_m=0.62),
                mott_B=0.0779,
                lethal_radius_m=30.0,
                auw_kg=45.0,
            ),
            sources=["S01", "S09", "S10"],
        ),
        Scenario(
            name="EOD Open-Detonation Fragment Hazard Quantile (L3)",
            target=TargetSpec(type="personnel", posture="standing", geometry={"exposed_site_distance_m": 600.0, "presented_area_m2": 0.56}),
            aim=AimSpec(range_m=0.0),
            fidelity="L3",
            n_occasions=80,
            rounds_per_occasion=1,
            hazard_quantile=0.99,
            fragments_per_detonation=150,
            safety_distance_mode=True,
        ),
    )

    # 14. ST-01: Notional Magazine Stack Accidental Explosion (QD Risk)
    presets["NOTIONAL-ST-01"] = (
        AmmoDefinition(
            id="NOTIONAL-ST-01",
            class_id="ST-01",
            name="Notional 1,080 kg NEQ Munition Stack Accidental Event (NOTIONAL)",
            description="Notional magazine stack accidental explosion combining TB-10 brick-house damage, TF-10 AASTP-1 stack fragment density, TF-14 RC debris, and TB-16 thermal dose.",
            calibre_m=0.155,
            mass_kg=6480.0,
            launch=LaunchSpec(mode="static", mv=MVSpec(mu=0.0, sigma_rr=0.0, sigma_lot=0.0), twist_cal=0.0),
            fuze=FuzeSpec(type="static", reliability=BetaReliabilitySpec(a=99.9, b=0.1)),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="TNT", mass_kg=1080.0),
                case=CaseSpec(mass_kg=34.0, t_in=0.58, d_in=6.10, length_m=0.62),
                auw_kg=6480.0,
                stack_total_items=150,
                stack_surface_items=42,
                storage_mode="ecm",
                rc_room_vol_m3=180.0,
                rc_wall_t_m=0.30,
                thermal_flux_kw_m2_at_100m=65.0,
                thermal_duration_s=3.2,
            ),
            sources=["S01", "S08"],
        ),
        Scenario(
            name="Magazine Stack Quantity-Distance (QD) Consequence Study",
            target=TargetSpec(type="structure", posture="brick_house", geometry={"exposed_site_distance_m": 400.0}),
            aim=AimSpec(range_m=0.0),
            fidelity="L3",
            n_occasions=100,
            rounds_per_occasion=1,
            safety_distance_mode=True,
        ),
    )

    # 15. SA-02: 12.7mm Notional Heavy Machine Gun AP-I
    presets["NOTIONAL-SA-02"] = (
        AmmoDefinition(
            id="NOTIONAL-SA-02",
            class_id="SA-02",
            name="12.7mm Notional Heavy MG Hard-Core AP-I (NOTIONAL)",
            description="Notional .50 cal / 12.7mm heavy machine gun armour-piercing incendiary projectile with G7 drag and Lambert-Jonas V50 light-armour perforation.",
            calibre_m=0.01295,
            mass_kg=0.0430,
            mass_sigma_kg=0.00018,
            length_m=0.0585,
            inertia=InertiaSpec(ix=9.1e-7, iy=1.15e-5),
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=0.94, sigma=0.014)),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=885.0, sigma_rr=3.2, sigma_lot=1.8),
                twist_cal=29.5,
                temp_coeff_mps_per_C=0.52,
            ),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="bullet_barrier",
                L_m=0.048,
                D_m=0.0108,
                rho=14800.0,
                v50_mps=510.0,
                sigma_v50_mps=22.0,
            ),
            sources=["S13", "S14", "S30"],
        ),
        Scenario(
            name="1200m Heavy MG Anti-Materiel Light Armour Engagement",
            target=TargetSpec(
                type="armour",
                posture="armoured",
                geometry={"half_width_x_m": 0.9, "half_width_y_m": 1.2, "presented_area_m2": 1.8, "obliquity_deg": 20.0},
            ),
            aim=AimSpec(range_m=1200.0, az_rad=0.0),
            met=MetSpec(wind_cross_right_mps=3.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.25, aim_qe_bias_sigma_mils=0.25, wind_cross_bias_sigma_mps=0.6),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.45, aim_qe_rand_sigma_mils=0.45, jump_az_sigma_mils=0.25, jump_qe_sigma_mils=0.25),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=35,
            rounds_per_occasion=6,
        ),
    )

    # 16. SA-03: 7.62mm Notional Pyrotechnic Tracer
    presets["NOTIONAL-SA-03"] = (
        AmmoDefinition(
            id="NOTIONAL-SA-03",
            class_id="SA-03",
            name="7.62mm Notional Pyrotechnic Tracer Cartridge (NOTIONAL)",
            description="Notional small-arms tracer round demonstrating base-drag reduction (EB-12) and mass loss during pyrotechnic trace burn.",
            calibre_m=0.00782,
            mass_kg=0.00920,
            mass_sigma_kg=0.00008,
            length_m=0.0340,
            inertia=InertiaSpec(ix=6.2e-8, iy=7.8e-7),
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.02, sigma=0.022)),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=835.0, sigma_rr=4.2, sigma_lot=2.5),
                twist_cal=30.0,
                base_bleed_burn_s=2.2,
                base_bleed_dcd=0.045,
            ),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="bullet_barrier",
                L_m=0.034,
                D_m=0.00782,
                rho=9800.0,
                v50_mps=390.0,
                sigma_v50_mps=20.0,
            ),
            sources=["S13", "S14"],
        ),
        Scenario(
            name="800m Tracer Trajectory & Burnout Dispersion",
            target=TargetSpec(
                type="armour",
                posture="standing",
                geometry={"half_width_x_m": 0.35, "half_width_y_m": 0.35, "presented_area_m2": 0.56},
            ),
            aim=AimSpec(range_m=800.0),
            met=MetSpec(wind_cross_right_mps=2.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.25, aim_qe_bias_sigma_mils=0.25),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.40, aim_qe_rand_sigma_mils=0.40, jump_az_sigma_mils=0.30, jump_qe_sigma_mils=0.30),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=40,
            rounds_per_occasion=5,
        ),
    )

    # 17. CN-04: 105mm Notional Tank HESH / HEP Spall Projectile
    presets["NOTIONAL-CN-04"] = (
        AmmoDefinition(
            id="NOTIONAL-CN-04",
            class_id="CN-04",
            name="105mm Notional Tank HESH / HEP Spall Projectile (NOTIONAL)",
            description="Notional spin-stabilised High-Explosive Squash-Head (HESH/HEP) round combining base-detonating PIBD fuze, high-mass plastic explosive blast, and spall/bunker breaching.",
            calibre_m=0.105,
            mass_kg=11.2,
            mass_sigma_kg=0.06,
            length_m=0.44,
            inertia=InertiaSpec(ix=0.018, iy=0.14),
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.18, sigma=0.025)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=730.0, sigma_rr=2.5, sigma_lot=1.6), twist_cal=18.0),
            fuze=FuzeSpec(type="PIBD", reliability=BetaReliabilitySpec(a=96.5, b=3.5), graze_threshold_deg=20.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Composition A3", mass_kg=3.1),
                case=CaseSpec(mass_kg=7.6, t_in=0.22, d_in=4.13, length_m=0.36),
                lethal_radius_m=18.5,
                carleton_rx_m=19.0,
                carleton_ry_m=18.0,
                auw_kg=11.2,
            ),
            sources=["S01", "S10", "S13"],
        ),
        Scenario(
            name="1500m Fortified Structure & Bunker Breaching Fire",
            target=TargetSpec(
                type="structure",
                posture="brick_house",
                geometry={"half_width_x_m": 3.5, "half_width_y_m": 4.0, "exposed_site_distance_m": 180.0},
            ),
            aim=AimSpec(range_m=1500.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.25, aim_qe_bias_sigma_mils=0.25),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=0.45, aim_qe_rand_sigma_mils=0.45),
            ),
            fidelity="L1",
            dt_s=0.01,
            n_occasions=40,
            rounds_per_occasion=3,
        ),
    )

    # 18. AR-04: 155mm Notional Base-Eject Pyrotechnic / Smoke Carrier
    ar04_ammo = presets["NOTIONAL-AR-01"][0].model_copy(deep=True)
    ar04_ammo.id = "NOTIONAL-AR-04"
    ar04_ammo.class_id = "AR-04"
    ar04_ammo.name = "155mm Notional Base-Eject Illumination / Smoke Carrier (NOTIONAL)"
    ar04_ammo.description = "Notional 155mm carrier shell with electronic time fuze (FZ-05) for high-altitude base-ejection of illumination/smoke canisters."
    ar04_ammo.fuze = FuzeSpec(
        type="time",
        reliability=BetaReliabilitySpec(a=95.0, b=5.0),
        time_set_s=28.0,
        time_sigma_s=0.12,
        hob=NormalSpec(mu=450.0, sigma=18.0),
    )
    ar04_scn = presets["NOTIONAL-AR-01"][1].model_copy(deep=True)
    ar04_scn.name = "11 km Time-Fuzed Base-Ejection Carrier Mission"
    ar04_scn.aim.range_m = 11000.0
    ar04_scn.n_occasions = 30
    ar04_scn.rounds_per_occasion = 4
    presets["NOTIONAL-AR-04"] = (ar04_ammo, ar04_scn)

    # 19. RK-02: 84mm Notional Shoulder-Launched Rocket-Assisted HEAT
    presets["NOTIONAL-RK-02"] = (
        AmmoDefinition(
            id="NOTIONAL-RK-02",
            class_id="RK-02",
            name="84mm Notional Shoulder-Launched Rocket-Assisted HEAT (NOTIONAL)",
            description="Notional infantry anti-armour rocket combining low recoil launch, short in-flight rocket sustain burn (EB-13), crosswind weathercocking, and shaped-charge jet perforation.",
            calibre_m=0.084,
            mass_kg=3.4,
            mass_sigma_kg=0.03,
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.22, sigma=0.035)),
            launch=LaunchSpec(
                mode="rocket",
                mv=MVSpec(mu=215.0, sigma_rr=3.5, sigma_lot=2.0),
                twist_cal=0.0,
                rocket_burn_s=0.55,
                rocket_mdot_kg_s=0.65,
                rocket_ve_mps=1850.0,
            ),
            fuze=FuzeSpec(type="PIBD", reliability=BetaReliabilitySpec(a=96.0, b=4.0), graze_threshold_deg=18.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Octol 75/25", mass_kg=0.72),
                case=CaseSpec(mass_kg=1.5, t_in=0.16, d_in=3.3, length_m=0.22),
                lethal_radius_m=8.5,
            ),
            penetrator=PenetratorSpec(
                type="shaped_charge",
                jet_length_m=0.42,
                jet_rho_kg_m3=8960.0,
                standoff_cd=3.8,
            ),
            sources=["S01", "S10", "S13"],
        ),
        Scenario(
            name="350m Shoulder-Launched Rocket-Assisted Anti-Armour Engagement",
            target=TargetSpec(
                type="armour",
                posture="armoured",
                geometry={"half_width_x_m": 1.15, "half_width_y_m": 1.6, "armour_thickness_m": 0.36, "obliquity_deg": 25.0},
            ),
            aim=AimSpec(range_m=350.0),
            met=MetSpec(wind_cross_right_mps=3.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.8, aim_qe_bias_sigma_mils=0.8),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=1.4, aim_qe_rand_sigma_mils=1.4, jump_az_sigma_mils=1.8, jump_qe_sigma_mils=1.6),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=45,
            rounds_per_occasion=2,
        ),
    )

    # 20. MS-01: Notional Precision Guided Anti-Tank / Strike Missile
    presets["NOTIONAL-MS-01"] = (
        AmmoDefinition(
            id="NOTIONAL-MS-01",
            class_id="MS-01",
            name="152mm Notional Precision Guided Anti-Armour Missile (NOTIONAL)",
            description="Notional boost-sustain guided anti-armour missile with terminal seeker guidance plugin (1.2m guided CEP with 4% seeker lock-loss fallback) and tandem shaped-charge warhead.",
            calibre_m=0.152,
            mass_kg=18.5,
            mass_sigma_kg=0.12,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.05, sigma=0.02)),
            launch=LaunchSpec(
                mode="rocket",
                mv=MVSpec(mu=55.0, sigma_rr=1.5, sigma_lot=1.0),
                twist_cal=0.0,
                rocket_burn_s=3.6,
                rocket_mdot_kg_s=1.45,
                rocket_ve_mps=2100.0,
            ),
            fuze=FuzeSpec(type="PIBD", reliability=BetaReliabilitySpec(a=98.0, b=2.0), graze_threshold_deg=12.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="HMX", mass_kg=3.4),
                case=CaseSpec(mass_kg=6.2, t_in=0.22, d_in=5.98, length_m=0.38),
                lethal_radius_m=16.0,
                carleton_rx_m=16.0,
                carleton_ry_m=16.0,
                auw_kg=18.5,
            ),
            penetrator=PenetratorSpec(
                type="shaped_charge",
                jet_length_m=0.74,
                jet_rho_kg_m3=8960.0,
                standoff_cd=4.2,
            ),
            guidance_plugin="course_correction",
            guidance_params={"cep_guided_m": 1.2, "max_authority_m": 350.0, "p_guidance_fail": 0.04, "gnss_bias_sigma_m": 0.35},
            sources=["S01", "S10", "S13"],
        ),
        Scenario(
            name="4000m Precision Guided Anti-Armour Missile Engagement",
            target=TargetSpec(
                type="armour",
                posture="armoured",
                geometry={"half_width_x_m": 1.3, "half_width_y_m": 1.8, "armour_thickness_m": 0.62, "obliquity_deg": 30.0},
            ),
            aim=AimSpec(range_m=4000.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=1.2, aim_qe_bias_sigma_mils=1.2),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=2.0, aim_qe_rand_sigma_mils=2.0, jump_az_sigma_mils=3.0, jump_qe_sigma_mils=2.5),
            ),
            fidelity="L1",
            dt_s=0.01,
            n_occasions=40,
            rounds_per_occasion=2,
        ),
    )

    # 21. AB-02: 900 kg Notional Hard-Target Penetrator Bomb
    presets["NOTIONAL-AB-02"] = (
        AmmoDefinition(
            id="NOTIONAL-AB-02",
            class_id="AB-02",
            name="900 kg Class Notional Hard-Target Penetrator Bomb (NOTIONAL)",
            description="Notional thick-walled steel case penetrator bomb with smart delay fuze, GNSS/INS guidance plugin, and high-confinement blast/fragmentation.",
            calibre_m=0.370,
            mass_kg=890.0,
            mass_sigma_kg=2.5,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=0.86, sigma=0.018)),
            launch=LaunchSpec(
                mode="bomb",
                mv=MVSpec(mu=260.0, sigma_rr=1.2, sigma_lot=0.5),
                twist_cal=0.0,
                release_alt_m=4500.0,
                release_tas_mps=260.0,
                release_dive_rad=0.2618,  # 15 deg dive
            ),
            fuze=FuzeSpec(type="delay", reliability=BetaReliabilitySpec(a=97.5, b=2.5), graze_threshold_deg=15.0, min_arm_vel_mps=120.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Tritonal", mass_kg=240.0),
                case=CaseSpec(mass_kg=620.0, t_in=1.45, d_in=14.5, length_m=2.4),
                lethal_radius_m=85.0,
                carleton_rx_m=88.0,
                carleton_ry_m=82.0,
                auw_kg=890.0,
            ),
            guidance_plugin="course_correction",
            guidance_params={"cep_guided_m": 6.5, "max_authority_m": 450.0, "p_guidance_fail": 0.025, "gnss_bias_sigma_m": 1.8},
            sources=["S01", "S02", "S10"],
        ),
        Scenario(
            name="4500m Release Hard-Target Guided Penetrator Delivery",
            target=TargetSpec(type="structure", posture="brick_house", geometry={"half_width_x_m": 18.0, "half_width_y_m": 18.0, "exposed_site_distance_m": 450.0}),
            aim=AimSpec(range_m=6200.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(tle_range_sigma_m=4.5, tle_deflection_sigma_m=4.5),
                random=RandomErrorsSpec(release_alt_sigma_m=10.0, release_tas_sigma_mps=1.5, release_time_sigma_s=0.05),
            ),
            fidelity="L1",
            dt_s=0.02,
            n_occasions=35,
            rounds_per_occasion=2,
        ),
    )

    # 22. AB-03: Notional Air-Delivered Cluster Submunition Dispenser
    presets["NOTIONAL-AB-03"] = (
        AmmoDefinition(
            id="NOTIONAL-AB-03",
            class_id="AB-03",
            name="400 kg Class Notional Air-Delivered Submunition Dispenser (NOTIONAL)",
            description="Notional tactical air-dropped dispenser releasing 140 submunitions with FZ-07 self-destruct redundancy and UXO footprint modeling.",
            calibre_m=0.400,
            mass_kg=415.0,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.12, sigma=0.025)),
            launch=LaunchSpec(
                mode="bomb",
                mv=MVSpec(mu=235.0, sigma_rr=1.5, sigma_lot=0.8),
                twist_cal=0.0,
                release_alt_m=2400.0,
                release_tas_mps=235.0,
                release_dive_rad=0.1396,
            ),
            fuze=FuzeSpec(
                type="submunition",
                reliability=BetaReliabilitySpec(a=96.0, b=4.0),
                sd_fail_prob=0.12,
                common_cause_beta=0.04,
                submunition_count=140,
                submunition_pattern_radius_m=95.0,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="RDX", mass_kg=42.0),
                case=CaseSpec(mass_kg=180.0, t_in=0.28, d_in=15.7, length_m=1.8),
                lethal_radius_m=68.0,
                carleton_rx_m=75.0,
                carleton_ry_m=65.0,
                auw_kg=415.0,
            ),
            sources=["S01", "S10", "S24"],
        ),
        Scenario(
            name="2400m Release Dispenser Footprint & UXO Contamination Study",
            target=TargetSpec(type="area", posture="standing", geometry={"half_width_x_m": 90.0, "half_width_y_m": 90.0, "presented_area_m2": 0.56}),
            aim=AimSpec(range_m=3800.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(tle_range_sigma_m=10.0, tle_deflection_sigma_m=10.0),
                random=RandomErrorsSpec(release_alt_sigma_m=12.0, release_tas_sigma_mps=2.0, release_time_sigma_s=0.08),
            ),
            fidelity="L1",
            dt_s=0.02,
            n_occasions=25,
            rounds_per_occasion=2,
        ),
    )

    # 23. GR-02: 40mm Notional High-Velocity Dual-Purpose HEDP Grenade
    presets["NOTIONAL-GR-02"] = (
        AmmoDefinition(
            id="NOTIONAL-GR-02",
            class_id="GR-02",
            name="40x53mm Notional High-Velocity HEDP Automatic Grenade (NOTIONAL)",
            description="Notional spin-stabilised 40mm high-velocity dual-purpose grenade combining a small shaped-charge liner (TP-02) with pre-fragmented steel body (TF-04) and self-destruct fuze.",
            calibre_m=0.040,
            mass_kg=0.245,
            mass_sigma_kg=0.002,
            length_m=0.112,
            inertia=InertiaSpec(ix=3.8e-5, iy=2.1e-4),
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.14, sigma=0.025)),
            launch=LaunchSpec(mode="gun", mv=MVSpec(mu=242.0, sigma_rr=2.2, sigma_lot=1.2), twist_cal=30.0),
            fuze=FuzeSpec(
                type="PIBD",
                reliability=BetaReliabilitySpec(a=96.0, b=4.0),
                sd_fail_prob=0.03,
                common_cause_beta=0.02,
                graze_threshold_deg=16.0,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Octol 75/25", mass_kg=0.042),
                case=CaseSpec(mass_kg=0.165, t_in=0.14, d_in=1.57, length_m=0.065),
                lethal_radius_m=5.0,
                carleton_rx_m=5.0,
                carleton_ry_m=5.0,
                auw_kg=0.245,
            ),
            penetrator=PenetratorSpec(
                type="shaped_charge",
                jet_length_m=0.095,
                jet_rho_kg_m3=8960.0,
                standoff_cd=2.5,
            ),
            sources=["S01", "S10", "S24"],
        ),
        Scenario(
            name="1100m Automatic Grenade Launcher Burst Engagement",
            target=TargetSpec(
                type="personnel",
                posture="standing",
                geometry={"half_width_x_m": 10.0, "half_width_y_m": 10.0, "presented_area_m2": 0.56, "armour_thickness_m": 0.05, "obliquity_deg": 20.0},
            ),
            aim=AimSpec(range_m=1100.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.8, aim_qe_bias_sigma_mils=0.8),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=1.5, aim_qe_rand_sigma_mils=1.5, jump_az_sigma_mils=1.2, jump_qe_sigma_mils=1.2),
            ),
            fidelity="L1",
            dt_s=0.008,
            n_occasions=35,
            rounds_per_occasion=6,
        ),
    )

    # 24. NV-01: 127mm Notional Naval Gun Multi-Role HE-VT
    presets["NOTIONAL-NV-01"] = (
        AmmoDefinition(
            id="NOTIONAL-NV-01",
            class_id="NV-01",
            name="127mm (5-inch) Notional Naval Gun Multi-Role HE-VT (NOTIONAL)",
            description="Notional 127mm naval gun spin-stabilised projectile modeled with STANAG 4355 MPM (L2), radar proximity airburst fuze (FZ-05), and high-velocity naval gun ballistics.",
            calibre_m=0.127,
            mass_kg=31.8,
            mass_sigma_kg=0.14,
            length_m=0.66,
            inertia=InertiaSpec(ix=0.072, iy=0.64),
            aero=AeroSpec(
                drag_ref="G7",
                form_factor=NormalSpec(mu=0.98, sigma=0.015, sigma_bias=0.01),
                mpm_coeffs=MPMCoeffsSpec(c_da2=3.9, c_la=2.05, c_ma=2.70, c_mag=0.22, c_spin=-0.013),
            ),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=808.0, sigma_rr=1.6, sigma_lot=1.1),
                twist_cal=22.0,
                temp_coeff_mps_per_C=0.40,
            ),
            fuze=FuzeSpec(
                type="proximity",
                reliability=BetaReliabilitySpec(a=98.0, b=2.0),
                hob=NormalSpec(mu=6.5, sigma=1.1),
                sd_fail_prob=0.02,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="PBXN-109", mass_kg=4.2),
                case=CaseSpec(mass_kg=25.4, t_in=0.52, d_in=5.00, length_m=0.52),
                mott_B=0.0779,
                lethal_radius_m=24.5,
                carleton_rx_m=26.0,
                carleton_ry_m=23.0,
                auw_kg=31.8,
            ),
            sources=["S01", "S02", "S10", "S14", "S32"],
        ),
        Scenario(
            name="18 km Naval Gun Surface Fire Support Salvo (L2 MPM)",
            target=TargetSpec(
                type="area",
                posture="standing",
                geometry={"half_width_x_m": 45.0, "half_width_y_m": 45.0, "presented_area_m2": 0.56},
            ),
            aim=AimSpec(range_m=18000.0, az_rad=0.20),
            met=MetSpec(wind_range_mps=3.0, wind_cross_right_mps=4.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(
                    tle_range_sigma_m=10.0,
                    tle_deflection_sigma_m=10.0,
                    aim_az_bias_sigma_mils=0.45,
                    aim_qe_bias_sigma_mils=0.45,
                    wind_range_bias_sigma_mps=1.0,
                    wind_cross_bias_sigma_mps=1.0,
                ),
                random=RandomErrorsSpec(
                    aim_az_rand_sigma_mils=0.65,
                    aim_qe_rand_sigma_mils=0.65,
                ),
            ),
            fidelity="L2",
            dt_s=0.02,
            n_occasions=35,
            rounds_per_occasion=5,
        ),
    )

    # 25. RK-03: 227mm Notional Heavy MLRS Extended-Range Rocket
    presets["NOTIONAL-RK-03"] = (
        AmmoDefinition(
            id="NOTIONAL-RK-03",
            class_id="RK-01",
            name="227mm Notional Heavy MLRS Extended-Range Rocket (NOTIONAL)",
            description="Notional heavy multiple-launch artillery rocket with high-thrust composite solid motor boost (EB-13), wrap-around tail fins, and 90 kg unitary HE-FRAG warhead.",
            calibre_m=0.227,
            mass_kg=306.0,
            mass_sigma_kg=1.8,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.12, sigma=0.025)),
            launch=LaunchSpec(
                mode="rocket",
                mv=MVSpec(mu=58.0, sigma_rr=1.8, sigma_lot=1.2),
                twist_cal=0.0,
                rocket_burn_s=4.2,
                rocket_mdot_kg_s=24.5,
                rocket_ve_mps=2350.0,
            ),
            fuze=FuzeSpec(
                type="proximity",
                reliability=BetaReliabilitySpec(a=97.0, b=3.0),
                hob=NormalSpec(mu=10.0, sigma=1.5),
                sd_fail_prob=0.02,
            ),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="PBXN-109", mass_kg=23.0),
                case=CaseSpec(mass_kg=67.0, t_in=0.48, d_in=8.94, length_m=0.88),
                lethal_radius_m=42.0,
                carleton_rx_m=44.0,
                carleton_ry_m=42.0,
                auw_kg=306.0,
            ),
            sources=["S01", "S10", "S13", "S24"],
        ),
        Scenario(
            name="32 km Heavy MLRS Rocket Battery Salvo",
            target=TargetSpec(
                type="area",
                posture="standing",
                geometry={"half_width_x_m": 150.0, "half_width_y_m": 150.0, "presented_area_m2": 0.56},
            ),
            aim=AimSpec(range_m=32000.0),
            met=MetSpec(wind_range_mps=2.5, wind_cross_right_mps=4.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=1.2, aim_qe_bias_sigma_mils=1.2, wind_cross_bias_sigma_mps=1.2),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=2.0, aim_qe_rand_sigma_mils=2.0, jump_az_sigma_mils=3.2, jump_qe_sigma_mils=2.8),
            ),
            fidelity="L1",
            dt_s=0.02,
            n_occasions=25,
            rounds_per_occasion=6,
        ),
    )

    # 26. RK-04: 70mm (2.75-in) Notional Folding-Fin Aerial/Ground Rocket
    presets["NOTIONAL-RK-04"] = (
        AmmoDefinition(
            id="NOTIONAL-RK-04",
            class_id="RK-01",
            name="70mm (2.75-in) Notional Folding-Fin Tactical Rocket (NOTIONAL)",
            description="Notional 70mm unguided folding-fin rocket with rapid 1.15s double-base solid motor boost (EB-13), fin deployment tip-off dispersion, and 4.5 kg HE-FRAG warhead.",
            calibre_m=0.070,
            mass_kg=10.6,
            mass_sigma_kg=0.08,
            aero=AeroSpec(drag_ref="G7", form_factor=NormalSpec(mu=1.16, sigma=0.03)),
            launch=LaunchSpec(
                mode="rocket",
                mv=MVSpec(mu=48.0, sigma_rr=2.0, sigma_lot=1.2),
                twist_cal=0.0,
                rocket_burn_s=1.15,
                rocket_mdot_kg_s=3.4,
                rocket_ve_mps=2050.0,
            ),
            fuze=FuzeSpec(type="PD", reliability=BetaReliabilitySpec(a=96.0, b=4.0), graze_threshold_deg=14.0),
            warhead=WarheadSpec(
                fill=FillSpec(explosive="Composition B", mass_kg=1.05),
                case=CaseSpec(mass_kg=3.2, t_in=0.22, d_in=2.75, length_m=0.26),
                lethal_radius_m=14.0,
                carleton_rx_m=15.0,
                carleton_ry_m=14.0,
                auw_kg=10.6,
            ),
            sources=["S01", "S10", "S13"],
        ),
        Scenario(
            name="4500m Folding-Fin Rocket Pod Salvo",
            target=TargetSpec(
                type="area",
                posture="standing",
                geometry={"half_width_x_m": 55.0, "half_width_y_m": 55.0, "presented_area_m2": 0.56},
            ),
            aim=AimSpec(range_m=4500.0),
            met=MetSpec(wind_cross_right_mps=3.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=1.8, aim_qe_bias_sigma_mils=1.8),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=2.8, aim_qe_rand_sigma_mils=2.8, jump_az_sigma_mils=4.5, jump_qe_sigma_mils=4.0),
            ),
            fidelity="L1",
            dt_s=0.01,
            n_occasions=30,
            rounds_per_occasion=7,
        ),
    )

    # 27. SG-01: 12-Gauge 00-Buckshot Tactical Load
    presets["NOTIONAL-SG-01"] = (
        AmmoDefinition(
            id="NOTIONAL-SG-01",
            class_id="SG-01",
            name="12-Gauge Notional 00-Buckshot Tactical (9 Pellets)",
            description="Notional 12-gauge 2-3/4\" combat buckshot load firing 9 lead pellets (8.38mm) at 405 m/s with choke dispersion.",
            calibre_m=0.01852,
            mass_kg=0.0314,
            mass_sigma_kg=0.0005,
            length_m=0.070,
            inertia=InertiaSpec(ix=1.5e-7, iy=1.5e-6),
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.05, sigma=0.04)),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=405.0, sigma_rr=5.0, sigma_lot=3.0),
                twist_cal=0.0,
                temp_coeff_mps_per_C=0.25,
            ),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="bullet_barrier",
                L_m=0.00838,
                D_m=0.00838,
                rho=11340.0,
                v50_mps=280.0,
                sigma_v50_mps=15.0,
            ),
            sources=["S13", "Lowry1970"],
        ),
        Scenario(
            name="35m Shotgun Tactical Target (30-in Circle & E-Type)",
            target=TargetSpec(
                type="personnel",
                posture="standing",
                geometry={"half_width_x_m": 0.225, "half_width_y_m": 0.225, "presented_area_m2": 0.45},
            ),
            aim=AimSpec(range_m=35.0, az_rad=0.0),
            met=MetSpec(wind_cross_right_mps=1.5),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.5, aim_qe_bias_sigma_mils=0.5),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=1.5, aim_qe_rand_sigma_mils=1.5),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=50,
            rounds_per_occasion=1,
        ),
    )

    # 28. SG-02: 12-Gauge #7.5 Target Clay / Skeet Load
    presets["NOTIONAL-SG-02"] = (
        AmmoDefinition(
            id="NOTIONAL-SG-02",
            class_id="SG-02",
            name="12-Gauge Notional #7.5 Clay / Trap Target (350 Pellets)",
            description="Notional 12-gauge 1-1/8 oz competition target load firing 350 #7.5 lead birdshot pellets (2.41mm) at 395 m/s.",
            calibre_m=0.01852,
            mass_kg=0.0318,
            mass_sigma_kg=0.0003,
            length_m=0.070,
            inertia=InertiaSpec(ix=1.5e-7, iy=1.5e-6),
            aero=AeroSpec(drag_ref="G1", form_factor=NormalSpec(mu=1.12, sigma=0.05)),
            launch=LaunchSpec(
                mode="gun",
                mv=MVSpec(mu=395.0, sigma_rr=4.0, sigma_lot=2.5),
                twist_cal=0.0,
                temp_coeff_mps_per_C=0.20,
            ),
            fuze=FuzeSpec(type="none", reliability=BetaReliabilitySpec(a=999.0, b=0.001, p_dud_mean=0.0)),
            penetrator=PenetratorSpec(
                type="bullet_barrier",
                L_m=0.00241,
                D_m=0.00241,
                rho=11340.0,
                v50_mps=160.0,
                sigma_v50_mps=10.0,
            ),
            sources=["S13", "Lowry1970"],
        ),
        Scenario(
            name="25m Clay Pigeon Breaking Arena",
            target=TargetSpec(
                type="personnel",
                posture="standing",
                geometry={"half_width_x_m": 0.055, "half_width_y_m": 0.055, "presented_area_m2": 0.01},
            ),
            aim=AimSpec(range_m=25.0, az_rad=0.0),
            met=MetSpec(wind_cross_right_mps=2.0),
            errors=ErrorBudgetSpec(
                bias=BiasErrorsSpec(aim_az_bias_sigma_mils=0.4, aim_qe_bias_sigma_mils=0.4),
                random=RandomErrorsSpec(aim_az_rand_sigma_mils=1.2, aim_qe_rand_sigma_mils=1.2),
            ),
            fidelity="L1",
            dt_s=0.005,
            n_occasions=50,
            rounds_per_occasion=1,
        ),
    )

    return presets


PRESET_CATALOG: dict[str, tuple[AmmoDefinition, Scenario]] = _build_presets()


def list_presets() -> list[dict[str, Any]]:
    """Return metadata for all shipped notional presets."""
    out = []
    for pid, (ammo, scn) in PRESET_CATALOG.items():
        out.append({
            "preset_id": pid,
            "class_id": ammo.class_id,
            "ammo_name": ammo.name,
            "scenario_name": scn.name,
            "fidelity": scn.fidelity,
            "notional": ammo.notional,
            "description": ammo.description,
        })
    return out


def get_preset(preset_id: str) -> tuple[AmmoDefinition, Scenario]:
    """Return deep copies of `(AmmoDefinition, Scenario)` for a given `preset_id`."""
    if preset_id not in PRESET_CATALOG:
        raise KeyError(f"Unknown preset '{preset_id}'. Available: {list(PRESET_CATALOG.keys())}")
    ammo, scn = PRESET_CATALOG[preset_id]
    return ammo.model_copy(deep=True), scn.model_copy(deep=True)
