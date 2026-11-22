"""Extract all reference tables from 'Ammunition Monte Carlo Knowledge Base.xlsx' into JSON.

Every exported record carries `source_id` and `verification_status` per App_Spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import openpyxl


def _clean(val: Any) -> Any:
    if isinstance(val, str):
        return val.strip()
    return val


def export_workbook(xlsx_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    # 1. Sources
    ws_src = wb["Sources"]
    sources = []
    for r in range(6, ws_src.max_row + 1):
        sid = _clean(ws_src.cell(r, 2).value)
        if not sid:
            continue
        sources.append({
            "source_id": sid,
            "title": _clean(ws_src.cell(r, 3).value),
            "type": _clean(ws_src.cell(r, 4).value),
            "url": _clean(ws_src.cell(r, 5).value),
            "used_for": _clean(ws_src.cell(r, 6).value),
            "verification_status": "VERIFIED-SRC",
        })

    # 2. Ammo_Taxonomy
    ws_tax = wb["Ammo_Taxonomy"]
    taxonomy = []
    for r in range(6, ws_tax.max_row + 1):
        cid = _clean(ws_tax.cell(r, 2).value)
        if not cid:
            continue
        req_ids_raw = _clean(ws_tax.cell(r, 11).value) or ""
        taxonomy.append({
            "class_id": cid,
            "family": _clean(ws_tax.cell(r, 3).value),
            "ammo_type": _clean(ws_tax.cell(r, 4).value),
            "calibre_mass_class": _clean(ws_tax.cell(r, 5).value),
            "launch_model": _clean(ws_tax.cell(r, 6).value),
            "flight_model": _clean(ws_tax.cell(r, 7).value),
            "stabilisation": _clean(ws_tax.cell(r, 8).value),
            "fuze_types": _clean(ws_tax.cell(r, 9).value),
            "terminal_mechanisms": _clean(ws_tax.cell(r, 10).value),
            "required_formula_ids": req_ids_raw,
            "dominant_stochastic_inputs": _clean(ws_tax.cell(r, 12).value),
            "key_mc_outputs": _clean(ws_tax.cell(r, 13).value),
            "source_id": "S01;S13;S26;S27",
            "verification_status": "STD-FORM",
        })

    # 3. Formulas
    ws_f = wb["Formulas"]
    formulas = []
    for r in range(6, ws_f.max_row + 1):
        fid = _clean(ws_f.cell(r, 2).value)
        if not fid:
            continue
        vstatus_raw = _clean(ws_f.cell(r, 13).value) or "STD-FORM"
        is_flag = "FLAG" in vstatus_raw.upper()
        formulas.append({
            "formula_id": fid,
            "module": _clean(ws_f.cell(r, 3).value),
            "name": _clean(ws_f.cell(r, 4).value),
            "equation_math": _clean(ws_f.cell(r, 5).value),
            "code_form": _clean(ws_f.cell(r, 6).value),
            "variables_units": _clean(ws_f.cell(r, 7).value),
            "validity_assumptions": _clean(ws_f.cell(r, 8).value),
            "mc_role": _clean(ws_f.cell(r, 9).value),
            "applies_to_classes": _clean(ws_f.cell(r, 10).value),
            "source_id": _clean(ws_f.cell(r, 11).value),
            "source_url": _clean(ws_f.cell(r, 12).value),
            "verification_status": vstatus_raw,
            "is_flagged": is_flag,
        })

    # 4. Drag_Tables
    ws_drag = wb["Drag_Tables"]
    g1_table = []
    g7_table = []
    frag_table = []
    for r in range(6, ws_drag.max_row + 1):
        m_g1 = ws_drag.cell(r, 2).value
        cd_g1 = ws_drag.cell(r, 3).value
        if m_g1 is not None and cd_g1 is not None:
            g1_table.append({
                "mach": float(m_g1),
                "cd": float(cd_g1),
                "source_id": "S15",
                "verification_status": "VERIFIED-SRC",
            })
        m_g7 = ws_drag.cell(r, 5).value
        cd_g7 = ws_drag.cell(r, 6).value
        if m_g7 is not None and cd_g7 is not None:
            g7_table.append({
                "mach": float(m_g7),
                "cd": float(cd_g7),
                "source_id": "S15",
                "verification_status": "VERIFIED-SRC",
            })
        m_from = ws_drag.cell(r, 8).value
        m_to = ws_drag.cell(r, 9).value
        cd_frag = ws_drag.cell(r, 10).value
        if isinstance(m_from, (int, float)) and isinstance(m_to, (int, float)) and isinstance(cd_frag, (int, float)):
            is_last_band = float(m_from) >= 5.5
            frag_table.append({
                "mach_from": float(m_from),
                "mach_to": float(m_to),
                "cd": float(cd_frag),
                "source_id": "S01",
                "verification_status": (
                    "VERIFIED-SRC (band 5.6-11.2 interpreted as contiguous from printed '5. - 11.2')"
                    if is_last_band else "VERIFIED-SRC"
                ),
            })
    drag_tables = {
        "G1": g1_table,
        "G7": g7_table,
        "natural_fragment": frag_table,
    }

    # 5. KB_Coefficients
    ws_kb = wb["KB_Coefficients"]
    kb_pieces = []
    param_map = {
        "Incident pressure Ps": "Ps",
        "Incident impulse Is": "Is",
        "Reflected pressure Pr": "Pr",
        "Reflected impulse Ir": "Ir",
        "Time of arrival ta": "ta",
        "Shock front velocity U": "U",
        "Positive duration td": "td",
    }
    for r in range(15, 25):
        pname = _clean(ws_kb.cell(r, 2).value)
        if not pname:
            continue
        coeffs = []
        for c in range(10, 25):
            val = ws_kb.cell(r, c).value
            coeffs.append(float(val) if val is not None else 0.0)
        # Trim trailing zeros while keeping at least C0
        while len(coeffs) > 1 and coeffs[-1] == 0.0:
            coeffs.pop()
        is_corrected_row = (r == 23)
        kb_pieces.append({
            "parameter_name": pname,
            "param_key": param_map[pname],
            "unit": _clean(ws_kb.cell(r, 3).value),
            "z_min": float(ws_kb.cell(r, 4).value),
            "z_max": float(ws_kb.cell(r, 5).value),
            "scale_by_w_cbrt": (_clean(ws_kb.cell(r, 6).value) == "Yes"),
            "multiplier": float(ws_kb.cell(r, 7).value),
            "k0": float(ws_kb.cell(r, 8).value),
            "k1": float(ws_kb.cell(r, 9).value),
            "c": coeffs,
            "source_id": "S01;S02;S40" if is_corrected_row else "S01;S02",
            "verification_status": (
                "VERIFIED-SRC (K0 sign corrected to -2.12492525216 per S40; validated V-02c/V-03b)"
                if is_corrected_row else "VERIFIED-SRC"
            ),
        })

    # 6. Explosives
    ws_exp = wb["Explosives"]
    tnt_equiv = []
    for r in range(6, 19):
        name = _clean(ws_exp.cell(r, 2).value)
        if not name:
            continue
        fp = ws_exp.cell(r, 3).value
        fi = ws_exp.cell(r, 4).value
        tnt_equiv.append({
            "explosive": name,
            "f_pressure": float(fp) if fp is not None else 1.0,
            "f_impulse": float(fi) if fi is not None else (float(fp) if fp is not None else 1.0),
            "pressure_range_mpa": _clean(ws_exp.cell(r, 5).value),
            "source_id": "S01;S11",
            "verification_status": "VERIFIED-SRC",
        })

    gurney_iatg = []
    for r in range(22, 31):
        name = _clean(ws_exp.cell(r, 2).value)
        if not name:
            continue
        gurney_iatg.append({
            "explosive": name,
            "density_g_cm3": float(ws_exp.cell(r, 3).value),
            "detonation_velocity_mps": float(ws_exp.cell(r, 4).value),
            "gurney_mps": float(ws_exp.cell(r, 5).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })

    gurney_alt = []
    for r in range(34, 47):
        name = _clean(ws_exp.cell(r, 2).value)
        if not name:
            continue
        gurney_alt.append({
            "explosive": name,
            "density_g_cm3": float(ws_exp.cell(r, 3).value),
            "detonation_velocity_mps": float(ws_exp.cell(r, 4).value),
            "gurney_mps": float(ws_exp.cell(r, 5).value),
            "source_id": "S05",
            "verification_status": "VERIFIED-SRC",
        })

    mott_table = []
    for r in range(50, 56):
        name = _clean(ws_exp.cell(r, 2).value)
        if not name:
            continue
        mott_table.append({
            "explosive": name,
            "mott_b": float(ws_exp.cell(r, 3).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })

    ufc_table_e = []
    for r in range(59, 71):
        name = _clean(ws_exp.cell(r, 2).value)
        if not name:
            continue
        ufc_table_e.append({
            "explosive": name,
            "printed_value_mj_kg": float(ws_exp.cell(r, 3).value),
            "hd_tnt_equiv": float(ws_exp.cell(r, 4).value),
            "source_id": "S01;S11",
            "verification_status": "FLAG - column header ambiguity in IATG Table E (see Explosives R57)",
        })

    explosives_data = {
        "tnt_equivalence": tnt_equiv,
        "gurney_iatg": gurney_iatg,
        "gurney_alt_s05": gurney_alt,
        "mott_constants": mott_table,
        "ufc_heat_of_detonation": ufc_table_e,
    }

    # 7. THOR_Coefficients
    ws_thor = wb["THOR_Coefficients"]
    thor_materials = []
    for r in range(6, 23):
        mat = _clean(ws_thor.cell(r, 2).value)
        if not mat:
            continue
        thor_materials.append({
            "material": mat,
            "c1": float(ws_thor.cell(r, 3).value),
            "alpha1": float(ws_thor.cell(r, 4).value),
            "beta1": float(ws_thor.cell(r, 5).value),
            "gamma1": float(ws_thor.cell(r, 6).value),
            "lambda1": float(ws_thor.cell(r, 7).value),
            "c2": float(ws_thor.cell(r, 8).value),
            "alpha2": float(ws_thor.cell(r, 9).value),
            "beta2": float(ws_thor.cell(r, 10).value),
            "gamma2": float(ws_thor.cell(r, 11).value),
            "lambda2": float(ws_thor.cell(r, 12).value),
            "source_id": "S07;S08",
            "verification_status": "FLAG - verify against original Project THOR report (see V-12)",
        })
    thor_data = {
        "materials": thor_materials,
        "shape_factors_cm2_g23": {
            "sphere": 0.3079,
            "cube": 0.3799,
            "random": 0.5199,
        },
    }

    # 8. Penetration
    ws_pen = wb["Penetration"]
    lanz_odermatt = []
    for r in range(6, 10):
        mat = _clean(ws_pen.cell(r, 2).value)
        if not mat:
            continue
        lanz_odermatt.append({
            "rod_material": mat,
            "mode": _clean(ws_pen.cell(r, 3).value),
            "a": float(ws_pen.cell(r, 4).value),
            "c0": float(ws_pen.cell(r, 5).value),
            "c1": float(ws_pen.cell(r, 6).value) if ws_pen.cell(r, 6).value is not None else None,
            "k_alpha": float(ws_pen.cell(r, 7).value) if ws_pen.cell(r, 7).value is not None else None,
            "n_beta": float(ws_pen.cell(r, 8).value) if ws_pen.cell(r, 8).value is not None else None,
            "b0": float(ws_pen.cell(r, 9).value),
            "b1": float(ws_pen.cell(r, 10).value),
            "m": float(ws_pen.cell(r, 11).value),
            "source_id": "S18;S39",
            "verification_status": "FLAG - s^2 unit convention inferred; UHSS b0 9874 vs 9876",
        })
    de_marre = [
        {
            "variant": "Wrought iron (1870s)",
            "equation": _clean(ws_pen.cell(14, 3).value),
            "units": _clean(ws_pen.cell(14, 4).value),
            "constant": float(ws_pen.cell(14, 5).value),
            "source_id": "S19",
            "verification_status": "VERIFIED-SRC",
        },
        {
            "variant": "Nickel-steel (c.1890)",
            "equation": _clean(ws_pen.cell(15, 3).value),
            "units": _clean(ws_pen.cell(15, 4).value),
            "constant": float(ws_pen.cell(15, 5).value),
            "source_id": "S19",
            "verification_status": "VERIFIED-SRC",
        },
    ]
    penetration_data = {
        "lanz_odermatt": lanz_odermatt,
        "de_marre": de_marre,
    }

    # 9. Effects_Criteria
    ws_eff = wb["Effects_Criteria"]
    scalar_criteria = []
    for r in range(6, 14):
        scalar_criteria.append({
            "criterion": _clean(ws_eff.cell(r, 2).value),
            "value": float(ws_eff.cell(r, 3).value),
            "unit": _clean(ws_eff.cell(r, 4).value),
            "source_clause": _clean(ws_eff.cell(r, 5).value),
            "source_id": "S01" if "S01" in str(ws_eff.cell(r, 5).value) else "S31;S32",
            "verification_status": "VERIFIED-SRC",
        })
    qd_z_constants = []
    for r in range(17, 21):
        qd_z_constants.append({
            "z_m_kg13": float(ws_eff.cell(r, 2).value),
            "purpose": _clean(ws_eff.cell(r, 3).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })
    brick_house_k = []
    for r in range(24, 29):
        brick_house_k.append({
            "category": _clean(ws_eff.cell(r, 2).value),
            "definition": _clean(ws_eff.cell(r, 3).value),
            "k_jarrett": float(ws_eff.cell(r, 4).value),
            "k_gilbert_lees_scilly": float(ws_eff.cell(r, 5).value),
            "p_fatality": float(ws_eff.cell(r, 6).value),
            "p_fatal_or_serious": float(ws_eff.cell(r, 7).value),
            "p_any_injury": float(ws_eff.cell(r, 8).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })
    glass_breakage = [
        {"window_size": "Small (<1 m2)", "key": "small", "c_a": -1.013, "slope_a": 3.356, "c_b": -2.558, "slope_b": 1.932, "source_id": "S01", "verification_status": "VERIFIED-SRC"},
        {"window_size": "Medium (1-3 m2)", "key": "medium", "c_a": 0.796, "slope_a": 3.356, "c_b": -0.788, "slope_b": 1.932, "source_id": "S01", "verification_status": "VERIFIED-SRC"},
        {"window_size": "Large (>3 m2)", "key": "large", "c_a": 2.674, "slope_a": 3.356, "c_b": 0.983, "slope_b": 1.932, "source_id": "S01", "verification_status": "VERIFIED-SRC"},
    ]
    axelsson_levels = []
    for r in range(51, 56):
        axelsson_levels.append({
            "injury_level": _clean(ws_eff.cell(r, 2).value),
            "asii_range": _clean(ws_eff.cell(r, 3).value),
            "v_mps_range": _clean(ws_eff.cell(r, 4).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })
    thermal_criteria = []
    for r in range(59, 63):
        thermal_criteria.append({
            "effect": _clean(ws_eff.cell(r, 2).value),
            "mean_dose": float(ws_eff.cell(r, 3).value),
            "range_dose": _clean(ws_eff.cell(r, 4).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })
    ground_shock = []
    for r in range(66, 70):
        ground_shock.append({
            "damage_degree": _clean(ws_eff.cell(r, 2).value),
            "ppv_mps": _clean(ws_eff.cell(r, 3).value),
            "source_id": "S01",
            "verification_status": "VERIFIED-SRC",
        })
    effects_criteria = {
        "scalars": scalar_criteria,
        "qd_z_constants": qd_z_constants,
        "brick_house_damage": brick_house_k,
        "glass_breakage": glass_breakage,
        "axelsson_injury_levels": axelsson_levels,
        "thermal_dose": thermal_criteria,
        "ground_shock": ground_shock,
    }

    # 10. Atmosphere reference table (0..20 km)
    ws_atm = wb["Atmosphere"]
    atm_rows = []
    for r in range(12, ws_atm.max_row + 1):
        h = ws_atm.cell(r, 2).value
        if h is None:
            continue
        atm_rows.append({
            "altitude_m": float(h),
            "T_K": float(ws_atm.cell(r, 3).value),
            "p_Pa": float(ws_atm.cell(r, 4).value),
            "rho_kg_m3": float(ws_atm.cell(r, 5).value),
            "a_mps": float(ws_atm.cell(r, 6).value),
            "rho_ratio": float(ws_atm.cell(r, 7).value),
            "source_id": "S16",
            "verification_status": "VERIFIED-SRC",
        })
    atmosphere_data = {
        "constants": {
            "R_gas": float(ws_atm.cell(7, 5).value),
            "gamma": float(ws_atm.cell(8, 5).value),
            "g0": float(ws_atm.cell(9, 5).value),
            "p0_Pa": float(ws_atm.cell(6, 5).value),
            "T0_K": 288.15,
        },
        "isa_table": atm_rows,
    }

    # 11. Validation_Cases
    ws_val = wb["Validation_Cases"]
    validation_cases = []
    for r in range(6, 28):
        tid = _clean(ws_val.cell(r, 2).value)
        if not tid:
            continue
        exp_v = ws_val.cell(r, 6).value
        xls_v = ws_val.cell(r, 7).value
        tol_v = ws_val.cell(r, 9).value
        validation_cases.append({
            "test_id": tid,
            "formula_id": _clean(ws_val.cell(r, 3).value),
            "case": _clean(ws_val.cell(r, 4).value),
            "expected_source": _clean(ws_val.cell(r, 5).value),
            "expected": float(exp_v) if exp_v is not None else None,
            "excel_value": float(xls_v) if xls_v is not None else None,
            "unit": _clean(ws_val.cell(r, 8).value),
            "rel_tolerance": float(tol_v) if tol_v is not None else None,
            "status_in_sheet": _clean(ws_val.cell(r, 11).value),
            "source_id": "S01;S02;S07;S08;S16;S17;S21;S22",
            "verification_status": "FLAG" if tid == "V-12" else "VERIFIED-SRC",
        })

    datasets = {
        "sources.json": sources,
        "ammo_taxonomy.json": taxonomy,
        "formulas.json": formulas,
        "drag_tables.json": drag_tables,
        "kb_coefficients.json": kb_pieces,
        "explosives.json": explosives_data,
        "thor_coefficients.json": thor_data,
        "penetration.json": penetration_data,
        "effects_criteria.json": effects_criteria,
        "atmosphere.json": atmosphere_data,
        "validation_cases.json": validation_cases,
    }

    for filename, payload in datasets.items():
        with open(out_dir / filename, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    return datasets


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    xlsx = root / "Ammunition Monte Carlo Knowledge Base.xlsx"
    out = root / "src" / "ammo_mc" / "data_tables"
    res = export_workbook(xlsx, out)
    print(f"Exported {len(res)} JSON tables to {out}")
