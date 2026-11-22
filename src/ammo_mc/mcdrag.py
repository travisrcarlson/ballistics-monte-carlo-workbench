"""McCoy (1974) MCDRAG drag-coefficient estimator ported verbatim from BASIC (Sheet 19: MCDRAG_Code).

Source: S15 (R. L. McCoy, BRL, December 1974, via JBM Ballistics).
Estimates zero-yaw drag CD0(M) and its components (head, skin friction, rotating band,
boattail, base, and base-pressure ratio) across 26 Mach numbers from 0.5 to 5.0.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

MCDRAG_MACH_GRID: tuple[float, ...] = (
    0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.1, 1.2, 1.3,
    1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
)


@dataclass
class McDragInputs:
    d_ref_mm: float  # D1: Reference diameter (mm)
    total_length_cal: float  # L1: Total length (calibres)
    nose_length_cal: float  # L2: Nose length (calibres)
    rt_over_r: float = 0.5  # R1: Tangent-radius / actual-ogive-radius (1=tangent ogive, 0=cone)
    boattail_length_cal: float = 0.0  # L3: Boattail length (calibres)
    base_dia_cal: float = 1.0  # D2: Base diameter (calibres)
    meplat_dia_cal: float = 0.0  # D3: Meplat diameter (calibres)
    band_dia_cal: float = 1.0  # D4: Rotating band diameter (calibres)
    xcg_nose_cal: float = 0.0  # X1: Optional CG from nose (calibres)
    boundary_layer_code: str = "L/T"  # K$: "L/L", "L/T", or "T/T"
    identification: str = "Projectile"


def estimate_mcdrag(inp: McDragInputs) -> dict[str, Any]:
    """Run McCoy's 1974 MCDRAG algorithm and return component curves + Cd table + diagnostics."""
    D1 = float(inp.d_ref_mm)
    L1 = float(inp.total_length_cal)
    L2 = float(inp.nose_length_cal)
    R1 = float(inp.rt_over_r)
    L3 = float(inp.boattail_length_cal)
    D2 = float(inp.base_dia_cal)
    D3 = float(inp.meplat_dia_cal)
    D4 = float(inp.band_dia_cal)
    K = inp.boundary_layer_code.strip().upper()
    if K not in ("L/L", "L/T", "T/T"):
        raise ValueError(f"boundary_layer_code must be 'L/L', 'L/T', or 'T/T', got '{inp.boundary_layer_code}'.")
    if D1 <= 0 or L1 <= 0 or L2 <= 0:
        raise ValueError("d_ref_mm, total_length_cal, and nose_length_cal must be positive.")

    rows: list[dict[str, float]] = []

    for M_val in MCDRAG_MACH_GRID:
        T1 = (1.0 - D3) / L2
        M2 = M_val**2
        R2 = 23296.3 * M_val * L1 * D1
        R3 = 0.4343 * math.log(R2)
        C7 = (1.328 / math.sqrt(R2)) * ((1.0 + 0.12 * M2) ** (-0.12))
        C8 = (0.455 / (R3**2.58)) * ((1.0 + 0.21 * M2) ** (-0.32))
        D5 = 1.0 + ((0.333 + (0.02 / (L2**2))) * R1)
        S1 = 1.5708 * L2 * D5 * (1.0 + (1.0 / (8.0 * (L2**2))))
        S2 = 3.1416 * (L1 - L2)
        S3 = S1 + S2

        if K == "L/L":
            C9 = 1.2732 * S3 * C7
            C10 = C9
        elif K == "L/T":
            C9 = 1.2732 * S3 * C7
            C10 = 1.2732 * S3 * C8
        else:  # "T/T"
            C9 = 1.2732 * S3 * C8
            C10 = C9

        C3 = (C9 * S1 + C10 * S2) / S3  # CDSF
        C15 = (M2 - 1.0) / (2.4 * M2)

        if M_val <= 1.0:
            P5 = (1.0 + 0.2 * M2) ** 3.5
        else:
            P5 = ((1.2 * M2) ** 3.5) * ((6.0 / (7.0 * M2 - 1.0)) ** 2.5)

        C16 = (1.122 * (P5 - 1.0) * (D3**2)) / M2
        if M_val <= 0.91:
            C18 = 0.0
        elif M_val >= 1.41:
            C18 = 0.85 * C16
        else:
            C18 = (0.254 + 2.88 * C15) * C16

        if M_val < 1.0:
            P2 = 1.0 / (1.0 + 0.1875 * M2 + 0.0531 * (M2**2))
        else:
            P2 = 1.0 / (1.0 + 0.2477 * M2 + 0.0345 * (M2**2))

        P4 = (1.0 + 0.09000001 * M2 * (1.0 - math.exp(L2 - L1))) * (1.0 + 0.25 * M2 * (1.0 - D2))
        P1 = max(0.0, P2 * P4)
        C6 = (1.4286 * (1.0 - P1) * (D2**2)) / M2  # CDB

        if M_val < 0.95:
            C4 = (M_val**12.5) * (D4 - 1.0)
        else:
            C4 = (0.21 + 0.28 / M2) * (D4 - 1.0)

        if M_val <= 1.0:
            # Subsonic and transonic speeds
            if L3 <= 0.0 or M_val <= 0.85:
                C5 = 0.0
            else:
                T2 = (1.0 - D2) / (2.0 * L3)
                T3 = 2.0 * (T2**2) + (T2**3)
                E1 = math.exp(-2.0 * L3)
                B4 = 1.0 - E1 + 2.0 * T2 * ((E1 * (L3 + 0.5)) - 0.5)
                C5 = 2.0 * T3 * B4 * (1.0 / (0.564 + 1250.0 * (C15**2)))

            X2 = (1.0 + 0.552 * (T1**0.8)) ** (-0.5) if T1 > 0 else 1.0
            if M_val <= X2 or T1 <= 0:
                C17 = 0.0
            else:
                C17 = 0.368 * (T1**1.8) + 1.6 * T1 * C15
            C2 = C17 + C18
        else:
            # Supersonic speeds
            B2 = M2 - 1.0
            B = math.sqrt(B2)
            Z = B
            S4 = 1.0 + 0.368 * (max(T1, 0.0) ** 1.85)
            if M_val < S4:
                Z = math.sqrt(S4**2 - 1.0)
            C11 = 0.7156 - 0.5313 * R1 + 0.595 * (R1**2)
            C12 = 0.0796 + 0.0779 * R1
            C13 = 1.587 + 0.049 * R1
            C14 = 0.1122 + 0.1658 * R1
            R4 = 1.0 / (Z**2) if Z > 0 else 0.0
            C17 = (
                (C11 - C12 * (T1**2)) * R4 * ((T1 * Z) ** (C13 + C14 * T1))
                if T1 > 0 and Z > 0
                else 0.0
            )
            C2 = C17 + C18

            if L3 <= 0.0:
                C5 = 0.0
            else:
                T2 = (1.0 - D2) / (2.0 * L3)
                if M_val <= 1.1:
                    T3 = 2.0 * (T2**2) + (T2**3)
                    E1 = math.exp(-2.0 * L3)
                    B4 = 1.0 - E1 + 2.0 * T2 * ((E1 * (L3 + 0.5)) - 0.5)
                    C5 = 2.0 * T3 * B4 * (1.774 - 9.3 * C15)
                else:
                    B3 = 0.85 / B
                    A12 = (
                        (5.0 * T1) / (6.0 * B)
                        + (0.5 * T1) ** 2
                        - (0.7435 / M2) * ((max(T1, 0.0) * M_val) ** 1.6)
                    )
                    A11 = (1.0 - ((0.6 * R1) / M_val)) * A12
                    E2 = math.exp((-1.1952 / M_val) * (L1 - L2 - L3))
                    X3 = ((2.4 * (M2**2) - 4.0 * B2) * (T2**2)) / (2.0 * (B2**2))
                    A1 = A11 * E2 - X3 + ((2.0 * T2) / B)
                    R5 = 1.0 / B3
                    E3 = math.exp(-B3 * L3)
                    A2 = 1.0 - E3 + (2.0 * T2 * (E3 * (L3 + R5) - R5))
                    C5 = 4.0 * A1 * T2 * A2 * R5

        C1 = C2 + C3 + C4 + C5 + C6
        rows.append({
            "mach": M_val,
            "cd0": C1,
            "cdh": C2,
            "cdsf": C3,
            "cdbnd": C4,
            "cdbt": C5,
            "cdb": C6,
            "pb_pinf": P1,
        })

    diagnostics: list[str] = []
    if L2 < 1.0:
        diagnostics.append("NOSE TOO SHORT. CDH IS TOO HIGH AT TRANSONIC AND SUPERSONIC SPEEDS.")
    if D3 > 0.5:
        diagnostics.append("NOSE TOO BLUNT. CDH IS TOO HIGH AT TRANSONIC AND SUPERSONIC SPEEDS.")
    if L3 >= 1.5:
        diagnostics.append("BOATTAIL TOO LONG. CDBT AND CDB MAY BE INCORRECT.")
    if D2 < 0.65:
        diagnostics.append("BOATTAIL TOO STEEP. CDBT AND CDB MAY BE INCORRECT.")
    elif D2 > 1.35:
        diagnostics.append("CONICAL FLARE TAIL TOO STEEP. CDBT AND CDB MAY BE INCORRECT.")

    # Prepend M=0.0 subsonic anchor equal to M=0.5 value for smooth table interpolation
    cd_table = [{"mach": 0.0, "cd": rows[0]["cd0"]}] + [
        {"mach": r["mach"], "cd": r["cd0"]} for r in rows
    ]

    return {
        "identification": inp.identification,
        "table": rows,
        "cd_table": cd_table,
        "diagnostics": diagnostics,
        "source_id": "S15",
        "verification_status": "VERIFIED-SRC (McCoy 1974 MCDRAG listing)",
    }
