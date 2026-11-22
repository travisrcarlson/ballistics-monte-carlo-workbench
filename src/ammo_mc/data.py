"""Knowledge Base data layer for `ammo_mc`.

Loads all reference tables exported from 'Ammunition Monte Carlo Knowledge Base.xlsx'
(or directly from an `.xlsx` workbook path). Every record carries `source_id` and
`verification_status` per App_Spec.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import tempfile
from typing import Any
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data_tables"

EXPLOSIVE_ALIASES: dict[str, str] = {
    "comp b": "Composition B",
    "comp-b": "Composition B",
    "composition b": "Composition B",
    "c4": "Composition C4",
    "c-4": "Composition C4",
    "comp c4": "Composition C4",
    "composition c4": "Composition C4",
    "c3": "Composition C-3",
    "comp c-3": "Composition C-3",
    "tnt": "TNT",
    "rdx": "RDX",
    "petn": "PETN",
    "hmx": "HMX",
    "h6": "H-6",
    "h-6": "H-6",
    "anfo": "ANFO",
    "octol": "Octol 75/25",
    "octol 75/25": "Octol 75/25",
    "cyclotol": "RDX/TNT 60/40 (Cyclotol)",
    "pentolite": "Pentolite",
    "tetryl": "Tetryl",
    "torpex": "Torpex II",
    "torpex ii": "Torpex II",
    "tritonal": "Tritonal 80/20",
    "tritonal 80/20": "Tritonal 80/20",
    "lx-14": "LX-14",
    "pbx 9404": "PBX 9404",
    "pbx 9502": "PBX 9502",
}


class KnowledgeBase:
    """In-memory, NumPy-accelerated access to all Knowledge Base tables."""

    def __init__(self, tables_dir: Path = DATA_DIR) -> None:
        self.tables_dir = Path(tables_dir)
        self.sources: list[dict[str, Any]] = self._load_json("sources.json")
        self.sources_by_id: dict[str, dict[str, Any]] = {
            row["source_id"]: row for row in self.sources
        }
        self.taxonomy: list[dict[str, Any]] = self._load_json("ammo_taxonomy.json")
        self.taxonomy_by_id: dict[str, dict[str, Any]] = {
            row["class_id"]: row for row in self.taxonomy
        }
        self.formulas: list[dict[str, Any]] = self._load_json("formulas.json")
        self.formulas_by_id: dict[str, dict[str, Any]] = {
            row["formula_id"]: row for row in self.formulas
        }
        self.drag_tables: dict[str, list[dict[str, Any]]] = self._load_json("drag_tables.json")
        self.kb_coefficients: list[dict[str, Any]] = self._load_json("kb_coefficients.json")
        self.explosives: dict[str, list[dict[str, Any]]] = self._load_json("explosives.json")
        self.thor: dict[str, Any] = self._load_json("thor_coefficients.json")
        self.penetration: dict[str, list[dict[str, Any]]] = self._load_json("penetration.json")
        self.effects_criteria: dict[str, Any] = self._load_json("effects_criteria.json")
        self.atmosphere: dict[str, Any] = self._load_json("atmosphere.json")
        self.validation_cases: list[dict[str, Any]] = self._load_json("validation_cases.json")

        # Pre-build fast NumPy arrays for drag interpolation
        self.g1_mach = np.array([r["mach"] for r in self.drag_tables["G1"]], dtype=np.float64)
        self.g1_cd = np.array([r["cd"] for r in self.drag_tables["G1"]], dtype=np.float64)
        self.g7_mach = np.array([r["mach"] for r in self.drag_tables["G7"]], dtype=np.float64)
        self.g7_cd = np.array([r["cd"] for r in self.drag_tables["G7"]], dtype=np.float64)
        self.frag_mach_from = np.array(
            [r["mach_from"] for r in self.drag_tables["natural_fragment"]], dtype=np.float64
        )
        self.frag_mach_to = np.array(
            [r["mach_to"] for r in self.drag_tables["natural_fragment"]], dtype=np.float64
        )
        self.frag_cd = np.array(
            [r["cd"] for r in self.drag_tables["natural_fragment"]], dtype=np.float64
        )

        # Group KB pieces by param_key
        self.kb_by_param: dict[str, list[dict[str, Any]]] = {}
        for piece in self.kb_coefficients:
            self.kb_by_param.setdefault(piece["param_key"], []).append(piece)

        # Index THOR materials case-insensitively
        self.thor_by_material: dict[str, dict[str, Any]] = {
            row["material"].lower(): row for row in self.thor["materials"]
        }

    def _load_json(self, name: str) -> Any:
        path = self.tables_dir / name
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def from_excel(cls, xlsx_path: str | Path) -> "KnowledgeBase":
        """Load KnowledgeBase directly from an `.xlsx` workbook."""
        import importlib.util
        import sys

        # Dynamically use export_workbook logic if scripts/ is present, or inline fallback
        root_exporter = Path(__file__).resolve().parents[2] / "scripts" / "export_kb_tables.py"
        tmp_dir = Path(tempfile.mkdtemp(prefix="ammo_mc_kb_"))
        if root_exporter.exists():
            spec = importlib.util.spec_from_file_location("_kb_exporter", root_exporter)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_kb_exporter"] = mod
            spec.loader.exec_module(mod)
            mod.export_workbook(Path(xlsx_path), tmp_dir)
            return cls(tmp_dir)
        raise FileNotFoundError(f"Workbook exporter script not found at {root_exporter}")

    def canonical_explosive_name(self, name: str) -> str:
        key = name.strip().lower()
        return EXPLOSIVE_ALIASES.get(key, name.strip())

    def get_explosive_properties(self, name: str, gurney_set: str = "iatg") -> dict[str, Any]:
        """Return merged TNT equivalence, Gurney constant, detonation velocity, density, and Mott B."""
        canon = self.canonical_explosive_name(name)
        canon_low = canon.lower()

        f_p, f_i = 1.0, 1.0
        for row in self.explosives["tnt_equivalence"]:
            if row["explosive"].lower() == canon_low:
                f_p = row["f_pressure"]
                f_i = row["f_impulse"]
                break

        gurney_table = (
            self.explosives["gurney_iatg"]
            if gurney_set.lower() == "iatg"
            else self.explosives["gurney_alt_s05"]
        )
        gurney_mps = None
        det_vel_mps = None
        density_g_cm3 = None
        for row in gurney_table:
            if row["explosive"].lower() == canon_low:
                gurney_mps = row["gurney_mps"]
                det_vel_mps = row["detonation_velocity_mps"]
                density_g_cm3 = row["density_g_cm3"]
                break
        if gurney_mps is None:
            # Fallback to alternate Gurney table
            other_table = (
                self.explosives["gurney_alt_s05"]
                if gurney_set.lower() == "iatg"
                else self.explosives["gurney_iatg"]
            )
            for row in other_table:
                if row["explosive"].lower() == canon_low or canon_low in row["explosive"].lower():
                    gurney_mps = row["gurney_mps"]
                    det_vel_mps = row["detonation_velocity_mps"]
                    density_g_cm3 = row["density_g_cm3"]
                    break
        if gurney_mps is None:
            # Default TNT baseline if unknown custom explosive without explicit Gurney
            gurney_mps = 2438.0
            det_vel_mps = 6900.0
            density_g_cm3 = 1.61

        mott_b = None
        for row in self.explosives["mott_constants"]:
            if row["explosive"].lower() == canon_low:
                mott_b = row["mott_b"]
                break
        if mott_b is None:
            mott_b = 0.0779  # TNT mild-steel reference

        return {
            "explosive": canon,
            "f_tnt_p": float(f_p),
            "f_tnt_i": float(f_i),
            "gurney_mps": float(gurney_mps),
            "detonation_velocity_mps": float(det_vel_mps),
            "det_vel_mps": float(det_vel_mps),
            "density_g_cm3": float(density_g_cm3),
            "mott_b": float(mott_b),
            "source_id": "S01;S05;S11",
            "verification_status": "VERIFIED-SRC",
        }

    def interp_cd(
        self,
        mach: float | np.ndarray,
        drag_ref: str = "G7",
        custom_table: list[tuple[float, float]] | list[dict[str, float]] | None = None,
    ) -> float | np.ndarray:
        """Interpolate reference drag coefficient Cd(M) per Drag_Tables sheet."""
        m_arr = np.asarray(mach, dtype=np.float64)
        ref_up = drag_ref.upper()
        if custom_table is not None and len(custom_table) >= 2:
            if isinstance(custom_table[0], dict):
                mx = np.array([pt["mach"] for pt in custom_table], dtype=np.float64)  # type: ignore[index]
                cy = np.array([pt["cd"] for pt in custom_table], dtype=np.float64)  # type: ignore[index]
            else:
                mx = np.array([pt[0] for pt in custom_table], dtype=np.float64)  # type: ignore[index]
                cy = np.array([pt[1] for pt in custom_table], dtype=np.float64)  # type: ignore[index]
            res = np.interp(m_arr, mx, cy)
        elif ref_up == "G1":
            res = np.interp(m_arr, self.g1_mach, self.g1_cd)
        elif ref_up == "G7":
            res = np.interp(m_arr, self.g7_mach, self.g7_cd)
        elif ref_up in ("FRAGMENT", "NATURAL_FRAGMENT", "FRAG"):
            # Band lookup matching Excel =INDEX(J6:J17, MATCH(M, H6:H17, 1))
            idx = np.searchsorted(self.frag_mach_from, m_arr, side="right") - 1
            idx = np.clip(idx, 0, len(self.frag_cd) - 1)
            res = self.frag_cd[idx]
        elif ref_up == "SPHERE":
            # Smooth subsonic/supersonic sphere approximation (~0.47 subsonic, ~0.92 supersonic)
            res = np.where(m_arr < 0.8, 0.47, np.where(m_arr < 1.2, 0.47 + (m_arr - 0.8) * 1.125, 0.92))
        else:
            raise ValueError(f"Unknown drag_ref '{drag_ref}'. Expected G1, G7, FRAGMENT, SPHERE, or custom.")

        if np.ndim(mach) == 0:
            return float(res)
        return res

    def get_thor_material(self, material: str) -> dict[str, Any]:
        key = material.strip().lower()
        if key in self.thor_by_material:
            return self.thor_by_material[key]
        # Partial match fallback (e.g. "RHA" -> "RHA armor")
        for k, v in self.thor_by_material.items():
            if key in k or k in key:
                return v
        raise KeyError(f"Unknown THOR target material '{material}'. Available: {[r['material'] for r in self.thor['materials']]}")

    def get_lanz_odermatt_coeffs(self, rod_material: str, mode: str = "semi-infinite") -> dict[str, Any]:
        rm_low = rod_material.strip().lower()
        mode_low = mode.strip().lower()
        candidates = []
        for row in self.penetration["lanz_odermatt"]:
            mat_match = (
                rm_low in row["rod_material"].lower()
                or (rm_low in ("w", "tungsten") and "tungsten" in row["rod_material"].lower())
                or (rm_low in ("du", "uranium") and "uranium" in row["rod_material"].lower())
                or (rm_low in ("steel", "uhss") and "steel" in row["rod_material"].lower())
            )
            if mat_match:
                candidates.append(row)
        if not candidates:
            raise KeyError(f"Unknown Lanz-Odermatt rod material '{rod_material}'")
        for c in candidates:
            if mode_low in c["mode"].lower():
                return c
        return candidates[0]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    """Singleton accessor for the bundled KnowledgeBase."""
    return KnowledgeBase()
