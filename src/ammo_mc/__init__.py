"""Ammunition Effects & Dispersion Monte Carlo Simulator (`ammo_mc`).

Core physics, ballistics, terminal effects, reliability, and stochastic analysis engine.
No UI code resides in core modules.
"""

from __future__ import annotations

from ammo_mc import formulas, mcdrag, plugins, presets
from ammo_mc.data import KnowledgeBase, get_kb
from ammo_mc.formulas_registry import (
    FORMULA_META,
    FORMULA_REGISTRY,
    USE_LIMITATION_NOTICE,
    ProvenanceTracker,
    validate_safety_distance_governance,
)
from ammo_mc.pipeline import GovernanceError, compute_governed_safety_distance, run_simulation
from ammo_mc.schemas import AmmoDefinition, Scenario, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AmmoDefinition",
    "Scenario",
    "SimulationResult",
    "KnowledgeBase",
    "get_kb",
    "run_simulation",
    "compute_governed_safety_distance",
    "GovernanceError",
    "FORMULA_REGISTRY",
    "FORMULA_META",
    "ProvenanceTracker",
    "USE_LIMITATION_NOTICE",
    "validate_safety_distance_governance",
    "formulas",
    "mcdrag",
    "plugins",
    "presets",
]
