"""Formula registry, provenance tracking, and FLAG governance enforcement.

Per App_Spec Governance rules:
- Every formula carries its ID, source_id, and verification_status.
- Formulas with status FLAG trigger a warning banner in the UI/reports and
  CANNOT be the sole basis of a safety distance output.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

from ammo_mc.data import get_kb

F = TypeVar("F", bound=Callable[..., Any])

USE_LIMITATION_NOTICE = (
    "Engineering estimate - not a substitute for approved danger-area templates, "
    "national QD regulations or competent-authority assessment."
)


@dataclass
class FormulaMetadata:
    formula_id: str
    function_name: str
    module: str
    name: str
    source_id: str
    source_url: str
    verification_status: str
    is_flagged: bool


FORMULA_REGISTRY: dict[str, Callable[..., Any]] = {}
FORMULA_META: dict[str, FormulaMetadata] = {}

_ACTIVE_TRACKER: ContextVar["ProvenanceTracker | None"] = ContextVar(
    "_ACTIVE_TRACKER", default=None
)


@dataclass
class ProvenanceTracker:
    """Collects formula IDs, sources, and FLAG warnings invoked during a run."""

    used_formula_ids: set[str] = field(default_factory=set)
    extra_warnings: list[str] = field(default_factory=list)
    _token: Any = field(default=None, init=False, repr=False)

    def __enter__(self) -> "ProvenanceTracker":
        self._token = _ACTIVE_TRACKER.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token is not None:
            _ACTIVE_TRACKER.reset(self._token)

    def record(self, formula_id: str) -> None:
        self.used_formula_ids.add(formula_id)

    def to_provenance_dict(self, seed: int | None, sampler: str, fidelity: str) -> dict[str, Any]:
        kb = get_kb()
        formulas_used = []
        source_ids: set[str] = set()
        flagged_formulas: list[dict[str, str]] = []

        for fid in sorted(self.used_formula_ids):
            row = kb.formulas_by_id.get(fid)
            if row is not None:
                vstat = row["verification_status"]
                is_flag = bool(row["is_flagged"])
                sids = [
                    s.strip()
                    for s in str(row["source_id"]).replace(",", ";").split(";")
                    if s.strip()
                ]
                for s in sids:
                    if s in kb.sources_by_id:
                        source_ids.add(s)
                formulas_used.append({
                    "formula_id": fid,
                    "name": row["name"],
                    "module": row["module"],
                    "source_id": row["source_id"],
                    "verification_status": vstat,
                    "is_flagged": is_flag,
                })
                if is_flag:
                    flagged_formulas.append({
                        "formula_id": fid,
                        "name": row["name"],
                        "warning": vstat,
                    })

        sources_detail = [
            kb.sources_by_id[sid] for sid in sorted(source_ids) if sid in kb.sources_by_id
        ]

        warnings = [
            f"[{f['formula_id']}] {f['name']}: {f['warning']}" for f in flagged_formulas
        ] + list(self.extra_warnings)

        return {
            "use_limitation": USE_LIMITATION_NOTICE,
            "seed": seed,
            "sampler": sampler,
            "fidelity_level": fidelity,
            "formulas_used": formulas_used,
            "sources": sources_detail,
            "flagged_formulas": flagged_formulas,
            "has_flag_warnings": len(warnings) > 0,
            "governance_warnings": warnings,
        }


def record_formula_use(formula_id: str) -> None:
    tracker = _ACTIVE_TRACKER.get()
    if tracker is not None:
        tracker.record(formula_id)


def register_formula(formula_id: str) -> Callable[[F], F]:
    """Decorator binding a pure Python function to its Knowledge Base Formula ID."""

    def decorator(func: F) -> F:
        kb = get_kb()
        row = kb.formulas_by_id.get(formula_id, {})
        meta = FormulaMetadata(
            formula_id=formula_id,
            function_name=func.__name__,
            module=row.get("module", ""),
            name=row.get("name", func.__name__),
            source_id=row.get("source_id", ""),
            source_url=row.get("source_url", ""),
            verification_status=row.get("verification_status", "STD-FORM"),
            is_flagged=bool(row.get("is_flagged", False)),
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker = _ACTIVE_TRACKER.get()
            if tracker is not None:
                tracker.record(formula_id)
            return func(*args, **kwargs)

        wrapper.formula_id = formula_id  # type: ignore[attr-defined]
        wrapper.formula_meta = meta  # type: ignore[attr-defined]
        FORMULA_REGISTRY[formula_id] = wrapper
        FORMULA_META[formula_id] = meta
        return wrapper  # type: ignore[return-value]

    return decorator


def validate_safety_distance_governance(
    basis_formula_ids: list[str],
) -> tuple[bool, str | None]:
    """Enforce App_Spec R25: FLAG formulas cannot be the sole basis of a safety distance."""
    kb = get_kb()
    if not basis_formula_ids:
        return False, "No basis formula provided for safety distance."
    non_flagged = []
    flagged = []
    for fid in basis_formula_ids:
        row = kb.formulas_by_id.get(fid, {})
        if row.get("is_flagged", False):
            flagged.append(fid)
        else:
            non_flagged.append(fid)
    if flagged and not non_flagged:
        return (
            False,
            f"GOVERNANCE BLOCK (App_Spec R25): Flagged formula(s) {flagged} have status FLAG "
            "and cannot be the sole basis of a safety distance output.",
        )
    return True, None
