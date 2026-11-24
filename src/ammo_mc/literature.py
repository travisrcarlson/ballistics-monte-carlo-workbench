"""Scientific literature catalog, formula provenance, and module-level traceability engine.

Pure Python domain module (zero UI dependencies) connecting all Knowledge Base sources
(`sources.json`), registered equations (`formulas.json`), empirical tables, and every
simulation module / workbench (`provenance_map.json`).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
from typing import Any

from ammo_mc.data import DATA_DIR, get_kb


def _load_provenance_payload(tables_dir: Path | None = None) -> dict[str, Any]:
    """Load `provenance_map.json` from custom directory or packaged data_tables."""
    base_dir = Path(tables_dir) if tables_dir is not None else DATA_DIR
    path = base_dir / "provenance_map.json"
    if not path.exists():
        path = DATA_DIR / "provenance_map.json"
    if not path.exists():
        return {"schema_version": "1.0", "modules": [], "data_tables": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _split_source_ids(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).replace(",", ";").split(";") if s.strip()]


def get_provenance_map() -> list[dict[str, Any]]:
    """Return full workbench-and-module traceability mapping records."""
    payload = _load_provenance_payload()
    return [dict(m) for m in payload.get("modules", [])]


def get_data_table_provenance() -> list[dict[str, Any]]:
    """Return empirical data table provenance enriched with live KB record counts."""
    kb = get_kb()
    payload = _load_provenance_payload()
    counts: dict[str, str] = {
        "sources.json": f"{len(kb.sources)} citations",
        "formulas.json": f"{len(kb.formulas)} equations ({sum(1 for f in kb.formulas if f.get('is_flagged'))} FLAGGED)",
        "drag_tables.json": f"G1:{len(kb.drag_tables.get('G1', []))}, G7:{len(kb.drag_tables.get('G7', []))}, Frag:{len(kb.drag_tables.get('natural_fragment', []))} Mach pts",
        "explosives.json": f"{len(kb.explosives.get('tnt_equivalence', []))} explosives",
        "kb_coefficients.json": f"{len(kb.kb_coefficients)} piecewise log-polynomials",
        "thor_coefficients.json": f"{len(kb.thor.get('materials', []))} materials / {len(kb.thor.get('shape_factors_cm2_g23', []))} shapes",
        "penetration.json": f"LO:{len(kb.penetration.get('lanz_odermatt', []))}, DM:{len(kb.penetration.get('de_marre', []))}",
        "effects_criteria.json": f"{len(kb.effects_criteria)} criteria groups",
        "atmosphere.json": f"{len(kb.atmosphere.get('isa_table', []))} altitude rows",
        "ammo_taxonomy.json": f"{len(kb.taxonomy)} classes",
        "validation_cases.json": f"{len(kb.validation_cases)} benchmark cases",
        "C-UAS & Small Arms Empirical Catalogs": "8+ domain catalogs",
    }
    rows: list[dict[str, Any]] = []
    for item in payload.get("data_tables", []):
        row = dict(item)
        tname = str(row.get("table_name", ""))
        row["record_summary"] = counts.get(tname, "Domain catalog")
        row["source_titles"] = [
            kb.sources_by_id[sid]["title"]
            for sid in row.get("source_ids", [])
            if sid in kb.sources_by_id
        ]
        rows.append(row)
    return rows


def get_literature_catalog() -> list[dict[str, Any]]:
    """Return all Knowledge Base literature sources enriched with linked formula & module counts."""
    kb = get_kb()
    prov_modules = get_provenance_map()

    formula_links: dict[str, list[str]] = {}
    for f in kb.formulas:
        fid = str(f["formula_id"])
        for sid in _split_source_ids(f.get("source_id", "")):
            formula_links.setdefault(sid, []).append(fid)

    module_links: dict[str, list[dict[str, str]]] = {}
    for mod in prov_modules:
        eid = str(mod["entry_id"])
        mname = str(mod["module_name"])
        for sid in mod.get("source_ids", []):
            module_links.setdefault(sid, []).append({"entry_id": eid, "module_name": mname})

    catalog: list[dict[str, Any]] = []
    for src in kb.sources:
        sid = str(src["source_id"])
        f_ids = sorted(set(formula_links.get(sid, [])))
        m_list = module_links.get(sid, [])
        doi = str(src.get("doi", "") or "").strip()
        row = {
            "source_id": sid,
            "title": str(src.get("title", "")),
            "authors": str(src.get("authors", "") or "Standard / Institutional"),
            "year": src.get("year"),
            "doi": doi,
            "has_doi": bool(doi),
            "category": str(src.get("category", "General Ballistics")),
            "type": str(src.get("type", "")),
            "url": str(src.get("url", "")),
            "used_for": str(src.get("used_for", "")),
            "workbenches": list(src.get("workbenches", [])),
            "verification_status": str(src.get("verification_status", "")),
            "linked_formula_ids": f_ids,
            "linked_formula_count": len(f_ids),
            "linked_module_ids": [m["entry_id"] for m in m_list],
            "used_by_modules": [m["module_name"] for m in m_list],
            "linked_module_count": len(m_list),
        }
        catalog.append(row)
    return catalog


def search_literature(
    query: str = "",
    category: str | None = None,
    source_type: str | None = None,
    workbench_slug: str | None = None,
    verified_only: bool = False,
    doi_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter enriched literature entries across free text, category, type, workbench, and DOI flags."""
    q = query.strip().lower()
    cat_filter = None if not category or category == "All" else category.strip().lower()
    type_filter = None if not source_type or source_type == "All" else source_type.strip().lower()
    wb_filter = None if not workbench_slug or workbench_slug == "All" else workbench_slug.strip().lower()

    results: list[dict[str, Any]] = []
    for item in get_literature_catalog():
        if cat_filter and item["category"].lower() != cat_filter:
            continue
        if type_filter and item["type"].lower() != type_filter:
            continue
        if wb_filter and wb_filter not in [w.lower() for w in item["workbenches"]]:
            continue
        if verified_only and "VERIFIED" not in item["verification_status"].upper():
            continue
        if doi_only and not item["has_doi"]:
            continue
        if q:
            haystack = " ".join(
                [
                    item["source_id"],
                    item["title"],
                    item["authors"],
                    str(item["year"] or ""),
                    item["doi"],
                    item["category"],
                    item["type"],
                    item["used_for"],
                    " ".join(item["used_by_modules"]),
                    " ".join(item["linked_formula_ids"]),
                ]
            ).lower()
            if q not in haystack:
                continue
        results.append(item)
    return results


def get_sources_for_module(entry_id: str) -> dict[str, Any]:
    """Return a complete provenance dossier for one mapped workbench module `entry_id`."""
    kb = get_kb()
    prov_map = get_provenance_map()
    target = next((m for m in prov_map if m["entry_id"] == entry_id), None)
    if target is None:
        raise KeyError(f"Unknown provenance module entry_id: '{entry_id}'")

    catalog_by_id = {r["source_id"]: r for r in get_literature_catalog()}
    resolved_sources = [catalog_by_id[sid] for sid in target.get("source_ids", []) if sid in catalog_by_id]
    resolved_formulas = [kb.formulas_by_id[fid] for fid in target.get("formula_ids", []) if fid in kb.formulas_by_id]

    table_prov_by_name = {t["table_name"]: t for t in get_data_table_provenance()}
    resolved_tables = [
        table_prov_by_name.get(tname, {"table_name": tname, "domain": "Domain Catalog", "description": tname})
        for tname in target.get("data_tables", [])
    ]

    return {
        **target,
        "sources": resolved_sources,
        "formulas": resolved_formulas,
        "resolved_tables": resolved_tables,
        "has_flagged_formulas": any(bool(f.get("is_flagged")) for f in resolved_formulas),
    }


def build_literature_audit_summary() -> dict[str, Any]:
    """Build executive summary metrics and category distributions over the Literature & Provenance KB."""
    kb = get_kb()
    catalog = get_literature_catalog()
    prov_map = get_provenance_map()

    cat_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for row in catalog:
        cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1

    doi_count = sum(1 for r in catalog if r["has_doi"])
    standards_count = sum(
        1
        for r in catalog
        if any(k in r["type"].lower() for k in ("standard", "regulation", "manual", "circular", "treaty", "criteria"))
    )
    flagged_formulas = [f["formula_id"] for f in kb.formulas if f.get("is_flagged")]
    all_source_ids = set(kb.sources_by_id.keys())
    referenced_by_formula_or_mod = {
        sid for r in catalog if (r["linked_formula_count"] > 0 or r["linked_module_count"] > 0) for sid in [r["source_id"]]
    }

    return {
        "total_sources": len(catalog),
        "verified_sources": sum(1 for r in catalog if "VERIFIED" in r["verification_status"].upper()),
        "doi_sources": doi_count,
        "standards_and_manuals": standards_count,
        "total_formulas": len(kb.formulas),
        "flagged_formula_count": len(flagged_formulas),
        "flagged_formula_ids": flagged_formulas,
        "mapped_modules": len(prov_map),
        "data_tables_documented": len(get_data_table_provenance()),
        "linked_source_coverage_pct": round(100.0 * len(referenced_by_formula_or_mod) / max(len(all_source_ids), 1), 1),
        "category_counts": dict(sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "type_counts": dict(sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _bib_escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def export_literature_bibtex(source_ids: list[str] | None = None) -> str:
    """Export selected (or all) Knowledge Base citations as standard BibTeX (.bib)."""
    catalog = get_literature_catalog()
    id_set = {s.strip() for s in source_ids} if source_ids else None

    entries: list[str] = []
    for row in catalog:
        sid = row["source_id"]
        if id_set is not None and sid not in id_set:
            continue
        t_low = row["type"].lower()
        bib_type = "article" if "paper" in t_low or "review" in t_low else ("book" if "textbook" in t_low or "handbook" in t_low else "misc")
        year = row["year"] or 2024
        first_tok = re.sub(r"[^a-z0-9]", "", row["authors"].split(",")[0].split()[0].lower()) if row["authors"] else "ref"
        cite_key = f"ammo_{sid.lower()}_{first_tok}_{year}"
        lines = [
            f"@{bib_type}{{{cite_key},",
            f"  title = {{{_bib_escape(row['title'])}}},",
            f"  author = {{{_bib_escape(row['authors'])}}},",
            f"  year = {{{year}}},",
            f"  note = {{{sid} | {_bib_escape(row['type'])} | {_bib_escape(row['verification_status'])}}},",
        ]
        if row["doi"]:
            lines.append(f"  doi = {{{row['doi']}}},")
        if row["url"]:
            lines.append(f"  howpublished = {{\\url{{{row['url']}}}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + ("\n" if entries else "")


def export_literature_csv(source_ids: list[str] | None = None) -> str:
    """Export selected (or all) Knowledge Base literature rows as RFC 4180 CSV."""
    catalog = get_literature_catalog()
    id_set = {s.strip() for s in source_ids} if source_ids else None
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "source_id",
            "title",
            "authors",
            "year",
            "doi",
            "category",
            "type",
            "verification_status",
            "linked_formula_count",
            "linked_module_count",
            "used_for",
            "url",
        ]
    )
    for row in catalog:
        if id_set is not None and row["source_id"] not in id_set:
            continue
        writer.writerow(
            [
                row["source_id"],
                row["title"],
                row["authors"],
                row["year"] or "",
                row["doi"],
                row["category"],
                row["type"],
                row["verification_status"],
                row["linked_formula_count"],
                row["linked_module_count"],
                row["used_for"],
                row["url"],
            ]
        )
    return buf.getvalue()
