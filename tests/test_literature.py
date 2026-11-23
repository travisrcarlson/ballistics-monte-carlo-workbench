"""Verification tests for expanded Scientific Literature Database & Model Provenance Map."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ammo_mc.api import app
from ammo_mc.cli import app as cli_app
from ammo_mc.data import get_kb
from ammo_mc.literature import (
    build_literature_audit_summary,
    export_literature_bibtex,
    export_literature_csv,
    get_data_table_provenance,
    get_literature_catalog,
    get_provenance_map,
    get_sources_for_module,
    search_literature,
)


def test_literature_catalog_coverage_and_zero_broken_refs() -> None:
    kb = get_kb()
    catalog = get_literature_catalog()
    assert len(catalog) >= 68
    sids = set(kb.sources_by_id.keys())
    fids = set(kb.formulas_by_id.keys())
    assert len(sids) >= 68

    prov_map = get_provenance_map()
    assert len(prov_map) >= 40
    for mod in prov_map:
        assert not (set(mod["source_ids"]) - sids), f"Unknown source IDs in {mod['entry_id']}"
        assert not (set(mod["formula_ids"]) - fids), f"Unknown formula IDs in {mod['entry_id']}"

    summary = build_literature_audit_summary()
    assert summary["total_sources"] >= 68
    assert summary["doi_sources"] >= 15
    assert summary["linked_source_coverage_pct"] == 100.0
    assert len(get_data_table_provenance()) >= 12


def test_literature_search_dossier_and_exports() -> None:
    singer_hits = search_literature(query="Singer", doi_only=True)
    assert any(r["source_id"] == "S44" for r in singer_hits)

    dossier = get_sources_for_module("cuas-01-fcs")
    assert len(dossier["sources"]) >= 4
    assert any(s["source_id"] == "S44" for s in dossier["sources"])

    bib = export_literature_bibtex(["S01", "S44"])
    assert "@misc{ammo_s01_" in bib or "@article{ammo_s44_" in bib
    assert "10.1109/TAES.1970.310128" in bib

    csv_txt = export_literature_csv(["S41", "S44"])
    assert "source_id,title,authors" in csv_txt
    assert "S41" in csv_txt and "S44" in csv_txt


def test_literature_api_and_cli() -> None:
    client = TestClient(app)
    res = client.get("/literature?workbench_slug=cuas")
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_sources"] >= 68
    assert len(data["sources"]) >= 10
    assert len(data["module_provenance"]) >= 40

    runner = CliRunner()
    cli_res = runner.invoke(cli_app, ["literature", "--workbench", "cuas", "--json"])
    assert cli_res.exit_code == 0
    parsed = json.loads(cli_res.output)
    assert parsed["summary"]["linked_source_coverage_pct"] == 100.0
