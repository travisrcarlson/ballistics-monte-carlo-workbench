"""Verification tests for SQLite ScenarioLibrary and Executive PDF/PPTX Briefings."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from pptx import Presentation

from ammo_mc.api import app
from ammo_mc.briefing_export import (
    generate_executive_briefing_pdf,
    generate_executive_briefing_pptx,
)
from ammo_mc.cuas_analytics import (
    analyze_cuas_sensitivity_tornado,
    compare_cuas_architectures_ab,
)
from ammo_mc.cuas_mission import (
    simulate_cuas_sustained_logistics_mc,
    simulate_multiturret_site_defense_mc,
)
from ammo_mc.scenario_library import ScenarioLibrary


def test_scenario_library_crud_diff_and_json_roundtrip(tmp_path) -> None:
    db_file = tmp_path / "test_scenarios.db"
    lib = ScenarioLibrary(db_path=db_file)

    s1 = lib.save_scenario(
        name="Baseline 4-Turret FOB",
        domain="cuas",
        config={"n_barrels": 6, "ammo": "12G_NO4_BUCK_C_UAS"},
        metrics={"zero_breach_pct": 78.0, "cost_per_kill_usd": 4.2},
        tags=["fob", "baseline"],
    )
    s2 = lib.save_scenario(
        name="Upgraded TSS Concentric",
        domain="cuas",
        config={"n_barrels": 8, "ammo": "12G_TSS_NO2_C_UAS"},
        metrics={"zero_breach_pct": 94.0, "cost_per_kill_usd": 14.5},
        tags=["fob", "tss"],
    )

    items = lib.list_scenarios(domain="cuas", search="Upgraded")
    assert len(items) == 1 and items[0]["id"] == s2["id"]

    diff = lib.diff_scenarios(s1["id"], s2["id"])
    assert len(diff["config_diffs"]) >= 2
    zb_row = next(r for r in diff["metric_diffs"] if r["metric"] == "zero_breach_pct")
    assert zb_row["delta"] == 16.0

    exported = lib.export_library_json()
    lib2 = ScenarioLibrary(db_path=tmp_path / "import_scenarios.db")
    assert lib2.import_library_json(exported) == 2
    assert lib2.delete_scenario(s1["id"]) is True
    assert len(lib2.list_scenarios()) == 1


def test_executive_pdf_and_pptx_generation_and_api() -> None:
    ab = compare_cuas_architectures_ab(n_swarm_drones=4, n_waves=2, drones_per_wave=2)
    sd = simulate_multiturret_site_defense_mc(n_raid_trials=12)
    tor = analyze_cuas_sensitivity_tornado(n_trials=16)
    lg = simulate_cuas_sustained_logistics_mc(n_waves=3, n_trials=15)

    pdf_bytes = generate_executive_briefing_pdf(
        title="Test C-UAS Executive Briefing",
        ab_result=ab,
        site_result=sd,
        tornado_result=tor,
        logistics_result=lg,
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 15_000

    pptx_bytes = generate_executive_briefing_pptx(
        title="Test C-UAS Executive Briefing",
        ab_result=ab,
        site_result=sd,
        tornado_result=tor,
        logistics_result=lg,
    )
    prs = Presentation(io.BytesIO(pptx_bytes))
    assert len(prs.slides) == 6

    client = TestClient(app)
    api_pdf = client.post("/cuas/briefing/pdf", json={"ab_result": ab, "site_result": sd})
    assert api_pdf.status_code == 200
    assert api_pdf.headers["content-type"].startswith("application/pdf")
    assert api_pdf.content.startswith(b"%PDF-")

    api_pptx = client.post("/cuas/briefing/pptx", json={"ab_result": ab, "tornado_result": tor})
    assert api_pptx.status_code == 200
    assert "presentationml" in api_pptx.headers["content-type"]
    prs_api = Presentation(io.BytesIO(api_pptx.content))
    assert len(prs_api.slides) >= 4
