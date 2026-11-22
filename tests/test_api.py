"""FastAPI TestClient verification for ammo_mc REST API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from ammo_mc.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_api_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "use_limitation" in data


def test_api_taxonomy(client: TestClient) -> None:
    res = client.get("/taxonomy")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 12
    class_ids = {row["class_id"] for row in data}
    assert "AR-01" in class_ids
    assert "SA-01" in class_ids


def test_api_formulas(client: TestClient) -> None:
    res = client.get("/formulas")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 80

    res_flag = client.get("/formulas?flagged_only=true")
    assert res_flag.status_code == 200
    data_flag = res_flag.json()
    assert len(data_flag) > 0
    assert all(f.get("is_flagged") for f in data_flag)


def test_api_presets(client: TestClient) -> None:
    res = client.get("/presets")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 12

    # Specific preset
    res_one = client.get("/presets/NOTIONAL-SA-01")
    assert res_one.status_code == 200
    payload = res_one.json()
    assert "ammo" in payload and "scenario" in payload
    assert payload["ammo"]["id"] == "NOTIONAL-SA-01"


def test_api_mcdrag(client: TestClient) -> None:
    payload = {
        "calibre_m": 0.155,
        "total_length_cal": 5.0,
        "nose_length_cal": 2.5,
        "rt_over_r": 0.5,
        "boattail_length_cal": 0.5,
        "base_dia_cal": 0.85,
        "meplat_dia_cal": 0.15,
        "band_dia_cal": 1.0,
        "boundary_layer_code": "L/T",
    }
    res = client.post("/mcdrag", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "table" in data
    assert len(data["table"]) >= 20
    assert data["table"][0]["mach"] == 0.5


def test_api_simulate_preset_and_geojson(client: TestClient) -> None:
    # Run small test simulation via API
    sim_req = {
        "preset_id": "NOTIONAL-SA-01",
        "fidelity_override": "L1",
        "trials_override": 20,
    }
    res = client.post("/simulate", json=sim_req)
    assert res.status_code == 200
    data = res.json()
    assert data["ammo_id"] == "NOTIONAL-SA-01"
    assert "dispersion" in data
    assert "effectiveness" in data
    assert "provenance" in data

    # GeoJSON export endpoint
    res_geo = client.post("/export/geojson", json=sim_req)
    assert res_geo.status_code == 200
    geo_data = res_geo.json()
    assert geo_data["type"] == "FeatureCollection"
    assert len(geo_data["features"]) >= 1
