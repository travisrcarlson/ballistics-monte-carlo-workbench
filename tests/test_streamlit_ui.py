"""Headless Streamlit UI regression tests (auth gate, contextual sidebar & URL deep links)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT_APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_ui_password_gate_reject_and_unlock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMMO_MC_PASSWORD", "ci-ui-pass")
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert not at.session_state["authenticated"]

    # Wrong password keeps screen locked with error
    at.text_input(key="auth_pw_field").set_value("wrong-password")
    at.button[0].click().run()
    assert not at.session_state["authenticated"]
    assert len(at.error) >= 1

    # Valid password unlocks into default Monte Carlo Simulator workbench
    at.text_input(key="auth_pw_field").set_value("ci-ui-pass")
    at.button[0].click().run()
    assert at.session_state["authenticated"] is True
    assert not at.exception
    assert any("Monte Carlo Simulator" in r.value for r in at.radio)


@pytest.mark.parametrize(
    ("wb_slug", "mod_slug", "expected_wb"),
    [
        ("cuas", "comparator", "🛸 C-UAS & Small Arms"),
        ("cuas", "site_defense", "🛸 C-UAS & Small Arms"),
        ("cuas", "sensitivity", "🛸 C-UAS & Small Arms"),
        ("cuas", "raid", "🛸 C-UAS & Small Arms"),
        ("cuas", "calibration", "🛸 C-UAS & Small Arms"),
        ("mcdrag", None, "🚀 MCDRAG & 3D CAD"),
        ("iatg", None, "🏛️ UNODA IATG Safety"),
        ("validation", None, "✅ Validation Suite"),
        ("literature", None, "📚 Literature & Sources"),
    ],
)
def test_ui_deeplinks_and_contextual_sidebars(
    wb_slug: str,
    mod_slug: str | None,
    expected_wb: str,
) -> None:
    at = AppTest.from_file(ROOT_APP, default_timeout=60)
    at.session_state["authenticated"] = True
    at.query_params["wb"] = wb_slug
    if mod_slug:
        at.query_params["mod"] = mod_slug
    at.run()

    assert not at.exception
    assert at.session_state["active_workbench"] == expected_wb
