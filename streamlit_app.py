"""Streamlit Community Cloud entrypoint for the Ballistics & Dispersion Monte Carlo Workbench."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ammo_mc.streamlit_app import main  # noqa: E402

main()
