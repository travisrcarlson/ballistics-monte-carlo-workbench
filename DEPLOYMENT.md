# Deployment Guide (`ammo-mc` Workbench & API)

This workbench is packaged cleanly for zero-DevOps cloud deployment on **Streamlit Community Cloud** as well as containerized / ASGI deployment via **FastAPI + Uvicorn**.

---

## 1. Streamlit Community Cloud (`share.streamlit.io`)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub** and select:
   - **Repository:** `travisrcarlson/ballistics-monte-carlo-workbench`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Open **Advanced settings → Secrets** and configure your custom launch password and optional SQLite scenario database path:
   ```toml
   APP_PASSWORD = "your-chosen-strong-password"
   ```
   - `verify_app_password()` checks `st.secrets["APP_PASSWORD"]` first, then the `AMMO_MC_PASSWORD` environment variable, then the built-in PBKDF2-SHA256 fallback digest (`200,000` iterations).
   - Never commit plaintext secrets or `.streamlit/secrets.toml` to version control.
   - Repeated failed logins trigger an in-session exponential cooldown (`LoginRateLimiter`: 5 attempts → 15 s / 30 s / 60 s ... up to 300 s).
4. Click **Deploy!** — Streamlit installs dependencies directly from `requirements.txt`.

### Deep-Linkable Views

Share direct links into any consolidated workbench or C-UAS stage module via query parameters:
- `?wb=home` — 3-Click C-UAS Turret Sizing Wizard, Task-First Quick-Start Home & Scenario Presets
- `?wb=cuas&mod=array` — Stage 1 Multi-Shotgun Array Designer, Custom Drone/Shell Builders, Hardware SWaP-C/OTM, Bolt-Shear FEA, DXF/SVG/STL CAD & Spec Sheets
- `?wb=cuas&mod=fcs` — Stage 2 Closed-Loop FCS & Evasive Drones (with Stage 1 Profile Sync)
- `?wb=cuas&mod=sdz` — Stage 3 Elevated Pellet Fallout, Acoustic Blast & Urban Collateral Safety Envelope
- `?wb=cuas&mod=raid` — Unified 3D Cooperative Multi-Turret Raid Simulator (with Nash Battlespace Handoff & Layered Effectors)
- `?wb=cuas&mod=calibration` — Closed-Loop Test Recalibration (Photo Digitizer, Multi-Range, Bayesian Lethality, Hardware Bench, Persistent SQLite Digital Twin, Design Fixes & Active Test Planner)
- `?wb=cuas&mod=site_defense` — Custom Site Builder, DEM LOS Masking & Real-World OSM GIS Map
- `?wb=cuas&mod=comparator` — A/B Turret Architecture Comparator & Layered Defense
- `?wb=cuas&mod=sensitivity` — Automated OAT Sensitivity & Bottleneck Tornado
- `?wb=cuas&mod=adversarial` — Stage 4 Adversarial Red-vs-Blue ML Optimizer (Hybrid/Robust, Dynamic RL Fire Control, Live HIL Telemetry Replay, 3D Battlespace ML & Beat-the-AI Duel)
- `?wb=sim` / `?wb=3d` / `?wb=firing` — Ballistics & 3D Trajectory Simulation Domain
- `?wb=physics` / `?wb=mcdrag` — Physics & Aero Labs + McCoy MCDRAG 3D CAD
- `?wb=iatg` — UNODA IATG 01.80 / 02.20 Safety & Quantity-Distance Workbench
- `?wb=validation` / `?wb=literature` — Automated V&V Suite (`40` checks & certificates) and Scientific Literature Provenance (`80` primary sources, `96` formulas)

---

## 2. Production REST API (`FastAPI` + `Uvicorn`)

Run locally or behind a reverse proxy:

```bash
pip install -e .
uvicorn ammo_mc.api:app --host 0.0.0.0 --port 8000
```

Enable Bearer / `X-API-Key` authentication for public API deployments:

```bash
export AMMO_MC_API_REQUIRE_AUTH=true
export AMMO_MC_PASSWORD="your-chosen-api-token"
uvicorn ammo_mc.api:app --host 0.0.0.0 --port 8000
```

Send requests with either `Authorization: Bearer <token>` or `X-API-Key: <token>`. Open interactive Swagger documentation at `/docs`.

---

## 3. Persistent Scenario Library Storage

Saved scenarios are stored in SQLite via `ScenarioLibrary`:
- Default location: `~/.ammo_mc/scenarios.db` (or `AMMO_MC_SCENARIO_DB` env var).
- Ephemeral cloud sandboxes without persistent home directories automatically fall back to `<temp>/ammo_mc_scenarios.db`. Use the **Export Library JSON** / **Download Briefing** buttons for permanent archival across container restarts on ephemeral hosts.
