# Ballistics, Dispersion & Counter-UAS Monte Carlo Workbench (`ammo_mc`)

![CI](https://github.com/travisrcarlson/ballistics-monte-carlo-workbench/actions/workflows/ci.yml/badge.svg)

A physics-grounded stochastic Monte Carlo simulation engine and Streamlit workbench for **artillery/small-arms ballistics**, **multi-barrel & shotgun dispersion**, **3D target drone vulnerability**, **scientific model provenance**, and a **13-module Counter-UAS (C-UAS) Multi-Shotgun Turret Engineering Suite**.

> **GOVERNANCE & USE LIMITATION NOTICE:**
> Models and estimators herein are for engineering analysis, range safety planning, and academic research based strictly on unclassified, peer-reviewed literature and standards (IATG, STANAG, McCoy, Lowry-Journée, BRL). All bundled munition presets represent NOTIONAL generic class archetypes.

---

## 🌟 Key Capabilities

### 1. Contextual Per-Workbench UX, Scientific Literature Tab & Counter-UAS Defense
- **Contextual Per-Workbench Popout Sidebar & URL Deep-Linking**: Stateful 9-workbench switcher (`Monte Carlo Simulator`, `C-UAS & Small Arms`, `3D Simulations`, `MCDRAG & 3D CAD`, `Physics & Ballistics Labs`, `UNODA IATG Safety`, `Firing Tables & Met`, `Validation Suite`, `📚 Literature & Sources`) executes only the active workbench, populates the left sidebar exclusively with relevant controls, and syncs shareable URL query params (e.g., `?wb=literature`, `?wb=cuas&mod=raid`, `?wb=cuas&mod=calibration`).
- **Scientific Literature & Provenance Database (`📚 Literature & Sources`)**: Full module-by-module traceability mapping **68 verified references (`S01`–`S68`)**, **96 governed equations** (3 explicit `FLAG` equations), **12 empirical tables/catalogs**, and **46 application modules**, with BibTeX/CSV/JSON exports (`GET /literature`, `ammo-mc literature`).
- **Flexible Spatial Barrel Placement & 3D Target Cross-Section (Modes 4 & 5)**: Circular ring, matrix grid, concentric dual-ring, hexagonal honeycomb, curtain, chevron V, and custom $(Y, Z)$ coordinates with convergent/parallel/splay/staggered pointing and exact 3-axis Euler projection ($\psi, \theta, \phi$).
- **Complete 13-Module C-UAS Turret Engineering Suite**:
  1. **Closed-Loop FCS & Evasive Drones**: Radar / EO-IR / Fused $\alpha\text{-}\beta\text{-}\gamma$ Kalman lead prediction, servo slew/acceleration limits, and sinusoidal/corkscrew/pop-up jink evasion.
  2. **Multi-Drone Swarm Saturation**: Timeline budgets, priority policies, sub-salvo allocation, and reload endurance.
  3. **Advanced Payloads & Programmable Airburst**: Lead #4 Buckshot, Steel T-Shot, Bismuth BB, Tungsten Super-Shot (TSS), Duplex, Saboted Flechettes, and time-fuzed electronic airburst canisters.
  4. **Turret Recoil, Gimbal Loads & Constrained Pareto Optimizer**: 2nd-order recoil buffers, 2-axis gimbal walk ODEs, and structural-limit-constrained optimization.
  5. **Elevated-Fire Pellet Fallout & SDZ Safety Zone**: Vectorized 3-DoF missed-pellet ballistics with ISA 1976 atmosphere, 1/7 wind shear, ocular ($0.06\;\text{J/mm}^2$), skin, and IATG $79\;\text{J}$ footprints.
  6. **Multi-Sensor Acquisition & Kill-Chain Timeline**: Aspect-RCS Ku-band radar, weather-attenuated EO/IR, acoustic array, and passive RF/C2 ESM (`0 m` vs fiber-optic threats).
  7. **Multi-Turret Site Defense, DEM Terrain LOS Masking & GIS Map**: Custom turret/obstacle/NFZ builder, synthetic/uploaded CSV & `.asc` / live Open-Meteo SRTM DEM ridge masking, terrain-adjusted Fallout SDZs, OpenStreetMap overlay, and GeoJSON/KML export.
  8. **3D Animated Intercept Replay**: Time-scrubbed WebGL animation showing evasive drone attitude, ripple shot-stringing, and component impacts.
  9. **Sustained Logistics, Scenario Library & Briefings (`.html`, `.pdf`, `.pptx`)**: Multi-wave thermal heating ($220^\circ\text{C}$ cook-off), magazine reloads, `$/Kill` comparison, SQLite Scenario Library save/diff, and presentation-grade multi-page PDF + 16:9 PowerPoint deck exports.
  10. **A/B Turret Comparator & Layered Defense**: Side-by-side multi-metric scorecard + paired Outer Soft/Hard-Kill vs Inner Multi-Shotgun waves.
  11. **Automated Sensitivity & Bottleneck Tornado**: One-at-a-time $\pm\%$ Tornado sweeps across C-UAS Kill Chain and Artillery `CEP50`.
  12. **Unified Dynamic 3D Tactical Raid Simulator (`?wb=cuas&mod=raid`)**: Time-stepped 3D cooperative multi-turret battlespace unifying sensors, evasive RF/fiber-optic swarms, Weapon-Target Assignment handoff, DEM/building LOS masking, reloads, and barrel heating.
  13. **Field-Test Calibration & Residual Diagnostics (`?wb=cuas&mod=calibration`)**: Upload range CSVs to fit Rayleigh shotgun pattern spread factors, chronograph MV & FM 6-40 $dMV/dT$, and exterior drag form factors via SciPy nonlinear least squares and apply directly to the simulator.

### 2. Multi-Stage Exterior, Interior & IATG Safety Engine (Dedicated Workbenches)
- **Stages 0–8 Ballistics Pipeline**: Lumped-parameter STANAG 4367 interior ballistics, L0–L3 exterior trajectories (3-DoF + STANAG 4355 Modified Point Mass with spin/Coriolis drift + terrain), McCoy MCDRAG (Mach 0.5–5.0), fuze reliability & Beta epistemic UXO estimation, Kingery-Bulmash blast, Gurney/Mott fragmentation, Carleton SSPK, and two-loop epistemic/aleatory UQ (LHS, Sobol, MC).
- **UNODA IATG 01.80 / 02.20 Safety Workbench**: Quantity-Distance (`PBD`, `PTRD`, `IBD`, `VBD`, `TB-15`), `TB-10` brick-house structural damage, `TB-11` Swiss glazing, `TF-10`/`TF-14` stack/RC debris, `TB-13` Axelsson thoracic biomechanics, propellant aging (`AOP-48`), and KML/CZML GIS export.

---

## ☁️ Deploying to Streamlit Community Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for full Streamlit Community Cloud & FastAPI production deployment notes.
1. Connect repository `travisrcarlson/ballistics-monte-carlo-workbench` (branch `main`, main file path `streamlit_app.py`) on `share.streamlit.io`.
2. **Access Password**: `src/ammo_mc/auth.py` enforces rate-limited password verification via `st.secrets["APP_PASSWORD"]` → `AMMO_MC_PASSWORD` environment variable → salted `PBKDF2-HMAC-SHA256` fallback digest (zero plaintext secrets stored in Git). Set `APP_PASSWORD = "..."` under **App settings → Secrets**.

---

## 🚀 Local Quick Start

```bash
# Install in editable mode (including pytest/httpx dev tools)
pip install -e ".[dev]"

# Launch the password-gated Streamlit workbench locally
streamlit run streamlit_app.py
```

### CLI & FastAPI REST Server

```bash
# Legacy ballistics & validation
ammo-mc simulate --preset NOTIONAL-AR-01 --fidelity L2 --trials 200
ammo-mc validate

# Counter-UAS CLI workflows (kill-chain, GIS exports, A/B comparator, layered, unified-raid, sweep, literature)
ammo-mc cuas kill-chain --fiber-optic
ammo-mc cuas site-defense --out-geojson exports/site.geojson --out-kml exports/site.kml
ammo-mc cuas compare-ab --out-csv exports/scorecard.csv
ammo-mc cuas layered --outer rf_gnss_jammer --fo-frac 0.30
ammo-mc cuas unified-raid --terrain rolling_hills_fob --drones 12
ammo-mc cuas sweep --param speed --start 15 --stop 40 --steps 6
ammo-mc literature --workbench cuas

# FastAPI REST server (/literature, /cuas/unified-raid, /cuas/briefing/pdf, /cuas/briefing/pptx, /calibrate/*, etc.)
# Optional Bearer / X-API-Key guard: set AMMO_MC_API_REQUIRE_AUTH=true and AMMO_MC_PASSWORD=<secret>
uvicorn ammo_mc.api:app --reload
```

---

## 🧪 Testing & Continuous Integration

Run the full test suite (all unit, calculator, Knowledge Base `V-01a..V-12`, 3D visualization, C-UAS turret, and authentication tests):

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs syntax verification and `pytest` automatically on every push and pull request to `main`.

---

## 📁 Repository Layout

```text
.
├── .github/workflows/ci.yml    # GitHub Actions CI (pytest + py_compile)
├── .streamlit/config.toml      # Dark-mode & headless server defaults
├── requirements.txt            # Streamlit Community Cloud runtime dependencies
├── streamlit_app.py            # Cloud & local root Streamlit entrypoint
├── DEPLOYMENT.md               # Streamlit Cloud & FastAPI deployment guide
├── pyproject.toml              # Package metadata and dependencies
├── Ammunition Monte Carlo Knowledge Base.xlsx
├── src/ammo_mc/
│   ├── data_tables/            # Knowledge Base JSON tables (68 sources, 96 formulas, provenance map)
│   ├── auth.py                 # Rate-limited password verification (Secrets / Env / PBKDF2)
│   ├── literature.py           # Scientific Literature Catalog, Provenance & BibTeX/CSV Engine
│   ├── terrain_dem.py          # DEM Terrain Grids, 3D Ridge LOS Masking & Slope Fallout SDZ
│   ├── cuas_defense.py         # 5-module C-UAS FCS, swarms, airburst, recoil & SDZ
│   ├── cuas_mission.py         # Kill chain, multi-turret site/NFZ, 3D replay, logistics & reports
│   ├── cuas_analytics.py       # A/B comparator, layered defense, GeoJSON/KML & Tornado sensitivity
│   ├── cuas_raid.py            # Unified Dynamic 3D Cooperative Multi-Turret Raid Simulator
│   ├── calibration.py          # Field-Test Pattern, Chronograph & Exterior Drag Form-Factor Fitting
│   ├── scenario_library.py     # SQLite Persistent Scenario Library (Save / Load / Diff / JSON Sync)
│   ├── briefing_export.py      # Multi-Page Landscape PDF & 16:9 PowerPoint (.pptx) Briefing Engine
│   ├── small_arms_pkg/         # Modular small-arms, shotgun pattern & anti-drone engines (+ facade small_arms.py)
│   ├── vis_3d_pkg/             # Modular 3D/2D WebGL Plotly builders (+ facade vis_3d.py)
│   ├── vis_cuas*.py            # C-UAS mission, analytics, raid & calibration visualizations
│   ├── streamlit_app.py        # Contextual workbench router & URL deep-link entrypoint
│   ├── ui/                     # Modular workbench UI views (9 workbenches incl. literature.py)
│   └── ...                     # Core Stages 0–8 ballistics, IATG, API & CLI modules
└── tests/                      # Automated pytest verification suites
```
