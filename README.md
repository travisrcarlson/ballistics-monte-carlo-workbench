# Ballistics, Dispersion & Counter-UAS Monte Carlo Workbench (`ammo_mc`)

![CI](https://github.com/travisrcarlson/ballistics-monte-carlo-workbench/actions/workflows/ci.yml/badge.svg)

A physics-grounded stochastic Monte Carlo simulation engine and Streamlit workbench for **artillery/small-arms ballistics**, **multi-barrel & shotgun dispersion**, **3D target drone vulnerability**, and an **11-module Counter-UAS (C-UAS) Multi-Shotgun Turret Engineering Suite**.

> **GOVERNANCE & USE LIMITATION NOTICE:**
> Models and estimators herein are for engineering analysis, range safety planning, and academic research based strictly on unclassified, peer-reviewed literature and standards (IATG, STANAG, McCoy, Lowry-Journée, BRL). All bundled munition presets represent NOTIONAL generic class archetypes.

---

## 🌟 Key Capabilities

### 1. Contextual Per-Workbench UX & Counter-UAS Defense (`🛸 C-UAS & Small Arms`)
- **Contextual Per-Workbench Popout Sidebar & URL Deep-Linking**: Stateful 8-workbench switcher (`Monte Carlo Simulator`, `C-UAS & Small Arms`, `3D Simulations`, `MCDRAG & 3D CAD`, `Physics & Ballistics Labs`, `UNODA IATG Safety`, `Firing Tables & Met`, `Validation Suite`) executes only the active workbench, dynamically populates the left sidebar exclusively with relevant controls, and syncs shareable URL query params (e.g., `?wb=cuas&mod=comparator`).
- **Flexible Spatial Barrel Placement Architectures (Mode 5)**: Circular ring, rectangular matrix grid, concentric dual-ring (heterogeneous TSS core + buckshot perimeter), hexagonal honeycomb, horizontal/vertical curtains, chevron V, and custom $(Y, Z)$ coordinates, with convergent, parallel, divergent splay net, and staggered tiling pointing modes plus simultaneous vs sequential/balanced-pair ripple firing.
- **3D Target Drone Cross-Section Presentation (Modes 4 & 5)**: Exact 3-axis Euler projection ($\psi, \theta, \phi$) and component vulnerability (avionics/fuselage core, LiPo battery/payload, structural arms, motors, finite-thickness rotor sweeps) across Head-On Frontal, Flank, Oblique Quartering, Tail, Overhead, and Underbelly aspects, plus multi-barrel regulation crossover and rotary Gatling dynamics.
- **Complete 11-Module C-UAS Turret Engineering Suite**:
  1. **Closed-Loop FCS & Evasive Drones**: Radar / EO-IR / Fused $\alpha\text{-}\beta\text{-}\gamma$ Kalman lead prediction, servo slew/acceleration limits, and sinusoidal/corkscrew/pop-up jink evasion.
  2. **Multi-Drone Swarm Saturation**: Timeline budgets, priority policies (shortest time-to-breach, nearest range, min slew), sub-salvo allocation, and reload endurance.
  3. **Advanced Payloads & Programmable Airburst**: Lead #4 Buckshot, Steel T-Shot, Bismuth BB, Tungsten Super-Shot (TSS), Duplex, Saboted Flechettes, and time-fuzed electronic airburst canisters.
  4. **Turret Recoil, Gimbal Loads & Constrained Pareto Optimizer**: 2nd-order recoil buffers, 2-axis gimbal walk ODEs, and structural-limit-constrained design optimization.
  5. **Elevated-Fire Pellet Fallout & SDZ Safety Zone**: Vectorized 3-DoF missed-pellet ballistics with ISA 1976 atmosphere, 1/7 wind shear, ocular ($0.06\;\text{J/mm}^2$), skin, and IATG $79\;\text{J}$ footprints.
  6. **Multi-Sensor Acquisition & Kill-Chain Timeline**: Aspect-RCS Ku-band radar, weather-attenuated EO/IR, wind-degraded acoustic array, and passive RF/C2 ESM (drops to $0\;\text{m}$ for fiber-optic/autonomous threats) with AI classification & human-in-the-loop latency budgets.
  7. **Multi-Turret Site Defense, Custom Builder & GIS Map**: Editable custom turret/obstacle/NFZ builder (`cuas_site_v1` JSON), 3D obstacle LOS masking, blind-spot detection, real-world OpenStreetMap overlay (`Scattermap`), and 1-click **GeoJSON & Google Earth 3D `.kml` export**.
  8. **3D Animated Intercept Replay**: Time-scrubbed WebGL animation (`▶ Play` / timeline slider) showing moving/evading 3D drone attitude, balanced-pair ripple shot-stringing, and component impacts.
  9. **Sustained Logistics, Barrel Heat & Cost-per-Kill (`$/Kill`) + Report Export**: Multi-wave thermal heating/Newton cooling ($220^\circ\text{C}$ cook-off limit), magazine reload/jam stoppages, 7-family `$/kill` & mass comparison, JSON Scenario Save/Load, and one-click standalone HTML/Print-to-PDF Trade-Study Report export.
  10. **A/B Turret Comparator & Layered C-UAS Defense**: Side-by-side multi-metric scorecard (`Kill-Chain Pk`, `Swarm Survival`, `Recoil`, `SDZ`, `Cook-off Margin`, `$/Kill`) on a normalized radar + delta table, plus paired Outer Soft/Hard-Kill (`RF Jammer`, `Interceptor Drone`, `30mm Airburst`, `10 kW HEL`) vs Inner Multi-Shotgun defense against mixed RF + fiber-optic waves.
  11. **Automated Sensitivity & Bottleneck Tornado**: One-at-a-time $\pm\%$ Tornado bottleneck sweeps across both C-UAS Kill Chain and Artillery `CEP50` with prioritized engineering recommendations.

### 2. Multi-Stage Exterior, Interior & IATG Safety Engine (Dedicated Workbenches)
- **Stages 0–8 Ballistics Pipeline**: Lumped-parameter STANAG 4367 interior ballistics, L0–L3 exterior trajectories (3-DoF + STANAG 4355 Modified Point Mass with spin/Coriolis drift + terrain), McCoy MCDRAG (Mach 0.5–5.0), fuze reliability & Beta epistemic UXO estimation, Kingery-Bulmash blast, Gurney/Mott fragmentation, Carleton SSPK, and two-loop epistemic/aleatory UQ (LHS, Sobol, MC).
- **UNODA IATG 01.80 / 02.20 Safety Workbench**: Quantity-Distance (`PBD`, `PTRD`, `IBD`, `VBD`, `TB-15`), `TB-10` brick-house structural damage, `TB-11` Swiss glazing, `TF-10`/`TF-14` stack/RC debris, `TB-13` Axelsson thoracic biomechanics, propellant aging (`AOP-48`), and KML/CZML GIS export.

---

## ☁️ Deploying to Streamlit Community Cloud

The repository is configured for zero-edit deployment on **Streamlit Community Cloud**:
1. Connect repository `travisrcarlson/ballistics-monte-carlo-workbench` (branch `main`, main file path `streamlit_app.py`) on `share.streamlit.io`.
2. **Access Password**: The launch screen requires a session password (`src/ammo_mc/auth.py` verifies via `st.secrets["APP_PASSWORD"]` → `AMMO_MC_PASSWORD` environment variable → salted `PBKDF2-HMAC-SHA256` fallback digest, with no plaintext secrets stored in Git). To override the password in Streamlit Cloud, set `APP_PASSWORD = "..."` under **App settings → Secrets**.

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

# Counter-UAS CLI workflows (kill-chain, GIS exports, A/B comparator, layered, sensitivity, sweep)
ammo-mc cuas kill-chain --fiber-optic
ammo-mc cuas site-defense --out-geojson exports/site.geojson --out-kml exports/site.kml
ammo-mc cuas compare-ab --out-csv exports/scorecard.csv
ammo-mc cuas layered --outer rf_gnss_jammer --fo-frac 0.30
ammo-mc cuas sweep --param speed --start 15 --stop 40 --steps 6

# FastAPI REST server (/cuas/array, /cuas/kill-chain, /cuas/site-defense, /cuas/compare-ab, /cuas/layered-defense, /cuas/sensitivity, /cuas/report)
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
├── pyproject.toml              # Package metadata and dependencies
├── Ammunition Monte Carlo Knowledge Base.xlsx
├── src/ammo_mc/
│   ├── data_tables/            # Exported Knowledge Base JSON tables
│   ├── auth.py                 # Password gate verification (Secrets / Env / PBKDF2)
│   ├── cuas_defense.py         # 5-module C-UAS FCS, swarms, airburst, recoil & SDZ
│   ├── cuas_mission.py         # Kill chain, multi-turret site/NFZ, 3D replay, logistics & reports
│   ├── cuas_analytics.py       # A/B comparator, layered defense, GeoJSON/KML & Tornado sensitivity
│   ├── small_arms.py           # Multi-barrel, shotgun patterns & 3D drone cross-section geometry
│   ├── vis_3d.py               # 3D WebGL trajectory, blast, drone & multi-barrel plots
│   ├── vis_cuas.py             # C-UAS turret dashboards
│   ├── vis_cuas_mission.py     # Kill-chain, site-map, 3D animation & logistics charts
│   ├── vis_cuas_analytics.py   # A/B radar, layered attrition, OSM Scattermap & Tornado plots
│   ├── streamlit_app.py        # Contextual workbench router & URL deep-link entrypoint
│   ├── ui/                     # Modular workbench UI views (simulator, cuas, 3D, mcdrag, physics, iatg, firing, validation)
│   └── ...                     # Core Stages 0–8 ballistics, IATG, API & CLI modules
└── tests/                      # Automated pytest verification suites
```
