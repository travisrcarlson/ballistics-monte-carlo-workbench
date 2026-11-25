# Ballistics, Dispersion & Counter-UAS Monte Carlo Workbench (`ammo_mc`)

![CI](https://github.com/travisrcarlson/ballistics-monte-carlo-workbench/actions/workflows/ci.yml/badge.svg)

A physics-grounded stochastic Monte Carlo simulation engine and Streamlit workbench for **artillery/small-arms ballistics**, **multi-barrel & shotgun dispersion**, **3D target drone vulnerability**, **verified scientific provenance**, and a **4-Stage Counter-UAS (C-UAS) Multi-Shotgun Turret Engineering Suite**.

> **GOVERNANCE & USE LIMITATION NOTICE:**
> Models and estimators herein are for engineering analysis, range safety planning, and academic research based strictly on unclassified, peer-reviewed literature and standards (IATG, STANAG, McCoy, Lowry-Journée, BRL). All bundled munition presets represent NOTIONAL generic class archetypes.

---

## 🌟 Key Capabilities

### 1. Consolidated 6-Domain Focus Mode UX & Counter-UAS 4-Stage Workflow
- **Task-First Navigation, 3-Click Turret Sizing Wizard, Focus Mode & URL Deep-Linking**: Streamlined 6-domain workbench switcher (`Home`, `C-UAS & Multi-Barrel`, `Ballistics & 3D Sim`, `Physics & Aero Labs`, `Safety & IATG`, `Audit & Sources`), instant **3-Click C-UAS Turret Sizing Wizard** on Home (outputs Top-3 feasible builds + 1-click PDF spec sheet & `.dxf` muzzle CAD), and global **Standard / Expert Mode (`Focus Mode`)** toggle. Supports shareable URL query parameters (`?wb=home`, `?wb=cuas&mod=array`, `?wb=cuas&mod=raid`, `?wb=cuas&mod=adversarial`, `?wb=validation`, `?wb=literature`).
- **Scientific Literature, V&V Certificates & Model Provenance (`Audit & Sources`)**: Full module-by-module traceability mapping **100 verified primary sources (`S01`–`S100`)**, **96 governed equations**, empirical catalogs, **44 core V&V checks (`B-01..B-22` + `V-01a..V-12`)**, and **46 deterministic C-UAS workbook oracles** (`WB-F01`/`WB-F02` salvo brackets, Walsh/NOL/CIPM-2007, Wen/FAA/THOR, SAAMI recoil, Wilson/Clopper-Pearson/Agresti-Coull CIs) with exportable V&V audit certificates (`GET /literature`, `ammo-mc cuas workbook-vv`).
- **v2.0 Scientific Rigor & Epistemic Transparency**: Every C-UAS stage carries **`V1`–`V4` weakest-link verification depth grading**, inline contextual **Counter-Evidence & Open Gaps** disclosures, **SE-F14** pellet-count physical consistency audits, **closing-target Usable Engagement Windows `[R_min_safe, R_max]`** with wreck carry-on inertia, explicit **`FC-T04` serial vision/actuator dead-time chains (~141 ms)**, **dual RCC 321-23 / Kneubuehl / AR 385-63 / UK JSP 403** safety overlays, and **Common Random Numbers (CRN)** paired A/B comparisons.
- **Stage 1 — Multi-Shotgun Array Designer, Hardware SWaP-C/OTM, 3D `.stl` CAD Ray-Cast & Muzzle-Plate CAD (`?wb=cuas&mod=array`)**:
  - Flexible muzzle mount geometries (`circular`, `grid`, `concentric`, `honeycomb`, `fan_horizontal`, `fan_vertical`, `v_shape`, and custom `(Y, Z, choke)` coordinates), homogeneous or **hybrid dual-ring mixed-load barrels**, convergent/parallel/splay/staggered pointing, ripple schedules, 3-axis Euler drone cross-section projection ($\psi, \theta, \phi$), **vectorized batch pellet impact kinematics**, **Recht-Ipson oblique ricochet $V_{50}(\theta)$ cutoff**, **short-range shot-cup/wad separation**, **watertight ASCII/binary `.stl` drone CAD mesh ray-casting**, and a **global High-Fidelity Physics Mode sidebar toggle** synced across Stages 1–4.
  - **Custom Drone & Custom Shell Builders**: Define custom quadcopter, hexacopter, octocopter, or fixed-wing threat geometries/lethality thresholds and custom pellet materials/diameters/charges/40-yd pattern percentages with JSON preset round-tripping.
  - **Effective Range ($R_{80}$ / $R_{50}$), Hardware SWaP-C, OTM Stabilization, Bolt-Shear FEA & 2D/3D CAD**: Compute single-load and multi-load $P_{\text{kill}}(R)$ curves, 2D crossing-speed $\times$ slant-range contours, full sub-assembly mass/inertia/servo power/BOM estimates, bolt-shear & ligament FEA sanity checks, off-road UGV/4×4 gyro stabilization envelopes, and 1-click **AutoCAD `.dxf`, `.svg`, and extruded 3D `.stl` muzzle-clamp manufacturing files**.
- **Stage 2 — Tracking, Swarms, Airburst & Structural Recoil**:
  - **Closed-Loop FCS & Evasive Drones (`?wb=cuas&mod=fcs`)**: Radar / EO-IR / Fused $\alpha\text{-}\beta\text{-}\gamma$ Kalman lead prediction, servo slew/acceleration limits, and sinusoidal/corkscrew/pop-up jink evasion.
  - **Multi-Drone Swarm Saturation (`?wb=cuas&mod=swarm`)**: Timeline budgets, priority policies, sub-salvo allocation, and reload endurance.
  - **Advanced Payloads & Programmable Airburst (`?wb=cuas&mod=payloads`)**: Lead #4 Buckshot, Steel T-Shot, Bismuth BB, Tungsten Super-Shot (TSS), Duplex, Saboted Flechettes, and time-fuzed airburst canisters.
  - **Turret Recoil, Gimbal Loads & Constrained Pareto Optimizer (`?wb=cuas&mod=recoil`)**: 2nd-order recoil buffers, 2-axis gimbal walk ODEs, and structural-limit-constrained optimization.
- **Stage 3 — Safety Envelopes, Kill Chain, Site Defense & 3D Raid**:
  - **Elevated-Fire Pellet Fallout, Acoustic Blast & Urban Collateral Safety (`?wb=cuas&mod=sdz`)**: Vectorized 3-DoF missed-pellet ballistics with ISA 1976 atmosphere, directional 140/165/175 dB muzzle-blast contours, friendly direct-fire danger cones, falling-pellet KE tiers (`Lethal >15 J`..`Safe <0.5 J`), and urban clearance verdicts.
  - **Multi-Sensor Acquisition & Kill-Chain Timeline (`?wb=cuas&mod=kill_chain`)**: Aspect-RCS Ku-band radar, weather-attenuated EO/IR, acoustic array, and passive RF/C2 ESM (`0 m` vs fiber-optic threats).
  - **Multi-Turret Site Defense, DEM Terrain LOS Masking & GIS Map (`?wb=cuas&mod=site_defense`)**: Custom turret/obstacle/NFZ builder, synthetic/uploaded CSV & `.asc` / live Open-Meteo SRTM DEM ridge masking, terrain-adjusted Fallout SDZs, and GeoJSON/KML export.
  - **Unified Dynamic 3D Tactical Raid & Intercept Replay (`?wb=cuas&mod=raid`)**: Time-stepped 3D cooperative multi-turret battlespace supporting heterogeneous mixed swarm packages, optional outer layered effectors (`RF Jammer`, `HPM Screen`, `35mm Airburst`), and animated 3D ripple shot-stringing replay.
- **Stage 4 — Trade Studies, Closed-Loop Test Recalibration, Autonomy & Briefings**:
  - **A/B Comparator, Layered Defense & Sensitivity Tornado (`?wb=cuas&mod=comparator`, `sensitivity`)**: Multi-criteria scorecard, paired Outer Soft/Hard-Kill vs Inner Multi-Shotgun waves, and one-at-a-time Tornado sweeps.
  - **Sustained Logistics, Scenario Library & Briefings (`.html`, `.pdf`, `.pptx`)**: Thermal cook-off margins, `$/Kill`, SQLite Scenario Library save/diff, multi-page PDF + 16:9 PowerPoint decks (including evolved Nash ML scorecard, Empirical Calibration Certificate, and Scientific V&V / Live SPRT slides).
  - **Closed-Loop Test Recalibration, Native Range-Instrument Auto-Import, 4-Fiducial Photo Rectification & Scientific V&V (`?wb=cuas&mod=calibration`)**: Organized into **4 clean workflow pillars** (`Ballistics & Range Instruments`, `Targets, Flight Tracks & Hardware`, `Digital Twin, Drift & Test Planner`, `Scientific V&V, UQ & Live SPRT`). Supports native auto-detection & ingestion for **LabRadar CSV**, **Garmin Xero C1 Pro (`.csv`/`.xlsx`)**, **Kestrel LiNK**, and **PX4/ArduPilot flight logs**, printable **4-fiducial checkerboard target board PDF** with pure-NumPy corner auto-detection & DLT rectification, **interactive Live Range-Day Wald SPRT tap-tracker** with exact Clopper-Pearson confidence intervals, ASME V&V 20 blind holdout validation, Ferson-Oberkampf AVM, and 2-loop LHS epistemic UQ P-boxes.
  - **Adversarial Red-vs-Blue Competing Monte Carlo, RL Fire Control & Live HIL Inspector (`?wb=cuas&mod=adversarial`)**: Co-evolve homogeneous or hybrid dual-ring shotgun platforms against evasive drone threats under calibrated/weather/OTM CVaR-20% uncertainty using Gaussian Process (Matérn-5/2) Bayesian surrogates and exact LP Nash equilibria. Includes **Antithetic ES Dynamic Shot-by-Shot RL Fire Control**, **Live Millisecond HIL Policy Telemetry Replay**, **Multi-Turret 3D Battlespace Co-Evolution** (with 1-click Raid handoff), and **Interactive "Beat the AI" Duel Root-Cause Attribution**.

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
ammo-mc cuas adversarial --generations 5 --episodes 18 --hybrid --robust --out-csv exports/adversarial_dataset.csv
ammo-mc cuas rl-policy --generations 10 --barrels 8 --hybrid
ammo-mc cuas battlespace-ml --terrain rolling_hills_fob --generations 2
ammo-mc cuas track-replay --scenario evasive_fpv_dive --barrels 6
ammo-mc cuas swap-c --barrels 8 --layout concentric --out-dxf exports/muzzle_plate.dxf --out-stl exports/muzzle_plate.stl
ammo-mc cuas wizard --drone fpv_quad_small --range-m 45 --mount light_ugv
ammo-mc cuas calibrate-twin --drone fpv_quad_small --range-m 45 --save-name Proto-V2
ammo-mc cuas safety-envelope --barrels 6 --elevation 35 --env urban_permissive
ammo-mc cuas hil-trace --scenario evasive_fpv_dive --dt-ms 100
ammo-mc cuas salvo-brackets --n-shots 4
ammo-mc cuas engagement-window --speed 25 --altitude 15 --drone-class fpv_cfrp
ammo-mc cuas rigor-check --ammo 12G_NO4_BUCK_C_UAS --barrels 6
ammo-mc cuas workbook-vv
ammo-mc literature --workbench cuas

# FastAPI REST server (/cuas/calibrate/*, /cuas/cad/stl, /cuas/hardware/fea, /cuas/safety-envelope, /cuas/hil-trace, /export/calibration-certificate)
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
│   ├── data_tables/            # Knowledge Base JSON tables (100 sources, 96 formulas, provenance map)
│   ├── auth.py                 # Rate-limited password verification (Secrets / Env / PBKDF2)
│   ├── literature.py           # Scientific Literature Catalog, Provenance & BibTeX/CSV Engine
│   ├── terrain_dem.py          # DEM Terrain Grids, 3D Ridge LOS Masking & Slope Fallout SDZ
│   ├── cuas_defense.py         # 5-module C-UAS FCS, swarms, airburst, recoil & SDZ
│   ├── cuas_mission.py         # Kill chain, site defense, 3D replay, logistics, JSON & HTML reports
│   ├── cuas_analytics.py       # A/B comparator, layered defense, GeoJSON/KML & Tornado sensitivity
│   ├── cuas_raid.py            # Unified Dynamic 3D Cooperative Multi-Turret Raid Simulator (Heterogeneous & Layered)
│   ├── cuas_adversarial.py     # Competing Monte Carlo Co-Evolution, Hybrid/Robust & GP ML Protection Optimizer
│   ├── cuas_rl_policy.py       # Dynamic Shot-by-Shot Neural Fire-Control Policy (Antithetic ES Trainer)
│   ├── cuas_battlespace_ml.py  # Multi-Turret 3D Battlespace & Mixed Swarm Co-Evolution Engine
│   ├── cuas_track_replay.py    # Recorded ENU / GPS-MAVLink / PX4-ArduPilot Track Engagement Replay Engine
│   ├── cuas_hardware.py        # SWaP-C Estimator, On-the-Move Gyro Stabilization, Bolt-Shear FEA & DXF/SVG/STL CAD
│   ├── cuas_calibration_twin.py # Closed-Loop Multi-Range/Fiducial-Photo/Bayesian/Bench Calibration & Persistent SQLite Digital Twin
│   ├── cuas_scientific_vv.py   # ASME V&V 20 Holdout, UQ P-Boxes, Live Wald SPRT, E-01..E-06 & Native Instrument Parsers
│   ├── cuas_safety_envelope.py # Directional Muzzle-Blast Acoustic Contours, Friendly Cone & Urban Collateral Safety
│   ├── cuas_wizard.py          # 3-Click Turret Sizing Wizard & Beat-the-AI Duel Root-Cause Diagnostics
│   ├── calibration.py          # Field-Test Pattern, Chronograph & Exterior Drag Form-Factor Fitting
│   ├── scenario_library.py     # SQLite Persistent Scenario Library (Save / Load / Diff / JSON Sync)
│   ├── briefing_export.py      # Multi-Page Landscape PDF (incl. Scientific V&V), Spec Sheet PDF & 16:9 PPTX Engine
│   ├── small_arms_pkg/         # Vectorized small-arms, oblique ricochet, STL ray-cast & range/speed envelopes (+ small_arms.py)
│   ├── vis_3d_pkg/             # Modular 3D/2D WebGL Plotly builders (+ facade vis_3d.py)
│   ├── vis_cuas*.py            # C-UAS mission, analytics, range/heatmap, raid, calibration, adversarial & V&V visualizations
│   ├── streamlit_app.py        # Contextual 6-domain workbench router & Global HF Physics / Digital Twin toggles
│   ├── ui/cuas_pkg/            # Modular 4-stage C-UAS UI package (+ facade ui/cuas.py)
│   └── ...                     # Core Stages 0–8 ballistics, IATG, V&V certificates, API & CLI modules
└── tests/                      # Automated pytest verification suites
```
