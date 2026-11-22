# Ammunition Monte Carlo Simulator (`ammo_mc`)

A high-performance, physics-grounded stochastic Monte Carlo simulation engine and workbench for ammunition effects, trajectory dispersion, fuze reliability, and blast/fragmentation hazard modeling.

> **GOVERNANCE & USE LIMITATION NOTICE:**
> Models and estimators herein are for engineering analysis, range safety planning, and academic research based strictly on unclassified, peer-reviewed literature and standards (IATG, STANAG, McCoy, BRL). All bundled munition presets represent NOTIONAL generic class archetypes.

---

## 🌟 Key Capabilities

1. **Multi-Stage Physical Pipeline (Stages 0–8)**
   - **Launch & Interior Ballistics**: Lumped-parameter ODE integrator (STANAG 4367), quadrant elevation (QE) solver, muzzle velocity variation.
   - **Exterior Trajectory Simulation**:
     - **L0**: Empirical bivariate dispersion modeling.
     - **L1**: 3-DoF vectorised point-mass integration (RK4 / adaptive RK45) with ISA 1976 atmosphere and altitude wind profiles.
     - **L2**: 4-DoF STANAG 4355 Modified Point Mass (MPM) with gyroscopic spin drift, Coriolis/centrifugal acceleration, crosswind drag, and yaw of repose.
     - **L3**: Fully coupled multi-round battery/salvo dispersion with occasion-shared biases and terrain elevation intersection.
   - **McCoy (1974) MCDRAG**: Axisymmetric projectile zero-yaw drag coefficient estimation across Mach 0.5–5.0 with full component breakdown (head, skin friction, base, boattail).
   - **Fuze Reliability & UXO**: Binomial detonation reliability, Beta epistemic dud rate estimation, time/HOB airburst dispersion, and unexploded ordnance mapping.
   - **Terminal Blast**: Hopkinson-Cranz scaling, Kingery-Bulmash 7-parameter polynomial blast curves, Kinney-Graham, incident/reflected pressure, Friedlander impulse, and IATG UN 01.80 safety distances.
   - **Fragmentation & Debris**: Gurney velocity (cylindrical, spherical, axial), Mott fragment distribution, Taylor ejection angle, fragment aerodynamic velocity decay, and 79 J hazardous fragment criteria.
   - **Target Vulnerability**: Carleton damage model, cookie-cutter, Single-Shot Probability of Kill (SSPK), multi-round salvo kill probability (correlated vs independent), and mechanism union.

2. **Rigorous Uncertainty Quantification**
   - Samplers: Latin Hypercube Sampling (LHS), Sobol quasi-random sequences, and pseudo-random Monte Carlo.
   - Occasion-level shared biases (calibration, weather, lot variation) vs round-level stochastic noise.
   - Two-loop epistemic/aleatory uncertainty quantification.
   - Adaptive confidence interval stopping rules (Wilson score 95% CI on SSPK and dud rates).
   - Global sensitivity analysis ranking (Sobol indices / Pearson correlation).

3. **Multi-Interface Architecture**
   - **Command Line (`ammo-mc`)**: Fast batch simulations, validation runs, MCDRAG estimates, and export generation.
   - **FastAPI REST API**: Headless endpoints for automated workflows, parameter querying, and remote simulations.
   - **Streamlit Workbench**: Interactive Plotly dashboard featuring POI scatter plots, CEP50/95 ellipses, trajectory profiles, 2D blast/frag hazard contours, and live formula calculators.

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install in editable mode with dependencies
pip install -e .
```

### 1. Command Line Interface (CLI)

```bash
# View available commands and notional presets
ammo-mc presets

# Run a Monte Carlo simulation (e.g. 155mm HE projectile, 200 rounds)
ammo-mc simulate --preset NOTIONAL-AR-01 --fidelity L2 --trials 200

# Generate an interactive 3D trajectory bundle & terrain simulation HTML
ammo-mc sim-3d --preset NOTIONAL-AR-01 --fidelity L2 --trials 100 --trajectories 12 --out ./sim_3d.html

# Export complete artifacts (CSV, JSON, GeoJSON, PNG report, PDF report, and 3D HTML)
ammo-mc simulate --preset NOTIONAL-SA-01 --trials 100 --export-all --export-3d --out-dir ./exports

# Execute McCoy MCDRAG aerodynamic drag analysis
ammo-mc mcdrag --calibre-m 0.155 --l-nose-cal 2.5 --l-cyl-cal 2.0 --l-bt-cal 0.5

# Run the 22-case Knowledge Base verification suite
ammo-mc validate
```

### 2. FastAPI REST Server

```bash
# Launch the API server
ammo-mc serve-api --host 127.0.0.1 --port 8000

# Or using uvicorn directly
uvicorn ammo_mc.api:app --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

Key Endpoints:
- `GET /health` — Service status and governance notice.
- `GET /taxonomy` — Munition taxonomy families.
- `GET /formulas` — 96 governing formulas with source IDs and FLAG warnings.
- `GET /presets` & `GET /presets/{preset_id}` — Shipped notional presets.
- `POST /mcdrag` — Aerodynamic zero-yaw drag calculations.
- `GET /validate` — Run the 22-case validation suite against Excel baselines.
- `POST /simulate` — Run stochastic Monte Carlo simulations.
- `POST /export/geojson` — Export RFC 7946 hazard and dispersion contours.

### 3. Interactive Streamlit Dashboard

```bash
# Launch the Streamlit Workbench
ammo-mc ui

# Or via streamlit directly
streamlit run src/ammo_mc/streamlit_app.py
```

Features:
- **Monte Carlo Simulator**: Interactive scenario tuning, terrain slope selection, POI scatter plot with target box, trajectory altitude profile, 2D hazard field, sensitivity tornado plot, and one-click artifact download buttons.
- **🌐 3D Simulations**: Interactive 3D spatial models featuring 3D Monte Carlo trajectory bundles with velocity/altitude colormaps, 3D digital elevation terrain meshes, 3D target bounding volumes, ground POI impact scatters, 3D blast overpressure iso-surfaces (100 kPa / 35 kPa / 14 kPa), 3D fragmentation spray cones, multiple camera perspectives, and one-click standalone interactive 3D HTML exports.
- **McCoy MCDRAG**: Geometric sliders and live drag breakdown chart across Mach numbers.
- **Formulas & Calculators**: Live calculators for Gurney velocity, Mott distribution, Kingery-Bulmash blast, and Carleton SSPK.
- **Validation Suite**: One-click execution comparing all 22 test cases against Excel and literature baselines.

---

## 🧪 Testing and Validation

Run the full pytest suite:

```bash
pytest
```

The test suite covers:
- `test_validation_cases.py`: All 22 test cases (V-01a .. V-12) checked against Excel cell formulas and source literature.
- `test_calculators.py`: Formula-level unit tests across interior, exterior, fuze, blast, fragmentation, and vulnerability.
- `test_formulas_all.py`: Verification of all registered Knowledge Base formulas.
- `test_pipeline_models.py`: Physics model-form checks (vacuum trajectory vs EB-11, L1 vs L2 MPM gyroscopic spin drift, correlated bias vs independent salvo Pk, terrain intersection, adaptive CI stopping, and vectorised throughput benchmarks).
- `test_api.py`: FastAPI REST endpoint contract and response verification.
- `test_cli.py`: Typer CLI runner execution tests.

---

## 📁 Repository Layout

```
.
├── src/ammo_mc/
│   ├── data_tables/            # Extracted JSON Knowledge Base data tables
│   ├── api.py                  # FastAPI REST endpoints
│   ├── cli.py                  # Typer CLI application
│   ├── data.py                 # Knowledge Base loader and table lookup
│   ├── environment.py          # Atmosphere (ISA 1976) & wind profiles
│   ├── export.py               # CSV, JSON, GeoJSON, PNG, PDF exporters
│   ├── exterior.py             # Trajectory integrators (L0, L1, L2 MPM, terrain)
│   ├── formulas_registry.py    # Formula registry & FLAG governance tracking
│   ├── fuze.py                 # Fuze reliability, dud rates & airburst geometry
│   ├── gis.py                  # DEM elevation grids & RFC 7946 GeoJSON contours
│   ├── interior.py             # Lumped-parameter interior ballistics ODE
│   ├── launch.py               # Aiming, jump, and QE firing solution solvers
│   ├── mcdrag.py               # McCoy (1974) MCDRAG drag estimator
│   ├── penetration.py          # Lanz-Odermatt, Lambert-Jonas, Shaped-Charge
│   ├── pipeline.py             # Stages 0-8 Monte Carlo orchestrator
│   ├── plugins.py              # 6-DOF and guidance extension plugin interfaces
│   ├── presets.py              # 14 notional munition presets
│   ├── schemas.py              # Pydantic data schemas
│   ├── stats.py                # Statistical distributions & sampling library
│   ├── streamlit_app.py        # Streamlit + Plotly interactive workbench
│   ├── terminal_blast.py       # Blast pressure, impulse & building damage
│   ├── terminal_frag.py        # Gurney, Mott, fragment spray & hazardous KE
│   ├── validation.py           # V-01a..V-12 validation suite runner
│   └── vulnerability.py        # Carleton, cookie-cutter, SSPK & salvo Pk
├── tests/                      # 71 comprehensive unit and integration tests
├── scripts/                    # KB table extractor scripts
├── pyproject.toml              # Build config and dependency definitions
└── README.md                   # Project documentation
```
