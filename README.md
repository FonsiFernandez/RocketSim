# Kinetica

![img.png](src%2Fres%2Fimg.png)

Kinetica is a modular Python-based rocket and mission simulation project focused on launch vehicle design, ascent analysis, orbital insertion, and simplified 3D mission planning.

The project started as a 2D ascent simulator and is evolving into a more complete mission analysis tool with:

- multi-stage launch vehicles
- 2D and 3D trajectory simulation
- simplified multi-body dynamics
- launch site modeling
- atmospheric drag
- stage separation
- mission phases such as burn, coast, target orbit, and SOI transitions
- interactive Streamlit dashboard for mission design and visualization

---

## Features

### Launch vehicle modeling
- Configurable multi-stage rockets
- Dry mass, propellant mass, thrust, Isp, and diameter per stage
- Payload mass and drag coefficient support
- Preset rockets for quick testing

### 2D ascent simulation
- Pitch program support
- RK4 integration
- Gravity and aerodynamic drag
- Stage depletion and staging
- Orbital element estimation from ascent result

### 3D mission simulation
- 3D state propagation
- Simplified n-body gravity
- Sphere of influence detection
- Rotating launch bodies
- Launch site initial velocity from planetary rotation
- Basic atmospheric drag model
- Stage-by-stage propellant consumption
- Mission phases:
  - `burn`
  - `coast`
  - `target_orbit`
  - `soi_change`

### Interactive dashboard
- Streamlit-based mission designer
- Rocket stage editor
- Mission phase editor
- Launch site configuration
- Earth-centered, Moon-centered, cislunar, and global trajectory views
- Event visualization
- English / Spanish UI support

---

## Project structure

```text
rocket-sim/
├─ src/
│  └─ Kinetica/
│     ├─ config/
│     │  ├─ rockets.py
│     │  └─ celestial_systems.py
│     ├─ models/
│     │  ├─ rocket.py
│     │  ├─ stage.py
│     │  ├─ planet.py
│     │  ├─ celestial_body.py
│     │  └─ mission.py
│     ├─ simulation/
│     │  ├─ trajectory2d.py
│     │  └─ mission3d.py
│     ├─ ui/
│     │  ├─ cli.py
│     │  ├─ dashboard.py
│     │  └─ i18n.py
│     └─ main.py
└─ README.md
```

---

## Requirements

Recommended Python version

- Python 3.10+

Install dependencies with:

```text
pip install -r requirements.txt
```

Typical dependencies include:

```
numpy
matplotlib
streamlit
plotly
```

If you are using a virtual environment on Windows:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Running the project

### CLI mode

```
python src/Kinetica/main.py --ui cli
```

### Streamlit dashboard

```
streamlit run src/Kinetica/ui/dashboard.py
```

If import resolution fails because of the src/ layout, make sure the project is run from the repository root and that src is available in PYTHONPATH, or keep the existing path bootstrap code inside the app files.

---

##Dashboard overview

The dashboard allows you to:

- select a rocket preset
- edit stage parameters
- configure payload and drag coefficien
- define launch latitude, longitude, altitude, and azimuth
- choose a mission profile or build a custom one
- simulate 3D missions and inspect:
  - altitude
  - velocity
  - mass
  - dominant body
  - Earth-centered trajectories
  - Moon-centered trajectorie
  - Earth-Moon transfer view
  - global 3D trajectory
  - mission events

The interface supports both English and Spanish.

---

## Mission model

Mission planning is phase-based.

Supported phase types:

### burn

A powered phase using thrust and Isp, with configurable direction modes such as:

- prograde
- retrograd
- radial_out
- radial_in
- normal
- antinormal

### coast

A ballistic propagation phase with no thrust.

### target_orbit

A simplified powered phase that attempts to reach a target periapsis and apoapsis around the currently dominant body.

### soi_change

A waiting or propagation phase used to continue the mission until the spacecraft enters the sphere of influence (SOI) of a target body.

---

## Physics model

Kinetica currently uses a simplified but useful physical model:

* RK4 numerical integration
* Newtonian gravity
* simplified multi-body superposition
* SOI-based mission interpretation
* exponential atmosphere model
* drag based on vehicle frontal area and drag coefficient
* planetary rotation for launch-site initial velocity
* real stage propellant depletion and dry-mass separation

This makes the simulator useful for prototyping and educational mission analysis, but it is not yet a high-fidelity astrodynamics tool.

---

## Current limitations

Kinetica is still under active development. Current limitations include:

* no Lambert solver
* no optimized translunar injection guidance
* no high-fidelity patched conics
* no precise gravity assist targeting
* no non-spherical gravity model
* no detailed engine throttling or restart logic
* no advanced ascent guidance or gravity turn autopilot
* atmosphere is simplified
* celestial body orientations are simplified
* no full ECI/ECEF reference frame system yet

---

### Notes

This project is intended for experimentation, learning, and iterative development.

Results should be interpreted as approximate unless validated against a higher-fidelity reference tool.

__License
MIT License
Author__

Created by **Alfonso Fernández**

