# RTC–BTES System Digital Twin with Model Predictive Control (MPC)

## 1. Introduction
This project presents a high-fidelity **digital twin** of a coupled thermo-fluid system composed of:
- a **Road Thermal Collector (RTC)**,
- a **Borehole Thermal Energy Storage (BTES)** system,
- and a **Dry Cooler (DC)**.

The model simulates the transient thermal behavior of the system and optimizes its operation through a **Model Predictive Control (MPC)** strategy.

---

## 2. Methodology

### 2.1 Thermal Model
The RTC is modeled using a **finite difference approach** applied to a multilayer structure:
- asphalt layer,
- pipe-embedded concrete layer,
- underlying thermal storage layers.

Heat transfer mechanisms include:
- conduction between layers,
- convection with ambient air,
- radiative exchange with the sky,
- convective heat exchange with the circulating fluid.

### 2.2 Radiative Model
The sky temperature is computed using the **Brutsaert (1975)** formulation, which accounts for atmospheric humidity.

This approach improves the estimation of longwave radiation losses, especially under high humidity conditions.

### 2.3 BTES Model
The Borehole Thermal Energy Storage is modeled using:
- **g-function methodology**,
- temporal superposition of thermal loads.

### 2.4 Control Strategy
An **MPC (Model Predictive Control)** framework is implemented with:
- finite prediction horizon,
- candidate mode evaluation,
- reward-based optimization.

The controller dynamically selects among:
- `charge` (energy storage),
- `discharge` (energy release),
- `solar_dc` (direct dissipation),
- `idle` (standby).

---

## 3. Input Data

The simulation requires the following input files:
- `Datiy2025.csv` → meteorological data
- `MIFT_ref.xlsx` → BTES g-function data

Weather data can be:
- read from local datasets, or
- retrieved via Open-Meteo API.

---

## 4. Output

The model produces:
- time-resolved thermal and energy variables,
- Excel output files,
- real-time CSV data for dashboard visualization,
- graphical outputs including:
  - system temperature evolution,
  - energy balance,
  - MPC performance analysis,
  - spatial temperature profiles.

---

## 5. Execution

Run the simulation with:

```bash
python RTC_BTES_MPC_v3.py
```

---

## 6. Dependencies

```txt
numpy
pandas
matplotlib
scipy
requests
openpyxl
```

---

## 7. Objectives

The main objectives of this work are:
- to model the thermal dynamics of an RTC–BTES system,
- to evaluate advanced control strategies for seasonal energy management,
- to compare MPC performance against rule-based control approaches.

---

## 8. Context

This code has been developed as part of a **university research project** in the field of:
- thermal energy storage,
- renewable energy systems,
- and intelligent control strategies.

---

## 9. Author
Academic project – Digital Twin for Energy Systems
