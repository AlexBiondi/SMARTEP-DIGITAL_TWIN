# Digital Twin of an RTCâ€“BTES System with Model Predictive Control

## Abstract
This work presents a high-fidelity digital twin of a coupled thermo-fluid energy system 
composed of a Road Thermal Collector (RTC), a Borehole Thermal Energy Storage (BTES) system, 
and a Dry Cooler (DC).  

The model simulates transient heat transfer processes using a finite difference approach and 
integrates a Model Predictive Control (MPC) strategy to optimize system operation.  

The proposed framework enables dynamic energy management, seasonal storage optimization, and 
comparison between predictive and rule-based control strategies.

---

## Keywords
Digital Twin Â· Thermal Energy Storage Â· MPC Â· RTC Â· BTES Â· Renewable Energy Systems

---

## 1. Introduction
The integration of thermal energy storage systems with renewable sources is a key aspect in 
improving energy efficiency and reducing environmental impact.

This project focuses on the development of a digital twin capable of:
- simulating the thermal behavior of a road-integrated solar collector,
- modeling underground thermal storage,
- optimizing system operation through predictive control.

---

## 2. System Description

The system consists of three main subsystems:

- **Road Thermal Collector (RTC)**  
  A multilayer structure with embedded pipes for heat extraction.

- **Borehole Thermal Energy Storage (BTES)**  
  Underground storage modeled using g-function methodology.

- **Dry Cooler (DC)**  
  Used for heat dissipation or auxiliary thermal management.

---

## 3. Methodology

### 3.1 Thermal Modeling
The RTC is modeled using a **finite difference scheme** applied to a multilayer domain.

Heat transfer mechanisms:
- conduction between layers,
- convection with ambient air,
- longwave radiation exchange,
- fluidâ€“structure heat transfer.

### 3.2 Radiative Model
The sky temperature is computed using the **Brutsaert (1975)** formulation:
- includes atmospheric humidity effects,
- improves longwave radiation estimation,
- more accurate under Mediterranean climatic conditions.

### 3.3 BTES Modeling
The BTES is simulated using:
- g-function interpolation,
- temporal superposition of thermal loads.

### 3.4 Control Strategy (MPC)
A **Model Predictive Control (MPC)** algorithm is implemented:
- finite prediction horizon,
- evaluation of candidate operating modes,
- reward-based optimization.

Operating modes:
- `charge` â†’ storage of thermal energy,
- `discharge` â†’ energy release,
- `solar_dc` â†’ direct dissipation,
- `idle` â†’ standby.

---

## 4. Repository Structure

```

project/
├── RTC_BTES_MPC_v3.py          # Main simulation model (MPC-based digital twin)
├── dashboard_dt_v2.py          # Streamlit dashboard for real-time monitoring
├── data/
│   ├── Datiy2025.csv           # Meteorological input data
│   ├── MIFT_ref.xlsx           # BTES g-function dataset
│   └── sensori_live.csv        # Simulated sensor data used to emulate real measurements
├── output/
│   └── RTC_BTES_output_MPC/
│       ├── live_results.csv          # Real-time simulation output (updated hourly)
│       ├── RTC_BTES_MPC_*.xlsx       # Full simulation results (time series data)
│       ├── MPC_v1_overview_*.png     # System temperature and energy overview plots
│       ├── MPC_v1_strati_*.png       # Temperature profiles of RTC layers
│       ├── MPC_v1_energia_*.png      # Cumulative energy balance plots
│       ├── MPC_v1_profilo_*.png      # Spatial fluid temperature profiles
│       └── MPC_v1_convergenza_*.png  # MPC convergence and performance analysis
└── README.md                         # Project documentation                                 
```

---

## 5. Input Data

Required files:
- `Datiy2025.csv` â†’ meteorological data
- `MIFT_ref.xlsx` â†’ BTES thermal response

Optional:
- Open-Meteo API for forecast data
- sensor data (`sensori_live.csv`) for validation

---

## 6. Output

The simulation produces:
- time-resolved temperature fields,
- energy fluxes and balances,
- Excel result files,
- real-time CSV output for dashboard,
- graphical analysis:
  - temperature evolution,
  - energy balance,
  - MPC performance,
  - spatial fluid profiles.

---

## 7. Dashboard

The project includes an interactive dashboard for real-time monitoring.

### Execution
```bash
streamlit run dashboard_dt_v2.py
```

Access via:
```
http://localhost:8501
```

### Features
- real-time system monitoring,
- energy and temperature visualization,
- MPC vs rule-based comparison,
- optional forecast integration,
- model validation with sensor data.

---

## 8. Installation

Install required dependencies:

```bash
pip install numpy pandas matplotlib scipy requests openpyxl streamlit plotly
```

---

## 9. Execution

Run the simulation:

```bash
python RTC_BTES_MPC_v3.py
```

---

## 10. Results and Objectives

The main objectives of this project are:
- to analyze the thermal dynamics of RTCâ€“BTES systems,
- to evaluate the effectiveness of MPC strategies,
- to optimize seasonal energy storage,
- to compare predictive control with rule-based approaches.

---

## 11. Conclusion

The developed digital twin demonstrates the potential of integrating:
- advanced thermal modeling,
- real-time monitoring,
- predictive control strategies,

for improving the performance of energy systems.

---

## 12. Author
Prof. Marco Beccali - University of Palermo "Digital Twin for Thermal Energy Systems"
