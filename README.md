# Smart Grid Bidirectional Power Flow
### Probabilistic Analysis · ML Forecasting · RL Control · Economic Dispatch

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![IEEE 33-Bus](https://img.shields.io/badge/Test%20System-IEEE%2033--Bus-orange)](https://ieeexplore.ieee.org)
[![Data](https://img.shields.io/badge/Data-OPSD%20Germany%202017--2018-green)](https://data.open-power-system-data.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What This Project Does

When solar panels and wind turbines generate more power than local demand, electricity flows **backward** — from homes toward the substation. This project quantifies that risk, predicts when it will happen, and automatically controls it.

**Five integrated modules, one executable pipeline:**

```
Real OPSD Data  →  Probabilistic Forecast  →  Monte Carlo Power Flow
                                                        ↓
                Economic Dispatch  ←  RL Voltage Control  ←  Bidirectional Flow Risk
```

---

## Key Results

| Module | Result |
|--------|--------|
| **Bidirectional flow** | 23 of 32 branches show reverse flow under high DER penetration |
| **ML surrogate** | Random Forest: 0.00033 pu RMSE (matches published deep NN benchmarks) |
| **RL control** | Overvoltage rate: **13.1% → 1.7%** with only 3.1% renewable curtailment |
| **Solar forecasting** | GBM: **85.1% skill** improvement over naive persistence (real OPSD data) |
| **Economic dispatch** | LP optimizer: 70%+ RE share, ramp/reserve/storage constraints |

---

## Architecture

```
sgbpf/
├── src/
│   ├── network.py               # IEEE 33-bus data + BFS power flow solver
│   ├── der_models.py            # Weibull wind / Beta-irradiance PV models
│   ├── monte_carlo.py           # LHS Monte Carlo probabilistic power flow
│   ├── advanced_forecasting.py  # Quantile regression forecaster (4 models)
│   ├── surrogate.py             # ML surrogate power-flow model
│   ├── rl_control.py            # Q-learning DER curtailment agent
│   ├── economic_dispatch.py     # Day-ahead LP economic dispatch
│   ├── data_loader.py           # OPSD real data loader + literature baselines
│   ├── pipeline.py              # End-to-end integration
│   └── dashboard.py             # All figure generation
├── main.py                      # Run everything
└── requirements.txt
```

---

## Methodology

### 1. Power Flow Engine
Backward/Forward Sweep (BFS) solver on the **IEEE 33-bus radial distribution test system** (Baran & Wu, 1989).
Validated: V_min = 0.9131 pu at bus 18, losses = 202.68 kW ✓

### 2. Probabilistic Power Flow
- 3000 Monte Carlo scenarios with Latin Hypercube Sampling
- Wind: Weibull(k=2.0, c=8.0) → 3-segment turbine curve
- Solar: Beta(α=2.5, β=1.5) → temperature-corrected PV model
- DER: 2000 kW wind at bus 18, 1800 kW PV at bus 33

### 3. Probabilistic Forecasting
- **Real data**: OPSD Germany 2017–2018 (load, wind, solar)
- Features: Fourier harmonics, lag-24h, rolling statistics
- Quantile regression → 80% prediction intervals
- Gaussian copula for correlated wind-solar scenario generation

### 4. ML Surrogate Model

| Model | RMSE (pu) | Speed (μs/sample) |
|-------|-----------|-------------------|
| Linear Regression | 0.00464 | 0.75 |
| **Random Forest** | **0.00033** | 57.7 |
| MLP (neural net) | 0.00157 | 2.35 |
| Full BFS solver | — | 1000 |

### 5. RL Curtailment Control
- Tabular Q-learning, 64 states (8×8 wind × PV)
- Actions: curtailment fraction {0%, 15%, 30%, 45%, 60%, 80%}
- 20,000 training episodes

### 6. Economic Dispatch
- Linear program (HiGHS solver), 192 variables (8 types × 24 hours)
- Ramp-rate limits, 10% spinning reserve, battery SoC dynamics

---

## Quick Start

```bash
git clone https://github.com/saba-aslani/smart-grid-bidirectional-power-flow.git
cd smart-grid-bidirectional-power-flow
pip install -r requirements.txt

# Optional: add real OPSD data
# Download time_series_60min_singleindex.csv from:
# https://data.open-power-system-data.org/time_series/2019-06-05/
# Place in: data/

python main.py
# Results saved to results/ (9 figures)
```

---

## Requirements

```
numpy · scipy · pandas · scikit-learn · matplotlib · networkx · seaborn
```

No GPU required. Python 3.9–3.13.

---

## Literature Comparison

| Method | Voltage RMSE (pu) | Reference |
|--------|-------------------|-----------|
| Point estimate | 0.0021 | Su, 2005 |
| Saddle-point approx. | 0.0012 | Mohammadi et al., 2018 |
| Deep NN surrogate | 0.0004 | Yang et al., 2020 |
| **This work — RF** | **0.00033** | — |

---

## Author

**Saba Aslani** — Independent Researcher, Vancouver, BC, Canada

Electrical Engineering · Data Engineering · Smart Grid Systems

---

## License

MIT — free to use with attribution.
