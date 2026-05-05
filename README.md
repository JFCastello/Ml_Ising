# Ising Model — Phase Transition Detection with Deep Learning

A reproduction of [Carrasquilla & Melko (2017)](https://arxiv.org/abs/1605.01735):
a neural network learns to identify the ferromagnetic phase transition of the 2D Ising
model using only raw spin configurations, with no prior physics knowledge.

**Live dashboard:** [mlising-elsmpmflvraqc6n2rv9jas.streamlit.app](https://mlising-elsmpmflvraqc6n2rv9jas.streamlit.app/)

---

## What it does

| Step | Component | Description |
|------|-----------|-------------|
| 1 | C++ `./main` | Metropolis-Hastings MC sampler generates spin configurations at temperatures across the phase transition |
| 2 | `python/Distributions.ipynb` | Energy and magnetization distributions are plotted for every temperature |
| 3 | `python/ML_Model.ipynb` | PyTorch neural network trained to classify ordered vs disordered phases |
| 4 | `streamlit_app.py` | Interactive dashboard explaining the physics, method, and results |

Steps 2–3 are executed automatically by `./main`. All output plots are saved to `plots/`.

---

## Quick start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile

```bash
make
```

### 3. Run the full pipeline

```bash
./main --Tc_seed 2.269 --Num_Samples 1000 --L 10 --T_ranges 0.1 4.0 8
```

This single command:
- Generates Monte Carlo samples for 8 temperatures in `[0.1, 4.0]` (a `linspace`)
- Writes data files to `data/`
- Executes both analysis notebooks
- Saves all plots to `plots/`

### 4. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

---

## CLI reference

```
./main --Tc_seed <val> --Num_Samples <val> --L <val> --T_ranges <T_min> <T_max> <T_n>
```

| Flag | Type | Description |
|------|------|-------------|
| `--Tc_seed` | double | Critical temperature estimate used to label configurations (ordered if `T < Tc_seed`) |
| `--Num_Samples` | int | Number of Monte Carlo samples per temperature |
| `--L` | int | Lattice size — system has `L×L` spins |
| `--T_ranges` | double double int | `T_min T_max T_n` — equivalent to `np.linspace(T_min, T_max, T_n)` |

Run `./main --help` for the full help message.

---

## Project structure

```
.
├── main.cpp                     # CLI entry point — pipeline orchestrator
├── src/Samples_Gen.cpp          # Metropolis-Hastings sampler
├── include/
│   ├── Samples_Gen.h            # microstate and Samples class definitions
│   └── Canonical.h              # partition function utilities
├── python/
│   ├── Distributions.ipynb      # energy & magnetization distribution analysis
│   ├── ML_Model.ipynb           # neural network training and Tc estimation
│   └── run_pipeline.py          # notebook executor (called by ./main)
├── streamlit_app.py             # interactive dashboard
├── data/                        # generated spin configuration files (auto-created)
├── plots/                       # all output figures (auto-created)
├── requirements.txt
└── Makefile
```

---

## Physics background

The 2D Ising Hamiltonian:

$$\mathcal{H} = -J \sum_{\langle i,j \rangle} s_i s_j$$

undergoes a second-order phase transition at the exact critical temperature
$T_c = 2J / k_B \ln(1+\sqrt{2}) \approx 2.269$ (Onsager, 1944).

Below $T_c$ spins align spontaneously (ferromagnetic / ordered phase).
Above $T_c$ thermal fluctuations destroy long-range order (paramagnetic / disordered phase).

---

## Tech stack

- **C++17** — Monte Carlo simulation, OpenMP-ready
- **Python 3.9+** — data analysis and ML pipeline
- **PyTorch** — fully-connected NN with Kaiming init, BatchNorm, Adam
- **scikit-learn** — metrics, train/test split
- **seaborn / matplotlib** — visualisation
- **Streamlit** — interactive dashboard

---

## Reference

> Carrasquilla, J., & Melko, R. G. (2017).
> *Machine learning phases of matter.*
> Nature Physics, 13(5), 431–434.
> [arXiv:1605.01735](https://arxiv.org/abs/1605.01735)
