"""
Interactive dashboard — 2D Ising Model Phase Transition via Deep Learning.
Run with: streamlit run streamlit_app.py
All numerical values are read from the last ./main run; no hardcoded parameters.
"""
import streamlit as st
from pathlib import Path
import json
import re

st.set_page_config(
    page_title="Ising Model — Phase Transition with Deep Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Typography — fonts and borders only, no color overrides (dark-mode safe) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Source+Code+Pro:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'EB Garamond', Georgia, serif;
}
.stMarkdown p, .stMarkdown li {
    font-size: 1.08rem;
    line-height: 1.95;
}
h1 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 2.05rem !important;
    font-weight: 600 !important;
    border-bottom: 1.5px solid currentColor;
    padding-bottom: 0.25em;
    margin-bottom: 0.7em !important;
    letter-spacing: 0.01em;
}
h2 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    margin-top: 1.8em !important;
    letter-spacing: 0.01em;
}
h3 {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    font-style: italic;
    margin-top: 1.3em !important;
}
code, pre {
    font-family: 'Source Code Pro', 'Courier New', monospace !important;
    font-size: 0.87rem !important;
}
[data-testid="metric-container"] {
    border-left: 3px solid currentColor;
    border-radius: 2px;
    padding: 0.7rem 1rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'EB Garamond', Georgia, serif !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    font-family: 'EB Garamond', Georgia, serif !important;
}
table {
    font-family: 'EB Garamond', Georgia, serif;
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 1.02rem;
}
thead th {
    padding: 0.45em 1.1em;
    font-weight: 500;
    text-align: left;
    letter-spacing: 0.04em;
    font-size: 0.88rem;
    text-transform: uppercase;
}
td {
    padding: 0.38em 1.1em;
    border-bottom: 1px solid rgba(128,128,128,0.3);
}
tr:last-child td { border-bottom: none; }
hr {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.4);
    margin: 1.8em 0;
}
blockquote {
    border-left: 3px solid rgba(128,128,128,0.6);
    padding-left: 1em;
    font-style: italic;
    margin: 1em 0;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
PLOTS = Path("plots")
DATA  = Path("data")
DOCS  = Path("docs")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_run_info():
    params_file = DATA / "run_params.json"
    if params_file.exists():
        with open(params_file) as f:
            p = json.load(f)
        L   = int(p["L"])
        N   = L * L
        temps = [round(t, 10) for t in p["temperatures"]]
        T_n = int(p["T_n"])
        num_samples = int(p["Num_Samples"])
        tc_seed = float(p["Tc_seed"])
        arch  = [N, int(.8*N), int(.6*N), int(.5*N), 2]
        total = num_samples * T_n
        return {
            "L": L, "N": N, "temps": temps,
            "T_min": float(p["T_min"]), "T_max": float(p["T_max"]), "T_n": T_n,
            "num_samples": num_samples, "total_samples": total,
            "train_samples": int(.75*total), "test_samples": total - int(.75*total),
            "tc_seed": tc_seed, "arch": arch,
        }
    # fallback — infer from filenames
    sample_files = sorted(DATA.glob("Ising*_Samples.txt"))
    if not sample_files:
        return None
    pat = re.compile(r'Ising(\d+)_T(\d+\.\d+)_Samples\.txt')
    L_vals, temps = set(), []
    for f in sample_files:
        m = pat.match(f.name)
        if m:
            L_vals.add(int(m.group(1))); temps.append(float(m.group(2)))
    if not L_vals:
        return None
    L = sorted(L_vals)[-1]; N = L*L; temps = sorted(temps); T_n = len(temps)
    first = DATA / f"Ising{L}_T{temps[0]:.3f}_Samples.txt"
    num_samples = sum(1 for _ in open(first)) if first.exists() else "?"
    tc_seed = None
    for t in temps:
        ds = DATA / f"Dataset{t:.3f}.txt"
        if ds.exists():
            with open(ds) as fh:
                if int(fh.readline().strip().split()[-1]) == 1:
                    tc_seed = t; break
    arch  = [N, int(.8*N), int(.6*N), int(.5*N), 2]
    total = (num_samples if isinstance(num_samples, int) else 0) * T_n
    return {
        "L": L, "N": N, "temps": temps,
        "T_min": temps[0], "T_max": temps[-1], "T_n": T_n,
        "num_samples": num_samples, "total_samples": total,
        "train_samples": int(.75*total), "test_samples": total - int(.75*total),
        "tc_seed": tc_seed, "arch": arch,
    }

@st.cache_data
def load_summary():
    p = PLOTS / "run_summary.json"
    return json.load(open(p)) if p.exists() else None

def fmt(v, d=3):
    if v is None: return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)

def show_plot(pattern: str, caption: str = ""):
    p = PLOTS / pattern
    if not p.exists():
        hits = sorted(PLOTS.glob(pattern))
        p = hits[0] if hits else None
    if p and p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Plot not yet available (`{pattern}`). Run `./main …` to generate it.")

# ── Load run data ─────────────────────────────────────────────────────────────
info    = load_run_info()
summary = load_summary()

def get(key, fallback="—"):
    if summary and key in summary: return summary[key]
    if info    and key in info:    return info[key]
    return fallback

L            = get("L",            "?")
N            = get("N",            "?")
T_n          = get("T_n",          "?")
T_min        = get("T_min",        "?")
T_max        = get("T_max",        "?")
num_samples  = summary.get("num_samples_per_T") if summary else get("num_samples", "?")
total_samp   = get("total_samples", "?")
train_samp   = get("train_samples", "?")
test_samp    = get("test_samples",  "?")
tc_seed      = get("tc_seed",  None)
arch         = get("arch",     None)
epochs       = get("epochs",   100)
final_val    = get("final_val_acc",   None)
final_train  = get("final_train_acc", None)
final_f1     = get("final_val_f1",    None)
estimated_tc = get("estimated_tc",    None)
temps_list   = get("temps", [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
sections = [
    "Project Overview",
    "The Ising Model",
    "Monte Carlo Simulation",
    "Energy & Magnetization Distributions",
    "Neural Network Architecture",
    "Training & Evaluation",
    "Critical Temperature Estimation",
]
st.sidebar.title("Navigation")
section = st.sidebar.radio("", sections)
st.sidebar.divider()
st.sidebar.markdown("**Last Run**")
if info:
    st.sidebar.markdown(
        f"*L* = {L}, *N* = {N}  \n"
        f"*T* ∈ [{fmt(T_min)}, {fmt(T_max)}], {T_n} pts  \n"
        f"Samples / *T* = {num_samples:,}\n" if isinstance(num_samples, int)
        else f"*L* = {L}, *N* = {N}  \n*T* ∈ [{fmt(T_min)}, {fmt(T_max)}], {T_n} pts  \n"
    )
    if summary:
        st.sidebar.markdown(
            f"Val. accuracy = {final_val:.2%}  \n"
            f"Est. *T*ᶜ = {fmt(estimated_tc)}" if final_val else ""
        )
else:
    st.sidebar.info("No data found. Run `./main …` first.")
st.sidebar.divider()
if st.sidebar.button("↺  Refresh"):
    st.cache_data.clear(); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
if section == "Project Overview":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Ising Model — Phase Transition Detection via Deep Learning")
    st.markdown("""
This project is a computational reproduction of the seminal work of
[Carrasquilla & Melko (2017)](https://www.nature.com/articles/nphys4035),
which demonstrated that a supervised neural network, trained solely on raw spin
configurations of the two-dimensional Ising model, learns to identify the
ferromagnetic phase transition without any prior knowledge of the underlying physics.
""")

    if info:
        st.markdown("#### Last Run — Parameters")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lattice size  L",  L)
        c2.metric("Spins  N = L²",    N)
        c3.metric("Temperatures",     T_n)
        c4.metric("Samples / T", f"{num_samples:,}" if isinstance(num_samples, int) else num_samples)
        c1.metric("T_min", fmt(T_min))
        c2.metric("T_max", fmt(T_max))
        c3.metric("Tc seed", fmt(tc_seed))
        c4.metric("Total samples", f"{total_samp:,}" if isinstance(total_samp, int) else total_samp)
    else:
        st.info("No run data found. Generate data with `./main …` first.")

    if summary:
        st.markdown("#### Last Run — Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train accuracy", f"{final_train:.2%}" if final_train else "—")
        c2.metric("Val. accuracy",  f"{final_val:.2%}"  if final_val  else "—")
        c3.metric("Val. F₁ score",  f"{final_f1:.4f}"   if final_f1   else "—")
        c4.metric("Estimated  Tᶜ",  fmt(estimated_tc))

    st.divider()
    st.markdown("#### Pipeline")
    ns  = f"{num_samples:,}" if isinstance(num_samples, int) else str(num_samples)
    tot = f"{total_samp:,}"  if isinstance(total_samp,  int) else str(total_samp)
    tr  = f"{train_samp:,}"  if isinstance(train_samp,  int) else str(train_samp)
    te  = f"{test_samp:,}"   if isinstance(test_samp,   int) else str(test_samp)
    st.markdown(f"""
| Step | Component | Description |
|---|---|---|
| 1 | C++ `./main` | Metropolis-Hastings MC generates {ns} configurations per temperature across {T_n} temperatures in $[{fmt(T_min)},\\,{fmt(T_max)}]$ on an $L={L}$ lattice |
| 2 | `Distributions.ipynb` | Energy and magnetisation distributions are plotted for every temperature |
| 3 | `ML_Model.ipynb` | PyTorch feed-forward network trained on {tot} labelled spin configurations ({tr} train / {te} test) |
| 4 | `ML_Model.ipynb` | Per-temperature accuracy curve yields $\\hat{{T}}_c \\approx {fmt(estimated_tc)}$ (exact: $2.269$) |
""")
    st.code(
        f"./main  --Tc_seed {fmt(tc_seed) if tc_seed else '2.269'}"
        f"  --Num_Samples {num_samples if isinstance(num_samples,int) else 1000}"
        f"  --L {L}"
        f"  --T_ranges {fmt(T_min) if T_min != '?' else '0.1'}"
        f" {fmt(T_max) if T_max != '?' else '4.0'}"
        f" {T_n if T_n != '?' else '8'}",
        language="bash",
    )

# ══════════════════════════════════════════════════════════════════════════════
elif section == "The Ising Model":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("The Two-Dimensional Ising Model")

    img_path = DOCS / "spin_lattice.png"
    if img_path.exists():
        st.image(str(img_path),
                 caption="Figure 1. A spin configuration on a 2D Ising lattice. "
                         "Each site carries a spin variable $s_i = \\pm 1$.",
                 use_container_width=True)
        st.markdown(
            "*Source: [Towards Data Science — Unsupervised Learning Meets Emergent Pattern]"
            "(https://towardsdatascience.com/unsupervised-learning-meets-emergent-pattern-ae5948a714f1/)*"
        )
    else:
        st.info("Save the lattice image to `docs/spin_lattice.png` to display it here.")

    st.markdown(f"""
### The Spin Lattice

Consider a regular $L \\times L$ square lattice — for this simulation $L = {L}$,
giving $N = {N}$ sites — where each site $i$ is occupied by a classical spin
variable $s_i \\in \\{{-1, +1\\}}$. A **microstate** is a complete assignment of
spin values to all $N$ sites. The microstate space contains $2^{{{N}}}$
configurations, an astronomically large set that rules out exact enumeration and
motivates the Monte Carlo approach.

### The Hamiltonian

The energy of a microstate is given by the **Ising Hamiltonian**:

$$
\\mathcal{{H}}\\bigl(\\{{s_i\\}}\\bigr) = -J \\sum_{{\\langle i,j \\rangle}} s_i \\, s_j,
$$

where $J > 0$ is the ferromagnetic exchange coupling (set to $J = 1$ throughout),
and the sum runs over all nearest-neighbour pairs $\\langle i,j \\rangle$
(left, right, up, down) with **periodic boundary conditions** — the lattice wraps
around like a torus, eliminating surface effects. Aligned neighbours ($s_i s_j = +1$)
lower the energy by $J$; anti-aligned neighbours raise it by $J$. The system
therefore tends to minimise energy by aligning spins, competing against the
entropic tendency of temperature to randomise them.

### Order Parameter and Phase Transition

The **total magnetisation** $M = \\sum_{{i=1}}^N s_i$ serves as the order parameter.
In the low-temperature **ferromagnetic phase**, exchange interactions dominate
thermal fluctuations and the spins spontaneously align: $|M| \\approx N$.
In the high-temperature **paramagnetic phase**, thermal energy overcomes
the exchange coupling and spins orient independently: $|M| \\approx 0$.

The competition between these two tendencies produces a sharp
**second-order phase transition** at the critical temperature

$$
T_c = \\frac{{2J}}{{k_B \\ln(1+\\sqrt{{2}})}} \\approx 2.269
\\quad (J = k_B = 1),
$$

an exact result first derived by Lars Onsager (1944). At $T_c$ the correlation
length $\\xi$ — the characteristic distance over which spins influence each other —
diverges, and the system exhibits **scale-invariant fluctuations**: magnetic domains
of all sizes coexist simultaneously, a hallmark of criticality.

### Relevance to Machine Learning

Each spin configuration is a binary image of {N} pixels with values $\\pm 1$.
When a neural network is shown only these raw images — labelled ordered or
disordered according to a seed temperature — it must implicitly discover the
concept of long-range spin correlation to perform well, with no physics encoded
in its architecture or loss function. The fact that it
succeeds{"" if not final_val else f" (reaching {final_val:.1%} test accuracy)"}
is a direct demonstration that deep networks can extract abstract physical
structure from microscopic data.
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "Monte Carlo Simulation":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Monte Carlo Simulation — Metropolis-Hastings Algorithm")

    burn_n = f"{1000*N:,}" if isinstance(N, int) else "1000 × N"
    thin_n = f"{100*N:,}"  if isinstance(N, int) else "100 × N"

    st.markdown(f"""
### Why Monte Carlo?

With $L = {L}$ there are $2^{{{N}}}$ possible microstates — direct enumeration is
impossible. The **Metropolis-Hastings algorithm** builds a Markov chain that
samples configurations from the Boltzmann distribution at temperature $T$ without
ever computing the partition function.

### The Algorithm

Starting from a random configuration, at each step a single spin $s_i$ is chosen
at random and a flip is proposed. The flip is accepted if it lowers the energy;
if it raises the energy by $\\Delta E > 0$ it is accepted with probability
$e^{{-\\Delta E / T}}$. This acceptance rule ensures the chain converges to the
correct thermal equilibrium distribution.

```cpp
if (deltaE <= 0 || u(gen) < std::exp(-beta * deltaE))
    current_state.flip(index);
```

### Sampling Protocol

**Burn-in ({burn_n} steps).** The chain runs without recording data, allowing the
system to reach thermal equilibrium from its random starting state.

**Production (every {thin_n} steps).** One configuration is saved every
$100 \\times N$ steps to reduce autocorrelation between successive samples.

### Labels

Each configuration is labelled for the classifier: **0 (ordered)** if
$T < T_{{c,\\text{{seed}}}} = {fmt(tc_seed)}$, **1 (disordered)** otherwise.

### Run Parameters

| Parameter | Value |
|---|---|
| Lattice size $L$ | {L} ({N} spins) |
| Temperature grid | linspace({fmt(T_min)}, {fmt(T_max)}, {T_n}) |
| Samples per temperature | {f"{num_samples:,}" if isinstance(num_samples,int) else num_samples} |
| $T_{{c,\\text{{seed}}}}$ | {fmt(tc_seed)} |
| Burn-in | {burn_n} steps |
| Thinning | {thin_n} steps |
| Boundary conditions | Periodic (toroidal) |
| Random engine | `std::mt19937` |
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "Energy & Magnetization Distributions":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Energy and Magnetisation Distributions")

    t_range = (f"$T \\in [{fmt(T_min)},\\,{fmt(T_max)}]$ ({T_n} temperatures)"
               if temps_list else "the simulated temperatures")

    st.markdown(f"""
For each temperature in {t_range} we record the energy $E$ and magnetisation $M$
of every sampled configuration. Their distributions reveal the statistical character
of each phase and how it changes across the transition.
""")

    st.markdown("### Energy Distributions")
    st.markdown(f"""
At low temperatures the energy distribution is narrow and centred at strongly
negative values — spins are mostly aligned. As the temperature rises toward
$T_c \\approx 2.269$ the distribution broadens, reaching its maximum variance at
criticality. At high temperatures configurations are essentially random and the
distribution is broad and centred near zero.
""")
    show_plot("energy_distributions_L*.png",
              f"Figure 2. Empirical energy distributions for $L = {L}$, "
              f"$T \\in [{fmt(T_min)}, {fmt(T_max)}]$.")

    st.markdown("### Magnetisation Distributions")
    st.markdown(f"""
The magnetisation is the clearest fingerprint of the phase transition. In the
**ordered phase** $P(M)$ is bimodal with peaks near $\\pm {N}$, reflecting
spontaneous symmetry breaking. Near $T_c$ the bimodal structure flattens as the
system fluctuates between the two ordered states. In the **disordered phase**
$P(M)$ collapses to a single peak at $M = 0$.
""")
    show_plot("magnetization_distributions_L*.png",
              f"Figure 3. Empirical magnetisation distributions for $L = {L}$, "
              f"$T \\in [{fmt(T_min)}, {fmt(T_max)}]$.")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "Neural Network Architecture":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Neural Network Architecture")

    arch_fmt = " → ".join(str(a) for a in arch) if arch else "N → … → 2"

    st.markdown(f"""
### Input

Each spin configuration is flattened to a vector of {N} values ($\\pm 1$), which
is fed directly to the network. No feature engineering is applied.

### Architecture

A fully-connected feed-forward network: **{arch_fmt}**.
Hidden layer widths decrease as fixed fractions of the input size $N = {N}$
(80 %, 60 %, 50 %), forcing the network to build progressively more compressed
representations.

### Design Choices

- **Kaiming initialisation** — weights scaled to keep activation variance stable
  through ReLU layers, preventing vanishing or exploding gradients.
- **Batch normalisation** after each hidden layer — stabilises training and acts
  as a regulariser.
- **ReLU activations** — computationally cheap, no saturation problem.
- **Cross-entropy loss** with **Adam optimiser** ($\\eta = 10^{{-4}}$,
  weight decay $10^{{-4}}$).

### Dataset

| | Fraction | Configurations |
|---|---|---|
| Training set | 75 % | {f"{train_samp:,}" if isinstance(train_samp,int) else train_samp} |
| Test set | 25 % | {f"{test_samp:,}" if isinstance(test_samp,int) else test_samp} |
| **Total** | — | {f"**{total_samp:,}**" if isinstance(total_samp,int) else f"**{total_samp}**"} |

{T_n} temperatures × {f"{num_samples:,}" if isinstance(num_samples,int) else num_samples} samples each,
split with stratification over temperature labels.
""")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "Training & Evaluation":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Training and Evaluation")

    ep = str(epochs) if epochs else "100"

    st.markdown("### Cross-Entropy Loss")
    st.markdown(f"""
Loss is tracked over {ep} epochs on both training and test sets. Parallel
decrease in both curves confirms the network is generalising, not memorising.
""")
    show_plot("training_cross_entropy.png",
              f"Figure 4. Cross-entropy loss over {ep} epochs.")

    st.markdown("### Classification Accuracy")
    final_str = (f"plateauing near **{final_val:.1%}**" if final_val
                 else "plateauing at high accuracy")
    st.markdown(f"""
Accuracy rises quickly in early epochs as the network picks up the dominant
spatial patterns separating the two phases, then {final_str}.
""")
    show_plot("training_accuracy.png",
              f"Figure 5. Classification accuracy over {ep} epochs.")

    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Final train accuracy", f"{final_train:.2%}" if final_train else "—")
        c2.metric("Final val accuracy",   f"{final_val:.2%}"  if final_val  else "—")
        c3.metric("Final val F₁",         f"{final_f1:.4f}"   if final_f1   else "—")

    st.markdown("### Precision, Recall, and F₁ Score")
    st.markdown("""
Macro-averaged precision, recall, and F₁ score on the validation set. For this
balanced binary problem all three metrics track closely together.
""")
    show_plot("validation_scores.png",
              f"Figure 6. Validation precision, recall, and F₁ over {ep} epochs.")

    st.markdown("### Confusion Matrix")
    test_str = (f"{test_samp:,} held-out configurations"
                if isinstance(test_samp, int) else "the held-out test set")
    tc_str   = f"$T < {fmt(tc_seed)}$" if tc_seed else "$T < T_{c,\\text{seed}}$"
    st.markdown(f"""
Evaluated on {test_str}. Diagonal entries close to 1 show the model makes very
few errors in both the ordered ({tc_str}) and disordered classes.
""")
    show_plot("confusion_matrix.png",
              "Figure 7. Row-normalised confusion matrix on the test set.")

# ══════════════════════════════════════════════════════════════════════════════
elif section == "Critical Temperature Estimation":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("Critical Temperature Estimation")

    tc_est_str = fmt(estimated_tc) if estimated_tc else r"\hat{T}_c"
    grid_res   = fmt((T_max - T_min) / (T_n - 1)) if (isinstance(T_min, float)
                 and isinstance(T_max, float) and isinstance(T_n, int)
                 and T_n > 1) else "ΔT"

    st.markdown(f"""
### Method

After training, the classifier is evaluated at each temperature independently.
Far from the transition the model is confident and accurate. Near $T_c$, however,
spin configurations from the two phases become statistically similar — even a
well-trained network cannot reliably tell them apart — so the per-temperature
accuracy drops to a minimum. That minimum is taken as the estimated critical
temperature:

$$
\\hat{{T}}_c = \\underset{{T}}{{\\arg\\min}}\\; \\mathrm{{Acc}}(T) \\approx {tc_est_str}.
$$

The exact value from Onsager's solution is $T_c = 2.269$.
""")

    show_plot("accuracy_vs_temperature.png",
              f"Figure 8. Per-temperature accuracy for $L = {L}$. "
              f"Blue dashed: estimated $\\hat{{T}}_c \\approx {tc_est_str}$. "
              f"Red dashed: seed $T_{{c,\\text{{seed}}}} = {fmt(tc_seed)}$. "
              "Green dotted: exact $T_c = 2.269$.")

    st.markdown(f"""
### Effect of the Seed

The network learns to separate phases relative to the seed temperature, not the
true $T_c$. When the seed is close to $2.269$ the labels are physically correct
and $\\hat{{T}}_c$ aligns well with the exact value. When the seed is far off, a
band of configurations near the true transition is mislabelled, the decision
boundary shifts toward the seed, and $\\hat{{T}}_c$ is biased accordingly.
""")

    if estimated_tc is not None and tc_seed is not None:
        seed_bias = abs(tc_seed - 2.269)
        est_err   = abs(estimated_tc - 2.269)
        st.markdown(f"""
> **This run:** seed $= {fmt(tc_seed)}$ (bias $= {seed_bias:.3f}$),
> estimated $\\hat{{T}}_c = {fmt(estimated_tc)}$ (error $= {est_err:.3f}$).
""")

    st.markdown(f"""
### Iterative Refinement

Because $\\hat{{T}}_c$ is systematically pulled toward a better value than the
seed, passing it as the new `--Tc_seed` on the next run improves the labels and
sharpens the accuracy minimum. Two or three iterations typically converge to
within the grid resolution $\\Delta T = {grid_res}$.

```bash
# Run 1 — initial guess
./main --Tc_seed 2.269 ...
# Run 2 — use the estimated Tc from plots/run_summary.json
./main --Tc_seed {tc_est_str} ...
```

---

### Reference

> Carrasquilla, J. & Melko, R. G. (2017).
> Machine learning phases of matter.
> *Nature Physics*, **13**, 431–434.
> [arXiv:1605.01735](https://arxiv.org/abs/1605.01735)
""")
