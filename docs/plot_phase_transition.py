from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def Read_Sample(filepath: Path) -> tuple[pd.Series, pd.Series]:
    # Read the last two columns of the file: Energy and Magnetization
    try:
        data = pd.read_csv(filepath, sep=r"\s+", header=None)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find: {filepath}")

    Energy = data.iloc[:, -2]
    Magnetization = data.iloc[:, -1]
    return Energy, Magnetization


def Panel_plot(L: int, Temperatures: np.ndarray) -> None:
    Temperatures = np.round(Temperatures, 3)

    # Data files are expected at <repo_root>/Data/
    data_dir = Path(__file__).resolve().parent.parent / "Data"

    # One column per temperature; two rows: Energy (top) and Magnetization (bottom)
    n = len(Temperatures)
    fig, axes = plt.subplots(nrows=2, ncols=n, sharey="row", figsize=(4 * n, 7))

    n_samples = None
    for i, T in enumerate(Temperatures):
        filepath = data_dir / f"Ising{L}_T{T:.3f}_Samples.txt"
        Energy, Magnetization = Read_Sample(filepath)
        n_samples = len(Energy)

        # Histogram + KDE curve; blue for Energy, red for Magnetization
        sns.histplot(Energy,        kde=True, color="steelblue", ax=axes[0][i])
        sns.histplot(Magnetization, kde=True, color="tomato",    ax=axes[1][i])

        axes[0][i].set_title(f"T = {T}")
        axes[0][i].set_xlabel("")
        axes[1][i].set_xlabel("")

    fig.suptitle(f"Ising Model  —  L = {L}  |  {n_samples} samples", fontsize=16, fontweight="bold")

    # Y-label only on the leftmost subplot (shared axis hides the rest automatically)
    axes[0][0].set_ylabel("Count")
    axes[1][0].set_ylabel("Count")

    # Row subtitles: placed just below each row using the actual axis positions
    plt.tight_layout()
    for row, label in enumerate(["Energy  (J = 1, k_B = 1)", "Magnetization  (J = 1, k_B = 1)"]):
        pos = axes[row][n // 2].get_position()
        fig.text(0.5, pos.y0 - 0.03, label, ha="center", va="top", fontsize=12, style="italic")

    output = Path(__file__).parent / f"Ising{L}_distributions.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ising model distributions")
    parser.add_argument("L", type=int, help="Lattice size (e.g. 10)")
    args = parser.parse_args()

    # --- Configuration: edit temperatures here ---
    Temperatures = np.array([0.1, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0])
    # ---------------------------------------------

    Panel_plot(args.L, Temperatures)


if __name__ == "__main__":
    main()
