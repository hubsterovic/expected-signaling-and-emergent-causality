from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def simulate():
    N = 10**4
    D_MIN = 2
    D_MAX = 8
    dims = range(D_MIN, D_MAX + 1)
    directions = ["A to B", "B to A"]
    dm_type = "pure"

    all_results = []

    for direction in directions:
        for d in dims:
            _, samples = haar_expected_mc_signaling_X_to_Y(
                N=N,
                d_A=d,
                d_B=d,
                dm_type=dm_type,
                direction=direction,  # type: ignore
            )
            for s in samples:
                all_results.append(
                    {"dimension": d, "signaling": s, "direction": direction}
                )

    df = pd.DataFrame(all_results)
    return {"df": df, "N": N}


def plot(data):
    apply_plot_style()
    df = data["df"]
    N = data["N"]

    palette = {"A to B": "blue", "B to A": "red"}

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="dimension",
        y="signaling",
        hue="direction",
        data=df,
        inner="quart",
        density_norm="area",
        split=True,
        gap=0.1,
        log_scale=(False, True),
        palette=palette,
    )
    plt.xlabel(r"$d_A = d_B$")
    plt.ylabel(r"$\mathcal{S}_{X \rightarrow Y}$")
    plt.title(f"Signaling Distribution for $d_A = d_B$ $(N=10^{int(np.log10(N))})$")
    plt.legend(title="Direction")
    plt.tight_layout()

    plt.savefig(
        f"plots/signaling_violin_N={N}__[dt={datetime.now().strftime('%Y%m%d_%H%M%S')}].png"
    )

    plt.show()


def main():
    data = simulate()
    plot(data)


if __name__ == "__main__":
    main()
