from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def simulate():
    N = 10**3
    d_B = 2
    d_A_MAX = 100

    d_As = sorted(
        set(
            int(round(i)) for i in np.logspace(np.log10(d_B), np.log10(d_A_MAX), num=10)
        )
    )
    directions = ["A to B", "B to A"]
    dm_type = "product"

    results = {}
    for direction in directions:
        means = []
        for d_A in d_As:
            mean_val, _ = haar_expected_mc_signaling_X_to_Y(
                N=N,
                d_A=d_A,
                d_B=d_B,
                dm_type=dm_type,
                direction=direction,  # type: ignore
            )
            means.append(mean_val)
        results[direction] = np.array(means)

    return {"d_As": d_As, "results": results, "N": N}


def plot(data):
    apply_plot_style()
    d_As = data["d_As"]
    results = data["results"]
    N = data["N"]

    log_d_As = np.log10(d_As)
    plt.figure(figsize=(8, 6))

    direction_colors = {"A to B": "blue", "B to A": "red"}
    markers = {"A to B": "o", "B to A": "s"}

    for direction in results:
        means = results[direction]
        color = direction_colors[direction]
        marker = markers[direction]

        plt.scatter(d_As, means, marker=marker, color=color, s=70)

        slope, intercept = np.polyfit(log_d_As, np.log10(means), 1)
        plt.plot(
            d_As,
            10 ** (intercept + slope * log_d_As),
            linestyle="--",
            color=color,
            label=rf"Fit $\langle \mathcal{{S}} \rangle_{{{direction.replace('to', r'\to')}}} \propto d_A^{{{slope:.2f}}}$",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$d_A$")
    plt.ylabel(r"$\langle \mathcal{S} \rangle_{X \rightarrow Y}$")
    plt.title(rf"Expected Signaling Log-Log $(N=10^{int(np.log10(N))})$")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"plots/expected_signaling_N={N}__[dt={datetime.now().strftime("%Y%m%d_%H%M%S")}].png")

    plt.show()


def main():
    data = simulate()
    plot(data)


if __name__ == "__main__":
    main()
