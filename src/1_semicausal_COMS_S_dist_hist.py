from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import qutip as qt


def simulate(N: int) -> dict:
    d_A = d_B = 2

    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    plus = (zero + one).unit()
    minus = (zero - one).unit()

    coms_S = [
        qt.tensor(zero, zero),
        qt.tensor(zero, one),
        qt.tensor(one, plus),
        qt.tensor(one, minus),
    ]

    data_AtoB = haar_expected_mc_signaling_X_to_Y(
        N=N, d_A=d_A, d_B=d_B, direction="A to B", dm_type="pure", fixed_coms=coms_S
    )
    data_BtoA = haar_expected_mc_signaling_X_to_Y(
        N=N, d_A=d_A, d_B=d_B, direction="B to A", dm_type="pure", fixed_coms=coms_S
    )

    return {
        "N": N,
        # Trace
        "tr_mean_AtoB": np.mean(data_AtoB["tr_dists"]),
        "tr_dists_AtoB": data_AtoB["tr_dists"],
        "tr_mean_BtoA": np.mean(data_BtoA["tr_dists"]),
        "tr_dists_BtoA": data_BtoA["tr_dists"],
        # H
        "h_mean_AtoB": np.mean(data_AtoB["h_dists"]),
        "h_dists_AtoB": data_AtoB["h_dists"],
        "h_mean_BtoA": np.mean(data_BtoA["h_dists"]),
        "h_dists_BtoA": data_BtoA["h_dists"],
    }


def plot(data: dict, metric: Literal["Tr", "H"]) -> None:
    apply_plot_style()
    BINS = 25
    N = data["N"]

    if metric == "Tr":
        metric_tex = r"\mathrm{Tr}"
        specs = [
            ("A to B", "tr_dists_AtoB", "tr_mean_AtoB", "blue", 1),
            ("B to A", "tr_dists_BtoA", "tr_mean_BtoA", "red", 2),
        ]
    elif metric == "H":
        metric_tex = r"\widehat{\mathrm{H}}"
        specs = [
            ("A to B", "h_dists_AtoB", "h_mean_AtoB", "blue", 1),
            ("B to A", "h_dists_BtoA", "h_mean_BtoA", "red", 2),
        ]
    else:
        raise ValueError("metric must be 'Tr' or 'H'")

    plt.figure(figsize=(10, 4))  # 1 row, 2 columns

    for direction_tex, dist_key, mean_key, color, subplot_idx in specs:
        plt.subplot(1, 2, subplot_idx)

        dist = data[dist_key]
        mean_val = data[mean_key]

        sns.histplot(
            dist,
            bins=BINS,
            stat="probability",
            element="step",
            color=color,
            log_scale=(True, False),
        )

        # format mean in scientific notation
        c, exp = f"{mean_val:.2e}".split("e")
        plt.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\langle \mathcal{{S}}^{{{metric_tex}}} \rangle_{{{direction_tex.replace('to', r'\to')}}} = {float(c):.2f} \times 10^{{{int(exp)}}}$",
        )

        plt.xlabel(
            rf"$\mathcal{{S}}^{{{metric_tex}}}_{{{direction_tex.replace('to', r'\to')}}}$"
        )
        plt.ylabel("Probability")
        plt.title(
            rf"${metric_tex}$-Signaling {direction_tex} $(N=10^{int(np.log10(N))})$"
        )
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"plots/{metric}_signaling_for_semicausal_COMS_N={N}__[dt={datetime.now().strftime('%Y%m%d_%H%M%S')}].png"
    )
    plt.show()


def main(full_sim: bool = False) -> None:
    N = 10**5 if full_sim else 10**3
    data = simulate(N=N)  # always has both metrics
    plot(data, "Tr")
    plot(data, "H")


if __name__ == "__main__":
    main()
