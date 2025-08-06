from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import qutip as qt


def simulate():
    N = 10**5
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

    mean_AtoB, dists_AtoB = haar_expected_mc_signaling_X_to_Y(
        N=N, d_A=d_A, d_B=d_B, direction="A to B", dm_type="pure", fixed_coms=coms_S
    )
    mean_BtoA, dists_BtoA = haar_expected_mc_signaling_X_to_Y(
        N=N, d_A=d_A, d_B=d_B, direction="B to A", dm_type="pure", fixed_coms=coms_S
    )

    return {
        "N": N,
        "mean_AtoB": mean_AtoB,
        "dists_AtoB": dists_AtoB,
        "mean_BtoA": mean_BtoA,
        "dists_BtoA": dists_BtoA,
    }


def plot(data):
    apply_plot_style()
    BINS = 50
    N = data["N"]

    plt.figure(figsize=(12, 5))

    for i, (direction, dist, mean, color) in enumerate(
        [
            ("A to B", data["dists_AtoB"], data["mean_AtoB"], "blue"),
            ("B to A", data["dists_BtoA"], data["mean_BtoA"], "red"),
        ]
    ):
        plt.subplot(1, 2, i + 1)
        sns.histplot(
            dist,
            bins=BINS,
            stat="probability",
            element="step",
            color=color,
            log_scale=(True, False),
        )
        c, exp = f"{mean:.2e}".split("e")
        plt.axvline(
            mean,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\langle \mathcal{{S}} \rangle_{{{direction.replace('to', r'\to')}}} = {float(c):.2f} \times 10^{{{int(exp)}}}$",
        )
        plt.xlabel(rf"$\mathcal{{S}}_{{{direction.replace('to', r'\to')}}}$")
        plt.ylabel("Probability")
        plt.title(
            rf"Signaling {direction} for fixed COMS $(N=10^{int(np.log10(N))})$"
        )
        plt.legend()

    plt.savefig(f"plots/signaling_for_semicausal_coms_N={N}__[dt={datetime.now().strftime("%Y%m%d_%H%M%S")}].png")
    plt.tight_layout()
    plt.show()


def main():
    data = simulate()
    plot(data)


if __name__ == "__main__":
    main()
