from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
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
        N=N,
        d_A=d_A,
        d_B=d_B,
        direction="A to B",
        dm_type="pure",
        fixed_coms=coms_S,
    )
    data_BtoA = haar_expected_mc_signaling_X_to_Y(
        N=N,
        d_A=d_A,
        d_B=d_B,
        direction="B to A",
        dm_type="pure",
        fixed_coms=coms_S,
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


def plot(data: dict) -> None:
    apply_plot_style()
    BINS = 25
    N = data["N"]

    plt.figure(figsize=(10, 8))  # smaller width, taller height for 2x2 layout

    # Top row = Trace signaling, Bottom row = H signaling
    # Left col = A→B, Right col = B→A
    plot_specs = [
        ("A to B", "tr_dists_AtoB", "tr_mean_AtoB", "blue", r"\mathrm{Tr}", 1),
        ("B to A", "tr_dists_BtoA", "tr_mean_BtoA", "red", r"\mathrm{Tr}", 2),
        ("A to B", "h_dists_AtoB", "h_mean_AtoB", "blue", r"\widehat{\mathrm{H}}", 3),
        ("B to A", "h_dists_BtoA", "h_mean_BtoA", "red", r"\widehat{\mathrm{H}}", 4),
    ]

    for direction_tex, dist_key, mean_key, color, metric_tex, subplot_idx in plot_specs:
        plt.subplot(2, 2, subplot_idx)

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
        f"plots/tr_and_h_signaling_for_semicausal_COMS_N={N}__[dt={datetime.now().strftime('%Y%m%d_%H%M%S')}].png"
    )
    plt.show()


def main(full_sim: bool = True) -> None:
    N = 10**5 if full_sim else 10**3
    data = simulate(N=N)
    plot(data)


if __name__ == "__main__":
    main()
