from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def simulate(N: int, dims: list[int], dm_type: str = "pure") -> dict:
    directions = ["A to B", "B to A"]
    rows = []

    for direction in directions:
        for d in dims:
            # returns (meta, SignalingData)
            sd = haar_expected_mc_signaling_X_to_Y(
                N=N,
                d_A=d,
                d_B=d,  # symmetric case: d_A = d_B = d
                direction=direction,  # type: ignore
            )

            # sd["tr_dists"], sd["h_dists"] are arrays of samples
            for s in np.asarray(sd["tr_dists"]):
                rows.append(
                    {
                        "dimension": d,
                        "signaling": float(s),
                        "direction": direction,
                        "metric": "Tr",
                    }
                )
            for s in np.asarray(sd["h_dists"]):
                rows.append(
                    {
                        "dimension": d,
                        "signaling": float(s),
                        "direction": direction,
                        "metric": "H",
                    }
                )

    df = pd.DataFrame.from_records(rows)
    return {"df": df, "N": N, "dims": dims, "dm_type": dm_type}


def _title_N(N: int) -> str:
    # pretty title like N=10^3 if N is a power of 10
    k = int(np.log10(N))
    return rf"$N=10^{{{k}}}$" if 10**k == N else f"$N={N}$"


def plot_violin_panel(data: dict) -> None:
    apply_plot_style()
    df = data["df"].copy()
    N = data["N"]

    # palettes: keep directions consistent across both panels
    palette = {"A to B": "blue", "B to A": "red"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    axes_map = {"Tr": axes[0], "H": axes[1]}
    metric_tex = {"Tr": r"\mathrm{Tr}", "H": r"\widehat{\mathrm{H}}"}

    for metric, ax in axes_map.items():
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            raise RuntimeError(f"No data for metric={metric}")

        # seaborn violin with split by direction
        sns.violinplot(
            x="dimension",
            y="signaling",
            hue="direction",
            data=sub,
            inner="quart",
            density_norm="area",
            split=True,
            gap=0.1,
            log_scale=(False, True),  # x linear, y log
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel(r"$d_A = d_B$")
        ax.set_ylabel(r"$\mathcal{S}_{X \to Y}$")
        ax.set_title(
            rf"${metric_tex[metric]}$-Signaling Distribution $(d_A = d_B,$ {_title_N(N)})"
        )
        ax.legend_.set_title("Direction")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"plots/signaling_violin_panel__N={N}__[dt={ts}].png"
    plt.savefig(out, dpi=300)
    print(f"Saved 2x1 panel to: {out}")
    plt.show()


def main(full_sim: bool = True):
    N = 10**3 if full_sim else 10**3
    DIMS = [2, 4, 8, 16] if full_sim else [2, 4, 8]

    data = simulate(N=N, dims=DIMS, dm_type="pure")

    plot_violin_panel(data)


if __name__ == "__main__":
    main()
