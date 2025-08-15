from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def simulate_over_dA(
    N: int,
    d_B: int,
    d_A_max: int = 50,
    num_d_A_points: int = 10,
) -> pd.DataFrame:
    # log-like spaced integer d_A grid
    d_As = sorted(
        set(
            int(round(i))
            for i in np.logspace(np.log10(2), np.log10(d_A_max), num=num_d_A_points)
        )
    )

    directions = ["A to B", "B to A"]
    dm_type = "product"
    rows: list[dict] = []

    for direction in directions:
        for d_A in d_As:
            res = haar_expected_mc_signaling_X_to_Y(
                N=N,
                d_A=d_A,
                d_B=d_B,
                dm_type=dm_type,
                direction=direction,  # type: ignore
            )

            metric_map = {
                "Tr": np.asarray(res["tr_dists"]),
                "H": np.asarray(res["h_dists"]),
            }
            for metric, samples in metric_map.items():
                q25, q50, q75 = np.percentile(samples, [25, 50, 75])
                rows.append(
                    {
                        "d_A": d_A,
                        "d_B": d_B,
                        "N": N,
                        "direction": direction,
                        "metric": metric,  # "Tr" or "H"
                        "mean": float(samples.mean()),
                        "median": float(q50),
                        "q25": float(q25),
                        "q75": float(q75),
                    }
                )

    return pd.DataFrame.from_records(rows)


def plot_metric_vs_dA(
    df_all: pd.DataFrame,
    N: int,
    d_B_low: int,
    d_B_high: int,
    metric: str,  # "Tr" or "H"
) -> None:
    if metric not in {"Tr", "H"}:
        raise ValueError("metric must be 'Tr' or 'H'")

    apply_plot_style()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metric_tex = r"\mathrm{Tr}" if metric == "Tr" else r"\widehat{\mathrm{H}}"

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = {
        "A to B": {"color": "blue", "marker": "o"},
        "B to A": {"color": "red", "marker": "s"},
    }

    summary_lines: list[str] = []
    for ax, d_B in zip(
        axes, [d_B_low, d_B_high]
    ):  # loop over each fixed d_B with its own plot/ax
        sub = df_all[(df_all["metric"] == metric) & (df_all["d_B"] == d_B)]
        if sub.empty:
            raise RuntimeError(f"No data for metric={metric} and d_B={d_B}")

        ax.set_xscale("log")
        ax.set_yscale("log")

        # plot both directions on same axes
        for direction in ["A to B", "B to A"]:
            dsub = sub[sub["direction"] == direction].sort_values("d_A")
            x = dsub["d_A"].to_numpy()
            y = dsub["mean"].to_numpy()
            yerr_lower = (dsub["median"] - dsub["q25"]).to_numpy()
            yerr_upper = (dsub["q75"] - dsub["median"]).to_numpy()

            # scatter + IQR error bars
            ax.scatter(
                x,
                y,
                color=styles[direction]["color"],
                marker=styles[direction]["marker"],
                s=70,
                label=direction,
            )
            ax.errorbar(
                x,
                y,
                yerr=[yerr_lower, yerr_upper],
                fmt="none",
                ecolor="gray",
                capsize=4,
                linewidth=1,
            )

            # Power-law fit in log-log
            log_x = np.log10(x)
            log_y = np.log10(np.maximum(y, np.finfo(float).tiny))
            slope, intercept = np.polyfit(log_x, log_y, 1)
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))

            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                color=styles[direction]["color"],
                label=rf"Fit: $d_A^{{{slope:.2f}}}$",
            )

            # Stats
            lin = stats.linregress(log_x, log_y)
            r2 = float(lin.rvalue**2)  # type: ignore
            stderr = float(lin.stderr)  # type: ignore
            pval = float(lin.pvalue)  # type: ignore
            prefactor = float(10**intercept)

            latex_eq = (
                rf"$\langle \mathcal{{S}}^{{{metric_tex}}} \rangle_{{{direction.replace('to', r'\to')}}} "
                rf"\approx {prefactor:.3g} \cdot d_A^{{{slope:.3f} \pm {stderr:.3f}}}$"
                f", $R^2 = {r2:.3f}$, $p = {pval:.2e}$"
            )

            summary_lines += [
                f"=== {metric} | d_B={d_B} | '{direction}' ===",
                f"Slope (scaling exponent): {slope:.6f} ± {stderr:.6f}",
                f"Intercept (log10 scale) : {intercept:.6f}",
                f"Prefactor (normal scale): {prefactor:.6e}",
                f"R-squared               : {r2:.6f}",
                f"p-value                 : {pval:.6e}",
                f"LaTeX fit expression    : {latex_eq}",
                "-" * 60,
                "",
            ]

        ax.set_title(
            rf"Expected ${metric_tex}$-Signaling $(d_B = {d_B}$, $N = 10^{{{int(np.log10(N))}}})$"
        )
        ax.set_xlabel(r"$d_A$")
        ax.set_ylabel(
            rf"$\langle \mathcal{{S}} \rangle_{{{direction.replace('to', r'\to')}}}^{{{metric_tex}}}$"  # type: ignore
        )
        ax.set_xlim(left=1.8, right=sub["d_A"].max() * 1.1)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(loc="lower left")

    plt.tight_layout()

    # save plot and stats
    plot_filename = f"plots/signaling_vs_dA_{metric}__[dt={timestamp}].png"
    plt.savefig(plot_filename)
    plt.show()

    stats_filename = (
        f"_stats/signaling_vs_dA_{metric}__[dt={timestamp}]__stat_summary.txt"
    )
    with open(stats_filename, "w") as f:
        f.write("\n".join(summary_lines))

    data_filename = f"_data/signaling_vs_dA_{metric}__[dt={timestamp}].csv"
    df_all.to_csv(data_filename, index=False)

    print("\n".join(summary_lines))
    print(f"\nSaved plot to:  {os.path.abspath(plot_filename)}")
    print(f"Saved stats to: {os.path.abspath(stats_filename)}")
    print(f"Saved data to: {os.path.abspath(data_filename)}")


def main(full_sim: bool = True):
    N = 10**3 if full_sim else 10**2
    D_A_MAX = 220 if full_sim else 40
    NUM_POINTS = 10 if full_sim else 5
    d_B_low, d_B_high = 2, 8 if full_sim else 4

    #  run simulations once per d_B
    df_low = simulate_over_dA(
        N=N, d_B=d_B_low, d_A_max=D_A_MAX, num_d_A_points=NUM_POINTS
    )
    df_high = simulate_over_dA(
        N=N,
        d_B=d_B_high,
        d_A_max=int(D_A_MAX * (d_B_low / d_B_high)),
        num_d_A_points=NUM_POINTS,
    )
    df_all = pd.concat([df_low, df_high], ignore_index=True)

    # plot separately per metric
    plot_metric_vs_dA(df_all, N=N, d_B_low=d_B_low, d_B_high=d_B_high, metric="Tr")
    plot_metric_vs_dA(df_all, N=N, d_B_low=d_B_low, d_B_high=d_B_high, metric="H")


if __name__ == "__main__":
    main()
