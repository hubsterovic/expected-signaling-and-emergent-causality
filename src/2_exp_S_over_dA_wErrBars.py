from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from datetime import datetime
from typing import Literal
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def simulate(
    N=100,
    d_B=2,
    d_A_MAX=50,
    num_d_A_points=10,
    distance: Literal["half-HS-squared", "trace"] = "half-HS-squared",
):
    # This is to get log like integers for d_A
    d_As = sorted(
        set(
            int(round(i))
            for i in np.logspace(np.log10(d_B), np.log10(d_A_MAX), num=num_d_A_points)
        )
    )

    directions = ["A to B", "B to A"]
    dm_type = "product"

    data = []
    for direction in directions:
        for d_A in d_As:
            mean_val, samples = haar_expected_mc_signaling_X_to_Y(
                N=N,
                d_A=d_A,
                d_B=d_B,
                dm_type=dm_type,
                direction=direction,  # type: ignore
                distance=distance,
            )
            samples = np.array(samples)
            log_samples = np.log10(samples)

            q25, q50, q75 = np.percentile(log_samples, [25, 50, 75])
            data.append(
                {
                    "d_A": d_A,
                    "mean": mean_val,
                    "median": 10**q50,
                    "q25": 10**q25,
                    "q75": 10**q75,
                    "direction": direction,
                }
            )

    return pd.DataFrame(data)


def plot(df, N, distance: Literal["half-HS-squared", "trace"] = "half-HS-squared"):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xscale("log")
    ax.set_yscale("log")

    styles = {
        "A to B": {"color": "blue", "marker": "o"},
        "B to A": {"color": "red", "marker": "s"},
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_lines = []

    for direction in df["direction"].unique():
        subdf = df[df["direction"] == direction]
        x = subdf["d_A"]
        y = subdf["mean"]
        yerr_lower = subdf["median"] - subdf["q25"]
        yerr_upper = subdf["q75"] - subdf["median"]

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

        log_x = np.log10(x)
        log_y = np.log10(y)

        slope, intercept = np.polyfit(log_x, log_y, 1)
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_fit = 10 ** (intercept + slope * np.log10(x_fit))

        linreg_res = stats.linregress(log_x, log_y)
        r2 = linreg_res.rvalue**2  # type: ignore
        stderr = linreg_res.stderr  # type: ignore
        pval = linreg_res.pvalue  # type: ignore

        prefactor = 10**intercept

        latex_eq = (
            rf"$\langle S \rangle_{{{direction.replace('to', r'\to')}}} "
            rf"\approx {prefactor:.3g} \cdot d_A^{{{slope:.3f} \pm {stderr:.3f}}}$"
            f", $R^2 = {r2:.3f}$, $p = {pval:.2e}$"
        )

        summary_lines.append(f"=== Fit statistics for '{direction}' ===")
        summary_lines.append(f"Slope (scaling exponent): {slope:.5f} ± {stderr:.5f}")
        summary_lines.append(f"Intercept (log10 scale) : {intercept:.5f}")
        summary_lines.append(f"Prefactor (normal scale): {prefactor:.5f}")
        summary_lines.append(f"R-squared               : {r2:.5f}")
        summary_lines.append(f"p-value                 : {pval:.5e}")
        summary_lines.append(f"LaTeX fit expression    : {latex_eq}")
        summary_lines.append("---------------------------------------------------\n")

        ax.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color=styles[direction]["color"],
            label=rf"Fit $\langle \mathcal{{S}} \rangle_{{{direction.replace('to', r'\to')}}} \propto d_A^{{{slope:.2f}}}$",
        )

    ax.set_xlabel(r"$d_A$")
    ax.set_ylabel(r"$\langle \mathcal{S} \rangle_{X \rightarrow Y}$")
    ax.set_xlim(left=1.8, right=df["d_A"].max() * 1.1)

    if distance == "half-HS-squared":
        title_str = rf"Expected Signaling (Log-Log) with IQR Error Bars and Fits $(N=10^{int(np.log10(N))})$"
    elif distance == "trace":
        title_str = rf"Expected Trace Signaling (Log-Log) with IQR Error Bars and Fits $(N=10^{int(np.log10(N))})$"

    ax.set_title(title_str)
    ax.legend(loc="lower left")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tight_layout()

    summary_str = "\n".join(summary_lines)
    summary_filename = (
        f"plots/expected_signaling_wEB__[dt={timestamp}]__stat_summary.txt"
    )
    with open(summary_filename, "w") as f:
        f.write(summary_str)

    print("\n" + summary_str)
    print(f"\nStatistical summary saved to: {os.path.abspath(summary_filename)}")

    plot_filename = f"plots/expected_signaling_wEB__[dt={timestamp}].png"
    plt.savefig(plot_filename)
    plt.show()


def main():
    N = 10**3
    D_A_MAX = 20
    NUM_POINTS = 10
    DISTANCE: Literal["half-HS-squared", "trace"] = "half-HS-squared"
    df = simulate(N=N, d_A_MAX=D_A_MAX, num_d_A_points=NUM_POINTS, distance=DISTANCE)
    plot(df, N)


if __name__ == "__main__":
    main()
