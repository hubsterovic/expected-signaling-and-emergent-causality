from qmc import haar_expected_mc_signaling_X_to_Y
from plot_styling import apply_plot_style
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def freedman_diaconis_bins(x: np.ndarray) -> int:
    # gpt gen
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 1
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(10, max(1, int(np.sqrt(n))))
    bin_width = 2 * iqr * (n ** (-1 / 3))
    if bin_width <= 0:
        return min(50, max(5, int(np.sqrt(n))))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(5, min(200, bins))


def histogram_mode(x: np.ndarray) -> float:
    # gpt gen
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.allclose(x, x[0]):
        return float(x[0])
    bins = freedman_diaconis_bins(x)
    counts, edges = np.histogram(x, bins=bins)
    idx = np.argmax(counts)
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def simulate_symmetric_one_direction(
    N: int = 10**4,
    d_min: int = 2,
    d_max: int = 20,
    dm_type: Literal["pure", "product", "mixed"] = "pure",
    metric: Literal["Tr", "H"] = "Tr",
    direction: Literal["A to B", "B to A"] = "A to B",
) -> pd.DataFrame:
    assert metric in {"Tr", "H"}
    dims = range(d_min, d_max + 1)
    rows: list[dict] = []

    for d in dims:
        sd = haar_expected_mc_signaling_X_to_Y(
            N=N,
            d_A=d,
            d_B=d,
            direction=direction,  # type: ignore
            dm_type=dm_type,
            fixed_coms=None,
        )
        samples = sd["tr_dists"] if metric == "Tr" else sd["h_dists"]
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]

        # linear-scale percentiles for IQR
        q25, med, q75 = (
            np.percentile(s, [25, 50, 75]) if s.size else (np.nan, np.nan, np.nan)
        )

        rows.append(
            {
                "dimension": d,
                "inv_dimension": 1.0 / d,
                "mean": float(np.mean(s)) if s.size else np.nan,
                "median": float(med),
                "mode": float(histogram_mode(s)),
                "q25": float(q25),
                "q75": float(q75),
                "iqr_lower": float(max(0.0, med - q25)) if s.size else np.nan,
                "iqr_upper": float(max(0.0, q75 - med)) if s.size else np.nan,
                "std": float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
                "var": float(np.var(s, ddof=1)) if s.size > 1 else 0.0,
                "n": int(s.size),
            }
        )

    df = pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)

    # print("\nPer-dimension summary (mean/median/mode, IQR, std, n):")
    # with pd.option_context("display.max_rows", None, "display.width", 120):
    #     print(df[["dimension","mean","median","mode","q25","q75","std","n"]].to_string(index=False))

    return df


def fit_power_exponent(d: np.ndarray, y: np.ndarray):
    # gpt gen
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (d > 0) & (y > 0) & np.isfinite(y)
    d_use, y_use = d[mask], y[mask]
    n = d_use.size
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan

    X = np.log(d_use)
    Y = np.log(y_use)

    # linear fit: Y = a + p X
    p, a = np.polyfit(X, Y, 1)
    Y_hat = a + p * X
    resid = Y - Y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # slope standard error: sqrt( sigma^2 / Sxx ), sigma^2 = RSS/(n-2), Sxx = sum((X - Xbar)^2)
    dof = max(1, n - 2)
    sigma2 = ss_res / dof
    sxx = float(np.sum((X - np.mean(X)) ** 2))
    p_stderr = float(np.sqrt(sigma2 / sxx)) if sxx > 0 else np.nan

    C = float(np.exp(a))
    return float(p), C, float(r2), int(n), p_stderr


def compute_fits(
    stats_df: pd.DataFrame,
    stats=("mean", "median", "mode"),
    min_fit_dim: int | None = None,
    max_fit_dim: int | None = None,
) -> pd.DataFrame:
    sub = stats_df.copy()
    if min_fit_dim is not None:
        sub = sub[sub["dimension"] > min_fit_dim]
    if max_fit_dim is not None:
        sub = sub[sub["dimension"] <= max_fit_dim]

    rows = []
    for stat in stats:
        d = sub["dimension"].to_numpy()
        y = sub[stat].to_numpy()
        p_fit, C_fit, r2, n, p_stderr = fit_power_exponent(d, y)
        rows.append(
            {
                "stat": stat,
                "p_fit": p_fit,
                "p_stderr": p_stderr,
                "C_fit": C_fit,
                "r2_loglog": r2,
                "n_points": n,
                "min_fit_dim": min_fit_dim,
                "max_fit_dim": max_fit_dim,
            }
        )
    return pd.DataFrame(rows)


def plot_compensated_metric(
    stats_df: pd.DataFrame,
    fits_df: pd.DataFrame,
    N: int,
    d_min: int,
    d_max: int,
    metric: Literal["Tr", "H"] = "Tr",
):
    # gpt gen
    apply_plot_style()

    metric_tex = r"\mathrm{Tr}" if metric == "Tr" else r"\widehat{\mathrm{H}}"
    markers = {"mean": "o", "median": "s", "mode": "^"}
    linestyles = {"mean": "-", "median": "--", "mode": "-."}

    plt.figure(figsize=(10, 6))
    sub = stats_df.sort_values("inv_dimension")
    d = sub["dimension"].to_numpy()
    x = sub["inv_dimension"].to_numpy()  # 1/d

    for stat in ["mean", "median", "mode"]:
        y = sub[stat].to_numpy()

        # Get fitted exponent and stderr for this stat
        row = fits_df[fits_df.stat == stat]
        if row.empty or not np.isfinite(row["p_fit"].values[0]):
            y_comp = np.full_like(y, np.nan, dtype=float)
            lower = upper = np.full_like(y, np.nan, dtype=float)
            p_str = "n/a"
        else:
            p_fit = float(row["p_fit"].values[0])
            p_se = float(row["p_stderr"].values[0])
            # Compensation: y'(d) = d^{-p_fit} * y(d)
            scale = d ** (-p_fit)
            y_comp = scale * y

            # IQR error bars computed from (q25, median, q75); reuse for each stat.
            # Transform by the same scale factor.
            q25 = sub["q25"].to_numpy()
            med = sub["median"].to_numpy()
            q75 = sub["q75"].to_numpy()
            y_q25 = scale * q25
            y_med = scale * med
            y_q75 = scale * q75
            lower = np.maximum(0.0, y_med - y_q25)
            upper = np.maximum(0.0, y_q75 - y_med)

            p_str = f"{p_fit:+.2f} \\pm {p_se:.2f}"

        label = f"{stat.capitalize()}  ($p={p_str}$)"
        # line + markers
        plt.plot(
            x,
            y_comp,
            linestyle=linestyles[stat],
            marker=markers[stat],
            label=label,
        )
        # IQR error bars (asymmetric)
        plt.errorbar(
            x,
            y_comp,
            yerr=[lower, upper],
            fmt="none",
            ecolor="gray",
            capsize=3,
            elinewidth=1,
            alpha=0.8,
        )

    plt.xlabel(r"$1/d_A$ (with $d_A=d_B$)")
    plt.ylabel(rf"Compensated ${metric_tex}$ Signaling $y'(d)=d^{{-p}} f(d)$")
    plt.title(
        rf"Compensated ${metric_tex}$-Signaling: mean / median / mode vs $1/d_A$ "
        rf"(log-$x$).  $N={N}$, $d \in [{d_min},{d_max}]$"
    )
    plt.xscale("log")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"plots/compensated_{metric}_mean_median_mode__N={N}__d{d_min}-{d_max}__[dt={ts}].png"
    plt.savefig(out, dpi=300)
    print(f"Saved: {out}")
    plt.show()




def main(full_sim: bool = True):
    N = 10**3 if full_sim else 10**3
    d_min, d_max = 2, (20 if full_sim else 12)
    min_fit_dim = int(d_max / 2)  # ignore small-d region in the exponent fit

    # --------- Tr. ---------
    tr_stats = simulate_symmetric_one_direction(
        N=N, d_min=d_min, d_max=d_max, metric="Tr", direction="A to B"
    )
    tr_fits = compute_fits(
        tr_stats, stats=("mean", "median", "mode"), min_fit_dim=min_fit_dim
    )
    csv_tr = f"_data/compensation_exponents_Tr__[dt={datetime.now().strftime('%Y%m%d_%H%M%S')}].csv"
    tr_fits.to_csv(csv_tr, index=False)
    print("\nTRACE: fitted exponents (y ≈ C d^{p}):")
    print(tr_fits.to_string(index=False))
    print(f"Saved CSV: {csv_tr}")
    plot_compensated_metric(
        tr_stats, tr_fits, N=N, d_min=d_min, d_max=d_max, metric="Tr"
    )

    # --- H (normalized half-HS-squared) ---
    h_stats = simulate_symmetric_one_direction(
        N=N, d_min=d_min, d_max=d_max, metric="H", direction="A to B"
    )
    h_fits = compute_fits(
        h_stats, stats=("mean", "median", "mode"), min_fit_dim=min_fit_dim
    )
    csv_h = f"_data/compensation_exponents_H__[dt={datetime.now().strftime('%Y%m%d_%H%M%S')}].csv"
    h_fits.to_csv(csv_h, index=False)
    print("\nH (half-HS-squared): fitted exponents (y ≈ C d^{p}):")
    print(h_fits.to_string(index=False))
    print(f"Saved CSV: {csv_h}")
    plot_compensated_metric(h_stats, h_fits, N=N, d_min=d_min, d_max=d_max, metric="H")


if __name__ == "__main__":
    main()
