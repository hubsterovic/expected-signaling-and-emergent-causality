import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from help import haar_expected_mc_signaling_X_to_Y


N = 100
d_B = 2
d_A_MAX = 20

start_exp = np.log10(d_B)
stop_exp = np.log10(d_A_MAX)
d_As = sorted(set(int(round(i)) for i in np.logspace(start_exp, stop_exp, num=10)))

# Data collection
means_atb = []
means_bta = []

for d_A in d_As:
    m_atb, _ = haar_expected_mc_signaling_X_to_Y(
        N, int(d_A), d_B, dm_type="pure", direction="A to B"
    )
    m_bta, _ = haar_expected_mc_signaling_X_to_Y(
        N, int(d_A), d_B, dm_type="pure", direction="B to A"
    )
    means_atb.append(m_atb)
    means_bta.append(m_bta)

# Convert to arrays
means_atb = np.array(means_atb)
means_bta = np.array(means_bta)

# Log-log fit for pure
log_d_As = np.log10(d_As)

log_means_atb = np.log10(means_atb)
slope_atb, intercept_atb = np.polyfit(log_d_As, log_means_atb, 1)
fitted_atb = 10 ** (intercept_atb + slope_atb * log_d_As)

# Log-log fit for bta
log_means_bta = np.log10(means_bta)
slope_bta, intercept_bta = np.polyfit(log_d_As, log_means_bta, 1)
fitted_bta = 10 ** (intercept_bta + slope_bta * log_d_As)

# Plotting
plt.figure(figsize=(8, 6))

# Plot raw data
sns.scatterplot(x=d_As, y=means_atb, label="A to B", marker="o")
sns.scatterplot(x=d_As, y=means_bta, label="B to A", marker="s")

# Plot fits
plt.plot(
    d_As,
    fitted_atb,
    linestyle="--",
    label=rf"A $\rightarrow$ B fit: $y \propto x^{{{slope_atb:.2f}}}$",
)
plt.plot(
    d_As,
    fitted_bta,
    linestyle="--",
    label=rf"B $\rightarrow$ A : $y \propto x^{{{slope_bta:.2f}}}$",
)

# Log-log scale
plt.xscale("log")
plt.yscale("log")

# Labels and styling
plt.xlabel(r"$d_A$ (log scale)")
plt.ylabel(r"$\langle \mathcal{S}_{X \rightarrow Y} \rangle$ (log scale)")
plt.title(f"Log-Log Power Law Fit for Signaling Strength (N={N})")
plt.legend()
sns.despine()
plt.tight_layout()
plt.show()
