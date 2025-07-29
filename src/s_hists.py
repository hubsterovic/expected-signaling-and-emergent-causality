from help import haar_expected_mc_signaling_X_to_Y
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


N = 100
params = [
    {"d_A": 2, "d_B": 32, "label": "A=2, B=32"},
    {"d_A": 4, "d_B": 16, "label": "A=4, B=16"},
    {"d_A": 8, "d_B": 8, "label": "A=8, B=8"},
    {"d_A": 16, "d_B": 4, "label": "A=16, B=4"},
    {"d_A": 32, "d_B": 2, "label": "A=32, B=2"},
]


results = []
for p in params:
    _, dists = haar_expected_mc_signaling_X_to_Y(
        N, p["d_A"], p["d_B"], direction="A to B", dm_type="pure"
    )
    results.append((p["label"], np.array(dists)))

plt.figure(figsize=(9, 6))
for label, dists in results:
    sns.histplot(
        dists,
        bins=40,
        stat="density",
        element="step",
        label=label,
        log_scale=(True, False),
    )

plt.xlabel(r"$\mathcal{S}_{A \rightarrow B}$ (log scale)")
plt.ylabel("Density")
plt.title(f"Comparison of Signaling Distributions (N={N})")
plt.legend(title="Subsystem Dimensions")
plt.tight_layout()
plt.show()
