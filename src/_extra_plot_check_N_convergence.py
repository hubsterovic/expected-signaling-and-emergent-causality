import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qmc import haar_expected_mc_signaling_X_to_Y

d_A = 4
d_B = 4
direction = "A to B"
dm_type = "pure"

total_samples = 10**6
chunk_size = 100
n_chunks = total_samples // chunk_size

running_means = []
running_stds = []
sample_counts = []

all_samples = []

for _ in tqdm(range(n_chunks), desc="checking convergence"):
    _, dists = haar_expected_mc_signaling_X_to_Y(
        N=chunk_size, d_A=d_A, d_B=d_B, direction=direction, dm_type=dm_type
    )

    all_samples.extend(dists)

    running_mean = np.mean(all_samples)
    running_std = np.std(all_samples) / np.sqrt(len(all_samples))

    running_means.append(running_mean)
    running_stds.append(running_std)
    sample_counts.append(len(all_samples))

plt.figure(figsize=(8, 5))
running_means = np.array(running_means)
running_stds = np.array(running_stds)

plt.plot(
    sample_counts, running_means, label=r"Running mean $\langle \mathcal{S} \rangle$"
)
plt.fill_between(
    sample_counts,
    running_means - running_stds,
    running_means + running_stds,
    alpha=0.3,
    label=r"$\pm$ SEM",
)

plt.xscale("log")
plt.xlabel("Number of Monte Carlo samples $N$")
plt.ylabel(r"Estimated $\langle \mathcal{S} \rangle$")
plt.title("Monte Carlo Convergence of Haar-Expected Signaling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
