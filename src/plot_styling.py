import seaborn as sns
import matplotlib.pyplot as plt


def apply_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.8,
    )
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.3,
        }
    )
