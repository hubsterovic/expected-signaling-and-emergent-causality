import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import os


def apply_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.5,
    )

    use_tex = shutil.which("pdflatex") is not None
    if use_tex:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.3,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    if use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": r"""
                    \usepackage{amsmath}
                    \usepackage{amssymb}
                """,
            }
        )
        print("[plot_styling] Using full LaTeX rendering via pdflatex.")
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "cm",
                "font.family": "serif",
            }
        )
        print("[plot_styling] LaTeX not found. Using mathtext fallback.")
