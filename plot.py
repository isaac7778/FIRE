
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from glob import glob

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT   = os.path.join(BASE_DIR, "csv_export")
PLOT_DIR   = os.path.join(BASE_DIR, "plotter", "plots", "wiml")

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Seeds (must match what export_data.py wrote) ────────────────────
SEEDS = [0, 5, 8]

# ── Plot styling ────────────────────────────────────────────────────
COLORS = {
    "rigl":   "red",
    "set":    "purple",
    "gmp":    "green",
    "static": "blue",
}
LABELS = {
    "rigl":   "RigL",
    "set":    "SET",
    "gmp":    "GMP",
    "static": "Static",
}

BLK_LINE_WIDTH   = 3
COLOR_LINE_WIDTH = 3
# BLK_LINE_WIDTH   = .5
# COLOR_LINE_WIDTH = .25
# X_LIM = (700, 1000) #zoom
X_LIM = (0, 1000) #reg
Y_LIM = (0, 0.85) #reg cifar10
# Y_LIM = (0, 0.33) #reg cifar100
# Y_LIM = (0.78,0.83) #zoom cifar10
# Y_LIM = (0,0.35) #zoom cifar100


# ── File-path builders (match export_data.py naming) ────────────────
def _dense_path(task, seed):
    return os.path.join(CSV_ROOT, task, "dense", f"dense_seed{seed}.csv")


def _static_path(task, seed, sparsity):
    return os.path.join(CSV_ROOT, task, "static",
                        f"static_seed{seed}_sp{sparsity}.csv")


def _gmp_path(task, seed, sparsity, nmu):
    return os.path.join(CSV_ROOT, task, "gmp",
                        f"gmp_seed{seed}_sp{sparsity}_nmu{nmu}.csv")


def _dst_path(task, sparsifier, seed, sparsity, pruning_ratio, nmu):
    """Path for rigl / set."""
    return os.path.join(CSV_ROOT, task, sparsifier,
                        f"{sparsifier}_seed{seed}_sp{sparsity}"
                        f"_pr{pruning_ratio}_nmu{nmu}.csv")


# ── Generic loader: aggregate across seeds ──────────────────────────
def load_and_aggregate(path_fn):
    """
    Args:
        path_fn: callable(seed) -> filepath

    Returns:
        (mean_array, std_array) or (None, None)
    """
    dfs = []
    for seed in SEEDS:
        fp = path_fn(seed)
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            dfs.append(df["accuracy"])
        else:
            pass  # silently skip missing seeds

    if not dfs:
        return None, None

    min_len = min(len(d) for d in dfs)
    dfs = [d[:min_len] for d in dfs]
    stacked = np.stack(dfs, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


# ── Subplot renderer ────────────────────────────────────────────────
def plot_subplot(ax, task, sparsity, nmu, pruning_ratio, dense_mean, dense_std):
    """Plot dense baseline + all sparsifiers on one axis."""

    # Dense baseline
    if dense_mean is not None:
        x = np.arange(len(dense_mean))
        ax.plot(x, dense_mean, label="Dense", color="black",
                linestyle="-", linewidth=BLK_LINE_WIDTH, alpha=0.7, zorder=10)
        ax.fill_between(x, dense_mean - dense_std, dense_mean + dense_std,
                        color="black", alpha=0.2, zorder=0)

    # Static (only varies by sparsity, not nmu)
    mean, std = load_and_aggregate(
        lambda s: _static_path(task, s, sparsity))
    if mean is not None:
        x = np.arange(len(mean))
        ax.plot(x, mean, label=LABELS["static"], color=COLORS["static"],
                linestyle="-", linewidth=COLOR_LINE_WIDTH, alpha=0.7, zorder=8)
        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS["static"], alpha=0.2, zorder=2)

    # GMP (varies by sparsity + nmu)
    mean, std = load_and_aggregate(
        lambda s: _gmp_path(task, s, sparsity, nmu))
    if mean is not None:
        x = np.arange(len(mean))
        ax.plot(x, mean, label=LABELS["gmp"], color=COLORS["gmp"],
                linestyle="-", linewidth=COLOR_LINE_WIDTH, alpha=0.7, zorder=8)
        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS["gmp"], alpha=0.2, zorder=2)

    # RigL / SET (vary by sparsity + pruning_ratio + nmu)
    for name in ("rigl", "set"):
        mean, std = load_and_aggregate(
            lambda s, _n=name: _dst_path(task, _n, s, sparsity, pruning_ratio, nmu))
        if mean is not None:
            x = np.arange(len(mean))
            ax.plot(x, mean, label=LABELS[name], color=COLORS[name],
                    linestyle="-", linewidth=COLOR_LINE_WIDTH, alpha=0.7, zorder=8)
            ax.fill_between(x, mean - std, mean + std,
                            color=COLORS[name], alpha=0.2, zorder=2)

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.grid(which="major", color="#666666", linestyle="-", linewidth=0.4)
    ax.grid(which="minor", color="#999999", linestyle=":", linewidth=0.3)
    # ax.minorticks_on()


# ── Grid plot (NMU rows × Sparsity cols) ────────────────────────────
def main_grid(task="CIFAR100", pruning_ratio=0.3,
              sparsity_list=None, nmu_list=None):
    """
    Create a grid of subplots for one task.
    Rows = nmu values, Columns = sparsity values.
    """
    if sparsity_list is None:
        sparsity_list = [0.1,0.3,0.5, 0.7, 0.9,0.95, 0.99]
    if nmu_list is None:
        nmu_list = [10, 100, 1000, 10000, 100000]

    n_rows = len(nmu_list)
    n_cols = len(sparsity_list)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5),
                             sharex=True, sharey=True)

    # Ensure axes is 2D even for 1-row or 1-col grids
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Dense baseline (independent of sparsity / nmu)
    dense_mean, dense_std = load_and_aggregate(
        lambda s: _dense_path(task, s))

    for row, nmu_val in enumerate(nmu_list):
        for col, sp_val in enumerate(sparsity_list):
            ax = axes[row, col]
            plot_subplot(ax, task, sp_val, nmu_val, pruning_ratio,
                         dense_mean, dense_std)

            if row == 0:
                ax.set_title(f"Sparsity: {sp_val}",
                             fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"NMU: {nmu_val}\nTest Accuracy",
                              fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Epochs", fontsize=9)

    # Shared legend from first subplot
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center",
                   ncol=len(leg_labels), fontsize=10, frameon=True,
                   bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        f"Test Accuracy vs. Epochs for {task}",
        # f"Rows: NMU  |  Columns: Sparsity",
        fontsize=25, fontweight="bold", y=1.02,
    )


    plt.tight_layout()
    # Uncomment to save:
    # plt.savefig(os.path.join(PLOT_DIR,
    #     f"{task}_grid_pr{pruning_ratio}_zoomed.svg"), bbox_inches="tight", dpi=150)
    # plt.savefig(os.path.join(PLOT_DIR,
    #     f"{task}_grid_pr{pruning_ratio}_zoomed.png"), bbox_inches="tight", dpi=150)
    plt.savefig(os.path.join(PLOT_DIR,
        f"{task}_grid_zoomed.svg"), bbox_inches="tight", dpi=150)
    plt.savefig(os.path.join(PLOT_DIR,
        f"{task}_grid_zoomed.png"), bbox_inches="tight", dpi=150)
    plt.close()

    
    # plt.tight_layout()
    # Uncomment to save:
    plt.savefig(os.path.join(PLOT_DIR,
        f"{task}_grid_pr{pruning_ratio}_zoomed.png"), bbox_inches="tight", dpi=150)
    plt.show()


# ── Single-plot mode ────────────────────────────────────────────────
def main(task="CIFAR10", sparsity=0.9, pruning_ratio=0.3, nmu=10000,
         zoom=False):
    """Single-panel plot for one (task, sparsity, pruning_ratio, nmu) combo."""
    if zoom:
        xlim, ylim, zoom_text = (700, 1000), (0.7, 0.9), "_Zoomed"
    else:
        xlim, ylim, zoom_text = X_LIM, Y_LIM, ""

    plt.figure(figsize=(20, 8))

    # Dense baseline
    dense_mean, dense_std = load_and_aggregate(
        lambda s: _dense_path(task, s))
    if dense_mean is not None:
        x = np.arange(len(dense_mean))
        plt.plot(x, dense_mean, label="Dense", color="black",
                 linestyle="-", linewidth=4, alpha=0.7, zorder=10)
        plt.fill_between(x, dense_mean - dense_std, dense_mean + dense_std,
                         color="black", alpha=0.2, zorder=0)
    print("done dense")
    # Static
    mean, std = load_and_aggregate(
        lambda s: _static_path(task, s, sparsity))
    if mean is not None:
        x = np.arange(len(mean))
        plt.plot(x, mean, label=LABELS["static"], color=COLORS["static"],
                 linestyle="-", linewidth=3, alpha=0.7, zorder=8)
        plt.fill_between(x, mean - std, mean + std,
                         color=COLORS["static"], alpha=0.2, zorder=2)
    print("done static")
    # GMP
    mean, std = load_and_aggregate(
        lambda s: _gmp_path(task, s, sparsity, nmu))
    if mean is not None:
        x = np.arange(len(mean))
        plt.plot(x, mean, label=LABELS["gmp"], color=COLORS["gmp"],
                 linestyle="-", linewidth=3, alpha=0.7, zorder=8)
        plt.fill_between(x, mean - std, mean + std,
                         color=COLORS["gmp"], alpha=0.2, zorder=2)
    print("done gmp")
    # RigL / SET
    for name in ("rigl", "set"):
        mean, std = load_and_aggregate(
            lambda s, _n=name: _dst_path(task, _n, s, sparsity, pruning_ratio, nmu))
        if mean is not None:
            x = np.arange(len(mean))
            plt.plot(x, mean, label=LABELS[name], color=COLORS[name],
                     linestyle="-", linewidth=3, alpha=0.7, zorder=8)
            plt.fill_between(x, mean - std, mean + std,
                             color=COLORS[name], alpha=0.2, zorder=2)
    print("done rigl/set")
    plt.legend(ncol=5, fontsize=12, frameon=True, loc="upper center")
    plt.grid(True)
    # plt.minorticks_on()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(which="minor", color="#999999", linestyle=":", linewidth=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title(f"{task}  |  Sparsity: {sparsity}  "
              f"Pruning Ratio: {pruning_ratio}  NMU: {nmu}", fontsize=20, fontweight="bold")
              
    # Uncomment to save:
    plt.savefig(os.path.join(PLOT_DIR,
        f"{task}_s{sparsity}_pr{pruning_ratio}_nmu{nmu}{zoom_text}.png"))
    plt.show()


if __name__ == "__main__":
    # --- Grid for each task ---
    sparsity_list = [0.1, 0.7, 0.95]
    nmu_list = [10, 1000, 10000]
    pruning_ratio = [0.5]
    # pruning_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in ("CIFAR100", ):#"CIFAR100"
        for r in pruning_ratio:
            main_grid(task=t, pruning_ratio=r, sparsity_list=sparsity_list, nmu_list=nmu_list)

    # --- Or single plot ---
    # main(task="CIFAR100", sparsity=0.5, pruning_ratio=0.9, nmu=1000)
    # main(task="CIFAR10", sparsity=0.3, pruning_ratio=0.1, nmu=1000)
