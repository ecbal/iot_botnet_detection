from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")
FIGURE_PATH = FIGURE_DIR / "combined_confusion_matrices.png"

MATRIX_PATHS = {
    "(a) RF Baseline": REPORT_DIR
    / "all_devices_stratified_random_forest_confusion_matrix.csv",
    "(b) RF Top-20": REPORT_DIR
    / "all_devices_stratified_random_forest_top20_confusion_matrix.csv",
    "(c) RF Top-20 + SMOTE": REPORT_DIR
    / "all_devices_stratified_random_forest_top20_smote_confusion_matrix.csv",
}

CLASS_NAMES = ["Benign", "Attack"]


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = CLASS_NAMES
    matrix.columns = CLASS_NAMES
    return matrix.astype(int)


def load_matrices() -> dict[str, pd.DataFrame]:
    matrices = {}
    for label, path in MATRIX_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing confusion matrix CSV: {path}")
        matrices[label] = read_matrix(path)
    return matrices


def print_matrices(matrices: dict[str, pd.DataFrame]) -> None:
    for label, matrix in matrices.items():
        print()
        print(label)
        print(matrix.to_string())


def draw_combined_figure(matrices: dict[str, pd.DataFrame]) -> None:
    sns.set_theme(style="white", font="DejaVu Sans")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    vmax = max(int(matrix.to_numpy().max()) for matrix in matrices.values())
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.3, 3.9),
        dpi=300,
        facecolor="white",
        constrained_layout=True,
    )

    for ax, (label, matrix) in zip(axes, matrices.items()):
        annotations = matrix.map(lambda value: f"{int(value):,}")
        sns.heatmap(
            matrix,
            ax=ax,
            annot=annotations,
            fmt="",
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=vmax,
            square=True,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"fontsize": 10, "fontweight": "bold"},
        )
        ax.set_xlabel("Predicted class", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual class", fontsize=10, labelpad=8)
        ax.set_xticklabels(CLASS_NAMES, rotation=0, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=9)
        ax.tick_params(length=0)
        ax.text(
            0.5,
            1.08,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    matrices = load_matrices()
    print_matrices(matrices)
    draw_combined_figure(matrices)
    print()
    print(f"Figure written to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
