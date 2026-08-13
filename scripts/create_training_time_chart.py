from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "publication_benchmark"

EXPERIMENTS = ["115-feature RF", "Top20 RF", "Top20+SMOTE"]
TRAINING_SECONDS = [256.46, 104.74, 246.32]


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.65, 3.67))
    bars = axis.bar(
        EXPERIMENTS,
        TRAINING_SECONDS,
        width=0.8,
        color="#1f77b4",
    )

    axis.set_title("Training Time Comparison", fontsize=13, pad=10)
    axis.set_ylabel("Seconds")
    axis.set_ylim(0, 300)
    axis.set_yticks(np.arange(0, 301, 50))
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}" for value in TRAINING_SECONDS],
        padding=3,
    )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "training_time_comparison_updated.png",
        dpi=200,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIRECTORY / "training_time_comparison_updated.svg",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
