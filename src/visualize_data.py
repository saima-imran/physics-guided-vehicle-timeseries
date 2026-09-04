"""Visualize the generated vehicle time-series data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "vehicle_timeseries.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "vehicle_signals.png"


def main() -> None:
    """Load the dataset and create an overview figure."""

    dataset = pd.read_csv(DATA_FILE)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    axes[0].plot(
        dataset["time_s"],
        dataset["throttle"],
        label="Throttle",
        color="green",
    )
    axes[0].plot(
        dataset["time_s"],
        dataset["brake"],
        label="Brake",
        color="red",
    )
    axes[0].set_ylabel("Driver input")
    axes[0].set_title("Synthetic Vehicle Driving Sequence")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        dataset["time_s"],
        dataset["speed_true_mps"],
        label="True speed",
        color="blue",
        linewidth=2,
    )
    axes[1].scatter(
        dataset["time_s"],
        dataset["speed_observed_mps"],
        label="Observed speed",
        color="orange",
        s=6,
        alpha=0.45,
    )
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        dataset["time_s"],
        dataset["acceleration_true_mps2"],
        label="True acceleration",
        color="purple",
        linewidth=2,
    )
    axes[2].plot(
        dataset["time_s"],
        dataset["acceleration_observed_mps2"],
        label="Observed acceleration",
        color="gray",
        linewidth=1,
        alpha=0.7,
    )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    figure.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=150)
    plt.show()

    print(f"Figure saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

    