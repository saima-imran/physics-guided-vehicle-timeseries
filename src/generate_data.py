"""Generate a simple synthetic vehicle time-series dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


# Experiment settings
RANDOM_SEED = 42
TIME_STEP_SECONDS = 0.1
DURATION_SECONDS = 80.0

# Simplified vehicle parameters
MAX_DRIVE_ACCELERATION = 3.0
MAX_BRAKE_DECELERATION = 5.5
ROLLING_DECELERATION = 0.10
DRAG_COEFFICIENT = 0.003

# Sensor-noise settings
SPEED_NOISE_STANDARD_DEVIATION = 0.25
ACCELERATION_NOISE_STANDARD_DEVIATION = 0.08

# Output location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "vehicle_timeseries.csv"


def create_time_axis() -> np.ndarray:
    """Create regularly spaced measurement times."""

    return np.arange(
        0.0,
        DURATION_SECONDS + TIME_STEP_SECONDS,
        TIME_STEP_SECONDS,
    )


def create_driver_inputs(
    time_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create simple throttle and braking commands."""

    throttle = np.zeros_like(time_seconds)
    brake = np.zeros_like(time_seconds)

    # First acceleration and cruising period
    throttle[(time_seconds >= 2) & (time_seconds < 18)] = 0.65
    throttle[(time_seconds >= 18) & (time_seconds < 32)] = 0.20

    # First braking period
    brake[(time_seconds >= 36) & (time_seconds < 44)] = 0.40

    # Second acceleration period
    throttle[(time_seconds >= 48) & (time_seconds < 62)] = 0.50

    # Second braking period
    brake[(time_seconds >= 68) & (time_seconds < 78)] = 0.55

    return throttle, brake


def simulate_vehicle(
    time_seconds: np.ndarray,
    throttle: np.ndarray,
    brake: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate true vehicle speed and acceleration."""

    speed = np.zeros_like(time_seconds)
    acceleration = np.zeros_like(time_seconds)

    for index in range(len(time_seconds) - 1):
        current_speed = speed[index]

        drag_deceleration = DRAG_COEFFICIENT * current_speed**2

        if current_speed == 0 and throttle[index] == 0:
            rolling_deceleration = 0.0
        else:
            rolling_deceleration = ROLLING_DECELERATION

        acceleration[index] = (
            MAX_DRIVE_ACCELERATION * throttle[index]
            - MAX_BRAKE_DECELERATION * brake[index]
            - rolling_deceleration
            - drag_deceleration
        )

        next_speed = (
            current_speed
            + acceleration[index] * TIME_STEP_SECONDS
        )

        # A vehicle's forward speed cannot become negative.
        speed[index + 1] = max(0.0, next_speed)

    # Reuse the final calculated acceleration value.
    acceleration[-1] = acceleration[-2]

    return speed, acceleration


def add_sensor_noise(
    speed: np.ndarray,
    acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy measurements while preserving the true values."""

    random_generator = np.random.default_rng(RANDOM_SEED)

    speed_noise = random_generator.normal(
        0.0,
        SPEED_NOISE_STANDARD_DEVIATION,
        size=len(speed),
    )

    acceleration_noise = random_generator.normal(
        0.0,
        ACCELERATION_NOISE_STANDARD_DEVIATION,
        size=len(acceleration),
    )

    observed_speed = np.maximum(0.0, speed + speed_noise)
    observed_acceleration = acceleration + acceleration_noise

    return observed_speed, observed_acceleration


def create_dataset() -> pd.DataFrame:
    """Run the simulation and return the complete dataset."""

    time_seconds = create_time_axis()

    throttle, brake = create_driver_inputs(time_seconds)

    true_speed, true_acceleration = simulate_vehicle(
        time_seconds,
        throttle,
        brake,
    )

    observed_speed, observed_acceleration = add_sensor_noise(
        true_speed,
        true_acceleration,
    )

    dataset = pd.DataFrame(
        {
            "time_s": time_seconds,
            "throttle": throttle,
            "brake": brake,
            "acceleration_true_mps2": true_acceleration,
            "speed_true_mps": true_speed,
            "acceleration_observed_mps2": observed_acceleration,
            "speed_observed_mps": observed_speed,
        }
    )

    # The target is the true vehicle speed at the next time step.
    dataset["speed_next_true_mps"] = dataset[
        "speed_true_mps"
    ].shift(-1)

    # Remove the final row because it has no next-step target.
    return dataset.iloc[:-1].copy()


def save_dataset(dataset: pd.DataFrame) -> None:
    """Save the generated dataset as a CSV file."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)


def main() -> None:
    """Generate, save and summarize the dataset."""

    dataset = create_dataset()
    save_dataset(dataset)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Number of rows: {len(dataset)}")
    print()
    print(dataset.head())
    print()
    print("Summary:")
    print(dataset.describe().round(3))


if __name__ == "__main__":
    main()
    