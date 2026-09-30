"""Create the project synthetic multi-episode network traffic sample."""

from pathlib import Path

from src.preprocessing.synthetic_data import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    destination = PROJECT_ROOT / "data" / "sample" / "sample_network_traffic.csv"
    rows = generate_synthetic_dataset(destination, num_episodes=20, seed=26153, minimum_records=1000)
    print(f"Generated {len(rows)} synthetic observations at {destination}")
