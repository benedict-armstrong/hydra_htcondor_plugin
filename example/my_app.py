"""Simple example demonstrating the HTCondor launcher with Hydra."""
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make sure the plugin package is importable when running from the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the plugin registers its configuration when running examples locally
import hydra_plugins.hydra_htcondor_launcher  # noqa: F401


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> float:
    """A simple task that simulates some work and returns a result."""
    print(f"Running job with: learning_rate={cfg.learning_rate}, batch_size={cfg.batch_size}")
    print(f"Training for {cfg.epochs} epochs...")

    # Simulate some work
    time.sleep(2)

    # Return a fake "loss" based on hyperparameters
    loss = 1.0 / (cfg.learning_rate * cfg.batch_size)
    print(f"Final loss: {loss:.4f}")

    return loss


if __name__ == "__main__":
    my_app()
