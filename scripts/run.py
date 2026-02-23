#!/usr/bin/env python3
"""Unified run script entry point.

This script provides a command-line interface. Behavior is determined by train_interval
(0 = rollout only, >0 = training) and config options (envs, load_train_dataset).

Usage:
    run [config_file]
    
Examples:
    run configs/train.yaml                 # Train from dataset
    run configs/deploy.yaml                # Rollout only (collect data)

Requirements:
    - PyTorch
    - Gymnasium (with Atari support for deploy mode)
    - WandB (for training mode)
    - Hugging Face datasets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loop import main as run_main

def main() -> None:
    """Main function."""
    # Allow config file to be specified as command line argument
    config_path = ""
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Call the main run function
    run_main(config_path)

if __name__ == "__main__":
    main()

