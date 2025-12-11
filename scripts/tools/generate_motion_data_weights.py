#!/usr/bin/env python3
"""
Generate motion data weights YAML file.

This script takes a directory containing motion data files and generates a YAML file
that maps each filename to a weight value.

Usage:
    python generate_motion_data_weights.py <input_directory> [--output OUTPUT] [--default-weight WEIGHT]

Example:
    python generate_motion_data_weights.py ./motion_data --output weights.yaml --default-weight 1.0
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path


def generate_weights(input_dir, output_path, default_weight=1.0):
    """
    Generate YAML weights file from files in input directory.

    Args:
        input_dir: Path to directory containing motion data files
        output_path: Path where YAML file should be saved
        default_weight: Default weight value for all files
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    # Get all files in directory (excluding subdirectories)
    files = []
    for item in input_path.iterdir():
        if item.is_file():
            files.append(item.name)

    if not files:
        print(f"Warning: No files found in directory '{input_dir}'")

    # Sort files alphabetically for consistent output
    files.sort()

    # Create weights dictionary (use filename without extension as key)
    weights = {}
    for filename in files:
        name_without_ext = Path(filename).stem
        weights[name_without_ext] = default_weight

    # Write YAML file
    with open(output_path, "w") as f:
        yaml.dump(weights, f, default_flow_style=False, sort_keys=True)

    # Write JSON file
    json_output_path = Path(output_path).with_suffix(".json")
    with open(json_output_path, "w") as f:
        json.dump(weights, f, indent=2, sort_keys=True)

    print(f"Generated weights YAML file: {output_path}")
    print(f"Generated weights JSON file: {json_output_path}")
    print(f"Total files processed: {len(files)}")
    print(f"Default weight: {default_weight}")

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion data weights YAML file"
    )
    parser.add_argument("input_dir", help="Directory containing motion data files")
    parser.add_argument(
        "--output",
        "-o",
        default="weights.yaml",
        help="Output YAML file path (default: weights.yaml)",
    )
    parser.add_argument(
        "--default-weight",
        "-w",
        type=float,
        default=1.0,
        help="Default weight value for all files (default: 1.0)",
    )

    args = parser.parse_args()

    generate_weights(args.input_dir, args.output, args.default_weight)


if __name__ == "__main__":
    main()
