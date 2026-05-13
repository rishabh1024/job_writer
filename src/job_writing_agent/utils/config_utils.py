"""
Utility functions for creating model configurations.
"""

import argparse
from typing import Dict, Any


def create_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Creates a model configuration dictionary from command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A dictionary with model configuration parameters.
    """
    model_config = {}
    if args.model:
        model_config["model_name"] = args.model
    if args.temp is not None:
        model_config["temperature"] = min(0.25, args.temp)
        model_config["precise_temperature"] = min(0.2, args.temp)
    return model_config
