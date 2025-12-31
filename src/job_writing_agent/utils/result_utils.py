"""
Utility functions for handling and saving workflow results.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def print_result(content_type: str, final_content: str):
    """
    Prints the final generated content to the console with formatting.

    Args:
        content_type: The type of content being printed (e.g., "cover_letter").
        final_content: The final generated content string.
    """
    print("\n" + "=" * 80)
    print(f"FINAL {content_type.upper()}:\n{final_content}")
    print("=" * 80)


def save_result(content_type: str, final_content: str) -> str:
    """
    Saves the final generated content to a timestamped text file.

    Args:
        content_type: The type of content being saved, used in the filename.
        final_content: The final generated content string.

    Returns:
        The path to the saved output file.
    """
    output_file = f"{content_type}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    logger.info("Saved to %s", output_file)
    return output_file
