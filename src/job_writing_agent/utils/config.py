"""
Configuration utilities for the job writer application.

This module provides functions for initializing and configuring
language models and other resources.
"""

# Standard library imports
import os

# Third-party imports
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


def init_models(
    config: dict[str, str | float] | None = None,
) -> tuple[BaseChatModel, BaseChatModel]:
    """
    Initialize language models based on configuration.

    Args:
        config: Optional configuration dictionary with keys:
            - model_name: Name of the model to use
            - temperature: Temperature for general LLM
            - precise_temperature: Temperature for precise LLM

    Returns:
        Tuple of (general_llm, precise_llm) instances
    """
    config = config or {}

    # Model configuration with defaults
    model_name = config.get("model_name", os.getenv("OLLAMA_MODEL", "llama3.2:latest"))
    temperature = float(config.get("temperature", "0.3"))
    precise_temperature = float(config.get("precise_temperature", "0.2"))

    # Initialize models
    general_llm = init_chat_model(f"ollama:{model_name}", temperature=temperature)
    precise_llm = init_chat_model(
        f"ollama:{model_name}", temperature=precise_temperature
    )

    return general_llm, precise_llm
