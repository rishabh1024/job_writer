"""
Configuration utilities for the job writer application.

This module provides functions for initializing and configuring 
language models and other resources.
"""

import os
from typing_extensions import Dict, Any, Tuple, Optional
from langchain.chat_models import init_chat_model

def init_models(config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """Initialize language models based on configuration."""
    config = config or {}
    
    # Model configuration with defaults
    model_name = config.get("model_name", os.getenv("OLLAMA_MODEL", "llama3.2:latest"))
    temperature = float(config.get("temperature", "0.3"))
    precise_temperature = float(config.get("precise_temperature", "0.2"))
    
    # Initialize models
    llm = init_chat_model(f"ollama:{model_name}", temperature=temperature)
    llm_precise = init_chat_model(f"ollama:{model_name}", temperature=precise_temperature)

    return llm, llm_precise