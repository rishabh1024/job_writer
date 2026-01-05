# -*- coding: utf-8 -*-
"""
System Initializer Module

This module provides the SystemInitializer class responsible for initializing
system messages in the workflow state. It follows the Single Responsibility
Principle by focusing solely on system message initialization.
"""

import logging
from typing import Optional

from langchain_core.messages import SystemMessage

from job_writing_agent.classes import DataLoadState
from job_writing_agent.prompts.templates import agent_system_prompt
from job_writing_agent.utils.logging.logging_decorators import log_async

logger = logging.getLogger(__name__)


class SystemInitializer:
    """
    Responsible for initializing system messages in workflow state.

    Example:
        >>> initializer = SystemInitializer()
        >>> state = await initializer.set_agent_system_message(initial_state)
        >>>
        >>> # With custom prompt for testing
        >>> custom_prompt = "Custom system prompt"
        >>> initializer = SystemInitializer(system_prompt=custom_prompt)
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize SystemInitializer with optional system prompt dependency injection.

        Parameters
        ----------
        system_prompt: Optional[str]
            System prompt text to use. Defaults to `agent_system_prompt` from
            prompts.templates. Can be injected for testing or custom prompts.
        """
        self._system_prompt = system_prompt or agent_system_prompt

    @log_async
    async def set_agent_system_message(self, state: DataLoadState) -> DataLoadState:
        """
        Add the system prompt to the conversation state.

        This method creates a SystemMessage from the configured prompt and
        adds it to the messages list in the workflow state.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state containing messages list.

        Returns
        -------
        DataLoadState
            Updated state with the system message added to messages list
            and current_node set to "initialize_system".
        """
        agent_initialization_system_message = SystemMessage(content=self._system_prompt)
        messages = state.get("messages", [])
        messages.append(agent_initialization_system_message)
        return {
            **state,
            "messages": messages,
            "current_node": "initialize_system",
        }
