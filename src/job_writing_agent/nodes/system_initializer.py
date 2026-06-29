# -*- coding: utf-8 -*-
"""
System Initializer Module

This module provides the SystemInitializer class responsible for initializing
system messages in the workflow state. It follows the Single Responsibility
Principle by focusing solely on system message initialization.
"""

from typing import Any

from langchain_core.messages import SystemMessage

from job_writing_agent.prompts.templates import default_agent_system_prompt


class SystemInitializer:
    """
    Responsible for initializing system messages in workflow state.
    """

    def __init__(self, system_prompt: str = default_agent_system_prompt):
        """
        Initialize SystemInitializer with optional system prompt dependency injection.

        Parameters
        ----------
        system_prompt: Optional[str]
            System prompt text to use. Defaults to `agent_system_prompt` from
            prompts.templates. Can be injected for testing or custom prompts.
        """
        self._system_prompt = system_prompt

    def set_initial_agent_state(self) -> dict[str, Any]:
        """
        Partial state update for the first data-loading node.

        Returns only keys merged into ``DataLoadState``:

        - ``messages``: a single ``SystemMessage`` (``add_messages`` merges it
          with any messages already on state from the invoke payload).
        - ``current_node``: bookkeeping for the workflow.

        ``workflow_inputs`` is seeded separately in the graph node from the
        invoke ``WorkflowInput``, because ``DataLoadState`` nests that model under
        ``workflow_inputs`` while ``input_schema=WorkflowInput`` supplies flat
        fields to channels before this node runs.
        """
        return {
            "messages": [SystemMessage(content=self._system_prompt)],
            "current_node": "initialize_system",
        }
