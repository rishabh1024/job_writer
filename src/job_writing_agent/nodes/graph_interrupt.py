# -*- coding: utf-8 -*-
"""
GraphInterrupt: LangGraph interrupt payload and request-input flow for human-in-the-loop.

Encapsulates JSON-serializable interrupt payloads and the "request input for a field"
pattern used in the data loading subgraph. The payload built here is what the client
sees in chunk.data["__interrupt__"][0].value when streaming with stream_subgraphs=True.
"""

import logging
from typing import Any

from langgraph.types import interrupt

from job_writing_agent.classes import CompanyResearchData, DataLoadState

logger = logging.getLogger(__name__)


class GraphInterrupt:
    """
    Encapsulates LangGraph interrupt payload and request-input flow for human-in-the-loop.

    Builds JSON-serializable payloads (per LangGraph docs) and provides
    request_input_for_field() which calls interrupt(), validates the resume value,
    and returns a state update or {} for re-prompt. Do not wrap interrupt() in try/except.
    """

    TYPE_INPUT_REQUIRED = "input_required"

    def payload_input_required(
        self, field: str, message: str, current_value: str = ""
    ) -> dict[str, Any]:
        """
        Build JSON-serializable payload for input_required (surfaces in __interrupt__).

        Returns
        -------
        dict[str, Any]
            Payload passed to interrupt(); client receives it as chunk.data["__interrupt__"][0].value.
        """
        return {
            "type": self.TYPE_INPUT_REQUIRED,
            "field": field,
            "message": message,
            "current_value": current_value,
        }

    def request_input_for_field(
        self,
        state: DataLoadState,
        field: str,
        message: str,
        log_label: str,
    ) -> dict[str, Any]:
        """
        Interrupt the workflow for user input on a field; validate non-empty; return state update or {}.

        Builds payload, calls interrupt(payload), validates the resume value. Returns {}
        if the user provided empty input so the graph router can re-route to the prompt node.
        """
        payload = self.payload_input_required(field, message)
        value = interrupt(payload)
        if not value or not str(value).strip():
            logger.warning("User provided empty %s, routing will re-prompt", log_label)
            return {}
        value = str(value).strip()
        logger.info("User provided %s: %d characters", log_label, len(value))
        cr = state.company_research_data or CompanyResearchData()
        return {"company_research_data": cr.model_copy(update={field: value})}
