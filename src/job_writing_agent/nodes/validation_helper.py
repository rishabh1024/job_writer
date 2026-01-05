# -*- coding: utf-8 -*-
"""
Validation Helper Module

This module provides the ValidationHelper class responsible for validating
workflow inputs and setting routing decisions. It follows the Single
Responsibility Principle by focusing solely on input validation.
"""

import logging

from job_writing_agent.classes import DataLoadState
from job_writing_agent.utils.logging.logging_decorators import (
    log_execution,
    log_errors,
)

logger = logging.getLogger(__name__)


class ValidationHelper:
    """
    Responsible for validating workflow inputs and setting routing decisions.


    Example:
        >>> validator = ValidationHelper()
        >>> validated_state = validator.verify_inputs(state)
        >>> next_node = validated_state.get("next_node")  # "load" or "research"
    """

    def __init__(self):
        """
        Initialize ValidationHelper.

        This class is stateless - no dependencies needed for validation logic.
        """
        pass

    @log_execution
    @log_errors
    def verify_inputs(self, state: DataLoadState) -> DataLoadState:
        """
        Validate inputs and set next_node for routing.

        This method validates that both resume and job description are present
        and non-empty in the state.
        Parameters
        ----------
        state: DataLoadState
            Current workflow state containing company_research_data.

        Returns
        -------
        DataLoadState
            Updated state with next_node set to "load" (if validation fails)
            or "research" (if validation passes).
        """
        logger.info("Verifying loaded inputs!")
        state["current_node"] = "verify"

        # Validate required fields using helper methods
        if not self._validate_resume(state):
            logger.error("Resume is missing or empty in company_research_data")
            state["next_node"] = "load"  # Loop back to load subgraph
            return state

        if not self._validate_job_description(state):
            logger.error("Job description is missing or empty in company_research_data")
            state["next_node"] = "load"  # Loop back to load subgraph
            return state

        # All validations passed
        state["next_node"] = "research"
        logger.info("Inputs verified successfully, proceeding to research")
        return state

    def _validate_resume(self, state: DataLoadState) -> bool:
        """
        Validate that resume is present and non-empty in company_research_data.

        Private helper method for better code organization.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state.

        Returns
        -------
        bool
            True if resume is present and non-empty, False otherwise.
        """
        company_research_data = state.get("company_research_data", {})
        resume = company_research_data.get("resume", "")
        # Handle various types: convert to string and check if non-empty
        if not resume:
            return False
        resume_str = str(resume).strip()
        return bool(resume_str)

    def _validate_job_description(self, state: DataLoadState) -> bool:
        """
        Validate that job description is present and non-empty in company_research_data.

        Private helper method for better code organization.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state.

        Returns
        -------
        bool
            True if job description is present and non-empty, False otherwise.
        """
        company_research_data = state.get("company_research_data", {})
        job_description = company_research_data.get("job_description", "")
        # Handle various types: convert to string and check if non-empty
        if not job_description:
            return False
        job_desc_str = str(job_description).strip()
        return bool(job_desc_str)
