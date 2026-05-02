"""Shared constants for the AgentQL job-description scraper.

Keeping ``ExtractionMethod`` and the query string in a dedicated module
prevents the circular import that would otherwise arise between
``agentql_job_scraper`` and the ``strategies`` sub-package:

    agentql_job_scraper -> strategies -> agentql_job_scraper  (cycle)

Both sides import from this module instead.
"""

from __future__ import annotations

from enum import StrEnum


class ExtractionMethod(StrEnum):
    """Supported extraction strategies.

    Attributes:
        AQL_WITH_CONTEXT: AQL query enriched with semantic context
            descriptions on every field and structural nesting for the main
            description section.
    """

    AQL_WITH_CONTEXT = "aql_with_context"


# ---------------------------------------------------------------------------
# Context-enriched AQL query. This is the only active extraction query.
# Applies both AgentQL best practices:
#   1. Semantic context — parentheses descriptions on every field
#   2. Structural context — list fields nested under job_description_section
# Reference: https://docs.agentql.com/agentql-query/best-practices
# ---------------------------------------------------------------------------
JOB_DESCRIPTION_QUERY_WITH_CONTEXT: str = """
{
    body[] {
    job_title(the h1 or prominent heading that names the open role)
    company_name(the name of the hiring organisation or employer)
    job_location(office city, region, country or remote label for the role)
    job_description_section(the main body section of the job posting) {
        job_summary(introductory paragraph or overview of the role)
        responsibilities(list of duties and day-to-day tasks for the role)[]
        requirements(mandatory qualifications, skills or experience needed)[]
        preferred_qualifications(nice-to-have or bonus qualifications)[]
        benefits(perks, compensation extras or employee benefits listed)[]
    }
    }
}
"""
