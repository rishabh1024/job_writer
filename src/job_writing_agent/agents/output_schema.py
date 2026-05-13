from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import dspy


class TavilyQuerySet(BaseModel):
    query1: Optional[List[str]] = Field(
        default=None,
        description="First search query and its rationale, e.g., ['query text']",
    )
    query2: Optional[List[str]] = Field(
        default=None, description="Second search query and its rationale"
    )
    query3: Optional[List[str]] = Field(
        default=None, description="Third search query and its rationale"
    )
    query4: Optional[List[str]] = Field(
        default=None, description="Fourth search query and its rationale"
    )
    query5: Optional[List[str]] = Field(
        default=None, description="Fifth search query and its rationale"
    )

    @field_validator("query1", "query2", "query3", "query4", "query5", mode="after")
    @classmethod
    def ensure_len_two(cls, v):
        """Ensure each provided query list contains exactly one strings: [query]."""
        if v is not None:  # Only validate if the list is actually provided
            if len(v) != 1:
                # Updated error message for clarity
                raise ValueError(
                    "Each query list, when provided, must contain exactly one string: the query text."
                )
        return v


class TavilySearchQueries(dspy.Signature):
    """Use the job description and company name
    to create exactly 5 search queries for the tavily search tool in JSON Format"""

    job_description = dspy.InputField(
        desc="Job description of the role that candidate is applying for."
    )
    company_name = dspy.InputField(
        desc="Name of the company the candidate is applying for."
    )
    search_queries = dspy.OutputField(
        desc="Dictionary of tavily search queries which will gather understanding of the company and it's culture",
        json=True,
    )
    search_query_relevance = dspy.OutputField(
        desc="Dictionary of relevance for each tavily search query that is generated",
        json=True,
    )


class CompanyResearchDataSummarizationSchema(dspy.Signature):
    """This schema is used to summarize the company research data into a concise summary to provide a clear understanding of the company."""

    company_research_data = dspy.InputField(
        desc="These are the results of the tavily search queries that were generated. They have been filtered for relevance and are now ready to be summarized."
    )
    company_research_data_summary = dspy.OutputField(
        desc="This is summary of the company research data that will be used by a job application writer to assist the candidate in writing content supporting the job application. The summary should be relevant to the job application and the company.",
    )


class CandidateJobFitAnalysis(dspy.Signature):
    """Analyze how a candidate's resume aligns with a job description.

    Identify matching qualifications, transferable skills, gaps, and key talking
    points for application materials (cover letter, resume bullets, LinkedIn note).
    """

    resume_text = dspy.InputField(
        desc="Full text of the candidate's resume including experience, skills, and education."
    )
    job_description = dspy.InputField(
        desc="Full text of the job posting including requirements, responsibilities, and qualifications."
    )
    company_name = dspy.InputField(desc="Name of the company hiring for this role.")

    # Outputs - structured for downstream content generation
    matching_qualifications = dspy.OutputField(
        desc="List of candidate qualifications that directly match job requirements."
    )
    transferable_skills = dspy.OutputField(
        desc="Skills from the resume that transfer to this role even if not explicitly required."
    )
    experience_highlights = dspy.OutputField(
        desc="2-3 specific experiences from the resume most relevant to this role with brief context."
    )
    potential_gaps = dspy.OutputField(
        desc="Requirements from the job description the resume doesn't clearly address."
    )
    unique_value_proposition = dspy.OutputField(
        desc="One sentence describing what makes this candidate stand out for this specific role."
    )
    talking_points = dspy.OutputField(
        desc="3-5 key points to emphasize in cover letter or interview, formatted as a list."
    )
