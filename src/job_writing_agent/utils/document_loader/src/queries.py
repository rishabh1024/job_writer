"""AQL query variants evaluated by the scraper experiment."""

from __future__ import annotations

FLAT_WITH_CONTEXT: str = """
{
    body[] {
    job_title(the h1 or prominent heading that names the open role)
    company_name(the name of the hiring organisation or employer)
    job_location(office city, region, country or remote label for the role)
    job_summary(introductory paragraph or overview of the role)
    responsibilities(list of duties and day-to-day tasks for the role)[]
    requirements(mandatory qualifications, skills or experience needed)[]
    preferred_qualifications(nice-to-have or bonus qualifications)[]
    benefits(perks, compensation extras or employee benefits listed)[]
    }
}
"""

FLAT_BARE: str = """
{
    body[] {
    job_title
    company_name
    job_location
    job_summary
    responsibilities
    requirements
    preferred_qualifications
    benefits
    }
}
"""

HEADING_BODY: str = """
    {
        Job_Posting_Description[]{
            Heading()
            Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
            }
    }
    """

NESTED_WITH_CONTEXT: str = """
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

STRATEGIES: dict[str, str] = {
    "flat_with_context": FLAT_WITH_CONTEXT,
    "flat_bare": FLAT_BARE,
    "heading_body": HEADING_BODY,
    "nested_with_context": NESTED_WITH_CONTEXT,
}
