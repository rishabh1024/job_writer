"""AgentQL query strings for job listing pages (Workday, Lever, Greenhouse, fallback)."""

WORKDAYJOB_AQL_QUERY = """
                            {
                                Job_Posting_Description[]{
                                    Heading()
                                    Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
                                    }
                            }
                        """

LEVER_AQL_QUERY = """
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

GREENHOUSEJOB_AQL_QUERY = """
                            {
                                Job_Posting_Description[]{
                                    Heading()
                                    Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
                                    }
                            }
                        """

DEFAULT_AQL_QUERY = """
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
