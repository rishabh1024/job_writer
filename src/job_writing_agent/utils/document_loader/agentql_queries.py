"""AgentQL query strings for job listing pages (Workday, Lever, Greenhouse, fallback)."""

WORKDAYJOB_AQL_QUERY = """
                            {
                                Job_Posting_Description[]{
                                    Heading()
                                    Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
                                    }
                                Company_Name(Only the name of the company that has posted this role.)
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
                                Company_Name(Only the name of the company that has posted this role.)
                            }
                        """

DEFAULT_AQL_QUERY = """
                            {
                                Job_Posting_Description[]{
                                    Heading()
                                    Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
                                    }
                                Company_Name(Only the name of the company that has posted this role.)
                            }
                        """
