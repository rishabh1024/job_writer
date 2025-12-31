from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompt_values import PromptValue
import dspy
import os
import mlflow

mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=True
    )

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("job description extract")


OPENROUTER_API_KEY="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"

dspy.configure(lm=dspy.LM(
                            "openrouter/qwen/qwen3-4b:free",
                            api_key=OPENROUTER_API_KEY,
                            temperature=0.1
                            ))



class TavilySearchQueries(dspy.Signature):
    """Use the job description and company name 
    to create search queries for the tavily search tool"""
    job_description_= dspy.InputField(desc="Job description of the role that candidate is applying for.")
    company_name = dspy.InputField(desc="Name of the company the candidate is applying for.")
    search_queries = dspy.OutputField(desc="Tavily Search Query")
    search_query_relevance = dspy.OutputField(desc="Relevance for each tavily search query that is generated")


tavily_query_generator = dspy.ChainOfThought(TavilySearchQueries)

job_description = """ Who are we?

Our mission is to scale intelligence to serve humanity. We’re training and deploying frontier models for developers and enterprises who are building AI systems to power magical experiences like content generation, semantic search, RAG, and agents. We believe that our work is instrumental to the widespread adoption of AI.

We obsess over what we build. Each one of us is responsible for contributing to increasing the capabilities of our models and the value they drive for our customers. We like to work hard and move fast to do what’s best for our customers.

Cohere is a team of researchers, engineers, designers, and more, who are passionate about their craft. Each person is one of the best in the world at what they do. We believe that a diverse range of perspectives is a requirement for building great products.

Join us on our mission and shape the future!

About North:

North is Cohere's cutting-edge AI workspace platform, designed to revolutionize the way enterprises utilize AI. It offers a secure and customizable environment, allowing companies to deploy AI while maintaining control over sensitive data. North integrates seamlessly with existing workflows, providing a trusted platform that connects AI agents with workplace tools and applications.

As a Senior/Staff Backend Engineer, you will:
Build and ship features for North, our AI workspace platform
Develop autonomous agents that talk to sensitive enterprise data
uns in low-resource environments, and has highly stringent deployment mechanisms
As security and privacy are paramount, you will sometimes need to re-invent the
 wheel, and won’t be able to use the most popular libraries or tooling
Collaborate with researchers to productionize state-of-the-art models and techniques
You may be a good fit if:
Have shipped (lots of) Python in production
You have built and deployed extremely performant client-side or server-side RAG/agentic 
applications to millions of users You have strong coding abilities and are comfortable working across the stack. You’re able to read and understand, and even fix issues outside of the main code base
You’ve worked in both large enterprises and startups
You excel in fast-paced environments and can execute while priorities and objectives are a moving target
If some of the above doesn’t line up perfectly with your experience, we still encourage you to apply! 
If you want to work really hard on a glorious mission with teammates that want the same thing, Cohere is the place for you."""

response = tavily_query_generator(job_description_=job_description, company_name="Cohere")

print(response)