import dspy
import os
import asyncio
import mlflow
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, AsyncChromiumLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
import trafilatura


os.environ['CEREBRAS_API_KEY'] = "csk-m28t6w8vk6pjn3rdrtwtdjkynjh5hxfe29dtx2hnjedft9he"


mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=True
    )

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("job description extract")


class ExtractJobDescription(dspy.Signature):
    """Clean and extract the job description from the provided scraped HTML of the job posting.
    Divide the job description into multiple sections under different headings.Company Overview,
    Role Introduction,Qualifications and Requirements, Prefrred Qualifications, Salary, Location.
    Do not alter the content of the job description.
    """
    job_description_html_content = dspy.InputField(desc="HTML content of the job posting.")
    job_description = dspy.OutputField(desc="Clean job description which is free of HTML tags and irrelevant information.")
    job_role = dspy.OutputField(desc="The job role in the posting.")
    company_name = dspy.OutputField(desc="Company Name of the Job listing.")
    location = dspy.OutputField(desc="The location for the provided job posting.")


def get_job_description(url: str):
    loader = WebBaseLoader(url)
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    document_splitted = loader.load_and_split(text_splitter=text_splitter)

    extracted_text = " ".join(doc.page_content for doc in document_splitted)
    return extracted_text


async def scrape_with_playwright(urls):
    loader = AsyncChromiumLoader(urls, headless=True, user_agent="Mozilla/5.0 (compatible)")
    docs = await loader.aload()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    # print(f"Docs transformed: {docs_transformed}")
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    extracted_content = splitter.split_documents(docs_transformed)

    return extracted_content


urls = ["https://jobs.ashbyhq.com/MotherDuck/c11f6d31-64e9-4964-85dd-c5b25eee55bc"]
asyncio.run(scrape_with_playwright(urls))
# print(extracted_content)