from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from job_writing_agent.utils.llm_provider_factory import LLMFactory


llm_provider = LLMFactory()
llm = llm_provider.create_langchain(
    "allenai/olmo-3.1-32b-think:free",
    provider="openrouter",
    temperature=0.1,
)


# Use PromptTemplate classes for variable interpolation
TEST_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        # Use SystemMessagePromptTemplate for SystemMessage with variables
        SystemMessagePromptTemplate.from_template(
            "You can answer any question that the user asks. If you don't know the answer, say 'I don't know' and don't make up an answer. Todays date is {current_date}.",
            input_variables=["current_date"],
        ),
        # Use AIMessagePromptTemplate for AIMessage with variables (if needed)
        # Or use AIMessage directly if no variables
        AIMessagePromptTemplate.from_template(
            "I am here to help you answer any question that you ask.",
            input_variables=["current_date"],
        ),
    ]
)

# Now the chain will work correctly
prompt_test_chain = ({"current_date": lambda x: x["current_date"]}) | TEST_PROMPT | llm

# Test it
print(TEST_PROMPT)


BULLET_POINTS_PROMPT = SystemMessagePromptTemplate.from_template(
    """You are an expert job application writer who
                                creates personalized application materials.

                                {persona_instruction}

                                Write 5-7 bullet points highlighting the candidate's
                                qualifications for this specific role.
                                Create content that genuinely reflects the candidate's
                                background and is tailored to the specific job.
                                Ensure the tone is professional, confident, and authentic.
                                Today is {current_date}.""",
    input_variables=["persona_instruction", "current_date"],
)

print(BULLET_POINTS_PROMPT)
