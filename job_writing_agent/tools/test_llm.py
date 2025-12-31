from job_writing_agent.utils.llm_provider_factory import LLMFactory
from langchain_cerebras import ChatCerebras

llm_provider = LLMFactory()

llm_cerebras =  ChatCerebras(
            model="llama3.1-8b",  # Direct name: "llama3.1-8b"
            temperature=0.3
        )
print(llm_cerebras.invoke("Hey! Can you hear me?"))