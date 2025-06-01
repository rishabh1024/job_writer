"""
LLM Client module for managing language model interactions.
"""

import os
from typing_extensions import Optional, Union


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .errors import ModelNotFoundError


class LLMClient:
    """
    Client for managing language model interactions.
    Provides a unified interface for different LLM backends.
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, model_provider: Optional[str] = None):
        """Get or create a singleton instance of the LLM client.
        
        Args:
            model_name: Optional model name to override the default
            
        Returns:
            LLMClient instance
        """
        if cls._instance is None:
            cls._instance = LLMClient(model_name, model_provider)
        elif model_name is not None and cls._instance.model_name != model_name:
            # Reinitialize if a different model is requested
            cls._instance = LLMClient(model_name)

        return cls._instance
    
    def __init__(self, model_name: Optional[str] = None, model_provider: Optional[str] = None):
        """Initialize the LLM client with the specified model.
        
        Args:
            model_name: Name of the model to use (default: from environment or "llama3.2:latest")
        """
        print("Initializing LLM Client with model:", model_name, "and provider:", model_provider)
        self.model_name = model_name or os.getenv("DEFAULT_LLM_MODEL", "llama3.2:latest")
        self.model_provider = model_provider or os.getenv("LLM_PROVIDER", "ollama").lower()
        self.llm = self._initialize_llm()
    
    def __str__(self):
        return f"LLMClient(model_name={self.model_name}, provider={self.model_provider})"
        
    def _initialize_llm(self) -> Union[BaseLLM, BaseChatModel]:
        """Initialize the appropriate LLM based on configuration.
        
        Returns:
            Initialized LLM instance
        """
        print(f"Initializing LLM with model {self.model_name} and provider {self.model_provider} in {__file__}")
        if self.model_provider == "ollama":
            return self._initialize_llama()
        elif self.model_provider == "openai":
            return self._initialize_openai()
        elif self.model_provider == "ollama_json":
            return self._initialize_jsonllm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.model_provider}")
    
    def _initialize_llama(self) -> BaseChatModel:
        """Initialize an Ollama LLM.
        
        Returns:
            Ollama LLM instance
        """
        try:
            # model = OllamaLLM(model=self.model_name, temperature=0.1, top_k=1, repeat_penalty=1.2)
            model: ChatOllama = ChatOllama(model=self.model_name, temperature=0.1, top_k=1, repeat_penalty=1.2)
            return model
        except Exception as e:
            raise ModelNotFoundError(f"Failed to initialize Ollama with model {self.model_name}: {e}") from e 


    def _initialize_jsonllm(self) -> BaseChatModel:
        """
        Initialize a Mistral chat model.
        Returns:
            Mistral chat model instance
        """
        try:
            model: ChatOllama = ChatOllama(model=self.model_name, format='json', temperature=0.1, top_k=1, repeat_penalty=1.2)
            return model
        except Exception as e:
            raise ModelNotFoundError(f"Failed to initialize Ollama with model {self.model_name}: {e}") from e 
    
    def _initialize_openai(self) -> BaseChatModel:
        """Initialize an OpenAI chat model.
        
        Returns:
            OpenAI chat model instance
        """
        api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcHAiLCJleHAiOjE3OTk5OTk5OTksInN1YiI6NjU1MDM3LCJhdWQiOiJXRUIiLCJpYXQiOjE2OTQwNzY4NTF9.hBcFcCqO1UF2Jb-m8Nv5u5zJPvQIuXUSZgyqggAD-ww"
        # api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        try:
            return ChatOpenAI(model_name=self.model_name, api_key=api_key)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to initialize Ollama with model {self.model_name}: {e}") from e
    

    def get_llm(self) -> Union[BaseLLM, BaseChatModel]:
        """Get the initialized LLM instance.
        
        Returns:
            LLM instance
        """
        if self.llm is None:
            raise RuntimeError("LLM client not initialized")
        return self.llm
    
    
    def reinitialize(self, model_name: Optional[str] = None, provider: Optional[str] = None) -> None:
        """Reinitialize the LLM with a different model or provider.
        
        Args:
            model_name: New model name to use
            provider: New provider to use
        """
        print(f"Reinitializing LLM client from {self.model_name} to {model_name}")
        if model_name:
            self.model_name = model_name
        if provider:
            self.model_provider = provider.lower()
        
        self.llm = self._initialize_llm()

