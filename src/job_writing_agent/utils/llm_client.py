import os
import logging

from abc import ABC, abstractmethod
from typing import Dict, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_cerebras import ChatCerebras
from pydantic import SecretStr
import dspy

logger = logging.getLogger(__name__)

__all__ = [
    "OllamaChatProvider",
    "CerebrasChatProvider",
    "OpenRouterChatProvider",
]


class LLMProvider(ABC):
    """Base class for LLM provider strategies."""

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_langchain_params(self) -> set[str]:
        pass

    @abstractmethod
    def get_dspy_params(self) -> set[str]:
        pass

    @abstractmethod
    def format_model_name_for_provider(self, model: str) -> str:
        """Convert model name to DSPy format.

        Different providers require different prefixes in DSPy.

        Args:
            model: Model name as used in LangChain

        Returns:
            Model name formatted for DSPy
        """
        pass

    @abstractmethod
    def validate_config(self, **config) -> Dict[str, Any]:
        pass

    def create_llm_instance(
        self,
        model: str,
        framework: Literal["langchain", "dspy"] = "langchain",
        **config,
    ) -> BaseChatModel | dspy.LM:
        """Create LLM instance for specified framework."""
        defaults = self.get_default_config()

        # Get framework-specific supported params
        if framework == "langchain":
            supported = self.get_langchain_params()
        else:
            supported = self.get_dspy_params()

        # Filter unsupported params
        filtered_config = {k: v for k, v in config.items() if k in supported}

        # Warn about ignored params
        ignored = set(config.keys()) - supported
        if ignored:
            logger.warning(
                f"Ignoring unsupported parameters for {framework}: {ignored}"
            )

        # Merge configs
        merged_config = {**defaults, **filtered_config}

        # Validate
        validated_config = self.validate_config(**merged_config)

        # Create instance based on framework
        if framework == "langchain":
            return self._create_langchain_instance(model, **validated_config)
        elif framework == "dspy":
            return self._create_dspy_instance(model, **validated_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @abstractmethod
    def _create_langchain_instance(self, model: str, **config) -> BaseChatModel:
        pass

    @abstractmethod
    def _create_dspy_instance(self, model: str, **config) -> dspy.LM:
        pass


class OpenRouterChatProvider(LLMProvider):
    """Provider for OpenRouter.

    Model format:
    - LangChain: "openai/gpt-4", "anthropic/claude-3-opus"
    - DSPy: Same - "openai/gpt-4", "anthropic/claude-3-opus"

    Docs: https://openrouter.ai/docs
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

    def get_default_config(self) -> Dict[str, Any]:
        return {"temperature": 0.2}

    def get_langchain_params(self) -> set[str]:
        return {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "stream",
        }

    def get_dspy_params(self) -> set[str]:
        return {"temperature", "max_tokens", "top_p", "stop", "n"}

    def format_model_name_for_provider(self, model: str) -> str:
        """OpenRouter models are used as-is in DSPy.

        Examples:
            "openai/gpt-4" -> "openai/gpt-4"
            "anthropic/claude-3-opus" -> "anthropic/claude-3-opus"
        """
        return f"{model}"  # ✅ Use as-is - already has provider/model format

    def validate_config(self, **config) -> Dict[str, Any]:
        if "temperature" in config:
            temp = config["temperature"]
            if not 0 <= temp <= 2:
                logger.warning(f"Temperature must be 0-2, got {temp}")

        if "api_key" not in config:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            config["api_key"] = api_key

        return config

    def _create_langchain_instance(self, model: str, **config) -> ChatOpenAI:
        """Create LangChain instance.

        Example model: "openai/gpt-4"
        """
        api_key = config.pop("api_key")

        return ChatOpenAI(
            model=self.format_model_name_for_provider(
                model
            ),  # ✅ Use model as-is: "openai/gpt-4"
            api_key=SecretStr(api_key),
            base_url=self.OPENROUTER_API_URL,
            **config,
        )

    def _create_dspy_instance(self, model: str, **config) -> dspy.LM:
        """Create DSPy instance.

        Example model: "openai/gpt-4"
        """
        api_key = config.pop("api_key")

        return dspy.LM(
            model=f"openrouter/{self.format_model_name_for_provider(model)}",  # ✅ Use as-is: "openai/gpt-4"
            api_key=api_key,
            api_base=self.OPENROUTER_API_URL,
            **config,
        )


class CerebrasChatProvider(LLMProvider):
    """Provider for Cerebras.

    Model format:
    - LangChain: "llama3.1-8b", "llama3.1-70b" (direct names)
    - DSPy: "openai/llama3.1-8b" (needs openai/ prefix for compatibility)

    Docs: https://inference-docs.cerebras.ai/
    """

    CEREBRAS_API_URL = "https://api.cerebras.ai/v1"

    def get_default_config(self) -> Dict[str, Any]:
        return {"temperature": 0.2, "max_tokens": 1024}

    def get_langchain_params(self) -> set[str]:
        return {"temperature", "max_tokens", "top_p", "stop", "stream", "seed"}

    def get_dspy_params(self) -> set[str]:
        return {"temperature", "max_tokens", "top_p", "stop"}

    def format_model_name_for_provider(self, model: str) -> str:
        """Cerebras models need 'cerebras/' prefix.

        Examples:
            "llama3.1-8b" -> "cerebras/llama3.1-8b"
            "llama3.1-70b" -> "cerebras/llama3.1-70b"
        """
        return f"cerebras/{model}"  # ✅ Add openai/ prefix for OpenAI-compatible API

    def validate_config(self, **config) -> Dict[str, Any]:
        if "temperature" in config:
            temp = config["temperature"]
            if not 0 <= temp <= 1.5:
                raise ValueError(f"Temperature must be 0-1.5, got {temp}")

        if "api_key" not in config:
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY not set")
            config["api_key"] = api_key

        return config

    def _create_langchain_instance(self, model: str, **config) -> ChatCerebras:
        """Create LangChain instance.

        Example model: "llama3.1-8b"
        """

        return ChatCerebras(
            model=model,  # Direct name: "llama3.1-8b"
            **config,
        )

    @DeprecationWarning
    def _create_langchain_instance_openaiclient(
        self, model: str, **config
    ) -> ChatOpenAI:
        """
        Create LangChain instance
        Example model: "llama3.1-8b"
        """

        api_key = config.pop("api_key")

        return ChatOpenAI(
            model=self.format_model_name_for_provider(
                model
            ),  # Direct name: "llama3.1-8b"
            api_key=SecretStr(api_key),
            base_url=self.CEREBRAS_API_URL,
            **config,
        )

    def _create_dspy_instance(self, model: str, **config) -> dspy.LM:
        """Create DSPy instance.

        Example model input: "llama3.1-8b"
        DSPy format: "openai/llama3.1-8b"
        """
        api_key = config.pop("api_key")

        return dspy.LM(
            model=self.format_model_name_for_provider(
                model
            ),  # With prefix: "openai/llama3.1-8b"
            api_key=api_key,
            api_base=self.CEREBRAS_API_URL,
            **config,
        )


class OllamaChatProvider(LLMProvider):
    """Provider for Ollama.

    Model format:
    - LangChain: "llama3.2", "llama3.2:latest" (direct names with optional tags)
    - DSPy: "ollama_chat/llama3.2" (needs ollama_chat/ prefix)

    Docs: https://ollama.com/
    """

    def get_default_config(self) -> Dict[str, Any]:
        return {"temperature": 0.2, "top_k": 40, "top_p": 0.9}

    def get_langchain_params(self) -> set[str]:
        return {
            "temperature",
            "top_k",
            "top_p",
            "repeat_penalty",
            "num_ctx",
            "num_predict",
            "format",
            "seed",
        }

    def get_dspy_params(self) -> set[str]:
        return {"temperature", "top_p", "num_ctx", "seed"}

    def format_model_name_for_provider(self, model: str) -> str:
        """Ollama models need 'ollama_chat/' prefix for DSPy.

        Examples:
            "llama3.2" -> "ollama_chat/llama3.2"
            "llama3.2:latest" -> "ollama_chat/llama3.2:latest"
        """
        return f"ollama_chat/{model}"  # ✅ Add ollama_chat/ prefix

    def validate_config(self, **config) -> Dict[str, Any]:
        if "temperature" in config:
            temp = config["temperature"]
            if not 0 <= temp <= 2:
                raise ValueError(f"Temperature must be 0-2, got {temp}")

        if "top_k" in config:
            if not isinstance(config["top_k"], int) or config["top_k"] < 1:
                raise ValueError("top_k must be positive integer")

        return config

    def _create_langchain_instance(self, model: str, **config) -> ChatOllama:
        return ChatOllama(model=self.format_model_name_for_provider(model), **config)

    def _create_dspy_instance(self, model: str, **config) -> dspy.LM:
        return dspy.LM(
            model=self.format_model_name_for_provider(
                model
            ),  # ✅ With prefix: "ollama_chat/llama3.2"
            **config,
        )
