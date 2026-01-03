import logging
from typing import Any, Dict, Literal

import dspy
from langchain_core.language_models.chat_models import BaseChatModel

from .llm_client import (
    CerebrasChatProvider,
    LLMProvider,
    OllamaChatProvider,
    OpenRouterChatProvider,
)
from .logging.logging_decorators import log_execution

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances supporting multiple frameworks and providers.

    Supports both LangChain and DSPy frameworks with automatic model name formatting
    for each provider.

    Example:
        >>> factory = LLMFactory()
        >>>
        >>> # LangChain usage
        >>> llm = factory.create_langchain("llama3.1-8b", provider="cerebras")
        >>> response = llm.invoke("Hello!")
        >>>
        >>> # DSPy usage
        >>> lm = factory.create_dspy("llama3.1-8b", provider="cerebras")
        >>> dspy.configure(lm=lm)
    """

    @log_execution
    def __init__(self, default_provider: str = "openrouter"):
        """Initialize factory with available providers.

        Args:
            default_provider: Default provider to use if not specified
        """
        self._providers: Dict[str, LLMProvider] = {
            "ollama": OllamaChatProvider(),
            "openrouter": OpenRouterChatProvider(),
            "cerebras": CerebrasChatProvider(),
        }
        self._default_provider = default_provider

        logger.info(
            f"LLMFactory initialized with providers: {list(self._providers.keys())}, "
            f"default: {default_provider}"
        )

    @log_execution
    def create(
        self,
        model: str,
        provider: str | None = None,
        framework: Literal["langchain", "dspy"] = "langchain",
        **config,
    ) -> BaseChatModel | dspy.LM:
        """Create LLM instance for specified framework and provider.

        Args:
            model: Model name/identifier (format depends on provider)
            provider: Provider name ('ollama', 'openrouter', 'cerebras',)
                     Defaults to factory's default_provider
            framework: 'langchain' or 'dspy' (default: 'langchain')
            **config: Additional configuration parameters (temperature, max_tokens, etc.)

        Returns:
            LangChain BaseChatModel or DSPy LM instance

        Raises:
            ValueError: If provider is unknown

        Examples:
            >>> factory = LLMFactory()
            >>>
            >>> # Create LangChain LLM
            >>> llm = factory.create(
            ...     "llama3.1-8b",
            ...     provider="cerebras",
            ...     framework="langchain",
            ...     temperature=0.7
            ... )
            >>>
            >>> # Create DSPy LM
            >>> lm = factory.create(
            ...     "openai/gpt-4",
            ...     provider="openrouter",
            ...     framework="dspy",
            ...     temperature=0.5
            ... )
        """
        provider = provider or self._default_provider

        if provider not in self._providers:
            available = ", ".join(self._providers.keys())
            logger.warning(
                f"Invalid provider '{provider}'. Available providers: {available}. "
                f"Falling back to default: {self._default_provider}"
            )
            provider = self._default_provider

        strategy = self._providers[provider]
        logger.debug(
            f"Creating {framework} LLM: provider={provider}, model={model}, config={config}"
        )

        return strategy.create_llm_instance(model, framework=framework, **config)

    def create_langchain(
        self, model: str, provider: str | None = None, **config
    ) -> BaseChatModel:
        """Convenience method to create LangChain LLM.

        Args:
            model: Model name/identifier
            provider: Provider name (defaults to factory default)
            **config: Configuration parameters

        Returns:
            LangChain BaseChatModel instance

        Example:
            >>> factory = LLMFactory()
            >>> llm = factory.create_langchain(
            ...     "llama3.1-8b",
            ...     provider="cerebras",
            ...     temperature=0.7,
            ...     max_tokens=2048
            ... )
            >>> response = llm.invoke("Explain quantum computing")
            >>> print(response.content)
        """
        return self.create(model, provider, framework="langchain", **config)

    def create_dspy(self, model: str, provider: str | None = None, **config) -> dspy.LM:
        """Convenience method to create DSPy LM.

        Args:
            model: Model name/identifier
            provider: Provider name (defaults to factory default)
            **config: Configuration parameters

        Returns:
            DSPy LM instance

        Example:
            >>> import dspy
            >>> factory = LLMFactory()
            >>>
            >>> lm = factory.create_dspy(
            ...     "llama3.1-8b",
            ...     provider="cerebras",
            ...     temperature=0.5
            ... )
            >>>
            >>> # Set as default LM for DSPy
            >>> dspy.configure(lm=lm)
            >>>
            >>> # Use in DSPy programs
            >>> class QA(dspy.Signature):
            ...     question = dspy.InputField()
            ...     answer = dspy.OutputField()
            >>>
            >>> qa = dspy.ChainOfThought(QA)
            >>> result = qa(question="What is AI?")
        """
        return self.create(model, provider, framework="dspy", **config)

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register a custom provider.

        Allows extending the factory with custom provider implementations.

        Args:
            name: Unique identifier for the provider
            provider: LLMProvider instance

        Example:
            >>> from abc import ABC
            >>>
            >>> class CustomProvider(LLMProvider):
            ...     def get_default_config(self):
            ...         return {'temperature': 0.1}
            ...
            ...     def get_langchain_params(self):
            ...         return {'temperature', 'max_tokens'}
            ...
            ...     def get_dspy_params(self):
            ...         return {'temperature'}
            ...
            ...     def format_model_name_for_dspy(self, model):
            ...         return f"custom/{model}"
            ...
            ...     def validate_config(self, **config):
            ...         return config
            ...
            ...     def _create_langchain_instance(self, model, **config):
            ...         return MyCustomChatModel(model=model, **config)
            ...
            ...     def _create_dspy_instance(self, model, **config):
            ...         return dspy.LM(model=f"custom/{model}", **config)
            >>>
            >>> factory = LLMFactory()
            >>> factory.register_provider('custom', CustomProvider())
            >>> llm = factory.create_langchain("my-model", provider="custom")
        """
        if name in self._providers:
            logger.warning(f"Overwriting existing provider: {name}")

        self._providers[name] = provider
        logger.info(f"Registered provider: {name}")

    def unregister_provider(self, name: str) -> None:
        """Remove a provider from the factory.

        Args:
            name: Provider name to remove

        Raises:
            ValueError: If provider doesn't exist or is the default provider
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")

        if name == self._default_provider:
            raise ValueError(f"Cannot unregister default provider '{name}'")

        del self._providers[name]
        logger.info(f"Unregistered provider: {name}")

    def list_providers(self) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of registered provider names

        Example:
            >>> factory = LLMFactory()
            >>> providers = factory.list_providers()
            >>> print(providers)
            ['ollama', 'openrouter', 'cerebras', 'groq']
        """
        return list(self._providers.keys())

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get detailed information about a provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary containing provider configuration details

        Raises:
            ValueError: If provider is unknown

        Example:
            >>> factory = LLMFactory()
            >>> info = factory.get_provider_info("cerebras")
            >>> print(info)
            {
                'name': 'cerebras',
                'default_config': {'temperature': 0.2, 'max_tokens': 1024},
                'langchain_params': ['temperature', 'max_tokens', ...],
                'dspy_params': ['temperature', 'max_tokens', ...]
            }
        """
        if provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider}'. Available providers: {available}"
            )

        strategy = self._providers[provider]
        return {
            "name": provider,
            "default_config": strategy.get_default_config(),
            "langchain_params": list(strategy.get_langchain_params()),
            "dspy_params": list(strategy.get_dspy_params()),
        }

    def get_all_providers_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered providers.

        Returns:
            Dictionary mapping provider names to their info

        Example:
            >>> factory = LLMFactory()
            >>> all_info = factory.get_all_providers_info()
            >>> for provider, info in all_info.items():
            ...     print(f"{provider}: {info['default_config']}")
        """
        return {name: self.get_provider_info(name) for name in self._providers.keys()}

    def set_default_provider(self, provider: str) -> None:
        """Change the default provider.

        Args:
            provider: Provider name to set as default

        Raises:
            ValueError: If provider is unknown

        Example:
            >>> factory = LLMFactory()
            >>> factory.set_default_provider('cerebras')
            >>> llm = factory.create_langchain("llama3.1-8b")  # Uses cerebras
        """
        if provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Cannot set unknown provider '{provider}' as default. "
                f"Available: {available}"
            )

        old_default = self._default_provider
        self._default_provider = provider
        logger.info(f"Changed default provider from '{old_default}' to '{provider}'")

    def get_default_provider(self) -> str:
        """Get the current default provider name.

        Returns:
            Name of the default provider
        """
        return self._default_provider

    def __repr__(self) -> str:
        """String representation of the factory."""
        return (
            f"LLMFactory("
            f"providers={list(self._providers.keys())}, "
            f"default='{self._default_provider}'"
            f")"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return (
            f"LLMFactory with {len(self._providers)} providers: "
            f"{', '.join(self._providers.keys())} "
            f"(default: {self._default_provider})"
        )
