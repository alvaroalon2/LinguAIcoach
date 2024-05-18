from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, SecretStr
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI


class ChainGenerator(ABC):

    def __init__(
        self,
        openai_api_key: SecretStr,
        chat_temperature: float = 0.3,
        chat_model: str = "gpt-3.5-turbo",
    ) -> None:
        """
        Initializes the class with the given parameters.

        Args:
            openai_api_key (SecretStr): The API key for OpenAI.
            chat_temperature (float, optional): The temperature to use for chat. Defaults to 0.3.
            chat_model (str, optional): The model to use for chat. Defaults to "gpt-3.5-turbo".

        Returns:
            None

        """
        self.openai_api_key = openai_api_key
        self.chat_temperature = chat_temperature

        self._initialize_chat_llm(chat_model)

    @abstractmethod
    def _get_system_prompt(self) -> ChatPromptTemplate:
        """Returns the system prompt for the exam.

        Returns:
            ChatPromptTemplate: System prompt.
        """

    @abstractmethod
    def _create_chain(self) -> Runnable[Any, Any]:
        """Creates the chain.

        Returns:
            RunnableSequence: Chain.
        """

    def _initialize_chat_llm(self, chat_model: str) -> None:
        """Initializes the ChatOpenAI language model."""
        self.chat_llm = ChatOpenAI(
            api_key=self.openai_api_key,
            temperature=self.chat_temperature,
            model=chat_model,
        )


class EvaluationChatModel(ChainGenerator):
    """Abstract base class for evaluation chat models."""

    def __init__(self, level: str, openai_api_key: SecretStr, chat_temperature: float) -> None:
        """Initialize the evaluation chat model.

        Args:
            level (str): Level of the exam.
            openai_api_key (SecretStr): OpenAI API key.
            chat_temperature (float): Temperature for the chat model.

        Returns:
            None
        """
        super().__init__(openai_api_key=openai_api_key, chat_temperature=chat_temperature)
        self.level = level

    @abstractmethod
    def _get_output_parser(self, pydantic_schema: Type[BaseModel]) -> PydanticOutputParser[Any]:
        """Get the output parser for the model.

        Args:
            pydantic_schema (BaseModel): The output schema of the model.

        Returns:
            PydanticOutputParser: The output parser for the model.
        """

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """
        Defines how the chain should be called in the predict method.

        Returns:
            Dict: The return value of the predict method.
        """


class ContentGenerator(ChainGenerator):
    """Abstract base class for generating content"""

    @abstractmethod
    def generate(self) -> str:
        """
        Defines how the chain should be called in the generate method.

        Returns:
            str: The return value of the generate method.
        """
