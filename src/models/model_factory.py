from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.schema import BaseMessage
from langchain_core.pydantic_v1 import SecretStr

from src.models.base.base_model import ChainGenerator, ContentGenerator, EvaluationChatModel
from src.models.evaluator.img_evaluator import EvaluationChatModelImg
from src.models.evaluator.text_evaluator import EvaluationChatModelQA
from src.models.gen.img_generator import ImgGenerator
from src.models.gen.text_generator import QuestionGenerator


class ModelFactory(ABC):

    @abstractmethod
    def create_model(self, *args: Any, **kwargs: Any) -> ChainGenerator:
        """
        An abstract method to create a model, with the return types EvaluationChatModel or ChainGenerator.
        """


class EvaluationChatModelFactory(ModelFactory):

    def create_model(
        self,
        model_class: str,
        openai_api_key: SecretStr,
        **kwargs: Any,
    ) -> EvaluationChatModel:
        """
        Create a model based on the provided model class and OpenAI API key.

        Args:
            model_class (str): The type of model to create.
            openai_api_key (SecretStr): The API key for OpenAI.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            EvaluationChatModel: The created evaluation chat model.

        Raises:
            ValueError: If an invalid model class is provided.
        """
        match model_class:
            case "qa":
                return EvaluationChatModelQA(
                    openai_api_key=openai_api_key,
                    **kwargs,
                )
            case "img_desc":
                return EvaluationChatModelImg(
                    openai_api_key=openai_api_key,
                    **kwargs,
                )
            case _:
                raise ValueError("Invalid model class provided")


class GeneratorModelFactory(ModelFactory):

    def create_model(
        self,
        model_class: str,
        openai_api_key: SecretStr,
        history_chat: Optional[List[BaseMessage]] = None,
        n_messages_memory: int = 20,
        prompt_model: str = "gpt-3.5-turbo",
        prompt_model_temperature: float = 0.3,
        img_size: str = "256x256",
        **kwargs: Any,
    ) -> ContentGenerator:
        """
        Generate a model based on the specified model class and parameters.

        Parameters:
            model_class (str): The class of the model to create.
            openai_api_key (SecretStr): The API key for OpenAI.
            history_chat (Optional[list], optional): List of chat history. Defaults to None.
            n_messages_memory (int, optional): Number of messages to keep in memory. Defaults to 20.
            prompt_model (str, optional): The prompt model. Defaults to "gpt-3.5-turbo".
            prompt_model_temperature (float, optional): The prompt model temperature. Defaults to 0.3.
            img_size (str, optional): The size of the image. Defaults to "256x256".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ContentGenerator: A generator for the specified model class.
        """
        match model_class:
            case "qa":
                return QuestionGenerator(
                    openai_api_key=openai_api_key,
                    history_chat=history_chat or [],
                    n_messages_memory=n_messages_memory,
                    **kwargs,
                )
            case "img_desc":
                return ImgGenerator(
                    openai_api_key=openai_api_key,
                    prompt_model=prompt_model,
                    prompt_model_temperature=prompt_model_temperature,
                    img_size=img_size,
                    **kwargs,
                )
            case _:
                raise ValueError("Invalid model class provided")
