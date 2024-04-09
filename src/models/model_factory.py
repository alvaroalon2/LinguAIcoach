from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain.schema import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import SecretStr

from src.models.generator import ImgGenerator, QuestionGenerator
from src.models.lc_base_model import ChainGenerator, ContentGenerator, EvaluationChatModel
from src.models.lc_img_desc_model import EvaluationChatModelImg
from src.models.lc_qa_model import EvaluationChatModelQA


class ModelFactory(ABC):

    @abstractmethod
    def create_model(self, *args: Any, **kwargs: Any) -> ChainGenerator:
        """
        An abstract method to create a model, with the return types EvaluationChatModel or ChainGenerator.
        """


class EvaluationChatModelFactory(ModelFactory):

    def create_model(self, model_class: str, openai_api_key: SecretStr, **kwargs: Any) -> EvaluationChatModel:
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
                return EvaluationChatModelQA(openai_api_key=openai_api_key, **kwargs)
            case "img_desc":
                return EvaluationChatModelImg(openai_api_key=openai_api_key, **kwargs)
            case _:
                raise ValueError("Invalid model class provided")


class GeneratorModelFactory(ModelFactory):

    def create_model(
        self,
        model_class: str,
        openai_api_key: SecretStr,
        history_chat: Optional[List[HumanMessage | AIMessage]] = None,
        img_size: str = "256x256",
        **kwargs: Any,
    ) -> ContentGenerator:
        """
        Generate a model based on the specified model class and parameters.

        Parameters:
            model_class (str): The class of the model to create.
            openai_api_key (SecretStr): The API key for OpenAI.
            history_chat (Optional[list], optional): List of chat history. Defaults to None.
            img_size (str, optional): The size of the image. Defaults to "256x256".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ContentGenerator: A generator for the specified model class.
        """
        match model_class:
            case "qa":
                return QuestionGenerator(openai_api_key=openai_api_key, history_chat=history_chat or [], **kwargs)
            case "img_desc":
                return ImgGenerator(openai_api_key=openai_api_key, img_size=img_size, **kwargs)
            case _:
                raise ValueError("Invalid model class provided")
