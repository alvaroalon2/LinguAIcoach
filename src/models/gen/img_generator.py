from typing import Any

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from src.models.base.base_model import ContentGenerator


class ImgGenerator(ContentGenerator):

    def __init__(  # noqa: PLR0913
        self,
        exam_prompt: str,
        level: str,
        description: str,
        openai_api_key: SecretStr,
        chat_model: str = "gpt-3.5-turbo",
        prompt_model: str = "gpt-3.5-turbo",
        chat_temperature: float = 0.3,
        prompt_model_temperature: float = 0.3,
        img_size: str = "256x256",
    ) -> None:
        """
        Initializes the class with the exam prompt, level, description, OpenAI API key, and optional image size.

        Parameters:
            exam_prompt (str): The prompt for the exam.
            level (str): The level of the exam.
            description (str): Description of the exam.
            openai_api_key (SecretStr): The OpenAI API key.
            chat_model (str, optional): The model to use for chat. Defaults to "gpt-3.5-turbo".
            prompt_model (str, optional): The model to use for prompting img. Defaults to "gpt-3.5-turbo".
            chat_temperature (float, optional): The temperature to use for chat. Defaults to 0.3.
            prompt_model_temperature (float, optional): The temperature to use for prompting img. Defaults to 0.3.
            img_size (str, optional): The size of the image. Default is "256x256".

        Returns:
            None
        """
        super().__init__(openai_api_key, chat_temperature, chat_model)
        self.level = level
        self.exam_prompt = exam_prompt
        self.description = description
        self.dalle = DallEAPIWrapper(
            size=img_size, api_key=self.openai_api_key.get_secret_value()
        )  # type: ignore[call-arg]
        self.img_prompt_model = ChatOpenAI(
            model=prompt_model, temperature=prompt_model_temperature, api_key=self.openai_api_key
        )
        self.chain = self._create_chain()

    def _get_system_prompt(self) -> ChatPromptTemplate:

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You will generate a short image description (one paragraph max)
                        which will be later used for generating an image taking into
                        account that this images will be used for evaluating the
                        user how well can describe this image in the context of
                        an {self.level} Speaking English exam.

                        {self.description}

                        Topics about image descriptions which can appear:
                        {self.exam_prompt}
                        """,
                ),
                ("human", "{base_input}"),
            ]
        )

    def _create_chain(self) -> Runnable[Any, Any]:

        img_prompt = PromptTemplate(
            input_variables=["image_desc"],
            template="""You will now act as a prompt generator. I will describe an image to you, and you will create a prompt
            that could be used for image-generation. The style must be realistic:
            Description: {image_desc}""",
        )

        return (
            {"image_desc": self._get_system_prompt() | self.chat_llm | StrOutputParser()}
            | img_prompt
            | self.img_prompt_model
            | StrOutputParser()
        )

    def generate(self) -> str:
        """
        Generate function to create and return an image URL based on input parameters.
        """
        img_url = self.dalle.run(self.chain.invoke({"base_input": "Generate image description"})[:999])
        return img_url
