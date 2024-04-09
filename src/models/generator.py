from operator import itemgetter
from typing import Any, List, Type

from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAI

from src.models.lc_base_model import ContentGenerator


class QuestionGenerator(ContentGenerator):
    class Question(BaseModel):
        question: str = Field(alias="Question", description="A question based on the given guidelines")

    def __init__(
        self,
        exam_prompt: str,
        level: str,
        description: str,
        history_chat: List[HumanMessage | AIMessage],
        openai_api_key: SecretStr,
        chat_temperature: float = 0.3,
    ) -> None:
        """
        Initializes the object with the given exam prompt, level, description, history chat, and OpenAI API key.

        Parameters:
            exam_prompt (str): The prompt for the exam.
            level (str): The level of the exam.
            description (str): The description of the exam.
            history_chat (List[HumanMessage | AIMessage]): List of chat messages from history.
            openai_api_key (SecretStr): The API key for OpenAI.

        Returns:
            None
        """
        super().__init__(openai_api_key=openai_api_key, chat_temperature=chat_temperature)
        self.level = level
        self.exam_prompt = exam_prompt
        self.description = description
        self.memory = ConversationBufferWindowMemory(
            chat_memory=ChatMessageHistory(messages=history_chat), return_messages=True, k=20
        )
        self.chain = self._create_chain()

    def _get_system_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{format_instructions}"
                    f"""
                    You are an excellent english teacher. You teach people to speak English by asking questions
                    to later evaluate its response.
                    You will do the following tasks for that purpose:
                    - Limit your questions to just once per interaction at a time.
                    - You will generate a question based on the following guidelines in order to
                    later evaluate the response given by the user.
                    - Don't repeat ANY previous question from the history.
                    - Don't ask too abstract or very specific questions.
                    - here below, some example structure questions are given. This is just a reference of
                    the kind of questions that is expected you provide. Feel free to ask another questions,
                    but always in the context of an {self.level} Speaking English exam. Remember to ask
                    just one question at a time and don't repeat ANY previous questions.

                    {self.description}
                    """,
                ),
                MessagesPlaceholder(
                    variable_name="history",
                ),
                (
                    "human",
                    "I'm ready to start, ask me a question. Do NOT repeat ANY previous AI questions from the history.",
                ),
            ]
        )
        return prompt

    def _get_output_parser(self, pydantic_schema: Type[BaseModel]) -> PydanticOutputParser[Any]:

        return PydanticOutputParser(pydantic_object=pydantic_schema)

    def _create_chain(self) -> Runnable[Any, Any]:
        response_parser = self._get_output_parser(self.Question)

        memory_runable = RunnablePassthrough.assign(
            history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
        )

        parsed_chain = (
            memory_runable
            | self._get_system_prompt().partial(format_instructions=response_parser.get_format_instructions())
            | self.chat_llm
            | response_parser
        )

        unparsed_chain = (
            memory_runable
            | self._get_system_prompt().partial(format_instructions="")
            | self.chat_llm
            | StrOutputParser()
        )

        chain = parsed_chain.with_fallbacks([unparsed_chain])

        return chain

    def generate(self) -> str:
        """
        A function that generates a response by invoking a chain, adding the AI message to chat memory, and returning a question from the response.
        """

        response = self.chain.invoke({})

        if isinstance(response, BaseModel):
            response = response.dict(by_alias=True)
            response = response["Question"]

        self.memory.chat_memory.add_ai_message(response)

        return response


class ImgGenerator(ContentGenerator):

    def __init__(
        self,
        exam_prompt: str,
        level: str,
        description: str,
        openai_api_key: SecretStr,
        chat_temperature: float = 0.3,
        img_size: str = "256x256",
    ) -> None:
        """
        Initializes the class with the exam prompt, level, description, OpenAI API key, and optional image size.

        Parameters:
            exam_prompt (str): The prompt for the exam.
            level (str): The level of the exam.
            description (str): Description of the exam.
            openai_api_key (SecretStr): The OpenAI API key.
            img_size (str, optional): The size of the image. Default is "256x256".

        Returns:
            None
        """
        super().__init__(openai_api_key, chat_temperature)
        self.level = level
        self.exam_prompt = exam_prompt
        self.description = description
        self.dalle = DallEAPIWrapper(size=img_size, api_key=self.openai_api_key.get_secret_value())  # type: ignore[call-arg]
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
            | OpenAI(temperature=0.7, api_key=self.openai_api_key)
            | StrOutputParser()
        )

    def generate(self) -> str:
        """
        Generate function to create and return an image URL based on input parameters.
        """
        img_url = self.dalle.run(self.chain.invoke({"base_input": "Generate image description"})[:999])
        return img_url
