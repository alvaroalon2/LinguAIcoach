import logging
from operator import itemgetter
from typing import Any, Dict, Optional, Type

# from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains import create_extraction_chain_pydantic, create_tagging_chain_pydantic
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from src.models.base.base_model import EvaluationChatModel

logger = logging.getLogger(__name__)


class EvaluationChatModelQA(EvaluationChatModel):

    class Input(BaseModel):
        response: str
        question: Optional[str] = Field(default="")

    class Output(BaseModel):
        evaluation: str = Field(
            alias="Evaluation",
            description="Summarize evaluation of the response (just if there is)",
            default="",
        )
        tips: str = Field(
            alias="Tips",
            description="tips about complexity and detailed mistakes correction (just if there are)",
            default="",
        )
        example: str = Field(
            alias="Example",
            description="Example of the response following the given  guidelines (just if there is)",
            default="",
        )

    class AnswerTagger(BaseModel):
        """
        Tags the answer considering the following aspects:

        - complexity
        """

        answer_complexity: int = Field(
            description="describes how complex the answer is. It is a number between 0 (simpler) and 10 (more complex)",
            enum=list(range(11)),
        )

    class MistakeExtractor(BaseModel):
        """
        Extracts the mistakes from the text.
        """

        grammar_mistake: Optional[str] = Field(
            description="Grammar syntax mistakes detected in text, just if there are, if not return empty string.",
            default="",
        )

    def __init__(
        self,
        level: str,
        openai_api_key: SecretStr,
        eval_model: str = "gpt-3.5-turbo",
        chat_temperature: float = 0.3,
        eval_temperature: float = 0.3,
    ) -> None:
        """
        Initializes the class with the given parameters.

        Args:
            exam_prompt (str): The prompt for the exam.
            level (str): The level of the exam.
            openai_api_key (SecretStr): The API key for OpenAI.
            eval_model (str, optional): The model to use for evaluation. Defaults to "gpt-3.5-turbo".
            chat_temperature (float, optional): The temperature to use for chat. Defaults to 0.3.
            eval_temperature (float, optional): The temperature to use for evaluation. Defaults to 0.3.

        Returns:
            None
        """
        super().__init__(level=level, openai_api_key=openai_api_key, chat_temperature=chat_temperature)

        self.checker_llm = ChatOpenAI(api_key=self.openai_api_key, temperature=eval_temperature, model=eval_model)
        self.prompt = self._get_system_prompt()

        self.chain = self._create_chain()

        self.config = RunnableConfig({})
        # {"callbacks": [ConsoleCallbackHandler()]}

    def _get_multi_chain_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing three chains for extracting mistakes, tagging responses, and retrieving relevant information from an item.

        The dictionary has the following keys:
        - "tags": A chain for tagging responses.
        - "extraction": A chain for extracting mistakes.
        - "base_response": A function that retrieves the "base_response" field from an item.
        - "question": A function that retrieves the "question" field from an item.

        The "tags" chain is created using the `create_tagging_chain_pydantic` function, with the `AnswerTagger` pydantic schema and a prompt template.
        The "extraction" chain is created using the `create_extraction_chain_pydantic` function, with the `MistakeExtractor` pydantic schema and a prompt template.

        Returns:
            dict: A dictionary containing the three chains and the relevant item getter functions.
        """
        chain_extractor = create_extraction_chain_pydantic(
            pydantic_schema=self.MistakeExtractor,
            llm=self.checker_llm,
            prompt=PromptTemplate(
                template="Extract the mistakes found on: {base_response}", input_variables=["base_response"]
            ),
        )

        chain_tagger = create_tagging_chain_pydantic(
            pydantic_schema=self.AnswerTagger,
            llm=self.checker_llm,
            prompt=PromptTemplate(
                template="Tag the given response: {base_response}", input_variables=["base_response"]
            ),
        )

        return {
            "tags": chain_tagger,
            "extraction": chain_extractor,
            "base_response": itemgetter("base_response"),
            "question": itemgetter("question"),
        }

    def _get_system_prompt(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{format_instructions}"
                    """You are an excellent english teacher. You teach spanish people to speak English.
                    You will do the following tasks for that purpose:
                    - You will evaluate the quality of the responses given by the user.
                    - if a non-related text is asked you will politely decline to answer and you will
                    suggest to stay on the topic.
                    - Limit your evaluation to just once per interaction at a time.
                    - guide client to adquire fluency for English exams.
                    """,
                ),
                ("ai", "AI Question: {question}"),
                ("human", "Human Response: {base_response}"),
                ("ai", "Tags:\n{tags}\n\\Extraction:\n{extraction}"),
                (
                    "system",
                    f"""Generate a final response given the AI Question, the Human Response and the detected Tags and Extraction:
                    - correct mistakes (just if there are) based on the Human Response
                        given by the MistakeExtractor according to the {self.level} english level.
                    - give relevant and related tips based on how complete the Human Response is given the punctuation of
                        the answer_complexity AnswerTagger Tags. Best responses are 7, 8 point responses since they are neither too simple
                        nor too complex.
                        - With too simple responses (1, 2, 3, 4 points) you must suggest an alternative response with a higher
                        degree of complexity.
                        - With too complex responses (9, 10 points) you must highlight which part of the response should be ignored.
                    - An excellent response must be grammatically correct, complete and clear.
                    - You will propose an excellent example answer to the AI Question given the above guidelines.
                    """,
                ),
            ]
        )
        return prompt

    def _get_output_parser(self, pydantic_schema: Type[BaseModel]) -> PydanticOutputParser[Any]:

        return PydanticOutputParser(pydantic_object=pydantic_schema)

    def _create_chain(self) -> Runnable[Any, Any]:
        response_parser = self._get_output_parser(self.Output)
        prompt = self.prompt.partial(format_instructions=response_parser.get_format_instructions())

        final_responder = prompt | self.chat_llm | response_parser

        return self._get_multi_chain_dict() | final_responder

    def predict(self, response: str, question: str) -> Dict[str, str]:

        input_model = self.Input(response=response, question=question)

        result = self.chain.invoke(
            {"base_response": input_model.response, "question": input_model.question}, config=self.config
        )

        return result.dict(by_alias=True)
