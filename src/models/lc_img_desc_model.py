from operator import itemgetter
from typing import Any, Dict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from typing_extensions import override

from src.models.lc_qa_model import EvaluationChatModelQA


class EvaluationChatModelImg(EvaluationChatModelQA):

    class Input(BaseModel):
        image_url: str
        user_desc: str

    class Output(BaseModel):
        evaluation: str = Field(
            alias="Evaluation", description="Summarize evaluation of the response (just if there is)"
        )
        tips: str = Field(
            alias="Tips", description="tips about complexity and detailed mistakes correction (just if there are)"
        )
        example: str = Field(
            alias="Example",
            description="Example of a response based on the AI image description following the given guidelines (just if there is)",
        )

    def __init__(self, level: str, openai_api_key: SecretStr, chat_temperature: float = 0.3) -> None:
        self.gpt4v = ChatOpenAI(temperature=0.3, model="gpt-4-vision-preview", max_tokens=1024, api_key=openai_api_key)
        super().__init__(level, openai_api_key=openai_api_key, chat_temperature=chat_temperature)

    def _get_system_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{format_instructions}"
                    f"""You will evaluate how close is the the image
                    description given by the user compared to the image
                    description given by the ai vision model. Take into account
                    that this image description will be used for evaluating
                    the user how well can describe this image in the context
                    of an {self.level} Speaking English exam. An Example of
                    an image description will also be provided based on the AI image description.""",
                ),
                ("ai", "AI image description: {ai_img_desc}"),
                ("human", "User image description: {base_response}"),
                ("ai", "Tags:\n{tags}\n\\Extraction:\n{extraction}"),
                (
                    "system",
                    f"""Generate a final response given the question, its response and the detected Tags and Extraction:
                    - correct mistakes (just if there are) based on the response
                        given by the MistakeExtractor according to the {self.level} english level.
                    - give relevant and related tips based on how complete the answer is given the punctuation of
                        the AnswerTagger. Best responses are 7, 8 point questions since they are neither too simple
                        nor too complex.
                        - With too simple questions (1, 2, 3, 4 points) you must suggest an alternative response with a higher
                        degree of complexity.
                        - With too complex questions (9, 10 points) you must highlight which part of the response should be ignored.
                    - An excellent response must be grammatically correct, complete and clear.
                    - Provide an example answer to the question given the above guidelines and the AI image description.
                    """,
                ),
            ]
        )

    def _get_multi_chain_dict(self) -> Dict[str, Any]:
        multi_chain_dict = super()._get_multi_chain_dict()
        multi_chain_dict = {key: multi_chain_dict[key] for key in ["tags", "extraction"]}

        multi_chain_dict.update(
            {
                "ai_img_desc": itemgetter("image_url") | self.gpt4v | StrOutputParser(),
                "base_response": itemgetter("base_response"),
            }
        )

        return multi_chain_dict

    @override
    def predict(self, user_desc: str, image_url: str) -> Dict[str, str]:
        """Make a prediction using the provided input.

        Args:
            user_desc (str): The user description.
            image_url (str): The image url.

        Returns:
            Dict: The output of the prediction.
        """
        input_model = self.Input(user_desc=user_desc, image_url=image_url)
        gpt4v_input = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What is this image showing?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": input_model.image_url, "detail": "auto"},
                    },
                ]
            )
        ]
        result = self.chain.invoke(
            {"base_response": input_model.user_desc, "image_url": gpt4v_input}, config=self.config
        )
        return result.dict(by_alias=True)
