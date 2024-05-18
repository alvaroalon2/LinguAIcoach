import logging
import os

import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import SecretStr

from src.models.model_factory import EvaluationChatModelFactory, GeneratorModelFactory
from src.utils import convert_json_to_str, read_json, read_plain_text
from src.whisper_transcription import whisper_stt

logging.basicConfig(level=os.getenv("ENV", "INFO"))

start_container = st.container()
with start_container:
    col1_head, col2_head = st.columns([0.15, 0.85])

col1_head.image("./docs/images/logo.png", width=80)
col2_head.title("LinguAIcoach")

st.subheader("_Your :red[AI] English Teacher_")
st.divider()


model_factory = EvaluationChatModelFactory()
gen_factory = GeneratorModelFactory()
exam_guides = read_json("./exam_guides/lessons.json")
config = read_json("./src/config.json")

input_text = None
input_voice = None

with st.sidebar as sidebar:
    exam_selection = st.sidebar.selectbox("Select Exam type", list(exam_guides.keys()))
    exam_selection = exam_selection if exam_selection else str(next(iter(exam_guides.keys())))
    exam_info = exam_guides[exam_selection]
    st.session_state.openai_api_key = SecretStr(
        os.getenv(
            "OPENAI_API_KEY",
            st.text_input("OpenAI API Key", key="chatbot_api_key", type="password"),
        )
    )

    st.markdown("[:red[Get your OpenAI API key]](https://platform.openai.com/account/api-keys)")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alvaroalon2/LinguAIcoach)"
    )

    logging.debug(f"Selected exam: {exam_selection}")
    logging.debug(f"Exam guides: {exam_guides}")

exam_content = read_plain_text(exam_info["file"])
exam_desc = exam_info["description"]
init_prompt = exam_info["init_prompt"]
level = exam_info["level"]

if "current_exam" not in st.session_state:
    st.session_state["current_exam"] = ""
if st.session_state["current_exam"] != exam_selection:
    logging.debug("Resetting state")
    input_text = None
    input_voice = None
    st.session_state["current_exam"] = exam_selection
    st.session_state["messages"] = [AIMessage(content=init_prompt)]
    st.session_state.first_run = True
    st.session_state["image_url"] = None
    st.session_state["question"] = []
    st.session_state["exam_type"] = exam_info["type"]

response_container = st.container()
start_container = st.container()
with start_container:
    col1_start, col2_start, col3_start = st.columns([0.4, 0.2, 0.4])
voice_container = st.container()
with voice_container:
    col1_voice, col2_voice, col3_voice = st.columns([0.39, 0.22, 0.39])

if not st.session_state.openai_api_key:
    logging.warning("Please add your OpenAI API key to continue.")
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

eval_model = config["EVALUATION_MODEL"][st.session_state["exam_type"]]["model"]
eval_temperature = config["EVALUATION_MODEL"][st.session_state["exam_type"]]["temperature"]

model_eval = model_factory.create_model(
    model_class=st.session_state["exam_type"],
    openai_api_key=st.session_state.openai_api_key,
    level=level,
    chat_temperature=config["OPENAI_TEMPERATURE_EVAL"],
    eval_model=eval_model,
    eval_temperature=eval_temperature,
)

generator = gen_factory.create_model(
    model_class=st.session_state["exam_type"],
    openai_api_key=st.session_state.openai_api_key,
    exam_prompt=exam_content,
    level=level,
    description=exam_desc,
    chat_temperature=config["OPENAI_TEMPERATURE_GEN"],
    n_messages_memory=config["N_MAX_HISTORY"],
    chat_model=config["MODEL_TEXT_GEN"],
    prompt_model=config["MODEL_IMG_PROMPT"],
    prompt_model_temperature=config["IMG_PROMPT_TEMPERATURE"],
    history_chat=[
        AIMessage(content=f"Previous question (Don't repeat): {q.content}")
        for q in st.session_state["question"][-config["N_MAX_HISTORY"] :]
    ],
    img_size=config["IMG_GEN_SIZE"],
)

if "messages" in st.session_state:
    logging.debug(f"Starting exercises for exam_type: {st.session_state['exam_type']}")
    placeholder_start = col2_start.empty()
    start_button = placeholder_start.button("Start exercises!", disabled=not (st.session_state.first_run))
    if start_button:
        logging.debug("Start button clicked, running exercise")
        st.session_state.first_run = False
        if st.session_state["exam_type"] == "qa":
            start_response = generator.generate()
            logging.info(f"Generated first question: {start_response}")
            st.session_state["question"].append(AIMessage(content=start_response))
            st.session_state["messages"].append(st.session_state["question"][-1])
        elif st.session_state["exam_type"] == "img_desc":
            st.session_state["image_url"] = generator.generate()
            logging.debug(f"Generated first image URL: {st.session_state['image_url']}")
            st.session_state["messages"].append(
                AIMessage(
                    content=st.session_state["image_url"],
                    response_metadata={"type": "image"},
                )
            )

if not st.session_state.first_run:
    placeholder_start.empty()

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        if getattr(msg, "response_metadata", None) and msg.response_metadata["type"] == "image":
            with response_container.chat_message("user"):
                response_container.image(str(msg.content), caption="AI generated image")
        else:
            response_container.chat_message("user").write(msg.content)
    elif getattr(msg, "response_metadata", None) and msg.response_metadata["type"] == "image":
        with response_container.chat_message("assistant"):
            response_container.write("Describe what you can see in the following image: ")
            response_container.image(msg.content, caption="AI generated image")
    else:
        response_container.chat_message("assistant").write(msg.content)

placeholder_input = st.empty()
if st.session_state.first_run:
    placeholder_input.empty()
    col2_voice.empty()
else:
    input_text = placeholder_input.chat_input(disabled=st.session_state.first_run) or None
    logging.debug(f"Input text: {input_text}")
    with col2_voice:
        input_voice = whisper_stt(language="en", n_max_retry=config["N_MAX_RETRY"])
        logging.debug(f"Input voice: {input_voice}")

input_prompt = input_text or input_voice

if input_prompt := input_text or input_voice:
    logging.info(f"Processing input: {input_prompt}")

    match st.session_state["exam_type"]:
        case "qa":
            response = model_eval.predict(input_prompt, st.session_state["question"][-1].content)
            logging.info(f"QA model response: {response}")
        case "img_desc":
            response = model_eval.predict(input_prompt, st.session_state["image_url"])
            logging.info(f"Image description model response: {response}")

    response_str = convert_json_to_str(response)

    with response_container:
        st.session_state.messages.append(HumanMessage(content=input_prompt))
        logging.debug(f"Adding user message to session state: {input_prompt}")
        response_container.chat_message("user").write(input_prompt)

        st.session_state.messages.append(AIMessage(content=response_str))
        logging.debug(f"Adding AI message to session state: {response_str}")
        response_container.chat_message("assistant").write(response_str)
        if st.session_state["exam_type"] == "qa":
            new_question = generator.generate()
            logging.info(f"Generated new question: {new_question}")
            st.session_state["question"].append(AIMessage(content=new_question))
            st.session_state.messages.append(AIMessage(content=new_question))
            response_container.chat_message("assistant").write(new_question)
        elif st.session_state["exam_type"] == "img_desc":
            new_image = generator.generate()
            st.session_state["image_url"] = new_image
            st.session_state.messages.append(AIMessage(content=new_image))
            logging.info(f"Generated new image URL: {st.session_state['image_url']}")
            response_container.chat_message("assistant").write("Describe what you can see in the following image: ")
            with response_container.chat_message("assistant"):
                response_container.image(st.session_state["image_url"], caption="AI generated image")
