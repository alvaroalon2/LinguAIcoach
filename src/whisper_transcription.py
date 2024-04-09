import io
from typing import Any, Callable, Dict, Optional, Tuple

import streamlit as st
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder


def whisper_stt(  # noqa: C901, PLR0912
    start_prompt: str = "Start recording ⏺️",
    stop_prompt: str = "Stop recording ⏹️",
    just_once: bool = False,
    key: Optional[str] = None,
    use_container_width: bool = False,
    language: Optional[str] = None,
    callback: Optional[Callable[..., None]] = None,
    n_max_retry: int = 3,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any]
) -> Optional[str]:
    """
    Generate speech-to-text (STT) from recorded audio using OpenAI.

    Args:
        start_prompt (str): The prompt to start recording.
        stop_prompt (str): The prompt to stop recording.
        just_once (bool): Flag to record audio just once or continuously.
        use_container_width (bool): Flag to use container width for the recording interface.
        language (Optional[str]): The language for the text transcription.
        callback (Optional[Callable[..., None]]): Callback function to execute after new output is generated.
        args (Tuple[Any, ...]): Positional arguments to pass to the callback function.
        kwargs (Dict[str, Any]): Keyword arguments to pass to the callback function.
        key (Optional[str]): Key to store the output in the session state.

    Returns:
        Optional[str]: The generated speech-to-text output or None if unsuccessful.
    """
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key.get_secret_value())
    if "_last_speech_to_text_transcript_id" not in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    if "_last_speech_to_text_transcript" not in st.session_state:
        st.session_state._last_speech_to_text_transcript = None
    if key and key + "_output" not in st.session_state:
        st.session_state[key + "_output"] = None

    audio = mic_recorder(
        start_prompt=start_prompt,
        stop_prompt=stop_prompt,
        just_once=just_once,
        use_container_width=use_container_width,
        key=key,
    )

    new_output = False
    if audio is None:
        output = None
    else:
        audio_id = audio["id"]
        new_output = audio_id > st.session_state._last_speech_to_text_transcript_id
        if new_output:
            output = None
            st.session_state._last_speech_to_text_transcript_id = audio_id
            audio_bio = io.BytesIO(audio["bytes"])
            audio_bio.name = "audio.mp3"
            success = False
            err = 0
            while not success and err < n_max_retry:
                try:
                    transcript = st.session_state.openai_client.audio.transcriptions.create(
                        model="whisper-1", file=audio_bio, language=language
                    )
                except Exception as e:
                    print(str(e))
                    err += 1
                else:
                    success = True
                    output = transcript.text
                    st.session_state._last_speech_to_text_transcript = output
        elif not just_once:
            output = st.session_state._last_speech_to_text_transcript
        else:
            output = None

    if key:
        st.session_state[key + "_output"] = output
    if new_output and callback:
        callback(*args, **kwargs)
    return output
