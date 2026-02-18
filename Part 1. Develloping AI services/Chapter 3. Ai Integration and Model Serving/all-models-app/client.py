import requests
import streamlit as st

st.title("FastAPI ChatBot")

mode = st.radio("Mode de génération", ["Texte", "Audio", "Image"], horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, bytes):
            # Distinguer audio et image via le type stocké
            if message.get("type") == "image":
                st.image(content)
            else:
                st.audio(content)
        else:
            st.markdown(content)

if prompt := st.chat_input("Write your prompt in this input field"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.text(prompt)

    if mode == "Image":
        response = requests.get(
            "http://localhost:8000/generate/image",
            params={"prompt": prompt},
        )
        response.raise_for_status()
        assistant_content = response.content
        with st.chat_message("assistant"):
            st.text("Here is your generated image")
            st.image(assistant_content)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content, "type": "image"}
        )

    elif mode == "Audio":
        response = requests.get(
            "http://localhost:8000/generate/audio",
            params={"prompt": prompt},
        )
        response.raise_for_status()
        assistant_content = response.content
        with st.chat_message("assistant"):
            st.text("Here is your generated audio")
            st.audio(assistant_content)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content, "type": "audio"}
        )

    else:
        response = requests.get(
            "http://localhost:8000/generate/text",
            params={"prompt": prompt},
        )
        response.raise_for_status()
        assistant_content = response.text
        with st.chat_message("assistant"):
            st.markdown(assistant_content)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content, "type": "text"}
        )
